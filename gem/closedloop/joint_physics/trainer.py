"""从随机初始化开始的单优化器联合训练，支持同步DDP和逐样本物理拒绝。

完整1000步扩散监督作用于常规batch，执行BPTT只用保留音乐的随机子批。DDP按全局有效
物理样本数归一化，各rank含零有效样本时仍执行同一backward。保存数据、噪声与窗口游标，
课程从零推进；诊断间隔才计算两条损失的参数梯度分解，日常仅测小型x0执行梯度。
真实学生无合法参考时如实记录physics_updates=0，绝不用GT替换学生预测。
"""

from collections import Counter
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .config import build_actor, build_adapter, seed_all
from .contracts import PhysicsRejected, digest, execution_contract
from .curriculum import Readiness, physics_noise_levels
from .data import JointDataStream, SidecarStore
from .checkpoint import read_checkpoint, restore_trainer, save_trainer

GRADIENT_GROUPS = ('future_encoder','stage1.denoiser','stage1.history_encoder',
                   'stage1.music_embedder','stage1.cond_exists_embedder','stage1.prefix_encoder')


def grad_summary(loss, actor, *, retain_graph=True):
    named = [(name, parameter) for name, parameter in actor.named_parameters() if parameter.requires_grad]
    values = torch.autograd.grad(loss, [p for _, p in named], retain_graph=retain_graph, allow_unused=True)
    result = {}
    for group in GRADIENT_GROUPS:
        terms = [g.float().square().sum() for (name, _), g in zip(named, values) if name.startswith(group) and g is not None]
        result[group] = float(torch.stack(terms).sum().sqrt()) if terms else 0.
    return result


def distributed_execution_scale(local_count, device, world_size):
    """DDP平均梯度与全局有效样本均值的精确换算；零样本rank仍参加collective。"""
    count = torch.tensor(float(local_count), device=device)
    if world_size > 1:
        dist.all_reduce(count)
    return count, world_size/count.clamp_min(1)


class JointTrainer:
    def __init__(self, cfg, device, *, rank=0, world_size=1, resume=None):
        self.cfg, self.device, self.rank, self.world_size = cfg, torch.device(device), rank, world_size
        resume_value = read_checkpoint(resume) if resume else None
        seed_all(cfg["training"]["seed"])
        self.actor, self.data_loss = build_actor(cfg, device)
        seed_all(cfg["training"]["seed"] + rank * 100003)
        self.stream = JointDataStream(cfg, rank=rank, world_size=world_size)
        self.adapter = build_adapter(cfg, device) if cfg["physics"]["enabled"] or cfg["data"]["sidecar"] else None
        from .prepare_rollouts import asset_contract
        from .execution import ValidatedSupport
        self.assets = asset_contract(cfg, self.adapter) if self.adapter is not None else {
            "stats_sha256": digest(cfg["stage1"]["endecoder"]["stats_path"]),
            "fk_sha256": self.actor.stage1.endecoder.kinematics.kinematics_sha256}
        self.frozen = self.adapter.bundle.frozen_snapshot() if self.adapter is not None else None
        self.store = SidecarStore(cfg["data"]["sidecar"], "train", self.assets,
            sample_contract=cfg["data"]["sample_contract"],
            snapshot=resume_value.get("sidecar_snapshot") if resume_value else None) if cfg["data"]["sidecar"] else None
        report = cfg["physics"].get("validation_report") if cfg["physics"]["enabled"] else None
        self.support = ValidatedSupport(report, self.assets, cfg["physics"]["horizon"], device, execution_contract(cfg)) if report else None
        self.noise_rng = torch.Generator().manual_seed(cfg["training"]["seed"]+1000+rank)
        self.window_rng = np.random.default_rng(cfg["training"]["seed"]+2000+rank)
        self.physics_noise_rng = torch.Generator().manual_seed(cfg["training"]["seed"]+5000+rank)
        self.future_noise_rng = torch.Generator().manual_seed(cfg["training"]["seed"]+7000+rank)
        self.readiness = Readiness(cfg["physics"])
        self.future_stream = None
        self.physics_stream = None
        self.physics_levels = None
        if cfg["physics"]["enabled"] and self.store is not None and self.support is not None:
            from .requests import PhysicsRequestStream
            self.physics_stream = PhysicsRequestStream(self.stream.catalog, self.store, self.adapter.bundle.spec.joint_names,
                cfg["physics"]["horizon"], cfg["training"]["seed"], rank, self.support,
                historical_mode=cfg["physics"].get("historical_support_mode", "strict"),
                noise_levels=cfg["physics"].get("noise_levels"))
            self.physics_levels = physics_noise_levels(self.actor.stage1.train_diffusion, cfg["physics"], self.support.report)
        if cfg.get("future_supervision", {}).get("enabled") and self.store is not None:
            from .requests import PhysicsRequestStream
            self.future_stream = PhysicsRequestStream(self.stream.catalog, self.store, self.adapter.bundle.spec.joint_names,
                1, cfg["training"]["seed"]+100000, rank, historical_mode="diagnostic")
        self.optimizer = torch.optim.AdamW((p for p in self.actor.parameters() if p.requires_grad), lr=cfg["training"]["learning_rate"],
                                           weight_decay=cfg["training"]["weight_decay"])
        tc = cfg["training"]
        def lr(step):
            if step < tc["lr_warmup_steps"]:
                return (step+1)/max(1, tc["lr_warmup_steps"])
            progress = min(1., (step-tc["lr_warmup_steps"])/max(1, tc["max_steps"]-tc["lr_warmup_steps"]))
            return tc["min_lr_ratio"]+(1-tc["min_lr_ratio"])*.5*(1+math.cos(math.pi*progress))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr)
        self.step = 0
        self.counters = {"supervised_updates": 0, "physics_updates": 0, "supervised_samples": 0, "physics_samples": 0,
                         "physics_attempts": 0, "physics_valid_samples": 0, "physics_effective_updates": 0,
                         "future_supervised_updates": 0, "effective_horizon_counts": {},
                         "physics_rejections_total": {}, "historical_diagnostics_total": {}}
        self.contract = {"assets": self.assets, "dataset": self.stream.catalog.identity,
            "sidecar": self.store.identity if self.store else None, "model": cfg["stage1"]["model"],
            "loss": cfg["stage1"]["loss"], "sampling": cfg["sampling"], "physics": cfg["physics"],
            "execution_loss": cfg["execution_loss"], "training": cfg["training"],
            "future_supervision": cfg.get("future_supervision"),
            "data_sample_contract": cfg["data"]["sample_contract"],
            "validation_sha256": digest(report) if report else None}
        if resume:
            restore_trainer(self, resume, resume_value)
        self.network = DistributedDataParallel(self.actor, device_ids=[self.device.index] if self.device.type == "cuda" else None,
                                              find_unused_parameters=True) if world_size > 1 else self.actor

    def update(self):
        from .reference import attach_sidecars
        from .execution import execute_window
        from .diagnostics import sample_record, summarize
        started = time.perf_counter()
        self.actor.train()
        self.optimizer.zero_grad(set_to_none=True)
        batch = self.stream.next(self.cfg['training']['batch_size'], self.device)
        # A/B完全相同的真实/代理历史；物理认证只影响执行分支，不能改变监督条件。
        batch, contexts, _, missing = attach_sidecars(batch, self.store, self.adapter,
            self.actor.stage1.endecoder.codec, self.cfg['physics']['horizon'])
        b = len(batch['decision_time'])
        timesteps = torch.randint(1000, (b,), generator=self.noise_rng).to(self.device)
        noise = torch.randn(b, 60, 30, generator=self.noise_rng).to(self.device)
        drop = (torch.rand(b, generator=self.noise_rng) < self.actor.stage1.music_mask_prob).to(self.device)
        pcfg = self.cfg['physics']
        if pcfg.get('readiness_hard_gate', True):
            probe = pcfg['enabled'] and (self.readiness.activated_step is not None or self.step % pcfg['probe_every'] == 0)
        else:
            probe = pcfg['enabled'] and self.step >= pcfg['start_step'] and (self.step-pcfg['start_step']) % pcfg.get('every_steps',4) == 0
        physical_inputs, physics_meta, windows, physics_missing = None, [], [], []
        selection_reason = 'curriculum_not_active' if pcfg['enabled'] else 'physics_disabled'
        if probe:
            if self.physics_stream is None:
                selection_reason = 'missing_sidecar_or_certification'
            elif not self.physics_stream.keys:
                selection_reason = 'no_certified_offline_requests'
            else:
                pbatch, physics_meta = self.physics_stream.next(pcfg['subbatch'], self.device)
                pbatch, pcontexts, windows, physics_missing = attach_sidecars(pbatch, self.store, self.adapter,
                    self.actor.stage1.endecoder.codec, pcfg['horizon'])
                n = len(physics_meta)
                choices = [[k for k in m['timesteps'] if k in self.physics_levels.tolist()] for m in physics_meta]
                if any(not c for c in choices):
                    raise ValueError('已认证类型没有满足物理SNR区间的实际噪声等级')
                k = torch.tensor([c[int(torch.randint(len(c),(1,),generator=self.physics_noise_rng))] for c in choices],device=self.device)
                epsilon = torch.randn(n,60,30,generator=self.physics_noise_rng).to(self.device)
                physical_inputs = dict(batch=pbatch, forecast_context=pcontexts, timesteps=k, noise=epsilon,
                                       drop_music_mask=torch.zeros(n,dtype=torch.bool,device=self.device))
        future_inputs, future_batch = None, None
        fcfg = self.cfg.get('future_supervision', {})
        if self.future_stream is not None and self.future_stream.keys and self.step % fcfg['every_steps'] == 0:
            future_batch, _ = self.future_stream.next(fcfg['batch_size'], self.device)
            future_batch, future_contexts, _, future_reasons = attach_sidecars(future_batch, self.store, self.adapter,
                self.actor.stage1.endecoder.codec, 1)
            if any(reason != 'available' for reason in future_reasons):
                raise ValueError('已索引Future监督请求的持久缓存合同损坏: '+str(future_reasons))
            n = len(future_contexts)
            future_inputs = dict(batch=future_batch, forecast_context=future_contexts,
                timesteps=torch.randint(1000,(n,),generator=self.future_noise_rng).to(self.device),
                noise=torch.randn(n,60,30,generator=self.future_noise_rng).to(self.device),
                drop_music_mask=(torch.rand(n,generator=self.future_noise_rng)<self.actor.stage1.music_mask_prob).to(self.device))
        output = self.network(batch, contexts, timesteps=timesteps, noise=noise, drop_music_mask=drop,
                              physics_inputs=physical_inputs, future_inputs=future_inputs)
        if future_inputs is not None:
            out, physical, future_out = output
        else:
            out, physical = output if physical_inputs is not None else (output, None)
        future_loss = out['pred_x_start'].new_zeros(())
        if future_inputs is not None:
            future_loss, _ = self.data_loss(future_batch, future_out['pred_x_start'], future_out['static_conf_logits'], self.step)
        bc, components = self.data_loss(batch, out['pred_x_start'], out['static_conf_logits'], self.step)
        accepted, rows, coverage = [], [], []
        if physical is not None:
            alpha = torch.as_tensor(self.actor.stage1.train_diffusion.alphas_cumprod,device=self.device)[k]
            snr = alpha/(1-alpha)
            finite_physics = torch.isfinite(physical['pred_x_start']).flatten(1).all(1) & torch.isfinite(physical['static_conf_logits']).flatten(1).all(1)
            # 一个合批前向中有NaN时，其余样本的零上游梯度也可能与NaN激活相乘污染参数。
            # 整个物理子批拒绝，分别记录自身非有限或同批连带拒绝；常规BC图保持完整。
            nonfinite_physics_batch = not bool(finite_physics.all())
            for i, metadata in enumerate(physics_meta):
                record = sample_record(pbatch,i,k[i],snr[i])
                record.update(tags=metadata['tags'],music_identity=metadata['music_identity'], stages=['sampled','static_metadata_passed','audited_noise'],
                              action_origin='not_executed')
                support = None
                try:
                    if nonfinite_physics_batch:
                        raise PhysicsRejected('nonfinite_denoised_x0_or_contact' if not bool(finite_physics[i])
                                              else 'physics_batch_rejected_nonfinite_peer',
                                              {'first_reject_tick':record['request_tick'],'action_origin':'not_executed',
                                               'failure_phase':'denoising_output'})
                    if windows[i] is None:
                        raise PhysicsRejected(physics_missing[i])
                    from .execution import DiagnosticSupport
                    support = (DiagnosticSupport(self.support, metadata['tags'])
                               if pcfg.get('historical_support_mode') == 'diagnostic' else self.support.for_tags(metadata['tags']))
                    record['stages'].append('execution_attempted')
                    result = execute_window(windows[i], physical['pred_x_start'][i:i+1], physical['static_conf_logits'][i:i+1],
                                            self.actor,self.adapter,self.cfg['execution_loss'],support=support,
                                            training_policy=pcfg.get('training_safety'))
                    derivative, = torch.autograd.grad(result['loss'],physical['pred_x_start'],retain_graph=True)
                    if not torch.isfinite(derivative).all() or not bool(derivative.norm()>0):
                        raise PhysicsRejected('zero_or_nonfinite_execution_gradient')
                    result['x0_gradient_norm'] = float(derivative.norm())
                    record['action_origin'] = 'frozen_sm_from_student_denoised_x0'
                    accepted.append(result)
                    coverage.append(result['coverage'])
                    record['reason'] = 'execution_gradient_completed'
                    record['effective_horizon'] = result['effective_horizon']
                    record['x0_gradient_norm'] = result['x0_gradient_norm']
                    record['historical_diagnostics'] = result['historical_diagnostics']
                    record['truncation'] = result['truncation']
                except PhysicsRejected as error:
                    record['reason'] = error.reason
                    record.update(error.details)
                    record['historical_diagnostics'] = dict(getattr(support,'counts',{}))
                rows.append(record)
        global_rows = rows
        if self.world_size > 1:
            gathered = [None]*self.world_size
            dist.all_gather_object(gathered, rows)
            global_rows = [row for rank_rows in gathered for row in rank_rows]
        readiness = self.readiness.observe(global_rows,self.step)
        weight = self.readiness.weight(self.step)
        count,scale = distributed_execution_scale(len(accepted),self.device,self.world_size)
        zero = out['pred_x_start'].sum()*0.
        # 两次前向使用同一Actor，全部参数仍通过完整BC图参加DDP。拒绝的物理输出不乘零：
        # NaN*0仍是NaN，不能因物理子批失败阻断本来有限的数据监督。
        local_exec = sum((r['loss'] for r in accepted),zero)
        total = bc+fcfg.get('weight',0.)*future_loss+scale*weight*local_exec
        finite = torch.tensor(int(torch.isfinite(total).all()),device=self.device)
        if self.world_size>1:
            dist.all_reduce(finite,op=dist.ReduceOp.MIN)
        if not bool(finite):
            raise FloatingPointError('数据或执行损失非有限，各rank同步停止')
        diagnostics, before = {}, {}
        diagnostic = (self.step % self.cfg['training']['diagnostics_every']==0 or self.step in self.cfg['training']['diagnostic_steps']
                      or (weight>0 and count>0 and self.counters['physics_effective_updates']<8))
        if diagnostic and self.world_size==1:
            diagnostics['data_gradient_norm'] = grad_summary(bc,self.actor)
            diagnostics['execution_gradient_norm'] = grad_summary(local_exec/max(1,len(accepted)),self.actor)
            diagnostics['weighted_execution_to_data_gradient_ratio'] = {
                key: weight*value/max(1e-12,diagnostics['data_gradient_norm'][key])
                for key,value in diagnostics['execution_gradient_norm'].items()}
            before = {name:p.detach().clone() for name,p in self.actor.named_parameters() if p.requires_grad}
        total.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(),self.cfg['training']['grad_clip'])
        if not torch.isfinite(norm):
            raise FloatingPointError('GENMO梯度非有限，未执行优化')
        self.optimizer.step()
        self.scheduler.step()
        if before:
            diagnostics['actual_parameter_delta'] = {}
            for group in GRADIENT_GROUPS:
                terms = [(p.detach()-before[name]).float().square().sum() for name,p in self.actor.named_parameters() if name.startswith(group) and name in before]
                diagnostics['actual_parameter_delta'][group] = float(torch.stack(terms).sum().sqrt()) if terms else 0.
        self.step += 1
        effective = bool(weight>0 and count>0)
        self.counters['future_supervised_updates'] += int(future_inputs is not None)
        self.counters['supervised_updates'] += 1
        self.counters['supervised_samples'] += b*self.world_size
        self.counters['physics_attempts'] += len(global_rows)
        self.counters['physics_valid_samples'] += int(count)
        self.counters['physics_effective_updates'] += int(effective)
        self.counters['physics_updates'] += int(effective)
        self.counters['physics_samples'] += int(count) if effective else 0
        reduced = torch.stack([bc.detach(),local_exec.detach(),future_loss.detach()])
        if self.world_size>1:
            dist.all_reduce(reduced)
        reasons = Counter(row['reason'] for row in global_rows if row['reason']!='execution_gradient_completed')
        horizons = Counter(str(row['effective_horizon']) for row in global_rows if row['reason']=='execution_gradient_completed' and effective)
        historical = Counter()
        for row in global_rows:
            historical.update(row.get('historical_diagnostics', {}))
        for key, values in [('effective_horizon_counts',horizons), ('physics_rejections_total',reasons), ('historical_diagnostics_total',historical)]:
            cumulative = Counter(self.counters[key]); cumulative.update(values); self.counters[key] = dict(cumulative)
        return {'step':self.step, 'lr':self.optimizer.param_groups[0]['lr'],
            'future_supervised_loss':float(reduced[2]/self.world_size),
            'physics_gradient_norm':max([row.get('x0_gradient_norm',0.) for row in global_rows] or [0.]),
            'physics_gradient_tensor':'student_pred_x_start',
            'effective_horizon_distribution':dict(horizons),**self.counters,'data_loss':float(reduced[0]/self.world_size),
            'execution_loss':float(reduced[1]/count.clamp_min(1)),'execution_safety':0.,
            'physics_valid_rate':float(count)/max(1,len(global_rows)),'lambda_exec':weight,'lambda_safety':0.,
            'physics_update_fraction':self.counters['physics_effective_updates']/self.step,
            'data_samples_with_inactive_physics_curriculum':b*self.world_size if weight==0 else 0,
            'physics_selection_status':selection_reason if not global_rows else 'independent_student_probe' if not weight else 'independent_student_training',
            'readiness':readiness,'rejections':dict(reasons),'physics_rejection_categories':dict(reasons),
            'sample_diagnostics':summarize(global_rows),'data_sidecar_status':dict(Counter(missing)),
            'timesteps':timesteps.tolist(),'sample_ids':[(m['sample_id'],m['decision_frame']) for m in batch['meta']],
            'music_dropout_mask':drop.tolist(),'history_sources':[m['proprio_source'] for m in batch['meta']],
            'physics_sample_ids':[(m['source'],m['frame']) for m in physics_meta],
            'physics_timesteps':k.tolist() if physical is not None else [],
            'data_terms':{key:float(v.detach().mean()) for key,v in components.items() if isinstance(v,torch.Tensor)},
            'gradient_norm':float(norm),**diagnostics,'coverage':coverage,
            'execution_x0_gradient_norm':[r['x0_gradient_norm'] for r in accepted],
            'wall_seconds':time.perf_counter()-started,
            'peak_memory_bytes':torch.cuda.max_memory_allocated(self.device) if self.device.type=='cuda' else 0}

    def refresh_sidecars(self):
        interval = self.cfg['data'].get('sidecar_refresh_every',0)
        if not interval or not self.step or self.step % interval or self.store is None:
            return
        changed = self.store.refresh()
        if self.world_size > 1:
            identities = [None]*self.world_size
            dist.all_gather_object(identities, self.store.identity)
            if any(value != identities[0] for value in identities):
                raise ValueError('各rank看到的离线缓存索引不一致，拒绝静默改变采样合同')
        if not changed:
            return
        from .requests import PhysicsRequestStream
        for name, seed, horizon, support in [('physics_stream',self.cfg['training']['seed'],self.cfg['physics']['horizon'],self.support),
                                             ('future_stream',self.cfg['training']['seed']+100000,1,None)]:
            old = getattr(self, name)
            if old is None:
                continue
            state = old.state_dict()
            new = PhysicsRequestStream(self.stream.catalog,self.store,self.adapter.bundle.spec.joint_names,horizon,seed,self.rank,support,
                historical_mode=self.cfg['physics'].get('historical_support_mode','strict') if name=='physics_stream' else 'diagnostic',
                noise_levels=self.cfg['physics'].get('noise_levels'))
            state['order'] = [new.keys.index(old.keys[index]) for index in state['order']]
            state['identity'] = new.identity
            new.load_state_dict(state)
            setattr(self,name,new)
        self.contract['sidecar'] = self.store.identity
        if self.rank==0:
            print(json.dumps({'sidecar_refresh_step':self.step,'physics_sources':len(self.physics_stream.keys),
                              'future_sources':len(self.future_stream.keys) if self.future_stream else 0}),flush=True)

    def numerical_diagnostics(self, output):
        """仅rank0运行完整DDIM20和去噪，保存输出；不仿真，不改变训练游标/RNG。"""
        from .checkpoint import rng_state, restore_rng, atomic_save
        from .actor import sampling_mode
        from .reference import attach_sidecars
        from gem.closedloop.contracts import STAGE1_CONDITION_KEYS
        if self.world_size>1:
            dist.barrier()
        if self.rank==0:
            state, rng = self.stream.state_dict(), rng_state()
            try:
                batch = self.stream.next(1,self.device)
                batch,contexts,_,_ = attach_sidecars(batch,self.store,self.adapter,self.actor.stage1.endecoder.codec,1)
                generator = torch.Generator().manual_seed(12345)
                noise = torch.randn(1,60,30,generator=generator).to(self.device)
                with torch.no_grad(), sampling_mode(self.actor):
                    rec = self.actor.denoise_supervised(batch,contexts,torch.tensor([250],device=self.device),noise,
                                                       torch.zeros(1,dtype=torch.bool,device=self.device))
                    loss,_ = self.data_loss(batch,rec['pred_x_start'],rec['static_conf_logits'],self.step)
                    conditions = {key:batch[key] for key in STAGE1_CONDITION_KEYS}
                    free = self.actor.sample_differentiable(conditions,contexts[0],noise)
                atomic_save({'step':self.step,'source':batch['meta'],'reconstruction_loss':float(loss),
                    'reconstruction_x0':rec['pred_x_start'].cpu(),'free_generation_qpos30':free['qpos30'].cpu(),
                    'free_generation_contact':free['contact'].cpu(),'sampling':self.cfg['sampling'],
                    'evidence':'numerical_only_no_newton'},output/f'numerical_{self.step:08d}.pt')
            finally:
                self.stream.load_state_dict(state)
                restore_rng(rng)
        if self.world_size>1:
            dist.barrier()

    def run(self, output, stop_step):
        output = Path(output)
        if self.rank == 0:
            output.mkdir(parents=True, exist_ok=False)
            (output/"config.json").write_text(json.dumps(self.cfg, ensure_ascii=False, indent=2))
        if self.world_size > 1:
            dist.barrier()
        if self.step == 0:
            save_trainer(self, output/"joint_00000000.pt")
        if self.rank == 0 and self.physics_stream is not None:
            (output/"physics_inventory.json").write_text(json.dumps(self.physics_stream.inventory, ensure_ascii=False, indent=2))
        recent_physics_norm = 0.
        try:
            while self.step < stop_step:
                self.refresh_sidecars()
                result = self.update()
                recent_physics_norm = max(recent_physics_norm, result['physics_gradient_norm'])
                if self.rank == 0:
                    with (output/"metrics.jsonl").open("a") as file:
                        file.write(json.dumps(result, ensure_ascii=False, allow_nan=False)+"\n")
                    if self.step <= 3 or self.step % self.cfg['training'].get('log_every',1)==0:
                        keys = ('step','lr','data_loss','data_terms','future_supervised_loss','execution_loss','lambda_exec',
                                'physics_attempts','physics_valid_samples','physics_effective_updates','physics_update_fraction',
                                'effective_horizon_counts','physics_rejections_total','wall_seconds','peak_memory_bytes')
                        print(json.dumps({**{k:result[k] for k in keys},'interval_max_physics_x0_gradient_norm':recent_physics_norm},ensure_ascii=False),flush=True)
                        recent_physics_norm = 0.
                if self.step % self.cfg["training"]["checkpoint_every"] == 0 or self.step == stop_step or self.step in self.cfg["training"]["diagnostic_steps"]:
                    save_trainer(self, output/f"joint_{self.step:08d}.pt")
                every = self.cfg['training'].get('numerical_diagnostics_every',0)
                if every and self.step % every == 0:
                    self.numerical_diagnostics(output)
        finally:
            if self.adapter is not None:
                self.adapter.bundle.assert_unchanged(self.frozen)
        if self.rank == 0:
            summary = {"global_step": self.step, **self.counters, "frozen_unchanged": True,
                       "status": "joint_student_updates_observed" if self.counters["physics_updates"] else "supervision_started_only"}
            (output/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        return self.counters
