"""有界A/B/C/D世界模型对照：同一冻结资产、固定真实标签和逐版本Newton审计。

prepare只采集训练/验证有限幅度参考响应，并暂存一次WM-BPTT Adam完整参数提案作为
数据干预；该提案不计为接受的Adapter更新。A/B/C共享随机参考数据，D额外包含实际
完整参数候选及反向候选。每组从原始WM初始化，最多150次监督，0/25/.../150记录固定
validation误差、原六项与完整奖励响应，独立audit歌曲的真实完整参数方向不进入拟合。
H64只验证；没有DDP、额外控制器或输出头，Critic终端关闭。达到预算完整保存并退出。
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def concatenate_responses(a, b):
    import torch
    if set(a) != set(b):
        raise ValueError('真实四条件字段不一致')
    return {k: torch.cat((a[k], b[k])) if isinstance(a[k], torch.Tensor) else a[k] for k in a}


def calibrate_training_losses(dynamics, control, windows, replay, cfg):
    """只用训练标签测量H8/16/32的真实梯度，预先限制新项的相对强度。"""
    import torch
    from .reference_response import consistency_loss, reward_response_loss
    settings, experiment = cfg['stage2a']['long_horizon'], cfg['wm_ablation']
    params = tuple(dynamics.model.net.parameters())
    rng = torch.cuda.get_rng_state_all()
    ids = torch.randint(len(windows['state']), (64,), device=windows['state'].device)
    long = {k: v[ids] if isinstance(v, torch.Tensor) else v for k, v in windows.items()}
    def prefix(h):
        return {k: v[:, :h] if isinstance(v, torch.Tensor) else v for k, v in long.items()}
    base, _ = dynamics.supervised_loss(prefix(8))
    one, _ = dynamics.one_step_loss(long)
    base = base+experiment['one_step_weight']*one
    for horizon, coefficient in settings['state_weights'].items():
        term, _ = dynamics.supervised_loss(prefix(int(horizon)))
        base = base+coefficient*term
    base_grad = torch.autograd.grad(base, params)
    base_norm = float(sum(g.square().sum() for g in base_grad).sqrt())
    rows = []
    for horizon in (8, 16, 32):
        pairs = replay.sample(32, horizon=horizon)
        if pairs is None:
            raise ValueError('梯度标定缺少训练集完整真实响应')
        response, _ = consistency_loss(dynamics, pairs, horizon, kin=control.runtime.kin,
                                      **settings['response'])
        reward, _ = reward_response_loss(dynamics, control, pairs, horizon,
                                        scale=experiment['reward_response_scale'])
        row = dict(horizon=horizon, base_gradient_norm=base_norm)
        for name, loss in (('reference', response), ('reward', reward)):
            grad = torch.autograd.grad(loss, params)
            norm = float(sum(g.square().sum() for g in grad).sqrt())
            cosine = float(sum((g*b).sum() for g, b in zip(grad, base_grad)))/max(norm*base_norm, 1e-12)
            row.update({name+'_gradient_norm': norm, name+'_base_cosine': cosine,
                        name+'_loss': float(loss.detach())})
        rows.append(row)
    # 同一组预先固定权重；取三个训练时域中最严格的上限，不按审计结果调参。
    fraction = experiment['reward_gradient_fraction']
    reward_weight = min(experiment['reward_response_weight'],
        *(fraction*base_norm/max(r['reward_gradient_norm'], 1e-12) for r in rows))
    reference_weight = min(cfg['stage2a']['reference_response_weight'],
        *(fraction*base_norm/max(r['reference_gradient_norm'], 1e-12) for r in rows))
    for row in rows:
        row.update(reward_weighted_fraction=reward_weight*row['reward_gradient_norm']/max(base_norm,1e-12),
                   reference_weighted_fraction=reference_weight*row['reference_gradient_norm']/max(base_norm,1e-12))
    torch.cuda.set_rng_state_all(rng)
    return dict(rows=rows, effective_weight=reward_weight, reference_weight=reference_weight,
        gradient_fraction_cap=fraction, selection='training_only_H8_H16_H32_no_validation_or_audit_tuning',
        new_terms='each_capped_separately_relative_to_full_state_plus_one_step_supervision')


def prepare_ablation_data(collector, replay, validation, extra, optimizers, critic, cfg, output):
    import torch
    from .vector_control import select_state
    from .vector_learning import adapter_update
    from .update_diagnostics import fixed_probes
    from .stage2a_physics import collect_reference_response
    from .reference_response import ReferenceResponseReplay, score_response_sequences, reward_response_loss
    from .checkpoint import shared_cpu_state
    from .vector_entry import json_write
    from supermimic.bridge.bundle import atomic_save
    control, dynamics = collector.control, collector.control.engine.dynamics
    source = fixed_probes(extra['high_error_starts'], per_song=1)
    if len(source['state']) != 64 or collector.num_envs < 256:
        raise ValueError('数据准备要求64训练歌及256物理环境')
    before = copy.deepcopy(control.adapter.state_dict())
    adam_before = copy.deepcopy(optimizers['adapter'].state_dict())
    random = ReferenceResponseReplay()
    collect = dict(horizon=64, replay=replay, intervention_steps=(1, 8, 32))
    random.add(collect_reference_response(collector, source, **collect))
    candidate, _ = adapter_update(control, dynamics, critic, optimizers['adapter'], source, 32,
        terminal_value=False, microbatch=32, probes=source, checkpoint_steps=True)
    after = copy.deepcopy(control.adapter.state_dict())
    candidate_adam = copy.deepcopy(optimizers['adapter'].state_dict())
    atomic_save(shared_cpu_state(dict(adapter=after, before=before, optimizer=candidate_adam,
        config=cfg, candidate=candidate, accepted_updates=0)), output/'data_only_Adam_candidate.pt')
    if not candidate['effective']:
        raise ValueError('完整Adam提案没有可辨识变化，不能伪造非零策略数据')
    # 非零提案只为真实分布采集，不标记为已验收策略。
    random.add(collect_reference_response(collector, source, **collect))
    train_windows = replay.windows(2048, 32)
    control.adapter.load_state_dict(before)
    parameters = ReferenceResponseReplay()
    parameters.add(collect_reference_response(collector, source, parameter_snapshots=(before, after), **collect))
    train_windows_d = replay.windows(2048, 32)
    validation_pairs = extra['response_validation']
    validation_source = {k.removeprefix('context_'): v[:, 0].clone()
                         for k, v in validation_pairs.items() if k.startswith('context_')}
    param_validation = collect_reference_response(collector, validation_source, horizon=64, split='validation',
        parameter_snapshots=(before, after))
    validation_pairs = concatenate_responses(validation_pairs, param_validation)
    control.adapter.load_state_dict(before); optimizers['adapter'].load_state_dict(adam_before)
    with torch.no_grad():
        reconstructed = score_response_sequences(dynamics, control, validation_pairs, 64, use_real_labels=True)
        valid = validation_pairs['trajectory_valid']
        parity = float((reconstructed-validation_pairs['all_rewards'])[valid].abs().max())
        if parity > 2e-6:
            raise ValueError(f'真实与WM奖励重放合同不一致: {parity}')
    params = tuple(dynamics.model.net.parameters())
    small = random.sample(8, horizon=8)
    base, _ = dynamics.supervised_loss({k: v[:32, :8] if isinstance(v, torch.Tensor) else v for k, v in train_windows.items()})
    one, _ = dynamics.one_step_loss({k: v[:32, :8] if isinstance(v, torch.Tensor) else v for k, v in train_windows.items()})
    reward, _ = reward_response_loss(dynamics, control, small, 8, scale=cfg['wm_ablation']['reward_response_scale'])
    base_grad = torch.autograd.grad(base+cfg['wm_ablation']['one_step_weight']*one, params)
    reward_grad = torch.autograd.grad(reward, params)
    base_norm = float(sum(g.square().sum() for g in base_grad).sqrt())
    reward_norm = float(sum(g.square().sum() for g in reward_grad).sqrt())
    effective_weight = min(cfg['wm_ablation']['reward_response_weight'],
        cfg['wm_ablation']['reward_gradient_fraction']*base_norm/max(reward_norm, 1e-12))
    calibration = dict(base_gradient_norm=base_norm, reward_gradient_norm=reward_norm,
        configured_weight=cfg['wm_ablation']['reward_response_weight'], effective_weight=effective_weight,
        candidate_weights={str(w): w*reward_norm/max(base_norm,1e-12) for w in (0.,.001,.01)},
        selection='training_only_gradient_fraction_cap_before_A_B_C_D_not_validation_selection')
    json_write(output/'reward_gradient_calibration.json', calibration)
    payload = dict(config=cfg, reward_gradient_calibration=calibration, train_windows=train_windows, train_windows_d=train_windows_d,
        validation_windows=validation.windows(256, 64), validation_pairs=validation_pairs,
        random_response=random.state_dict(), parameter_response=parameters.state_dict(),
        zero_adapter=before, candidate_adapter=after, candidate_optimizer=candidate_adam,
        candidate_report=candidate, accepted_adapter_updates=0, source=source,
        real_transitions=collector.real_transitions, real_by_role=collector.real_by_role,
        reward_replay_max_abs=parity, boundary='fixed_real_labels_candidate_not_accepted_training_policy')
    path=output/'fixed_ablation_data.pt'
    atomic_save(shared_cpu_state(payload), path)
    json_write(output/'data_summary.json', dict(path=str(path), random_coverage=random.coverage(),
        parameter_coverage=parameters.coverage(), real_transitions=collector.real_transitions,
        real_by_role=collector.real_by_role, reward_replay_max_abs=parity, candidate=candidate,
        accepted_adapter_updates=0, data_gate='contact_changes_allowed_finite_amplitude_only'))
    return path


def main():
    from omegaconf import OmegaConf
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--data', required=True)
    p.add_argument('--group', choices=('A','B','C','D'), required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--calibrate-only', action='store_true', help='只测训练标签H8/16/32梯度，不更新或运行审计')
    p.add_argument('--candidate-only', action='store_true', help='当前WM提出全新H32 Adam参数步，仅真实审计、不接受策略更新')
    p.add_argument('--wm-initialize', help='候选诊断使用的确切WM版本；不延长150次监督预算')
    first,_=p.parse_known_args()
    cfg=OmegaConf.to_container(OmegaConf.load(first.config), resolve=True)
    sys.path.insert(0,cfg['supermimic']['repo'])
    from isaaclab.app import add_launcher_args, launch_simulation
    add_launcher_args(p);args=p.parse_args()
    import torch
    from .application import build_engine, read_tasks, seed_all, direct_reference_config
    from .vector_bank import VectorBank
    from .vector_control import VectorControl
    from .vector_collector import VectorCollector
    from .vector_learning import GroupNormalizedCritic, fit_wm
    from .precision import PrecisionObjective
    from .reference_response import ReferenceResponseReplay, validate_response, reward_response_loss
    from .parameter_audit import parameter_update_audit
    from .update_diagnostics import parameter_identity
    from .vector_entry import json_write, append_json, compile_kinematics, gpu_usage
    from .checkpoint import shared_cpu_state
    from supermimic.bridge.bundle import atomic_save, file_digest
    from supermimic.bridge.streaming_env import StreamingPhysicalEnv
    from supermimic.env import make_sim
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    seed_all(cfg['training']['seed']);torch.set_num_threads(1)
    cfg=direct_reference_config(cfg)
    engine=build_engine(cfg,'cuda:0')
    if bool(args.wm_initialize) != bool(args.candidate_only):
        raise ValueError('指定已有WM仅用于明确的候选诊断，不能暗中接续监督预算')
    if args.wm_initialize:
        wm=torch.load(args.wm_initialize,map_location='cpu',weights_only=False,mmap=True)
        engine.dynamics.load_state_dict(wm['dynamics'])
        engine.dynamics.model_version=wm['wm_version']
    bank=VectorBank(read_tasks(cfg['tasks']),engine,'cuda:0')
    control=VectorControl(engine,bank);control.adapter.disable_root_xy=True
    scales=json.loads(Path(cfg['stage2a']['calibration']).read_text())['scales']
    control.precision=PrecisionObjective(scales,**cfg['stage2a']['precision'])
    control.soft_joint_limit_cost=False;control.soft_imagination_continuation=False
    critic=GroupNormalizedCritic(2835,control.spec).cuda();critic.requires_grad_(False)
    data=torch.load(args.data,map_location='cuda:0',weights_only=False)
    replay=ReferenceResponseReplay();replay.load_state_dict(data['random_response'],'cuda:0')
    if args.group=='D':
        more=ReferenceResponseReplay();more.load_state_dict(data['parameter_response'],'cuda:0')
        for b in more.buckets.values():
            replay.add(dict(origin='current_policy_real_reference_response',split='train',rejected={},
                            **{k:v for k,v in b.items() if k!='_sequence'}))
    windows=data['train_windows_d' if args.group=='D' else 'train_windows']
    validation=data['validation_windows'];vpairs=data['validation_pairs']
    if args.calibrate_only:
        compile_kinematics(engine,bank)
        result=calibrate_training_losses(engine.dynamics,control,windows,replay,cfg)
        json_write(out/'training_gradient_calibration.json',result)
        print(json.dumps(result),flush=True)
        return 0
    zero=copy.deepcopy(control.adapter.state_dict())
    if args.candidate_only:
        from .vector_learning import adapter_update
        adapter_optimizer=torch.optim.Adam(control.adapter.parameters(),lr=cfg['training']['adapter_lr'])
        candidate,_=adapter_update(control,engine.dynamics,critic,adapter_optimizer,data['source'],32,
            terminal_value=False,microbatch=32,probes=data['source'],checkpoint_steps=True)
        probe=copy.deepcopy(control.adapter.state_dict())
        atomic_save(shared_cpu_state(dict(adapter=probe,before=zero,dynamics=engine.dynamics.state_dict(),
            wm_version=engine.dynamics.model_version,optimizer=adapter_optimizer.state_dict(),config=cfg,
            candidate=candidate,state=dict(accepted_updates=0),
            boundary='new_H32_Adam_training_data_proposal_not_accepted_update')),out/'new_Adam_candidate.pt')
        json_write(out/'candidate_report.json',candidate)
        control.adapter.load_state_dict(zero)
    else:
        old=torch.load(cfg['wm_ablation']['direction_probe'],map_location='cpu',weights_only=False,mmap=True)
        probe=old['adapter']
    optimizer=torch.optim.Adam(engine.dynamics.model.net.parameters(),lr=cfg['training']['wm_lr'])
    criteria=json.loads(Path(cfg['physical_criteria']).read_text()) | cfg['training']['wm_quality']
    settings=cfg['stage2a']['long_horizon'];ablation=cfg['wm_ablation']
    one_weight=ablation['one_step_weight'] if args.group!='A' else 0.
    calibration=(json.loads(Path(ablation['calibration_file']).read_text()) if ablation.get('calibration_file')
                 else data['reward_gradient_calibration'])
    reward_weight=calibration['effective_weight'] if args.group in ('C','D') else 0.
    reference_weight=calibration.get('reference_weight',cfg['stage2a']['reference_response_weight'])
    data_sha=file_digest(args.data);frozen=engine.step.bundle.frozen_snapshot()
    json_write(out/'manifest.json',dict(config=cfg,group=args.group,data_sha256=data_sha,
        initial_WM=parameter_identity(engine.dynamics),one_step_weight=one_weight,reward_response_weight=reward_weight,
        reference_response_weight=reference_weight,training_gradient_calibration=calibration,
        shared_changes='remove_contact_veto_bounded_action_normalization_Huber_response',
        command=[sys.executable,*sys.argv],source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    compile_kinematics(engine,bank)
    collector=None;started=time.time()
    def save(label):
        atomic_save(shared_cpu_state(dict(dynamics=engine.dynamics.state_dict(),wm_version=engine.dynamics.model_version,
            adapter=zero,critic=critic.state_dict(),optimizer=optimizer.state_dict(),config=cfg,
            rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),data=args.data,data_sha256=data_sha,
            state=dict(wm_updates=engine.dynamics.model_version,accepted_updates=0),
            source='bounded_WM_ablation_not_formal_Adapter_training')),out/(label+'.pt'))
    def status(phase):
        json_write(out/'status.json',dict(pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),group=args.group,
            phase=phase,wm_updates=engine.dynamics.model_version,accepted_updates=0,
            num_audit_envs=16,data_physical_transitions=data['real_transitions'],
            new_audit_transitions=collector.real_transitions if collector else 0,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,gpu_process_mib=gpu_usage(),
            elapsed_seconds=time.time()-started,response_coverage=replay.coverage()),replace=True)
    def evaluate():
        with torch.no_grad():
            one,one_report=engine.dynamics.one_step_loss(validation)
            state_report=engine.dynamics.validate_real(validation,kin=control.runtime.kin)
            response_report=validate_response(engine.dynamics,vpairs,kin=control.runtime.kin)
            _,reward_report=reward_response_loss(engine.dynamics,control,vpairs,64,
                                                 scale=ablation['reward_response_scale'])
        control.adapter.load_state_dict(probe)
        direction=parameter_update_audit(collector,control,engine.dynamics,critic,zero,
                                         horizons=(8,32,64),criteria=criteria)
        control.adapter.load_state_dict(zero)
        result=dict(wm_updates=engine.dynamics.model_version,wm_sha256=parameter_identity(engine.dynamics),
            true_state_one_step=one_report,state_errors=state_report,reference_response=response_report,
            reward_response=reward_report,physical_parameter_direction=direction,
            accepted_adapter_updates=0,validation_label_sha256=data_sha,
            direction_scope=('new_H32_Adam_complete_parameter_probe_current_WM_zero_accepted_updates' if args.candidate_only
                else 'same_known_complete_parameter_probe_current_WM_new_Newton_replay_not_new_Adam_acceptance'))
        json_write(out/f'validation_{engine.dynamics.model_version:03d}.json',result)
        save(f'wm_{engine.dynamics.model_version:03d}')
        status('checkpoint_evaluated')
        print(json.dumps(dict(group=args.group,wm_updates=engine.dynamics.model_version,
            H32=direction['gates']['32'],H64=direction['gates']['64'])),flush=True)
    try:
        with launch_simulation(make_sim('cuda:0',args),args):
            env=StreamingPhysicalEnv(make_sim('cuda:0',args),16,engine.step.bundle)
            collector=VectorCollector(env,control,seed=cfg['vector']['sampling_seed'])
            collector.reset(perturb=False)
            evaluate()
            if args.candidate_only:
                engine.step.bundle.assert_unchanged(frozen)
                status('new_Adam_candidate_audited_zero_accepted_Adapter_updates')
                return 0
            for update in range(1,ablation['updates']+1):
                ids=torch.randint(len(windows['state']),(ablation['state_batch'],),device='cuda')
                long={k:v[ids] if isinstance(v,torch.Tensor) else v for k,v in windows.items()}
                ids=torch.randint(len(windows['state']),(cfg['training']['wm_batch'],),device='cuda')
                short={k:v[ids,:8] if isinstance(v,torch.Tensor) else v for k,v in windows.items()}
                horizon=(8,16,32)[(update-1)%3]
                pairs=replay.sample(ablation['response_batch'],horizon=horizon)
                report=fit_wm(engine.dynamics,optimizer,short,None,0.,reference_pairs=pairs,
                    reference_weight=reference_weight,reference_kin=control.runtime.kin,
                    response_settings=settings['response'],long_windows=long,long_weights=settings['state_weights'],
                    checkpoint_steps=True,one_step_weight=one_weight,reward_response_weight=reward_weight,
                    reward_control=control,reward_response_scale=ablation['reward_response_scale'])
                append_json(out/'wm.jsonl',report)
                if update%25==0:evaluate()
                elif update%5==0:status('bounded_WM_supervision')
            engine.step.bundle.assert_unchanged(frozen)
            status('completed_150_WM_updates_zero_Adapter_updates')
        return 0
    except BaseException:
        json_write(out/'failure.json',dict(traceback=traceback.format_exc(),accepted_updates=0))
        save('failure_last_WM');status('failed_saved')
        raise
