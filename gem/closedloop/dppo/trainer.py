"""第九步 DPPO 的模型装配、固定回报目标和单卡／协同多卡有限优化。

Actor 从已验收 Stage1 权重初始化，独立 Critic 与优化器从零建立。采集策略固定，
价值目标在更新前一次性计算；DPPO 按内部每个去噪转移的联合概率比累计梯度，不对
最终动作文件套 PPO，也不反传穿过 GMT 或仿真。每轮默认只执行一次 Actor 优化步，
监督保持使用原 Stage1 loss 并独立记录监督更新次数；所有异常概率和梯度立即报错。

可选 distributed 协作者让各卡分担同一批上层转移的计算：每张卡按全局分母累计
局部损失，随后对梯度求和，因而保持单卡全局 batch 的更新语义。Critic 使用同一
全局 minibatch 的分片，Actor 按完整去噪链分片，KL 与概率统计恢复为完整全局顺序。
BC 只由 rank 0 执行一次；同步后的 PPO 梯度与该次 BC 梯度相加后统一裁剪、更新。
采集、运行目录和 checkpoint 仍由运行层统一管理，默认 distributed=None 保持旧路径。
"""
from __future__ import annotations

import math
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from gem.closedloop.checkpoint import load_stage1_checkpoint
from gem.closedloop.dppo.returns import compute_gae
from gem.closedloop.dppo.lr_calibration import calibrated_optimizer_step
from gem.closedloop.frozen_actor import _fingerprint
from gem.closedloop.training import build_stage1_actor, build_stage1_losses, batch_to_device


def load_actor(config):
    paths = config['paths']
    payload = torch.load(paths['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    train = OmegaConf.create(payload['config'])
    train.endecoder.stats_path = paths['stats']
    train.endecoder.kinematics_path = paths['kinematics']
    train.endecoder.allow_placeholder_stats = False
    data = OmegaConf.create(dict(qpos30_stats=dict(path=paths['stats']),
        dataset_defaults=dict(kinematics_path=paths['kinematics']),sample_contract=dict(history_steps=50),datasets={}))
    actor = build_stage1_actor(train,data)
    report = load_stage1_checkpoint(actor,payload)
    for key, value in actor.state_dict().items():
        if not torch.isfinite(value).all():
            raise FloatingPointError(f'Nonfinite checkpoint parameter: {key}')
    report.update(parameter_count=sum(p.numel() for p in actor.parameters()),
                  source_global_step=payload.get('global_step'),optimizer_restored=False,
                  global_step_restored=False)
    return actor.float().eval().requires_grad_(True).to(config['runtime']['genmo_device']), train, report


def batch_context(transitions, device, *, next_state=False):
    contexts = [t.next_context if next_state else t.context for t in transitions]
    return {key:torch.cat([c[key] for c in contexts],0).to(device) for key in contexts[0]}


def populate_values(transitions, critic, device):
    """使用当前冻结价值参数填入旧/下一价值，供逐条持久化和固定目标共同复用。"""
    critic.eval()
    with torch.no_grad():
        for row in transitions:
            context = {k:v.to(device) for k,v in row.context.items()}
            row.old_value = float(critic(context, torch.tensor([row.metadata['remaining_music_seconds']],device=device))[0])
            row.next_value = 0.
            if row.next_context is not None and not row.terminated:
                nxt = {k:v.to(device) for k,v in row.next_context.items()}
                row.next_value = float(critic(nxt,torch.tensor([row.metadata['next_remaining_music_seconds']],device=device))[0])


def fixed_targets(transitions, critic, device, *, gamma_upper=.99, lambda_upper=.95):
    populate_values(transitions, critic, device)
    continuation = [False]*len(transitions)
    for i in range(len(transitions)-1):
        a,b=transitions[i:i+2]
        continuation[i] = (a.transition_valid and b.transition_valid and not a.terminated and not a.truncated
            and all(a.identity[k]==b.identity[k] for k in ('backend_session_id','episode_id','policy_version'))
            and a.control_tick_end==b.control_tick_begin)
    return compute_gae(rewards=[t.rewards for t in transitions],values=[t.old_value for t in transitions],
        next_values=[t.next_value for t in transitions],executed_steps=[t.executed_control_steps for t in transitions],
        bootstrap_mask=[not t.terminated and t.next_context is not None for t in transitions],
        continuation_mask=continuation,valid=[t.transition_valid for t in transitions],
        event_rewards=[t.metadata.get('event_reward',0.) for t in transitions],
        gamma_upper=gamma_upper,lambda_upper=lambda_upper)


def critic_update(critic, optimizer, transitions, targets, *, steps=20, batch_size=32, generator=None,
                  grad_clip_norm=1., distributed=None):
    if type(steps) is not int or steps < 1 or type(batch_size) is not int or batch_size < 1:
        raise ValueError('Critic steps and batch size must be positive integers')
    if not math.isfinite(float(grad_clip_norm)) or grad_clip_norm <= 0:
        raise ValueError('Critic gradient clip must be finite and positive')
    device = next(critic.parameters()).device
    critic.train()
    target = targets['returns'].detach().float().to(device)
    losses, norms = [], []
    before = _fingerprint(critic)
    with torch.no_grad():
        initial_values = critic(batch_context(transitions, device), torch.tensor(
            [t.metadata['remaining_music_seconds'] for t in transitions], device=device))
        initial_mse = float((initial_values-target).square().mean())
        initial_variance = float(torch.var(target, unbiased=False))
        initial_ev = None if initial_variance <= 1e-12 else 1.-float(
            torch.var(target-initial_values, unbiased=False))/initial_variance
    for _ in range(steps):
        indices = torch.randperm(len(transitions),generator=generator)[:batch_size].tolist()
        if distributed is not None:
            indices = distributed.broadcast_object(indices if distributed.rank == 0 else None)
        local_indices = indices if distributed is None else indices[distributed.rank::distributed.world_size]
        optimizer.zero_grad(set_to_none=True)
        if local_indices:
            items = [transitions[i] for i in local_indices]
            value = critic(batch_context(items,device),torch.tensor(
                [t.metadata['remaining_music_seconds'] for t in items],device=device))
            squared_error = (value-target[local_indices]).square()
            loss = (.5*squared_error.mean() if distributed is None
                    else .5*squared_error.sum()/len(indices))
        else:
            loss = torch.zeros((), device=device)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite value loss')
        if local_indices:
            loss.backward()
        if distributed is not None:
            distributed.sum_gradients(critic)
            loss = distributed.sum_tensor(loss.detach())
        norms.append(float(torch.nn.utils.clip_grad_norm_(critic.parameters(),grad_clip_norm,error_if_nonfinite=True)))
        optimizer.step()
        losses.append(float(loss.detach()))
    critic.eval()
    with torch.no_grad():
        values = critic(batch_context(transitions,device),torch.tensor([t.metadata['remaining_music_seconds'] for t in transitions],device=device))
        variance = float(torch.var(target,unbiased=False))
        ev = None if variance <= 1e-12 else 1.-float(torch.var(target-values,unbiased=False))/variance
    return dict(losses=losses,gradient_norms=norms,parameters_changed=before!=_fingerprint(critic),
                initial_mse=initial_mse,initial_explained_variance=initial_ev,
                mse=float((values-target).square().mean()),explained_variance=ev,
                value_range=[float(values.min()),float(values.max())],return_range=[float(target.min()),float(target.max())])


def _validate_actor_transitions(policy, transitions):
    """拒绝不属于当前随机核的链；不允许用参数版本相同掩盖条件或掩码漂移。"""
    if not transitions:
        raise ValueError('Actor diagnostics require at least one transition')
    for item in transitions:
        if not item.transition_valid:
            raise ValueError('Invalid transitions cannot enter Actor optimization')
        if item.chain.shape != (policy.steps + 1, 120, 30):
            raise ValueError('Rollout chain differs from the configured denoising steps')
        if item.chain.dtype != torch.float32 or item.old_log_prob.dtype != torch.float64:
            raise TypeError('Rollout chain/log probability precision changed')
        if item.old_log_prob.shape != (policy.steps,):
            raise ValueError('Old log probabilities must match all denoising steps')
        expected = item.context['future_valid'][..., None] & ~item.context['known_qpos30_mask']
        if not torch.equal(item.free_mask.cpu(), expected[0].cpu()):
            raise ValueError('Stored free mask differs from the original conditions')
        trace = item.metadata['sampler_trace']
        if trace['kernel_config'] != policy.kernel_config:
            raise ValueError('Rollout and learner stochastic kernel configuration differs')
        if trace['timestep_map'].tolist() != list(policy.timestep_map):
            raise ValueError('Rollout timestep map differs from the learner')


def _ratio_report(log_ratios, *, clip):
    """按完整联合比值报告分位数，不裁剪 log_ratio 来制造有限结果。"""
    values = torch.as_tensor(log_ratios, dtype=torch.float64).detach().cpu()
    ratios = values.exp()
    if values.ndim != 2 or not torch.isfinite(values).all() or not torch.isfinite(ratios).all():
        raise FloatingPointError('Nonfinite probability ratio diagnostics')
    def summary(logs):
        current = logs.exp().flatten()
        points = torch.quantile(current, torch.tensor([0., .05, .5, .95, 1.], dtype=torch.float64))
        return dict(ratio_quantiles=dict(zip(('min', 'p05', 'p50', 'p95', 'max'), points.tolist())),
                    clip_fraction=float(((current < 1-clip) | (current > 1+clip)).double().mean()),
                    log_ratio_range=[float(logs.min()), float(logs.max())],
                    mean_log_ratio=float(logs.mean()))
    return {**summary(values), 'per_denoising_step': [dict(step_index=i, **summary(values[:, i]))
                                                     for i in range(values.shape[1])]}


def probability_check(policy, transitions, *, clip=.01):
    _validate_actor_transitions(policy, transitions)
    device = next(policy.actor.parameters()).device
    differences = []
    analytic_error = 0.
    with torch.no_grad():
        for item in transitions:
            context = {k:v.to(device) for k,v in item.context.items()}
            row = []
            for step in range(policy.steps):
                current = policy.evaluate_log_probs(context,item.chain[step:step+1].to(device),
                    item.chain[step+1:step+2].to(device),step)
                delta = current-item.old_log_prob[step].to(device)
                row.append(float(delta))
                trace=item.metadata['sampler_trace']
                old=torch.distributions.Normal(trace['old_means'][:,step].to(device).double(),
                                               trace['old_stds'][:,step].to(device).double())
                independent=old.log_prob(item.chain[step+1:step+2].to(device).double())
                independent=independent.masked_fill(~item.free_mask[None].to(device),0.).sum((-2,-1))
                error=(independent-item.old_log_prob[step].to(device)).abs()
                if not torch.isfinite(error).all():
                    raise FloatingPointError('Nonfinite stored Gaussian probability')
                analytic_error=max(analytic_error,float(error.max()))
            differences.append(row)
    values = torch.tensor(differences, dtype=torch.float64)
    if not torch.isfinite(values).all():
        raise FloatingPointError('Nonfinite zero-update probability difference')
    maximum = float(values.abs().max())
    ratio_error = float((values.exp()-1).abs().max())
    if maximum>1e-4 or ratio_error>1e-3 or analytic_error>1e-8 or not math.isfinite(maximum+ratio_error):
        raise FloatingPointError(f'Zero-update probability mismatch: {maximum}, {ratio_error}')
    return dict(max_abs_log_probability_difference=maximum,max_abs_ratio_minus_one=ratio_error,
                max_abs_independent_gaussian_difference=analytic_error,
                passed=True, **_ratio_report(values, clip=clip))


def analytic_kl(policy, transitions, *, distributed=None):
    _validate_actor_transitions(policy, transitions)
    values, per_dimension = [], []
    device = next(policy.actor.parameters()).device
    selected = [item for item in transitions if item.free_mask.any()]
    if not selected:
        raise ValueError('Analytic KL requires at least one free action coordinate')
    indices = list(range(len(selected)))
    if distributed is not None:
        indices = indices[distributed.rank::distributed.world_size]
    with torch.no_grad():
        for index in indices:
            item = selected[index]
            count = int(item.free_mask.sum())
            if not count:
                continue
            context={k:v.to(device) for k,v in item.context.items()}
            trace=item.metadata['sampler_trace']
            row, dimension_row = [], []
            for step in range(policy.steps):
                params=policy.transition_parameters(context,item.chain[step:step+1].to(device),step)
                old_mu=trace['old_means'][:,step].to(device).double()
                old_std=trace['old_stds'][:,step].to(device).double()
                new_mu=params['mean'].double(); new_std=params['std'].double()
                terms=(new_std/old_std).log()+(old_std.square()+(old_mu-new_mu).square())/(2*new_std.square())-.5
                mask=item.free_mask[None].to(device)
                joint=terms.masked_select(mask).sum()
                row.append(float(joint));dimension_row.append(float(joint)/count)
            values.append(row);per_dimension.append(dimension_row)
    result=torch.tensor(values,dtype=torch.float64).reshape(-1, policy.steps)
    dimensions=torch.tensor(per_dimension,dtype=torch.float32).reshape(-1, policy.steps)
    if distributed is not None:
        result = distributed.gather_rows(result, indices, len(selected))
        dimensions = distributed.gather_rows(dimensions, indices, len(selected))
    if not torch.isfinite(result).all():
        raise FloatingPointError('Nonfinite analytic joint KL')
    return dict(mean_joint_kl=float(result.mean()),p95_joint_kl=float(torch.quantile(result,.95)),
                max_joint_kl=float(result.max()),mean_per_dimension_kl=float(dimensions.mean()),
                joint_kl_scope='sum_free_coordinates_per_internal_transition_then_mean',
                mean_chain_joint_kl=float(result.sum(1).mean()),
                per_denoising_step=[dict(step_index=i, mean_joint_kl=float(result[:, i].mean()),
                    max_joint_kl=float(result[:, i].max()),
                    mean_per_dimension_kl=float(dimensions[:, i].mean()))
                    for i in range(policy.steps)])


def actor_update(policy, optimizer, transitions, targets, *, bc=None, bc_weight=.1, clip=.01,
                 gamma_denoising=.99, grad_clip_norm=1., learning_rate_candidates=None,
                 kl_limit=.02, reserve_attempt=None, calibration_progress=None, distributed=None):
    _validate_actor_transitions(policy, transitions)
    if not math.isfinite(clip) or not 0 < clip < 1 or not 0 < gamma_denoising <= 1:
        raise ValueError('Invalid PPO clip or denoising discount')
    if not math.isfinite(bc_weight) or bc_weight < 0 or not math.isfinite(grad_clip_norm) or grad_clip_norm <= 0:
        raise ValueError('Invalid BC weight or gradient bound')
    actor=policy.actor
    actor.eval()
    optimizer.zero_grad(set_to_none=True)
    device=next(actor.parameters()).device
    initial=_fingerprint(actor)
    valid = torch.as_tensor(targets.get('valid', torch.ones(len(transitions), dtype=torch.bool))).bool()
    advantages = torch.as_tensor(targets['advantages']).detach()
    if valid.shape != (len(transitions),) or advantages.shape != valid.shape or not torch.isfinite(advantages[valid]).all():
        raise ValueError('Fixed advantages must match the rollout and remain finite')
    selected = [(i, t) for i, t in enumerate(transitions) if valid[i] and t.free_mask.any()]
    if not selected:
        raise ValueError('No valid free action remains for Actor update')
    total=len(selected)*policy.steps
    loss_sum=0.; all_log_ratios=[]; step_losses=[0.]*policy.steps
    local_positions = list(range(len(selected)))
    if distributed is not None:
        local_positions = local_positions[distributed.rank::distributed.world_size]
    for position in local_positions:
        index, item = selected[position]
        context={k:v.to(device) for k,v in item.context.items()}
        row=[]
        for step in range(policy.steps):
            new=policy.evaluate_log_probs(context,item.chain[step:step+1].to(device),
                item.chain[step+1:step+2].to(device),step)
            log_ratio=new-item.old_log_prob[step].detach().to(device)
            ratio=log_ratio.exp()
            if not torch.isfinite(ratio).all():
                raise FloatingPointError('Nonfinite PPO ratio; no optimizer step performed')
            advantage=advantages[index].to(device)*(gamma_denoising**(policy.steps-1-step))
            objective=torch.minimum(ratio*advantage,ratio.clamp(1-clip,1+clip)*advantage)
            loss=-objective.mean()/total
            loss.backward()
            loss_sum+=float(loss.detach())
            step_losses[step]+=float(loss.detach())
            row.append(float(log_ratio.detach()))
        all_log_ratios.append(row)
        if (index+1)%8==0 and (distributed is None or distributed.rank == 0):
            print(f'[DPPO] accumulated {index+1}/{len(transitions)} upper transitions',flush=True)
    if distributed is not None:
        distributed.sum_gradients(actor)
        totals = distributed.sum_tensor(torch.tensor([loss_sum, *step_losses], device=device, dtype=torch.float64))
        loss_sum, *step_losses = totals.tolist()
        all_log_ratios = distributed.gather_rows(
            torch.tensor(all_log_ratios, dtype=torch.float64).reshape(-1, policy.steps),
            local_positions, len(selected))
    # 在加入BC之前核验，防止只有监督损失产生梯度却被误称DPPO成功。
    ppo_norm=torch.nn.utils.clip_grad_norm_(actor.parameters(),float('inf'),error_if_nonfinite=True)
    if not float(ppo_norm)>0:
        raise RuntimeError('DPPO-only Actor gradient is zero')
    ppo_modules={prefix:sum(float(p.grad.detach().double().square().sum()) for name,p in actor.named_parameters()
        if name.startswith(prefix) and p.grad is not None)**.5 for prefix in ('denoiser','history_encoder','prefix_encoder','music_embedder')}
    bc_report=None
    bc_enabled = bc is not None and bc_weight>0
    if distributed is not None:
        bc_enabled = distributed.broadcast_object(bc_enabled if distributed.rank == 0 else None)
    if bc_enabled:
        if distributed is None or distributed.rank == 0:
            bc_report=bc.backward(actor,weight=bc_weight)
        else:
            # rank 0 已持有完整 PPO 梯度，其余 rank 必须清空，避免第二次 SUM 乘卡数。
            optimizer.zero_grad(set_to_none=True)
        if distributed is not None:
            distributed.sum_gradients(actor)
            bc_report = distributed.broadcast_object(bc_report if distributed.rank == 0 else None)
    actor.eval()
    gradient=float(torch.nn.utils.clip_grad_norm_(actor.parameters(),grad_clip_norm,error_if_nonfinite=True))
    calibration = None
    if learning_rate_candidates:
        def progress(event):
            if distributed is None or distributed.rank == 0:
                print('[LR_CALIBRATION] '+str({key:value for key,value in event.items()
                    if key!='candidate'}), flush=True)
            if calibration_progress is not None:
                calibration_progress(event)
        calibration = calibrated_optimizer_step(actor, optimizer,
            evaluate_kl=lambda: (analytic_kl(policy, transitions) if distributed is None
                                 else analytic_kl(policy, transitions, distributed=distributed)),
            candidates=learning_rate_candidates,
            kl_limit=kl_limit, reserve_attempt=reserve_attempt, progress=progress)
    else:
        if reserve_attempt is not None:
            reserve_attempt()
        optimizer.step()
    changed=initial!=_fingerprint(actor)
    if not changed:
        raise RuntimeError('Actor optimizer step did not change any parameter')
    ratios = _ratio_report(all_log_ratios, clip=clip)
    for row, loss in zip(ratios['per_denoising_step'], step_losses):
        row.update(ppo_loss_contribution=loss, denoising_discount=gamma_denoising**(policy.steps-1-row['step_index']))
    return dict(ppo_loss=loss_sum,ppo_only_gradient_norm=float(ppo_norm),ppo_module_gradients=ppo_modules,
                total_gradient_norm=gradient,parameters_changed=changed,optimizer_steps=1,
                lr_calibration=calibration,
                ratio_range=[ratios['ratio_quantiles']['min'],ratios['ratio_quantiles']['max']],bc=bc_report,
                included_upper_transitions=len(selected), excluded_upper_transitions=len(transitions)-len(selected),
                ratio_scope='before_single_optimizer_step', **ratios)


class SupervisedAnchor:
    """原Stage1 Dataset/loss监督支路；独立随机状态和bc_update_steps可完整恢复。"""
    def __init__(self, config, actor, train_config):
        from gem.closedloop.stage1_dataset import BumiClosedLoopStage1Dataset
        data=OmegaConf.load(Path(config['paths']['genmo_repo'])/'configs/closedloop/stage1_dataset_server1_fourset_90505_v1.yaml')
        data.server1_example.root=config['stage9']['bc_data_root']
        data.qpos30_stats.path=config['paths']['stats']
        data.dataset_defaults.kinematics_path=config['paths']['kinematics']
        prefix=config['stage9'].get('bc_prefix',dict(min_frames=6,max_frames=18,zero_probability=.15))
        if (type(prefix['min_frames']) is not int or type(prefix['max_frames']) is not int
                or not 1 <= prefix['min_frames'] <= prefix['max_frames'] < 120
                or not 0 <= prefix['zero_probability'] < 1):
            raise ValueError('BC prefix range must preserve at least one free future frame')
        data.sample_contract.update(prefix_min_frames=prefix['min_frames'],prefix_max_frames=prefix['max_frames'],
                                    prefix_zero_probability=prefix['zero_probability'])
        self.prefix_contract=dict(prefix)
        self.datasets=[]
        for entry in data.datasets.train.values():
            opts=OmegaConf.to_container(entry,resolve=True)
            opts.pop('_target_',None)
            opts['kinematics_path']=config['paths']['kinematics']
            self.datasets.append(BumiClosedLoopStage1Dataset(**opts))
        self.losses=build_stage1_losses(actor,train_config).to(next(actor.parameters()).device)
        self.generator=torch.Generator().manual_seed(int(config['stage9']['seed'])+1001)
        self.bc_update_steps=0
        self.batch_size=int(config['stage9'].get('bc_batch',2))
        if self.batch_size < 1:
            raise ValueError('BC batch must be positive')
        self.torch_rng=torch.Generator().manual_seed(int(config['stage9']['seed'])+1002).get_state()
        self.numpy_rng=np.random.RandomState(int(config['stage9']['seed'])+1003).get_state()
        self.python_rng=random.Random(int(config['stage9']['seed'])+1004).getstate()
        device=next(actor.parameters()).device
        self.cuda_rng=(torch.Generator(device=device).manual_seed(int(config['stage9']['seed'])+1005).get_state()
                       if device.type=='cuda' else None)

    @contextmanager
    def _rng_scope(self, actor):
        """监督采样、数据窗口与dropout拥有独立且可恢复的随机状态。"""
        device=next(actor.parameters()).device
        cuda_devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
        old_numpy,old_python=np.random.get_state(),random.getstate()
        with torch.random.fork_rng(devices=cuda_devices):
            torch.set_rng_state(self.torch_rng)
            np.random.set_state(self.numpy_rng);random.setstate(self.python_rng)
            if cuda_devices:
                torch.cuda.set_rng_state(self.cuda_rng,cuda_devices[0])
            try:
                yield
            finally:
                self.torch_rng=torch.get_rng_state()
                self.numpy_rng=np.random.get_state();self.python_rng=random.getstate()
                if cuda_devices:
                    self.cuda_rng=torch.cuda.get_rng_state(cuda_devices[0])
                np.random.set_state(old_numpy);random.setstate(old_python)

    def backward(self, actor, weight):
        from gem.closedloop.stage1_dataset import collate_stage1_training_samples
        values=[]; samples=[]; micro_gradient_norms=[]; warmup_factors=[]
        actor.train()
        try:
            with self._rng_scope(actor), torch.autocast(device_type=next(actor.parameters()).device.type,enabled=False):
                for _ in range(self.batch_size):
                    source=int(torch.multinomial(torch.tensor([.2,.35,.25,.2]),1,generator=self.generator))
                    dataset=self.datasets[source]
                    index=int(torch.randint(len(dataset),(1,),generator=self.generator))
                    batch=batch_to_device(collate_stage1_training_samples([dataset[index]]),next(actor.parameters()).device)
                    prediction=actor.training_forward(batch)
                    loss,details=self.losses(batch,prediction['pred_x_start'],prediction['static_conf_logits'],global_step=self.bc_update_steps)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite Stage1 supervised retention loss')
                    squares=[]
                    hooks=[parameter.register_hook(lambda gradient: squares.append(gradient.detach().double().square().sum()))
                           for parameter in actor.parameters() if parameter.requires_grad]
                    try:
                        (weight*loss/self.batch_size).backward()
                    finally:
                        for hook in hooks:
                            hook.remove()
                    micro_gradient_norms.append(float(torch.stack(squares).sum().sqrt()) if squares else 0.)
                    warmup_factors.append({key:float(value.detach()) for key,value in details.items() if key.endswith('warmup_factor')})
                    values.append(float(loss.detach()));samples.append(batch['meta'])
            self.bc_update_steps+=1
        finally:
            actor.eval()
        return dict(loss=sum(values)/self.batch_size,weight=weight,bc_update_steps=self.bc_update_steps,samples=samples,
                    batch_size=self.batch_size,prefix_contract=self.prefix_contract,warmup_step=self.bc_update_steps-1,
                    warmup_origin='independent_bc_updates_from_zero',warmup_factors=warmup_factors,
                    weighted_gradient_norm_per_microbatch=micro_gradient_norms)

    def state_dict(self):
        return dict(generator=self.generator.get_state(),bc_update_steps=self.bc_update_steps,
                    torch_rng=self.torch_rng.clone(),numpy_rng=self.numpy_rng,python_rng=self.python_rng,
                    cuda_rng=self.cuda_rng.clone() if self.cuda_rng is not None else None,batch_size=self.batch_size)

    def load_state_dict(self,state):
        if state['batch_size']!=self.batch_size or int(state['bc_update_steps'])<0:
            raise ValueError('BC state batch or update count differs')
        self.generator.set_state(state['generator']);self.bc_update_steps=int(state['bc_update_steps'])
        self.torch_rng=state['torch_rng'].clone();self.numpy_rng=state['numpy_rng'];self.python_rng=state['python_rng']
        if (state['cuda_rng'] is None)!=(self.cuda_rng is None):
            raise ValueError('BC CUDA RNG topology differs')
        self.cuda_rng=state['cuda_rng'].clone() if state['cuda_rng'] is not None else None
