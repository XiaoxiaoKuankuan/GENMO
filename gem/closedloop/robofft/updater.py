"""固定 MC 的动作级 RoboFFT-D PPO 更新及全 rank 一致诊断。

输入是已全局标准化一次的优势、compact 最终动作和原 Critic cache。第一步更新前
固定全 rollout 的 eps/t/old_loss；每个全局动作 minibatch 先无梯度求所有 MC 新损失，
对动作目标求系数，再逐 MC 微批重算并反传。链式法则只除一次实际全局动作数，
保留 SUM 梯度语义、尾批和空 rank，不按去噪步或 world_size 再平均。

所有数值阶段使用原跨 rank 错误传播；失败由 training 的整轮 Actor/Critic/Adam/RNG
快照回滚。更新不调用 Gaussian PPO、Gaussian KL 或学习率回溯；可选 surrogate_kl
早停仅停止剩余小批次，不伪称真实 KL，也不把学习率自动缩小到接近零。
固定噪声输出探针只测新旧生成参考变化，不推进物理世界、不替代跟踪质量评估。
逐步诊断只覆盖当前全局动作小批与本轮固定的最多 64 个哨兵的去重并集；哨兵从
有效 manifest 等距确定，不消耗任何 RNG，不进入 PPO 反传。初始与整轮结束各做
一次全量诊断；可选早停使用明确标注范围的逐步 surrogate，不冒称全量或真实 KL。
"""
from __future__ import annotations

import time
import torch

from . import ALGORITHM, OBJECTIVE
from .records import MCRecord, validate_action
from .surrogate import action_objective
from ..dppo.execution_checks import policy_phase
from ..dppo.optimization_support import _owned, _local_phase, balanced_epoch_order
from ..dppo.actor_optimizer import actor_base_lr, validate_actor_optimizer


def _losses(policy, rows, cache, start, end):
    if _cache_signature(cache) != getattr(cache, 'robofft_input_signature', None):
        raise ValueError('Immutable RoboFFT action/condition/advantage cache was modified')
    for row in rows:
        row.mc_record.validate()
    records = [row.mc_record for row in rows]
    return policy.losses(cache.context(rows), cache.get('action_x0', rows),
        torch.stack([r.eps[start:end] for r in records]),
        torch.stack([r.sampling_indices[start:end] for r in records]),
        torch.stack([r.original_timesteps[start:end] for r in records]))


def _cache_signature(cache):
    return tuple((key,id(value),value._version) for key,value in sorted(cache.host.items()))


@torch.no_grad()
def prepare_mc(policy, rows, cache, generator, *, microbatch=8):
    """只允许为新 rollout 准备一次，旧数据在整轮 PPO 的所有 epoch 内固定。"""
    if any(row.mc_record is not None for row in rows):
        raise ValueError('MC records already fixed; old losses must not be regenerated')
    m = policy.robofft['mc_samples']
    chunk = policy.robofft['mc_microbatch']
    for start in range(0, len(rows), microbatch):
        selected = rows[start:start + microbatch]
        for row in selected:
            validate_action(row, policy)
        action = cache.get('action_x0', selected)
        mc = policy.draw_mc(action, generator)
        losses = []
        for k in range(0, m, chunk):
            losses.append(policy.losses(cache.context(selected), action,
                mc['eps'][:, k:k+chunk], mc['sampling_indices'][:, k:k+chunk],
                mc['original_timesteps'][:, k:k+chunk]).cpu())
        old = torch.cat(losses, dim=1)
        for i, row in enumerate(selected):
            row.mc_record = MCRecord(mc['eps'][i], mc['sampling_indices'][i],
                mc['original_timesteps'][i], old[i], torch.arange(m))
    cache.robofft_input_signature = _cache_signature(cache)
    return dict(algorithm=ALGORITHM, local_actions=len(rows), mc_samples=m,
        old_loss_fixed_before_any_optimizer=True, gaussian_probability_evaluations=0)


def _options(policy, clip):
    return dict(clip=clip, **{k: policy.robofft[k] for k in
        ('beta_neg', 'logprob_normalizer', 'logratio_alpha')})


def objective_batch(policy, rows, cache, *, clip, global_count, backward):
    """逐动作两遍流式反传；返回诊断总和，允许调用者作全局 SUM。"""
    m, chunk = policy.robofft['mc_samples'], policy.robofft['mc_microbatch']
    with torch.no_grad():
        current = torch.cat([_losses(policy, rows, cache, k, min(k+chunk, m))
                             for k in range(0, m, chunk)], dim=1)
    leaf = current.detach().requires_grad_(backward)
    old = torch.stack([r.mc_record.old_loss for r in rows]).to(leaf)
    metrics = action_objective(leaf, old, cache.get('advantages', rows), **_options(policy, clip))
    if backward:
        coefficients, = torch.autograd.grad(metrics['loss'].sum() / global_count, leaf)
        for k in range(0, m, chunk):
            loss = _losses(policy, rows, cache, k, min(k+chunk, m))
            (loss * coefficients[:, k:k+chunk]).sum().backward()
    return torch.stack([metrics[key].detach().sum() for key in METRICS])


METRICS = ('loss', 'ratio', 'surrogate_kl', 'ratio_clip_fraction', 'logratio_clamp_fraction',
           'new_logp_range_fraction', 'old_logp_range_fraction')


def diagnostics(policy, rows, cache, manifest, distributed, *, microbatch, clip,
                indices=None, scope='full_rollout'):
    with _local_phase(distributed, 'robofft_action_diagnostics'), policy_phase(policy):
        eligible = [i for i, item in enumerate(manifest) if item['valid'] and item['has_free']]
        selected = eligible if indices is None else list(indices)
        if len(set(selected)) != len(selected) or not set(selected).issubset(eligible):
            raise ValueError('RoboFFT diagnostics require unique valid free action indices')
        local = [rows[i] for _, i in _owned(selected, manifest, distributed)]
        total = torch.zeros(len(METRICS), device=cache.device, dtype=torch.float64)
        for start in range(0, len(local), microbatch):
            total += objective_batch(policy, local[start:start+microbatch], cache,
                                     clip=clip, global_count=max(len(selected), 1), backward=False)
    total = distributed.sum_tensor(total)
    if not bool(torch.isfinite(total).all()):
        raise FloatingPointError('Nonfinite global RoboFFT diagnostics')
    result = dict(zip(METRICS, (total / max(len(selected), 1)).cpu().tolist()))
    return dict(result, algorithm=ALGORITHM, covered_actions=len(selected), scope=scope,
                ratio_is_probability=False, kl_kind='surrogate_kl', gaussian_evaluations=0)


def fixed_sentinel_indices(selected, maximum=64):
    """从稳定的全局有效动作顺序等距取中点；整轮固定，独立于 epoch 重排和所有 RNG。"""
    if type(maximum) is not int or maximum < 1:
        raise ValueError('Diagnostic sentinel action count must be positive')
    count = min(len(selected), maximum)
    return [selected[(2*i+1)*len(selected)//(2*count)] for i in range(count)]


@torch.no_grad()
def output_probe(policy, rows, cache):
    selected = rows[:2]
    if not selected:
        return None
    generators = [torch.Generator(device=cache.device).manual_seed(80917+i) for i in range(len(selected))]
    trace = policy.sample_rollout(cache.context(selected), generator=generators)
    return {k: trace[k].detach().clone() for k in ('normalized', 'qpos', 'free_mask')}


def compare_outputs(before, after):
    if before is None:
        return dict(actions=0)
    free = before['free_mask']
    delta = (after['normalized'] - before['normalized'])[free].double()
    joint_before, joint_after = before['qpos'][..., 7:], after['qpos'][..., 7:]
    joint_free = free[..., 9:]
    change = (joint_after - joint_before)[joint_free].double()
    rms=lambda v: float(v.double().square().mean().sqrt()) if v.numel() else None
    maximum=lambda v: float(v.abs().max()) if v.numel() else None
    return dict(actions=len(before['normalized']), normalized_change_rms=rms(delta),
        normalized_change_max=maximum(delta), joint_change_rms_rad=rms(change),
        joint_change_max_rad=maximum(change),
        joint_amplitude_before_rad=rms(joint_before[joint_free]),
        joint_amplitude_after_rad=rms(joint_after[joint_free]),
        scope='same_conditions_fixed_independent_noise_reference_change_not_physics')


def actor_update(policy, optimizer, rows, targets, cache, manifest, *, distributed, generator,
                 ppo_epochs=1, minibatch_actions=8192, microbatch=8, clip=.01, grad_clip_norm=1.,
                 reserve_attempt=None, step_observer=None, diagnostic_sentinel_actions=64):
    if not targets.get('advantages_normalized') or targets.get('advantage_normalization_scope') != 'global_valid_upper_transitions':
        raise ValueError('RoboFFT requires advantages normalized globally exactly once')
    validate_actor_optimizer(policy.actor, optimizer)
    selected = [i for i, item in enumerate(manifest) if item['valid'] and item['has_free']]
    local = [rows[i] for _, i in _owned(selected, manifest, distributed)]
    with _local_phase(distributed, 'robofft_frozen_mc_validation'):
        for row in local:
            validate_action(row, policy)
            if row.mc_record is None:
                raise ValueError('Every action must have fixed MC and old loss before PPO')
    reports, stop_reason, orders = [], None, []
    sentinels = fixed_sentinel_indices(selected, diagnostic_sentinel_actions)
    initial = diagnostics(policy, rows, cache, manifest, distributed, microbatch=microbatch, clip=clip)
    with _local_phase(distributed, 'robofft_output_probe'), policy_phase(policy):
        before = output_probe(policy, local, cache)
    for epoch in range(ppo_epochs):
        order = balanced_epoch_order(selected, manifest, generator) if distributed.rank == 0 else None
        order = distributed.broadcast_object(order)
        orders.append(order)
        for start in range(0, len(order), minibatch_actions):
            indices = order[start:start+minibatch_actions]
            owned = [rows[i] for _, i in _owned(indices, manifest, distributed)]
            beginning = time.perf_counter()
            if reserve_attempt is not None:
                reserve_attempt()
            optimizer.zero_grad(set_to_none=True)
            with _local_phase(distributed, 'robofft_streaming_backward'), policy_phase(policy):
                for j in range(0, len(owned), microbatch):
                    objective_batch(policy, owned[j:j+microbatch], cache, clip=clip,
                                    global_count=len(indices), backward=True)
            distributed.sum_gradients(policy.actor)
            with _local_phase(distributed, 'robofft_clip_and_step'):
                norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in policy.actor.parameters() if p.requires_grad], grad_clip_norm, error_if_nonfinite=True)
                optimizer.step()
                if not all(bool(torch.isfinite(p).all()) for p in policy.actor.parameters()):
                    raise FloatingPointError('Nonfinite RoboFFT Actor update')
            # 诊断子集不进入上面的目标或全局动作分母；重叠哨兵只前向一次。
            diagnostic_indices = list(dict.fromkeys([*indices, *sentinels]))
            diagnostic_start = time.perf_counter()
            post = diagnostics(policy, rows, cache, manifest, distributed, microbatch=microbatch, clip=clip,
                               indices=diagnostic_indices, scope='current_minibatch_union_fixed_sentinel')
            post.update(minibatch_actions=len(indices), sentinel_actions=len(sentinels),
                        overlapping_actions=len(indices)+len(sentinels)-len(diagnostic_indices))
            report = dict(optimizer_step=len(reports)+1, epoch=epoch, global_action_count=len(indices),
                learning_rate=actor_base_lr(optimizer), gradient_norm=float(norm), post_step_surrogate=post,
                diagnostic_seconds=time.perf_counter()-diagnostic_start,
                wall_seconds=time.perf_counter()-beginning, algorithm=ALGORITHM)
            reports.append(report)
            if step_observer is not None:
                step_observer(report)
            threshold = policy.robofft.get('surrogate_kl_stop')
            if threshold is not None and post['surrogate_kl'] > threshold:
                stop_reason = 'surrogate_kl_early_stop'
                break
        if stop_reason:
            break
    with _local_phase(distributed, 'robofft_final_reference'), policy_phase(policy):
        reference = compare_outputs(before, output_probe(policy, local, cache))
    final = diagnostics(policy, rows, cache, manifest, distributed, microbatch=microbatch, clip=clip)
    return dict(algorithm=ALGORITHM, objective=OBJECTIVE, optimizer_steps=len(reports),
        accepted_optimizer_steps=len(reports), steps=reports, epochs=ppo_epochs,
        eligible_actions=len(selected), eligible_chains=len(selected), orders=orders, stop_reason=stop_reason,
        initial_surrogate=initial, final_surrogate=final, reference_change=distributed.all_gather_object(reference),
        diagnostic_policy=dict(sentinel_manifest_indices=sentinels,
            selection='evenly_spaced_valid_manifest_midpoints_no_rng',
            post_step_scope='current_minibatch_union_fixed_sentinel', final_scope='full_rollout',
            early_stop_scope='current_minibatch_union_fixed_sentinel'),
        acceptance_scope='finite_action_surrogate_no_Gaussian_KL',
        global_normalization='actual_valid_free_actions_once_then_SUM_gradients', mc_samples=policy.robofft['mc_samples'])
