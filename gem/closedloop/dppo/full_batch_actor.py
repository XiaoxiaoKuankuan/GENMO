"""当前单次on-policy全批量joint-score Actor更新及原Stage1 BC。

每次更新使用完整真实rollout、GAE、BC、统一学习率和固定梯度KL回溯。score目标为：
每条链四步真实joint_log_ratio求和乘同一GAE优势，再除以实际链数；不取joint
指数、不做PPO clip、不额外乘去噪折扣，不将其伪称为联合概率PPO。

只接受更新前旧链概率一致的完整批量和一次optimizer提交。16384表示全局链容量，
128只是网络计算微批；空rank仍参加同一SUM和KL通信。BC按实际全局1024样本加权，
PPO/BC在本地累计后统一SUM、统一裁剪，所有Actor参数共享一个AdamW学习率组。
每次尝试保存参数与输出诊断；回溯失败完整恢复，消耗的计算预算不撤销。
PPO符号裁剪只在KL报告中作为反事实诊断，不提供旧坐标PPO训练分支。
"""
from __future__ import annotations

import copy
import time
from collections import Counter
import torch

from .actor_backtracking import backtracked_step
from .actor_optimizer import validate_actor_optimizer, set_actor_base_lr
from .actor_step_diagnostics import StepDiagnostics, output_probe, compare_probes, advantage_diagnostics
from .dense_policy import dense_precision_scope
from .execution_checks import policy_phase
from .gaussian_probability import gaussian_statistics
ACTOR_OBJECTIVE='joint_score_pg'
from .gradient_merge import merge_global_gradients, take_gradients
from .optimization_support import _local_phase, _owned, _parameters, _old_kernel
from .parallel_support import capture_local_rng, restore_local_rng, local_call
from .tensor_cache import ConditionGraphCache
from .updater import _eligible, final_kl

def joint_score_objective(statistics, advantages):
    """完整联合score差；不指数化，也不截断极高维联合log-ratio。"""
    return torch.where(statistics['free_count'] > 0,
        statistics['joint_log_ratio']*advantages.detach().to(statistics['joint_log_ratio']), 0.)


def prepare_full_batch_gradient(policy, optimizer, rows, targets, cache, manifest, controller, *,
        distributed, anchor, minibatch_chains=16384, microbatch=128,
        grad_clip_norm=1., bc_samples_per_rollout=1024, detailed_diagnostics=True,
        gradient_snr=False):
    if tuple(policy.trainable_steps) != (15,16,17,18):
        raise ValueError('Explicit four-step full-batch objective required')
    selected = _eligible(manifest)
    if not selected or len(selected) > minibatch_chains:
        raise ValueError('Single on-policy update requires all valid chains in one global minibatch')
    validate_actor_optimizer(policy.actor, optimizer)
    actor = policy.actor
    owned = _owned(selected, manifest, distributed)
    local = [rows[i] for _, i in owned]
    device = next(actor.parameters()).device
    probe_before = output_probe(policy, local, cache) if detailed_diagnostics else None
    observation = StepDiagnostics(actor) if detailed_diagnostics else None
    optimizer.zero_grad(set_to_none=True)
    actor.eval()
    pairs = [(i,s) for _,i in owned for s in policy.trainable_steps]
    totals = torch.zeros(5, device=device, dtype=torch.float64)
    max_log = torch.zeros((), device=device, dtype=torch.float64)
    started = time.perf_counter()
    with _local_phase(distributed, 'full_batch_actor_gradient'), policy_phase(policy):
        conditions = ConditionGraphCache(policy, cache, include_frozen=False)
        conditions.prime(local)
        for offset in range(0, len(pairs), microbatch):
            part = pairs[offset:offset+microbatch]
            samples, steps = [rows[i] for i,_ in part], [s for _,s in part]
            params, mask = _parameters(policy, samples, steps, device, cache, conditions)
            old_mean, old_std = _old_kernel(samples, steps, device, cache)
            stats = gaussian_statistics(cache.get('chain', samples, [s+1 for s in steps]),
                old_mean, old_std, params['mean'], params['std'], mask)
            advantages = cache.get('advantages', samples)
            loss = -joint_score_objective(stats, advantages).sum()/len(selected)
            loss.backward()
            with torch.no_grad():
                max_log = torch.maximum(max_log, stats['joint_log_ratio'].abs().max())
                totals += torch.stack((loss.detach(), stats['joint_kl'].sum(), stats['mean_coordinate_kl'].sum(),
                    stats['free_count'].sum(), loss.new_tensor(len(part))))
        conditions.backward()
    totals = distributed.sum_tensor(totals).cpu().tolist()
    maximum = max(distributed.all_gather_object(float(max_log)))
    # 整个score实验只允许在真实行为位置做一步，不能把已更新Actor当作on-policy起点。
    if maximum > 1e-4:
        optimizer.zero_grad(set_to_none=True)
        raise ValueError(f'Full-batch single-step starting policy differs from behavior: joint log ratio {maximum}')
    ppo = take_gradients(actor)
    with _local_phase(distributed, 'full_batch_actor_bc'):
        anchor.batch_size = bc_samples_per_rollout//distributed.world_size + (distributed.rank < bc_samples_per_rollout%distributed.world_size)
        with dense_precision_scope(policy.precision):
            bc_report = anchor.backward(actor, weight=anchor.batch_size/bc_samples_per_rollout)
        bc = take_gradients(actor)
    contributions = merge_global_gradients(actor, ppo, bc, controller, distributed=distributed,
        half_split=gradient_snr, component_observer=None if observation is None else observation.capture_component)
    with _local_phase(distributed, 'full_batch_actor_clip'):
        norm = float(torch.nn.utils.clip_grad_norm_([p for p in actor.parameters() if p.requires_grad],
            grad_clip_norm, error_if_nonfinite=True))
    factor = min(1., grad_clip_norm/(norm+1e-6))
    sources = Counter(meta['dataset_id'] for sample in bc_report.get('samples',[]) for meta in sample)
    reports = distributed.all_gather_object(dict(count=bc_report['batch_size'], loss=bc_report['loss'], sources=dict(sources)))
    all_sources = Counter()
    for r in reports:all_sources.update(r['sources'])
    report = dict(objective=ACTOR_OBJECTIVE, ppo_loss=totals[0], global_chain_count=len(selected),
        denominator=len(selected),
        internal_sample_visits=int(totals[4]), actual_free_coordinates=int(totals[3]),
        pre_step_mean_joint_kl=totals[1]/totals[4], pre_step_mean_coordinate_kl=totals[2]/totals[4],
        zero_update_max_joint_log_ratio=maximum, total_gradient_norm=norm, gradient_clip_factor=factor,
        gradient_contributions=contributions, bc_global_samples=sum(r['count'] for r in reports),
        bc_loss=sum(r['count']*r['loss'] for r in reports)/bc_samples_per_rollout,
        bc_samples_by_source=dict(all_sources), gradient_wall_seconds=max(distributed.all_gather_object(time.perf_counter()-started)),
        ppo_clip_applied=False, joint_ratio_exponentiated=False, denoising_discount=None)
    return report, observation, probe_before, local


def full_batch_actor_update(policy, optimizer, rows, targets, cache, manifest, controller, *,
        distributed, anchor, generator, backtracking, kl_limits, kl_microbatch=128,
        ppo_epochs=1, minibatch_chains=16384, microbatch=128, clip=.003,
        grad_clip_norm=1., reserve_attempt=None, bc_samples_per_rollout=1024, proposed_lr=None,
        detailed_diagnostics=True, snr=False):
    if ppo_epochs != 1:
        raise ValueError('Full-batch experiment permits exactly one on-policy update per rollout')
    policy._accepted_kl_receipt=None
    if proposed_lr is not None:set_actor_base_lr(optimizer,proposed_lr)
    control_before = copy.deepcopy(controller.state_dict())
    anchor_before = copy.deepcopy(anchor.state_dict())
    rng_before = capture_local_rng(dict(actor=generator))
    gradient, observation, before, local = prepare_full_batch_gradient(policy, optimizer, rows, targets, cache,
        manifest, controller, distributed=distributed, anchor=anchor,
        minibatch_chains=minibatch_chains, microbatch=microbatch,
        grad_clip_norm=grad_clip_norm, bc_samples_per_rollout=bc_samples_per_rollout,
        detailed_diagnostics=detailed_diagnostics, gradient_snr=snr)
    def evaluate():
        result = final_kl(policy, rows, cache, manifest, distributed=distributed, microbatch=kl_microbatch,
            clip=clip, objective_reduction='free_coordinate_mean')
        result['clip_is_counterfactual_dppo_diagnostic'] = True
        return result
    def observe():
        parameter = local_call(distributed,lambda:observation.finish(optimizer, controller.value, gradient['gradient_clip_factor']))
        after=local_call(distributed,lambda:output_probe(policy,local,cache))
        outputs = compare_probes(before, after, distributed)
        # 各rank模型更新必须相同；小型聚合结果差异即拒绝，不单独接受某张卡。
        parameters = distributed.all_gather_object(parameter)
        if any(p != parameters[0] for p in parameters):
            raise RuntimeError('Actor parameter diagnostics differ across ranks')
        return dict(parameters=parameter, outputs=outputs)
    transaction = backtracked_step(policy.actor, optimizer, controller, evaluate=evaluate, limits=kl_limits,
        distributed=distributed, generators=dict(actor=generator), max_attempts=backtracking['max_attempts'],
        factor=backtracking['factor'], reserve_attempt=reserve_attempt,
        observe=observe if detailed_diagnostics else None)
    if not transaction['accepted']:
        controller.load_state_dict(control_before)
        anchor.load_state_dict(anchor_before)
        restore_local_rng(rng_before, dict(actor=generator))
    accepted = int(transaction['accepted'])
    step = dict(gradient, optimizer_step=1, learning_rate=transaction['accepted_lr'], backtracking=transaction)
    result = dict(optimizer_steps=accepted, planned_optimizer_steps=1, steps=[step] if accepted else [],
        rejected_step=step if not accepted else None, step_rejected=not bool(accepted),
        attempted_optimizer_steps=transaction['attempt_count'], objective=ACTOR_OBJECTIVE,
        eligible_chains=gradient['global_chain_count'], applied_unique_chains=gradient['global_chain_count']*accepted,
        applied_chain_fraction=float(accepted), actual_global_bc_samples=gradient['bc_global_samples'],
        bc_samples_by_source=gradient['bc_samples_by_source'], internal_sample_visits=gradient['internal_sample_visits'],
        early_stop_reason=None if accepted else 'optimizer_step_backtracking_exhausted',
        advantage_diagnostics=advantage_diagnostics(rows, targets, distributed),
        old_statistics_fixed=True, final_trainable4_kl_pending=True,
        detailed_diagnostics=bool(detailed_diagnostics))
    if accepted:
        from .kl_reuse import seal_accepted_kl
        policy._accepted_kl_receipt = seal_accepted_kl(policy, cache, manifest,
            transaction['attempts'][-1]['kl'], clip=clip, reduction='free_coordinate_mean')
        result['complete_accepted_kl_available']=True
    return result
