"""第二阶段按本地采样分片工作的多 minibatch PPO／Critic 更新器。

本模块把一次 rollout、一次真正 optimizer.step 与一次显存微批明确分开。全局清单只
包含 owner_rank/local_index/valid/has_free，原始链留在所属进程；每个 epoch 对完整
上层链打乱，再按内部转移数分组。旧概率、旧核均值与方差、条件、自由掩码和固定优势
始终只读，后续 minibatch 必须用已经更新的 Actor 重新计算概率。局部 objective 求和
除以当前全局内部转移数，随后 SUM 梯度，不能再除以微批大小或 GPU 数量。

BC 每个真实 Actor 更新只由 rank 0 计算全局监督批，与同步后的 PPO 梯度组合再裁剪。
更新后的当前 minibatch KL 只用于软停止；它不能证明全 rollout 满足硬限制。调用方
必须在发布策略前调用 analytic_kl_local 做完整检查，并负责整轮状态快照／回滚。
本模块不保存 checkpoint、不启动环境、不改变学习率，也不修改 GMT 或采样链。

保留真实 FP64 联合概率作为诊断，free_coordinate_mean 仅是显式的替代优化目标。
链 KL 是旧策略中间状态上的条件路径 KL Monte Carlo 估计，不冒充最终动作分布 KL。
"""
from __future__ import annotations

import math
from collections import deque
from contextlib import contextmanager

import torch

from gem.closedloop.dppo.policy import masked_joint_log_prob
from .performance import measure, profiled
from .execution_checks import policy_phase, require_tensor, checked_policy_phase
from .tensor_cache import ConditionGraphCache, KLResultCache


@contextmanager
def _local_phase(distributed, phase):
    """在下一次张量 collective 前同步本地异常；块内严格禁止任何 collective。

    一卡的有限性校验、前反向 OOM 或 Adam 分配失败不能让其他卡进入下一次梯度归约。
    整轮回滚仍由外层事务处理，此处只传播一致失败，不擅自回退已消耗的预算。
    """
    error = None
    try:
        with measure(f'compute.{phase}', gpu=True):
            yield
    except BaseException as caught:
        error = caught
    if distributed is None:
        if error is not None:
            raise error
        return
    local = None if error is None else dict(rank=distributed.rank, type=type(error).__name__, message=str(error))
    with measure(f'wait.{phase}'):
        failures = [item for item in distributed.all_gather_object(local) if item is not None]
    if failures:
        raise RuntimeError(f'Cooperative local phase failed ({phase}): {failures}') from error


def _rank(distributed):
    return 0 if distributed is None else distributed.rank


def _manifest(transitions, targets, distributed, global_manifest):
    rank = _rank(distributed)
    valid = torch.as_tensor(targets.get('valid', [row.transition_valid for row in transitions])).bool()
    if valid.shape != (len(transitions),):
        raise ValueError('Targets must align with the local transition shard')
    if global_manifest is None:
        world = 1 if distributed is None else distributed.world_size
        result = [dict(owner_rank=i % world, local_index=i, valid=bool(valid[i]),
                       has_free=bool(row.free_mask.any())) for i, row in enumerate(transitions)]
    else:
        result = list(global_manifest)
    seen = set()
    for item in result:
        owner, index = item['owner_rank'], item['local_index']
        world = 1 if distributed is None else distributed.world_size
        if type(owner) is not int or not 0 <= owner < world or type(index) is not int or index < 0:
            raise ValueError('Invalid owner/local index in global rollout manifest')
        if (owner, index) in seen:
            raise ValueError('Duplicate owner/local index in global rollout manifest')
        seen.add((owner, index))
        if owner == rank:
            if index >= len(transitions) or bool(valid[index]) != bool(item['valid']):
                raise ValueError('Global rollout manifest differs from local validity')
            if bool(transitions[index].free_mask.any()) != bool(item['has_free']):
                raise ValueError('Global rollout manifest differs from local free mask')
    if global_manifest is not None:
        actual_local_indices = {index for owner, index in seen if owner == rank}
        if actual_local_indices != set(range(len(transitions))):
            raise ValueError('Global rollout manifest must cover the complete local shard exactly once')
    return result


def _owned(global_indices, manifest, distributed):
    return [(position, manifest[index]['local_index']) for position, index in enumerate(global_indices)
            if manifest[index]['owner_rank'] == _rank(distributed)]


def balanced_epoch_order(selected, manifest, generator):
    """每rank独立打乱后轮询交织；完整覆盖、无丢弃/重复，随机状态可恢复。"""
    buckets = {}
    for index in selected:
        buckets.setdefault(manifest[index]['owner_rank'], []).append(index)
    for rank, values in buckets.items():
        buckets[rank] = deque(values[i] for i in torch.randperm(len(values), generator=generator).tolist())
    ranks = sorted(buckets)
    ranks = [ranks[i] for i in torch.randperm(len(ranks), generator=generator).tolist()]
    result = []
    while len(result) < len(selected):
        for rank in ranks:
            if buckets[rank]:
                result.append(buckets[rank].popleft())
    return result


@profiled('learning.prepare_context', gpu=True)
def _context(rows, device):
    return {key: torch.cat([row.context[key] for row in rows], 0).to(device) for key in rows[0].context}


@profiled('learning.parameters', gpu=True)
def _parameters(policy, rows, steps, device, cache=None, conditions=None):
    context = _context(rows, device) if cache is None else cache.context(rows)
    state = (torch.stack([row.chain[step] for row, step in zip(rows, steps)]).to(device)
             if cache is None else cache.get('chain', rows, steps))
    indices = torch.tensor(steps, device=device, dtype=torch.long) if cache is None else cache.step_indices(steps)
    if hasattr(policy, 'prepare_conditions'):
        prepared = policy.prepare_conditions(context) if conditions is None else conditions.prepare(rows, context)
        result = policy.transition_parameters(context, state, indices, prepared=prepared)
    else:
        result = policy.transition_parameters(context, state, indices)
    mask = torch.stack([row.free_mask for row in rows]).to(device) if cache is None else cache.get('free_mask', rows)
    if 'free_mask' in result:
        require_tensor((result['free_mask'] == mask).all(), 'Current policy changed the fixed rollout free-coordinate mask')
    return result, mask


def _old_kernel(rows, steps, device, cache=None):
    if cache is not None:
        return cache.get('old_means', rows, steps).double(), cache.get('old_stds', rows, steps).double()
    means = torch.cat([row.metadata['sampler_trace']['old_means'][:, step]
                       for row, step in zip(rows, steps)], 0).to(device).double()
    stds = torch.cat([row.metadata['sampler_trace']['old_stds'][:, step]
                      for row, step in zip(rows, steps)], 0).to(device).double()
    return means, stds


def _joint_kl(parameters, mask, old_mean, old_std):
    mean, std = parameters['mean'].double(), parameters['std'].double()
    terms = (std / old_std).log() + (old_std.square() + (old_mean - mean).square()) / (2 * std.square()) - .5
    result = terms.masked_fill(~mask, 0.).sum((-2, -1))
    require_tensor(torch.isfinite(result).all() & (result >= -1e-10).all(),
                   'Nonfinite or negative conditional Gaussian KL', FloatingPointError)
    return result.clamp_min(0.)


def _kl_report(values, free_counts):
    if values.ndim != 2 or not values.numel() or not torch.isfinite(values).all():
        raise ValueError('KL report requires finite nonempty [upper, denoising_step] values')
    values = values.double()
    dimensions = values / free_counts[:, None].double()
    chain = values.sum(1)
    return dict(mean_joint_kl=float(values.mean()), mean_internal_joint_kl=float(values.mean()),
        p95_joint_kl=float(torch.quantile(values, .95)), max_joint_kl=float(values.max()),
        max_internal_joint_kl=float(values.max()), max_step_mean_joint_kl=float(values.mean(0).max()),
        mean_chain_joint_kl=float(chain.mean()), max_chain_joint_kl=float(chain.max()),
        mean_per_dimension_kl=float(dimensions.mean()),
        joint_kl_scope='sum_free_coordinates_per_internal_transition_then_mean',
        chain_kl_scope='old_path_monte_carlo_conditional_chain_kl',
        per_denoising_step=[dict(step_index=i, mean_joint_kl=float(values[:, i].mean()),
            max_joint_kl=float(values[:, i].max()), mean_per_dimension_kl=float(dimensions[:, i].mean()))
            for i in range(values.shape[1])], included_upper_transitions=len(values))


@torch.no_grad()
@profiled('learning.analytic_kl', gpu=True)
def analytic_kl_local(policy, transitions, *, global_manifest=None, distributed=None,
                      denoising_microbatch=4, global_indices=None, tensor_cache=None,
                      reuse_cache=None, return_cache=False):
    """按各卡本地链计算精确条件高斯 KL；跨卡仅收集小型逐步 KL 矩阵。"""
    from gem.closedloop.dppo.trainer import _validate_actor_transitions
    if type(denoising_microbatch) is not int or denoising_microbatch < 1:
        raise ValueError('Denoising microbatch must be a positive integer')
    with _local_phase(distributed, 'analytic_kl_local_forward'), policy_phase(policy):
        manifest = _manifest(transitions, {}, distributed, global_manifest)
        selected = ([i for i, item in enumerate(manifest) if item['valid'] and item['has_free']]
                    if global_indices is None else list(global_indices))
        if not selected or any(not manifest[i]['valid'] or not manifest[i]['has_free'] for i in selected):
            raise ValueError('Analytic KL requires valid transitions with free coordinates')
        owned = _owned(selected, manifest, distributed)
        local_rows = [transitions[index] for _, index in owned]
        if local_rows:
            _validate_actor_transitions(policy, local_rows)
        device = next(policy.actor.parameters()).device
        output = torch.zeros((len(owned), policy.steps), dtype=torch.float64, device=device)
        # 与 KL 共用本次前向；不为噪声核和输出位移日志再遍历整条链。
        diagnostics = torch.zeros((len(owned), 4, policy.steps), dtype=torch.float64, device=device)
        counts = torch.tensor([int(row.free_mask.sum()) for row in local_rows], device=device, dtype=torch.float64)
        if reuse_cache is not None:
            reuse_cache.validate(policy, transitions, manifest)
        reused = set()
        for index, (position, _) in enumerate(owned):
            global_index = selected[position]
            if reuse_cache is not None and global_index in reuse_cache.records:
                output[index], diagnostics[index], counts[index] = reuse_cache.records[global_index]
                reused.add(index)
        flat = [(index, step) for index in range(len(local_rows)) if index not in reused for step in range(policy.steps)]
        conditions = ConditionGraphCache(policy, tensor_cache) if tensor_cache is not None and hasattr(policy, 'prepare_conditions') else None
        if conditions is not None and getattr(policy, 'defer_checks', False):
            conditions.prime([local_rows[i] for i in range(len(local_rows)) if i not in reused])
        for start in range(0, len(flat), denoising_microbatch):
            chunk = flat[start:start + denoising_microbatch]
            rows = [local_rows[index] for index, _ in chunk]
            steps = [step for _, step in chunk]
            parameters, mask = _parameters(policy, rows, steps, device, tensor_cache, conditions)
            old_mean, old_std = _old_kernel(rows, steps, device, tensor_cache)
            values = _joint_kl(parameters, mask, old_mean, old_std)
            count = mask.sum((-2, -1))
            shift = ((parameters['mean'].double() - old_mean).square().masked_fill(~mask, 0.).sum((-2, -1)) / count).sqrt()
            current_std = parameters['std'].double().reshape(len(rows), -1).mean(1)
            base_std = parameters.get('base_std', parameters['std']).double().reshape(len(rows), -1).mean(1)
            noise = torch.stack((old_std.reshape(len(rows), -1).mean(1), current_std, base_std, shift), 1)
            # 批量散写小型KL/噪声矩阵，避免每个内部样本启动五个小拷贝kernel。
            chain_index = torch.tensor([index for index, _ in chunk], device=device)
            step_index = torch.tensor(steps, device=device) if tensor_cache is None else tensor_cache.step_indices(steps)
            output[chain_index, step_index] = values
            diagnostics[chain_index, :, step_index] = noise
        cache = (KLResultCache(policy, transitions, {selected[position]: (output[i], diagnostics[i], counts[i])
                 for i, (position, _) in enumerate(owned)}, manifest) if return_cache else None)
    if distributed is not None:
        combined = distributed.gather_rows(torch.cat((output, diagnostics.reshape(len(owned), 4 * policy.steps), counts[:, None]), 1),
                                            [position for position, _ in owned], len(selected))
        output, counts = combined[:, :policy.steps], combined[:, -1]
        diagnostics = combined[:, policy.steps:-1].reshape(len(selected), 4, policy.steps)
    report = _kl_report(output.cpu(), counts.cpu())
    diagnostics = diagnostics.cpu()
    for step, entry in enumerate(report['per_denoising_step']):
        entry.update(mean_old_std=float(diagnostics[:, 0, step].mean()),
                     mean_std=float(diagnostics[:, 1, step].mean()), mean_base_std=float(diagnostics[:, 2, step].mean()),
                     mean_normalized_mean_shift_rms=float(diagnostics[:, 3, step].mean()))
    report['free_coordinate_count'] = dict(min=int(counts.min()), max=int(counts.max()), mean=float(counts.mean()))
    reuse_count = torch.tensor(len(reused), device=device, dtype=torch.long)
    if distributed is not None:
        reuse_count = distributed.sum_tensor(reuse_count)
    report['reused_upper_transitions'] = int(reuse_count)
    report['fresh_internal_forwards'] = (len(selected)-int(reuse_count))*policy.steps
    return (report, cache) if return_cache else report


@torch.no_grad()
@profiled('learning.probability_check', gpu=True)
def probability_check_local(policy, transitions, *, global_manifest=None, distributed=None,
                            denoising_microbatch=4, global_indices=None, tensor_cache=None):
    """完整零更新检查仅用于启动／周期诊断；原始概率、ratio 和独立公式门槛不放宽。"""
    from gem.closedloop.dppo.trainer import _validate_actor_transitions
    if type(denoising_microbatch) is not int or denoising_microbatch < 1:
        raise ValueError('Denoising microbatch must be a positive integer')
    with _local_phase(distributed, 'initial_probability_local_forward'):
        manifest = _manifest(transitions, {}, distributed, global_manifest)
        selected = ([i for i, item in enumerate(manifest) if item['valid'] and item['has_free']]
                    if global_indices is None else list(global_indices))
        if (not selected or len(set(selected)) != len(selected)
                or any(not manifest[i]['valid'] or not manifest[i]['has_free'] for i in selected)):
            raise ValueError('Zero-update probability check requires free valid transitions')
        owned = _owned(selected, manifest, distributed)
        rows = [transitions[index] for _, index in owned]
        if rows:
            _validate_actor_transitions(policy, rows)
        device = next(policy.actor.parameters()).device
        checks = torch.zeros((len(rows), 3), dtype=torch.float64, device=device)
        flat = [(index, step) for index in range(len(rows)) for step in range(policy.steps)]
        conditions = ConditionGraphCache(policy, tensor_cache) if tensor_cache is not None and hasattr(policy, 'prepare_conditions') else None
        for start in range(0, len(flat), denoising_microbatch):
            chunk = flat[start:start + denoising_microbatch]
            samples, steps = [rows[i] for i, _ in chunk], [step for _, step in chunk]
            parameters, mask = _parameters(policy, samples, steps, device, tensor_cache, conditions)
            observed = (torch.stack([row.chain[step + 1] for row, step in zip(samples, steps)]).to(device)
                        if tensor_cache is None else tensor_cache.get('chain', samples, [step+1 for step in steps]))
            stored = (torch.stack([row.old_log_prob[step] for row, step in zip(samples, steps)]).to(device)
                      if tensor_cache is None else tensor_cache.get('old_log_prob', samples, steps))
            current = masked_joint_log_prob(observed, parameters['mean'], parameters['std'], mask)
            old_mean, old_std = _old_kernel(samples, steps, device, tensor_cache)
            independent = torch.distributions.Normal(old_mean, old_std).log_prob(observed.double())
            independent = independent.masked_fill(~mask, 0.).sum((-2, -1))
            batch = torch.stack(((current - stored).abs(), ((current - stored).exp() - 1).abs(),
                                 (independent - stored).abs()), 1)
            for offset, (index, _) in enumerate(chunk):
                checks[index] = torch.maximum(checks[index], batch[offset])
    if distributed is not None:
        checks = distributed.gather_rows(checks, [position for position, _ in owned], len(selected))
    maximum = checks.amax(0).cpu()
    if not torch.isfinite(maximum).all() or (maximum > torch.tensor([1e-4, 1e-3, 1e-8])).any():
        raise FloatingPointError(f'Zero-update probability mismatch: {maximum.tolist()}')
    return dict(passed=True, max_abs_log_probability_difference=float(maximum[0]),
                max_abs_ratio_minus_one=float(maximum[1]), max_abs_independent_gaussian_difference=float(maximum[2]))


def _gradient_pair_report(actor, ppo_gradients, *, module_details=True):
    """在同一参数版本上分离 PPO 与加权 BC，公共分支与独立头分开报告。"""
    device = next(actor.parameters()).device
    groups = {key: torch.zeros(5, dtype=torch.float64, device=device) for key in ('all', 'shared')}
    for name, parameter in actor.named_parameters():
        if not parameter.requires_grad:
            continue
        old = ppo_gradients.get(name)
        merged = parameter.grad
        if old is None and merged is None:
            continue
        ppo = torch.zeros_like(merged) if old is None else old
        total = torch.zeros_like(ppo) if merged is None else merged
        bc = total - ppo
        ppo64, bc64, total64 = ppo.double(), bc.double(), total.double()
        numbers = torch.stack((ppo64.square().sum(), bc64.square().sum(),
                               (ppo64 * bc64).sum(), total64.square().sum(),
                               ppo.new_tensor(1., dtype=torch.float64)))
        labels = ['all', name.split('.')[0]] if module_details else ['all']
        if old is not None:
            groups['shared'] += numbers*(numbers[1] > 0)
        if 'contact' in name or 'static_conf' in name:
            labels.append('contact_head')
        for label in set(labels):
            if label not in groups:
                groups[label] = torch.zeros_like(numbers)
            groups[label] += numbers
    result = {}
    for group, (ppo2, bc2, dot, merged2, count) in zip(groups, torch.stack(list(groups.values())).cpu().tolist()):
        result[group] = dict(ppo_norm=math.sqrt(ppo2), weighted_bc_norm=math.sqrt(bc2),
            cosine=None if ppo2 <= 0 or bc2 <= 0 else max(-1., min(1., dot / math.sqrt(ppo2 * bc2))),
            merged_norm=math.sqrt(merged2), contributing_parameter_tensors=int(count))
    return result


def actor_update_v2(policy, optimizer, transitions, targets, *, global_manifest=None,
                    distributed=None, ppo_epochs=2, actor_minibatch_internal_transitions=1600,
                    denoising_microbatch=4, max_optimizer_steps=4, generator=None, epoch_orders=None,
                    bc=None, bc_weight=.1, clip=.01, gamma_denoising=.99, grad_clip_norm=1.,
                    soft_kl_limit=.015, objective_logprob_reduction='joint_sum',
                    reserve_attempt=None, verify_initial_probability=False,
                    gradient_diagnostics=True, step_callback=None, tensor_cache=None,
                    balanced_minibatches=False, gradient_module_details=True, kl_cache_sink=None,
                    kl_check_mode='post_step_full', gradient_observer=None):
    """执行真实多次 PPO 参数更新；完整硬 KL 与整轮回滚明确由外层事务负责。"""
    from gem.closedloop.dppo.trainer import _validate_actor_transitions
    for name, value in (('ppo_epochs', ppo_epochs), ('actor minibatch', actor_minibatch_internal_transitions),
                        ('microbatch', denoising_microbatch), ('max optimizer steps', max_optimizer_steps)):
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if actor_minibatch_internal_transitions % policy.steps:
        raise ValueError('Actor optimizer minibatch must contain complete denoising chains')
    if not 0 < clip < 1 or not 0 < gamma_denoising <= 1 or not math.isfinite(grad_clip_norm) or grad_clip_norm <= 0:
        raise ValueError('Invalid PPO clipping, denoising discount, or gradient clipping')
    if not math.isfinite(bc_weight) or bc_weight < 0 or soft_kl_limit is not None and (not math.isfinite(soft_kl_limit) or soft_kl_limit <= 0):
        raise ValueError('Invalid BC weight or soft KL limit')
    if objective_logprob_reduction not in ('joint_sum', 'free_coordinate_mean'):
        raise ValueError('Unknown PPO objective probability reduction')
    if kl_check_mode not in ('post_step_full', 'pre_step_plus_final'):
        raise ValueError('Unknown KL check mode')
    if kl_check_mode == 'pre_step_plus_final' and kl_cache_sink is not None:
        kl_cache_sink.clear()
    discarded = None
    actor = policy.actor
    actor.eval()
    if any(not parameter.requires_grad for group in optimizer.param_groups for parameter in group['params']):
        raise ValueError('New Actor optimizer must exclude frozen architecture parameters')
    with _local_phase(distributed, 'actor_local_shard_validation'):
        manifest = _manifest(transitions, targets, distributed, global_manifest)
        selected = [i for i, item in enumerate(manifest) if item['valid'] and item['has_free']]
        if not selected:
            raise ValueError('No free valid upper transition remains for Actor optimization')
        if epoch_orders is not None and len(epoch_orders) != ppo_epochs:
            raise ValueError('Explicit epoch plans must match ppo_epochs')
        local_rows = [transitions[index] for _, index in _owned(selected, manifest, distributed)]
        if local_rows:
            _validate_actor_transitions(policy, local_rows)
        advantages = torch.as_tensor(targets['advantages'], dtype=torch.float64).detach()
        if advantages.shape != (len(transitions),) or any(not torch.isfinite(advantages[index]) for _, index in _owned(selected, manifest, distributed)):
            raise ValueError('Actor requires fixed finite advantages aligned with local transitions')
    probability = (probability_check_local(policy, transitions, global_manifest=global_manifest, distributed=distributed,
                    denoising_microbatch=denoising_microbatch, tensor_cache=tensor_cache) if verify_initial_probability else None)
    device = next(actor.parameters()).device
    upper_batch = actor_minibatch_internal_transitions // policy.steps
    reports, orders = [], []
    stopped = None
    for epoch in range(ppo_epochs):
        if epoch_orders is not None:
            order = list(epoch_orders[epoch])
        elif distributed is None or _rank(distributed) == 0:
            order = (balanced_epoch_order(selected, manifest, generator) if balanced_minibatches else
                     [selected[i] for i in torch.randperm(len(selected), generator=generator).tolist()])
        else:
            order = None
        if distributed is not None:
            order = distributed.broadcast_object(order if _rank(distributed) == 0 else None)
        if len(order) != len(selected) or set(order) != set(selected):
            raise ValueError('Every PPO epoch must cover every eligible upper chain exactly once')
        orders.append(order)
        for offset in range(0, len(order), upper_batch):
            if len(reports) >= max_optimizer_steps:
                stopped = 'max_optimizer_steps'
                break
            indices = order[offset:offset + upper_batch]
            owned = _owned(indices, manifest, distributed)
            pairs = [(index, step) for _, index in owned for step in range(policy.steps)]
            denominator = len(indices) * policy.steps
            optimizer.zero_grad(set_to_none=True)
            summary = torch.zeros(7, dtype=torch.float64, device=device)
            with _local_phase(distributed, 'actor_minibatch_forward_backward'), policy_phase(policy):
                from .batch_execution import ROW_BMM, MicrobatchGradientAccumulator
                gradient_sum = (MicrobatchGradientAccumulator(actor)
                                if getattr(policy,'numerical_layout',None)==ROW_BMM else None)
                conditions = ConditionGraphCache(policy, tensor_cache) if tensor_cache is not None and hasattr(policy, 'prepare_conditions') else None
                if conditions is not None and getattr(policy, 'defer_checks', False):
                    conditions.prime([transitions[index] for _, index in owned])
                for start in range(0, len(pairs), denoising_microbatch):
                    chunk = pairs[start:start + denoising_microbatch]
                    rows, steps = [transitions[index] for index, _ in chunk], [step for _, step in chunk]
                    parameters, mask = _parameters(policy, rows, steps, device, tensor_cache, conditions)
                    observed = (torch.stack([row.chain[step + 1] for row, step in zip(rows, steps)]).to(device)
                        if tensor_cache is None else tensor_cache.get('chain', rows, [step+1 for step in steps]))
                    new = masked_joint_log_prob(observed, parameters['mean'], parameters['std'], mask)
                    old = (torch.stack([row.old_log_prob[step] for row, step in zip(rows, steps)]).to(device).detach()
                        if tensor_cache is None else tensor_cache.get('old_log_prob', rows, steps))
                    log_ratio = new - old
                    if objective_logprob_reduction == 'free_coordinate_mean':
                        log_ratio = log_ratio / mask.sum((-2, -1))
                    ratio = log_ratio.exp()
                    require_tensor(torch.isfinite(ratio).all(), 'Nonfinite PPO ratio before optimizer step', FloatingPointError)
                    advantage = (torch.stack([advantages[index] * gamma_denoising ** (policy.steps - 1 - step)
                                             for index, step in chunk]).to(device) if tensor_cache is None else
                        tensor_cache.get('advantages', rows) * tensor_cache.denoising_discounts(steps, policy.steps, gamma_denoising))
                    objective = torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
                    loss = -objective.sum() / denominator
                    with measure('actor.ppo_backward', gpu=True):
                        loss.backward()
                        if gradient_sum is not None:
                            gradient_sum.add()
                    old_mean, old_std = _old_kernel(rows, steps, device, tensor_cache)
                    kl = _joint_kl(parameters, mask, old_mean, old_std).detach()
                    summary += torch.stack((loss.detach(), ((ratio < 1 - clip) | (ratio > 1 + clip)).double().sum(),
                                            ratio.detach().sum(), log_ratio.detach().abs().sum(), kl.sum(),
                                            ratio.new_tensor(ratio.numel(), dtype=torch.float64),
                                            (((advantage > 0) & (ratio > 1 + clip)) |
                                            ((advantage < 0) & (ratio < 1 - clip))).double().sum()))
                if gradient_sum is not None:
                    gradient_sum.finish()
                if conditions is not None:
                    conditions.backward()
            if distributed is not None:
                distributed.sum_gradients(actor)
                summary = distributed.sum_tensor(summary)
            # 当前 minibatch 已经计算出真实 KL；两种模式都必须遵守软停止。
            # 上一 minibatch 的更新后 KL 不能代替当前不同样本上的更新前 KL。
            if (soft_kl_limit is not None
                    and float(summary[4] / summary[5]) >= soft_kl_limit):
                discarded = dict(epoch=epoch, global_upper_indices=indices,
                    mean_joint_kl=float(summary[4] / summary[5]), optimizer_step_executed=False,
                    scope='current_parameters_before_pending_step_against_fixed_rollout')
                optimizer.zero_grad(set_to_none=True)
                stopped = 'pre_minibatch_soft_kl'
                break
            with _local_phase(distributed, 'actor_ppo_gradient_validation'):
                ppo_norm = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), float('inf'), error_if_nonfinite=True))
                ppo_gradients = ({name: parameter.grad.detach().clone() for name, parameter in actor.named_parameters()
                                  if parameter.grad is not None} if gradient_diagnostics and _rank(distributed) == 0 else {})
            bc_enabled = bc is not None and bc_weight > 0
            if distributed is not None:
                bc_enabled = distributed.broadcast_object(bc_enabled if _rank(distributed) == 0 else None)
            bc_report = None
            if bc_enabled:
                sharded_bc = bool(getattr(bc, 'distributed_global_mean', False))
                if distributed is not None:
                    sharded_bc = distributed.broadcast_object(sharded_bc if _rank(distributed) == 0 else None)
                with _local_phase(distributed, 'bc_local_backward'):
                    # PPO已经SUM，第二次通信前只保留rank0的一份；BC各rank按全局均值贡献。
                    if _rank(distributed) != 0:
                        optimizer.zero_grad(set_to_none=True)
                    if sharded_bc or _rank(distributed) == 0:
                        if bc is None: raise ValueError('Distributed BC requires an independent anchor on every rank')
                        from .numerical_execution import precision_scope
                        with precision_scope(getattr(policy, 'numerical_execution', None)):
                            bc_report = bc.backward(actor, weight=bc_weight/(distributed.world_size if sharded_bc and distributed else 1))
                if distributed is not None:
                    distributed.sum_gradients(actor)
                    if sharded_bc:
                        shards = distributed.all_gather_object(bc_report)
                        bc_report = dict(weight=bc_weight, batch_size=sum(item['batch_size'] for item in shards),
                            loss=sum(item['loss']*item['batch_size'] for item in shards)/sum(item['batch_size'] for item in shards),
                            distribution='all_ranks_global_mean_gradient_sum', ranks=shards,
                            bc_update_steps=shards[0]['bc_update_steps'])
                    else:
                        bc_report = distributed.broadcast_object(bc_report if _rank(distributed) == 0 else None)
            actor.eval()
            with _local_phase(distributed, 'actor_gradient_diagnostics'):
                diagnostics = (_gradient_pair_report(actor, ppo_gradients, module_details=gradient_module_details)
                               if gradient_diagnostics and _rank(distributed) == 0 else None)
                del ppo_gradients
            if distributed is not None and gradient_diagnostics:
                diagnostics = distributed.broadcast_object(diagnostics)
            with _local_phase(distributed, 'actor_gradient_clipping'):
                # 仅显式验收时读取完整、同步后且尚未裁剪的梯度；默认路径没有拷贝。
                if gradient_observer is not None:
                    gradient_observer(actor, len(reports))
                total_norm = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), grad_clip_norm, error_if_nonfinite=True))
            if reserve_attempt is not None:
                reserve_attempt()
            with _local_phase(distributed, 'actor_optimizer_step'):
                optimizer.step()
            post = None
            if kl_check_mode == 'post_step_full':
                post = analytic_kl_local(policy, transitions, global_manifest=global_manifest, distributed=distributed,
                                          denoising_microbatch=denoising_microbatch, global_indices=indices, tensor_cache=tensor_cache,
                                          return_cache=kl_cache_sink is not None)
                if kl_cache_sink is not None:
                    post, kl_cache_sink['cache'] = post
            record = dict(epoch=epoch, optimizer_step=len(reports) + 1, global_upper_indices=indices,
                internal_transitions=denominator, ppo_loss=float(summary[0]),
                clip_fraction=float(summary[1] / summary[5]), mean_ratio=float(summary[2] / summary[5]),
                objective_clipped_fraction=float(summary[6] / summary[5]),
                mean_abs_log_ratio=float(summary[3] / summary[5]), pre_minibatch_mean_joint_kl=float(summary[4] / summary[5]),
                ratio_scope='before_this_minibatch_step_against_fixed_rollout_old_policy',
                ppo_only_gradient_norm=ppo_norm, total_gradient_norm=total_norm,
                grad_clip_factor=min(1., grad_clip_norm / (total_norm + 1e-6)), bc=bc_report,
                gradient_contributions=diagnostics, post_minibatch_kl=post,
                post_kl_extra_internal_forwards=denominator if post is not None else 0,
                kl_check_mode=kl_check_mode,
                learning_rates=[group['lr'] for group in optimizer.param_groups])
            reports.append(record)
            with _local_phase(distributed, 'actor_step_callback'):
                if step_callback is not None:
                    step_callback(record)
            if post is not None and soft_kl_limit is not None and post['mean_joint_kl'] >= soft_kl_limit:
                stopped = 'minibatch_soft_kl'
                break
        if stopped:
            break
    return dict(optimizer_steps=len(reports), optimizer_attempts=len(reports), steps=reports,
        epoch_orders=orders, epochs_started=len(orders), early_stop_reason=stopped,
        included_upper_transitions=len(selected), excluded_upper_transitions=len(manifest) - len(selected),
        probability_check=probability, objective_logprob_reduction=objective_logprob_reduction,
        kl_check_mode=kl_check_mode, discarded_pending_minibatch=discarded,
        old_statistics_fixed=True, hard_kl_pending=True, rollback_scope='caller_owned_whole_rollout',
        minibatch_order_contract='owner_balanced_complete_chains.v1' if balanced_minibatches else 'global_shuffle.v1',
        bc_global_samples=sum((item['bc'] or {}).get('batch_size', 0) for item in reports))


def critic_update_local(critic, optimizer, transitions, targets, *, global_manifest=None,
                        distributed=None, steps=80, batch_size=32, generator=None, grad_clip_norm=1., tensor_cache=None,
                        epochs=None):
    """小 Critic 用全局索引、本地条件和固定目标学习；前后指标明确属于本批数据。"""
    if type(steps) is not int or steps < 1 or type(batch_size) is not int or batch_size < 1:
        raise ValueError('Critic steps and global batch size must be positive integers')
    if not math.isfinite(grad_clip_norm) or grad_clip_norm <= 0:
        raise ValueError('Critic gradient clip must be finite and positive')
    with _local_phase(distributed, 'critic_local_shard_validation'):
        manifest = _manifest(transitions, targets, distributed, global_manifest)
        selected = [i for i, item in enumerate(manifest) if item['valid']]
        if not selected:
            raise ValueError('No valid transition remains for Critic update')
        device = next(critic.parameters()).device
        target = torch.as_tensor(targets['returns'], device=device).detach().float()
        owned = _owned(selected, manifest, distributed)
        if target.shape != (len(transitions),) or any(not torch.isfinite(target[i]) for _, i in owned):
            raise ValueError('Critic requires aligned finite fixed returns')

    def forward(indices):
        rows = [transitions[index] for index in indices]
        return critic(_context(rows, device), torch.tensor([row.metadata['remaining_music_seconds'] for row in rows], device=device)) if tensor_cache is None else critic(
            tensor_cache.context(rows), tensor_cache.get('remaining_music_seconds', rows))

    @torch.no_grad()
    def metrics():
        critic.eval()
        with _local_phase(distributed, 'critic_metrics_forward'):
            indices = [index for _, index in owned]
            sums = torch.zeros(6, device=device, dtype=torch.float64)
            if indices:
                expected = target[indices].double()
                difference = expected - forward(indices).double()
                sums = torch.stack((expected.new_tensor(len(indices)), expected.sum(), expected.square().sum(),
                                    difference.sum(), difference.square().sum(), difference.abs().sum()))
        if distributed is not None:
            sums = distributed.sum_tensor(sums)
        count, total, squared, error, error_squared, absolute = sums
        variance = (squared / count - (total / count).square()).clamp_min(0.)
        error_variance = (error_squared / count - (error / count).square()).clamp_min(0.)
        return dict(mse=float(error_squared / count), bias=float(error / count), mae=float(absolute / count),
            explained_variance=None if variance <= 1e-12 else float(1. - error_variance / variance),
            target_variance=float(variance), count=int(count))

    initial = metrics()
    losses, norms = [], []
    plan = None
    if epochs is not None:
        if type(epochs) is not int or epochs < 1: raise ValueError('Critic epochs must be positive')
        if distributed is None or _rank(distributed) == 0:
            plan = []
            for _ in range(epochs):
                order = balanced_epoch_order(selected, manifest, generator)
                plan.extend(order[start:start+batch_size] for start in range(0,len(order),batch_size))
        if distributed is not None: plan = distributed.broadcast_object(plan)
        steps = len(plan)
    for step in range(steps):
        if distributed is None or _rank(distributed) == 0:
            global_indices = plan[step] if plan is not None else [selected[i] for i in torch.randperm(len(selected), generator=generator)[:batch_size].tolist()]
        else:
            global_indices = None
        if distributed is not None:
            global_indices = distributed.broadcast_object(global_indices)
        local_indices = [index for _, index in _owned(global_indices, manifest, distributed)]
        optimizer.zero_grad(set_to_none=True)
        critic.train()
        loss = torch.zeros((), device=device)
        with _local_phase(distributed, 'critic_minibatch_forward_backward'):
            if local_indices:
                loss = .5 * (forward(local_indices) - target[local_indices]).square().sum() / len(global_indices)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite Critic loss')
                loss.backward()
        if distributed is not None:
            distributed.sum_gradients(critic)
            loss = distributed.sum_tensor(loss.detach())
        with _local_phase(distributed, 'critic_gradient_clipping_and_optimizer_step'):
            norms.append(float(torch.nn.utils.clip_grad_norm_(critic.parameters(), grad_clip_norm, error_if_nonfinite=True)))
            optimizer.step()
        losses.append(float(loss.detach()))
    final = metrics()
    return dict(losses=losses, gradient_norms=norms, optimizer_steps=steps,
        epochs=epochs, sampling='complete_epoch_without_replacement' if epochs else 'legacy_repeated_minibatch',
        global_sample_visits=len(selected)*epochs if epochs else sum(min(batch_size,len(selected)) for _ in losses),
        initial_mse=initial['mse'], initial_explained_variance=initial['explained_variance'],
        mse=final['mse'], explained_variance=final['explained_variance'], before=initial, after=final,
        metric_scope='current_rollout_fixed_bootstrapped_targets_not_independent_value_ground_truth')
