"""当前 DPPO 更新使用的分片分配、批量重算和零更新概率检查。

完整行为链、自由掩码与旧高斯参数保持只读，每个 rank 只计算所属样本；条件缓存
与概率检查共用同一前向。局部异常在下一次 collective 前一致传播，防止单卡退出
后其余卡挂起。这里只保留当前更新器需要的辅助函数，不包含旧 PPO 或 Critic 更新。
"""
from __future__ import annotations
from collections import deque
from contextlib import contextmanager
import torch

from .policy import masked_joint_log_prob
from .performance import measure, profiled
from .execution_checks import policy_phase, require_tensor
from .tensor_cache import ConditionGraphCache

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
        if all(item['type']=='FloatingPointError' for item in failures):
            raise FloatingPointError(f'Cooperative numerical phase failed ({phase}): {failures}') from error
        raise RuntimeError(f'Cooperative local phase failed ({phase}): {failures}') from error



def _rank(distributed):
    return 0 if distributed is None else distributed.rank



def _manifest(transitions, targets, distributed, global_manifest):
    from .executed_action_mask import action_mask
    rank = _rank(distributed)
    valid = torch.as_tensor(targets.get('valid', [row.transition_valid for row in transitions])).bool()
    if valid.shape != (len(transitions),):
        raise ValueError('Targets must align with the local transition shard')
    if global_manifest is None:
        world = 1 if distributed is None else distributed.world_size
        result = [dict(owner_rank=i % world, local_index=i, valid=bool(valid[i]),
                       has_free=bool(action_mask(row).any())) for i, row in enumerate(transitions)]
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
            if bool(action_mask(transitions[index]).any()) != bool(item['has_free']):
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
def _parameters(policy, rows, steps, device, cache=None, conditions=None, *, executed_only=False):
    prepared_only = (conditions is not None and conditions.bank is not None
                     and hasattr(policy, '_transition_prepared'))
    context = None if prepared_only else (_context(rows, device) if cache is None else cache.context(rows))
    state = (torch.stack([row.chain[step] for row, step in zip(rows, steps)]).to(device)
             if cache is None else cache.get('chain', rows, steps))
    indices = torch.tensor(steps, device=device, dtype=torch.long) if cache is None else cache.step_indices(steps)
    if prepared_only:
        result = conditions.transition(rows, state, indices)
    elif hasattr(policy, 'prepare_conditions'):
        prepared = policy.prepare_conditions(context) if conditions is None else conditions.prepare(rows, context)
        result = policy.transition_parameters(context, state, indices, prepared=prepared)
    else:
        result = policy.transition_parameters(context, state, indices)
    mask = torch.stack([row.free_mask for row in rows]).to(device) if cache is None else cache.get('free_mask', rows)
    if 'free_mask' in result:
        require_tensor((result['free_mask'] == mask).all(), 'Current policy changed the fixed rollout free-coordinate mask')
    if executed_only:
        from .executed_action_mask import action_mask
        selected=(torch.stack([action_mask(row) for row in rows]).to(device) if cache is None
                  else cache.get('ppo_action_mask',rows))
        require_tensor((~selected | mask).all(),'Executed PPO coordinates escaped the sampled free mask')
        mask=selected
    return result, mask



def _old_kernel(rows, steps, device, cache=None):
    if cache is not None:
        return cache.get('old_means', rows, steps).double(), cache.get('old_stds', rows, steps).double()
    means = torch.cat([row.metadata['sampler_trace']['old_means'][:, step]
                       for row, step in zip(rows, steps)], 0).to(device).double()
    stds = torch.cat([row.metadata['sampler_trace']['old_stds'][:, step]
                      for row, step in zip(rows, steps)], 0).to(device).double()
    return means, stds



@torch.no_grad()
@profiled('learning.probability_check', gpu=True)
def probability_check_local(policy, transitions, *, global_manifest=None, distributed=None,
                            denoising_microbatch=4, global_indices=None, tensor_cache=None, include_frozen=True):
    """完整零更新检查仅用于启动／周期诊断；原始概率、ratio 和独立公式门槛不放宽。"""
    from gem.closedloop.dppo.trainer import _validate_actor_transitions
    if type(denoising_microbatch) is not int or denoising_microbatch < 1:
        raise ValueError('Denoising microbatch must be a positive integer')
    with _local_phase(distributed, 'initial_probability_local_forward'), policy_phase(policy):
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
        flat = [(index, step) for index in range(len(rows)) for step in (policy.stochastic_steps if include_frozen else policy.trainable_steps)]
        conditions = ConditionGraphCache(policy, tensor_cache,include_frozen=include_frozen) if tensor_cache is not None and hasattr(policy, 'prepare_conditions') else None
        if conditions is not None and getattr(policy, 'defer_checks', False):
            conditions.prime(rows)
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
            # 同链20步按amax一次归约，避免每个内部转移启动索引/maximum/写回小算子。
            # amax不改变浮点求和顺序，重复链索引和尾批均保留完整覆盖。
            indices = torch.tensor([index for index, _ in chunk], device=device)[:, None].expand(-1, 3)
            checks.scatter_reduce_(0, indices, batch, reduce='amax', include_self=True)
        if conditions is not None:
            conditions.validate_inputs()
    if distributed is not None:
        checks = distributed.gather_rows(checks, [position for position, _ in owned], len(selected))
    maximum = checks.amax(0).cpu()
    if not torch.isfinite(maximum).all() or (maximum > torch.tensor([1e-4, 1e-3, 1e-8])).any():
        raise FloatingPointError(f'Zero-update probability mismatch: {maximum.tolist()}')
    return dict(passed=True,checked_steps=list(policy.stochastic_steps if include_frozen else policy.trainable_steps),
                frozen_identity=policy.assert_frozen_identity(),max_abs_log_probability_difference=float(maximum[0]),
                max_abs_ratio_minus_one=float(maximum[1]), max_abs_independent_gaussian_difference=float(maximum[2]))
