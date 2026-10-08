"""第九步按实际控制时间计算的半马尔可夫回报和 GAE。

输入 rewards 已经是每个 20ms 控制区间的积分奖励，本模块不再乘 dt。0.5 秒的 gamma
和 lambda 分别换算到 50Hz；变长 latency 转移使用实际控制步数，不能把去噪步数当成
环境时间。bootstrap_mask 决定下一状态可否自举，continuation_mask 独立决定能否向后
递推优势。无效转移切断两侧递推，不以零状态为无效终态计算价值。

返回 target 始终由未标准化的优势加采集时 old value 得到，全部 detach 后固定；Actor
优势只在整批有效上层转移上标准化一次。独立事件奖励在最后执行时刻折扣，零步终止
仍可有事件代价，不能除以实际步数或伪造一步。
"""
from __future__ import annotations

import torch


def compute_gae(rewards, values, next_values, executed_steps, bootstrap_mask, continuation_mask,
                valid=None, *, gamma_upper=.99, lambda_upper=.95, event_rewards=None, normalize=True):
    if not 0 < float(gamma_upper) <= 1 or not 0 < float(lambda_upper) <= 1:
        raise ValueError("gamma_upper and lambda_upper must be in (0,1]")
    n = len(rewards)
    def vector(value, dtype, name):
        result = torch.as_tensor(value, dtype=dtype).detach().cpu().clone()
        if result.shape != (n,):
            raise ValueError(f"{name} must have shape [N]")
        return result
    old = vector(values, torch.float64, "values")
    nxt = vector(next_values, torch.float64, "next_values")
    raw_steps = torch.as_tensor(executed_steps).detach().cpu()
    steps = vector(executed_steps, torch.int64, "executed_steps")
    if raw_steps.dtype == torch.bool or raw_steps.shape != steps.shape or not torch.equal(raw_steps, steps) or (steps < 0).any():
        raise ValueError("executed_steps must contain nonnegative integers")
    b = vector(bootstrap_mask, torch.bool, "bootstrap_mask")
    q = vector(continuation_mask, torch.bool, "continuation_mask")
    mask = torch.ones(n, dtype=torch.bool) if valid is None else vector(valid, torch.bool, "valid")
    events = torch.zeros(n, dtype=torch.float64) if event_rewards is None else vector(event_rewards, torch.float64, "event_rewards")
    if n and q[-1]:
        raise ValueError("last rollout transition cannot continue beyond its buffer")
    if not torch.isfinite(old[mask]).all() or not torch.isfinite(nxt[mask & b]).all() or not torch.isfinite(events[mask]).all():
        raise ValueError("nonfinite value or event reward in valid transition")
    gamma_low, lambda_low = gamma_upper ** (1 / 25), lambda_upper ** (1 / 25)
    discounted = torch.zeros(n, dtype=torch.float64)
    advantage = torch.zeros(n, dtype=torch.float64)
    for i, item in enumerate(rewards):
        r = torch.as_tensor(item, dtype=torch.float64).detach().cpu()
        if r.shape != (int(steps[i]),):
            raise ValueError("reward length must equal actual executed steps")
        if mask[i]:
            if not torch.isfinite(r).all():
                raise ValueError("nonfinite actual reward")
            discounted[i] = (r * gamma_low ** torch.arange(len(r), dtype=torch.float64)).sum()
            discounted[i] += gamma_low ** max(int(steps[i]) - 1, 0) * events[i]
    for i in range(n - 1, -1, -1):
        if not mask[i]:
            continue
        gamma = gamma_low ** int(steps[i])
        delta = discounted[i] - old[i]
        if b[i]:
            delta += gamma * nxt[i]
        advantage[i] = delta
        if q[i] and i + 1 < n and mask[i + 1]:
            advantage[i] += gamma * lambda_low ** int(steps[i]) * advantage[i + 1]
    targets = torch.where(mask, advantage + torch.where(mask, old, 0.), 0.)
    normalized = torch.zeros_like(advantage)
    if mask.any() and normalize:
        selected = advantage[mask]
        std = selected.std(unbiased=False)
        normalized[mask] = (selected - selected.mean()) / std.clamp_min(1e-8)
    elif mask.any():
        normalized[mask] = advantage[mask]
    if not all(torch.isfinite(value[mask]).all() for value in (discounted, advantage, targets, normalized)):
        raise FloatingPointError("nonfinite return or advantage; refusing a corrupt optimization target")
    return {"discounted_rewards": discounted, "advantages_raw": advantage, "returns": targets,
            "advantages": normalized, "valid": mask, "gamma_low": gamma_low, "lambda_low": lambda_low,
            "advantages_normalized": bool(normalize)}


def normalize_advantages_global(targets, *, distributed=None):
    """只汇总有效上层状态的 FP64 统计量；不跨环境拼接 GAE，不按 Actor 自由度筛选。

    调用前各卡先对独立连续轨迹计算 normalize=False 的 GAE。返回新的映射，不覆盖
    old value、returns 或 raw advantage；方差采用全局总体方差，空卡也参加相同归约。
    """
    raw = torch.as_tensor(targets['advantages_raw'], dtype=torch.float64).detach()
    valid = torch.as_tensor(targets['valid'], dtype=torch.bool, device=raw.device)
    if raw.shape != valid.shape or not torch.isfinite(raw[valid]).all():
        raise ValueError('Global advantage normalization requires finite aligned valid values')
    selected = raw[valid]
    statistics = torch.stack((raw.new_tensor(selected.numel()), selected.sum(), selected.square().sum()))
    if distributed is not None:
        statistics = distributed.sum_tensor(statistics)
    if not torch.isfinite(statistics).all():
        raise FloatingPointError('Nonfinite global advantage normalization statistics')
    count, total, squared = statistics
    if count <= 0:
        raise ValueError('No valid upper transition remains for global advantage normalization')
    mean = total / count
    variance = (squared / count - mean.square()).clamp_min(0.)
    normalized = torch.zeros_like(raw)
    normalized[valid] = (raw[valid] - mean) / variance.sqrt().clamp_min(1e-8)
    if not torch.isfinite(normalized[valid]).all():
        raise FloatingPointError('Nonfinite globally normalized advantages')
    return {**targets, 'advantages': normalized, 'advantages_normalized': True,
            'advantage_normalization_scope': 'global_valid_upper_transitions',
            'advantage_global_count': int(count), 'advantage_global_mean': float(mean),
            'advantage_global_std': float(variance.sqrt())}
