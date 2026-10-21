"""完整真实回报及可变时长N-step目标。

MC调用从旧实现抽取的公共时间审计与折扣算术，奖励已经积分，绝不再乘dt。
FVI仅读取完整Episode、真实区间奖励和上一轮detach价值；到终止清零尾值，
不跨Episode，也不把FVI目标写回真实MC标签。所有回报运算使用CPU FP64。
"""
from __future__ import annotations

import torch
from gem.closedloop.common.complete_episode import complete_mc_returns


def mc_returns(episode, gamma_upper=.99):
    episode.validate()
    values = complete_mc_returns(episode.rows, gamma_upper=gamma_upper)
    episode.returns = values
    return values


def interval_rewards(rows, gamma_upper=.99):
    if not 0 < gamma_upper <= 1:
        raise ValueError("Invalid upper discount")
    gamma = gamma_upper ** (1 / 25)
    rewards, discounts = [], []
    for row in rows:
        n = row.executed_control_steps
        r = row.rewards.detach().cpu().double()
        event = float(row.metadata.get("event_reward", 0.))
        rewards.append((r * gamma ** torch.arange(n, dtype=torch.float64)).sum()
                       + gamma ** max(n - 1, 0) * event)
        discounts.append(gamma ** n)
    return torch.stack(rewards), torch.tensor(discounts, dtype=torch.float64)


def nstep_targets(episode, previous_values, nstep=16, gamma_upper=.99):
    episode.validate()
    if type(nstep) is not int or nstep < 1:
        raise ValueError("FVI lookahead must be positive")
    if torch.is_tensor(previous_values) and previous_values.requires_grad:
        raise ValueError("FVI target values must be frozen/detached")
    values = torch.as_tensor(previous_values, dtype=torch.float64).detach().cpu()
    count = len(episode.rows)
    if values.shape != (count,) or not torch.isfinite(values).all():
        raise ValueError("FVI values must match this complete episode")
    rewards, discounts = interval_rewards(episode.rows, gamma_upper)
    targets = torch.zeros(count, dtype=torch.float64)
    for start in range(count):
        scale = 1.
        stop = min(start + nstep, count)
        for index in range(start, stop):
            targets[start] += scale * rewards[index]
            scale *= discounts[index]
        if stop < count:
            targets[start] += scale * values[stop]
    return targets


def nstep_advantages(episode, crossfit_values, nstep=4, gamma_upper=.99):
    """Actor标签专用N-step优势；当前和尾部Value必须来自同一交叉Critic快照。"""
    if torch.is_tensor(crossfit_values) and crossfit_values.requires_grad:
        raise ValueError("Actor advantage requires frozen cross-fitted values")
    value = torch.as_tensor(crossfit_values).detach().cpu().double()
    if nstep == "full_mc":
        return mc_returns(episode, gamma_upper)-value
    if nstep not in (4, 8, 16):
        raise ValueError("Actor advantage horizon must be 4, 8, 16 or full_mc")
    return nstep_targets(episode, value, nstep, gamma_upper)-value
