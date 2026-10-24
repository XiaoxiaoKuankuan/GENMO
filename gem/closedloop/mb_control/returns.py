"""按实际50Hz控制步积分的变长区间回报与上层lambda targets。

即时奖励进入本模块前已乘dt一次。区间内累计gamma和软存活，跨区间用Gamma*C连接；
真实终止bootstrap为零，行政截断保留终点价值。Actor目标保持图，Critic目标整体detach。
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DecisionInterval:
    start: object
    end: object
    reward: torch.Tensor
    survival: torch.Tensor
    steps: int
    gamma: float
    terminal: bool
    reason: str
    event: dict

    @property
    def continuation(self):
        return self.survival * (self.gamma ** self.steps) * (0. if self.terminal else 1.)


class IntervalAccumulator:
    def __init__(self, like, gamma):
        self.reward = torch.zeros_like(like)
        self.survival = torch.ones_like(like)
        self.steps, self.gamma = 0, gamma

    def add(self, reward_dt, alive):
        self.reward = self.reward + (self.gamma ** self.steps) * self.survival * reward_dt
        self.survival = self.survival * alive
        self.steps += 1


def actor_return(intervals, terminal_value):
    value = terminal_value
    for interval in reversed(intervals):
        value = interval.reward + interval.continuation * value
    return value


def lambda_targets(intervals, next_values, lambda_per_half_second=.95):
    if len(intervals) != len(next_values) or not intervals:
        raise ValueError("每个真实区间需要一个固定旧next value")
    result = []
    with torch.no_grad():
        target = next_values[-1].detach()
        for interval, next_value in reversed(list(zip(intervals, next_values))):
            lam = lambda_per_half_second ** (interval.steps * .02 / .5)
            target = interval.reward.detach() + interval.continuation.detach() * (
                (1 - lam) * next_value.detach() + lam * target)
            result.append(target)
    return list(reversed(result))
