"""按实际50Hz控制步积分的变长区间回报与上层lambda targets。

即时奖励进入本模块前已乘dt一次。区间内累计gamma和软存活，跨区间用Gamma*C连接；
真实终止bootstrap为零，行政截断保留终点价值。Actor目标保持图，Critic目标整体detach。
"""

from dataclasses import dataclass, replace

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


def close_real_mc_episode(intervals, terminal_event):
    """把零控制步的真实任务终止关联到最后一个已执行步，拒绝行政截断。

    音乐结束、物理失败、参考耗尽或连续超时都可以在请求边界被发现；此时不能
    虚构新状态转移，但已有最后一步的bootstrap必须为零。终止时间必须恰好相同。
    """
    allowed = terminal_event.reason in ("music_end", "physical_failure", "consecutive_timeouts") or (
        terminal_event.reason.startswith("reference_exhausted:"))
    if not intervals or not terminal_event.terminal or not allowed:
        raise RuntimeError("MC需要真实episode终点，不能使用行政截断或空轨迹")
    if intervals[-1].end.tick != terminal_event.end.tick:
        raise RuntimeError("MC终止时间不属于最后一个已记录的真实控制步")
    intervals[-1] = replace(intervals[-1], terminal=True, reason=terminal_event.reason)
    return intervals


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
