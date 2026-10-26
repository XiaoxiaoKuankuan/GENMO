"""真实与想象共用的因果生成预约、到达和原子参考接管规则。

预计延迟只读已发生记录，实际延迟由外生过程或真实worker完成事件提供，不进入请求
条件。每环境只允许一个在途请求，晚到侵犯保护支持点即丢弃；不会平移计划或暂停物理。
首次影响由实际支持集计算，中心消费时刻另记。时间和分支是外生离散量，参考保持可微。
"""

from dataclasses import dataclass, replace
import math

from .contracts import RequestTicket, ReferenceExhausted


def ceil_grid(tick, grid):
    return ((tick + grid - 1) // grid) * grid


@dataclass(frozen=True)
class Scheduler:
    bootstrap_latency_ticks: int
    latency_margin_ticks: int = 0
    request_interval_ticks: int = 300
    max_consecutive_timeouts: int = 3
    request_mode: str = "eager"

    def __post_init__(self):
        if type(self.bootstrap_latency_ticks) is not int or self.bootstrap_latency_ticks < 0:
            raise ValueError("需要显式新框架延迟标定；禁止默用旧后端延迟")
        if self.request_mode not in ("eager", "fixed_interval"):
            raise ValueError("调度模式只能是eager或fixed_interval")
        if self.request_interval_ticks < 60 or self.request_interval_ticks % 60:
            raise ValueError("生成请求间隔必须为60tick的正整数倍")

    def estimated_latency(self, state):
        history = state.latency_history[-16:]
        value = math.ceil(sum(history) / len(history)) if history else self.bootstrap_latency_ticks
        return value + self.latency_margin_ticks

    def reserve(self, state):
        state.validate(decision=True)
        ready = ceil_grid(state.tick + self.estimated_latency(state), 12)
        last_control = max(state.tick - 12, ready - 12)
        support = state.reference.dependency_support(last_control) if last_control >= state.reference.lower else None
        protection = int(support.max()) if support is not None else state.tick - 20
        first = max(state.tick + 6 * 20, ceil_grid(protection + 1, 20))
        prefix = (first - state.tick) // 20
        if prefix > 30:
            raise ReferenceExhausted("部署延迟所需保护区超过Stage1前缀范围")
        influence = state.reference.first_influence(ready, first)
        return RequestTicket(state.decision_id, state.tick, ready, first, influence, prefix, state.reference.version)

    def commit(self, state, plan, *, actual_ready_tick):
        if not state.pending or state.tick < actual_ready_tick:
            raise ValueError("只能在已到达的pending事件提交")
        ticket = plan.ticket
        if state.decision_id != ticket.decision_id:
            raise ValueError("重复或跨决策提交")
        protection = int(state.reference.dependency_support(state.tick - 12).max()) if state.tick > ticket.request_tick else state.tick - 20
        accepted = protection < ticket.first_mutable_source_tick
        cache = state.reference
        if accepted:
            cache = cache.merge_plan(plan, state.tick, protection)
        timeouts = 0 if accepted else state.consecutive_timeouts + 1
        history = (*state.latency_history, actual_ready_tick - ticket.request_tick)[-16:]
        event = {"request_tick": ticket.request_tick, "predicted_ready_tick": ticket.predicted_ready_tick,
                 "actual_ready_tick": actual_ready_tick, "commit_tick": state.tick if accepted else None,
                 "rejection_tick": None if accepted else state.tick,
                 "first_mutable_source_tick": ticket.first_mutable_source_tick,
                 "first_influence_tick": cache.first_influence(state.tick, ticket.first_mutable_source_tick) if accepted else None,
                 "first_center_consumed_tick": cache.first_center_consumed(state.tick, ticket.first_mutable_source_tick) if accepted else None,
                 "terminal_timeout": timeouts >= self.max_consecutive_timeouts,
                 "accepted": accepted, "reason": "committed" if accepted else "late_protected_support"}
        return replace(state, reference=cache, pending=False, pending_request=None, latency_history=history,
                       protected_until=protection, consecutive_timeouts=timeouts), event

    def next_request_tick(self, ticket, current_tick):
        # 至少前进一个公共网格，避免零延迟样本在同一tick无限请求。
        minimum = ticket.request_tick + (60 if self.request_mode == "eager" else self.request_interval_ticks)
        return ceil_grid(max(minimum, current_tick), 60)
