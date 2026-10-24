"""多次生成、外生到达、参考覆盖和闭环WM推进的完整可微想象。

真实起点仅detach一次；后续预测、历史、前缀、参考合并及终点价值输入全程保留图。
每区间重新调用当前GENMO，等待期间照常执行旧参考，每控制步奖励仅计一次。
无pending公共网格作为非终止区间边界；真音乐结束/缓存耗尽取消pending并置零bootstrap。
"""

from dataclasses import replace

import torch

from .contracts import ForecastContext, ReferenceExhausted
from .forecast import forecast_rollout
from .returns import DecisionInterval, IntervalAccumulator
from .scheduler import ceil_grid


class Imagination:
    def __init__(self, actor, adapter, scheduler, rewards, *, decisions=3, gamma=.99,
                 forecast_enabled=True, max_model_steps=None):
        if decisions < 2:
            raise ValueError("正式想象必须包含多次生成；单计划调试请单独调用forecast")
        self.actor, self.adapter, self.scheduler, self.rewards = actor, adapter, scheduler, rewards
        self.decisions, self.gamma = decisions, gamma
        self.forecast_enabled, self.max_model_steps = forecast_enabled, max_model_steps

    def rollout(self, initial, noises, actual_latency_ticks, *, detach_initial=True,
                transition_fn=None, on_step=None):
        """transition_fn仅供离线配对物理审计；正式异步采集由PhysicalCollector执行。"""
        if len(noises) != self.decisions or len(actual_latency_ticks) != self.decisions:
            raise ValueError("每个决策必须有独立重参数噪声和外生延迟")
        z = initial.detached() if detach_initial else initial
        z.validate(decision=True)
        intervals = []
        model_steps = 0
        state_depth, reference_depth = 0, 0
        forecast_branches = []
        for noise, latency in zip(noises, actual_latency_ticks):
            if type(latency) is not int or latency < 0:
                raise ValueError("实际延迟是非负整数外生tick")
            start = z
            try:
                ticket = self.scheduler.reserve(z)
            except ReferenceExhausted as error:
                zero = torch.zeros_like(z.state[:, 0])
                intervals.append(DecisionInterval(start, z, zero, zero, 0, self.gamma, True,
                                                  f"reference_exhausted:{error}", {}))
                break
            if self.forecast_enabled:
                forecast = forecast_rollout(z, self.adapter, ticket.first_influence_tick)
                context = forecast.context
                plan_depth = max(state_depth, reference_depth) + forecast.control_steps
                forecast_branches.append({"control_steps": forecast.control_steps, "dependency_steps": plan_depth})
                if self.max_model_steps is not None and plan_depth > self.max_model_steps:
                    raise ValueError("预测分支超过已验证WM时域")
            else:
                context = ForecastContext.invalid(z.state, z.tick)
                plan_depth = max(state_depth, reference_depth)
            plan = self.actor.generate(z, ticket, context, noise)
            # latency从此刻才进入事件系统，不能进入上面的forecast/condition。
            ready = ticket.request_tick + latency
            z = replace(z, pending=True)
            totals = IntervalAccumulator(z.state[:, 0], self.gamma)
            event, terminal, reason = {}, False, "decision_boundary"
            try:
                while True:
                    if z.tick >= z.music_end_tick:
                        terminal, reason = True, "music_end"
                        break
                    if z.pending and z.tick >= ready:
                        z, event = self.scheduler.commit(z, plan, actual_ready_tick=ready)
                        if event["accepted"]:
                            reference_depth = max(reference_depth, plan_depth)
                        if event["terminal_timeout"]:
                            terminal, reason = True, "consecutive_timeouts"
                            break
                    boundary = self.scheduler.next_request_tick(ticket, ceil_grid(ready, 12))
                    if not z.pending and z.tick >= boundary:
                        break
                    next_depth = max(state_depth, reference_depth) + 1
                    if self.max_model_steps is not None and next_depth > self.max_model_steps:
                        raise ValueError("多决策展开超过已验证WM时域")
                    z = self.adapter.refresh_current_observation(z)
                    _, _, goal = z.reference.read_current_future(z.tick)
                    with torch.profiler.record_function("mb/sm_actor"):
                        action = self.adapter.bundle.act(z.sm_history, goal)
                    with torch.profiler.record_function("mb/world_model_or_physics"):
                        state1 = (self.adapter.bundle.step_model(z.state, action, z.xi) if transition_fn is None
                                  else transition_fn(action))
                    ref1, _, _ = z.reference.read_current_future(z.tick + 12)
                    reward, _ = self.rewards(state1, action, z.previous_action, ref1)
                    failed = self.adapter.runtime.termination(state1, ref1)
                    alive = (1 - self.adapter.runtime.termination(state1, ref1, soft=True) if transition_fn is None
                             else (~failed).to(reward))
                    totals.add(reward, alive)
                    z = self.adapter.advance(z, state1, action, origin="imagined" if transition_fn is None else "real")
                    model_steps += 1
                    state_depth = next_depth
                    if on_step is not None:
                        on_step(z, reward)
                    if transition_fn is not None and bool(failed.any()):
                        terminal, reason = True, "physical_failure"
                        break
            except ReferenceExhausted as error:
                terminal, reason = True, f"reference_exhausted:{error}"
            z = replace(z, pending=False, decision_id=z.decision_id + 1)
            event.update(cumulative_control_steps=model_steps, longest_model_dependency_steps=state_depth,
                         forecast_branches=list(forecast_branches),
                         maximum_forecast_dependency_steps=max((x["dependency_steps"] for x in forecast_branches), default=0))
            intervals.append(DecisionInterval(start, z, totals.reward, totals.survival, totals.steps,
                                              self.gamma, terminal, reason, event))
            if terminal:
                break
        return intervals
