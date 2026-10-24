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

    def rollout(self, initial, noises, actual_latency_ticks, *, detach_initial=True):
        if len(noises) != self.decisions or len(actual_latency_ticks) != self.decisions:
            raise ValueError("每个决策必须有独立重参数噪声和外生延迟")
        z = initial.detached() if detach_initial else initial
        z.validate(decision=True)
        intervals = []
        model_steps = 0
        for noise, latency in zip(noises, actual_latency_ticks):
            if type(latency) is not int or latency < 0:
                raise ValueError("实际延迟是非负整数外生tick")
            start = z
            ticket = self.scheduler.reserve(z)
            if self.forecast_enabled:
                forecast = forecast_rollout(z, self.adapter, ticket.first_influence_tick)
                context = forecast.context
                if self.max_model_steps is not None and model_steps + forecast.control_steps > self.max_model_steps:
                    raise ValueError("预测分支超过已验证WM时域")
            else:
                context = ForecastContext.invalid(z.state, z.tick)
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
                    boundary = self.scheduler.next_request_tick(ticket, ceil_grid(ready, 12))
                    if not z.pending and z.tick >= boundary:
                        break
                    if self.max_model_steps is not None and model_steps >= self.max_model_steps:
                        raise ValueError("多决策展开超过已验证WM时域")
                    z = self.adapter.refresh_current_observation(z)
                    _, _, goal = z.reference.read_current_future(z.tick)
                    action = self.adapter.bundle.act(z.sm_history, goal)
                    state1 = self.adapter.bundle.step_model(z.state, action, z.xi)
                    ref1, _, _ = z.reference.read_current_future(z.tick + 12)
                    reward, _ = self.rewards(state1, action, z.previous_action, ref1)
                    alive = 1 - self.adapter.runtime.termination(state1, ref1, soft=True)
                    totals.add(reward, alive)
                    z = self.adapter.advance(z, state1, action, origin="imagined")
                    model_steps += 1
            except ReferenceExhausted as error:
                terminal, reason = True, f"reference_exhausted:{error}"
            z = replace(z, pending=False, decision_id=z.decision_id + 1)
            intervals.append(DecisionInterval(start, z, totals.reward, totals.survival, totals.steps,
                                              self.gamma, terminal, reason, event))
            if terminal:
                break
        return intervals
