"""多次生成、外生到达、参考覆盖和闭环WM推进的完整可微想象。

真实起点仅detach一次；后续预测、历史、前缀、参考合并及终点价值输入全程保留图。
每区间重新调用当前GENMO，等待期间照常执行旧参考，每控制步奖励仅计一次。
无pending公共网格作为非终止区间边界；真音乐结束/缓存耗尽取消pending并置零bootstrap。
"""

from dataclasses import replace

import torch

from .contracts import ForecastContext, ReferenceExhausted, PendingGeneration
from .forecast import forecast_rollout
from .returns import DecisionInterval, IntervalAccumulator
from .scheduler import ceil_grid
from .dependency import ReferenceDependencies


class Imagination:
    def __init__(self, actor, adapter, scheduler, rewards, *, decisions=3, gamma=.99,
                 forecast_enabled=True, max_model_steps=None):
        if decisions < 2:
            raise ValueError("正式想象必须包含多次生成；单计划调试请单独调用forecast")
        self.actor, self.adapter, self.scheduler, self.rewards = actor, adapter, scheduler, rewards
        self.decisions, self.gamma = decisions, gamma
        self.forecast_enabled, self.max_model_steps = forecast_enabled, max_model_steps
        self.support_guard = None

    def rollout(self, initial, noises, actual_latency_ticks, *, detach_initial=True,
                transition_fn=None, on_step=None, on_plan=None, on_transition=None, on_request=None):
        """transition_fn仅供离线配对物理审计；正式异步采集由PhysicalCollector执行。"""
        if len(noises) != self.decisions or len(actual_latency_ticks) != self.decisions:
            raise ValueError("每个决策必须有独立重参数噪声和外生延迟")
        z = initial.detached() if detach_initial else initial
        z.validate(decision=True)
        if self.support_guard is not None:
            self.support_guard(z.state)
        intervals = []
        model_steps = 0
        state_depth = 0
        dependencies = ReferenceDependencies(z.reference)
        forecast_branches = []
        for decision_index, (noise, latency) in enumerate(zip(noises, actual_latency_ticks)):
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
            if on_request is not None:
                on_request(z, ticket, decision_index)
            plan_depth = max(state_depth, dependencies.prefix_depth(z.reference, ticket))
            if self.forecast_enabled:
                forecast = forecast_rollout(z, self.adapter, ticket.first_influence_tick, support_guard=self.support_guard,
                                            reference_dependencies=dependencies, initial_dependency_steps=state_depth)
                context = forecast.context
                plan_depth = max(plan_depth, forecast.dependency_steps)
                forecast_branches.append({"request_tick": z.tick, "forecast_tick": ticket.first_influence_tick,
                                          "control_steps": forecast.control_steps, "dependency_steps": plan_depth})
                if self.max_model_steps is not None and plan_depth > self.max_model_steps:
                    raise ValueError("预测分支超过已验证WM时域")
            else:
                context = ForecastContext.invalid(z.state, z.tick)
            plan = self.actor.generate(z, ticket, context, noise)
            if on_plan is not None:
                on_plan(plan)
            # latency从此刻才进入事件系统，不能进入上面的forecast/condition。
            ready = ticket.request_tick + latency
            z = replace(z, pending=True, pending_request=PendingGeneration(z, ticket, noise, -1))
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
                            dependencies.register(z.reference, plan_depth)
                        if event["terminal_timeout"]:
                            terminal, reason = True, "consecutive_timeouts"
                            break
                    boundary = self.scheduler.next_request_tick(ticket, ceil_grid(ready, 12))
                    if not z.pending and z.tick >= boundary:
                        break
                    next_depth = max(state_depth, dependencies.control_depth(z.reference, z.tick)) + 1
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
                    if self.support_guard is not None:
                        self.support_guard(state1)
                    if on_transition is not None:
                        on_transition(z, action, state1, ref1)
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
            z = replace(z, pending=False, pending_request=None, decision_id=z.decision_id + 1)
            event.update(cumulative_control_steps=model_steps, longest_model_dependency_steps=state_depth,
                         terminal_cache_dependency_steps=dependencies.cache_depth(z.reference),
                         forecast_branches=list(forecast_branches),
                         maximum_forecast_dependency_steps=max((x["dependency_steps"] for x in forecast_branches), default=0))
            intervals.append(DecisionInterval(start, z, totals.reward, totals.survival, totals.steps,
                                              self.gamma, terminal, reason, event))
            if terminal:
                break
        return intervals


def replace_plan_version(cache, plan, version):
    """只重建该请求实际保留下来的源点；后来已覆盖的源点属于固定行为续接。"""
    indices = (cache.ticks-plan.ticket.request_tick)//20
    selected = cache.source_versions == version
    if bool((selected & ((indices < plan.ticket.prefix_frames) | (indices >= 60))).any()):
        raise ValueError("参考来源版本与原计划的真实时间范围不一致")
    indices = indices.clamp(0, 59)
    return replace(cache,
        world_qpos=torch.where(selected[None, :, None], plan.world_qpos[:, indices], cache.world_qpos),
        contact=torch.where(selected[None, :, None], plan.contact[:, indices], cache.contact),
        prepared_reference=None, prepared_source=None, prepared_start=None, prepared_with_grad=False)


class PhysicalAnchorImagination:
    """原请求条件重生成、真实首次接管状态启动的短时路径导数。

    H内后来实际提交的其他计划作为固定行为续接重放，保持参考覆盖及其时刻；只优化
    当前被选中的请求，这是单请求的局部策略导数，不冒充三决策全策略BPTT。以后请求
    的Critic在途状态由对应模型时刻重建，绝不复制未来实测state55/历史。真实接管
    状态只初始化SM/WM；传给GENMO的始终是原请求快照、原噪声及原预测条件。
    """

    mode = "physical_anchor_short_bptt"

    def __init__(self, actor, adapter, scheduler, rewards, *, horizon_steps, gamma=.99, forecast_enabled=True):
        if horizon_steps not in (8, 16, 24, 32):
            raise ValueError("短时审计只实现8/16/24/32步，正式时域须另经真实验收")
        self.actor, self.adapter, self.scheduler, self.rewards = actor, adapter, scheduler, rewards
        self.horizon_steps, self.gamma, self.forecast_enabled = horizon_steps, gamma, forecast_enabled
        self.support_guard = None
        self.decisions = 1

    def rollout_anchor(self, sample, *, noise=None, plan_override=None, transition_fn=None, on_step=None):
        sample.validate(self.horizon_steps)
        record = sample.generation
        request = record.request
        if noise is None:
            noise = request.noise.detach().clone().requires_grad_(torch.is_grad_enabled())
        plan = (self.actor.generate(request.snapshot, request.ticket, record.forecast, noise)
                if plan_override is None else plan_override)
        z = sample.state.detached()
        start, totals = z, IntervalAccumulator(z.state[:, 0], self.gamma)
        states_by_tick = {z.tick: z}
        pending_by_id = {}
        version = record.reference_version
        consumed_steps = 0

        def schedule(current, recorded):
            pending = recorded.pending_request
            cache = replace_plan_version(recorded.reference, plan, version)
            if pending is not None and pending.ticket.request_tick >= start.tick:
                key = pending.ticket.decision_id
                if key not in pending_by_id:
                    if pending.ticket.request_tick not in states_by_tick:
                        raise ValueError("短时未来请求缺少对应模型时刻，禁止借用未来真实状态")
                    model_request = states_by_tick[pending.ticket.request_tick]
                    snapshot = replace(model_request, reference=cache, pending=False, pending_request=None,
                        decision_id=pending.ticket.decision_id, latency_history=recorded.latency_history,
                        protected_until=recorded.protected_until, consecutive_timeouts=recorded.consecutive_timeouts)
                    pending_by_id[key] = replace(pending, snapshot=snapshot)
                pending = pending_by_id[key]
            return replace(current, reference=cache, pending=recorded.pending, pending_request=pending,
                latency_history=recorded.latency_history, protected_until=recorded.protected_until,
                decision_id=recorded.decision_id, consecutive_timeouts=recorded.consecutive_timeouts)

        terminal, reason = False, "short_model_horizon"
        for index, step in enumerate(sample.transitions[:self.horizon_steps]):
            z = schedule(z, step.start)
            z = self.adapter.refresh_current_observation(z)
            if self.support_guard is not None:
                self.support_guard(z.state)
            support = z.reference.dependency_support(z.tick)
            consumed_steps += int(bool((z.reference.source_versions[(support-z.reference.lower)//20] == version).any()))
            _, _, goal = z.reference.read_current_future(z.tick)
            action = self.adapter.bundle.act(z.sm_history, goal)
            state1 = (self.adapter.bundle.step_model(z.state, action, z.xi) if transition_fn is None else transition_fn(action))
            if self.support_guard is not None:
                self.support_guard(state1)
            ref1, _, _ = z.reference.read_current_future(z.tick+12)
            reward, _ = self.rewards(state1, action, z.previous_action, ref1)
            failed = self.adapter.runtime.termination(state1, ref1)
            alive = (1-self.adapter.runtime.termination(state1, ref1, soft=True) if transition_fn is None
                     else (~failed).to(reward))
            totals.add(reward, alive)
            z = self.adapter.advance(z, state1, action, origin="imagined" if transition_fn is None else "real")
            states_by_tick[z.tick] = z
            z = schedule(z, step.end)
            if on_step is not None:
                on_step(z, reward)
            if transition_fn is not None and bool(failed.any()):
                terminal, reason = True, "physical_failure"
                break
            if z.tick >= z.music_end_tick:
                terminal, reason = True, "music_end"
                break
        if consumed_steps == 0:
            raise ValueError("新计划没有影响实际SM控制支持点，禁止伪造执行梯度")
        event = {"accepted": True, "reference_version": version, "actor_version": request.actor_version,
                 "request_tick": request.ticket.request_tick, "real_anchor_tick": start.tick,
                 "consumed_control_steps": consumed_steps, "actual_control_steps": totals.steps,
                 "longest_model_dependency_steps": totals.steps, "future_reference_policy": "fixed_recorded_behavior_continuation",
                 "genmo_condition_tick": request.snapshot.tick, "forecast_condition_is_recorded_prediction": True,
                 "no_previous_consumption_verified": True}
        interval = DecisionInterval(start, z, totals.reward, totals.survival, totals.steps, self.gamma, terminal, reason, event)
        return interval, plan, noise
