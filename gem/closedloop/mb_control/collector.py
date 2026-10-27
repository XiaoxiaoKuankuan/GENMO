"""真实SM物理采集与异步GENMO工作线程，持续执行旧参考直到新计划真实到达。

每次请求冻结真实快照，worker计算前瞻/条件/DDIM/解码，主线程50Hz推进同款物理，
不等待生成暂停物理。实际延迟在worker完成后记录，超时由共享scheduler处理；慢物理
单独报告墙钟超期，不能冒充实时性。首版真实采集batch=1，行政窗口只在无pending
边界返回，不自动复位；失败/音乐结束后的reset必须由调用者显式发起。
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from collections import deque
import math
import time

import torch

from .contracts import (ClosedLoopState, ForecastContext, ReferenceExhausted, PendingGeneration,
                        GenerationRecord, PhysicalAnchor)
from .forecast import forecast_rollout
from .reference_buffer import ReferenceBuffer
from .returns import DecisionInterval, IntervalAccumulator


class GenerationWorker:
    def __init__(self, actor, adapter, forecast_enabled):
        self.actor, self.adapter, self.forecast_enabled = actor, adapter, forecast_enabled
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="genmo-plan")
        self.future = None

    def submit(self, snapshot, ticket, noise, *, request_wall=None):
        if self.future is not None:
            raise RuntimeError("每环境只允许一个在途请求")
        self.started = time.perf_counter() if request_wall is None else request_wall
        # 张量克隆在调用线程完成，worker流必须等待这些输入准备完成。
        producer_stream = torch.cuda.current_stream(snapshot.state.device) if snapshot.state.is_cuda else None
        def generate():
            def run():
                with torch.no_grad():
                    begin = time.perf_counter()
                    context = (forecast_rollout(snapshot, self.adapter, ticket.first_influence_tick,
                               reuse_current_observation=getattr(self.actor,"fast_forecast",False)).context
                               if self.forecast_enabled else ForecastContext.invalid(snapshot.state, snapshot.tick))
                    if snapshot.state.is_cuda:
                        torch.cuda.current_stream(snapshot.state.device).synchronize()
                    forecast_end = time.perf_counter()
                    plan = self.actor.generate(snapshot, ticket, context, noise)
                    if snapshot.state.is_cuda:
                        torch.cuda.current_stream(snapshot.state.device).synchronize()
                    finished = time.perf_counter()
                    return plan, finished, {"queue_seconds": begin - self.started,
                                            "forecast_seconds": forecast_end - begin,
                                            "generation_decode_seconds": finished - forecast_end,
                                            "total_seconds": finished - self.started}, context
            if producer_stream is None:
                return run()
            stream = torch.cuda.Stream(device=snapshot.state.device)
            stream.wait_stream(producer_stream)
            with torch.cuda.stream(stream):
                return run()
        self.future = self.executor.submit(generate)

    def poll(self):
        return self.future.result() if self.future is not None and self.future.done() else None

    def clear(self):
        if self.future is None or not self.future.done():
            raise RuntimeError("不能清除未完成任务")
        self.future = None

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.future = None


class PhysicalCollector:
    def __init__(self, env, actor, adapter, scheduler, rewards, *, forecast_enabled=True, gamma=.99,
                 record_trajectory=False):
        if env.N != 1:
            raise ValueError("首版真实采集按单机器人延迟；批量采集需要独立每环境调度")
        self.env, self.actor, self.adapter = env, actor, adapter
        self.scheduler, self.rewards, self.gamma = scheduler, rewards, gamma
        self.worker = GenerationWorker(actor, adapter, forecast_enabled)
        self.record_trajectory = record_trajectory
        # 全量离线评价保存当时实际参考速度，避免后处理用差分速度代替SM原目标。
        self.record_reference_derivatives = False
        self.reference_derivatives = []
        self.state, self.terminal = None, False
        self.events, self.trajectory, self.metrics = [], [], []
        self.cancelled_requests = []
        self.overruns, self.wall_started = 0, None
        self.control_wall_seconds = []
        # 可选推理优化：advance已经重建当前观测，只有参考实际接管后需要重算。
        # 不修改SM观测定义；不可变参考在决策边界裁剪，避免每20ms重复校验/拷贝。
        self.reuse_control_observation = False
        self.profile_control = False
        self.profile_control_synchronize = False
        self.control_profiles = []
        self.actor_version = 0
        self.collector_id, self.episode_index = 0, -1
        self.anchor_horizon = 0
        self.record_controls = False
        self.record_contacts = False
        self.record_contact_metrics = False
        self.contact_counts, self.previous_contact = {}, None
        self.controls = []
        self.generations = {}
        self.consumed_versions = set()
        self.active_anchors = {}
        self.ready_anchors = deque(maxlen=128)
        self.noise_seed = None
        self.timing_mode = "wall_clock_async"
        self.delay_events = None
        self.request_distribution = []
        self.anchor_outcomes = []

    @torch.no_grad()
    def reset(self, music, initial_state, *, music_start_tick=0, initial_reference=None):
        if self.worker.future is not None:
            raise RuntimeError("复位前必须显式结束旧请求")
        dev = initial_state.device
        teacher = getattr(self.rewards, "teacher", None)
        teacher_reference = teacher.timeline(music, initial_state) if teacher is not None else None
        self.delay_events = (teacher_reference.identity.get("delivery_events")
            if self.timing_mode=="fixed_delay_replay" and teacher_reference is not None else None)
        self.episode_index += 1
        self.env.reset(torch.arange(1, device=dev), initial_state)
        observed = self.env.observe()
        if initial_reference is None:
            s = observed["state"]
            pose = torch.cat((s[:, :7], s[:, 13:34][:, self.adapter.to_genmo]), -1)
            # 有限启动保持轨迹只支持59/30秒，耗尽即终止，不能无限续接假参考。
            initial_reference = pose[:, None].expand(-1, 60, -1).clone()
        ticks = torch.arange(initial_reference.shape[1], device=dev, dtype=torch.long) * 20
        ref = ReferenceBuffer(ticks, initial_reference, initial_reference.new_zeros((1, len(ticks), 2)),
                              torch.zeros_like(ticks), self.adapter.runtime.kin, self.adapter.to_sm)
        current, _, _ = ref.read_current_future(0)
        obs = self.adapter.runtime.observation(observed["state"], observed["previous_action"], observed["xi"], current)
        history = initial_state.new_zeros((1, 49, 48))
        history = torch.cat((history, self.adapter.proprio(observed["state"], observed["xi"])[:, None]), 1)
        mask = torch.cat((torch.zeros(1, 49, device=dev, dtype=torch.bool), torch.ones(1, 1, device=dev, dtype=torch.bool)), 1)
        self.state = ClosedLoopState(0, observed["state"], observed["xi"], observed["previous_action"],
                                     obs[:, None].expand(-1, 10, -1).clone(), history, mask, ref, music,
                                     music_start_tick, music_start_tick + music.shape[1] * 20)
        task = getattr(self.adapter, "current_task", {})
        self.state = replace(self.state, teacher_reference=teacher_reference, task_id=str(task.get("id", "")),
            task_source=task.get("source", "unknown"), motion_class=task.get("motion_class", "unlabelled"))
        if teacher_reference is not None and self.rewards.config.get("episode_dance_weight",0.):
            from .episode_objective import with_episode_alignment
            from .dance_losses import AnchorAlignment
            alignment=AnchorAlignment.at_anchor(observed["state"],teacher_reference.read_current_future(0,0)[0])
            self.state=with_episode_alignment(self.state,alignment)
        self.terminal = False
        self.events, self.trajectory, self.metrics = [], [], []
        self.cancelled_requests = []
        self.overruns = 0
        self.control_wall_seconds = []
        self.control_profiles = []
        self.reference_derivatives = []
        self.controls = []
        self.generations, self.consumed_versions, self.active_anchors = {}, set(), {}
        self.contact_counts, self.previous_contact = {}, None
        self.ready_anchors.clear()
        self.request_distribution, self.anchor_outcomes = [], []
        self.wall_started = time.perf_counter()
        return self.state.detached()

    @torch.no_grad()
    def collect(self, decisions=1, replay=None):
        if self.state is None or self.terminal:
            raise RuntimeError("episode结束后必须显式reset")
        intervals = []
        for _ in range(decisions):
            z = self.state
            z.validate(decision=True)
            request_wall = time.perf_counter()
            start = z.detached()
            if replay is not None and not self.anchor_horizon:
                replay.add(start)
            try:
                ticket = self.scheduler.reserve(z)
            except ReferenceExhausted as error:
                self.terminal = True
                zero = torch.zeros_like(z.state[:, 0])
                intervals.append(DecisionInterval(start, z.detached(), zero, zero, 0, self.gamma, True,
                                                  f"reference_exhausted:{error}", {}))
                break
            generator = (torch.Generator(device=z.state.device).manual_seed(self.noise_seed+(z.tick//60)*1009)
                         if self.noise_seed is not None else None)
            noise = torch.randn(len(z.state), 60, 30, device=z.state.device, generator=generator)
            from .sample_distribution import describe_request
            description=describe_request(start,self.adapter,self.collector_id,self.episode_index)
            description["actor_version"]=self.actor_version
            if self.record_contacts:
                contact=self.env.contact_events()
                description["contact"]="double_support" if contact["left_foot"] and contact["right_foot"] else "single_support" if contact["left_foot"]!=contact["right_foot"] else "airborne"
            self.request_distribution.append(description)
            self.worker.submit(start, ticket, noise, request_wall=request_wall)
            # 离线固定事件对照：先准备数值计划再按记录事件推进仿真时钟；不可称实时部署。
            replay_arrival = None
            delivery_event = None
            if self.timing_mode == "fixed_delay_replay":
                if self.delay_events is None or str(ticket.request_tick) not in self.delay_events:
                    raise ValueError(f"固定延迟回放缺少请求{ticket.request_tick}的外生事件，不能猜测或重复末事件")
                delivery_event=self.delay_events[str(ticket.request_tick)]
                if not isinstance(delivery_event,dict):raise ValueError("回放需要实际完成/接收两个时刻，不能仅按完成延迟推断接管")
                replay_arrival = self.worker.future.result()
            request = PendingGeneration(start, ticket, noise.detach().clone(), self.actor_version)
            z = replace(z, pending=True, pending_request=request)
            totals = IntervalAccumulator(z.state[:, 0], self.gamma)
            reason, event = "decision_boundary", {}
            next_boundary = None
            wall_origin, tick_origin = request_wall, z.tick
            try:
                while True:
                    control_started = time.perf_counter()
                    phase_started, profile = control_started, {}
                    def mark(name):
                        nonlocal phase_started
                        if self.profile_control:
                            if self.profile_control_synchronize and z.state.is_cuda:
                                torch.cuda.current_stream(z.state.device).synchronize()
                            now = time.perf_counter()
                            profile[name], phase_started = now-phase_started, now
                    reference_changed = False
                    if z.tick >= z.music_end_tick:
                        self.terminal, reason = True, "music_end"
                        break
                    if self.timing_mode=="fixed_delay_replay":
                        arrival=(replay_arrival if z.pending and delivery_event["delivered"] and
                            z.tick>=delivery_event.get("delivery_tick",delivery_event["commit_tick"]) else None)
                    else:
                        arrival=self.worker.poll() if z.pending else None
                    if arrival is not None:
                        plan, finished, timings, context = arrival
                        actual_ready = ticket.request_tick + math.ceil((finished - self.worker.started) * 600)
                        if self.timing_mode == "fixed_delay_replay":
                            actual_ready = delivery_event["actual_ready_tick"]
                        if z.tick >= actual_ready:
                            z, event = self.scheduler.commit(z, plan, actual_ready_tick=actual_ready)
                            reference_changed = event["accepted"]
                            event["timings"] = timings
                            event["timing_mode"] = self.timing_mode
                            event["actor_version"] = self.actor_version
                            event["reference_version"] = z.reference.version if event["accepted"] else None
                            description.update(generation_accepted=event["accepted"],generation_event=event.get("reason"),
                                actual_ready_tick=actual_ready,commit_tick=event["commit_tick"])
                            self.events.append(event)
                            self.generations[ticket.decision_id] = GenerationRecord(request, context, plan,
                                actual_ready, event["commit_tick"], event["reference_version"], event["accepted"])
                            self.worker.clear()
                            next_boundary = self.scheduler.next_request_tick(ticket, z.tick)
                            if event["terminal_timeout"]:
                                self.terminal, reason = True, "consecutive_timeouts"
                                break
                    if not z.pending and z.tick >= next_boundary:
                        break
                    mark('request_handling')
                    if not self.reuse_control_observation or reference_changed:
                        z = self.adapter.refresh_current_observation(z)
                    before_control = z.detached() if self.record_controls or self.anchor_horizon else None
                    if self.anchor_horizon:
                        self._find_first_consumption(before_control)
                    _, _, goal = z.reference.read_current_future(z.tick)
                    with torch.profiler.record_function("mb/physical_sm_actor"):
                        action = self.adapter.bundle.act(z.sm_history, goal)
                    # 先核对下一步目标支持，物理推进后不偷偷reset或丢弃终态。
                    ref1, _, _ = z.reference.read_current_future(z.tick + 12)
                    mark('reference_observation_action')
                    with torch.profiler.record_function("mb/physics"):
                        state1 = self.env.step_physics(action)
                    mark('newton_step')
                    target = (z.teacher_reference.read_current_future(z.tick+12, future_steps=0)[0]
                              if z.teacher_reference is not None else None)
                    from .episode_objective import episode_alignment
                    reward, metrics = self.rewards(state1, action, z.previous_action, ref1, teacher_reference=target,
                        episode_alignment=episode_alignment(z,required=bool(self.rewards.config.get("episode_dance_weight",0.))))
                    failed = bool(self.adapter.runtime.termination(state1, ref1).any())
                    mark('reward_termination')
                    totals.add(reward, torch.zeros_like(reward) if failed else torch.ones_like(reward))
                    z = self.adapter.advance(z, state1, action, origin="real",
                        trim_reference=not self.reuse_control_observation)
                    mark('history_advance')
                    contact = self.env.contact_events() if self.record_contacts or self.record_contact_metrics else None
                    if self.record_contact_metrics:
                        self._accumulate_contacts(contact)
                    if before_control is not None:
                        end = z.detached()
                        control_event = {"contacts": contact} if self.record_contacts else {}
                        step = DecisionInterval(before_control, end, reward.detach().clone(),
                            torch.zeros_like(reward) if failed else torch.ones_like(reward), 1, self.gamma,
                            failed or z.tick >= z.music_end_tick, "physical_failure" if failed else "control_step", control_event)
                        self.controls.append(step)
                        if self.anchor_horizon:
                            self._append_anchor_step(step, replay)
                    metric_keys=list(metrics)
                    metric_values=torch.stack([metrics[k].mean() for k in metric_keys]).detach().cpu().tolist()
                    self.metrics.append(dict(zip(metric_keys,metric_values)))
                    if self.record_trajectory:
                        # 保存该控制帧实际读取的参考，供影子渲染；不能事后用新缓存
                        # 重建已被覆盖的旧参考，也不把参考根平移到实际机器人位置。
                        self.trajectory.append((z.tick, state1.cpu(), ref1.root_p.cpu(), action.cpu(),
                                                ref1.root_R.cpu(), ref1.q.cpu()))
                        if self.record_reference_derivatives:
                            self.reference_derivatives.append((ref1.root_v.cpu(), ref1.root_w.cpu(), ref1.qd.cpu()))
                    self.control_wall_seconds.append(time.perf_counter()-control_started)
                    mark('metrics_recording')
                    if self.profile_control:
                        self.control_profiles.append(profile)
                    if failed:
                        self.terminal, reason = True, "physical_failure"
                        break
                    deadline = wall_origin + (z.tick - tick_origin) / 600
                    slack = deadline - time.perf_counter()
                    if self.timing_mode == "fixed_delay_replay":
                        pass
                    elif slack > 0:
                        time.sleep(slack)
                    else:
                        self.overruns += 1
            except ReferenceExhausted as error:
                self.terminal, reason = True, f"reference_exhausted:{error}"
            if self.terminal and self.worker.future is not None:
                # episode已经停止后收束worker，确保优化器不会与生成并发改权重。
                _, finished, timings, _ = self.worker.future.result()
                self.cancelled_requests.append({"request_tick": ticket.request_tick, "terminal_tick": z.tick,
                    "actual_ready_tick": ticket.request_tick + math.ceil((finished - self.worker.started) * 600),
                    "reason": reason, "timings": timings, "consumed_by_scheduler": False})
                self.worker.clear()
            if self.reuse_control_observation:
                z = replace(z, reference=z.reference.trim(z.tick))
            z = replace(z, pending=False, pending_request=None, decision_id=z.decision_id + 1)
            if self.terminal and self.controls:
                self.controls[-1] = replace(self.controls[-1], terminal=True, reason=reason)
            if self.terminal:
                for record,anchor,transitions,_,_ in self.active_anchors.values():
                    self.anchor_outcomes.append({"stage":"B_incomplete_H_anchor","request_tick":record.request.ticket.request_tick,
                        "steps":len(transitions),"required":self.anchor_horizon,"reason":reason,
                        "task":anchor.task_id,"source":anchor.task_source,"collector":self.collector_id,
                        "episode":self.episode_index,"actor_version":record.request.actor_version})
                self.active_anchors.clear()
            self.state = z
            intervals.append(DecisionInterval(start, z.detached(), totals.reward, totals.survival, totals.steps,
                                              self.gamma, self.terminal, reason, event))
            if self.terminal:
                break
        return intervals

    def take_controls(self):
        controls, self.controls = self.controls, []
        return controls

    def take_distribution(self):
        result={"A_all_real_requests":self.request_distribution,"B_anchor_outcomes":self.anchor_outcomes}
        self.request_distribution,self.anchor_outcomes=[],[]
        return result

    def _accumulate_contacts(self, event):
        left, right = event["left_foot"], event["right_foot"]
        previous = self.previous_contact
        increments = {"frames": 1, "double_support_frames": int(left and right),
            "single_support_frames": int(left != right), "airborne_frames": int(not left and not right),
            "left_landings": int(previous is not None and left and not previous["left_foot"]),
            "right_landings": int(previous is not None and right and not previous["right_foot"]),
            "pair_changes": int(previous is not None and event["pair_signature"] != previous["pair_signature"])}
        for key, value in increments.items():
            self.contact_counts[key] = self.contact_counts.get(key, 0)+value
        self.previous_contact = event

    def _find_first_consumption(self, z):
        support = z.reference.dependency_support(z.tick)
        indices = (support-z.reference.lower)//20
        versions = set(z.reference.source_versions[indices].tolist())
        by_version = {g.reference_version: g for g in self.generations.values() if g.accepted}
        for version in versions - self.consumed_versions:
            if version in by_version:
                record = by_version[version]
                source_ticks = support[z.reference.source_versions[indices] == version].tolist()
                self.active_anchors[version] = [record, z, [], set(source_ticks), tuple(sorted(self.consumed_versions))]
                for event in self.events:
                    if event.get("reference_version") == version:
                        event["observed_first_control_tick"] = z.tick
                        event["no_previous_consumption_verified"] = True
        self.consumed_versions.update(versions)

    def _append_anchor_step(self, step, replay):
        finished = []
        for version, values in self.active_anchors.items():
            record, anchor, transitions, consumed, prior = values
            transitions.append(step)
            ref = step.start.reference
            support = ref.dependency_support(step.start.tick)
            indices = (support-ref.lower)//20
            consumed.update(support[ref.source_versions[indices] == version].tolist())
            if len(transitions) == self.anchor_horizon:
                sample = PhysicalAnchor(record, anchor, tuple(transitions), tuple(sorted(consumed)), prior,
                                        self.collector_id, self.episode_index)
                sample.validate(self.anchor_horizon)
                self.ready_anchors.append(sample)
                from .sample_distribution import describe_sample
                self.anchor_outcomes.append(dict(describe_sample(sample,self.adapter),stage="B_formed_H_anchor"))
                if replay is not None:
                    replay.add(sample)
                finished.append(version)
            elif step.terminal:
                self.anchor_outcomes.append({"stage":"B_incomplete_H_anchor","request_tick":record.request.ticket.request_tick,
                    "steps":len(transitions),"required":self.anchor_horizon,"reason":step.reason,
                    "task":anchor.task_id,"source":anchor.task_source})
                finished.append(version)
        for version in finished:
            del self.active_anchors[version]

    def close(self):
        self.worker.close()
