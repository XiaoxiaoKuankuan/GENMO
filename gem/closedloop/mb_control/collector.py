"""真实SM物理采集与异步GENMO工作线程，持续执行旧参考直到新计划真实到达。

每次请求冻结真实快照，worker计算前瞻/条件/DDIM/解码，主线程50Hz推进同款物理，
不等待生成暂停物理。实际延迟在worker完成后记录，超时由共享scheduler处理；慢物理
单独报告墙钟超期，不能冒充实时性。首版真实采集batch=1，行政窗口只在无pending
边界返回，不自动复位；失败/音乐结束后的reset必须由调用者显式发起。
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import math
import time

import torch

from .contracts import ClosedLoopState, ForecastContext, ReferenceExhausted
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
                    context = (forecast_rollout(snapshot, self.adapter, ticket.first_influence_tick).context
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
                                            "total_seconds": finished - self.started}
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
        self.state, self.terminal = None, False
        self.events, self.trajectory, self.metrics = [], [], []
        self.overruns, self.wall_started = 0, None

    @torch.no_grad()
    def reset(self, music, initial_state, *, music_start_tick=0, initial_reference=None):
        if self.worker.future is not None:
            raise RuntimeError("复位前必须显式结束旧请求")
        dev = initial_state.device
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
        self.terminal = False
        self.events, self.trajectory, self.metrics = [], [], []
        self.overruns = 0
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
            if replay is not None:
                replay.add(start)
            try:
                ticket = self.scheduler.reserve(z)
            except ReferenceExhausted as error:
                self.terminal = True
                zero = torch.zeros_like(z.state[:, 0])
                intervals.append(DecisionInterval(start, z.detached(), zero, zero, 0, self.gamma, True,
                                                  f"reference_exhausted:{error}", {}))
                break
            noise = torch.randn(len(z.state), 60, 30, device=z.state.device)
            self.worker.submit(start, ticket, noise, request_wall=request_wall)
            z = replace(z, pending=True)
            totals = IntervalAccumulator(z.state[:, 0], self.gamma)
            reason, event = "decision_boundary", {}
            next_boundary = None
            wall_origin, tick_origin = request_wall, z.tick
            try:
                while True:
                    if z.tick >= z.music_end_tick:
                        self.terminal, reason = True, "music_end"
                        break
                    arrival = self.worker.poll() if z.pending else None
                    if arrival is not None:
                        plan, finished, timings = arrival
                        actual_ready = ticket.request_tick + math.ceil((finished - self.worker.started) * 600)
                        if z.tick >= actual_ready:
                            z, event = self.scheduler.commit(z, plan, actual_ready_tick=actual_ready)
                            event["timings"] = timings
                            self.events.append(event)
                            self.worker.clear()
                            next_boundary = self.scheduler.next_request_tick(ticket, z.tick)
                            if event["terminal_timeout"]:
                                self.terminal, reason = True, "consecutive_timeouts"
                                break
                    if not z.pending and z.tick >= next_boundary:
                        break
                    z = self.adapter.refresh_current_observation(z)
                    _, _, goal = z.reference.read_current_future(z.tick)
                    with torch.profiler.record_function("mb/physical_sm_actor"):
                        action = self.adapter.bundle.act(z.sm_history, goal)
                    # 先核对下一步目标支持，物理推进后不偷偷reset或丢弃终态。
                    ref1, _, _ = z.reference.read_current_future(z.tick + 12)
                    with torch.profiler.record_function("mb/physics"):
                        state1 = self.env.step_physics(action)
                    reward, metrics = self.rewards(state1, action, z.previous_action, ref1)
                    failed = bool(self.adapter.runtime.termination(state1, ref1).any())
                    totals.add(reward, torch.zeros_like(reward) if failed else torch.ones_like(reward))
                    z = self.adapter.advance(z, state1, action, origin="real")
                    self.metrics.append({k: float(v.mean()) for k, v in metrics.items()})
                    if self.record_trajectory:
                        self.trajectory.append((z.tick, state1.cpu(), ref1.root_p.cpu(), action.cpu()))
                    if failed:
                        self.terminal, reason = True, "physical_failure"
                        break
                    deadline = wall_origin + (z.tick - tick_origin) / 600
                    slack = deadline - time.perf_counter()
                    if slack > 0:
                        time.sleep(slack)
                    else:
                        self.overruns += 1
            except ReferenceExhausted as error:
                self.terminal, reason = True, f"reference_exhausted:{error}"
            if self.terminal and self.worker.future is not None:
                # episode已经停止后收束worker，确保优化器不会与生成并发改权重。
                self.worker.future.result()
                self.worker.clear()
            z = replace(z, pending=False, decision_id=z.decision_id + 1)
            self.state = z
            intervals.append(DecisionInterval(start, z.detached(), totals.reward, totals.survival, totals.steps,
                                              self.gamma, self.terminal, reason, event))
            if self.terminal:
                break
        return intervals

    def close(self):
        self.worker.close()
