"""冻结GENMO异步名义生成与新Adapter真实控制的滚动连接器。

复用原GENMO预约/保护规则，只保护生成器的名义源点；Adapter仍每20ms重规划十帧。
GENMO只读取已发生真实历史，worker显式关闭WM forecast；在途请求快照只记录当时
可见输入。想象不重放未来真实生成结果：从pending前或下一预约边界截断bootstrap。
本模块不启动训练；调用者提供明确的有限启动名义缓存和真实环境，控制线程逐步调用。
"""

from dataclasses import replace
import math
import time

import torch

from gem.closedloop.mb_control.contracts import ClosedLoopState, PendingGeneration
from gem.closedloop.mb_control.collector import GenerationWorker
from gem.closedloop.mb_control.sm_adapter import SMAdapter


class RollingCollector:
    def __init__(self, collector, frozen_genmo, scheduler):
        if any(p.requires_grad for p in frozen_genmo.parameters()):
            raise ValueError("滚动GENMO及未训练FutureState分支必须全部冻结")
        self.collector, self.genmo, self.scheduler = collector, frozen_genmo, scheduler
        device = str(next(frozen_genmo.parameters()).device)
        self.sm = SMAdapter(collector.step.bundle, frozen_genmo.stage1.endecoder.kinematics.joint_order, device)
        self.worker = GenerationWorker(frozen_genmo, self.sm, forecast_enabled=False)
        self.pending = None
        self.events = []
        self.decision, self.next_request_tick = 0, 0
        self.latencies = ()
        self.history = self.history_valid = None

    @torch.no_grad()
    def reset(self, initial, bootstrap_nominal, audio, **metadata):
        if self.pending is not None:
            raise RuntimeError("不能丢弃在途GENMO请求；先close或结算")
        z = self.collector.reset(initial, bootstrap_nominal, audio, mode="rolling", **metadata)
        self.history = z.executed_state.new_zeros(1, 50, 48)
        self.history[:, -1] = self.sm.proprio(z.executed_state, z.xi)
        self.history_valid = torch.zeros(1, 50, dtype=torch.bool, device=z.executed_state.device)
        self.history_valid[:, -1] = True
        self.decision, self.next_request_tick, self.latencies, self.events = 0, z.tick, (), []
        return z

    def _source_state(self):
        z = self.collector.state
        return ClosedLoopState(z.tick, z.executed_state, z.xi, z.previous_action, z.control_history,
                               self.history, self.history_valid, replace(z.nominal_reference, kin=self.sm.runtime.kin),
                               z.audio.features, z.audio.start_tick, z.audio.end_tick,
                               latency_history=self.latencies, decision_id=self.decision,
                               pending=self.pending is not None, pending_request=self.pending)

    @torch.no_grad()
    def collect(self, count, replay=None, *, zero=False, pace_realtime=False):
        rows = []
        for _ in range(count):
            started = time.perf_counter()
            z = self.collector.state
            if self.collector.terminal:
                break
            arrival = self.worker.poll() if self.pending is not None else None
            if arrival is not None:
                plan, finished, timings, _ = arrival
                ready = self.pending.ticket.request_tick + math.ceil((finished-self.worker.started)*600)
                if z.tick >= ready:
                    source, event = self.scheduler.commit(self._source_state(), plan, actual_ready_tick=ready)
                    self.events.append(event | {"timings": timings})
                    if event["accepted"]:
                        self.collector.install_nominal(source.reference, known_at_tick=z.tick)
                    self.latencies = source.latency_history
                    self.next_request_tick = self.scheduler.next_request_tick(plan.ticket, z.tick)
                    self.worker.clear()
                    self.pending = None
                    self.decision += 1
                    if event["terminal_timeout"]:
                        raise RuntimeError("冻结GENMO连续超时，不能外推缺失名义轨迹")
            if self.pending is None and z.tick >= self.next_request_tick and z.tick % 60 == 0:
                snapshot = self._source_state().detached()
                ticket = self.scheduler.reserve(snapshot)
                noise = torch.randn(1, 60, 30, device=z.executed_state.device)
                self.worker.submit(snapshot, ticket, noise)
                self.pending = PendingGeneration(snapshot, ticket, noise, 0)
            z = self.collector.state
            boundary = z.tick if self.pending is not None else self.next_request_tick
            schedule = dict(pending_request=self.pending, next_request_tick=self.next_request_tick,
                            latency_history=self.latencies, genmo_history=self.history, genmo_history_valid=self.history_valid)
            self.collector.state = replace(z, plan_boundary_tick=boundary, scheduling=schedule)
            part = self.collector.collect(1, replay, zero=zero)
            if not part:
                raise RuntimeError("滚动缓存不足当前/未来/halo支持；停止而不是伪造参考")
            row = part[0]
            self.history = torch.cat((self.history[:, 1:], self.sm.proprio(row.end.executed_state, row.end.xi)[:, None]), 1)
            self.history_valid = torch.cat((self.history_valid[:, 1:], torch.ones_like(self.history_valid[:, :1])), 1)
            rows.append(row)
            if pace_realtime:
                time.sleep(max(0., .02-(time.perf_counter()-started)))
        return rows

    def close(self):
        self.worker.close()
        self.pending = None
