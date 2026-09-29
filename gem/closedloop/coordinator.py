"""第8步冻结闭环协调器：音乐、请求快照、延迟事件与有限物理转移。

本模块不创建优化器，不直接调用 Isaac，也不拥有可改写的参考缓存。每个规划请求先向
GMT预留前缀，再把同一时刻实际历史交给现有Actor。暂停模式在当前控制边界发布；延迟
模式把完整生成/通信/转换wall耗时映射到600Hz仿真时间，等待期间消费旧参考。一次最多
一个候选，跨决策周期仍未到达时记录漏决策；永远不补发旧时刻条件、不回填过去。
advance RPC具有唯一编号和episode身份，反馈逐条交给recorder，失败后不自动reset。
音乐长度到控制步数使用整数帧换算，并与统计报告共用边界，避免浮点舍入造成正常
短音乐少执行20ms后被错误判为未完成。
"""
from __future__ import annotations

import math
import time

import numpy as np

from gem.runtime.closedloop_protocol import CLOCK_HZ, CONTROL_TICKS, DECISION_TICKS, RemoteError
from gem.closedloop.evaluation_music import music_control_steps


def ceil_control_tick(tick):
    return int(math.ceil(float(tick) / CONTROL_TICKS)) * CONTROL_TICKS


def numpy_conditions(conditions):
    return {key: (value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value))
            for key, value in conditions.items()}


class ClosedLoopCoordinator:
    def __init__(self, config, actor, backend, condition_builder, recorder):
        self.config, self.actor, self.backend = config, actor, backend
        self.builder, self.recorder = condition_builder, recorder
        self.advance_counter = 0
        self.guard = float(config["timing"]["latency_guard_s"])
        self.min_prefix = int(config["timing"]["min_prefix_frames"])
        self.warmup_ticks = int(config["timing"]["warmup_ticks"])
        if self.warmup_ticks % DECISION_TICKS:
            raise ValueError("Warmup must end on the common decision grid")

    def _advance(self, snapshot, steps, *, phase, decision_id=None):
        if not 1 <= steps <= 25:
            raise ValueError("Bounded advance requires 1..25 control steps")
        self.advance_counter += 1
        result = self.backend.call("advance", control_steps=steps,
                    advance_id=f"{snapshot['episode_id']}:advance:{self.advance_counter}",
                    expected_episode_id=snapshot["episode_id"])
        actual = int(result["executed_control_steps"])
        physics = int(result["executed_physics_steps"])
        new = result["snapshot"]
        if new["episode_id"] != snapshot["episode_id"]:
            raise RuntimeError("Backend reset inside an upper transition")
        if not 0 <= actual <= steps or physics != 4 * actual:
            raise RuntimeError("Invalid actual control/physics counts")
        if int(new["tick"]) - int(snapshot["tick"]) != actual * CONTROL_TICKS:
            raise RuntimeError("Feedback consumed time differs from actual control steps")
        if actual < steps and not new["done"]:
            raise RuntimeError("Backend returned a short nonterminal transition")
        for row in result["trace"]:
            if row["episode_id"] != snapshot["episode_id"]:
                raise RuntimeError("Trace contains a different episode")
            self.recorder.step({**row, "phase": phase, "decision_id": decision_id})
        self.recorder.event("advance", phase=phase, decision_id=decision_id,
            episode_id=new["episode_id"], begin_tick=snapshot["tick"], end_tick=new["tick"],
            requested_control_steps=steps, executed_control_steps=actual,
            executed_physics_steps=physics, done=new["done"], reason=new.get("reason"),
            reference_valid_end_tick=new.get("reference_valid_end_tick"),
            protected_end_tick=new.get("protected_end_tick"),
            read_watermark_tick=new.get("read_watermark_tick"),
            gmt_history_update_count=new.get("gmt_history_update_count"),
            proprio_history_update_count=new.get("history_append_count"))
        return new

    def _start(self, sample, seed, mode):
        snapshot = self.backend.call("reset_episode", seed=int(seed),
            episode_spec={"sample_id": sample["row"]["sample_id"], "dataset": sample["dataset"], "mode": mode})
        self.recorder.start_episode(sample, seed, mode, snapshot)
        while int(snapshot["tick"]) < self.warmup_ticks and not snapshot["done"]:
            steps = min(25, (self.warmup_ticks - int(snapshot["tick"])) // CONTROL_TICKS)
            snapshot = self._advance(snapshot, steps, phase="warmup")
        return snapshot

    def _generate(self, snapshot, music, *, seed, decision_id, budget_s):
        tick = int(snapshot["tick"])
        request_id = f"{snapshot['episode_id']}:decision:{decision_id}"
        start = time.perf_counter()
        reservation = self.backend.call("reserve_prefix", request={
            "env_id": snapshot["env_id"], "episode_id": snapshot["episode_id"],
            "request_id": request_id, "decision_tick": tick,
            "deadline_tick": tick + ceil_control_tick(budget_s * CLOCK_HZ),
            "min_prefix": self.min_prefix})
        conditions, meta = self.builder.build(snapshot, reservation, music, music_start_tick=self.warmup_ticks)
        meta.update(seed=int(seed), decision_id=decision_id, plan_id=f"{request_id}:plan")
        generated = self.actor.call("generate", conditions=numpy_conditions(conditions), meta=meta)
        prepared = self.backend.call("prepare_plan", generated_plan=generated)
        elapsed = time.perf_counter() - start
        self.recorder.plan(generated)
        self.recorder.event("plan_prepared", episode_id=snapshot["episode_id"],
            request_id=request_id, plan_id=meta["plan_id"], parent_plan_id=meta["parent_plan_id"],
            decision_id=decision_id, request_tick=tick, deadline_tick=reservation["deadline_tick"],
            prefix_frames=reservation["prefix_frames"], protected_end_tick=reservation["protected_end_tick"],
            end_to_end_seconds=elapsed, actor_seconds=generated.get("latency_s"),
            prepare_seconds=prepared.get("prepare_seconds"),
            max_position_error=prepared.get("max_position_error"),
            max_velocity_error=prepared.get("max_velocity_error"))
        return {"prepared_plan_id": prepared.get("prepared_plan_id", prepared.get("ticket_id")),
                "plan_id": meta["plan_id"], "request_id": request_id,
                "request_tick": tick, "deadline_tick": reservation["deadline_tick"],
                "end_to_end_seconds": elapsed}

    def calibrate(self, sample, music, *, warmup=None, samples=None):
        settings = self.config["timing"]
        warmup = settings["calibration_warmup"] if warmup is None else int(warmup)
        samples = settings["calibration_samples"] if samples is None else int(samples)
        if warmup < 0 or samples <= 0:
            raise ValueError("Calibration requires nonnegative warmup and positive samples")
        snapshot = self._start(sample, 42, "calibration")
        if snapshot["done"]:
            report = self.recorder.finish_episode(snapshot, "startup_failure")
            raise RuntimeError(f"Calibration startup failed: {report['reason']}")
        if self.config.get("diagnostics", {}).get("verify_actor_condition_edges", False):
            self.verify_actor_conditions(snapshot, music)
        durations = []
        # 静止于同一真实状态，仅测量请求成本，不把候选发布到参考时间线。
        for index in range(warmup + samples):
            pending = self._generate(snapshot, music, seed=42, decision_id=f"calibration:{index}", budget_s=.2)
            self.backend.call("discard_plan", prepared_plan_id=pending["prepared_plan_id"])
            if index >= warmup:
                durations.append(pending["end_to_end_seconds"])
            if index % 10 == 0:
                print(f"[CALIBRATION] {index+1}/{warmup+samples}", flush=True)
        p95 = float(np.percentile(durations, 95))
        result = {"warmup_requests": warmup, "measured_requests": samples,
                  "end_to_end_seconds": durations, "p95_seconds": p95,
                  "guard_seconds": self.guard, "latency_budget_seconds": p95 + self.guard,
                  "full_calibration": warmup == 10 and samples == 100}
        self.recorder.event("calibration", **result)
        self.recorder.finish_episode(snapshot, "calibration_complete")
        return result

    def verify_actor_conditions(self, snapshot, music):
        """用同一个完整冻结模型核验P=12/P=0/空历史及masked占位不影响输出。"""
        reservation = self.backend.call("reserve_prefix", request={
            "env_id": snapshot["env_id"], "episode_id": snapshot["episode_id"],
            "request_id": f"{snapshot['episode_id']}:condition_edges",
            "decision_tick": int(snapshot["tick"]), "deadline_tick": int(snapshot["tick"]) + 60,
            "min_prefix": 12})
        conditions, meta = self.builder.build(snapshot, reservation, music, music_start_tick=self.warmup_ticks)
        meta.update(seed=42, decision_id="condition_edges", plan_id=f"{reservation['request_id']}:plan")
        values = numpy_conditions(conditions)
        result = self.actor.call("generate", conditions=values, meta=meta)
        mask = values["known_qpos30_mask"][0]
        if not np.array_equal(result["qpos30"][mask], values["known_qpos30"][0][mask]):
            raise RuntimeError("Full Actor changed known physical coordinates")
        for name in ("qpos30", "qpos_world", "contact"):
            if not np.isfinite(result[name]).all():
                raise RuntimeError(f"Full Actor nonfinite output: {name}")
        empty = {key: value.copy() for key, value in values.items()}
        empty["known_qpos30_mask"][:] = False
        empty["known_qpos30"][:] = 0
        empty["proprio_history_valid"][:] = False
        empty["proprio_history"][:] = 0
        empty_meta = {**meta, "prefix_frames": 0}
        first = self.actor.call("generate", conditions=empty, meta=empty_meta)
        poisoned = {key: value.copy() for key, value in empty.items()}
        poisoned["known_qpos30"][:] = 1234.
        poisoned["proprio_history"][:] = -987.
        second = self.actor.call("generate", conditions=poisoned, meta=empty_meta)
        differences = {name: float(np.max(np.abs(first[name]-second[name])))
                       for name in ("qpos30", "qpos_world", "contact")}
        if any(not math.isfinite(value) or value > 1e-6 for value in differences.values()):
            raise RuntimeError(f"Masked placeholders affected full Actor output: {differences}")
        self.recorder.event("actor_condition_edges", episode_id=snapshot["episode_id"],
                            known_prefix_frames=int(meta["prefix_frames"]), known_physical_exact=True,
                            p0_empty_history_finite=True, masked_placeholder_max_errors=differences)

    def run_episode(self, sample, music, *, seed, mode, latency_budget_s, seconds=None):
        if mode not in ("paused", "latency"):
            raise ValueError("mode must be paused or latency")
        if not math.isfinite(latency_budget_s) or latency_budget_s <= 0:
            raise ValueError("latency budget must be finite and positive")
        requested = float(self.config["evaluation"]["seconds"] if seconds is None else seconds)
        if requested <= 0:
            raise ValueError("Episode seconds must be positive")
        end_tick = self.warmup_ticks + music_control_steps(len(music), requested) * CONTROL_TICKS
        snapshot = self._start(sample, seed, mode)
        if snapshot["done"]:
            return self.recorder.finish_episode(snapshot, "startup_failure")
        next_decision, pending = self.warmup_ticks, None
        decision_id = -1
        stop_reason = "duration_limit" if requested < len(music)/30. else "music_end"
        while int(snapshot["tick"]) < end_tick and not snapshot["done"]:
            tick = int(snapshot["tick"])
            if pending is not None and pending["arrival_tick"] <= tick:
                try:
                    ack = self.backend.call("commit_plan", prepared_plan_id=pending["prepared_plan_id"],
                                            expected_control_tick=tick)
                    self.recorder.event("plan_committed", **pending, effective_tick=tick, acknowledgement=ack)
                except RemoteError as exc:
                    self.recorder.event("plan_rejected", **pending, effective_tick=tick, code=exc.code, detail=str(exc))
                pending = None
            if tick == next_decision:
                decision_id = (tick-self.warmup_ticks)//DECISION_TICKS
                if pending is not None:
                    self.recorder.event("decision_missed", episode_id=snapshot["episode_id"],
                                        decision_id=decision_id, tick=tick, reason="request_in_flight")
                else:
                    try:
                        pending = self._generate(snapshot, music, seed=seed,
                                                 decision_id=decision_id, budget_s=latency_budget_s)
                        pending["arrival_tick"] = tick if mode == "paused" else ceil_control_tick(
                            tick + pending["end_to_end_seconds"] * CLOCK_HZ)
                    except RemoteError as exc:
                        self.recorder.event("request_rejected", episode_id=snapshot["episode_id"],
                                            decision_id=decision_id, tick=tick, code=exc.code, detail=str(exc))
                next_decision += DECISION_TICKS
                # 暂停模式在当前控制边界提交，绝不先偷跑一个控制步。
                if pending is not None and pending["arrival_tick"] == tick:
                    continue
            stop_tick = min(end_tick, next_decision,
                            pending["arrival_tick"] if pending is not None else end_tick)
            if stop_tick <= tick:
                raise RuntimeError("Coordinator failed to make monotonic progress")
            steps = min(25, (stop_tick-tick)//CONTROL_TICKS)
            snapshot = self._advance(snapshot, steps, phase="music", decision_id=decision_id)
        if pending is not None:
            self.recorder.event("pending_invalidated", **pending, reason="episode_ended")
        return self.recorder.finish_episode(snapshot, snapshot.get("reason") or stop_reason)
