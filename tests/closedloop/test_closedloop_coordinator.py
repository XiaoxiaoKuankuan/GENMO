"""验证冻结闭环协调器的整数时钟、异步到达与有界转移语义。

使用可控墙钟、模拟 RPC 后端和模拟 Actor，不创建真实模型、Isaac 或 GPU 任务。条件
构造仍使用现有 Bumi codec 和 OnlineConditionBuilder，确保协调器实际发送的是原十个
Stage1 条件字段。fake backend 明确记录每次控制所消费的计划，晚到请求在提交时拒绝，
测试不会用修改过去时间的方法掩盖错误。覆盖暂停前发布、延迟边界、漏决策、短转移、
跨 episode 错误及 25 控制步/100 物理步关系。全部数据只在内存中构造。
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

import gem.closedloop.coordinator as coordinator_module
from gem.closedloop.contracts import STAGE1_CONDITION_KEYS
from gem.closedloop.coordinator import ClosedLoopCoordinator, ceil_control_tick
from gem.closedloop.online_conditions import OnlineConditionBuilder
from gem.robots.bumi.feature_codec import BumiMotionFeatureCodec
from gem.robots.bumi.kinematics import BumiKinematics
from gem.runtime.closedloop_protocol import RemoteError

REPO = Path(__file__).resolve().parents[2]


class FakeClock:
    def __init__(self):
        self.wall = 0.0

    def now(self):
        return self.wall


class RecorderSpy:
    def __init__(self):
        self.events, self.steps, self.plans, self.episodes = [], [], [], []

    def start_episode(self, sample, seed, mode, snapshot):
        self.episodes.append({"seed": seed, "mode": mode, "episode_id": snapshot["episode_id"]})

    def event(self, kind, **fields):
        self.events.append({"event": kind, **copy.deepcopy(fields)})

    def step(self, row):
        self.steps.append(copy.deepcopy(row))

    def plan(self, generated):
        self.plans.append(copy.deepcopy(generated))

    def finish_episode(self, snapshot, reason):
        return {"snapshot": copy.deepcopy(snapshot), "reason": reason}

    def of_type(self, kind):
        return [event for event in self.events if event["event"] == kind]


class ActorSpy:
    def __init__(self, clock, durations):
        self.clock, self.durations = clock, list(durations)
        self.requests = []

    def call(self, method, *, conditions, meta):
        assert method == "generate"
        assert set(conditions) == set(STAGE1_CONDITION_KEYS)
        assert all(isinstance(value, np.ndarray) for value in conditions.values())
        self.requests.append((copy.deepcopy(conditions), copy.deepcopy(meta)))
        elapsed = self.durations[min(len(self.requests) - 1, len(self.durations) - 1)]
        self.clock.wall += elapsed
        return {**meta, "qpos_world": np.zeros((120, 28), dtype=np.float32),
                "qpos30": np.zeros((120, 30), dtype=np.float32),
                "contact": np.zeros((120, 2), dtype=np.float32), "latency_s": elapsed}


class BackendSpy:
    def __init__(self, qpos, clock, *, warmup=600):
        self.qpos, self.clock, self.warmup = qpos, clock, warmup
        self.tick, self.episode_id = 0, 0
        self.done, self.reason = False, None
        self.active_plan = "bootstrap"
        self.calls, self.prepared, self.commits = [], {}, []
        self.failure_tick = None
        self.fault = None
        self.prepare_seconds = 0.0

    def snapshot(self):
        ticks = self.tick - np.arange(49, -1, -1, dtype=np.int64) * 12
        valid = ticks >= 0
        history = np.repeat(ticks[:, None] / 600.0, 48, axis=1).astype(np.float32)
        history[~valid] = 0
        return {"env_id": 0, "episode_id": self.episode_id, "tick": self.tick,
                "history_values": history, "history_valid": valid, "history_ticks": ticks,
                "done": self.done, "reason": self.reason, "terminated": self.done,
                "truncated": False}

    def call(self, method, **fields):
        self.calls.append((method, copy.deepcopy(fields), self.tick))
        if method == "reset_episode":
            self.episode_id += 1
            self.tick, self.done, self.reason, self.active_plan = 0, False, None, "bootstrap"
            self.prepared.clear()
            return self.snapshot()
        if method == "reserve_prefix":
            request = fields["request"]
            assert request["decision_tick"] == self.tick
            prefix = max(request["min_prefix"], int(np.ceil((request["deadline_tick"] - self.tick + 132) / 20)) + 1)
            return {**request, "parent_plan_id": self.active_plan, "prefix_frames": prefix,
                    "protected_end_tick": self.tick + (prefix - 2) * 20,
                    "source_qpos": np.repeat(self.qpos[None], prefix, axis=0),
                    "source_ticks": self.tick + np.arange(prefix, dtype=np.int64) * 20,
                    "reference_anchor_qpos": self.qpos.copy()}
        if method == "prepare_plan":
            generated = fields["generated_plan"]
            self.clock.wall += self.prepare_seconds
            ticket = f"ticket:{len(self.prepared)}"
            self.prepared[ticket] = copy.deepcopy(generated)
            return {"prepared_plan_id": ticket, "prepare_seconds": self.prepare_seconds,
                    "max_position_error": 0.0, "max_velocity_error": 0.0}
        if method == "commit_plan":
            generated = self.prepared[fields["prepared_plan_id"]]
            assert fields["expected_control_tick"] == self.tick
            if self.tick > generated["deadline_tick"]:
                raise RemoteError({"code": "deadline_exceeded", "message": "candidate arrived too late"})
            if generated["episode_id"] != self.episode_id:
                raise RemoteError({"code": "stale_episode", "message": "candidate belongs to old episode"})
            self.active_plan = generated["plan_id"]
            self.commits.append((self.tick, self.active_plan))
            return {"plan_id": self.active_plan, "effective_tick": self.tick}
        if method == "discard_plan":
            del self.prepared[fields["prepared_plan_id"]]
            return {"discarded": True}
        if method == "advance":
            assert fields["expected_episode_id"] == self.episode_id
            requested = fields["control_steps"]
            count = requested - 1 if self.fault == "short_nonterminal" else requested
            trace = []
            for _ in range(count):
                if self.done:
                    break
                begin = self.tick
                self.tick += 12
                trace.append({"episode_id": self.episode_id, "control_tick": begin,
                              "reference_tick": begin, "reference_plan_id": self.active_plan})
                if self.failure_tick is not None and self.tick >= self.failure_tick:
                    self.done, self.reason = True, "global_anchor_ori"
            snapshot = self.snapshot()
            actual = len(trace)
            physics = 4 * actual
            if self.fault == "wrong_physics":
                physics += 1
            elif self.fault == "wrong_tick":
                snapshot["tick"] += 1
            elif self.fault == "reset_inside":
                snapshot["episode_id"] += 1
            elif self.fault == "wrong_trace_episode" and trace:
                trace[-1]["episode_id"] += 1
            return {"snapshot": snapshot, "trace": trace,
                    "executed_control_steps": actual, "executed_physics_steps": physics}
        raise AssertionError(method)


@pytest.fixture
def harness(monkeypatch):
    def make(durations=(0.1,), *, warmup=600):
        codec = BumiMotionFeatureCodec(BumiKinematics(REPO / "configs/bumi/bumi_kinematics_robot_retargeter_fe934_v1.json"))
        builder = OnlineConditionBuilder(codec)
        clock = FakeClock()
        monkeypatch.setattr(coordinator_module.time, "perf_counter", clock.now)
        actor = ActorSpy(clock, durations)
        backend = BackendSpy(codec.kinematics.default_qpos.numpy().copy(), clock, warmup=warmup)
        recorder = RecorderSpy()
        config = {"timing": {"latency_guard_s": .04, "min_prefix_frames": 12,
                             "warmup_ticks": warmup, "calibration_warmup": 10, "calibration_samples": 100},
                  "evaluation": {"seconds": 30}}
        coordinator = ClosedLoopCoordinator(config, actor, backend, builder, recorder)
        return coordinator, actor, backend, recorder
    return make


SAMPLE = {"row": {"sample_id": "mine_test_song"}, "dataset": "Mine"}
MUSIC = np.ones((300, 35), dtype=np.float32)


def test_paused_plan_commits_before_first_music_step_with_25_100_counts(harness):
    coordinator, actor, backend, recorder = harness([.7])
    report = coordinator.run_episode(SAMPLE, MUSIC, seed=42, mode="paused", latency_budget_s=.2, seconds=.5)
    assert backend.commits == [(600, "1:decision:0:plan")]
    music_steps = [row for row in recorder.steps if row["phase"] == "music"]
    assert len(music_steps) == 25
    assert all(row["reference_plan_id"] == "1:decision:0:plan" for row in music_steps)
    assert recorder.of_type("advance")[-1]["executed_physics_steps"] == 100
    assert recorder.of_type("advance")[-1]["executed_control_steps"] == 25
    assert report["snapshot"]["tick"] == 900
    assert len(actor.requests) == 1
    conditions, meta = actor.requests[0]
    assert set(conditions) == set(STAGE1_CONDITION_KEYS)
    assert conditions["decision_time"].item() == 1.0
    assert meta["decision_tick"] == 600 and meta["seed"] == 42


def test_latency_includes_prepare_and_commits_at_next_control_boundary(harness):
    coordinator, actor, backend, recorder = harness([.12])
    backend.prepare_seconds = .011
    coordinator.run_episode(SAMPLE, MUSIC, seed=42, mode="latency", latency_budget_s=.2, seconds=.5)
    assert backend.commits == [(684, "1:decision:0:plan")]
    prepared = recorder.of_type("plan_prepared")[0]
    assert prepared["end_to_end_seconds"] == pytest.approx(.131)
    assert prepared["deadline_tick"] == 720
    music_steps = [row for row in recorder.steps if row["phase"] == "music"]
    assert all(row["reference_plan_id"] == "bootstrap" for row in music_steps if row["control_tick"] < 684)
    assert all(row["reference_plan_id"] == "1:decision:0:plan" for row in music_steps if row["control_tick"] >= 684)
    assert sum(event["executed_physics_steps"] for event in recorder.of_type("advance") if event["phase"] == "music") == 100


def test_late_plan_is_rejected_without_rewriting_already_consumed_reference(harness):
    coordinator, actor, backend, recorder = harness([.31])
    coordinator.run_episode(SAMPLE, MUSIC, seed=42, mode="latency", latency_budget_s=.1, seconds=.5)
    assert backend.commits == []
    rejected = recorder.of_type("plan_rejected")
    assert len(rejected) == 1 and rejected[0]["code"] == "deadline_exceeded"
    assert rejected[0]["effective_tick"] == 792
    assert all(row["reference_plan_id"] == "bootstrap" for row in recorder.steps)
    ticks = [row["reference_tick"] for row in recorder.steps]
    assert ticks == list(range(0, 900, 12))


def test_inflight_request_crossing_decision_grid_records_misses_without_catchup(harness):
    coordinator, actor, backend, recorder = harness([1.1])
    coordinator.run_episode(SAMPLE, MUSIC, seed=42, mode="latency", latency_budget_s=1.2, seconds=1.5)
    assert len(actor.requests) == 1
    assert [event["tick"] for event in recorder.of_type("decision_missed")] == [900, 1200]
    assert backend.commits == [(1260, "1:decision:0:plan")]
    assert all(1 <= event["requested_control_steps"] <= 25 for event in recorder.of_type("advance"))


@pytest.mark.parametrize("failure_step", [1, 7, 24])
def test_early_failure_stops_current_episode_without_reset_or_tail_steps(harness, failure_step):
    coordinator, actor, backend, recorder = harness([.01])
    backend.failure_tick = 600 + failure_step * 12
    report = coordinator.run_episode(SAMPLE, MUSIC, seed=42, mode="paused", latency_budget_s=.2, seconds=1.)
    last = recorder.of_type("advance")[-1]
    assert last["executed_control_steps"] == failure_step
    assert last["executed_physics_steps"] == failure_step * 4
    assert report["snapshot"]["episode_id"] == 1
    assert report["snapshot"]["tick"] == 600 + failure_step * 12
    assert report["reason"] == "global_anchor_ori"
    assert sum(method == "reset_episode" for method, _, _ in backend.calls) == 1
    assert len(actor.requests) == 1


def test_failure_while_waiting_invalidates_candidate_and_never_commits(harness):
    coordinator, _, backend, recorder = harness([.3])
    backend.failure_tick = 624
    coordinator.run_episode(SAMPLE, MUSIC, seed=42, mode="latency", latency_budget_s=.4, seconds=1.)
    assert not backend.commits
    assert len(recorder.of_type("pending_invalidated")) == 1
    assert all(row["reference_plan_id"] == "bootstrap" for row in recorder.steps)


@pytest.mark.parametrize("fault,message", [
    ("wrong_physics", "control/physics"),
    ("wrong_tick", "consumed time"),
    ("reset_inside", "reset inside"),
    ("wrong_trace_episode", "different episode"),
    ("short_nonterminal", "short nonterminal"),
])
def test_invalid_backend_feedback_is_rejected(harness, fault, message):
    coordinator, _, backend, _ = harness()
    backend.fault = fault
    with pytest.raises(RuntimeError, match=message):
        coordinator._advance(backend.snapshot(), 25, phase="music")


def test_startup_failure_is_reported_and_does_not_sample_actor(harness):
    coordinator, actor, backend, _ = harness()
    backend.failure_tick = 84
    report = coordinator.run_episode(SAMPLE, MUSIC, seed=42, mode="paused", latency_budget_s=.2)
    assert report["reason"] == "startup_failure"
    assert report["snapshot"]["tick"] == 84 and not actor.requests


def test_calibration_does_not_advance_or_publish_each_candidate(harness):
    coordinator, actor, backend, recorder = harness([.01, .02, .1, .2, .3])
    result = coordinator.calibrate(SAMPLE, MUSIC, warmup=2, samples=3)
    assert len(actor.requests) == 5 and not backend.commits
    assert not backend.prepared
    assert backend.tick == 600
    assert result["p95_seconds"] == pytest.approx(.29)
    assert result["latency_budget_seconds"] == pytest.approx(.33)
    assert not result["full_calibration"]
    assert len(recorder.of_type("calibration")) == 1


def test_reference_consumption_remains_associated_with_old_plan_until_arrival(harness):
    coordinator, _, backend, recorder = harness([.08, .12])
    coordinator.run_episode(SAMPLE, MUSIC, seed=43, mode="latency", latency_budget_s=.2, seconds=1.)
    assert backend.commits == [(648, "1:decision:0:plan"), (972, "1:decision:1:plan")]
    waiting = [row for row in recorder.steps if 900 <= row["control_tick"] < 972]
    assert waiting and all(row["reference_plan_id"] == "1:decision:0:plan" for row in waiting)


@pytest.mark.parametrize("value,expected", [(0, 0), (600, 600), (600.01, 612), (611.99, 612), (612, 612)])
def test_control_tick_ceiling(value, expected):
    assert ceil_control_tick(value) == expected
