"""第九步上层环境适配器的 CPU 状态机及奖励对接测试。

用可计数的冻结执行替身代替 Isaac，在相同真实 trace 契约下检查 latency 等待旧参考、
错过决策栅格、计划拒绝、音乐终止、合法截断与缺失执行记录。样本替身固定已采集
去噪链而不调用神经网络，因此这些测试证明协调逻辑，不证明真实模型或动力学表现。
磁盘输出全部位于 pytest tmp_path；物理步数直接来自替身的推进记录，不能伪造 GPU
或机器人验收。
"""
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from gem.closedloop.dppo.buffer import RolloutBuffer
from gem.closedloop.dppo.env_adapter import ExecutionIntegrityError, UpperEnvironment
from gem.closedloop.dppo.rewards import ExecutionReward
from gem.runtime.closedloop_protocol import RemoteError
from tests.closedloop.dppo.test_data_learning import actual_step, context, music_fixture


class FakeBudget:
    def __init__(self):
        self.reservations = []
        self.actual = []

    def reserve(self, phase, **values):
        self.reservations.append((phase, values))

    def settle_control(self, phase, requested, result):
        self.actual.append(result["executed_control_steps"])


class FakeBackend:
    def __init__(self):
        self.sequence = 0
        self.session_id = "s"
        self.tick = 600
        self.calls = []
        self.corrupt_trace = False
        self.fail_at = None
        self.reject_commit = False

    def snapshot(self):
        failed = self.fail_at is not None and self.tick >= self.fail_at
        return dict(env_id=0, episode_id="e", tick=self.tick, done=failed, terminated=failed,
                    truncated=False, reason="physical_failure" if failed else None)

    def call(self, method, **payload):
        self.calls.append((method, copy.deepcopy(payload)))
        if method == "preview_prefix":
            return payload["request"]
        self.sequence += 1
        if method == "advance":
            begin = self.tick
            rows = []
            for _ in range(payload["control_steps"]):
                self.tick += 12
                rows.append(actual_step((self.tick - 612) // 12))
                if self.snapshot()["done"]:
                    break
            m = len(rows)
            if self.corrupt_trace:
                rows = rows[:-1]
            return dict(snapshot=self.snapshot(), trace=rows, executed_control_steps=m,
                        executed_physics_steps=4 * m, control_tick_begin=begin, control_tick_end=self.tick,
                        transition_valid=True, physics_count_exact=True)
        if method == "commit_plan":
            if self.reject_commit:
                raise RemoteError({"code": "late_plan", "message": "test rejection"})
            return dict(plan_id="p", installed_at=self.tick)
        if method == "discard_plan":
            return dict(discarded=True)
        raise AssertionError(method)


class FakeBuilder:
    def build(self, snapshot, reservation, music, music_start_tick):
        conditions = context()
        offset = snapshot["tick"] / 600. - 1.
        for key in ("proprio_history_times", "future_times", "decision_time"):
            conditions[key] += offset
        return conditions, {"request_id": reservation["request_id"]}


def adapter(tmp_path, *, elapsed=.75, mode="latency", prepared=True):
    backend = FakeBackend()
    config = {"stage9": {"latency_budget_s": 1., "execution_mode": mode, "episode_seconds": 30.,
                          "seed": 42, "run_id": "r"}}
    env = UpperEnvironment(config, backend, FakeBuilder(), SimpleNamespace(), FakeBudget(), tmp_path)
    env.snapshot = backend.snapshot()
    env.music = music_fixture()
    env.music_end_tick = 600 + 3000
    env.soft_end_tick = 600 + 30 * 600
    env.reward = ExecutionReward(music_features=env.music)
    sample = dict(context=context(), meta={"request_id": "req", "plan_id": "p", "parent_plan_id": "old"},
                  generated={}, trace={"chain": torch.zeros(1, 3, 120, 30), "old_log_probs": torch.zeros(1, 2, dtype=torch.float64),
                                       "free_mask": torch.ones(1, 120, 30, dtype=torch.bool)},
                  prepared={"prepared_plan_id": "ticket"} if prepared else None,
                  rejection=None if prepared else {"code": "invalid_plan_output"}, elapsed=elapsed,
                  seed=42, raw_path="synthetic_only")
    env.generate = lambda **kwargs: copy.deepcopy(sample)
    return env, backend


def test_latency_spans_multiple_control_batches_and_missed_decision(tmp_path):
    env, backend = adapter(tmp_path)
    result = env.step()
    assert result.executed_control_steps == 50 and result.executed_physics_steps == 200
    assert result.control_tick_begin == 600 and result.control_tick_end == 1200
    assert result.metadata["events"] == [{"kind": "decision_missed", "tick": 900}]
    commit_calls = [item for item in backend.calls if item[0] == "commit_plan"]
    assert len(commit_calls) == 1 and commit_calls[0][1]["expected_control_tick"] == 1056
    assert all(item[1]["control_steps"] <= 25 for item in backend.calls if item[0] == "advance")
    assert len(result.rewards) == 50
    assert sum(env.budget.actual) == 50
    RolloutBuffer().append(result)


def test_paused_reference_regression_stays_at_25_controls(tmp_path):
    env, backend = adapter(tmp_path, mode="paused")
    result = env.step()
    assert result.executed_control_steps == 25
    assert next(payload for name, payload in backend.calls if name == "commit_plan")["expected_control_tick"] == 600


@pytest.mark.parametrize("phase", ["prepare", "commit"])
def test_finite_rejected_plan_retains_chain_and_exactly_one_event(tmp_path, phase):
    env, backend = adapter(tmp_path, mode="paused", prepared=phase != "prepare")
    backend.reject_commit = phase == "commit"
    result = env.step()
    assert result.metadata["rejection"]
    assert result.chain.shape == (3, 120, 30)
    baseline_sum = sum(item["reward"] for item in result.metadata["reward_details"])
    assert result.rewards.sum().item() == pytest.approx(baseline_sum - 1.)
    assert result.metadata["event_reward"] == 0
    assert result.transition_valid


def test_physical_failure_while_waiting_keeps_failure_step_and_discards_pending(tmp_path):
    env, backend = adapter(tmp_path)
    backend.fail_at = 720
    result = env.step()
    assert result.terminated and not result.truncated and result.next_context is None
    assert result.executed_control_steps == 10 and len(result.rewards) == 10
    assert not any(name == "commit_plan" for name, _ in backend.calls)
    assert any(name == "discard_plan" for name, _ in backend.calls)
    baseline = sum(item["reward"] for item in result.metadata["reward_details"])
    assert result.rewards.sum() == pytest.approx(baseline - 5.)
    RolloutBuffer().append(result)


def test_music_end_and_admin_cutoff_have_different_bootstrap(tmp_path):
    env, backend = adapter(tmp_path / "music")
    env.music_end_tick = 720
    result = env.step()
    assert result.terminated and result.reason == "music_end" and result.next_context is None
    assert result.rewards.sum() == pytest.approx(sum(item["reward"] for item in result.metadata["reward_details"]))
    capped, _ = adapter(tmp_path / "capped")
    capped.soft_end_tick = 900
    result = capped.step()
    assert result.truncated and not result.terminated
    assert result.control_tick_end == 1200 and result.next_context is not None
    RolloutBuffer().append(result)


def test_missing_execution_trace_cannot_enter_rewards_or_buffer(tmp_path):
    env, backend = adapter(tmp_path)
    backend.corrupt_trace = True
    with pytest.raises(ExecutionIntegrityError, match=r"len\(trace\)"):
        env.step()
    assert env.snapshot["tick"] == 600
    assert backend.tick > 600


def test_bootstrap_preview_does_not_reserve_another_prefix(tmp_path):
    env, backend = adapter(tmp_path)
    result = env.step()
    assert result.next_context is not None
    assert [name for name, _ in backend.calls].count("preview_prefix") == 1
    assert [name for name, _ in backend.calls].count("reserve_prefix") == 0


def test_comparison_initial_noise_matches_despite_different_run_counters(tmp_path):
    def prepare_env(directory, *, attempt, iteration):
        env, backend = adapter(directory)
        del env.generate  # 使用真实 generate；只替换 Actor 和执行器，核对实际 seed 路径。
        original_call = backend.call
        def call(method, **payload):
            if method == "reserve_prefix":
                return payload["request"]
            if method == "prepare_plan":
                return {"prepared_plan_id": "ticket"}
            return original_call(method, **payload)
        backend.call = call
        env.builder = SimpleNamespace(build=lambda snapshot, reservation, *args, **kwargs:
            (context(), {**reservation, "parent_plan_id": "old", "world_anchor": [0., 0., 0., 0.]}))
        actor = torch.nn.Linear(1, 1)
        actor.endecoder = SimpleNamespace(codec=SimpleNamespace(apply_world_anchor=lambda q, anchor: q))
        def output(noise):
            qpos = torch.zeros(1, 120, 28)
            qpos[..., 3] = 1.
            return {"qpos": qpos, "qpos30": noise, "contact": torch.full((1, 120, 2), .5)}
        actor.sample = lambda batch, *, steps, guidance_scale, noise: output(noise)
        env.policy = SimpleNamespace(actor=actor, steps=20, guidance_scale=2.5,
            sample_rollout=lambda batch, *, generator: output(torch.randn(1, 120, 30, generator=generator)))
        env.attempt, env.iteration = attempt, iteration
        env.episode_count = attempt + 2
        env.comparison_noise_index = 0
        return env
    baseline = prepare_env(tmp_path / "deterministic", attempt=0, iteration=0)
    stochastic = prepare_env(tmp_path / "stochastic", attempt=88, iteration=1)
    a = baseline.generate(deterministic=True)
    b = stochastic.generate()
    assert a["seed"] == b["seed"]
    assert a["meta"]["seed"] == b["meta"]["seed"]
    torch.testing.assert_close(torch.from_numpy(a["generated"]["qpos30"]), torch.from_numpy(b["generated"]["qpos30"]), rtol=0, atol=0)
    stochastic.comparison_noise_index = None
    training = stochastic.generate()
    assert training["seed"] != b["seed"]
