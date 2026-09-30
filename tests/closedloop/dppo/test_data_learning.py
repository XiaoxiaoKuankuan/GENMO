"""第九步数据、奖励、独立 Critic、时间回报和训练音乐的 CPU 验收。

这些测试使用小型可解析高斯链、手算变长回报、可控实际执行轨迹及四库 train 音乐
夹具。重点验证 RPC 分段不改变奖励、缺失诊断不能成为有效样本、Buffer 与活跃历史
断开、执行日志在重开后仍幂等、Critic 不受未知占位污染，以及恢复音乐采样的精确
后续顺序。测试不加载真实大模型、不运行 Isaac/GPU、不把合成结果当成动力学验收。
全部磁盘夹具只写 pytest tmp_path，缓存由本轮统一命令放入系统临时目录并清理。
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

import numpy as np
import pytest
import torch

from gem.closedloop.dppo.buffer import RolloutBuffer, StepJournal, UpperTransition
from gem.closedloop.dppo.critic import UpperCritic
from gem.closedloop.dppo.music_tasks import SOURCES, TrainMusicSampler
from gem.closedloop.dppo.returns import compute_gae
from gem.closedloop.dppo.rewards import ExecutionReward


def context(batch=1):
    return {
        "music_features": torch.zeros(batch, 120, 35), "music_valid": torch.ones(batch, 120, dtype=torch.bool),
        "proprio_history": torch.zeros(batch, 50, 48), "proprio_history_valid": torch.ones(batch, 50, dtype=torch.bool),
        "proprio_history_times": (torch.arange(50, dtype=torch.float64) / 50 + .02).expand(batch, -1).clone(),
        "known_qpos30": torch.zeros(batch, 120, 30), "known_qpos30_mask": torch.zeros(batch, 120, 30, dtype=torch.bool),
        "future_valid": torch.ones(batch, 120, dtype=torch.bool),
        "future_times": (torch.arange(120, dtype=torch.float64) / 30 + 1.).expand(batch, -1).clone(),
        "decision_time": torch.ones(batch, dtype=torch.float64),
    }


def transition():
    return UpperTransition(
        identity={"run_id": "r", "backend_session_id": "s", "episode_id": "e", "decision_id": 1, "policy_version": 0},
        context=context(), next_context=context(), chain=torch.zeros(3, 120, 30, requires_grad=True),
        old_log_prob=torch.zeros(2, dtype=torch.float64), free_mask=torch.ones(120, 30, dtype=torch.bool),
        rewards=torch.tensor([.1, .2]), old_value=0., next_value=1., control_tick_begin=600,
        control_tick_end=624, executed_control_steps=2, executed_physics_steps=8,
        metadata={"mutable": np.zeros(3)},
    )


def test_buffer_detaches_copies_and_keeps_one_policy():
    item = transition()
    buffer = RolloutBuffer(2)
    buffer.append(item)
    with torch.no_grad():
        item.chain.add_(1)
    item.context["music_features"].add_(5)
    item.metadata["mutable"][0] = 8
    stored = buffer.transitions[0]
    assert not stored.chain.requires_grad and stored.chain.device.type == "cpu"
    assert stored.chain.sum() == 0 and stored.context["music_features"].sum() == 0
    assert stored.metadata["mutable"][0] == 0
    item.identity["policy_version"] = 1
    with pytest.raises(ValueError, match="policy versions"):
        buffer.append(item)
    buffer.clear()
    buffer.append(item)
    assert len(buffer) == 1


@pytest.mark.parametrize("field,value", [("executed_physics_steps", 7), ("control_tick_end", 636),
                                         ("old_log_prob", torch.zeros(2)), ("chain", torch.zeros(3, 120, 30, dtype=torch.float16))])
def test_buffer_rejects_fake_complete_or_quantized_data(field, value):
    item = transition()
    setattr(item, field, value)
    with pytest.raises(ValueError):
        RolloutBuffer().append(item)


def test_single_environment_policy_shapes_are_normalized():
    item = transition()
    item.chain, item.old_log_prob, item.free_mask = item.chain[None], item.old_log_prob[None], item.free_mask[None]
    buffer = RolloutBuffer(1)
    buffer.append(item)
    assert buffer.transitions[0].chain.shape == (3, 120, 30)
    with pytest.raises(OverflowError):
        buffer.append(item)


def test_execution_journal_durable_replay_and_conflicting_reply(tmp_path):
    path = tmp_path / "steps.sqlite"
    reply = {"backend_session_id": "session1", "mutation_seq": 4, "ok": True,
             "result": {"episode_id": "episode1", "trace": [{"tick": 612, "q": np.arange(3.)}]}}
    with StepJournal(path) as journal:
        assert journal.append_result(reply)
        assert not journal.append_result(copy.deepcopy(reply))
    with StepJournal(path) as journal:
        assert len(journal) == 1
        assert not journal.append_result(reply)
        corrupt = copy.deepcopy(reply)
        corrupt["result"]["trace"][0]["tick"] = 624
        with pytest.raises(ValueError, match="different payload"):
            journal.append_result(corrupt)
        failed = {"backend_session_id": "session1", "mutation_seq": 5, "ok": False,
                  "result": None, "error": {"partial_omega": float("nan")}}
        assert journal.append_result(failed)
        assert len(journal) == 2
        assert "__nonfinite__" in journal.connection.execute("SELECT payload FROM replies WHERE payload LIKE '%__nonfinite__%'").fetchone()[0]


def test_semimarkov_returns_manual_variable_steps_and_fixed_targets():
    gamma, lam = .99 ** (1 / 25), .95 ** (1 / 25)
    out = compute_gae([[1., 2.], [3.]], [4., 5.], [5., 7.], [2, 1], [True, False], [True, False])
    d1 = 3 - 5
    d0 = 1 + gamma * 2 + gamma ** 2 * 5 - 4
    a0 = d0 + (gamma * lam) ** 2 * d1
    torch.testing.assert_close(out["advantages_raw"], torch.tensor([a0, d1], dtype=torch.float64))
    torch.testing.assert_close(out["returns"], torch.tensor([a0 + 4, 3], dtype=torch.float64))
    assert out["advantages"].mean().abs() < 1e-12
    assert not out["returns"].requires_grad


def test_time_discount_reward_split_invariance():
    gamma = .99 ** (1 / 25)
    rewards = torch.arange(25, dtype=torch.float64) / 100
    full = compute_gae([rewards], [0.], [0.], [25], [False], [False])
    parts = compute_gae([rewards[:7], rewards[7:]], [0., 0.], [0., 0.], [7, 18], [False, False], [False, False])
    torch.testing.assert_close(full["discounted_rewards"][0], parts["discounted_rewards"][0] + gamma ** 7 * parts["discounted_rewards"][1])


def test_truncation_bootstrap_is_separate_from_recursion_and_invalid_barrier():
    out = compute_gae([[1.], [], [2.]], [0., float("nan"), 0.], [9., float("nan"), 4.], [1, 0, 1],
                      [True, False, True], [True, False, False], valid=[True, False, True])
    assert out["advantages_raw"][0] == pytest.approx(1 + .99 ** (1 / 25) * 9)
    assert out["advantages_raw"][1] == 0
    assert out["returns"][2] == pytest.approx(2 + .99 ** (1 / 25) * 4)
    assert torch.isfinite(out["returns"]).all()


def test_zero_step_rejection_event_and_event_discount():
    out = compute_gae([[], [0., 0.]], [0., 0.], [0., 0.], [0, 2], [False, False], [False, False], event_rewards=[-1., -5.])
    assert out["returns"][0] == -1
    assert out["returns"][1] == pytest.approx(-5 * .99 ** (1 / 25))
    with pytest.raises(ValueError, match="actual executed"):
        compute_gae([[1]], [0.], [0.], [2], [False], [False])
    with pytest.raises(ValueError, match="beyond its buffer"):
        compute_gae([[1]], [0.], [0.], [1], [False], [True])


def test_critic_masks_unknown_padding_and_empty_history():
    torch.manual_seed(13)
    critic = UpperCritic()
    original = context()
    original["music_valid"][:, 60:] = False
    original["proprio_history_valid"].zero_()
    changed = copy.deepcopy(original)
    changed["proprio_history"].fill_(10000)
    changed["known_qpos30"].fill_(-10000)
    changed["music_features"][:, 60:] = 10000
    torch.testing.assert_close(critic(original, [10.]), critic(changed, [10.]), atol=0, rtol=0)
    assert critic(original, [0.]).shape == (1,)
    assert not torch.equal(critic(original, [0.]), critic(original, [10.]))
    with pytest.raises(ValueError, match="ten Actor"):
        critic({**original, "target_qpos30": torch.zeros(1, 120, 30)}, [10.])


def test_value_loss_updates_only_critic_and_copies_statistics():
    actor = torch.nn.Linear(30, 30)
    before = copy.deepcopy(actor.state_dict())
    mean = torch.arange(30.)
    critic = UpperCritic(qpos_mean=mean)
    mean.add_(100)
    assert critic.qpos_mean[0] == 0
    opt = torch.optim.Adam(critic.parameters(), lr=1e-4)
    old = critic.value_head[-1].weight.detach().clone()
    loss = (critic(context(), [8.]) - torch.tensor([2.])).square().mean()
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0 for p in critic.parameters())
    assert all(p.grad is None for p in actor.parameters())
    opt.step()
    assert not torch.equal(old, critic.value_head[-1].weight)
    assert all(torch.equal(value, before[key]) for key, value in actor.state_dict().items())


def actual_step(index=0, *, speed=0., position=0.):
    names = ["root", "left_foot", "right_foot", "left_elbow", "right_elbow"]
    tick = 612 + 12 * index
    physical = {"body_names": names, "foot_body_names": names[1:3],
                "body_link_lin_vel_w": np.zeros((5, 3)), "pd_torque_estimate_nm": np.zeros(21),
                "joint_effort_limits_nm": np.ones(21) * 40, "joint_velocity_limits_rad_s": np.ones(21) * 10,
                "joint_pos_limits_rad": np.tile([-2., 2.], (21, 1)), "soft_joint_pos_limits_rad": None,
                "foot_net_contact_forces_w_n": np.array([[0., 0., 20.], [0., 0., 20.]]),
                "foot_min_support_clearance_m": np.zeros(2), "contact_body_names": names,
                "allowed_contact_body_names": names[1:], "undesired_contact_force_threshold_n": 1.,
                "foot_contact_force_threshold_n": 10.,
                "net_contact_forces_w_n": np.array([[0., 0., 0.], [0., 0., 20.], [0., 0., 20.], [0., 0., 0.], [0., 0., 0.]])}
    substeps = []
    for substep in range(4):
        physics_tick = tick - 12 + (substep + 1) * 3
        diagnostic = copy.deepcopy(physical)
        diagnostic["mechanical_power_pd_estimate"] = {"torque_sample_tick": physics_tick - 3,
            "velocity_sample_tick": physics_tick, "time_s": physics_tick / 600,
            "sampling_synchronized": False}
        substeps.append({"physics_tick": physics_tick, "dt_s": .005, "physical_diagnostics": diagnostic,
                         "joint_vel_gmt": np.full(21, speed), "joint_position_target": np.zeros(21)})
    physical["mechanical_power_pd_estimate"] = copy.deepcopy(substeps[-1]["physical_diagnostics"]["mechanical_power_pd_estimate"])
    return {"episode_id": "e", "tick": tick, "control_tick_begin": tick - 12,
            "completed_physics_steps": 4, "transition_valid": True, "state_valid": True,
            "terminated": False, "truncated": False,
            "actual_joint_pos_gmt": np.full(21, position), "actual_joint_vel_gmt": np.full(21, speed),
            "actual_qpos": np.r_[0., 0., 1., 1., 0., 0., 0., np.full(21, position)],
            "actual_root_ang_vel_b": np.zeros(3), "joint_position_target": np.zeros(21),
            "reference": {"joint_pos": np.full(21, position), "joint_vel": np.full(21, speed),
                          "body_pos_w": np.array([[0., 0., 1.]] * 3), "body_quat_w": np.array([[1., 0., 0., 0.]] * 3)},
            "errors": {"joint_position_rmse_rad": 0., "joint_velocity_rmse_rad_s": 0., "root_position_error_m": 0.,
                       "root_height_error_m": 0., "non_yaw_orientation_error_rad": 0.,
                       "yaw_error_rad": 0., "end_effector_relative_height_error_m": 0.},
            "reference_consistency": {"valid": True, "joint_vel_rms_rad_s": 0., "root_lin_vel_rms_m_s": 0., "root_ang_vel_rms_rad_s": 0.},
            "physical_diagnostics": physical, "physics_substeps": substeps}


def target_activity_fixture(tick):
    count = min(25, (tick - 600) // 12)
    return {"valid": True, "activity_rad_s": 1., "window_count": count, "window_complete": count == 25,
            "window_begin_tick": tick - count * 12, "window_end_tick": tick,
            "source": {"kind": "paired_train_fixture", "sample_id": "fixture", "velocity_unit": "rad/s"}}


def music_fixture():
    features = np.zeros((150, 35))
    features[:, 0] = .5
    features[::15, 34] = 1.
    return features


def test_actual_tracking_and_stability_reward_prefers_controlled_pose():
    good = ExecutionReward(music_features=music_fixture(), target_activity=target_activity_fixture).evaluate_step(actual_step(speed=1.))
    bad_row = actual_step(speed=1.)
    bad_row["errors"]["root_height_error_m"] = .5
    bad_row["errors"]["non_yaw_orientation_error_rad"] = 1.
    bad_row["errors"]["joint_position_rmse_rad"] = 1.
    bad = ExecutionReward(music_features=music_fixture(), target_activity=target_activity_fixture).evaluate_step(bad_row)
    assert good["transition_valid"] and bad["transition_valid"]
    assert good["reward"] == pytest.approx(.092)
    assert bad["reward"] < good["reward"]


def test_missing_or_nonfinite_actual_diagnostics_marks_invalid():
    missing = actual_step()
    del missing["errors"]["root_height_error_m"]
    result = ExecutionReward(music_features=music_fixture(), target_activity=target_activity_fixture).evaluate_step(missing)
    assert not result["transition_valid"] and not result["components"]["stable"]["valid"]
    invalid = actual_step()
    invalid["physics_substeps"][0]["physical_diagnostics"]["pd_torque_estimate_nm"][0] = float("nan")
    result = ExecutionReward(music_features=music_fixture(), target_activity=target_activity_fixture).evaluate_step(invalid)
    assert not result["transition_valid"] and any("nonfinite" in error for error in result["errors"])


def test_substep_peak_is_not_replaced_with_last_control_snapshot():
    row = actual_step()
    row["physics_substeps"][0]["physical_diagnostics"]["pd_torque_estimate_nm"][0] = 40
    result = ExecutionReward(music_features=music_fixture(), target_activity=target_activity_fixture).evaluate_step(row)
    assert result["transition_valid"]
    raw = result["components"]["torque"]["raw"]
    assert raw["max_torque_ratio"] == 1
    assert raw["sampling"] == "four_physics_substeps"
    row.pop("physics_substeps")
    assert not ExecutionReward(music_features=music_fixture(), target_activity=target_activity_fixture).evaluate_step(row)["transition_valid"]
    result = ExecutionReward({"require_substeps": False}, music_features=music_fixture(), target_activity=target_activity_fixture).evaluate_step(row)
    assert result["components"]["torque"]["raw"]["sampling"] == "control_end_proxy"


def test_stationary_music_reward_is_gated_and_high_activity_bounded():
    stationary = ExecutionReward(music_features=music_fixture(), target_activity=target_activity_fixture)
    active = ExecutionReward(music_features=music_fixture(), target_activity=target_activity_fixture)
    for i in range(70):
        zero = stationary.evaluate_step(actual_step(i))
        moving = active.evaluate_step(actual_step(i, speed=1e3 * (1. + abs(math.sin(i))), position=math.sin(i)))
    assert zero["components"]["music"]["valid"]
    assert zero["components"]["music"]["weighted_rate"] == 0
    assert zero["components"]["track"]["weighted_rate"] == 0
    assert 0 <= moving["components"]["music"]["score"] <= 1


def test_music_reward_is_causal_and_rpc_segmentation_independent():
    traces = [actual_step(i, speed=abs(math.sin(i / 4)), position=.3 * math.cos(i / 4)) for i in range(75)]
    one = ExecutionReward(music_features=music_fixture(), target_activity=target_activity_fixture)
    split = ExecutionReward(music_features=music_fixture(), target_activity=target_activity_fixture)
    expected = [one.evaluate_step(row)["reward"] for row in traces]
    actual = []
    for segment in (traces[:7], traces[7:25], traces[25:]):
        actual.extend(split.evaluate_step(row)["reward"] for row in segment)
    np.testing.assert_array_equal(actual, expected)
    changed = music_fixture()
    changed[100:] = 300
    other = ExecutionReward(music_features=changed, target_activity=target_activity_fixture)
    assert [other.evaluate_step(row)["reward"] for row in traces] == expected
    with pytest.raises(ValueError, match="continuous"):
        split.evaluate_step(traces[-1])
    assert one.event_reward("plan_rejected") == -.5 and one.event_reward("physical_failure") == -5


def write_music_tasks(tmp_path):
    selection = []
    for source in SOURCES:
        directory = tmp_path / "data" / source
        (directory / "manifests").mkdir(parents=True)
        rows = []
        for index in range(2):
            relative = f"music_{index}.pt"
            torch.save(torch.zeros(150, 35), directory / relative)
            group = f"{source}:{index}"
            row = {"sample_id": str(index), "split": "train", "fps": 30, "num_frames": 150,
                   "music_feature_path": relative, "source_music_feature_sha256": hashlib.sha256((directory / relative).read_bytes()).hexdigest(),
                   "source_audio_sha256": hashlib.sha256(group.encode()).hexdigest(),
                   "resplit_provenance": {"group_id": group}}
            rows.append(row)
        manifest = directory / "manifests/train.jsonl"
        manifest.write_text("\n".join(json.dumps(row) for row in rows))
        for row in rows:
            selection.append({"dataset": source, "group_id": row["resplit_provenance"]["group_id"],
                              "row": row, "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()})
    path = tmp_path / "corrected_selection.json"
    path.write_text(json.dumps(selection))
    return tmp_path / "data", path


def test_train_music_first_four_cover_sources_and_resume_exactly(tmp_path):
    root, selection = write_music_tasks(tmp_path)
    sampler = TrainMusicSampler(root, selection)
    assert [sampler.next_task()["dataset"] for _ in range(4)] == list(SOURCES)
    state = sampler.state_dict()
    expected = [sampler.next_task()["group_id"] for _ in range(25)]
    restored = TrainMusicSampler(root, selection, seed=777)
    restored.load_state_dict(state)
    assert [restored.next_task()["group_id"] for _ in range(25)] == expected
    sample = restored.next_task()
    assert sample["music"].shape == (150, 35)
    assert sample["sample"]["row"]["split"] == "train"
    # 配对动作只由奖励路径显式加载：音乐采样本身仍能在没有任何动作文件时运行。
    assert not any(root.glob("*/motions"))
    assert "target_activity" not in sample and "activity_target" not in sample
    assert "target_qpos30" not in sample and "qpos" not in sample


def test_train_music_cannot_accept_val_or_modified_features(tmp_path):
    root, path = write_music_tasks(tmp_path)
    selection = json.loads(path.read_text())
    selection[0]["row"]["split"] = "val"
    path.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="current train manifest"):
        TrainMusicSampler(root, path)
    selection[0]["row"]["split"] = "train"
    path.write_text(json.dumps(selection))
    (root / SOURCES[0] / "music_0.pt").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="SHA mismatch"):
        TrainMusicSampler(root, path)
