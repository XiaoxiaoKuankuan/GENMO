"""验证在线条件的旧参考 anchor、真实历史时间轴和监督隔离。

测试仅在 CPU 使用仓库真实 fe934 codec/FK，把显式合成的物理姿态和观测转换成现有十
字段；不加载完整 checkpoint，不运行 Isaac 或 GPU。覆盖 P=0/P=1/可变 P、前缀末端
root XY 两坐标无效、音乐边界、实际历史缺失及重复时间，并检查条件构造不会修改来源。
这些测试只证明数据适配，不替代真实模型和 Isaac 动力学的闭环验收。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from gem.closedloop.contracts import STAGE1_CONDITION_KEYS, validate_stage1_condition_batch
from gem.closedloop.online_conditions import OnlineConditionBuilder
from gem.robots.bumi.feature_codec import BumiMotionFeatureCodec
from gem.robots.bumi.kinematics import BumiKinematics
from gem.utils.rotation_conversions import axis_angle_to_quaternion

REPO = Path(__file__).resolve().parents[2]
KINEMATICS = REPO / "configs/bumi/bumi_kinematics_robot_retargeter_fe934_v1.json"


@pytest.fixture
def builder():
    return OnlineConditionBuilder(BumiMotionFeatureCodec(BumiKinematics(KINEMATICS)))


def make_inputs(builder, prefix=12, *, tick=600, history=50):
    qpos = builder.codec.kinematics.default_qpos.repeat(max(prefix, 1), 1)
    frame = torch.arange(max(prefix, 1), dtype=torch.float32)
    qpos[:, 0] = 2.5 + frame * 0.01
    qpos[:, 1] = -1.2 - frame * 0.004
    qpos[:, 2] += 0.08
    angles = torch.zeros((max(prefix, 1), 3))
    angles[:, 2] = 3.1 + frame * 0.02
    qpos[:, 3:7] = axis_angle_to_quaternion(angles)
    snapshot = {
        "tick": tick, "env_id": 0, "episode_id": "ep7", "seed": 42,
        "history_values": np.arange(history * 48, dtype=np.float32).reshape(history, 48) / 100,
        "history_valid": np.ones(history, dtype=bool),
        "history_ticks": tick - np.arange(history - 1, -1, -1, dtype=np.int64) * 12,
        # 适配器不得用实际状态重锚；这个明显不同的位置只用于发现误用。
        "actual_qpos": np.full(28, 999.0, dtype=np.float32),
    }
    reservation = {
        "source_qpos": qpos[:prefix].numpy().copy(),
        "source_ticks": tick + np.arange(prefix, dtype=np.int64) * 20,
        "reference_anchor_qpos": qpos[0].numpy().copy(),
        "prefix_frames": prefix,
        "decision_tick": tick,
        "env_id": 0,
        "episode_id": "ep7",
        "decision_id": 3,
        "plan_id": "p3",
        "parent_plan_id": "p2",
        "request_id": "r3",
        "protected_end": tick + 240,
        "deadline": tick + 120,
    }
    music = np.arange(200 * 35, dtype=np.float32).reshape(200, 35) / 1000
    return snapshot, reservation, music


@pytest.mark.parametrize("prefix", [0, 1, 6, 12, 18, 35, 119])
def test_physical_prefix_masks_and_common_anchor(builder, prefix):
    snapshot, reservation, music = make_inputs(builder, prefix)
    original = reservation["source_qpos"].copy()
    conditions, meta = builder.build(snapshot, reservation, music)
    assert set(conditions) == set(STAGE1_CONDITION_KEYS)
    validate_stage1_condition_batch(conditions)
    mask = conditions["known_qpos30_mask"][0]
    assert mask[:prefix, 2:].all()
    assert not mask[prefix:].any()
    assert not mask[-1, :2].any()
    assert int(mask.any(-1).sum()) == prefix
    assert torch.count_nonzero(conditions["known_qpos30"][~conditions["known_qpos30_mask"]]) == 0
    if prefix:
        assert not mask[prefix - 1, :2].any()
        assert mask[:prefix - 1, :2].all()
        physical = conditions["known_qpos30"][0, :prefix]
        world = builder.codec.apply_world_anchor(builder.codec.decode_to_canonical_qpos(physical), torch.tensor(meta["world_anchor"]))
        np.testing.assert_allclose(world[:, :3], original[:, :3], atol=1e-6, rtol=0)
        np.testing.assert_allclose(np.abs((world[:, 3:7].numpy() * original[:, 3:7]).sum(-1)), 1.0, atol=1e-6)
        np.testing.assert_allclose(world[:, 7:], original[:, 7:], atol=0, rtol=0)
    assert meta["world_anchor"][0] == pytest.approx(2.5)
    assert meta["world_anchor"][2] == pytest.approx(float(builder.codec.default_root_height))
    assert meta["plan_id"] == "p3" and meta["parent_plan_id"] == "p2"
    assert meta["seed"] == 42
    np.testing.assert_array_equal(reservation["source_qpos"], original)


def test_p0_requires_explicit_old_reference_anchor(builder):
    snapshot, reservation, music = make_inputs(builder, 0)
    del reservation["reference_anchor_qpos"]
    with pytest.raises(ValueError, match="reference_anchor_qpos"):
        builder.build(snapshot, reservation, music)


def test_empty_history_and_invalid_slots_cannot_leak(builder):
    snapshot, reservation, music = make_inputs(builder, history=0)
    conditions, _ = builder.build(snapshot, reservation, music)
    assert not conditions["proprio_history_valid"].any()
    assert not conditions["proprio_history"].any()
    snapshot, reservation, music = make_inputs(builder)
    snapshot["history_valid"][:25] = False
    snapshot["history_values"][:25] = np.nan
    conditions, _ = builder.build(snapshot, reservation, music)
    assert not conditions["proprio_history"][0, :25].any()
    np.testing.assert_array_equal(conditions["proprio_history"][0, 25:].numpy(), snapshot["history_values"][25:])
    assert np.isnan(snapshot["history_values"][:25]).all()


def test_partial_history_is_left_padded_without_fabricating_valid_data(builder):
    snapshot, reservation, music = make_inputs(builder, history=4)
    conditions, _ = builder.build(snapshot, reservation, music)
    assert not conditions["proprio_history_valid"][0, :46].any()
    assert conditions["proprio_history_valid"][0, 46:].all()
    np.testing.assert_array_equal(conditions["proprio_history"][0, 46:].numpy(), snapshot["history_values"])
    assert float(conditions["proprio_history_times"][0, -1]) == 1.0


def test_music_start_and_end_use_masks_without_padding_influence(builder):
    snapshot, reservation, music = make_inputs(builder, tick=900)
    conditions, _ = builder.build(snapshot, reservation, music[:25], music_start_tick=600)
    np.testing.assert_array_equal(conditions["music_features"][0, :10].numpy(), music[15:25])
    assert conditions["music_valid"][0, :10].all()
    assert not conditions["music_valid"][0, 10:].any()
    assert not conditions["music_features"][0, 10:].any()
    assert conditions["future_valid"].all()


@pytest.mark.parametrize("mutation,error", [
    ("future_history", "later"), ("duplicate_history", "unique"),
    ("shift_source", "source ticks"), ("different_episode", "episode_id"),
    ("bad_snapshot_time", "snapshot tick"), ("target", "supervision"),
])
def test_rejects_inconsistent_request_inputs(builder, mutation, error):
    snapshot, reservation, music = make_inputs(builder)
    if mutation == "future_history":
        snapshot["history_ticks"] += 12
    elif mutation == "duplicate_history":
        snapshot["history_ticks"][-1] = snapshot["history_ticks"][-2]
    elif mutation == "shift_source":
        reservation["source_ticks"] += 20
    elif mutation == "different_episode":
        reservation["episode_id"] = "other"
    elif mutation == "bad_snapshot_time":
        snapshot["tick"] += 12
    elif mutation == "target":
        reservation["target_qpos30"] = np.zeros((120, 30))
    with pytest.raises(ValueError, match=error):
        builder.build(snapshot, reservation, music)


def test_long_episode_times_keep_30hz_and_50hz_resolution(builder):
    snapshot, reservation, music = make_inputs(builder, tick=600 * 10000)
    conditions, _ = builder.build(snapshot, reservation, music, music_start_tick=snapshot["tick"])
    assert conditions["future_times"].dtype == torch.float64
    validate_stage1_condition_batch(conditions)
