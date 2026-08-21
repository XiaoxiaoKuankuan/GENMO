"""验证 BUMI3 SONIC 50 Hz NPZ 物理预检查的核心契约与时间尺度。

测试构造可解释的 22-body、21 关节、七字段 float32 轨迹，覆盖严格 NPZ 键与帧率、
Isaac-Lab publish-order 关节限位、50 Hz 连续帧阈值、左右肩躯干代理、严重速度峰值、
速度字段只诊断不改变状态以及“忽略关节限位”的反事实状态。测试不依赖外部 GMR
或 Isaac-Lab，因此用于证明规则方向和数据契约；生产资产 SHA 与实际全量结果由脚本
启动时的 asset verification 和全量报告负责验证。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gem.robots.bumi.quality_filter import QualityStatus
from tools.data.bumi.filter_sonic_npz_motions import (
    DEFAULT_CONFIG,
    evaluate_motion,
    load_config,
    load_motion_npz,
)


@pytest.fixture(scope="module")
def quality_config():
    return load_config(DEFAULT_CONFIG)


def _central_difference(values: np.ndarray, fps: float = 50.0) -> np.ndarray:
    result = np.zeros_like(values)
    result[0] = (values[1] - values[0]) * fps
    result[-1] = (values[-1] - values[-2]) * fps
    result[1:-1] = (values[2:] - values[:-2]) * (fps / 2.0)
    return result


def _arrays(quality_config, frames: int = 100) -> dict[str, np.ndarray]:
    joint_mid = (quality_config.joint_lower_limits + quality_config.joint_upper_limits) / 2.0
    joint_pos = np.repeat(joint_mid[None], frames, axis=0).astype(np.float32)
    body_pos = np.zeros((frames, 22, 3), dtype=np.float32)
    body_pos[:, :, 2] = 0.5
    lookup = {name: index for index, name in enumerate(quality_config.body_order)}
    body_pos[:, 0, 2] = 0.5
    body_pos[:, lookup["l_ankle_roll_link"], 2] = 0.05
    body_pos[:, lookup["r_ankle_roll_link"], 2] = 0.05
    body_pos[:, lookup["l_arm_pitch_link"], 2] = 0.75
    body_pos[:, lookup["r_arm_pitch_link"], 2] = 0.75
    for name in quality_config.upper_non_hand_bodies:
        body_pos[:, lookup[name], 2] = 0.55
    quaternion = np.zeros((frames, 22, 4), dtype=np.float32)
    quaternion[..., 0] = 1.0
    return {
        "fps": np.asarray([50.0], dtype=np.float32),
        "joint_pos": joint_pos,
        "joint_vel": _central_difference(joint_pos),
        "body_pos_w": body_pos,
        "body_quat_w": quaternion,
        "body_lin_vel_w": _central_difference(body_pos),
        "body_ang_vel_w": np.zeros((frames, 22, 3), dtype=np.float32),
    }


def _save(path: Path, arrays: dict[str, np.ndarray]) -> Path:
    np.savez(path, **arrays)
    return path


def test_strict_npz_contract_requires_exact_50hz_float32(tmp_path, quality_config) -> None:
    source = _save(tmp_path / "valid.npz", _arrays(quality_config))
    loaded = load_motion_npz(source, quality_config)
    assert loaded["joint_pos"].shape == (100, 21)
    wrong = _arrays(quality_config)
    wrong["fps"] = np.asarray([30.0], dtype=np.float32)
    with pytest.raises(ValueError, match="fps 必须精确"):
        load_motion_npz(_save(tmp_path / "wrong.npz", wrong), quality_config)


def test_normal_motion_passes(quality_config) -> None:
    decision = evaluate_motion(_arrays(quality_config), quality_config)
    assert decision["status"] == QualityStatus.PASS.value
    assert decision["status_without_joint_limit"] == QualityStatus.PASS.value


def test_publish_order_joint_limit_is_hard_reject_but_counterfactual_passes(
    quality_config,
) -> None:
    arrays = _arrays(quality_config)
    joint_index = quality_config.joint_order.index("r_arm_roll_joint")
    arrays["joint_pos"][:, joint_index] = np.float32(
        quality_config.joint_upper_limits[joint_index] + 0.02
    )
    arrays["joint_vel"] = _central_difference(arrays["joint_pos"])
    decision = evaluate_motion(arrays, quality_config)
    assert decision["status"] == QualityStatus.REJECT.value
    assert decision["status_without_joint_limit"] == QualityStatus.PASS.value
    assert "SOURCE_JOINT_LIMIT" in decision["reason_codes"]


def test_torso_proxy_uses_50hz_sustained_duration(quality_config) -> None:
    arrays = _arrays(quality_config, frames=80)
    lookup = {name: index for index, name in enumerate(quality_config.body_order)}
    arrays["body_pos_w"][:, 0, 2] = 0.2
    for name in quality_config.torso_proxy_bodies:
        arrays["body_pos_w"][:24, lookup[name], 2] = 0.1
    decision = evaluate_motion(arrays, quality_config)
    assert decision["status"] == QualityStatus.REVIEW.value
    assert "FLOOR_STYLE_SUSTAINED" not in decision["reason_codes"]

    for name in quality_config.torso_proxy_bodies:
        arrays["body_pos_w"][24, lookup[name], 2] = 0.1
    decision = evaluate_motion(arrays, quality_config)
    assert decision["status"] == QualityStatus.REJECT.value
    assert "FLOOR_STYLE_SUSTAINED" in decision["reason_codes"]


def test_severe_joint_velocity_is_rejected(quality_config) -> None:
    arrays = _arrays(quality_config, frames=30)
    index = quality_config.joint_order.index("waist_yaw_joint")
    arrays["joint_pos"][14, index] = np.float32(-1.0)
    arrays["joint_pos"][15, index] = np.float32(1.0)
    arrays["joint_vel"] = _central_difference(arrays["joint_pos"])
    decision = evaluate_motion(arrays, quality_config)
    assert decision["status"] == QualityStatus.REJECT.value
    assert "JOINT_VELOCITY_L2_SEVERE" in decision["reason_codes"]


def test_velocity_field_difference_is_diagnostic_only(quality_config) -> None:
    arrays = _arrays(quality_config)
    arrays["body_lin_vel_w"][:, 5, 0] = 123.0
    decision = evaluate_motion(arrays, quality_config)
    consistency = decision["metrics"]["stored_velocity_consistency"]
    assert decision["status"] == QualityStatus.PASS.value
    assert consistency["all_body_linear_velocity_central_difference_max_abs_error"] == 123.0
    assert consistency["status_affecting"] is False
