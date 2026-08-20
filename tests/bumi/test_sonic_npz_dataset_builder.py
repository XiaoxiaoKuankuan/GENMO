"""验证 50 Hz SONIC NPZ 到当前 GENMO BUMI qpos28 的关键转换语义。

这些测试不依赖外部 GMR、MuJoCo 或服务器数据。它们用可解释的合成轨迹覆盖持续时间
保持帧数公式、30 Hz 最后一帧保持、按完整关节名重排、wxyz 四元数连续化/SLERP、
AIST++ ``_armfix`` variant 映射，以及使用目标 GENMO kinematics 做 body-origin 地面
归一化。全量音频、manifest、SHA 和统计量仍由服务器2正式构建与严格 validator 验收。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gem.robots.bumi.kinematics import BumiKinematics
from tools.data.bumi.build_bumi_music_dataset_from_sonic_npz import (
    canonical_human_sample_id,
    deduplicate_quality_variants,
    expected_50hz_frames,
    make_quaternion_continuous_np,
    normalize_body_origin_ground,
    resample_sonic_qpos_to_30hz,
)


def _arrays(frames: int, source_names: tuple[str, ...]) -> dict[str, np.ndarray]:
    joint = np.repeat(np.arange(21, dtype=np.float32)[None], frames, axis=0)
    body_pos = np.zeros((frames, 22, 3), dtype=np.float32)
    body_pos[:, 0, 0] = np.arange(frames, dtype=np.float32) / 50.0
    body_quat = np.zeros((frames, 22, 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    assert len(source_names) == joint.shape[1]
    return {"joint_pos": joint, "body_pos_w": body_pos, "body_quat_w": body_quat}


def test_duration_preserving_frame_formula_and_last_frame_hold() -> None:
    target_names = tuple(f"joint_{index}" for index in range(21))
    source_names = tuple(reversed(target_names))
    assert expected_50hz_frames(5) == 7
    assert expected_50hz_frames(4) == 5  # 右端点不包含，整秒网格不能再额外加一帧。
    arrays = _arrays(7, source_names)
    qpos = resample_sonic_qpos_to_30hz(
        arrays,
        source_joint_order=source_names,
        target_joint_order=target_names,
        target_frames=5,
    )
    assert qpos.shape == (5, 28)
    torch.testing.assert_close(qpos[:, 0], torch.tensor([0.0, 1.0 / 30.0, 2.0 / 30.0, 0.1, 0.12]))
    torch.testing.assert_close(qpos[0, 7:], torch.arange(20, -1, -1).float())
    with pytest.raises(ValueError, match="帧数不一致"):
        resample_sonic_qpos_to_30hz(
            _arrays(8, source_names),
            source_joint_order=source_names,
            target_joint_order=target_names,
            target_frames=5,
        )


def test_quaternion_sign_is_continuous_before_slerp() -> None:
    quaternion = np.asarray(
        [[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0]],
        dtype=np.float32,
    )
    result = make_quaternion_continuous_np(quaternion)
    assert np.all(np.sum(result[1:] * result[:-1], axis=-1) >= 0.0)
    np.testing.assert_allclose(np.linalg.norm(result, axis=-1), 1.0)


def test_target_kinematics_ground_normalization(test_kinematics_path) -> None:
    kinematics = BumiKinematics(test_kinematics_path)
    qpos = torch.zeros(4, 28)
    qpos[:, 2] = 1.0
    qpos[:, 3] = 1.0
    normalized, before, after = normalize_body_origin_ground(qpos, kinematics)
    assert before == pytest.approx(0.5)
    assert after == pytest.approx(0.0, abs=1.0e-6)
    torch.testing.assert_close(normalized[:, 2], torch.full((4,), 0.5))


def test_only_known_aist_armfix_variant_changes_pairing_id() -> None:
    assert (
        canonical_human_sample_id("aistpp", "gBR_sBM_cAll_d04_mBR0_ch02_armfix")
        == "gBR_sBM_cAll_d04_mBR0_ch02"
    )
    assert canonical_human_sample_id("finedance", "001_armfix") == "001_armfix"
    selected, superseded = deduplicate_quality_variants(
        [
            {"sample_id": "aistpp/gBR_sBM_cAll_d04_mBR0_ch02"},
            {"sample_id": "aistpp/gBR_sBM_cAll_d04_mBR0_ch02_armfix"},
            {"sample_id": "finedance/001"},
        ]
    )
    assert [row["sample_id"] for row in selected] == [
        "aistpp/gBR_sBM_cAll_d04_mBR0_ch02_armfix",
        "finedance/001",
    ]
    assert superseded == [
        {
            "dataset": "aistpp",
            "canonical_sample_id": "gBR_sBM_cAll_d04_mBR0_ch02",
            "selected_variant_id": "gBR_sBM_cAll_d04_mBR0_ch02_armfix",
            "superseded_variant_id": "gBR_sBM_cAll_d04_mBR0_ch02",
        }
    ]
