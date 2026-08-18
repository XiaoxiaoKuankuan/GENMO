"""验证 legacy BUMI3 预筛选的坐标契约、风格规则和安全物化。

测试使用可解释的合成 25-link 动作覆盖最容易出错的边界：NumPy 2 pickle 在
NumPy 1 环境读取、xyzw/wxyz 与关节名重排、单纯手撑地、低 Root 深蹲、持续
躯干贴地、碎片化贴地、严重动力学峰值、安全区间生成以及 hardlink/copy 物化。
这些用例不依赖真实数据或 MuJoCo，因此可以在 CI 中稳定证明筛选规则的方向，真实
阈值的最终精度仍由全量 dry-run 和候选视频人工复核确认。
"""

from __future__ import annotations

import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gem.robots.bumi.legacy_motion import (
    LEGACY_BUMI_BODY_ORDER,
    LEGACY_BUMI_JOINT_ORDER,
    LegacyBumiMotion,
    load_legacy_bumi_motion,
    reorder_joints,
    sha256_file,
)
from gem.robots.bumi.quality_filter import (
    QualityStatus,
    evaluate_legacy_bumi_motion,
    load_bumi_quality_config,
    mask_to_intervals,
    safe_intervals_from_bad_mask,
)
from tools.data.bumi.filter_legacy_bumi_motions import (
    build_summary,
    materialize_selection,
    scan_motions,
    verify_source_mjcf,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
QUALITY_CONFIG = REPO_ROOT / "configs" / "bumi" / "quality_filter_v1.yaml"


@pytest.fixture(scope="module")
def quality_config():
    return load_bumi_quality_config(QUALITY_CONFIG)


def _midpoint_dof(quality_config, frames: int) -> np.ndarray:
    midpoint = (quality_config.joint_lower_limits + quality_config.joint_upper_limits) / 2.0
    return np.repeat(midpoint[None, :], frames, axis=0)


def _motion(
    quality_config,
    *,
    frames: int = 60,
    root_height: float = 0.5,
    hand_on_ground: bool = False,
    torso_on_ground: bool = False,
    fragmented_floor: np.ndarray | None = None,
) -> LegacyBumiMotion:
    root_pos = np.zeros((frames, 3), dtype=np.float64)
    root_pos[:, 2] = root_height
    root_rot = np.zeros((frames, 4), dtype=np.float64)
    root_rot[:, 3] = 1.0
    local = np.zeros((frames, len(LEGACY_BUMI_BODY_ORDER), 3), dtype=np.float64)
    lookup = {name: index for index, name in enumerate(LEGACY_BUMI_BODY_ORDER)}
    local[:, lookup["l_ankle_roll_link"], 2] = -root_height
    local[:, lookup["r_ankle_roll_link"], 2] = -root_height
    local[:, lookup["torso_link_virtual"], 2] = 0.4
    for name in (
        "l_arm_pitch_link",
        "l_arm_roll_link",
        "l_arm_yaw_link",
        "l_elbow_pitch_link",
        "r_arm_pitch_link",
        "r_arm_roll_link",
        "r_arm_yaw_link",
        "r_elbow_pitch_link",
    ):
        local[:, lookup[name], 2] = 0.3
    local[:, lookup["l_arm_hand_link_virtual"], 2] = 0.2
    local[:, lookup["r_arm_hand_link_virtual"], 2] = 0.2
    if hand_on_ground:
        local[:, lookup["l_arm_hand_link_virtual"], 2] = -root_height
        local[:, lookup["r_arm_hand_link_virtual"], 2] = -root_height
    if torso_on_ground:
        local[:, lookup["torso_link_virtual"], 2] = 0.05
    if fragmented_floor is not None:
        selected = np.asarray(fragmented_floor, dtype=np.int64)
        local[selected, lookup["torso_link_virtual"], 2] = -0.25
        local[selected, lookup["l_ankle_roll_link"], 2] = -0.4
        local[selected, lookup["r_ankle_roll_link"], 2] = -0.4
    return LegacyBumiMotion(
        path=Path("synthetic.pkl"),
        fps=30,
        root_pos=root_pos,
        root_rot_xyzw=root_rot,
        dof_pos=_midpoint_dof(quality_config, frames),
        local_body_pos=local,
        body_names=LEGACY_BUMI_BODY_ORDER,
    )


def _payload(motion: LegacyBumiMotion) -> dict[str, object]:
    return {
        "fps": motion.fps,
        "root_pos": motion.root_pos,
        "root_rot": motion.root_rot_xyzw,
        "dof_pos": motion.dof_pos,
        "local_body_pos": motion.local_body_pos.astype(np.float32),
        "link_body_list": list(motion.body_names),
    }


def _write_pickle(path: Path, motion: LegacyBumiMotion, *, numpy2_names: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = pickle.dumps(_payload(motion), protocol=0)
    if numpy2_names:
        assert b"numpy.core" in data
        data = data.replace(b"numpy.core", b"numpy._core")
    path.write_bytes(data)
    return path


def test_legacy_numpy2_pickle_and_explicit_qpos_reorder(tmp_path, quality_config) -> None:
    source = _write_pickle(
        tmp_path / "dataset" / "motion.pkl",
        _motion(quality_config, frames=8),
        numpy2_names=True,
    )
    loaded = load_legacy_bumi_motion(source)
    target_order = tuple(reversed(LEGACY_BUMI_JOINT_ORDER))
    qpos = loaded.qpos_wxyz(target_order)
    assert qpos.shape == (8, 28)
    np.testing.assert_allclose(qpos[:, 3:7], np.repeat([[1.0, 0.0, 0.0, 0.0]], len(qpos), axis=0))
    np.testing.assert_allclose(
        qpos[:, 7:],
        reorder_joints(loaded.dof_pos, LEGACY_BUMI_JOINT_ORDER, target_order),
    )


def test_hand_only_ground_contact_is_not_floor_style(quality_config) -> None:
    decision = evaluate_legacy_bumi_motion(
        _motion(quality_config, hand_on_ground=True), quality_config
    )
    assert decision.status is QualityStatus.PASS
    assert decision.metrics["floor_style"]["frame_count"] == 0
    assert decision.metrics["floor_style"]["hand_below_upper_threshold_frame_count"] == 60


def test_quaternion_sign_flips_do_not_create_angular_velocity(quality_config) -> None:
    motion = _motion(quality_config, frames=12)
    quaternion = motion.root_rot_xyzw.copy()
    quaternion[1::2] *= -1.0
    decision = evaluate_legacy_bumi_motion(
        replace(motion, root_rot_xyzw=quaternion), quality_config
    )
    assert decision.status is QualityStatus.PASS
    assert decision.metrics["quaternion_sign_flip_count"] == 11
    assert decision.metrics["dynamics"]["root_angular_velocity"]["max"] == 0.0


def test_low_root_without_upper_body_contact_is_review_not_reject(quality_config) -> None:
    decision = evaluate_legacy_bumi_motion(_motion(quality_config, root_height=0.2), quality_config)
    assert decision.status is QualityStatus.REVIEW
    assert decision.reason_codes == ("LOW_ROOT_REVIEW",)


def test_sustained_torso_ground_contact_is_rejected(quality_config) -> None:
    decision = evaluate_legacy_bumi_motion(
        _motion(quality_config, root_height=0.2, torso_on_ground=True), quality_config
    )
    assert decision.status is QualityStatus.REJECT
    assert "FLOOR_STYLE_SUSTAINED" in decision.reason_codes
    assert decision.floor_intervals == ((0, 60),)


def test_fragmented_floor_contact_is_review(quality_config) -> None:
    selected = np.arange(0, 600, 40, dtype=np.int64)
    assert len(selected) == 15
    decision = evaluate_legacy_bumi_motion(
        _motion(quality_config, frames=600, fragmented_floor=selected), quality_config
    )
    assert decision.status is QualityStatus.REVIEW
    assert "FLOOR_STYLE_FRAGMENTED" in decision.reason_codes
    assert decision.metrics["floor_style"]["max_consecutive_frames"] == 1


def test_severe_joint_velocity_spike_is_rejected(quality_config) -> None:
    motion = _motion(quality_config, frames=20)
    dof = motion.dof_pos.copy()
    dof[10, 0] = quality_config.joint_upper_limits[0]
    dof[11, 0] = quality_config.joint_lower_limits[0]
    decision = evaluate_legacy_bumi_motion(replace(motion, dof_pos=dof), quality_config)
    assert decision.status is QualityStatus.REJECT
    assert "JOINT_VELOCITY_L2_SEVERE" in decision.reason_codes


def test_interval_helpers_use_half_open_ranges_and_halo() -> None:
    mask = np.zeros(400, dtype=np.bool_)
    mask[150:170] = True
    assert mask_to_intervals(mask) == ((150, 170),)
    assert safe_intervals_from_bad_mask(mask, halo_frames=15, minimum_frames=120) == (
        (0, 135),
        (185, 400),
    )


def test_source_mjcf_hash_mismatch_fails_before_scanning(tmp_path) -> None:
    mjcf = tmp_path / "bumi3.xml"
    mjcf.write_text("<mujoco/>", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_source_mjcf(mjcf, "0" * 64)


def test_scan_and_materialize_pass_only_without_modifying_source(tmp_path, quality_config) -> None:
    input_root = tmp_path / "source"
    normal = _write_pickle(
        input_root / "aistpp" / "normal.pkl", _motion(quality_config, frames=120)
    )
    rejected = _write_pickle(
        input_root / "finedance" / "floor.pkl",
        _motion(quality_config, frames=120, root_height=0.2, torso_on_ground=True),
    )
    hashes_before = {path: sha256_file(path) for path in (normal, rejected)}
    config_sha = sha256_file(QUALITY_CONFIG)
    rows = scan_motions(
        [normal, rejected],
        input_root=input_root.resolve(),
        config=quality_config,
        config_sha256=config_sha,
        source_mjcf_sha256=quality_config.source_mjcf_sha256,
        workers=1,
        show_progress=False,
    )
    assert [row["status"] for row in rows] == ["PASS", "REJECT"]
    summary = build_summary(
        rows,
        input_root=input_root.resolve(),
        config_path=QUALITY_CONFIG,
        config_sha256=config_sha,
        source_mjcf=tmp_path / "source.xml",
        source_mjcf_sha256=quality_config.source_mjcf_sha256,
    )
    output = tmp_path / "selected"
    result = materialize_selection(
        rows,
        input_root=input_root.resolve(),
        output_root=output,
        include_review=False,
        mode="auto",
        summary=summary,
    )
    assert result["selected_sequences"] == 1
    assert (output / "motions" / "aistpp" / "normal.pkl").is_file()
    assert not (output / "motions" / "finedance" / "floor.pkl").exists()
    assert {path: sha256_file(path) for path in (normal, rejected)} == hashes_before
