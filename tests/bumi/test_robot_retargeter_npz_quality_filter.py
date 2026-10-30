"""测试 BUMI3 30 Hz 输入契约、历史三态和保留舞蹈的新二分类规则。

本文件构造最小的 13 字段 Mimic NPZ，不依赖服务器数据，重点覆盖正式批处理最容易
静默出错的边界：帧率与键顺序必须精确、名称数组不能换序、数值数组必须 float32，
以及 PASS / REVIEW / REJECT 必须分别响应正常站立、Root 倾角异常和严重关节突变。
新增合成地板、倒立、跳跃和快转样例，验证新规则只诊断、不切片，报告无 REVIEW；
关节限位和损坏数值仍拒绝。只验证离线数组规则，不代替全库统计或物理回放。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tools.data.bumi.filter_robot_retargeter_npz_motions import (
    DANCE_CONFIG,
    DEFAULT_CONFIG,
    REPORT_VERSION,
    evaluate_path,
    load_config,
    load_motion_npz,
)
from tools.data.bumi.npz_quality_utils import build_summary, evaluate_motion, write_reports


@pytest.fixture(scope="module")
def quality_config():
    return load_config(DEFAULT_CONFIG)


@pytest.fixture(scope="module")
def dance_config():
    return load_config(DANCE_CONFIG)


def _arrays(config, frames: int = 120) -> dict[str, np.ndarray]:
    body_pos = np.zeros((frames, 22, 3), dtype=np.float32)
    body_pos[..., 2] = np.float32(0.5)
    body_pos[:, 0, 2] = np.float32(0.8)
    for name in config.ankle_bodies:
        body_pos[:, config.body_order.index(name), 2] = np.float32(0.05)
    body_quat = np.zeros((frames, 22, 4), dtype=np.float32)
    body_quat[..., 0] = np.float32(1.0)
    return {
        "fps": np.asarray(30.0, dtype=np.float64),
        "joint_pos": np.zeros((frames, 21), dtype=np.float32),
        "joint_vel": np.zeros((frames, 21), dtype=np.float32),
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_lin_vel_w": np.zeros((frames, 22, 3), dtype=np.float32),
        "body_ang_vel_w": np.zeros((frames, 22, 3), dtype=np.float32),
        "joint_names": np.asarray(config.joint_order),
        "body_names": np.asarray(config.body_order),
        "anchor_body_name": np.asarray(config.anchor_body_name),
        "source_motion": np.asarray("/trusted/source/example.npz"),
        "robot_name": np.asarray(config.robot_name),
        "quaternion_order": np.asarray("wxyz"),
    }


def _save_npz(path: Path, arrays: dict[str, np.ndarray], key_order: tuple[str, ...]) -> None:
    np.savez(path, **{name: arrays[name] for name in key_order})


def test_strict_13_field_contract_accepts_exact_npz(tmp_path: Path, quality_config) -> None:
    path = tmp_path / "motion.npz"
    expected = _arrays(quality_config)
    _save_npz(path, expected, quality_config.required_npz_keys)

    actual = load_motion_npz(path, quality_config)

    assert tuple(actual) == quality_config.required_npz_keys
    assert actual["joint_pos"].shape == (120, 21)
    assert float(actual["fps"]) == 30.0


@pytest.mark.parametrize("mutation", ["fps", "key_order", "joint_order"])
def test_strict_contract_rejects_implicit_conversion(
    tmp_path: Path, quality_config, mutation: str
) -> None:
    path = tmp_path / f"bad_{mutation}.npz"
    arrays = _arrays(quality_config)
    keys = quality_config.required_npz_keys
    if mutation == "fps":
        arrays["fps"] = np.asarray([30.0], dtype=np.float32)
    elif mutation == "key_order":
        keys = (keys[1], keys[0], *keys[2:])
    else:
        arrays["joint_names"] = arrays["joint_names"][::-1]
    _save_npz(path, arrays, keys)

    with pytest.raises(ValueError):
        load_motion_npz(path, quality_config)


def test_quality_states_cover_pass_review_reject(quality_config) -> None:
    standing = _arrays(quality_config)
    assert evaluate_motion(standing, quality_config)["status"] == "PASS"

    tilted = _arrays(quality_config)
    angle = np.deg2rad(35.0)
    tilted["body_quat_w"][:, 0, :2] = np.asarray(
        [np.cos(angle / 2.0), np.sin(angle / 2.0)], dtype=np.float32
    )
    review = evaluate_motion(tilted, quality_config)
    assert review["status"] == "REVIEW"
    assert "ROOT_TILT_DISTRIBUTION_REVIEW" in review["reason_codes"]

    violent = _arrays(quality_config)
    violent["joint_pos"][1, 0] = np.float32(2.0)
    violent["joint_pos"][2, 0] = np.float32(-2.0)
    reject = evaluate_motion(violent, quality_config)
    assert reject["status"] == "REJECT"
    assert any(code.endswith("_SEVERE") for code in reject["reason_codes"])


def test_sustained_sideways_root_is_rejected(quality_config) -> None:
    arrays = _arrays(quality_config)
    angle = np.deg2rad(90.0)
    arrays["body_quat_w"][:, 0, :2] = np.asarray(
        [np.cos(angle / 2.0), np.sin(angle / 2.0)], dtype=np.float32
    )

    decision = evaluate_motion(arrays, quality_config)

    assert decision["status"] == "REJECT"
    assert "ROOT_TILT_DISTRIBUTION_REJECT" in decision["reason_codes"]
    assert decision["metrics"]["root_orientation"]["median_degrees"] == pytest.approx(90.0)


def test_robot_retargeter_record_overrides_generic_report_version(
    tmp_path: Path, quality_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """中性评估结果不能覆盖 robot_retargeter 边界的逐条报告版本。"""

    source_root = tmp_path / "root"
    source = source_root / "aistpp" / "mimic_npz" / "bumi3" / "sample.npz"
    source.parent.mkdir(parents=True)
    _save_npz(source, _arrays(quality_config), quality_config.required_npz_keys)
    monkeypatch.setattr(
        "tools.data.bumi.filter_robot_retargeter_npz_motions._validate_sidecars",
        lambda **_kwargs: {},
    )

    row = evaluate_path(
        source,
        input_root=source_root,
        config=quality_config,
        config_sha256="a" * 64,
        release_row={},
    )

    assert row["status"] == "PASS"
    assert row["report_contract_version"] == REPORT_VERSION


@pytest.mark.parametrize(
    "motion",
    ["floor", "inverted", "crouch", "jump", "fast_spin", "fast_joints", "fast_translation"],
)
def test_binary_preserves_difficult_dance_and_entire_timeline(dance_config, motion):
    arrays = _arrays(dance_config)
    if motion == "floor":
        arrays["body_pos_w"][..., 2] = 0.04
        arrays["body_pos_w"][:, 0, 2] = 0.15
        arrays["body_quat_w"][:, 0] = [np.sqrt(0.5), np.sqrt(0.5), 0, 0]
    elif motion == "inverted":
        arrays["body_quat_w"][:, 0] = [0, 1, 0, 0]
    elif motion == "crouch":
        arrays["body_pos_w"][:, 0, 2] = 0.15
    elif motion == "jump":
        arrays["body_pos_w"][:, 0, 2] = 2.5
    elif motion == "fast_spin":
        angle = np.arange(120) * 1.4
        arrays["body_quat_w"][:, 0, 0] = np.cos(angle / 2)
        arrays["body_quat_w"][:, 0, 3] = np.sin(angle / 2)
    elif motion == "fast_joints":
        arrays["joint_pos"][:, 0] = 1.9 * np.sin(np.arange(120) * np.pi / 3)
    else:
        arrays["body_pos_w"][:, 0, 0] = np.arange(120) * 0.4
    result = evaluate_motion(arrays, dance_config)
    assert result["status"] == result["status_without_joint_limit"] == "PASS"
    assert result["quality_accepted"] is True
    assert result["reason_codes"] == []
    assert result["diagnostic_codes"]
    assert all("REVIEW" not in code for code in result["diagnostic_codes"])
    assert result["valid_intervals"] == [[0, 120]]
    assert result["metrics"]["root_orientation"]["status_affecting"] is False
    if motion == "floor":
        assert result["floor_intervals"] == [[0, 120]]


@pytest.mark.parametrize("motion", ["joint_limit", "quaternion"])
def test_binary_rejects_definite_kinematic_errors(dance_config, motion):
    arrays = _arrays(dance_config)
    if motion == "joint_limit":
        arrays["joint_pos"][40, 0] = dance_config.joint_upper_limits[0] + 0.001
    else:
        arrays["body_quat_w"][40, 0, 0] = 2.0
    result = evaluate_motion(arrays, dance_config)
    assert result["status"] == "REJECT"
    assert result["valid_intervals"] == []
    assert set(result["reason_statuses"].values()) == {"REJECT"}


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_binary_rejects_nonfinite_at_input_boundary(tmp_path, dance_config, bad_value, monkeypatch):
    root = tmp_path / "source"
    path = root / "aistpp/mimic_npz/bumi3/sample.npz"
    path.parent.mkdir(parents=True)
    arrays = _arrays(dance_config)
    arrays["joint_pos"][10, 0] = bad_value
    _save_npz(path, arrays, dance_config.required_npz_keys)
    monkeypatch.setattr(
        "tools.data.bumi.filter_robot_retargeter_npz_motions._validate_sidecars", lambda **kw: {}
    )
    row = evaluate_path(
        path, input_root=root, config=dance_config, config_sha256="a" * 64, release_row={}
    )
    assert row["status"] == "REJECT"
    assert row["reason_codes"] == ["MOTION_CONTRACT_ERROR"]


def test_binary_short_valid_clip_not_lost_and_quaternion_sign_is_not_spin(dance_config):
    arrays = _arrays(dance_config, frames=4)
    arrays["body_quat_w"][1::2] *= -1
    result = evaluate_motion(arrays, dance_config)
    assert result["status"] == "PASS"
    assert result["valid_intervals"] == [[0, 4]]
    assert result["metrics"]["dynamics"]["root_angular_velocity"]["max"] == 0


def test_binary_report_contains_only_pass_reject(tmp_path, dance_config):
    import json

    good, bad = _arrays(dance_config), _arrays(dance_config)
    bad["joint_pos"][40, 0] = 3.0
    rows = [
        {
            **evaluate_motion(arrays, dance_config),
            "sample_id": str(i),
            "dataset": "aistpp",
            "source_relative_path": f"{i}.npz",
        }
        for i, arrays in enumerate((good, bad))
    ]
    summary = build_summary(
        rows,
        input_root=tmp_path,
        config_path=DANCE_CONFIG,
        config_sha256="a" * 64,
        config=dance_config,
        assets={},
    )
    assert summary["status_counts"] == {"PASS": 1, "REJECT": 1}
    assert summary["result_states"] == ["PASS", "REJECT"]
    write_reports(tmp_path / "reports", rows, summary, DANCE_CONFIG, overwrite=False)
    assert not list((tmp_path / "reports").glob("*review*"))
    assert "REVIEW" not in (tmp_path / "reports/quality_report.jsonl").read_text()
    assert "REVIEW" not in json.dumps(summary)
    assert (tmp_path / "reports/strict_pass.txt").read_text() == "0.npz\n"
    assert (tmp_path / "reports/strict_reject.txt").read_text() == "1.npz\n"
