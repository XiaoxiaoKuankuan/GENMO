#!/usr/bin/env python3
"""对实际 50 Hz BUMI3 SONIC/Isaac-Lab NPZ 执行自动物理预检查。

这个脚本用于检查离线 GMR 导出并可能经过平滑处理的部署轨迹，而不是旧的 30 Hz
legacy pickle。它严格验证七字段 NPZ 的键、float32、帧率、帧数、形状、有限值与
wxyz 四元数；按照 GMR preset 的 Isaac-Lab publish order 检查 21 个关节限位；
直接使用文件中的 22-body 世界状态检查根高度、根倾角和贴地风格；最后从真实
50 Hz ``joint_pos``、根位置和根朝向重新计算速度、加速度与 jerk。存储的速度字段
与中心差分之间的误差会被记录，但由于部署契约没有规定唯一离散化实现，v1 不用
该误差单独改变状态。

阈值来自版本化 YAML。动力学阈值保持物理单位不变，所有持续帧阈值按时间由旧
30 Hz 规则换算到 50 Hz。当前 22-body 契约没有 legacy virtual torso/hand：手部
原本不参与拒绝；躯干高度改用左右肩根 body 的平均世界高度作为代理。GMR 使用
0.05 m ground clearance，因此旧的“body origin 最低点必须为 0”不再适用，只作为
诊断值保存。脚本只读源 NPZ，在独立目录原子写 JSONL、CSV、汇总和配置快照；不会
删除、移动、重写或物化任何动作。这里的 PASS 只表示离线数据预检查通过，不代表
已经通过 Isaac-Lab 物理 rollout 或 SONIC 策略跟踪测试。

典型用法：

.. code-block:: bash

   python tools/data/bumi/filter_sonic_npz_motions.py \
     --input-root data/motions_npz_bumi3_smooth_q1 \
     --output-dir outputs/bumi_quality_smooth_q1_50hz_v1
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gem.robots.bumi.quality_filter import (  # noqa: E402
    QualityStatus,
    mask_to_intervals,
    safe_intervals_from_bad_mask,
)

CONFIG_VERSION = "genmo.bumi_sonic_npz_quality_config.v1"
REPORT_VERSION = "genmo.bumi_quality_report.sonic_npz_50hz.v1"
DEFAULT_CONFIG = REPO_ROOT / "configs/bumi/quality_filter_sonic_npz_50hz_v1.yaml"
DEFAULT_PRESET = Path("/home/weili/GMR-CPP_e1jump_lowdpi/config/robot_presets/bumi3.json")
DEFAULT_ROBOT_XML = Path("/home/weili/GMR-CPP_e1jump_lowdpi/assets/bumi3/mjcf/bumi3.xml")
DEFAULT_KINEMATICS = Path("/home/weili/OMG/assets/robots/bumi/bumi_kinematics.json")
REPORT_FILENAMES = (
    "quality_report.jsonl",
    "quality_report.csv",
    "quality_summary.json",
    "review_candidates.jsonl",
    "quality_config.snapshot.yaml",
    "strict_pass.txt",
    "strict_reject.txt",
    "without_joint_limit_review.txt",
    "without_joint_limit_reject.txt",
)


@dataclass(frozen=True)
class SonicNpzQualityConfig:
    """已经完成结构和语义校验的 50 Hz NPZ 质量规则。"""

    motion_contract_version: str
    fps: int
    required_keys: tuple[str, ...]
    robot_xml_sha256: str
    preset_sha256: str
    kinematics_sha256: str
    joint_order: tuple[str, ...]
    joint_lower_limits: np.ndarray
    joint_upper_limits: np.ndarray
    body_order: tuple[str, ...]
    minimum_frames: int
    quaternion_norm_error_max: float
    joint_limit_violation_max: float
    minimum_joint_limit_margin_warn: float
    root_height_min_absolute: float
    root_height_max_absolute: float
    exceed_ratio_max: float
    consecutive_exceed_frames: int
    severe_multiplier: float
    dynamics: Mapping[str, tuple[float, str]]
    root_low_height: float
    root_low_tilt_degrees: float
    torso_ground_height: float
    upper_body_ground_height: float
    floor_gate_root_height: float
    floor_gate_tilt_degrees: float
    ankles_airborne_height: float
    floor_reject_consecutive_frames: int
    floor_review_ratio: float
    floor_review_min_frames: int
    low_root_review_height: float
    low_root_review_consecutive_frames: int
    safe_interval_halo_frames: int
    minimum_safe_interval_frames: int
    torso_proxy_bodies: tuple[str, ...]
    upper_non_hand_bodies: tuple[str, ...]
    ankle_bodies: tuple[str, ...]


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} 必须是 mapping")
    return value


def _finite_float(parent: Mapping[str, Any], key: str) -> float:
    try:
        value = float(parent[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{key} 必须是有限数值") from exc
    if not np.isfinite(value):
        raise ValueError(f"{key} 必须是有限数值")
    return value


def _integer(parent: Mapping[str, Any], key: str, *, allow_zero: bool = False) -> int:
    value = parent.get(key)
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{key} 必须是 >= {minimum} 的整数")
    return value


def _names(parent: Mapping[str, Any], key: str, length: int | None = None) -> tuple[str, ...]:
    values = tuple(map(str, parent.get(key, ())))
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{key} 必须是非空且不重复的名称列表")
    if length is not None and len(values) != length:
        raise ValueError(f"{key} 长度必须为 {length}，实际为 {len(values)}")
    return values


def _vector(parent: Mapping[str, Any], key: str, length: int) -> np.ndarray:
    value = np.asarray(parent.get(key), dtype=np.float64)
    if value.shape != (length,) or not np.isfinite(value).all():
        raise ValueError(f"{key} 必须是长度 {length} 的有限向量")
    return value


def load_config(path: str | Path) -> SonicNpzQualityConfig:
    """读取并严格验证 50 Hz 规则，拒绝不完整或自相矛盾的配置。"""

    config_path = Path(path).expanduser().resolve(strict=True)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("contract_version") != CONFIG_VERSION:
        raise ValueError(f"质量配置 contract_version 必须为 {CONFIG_VERSION}")
    source = _mapping(raw, "source")
    hard = _mapping(raw, "hard_thresholds")
    soft = _mapping(raw, "soft_policy")
    floor = _mapping(raw, "floor_style")
    body_groups = _mapping(floor, "body_groups")
    fps = _integer(source, "fps")
    if fps != 50:
        raise ValueError(f"SONIC 离线质量检查固定要求 50 Hz，实际为 {fps}")
    if source.get("quaternion_convention") != "wxyz":
        raise ValueError("source.quaternion_convention 必须为 wxyz")
    joint_order = _names(source, "joint_order", 21)
    body_order = _names(source, "body_order", 22)
    lower = _vector(source, "joint_lower_limits", 21)
    upper = _vector(source, "joint_upper_limits", 21)
    if np.any(lower >= upper):
        raise ValueError("每个关节必须满足 lower < upper")
    dynamics_raw = _mapping(raw, "dynamics")
    expected_dynamics = {
        "joint_velocity_l2",
        "joint_acceleration_l2",
        "joint_jerk_l2",
        "root_linear_velocity",
        "root_angular_velocity",
    }
    if set(dynamics_raw) != expected_dynamics:
        raise ValueError(f"dynamics 键必须严格为 {sorted(expected_dynamics)}")
    dynamics: dict[str, tuple[float, str]] = {}
    for name in sorted(expected_dynamics):
        item = _mapping(dynamics_raw, name)
        threshold = _finite_float(item, "threshold")
        unit = str(item.get("unit", "")).strip()
        if threshold <= 0.0 or not unit:
            raise ValueError(f"dynamics.{name} 的 threshold/unit 非法")
        dynamics[name] = (threshold, unit)
    torso = _names(body_groups, "torso_proxy", 2)
    upper_bodies = _names(body_groups, "upper_non_hand")
    ankles = _names(body_groups, "ankles", 2)
    missing = set(torso + upper_bodies + ankles) - set(body_order)
    if missing:
        raise ValueError(f"floor_style 使用未知 body: {sorted(missing)}")
    status = str(soft.get("soft_failure_status", "")).upper()
    if status != QualityStatus.REVIEW.value:
        raise ValueError("v1 soft_failure_status 必须为 REVIEW")
    exceed_ratio = _finite_float(soft, "exceed_ratio_max")
    review_ratio = _finite_float(floor, "review_ratio")
    severe_multiplier = _finite_float(soft, "severe_multiplier")
    if not 0.0 <= exceed_ratio <= 1.0 or not 0.0 <= review_ratio <= 1.0:
        raise ValueError("ratio 必须位于 [0,1]")
    if severe_multiplier <= 1.0:
        raise ValueError("severe_multiplier 必须大于 1")
    root_min = _finite_float(hard, "root_height_min_absolute")
    root_max = _finite_float(hard, "root_height_max_absolute")
    if root_min >= root_max:
        raise ValueError("root height absolute bounds 非法")
    return SonicNpzQualityConfig(
        motion_contract_version=str(source["motion_contract_version"]),
        fps=fps,
        required_keys=_names(source, "required_keys", 7),
        robot_xml_sha256=str(source["robot_xml_sha256"]),
        preset_sha256=str(source["preset_sha256"]),
        kinematics_sha256=str(source["kinematics_sha256"]),
        joint_order=joint_order,
        joint_lower_limits=lower,
        joint_upper_limits=upper,
        body_order=body_order,
        minimum_frames=_integer(hard, "minimum_frames"),
        quaternion_norm_error_max=_finite_float(hard, "quaternion_norm_error_max"),
        joint_limit_violation_max=_finite_float(hard, "joint_limit_violation_max"),
        minimum_joint_limit_margin_warn=_finite_float(hard, "minimum_joint_limit_margin_warn"),
        root_height_min_absolute=root_min,
        root_height_max_absolute=root_max,
        exceed_ratio_max=exceed_ratio,
        consecutive_exceed_frames=_integer(soft, "consecutive_exceed_frames"),
        severe_multiplier=severe_multiplier,
        dynamics=dynamics,
        root_low_height=_finite_float(floor, "root_low_height"),
        root_low_tilt_degrees=_finite_float(floor, "root_low_tilt_degrees"),
        torso_ground_height=_finite_float(floor, "torso_ground_height"),
        upper_body_ground_height=_finite_float(floor, "upper_body_ground_height"),
        floor_gate_root_height=_finite_float(floor, "gate_root_height"),
        floor_gate_tilt_degrees=_finite_float(floor, "gate_tilt_degrees"),
        ankles_airborne_height=_finite_float(floor, "ankles_airborne_height"),
        floor_reject_consecutive_frames=_integer(floor, "reject_consecutive_frames"),
        floor_review_ratio=review_ratio,
        floor_review_min_frames=_integer(floor, "review_min_frames"),
        low_root_review_height=_finite_float(floor, "low_root_review_height"),
        low_root_review_consecutive_frames=_integer(floor, "low_root_review_consecutive_frames"),
        safe_interval_halo_frames=_integer(floor, "safe_interval_halo_frames", allow_zero=True),
        minimum_safe_interval_frames=_integer(floor, "minimum_safe_interval_frames"),
        torso_proxy_bodies=torso,
        upper_non_hand_bodies=upper_bodies,
        ankle_bodies=ankles,
    )


def sha256_file(path: str | Path) -> str:
    """流式计算资产或动作 SHA256，避免一次性复制压缩 NPZ。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_assets(
    config: SonicNpzQualityConfig,
    *,
    preset_path: Path,
    robot_xml_path: Path,
    kinematics_path: Path,
) -> dict[str, str]:
    """同时校验三个生产资产的哈希、名称顺序和关节限位。"""

    paths = {
        "preset": preset_path.expanduser().resolve(strict=True),
        "robot_xml": robot_xml_path.expanduser().resolve(strict=True),
        "kinematics": kinematics_path.expanduser().resolve(strict=True),
    }
    expected_hashes = {
        "preset": config.preset_sha256,
        "robot_xml": config.robot_xml_sha256,
        "kinematics": config.kinematics_sha256,
    }
    actual_hashes = {name: sha256_file(path) for name, path in paths.items()}
    for name, expected in expected_hashes.items():
        if actual_hashes[name] != expected:
            raise ValueError(
                f"{name} SHA256 不匹配: expected={expected}, actual={actual_hashes[name]}"
            )
    preset = json.loads(paths["preset"].read_text(encoding="utf-8"))
    if tuple(map(str, preset["joint_names_publish_order"])) != config.joint_order:
        raise ValueError("preset publish order 与质量配置不一致")
    kinematics = json.loads(paths["kinematics"].read_text(encoding="utf-8"))
    native_names = tuple(map(str, kinematics["joint_order"]))
    if tuple(map(str, kinematics["body_order"])) != config.body_order:
        raise ValueError("kinematics body order 与质量配置不一致")
    lower_by_name = dict(zip(native_names, map(float, kinematics["joint_lower_limits"])))
    upper_by_name = dict(zip(native_names, map(float, kinematics["joint_upper_limits"])))
    np.testing.assert_allclose(
        [lower_by_name[name] for name in config.joint_order],
        config.joint_lower_limits,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        [upper_by_name[name] for name in config.joint_order],
        config.joint_upper_limits,
        rtol=0.0,
        atol=1e-12,
    )
    return {
        **{f"{name}_path": str(path) for name, path in paths.items()},
        **{f"{name}_sha256": value for name, value in actual_hashes.items()},
    }


def discover_motion_paths(input_root: str | Path) -> list[Path]:
    """确定性发现 NPZ，并拒绝指向输入 root 外部的符号链接。"""

    root = Path(input_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    paths: list[Path] = []
    for candidate in root.rglob("*.npz"):
        if candidate.is_file():
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
            paths.append(resolved)
    paths.sort(key=lambda value: value.relative_to(root).as_posix())
    if not paths:
        raise ValueError(f"{root} 下没有 NPZ")
    return paths


def load_motion_npz(path: str | Path, config: SonicNpzQualityConfig) -> dict[str, np.ndarray]:
    """加载一条部署 NPZ，并执行不允许隐式转换的严格数据契约校验。"""

    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        if tuple(archive.files) != config.required_keys:
            raise ValueError(
                f"NPZ keys/order 应为 {config.required_keys}，实际为 {tuple(archive.files)}"
            )
        arrays = {name: archive[name] for name in config.required_keys}
    frames = int(arrays["joint_pos"].shape[0]) if arrays["joint_pos"].ndim == 2 else -1
    expected_shapes = {
        "fps": (1,),
        "joint_pos": (frames, 21),
        "joint_vel": (frames, 21),
        "body_pos_w": (frames, 22, 3),
        "body_quat_w": (frames, 22, 4),
        "body_lin_vel_w": (frames, 22, 3),
        "body_ang_vel_w": (frames, 22, 3),
    }
    if frames < config.minimum_frames:
        raise ValueError(f"帧数 {frames} 小于最低要求 {config.minimum_frames}")
    for name in config.required_keys:
        value = arrays[name]
        if value.shape != expected_shapes[name]:
            raise ValueError(f"{name} shape 应为 {expected_shapes[name]}，实际为 {value.shape}")
        if value.dtype != np.float32:
            raise ValueError(f"{name} dtype 应为 float32，实际为 {value.dtype}")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} 含 NaN/Inf")
    expected_fps = np.asarray([config.fps], dtype=np.float32)
    if arrays["fps"].tobytes() != expected_fps.tobytes():
        raise ValueError(f"fps 必须精确为 float32 [{config.fps}.0]")
    return arrays


def _longest_true_run(mask: np.ndarray) -> int:
    longest = current = 0
    for value in np.asarray(mask, dtype=np.bool_).reshape(-1):
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return longest


def _signal_metrics(values: np.ndarray, threshold: float) -> dict[str, float | int]:
    sequence = np.asarray(values, dtype=np.float64).reshape(-1)
    if not sequence.size:
        return {
            "sample_count": 0,
            "max": 0.0,
            "p95": 0.0,
            "exceed_ratio": 0.0,
            "max_consecutive_exceed_frames": 0,
        }
    exceed = sequence > threshold
    return {
        "sample_count": int(sequence.size),
        "max": float(np.max(sequence)),
        "p95": float(np.percentile(sequence, 95.0)),
        "exceed_ratio": float(np.mean(exceed)),
        "max_consecutive_exceed_frames": _longest_true_run(exceed),
    }


def _central_difference(values: np.ndarray, fps: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    output = np.zeros_like(array)
    if len(array) <= 1:
        return output
    output[0] = (array[1] - array[0]) * np.float32(fps)
    output[-1] = (array[-1] - array[-2]) * np.float32(fps)
    if len(array) > 2:
        output[1:-1] = (array[2:] - array[:-2]) * np.float32(fps / 2.0)
    return output


def _normalize_wxyz(quaternions: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternions, dtype=np.float64)
    norms = np.linalg.norm(value, axis=-1, keepdims=True)
    return value / np.clip(norms, 1e-12, None)


def _angular_speed_wxyz(quaternions: np.ndarray, fps: float) -> np.ndarray:
    normalized = _normalize_wxyz(quaternions)
    if len(normalized) <= 1:
        return np.zeros((0,), dtype=np.float64)
    dots = np.clip(np.abs(np.sum(normalized[:-1] * normalized[1:], axis=-1)), 0.0, 1.0)
    return 2.0 * np.arccos(dots) * fps


def _priority(status: QualityStatus) -> int:
    return {
        QualityStatus.PASS: 0,
        QualityStatus.REVIEW: 1,
        QualityStatus.REJECT: 2,
    }[status]


def evaluate_motion(
    arrays: Mapping[str, np.ndarray], config: SonicNpzQualityConfig
) -> dict[str, Any]:
    """在已通过契约校验的实际数组上计算物理指标和三态结论。"""

    status = QualityStatus.PASS
    flags: list[tuple[str, QualityStatus]] = []

    def flag(code: str, requested: QualityStatus) -> None:
        nonlocal status
        if code not in {value[0] for value in flags}:
            flags.append((code, requested))
        if _priority(requested) > _priority(status):
            status = requested

    fps = float(config.fps)
    joint_pos = np.asarray(arrays["joint_pos"], dtype=np.float64)
    body_pos = np.asarray(arrays["body_pos_w"], dtype=np.float64)
    body_quat = np.asarray(arrays["body_quat_w"], dtype=np.float64)
    frames = int(len(joint_pos))
    metrics: dict[str, Any] = {"finite": True, "num_frames": frames, "fps": fps}

    quaternion_norms = np.linalg.norm(body_quat, axis=-1)
    quaternion_error = np.abs(quaternion_norms - 1.0)
    root_quat = _normalize_wxyz(body_quat[:, 0])
    adjacent_dot = np.sum(root_quat[:-1] * root_quat[1:], axis=-1)
    metrics["body_quaternion_norm_max_error"] = float(np.max(quaternion_error))
    metrics["root_quaternion_adjacent_dot_min"] = (
        float(np.min(adjacent_dot)) if adjacent_dot.size else 1.0
    )
    metrics["root_quaternion_sign_flip_count"] = int(np.count_nonzero(adjacent_dot < 0.0))
    if metrics["body_quaternion_norm_max_error"] > config.quaternion_norm_error_max:
        flag("BODY_QUATERNION_NORM", QualityStatus.REJECT)

    lower = config.joint_lower_limits[None, :]
    upper = config.joint_upper_limits[None, :]
    limit_violation = np.maximum(np.maximum(lower - joint_pos, joint_pos - upper), 0.0)
    margin = np.minimum(joint_pos - lower, upper - joint_pos)
    per_joint_violation = np.max(limit_violation, axis=0)
    metrics["joint_limit_violation_max"] = float(np.max(per_joint_violation))
    metrics["joint_limit_violation_max_by_joint"] = {
        name: float(value)
        for name, value in zip(config.joint_order, per_joint_violation)
        if value > 0.0
    }
    metrics["minimum_joint_limit_margin"] = float(np.min(margin))
    metrics["minimum_joint_limit_margin_warned"] = bool(
        metrics["minimum_joint_limit_margin"] < config.minimum_joint_limit_margin_warn
    )
    if metrics["joint_limit_violation_max"] > config.joint_limit_violation_max:
        flag("SOURCE_JOINT_LIMIT", QualityStatus.REJECT)

    root_pos = body_pos[:, 0]
    root_height = root_pos[:, 2]
    metrics["root_height_min"] = float(np.min(root_height))
    metrics["root_height_p05"] = float(np.percentile(root_height, 5.0))
    metrics["root_height_max"] = float(np.max(root_height))
    metrics["body_origin_ground_min"] = float(np.min(body_pos[..., 2]))
    if metrics["root_height_min"] < config.root_height_min_absolute:
        flag("ROOT_HEIGHT_BELOW_ABSOLUTE_BOUND", QualityStatus.REJECT)
    if metrics["root_height_max"] > config.root_height_max_absolute:
        flag("ROOT_HEIGHT_ABOVE_ABSOLUTE_BOUND", QualityStatus.REJECT)

    # R[2,2] 是 root 局部 +Z 与世界 +Z 的夹角余弦；wxyz 下为 1-2(x²+y²)。
    up_dot = 1.0 - 2.0 * (root_quat[:, 1] ** 2 + root_quat[:, 2] ** 2)
    root_tilt = np.degrees(np.arccos(np.clip(up_dot, -1.0, 1.0)))
    body_index = {name: index for index, name in enumerate(config.body_order)}
    torso_height = np.mean(
        body_pos[:, [body_index[name] for name in config.torso_proxy_bodies], 2], axis=1
    )
    upper_height = np.min(
        body_pos[:, [body_index[name] for name in config.upper_non_hand_bodies], 2], axis=1
    )
    ankle_height = np.min(
        body_pos[:, [body_index[name] for name in config.ankle_bodies], 2], axis=1
    )
    root_low_tilt = (root_height < config.root_low_height) & (
        root_tilt > config.root_low_tilt_degrees
    )
    torso_ground = torso_height < config.torso_ground_height
    upper_ground = upper_height < config.upper_body_ground_height
    floor_evidence = root_low_tilt | torso_ground | upper_ground
    floor_gate = (
        (root_height < config.floor_gate_root_height)
        | (root_tilt > config.floor_gate_tilt_degrees)
        | (ankle_height > config.ankles_airborne_height)
    )
    floor_mask = floor_evidence & floor_gate
    low_root_mask = root_height < config.low_root_review_height
    floor_count = int(np.count_nonzero(floor_mask))
    floor_ratio = float(np.mean(floor_mask))
    floor_run = _longest_true_run(floor_mask)
    low_root_run = _longest_true_run(low_root_mask)
    metrics["floor_style"] = {
        "torso_proxy": "mean(l_arm_pitch_link, r_arm_pitch_link)",
        "frame_count": floor_count,
        "frame_ratio": floor_ratio,
        "max_consecutive_frames": floor_run,
        "root_low_tilt_frame_count": int(np.count_nonzero(root_low_tilt)),
        "torso_proxy_ground_frame_count": int(np.count_nonzero(torso_ground)),
        "upper_non_hand_ground_frame_count": int(np.count_nonzero(upper_ground)),
        "low_root_frame_ratio": float(np.mean(low_root_mask)),
        "low_root_max_consecutive_frames": low_root_run,
        "root_tilt_p95_degrees": float(np.percentile(root_tilt, 95.0)),
        "root_tilt_max_degrees": float(np.max(root_tilt)),
        "torso_proxy_height_min": float(np.min(torso_height)),
        "upper_non_hand_height_min": float(np.min(upper_height)),
        "ankle_height_min": float(np.min(ankle_height)),
    }
    if floor_run >= config.floor_reject_consecutive_frames:
        flag("FLOOR_STYLE_SUSTAINED", QualityStatus.REJECT)
    elif floor_count >= config.floor_review_min_frames and floor_ratio >= config.floor_review_ratio:
        flag("FLOOR_STYLE_FRAGMENTED", QualityStatus.REVIEW)
    if (
        low_root_run >= config.low_root_review_consecutive_frames
        and floor_run < config.floor_reject_consecutive_frames
    ):
        flag("LOW_ROOT_REVIEW", QualityStatus.REVIEW)

    joint_velocity = np.diff(joint_pos, axis=0) * fps
    joint_acceleration = np.diff(joint_velocity, axis=0) * fps
    joint_jerk = np.diff(joint_acceleration, axis=0) * fps
    root_velocity = np.diff(root_pos, axis=0) * fps
    signals = {
        "joint_velocity_l2": np.linalg.norm(joint_velocity, axis=-1),
        "joint_acceleration_l2": np.linalg.norm(joint_acceleration, axis=-1),
        "joint_jerk_l2": np.linalg.norm(joint_jerk, axis=-1),
        "root_linear_velocity": np.linalg.norm(root_velocity, axis=-1),
        "root_angular_velocity": _angular_speed_wxyz(body_quat[:, 0], fps),
    }
    dynamic_metrics: dict[str, Any] = {}
    for name, values in signals.items():
        threshold, unit = config.dynamics[name]
        summary = _signal_metrics(values, threshold)
        summary.update({"threshold": threshold, "unit": unit})
        dynamic_metrics[name] = summary
        severe = float(summary["max"]) > threshold * config.severe_multiplier
        broad = (
            float(summary["p95"]) > threshold
            and float(summary["exceed_ratio"]) > config.exceed_ratio_max
        )
        sustained = (
            int(summary["max_consecutive_exceed_frames"]) >= config.consecutive_exceed_frames
        )
        code = name.upper()
        if severe:
            flag(f"{code}_SEVERE", QualityStatus.REJECT)
        elif broad or sustained:
            flag(f"{code}_SOFT", QualityStatus.REVIEW)
    metrics["dynamics"] = dynamic_metrics

    joint_vel_expected = _central_difference(arrays["joint_pos"], fps)
    body_lin_expected = _central_difference(arrays["body_pos_w"], fps)
    metrics["stored_velocity_consistency"] = {
        "joint_velocity_central_difference_max_abs_error": float(
            np.max(np.abs(arrays["joint_vel"] - joint_vel_expected))
        ),
        "root_linear_velocity_central_difference_max_abs_error": float(
            np.max(np.abs(arrays["body_lin_vel_w"][:, 0] - body_lin_expected[:, 0]))
        ),
        "all_body_linear_velocity_central_difference_max_abs_error": float(
            np.max(np.abs(arrays["body_lin_vel_w"] - body_lin_expected))
        ),
        "status_affecting": False,
    }
    floor_intervals = mask_to_intervals(floor_mask)
    valid_intervals = safe_intervals_from_bad_mask(
        floor_mask,
        halo_frames=config.safe_interval_halo_frames,
        minimum_frames=config.minimum_safe_interval_frames,
    )
    non_limit_status = QualityStatus.PASS
    for code, requested in flags:
        if code != "SOURCE_JOINT_LIMIT" and _priority(requested) > _priority(non_limit_status):
            non_limit_status = requested
    return {
        "report_contract_version": REPORT_VERSION,
        "status": status.value,
        "status_without_joint_limit": non_limit_status.value,
        "quality_accepted": status is QualityStatus.PASS,
        "reason_codes": [code for code, _ in flags],
        "reason_statuses": {code: requested.value for code, requested in flags},
        "metrics": metrics,
        "floor_intervals": [list(interval) for interval in floor_intervals],
        "valid_intervals": [list(interval) for interval in valid_intervals],
    }


def _identity(path: Path, input_root: Path) -> dict[str, str]:
    relative = path.relative_to(input_root)
    if len(relative.parts) < 2:
        raise ValueError(f"动作必须位于数据集子目录下: {relative}")
    return {
        "dataset": relative.parts[0],
        "sample_id": relative.with_suffix("").as_posix(),
        "source_relative_path": relative.as_posix(),
    }


def evaluate_path(
    path: Path,
    *,
    input_root: Path,
    config: SonicNpzQualityConfig,
    config_sha256: str,
) -> dict[str, Any]:
    """把文件读取或契约异常稳定映射为一条 REJECT 记录。"""

    identity = _identity(path, input_root)
    base = {
        "report_contract_version": REPORT_VERSION,
        "source_motion_contract_version": config.motion_contract_version,
        **identity,
        "quality_config_sha256": config_sha256,
    }
    try:
        source_sha = sha256_file(path)
        arrays = load_motion_npz(path, config)
        decision = evaluate_motion(arrays, config)
    except Exception as exc:
        return {
            **base,
            "source_sha256": locals().get("source_sha"),
            "status": QualityStatus.REJECT.value,
            "status_without_joint_limit": QualityStatus.REJECT.value,
            "quality_accepted": False,
            "reason_codes": ["MOTION_CONTRACT_ERROR"],
            "reason_statuses": {"MOTION_CONTRACT_ERROR": QualityStatus.REJECT.value},
            "metrics": {},
            "floor_intervals": [],
            "valid_intervals": [],
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:2000],
        }
    return {
        **base,
        "source_sha256": source_sha,
        **decision,
        "error_type": None,
        "error_message": None,
    }


def _nested(value: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _percentiles(values: Iterable[Any]) -> dict[str, float | int] | None:
    finite = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and np.isfinite(float(value))
    ]
    if not finite:
        return None
    array = np.asarray(finite, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p50": float(np.percentile(array, 50.0)),
        "p90": float(np.percentile(array, 90.0)),
        "p95": float(np.percentile(array, 95.0)),
        "p99": float(np.percentile(array, 99.0)),
        "max": float(np.max(array)),
    }


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_summary(
    rows: list[dict[str, Any]],
    *,
    input_root: Path,
    config_path: Path,
    config_sha256: str,
    config: SonicNpzQualityConfig,
    assets: Mapping[str, str],
) -> dict[str, Any]:
    """汇总严格状态、去限位反事实状态、原因、分数据集结果和指标分布。"""

    status_counts = Counter(str(row["status"]) for row in rows)
    no_limit_counts = Counter(str(row["status_without_joint_limit"]) for row in rows)
    reason_counts = Counter(str(code) for row in rows for code in row["reason_codes"])
    contract_error_count = reason_counts.get("MOTION_CONTRACT_ERROR", 0)

    def rates(counts: Mapping[str, int], total: int) -> dict[str, float]:
        return {
            name: (float(count) / total if total else 0.0) for name, count in sorted(counts.items())
        }

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["dataset"])].append(row)
    by_dataset: dict[str, Any] = {}
    for dataset, values in sorted(groups.items()):
        frames = sum(int(_nested(row, "metrics", "num_frames") or 0) for row in values)
        by_dataset[dataset] = {
            "sequences": len(values),
            "frames": frames,
            "hours": frames / config.fps / 3600.0,
            "status_counts": dict(dataset_status := Counter(str(row["status"]) for row in values)),
            "status_rates": rates(dataset_status, len(values)),
            "status_without_joint_limit_counts": dict(
                dataset_no_limit := Counter(
                    str(row["status_without_joint_limit"]) for row in values
                )
            ),
            "status_without_joint_limit_rates": rates(dataset_no_limit, len(values)),
            "reason_counts": dict(
                Counter(str(code) for row in values for code in row["reason_codes"])
            ),
        }
    distribution_paths = {
        "joint_limit_violation_max": ("metrics", "joint_limit_violation_max"),
        "root_height_p05": ("metrics", "root_height_p05"),
        "body_origin_ground_min": ("metrics", "body_origin_ground_min"),
        "floor_frame_ratio": ("metrics", "floor_style", "frame_ratio"),
        "floor_max_consecutive_frames": (
            "metrics",
            "floor_style",
            "max_consecutive_frames",
        ),
        "joint_velocity_l2_p95": (
            "metrics",
            "dynamics",
            "joint_velocity_l2",
            "p95",
        ),
        "joint_acceleration_l2_p95": (
            "metrics",
            "dynamics",
            "joint_acceleration_l2",
            "p95",
        ),
        "joint_jerk_l2_p95": ("metrics", "dynamics", "joint_jerk_l2", "p95"),
        "root_linear_velocity_p95": (
            "metrics",
            "dynamics",
            "root_linear_velocity",
            "p95",
        ),
        "root_angular_velocity_p95": (
            "metrics",
            "dynamics",
            "root_angular_velocity",
            "p95",
        ),
        "joint_velocity_field_max_abs_error": (
            "metrics",
            "stored_velocity_consistency",
            "joint_velocity_central_difference_max_abs_error",
        ),
        "all_body_linear_velocity_field_max_abs_error": (
            "metrics",
            "stored_velocity_consistency",
            "all_body_linear_velocity_central_difference_max_abs_error",
        ),
        "root_linear_velocity_field_max_abs_error": (
            "metrics",
            "stored_velocity_consistency",
            "root_linear_velocity_central_difference_max_abs_error",
        ),
    }
    distributions = {
        name: result
        for name, keys in distribution_paths.items()
        if (result := _percentiles(_nested(row, *keys) for row in rows)) is not None
    }
    frames = sum(int(_nested(row, "metrics", "num_frames") or 0) for row in rows)
    joint_limit_by_joint: dict[str, Any] = {}
    for joint_name in config.joint_order:
        values = np.asarray(
            [
                float(
                    (_nested(row, "metrics", "joint_limit_violation_max_by_joint") or {}).get(
                        joint_name, 0.0
                    )
                )
                for row in rows
                if _nested(row, "metrics", "joint_limit_violation_max_by_joint") is not None
            ],
            dtype=np.float64,
        )
        joint_limit_by_joint[joint_name] = {
            "violating_sequences": int(np.count_nonzero(values > config.joint_limit_violation_max)),
            "violating_rate": float(
                np.mean(values > config.joint_limit_violation_max) if values.size else 0.0
            ),
            "max": float(np.max(values) if values.size else 0.0),
            "p95": float(np.percentile(values, 95.0) if values.size else 0.0),
        }
    tolerance_sensitivity: dict[str, Any] = {}
    for tolerance in (0.0001, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2):
        counts: Counter[str] = Counter()
        for row in rows:
            violation = _nested(row, "metrics", "joint_limit_violation_max")
            if violation is None or float(violation) > tolerance:
                counts[QualityStatus.REJECT.value] += 1
            else:
                counts[str(row["status_without_joint_limit"])] += 1
        tolerance_sensitivity[f"{tolerance:.4g}"] = {
            "status_counts": dict(counts),
            "status_rates": rates(counts, len(rows)),
        }
    return {
        "report_contract_version": REPORT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_root": str(input_root),
        "source_files_modified": False,
        "source_motion_contract_version": config.motion_contract_version,
        "fps": config.fps,
        "quality_config": str(config_path),
        "quality_config_sha256": config_sha256,
        "verified_assets": dict(assets),
        "sequences": len(rows),
        "motion_contract_valid_sequences": len(rows) - contract_error_count,
        "motion_contract_error_sequences": contract_error_count,
        "frames": frames,
        "hours": frames / config.fps / 3600.0,
        "quality_accepted_sequences": status_counts.get(QualityStatus.PASS.value, 0),
        "status_counts": dict(status_counts),
        "status_rates": rates(status_counts, len(rows)),
        "status_without_joint_limit_counts": dict(no_limit_counts),
        "status_without_joint_limit_rates": rates(no_limit_counts, len(rows)),
        "reason_counts": dict(reason_counts),
        "by_dataset": by_dataset,
        "metric_distributions": distributions,
        "joint_limit_by_joint": joint_limit_by_joint,
        "joint_limit_tolerance_sensitivity": tolerance_sensitivity,
        "decision_scope": ("离线 50Hz 运动学/动力学预检查；未执行 Isaac-Lab rollout 或 SONIC 跟踪"),
        "compatibility_notes": {
            "continuous_frame_thresholds_scaled_from_30hz_to_50hz": True,
            "torso_proxy": "mean(l_arm_pitch_link, r_arm_pitch_link)",
            "virtual_hand_rule_removed": "legacy hands only supplied diagnostics",
            "legacy_zero_body_origin_ground_gauge_disabled": (
                "GMR export uses offset_to_ground with 0.05m clearance"
            ),
            "stored_velocity_consistency_is_diagnostic_only": True,
            "status_without_joint_limit_is_diagnostic_only": True,
        },
    }


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    try:
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _flat_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset": row.get("dataset"),
        "sample_id": row.get("sample_id"),
        "source_relative_path": row.get("source_relative_path"),
        "source_sha256": row.get("source_sha256"),
        "status": row.get("status"),
        "status_without_joint_limit": row.get("status_without_joint_limit"),
        "quality_accepted": row.get("quality_accepted"),
        "reason_codes": "|".join(map(str, row.get("reason_codes", ()))),
        "num_frames": _nested(row, "metrics", "num_frames"),
        "joint_limit_violation_max": _nested(row, "metrics", "joint_limit_violation_max"),
        "root_height_min": _nested(row, "metrics", "root_height_min"),
        "root_height_p05": _nested(row, "metrics", "root_height_p05"),
        "body_origin_ground_min": _nested(row, "metrics", "body_origin_ground_min"),
        "floor_frame_ratio": _nested(row, "metrics", "floor_style", "frame_ratio"),
        "floor_max_run": _nested(row, "metrics", "floor_style", "max_consecutive_frames"),
        "joint_velocity_p95": _nested(row, "metrics", "dynamics", "joint_velocity_l2", "p95"),
        "joint_acceleration_p95": _nested(
            row, "metrics", "dynamics", "joint_acceleration_l2", "p95"
        ),
        "joint_jerk_p95": _nested(row, "metrics", "dynamics", "joint_jerk_l2", "p95"),
        "root_linear_velocity_p95": _nested(
            row, "metrics", "dynamics", "root_linear_velocity", "p95"
        ),
        "root_angular_velocity_p95": _nested(
            row, "metrics", "dynamics", "root_angular_velocity", "p95"
        ),
        "body_linear_velocity_field_max_abs_error": _nested(
            row,
            "metrics",
            "stored_velocity_consistency",
            "all_body_linear_velocity_central_difference_max_abs_error",
        ),
        "error_type": row.get("error_type"),
        "error_message": row.get("error_message"),
    }


def write_reports(
    output_dir: Path,
    rows: list[dict[str, Any]],
    summary: Mapping[str, Any],
    config_path: Path,
    *,
    overwrite: bool,
) -> None:
    """原子写出完整报告；已有结果必须显式 --overwrite 才能替换。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    occupied = [output_dir / name for name in REPORT_FILENAMES if (output_dir / name).exists()]
    if occupied and not overwrite:
        raise FileExistsError(f"报告已存在，需 --overwrite: {occupied}")
    _atomic_write(
        output_dir / "quality_report.jsonl",
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
    )
    review = [row for row in rows if row["status"] != QualityStatus.PASS.value]
    review.sort(
        key=lambda row: (
            0 if row["status"] == QualityStatus.REJECT.value else 1,
            -float(_nested(row, "metrics", "joint_limit_violation_max") or 0.0),
            str(row["sample_id"]),
        )
    )
    _atomic_write(
        output_dir / "review_candidates.jsonl",
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in review),
    )
    _atomic_write(
        output_dir / "quality_summary.json",
        json.dumps(dict(summary), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )
    _atomic_write(
        output_dir / "quality_config.snapshot.yaml",
        config_path.read_text(encoding="utf-8"),
    )
    list_specs = {
        "strict_pass.txt": lambda row: row["status"] == QualityStatus.PASS.value,
        "strict_reject.txt": lambda row: row["status"] == QualityStatus.REJECT.value,
        "without_joint_limit_review.txt": lambda row: (
            row["status_without_joint_limit"] == QualityStatus.REVIEW.value
        ),
        "without_joint_limit_reject.txt": lambda row: (
            row["status_without_joint_limit"] == QualityStatus.REJECT.value
        ),
    }
    for filename, selected in list_specs.items():
        _atomic_write(
            output_dir / filename,
            "".join(f"{row['source_relative_path']}\n" for row in rows if selected(row)),
        )
    flat = [_flat_row(row) for row in rows]
    csv_path = output_dir / "quality_report.csv"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=output_dir,
        prefix=f".{csv_path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    try:
        os.replace(temporary, csv_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root", type=Path, default=REPO_ROOT / "data/motions_npz_bumi3_smooth_q1"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preset", type=Path, default=DEFAULT_PRESET)
    parser.add_argument("--robot-xml", type=Path, default=DEFAULT_ROBOT_XML)
    parser.add_argument("--kinematics", type=Path, default=DEFAULT_KINEMATICS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="仅检查排序后的前 N 条，用于代码验证")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """绑定生产资产、扫描真实 NPZ 并输出只读检查报告。"""

    args = parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit 必须为正整数")
    input_root = args.input_root.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir == input_root or input_root in output_dir.parents:
        raise ValueError("--output-dir 必须位于输入目录之外")
    config_path = args.config.expanduser().resolve(strict=True)
    config = load_config(config_path)
    assets = verify_assets(
        config,
        preset_path=args.preset,
        robot_xml_path=args.robot_xml,
        kinematics_path=args.kinematics,
    )
    config_sha256 = sha256_file(config_path)
    paths = discover_motion_paths(input_root)
    if args.limit is not None:
        paths = paths[: args.limit]
    rows = [
        evaluate_path(
            path,
            input_root=input_root,
            config=config,
            config_sha256=config_sha256,
        )
        for path in tqdm(paths, desc="BUMI 50Hz physical quality")
    ]
    summary = build_summary(
        rows,
        input_root=input_root,
        config_path=config_path,
        config_sha256=config_sha256,
        config=config,
        assets=assets,
    )
    write_reports(output_dir, rows, summary, config_path, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "report_dir": str(output_dir),
                "sequences": len(rows),
                "status_counts": summary["status_counts"],
                "status_without_joint_limit_counts": summary["status_without_joint_limit_counts"],
                "source_files_modified": False,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
