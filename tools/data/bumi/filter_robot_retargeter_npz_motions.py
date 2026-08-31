#!/usr/bin/env python3
"""筛选 robot_retargeter 生成的 BUMI3 30 Hz Mimic NPZ。

本工具是新 ``robot_retargeter`` 与 GENMO 正式 BUMI 音乐训练数据之间的只读质量
边界。输入必须是四库全量 release 目录：每条成功动作同时具有 30 Hz Mimic NPZ、
同名重定向 metadata 和验证报告；目录根还必须具有
``robot_retargeter.bumi3_full_release.v1`` release report。脚本先核对 release 中明确
排除的上游失败项，再验证 MJCF、重定向配置、GENMO 运动学 JSON 的 SHA256，防止旧
GMR/旧机器人资产或错误关节顺序被静默复用。

对每条剩余动作，工具严格检查 13 字段 NPZ、30 Hz、float32 数值数组、wxyz 四元数、
21 关节和 22 body 名称；同时要求逐条报告确认输入和输出都是右手 Z-up 米制坐标。
随后从最终 NPZ 重新计算关节限位、速度、加速度、jerk、Root 高度/倾角以及倒地动作，
给出 ``PASS / REVIEW / REJECT`` 三态结果。重定向报告中的软警告只作为诊断保留，
不替代这里对最终数据的重新计算。正式训练构建器只读取 PASS，REVIEW/REJECT 永不
进入 manifest。

脚本不会删除、移动或改写源动作。JSONL、CSV、汇总、严格 PASS 列表和上游8条排除
记录都写入独立输出目录；已有报告只有显式 ``--overwrite`` 才会被原子替换。这里的
PASS 仅代表离线运动学/动力学数据门禁，不代表控制器、仿真 rollout 或实机安全通过。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from gem.robots.bumi.kinematics import BumiKinematics  # noqa: E402
from gem.robots.bumi.legacy_motion import sha256_file  # noqa: E402
from gem.robots.bumi.quality_filter import QualityStatus  # noqa: E402
from tools.data.bumi.filter_sonic_npz_motions import (  # noqa: E402
    build_summary,
    evaluate_motion,
    write_reports,
)

CONFIG_VERSION = "genmo.bumi_robot_retargeter_npz_quality_config.v1"
REPORT_VERSION = "genmo.bumi_quality_report.robot_retargeter_npz_30hz.v1"
SOURCE_CONTRACT_VERSION = "robot_retargeter.bumi3_mimic_npz_30hz.v1"
DEFAULT_CONFIG = REPO_ROOT / "configs/bumi/quality_filter_robot_retargeter_30hz_v1.yaml"
EXPECTED_DATASETS = ("aioz_gdance", "aistpp", "compas3d", "finedance")


@dataclass(frozen=True)
class RobotRetargeterQualityConfig:
    """完成结构验证的30 Hz质量规则；字段兼容通用离线评估核心。"""

    motion_contract_version: str
    release_report_schema: str
    fps: int
    required_npz_keys: tuple[str, ...]
    robot_name: str
    anchor_body_name: str
    robot_xml_sha256: str
    retarget_config_sha256: str
    kinematics_sha256: str
    coordinate_contract: Mapping[str, Any]
    joint_order: tuple[str, ...]
    joint_lower_limits: np.ndarray
    joint_upper_limits: np.ndarray
    body_order: tuple[str, ...]
    expected_selected: int
    expected_completed: int
    expected_failed: int
    expected_completed_by_dataset: Mapping[str, int]
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
    root_tilt_review_median_degrees: float
    root_tilt_reject_median_degrees: float
    root_tilt_review_p95_degrees: float
    root_tilt_reject_p95_degrees: float
    root_tilt_review_over_45_fraction: float
    root_tilt_reject_over_45_fraction: float
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


def _finite(parent: Mapping[str, Any], key: str) -> float:
    try:
        value = float(parent[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{key} 必须是有限数值") from exc
    if not math.isfinite(value):
        raise ValueError(f"{key} 必须是有限数值")
    return value


def _positive_int(parent: Mapping[str, Any], key: str) -> int:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} 必须是正整数")
    return value


def _names(parent: Mapping[str, Any], key: str, length: int | None = None) -> tuple[str, ...]:
    value = tuple(map(str, parent.get(key, ())))
    if not value or len(value) != len(set(value)):
        raise ValueError(f"{key} 必须是非空且不重复的名称列表")
    if length is not None and len(value) != length:
        raise ValueError(f"{key} 长度必须为 {length}，实际为 {len(value)}")
    return value


def _vector(parent: Mapping[str, Any], key: str, length: int) -> np.ndarray:
    value = np.asarray(parent.get(key), dtype=np.float64)
    if value.shape != (length,) or not np.isfinite(value).all():
        raise ValueError(f"{key} 必须是长度 {length} 的有限向量")
    return value


def load_config(path: str | Path) -> RobotRetargeterQualityConfig:
    """读取并严格验证30 Hz契约，拒绝隐式默认和互相矛盾的阈值。"""

    config_path = Path(path).expanduser().resolve(strict=True)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("contract_version") != CONFIG_VERSION:
        raise ValueError(f"质量配置 contract_version 必须为 {CONFIG_VERSION}")
    source = _mapping(raw, "source")
    upstream = _mapping(raw, "upstream_release")
    hard = _mapping(raw, "hard_thresholds")
    soft = _mapping(raw, "soft_policy")
    root = _mapping(raw, "root_orientation")
    floor = _mapping(raw, "floor_style")
    body_groups = _mapping(floor, "body_groups")
    fps = _positive_int(source, "fps")
    if fps != 30:
        raise ValueError(f"robot_retargeter 输入契约固定要求30 Hz，实际为 {fps}")
    if source.get("quaternion_convention") != "wxyz":
        raise ValueError("source.quaternion_convention 必须为 wxyz")
    if source.get("motion_contract_version") != SOURCE_CONTRACT_VERSION:
        raise ValueError(f"source.motion_contract_version 必须为 {SOURCE_CONTRACT_VERSION}")
    required_keys = _names(source, "required_npz_keys", 13)
    joints = _names(source, "joint_order", 21)
    bodies = _names(source, "body_order", 22)
    lower = _vector(source, "joint_lower_limits", 21)
    upper = _vector(source, "joint_upper_limits", 21)
    if np.any(lower >= upper):
        raise ValueError("每个关节必须满足 lower < upper")
    coordinate = _mapping(source, "coordinate_contract")
    expected_coordinate = {
        "source_coordinate_system": "right_handed_z_up_metric",
        "requested_up_axis": "z",
        "y_up_to_z_up_conversion_applied": False,
        "output_coordinate_system": "right_handed_z_up_metric",
    }
    if dict(coordinate) != expected_coordinate:
        raise ValueError(f"source.coordinate_contract 必须精确为 {expected_coordinate}")
    expected_counts = {str(k): int(v) for k, v in _mapping(upstream, "expected_completed_by_dataset").items()}
    if set(expected_counts) != set(EXPECTED_DATASETS) or any(v <= 0 for v in expected_counts.values()):
        raise ValueError("expected_completed_by_dataset 必须完整覆盖四库且数量为正")
    completed = _positive_int(upstream, "expected_completed")
    failed = _positive_int(upstream, "expected_failed")
    selected = _positive_int(upstream, "expected_selected")
    if completed != sum(expected_counts.values()) or selected != completed + failed:
        raise ValueError("upstream_release 数量自相矛盾")
    expected_dynamics = {
        "joint_velocity_l2",
        "joint_acceleration_l2",
        "joint_jerk_l2",
        "root_linear_velocity",
        "root_angular_velocity",
    }
    dynamics_raw = _mapping(raw, "dynamics")
    if set(dynamics_raw) != expected_dynamics:
        raise ValueError(f"dynamics 键必须精确为 {sorted(expected_dynamics)}")
    dynamics: dict[str, tuple[float, str]] = {}
    for name in sorted(expected_dynamics):
        item = _mapping(dynamics_raw, name)
        threshold = _finite(item, "threshold")
        unit = str(item.get("unit", "")).strip()
        if threshold <= 0.0 or not unit:
            raise ValueError(f"dynamics.{name} 的 threshold/unit 非法")
        dynamics[name] = (threshold, unit)
    if str(soft.get("soft_failure_status", "")).upper() != QualityStatus.REVIEW.value:
        raise ValueError("soft_failure_status 必须为 REVIEW")
    review_values = (
        _finite(root, "review_median_degrees"),
        _finite(root, "review_p95_degrees"),
        _finite(root, "review_over_45deg_fraction"),
    )
    reject_values = (
        _finite(root, "reject_median_degrees"),
        _finite(root, "reject_p95_degrees"),
        _finite(root, "reject_over_45deg_fraction"),
    )
    if any(review >= reject for review, reject in zip(review_values, reject_values, strict=True)):
        raise ValueError("每项 Root REVIEW 阈值必须严格小于 REJECT 阈值")
    if not 0.0 <= review_values[2] < reject_values[2] <= 1.0:
        raise ValueError("Root over45 fraction 阈值必须位于 [0,1]")
    torso = _names(body_groups, "torso_proxy", 2)
    upper_bodies = _names(body_groups, "upper_non_hand")
    ankles = _names(body_groups, "ankles", 2)
    missing_bodies = set(torso + upper_bodies + ankles) - set(bodies)
    if missing_bodies:
        raise ValueError(f"floor_style 使用未知 body: {sorted(missing_bodies)}")
    exceed_ratio = _finite(soft, "exceed_ratio_max")
    floor_ratio = _finite(floor, "review_ratio")
    severe = _finite(soft, "severe_multiplier")
    if not 0.0 <= exceed_ratio <= 1.0 or not 0.0 <= floor_ratio <= 1.0:
        raise ValueError("比例阈值必须位于 [0,1]")
    if severe <= 1.0:
        raise ValueError("severe_multiplier 必须大于1")
    return RobotRetargeterQualityConfig(
        motion_contract_version=str(source["motion_contract_version"]),
        release_report_schema=str(source.get("release_report_schema", "")),
        fps=fps,
        required_npz_keys=required_keys,
        robot_name=str(source.get("robot_name", "")),
        anchor_body_name=str(source.get("anchor_body_name", "")),
        robot_xml_sha256=str(source.get("robot_xml_sha256", "")),
        retarget_config_sha256=str(source.get("retarget_config_sha256", "")),
        kinematics_sha256=str(source.get("kinematics_sha256", "")),
        coordinate_contract=dict(coordinate),
        joint_order=joints,
        joint_lower_limits=lower,
        joint_upper_limits=upper,
        body_order=bodies,
        expected_selected=selected,
        expected_completed=completed,
        expected_failed=failed,
        expected_completed_by_dataset=expected_counts,
        minimum_frames=_positive_int(hard, "minimum_frames"),
        quaternion_norm_error_max=_finite(hard, "quaternion_norm_error_max"),
        joint_limit_violation_max=_finite(hard, "joint_limit_violation_max"),
        minimum_joint_limit_margin_warn=_finite(hard, "minimum_joint_limit_margin_warn"),
        root_height_min_absolute=_finite(hard, "root_height_min_absolute"),
        root_height_max_absolute=_finite(hard, "root_height_max_absolute"),
        exceed_ratio_max=exceed_ratio,
        consecutive_exceed_frames=_positive_int(soft, "consecutive_exceed_frames"),
        severe_multiplier=severe,
        dynamics=dynamics,
        root_tilt_review_median_degrees=review_values[0],
        root_tilt_reject_median_degrees=reject_values[0],
        root_tilt_review_p95_degrees=review_values[1],
        root_tilt_reject_p95_degrees=reject_values[1],
        root_tilt_review_over_45_fraction=review_values[2],
        root_tilt_reject_over_45_fraction=reject_values[2],
        root_low_height=_finite(floor, "root_low_height"),
        root_low_tilt_degrees=_finite(floor, "root_low_tilt_degrees"),
        torso_ground_height=_finite(floor, "torso_ground_height"),
        upper_body_ground_height=_finite(floor, "upper_body_ground_height"),
        floor_gate_root_height=_finite(floor, "gate_root_height"),
        floor_gate_tilt_degrees=_finite(floor, "gate_tilt_degrees"),
        ankles_airborne_height=_finite(floor, "ankles_airborne_height"),
        floor_reject_consecutive_frames=_positive_int(floor, "reject_consecutive_frames"),
        floor_review_ratio=floor_ratio,
        floor_review_min_frames=_positive_int(floor, "review_min_frames"),
        low_root_review_height=_finite(floor, "low_root_review_height"),
        low_root_review_consecutive_frames=_positive_int(
            floor, "low_root_review_consecutive_frames"
        ),
        safe_interval_halo_frames=_positive_int(floor, "safe_interval_halo_frames"),
        minimum_safe_interval_frames=_positive_int(floor, "minimum_safe_interval_frames"),
        torso_proxy_bodies=torso,
        upper_non_hand_bodies=upper_bodies,
        ankle_bodies=ankles,
    )


def verify_assets(
    config: RobotRetargeterQualityConfig,
    *,
    robot_xml: Path,
    retarget_config: Path,
    kinematics: Path,
) -> dict[str, str]:
    """核对三项生产资产及运动学内部来源，拒绝同名不同内容。"""

    paths = {
        "robot_xml": robot_xml.expanduser().resolve(strict=True),
        "retarget_config": retarget_config.expanduser().resolve(strict=True),
        "kinematics": kinematics.expanduser().resolve(strict=True),
    }
    expected = {
        "robot_xml": config.robot_xml_sha256,
        "retarget_config": config.retarget_config_sha256,
        "kinematics": config.kinematics_sha256,
    }
    actual = {name: sha256_file(path) for name, path in paths.items()}
    mismatch = {
        name: {"expected": expected[name], "actual": actual[name]}
        for name in paths
        if actual[name] != expected[name]
    }
    if mismatch:
        raise ValueError(f"robot_retargeter 30 Hz资产 SHA 不匹配: {mismatch}")
    kin = BumiKinematics(paths["kinematics"])
    if kin.source_mjcf_sha256 != config.robot_xml_sha256:
        raise ValueError("运动学 JSON 不是从当前 robot_retargeter MJCF 导出")
    if set(kin.joint_order) != set(config.joint_order):
        raise ValueError("运动学与 NPZ 的关节名称集合不一致")
    if set(kin.body_order) != set(config.body_order):
        raise ValueError("运动学与 NPZ 的 body 名称集合不一致")
    return {
        **{f"{name}_path": str(path) for name, path in paths.items()},
        **{f"{name}_sha256": digest for name, digest in actual.items()},
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 必须是 object: {path}")
    return value


def verify_release_report(
    path: Path,
    *,
    config: RobotRetargeterQualityConfig,
    input_root: Path,
) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """核对上游3162=3154+8集合，并返回成功项和明确排除项。"""

    release_path = path.expanduser().resolve(strict=True)
    release = _read_json(release_path)
    if release.get("schema") != config.release_report_schema:
        raise ValueError("release report schema 不匹配")
    contracts = _mapping(release, "contracts")
    if contracts.get("config_sha256") != config.retarget_config_sha256:
        raise ValueError("release report retarget config SHA 不匹配")
    if contracts.get("mjcf_sha256") != config.robot_xml_sha256:
        raise ValueError("release report MJCF SHA 不匹配")
    counts = _mapping(release, "counts")
    expected_counts = {
        "selected": config.expected_selected,
        "completed": config.expected_completed,
        "failed": config.expected_failed,
    }
    for key, expected in expected_counts.items():
        if int(counts.get(key, -1)) != expected:
            raise ValueError(f"release counts.{key} 应为 {expected}，实际 {counts.get(key)!r}")
    results = release.get("results")
    failures = release.get("failures")
    if not isinstance(results, list) or not isinstance(failures, list):
        raise ValueError("release results/failures 必须是列表")
    completed_rows = [row for row in results if row.get("status") == "completed"]
    if len(completed_rows) != config.expected_completed or len(failures) != config.expected_failed:
        raise ValueError("release 成功/失败明细数量与汇总不一致")
    completed_by_dataset = Counter(str(row.get("dataset")) for row in completed_rows)
    if dict(completed_by_dataset) != dict(config.expected_completed_by_dataset):
        raise ValueError(
            "release 成功四库数量不一致: "
            f"expected={dict(config.expected_completed_by_dataset)}, actual={dict(completed_by_dataset)}"
        )
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in completed_rows:
        dataset = str(row.get("dataset", ""))
        stem = str(row.get("stem", ""))
        if dataset not in EXPECTED_DATASETS or not stem or Path(stem).name != stem:
            raise ValueError(f"release completed identity 非法: {dataset}/{stem}")
        key = (dataset, stem)
        if key in index:
            raise ValueError(f"release completed 重复: {dataset}/{stem}")
        index[key] = dict(row)
    expected_paths = {
        (input_root / dataset / "mimic_npz" / "bumi3" / f"{stem}.npz").resolve()
        for dataset, stem in index
    }
    actual_paths = {
        path.resolve()
        for dataset in EXPECTED_DATASETS
        for path in (input_root / dataset / "mimic_npz" / "bumi3").glob("*.npz")
        if path.is_file()
    }
    if actual_paths != expected_paths:
        raise ValueError(
            "NPZ集合与 release completed 集合不一致: "
            f"missing={len(expected_paths - actual_paths)}, extra={len(actual_paths - expected_paths)}"
        )
    return index, [dict(row) for row in failures], {
        "release_report": str(release_path),
        "release_report_sha256": sha256_file(release_path),
        "upstream_selected": config.expected_selected,
        "upstream_completed": config.expected_completed,
        "upstream_failed_excluded": config.expected_failed,
    }


def load_motion_npz(
    path: str | Path, config: RobotRetargeterQualityConfig
) -> dict[str, np.ndarray]:
    """严格加载13字段30 Hz NPZ，不对 dtype、帧率或名称做隐式转换。"""

    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        if tuple(archive.files) != config.required_npz_keys:
            raise ValueError(
                f"NPZ keys/order 应为 {config.required_npz_keys}，实际为 {tuple(archive.files)}"
            )
        arrays = {name: archive[name] for name in config.required_npz_keys}
    if arrays["fps"].shape != () or arrays["fps"].dtype != np.float64:
        raise ValueError("fps 必须是 float64 scalar")
    if float(arrays["fps"]) != float(config.fps):
        raise ValueError(f"fps 必须精确为 {config.fps}.0")
    frames = int(arrays["joint_pos"].shape[0]) if arrays["joint_pos"].ndim == 2 else -1
    expected_shapes = {
        "joint_pos": (frames, 21),
        "joint_vel": (frames, 21),
        "body_pos_w": (frames, 22, 3),
        "body_quat_w": (frames, 22, 4),
        "body_lin_vel_w": (frames, 22, 3),
        "body_ang_vel_w": (frames, 22, 3),
    }
    if frames < config.minimum_frames:
        raise ValueError(f"帧数 {frames} 小于最低要求 {config.minimum_frames}")
    for name, shape in expected_shapes.items():
        value = arrays[name]
        if value.shape != shape or value.dtype != np.float32:
            raise ValueError(f"{name} 应为 float32 {shape}，实际 {value.dtype} {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} 含 NaN/Inf")
    scalar_expectations = {
        "anchor_body_name": config.anchor_body_name,
        "robot_name": config.robot_name,
        "quaternion_order": "wxyz",
    }
    for name, expected in scalar_expectations.items():
        if arrays[name].shape != () or str(arrays[name].item()) != expected:
            raise ValueError(f"{name} 必须为 {expected!r}")
    if arrays["source_motion"].shape != () or not str(arrays["source_motion"].item()):
        raise ValueError("source_motion 必须是非空 scalar string")
    if tuple(map(str, arrays["joint_names"].tolist())) != config.joint_order:
        raise ValueError("joint_names 与30 Hz契约顺序不一致")
    if tuple(map(str, arrays["body_names"].tolist())) != config.body_order:
        raise ValueError("body_names 与30 Hz契约顺序不一致")
    return arrays


def _warning_code(value: Any) -> str:
    text = str(value)
    return text.split(":", 1)[0].strip()


def _validate_sidecars(
    *,
    input_root: Path,
    dataset: str,
    stem: str,
    arrays: Mapping[str, np.ndarray],
    config: RobotRetargeterQualityConfig,
    release_row: Mapping[str, Any],
) -> dict[str, Any]:
    """复核 metadata、逐条报告与 release 身份，返回可审计诊断摘要。"""

    metadata_path = input_root / dataset / "robot_motion" / f"{stem}_bumi3.meta.json"
    report_path = input_root / dataset / "reports" / f"{stem}_bumi3.json"
    if not metadata_path.is_file() or not report_path.is_file():
        raise FileNotFoundError(f"{dataset}/{stem}: 缺少 metadata 或逐条报告")
    metadata = _read_json(metadata_path)
    report = _read_json(report_path)
    frames = int(arrays["joint_pos"].shape[0])
    expected_metadata = {
        "robot": config.robot_name,
        "fps": float(config.fps),
        "num_frames": frames,
        "qpos_size": 28,
        "robot_xml_sha256": config.robot_xml_sha256,
        "config_sha256": config.retarget_config_sha256,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"{dataset}/{stem}: metadata.{key} 契约不匹配")
    if tuple(map(str, metadata.get("isaac_joint_names", ()))) != config.joint_order:
        raise ValueError(f"{dataset}/{stem}: metadata isaac_joint_names 不匹配")
    if tuple(map(str, metadata.get("isaac_body_names", ()))) != config.body_order:
        raise ValueError(f"{dataset}/{stem}: metadata isaac_body_names 不匹配")
    if str(metadata.get("source_motion")) != str(arrays["source_motion"].item()):
        raise ValueError(f"{dataset}/{stem}: metadata/NPZ source_motion 不一致")
    if str(release_row.get("source_path")) != str(arrays["source_motion"].item()):
        raise ValueError(f"{dataset}/{stem}: release/NPZ source_motion 不一致")
    if report.get("status") != "passed" or report.get("failures") != []:
        raise ValueError(f"{dataset}/{stem}: 上游逐条验证没有 passed")
    motion = _mapping(_mapping(report, "checks"), "motion")
    if dict(_mapping(motion, "coordinate_contract")) != dict(config.coordinate_contract):
        raise ValueError(f"{dataset}/{stem}: 坐标契约不匹配")
    if list(motion.get("csv_shape", ())) != [frames, 28] or int(motion.get("nan_inf_count", -1)) != 0:
        raise ValueError(f"{dataset}/{stem}: 上游 CSV shape/finite 门禁不匹配")
    warnings = list(map(str, report.get("warnings", ())))
    return {
        "retarget_metadata_path": str(metadata_path),
        "retarget_report_path": str(report_path),
        "retarget_report_warnings": warnings,
        "retarget_report_warning_codes": sorted({_warning_code(value) for value in warnings}),
        "retarget_report_warning_count": len(warnings),
    }


def evaluate_path(
    path: Path,
    *,
    input_root: Path,
    config: RobotRetargeterQualityConfig,
    config_sha256: str,
    release_row: Mapping[str, Any],
) -> dict[str, Any]:
    """把一条源文件及其 sidecar 稳定映射成三态记录。"""

    relative = path.relative_to(input_root)
    dataset = relative.parts[0]
    stem = path.stem
    base = {
        "report_contract_version": REPORT_VERSION,
        "source_motion_contract_version": config.motion_contract_version,
        "dataset": dataset,
        "sample_id": f"{dataset}/{stem}",
        "source_relative_path": relative.as_posix(),
        "quality_config_sha256": config_sha256,
        "source_mjcf_sha256": config.robot_xml_sha256,
        "retarget_config_sha256": config.retarget_config_sha256,
    }
    try:
        source_sha = sha256_file(path)
        arrays = load_motion_npz(path, config)
        sidecars = _validate_sidecars(
            input_root=input_root,
            dataset=dataset,
            stem=stem,
            arrays=arrays,
            config=config,
            release_row=release_row,
        )
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
        **sidecars,
        "error_type": None,
        "error_message": None,
    }


def _worker(argument: tuple[str, str, RobotRetargeterQualityConfig, str, dict[str, Any]]) -> dict[str, Any]:
    path, input_root, config, config_sha256, release_row = argument
    return evaluate_path(
        Path(path),
        input_root=Path(input_root),
        config=config,
        config_sha256=config_sha256,
        release_row=release_row,
    )


def _atomic_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write("".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    try:
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--release-report", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--robot-xml", required=True, type=Path)
    parser.add_argument("--retarget-config", required=True, type=Path)
    parser.add_argument("--kinematics", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 8))
    parser.add_argument("--limit", type=int, help="只检查确定性排序后的前N条，用于临时测试")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """绑定生产资产，完成只读全量筛选并发布审计报告。"""

    args = parse_args(argv)
    if args.workers <= 0:
        raise ValueError("--workers 必须为正整数")
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
        robot_xml=args.robot_xml,
        retarget_config=args.retarget_config,
        kinematics=args.kinematics,
    )
    release_index, upstream_failures, release_audit = verify_release_report(
        args.release_report,
        config=config,
        input_root=input_root,
    )
    tasks = [
        (
            input_root / dataset / "mimic_npz" / "bumi3" / f"{stem}.npz",
            release_row,
        )
        for (dataset, stem), release_row in sorted(release_index.items())
    ]
    if args.limit is not None:
        tasks = tasks[: args.limit]
    config_sha = sha256_file(config_path)
    arguments = [
        (str(path), str(input_root), config, config_sha, dict(release_row))
        for path, release_row in tasks
    ]
    if args.workers == 1:
        rows = [_worker(argument) for argument in tqdm(arguments, desc="BUMI 30Hz quality")]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            rows = list(
                tqdm(
                    executor.map(_worker, arguments, chunksize=4),
                    total=len(arguments),
                    desc="BUMI 30Hz quality",
                )
            )
    summary = build_summary(
        rows,
        input_root=input_root,
        config_path=config_path,
        config_sha256=config_sha,
        config=config,
        assets={**assets, **release_audit},
        report_version=REPORT_VERSION,
        decision_scope=(
            "robot_retargeter 30Hz最终NPZ离线运动学/动力学与倒地预筛选；"
            "未执行控制器、仿真rollout或实机测试"
        ),
        compatibility_notes={
            "input_contract": SOURCE_CONTRACT_VERSION,
            "upstream_failed_sequences_excluded_before_scan": config.expected_failed,
            "retarget_report_warnings_status_affecting": False,
            "stored_velocity_consistency_is_diagnostic_only": True,
            "root_tilt_distribution_gate_enabled": True,
            "formal_builder_accepts_only_pass": True,
        },
    )
    summary["upstream_release"] = release_audit
    summary["upstream_excluded_failures"] = upstream_failures
    excluded_path = output_dir / "upstream_excluded.jsonl"
    if excluded_path.exists() and not args.overwrite:
        raise FileExistsError(f"上游排除报告已存在，需 --overwrite: {excluded_path}")
    write_reports(output_dir, rows, summary, config_path, overwrite=args.overwrite)
    _atomic_jsonl(excluded_path, upstream_failures)
    print(
        json.dumps(
            {
                "report_dir": str(output_dir),
                "sequences": len(rows),
                "status_counts": summary["status_counts"],
                "upstream_failed_excluded": len(upstream_failures),
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
