"""BUMI NPZ 质量评估、汇总和报告发布的 producer 中性实现。

本模块集中保存当前 robot_retargeter 与 UMR qpos 共同使用的纯数组质量算法：
中心差分、四元数角速度、关节限位与倒地/Root 倾角判定、三态决策、跨数据集统计
汇总，以及 JSONL/CSV/文本清单的原子报告写出。模块只要求配置对象提供约定字段，
不导入任何具体 producer 的配置 dataclass、NPZ reader、资产路径或命令行入口。

通用评估只发布中性内部版本；robot_retargeter 和 UMR 边界在各自入口显式覆盖最终
契约版本。报告写出规则、字段顺序、容差扫描及 PASS/REVIEW/REJECT 排序保持迁移前
行为，不扩大质量结论到仿真 rollout、控制器可跟踪性或实机安全。
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from gem.robots.bumi.quality_common import (
    QualityStatus,
    mask_to_intervals,
    safe_intervals_from_bad_mask,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_VERSION = "genmo.bumi_quality_report.npz_common.v1"
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


def central_difference(values: np.ndarray, fps: float) -> np.ndarray:
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


def evaluate_motion(arrays: Mapping[str, np.ndarray], config: Any) -> dict[str, Any]:
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
    root_tilt_median = float(np.median(root_tilt))
    root_tilt_p95 = float(np.percentile(root_tilt, 95.0))
    root_tilt_over_45_fraction = float(np.mean(root_tilt > 45.0))
    metrics["root_orientation"] = {
        "median_degrees": root_tilt_median,
        "p95_degrees": root_tilt_p95,
        "maximum_degrees": float(np.max(root_tilt)),
        "over_45deg_fraction": root_tilt_over_45_fraction,
    }
    # Root 分布门禁是可选配置能力。robot_retargeter 30 Hz 契约显式提供两级阈值：
    # 普通异常进入 REVIEW，明显躺倒进入 REJECT；正式数据构建只消费 PASS。
    root_review = (
        getattr(config, "root_tilt_review_median_degrees", None),
        getattr(config, "root_tilt_review_p95_degrees", None),
        getattr(config, "root_tilt_review_over_45_fraction", None),
    )
    root_reject = (
        getattr(config, "root_tilt_reject_median_degrees", None),
        getattr(config, "root_tilt_reject_p95_degrees", None),
        getattr(config, "root_tilt_reject_over_45_fraction", None),
    )
    root_values = (root_tilt_median, root_tilt_p95, root_tilt_over_45_fraction)
    if all(value is not None for value in root_reject) and any(
        actual > float(limit) for actual, limit in zip(root_values, root_reject, strict=True)
    ):
        flag("ROOT_TILT_DISTRIBUTION_REJECT", QualityStatus.REJECT)
    elif all(value is not None for value in root_review) and any(
        actual > float(limit) for actual, limit in zip(root_values, root_review, strict=True)
    ):
        flag("ROOT_TILT_DISTRIBUTION_REVIEW", QualityStatus.REVIEW)
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

    joint_vel_expected = central_difference(arrays["joint_pos"], fps)
    body_lin_expected = central_difference(arrays["body_pos_w"], fps)
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
    config: Any,
    assets: Mapping[str, str],
    report_version: str = REPORT_VERSION,
    decision_scope: str | None = None,
    compatibility_notes: Mapping[str, Any] | None = None,
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
        "root_tilt_median_degrees": (
            "metrics",
            "root_orientation",
            "median_degrees",
        ),
        "root_tilt_p95_degrees": (
            "metrics",
            "root_orientation",
            "p95_degrees",
        ),
        "root_tilt_over_45deg_fraction": (
            "metrics",
            "root_orientation",
            "over_45deg_fraction",
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
        "report_contract_version": report_version,
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
        "decision_scope": decision_scope
        or "离线 NPZ 运动学/动力学预检查；未执行控制器、仿真 rollout 或实机测试",
        "compatibility_notes": dict(compatibility_notes)
        if compatibility_notes is not None
        else {
            "producer_specific_compatibility_notes_required": True,
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
