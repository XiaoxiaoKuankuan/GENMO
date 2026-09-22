"""BUMI 动作质量统计的公共数值核心。

接收调用方已核对关节、刚体顺序和帧率的数组，计算连续区间、限位、姿态和差分指标。
阈值由 UMR 规则构造，不加载 SONIC 配置、不访问旧机器人资产。保留历史报告版本字符串
仅用于数值结果和已交付报告的兼容，不表示输入必须为旧 50 Hz 数据。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

REPORT_VERSION = "genmo.bumi_quality_report.sonic_npz_50hz.v1"


class QualityStatus(str, Enum):
    """质量决策优先级：REJECT > REVIEW > PASS。"""

    PASS = "PASS"
    REVIEW = "REVIEW"
    REJECT = "REJECT"


def mask_to_intervals(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    """把布尔帧 mask 转为左闭右开 ``[start,end)`` 区间。"""

    values = np.asarray(mask, dtype=np.bool_).reshape(-1)
    if not values.size:
        return ()
    changes = np.diff(np.pad(values.astype(np.int8), (1, 1)))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return tuple((int(start), int(end)) for start, end in zip(starts, ends))


def safe_intervals_from_bad_mask(
    mask: np.ndarray,
    *,
    halo_frames: int,
    minimum_frames: int,
) -> tuple[tuple[int, int], ...]:
    """扩张坏帧区间后，返回足够长的安全片段。"""

    values = np.asarray(mask, dtype=np.bool_).reshape(-1)
    expanded = np.zeros_like(values)
    for start, end in mask_to_intervals(values):
        expanded[max(0, start - halo_frames) : min(len(values), end + halo_frames)] = True
    return tuple(
        interval
        for interval in mask_to_intervals(~expanded)
        if interval[1] - interval[0] >= minimum_frames
    )


@dataclass(frozen=True)
class MotionQualityConfig:
    """由调用方提供帧率、资产身份和阈值的运动质量规则。"""

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
    arrays: Mapping[str, np.ndarray], config: MotionQualityConfig
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
