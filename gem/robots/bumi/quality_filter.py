"""BUMI3 离线动作质量指标、贴地风格识别与三态决策。

实现思路参考 OMG 的 MuJoCo 数据质量检查，但针对当前历史 pickle 能提供的信息做了
明确裁剪：契约/有限值/源关节限位属于硬门禁；速度、加速度和 jerk 使用“P95 +
超限比例 + 连续超限 + 严重峰值”的统计策略；躺地和滚地则结合根高度、根倾角、
虚拟躯干及非手部上身 link 高度判断。单纯手撑地或低 Root 不会被直接当作滚地，
从而减少对深蹲、跪姿和正常地板编舞的误杀。

模块只做确定性的 NumPy 计算，不依赖 MuJoCo，也不修改源动作。最终决策分为
``PASS``、``REVIEW``、``REJECT``；正式数据物化默认只接收 ``PASS``，而完整指标、
触发区间和安全区间会保留在 JSONL 中，便于人工复核和阈值重放。
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .legacy_motion import (
    LEGACY_BUMI_BODY_ORDER,
    LEGACY_BUMI_JOINT_ORDER,
    LEGACY_BUMI_MOTION_CONTRACT_VERSION,
    LEGACY_BUMI_QUATERNION_CONVENTION,
    LegacyBumiMotion,
    normalize_xyzw,
)

BUMI_QUALITY_CONFIG_VERSION = "genmo.bumi_quality_config.v1"
BUMI_QUALITY_REPORT_VERSION = "genmo.bumi_quality_report.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class QualityStatus(str, Enum):
    """质量决策优先级：REJECT > REVIEW > PASS。"""

    PASS = "PASS"
    REVIEW = "REVIEW"
    REJECT = "REJECT"


@dataclass(frozen=True)
class SignalThreshold:
    """一个逐帧软指标及其物理单位阈值。"""

    threshold: float
    unit: str


@dataclass(frozen=True)
class BumiQualityConfig:
    """已验证且可直接用于质量计算的 v1 配置。"""

    source_mjcf_sha256: str
    fps: int
    joint_lower_limits: np.ndarray
    joint_upper_limits: np.ndarray
    quaternion_norm_error_max: float
    joint_limit_violation_max: float
    minimum_joint_limit_margin_warn: float
    source_ground_abs_error_max: float
    root_height_min_absolute: float
    root_height_max_absolute: float
    exceed_ratio_max: float
    consecutive_exceed_frames: int
    severe_multiplier: float
    soft_failure_status: QualityStatus
    dynamics: Mapping[str, SignalThreshold]
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
    torso_body: str
    upper_non_hand_bodies: tuple[str, ...]
    hand_bodies: tuple[str, ...]
    ankle_bodies: tuple[str, ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> BumiQualityConfig:
        if not isinstance(raw, Mapping):
            raise ValueError("BUMI quality config must be a mapping")
        if raw.get("contract_version") != BUMI_QUALITY_CONFIG_VERSION:
            raise ValueError(
                f"quality contract_version must be {BUMI_QUALITY_CONFIG_VERSION!r}, "
                f"got {raw.get('contract_version')!r}"
            )
        source = _mapping(raw, "source")
        expected_source = {
            "motion_contract_version": LEGACY_BUMI_MOTION_CONTRACT_VERSION,
            "quaternion_convention": LEGACY_BUMI_QUATERNION_CONVENTION,
        }
        for key, expected in expected_source.items():
            if source.get(key) != expected:
                raise ValueError(f"source.{key} must be {expected!r}, got {source.get(key)!r}")
        source_sha = str(source.get("mjcf_sha256", ""))
        if _SHA256_RE.fullmatch(source_sha) is None:
            raise ValueError("source.mjcf_sha256 must be a lowercase SHA256 digest")
        fps = _positive_int(source, "fps")
        if fps != 30:
            raise ValueError(f"legacy BUMI quality source.fps must be 30, got {fps}")
        if tuple(map(str, source.get("joint_order", ()))) != LEGACY_BUMI_JOINT_ORDER:
            raise ValueError("source.joint_order must exactly match the legacy pickle qpos order")
        if tuple(map(str, source.get("body_order", ()))) != LEGACY_BUMI_BODY_ORDER:
            raise ValueError("source.body_order must exactly match legacy local_body_pos")
        joint_lower = _finite_vector(source, "joint_lower_limits", len(LEGACY_BUMI_JOINT_ORDER))
        joint_upper = _finite_vector(source, "joint_upper_limits", len(LEGACY_BUMI_JOINT_ORDER))
        if np.any(joint_lower >= joint_upper):
            raise ValueError("every source joint limit must satisfy lower < upper")

        hard = _mapping(raw, "hard_thresholds")
        policy = _mapping(raw, "soft_policy")
        dynamics_raw = _mapping(raw, "dynamics")
        required_signals = {
            "joint_velocity_l2",
            "joint_acceleration_l2",
            "joint_jerk_l2",
            "root_linear_velocity",
            "root_angular_velocity",
        }
        if set(dynamics_raw) != required_signals:
            raise ValueError(
                "dynamics keys must exactly be "
                f"{sorted(required_signals)}, got {sorted(dynamics_raw)}"
            )
        dynamics: dict[str, SignalThreshold] = {}
        for name in sorted(required_signals):
            item = _mapping(dynamics_raw, name)
            threshold = _positive_float(item, "threshold")
            unit = str(item.get("unit", "")).strip()
            if not unit:
                raise ValueError(f"dynamics.{name}.unit must be non-empty")
            dynamics[name] = SignalThreshold(threshold=threshold, unit=unit)

        floor = _mapping(raw, "floor_style")
        bodies = _mapping(floor, "body_groups")
        torso_body = str(bodies.get("torso", ""))
        upper_bodies = tuple(map(str, bodies.get("upper_non_hand", ())))
        hands = tuple(map(str, bodies.get("hands", ())))
        ankles = tuple(map(str, bodies.get("ankles", ())))
        for name, values, expected_length in (
            ("upper_non_hand", upper_bodies, None),
            ("hands", hands, 2),
            ("ankles", ankles, 2),
        ):
            if not values or (expected_length is not None and len(values) != expected_length):
                raise ValueError(f"floor_style.body_groups.{name} has invalid length")
            if len(values) != len(set(values)):
                raise ValueError(f"floor_style.body_groups.{name} contains duplicate bodies")
            missing = set(values) - set(LEGACY_BUMI_BODY_ORDER)
            if missing:
                raise ValueError(f"floor_style.body_groups.{name} has unknown bodies {missing}")
        if torso_body not in LEGACY_BUMI_BODY_ORDER:
            raise ValueError(f"floor_style torso body is unknown: {torso_body!r}")
        if set(upper_bodies) & set(hands):
            raise ValueError("upper_non_hand must not include hand bodies")

        status_text = str(policy.get("soft_failure_status", "REVIEW")).upper()
        try:
            soft_status = QualityStatus(status_text)
        except ValueError as exc:
            raise ValueError(
                "soft_policy.soft_failure_status must be PASS, REVIEW or REJECT"
            ) from exc
        if soft_status is QualityStatus.PASS:
            raise ValueError("soft_policy.soft_failure_status cannot be PASS")

        exceed_ratio = _unit_interval(policy, "exceed_ratio_max")
        severe_multiplier = _positive_float(policy, "severe_multiplier")
        if severe_multiplier <= 1.0:
            raise ValueError("soft_policy.severe_multiplier must be greater than 1")
        floor_review_ratio = _unit_interval(floor, "review_ratio")
        root_height_min = _finite_float(hard, "root_height_min_absolute")
        root_height_max = _finite_float(hard, "root_height_max_absolute")
        if root_height_min >= root_height_max:
            raise ValueError(
                "root_height_min_absolute must be smaller than root_height_max_absolute"
            )
        root_low_height = _nonnegative_float(floor, "root_low_height")
        floor_gate_root_height = _nonnegative_float(floor, "gate_root_height")
        low_root_review_height = _nonnegative_float(floor, "low_root_review_height")
        if root_low_height > floor_gate_root_height:
            raise ValueError("floor_style.root_low_height must not exceed gate_root_height")
        if low_root_review_height > root_low_height:
            raise ValueError("floor_style.low_root_review_height must not exceed root_low_height")
        root_low_tilt = _degrees(floor, "root_low_tilt_degrees")
        floor_gate_tilt = _degrees(floor, "gate_tilt_degrees")
        if root_low_tilt < floor_gate_tilt:
            raise ValueError("root_low_tilt_degrees must be at least gate_tilt_degrees")
        return cls(
            source_mjcf_sha256=source_sha,
            fps=fps,
            joint_lower_limits=joint_lower,
            joint_upper_limits=joint_upper,
            quaternion_norm_error_max=_nonnegative_float(hard, "quaternion_norm_error_max"),
            joint_limit_violation_max=_nonnegative_float(hard, "joint_limit_violation_max"),
            minimum_joint_limit_margin_warn=_nonnegative_float(
                hard, "minimum_joint_limit_margin_warn"
            ),
            source_ground_abs_error_max=_nonnegative_float(hard, "source_ground_abs_error_max"),
            root_height_min_absolute=root_height_min,
            root_height_max_absolute=root_height_max,
            exceed_ratio_max=exceed_ratio,
            consecutive_exceed_frames=_positive_int(policy, "consecutive_exceed_frames"),
            severe_multiplier=severe_multiplier,
            soft_failure_status=soft_status,
            dynamics=dynamics,
            root_low_height=root_low_height,
            root_low_tilt_degrees=root_low_tilt,
            torso_ground_height=_nonnegative_float(floor, "torso_ground_height"),
            upper_body_ground_height=_nonnegative_float(floor, "upper_body_ground_height"),
            floor_gate_root_height=floor_gate_root_height,
            floor_gate_tilt_degrees=floor_gate_tilt,
            ankles_airborne_height=_nonnegative_float(floor, "ankles_airborne_height"),
            floor_reject_consecutive_frames=_positive_int(floor, "reject_consecutive_frames"),
            floor_review_ratio=floor_review_ratio,
            floor_review_min_frames=_positive_int(floor, "review_min_frames"),
            low_root_review_height=low_root_review_height,
            low_root_review_consecutive_frames=_positive_int(
                floor, "low_root_review_consecutive_frames"
            ),
            safe_interval_halo_frames=_nonnegative_int(floor, "safe_interval_halo_frames"),
            minimum_safe_interval_frames=_positive_int(floor, "minimum_safe_interval_frames"),
            torso_body=torso_body,
            upper_non_hand_bodies=upper_bodies,
            hand_bodies=hands,
            ankle_bodies=ankles,
        )


@dataclass(frozen=True)
class QualityDecision:
    """一条动作的可序列化质量结论。"""

    status: QualityStatus
    reason_codes: tuple[str, ...]
    metrics: Mapping[str, Any]
    floor_intervals: tuple[tuple[int, int], ...]
    valid_intervals: tuple[tuple[int, int], ...]

    @property
    def accepted(self) -> bool:
        return self.status is QualityStatus.PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_contract_version": BUMI_QUALITY_REPORT_VERSION,
            "status": self.status.value,
            "quality_accepted": self.accepted,
            "reason_codes": list(self.reason_codes),
            "metrics": dict(self.metrics),
            "floor_intervals": [list(value) for value in self.floor_intervals],
            "valid_intervals": [list(value) for value in self.valid_intervals],
        }


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _finite_vector(parent: Mapping[str, Any], key: str, length: int) -> np.ndarray:
    value = np.asarray(parent.get(key), dtype=np.float64)
    if value.shape != (length,) or not np.isfinite(value).all():
        raise ValueError(f"{key} must be a finite vector of length {length}")
    return value


def _positive_int(parent: Mapping[str, Any], key: str) -> int:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _nonnegative_int(parent: Mapping[str, Any], key: str) -> int:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value


def _finite_float(parent: Mapping[str, Any], key: str) -> float:
    try:
        value = float(parent[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a finite number") from exc
    if not np.isfinite(value):
        raise ValueError(f"{key} must be a finite number")
    return value


def _positive_float(parent: Mapping[str, Any], key: str) -> float:
    value = _finite_float(parent, key)
    if value <= 0.0:
        raise ValueError(f"{key} must be finite and positive")
    return value


def _nonnegative_float(parent: Mapping[str, Any], key: str) -> float:
    value = _finite_float(parent, key)
    if value < 0.0:
        raise ValueError(f"{key} must be finite and non-negative")
    return value


def _unit_interval(parent: Mapping[str, Any], key: str) -> float:
    value = _finite_float(parent, key)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{key} must be in [0,1]")
    return value


def _degrees(parent: Mapping[str, Any], key: str) -> float:
    value = _nonnegative_float(parent, key)
    if value > 180.0:
        raise ValueError(f"{key} must be in [0,180]")
    return value


def load_bumi_quality_config(path: str | Path) -> BumiQualityConfig:
    """从 YAML 加载并严格验证质量配置。"""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"{config_path}: root must be a mapping")
    return BumiQualityConfig.from_mapping(raw)


def _longest_true_run(values: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in np.asarray(values, dtype=np.bool_).reshape(-1):
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return longest


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


def _signal_metrics(values: np.ndarray, threshold: float) -> dict[str, float | int]:
    sequence = np.asarray(values, dtype=np.float64).reshape(-1)
    if sequence.size == 0:
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


def _angular_speed_xyzw(quaternion: np.ndarray, fps: float) -> np.ndarray:
    normalized = normalize_xyzw(quaternion)
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


def _raise_status(current: QualityStatus, requested: QualityStatus) -> QualityStatus:
    return requested if _priority(requested) > _priority(current) else current


def evaluate_legacy_bumi_motion(
    motion: LegacyBumiMotion,
    config: BumiQualityConfig,
) -> QualityDecision:
    """计算一条动作的完整指标和三态结论，不修改输入数组。"""

    if motion.fps != config.fps:
        raise ValueError(f"motion fps={motion.fps} differs from quality config fps={config.fps}")
    status = QualityStatus.PASS
    reasons: list[str] = []

    def flag(code: str, requested: QualityStatus) -> None:
        nonlocal status
        if code not in reasons:
            reasons.append(code)
        status = _raise_status(status, requested)

    metrics: dict[str, Any] = {
        "finite": True,
        "num_frames": motion.num_frames,
        "fps": motion.fps,
    }
    quaternion_norm = np.linalg.norm(motion.root_rot_xyzw, axis=-1)
    normalized_quat = normalize_xyzw(motion.root_rot_xyzw)
    adjacent_dot = np.sum(normalized_quat[:-1] * normalized_quat[1:], axis=-1)
    metrics["quaternion_norm_max_error"] = float(np.max(np.abs(quaternion_norm - 1.0)))
    metrics["quaternion_adjacent_dot_min"] = (
        float(np.min(adjacent_dot)) if adjacent_dot.size else 1.0
    )
    metrics["quaternion_sign_flip_count"] = int(np.count_nonzero(adjacent_dot < 0.0))
    if metrics["quaternion_norm_max_error"] > config.quaternion_norm_error_max:
        flag("QUATERNION_NORM", QualityStatus.REJECT)

    lower = config.joint_lower_limits[None, :]
    upper = config.joint_upper_limits[None, :]
    limit_violation = np.maximum(np.maximum(lower - motion.dof_pos, motion.dof_pos - upper), 0.0)
    margin = np.minimum(motion.dof_pos - lower, upper - motion.dof_pos)
    per_joint_violation = np.max(limit_violation, axis=0)
    metrics["joint_limit_violation_max"] = float(np.max(per_joint_violation))
    metrics["joint_limit_violation_max_by_joint"] = {
        name: float(value)
        for name, value in zip(LEGACY_BUMI_JOINT_ORDER, per_joint_violation)
        if value > 0.0
    }
    metrics["minimum_joint_limit_margin"] = float(np.min(margin))
    metrics["minimum_joint_limit_margin_warned"] = bool(
        metrics["minimum_joint_limit_margin"] < config.minimum_joint_limit_margin_warn
    )
    if metrics["joint_limit_violation_max"] > config.joint_limit_violation_max:
        flag("SOURCE_JOINT_LIMIT", QualityStatus.REJECT)

    world_body = motion.world_body_positions()
    body_lookup = motion.body_name_to_index
    source_ground = float(np.min(world_body[..., 2]))
    metrics["source_body_origin_ground_min"] = source_ground
    metrics["source_body_origin_ground_abs_error"] = abs(source_ground)
    if abs(source_ground) > config.source_ground_abs_error_max:
        flag("SOURCE_GROUND_GAUGE", QualityStatus.REJECT)

    root_height = motion.root_pos[:, 2]
    metrics["root_height_min"] = float(np.min(root_height))
    metrics["root_height_p05"] = float(np.percentile(root_height, 5.0))
    metrics["root_height_max"] = float(np.max(root_height))
    if metrics["root_height_min"] < config.root_height_min_absolute:
        flag("ROOT_HEIGHT_BELOW_ABSOLUTE_BOUND", QualityStatus.REJECT)
    if metrics["root_height_max"] > config.root_height_max_absolute:
        flag("ROOT_HEIGHT_ABOVE_ABSOLUTE_BOUND", QualityStatus.REJECT)

    tilt = motion.root_tilt_degrees()
    torso_height = world_body[:, body_lookup[config.torso_body], 2]
    upper_height = np.min(
        world_body[:, [body_lookup[name] for name in config.upper_non_hand_bodies], 2],
        axis=1,
    )
    hand_height = np.min(
        world_body[:, [body_lookup[name] for name in config.hand_bodies], 2], axis=1
    )
    ankle_height = np.min(
        world_body[:, [body_lookup[name] for name in config.ankle_bodies], 2], axis=1
    )
    root_low_tilt = (root_height < config.root_low_height) & (tilt > config.root_low_tilt_degrees)
    torso_ground = torso_height < config.torso_ground_height
    upper_ground = upper_height < config.upper_body_ground_height
    floor_evidence = root_low_tilt | torso_ground | upper_ground
    floor_gate = (
        (root_height < config.floor_gate_root_height)
        | (tilt > config.floor_gate_tilt_degrees)
        | (ankle_height > config.ankles_airborne_height)
    )
    floor_mask = floor_evidence & floor_gate
    low_root_mask = root_height < config.low_root_review_height
    floor_intervals = mask_to_intervals(floor_mask)
    floor_count = int(np.count_nonzero(floor_mask))
    floor_ratio = float(np.mean(floor_mask))
    floor_run = _longest_true_run(floor_mask)
    low_root_run = _longest_true_run(low_root_mask)
    metrics["floor_style"] = {
        "frame_count": floor_count,
        "frame_ratio": floor_ratio,
        "max_consecutive_frames": floor_run,
        "root_low_tilt_frame_count": int(np.count_nonzero(root_low_tilt)),
        "torso_ground_frame_count": int(np.count_nonzero(torso_ground)),
        "upper_non_hand_ground_frame_count": int(np.count_nonzero(upper_ground)),
        "hand_below_upper_threshold_frame_count": int(
            np.count_nonzero(hand_height < config.upper_body_ground_height)
        ),
        "low_root_frame_ratio": float(np.mean(low_root_mask)),
        "low_root_max_consecutive_frames": low_root_run,
        "root_tilt_p95_degrees": float(np.percentile(tilt, 95.0)),
        "root_tilt_max_degrees": float(np.max(tilt)),
        "torso_height_min": float(np.min(torso_height)),
        "upper_non_hand_height_min": float(np.min(upper_height)),
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

    fps = float(config.fps)
    joint_velocity = np.diff(motion.dof_pos, axis=0) * fps
    joint_acceleration = np.diff(joint_velocity, axis=0) * fps
    joint_jerk = np.diff(joint_acceleration, axis=0) * fps
    root_velocity = np.diff(motion.root_pos, axis=0) * fps
    signals = {
        "joint_velocity_l2": np.linalg.norm(joint_velocity, axis=-1),
        "joint_acceleration_l2": np.linalg.norm(joint_acceleration, axis=-1),
        "joint_jerk_l2": np.linalg.norm(joint_jerk, axis=-1),
        "root_linear_velocity": np.linalg.norm(root_velocity, axis=-1),
        "root_angular_velocity": _angular_speed_xyzw(motion.root_rot_xyzw, fps),
    }
    dynamic_metrics: dict[str, Any] = {}
    for name, values in signals.items():
        signal_config = config.dynamics[name]
        summary = _signal_metrics(values, signal_config.threshold)
        summary["threshold"] = signal_config.threshold
        summary["unit"] = signal_config.unit
        dynamic_metrics[name] = summary
        severe = float(summary["max"]) > signal_config.threshold * config.severe_multiplier
        broad = (
            float(summary["p95"]) > signal_config.threshold
            and float(summary["exceed_ratio"]) > config.exceed_ratio_max
        )
        sustained = (
            int(summary["max_consecutive_exceed_frames"]) >= config.consecutive_exceed_frames
        )
        code = name.upper()
        if severe:
            flag(f"{code}_SEVERE", QualityStatus.REJECT)
        elif broad or sustained:
            flag(f"{code}_SOFT", config.soft_failure_status)
    metrics["dynamics"] = dynamic_metrics

    valid_intervals = safe_intervals_from_bad_mask(
        floor_mask,
        halo_frames=config.safe_interval_halo_frames,
        minimum_frames=config.minimum_safe_interval_frames,
    )
    return QualityDecision(
        status=status,
        reason_codes=tuple(reasons),
        metrics=metrics,
        floor_intervals=floor_intervals,
        valid_intervals=valid_intervals,
    )


def quality_config_as_json(config: BumiQualityConfig) -> str:
    """调试辅助：以稳定 JSON 展示已经解析的关键配置。"""

    value = {
        "source_mjcf_sha256": config.source_mjcf_sha256,
        "fps": config.fps,
        "soft_failure_status": config.soft_failure_status.value,
        "dynamics": {
            name: {"threshold": item.threshold, "unit": item.unit}
            for name, item in sorted(config.dynamics.items())
        },
    }
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
