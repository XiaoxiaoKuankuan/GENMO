"""Kinematic quality metrics for generated BUMI trajectories.

These metrics do not run ``mj_step`` and do not imply dynamic trackability,
balance, torque feasibility, or GMT controller success.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from gem.utils.rotation_conversions import quaternion_to_matrix

from .feature_codec import normalize_quaternion_wxyz
from .kinematics import BumiKinematics
from .losses import so3_geodesic_angle, temporal_difference_mask


def _as_batched_qpos(qpos: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if not isinstance(qpos, torch.Tensor) or qpos.shape[-1] != 28:
        raise ValueError(f"qpos must be a tensor ending in 28, got {getattr(qpos, 'shape', None)}")
    if qpos.ndim == 2:
        return qpos.unsqueeze(0), True
    if qpos.ndim != 3:
        raise ValueError(f"qpos must have shape [T,28] or [B,T,28], got {qpos.shape}")
    return qpos, False


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.bool()
    while expanded.ndim < value.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(value)
    weight = expanded.to(value)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def _masked_p95(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.bool()
    while expanded.ndim < value.ndim:
        expanded = expanded.unsqueeze(-1)
    selected = value[expanded.expand_as(value)]
    if selected.numel() == 0:
        return value.new_zeros(())
    return torch.quantile(selected.float(), 0.95).to(value)


def _derivative(value: torch.Tensor, order: int, fps: int) -> torch.Tensor:
    return torch.diff(value, n=order, dim=1) * (float(fps) ** order)


def _derive_motion_beats(joint_speed: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Use local minima of mean joint speed as a deterministic motion-beat proxy."""

    energy = joint_speed.mean(dim=-1)
    beats = torch.zeros_like(valid)
    if energy.shape[1] >= 3:
        minima = (energy[:, 1:-1] <= energy[:, :-2]) & (energy[:, 1:-1] < energy[:, 2:])
        beats[:, 1:-1] = minima & valid[:, 1:-1]
    return beats


def _beat_alignment(
    music_beats: torch.Tensor,
    motion_beats: torch.Tensor,
    valid: torch.Tensor,
    fps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    distances = []
    for batch_index in range(valid.shape[0]):
        music_index = torch.nonzero(
            music_beats[batch_index].bool() & valid[batch_index], as_tuple=False
        ).flatten()
        motion_index = torch.nonzero(
            motion_beats[batch_index].bool() & valid[batch_index], as_tuple=False
        ).flatten()
        if music_index.numel() == 0 or motion_index.numel() == 0:
            continue
        nearest = (music_index[:, None] - motion_index[None, :]).abs().amin(dim=1)
        distances.append(nearest.float() / float(fps))
    if not distances:
        zero = music_beats.new_zeros((), dtype=torch.float32)
        return zero, zero
    distance = torch.cat(distances).mean()
    return distance, torch.exp(-distance / 0.1)


@torch.no_grad()
def compute_bumi_kinematic_metrics(
    pred_qpos: torch.Tensor,
    kinematics: BumiKinematics,
    *,
    target_qpos: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    target_contact: torch.Tensor | None = None,
    pred_contact_logits: torch.Tensor | None = None,
    music_beats: torch.Tensor | None = None,
    fps: int = 30,
    ground_height: float | torch.Tensor = 0.0,
    contact_height_threshold: float = 0.025,
    contact_velocity_threshold: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Compute future-facing BUMI kinematic metrics on one or more sequences."""

    if int(fps) != 30:
        raise ValueError(f"BUMI music metrics require 30 FPS, got {fps}")
    pred, _ = _as_batched_qpos(pred_qpos)
    pred = pred.clone()
    pred[..., 3:7] = normalize_quaternion_wxyz(pred[..., 3:7])
    batch_size, length = pred.shape[:2]
    if valid_mask is None:
        valid = torch.ones((batch_size, length), dtype=torch.bool, device=pred.device)
    else:
        valid = valid_mask.to(device=pred.device).bool()
        if valid.ndim == 1:
            valid = valid.unsqueeze(0)
        if tuple(valid.shape) != (batch_size, length):
            raise ValueError(
                f"valid_mask must have shape {(batch_size, length)}, got {valid.shape}"
            )
    fk = kinematics.forward_kinematics(pred)
    sole = kinematics.aggregate_sole_by_foot(fk["body_pos_w"], fk["body_quat_w"])
    ground = torch.as_tensor(ground_height, dtype=pred.dtype, device=pred.device)
    if not bool(torch.isfinite(ground).all()):
        raise ValueError("ground_height contains NaN or Inf")
    try:
        ground = ground.expand(batch_size)
    except RuntimeError as exc:
        raise ValueError(
            f"ground_height shape {tuple(ground.shape)} cannot broadcast to batch {batch_size}"
        ) from exc
    ground_per_foot = ground[:, None, None]
    joints = pred[..., 7:]
    lower = kinematics.joint_lower_limits.to(joints)
    upper = kinematics.joint_upper_limits.to(joints)
    violations = (joints < lower) | (joints > upper)
    margin = torch.minimum(joints - lower, upper - joints)
    root_rotation = quaternion_to_matrix(pred[..., 3:7])
    root_up = root_rotation[..., :, 2]
    world_up = torch.zeros_like(root_up)
    world_up[..., 2] = 1.0
    root_tilt = torch.acos((root_up * world_up).sum(dim=-1).clamp(-1.0, 1.0))

    joint_velocity = _derivative(joints, 1, fps)
    joint_acceleration = _derivative(joints, 2, fps)
    joint_jerk = _derivative(joints, 3, fps)
    root_velocity = _derivative(pred[..., :3], 1, fps)
    relative_root = root_rotation[:, 1:] @ root_rotation[:, :-1].transpose(-1, -2)
    identity = torch.eye(3, dtype=pred.dtype, device=pred.device).expand_as(relative_root)
    root_angular_velocity = so3_geodesic_angle(relative_root, identity) * float(fps)
    foot_velocity = _derivative(sole["foot_points_w"][..., :2], 1, fps)
    foot_speed = torch.linalg.vector_norm(foot_velocity, dim=-1)
    if length <= 1:
        padded_foot_speed = pred.new_zeros((batch_size, length, 2))
    else:
        padded_foot_speed = torch.cat((foot_speed[:, :1], foot_speed), dim=1)
    inferred_contact = (
        (sole["foot_bottom_height"] <= ground_per_foot + float(contact_height_threshold))
        & padded_foot_speed.le(contact_velocity_threshold)
        & valid[..., None]
    )
    predicted_contact = None
    if pred_contact_logits is not None:
        predicted_contact = pred_contact_logits.to(pred).sigmoid() >= 0.5
        if predicted_contact.ndim == 2:
            predicted_contact = predicted_contact.unsqueeze(0)
        if predicted_contact.shape != inferred_contact.shape:
            raise ValueError(
                "pred_contact_logits must have shape "
                f"{inferred_contact.shape}, got {predicted_contact.shape}"
            )
    if target_contact is None:
        # 有接触 head 时必须按模型声明的支撑区测脚滑，不能用“速度已经很低”反推接触，
        # 否则快速滑动的脚会因超过速度阈值而被排除，产生虚假的零 foot-slide 指标。
        contact_gate = inferred_contact if predicted_contact is None else predicted_contact
    else:
        contact_gate = target_contact.to(pred).bool()
        if contact_gate.ndim == 2:
            contact_gate = contact_gate.unsqueeze(0)
        if contact_gate.shape != inferred_contact.shape:
            raise ValueError(
                f"target_contact must have shape {inferred_contact.shape}, got {contact_gate.shape}"
            )
    slide_mask = (
        contact_gate[:, 1:] & contact_gate[:, :-1] & temporal_difference_mask(valid, 1)[..., None]
    )
    penetration = F.relu(ground_per_foot - sole["foot_bottom_height"])

    metrics = {
        "joint_limit_violation_rate": _masked_mean(violations.float(), valid),
        "minimum_joint_margin_rad": margin.masked_fill(~valid[..., None], torch.inf).amin(),
        "foot_penetration_mean_m": _masked_mean(penetration, valid),
        "foot_penetration_max_m": penetration.masked_fill(~valid[..., None], 0.0).amax(),
        "foot_sliding_mean_mps": _masked_mean(foot_speed, slide_mask),
        "root_height_mean_m": _masked_mean(pred[..., 2], valid),
        "root_height_min_m": pred[..., 2].masked_fill(~valid, torch.inf).amin(),
        "root_tilt_mean_rad": _masked_mean(root_tilt, valid),
        "root_tilt_max_rad": root_tilt.masked_fill(~valid, 0.0).amax(),
        "joint_velocity_p95_radps": _masked_p95(
            joint_velocity.abs(), temporal_difference_mask(valid, 1)
        ),
        "joint_acceleration_p95_radps2": _masked_p95(
            joint_acceleration.abs(), temporal_difference_mask(valid, 2)
        ),
        "joint_jerk_p95_radps3": _masked_p95(joint_jerk.abs(), temporal_difference_mask(valid, 3)),
        "root_linear_velocity_p95_mps": _masked_p95(
            torch.linalg.vector_norm(root_velocity, dim=-1), temporal_difference_mask(valid, 1)
        ),
        "root_angular_velocity_p95_radps": _masked_p95(
            root_angular_velocity, temporal_difference_mask(valid, 1)
        ),
    }
    if target_qpos is not None:
        target, _ = _as_batched_qpos(target_qpos.to(pred))
        if target.shape != pred.shape:
            raise ValueError(f"target_qpos shape {target.shape} does not match pred {pred.shape}")
        target[..., 3:7] = normalize_quaternion_wxyz(target[..., 3:7])
        target_fk = kinematics.forward_kinematics(target)
        metrics.update(
            {
                "joint_angle_mae_rad": _masked_mean((joints - target[..., 7:]).abs(), valid),
                "root_trajectory_error_m": _masked_mean(
                    torch.linalg.vector_norm(pred[..., :3] - target[..., :3], dim=-1), valid
                ),
                "fk_body_position_error_m": _masked_mean(
                    torch.linalg.vector_norm(fk["body_pos_w"] - target_fk["body_pos_w"], dim=-1),
                    valid,
                ),
            }
        )
    if target_contact is not None and predicted_contact is not None:
        target_contact_bool = target_contact.to(pred).bool()
        if target_contact_bool.ndim == 2:
            target_contact_bool = target_contact_bool.unsqueeze(0)
        if predicted_contact.shape != target_contact_bool.shape:
            raise ValueError(
                "contact prediction/target shape mismatch: "
                f"{predicted_contact.shape}/{target_contact_bool.shape}"
            )
        metrics["contact_accuracy"] = _masked_mean(
            (predicted_contact == target_contact_bool).float(), valid
        )
    if music_beats is not None:
        beats = music_beats.to(device=pred.device)
        if beats.ndim == 1:
            beats = beats.unsqueeze(0)
        if beats.shape != valid.shape:
            raise ValueError(f"music_beats must have shape {valid.shape}, got {beats.shape}")
        joint_speed = joint_velocity.abs()
        if length <= 1:
            padded_joint_speed = pred.new_zeros((batch_size, length, 21))
        else:
            padded_joint_speed = torch.cat((joint_speed[:, :1], joint_speed), dim=1)
        motion_beats = _derive_motion_beats(padded_joint_speed, valid)
        distance, score = _beat_alignment(beats, motion_beats, valid, fps)
        metrics["beat_alignment_mean_distance_s"] = distance
        metrics["beat_alignment_score"] = score
    if batch_size > 1:
        flattened = joints.reshape(batch_size, -1)
        pairwise = torch.pdist(flattened, p=2) / max(length * 21, 1) ** 0.5
        metrics["diversity_joint_rms_rad"] = pairwise.mean()
    else:
        metrics["diversity_joint_rms_rad"] = pred.new_zeros(())
    return metrics


def metrics_to_json(metrics: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"Metric {key} is not scalar: {value.shape}")
            result[key] = float(value.detach().cpu())
        else:
            result[key] = float(value)
    return result


__all__ = ["compute_bumi_kinematic_metrics", "metrics_to_json"]
