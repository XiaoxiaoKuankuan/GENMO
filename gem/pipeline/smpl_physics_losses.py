# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Low-weight physical temporal losses for the 151D SMPL representation."""

from __future__ import annotations

import re
from typing import Any

import torch
import torch.nn.functional as F

from gem.utils.ground_sidecar import SOLE_V437_INDICES
from gem.utils.motion_utils import get_local_transl_vel, rollout_local_transl_vel
from gem.utils.rotation_conversions import axis_angle_to_matrix, matrix_to_axis_angle


def consecutive_valid_mask(frame_valid: torch.Tensor, order: int) -> torch.Tensor:
    """Require ``order + 1`` consecutive real frames for a derivative."""
    if order < 1:
        raise ValueError("derivative order must be at least one")
    frame_valid = torch.as_tensor(frame_valid, dtype=torch.bool)
    if frame_valid.ndim != 2:
        raise ValueError("frame_valid must have shape [B,T]")
    if frame_valid.shape[1] <= order:
        return frame_valid[:, :0]
    return frame_valid.unfold(1, order + 1, 1).all(dim=-1)


def derivative_valid_mask(
    frame_valid: torch.Tensor,
    order: int,
    accepted_velocity: torch.Tensor | None = None,
) -> torch.Tensor:
    """Derivative mask with optional first-order data-quality intervals."""
    valid = consecutive_valid_mask(frame_valid, order)
    if accepted_velocity is None:
        return valid
    accepted_velocity = torch.as_tensor(accepted_velocity, dtype=torch.bool)
    if accepted_velocity.shape[:2] != (
        frame_valid.shape[0],
        frame_valid.shape[1] - 1,
    ):
        raise ValueError("accepted_velocity must start with shape [B,T-1]")
    accepted = accepted_velocity.unfold(1, order, 1).all(dim=-1)
    while valid.ndim < accepted.ndim:
        valid = valid.unsqueeze(-1)
    return valid & accepted


def finite_difference(value: torch.Tensor, order: int, fps: float = 30.0) -> torch.Tensor:
    """Forward finite difference in physical seconds, for orders one to three."""
    if order not in (1, 2, 3):
        raise ValueError("only first-, second-, and third-order differences are supported")
    if fps <= 0:
        raise ValueError("fps must be positive")
    result = value
    for _ in range(order):
        result = (result[:, 1:] - result[:, :-1]) * float(fps)
    return result


def so3_angular_velocity(rotation: torch.Tensor, fps: float = 30.0) -> torch.Tensor:
    """SO(3) log of ``R_t^T R_{t+1}``, expressed in radians/second."""
    if rotation.ndim < 4 or rotation.shape[-2:] != (3, 3):
        raise ValueError("rotation must end in [3,3] and contain a time dimension")
    relative = rotation[:, :-1].mT @ rotation[:, 1:]
    return matrix_to_axis_angle(relative) * float(fps)


def rollout_canonical_root(
    local_transl_vel: torch.Tensor, gravity_root_orient: torch.Tensor
) -> torch.Tensor:
    """Use GEM inference semantics and a fixed zero origin for root rollout."""
    origin = local_transl_vel[:, :1].new_zeros(local_transl_vel.shape[0], 1, 3)
    return rollout_local_transl_vel(
        local_transl_vel, gravity_root_orient, transl_0=origin
    )


def sole_penetration_loss(
    pred_sole_y: torch.Tensor,
    ground_y_local: torch.Tensor,
    frame_valid: torch.Tensor,
    ground_valid: torch.Tensor,
    *,
    tolerance_m: float = 0.01,
    scale_m: float = 0.03,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized SmoothL1 penetration and mean penetration depth."""
    if pred_sole_y.ndim != 3:
        raise ValueError("pred_sole_y must have shape [B,T,P]")
    ground_y_local = ground_y_local.to(pred_sole_y).reshape(-1, 1, 1)
    valid = frame_valid.bool() & ground_valid.bool().reshape(-1, 1)
    valid = valid.unsqueeze(-1).expand_as(pred_sole_y)
    depth = F.relu(ground_y_local - pred_sole_y - float(tolerance_m))
    normalized = F.smooth_l1_loss(
        depth / float(scale_m), torch.zeros_like(depth), beta=1.0, reduction="none"
    )
    return _masked_mean(normalized, valid), _masked_mean(depth, valid)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(value)
    denominator = mask.sum().to(value.dtype)
    # torch.where also prevents a masked NaN from leaking through ``NaN * 0``.
    numerator = torch.where(mask, value, torch.zeros_like(value)).sum()
    return numerator / denominator.clamp_min(1.0)


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _term_config(config: Any, name: str) -> tuple[float, float]:
    term = _cfg_get(config, name)
    if term is None:
        raise ValueError(f"physics loss config is missing {name!r}")
    return float(_cfg_get(term, "scale")), float(_cfg_get(term, "weight"))


def _normalized_match(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual = pred - target
    normalized = F.smooth_l1_loss(
        residual / float(scale),
        torch.zeros_like(residual),
        beta=1.0,
        reduction="none",
    ).mean(dim=-1)
    raw = torch.linalg.vector_norm(residual, dim=-1)
    return _masked_mean(normalized, mask), _masked_mean(raw, mask)


def _quality_rate(
    frame_valid: torch.Tensor, accepted: torch.Tensor, sample_mask: torch.Tensor
) -> torch.Tensor:
    base = consecutive_valid_mask(frame_valid, 1)
    while base.ndim < accepted.ndim:
        base = base.unsqueeze(-1)
    sample_mask = sample_mask.bool()
    while sample_mask.ndim < accepted.ndim:
        sample_mask = sample_mask.unsqueeze(-1)
    base = base.expand_as(accepted) & sample_mask
    denominator = base.sum().float()
    rejected = base & ~accepted
    return rejected.sum().float() / denominator.clamp_min(1.0)


def _quality_counts(
    frame_valid: torch.Tensor, accepted: torch.Tensor, sample_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    base = consecutive_valid_mask(frame_valid, 1)
    while base.ndim < accepted.ndim:
        base = base.unsqueeze(-1)
    while sample_mask.ndim < accepted.ndim:
        sample_mask = sample_mask.unsqueeze(-1)
    base = base.expand_as(accepted) & sample_mask.bool()
    return (base & ~accepted).sum().float(), base.sum().float()


def _metric_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def compute_smpl_physics_losses(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    pipeline: Any,
    *,
    global_step: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute physics-v1 temporal and sole losses in FP32.

    The model architecture is unchanged.  This function consumes decoded 151D
    fields and the 437-vertex camera-space forward pass already used by the
    baseline geometric losses.
    """
    config = _cfg_get(pipeline.args, "physics_losses")
    decode = outputs["decode_dict"]
    device = decode["body_pose"].device
    with torch.autocast(device_type=device.type, enabled=False):
        fps = float(_cfg_get(config, "fps", 30.0))
        warmup_steps = max(int(_cfg_get(config, "warmup_steps", 10000)), 1)
        ramp = min(max(float(global_step) / warmup_steps, 0.0), 1.0)
        frame_valid = inputs["mask"]["valid"].bool().clone()
        frame_valid[inputs["mask"]["spv_incam_only"].bool()] = False
        frame_valid[inputs["mask"]["2d_only"].bool()] = False

        pred_body_pose = decode["body_pose"].float()
        pred_betas = decode["betas"].float()
        pred_root_gv = decode["global_orient_gv"].float()
        pred_local_velocity = decode["local_transl_vel"].float()

        gt_body_pose = inputs["smpl_params_w"]["body_pose"].float()
        gt_betas = inputs["smpl_params_w"]["betas"].float()
        gt_root_world = inputs["smpl_params_w"]["global_orient"].float()
        gt_translation_world = inputs["smpl_params_w"]["transl"].float()
        gt_root_camera = inputs["smpl_params_c"]["global_orient"].float()
        gt_root_gv_R = inputs["R_c2gv"].float() @ axis_angle_to_matrix(gt_root_camera)
        gt_root_gv = matrix_to_axis_angle(gt_root_gv_R)
        gt_local_velocity = get_local_transl_vel(
            gt_translation_world, gt_root_world
        ).float()

        pred_root_position = rollout_canonical_root(
            pred_local_velocity, pred_root_gv
        )
        gt_root_position = rollout_canonical_root(gt_local_velocity, gt_root_gv)

        pred_body_R = axis_angle_to_matrix(
            pred_body_pose.reshape(*pred_body_pose.shape[:2], 21, 3)
        )
        gt_body_R = axis_angle_to_matrix(
            gt_body_pose.reshape(*gt_body_pose.shape[:2], 21, 3)
        )
        pred_joint_angular_velocity = so3_angular_velocity(pred_body_R, fps)
        gt_joint_angular_velocity = so3_angular_velocity(gt_body_R, fps)

        pred_fk = pipeline.endecoder.fk_v2(
            body_pose=pred_body_pose,
            betas=pred_betas,
            global_orient=pred_root_gv,
            transl=pred_root_position,
        ).float()
        gt_fk = pipeline.endecoder.fk_v2(
            body_pose=gt_body_pose,
            betas=gt_betas,
            global_orient=gt_root_gv,
            transl=gt_root_position,
        ).float()

        gt_root_velocity = finite_difference(gt_root_position, 1, fps)
        gt_fk_velocity = finite_difference(gt_fk, 1, fps)
        root_accepted = (
            torch.linalg.vector_norm(gt_root_velocity, dim=-1)
            <= float(_cfg_get(config, "max_gt_root_speed", 6.0))
        )
        joint_accepted = (
            torch.linalg.vector_norm(gt_joint_angular_velocity, dim=-1)
            <= float(_cfg_get(config, "max_gt_joint_angular_speed", 30.0))
        )
        fk_accepted = (
            torch.linalg.vector_norm(gt_fk_velocity, dim=-1)
            <= float(_cfg_get(config, "max_gt_fk_speed", 9.0))
        )

        total = pred_body_pose.sum() * 0.0
        result: dict[str, torch.Tensor] = {
            "physics_weight_ramp_metric": pred_body_pose.new_tensor(ramp)
        }

        def add_match(
            name: str,
            pred: torch.Tensor,
            target: torch.Tensor,
            mask: torch.Tensor,
        ) -> None:
            nonlocal total
            scale, target_weight = _term_config(config, name)
            normalized, raw = _normalized_match(pred, target, mask, scale)
            weighted = normalized * (target_weight * ramp)
            total = total + weighted
            result[f"physics_{name}_raw_loss"] = raw.detach()
            result[f"physics_{name}_normalized_loss"] = normalized.detach()
            result[f"physics_{name}_weighted_loss"] = weighted.detach()

        for order, suffix in ((1, "velocity"), (2, "acceleration"), (3, "jerk")):
            add_match(
                f"root_{suffix}",
                finite_difference(pred_root_position, order, fps),
                finite_difference(gt_root_position, order, fps),
                derivative_valid_mask(frame_valid, order, root_accepted),
            )

        add_match(
            "joint_angular_velocity",
            pred_joint_angular_velocity,
            gt_joint_angular_velocity,
            derivative_valid_mask(frame_valid, 1, joint_accepted),
        )
        add_match(
            "joint_angular_acceleration",
            finite_difference(pred_joint_angular_velocity, 1, fps),
            finite_difference(gt_joint_angular_velocity, 1, fps),
            derivative_valid_mask(frame_valid, 2, joint_accepted),
        )

        for order, suffix in ((1, "velocity"), (2, "acceleration"), (3, "jerk")):
            add_match(
                f"fk_{suffix}",
                finite_difference(pred_fk, order, fps),
                finite_difference(gt_fk, order, fps),
                derivative_valid_mask(frame_valid, order, fk_accepted),
            )

        penetration_scale, penetration_weight = _term_config(
            config, "sole_penetration"
        )
        if "physics" not in inputs:
            raise ValueError(
                "physics-v1 requires batch['physics']; build and configure ground sidecars"
            )
        if "_pred_c_verts437" not in outputs:
            raise ValueError(
                "physics-v1 requires the cached 437-vertex camera-space prediction"
            )
        pred_c_verts = outputs["_pred_c_verts437"].float()
        pred_c_transl = outputs["pred_body_params_incam"]["transl"].float()
        pred_root_c_R = axis_angle_to_matrix(decode["global_orient"].float())
        pred_root_gv_R = axis_angle_to_matrix(pred_root_gv)
        pred_c2gv_R = pred_root_gv_R @ pred_root_c_R.mT
        sole_indices = torch.as_tensor(SOLE_V437_INDICES, device=device)
        pred_c_sole = pred_c_verts.index_select(-2, sole_indices)
        pred_relative_sole = pred_c_sole - pred_c_transl.unsqueeze(-2)
        pred_gv_sole = torch.einsum(
            "...ij,...vj->...vi", pred_c2gv_R, pred_relative_sole
        ) + pred_root_position.unsqueeze(-2)
        pred_sole_y = pred_gv_sole[..., 1]
        penetration_normalized, penetration_raw = sole_penetration_loss(
            pred_sole_y,
            inputs["physics"]["ground_y_local"].float(),
            frame_valid,
            inputs["physics"]["ground_valid"].bool(),
            tolerance_m=float(_cfg_get(config, "sole_tolerance_m", 0.01)),
            scale_m=penetration_scale,
        )
        penetration_weighted = penetration_normalized * (penetration_weight * ramp)
        total = total + penetration_weighted
        result["physics_sole_penetration_raw_loss"] = penetration_raw.detach()
        result["physics_sole_penetration_normalized_loss"] = (
            penetration_normalized.detach()
        )
        result["physics_sole_penetration_weighted_loss"] = (
            penetration_weighted.detach()
        )

        all_samples = torch.ones(frame_valid.shape[0], dtype=torch.bool, device=device)
        quality = {
            "root_speed": root_accepted,
            "joint_angular_speed": joint_accepted,
            "fk_speed": fk_accepted,
        }
        accepted_frame_map = {
            "root_speed": root_accepted,
            "joint_angular_speed": joint_accepted,
            "fk_speed": fk_accepted,
        }
        for quality_name, accepted in accepted_frame_map.items():
            result[f"physics_{quality_name}_masked_rate_metric"] = _quality_rate(
                frame_valid, accepted, all_samples
            ).detach()
            masked_count, candidate_count = _quality_counts(
                frame_valid, accepted, all_samples
            )
            result[f"physics_{quality_name}_masked_count_metric"] = masked_count
            result[f"physics_{quality_name}_candidate_count_metric"] = candidate_count

        dataset_ids = [str(value) for value in _cfg_get(config, "dataset_ids", [])]
        meta = inputs.get("meta", [])
        batch_dataset_ids = [
            str(item.get("dataset_id", item.get("data_name", ""))) for item in meta
        ]
        for dataset_id in dataset_ids:
            sample_mask = torch.tensor(
                [value == dataset_id for value in batch_dataset_ids],
                dtype=torch.bool,
                device=device,
            )
            slug = _metric_slug(dataset_id)
            result[f"physics_{slug}_present_metric"] = sample_mask.any().float()
            for quality_name, accepted in quality.items():
                result[
                    f"physics_{slug}_{quality_name}_masked_rate_metric"
                ] = _quality_rate(frame_valid, accepted, sample_mask).detach()
                masked_count, candidate_count = _quality_counts(
                    frame_valid, accepted, sample_mask
                )
                result[
                    f"physics_{slug}_{quality_name}_masked_count_metric"
                ] = masked_count
                result[
                    f"physics_{slug}_{quality_name}_candidate_count_metric"
                ] = candidate_count

        result["physics_total_loss"] = total
        return total, result


__all__ = [
    "compute_smpl_physics_losses",
    "consecutive_valid_mask",
    "derivative_valid_mask",
    "finite_difference",
    "rollout_canonical_root",
    "so3_angular_velocity",
    "sole_penetration_loss",
]
