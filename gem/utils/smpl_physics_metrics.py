# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Evaluation metrics shared by baseline/physics-v1 held-out comparisons."""

from __future__ import annotations

import torch

from gem.pipeline.smpl_physics_losses import consecutive_valid_mask, finite_difference
from gem.utils.rotation_conversions import matrix_to_axis_angle


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(value)
    return torch.where(mask, value, torch.zeros_like(value)).sum() / mask.sum().clamp_min(1)


def _vector_error(
    pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    return _masked_mean(torch.linalg.vector_norm(pred - target, dim=-1), valid)


@torch.no_grad()
def temporal_quality_metrics(
    *,
    pred_root_position: torch.Tensor,
    gt_root_position: torch.Tensor,
    pred_fk_position: torch.Tensor,
    gt_fk_position: torch.Tensor,
    pred_body_rotation: torch.Tensor,
    gt_body_rotation: torch.Tensor,
    frame_valid: torch.Tensor,
    fps: float = 30.0,
) -> dict[str, float]:
    """Report Root/FK derivative errors plus the existing FK/pose errors."""
    values = {}
    for order, label in ((1, "velocity"), (2, "acceleration"), (3, "jerk")):
        derivative_mask = consecutive_valid_mask(frame_valid, order)
        values[f"root_{label}_error"] = float(
            _vector_error(
                finite_difference(pred_root_position, order, fps),
                finite_difference(gt_root_position, order, fps),
                derivative_mask,
            )
        )
        values[f"fk_{label}_error"] = float(
            _vector_error(
                finite_difference(pred_fk_position, order, fps),
                finite_difference(gt_fk_position, order, fps),
                derivative_mask,
            )
        )
    values["fk_position_error_m"] = float(
        _vector_error(pred_fk_position, gt_fk_position, frame_valid)
    )
    relative_rotation = pred_body_rotation.mT @ gt_body_rotation
    pose_error = torch.linalg.vector_norm(matrix_to_axis_angle(relative_rotation), dim=-1)
    values["pose_geodesic_error_rad"] = float(_masked_mean(pose_error, frame_valid))
    return values


@torch.no_grad()
def sole_penetration_metrics(
    pred_sole_y: torch.Tensor,
    *,
    ground_y_local: torch.Tensor,
    frame_valid: torch.Tensor,
    ground_valid: torch.Tensor,
    tolerance_m: float = 0.01,
) -> dict[str, float]:
    """Report maximum depth and ratio of valid frames with any sole penetration."""
    ground = ground_y_local.to(pred_sole_y).reshape(-1, 1, 1)
    valid_frame = frame_valid.bool() & ground_valid.bool().reshape(-1, 1)
    depth = torch.relu(ground - pred_sole_y - float(tolerance_m))
    valid_proxy = valid_frame.unsqueeze(-1).expand_as(depth)
    valid_depth = torch.where(valid_proxy, depth, torch.zeros_like(depth))
    penetrating_frame = (valid_depth > 0).any(dim=-1) & valid_frame
    ratio = penetrating_frame.sum().float() / valid_frame.sum().clamp_min(1).float()
    return {
        "sole_max_penetration_depth_m": float(valid_depth.max()),
        "sole_penetration_frame_ratio": float(ratio),
    }


def relative_regression(new_value: float, baseline_value: float) -> float:
    """Relative metric change, with a stable zero-baseline convention."""
    if baseline_value == 0:
        return 0.0 if new_value == 0 else float("inf")
    return (float(new_value) - float(baseline_value)) / float(baseline_value)


__all__ = [
    "relative_regression",
    "sole_penetration_metrics",
    "temporal_quality_metrics",
]
