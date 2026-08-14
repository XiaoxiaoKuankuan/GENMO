"""Physical sanity checks for decoded music-only SMPL motion."""

from __future__ import annotations

from typing import Any

import torch

from gem.utils.rotation_conversions import axis_angle_to_matrix


def evaluate_global_motion_sanity(
    body_params: dict[str, torch.Tensor],
    *,
    max_mean_root_step_m: float = 0.25,
    min_mean_body_up_y: float = 0.5,
) -> dict[str, Any]:
    """Measure gross scale/orientation failures without claiming dance quality.

    Thresholds are intentionally generous.  They detect unit/coordinate errors
    such as metre-scale frame-to-frame jumps or a persistently horizontal body;
    they are not a replacement for perceptual evaluation.
    """
    transl = body_params["transl"].detach().float().cpu()
    orient = body_params["global_orient"].detach().float().cpu()
    if transl.ndim == 3 and transl.shape[0] == 1:
        transl = transl[0]
    if orient.ndim == 3 and orient.shape[0] == 1:
        orient = orient[0]
    if transl.ndim != 2 or transl.shape[1] != 3:
        raise ValueError(f"transl must be [T,3], got {tuple(transl.shape)}")
    if orient.shape != transl.shape:
        raise ValueError(
            f"global_orient must match transl {tuple(transl.shape)}, got {tuple(orient.shape)}"
        )
    if transl.shape[0] > 1:
        root_steps = torch.linalg.vector_norm(transl[1:] - transl[:-1], dim=-1)
    else:
        root_steps = torch.zeros(1)
    body_up_y = axis_angle_to_matrix(orient)[..., 1, 1]
    issues: list[str] = []
    mean_step = float(root_steps.mean())
    mean_up = float(body_up_y.mean())
    if mean_step > max_mean_root_step_m:
        issues.append(
            "mean root displacement per frame is too large: "
            f"{mean_step:.6f}m > {max_mean_root_step_m:.6f}m"
        )
    if mean_up < min_mean_body_up_y:
        issues.append(
            "body up-axis is not predominantly aligned with +Y: "
            f"mean dot={mean_up:.6f} < {min_mean_body_up_y:.6f}"
        )
    translation_range = transl.amax(dim=0) - transl.amin(dim=0)
    return {
        "finite": bool(torch.isfinite(transl).all() and torch.isfinite(orient).all()),
        "mean_root_step_m": mean_step,
        "p95_root_step_m": float(torch.quantile(root_steps, 0.95)),
        "max_root_step_m": float(root_steps.max()),
        "translation_range_m": translation_range.tolist(),
        "mean_body_up_y_dot": mean_up,
        "p05_body_up_y_dot": float(torch.quantile(body_up_y, 0.05)),
        "upright_frame_fraction": float((body_up_y > 0.5).float().mean()),
        "thresholds": {
            "max_mean_root_step_m": max_mean_root_step_m,
            "min_mean_body_up_y": min_mean_body_up_y,
        },
        "issues": issues,
        "physical_sanity_pass": not issues,
    }
