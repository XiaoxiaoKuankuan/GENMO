"""Mask-correct physical and representation losses for BUMI music motion."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from gem.utils.rotation_conversions import rotation_6d_to_matrix

from .endecoder import BumiEndecoder
from .feature_codec import BUMI_FEATURE_SLICES

BUMI_LOSS_NAMES = (
    "repr_root_pos",
    "repr_root_rot",
    "repr_joint",
    "repr_body_pos",
    "root_pos",
    "root_rot",
    "joint_dof",
    "fk_body_pos",
    "fk_consistency",
    "joint_velocity",
    "joint_acceleration",
    "joint_jerk",
    "joint_limit",
    "contact_bce",
    "foot_slide",
    "penetration",
    "root_height",
)

BUMI_PHYSICAL_V1_SCALES = {
    "root_pos": 1.0,
    "root_rot": torch.pi,
    "joint_dof": 1.0,
    "fk_body_pos": 1.0,
    "fk_consistency": 1.0,
    "joint_velocity": 6.0,
    "joint_acceleration": 180.0,
    "joint_jerk": 600.0,
    "joint_limit": 0.1,
    "root_height": 1.0,
}


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise TypeError("_masked_mean expects tensor value and mask")
    expanded = mask.bool()
    while expanded.ndim < value.ndim:
        expanded = expanded.unsqueeze(-1)
    try:
        expanded = expanded.expand_as(value)
    except RuntimeError as exc:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} cannot supervise value shape {tuple(value.shape)}"
        ) from exc
    weights = expanded.to(dtype=value.dtype)
    denominator = weights.sum().clamp_min(1.0)
    return (value * weights).sum() / denominator


def temporal_difference_mask(valid: torch.Tensor, order: int) -> torch.Tensor:
    """Require every frame participating in an order-N finite difference."""

    if valid.ndim != 2:
        raise ValueError(f"valid mask must have shape [B,T], got {valid.shape}")
    if order < 1:
        raise ValueError("temporal difference order must be positive")
    length = valid.shape[1]
    if length <= order:
        return valid[:, :0]
    result = torch.ones(
        (valid.shape[0], length - order), dtype=torch.bool, device=valid.device
    )
    for offset in range(order + 1):
        result &= valid[:, offset : offset + length - order]
    return result


def so3_geodesic_angle(pred_rotation: torch.Tensor, target_rotation: torch.Tensor) -> torch.Tensor:
    if pred_rotation.shape != target_rotation.shape or pred_rotation.shape[-2:] != (3, 3):
        raise ValueError(
            f"SO(3) inputs must have matching [...,3,3] shapes, got "
            f"{pred_rotation.shape}/{target_rotation.shape}"
        )
    relative = pred_rotation @ target_rotation.transpose(-1, -2)
    skew = torch.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        dim=-1,
    )
    sin_twice = torch.linalg.vector_norm(skew, dim=-1)
    cos_twice = relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0
    return torch.atan2(sin_twice, cos_twice)


class BumiRobotLosses(nn.Module):
    """Compute every BUMI loss term and combine YAML-provided weights."""

    def __init__(
        self,
        endecoder: BumiEndecoder,
        weights: Mapping[str, float],
        fps: int = 30,
        contract_version: str = "legacy_v0",
        auxiliary_warmup_steps: int = 0,
        ground_semantics: str | None = None,
    ) -> None:
        super().__init__()
        self.endecoder = endecoder
        self.kinematics = endecoder.kinematics
        self.fps = int(fps)
        self.contract_version = str(contract_version)
        self.auxiliary_warmup_steps = int(auxiliary_warmup_steps)
        self.ground_semantics = ground_semantics
        if self.fps != 30:
            raise ValueError(f"BUMI losses require 30 FPS, got {fps}")
        self.weights = {name: float(weights.get(name, 0.0)) for name in BUMI_LOSS_NAMES}
        unknown = set(weights) - set(BUMI_LOSS_NAMES)
        if unknown:
            raise ValueError(f"Unknown BUMI loss weights: {sorted(unknown)}")
        if any(not torch.isfinite(torch.tensor(value)) or value < 0.0 for value in self.weights.values()):
            raise ValueError("BUMI loss weights must be finite and non-negative")
        if self.contract_version not in {"legacy_v0", "physical_v1"}:
            raise ValueError(
                "BUMI loss contract_version must be 'legacy_v0' or 'physical_v1'"
            )
        if self.auxiliary_warmup_steps < 0:
            raise ValueError("auxiliary_warmup_steps must be non-negative")
        if self.contract_version == "physical_v1":
            if self.ground_semantics != "legacy_body_origin_min_zero":
                raise ValueError(
                    "BUMI physical_v1 requires ground_semantics="
                    "'legacy_body_origin_min_zero'"
                )
            forbidden = {
                name: self.weights[name]
                for name in ("contact_bce", "foot_slide", "penetration")
                if self.weights[name] != 0.0
            }
            if forbidden:
                raise ValueError(
                    "Unadjusted legacy-body-origin data cannot enable contact/slide/"
                    f"penetration losses: {forbidden}"
                )

    @staticmethod
    def _repr_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        start, end = BUMI_FEATURE_SLICES[name]
        return _masked_mean((pred[..., start:end] - target[..., start:end]).square(), valid)

    def _forward_legacy(
        self,
        inputs: Mapping[str, Any],
        model_output: Mapping[str, torch.Tensor | None],
        decode_dict: Mapping[str, torch.Tensor],
        pred_qpos_canonical: torch.Tensor,
        pred_body_link_pos_local_fk: torch.Tensor,
        pred_body_quat_local_fk: torch.Tensor,
        global_step: int = 0,
    ) -> dict[str, torch.Tensor]:
        del global_step
        required = (
            "target_x",
            "target_physical_features",
            "target_qpos_canonical",
            "target_body_link_pos_local",
            "target_foot_contact",
            "target_foot_contact_mask",
            "mask",
        )
        missing = [key for key in required if key not in inputs]
        if missing:
            raise KeyError(f"BUMI loss inputs are missing {missing}")
        valid = inputs["mask"]["valid"].bool()
        pred_norm = model_output.get("pred_x")
        if pred_norm is None:
            pred_norm = model_output.get("pred_x_start")
        if not isinstance(pred_norm, torch.Tensor):
            raise KeyError("BUMI model output is missing pred_x/pred_x_start")
        target_norm = inputs["target_x"]
        target_physical = inputs["target_physical_features"]
        target_components = self.endecoder.codec.split_features(target_physical)
        pred_body_raw = decode_dict["body_link_pos_local_raw"]

        terms: dict[str, torch.Tensor] = {}
        terms["repr_root_pos"] = self._repr_loss(
            pred_norm, target_norm, valid, "root_pos_local"
        )
        terms["repr_root_rot"] = self._repr_loss(
            pred_norm, target_norm, valid, "root_rot_local"
        )
        terms["repr_joint"] = self._repr_loss(pred_norm, target_norm, valid, "joint_dof")
        terms["repr_body_pos"] = self._repr_loss(
            pred_norm, target_norm, valid, "body_link_pos_local"
        )
        terms["root_pos"] = _masked_mean(
            F.smooth_l1_loss(
                decode_dict["root_pos_local"],
                target_components.root_pos_local,
                reduction="none",
            ),
            valid,
        )
        pred_rotation = rotation_6d_to_matrix(decode_dict["root_rot_local_6d"])
        rot_start, rot_end = BUMI_FEATURE_SLICES["root_rot_local"]
        target_rotation = rotation_6d_to_matrix(target_physical[..., rot_start:rot_end])
        terms["root_rot"] = _masked_mean(
            so3_geodesic_angle(pred_rotation, target_rotation), valid
        )
        terms["joint_dof"] = _masked_mean(
            F.smooth_l1_loss(
                decode_dict["joint_dof"], target_components.joint_dof, reduction="none"
            ),
            valid,
        )
        terms["fk_body_pos"] = _masked_mean(
            F.smooth_l1_loss(
                pred_body_link_pos_local_fk,
                inputs["target_body_link_pos_local"],
                reduction="none",
            ),
            valid,
        )
        terms["fk_consistency"] = _masked_mean(
            F.smooth_l1_loss(
                pred_body_raw, pred_body_link_pos_local_fk, reduction="none"
            ),
            valid,
        )

        for order, name in (
            (1, "joint_velocity"),
            (2, "joint_acceleration"),
            (3, "joint_jerk"),
        ):
            pred_delta = torch.diff(decode_dict["joint_dof"], n=order, dim=1) * (
                float(self.fps) ** order
            )
            target_delta = torch.diff(target_components.joint_dof, n=order, dim=1) * (
                float(self.fps) ** order
            )
            terms[name] = _masked_mean(
                F.smooth_l1_loss(pred_delta, target_delta, reduction="none"),
                temporal_difference_mask(valid, order),
            )

        lower = self.kinematics.joint_lower_limits.to(decode_dict["joint_dof"])
        upper = self.kinematics.joint_upper_limits.to(decode_dict["joint_dof"])
        joint_violation = F.relu(lower - decode_dict["joint_dof"]) + F.relu(
            decode_dict["joint_dof"] - upper
        )
        terms["joint_limit"] = _masked_mean(joint_violation, valid)

        contact_logits = model_output.get("static_conf_logits")
        if contact_logits is None:
            if self.weights["contact_bce"] > 0.0:
                raise RuntimeError(
                    "contact_bce weight is non-zero but the denoiser has no 2D static_conf_logits"
                )
            terms["contact_bce"] = pred_norm.new_zeros(())
        else:
            target_contact = inputs["target_foot_contact"].to(contact_logits)
            if tuple(contact_logits.shape) != tuple(target_contact.shape):
                raise ValueError(
                    f"BUMI contact logits/target shape mismatch: "
                    f"{contact_logits.shape}/{target_contact.shape}"
                )
            terms["contact_bce"] = _masked_mean(
                F.binary_cross_entropy_with_logits(
                    contact_logits, target_contact, reduction="none"
                ),
                inputs["target_foot_contact_mask"].bool(),
            )

        pred_body_pos_all = torch.cat(
            (pred_qpos_canonical[..., None, :3], pred_body_link_pos_local_fk), dim=-2
        )
        pred_sole = self.kinematics.aggregate_sole_by_foot(
            pred_body_pos_all, pred_body_quat_local_fk
        )
        foot_xy = pred_sole["foot_points_w"][..., :2]
        foot_velocity = torch.diff(foot_xy, dim=1) * float(self.fps)
        foot_speed = torch.linalg.vector_norm(foot_velocity, dim=-1)
        contact = inputs["target_foot_contact"].bool()
        contact_valid = inputs["target_foot_contact_mask"].bool()
        slide_gate = (
            contact[:, 1:]
            & contact[:, :-1]
            & contact_valid[:, 1:]
            & contact_valid[:, :-1]
            & temporal_difference_mask(valid, 1)[..., None]
        )
        terms["foot_slide"] = _masked_mean(foot_speed, slide_gate)

        # Canonical Z=0 is the default-root-height plane; world ground is
        # therefore at -default_root_height in canonical coordinates.
        ground_height_local = -self.kinematics.default_qpos[2].to(pred_norm)
        penetration = F.relu(
            ground_height_local - pred_sole["foot_bottom_height"]
        )
        terms["penetration"] = _masked_mean(penetration, valid)
        terms["root_height"] = _masked_mean(
            F.smooth_l1_loss(
                decode_dict["root_pos_local"][..., 2],
                target_components.root_pos_local[..., 2],
                reduction="none",
            ),
            valid,
        )

        total = pred_norm.new_zeros(())
        output: dict[str, torch.Tensor] = {}
        for name in BUMI_LOSS_NAMES:
            value = terms[name]
            output[f"{name}_loss"] = value
            total = total + value * self.weights[name]
        output["loss"] = total
        return output

    @staticmethod
    def _smooth_l1_pair(
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        error = prediction - target
        zero = torch.zeros_like(error)
        raw = _masked_mean(
            F.smooth_l1_loss(error, zero, beta=1.0, reduction="none"), mask
        )
        normalized = _masked_mean(
            F.smooth_l1_loss(error / float(scale), zero, beta=1.0, reduction="none"),
            mask,
        )
        return raw, normalized

    def _forward_physical_v1(
        self,
        inputs: Mapping[str, Any],
        model_output: Mapping[str, torch.Tensor | None],
        decode_dict: Mapping[str, torch.Tensor],
        pred_qpos_canonical: torch.Tensor,
        pred_body_link_pos_local_fk: torch.Tensor,
        pred_body_quat_local_fk: torch.Tensor,
        global_step: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Low-weight GT derivative losses for the unadjusted BUMI corpus."""

        del pred_qpos_canonical, pred_body_quat_local_fk
        required = (
            "target_x",
            "target_physical_features",
            "target_body_link_pos_local",
            "mask",
        )
        missing = [key for key in required if key not in inputs]
        if missing:
            raise KeyError(f"BUMI physical_v1 loss inputs are missing {missing}")
        valid = inputs["mask"]["valid"].bool()
        pred_norm_source = model_output.get("pred_x")
        if pred_norm_source is None:
            pred_norm_source = model_output.get("pred_x_start")
        if not isinstance(pred_norm_source, torch.Tensor):
            raise KeyError("BUMI model output is missing pred_x/pred_x_start")

        # Autocast may be active around the Lightning forward. Explicit casts
        # preserve the graph while forcing every physical calculation to FP32.
        pred_norm = pred_norm_source.float()
        target_norm = inputs["target_x"].float()
        target_physical = inputs["target_physical_features"].float()
        pred_root_pos = decode_dict["root_pos_local"].float()
        pred_root_rot6d = decode_dict["root_rot_local_6d"].float()
        pred_joint = decode_dict["joint_dof"].float()
        pred_body_raw = decode_dict["body_link_pos_local_raw"].float()
        pred_body_fk = pred_body_link_pos_local_fk.float()
        target_body_fk = inputs["target_body_link_pos_local"].float()
        target_components = self.endecoder.codec.split_features(target_physical)

        raw: dict[str, torch.Tensor] = {}
        normalized: dict[str, torch.Tensor] = {}
        for loss_name, feature_name in (
            ("repr_root_pos", "root_pos_local"),
            ("repr_root_rot", "root_rot_local"),
            ("repr_joint", "joint_dof"),
            ("repr_body_pos", "body_link_pos_local"),
        ):
            value = self._repr_loss(pred_norm, target_norm, valid, feature_name)
            raw[loss_name] = value
            normalized[loss_name] = value

        raw["root_pos"], normalized["root_pos"] = self._smooth_l1_pair(
            pred_root_pos,
            target_components.root_pos_local.float(),
            valid,
            BUMI_PHYSICAL_V1_SCALES["root_pos"],
        )
        pred_rotation = rotation_6d_to_matrix(pred_root_rot6d)
        rot_start, rot_end = BUMI_FEATURE_SLICES["root_rot_local"]
        target_rotation = rotation_6d_to_matrix(
            target_physical[..., rot_start:rot_end]
        )
        root_angle = so3_geodesic_angle(pred_rotation, target_rotation)
        raw["root_rot"] = _masked_mean(root_angle, valid)
        normalized["root_rot"] = _masked_mean(
            root_angle / float(BUMI_PHYSICAL_V1_SCALES["root_rot"]), valid
        )
        raw["joint_dof"], normalized["joint_dof"] = self._smooth_l1_pair(
            pred_joint,
            target_components.joint_dof.float(),
            valid,
            BUMI_PHYSICAL_V1_SCALES["joint_dof"],
        )
        raw["fk_body_pos"], normalized["fk_body_pos"] = self._smooth_l1_pair(
            pred_body_fk,
            target_body_fk,
            valid,
            BUMI_PHYSICAL_V1_SCALES["fk_body_pos"],
        )
        raw["fk_consistency"], normalized["fk_consistency"] = self._smooth_l1_pair(
            pred_body_raw,
            pred_body_fk,
            valid,
            BUMI_PHYSICAL_V1_SCALES["fk_consistency"],
        )

        target_joint = target_components.joint_dof.float()
        for order, name in (
            (1, "joint_velocity"),
            (2, "joint_acceleration"),
            (3, "joint_jerk"),
        ):
            multiplier = float(self.fps) ** order
            pred_delta = torch.diff(pred_joint, n=order, dim=1) * multiplier
            target_delta = torch.diff(target_joint, n=order, dim=1) * multiplier
            derivative_mask = temporal_difference_mask(valid, order)
            raw[name], normalized[name] = self._smooth_l1_pair(
                pred_delta,
                target_delta,
                derivative_mask,
                BUMI_PHYSICAL_V1_SCALES[name],
            )

        lower = self.kinematics.joint_lower_limits.to(pred_joint)
        upper = self.kinematics.joint_upper_limits.to(pred_joint)
        violation = F.relu(lower - pred_joint) + F.relu(pred_joint - upper)
        raw["joint_limit"] = _masked_mean(violation, valid)
        normalized["joint_limit"] = _masked_mean(
            F.smooth_l1_loss(
                violation / BUMI_PHYSICAL_V1_SCALES["joint_limit"],
                torch.zeros_like(violation),
                beta=1.0,
                reduction="none",
            ),
            valid,
        )
        raw["root_height"], normalized["root_height"] = self._smooth_l1_pair(
            pred_root_pos[..., 2],
            target_components.root_pos_local[..., 2].float(),
            valid,
            BUMI_PHYSICAL_V1_SCALES["root_height"],
        )

        # These are contractually disabled until root height has true sole-ground
        # semantics and audited labels exist. Keep explicit zero logs so a run
        # cannot be mistaken for one that trained these objectives.
        zero = pred_norm.new_zeros(())
        for name in ("contact_bce", "foot_slide", "penetration"):
            raw[name] = zero
            normalized[name] = zero

        if self.auxiliary_warmup_steps <= 0:
            warmup = 1.0
        else:
            warmup = min(max(float(global_step), 0.0) / self.auxiliary_warmup_steps, 1.0)
        representation = {
            "repr_root_pos",
            "repr_root_rot",
            "repr_joint",
            "repr_body_pos",
        }
        total = zero
        output: dict[str, torch.Tensor] = {
            "auxiliary_warmup_factor": pred_norm.new_tensor(warmup)
        }
        for name in BUMI_LOSS_NAMES:
            factor = 1.0 if name in representation else warmup
            weighted = normalized[name] * (self.weights[name] * factor)
            output[f"raw_{name}_loss"] = raw[name]
            output[f"normalized_{name}_loss"] = normalized[name]
            output[f"weighted_{name}_loss"] = weighted
            total = total + weighted
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("BUMI physical_v1 total loss is NaN or Inf")
        output["loss"] = total
        return output

    def forward(
        self,
        inputs: Mapping[str, Any],
        model_output: Mapping[str, torch.Tensor | None],
        decode_dict: Mapping[str, torch.Tensor],
        pred_qpos_canonical: torch.Tensor,
        pred_body_link_pos_local_fk: torch.Tensor,
        pred_body_quat_local_fk: torch.Tensor,
        global_step: int = 0,
    ) -> dict[str, torch.Tensor]:
        implementation = (
            self._forward_physical_v1
            if self.contract_version == "physical_v1"
            else self._forward_legacy
        )
        return implementation(
            inputs,
            model_output,
            decode_dict,
            pred_qpos_canonical,
            pred_body_link_pos_local_fk,
            pred_body_quat_local_fk,
            global_step=global_step,
        )


__all__ = [
    "BUMI_LOSS_NAMES",
    "BUMI_PHYSICAL_V1_SCALES",
    "BumiRobotLosses",
    "so3_geodesic_angle",
    "temporal_difference_mask",
]
