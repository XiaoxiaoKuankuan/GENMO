"""BUMI qpos30 模型的表示、FK、接触、脚滑与根倾斜训练损失。

网络监督只覆盖能决定 qpos28 的 30 维，不存在可与 qpos 冲突的 link 回归分支。所有 link、
鞋底和穿透几何均由预测 qpos 经过固定可微 FK 得到。Root rotation 使用完整 SO(3) 测地
误差；额外 root tilt 项从 ZYX 根旋转中显式提取 roll/pitch 环绕角，并惩罚超过 GT 合理
倾斜包络的异常大倾角，因此防躺倒由模型损失完成，而不是部署时强制改根四元数。

左右接触 head 使用版本化 FK 足底标签做 BCE；foot-slide loss 只在 GT 连续接触且两帧都
有效时惩罚预测鞋底水平速度，避免模型通过把接触概率降为零逃避脚滑约束。所有物理项在
FP32 中计算，并同时记录原始量、按物理尺度归一化量和加权量。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from gem.utils.rotation_conversions import rotation_6d_to_matrix

from .endecoder import BumiEndecoder
from .feature_codec import BUMI_FEATURE_SLICES

BUMI_LOSS_CONTRACT_VERSION = "physical_qpos30_contact_v2"
BUMI_LOSS_NAMES = (
    "repr_root_pos",
    "repr_root_rot",
    "repr_joint",
    "root_pos",
    "root_rot",
    "root_tilt",
    "joint_dof",
    "fk_body_pos",
    "joint_velocity",
    "joint_acceleration",
    "joint_jerk",
    "joint_limit",
    "contact_bce",
    "foot_slide",
    "penetration",
    "root_height",
)

BUMI_PHYSICAL_V2_SCALES = {
    "root_pos": 1.0,
    "root_rot": torch.pi,
    "root_tilt": 0.35,
    "joint_dof": 1.0,
    "fk_body_pos": 1.0,
    "joint_velocity": 6.0,
    "joint_acceleration": 180.0,
    "joint_jerk": 600.0,
    "joint_limit": 0.1,
    "contact_bce": 1.0,
    "foot_slide": 1.0,
    "penetration": 0.05,
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
    return (value * weights).sum() / weights.sum().clamp_min(1.0)


def temporal_difference_mask(valid: torch.Tensor, order: int) -> torch.Tensor:
    """N 阶差分只有在参与的 N+1 帧都有效时才计入损失。"""

    if valid.ndim != 2:
        raise ValueError(f"valid mask must have shape [B,T], got {valid.shape}")
    if int(order) < 1:
        raise ValueError("temporal difference order must be positive")
    length = valid.shape[1]
    if length <= order:
        return valid[:, :0]
    result = torch.ones((valid.shape[0], length - order), dtype=torch.bool, device=valid.device)
    for offset in range(order + 1):
        result &= valid[:, offset : offset + length - order]
    return result


def so3_geodesic_angle(pred_rotation: torch.Tensor, target_rotation: torch.Tensor) -> torch.Tensor:
    if pred_rotation.shape != target_rotation.shape or pred_rotation.shape[-2:] != (3, 3):
        raise ValueError(
            "SO(3) inputs must have matching [...,3,3] shapes, got "
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
    # ``vector_norm(0)`` 与 ``atan2(0, positive)`` 的组合在完全正确的单位旋转处会产生
    # 未定义反向梯度。极小平方下限只用于数值求导，角度偏差低于 1e-6 rad。
    sin_twice = torch.sqrt(skew.square().sum(dim=-1) + 1.0e-12)
    cos_twice = relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0
    return torch.atan2(sin_twice, cos_twice)


def root_tilt_loss_values(
    pred_rotation: torch.Tensor,
    target_rotation: torch.Tensor,
    valid: torch.Tensor,
    *,
    upright_allowance_rad: float = 0.35,
    target_margin_rad: float = 0.10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 roll/pitch 定向误差和异常大倾角安全项。"""

    def roll_pitch_zyx(rotation: torch.Tensor) -> torch.Tensor:
        # 对 R = Rz(yaw) @ Ry(pitch) @ Rx(roll) 显式取 roll/pitch；左乘任意 yaw
        # 不会改变结果，因此该监督不重复惩罚舞蹈中的水平转向。
        roll = torch.atan2(rotation[..., 2, 1], rotation[..., 2, 2])
        pitch = torch.atan2(
            -rotation[..., 2, 0],
            torch.sqrt(rotation[..., 0, 0].square() + rotation[..., 1, 0].square() + 1.0e-12),
        )
        return torch.stack((roll, pitch), dim=-1)

    pred_roll_pitch = roll_pitch_zyx(pred_rotation)
    target_roll_pitch = roll_pitch_zyx(target_rotation)
    angle_delta = torch.atan2(
        torch.sin(pred_roll_pitch - target_roll_pitch),
        torch.cos(pred_roll_pitch - target_roll_pitch),
    )
    direction_error = F.smooth_l1_loss(
        angle_delta,
        torch.zeros_like(angle_delta),
        reduction="none",
    )
    pred_up_xy = pred_rotation[..., :2, 2]
    target_up_xy = target_rotation[..., :2, 2]
    pred_tilt = torch.atan2(
        torch.sqrt(pred_up_xy.square().sum(dim=-1) + 1.0e-12),
        pred_rotation[..., 2, 2],
    )
    target_tilt = torch.atan2(
        torch.sqrt(target_up_xy.square().sum(dim=-1) + 1.0e-12),
        target_rotation[..., 2, 2],
    )
    allowance = torch.maximum(
        target_tilt + float(target_margin_rad),
        target_tilt.new_full((), float(upright_allowance_rad)),
    )
    excessive_tilt = F.relu(pred_tilt - allowance)
    raw = _masked_mean(direction_error, valid) + _masked_mean(excessive_tilt, valid)
    normalized = _masked_mean(direction_error / float(upright_allowance_rad), valid) + _masked_mean(
        excessive_tilt / float(upright_allowance_rad), valid
    )
    return raw, normalized


class BumiRobotLosses(nn.Module):
    """计算 qpos30 表示损失与全部 FK 机器人辅助损失。"""

    def __init__(
        self,
        endecoder: BumiEndecoder,
        weights: Mapping[str, float],
        fps: int = 30,
        contract_version: str = BUMI_LOSS_CONTRACT_VERSION,
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
        if self.contract_version != BUMI_LOSS_CONTRACT_VERSION:
            raise ValueError(
                "qpos30 BUMI only supports loss contract "
                f"{BUMI_LOSS_CONTRACT_VERSION!r}, got {self.contract_version!r}"
            )
        if self.ground_semantics not in {
            "gmr_foot_sole_ground_zero_v1",
            "legacy_body_origin_min_zero",
            "mixed_floor_zero_fk_contact_v2",
        }:
            raise ValueError(
                "qpos30 contact loss requires an explicit FK-ground contract; "
                f"got {self.ground_semantics!r}"
            )
        if self.auxiliary_warmup_steps < 0:
            raise ValueError("auxiliary_warmup_steps must be non-negative")
        self.weights = {name: float(weights.get(name, 0.0)) for name in BUMI_LOSS_NAMES}
        unknown = set(weights) - set(BUMI_LOSS_NAMES)
        if unknown:
            raise ValueError(f"Unknown BUMI loss weights: {sorted(unknown)}")
        if any(
            not torch.isfinite(torch.tensor(value)) or value < 0.0
            for value in self.weights.values()
        ):
            raise ValueError("BUMI loss weights must be finite and non-negative")
        if self.weights["contact_bce"] <= 0.0 or self.weights["foot_slide"] <= 0.0:
            raise ValueError("qpos30 contact contract requires positive contact_bce and foot_slide")

    @staticmethod
    def _repr_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        start, end = BUMI_FEATURE_SLICES[name]
        return _masked_mean((pred[..., start:end] - target[..., start:end]).square(), valid)

    @staticmethod
    def _smooth_l1_pair(
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        error = prediction - target
        zero = torch.zeros_like(error)
        raw = _masked_mean(F.smooth_l1_loss(error, zero, beta=1.0, reduction="none"), mask)
        normalized = _masked_mean(
            F.smooth_l1_loss(error / float(scale), zero, beta=1.0, reduction="none"),
            mask,
        )
        return raw, normalized

    def forward(
        self,
        inputs: Mapping[str, Any],
        model_output: Mapping[str, torch.Tensor | None],
        decode_dict: Mapping[str, torch.Tensor],
        pred_qpos_canonical: torch.Tensor,
        pred_fk: Mapping[str, torch.Tensor],
        *,
        global_step: int = 0,
    ) -> dict[str, torch.Tensor]:
        required = (
            "target_x",
            "target_physical_features",
            "target_qpos_canonical",
            "target_body_link_pos_root",
            "target_foot_contact",
            "target_foot_contact_mask",
            "target_contact_ground_height",
            "mask",
        )
        missing = [key for key in required if key not in inputs]
        if missing:
            raise KeyError(f"BUMI qpos30 loss inputs are missing {missing}")
        valid = inputs["mask"]["valid"].bool()
        pred_source = model_output.get("pred_x")
        if pred_source is None:
            pred_source = model_output.get("pred_x_start")
        if not isinstance(pred_source, torch.Tensor):
            raise KeyError("BUMI model output is missing pred_x/pred_x_start")

        pred_norm = pred_source.float()
        target_norm = inputs["target_x"].float()
        target_physical = inputs["target_physical_features"].float()
        pred_qpos = pred_qpos_canonical.float()
        pred_joint = decode_dict["joint_dof"].float()
        target_components = self.endecoder.codec.split_features(target_physical)
        pred_body_root = self.endecoder.codec.body_positions_in_root_frame(
            pred_qpos[..., :3],
            pred_qpos[..., 3:7],
            pred_fk["body_pos_w"].float()[..., 1:, :],
        )

        raw: dict[str, torch.Tensor] = {}
        normalized: dict[str, torch.Tensor] = {}
        raw["repr_root_pos"] = _masked_mean(
            (pred_norm[..., :3] - target_norm[..., :3]).square(), valid
        )
        normalized["repr_root_pos"] = raw["repr_root_pos"]
        raw["repr_root_rot"] = self._repr_loss(pred_norm, target_norm, valid, "root_rot_local")
        normalized["repr_root_rot"] = raw["repr_root_rot"]
        raw["repr_joint"] = self._repr_loss(pred_norm, target_norm, valid, "joint_dof")
        normalized["repr_joint"] = raw["repr_joint"]

        raw["root_pos"], normalized["root_pos"] = self._smooth_l1_pair(
            pred_qpos[..., :3],
            inputs["target_qpos_canonical"].float()[..., :3],
            valid,
            BUMI_PHYSICAL_V2_SCALES["root_pos"],
        )
        pred_rotation = rotation_6d_to_matrix(decode_dict["root_rot_local_6d"].float())
        rot_start, rot_end = BUMI_FEATURE_SLICES["root_rot_local"]
        target_rotation = rotation_6d_to_matrix(target_physical[..., rot_start:rot_end])
        root_angle = so3_geodesic_angle(pred_rotation, target_rotation)
        raw["root_rot"] = _masked_mean(root_angle, valid)
        normalized["root_rot"] = _masked_mean(
            root_angle / float(BUMI_PHYSICAL_V2_SCALES["root_rot"]), valid
        )
        raw["root_tilt"], normalized["root_tilt"] = root_tilt_loss_values(
            pred_rotation,
            target_rotation,
            valid,
        )
        raw["joint_dof"], normalized["joint_dof"] = self._smooth_l1_pair(
            pred_joint,
            target_components.joint_dof.float(),
            valid,
            BUMI_PHYSICAL_V2_SCALES["joint_dof"],
        )
        raw["fk_body_pos"], normalized["fk_body_pos"] = self._smooth_l1_pair(
            pred_body_root,
            inputs["target_body_link_pos_root"].float(),
            valid,
            BUMI_PHYSICAL_V2_SCALES["fk_body_pos"],
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
            raw[name], normalized[name] = self._smooth_l1_pair(
                pred_delta,
                target_delta,
                temporal_difference_mask(valid, order),
                BUMI_PHYSICAL_V2_SCALES[name],
            )

        lower = self.kinematics.joint_lower_limits.to(pred_joint)
        upper = self.kinematics.joint_upper_limits.to(pred_joint)
        violation = F.relu(lower - pred_joint) + F.relu(pred_joint - upper)
        raw["joint_limit"] = _masked_mean(violation, valid)
        normalized["joint_limit"] = _masked_mean(
            F.smooth_l1_loss(
                violation / BUMI_PHYSICAL_V2_SCALES["joint_limit"],
                torch.zeros_like(violation),
                reduction="none",
            ),
            valid,
        )

        contact_logits = model_output.get("static_conf_logits")
        if (
            not isinstance(contact_logits, torch.Tensor)
            or contact_logits.shape != inputs["target_foot_contact"].shape
        ):
            raise RuntimeError(
                "qpos30 contact loss requires static_conf_logits matching target_foot_contact"
            )
        target_contact = inputs["target_foot_contact"].to(contact_logits)
        contact_mask = inputs["target_foot_contact_mask"].bool()
        raw["contact_bce"] = _masked_mean(
            F.binary_cross_entropy_with_logits(
                contact_logits.float(), target_contact.float(), reduction="none"
            ),
            contact_mask,
        )
        normalized["contact_bce"] = raw["contact_bce"]

        pred_sole = self.kinematics.aggregate_sole_by_foot(
            pred_fk["body_pos_w"].float(), pred_fk["body_quat_w"].float()
        )
        foot_velocity = torch.diff(pred_sole["foot_points_w"][..., :2], dim=1) * float(self.fps)
        foot_speed = torch.linalg.vector_norm(foot_velocity, dim=-1)
        contact_bool = target_contact >= 0.5
        slide_gate = (
            contact_bool[:, 1:]
            & contact_bool[:, :-1]
            & contact_mask[:, 1:]
            & contact_mask[:, :-1]
            & temporal_difference_mask(valid, 1)[..., None]
        )
        raw["foot_slide"] = _masked_mean(foot_speed, slide_gate)
        normalized["foot_slide"] = raw["foot_slide"] / BUMI_PHYSICAL_V2_SCALES["foot_slide"]

        ground_height_local = inputs["target_contact_ground_height"].to(pred_norm)
        ground_height_local = ground_height_local - self.kinematics.default_qpos[2].to(pred_norm)
        while ground_height_local.ndim < pred_sole["foot_bottom_height"].ndim:
            ground_height_local = ground_height_local.unsqueeze(-1)
        penetration = F.relu(ground_height_local - pred_sole["foot_bottom_height"])
        raw["penetration"] = _masked_mean(penetration, valid)
        normalized["penetration"] = raw["penetration"] / BUMI_PHYSICAL_V2_SCALES["penetration"]
        raw["root_height"], normalized["root_height"] = self._smooth_l1_pair(
            decode_dict["root_height_offset"].float()[..., 0],
            target_components.root_height_offset.float()[..., 0],
            valid,
            BUMI_PHYSICAL_V2_SCALES["root_height"],
        )

        if self.auxiliary_warmup_steps <= 0:
            warmup = 1.0
        else:
            warmup = min(max(float(global_step), 0.0) / self.auxiliary_warmup_steps, 1.0)
        always_on = {
            "repr_root_pos",
            "repr_root_rot",
            "repr_joint",
            "root_rot",
            "root_tilt",
            "contact_bce",
        }
        total = pred_norm.new_zeros(())
        output: dict[str, torch.Tensor] = {"auxiliary_warmup_factor": pred_norm.new_tensor(warmup)}
        for name in BUMI_LOSS_NAMES:
            factor = 1.0 if name in always_on else warmup
            weighted = normalized[name] * (self.weights[name] * factor)
            output[f"raw_{name}_loss"] = raw[name]
            output[f"normalized_{name}_loss"] = normalized[name]
            output[f"weighted_{name}_loss"] = weighted
            total = total + weighted
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("BUMI qpos30 total loss is NaN or Inf")
        output["loss"] = total
        return output


__all__ = [
    "BUMI_LOSS_CONTRACT_VERSION",
    "BUMI_LOSS_NAMES",
    "BUMI_PHYSICAL_V2_SCALES",
    "BumiRobotLosses",
    "root_tilt_loss_values",
    "so3_geodesic_angle",
    "temporal_difference_mask",
]
