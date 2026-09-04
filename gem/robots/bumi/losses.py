"""BUMI qpos30 模型的表示、FK、接触、脚滑、限位与动态连续性训练损失。

网络监督只覆盖能决定 qpos28 的 30 维，不存在可与 qpos 冲突的 link 回归分支。所有 link、
鞋底和穿透几何均由预测 qpos 经过固定可微 FK 得到。Root rotation 使用完整 SO(3) 测地
误差；额外 root tilt 项从 ZYX 根旋转中显式提取 roll/pitch 环绕角，并惩罚超过 GT 合理
倾斜包络的异常大倾角，因此防躺倒由模型损失完成，而不是部署时强制改根四元数。

左右接触 head 使用版本化 FK 足底标签做 BCE；foot-slide loss 只在 GT 连续接触且两帧都
有效时惩罚预测鞋底水平速度，避免模型通过把接触概率降为零逃避脚滑约束。所有物理项在
FP32 中计算，并同时记录原始量、按物理尺度归一化量和加权量。v5 进一步监督根位姿与
全身 FK 的速度/加速度，并对关节跳变、限位安全边距、脚滑、接触脚高度、穿地和异常
倾斜保留 top-k/max 长尾；这些仍是运动学与有限差分代理，不包含力矩、接触力或闭环控制。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from gem.utils.rotation_conversions import matrix_to_axis_angle, rotation_6d_to_matrix

from .endecoder import BumiEndecoder
from .feature_codec import BUMI_FEATURE_SLICES

BUMI_LOSS_CONTRACT_VERSION = "physical_qpos30_contact_v2"
BUMI_LOSS_CONTRACT_V3 = "physical_qpos30_contact_v3"
BUMI_LOSS_CONTRACT_V4 = "physical_qpos30_contact_v4"
BUMI_LOSS_CONTRACT_V5 = "physical_qpos30_contact_v5"
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
BUMI_EXCESS_LOSS_NAMES = (
    "joint_acceleration_excess",
    "joint_jerk_excess",
)
BUMI_ROBUST_JOINT_LIMIT_LOSS_NAMES = (
    "joint_limit_margin",
    "joint_limit_topk",
    "joint_limit_max",
)
BUMI_ADVANCED_PHYSICS_LOSS_NAMES = (
    "root_velocity",
    "root_acceleration",
    "root_angular_velocity",
    "root_angular_acceleration",
    "fk_velocity",
    "fk_acceleration",
    "joint_acceleration_excess_topk",
    "joint_jerk_excess_topk",
    "joint_limit_margin_topk",
    "foot_slide_topk",
    "foot_slide_max",
    "foot_contact_height",
    "foot_contact_height_topk",
    "penetration_topk",
    "penetration_max",
    "root_tilt_excess_topk",
    "root_tilt_excess_max",
)
BUMI_LOSS_NAMES_BY_CONTRACT = {
    BUMI_LOSS_CONTRACT_VERSION: BUMI_LOSS_NAMES,
    BUMI_LOSS_CONTRACT_V3: BUMI_LOSS_NAMES + BUMI_EXCESS_LOSS_NAMES,
    BUMI_LOSS_CONTRACT_V4: (
        BUMI_LOSS_NAMES + BUMI_EXCESS_LOSS_NAMES + BUMI_ROBUST_JOINT_LIMIT_LOSS_NAMES
    ),
    BUMI_LOSS_CONTRACT_V5: (
        BUMI_LOSS_NAMES
        + BUMI_EXCESS_LOSS_NAMES
        + BUMI_ROBUST_JOINT_LIMIT_LOSS_NAMES
        + BUMI_ADVANCED_PHYSICS_LOSS_NAMES
    ),
}

BUMI_PHYSICAL_V2_SCALES = {
    "root_pos": 1.0,
    "root_rot": torch.pi,
    "root_tilt": 0.35,
    "joint_dof": 1.0,
    "fk_body_pos": 1.0,
    "joint_velocity": 6.0,
    "joint_acceleration": 180.0,
    "joint_jerk": 600.0,
    "joint_acceleration_excess": 180.0,
    "joint_jerk_excess": 600.0,
    "joint_limit": 0.1,
    "joint_limit_margin": 0.1,
    "joint_limit_topk": 0.1,
    "joint_limit_max": 0.1,
    "contact_bce": 1.0,
    "foot_slide": 1.0,
    "penetration": 0.05,
    "root_height": 1.0,
    "root_velocity": 2.0,
    "root_acceleration": 30.0,
    "root_angular_velocity": 6.0,
    "root_angular_acceleration": 180.0,
    "fk_velocity": 3.0,
    "fk_acceleration": 90.0,
    "joint_acceleration_excess_topk": 180.0,
    "joint_jerk_excess_topk": 600.0,
    "joint_limit_margin_topk": 0.1,
    "foot_slide_topk": 1.0,
    "foot_slide_max": 1.0,
    "foot_contact_height": 0.03,
    "foot_contact_height_topk": 0.03,
    "penetration_topk": 0.05,
    "penetration_max": 0.05,
    "root_tilt_excess_topk": 0.35,
    "root_tilt_excess_max": 0.35,
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


def _masked_topk_mean(
    value: torch.Tensor,
    mask: torch.Tensor,
    fraction: float,
) -> torch.Tensor:
    """逐条序列聚合最严重的一小部分有效关节帧，避免稀疏尖峰被全局平均稀释。"""

    expanded = mask.bool()
    while expanded.ndim < value.ndim:
        expanded = expanded.unsqueeze(-1)
    try:
        expanded = expanded.expand_as(value)
    except RuntimeError as exc:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} cannot supervise value shape {tuple(value.shape)}"
        ) from exc
    flat_value = value.reshape(value.shape[0], -1)
    flat_mask = expanded.reshape(value.shape[0], -1)
    max_k = max(1, math.ceil(flat_value.shape[1] * float(fraction)))
    masked_value = flat_value.masked_fill(~flat_mask, float("-inf"))
    top_values = torch.topk(masked_value, k=max_k, dim=-1, sorted=True).values
    valid_counts = flat_mask.sum(dim=-1)
    selected_counts = torch.ceil(valid_counts.to(torch.float32) * float(fraction)).to(torch.long)
    selected_counts = selected_counts.clamp(min=1, max=max_k)
    ranks = torch.arange(max_k, device=value.device).unsqueeze(0)
    selected = (ranks < selected_counts.unsqueeze(-1)) & (valid_counts.unsqueeze(-1) > 0)
    selected_values = torch.where(selected, top_values, torch.zeros_like(top_values))
    per_sequence = selected_values.sum(dim=-1) / selected.sum(dim=-1).clamp_min(1)
    valid_sequences = valid_counts > 0
    return (
        per_sequence * valid_sequences.to(per_sequence)
    ).sum() / valid_sequences.sum().clamp_min(1)


def _masked_max_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """先取每条序列最严重的有效关节帧，再跨 batch 求均值。"""

    expanded = mask.bool()
    while expanded.ndim < value.ndim:
        expanded = expanded.unsqueeze(-1)
    try:
        expanded = expanded.expand_as(value)
    except RuntimeError as exc:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} cannot supervise value shape {tuple(value.shape)}"
        ) from exc
    flat_value = value.reshape(value.shape[0], -1)
    flat_mask = expanded.reshape(value.shape[0], -1)
    valid_sequences = flat_mask.any(dim=-1)
    per_sequence = flat_value.masked_fill(~flat_mask, float("-inf")).max(dim=-1).values
    per_sequence = torch.where(valid_sequences, per_sequence, torch.zeros_like(per_sequence))
    return (
        per_sequence * valid_sequences.to(per_sequence)
    ).sum() / valid_sequences.sum().clamp_min(1)


def _masked_tail_pairs(
    raw_value: torch.Tensor,
    normalized_value: torch.Tensor,
    mask: torch.Tensor,
    fraction: float,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """聚合逐序列 top-k 与最大值，两个分支必须使用相同的有效掩码。"""

    if raw_value.shape != normalized_value.shape:
        raise ValueError(
            "tail raw/normalized values must have matching shapes, got "
            f"{raw_value.shape}/{normalized_value.shape}"
        )
    if not math.isfinite(float(fraction)) or not (0.0 < float(fraction) <= 1.0):
        raise ValueError("tail fraction must be finite and in (0, 1]")
    return {
        "topk": (
            _masked_topk_mean(raw_value, mask, fraction),
            _masked_topk_mean(normalized_value, mask, fraction),
        ),
        "max": (
            _masked_max_mean(raw_value, mask),
            _masked_max_mean(normalized_value, mask),
        ),
    }


def nonnegative_tail_loss_values(
    value: torch.Tensor,
    mask: torch.Tensor,
    *,
    scale: float,
    fraction: float,
    smooth_l1: bool,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """把非负物理量转换为有量纲/归一化的逐序列长尾损失。"""

    if not math.isfinite(float(scale)) or float(scale) <= 0.0:
        raise ValueError("tail scale must be finite and positive")
    if smooth_l1:
        raw_value = F.smooth_l1_loss(
            value,
            torch.zeros_like(value),
            beta=1.0,
            reduction="none",
        )
        normalized_value = F.smooth_l1_loss(
            value / float(scale),
            torch.zeros_like(value),
            beta=1.0,
            reduction="none",
        )
    else:
        raw_value = value
        normalized_value = value / float(scale)
    return _masked_tail_pairs(raw_value, normalized_value, mask, fraction)


def joint_limit_loss_values(
    pred_joint: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    valid: torch.Tensor,
    *,
    margin_rad: float,
    topk_fraction: float,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """同时计算平均越限、安全边距、逐序列 top-k 和逐序列最大越限损失。"""

    if not math.isfinite(float(margin_rad)) or float(margin_rad) < 0.0:
        raise ValueError("joint limit margin must be finite and non-negative")
    if not math.isfinite(float(topk_fraction)) or not (0.0 < float(topk_fraction) <= 1.0):
        raise ValueError("joint limit top-k fraction must be in (0, 1]")
    scale = BUMI_PHYSICAL_V2_SCALES["joint_limit"]
    violation = F.relu(lower - pred_joint) + F.relu(pred_joint - upper)
    margin_violation = F.relu(lower + float(margin_rad) - pred_joint) + F.relu(
        pred_joint - (upper - float(margin_rad))
    )
    violation_normalized = F.smooth_l1_loss(
        violation / scale,
        torch.zeros_like(violation),
        reduction="none",
    )
    margin_normalized = F.smooth_l1_loss(
        margin_violation / scale,
        torch.zeros_like(margin_violation),
        reduction="none",
    )
    return {
        "joint_limit": (
            _masked_mean(violation, valid),
            _masked_mean(violation_normalized, valid),
        ),
        "joint_limit_margin": (
            _masked_mean(margin_violation, valid),
            _masked_mean(margin_normalized, valid),
        ),
        "joint_limit_margin_topk": (
            _masked_topk_mean(margin_violation, valid, topk_fraction),
            _masked_topk_mean(margin_normalized, valid, topk_fraction),
        ),
        "joint_limit_topk": (
            _masked_topk_mean(violation, valid, topk_fraction),
            _masked_topk_mean(violation_normalized, valid, topk_fraction),
        ),
        "joint_limit_max": (
            _masked_max_mean(violation, valid),
            _masked_max_mean(violation_normalized, valid),
        ),
    }


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


def finite_difference(value: torch.Tensor, order: int, fps: int) -> torch.Tensor:
    """沿时间轴计算一至三阶、带真实秒单位的前向有限差分。"""

    if int(order) not in (1, 2, 3):
        raise ValueError("finite difference order must be 1, 2 or 3")
    if int(fps) <= 0:
        raise ValueError("finite difference fps must be positive")
    return torch.diff(value, n=int(order), dim=1) * (float(fps) ** int(order))


def so3_angular_velocity(rotation: torch.Tensor, fps: int) -> torch.Tensor:
    """用相邻姿态的 SO(3) 对数映射计算局部角速度向量。"""

    if rotation.ndim < 4 or rotation.shape[-2:] != (3, 3):
        raise ValueError("rotation must contain a time axis and end in [3,3]")
    if int(fps) <= 0:
        raise ValueError("angular velocity fps must be positive")
    relative = rotation[:, :-1].transpose(-1, -2) @ rotation[:, 1:]
    return matrix_to_axis_angle(relative) * float(fps)


def derivative_excess_loss_values(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """只惩罚预测导数幅值超过同帧 GT 幅值的部分。"""

    if prediction.shape != target.shape:
        raise ValueError(
            "derivative excess inputs must have matching shapes, got "
            f"{prediction.shape}/{target.shape}"
        )
    if not torch.isfinite(torch.tensor(float(scale))) or float(scale) <= 0.0:
        raise ValueError(f"derivative excess scale must be positive and finite, got {scale}")
    excess = F.relu(prediction.abs() - target.abs())
    zero = torch.zeros_like(excess)
    raw = _masked_mean(F.smooth_l1_loss(excess, zero, beta=1.0, reduction="none"), mask)
    normalized = _masked_mean(
        F.smooth_l1_loss(excess / float(scale), zero, beta=1.0, reduction="none"),
        mask,
    )
    return raw, normalized


def derivative_excess_tail_loss_values(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    scale: float,
    fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """保留预测导数超出 GT 幅值最严重的一小部分关节帧。"""

    if prediction.shape != target.shape:
        raise ValueError(
            "derivative excess tail inputs must have matching shapes, got "
            f"{prediction.shape}/{target.shape}"
        )
    excess = F.relu(prediction.abs() - target.abs())
    values = nonnegative_tail_loss_values(
        excess,
        mask,
        scale=scale,
        fraction=fraction,
        smooth_l1=True,
    )
    return values["topk"]


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


def root_tilt_components(
    pred_rotation: torch.Tensor,
    target_rotation: torch.Tensor,
    *,
    upright_allowance_rad: float = 0.35,
    target_margin_rad: float = 0.10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回逐轴 roll/pitch 误差与超出 GT 包络的根倾角。"""

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
    return direction_error, excessive_tilt


def root_tilt_loss_values(
    pred_rotation: torch.Tensor,
    target_rotation: torch.Tensor,
    valid: torch.Tensor,
    *,
    upright_allowance_rad: float = 0.35,
    target_margin_rad: float = 0.10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 roll/pitch 定向误差和异常大倾角安全项。"""

    direction_error, excessive_tilt = root_tilt_components(
        pred_rotation,
        target_rotation,
        upright_allowance_rad=upright_allowance_rad,
        target_margin_rad=target_margin_rad,
    )
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
        joint_limit_margin_rad: float = 0.0,
        joint_limit_topk_fraction: float = 0.01,
        robust_joint_limit_start_step: int = 0,
        robust_joint_limit_warmup_steps: int = 0,
        advanced_physics_start_step: int = 0,
        advanced_physics_warmup_steps: int = 0,
        advanced_physics_topk_fraction: float = 0.05,
        root_tilt_upright_allowance_rad: float = 0.35,
        root_tilt_target_margin_rad: float = 0.10,
    ) -> None:
        super().__init__()
        self.endecoder = endecoder
        self.kinematics = endecoder.kinematics
        self.fps = int(fps)
        self.contract_version = str(contract_version)
        self.auxiliary_warmup_steps = int(auxiliary_warmup_steps)
        self.ground_semantics = ground_semantics
        self.joint_limit_margin_rad = float(joint_limit_margin_rad)
        self.joint_limit_topk_fraction = float(joint_limit_topk_fraction)
        self.robust_joint_limit_start_step = int(robust_joint_limit_start_step)
        self.robust_joint_limit_warmup_steps = int(robust_joint_limit_warmup_steps)
        self.advanced_physics_start_step = int(advanced_physics_start_step)
        self.advanced_physics_warmup_steps = int(advanced_physics_warmup_steps)
        self.advanced_physics_topk_fraction = float(advanced_physics_topk_fraction)
        self.root_tilt_upright_allowance_rad = float(root_tilt_upright_allowance_rad)
        self.root_tilt_target_margin_rad = float(root_tilt_target_margin_rad)
        if self.fps != 30:
            raise ValueError(f"BUMI losses require 30 FPS, got {fps}")
        if self.contract_version not in BUMI_LOSS_NAMES_BY_CONTRACT:
            raise ValueError(
                "qpos30 BUMI only supports loss contracts "
                f"{sorted(BUMI_LOSS_NAMES_BY_CONTRACT)!r}, got {self.contract_version!r}"
            )
        if self.ground_semantics not in {
            "gmr_foot_sole_ground_zero_v1",
            "robot_retargeter_floor_zero_v1",
            "legacy_body_origin_min_zero",
            "mixed_floor_zero_fk_contact_v2",
        }:
            raise ValueError(
                "qpos30 contact loss requires an explicit FK-ground contract; "
                f"got {self.ground_semantics!r}"
            )
        if self.auxiliary_warmup_steps < 0:
            raise ValueError("auxiliary_warmup_steps must be non-negative")
        if self.robust_joint_limit_start_step < 0:
            raise ValueError("robust_joint_limit_start_step must be non-negative")
        if self.robust_joint_limit_warmup_steps < 0:
            raise ValueError("robust_joint_limit_warmup_steps must be non-negative")
        if self.advanced_physics_start_step < 0:
            raise ValueError("advanced_physics_start_step must be non-negative")
        if self.advanced_physics_warmup_steps < 0:
            raise ValueError("advanced_physics_warmup_steps must be non-negative")
        if self.contract_version in {BUMI_LOSS_CONTRACT_V4, BUMI_LOSS_CONTRACT_V5}:
            if not math.isfinite(self.joint_limit_margin_rad) or self.joint_limit_margin_rad <= 0.0:
                raise ValueError("v4/v5 joint_limit_margin_rad must be finite and positive")
            if not math.isfinite(self.joint_limit_topk_fraction) or not (
                0.0 < self.joint_limit_topk_fraction <= 1.0
            ):
                raise ValueError("v4/v5 joint_limit_topk_fraction must be in (0, 1]")
            joint_width = self.kinematics.joint_upper_limits - self.kinematics.joint_lower_limits
            if bool((joint_width <= 2.0 * self.joint_limit_margin_rad).any()):
                raise ValueError("v4/v5 joint limit margin must leave a non-empty safe interval")
        if self.contract_version == BUMI_LOSS_CONTRACT_V5:
            if not math.isfinite(self.advanced_physics_topk_fraction) or not (
                0.0 < self.advanced_physics_topk_fraction <= 1.0
            ):
                raise ValueError("v5 advanced_physics_topk_fraction must be in (0, 1]")
            if (
                not math.isfinite(self.root_tilt_upright_allowance_rad)
                or self.root_tilt_upright_allowance_rad <= 0.0
            ):
                raise ValueError("v5 root tilt upright allowance must be finite and positive")
            if (
                not math.isfinite(self.root_tilt_target_margin_rad)
                or self.root_tilt_target_margin_rad < 0.0
            ):
                raise ValueError("v5 root tilt target margin must be finite and non-negative")
        self.loss_names = BUMI_LOSS_NAMES_BY_CONTRACT[self.contract_version]
        self.weights = {name: float(weights.get(name, 0.0)) for name in self.loss_names}
        unknown = set(weights) - set(self.loss_names)
        if unknown:
            raise ValueError(f"Unknown BUMI loss weights: {sorted(unknown)}")
        if any(
            not torch.isfinite(torch.tensor(value)) or value < 0.0
            for value in self.weights.values()
        ):
            raise ValueError("BUMI loss weights must be finite and non-negative")
        if self.weights["contact_bce"] <= 0.0 or self.weights["foot_slide"] <= 0.0:
            raise ValueError("qpos30 contact contract requires positive contact_bce and foot_slide")
        if self.contract_version in {BUMI_LOSS_CONTRACT_V4, BUMI_LOSS_CONTRACT_V5} and any(
            self.weights[name] <= 0.0 for name in BUMI_ROBUST_JOINT_LIMIT_LOSS_NAMES
        ):
            raise ValueError("v4/v5 require positive margin, top-k and max joint-limit weights")
        if self.contract_version == BUMI_LOSS_CONTRACT_V5 and any(
            self.weights[name] <= 0.0 for name in BUMI_ADVANCED_PHYSICS_LOSS_NAMES
        ):
            raise ValueError("v5 requires every advanced physics weight to be positive")

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
        target_qpos = inputs["target_qpos_canonical"].float()
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
            target_qpos[..., :3],
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
            upright_allowance_rad=self.root_tilt_upright_allowance_rad,
            target_margin_rad=self.root_tilt_target_margin_rad,
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
        temporal_values: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for order, name in (
            (1, "joint_velocity"),
            (2, "joint_acceleration"),
            (3, "joint_jerk"),
        ):
            multiplier = float(self.fps) ** order
            pred_delta = torch.diff(pred_joint, n=order, dim=1) * multiplier
            target_delta = torch.diff(target_joint, n=order, dim=1) * multiplier
            difference_mask = temporal_difference_mask(valid, order)
            raw[name], normalized[name] = self._smooth_l1_pair(
                pred_delta,
                target_delta,
                difference_mask,
                BUMI_PHYSICAL_V2_SCALES[name],
            )
            temporal_values[name] = (pred_delta, target_delta, difference_mask)

        if self.contract_version in {
            BUMI_LOSS_CONTRACT_V3,
            BUMI_LOSS_CONTRACT_V4,
            BUMI_LOSS_CONTRACT_V5,
        }:
            for source_name, excess_name in (
                ("joint_acceleration", "joint_acceleration_excess"),
                ("joint_jerk", "joint_jerk_excess"),
            ):
                pred_delta, target_delta, difference_mask = temporal_values[source_name]
                raw[excess_name], normalized[excess_name] = derivative_excess_loss_values(
                    pred_delta,
                    target_delta,
                    difference_mask,
                    BUMI_PHYSICAL_V2_SCALES[excess_name],
                )

        if self.contract_version == BUMI_LOSS_CONTRACT_V5:
            for order, name in ((1, "root_velocity"), (2, "root_acceleration")):
                difference_mask = temporal_difference_mask(valid, order)
                raw[name], normalized[name] = self._smooth_l1_pair(
                    finite_difference(pred_qpos[..., :3], order, self.fps),
                    finite_difference(target_qpos[..., :3], order, self.fps),
                    difference_mask,
                    BUMI_PHYSICAL_V2_SCALES[name],
                )

            pred_root_angular_velocity = so3_angular_velocity(pred_rotation, self.fps)
            target_root_angular_velocity = so3_angular_velocity(target_rotation, self.fps)
            raw["root_angular_velocity"], normalized["root_angular_velocity"] = (
                self._smooth_l1_pair(
                    pred_root_angular_velocity,
                    target_root_angular_velocity,
                    temporal_difference_mask(valid, 1),
                    BUMI_PHYSICAL_V2_SCALES["root_angular_velocity"],
                )
            )
            raw["root_angular_acceleration"], normalized["root_angular_acceleration"] = (
                self._smooth_l1_pair(
                    finite_difference(pred_root_angular_velocity, 1, self.fps),
                    finite_difference(target_root_angular_velocity, 1, self.fps),
                    temporal_difference_mask(valid, 2),
                    BUMI_PHYSICAL_V2_SCALES["root_angular_acceleration"],
                )
            )

            target_body_root = inputs["target_body_link_pos_root"].float()
            for order, name in ((1, "fk_velocity"), (2, "fk_acceleration")):
                raw[name], normalized[name] = self._smooth_l1_pair(
                    finite_difference(pred_body_root, order, self.fps),
                    finite_difference(target_body_root, order, self.fps),
                    temporal_difference_mask(valid, order),
                    BUMI_PHYSICAL_V2_SCALES[name],
                )

            for source_name, tail_name in (
                ("joint_acceleration", "joint_acceleration_excess_topk"),
                ("joint_jerk", "joint_jerk_excess_topk"),
            ):
                pred_delta, target_delta, difference_mask = temporal_values[source_name]
                raw[tail_name], normalized[tail_name] = derivative_excess_tail_loss_values(
                    pred_delta,
                    target_delta,
                    difference_mask,
                    scale=BUMI_PHYSICAL_V2_SCALES[tail_name],
                    fraction=self.advanced_physics_topk_fraction,
                )

        lower = self.kinematics.joint_lower_limits.to(pred_joint)
        upper = self.kinematics.joint_upper_limits.to(pred_joint)
        if self.contract_version in {BUMI_LOSS_CONTRACT_V4, BUMI_LOSS_CONTRACT_V5}:
            limit_losses = joint_limit_loss_values(
                pred_joint,
                lower,
                upper,
                valid,
                margin_rad=self.joint_limit_margin_rad,
                topk_fraction=self.joint_limit_topk_fraction,
            )
            raw["joint_limit"], normalized["joint_limit"] = limit_losses["joint_limit"]
            for name in BUMI_ROBUST_JOINT_LIMIT_LOSS_NAMES:
                raw[name], normalized[name] = limit_losses[name]
            if self.contract_version == BUMI_LOSS_CONTRACT_V5:
                raw["joint_limit_margin_topk"], normalized["joint_limit_margin_topk"] = (
                    limit_losses["joint_limit_margin_topk"]
                )
        else:
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
        if self.contract_version == BUMI_LOSS_CONTRACT_V5:
            slide_tail = nonnegative_tail_loss_values(
                foot_speed,
                slide_gate,
                scale=BUMI_PHYSICAL_V2_SCALES["foot_slide_topk"],
                fraction=self.advanced_physics_topk_fraction,
                smooth_l1=False,
            )
            raw["foot_slide_topk"], normalized["foot_slide_topk"] = slide_tail["topk"]
            raw["foot_slide_max"], normalized["foot_slide_max"] = slide_tail["max"]

        ground_height_local = inputs["target_contact_ground_height"].to(pred_norm)
        ground_height_local = ground_height_local - self.kinematics.default_qpos[2].to(pred_norm)
        while ground_height_local.ndim < pred_sole["foot_bottom_height"].ndim:
            ground_height_local = ground_height_local.unsqueeze(-1)
        if self.contract_version == BUMI_LOSS_CONTRACT_V5:
            contact_height_error = (pred_sole["foot_bottom_height"] - ground_height_local).abs()
            contact_height_gate = contact_bool & contact_mask & valid[..., None]
            contact_height_raw = F.smooth_l1_loss(
                contact_height_error,
                torch.zeros_like(contact_height_error),
                beta=1.0,
                reduction="none",
            )
            contact_height_normalized = F.smooth_l1_loss(
                contact_height_error / BUMI_PHYSICAL_V2_SCALES["foot_contact_height"],
                torch.zeros_like(contact_height_error),
                beta=1.0,
                reduction="none",
            )
            raw["foot_contact_height"] = _masked_mean(contact_height_raw, contact_height_gate)
            normalized["foot_contact_height"] = _masked_mean(
                contact_height_normalized, contact_height_gate
            )
            contact_height_tail = _masked_tail_pairs(
                contact_height_raw,
                contact_height_normalized,
                contact_height_gate,
                self.advanced_physics_topk_fraction,
            )
            (
                raw["foot_contact_height_topk"],
                normalized["foot_contact_height_topk"],
            ) = contact_height_tail["topk"]
        penetration = F.relu(ground_height_local - pred_sole["foot_bottom_height"])
        raw["penetration"] = _masked_mean(penetration, valid)
        normalized["penetration"] = raw["penetration"] / BUMI_PHYSICAL_V2_SCALES["penetration"]
        if self.contract_version == BUMI_LOSS_CONTRACT_V5:
            penetration_tail = nonnegative_tail_loss_values(
                penetration,
                valid,
                scale=BUMI_PHYSICAL_V2_SCALES["penetration_topk"],
                fraction=self.advanced_physics_topk_fraction,
                smooth_l1=False,
            )
            raw["penetration_topk"], normalized["penetration_topk"] = penetration_tail["topk"]
            raw["penetration_max"], normalized["penetration_max"] = penetration_tail["max"]

            _, excessive_tilt = root_tilt_components(
                pred_rotation,
                target_rotation,
                upright_allowance_rad=self.root_tilt_upright_allowance_rad,
                target_margin_rad=self.root_tilt_target_margin_rad,
            )
            tilt_tail = nonnegative_tail_loss_values(
                excessive_tilt,
                valid,
                scale=BUMI_PHYSICAL_V2_SCALES["root_tilt_excess_topk"],
                fraction=self.advanced_physics_topk_fraction,
                smooth_l1=False,
            )
            raw["root_tilt_excess_topk"], normalized["root_tilt_excess_topk"] = tilt_tail["topk"]
            raw["root_tilt_excess_max"], normalized["root_tilt_excess_max"] = tilt_tail["max"]
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
        if self.robust_joint_limit_warmup_steps <= 0:
            robust_joint_limit_warmup = 1.0
        else:
            robust_joint_limit_warmup = min(
                max(float(global_step - self.robust_joint_limit_start_step), 0.0)
                / self.robust_joint_limit_warmup_steps,
                1.0,
            )
        if self.advanced_physics_warmup_steps <= 0:
            advanced_physics_warmup = 1.0
        else:
            advanced_physics_warmup = min(
                max(float(global_step - self.advanced_physics_start_step), 0.0)
                / self.advanced_physics_warmup_steps,
                1.0,
            )
        always_on = {
            "repr_root_pos",
            "repr_root_rot",
            "repr_joint",
            "root_rot",
            "root_tilt",
            "contact_bce",
        }
        total = pred_norm.new_zeros(())
        output: dict[str, torch.Tensor] = {
            "auxiliary_warmup_factor": pred_norm.new_tensor(warmup),
            "robust_joint_limit_warmup_factor": pred_norm.new_tensor(robust_joint_limit_warmup),
            "advanced_physics_warmup_factor": pred_norm.new_tensor(advanced_physics_warmup),
        }
        for name in self.loss_names:
            if name in BUMI_ADVANCED_PHYSICS_LOSS_NAMES:
                factor = advanced_physics_warmup
            elif name in BUMI_ROBUST_JOINT_LIMIT_LOSS_NAMES:
                factor = robust_joint_limit_warmup
            else:
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
    "BUMI_ADVANCED_PHYSICS_LOSS_NAMES",
    "BUMI_LOSS_CONTRACT_VERSION",
    "BUMI_LOSS_CONTRACT_V3",
    "BUMI_LOSS_CONTRACT_V4",
    "BUMI_LOSS_CONTRACT_V5",
    "BUMI_EXCESS_LOSS_NAMES",
    "BUMI_LOSS_NAMES",
    "BUMI_LOSS_NAMES_BY_CONTRACT",
    "BUMI_PHYSICAL_V2_SCALES",
    "BUMI_ROBUST_JOINT_LIMIT_LOSS_NAMES",
    "BumiRobotLosses",
    "derivative_excess_loss_values",
    "derivative_excess_tail_loss_values",
    "finite_difference",
    "joint_limit_loss_values",
    "nonnegative_tail_loss_values",
    "root_tilt_components",
    "root_tilt_loss_values",
    "so3_angular_velocity",
    "so3_geodesic_angle",
    "temporal_difference_mask",
]
