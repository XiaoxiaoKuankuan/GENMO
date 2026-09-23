"""Stage 1 条件扩散的 qpos30/contact2 逐坐标监督与 BUMI v5 物理损失。

本模块只适配第 3 步已有 batch，不创建第二套监督字段，也不重新编码动作或改变 anchor。
网络输出处于现有 BumiEndecoder 的 normalized x0 空间；主重建只对有效未知坐标求均值。
物理项先把已知 physical qpos30 和生成后续逐坐标合成，再用原 decode、积分和可微 FK
还原完整 120 帧。时序导数保留已知前缀到未知后续的支持点，禁止把前缀边界速度置零。

旧 BumiRobotLosses.forward 使用整帧 mask，不能直接用于未知逐坐标重建。本实现继承其
严格 v5 权重/配置校验，并复用 SO(3)、有限差分、限位、长尾和物理尺度公式；所有归约
在这里明确传入适用的坐标或支持点 mask。无效 target、padding 和未知条件占位在算术、
rotation/FK 之前均替换为有限的中性值，末帧无 halo 的 root XY 不进入任何监督。

contact 仅使用独立标签及有效 mask，不进入前缀条件。本路径的地面必须明确为 floor-zero；
旧 legacy_body_origin_min_zero 需要未裁剪序列的地面估计，本 batch 未携带该量，因此拒绝
该配置，不从当前验证窗口伪造地面。这里全部为运动学监督，不包含 GMT、Critic 或动力学。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from gem.closedloop.contracts import validate_stage1_training_batch
from gem.robots.bumi.endecoder import BumiEndecoder
from gem.robots.bumi.losses import (
    BUMI_ADVANCED_PHYSICS_LOSS_NAMES,
    BUMI_PHYSICAL_SCALES,
    BUMI_ROBUST_JOINT_LIMIT_LOSS_NAMES,
    BumiRobotLosses,
    _masked_mean,
    _masked_tail_pairs,
    derivative_excess_loss_values,
    derivative_excess_tail_loss_values,
    finite_difference,
    joint_limit_loss_values,
    nonnegative_tail_loss_values,
    root_tilt_components,
    root_tilt_loss_values,
    so3_angular_velocity,
    so3_geodesic_angle,
)
from gem.utils.rotation_conversions import rotation_6d_to_matrix

STAGE1_LOSS_CONTRACT_VERSION = "genmo.bumi_closedloop.masked_physical_qpos30_contact_v1"


def coordinate_difference_mask(valid: torch.Tensor, order: int) -> torch.Tensor:
    """任意尾部维度的逐坐标差分必须拥有全部 N+1 个有效支持点。"""
    if valid.ndim < 2 or order not in (1, 2, 3):
        raise ValueError("difference mask requires [B,T,...] and order in (1,2,3)")
    length = valid.shape[1] - order
    if length <= 0:
        return valid[:, :0]
    result = valid[:, :length].clone()
    for offset in range(1, order + 1):
        result &= valid[:, offset : offset + length]
    return result


class Stage1BumiLosses(BumiRobotLosses):
    """复用全部 v5 loss 项，但独立实现条件扩散的有效性与逐坐标归约。"""

    def __init__(
        self,
        endecoder: BumiEndecoder,
        weights: Mapping[str, float],
        *,
        ground_semantics: str = "mixed_floor_zero_fk_contact_v2",
        **kwargs: Any,
    ) -> None:
        if ground_semantics == "legacy_body_origin_min_zero":
            raise ValueError("Stage1 losses require explicit floor-zero ground semantics")
        super().__init__(endecoder, weights, ground_semantics=ground_semantics, **kwargs)
        self.stage1_contract_version = STAGE1_LOSS_CONTRACT_VERSION

    def forward(
        self,
        batch: Mapping[str, Any],
        pred_x_start: torch.Tensor,
        contact_logits: torch.Tensor,
        global_step: int = 0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """返回可反传总损失和包含 raw/normalized/weighted 的独立诊断字典。"""
        validate_stage1_training_batch(batch, history_steps=int(batch["proprio_history"].shape[1]))
        if pred_x_start.shape != batch["target_qpos30"].shape:
            raise ValueError("pred_x_start must match target_qpos30 [B,120,30]")
        if contact_logits.shape != batch["target_contact"].shape:
            raise ValueError("contact_logits must match target_contact [B,120,2]")
        # 原物理公式也在 FP32 中运行，避免低精度 SO(3)/差分/FK 放大误差。
        with torch.autocast(device_type=pred_x_start.device.type, enabled=False):
            return self._forward_float(
                batch, pred_x_start.float(), contact_logits.float(), global_step
            )

    def _forward_float(
        self, batch: Mapping[str, Any], pred: torch.Tensor, logits: torch.Tensor, step: int
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        valid = batch["target_qpos30_valid"]
        known_mask = batch["known_qpos30_mask"]
        unknown = valid & ~known_mask
        # physical 零不是 normalized 零；且零 rot6d 会产生无定义姿态，因此显式使用单位姿态。
        neutral = pred.new_zeros(30)
        neutral[3] = neutral[7] = 1.0
        neutral[9:] = self.kinematics.default_qpos[7:].to(pred)
        target_physical = torch.where(valid, batch["target_qpos30"].float(), neutral)
        known_physical = torch.where(known_mask, batch["known_qpos30"].float(), neutral)
        target_norm = self.endecoder.normalize(target_physical)
        known_norm = self.endecoder.normalize(known_physical)
        neutral_norm = self.endecoder.normalize(neutral)
        combined_norm = torch.where(known_mask, known_norm, pred)
        combined_norm = torch.where(valid, combined_norm, neutral_norm)
        # 一次共享时间轴的 decode/integration，绝不分别编码前缀与未来。
        decoded = self.endecoder.decode(combined_norm)
        target_decoded = self.endecoder.decode(target_norm)
        pred_qpos = self.endecoder.compose_qpos(decoded)
        target_qpos = self.endecoder.compose_qpos(target_decoded)
        pred_fk = self.kinematics.forward_kinematics(pred_qpos)
        target_fk = self.kinematics.forward_kinematics(target_qpos)
        pred_body = self.endecoder.codec.body_positions_in_root_frame(
            pred_qpos[..., :3], pred_qpos[..., 3:7], pred_fk["body_pos_w"][..., 1:, :]
        )
        target_body = self.endecoder.codec.body_positions_in_root_frame(
            target_qpos[..., :3], target_qpos[..., 3:7], target_fk["body_pos_w"][..., 1:, :]
        )
        # XY[t] 依赖全部较早的 delta 和 heading；Z[t] 只依赖本帧高度。
        rot_valid = valid[..., 3:9].all(-1)
        delta_support = valid[..., :2].all(-1) & rot_valid
        xy_valid = (
            torch.cat((torch.ones_like(delta_support[:, :1]), delta_support[:, :-1]), dim=1)
            .to(torch.int64)
            .cumprod(dim=1)
            .bool()
        )
        xy_valid &= batch["future_valid"]
        root_valid = torch.cat((xy_valid[..., None].expand(-1, -1, 2), valid[..., 2:3]), -1)
        joint_valid = valid[..., 9:]
        body_valid = valid[..., 2:].all(-1)
        world_body_valid = body_valid & xy_valid
        pred_joint, target_joint = decoded["joint_dof"], target_decoded["joint_dof"]
        pred_rot = rotation_6d_to_matrix(decoded["root_rot_local_6d"])
        target_rot = rotation_6d_to_matrix(target_decoded["root_rot_local_6d"])
        raw: dict[str, torch.Tensor] = {}
        normalized: dict[str, torch.Tensor] = {}

        def put(name: str, values: tuple[torch.Tensor, torch.Tensor]) -> None:
            raw[name], normalized[name] = values

        def smooth(
            name: str, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
        ) -> None:
            put(name, self._smooth_l1_pair(prediction, target, mask, BUMI_PHYSICAL_SCALES[name]))

        # 在减法之前屏蔽，不允许未知条件/无效标签占位参与任何主重建计算。
        reconstruction_pred = torch.where(unknown, pred, target_norm)
        reconstruction_error = (reconstruction_pred - target_norm).square()
        for name, part in (
            ("repr_root_pos", slice(0, 3)),
            ("repr_root_rot", slice(3, 9)),
            ("repr_joint", slice(9, 30)),
        ):
            value = _masked_mean(reconstruction_error[..., part], unknown[..., part])
            put(name, (value, value))
        smooth("root_pos", pred_qpos[..., :3], target_qpos[..., :3], root_valid)
        angle = so3_geodesic_angle(pred_rot, target_rot)
        put(
            "root_rot",
            (
                _masked_mean(angle, rot_valid),
                _masked_mean(angle / BUMI_PHYSICAL_SCALES["root_rot"], rot_valid),
            ),
        )
        put(
            "root_tilt",
            root_tilt_loss_values(
                pred_rot,
                target_rot,
                rot_valid,
                upright_allowance_rad=self.root_tilt_upright_allowance_rad,
                target_margin_rad=self.root_tilt_target_margin_rad,
            ),
        )
        smooth("joint_dof", pred_joint, target_joint, joint_valid)
        smooth("fk_body_pos", pred_body, target_body, body_valid)
        smooth(
            "root_height",
            decoded["root_height_offset"],
            target_decoded["root_height_offset"],
            valid[..., 2:3],
        )

        temporal: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for order, name in ((1, "joint_velocity"), (2, "joint_acceleration"), (3, "joint_jerk")):
            p = finite_difference(pred_joint, order, self.fps)
            t = finite_difference(target_joint, order, self.fps)
            support = coordinate_difference_mask(joint_valid, order)
            smooth(name, p, t, support)
            temporal[name] = p, t, support
        for source, name in (
            ("joint_acceleration", "joint_acceleration_excess"),
            ("joint_jerk", "joint_jerk_excess"),
        ):
            p, t, support = temporal[source]
            put(name, derivative_excess_loss_values(p, t, support, BUMI_PHYSICAL_SCALES[name]))
            tail_name = name + "_topk"
            put(
                tail_name,
                derivative_excess_tail_loss_values(
                    p,
                    t,
                    support,
                    scale=BUMI_PHYSICAL_SCALES[tail_name],
                    fraction=self.advanced_physics_topk_fraction,
                ),
            )
        for order, name in ((1, "root_velocity"), (2, "root_acceleration")):
            smooth(
                name,
                finite_difference(pred_qpos[..., :3], order, self.fps),
                finite_difference(target_qpos[..., :3], order, self.fps),
                coordinate_difference_mask(root_valid, order),
            )
        p_angular = so3_angular_velocity(pred_rot, self.fps)
        t_angular = so3_angular_velocity(target_rot, self.fps)
        smooth(
            "root_angular_velocity", p_angular, t_angular, coordinate_difference_mask(rot_valid, 1)
        )
        smooth(
            "root_angular_acceleration",
            finite_difference(p_angular, 1, self.fps),
            finite_difference(t_angular, 1, self.fps),
            coordinate_difference_mask(rot_valid, 2),
        )
        for order, name in ((1, "fk_velocity"), (2, "fk_acceleration")):
            smooth(
                name,
                finite_difference(pred_body, order, self.fps),
                finite_difference(target_body, order, self.fps),
                coordinate_difference_mask(body_valid, order),
            )
        for name, values in joint_limit_loss_values(
            pred_joint,
            self.kinematics.joint_lower_limits.to(pred_joint),
            self.kinematics.joint_upper_limits.to(pred_joint),
            joint_valid,
            margin_rad=self.joint_limit_margin_rad,
            topk_fraction=self.joint_limit_topk_fraction,
        ).items():
            put(name, values)

        contact_valid = batch["target_contact_valid"]
        contact = torch.where(contact_valid, batch["target_contact"].float(), 0.0)
        logits = torch.where(contact_valid, logits, 0.0)
        bce = _masked_mean(
            F.binary_cross_entropy_with_logits(logits, contact, reduction="none"), contact_valid
        )
        put("contact_bce", (bce, bce))
        sole = self.kinematics.aggregate_sole_by_foot(pred_fk["body_pos_w"], pred_fk["body_quat_w"])
        foot_speed = torch.linalg.vector_norm(
            torch.diff(sole["foot_points_w"][..., :2], dim=1) * self.fps, dim=-1
        )
        contact_bool = contact >= 0.5
        slide_gate = (
            contact_bool[:, 1:]
            & contact_bool[:, :-1]
            & coordinate_difference_mask(contact_valid, 1)
            & coordinate_difference_mask(world_body_valid, 1)[..., None]
        )
        slide = _masked_mean(foot_speed, slide_gate)
        put("foot_slide", (slide, slide / BUMI_PHYSICAL_SCALES["foot_slide"]))
        slide_tail = nonnegative_tail_loss_values(
            foot_speed,
            slide_gate,
            scale=BUMI_PHYSICAL_SCALES["foot_slide_topk"],
            fraction=self.advanced_physics_topk_fraction,
            smooth_l1=False,
        )
        put("foot_slide_topk", slide_tail["topk"])
        put("foot_slide_max", slide_tail["max"])
        # codec 的 canonical Z = world Z - default_root_height；统一 floor-zero 的世界地面为 0。
        ground = -self.endecoder.codec.default_root_height.to(pred)
        height_error = (sole["foot_bottom_height"] - ground).abs()
        height_gate = contact_bool & contact_valid & body_valid[..., None]
        height_raw = F.smooth_l1_loss(
            height_error, torch.zeros_like(height_error), reduction="none"
        )
        height_normalized = F.smooth_l1_loss(
            height_error / BUMI_PHYSICAL_SCALES["foot_contact_height"],
            torch.zeros_like(height_error),
            reduction="none",
        )
        put(
            "foot_contact_height",
            (_masked_mean(height_raw, height_gate), _masked_mean(height_normalized, height_gate)),
        )
        put(
            "foot_contact_height_topk",
            _masked_tail_pairs(
                height_raw, height_normalized, height_gate, self.advanced_physics_topk_fraction
            )["topk"],
        )
        penetration = F.relu(ground - sole["foot_bottom_height"])
        p_mean = _masked_mean(penetration, body_valid)
        put("penetration", (p_mean, p_mean / BUMI_PHYSICAL_SCALES["penetration"]))
        p_tail = nonnegative_tail_loss_values(
            penetration,
            body_valid,
            scale=BUMI_PHYSICAL_SCALES["penetration_topk"],
            fraction=self.advanced_physics_topk_fraction,
            smooth_l1=False,
        )
        put("penetration_topk", p_tail["topk"])
        put("penetration_max", p_tail["max"])
        _, tilt = root_tilt_components(
            pred_rot,
            target_rot,
            upright_allowance_rad=self.root_tilt_upright_allowance_rad,
            target_margin_rad=self.root_tilt_target_margin_rad,
        )
        tilt_tail = nonnegative_tail_loss_values(
            tilt,
            rot_valid,
            scale=BUMI_PHYSICAL_SCALES["root_tilt_excess_topk"],
            fraction=self.advanced_physics_topk_fraction,
            smooth_l1=False,
        )
        put("root_tilt_excess_topk", tilt_tail["topk"])
        put("root_tilt_excess_max", tilt_tail["max"])

        def ramp(start: int, warmup: int) -> float:
            return 1.0 if warmup <= 0 else min(max(float(step - start), 0.0) / warmup, 1.0)

        aux = ramp(0, self.auxiliary_warmup_steps)
        robust = ramp(self.robust_joint_limit_start_step, self.robust_joint_limit_warmup_steps)
        advanced = ramp(self.advanced_physics_start_step, self.advanced_physics_warmup_steps)
        always_on = {
            "repr_root_pos",
            "repr_root_rot",
            "repr_joint",
            "root_rot",
            "root_tilt",
            "contact_bce",
        }
        output = {
            "auxiliary_warmup_factor": pred.new_tensor(aux),
            "robust_joint_limit_warmup_factor": pred.new_tensor(robust),
            "advanced_physics_warmup_factor": pred.new_tensor(advanced),
            "unknown_qpos30_elements": unknown.sum().to(pred),
            "contact_supervision_elements": contact_valid.sum().to(pred),
            "raw_reconstruction_loss": _masked_mean(reconstruction_error, unknown),
        }
        total = pred.new_zeros(())
        for name in self.loss_names:
            if name in BUMI_ADVANCED_PHYSICS_LOSS_NAMES:
                factor = advanced
            elif name in BUMI_ROBUST_JOINT_LIMIT_LOSS_NAMES:
                factor = robust
            else:
                factor = 1.0 if name in always_on else aux
            weighted = normalized[name] * (self.weights[name] * factor)
            output[f"raw_{name}_loss"] = raw[name]
            output[f"normalized_{name}_loss"] = normalized[name]
            output[f"weighted_{name}_loss"] = weighted
            total = total + weighted
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("Stage1 BUMI total loss is NaN or Inf")
        output["loss"] = total
        return total, output


__all__ = ["STAGE1_LOSS_CONTRACT_VERSION", "Stage1BumiLosses", "coordinate_difference_mask"]
