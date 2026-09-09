"""SMPL physics_v4 的接触支撑、平脚和异常时序损失。

本模块不增加网络输出、不修改数据文件，也不使用模型预测的接触概率作为惩罚开关。
标签只由已有真实动作足底代理点、可信地面和连续有效帧构造：贴地且竖直稳定的点用于
离地约束，水平也稳定的点用于脚滑约束，四个点都稳定且高度差足够小的脚用于平脚约束。
因此正常滑步不被强行锁住，踮脚时不会把已经抬起的脚跟当作接地点压下去。

异常加速度/jerk 在真实动作幅度及绝对容许量之外惩罚额外尖峰，保留旧的 GT 匹配损失。
各项在 FP32 下计算非负 SmoothL1，同时记录有效均值、最差部分均值、有效/激活计数。
最差部分只统计有效位置，padding 和无效地面不充当零误差样本。权重与容差是可配置的
候选值，代码/单元测试通过不代表已经校准动作质量或保证机器人动力学可行性。
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from gem.pipeline.smpl_physics_losses import derivative_valid_mask, finite_difference
from gem.utils.ground_sidecar import SOLE_V437_INDICES


def canonical_sole_vertices(vertices, translation, camera_to_gravity, root_position):
    """将已有相机空间 v437 足底点转到与旧 physics 相同的 Y-up、零起点根轨迹坐标。"""
    if vertices.ndim != 4 or vertices.shape[-2:] != (437, 3):
        raise ValueError("support losses require gt_c_verts437 [B,T,437,3]")
    indices = torch.as_tensor(SOLE_V437_INDICES, device=vertices.device)
    relative = vertices.float().index_select(-2, indices) - translation.float().unsqueeze(-2)
    return torch.einsum(
        "...ij,...vj->...vi", camera_to_gravity.float(), relative
    ) + root_position.float().unsqueeze(-2)


def _number(config: Any, name: str, *, positive: bool = False) -> float:
    value = float(config[name])
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(
            f"support loss {name} must be finite and {'positive' if positive else 'nonnegative'}"
        )
    return value


def validate_support_config(config: Any) -> None:
    """校验新配置，拒绝无效比例、单位尺度或悄悄退化为零阈值。"""
    if config.get("contract_version") != "genmo.smpl_support_losses.v1":
        raise ValueError("unsupported SMPL support loss contract")
    if "warmup_start_step" in config:
        _number(config, "warmup_start_step")
    for name in (
        "warmup_steps",
        "contact_height_m",
        "contact_vertical_speed_mps",
        "planted_speed_mps",
        "flat_gt_span_m",
    ):
        _number(config, name, positive=True)
    for name in ("slide_margin_mps", "height_margin_m", "flat_margin_m", "flat_tolerance_m"):
        _number(config, name)
    fraction = _number(config, "topk_fraction", positive=True)
    if fraction > 1:
        raise ValueError("topk_fraction must be in (0,1]")
    for name in (
        "foot_slide",
        "foot_contact_height",
        "foot_flat_support",
        "root_acceleration_excess",
        "root_jerk_excess",
        "joint_angular_acceleration_excess",
        "joint_angular_jerk_excess",
        "fk_acceleration_excess",
        "fk_jerk_excess",
    ):
        term = config[name]
        _number(term, "scale", positive=True)
        _number(term, "weight")
        _number(term, "topk_weight")
        if name.endswith("_excess"):
            _number(term, "allowance", positive=True)
            _number(term, "target_margin")
            if _number(term, "target_multiplier", positive=True) < 1:
                raise ValueError("target_multiplier must be >= 1 to preserve valid target dynamics")


def nonnegative_penalty(error, mask, scale: float, topk_fraction: float):
    """返回有效均值、有效最差比例均值、原单位均值；空 mask 返回可反传的零。

    topk 先选静态最大容量，再用设备端有效计数截取真正比例，无需每项调用 .item()
    同步 GPU。误差均非负，因此无效位置填零不改变尾部正误差或全零情况的结果。
    """
    mask = mask.bool().expand_as(error)
    value = torch.where(mask, error, torch.zeros_like(error))
    normalized = F.smooth_l1_loss(value / scale, torch.zeros_like(value), reduction="none")
    count = mask.sum()
    denominator = count.to(error.dtype).clamp_min(1)
    mean = normalized.sum() / denominator
    raw = value.sum() / denominator
    if value.numel() == 0:
        return mean, mean, raw
    capacity = max(1, math.ceil(value.numel() * topk_fraction))
    tail = torch.topk(normalized.flatten(), capacity, sorted=True).values
    tail_count = torch.ceil(count.to(error.dtype) * topk_fraction)
    ranks = torch.arange(capacity, device=error.device)
    tail_mean = torch.where(
        ranks < tail_count, tail, torch.zeros_like(tail)
    ).sum() / tail_count.clamp_min(1)
    return mean, tail_mean, raw


@torch.no_grad()
def support_masks(gt_sole, ground_y_local, ground_valid, frame_valid, *, fps, config):
    """仅用 GT 构造逐点接触/稳定支撑及逐脚平放标签；至少两帧连续接触。"""
    if gt_sole.ndim != 4 or gt_sole.shape[-2:] != (8, 3):
        raise ValueError("sole points must have shape [B,T,8,3]")
    valid = frame_valid.bool() & ground_valid.bool().reshape(-1, 1)
    valid = valid & torch.isfinite(gt_sole).all(dim=(-1, -2))
    valid = valid & torch.isfinite(ground_y_local).reshape(-1, 1)
    clean = torch.where(valid[..., None, None], gt_sole.detach(), torch.zeros_like(gt_sole))
    height = clean[..., 1] - ground_y_local.reshape(-1, 1, 1)
    if gt_sole.shape[1] < 2:
        contact = torch.zeros_like(height, dtype=torch.bool)
        return contact, contact, contact.reshape(*contact.shape[:2], 2, 4).all(-1)
    velocity = finite_difference(clean, 1, fps)
    interval_valid = valid[:, :-1] & valid[:, 1:]
    horizontal = torch.linalg.vector_norm(velocity[..., [0, 2]], dim=-1)
    vertical = velocity[..., 1].abs()

    def frame_maximum(interval):
        left = torch.cat((interval[:, :1], interval), dim=1)
        right = torch.cat((interval, interval[:, -1:]), dim=1)
        return torch.maximum(left, right)

    adjacent_valid = torch.cat((interval_valid[:, :1], interval_valid), dim=1)
    adjacent_valid = adjacent_valid & torch.cat((interval_valid, interval_valid[:, -1:]), dim=1)
    contact = valid[..., None] & adjacent_valid[..., None]
    contact = contact & (height.abs() <= float(config["contact_height_m"]))
    contact = contact & (frame_maximum(vertical) <= float(config["contact_vertical_speed_mps"]))
    # 单帧贴地脉冲不是可靠支撑；不跨 padding 或中断帧连接接触段。
    pairs = contact[:, :-1] & contact[:, 1:]
    empty = torch.zeros_like(contact[:, :1])
    contact = contact & (torch.cat((empty, pairs), 1) | torch.cat((pairs, empty), 1))
    planted = contact & (frame_maximum(horizontal) <= float(config["planted_speed_mps"]))
    foot_height = height.reshape(*height.shape[:2], 2, 4)
    gt_span = foot_height.amax(-1) - foot_height.amin(-1)
    flat = planted.reshape(*planted.shape[:2], 2, 4).all(-1)
    flat = flat & (gt_span <= float(config["flat_gt_span_m"]))
    return contact, planted, flat


def derivative_excess(pred, target, term):
    """只罚超出 GT 幅度及绝对容许量的额外变化，不把正常踢腿/快速动作压向零。"""
    target_size = torch.linalg.vector_norm(target.detach(), dim=-1)
    permitted = (
        target_size * float(term["target_multiplier"]) + float(term["target_margin"])
    ).clamp_min(float(term["allowance"]))
    return F.relu(torch.linalg.vector_norm(pred, dim=-1) - permitted)


def compute_smpl_support_losses(
    *,
    pred_sole,
    gt_sole,
    ground_y_local,
    ground_valid,
    frame_valid,
    pred_root,
    gt_root,
    pred_fk,
    gt_fk,
    pred_joint_velocity,
    gt_joint_velocity,
    root_accepted,
    fk_accepted,
    joint_accepted,
    config,
    global_step,
    fps,
):
    """计算可选 v4 九项损失；复用旧 physics 的 FK/角速度和 GT 异常区间 mask。"""
    validate_support_config(config)
    if not math.isfinite(float(fps)) or fps <= 0:
        raise ValueError("support fps must be finite and positive")
    with torch.autocast(device_type=pred_sole.device.type, enabled=False):
        pred_sole = pred_sole.float()
        gt_sole = gt_sole.detach().float()
        # 完整 resume 保留旧 global_step；仅新增损失按本阶段起点升权，旧配置默认从零起算。
        elapsed_steps = float(global_step) - float(config.get("warmup_start_step", 0))
        ramp = min(max(elapsed_steps, 0) / float(config["warmup_steps"]), 1.0)
        fraction = float(config["topk_fraction"])
        contact, planted, flat = support_masks(
            gt_sole, ground_y_local, ground_valid, frame_valid, fps=fps, config=config
        )
        # 无效地面/填充帧在求导数和向量范数前清零，避免被 mask 的 NaN 污染反传。
        sole_valid = frame_valid.bool() & ground_valid.bool().reshape(-1, 1)
        sole_valid = sole_valid & torch.isfinite(gt_sole).all(dim=(-1, -2))
        sole_valid = sole_valid & torch.isfinite(ground_y_local).reshape(-1, 1)
        pred_sole = torch.where(sole_valid[..., None, None], pred_sole, torch.zeros_like(pred_sole))
        gt_sole = torch.where(sole_valid[..., None, None], gt_sole, torch.zeros_like(gt_sole))
        total = pred_sole[:, :0].sum()
        logs = {"physics_v4_weight_ramp_metric": pred_sole.new_tensor(ramp)}

        def add(name, error, mask):
            nonlocal total
            term = config[name]
            mean, tail, raw = nonnegative_penalty(error, mask, float(term["scale"]), fraction)
            weighted = ramp * (float(term["weight"]) * mean + float(term["topk_weight"]) * tail)
            total = total + weighted
            prefix = "physics_v4_" + name
            logs[prefix + "_raw_loss"] = raw.detach()
            logs[prefix + "_normalized_loss"] = mean.detach()
            logs[prefix + "_topk_normalized_loss"] = tail.detach()
            logs[prefix + "_weighted_loss"] = weighted.detach()
            logs[prefix + "_candidate_count_metric"] = mask.sum().float()
            logs[prefix + "_active_count_metric"] = (mask & (error.detach() > 0)).sum().float()

        # 高度只约束本来应接地的点，容许 GT 自身的微小足弓高度，不压正常翘起的脚跟。
        ground = ground_y_local.to(pred_sole)
        ground = torch.where(torch.isfinite(ground), ground, torch.zeros_like(ground)).reshape(
            -1, 1, 1
        )
        gt_height = (gt_sole[..., 1] - ground).clamp_min(0)
        height_error = F.relu(
            pred_sole[..., 1] - ground - gt_height - float(config["height_margin_m"])
        )
        add("foot_contact_height", height_error, contact)

        # 支撑点允许 GT 本来的小幅移动；不约束有意滑步的非稳定点，也不约束脚的 yaw。
        pred_velocity = finite_difference(pred_sole, 1, fps)
        target_velocity = finite_difference(gt_sole, 1, fps)
        slide = torch.linalg.vector_norm((pred_velocity - target_velocity)[..., [0, 2]], dim=-1)
        slide = F.relu(slide - float(config["slide_margin_mps"]))
        add("foot_slide", slide, planted[:, :-1] & planted[:, 1:])

        pred_y = pred_sole[..., 1].reshape(*pred_sole.shape[:2], 2, 4)
        gt_y = gt_sole[..., 1].reshape(*gt_sole.shape[:2], 2, 4)
        pred_span = pred_y.amax(-1) - pred_y.amin(-1)
        gt_span = gt_y.amax(-1) - gt_y.amin(-1)
        permitted_span = gt_span.clamp_min(float(config["flat_tolerance_m"])) + float(
            config["flat_margin_m"]
        )
        add("foot_flat_support", F.relu(pred_span - permitted_span), flat)

        for label, pred, target, accepted in (
            ("root", pred_root, gt_root, root_accepted),
            ("fk", pred_fk, gt_fk, fk_accepted),
            ("joint_angular", pred_joint_velocity, gt_joint_velocity, joint_accepted),
        ):
            valid_values = frame_valid.bool()
            if label == "joint_angular":
                valid_values = valid_values[:, :-1] & valid_values[:, 1:]
            while valid_values.ndim < pred.ndim:
                valid_values = valid_values.unsqueeze(-1)
            pred = torch.where(
                valid_values, pred.float(), torch.zeros_like(pred, dtype=torch.float32)
            )
            target = torch.where(
                valid_values, target.detach().float(), torch.zeros_like(target, dtype=torch.float32)
            )
            for order, suffix in ((2, "acceleration"), (3, "jerk")):
                name = f"{label}_{suffix}_excess"
                difference_order = order - 1 if label == "joint_angular" else order
                predicted = finite_difference(pred.float(), difference_order, fps)
                expected = finite_difference(target.detach().float(), difference_order, fps)
                # 极短片段没有完整差分区间，不对共享旧 mask 工具调用过长的 unfold。
                mask = (
                    derivative_valid_mask(frame_valid, order, accepted)
                    if frame_valid.shape[1] > order
                    else accepted[:, :0]
                )
                error = derivative_excess(predicted, expected, config[name])
                add(name, error, mask)
        logs["physics_v4_ground_valid_count_metric"] = (
            (frame_valid & ground_valid.bool().reshape(-1, 1)).sum().float()
        )
        logs["physics_v4_frame_count_metric"] = frame_valid.sum().float()
        logs["physics_v4_total_loss"] = total
        return total, logs
