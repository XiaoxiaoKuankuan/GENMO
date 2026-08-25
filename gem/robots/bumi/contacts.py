"""基于权威 BUMI FK 生成可靠的左右足底接触标签。

本模块只从 qpos28、固定运动学和真实鞋底代理推导监督，不从网络预测的 link 位置或未标注
的零值猜测接触。标签同时检查足底相对地面的高度与水平速度，并使用进入/退出双阈值形成
迟滞，最后删除过短的单帧脉冲。对于 GMR 足底归零数据，地面严格固定为世界 Z=0；对于
历史 body-origin 归零数据，可以在完整序列上用足底最低分位数估计该序列的等效地面。

输出包含标签、有效掩码、实际使用的地面高度和诊断信号，数据读取、训练 loss 与测试可以
共用同一份契约，避免各处阈值不一致。时间维固定为倒数第二维，足顺序固定为 left/right。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from .kinematics import BumiKinematics

BUMI_CONTACT_CONTRACT_VERSION = "genmo.bumi_foot_contact.fk_sole_hysteresis.v1"


@dataclass(frozen=True)
class BumiFootContactTargets:
    contact: torch.Tensor
    valid_mask: torch.Tensor
    ground_height: torch.Tensor
    foot_speed_xy: torch.Tensor
    foot_bottom_height: torch.Tensor


def _validate_thresholds(
    *,
    enter_height: float,
    exit_height: float,
    enter_speed: float,
    exit_speed: float,
    ground_quantile: float,
    min_contact_frames: int,
) -> None:
    values = (enter_height, exit_height, enter_speed, exit_speed, ground_quantile)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("BUMI contact thresholds must be finite")
    if enter_height < 0.0 or exit_height < enter_height:
        raise ValueError("contact height thresholds require 0 <= enter_height <= exit_height")
    if enter_speed < 0.0 or exit_speed < enter_speed:
        raise ValueError("contact speed thresholds require 0 <= enter_speed <= exit_speed")
    if not 0.0 <= ground_quantile <= 0.25:
        raise ValueError("ground_quantile must be in [0,0.25]")
    if int(min_contact_frames) < 1:
        raise ValueError("min_contact_frames must be positive")


def _forward_foot_speed(foot_xy: torch.Tensor, fps: int) -> torch.Tensor:
    if foot_xy.shape[-3] <= 1:
        return torch.zeros(foot_xy.shape[:-1], dtype=foot_xy.dtype, device=foot_xy.device)
    delta = torch.diff(foot_xy, dim=-3) * float(fps)
    speed = torch.linalg.vector_norm(delta, dim=-1)
    return torch.cat((speed, speed[..., -1:, :]), dim=-2)


def _resolve_ground_height(
    bottom_height: torch.Tensor,
    valid: torch.Tensor,
    ground_height: float | torch.Tensor | None,
    ground_quantile: float,
    estimate_ground_mask: torch.Tensor | None,
) -> torch.Tensor:
    batch_shape = bottom_height.shape[:-2]
    frames = bottom_height.shape[-2]
    flat_height = bottom_height.reshape(-1, frames, 2)
    flat_valid = valid.reshape(-1, frames)
    if ground_height is None:
        resolved = bottom_height.new_zeros(batch_shape)
        estimate = torch.ones(batch_shape, dtype=torch.bool, device=bottom_height.device)
    else:
        resolved = torch.as_tensor(
            ground_height,
            dtype=bottom_height.dtype,
            device=bottom_height.device,
        )
        if not bool(torch.isfinite(resolved).all()):
            raise ValueError("ground_height contains NaN or Inf")
        try:
            resolved = resolved.expand(batch_shape).clone()
        except RuntimeError as exc:
            raise ValueError(
                f"ground_height shape {tuple(resolved.shape)} cannot broadcast to {batch_shape}"
            ) from exc
        estimate = torch.zeros(batch_shape, dtype=torch.bool, device=bottom_height.device)
    if estimate_ground_mask is not None:
        supplied_mask = torch.as_tensor(
            estimate_ground_mask,
            dtype=torch.bool,
            device=bottom_height.device,
        )
        try:
            estimate = supplied_mask.expand(batch_shape)
        except RuntimeError as exc:
            raise ValueError(
                "estimate_ground_mask shape "
                f"{tuple(supplied_mask.shape)} cannot broadcast to {batch_shape}"
            ) from exc

    # 无效帧改成 NaN 后一次性计算每个样本的足底分位数；这与逐样本筛选完全等价，
    # 但不会在 8 卡训练中因 ``bool(cuda_tensor)`` 产生逐样本 host/device 同步。
    masked_height = flat_height.masked_fill(~flat_valid[..., None], torch.nan)
    estimated = torch.nanquantile(
        masked_height.reshape(flat_height.shape[0], -1),
        float(ground_quantile),
        dim=1,
    )
    estimated = torch.nan_to_num(estimated, nan=0.0)
    flat_resolved = torch.where(estimate.reshape(-1), estimated, resolved.reshape(-1))
    return flat_resolved.reshape(batch_shape)


def _apply_hysteresis_and_minimum_run(
    enter: torch.Tensor,
    stay: torch.Tensor,
    valid: torch.Tensor,
    min_contact_frames: int,
) -> torch.Tensor:
    frames = enter.shape[-2]
    flat_enter = enter.reshape(-1, frames, 2)
    flat_stay = stay.reshape(-1, frames, 2)
    flat_valid = valid.reshape(-1, frames)
    active = torch.zeros((flat_enter.shape[0], 2), dtype=torch.bool, device=enter.device)
    timeline: list[torch.Tensor] = []
    # 只沿固定 120 帧时间轴递归；batch 和左右脚完全向量化，避免逐元素读取 CUDA bool
    # 造成数万次 host/device 同步。
    for frame_index in range(frames):
        frame_valid = flat_valid[:, frame_index, None]
        active = frame_valid & torch.where(
            active,
            flat_stay[:, frame_index],
            flat_enter[:, frame_index],
        )
        timeline.append(active)
    flat_output = torch.stack(timeline, dim=1)
    if min_contact_frames <= 1:
        return flat_output.reshape_as(enter)
    if frames < min_contact_frames:
        return torch.zeros_like(enter, dtype=torch.bool)

    sequences = flat_output.permute(0, 2, 1).reshape(-1, frames)
    full_windows = sequences.unfold(1, min_contact_frames, 1).all(dim=-1)
    keep = torch.zeros_like(sequences)
    window_count = full_windows.shape[1]
    for offset in range(min_contact_frames):
        keep[:, offset : offset + window_count] |= full_windows
    filtered = keep.reshape(flat_output.shape[0], 2, frames).permute(0, 2, 1)
    return filtered.reshape_as(enter)


@torch.no_grad()
def derive_bumi_foot_contact(
    qpos: torch.Tensor,
    kinematics: BumiKinematics,
    *,
    valid_mask: torch.Tensor | None = None,
    fps: int = 30,
    ground_height: float | torch.Tensor | None = 0.0,
    estimate_ground_mask: torch.Tensor | None = None,
    ground_quantile: float = 0.02,
    enter_height: float = 0.035,
    exit_height: float = 0.055,
    enter_speed: float = 0.15,
    exit_speed: float = 0.25,
    min_contact_frames: int = 2,
    fk: Mapping[str, torch.Tensor] | None = None,
) -> BumiFootContactTargets:
    """从 ``[...,T,28]`` qpos 生成 left/right FK 足底接触标签。"""

    if not isinstance(kinematics, BumiKinematics):
        raise TypeError("derive_bumi_foot_contact requires BumiKinematics")
    if not isinstance(qpos, torch.Tensor) or qpos.ndim < 2 or qpos.shape[-1] != 28:
        raise ValueError(f"qpos must have shape [...,T,28], got {getattr(qpos, 'shape', None)}")
    if qpos.shape[-2] <= 0 or not bool(torch.isfinite(qpos).all()):
        raise ValueError("qpos must contain at least one finite frame")
    if int(fps) <= 0:
        raise ValueError("fps must be positive")
    _validate_thresholds(
        enter_height=enter_height,
        exit_height=exit_height,
        enter_speed=enter_speed,
        exit_speed=exit_speed,
        ground_quantile=ground_quantile,
        min_contact_frames=min_contact_frames,
    )
    if valid_mask is None:
        valid = torch.ones(qpos.shape[:-1], dtype=torch.bool, device=qpos.device)
    else:
        if not isinstance(valid_mask, torch.Tensor) or tuple(valid_mask.shape) != tuple(
            qpos.shape[:-1]
        ):
            raise ValueError(
                f"valid_mask must have shape {tuple(qpos.shape[:-1])}, "
                f"got {getattr(valid_mask, 'shape', None)}"
            )
        valid = valid_mask.to(device=qpos.device).bool()

    resolved_fk = kinematics.forward_kinematics(qpos) if fk is None else fk
    body_pos = resolved_fk.get("body_pos_w")
    body_quat = resolved_fk.get("body_quat_w")
    expected_prefix = qpos.shape[:-1]
    if not isinstance(body_pos, torch.Tensor) or tuple(body_pos.shape) != (
        *expected_prefix,
        22,
        3,
    ):
        raise ValueError(f"FK body_pos_w does not match qpos: {getattr(body_pos, 'shape', None)}")
    if not isinstance(body_quat, torch.Tensor) or tuple(body_quat.shape) != (
        *expected_prefix,
        22,
        4,
    ):
        raise ValueError(f"FK body_quat_w does not match qpos: {getattr(body_quat, 'shape', None)}")
    sole = kinematics.aggregate_sole_by_foot(body_pos, body_quat)
    bottom_height = sole["foot_bottom_height"]
    speed = _forward_foot_speed(sole["foot_points_w"][..., :2], int(fps))
    resolved_ground = _resolve_ground_height(
        bottom_height,
        valid,
        ground_height,
        float(ground_quantile),
        estimate_ground_mask,
    )
    ground = resolved_ground
    while ground.ndim < bottom_height.ndim:
        ground = ground.unsqueeze(-1)
    relative_height = bottom_height - ground
    enter = (
        (relative_height <= float(enter_height)) & (speed <= float(enter_speed)) & valid[..., None]
    )
    stay = (relative_height <= float(exit_height)) & (speed <= float(exit_speed)) & valid[..., None]
    contact = _apply_hysteresis_and_minimum_run(
        enter,
        stay,
        valid,
        int(min_contact_frames),
    )
    contact_mask = valid[..., None].expand_as(contact)
    return BumiFootContactTargets(
        contact=contact.float(),
        valid_mask=contact_mask,
        ground_height=resolved_ground,
        foot_speed_xy=speed,
        foot_bottom_height=bottom_height,
    )


__all__ = [
    "BUMI_CONTACT_CONTRACT_VERSION",
    "BumiFootContactTargets",
    "derive_bumi_foot_contact",
]
