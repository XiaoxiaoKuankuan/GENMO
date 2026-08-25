"""BUMI 专用的 FK 足底锁定后处理。

该后处理只解决模型已经判定为支撑脚时的水平脚滑：先用 qpos28 和固定运动学得到左右鞋底
世界坐标，再在连续接触区维护足底锚点，最后只修正 floating root 的世界 XY 平移。根 Z、
根四元数和 21 个关节角逐值保持不变，因此它不会通过强制直立、改关节或抬高身体掩盖模型
的躺倒问题。Root roll/pitch 是否正确必须由训练中的 root rotation 与 root tilt loss 解决。

接触状态来自网络两维 contact head，采用进入/退出双阈值避免概率在 0.5 附近抖动；单帧
根修正还带有位移上限，防止错误接触预测造成瞬时跳变。返回值同时保留原始/锁定后的足底
滑动统计和每帧修正量，便于 demo artifact 明确披露后处理影响。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .kinematics import BumiKinematics

BUMI_FOOT_LOCK_CONTRACT_VERSION = "genmo.bumi_fk_foot_lock_xy.v1"


@dataclass(frozen=True)
class BumiFootLockResult:
    qpos: torch.Tensor
    correction_xy: torch.Tensor
    active_contact: torch.Tensor
    contact_probability: torch.Tensor
    mean_contact_slide_before_mps: torch.Tensor
    mean_contact_slide_after_mps: torch.Tensor


def _contact_hysteresis(
    probability: torch.Tensor,
    *,
    enter_threshold: float,
    exit_threshold: float,
) -> torch.Tensor:
    frames = probability.shape[-2]
    flat_probability = probability.reshape(-1, frames, 2)
    active = torch.zeros(
        (flat_probability.shape[0], 2), dtype=torch.bool, device=probability.device
    )
    timeline: list[torch.Tensor] = []
    for frame_index in range(frames):
        frame_probability = flat_probability[:, frame_index]
        active = torch.where(
            active,
            frame_probability >= float(exit_threshold),
            frame_probability >= float(enter_threshold),
        )
        timeline.append(active)
    return torch.stack(timeline, dim=1).reshape_as(probability)


def _masked_slide_mean(
    foot_xy: torch.Tensor,
    active: torch.Tensor,
    fps: int,
) -> torch.Tensor:
    if foot_xy.shape[-3] <= 1:
        return foot_xy.new_zeros(())
    speed = torch.linalg.vector_norm(torch.diff(foot_xy, dim=-3) * float(fps), dim=-1)
    gate = active[..., 1:, :] & active[..., :-1, :]
    weights = gate.to(speed)
    return (speed * weights).sum() / weights.sum().clamp_min(1.0)


@torch.no_grad()
def lock_bumi_foot_contacts(
    qpos: torch.Tensor,
    contact: torch.Tensor,
    kinematics: BumiKinematics,
    *,
    contact_is_logits: bool = True,
    enter_threshold: float = 0.60,
    exit_threshold: float = 0.40,
    max_correction_per_frame: float = 0.08,
    fps: int = 30,
) -> BumiFootLockResult:
    """只修正 root XY，使预测接触区的 FK 足底尽量保持在首次接触锚点。"""

    if not isinstance(kinematics, BumiKinematics):
        raise TypeError("lock_bumi_foot_contacts requires BumiKinematics")
    if not isinstance(qpos, torch.Tensor) or qpos.ndim < 2 or qpos.shape[-1] != 28:
        raise ValueError(f"qpos must have shape [...,T,28], got {getattr(qpos, 'shape', None)}")
    expected_contact_shape = (*qpos.shape[:-1], 2)
    if not isinstance(contact, torch.Tensor) or tuple(contact.shape) != expected_contact_shape:
        raise ValueError(
            f"contact must have shape {expected_contact_shape}, "
            f"got {getattr(contact, 'shape', None)}"
        )
    if not bool(torch.isfinite(qpos).all()) or not bool(torch.isfinite(contact).all()):
        raise ValueError("qpos/contact contains NaN or Inf")
    if not 0.0 <= float(exit_threshold) < float(enter_threshold) <= 1.0:
        raise ValueError("contact thresholds require 0 <= exit < enter <= 1")
    if not math.isfinite(float(max_correction_per_frame)) or max_correction_per_frame <= 0.0:
        raise ValueError("max_correction_per_frame must be finite and > 0")
    if int(fps) <= 0:
        raise ValueError("fps must be positive")

    probability = torch.sigmoid(contact) if contact_is_logits else contact.clamp(0.0, 1.0)
    active = _contact_hysteresis(
        probability,
        enter_threshold=float(enter_threshold),
        exit_threshold=float(exit_threshold),
    )
    fk = kinematics.forward_kinematics(qpos)
    sole = kinematics.aggregate_sole_by_foot(fk["body_pos_w"], fk["body_quat_w"])
    foot_xy = sole["foot_points_w"][..., :2]
    frames = qpos.shape[-2]
    batch_shape = qpos.shape[:-2]
    # 锁定包含不可并行的接触锚点状态。一次性搬运这些小张量到 CPU 后递归，避免长音乐
    # 每帧 ``bool(cuda_tensor)`` 强制同步；最终仅把 [T,2] correction 传回原设备。
    flat_foot = foot_xy.reshape(-1, frames, 2, 2).detach().cpu()
    flat_probability = probability.reshape(-1, frames, 2).detach().cpu()
    flat_active = active.reshape(-1, frames, 2).detach().cpu()
    flat_correction = torch.zeros(
        (flat_foot.shape[0], frames, 2),
        dtype=flat_foot.dtype,
        device="cpu",
    )
    max_step = float(max_correction_per_frame)
    for batch_index in range(flat_foot.shape[0]):
        anchors = torch.zeros((2, 2), dtype=flat_foot.dtype)
        anchored = [False, False]
        previous = torch.zeros(2, dtype=flat_foot.dtype)
        for frame_index in range(frames):
            for foot_index in range(2):
                is_active = bool(flat_active[batch_index, frame_index, foot_index])
                if is_active and not anchored[foot_index]:
                    anchors[foot_index] = flat_foot[batch_index, frame_index, foot_index] + previous
                    anchored[foot_index] = True
                elif not is_active:
                    anchored[foot_index] = False

            active_indices = [
                foot_index
                for foot_index in range(2)
                if bool(flat_active[batch_index, frame_index, foot_index]) and anchored[foot_index]
            ]
            if active_indices:
                index = torch.tensor(active_indices, dtype=torch.long)
                errors = anchors.index_select(0, index) - flat_foot[
                    batch_index, frame_index
                ].index_select(0, index)
                weights = flat_probability[batch_index, frame_index].index_select(0, index)
                weights = weights.clamp_min(1.0e-6)
                target = (errors * weights[:, None]).sum(dim=0) / weights.sum()
                delta = target - previous
                length = torch.linalg.vector_norm(delta)
                if float(length) > max_step:
                    delta = delta * (max_step / length.clamp_min(1.0e-8))
                previous = previous + delta
            flat_correction[batch_index, frame_index] = previous

    correction = flat_correction.reshape(*batch_shape, frames, 2).to(qpos)
    locked_qpos = qpos.clone()
    locked_qpos[..., :2] = locked_qpos[..., :2] + correction
    locked_foot_xy = foot_xy + correction.unsqueeze(-2)
    return BumiFootLockResult(
        qpos=locked_qpos.contiguous(),
        correction_xy=correction,
        active_contact=active,
        contact_probability=probability,
        mean_contact_slide_before_mps=_masked_slide_mean(foot_xy, active, int(fps)),
        mean_contact_slide_after_mps=_masked_slide_mean(locked_foot_xy, active, int(fps)),
    )


__all__ = [
    "BUMI_FOOT_LOCK_CONTRACT_VERSION",
    "BumiFootLockResult",
    "lock_bumi_foot_contacts",
]
