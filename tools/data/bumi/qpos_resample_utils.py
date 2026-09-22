"""BUMI qpos 四元数插值与 body-origin 落地共享工具。

本模块承载与数据来源无关的数值步骤：规范化连续 ``wxyz`` 根四元数、批量最短弧
SLERP，以及通过绑定 BUMI kinematics 对整条轨迹施加常量 Root-Z 偏移。当前 CSV
producer 直接依赖这些数学 helper；robot_retargeter producer 也复用同一四元数连续化，
因此它们不需要从任一特定历史数据入口间接导入公共数值逻辑。

这些函数不会猜测关节顺序、调整关节限位或生成接触标签。落地只移动 Root Z，并继续
以 FK 后所有 body origin 的全局最小 Z 为零作为兼容语义，因此迁移不改变当前 CSV
producer 的既有数据数值。
"""

from __future__ import annotations

import numpy as np
import torch

from gem.robots.bumi.kinematics import BumiKinematics


def make_quaternion_continuous_np(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """归一化 ``wxyz`` 四元数并消除逐帧 ``q/-q`` 符号跳变。"""

    value = np.asarray(quaternion_wxyz, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 4 or value.shape[0] <= 0:
        raise ValueError(f"quaternion 必须为 [T,4]，实际 {value.shape}")
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    if not np.isfinite(value).all() or np.any(norm < 1.0e-8):
        raise ValueError("quaternion 包含非有限值或零范数")
    result = value / norm
    for index in range(1, len(result)):
        if float(np.dot(result[index - 1], result[index])) < 0.0:
            result[index] *= -1.0
    return result


def slerp_pairs(q0: np.ndarray, q1: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """对同形状 ``wxyz`` 四元数对执行向量化最短弧 SLERP。"""

    dot = np.sum(q0 * q1, axis=-1)
    q1_short = q1.copy()
    negative = dot < 0.0
    q1_short[negative] *= -1.0
    dot = np.abs(dot).clip(0.0, 1.0)
    angle = np.arccos(dot)
    sin_angle = np.sin(angle)
    use_linear = sin_angle < 1.0e-7
    a = np.asarray(alpha, dtype=np.float64)
    weight0 = np.empty_like(a)
    weight1 = np.empty_like(a)
    weight0[use_linear] = 1.0 - a[use_linear]
    weight1[use_linear] = a[use_linear]
    stable = ~use_linear
    weight0[stable] = np.sin((1.0 - a[stable]) * angle[stable]) / sin_angle[stable]
    weight1[stable] = np.sin(a[stable] * angle[stable]) / sin_angle[stable]
    result = weight0[:, None] * q0 + weight1[:, None] * q1_short
    return result / np.linalg.norm(result, axis=-1, keepdims=True)


def normalize_body_origin_ground(
    qpos: torch.Tensor, kinematics: BumiKinematics
) -> tuple[torch.Tensor, float, float]:
    """施加常量 Root-Z 偏移，使 FK 的 body-origin 全局最小 Z 为零。"""

    value = qpos.detach().cpu().float().clone()
    with torch.no_grad():
        before = float(kinematics.forward_kinematics(value)["body_pos_w"][..., 2].amin().item())
        value[:, 2] -= before
        after = float(kinematics.forward_kinematics(value)["body_pos_w"][..., 2].amin().item())
    if abs(after) > 2.0e-5:
        raise RuntimeError(f"root-Z 归一化后 body-origin ground={after:.8g}，不接近 0")
    return value.contiguous(), before, after
