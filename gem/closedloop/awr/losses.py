"""逐样本优势加权的qpos30去噪回归。

先在每条样本自己的有效未知坐标上求平均MSE，再乘固定AWR权重，最后按真实
全局样本数归约。目标为生成器原始参考而非GMT实际姿态；本模块没有Contact、
FK伪标签或PPO概率。BC由独立原Stage1损失支路负责，两个全局分母不能混用。
"""
from __future__ import annotations

import torch


def per_sample_denoising(prediction, target, mask):
    if prediction.shape != target.shape or mask.shape != target.shape or mask.dtype != torch.bool:
        raise ValueError("AWR prediction/target/mask must share [B,T,30]")
    if prediction.ndim != 3 or prediction.shape[-1] != 30:
        raise ValueError("AWR target is qpos30")
    counts = mask.flatten(1).sum(1)
    if (counts == 0).any():
        raise ValueError("AWR Actor samples require unknown valid coordinates")
    # where在相减之前屏蔽无效坐标，防止padding中的占位NaN污染有效样本。
    difference = torch.where(mask, prediction, 0.) - torch.where(mask, target, 0.)
    if not torch.isfinite(difference).all():
        raise FloatingPointError("Nonfinite AWR reconstruction")
    return difference.square().flatten(1).sum(1) / counts


def weighted_denoising(prediction, target, mask, weights, *, global_count=None):
    losses = per_sample_denoising(prediction, target, mask)
    w = weights.to(device=losses.device, dtype=torch.float64)
    if w.shape != losses.shape or not torch.isfinite(w).all() or not (w > 0).all():
        raise ValueError("One positive finite AWR weight required per sample")
    denominator = len(losses) if global_count is None else global_count
    if denominator < len(losses) or denominator <= 0:
        raise ValueError("Invalid global sample denominator")
    return (losses.double() * w).sum() / denominator, losses
