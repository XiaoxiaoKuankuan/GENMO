"""完整动作训练的有效元素损失归一化与时序边界 mask。

新策略先在每条样本内对广播后的有效残差元素求平均，再做 batch 平均，并在样本
平均后乘扩散采样 importance weight；空 mask 给出可求导零值。旧策略保留原有
乘 mask 后整体 mean 的数值行为。权重不是有效元素计数，不会偷偷进入平均分母。
此模块不吞掉有效位置的 NaN，也不新增监督目标。
"""

import torch


def temporal_valid_mask(valid, order=1, *, pad_last=False):
    """第 t 项差分仅当 t 到 t+order 都有效时有效；末尾无观测差分为 False。"""
    if order < 1:
        raise ValueError("差分阶数必须为正")
    count = max(valid.shape[1] - order, 0)
    result = valid[:, :count].clone()
    for offset in range(1, order + 1):
        result &= valid[:, offset : offset + count]
    if pad_last:
        result = torch.cat([result, torch.zeros_like(valid[:, count:])], dim=1)
    return result


def masked_reduce(residual, mask, *, strategy="legacy", sample_weights=None):
    if strategy == "legacy":
        return (residual * mask).mean()
    if strategy != "valid_per_sample":
        raise ValueError(f"未知 loss reduction: {strategy}")
    mask = torch.broadcast_to(mask.bool(), residual.shape)
    # 在 FP32 累加，避免 BF16/FP16 下大顶点张量的有效元素计数或求和溢出。
    sums = torch.where(mask, residual, 0).float().flatten(1).sum(1)
    counts = mask.flatten(1).sum(1).clamp_min(1)
    per_sample = sums / counts
    if sample_weights is not None:
        if sample_weights.shape != per_sample.shape:
            raise ValueError("扩散 t_weights 必须与 batch 对齐")
        per_sample = per_sample * sample_weights
    return per_sample.mean()
