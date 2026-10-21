"""AWR去噪训练的固定扩散时间权重与来源分布校正。

输入为原Stage1训练扩散表alphas_cumprod和本轮不可变Replay来源；不改变采样器、
加噪公式、条件、目标或物理奖励。x0预测的Min-SNR-gamma使用min(SNR(t),gamma)，
可按完整时间表一次均值归一化，禁止按microbatch重算。来源校正为N/(S*N_source)，
使均匀遍历Replay对应等权来源目标；不按优势优先抽样，原AWR标签保持不变。
这些是可配置的监督目标选项，不保证消除CFG采样/再加噪的全部统计差异。所有因子
独立记录并绑定checkpoint身份；uniform和none完整保留旧目标作对照。
x0权重依据Hang等ICCV2023的Min-SNR公式：https://arxiv.org/abs/2303.09556；
整张时间表的均值归一化是本项目工程选项，gamma=5是本轮候选，非已验证最优值。
"""
from __future__ import annotations

from collections import Counter
import math
import torch


def timestep_weights(alphas_cumprod, config=None):
    config = dict(config or {})
    if set(config) - {"mode", "gamma", "mean_normalize"}:
        raise ValueError("Unknown diffusion timestep weighting option")
    alpha = torch.as_tensor(alphas_cumprod, dtype=torch.float64).clone()
    if alpha.ndim != 1 or not len(alpha) or not torch.isfinite(alpha).all() or not ((alpha > 0) & (alpha < 1)).all():
        raise ValueError("Original diffusion alpha table must be finite and strictly between zero and one")
    mode = config.get("mode", "uniform")
    if mode == "uniform":
        return torch.ones_like(alpha)
    if mode != "min_snr":
        raise ValueError("Use uniform or min_snr for x0 timestep weighting")
    gamma = config.get("gamma", 5.)
    normalize = config.get("mean_normalize", True)
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)) or not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("Min-SNR gamma must be finite and positive")
    if type(normalize) is not bool:
        raise ValueError("Timestep mean_normalize must be boolean")
    weights = (alpha/(1.-alpha)).clamp_max(gamma)
    if normalize:
        weights = weights/weights.mean()
    if not torch.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError("Invalid positive x0 timestep weights")
    return weights


def source_weights(sources, mode="none"):
    counts = Counter(sources)
    if not counts or any(not isinstance(s, str) or not s for s in sources):
        raise ValueError("Replay source identity is required")
    if mode == "none":
        return torch.ones(len(sources), dtype=torch.float64)
    if mode != "uniform_sources":
        raise ValueError("Use none or uniform_sources for the Replay objective")
    return torch.tensor([len(sources)/(len(counts)*counts[s]) for s in sources], dtype=torch.float64)
