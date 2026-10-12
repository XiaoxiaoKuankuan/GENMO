"""Stage10 v3 的稳定高斯比值、坐标平均代理目标和真实联合 KL。

行为策略仍保存 normalized qpos30 的完整 FP64 联合概率、旧均值和实际标准差。
PPO 主目标只将每个内部转移的 log-ratio 除以自由有效坐标数；这是显式的新
代理目标，绝不把 exp(mean(log-ratio)) 解释为联合密度比。已承诺前缀和 padding
从两种归约中同时排除，完整四秒自由动作仍参与，没有按短执行窗口截断目标。

比值直接使用旧核的标准化残差和均值变化计算，不先求两个大联合 log-prob 再
相减；方差变化采用 expm1，固定方差时恰好退化为 z*d-d*d/2。KL 使用同一实际
高斯核计算 old||new，保持完整联合 KL 与坐标平均 KL 两套命名和量纲。这里不
做概率截断、不修改行为数据、不处理优化器，也不改变独立概率验收门槛。
"""
from __future__ import annotations

import torch

from .execution_checks import require_tensor


def gaussian_statistics(value, old_mean, old_std, new_mean, new_std, free_mask):
    """返回 FP64 [B] 的直接联合 log-ratio、联合 KL 与自由坐标数。"""
    if value.shape != old_mean.shape or value.shape != new_mean.shape or value.shape != free_mask.shape:
        raise ValueError('Gaussian values, means and mask must have equal [B,T,D] shapes')
    if value.ndim != 3 or free_mask.dtype != torch.bool:
        raise ValueError('Expected [B,T,D] values and boolean free mask')
    for scale in (old_std, new_std):
        if scale.shape not in ((len(value), 1, 1), value.shape):
            raise ValueError('Actual Gaussian std must be [B,1,1] or [B,T,D]')
        require_tensor(torch.isfinite(scale).all() & (scale > 0).all(), 'Invalid actual Gaussian standard deviation')
    x = torch.where(free_mask, value.detach(), 0.).double()
    old = torch.where(free_mask, old_mean.detach(), 0.).double()
    new = torch.where(free_mask, new_mean, 0.).double()
    sigma0, sigma = old_std.detach().double(), new_std.double()
    # log1p在新旧标准差很接近时避免先做除法再log的额外消减。
    r = torch.log1p((sigma-sigma0)/sigma0)
    a = torch.exp(-r)
    z, d = (x-old)/sigma0, (new-old)/sigma
    ratio_terms = -r - .5*torch.expm1(-2*r)*z.square() + a*z*d - .5*d.square()
    kl_terms = r + .5*torch.expm1(-2*r) + .5*d.square()
    joint_ratio = ratio_terms.masked_fill(~free_mask, 0.).sum((-2,-1))
    joint_kl = kl_terms.masked_fill(~free_mask, 0.).sum((-2,-1))
    counts = free_mask.sum((-2,-1))
    require_tensor(torch.isfinite(joint_ratio).all() & torch.isfinite(joint_kl).all()
        & (joint_kl >= -1e-10).all(), 'Nonfinite Gaussian ratio or negative KL', FloatingPointError)
    return dict(joint_log_ratio=joint_ratio, joint_kl=joint_kl.clamp_min(0.), free_count=counts,
        mean_coordinate_log_ratio=joint_ratio/counts.clamp_min(1),
        mean_coordinate_kl=joint_kl.clamp_min(0.)/counts.clamp_min(1))


def ppo_surrogate(statistics, advantages, *, clip, reduction='free_coordinate_mean'):
    """执行真正的符号相关 PPO clipping；零自由度转移不贡献目标。"""
    if reduction not in ('joint_sum','free_coordinate_mean') or not 0 < clip < 1:
        raise ValueError('Unknown proxy reduction or invalid PPO clip')
    log_ratio = statistics['joint_log_ratio' if reduction == 'joint_sum' else 'mean_coordinate_log_ratio']
    ratio = log_ratio.exp()
    require_tensor(torch.isfinite(ratio).all(), 'PPO ratio overflow; no silent log-ratio clamp', FloatingPointError)
    advantage = advantages.detach().to(ratio)
    valid = statistics['free_count'] > 0
    objective = torch.minimum(ratio*advantage, ratio.clamp(1-clip,1+clip)*advantage)
    active_clip = ((advantage > 0)&(ratio > 1+clip)) | ((advantage < 0)&(ratio < 1-clip))
    return dict(objective=torch.where(valid,objective,0.), ratio=ratio, log_ratio=log_ratio,
        clipped=((ratio < 1-clip)|(ratio > 1+clip))&valid, objective_clipped=active_clip&valid,
        reduction=reduction, ratio_is_joint_probability=reduction=='joint_sum')
