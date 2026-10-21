"""Stage2 的稳定高斯统计和完整随机链联合 PPO 概率。

行为策略仍保存 normalized qpos30 的完整 FP64 联合概率、旧均值和实际标准差。
训练目标将15～18四个随机转移的自由坐标 log-ratio 在FP64求和，每条链只裁剪一次，
对应一个上层GAE优势。坐标平均值仅作诊断，已承诺前缀和padding不参与概率；
冻结前15步通过身份校验抵消，确定性末步不具有高斯密度，不参与联合概率。

比值直接使用旧核的标准化残差和均值变化计算，不先求两个大联合 log-prob 再
相减；方差变化采用 expm1，固定方差时恰好退化为 z*d-d*d/2。KL 使用同一实际
高斯核计算 old||new，保持完整联合 KL 与坐标平均 KL 两套命名和量纲。这里不
做概率截断、不修改行为数据、不处理优化器，也不改变独立概率验收门槛。
"""
from __future__ import annotations

import math
import json
import torch

from .execution_checks import require_tensor


def gaussian_statistics(value, old_mean, old_std, new_mean, new_std, free_mask, *, coordinate_details=False):
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
    result=dict(joint_log_ratio=joint_ratio, joint_kl=joint_kl.clamp_min(0.), free_count=counts,
        mean_coordinate_log_ratio=joint_ratio/counts.clamp_min(1),
        mean_coordinate_kl=joint_kl.clamp_min(0.)/counts.clamp_min(1))
    if coordinate_details:
        # 诊断复用同一完整异方差公式；普通训练不分配这些额外输出。
        result['coordinates']={key:torch.where(free_mask,tensor,0.) for key,tensor in dict(
            kl=kl_terms,log_ratio=ratio_terms,mean_delta=new-old,
            standardized_mean_delta=(new-old)/sigma0,residual=z,
            actual_std=sigma0.expand_as(x),std_log_change=r.expand_as(x)).items()}
        result['coordinates']['mask']=free_mask
    return result


def ppo_surrogate(statistics, advantages, *, clip, reduction='joint_sum'):
    """执行真正的符号相关 PPO clipping；零自由度转移不贡献目标。"""
    if reduction not in ('joint_sum','free_coordinate_mean') or not 0 < clip < 1:
        raise ValueError('Unknown proxy reduction or invalid PPO clip')
    log_ratio = statistics['joint_log_ratio' if reduction == 'joint_sum' else 'mean_coordinate_log_ratio']
    advantage = advantages.detach().to(log_ratio)
    valid = statistics['free_count'] > 0
    bad=~torch.isfinite(log_ratio) | ~torch.isfinite(advantage)
    overflow=log_ratio>math.log(torch.finfo(torch.float64).max)
    underflow=log_ratio<math.log(float.fromhex('0x0.0000000000001p-1022'))
    if bool((bad|overflow|underflow).any()):
        raise FloatingPointError('PPO candidate must be rolled back: '+json.dumps(dict(
            nonfinite_fraction=float(bad.double().mean()),ratio_overflow_fraction=float(overflow.double().mean()),
            ratio_underflow_fraction=float(underflow.double().mean()),
            action='reject_candidate_without_ratio_clamp_or_sample_filter')))
    lower, upper = math.log1p(-clip), math.log1p(clip)
    active_clip = ((advantage > 0)&(log_ratio > upper)) | ((advantage < 0)&(log_ratio < lower))
    # 先在log空间选择数学上正确的分支；被裁剪分支不对极端log-ratio取指数。
    objective_log = torch.where(active_clip, torch.where(advantage > 0, upper, lower), log_ratio)
    objective = objective_log.exp()*advantage
    ratio = log_ratio.exp()
    # 任何溢出/下溢均显式拒绝，而非改写ratio、删除样本或静默产生零梯度。
    require_tensor(torch.isfinite(log_ratio).all() & torch.isfinite(advantage).all()
        & torch.isfinite(ratio).all() & (ratio > 0).all() & torch.isfinite(objective).all(),
        'PPO joint ratio/objective nonfinite or underflow; candidate must be rolled back', FloatingPointError)
    return dict(objective=torch.where(valid,objective,0.), ratio=ratio, log_ratio=log_ratio,
        clipped=((log_ratio < lower)|(log_ratio > upper))&valid, objective_clipped=active_clip&valid,
        objective_log_gradient=torch.where(valid & ~active_clip, objective, 0.).detach(),
        reduction=reduction, ratio_is_joint_probability=reduction=='joint_sum')


def chain_statistics(joint_log_ratios, joint_kls, free_counts, *, steps=(15,16,17,18)):
    """四个旧路径条件高斯组成一个扩散链；不除坐标数、不做去噪折扣。"""
    if tuple(steps)!=(15,16,17,18) or joint_log_ratios.ndim!=2 or joint_log_ratios.shape[1]!=4:
        raise ValueError('Exactly four stochastic transitions 15..18 required per complete chain')
    if joint_kls.shape!=joint_log_ratios.shape or free_counts.shape!=joint_log_ratios.shape:
        raise ValueError('Chain log ratios, KL and free counts must have identical [chains,4] shape')
    counts=free_counts.sum(1)
    logratio=joint_log_ratios.double().sum(1)
    kl=joint_kls.double().sum(1)
    return dict(joint_log_ratio=logratio,joint_kl=kl,free_count=counts,
        mean_coordinate_log_ratio=logratio/counts.clamp_min(1),mean_coordinate_kl=kl/counts.clamp_min(1))


def distribution(values):
    """小型诊断向量转为JSON有限标量；拒绝非有限值，不滤掉异常样本。"""
    values=values.detach().double().flatten()
    if not len(values) or not torch.isfinite(values).all():
        raise FloatingPointError('Cannot summarize empty or nonfinite probability population')
    quantiles=torch.quantile(values,values.new_tensor([0.,.01,.05,.5,.95,.99,1.])).cpu().tolist()
    return dict(zip(('min','p01','p05','median','p95','p99','max'),quantiles),mean=float(values.mean()))


def chain_diagnostics(statistics, advantages, *, clip):
    """真实联合ratio、符号相关裁剪及联合KL；每条完整链只统计一次。"""
    surrogate=ppo_surrogate(statistics,advantages,clip=clip,reduction='joint_sum')
    valid=statistics['free_count']>0
    logratio=surrogate['log_ratio'][valid].detach()
    return dict(chain_joint_log_ratio=distribution(logratio),
        objective_mean=float(surrogate['objective'][valid].mean()),
        behavior_objective_mean=float(advantages.detach().to(logratio)[valid].mean()),
        objective_gain_mean=float((surrogate['objective'][valid]-advantages.detach().to(logratio)[valid]).mean()),
        joint_score_gain_mean=float((advantages.detach().to(logratio)[valid]*logratio).mean()),
        chain_joint_ratio=distribution(surrogate['ratio'][valid]),chain_joint_kl=distribution(statistics['joint_kl'][valid]),
        clip_fraction=float(surrogate['clipped'][valid].double().mean()),
        active_clip_fraction=float(surrogate['objective_clipped'][valid].double().mean()),
        nonfinite_fraction=0.,ratio_underflow_fraction=0.,
        ratio_is_joint_probability=True,denominator='one_complete_chain_one_upper_GAE_advantage')
