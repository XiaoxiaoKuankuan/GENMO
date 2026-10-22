"""原坐标平均DPPO的固定真实数据目标与梯度诊断数学。

输入为每条真实链四个训练步的联合log-ratio、联合KL和自由坐标数，以及冻结的
上层GAE优势。生产目标严格复现原updater：每步先除自由坐标数，使用原去噪折扣，
每步独立clip，最后以全局有效链数乘四为分母。联合链目标仅作另一量纲的诊断，
不会替换生产目标，不截断真实概率或删除异常样本。

本模块只做张量运算与统计，不采物理数据、不更新模型、不修改优势、奖励或旧链。
其单独测试覆盖非零策略偏移、符号相关裁剪、全局分母及独立Normal公式梯度。
"""
from __future__ import annotations

import math
import torch

from gem.closedloop.dppo.gaussian_probability import ppo_surrogate


STEPS = (15, 16, 17, 18)


def production_surrogate(records, advantages, *, clip, gamma_denoising=.99):
    if records.ndim != 3 or records.shape[1:] != (4, 3) or advantages.shape != records.shape[:1]:
        raise ValueError('需要[N,4,3]真实统计与[N]固定优势')
    count = records[..., 2]
    stats = dict(joint_log_ratio=records[..., 0], joint_kl=records[..., 1],
                 free_count=count, mean_coordinate_log_ratio=records[..., 0]/count.clamp_min(1))
    discount = records.new_tensor([gamma_denoising**(19-step) for step in STEPS])
    return ppo_surrogate(stats, advantages[:, None]*discount, clip=clip, reduction='free_coordinate_mean')


def quantiles(values):
    values = values.detach().double().flatten()
    if not len(values) or not torch.isfinite(values).all():
        raise ValueError('统计必须覆盖非空有限总体，不能筛掉异常值')
    q = torch.quantile(values, values.new_tensor([0., .5, .95, .99, 1.])).cpu().tolist()
    return dict(zip(('min', 'p50', 'p95', 'p99', 'max'), q), mean=float(values.mean()))


def summarize_records(records, advantages, *, clip, gamma_denoising=.99):
    proxy = production_surrogate(records, advantages, clip=clip, gamma_denoising=gamma_denoising)
    count = records[..., 2]
    coordinate_kl = records[..., 1]/count.clamp_min(1)
    joint_stats = dict(joint_log_ratio=records[..., 0].sum(1),
                       mean_coordinate_log_ratio=records[..., 0].sum(1)/count.sum(1).clamp_min(1),
                       free_count=count.sum(1))
    result = dict(production_objective=float(proxy['objective'].mean()),
                  active_clip_fraction=float(proxy['objective_clipped'].double().mean()),
                  ordinary_clip_fraction=float(proxy['clipped'].double().mean()),
                  coordinate_ratio=quantiles(proxy['ratio']), coordinate_log_ratio=quantiles(proxy['log_ratio']),
                  mean_coordinate_kl=float(coordinate_kl.mean()),
                  maximum_step_mean_coordinate_kl=float(coordinate_kl.mean(0).max()),
                  coordinate_kl=quantiles(coordinate_kl), chain_joint_kl=quantiles(records[..., 1].sum(1)),
                  mean_joint_kl=float(records[..., 1].mean()),
                  joint_log_ratio=quantiles(joint_stats['joint_log_ratio']),
                  valid_chains=len(records), internal_transitions=4*len(records),
                  clip=clip, gamma_denoising=gamma_denoising,
                  objective_scope='actual_production_coordinate_mean_independent_four_step_clipped_surrogate')
    try:
        joint = ppo_surrogate(joint_stats, advantages, clip=clip, reduction='joint_sum')
        result['counterfactual_joint_objective'] = float(joint['objective'].mean())
        result['counterfactual_joint_active_clip_fraction'] = float(joint['objective_clipped'].double().mean())
    except FloatingPointError as error:
        result['counterfactual_joint_objective'] = None
        result['counterfactual_joint_error'] = str(error)
    result['counterfactual_joint_scope'] = 'different_objective_diagnostic_not_the_training_loss'
    return result


def vector_relation(first, second):
    names = sorted(set(first)|set(second))
    device = next(iter(first.values())).device
    sums = torch.zeros(4, dtype=torch.float64, device=device)
    for name in names:
        a, b = first.get(name), second.get(name)
        if a is not None:
            a = a.double();sums[0] += a.square().sum()
        if b is not None:
            b = b.double();sums[1] += b.square().sum()
        if a is not None and b is not None:
            sums[2] += (a*b).sum();sums[3] += (a-b).square().sum()
        elif a is not None:
            sums[3] += a.square().sum()
        elif b is not None:
            sums[3] += b.square().sum()
    aa, bb, dot, diff = sums.cpu().tolist()
    return dict(first_norm=math.sqrt(aa), second_norm=math.sqrt(bb), dot=dot,
                cosine=dot/math.sqrt(aa*bb) if aa and bb else None,
                relative_difference=math.sqrt(diff/max(aa, 1e-300)))


@torch.no_grad()
def actual_update_direction(actor, before, loss_gradient):
    sums = torch.zeros(3, dtype=torch.float64, device=next(actor.parameters()).device)
    for name, p in actor.named_parameters():
        if name not in before:
            continue
        delta = p.double()-before[name].double();sums[0] += delta.square().sum()
        grad = loss_gradient.get(name)
        if grad is not None:
            grad = grad.double();sums[1] += grad.square().sum();sums[2] += (grad*delta).sum()
    delta2, grad2, dot = sums.cpu().tolist()
    return dict(full_objective_first_order_gain=-dot, actual_parameter_delta_norm=math.sqrt(delta2),
                full_loss_gradient_norm=math.sqrt(grad2),
                update_cosine_with_objective_ascent=-dot/math.sqrt(delta2*grad2) if delta2 and grad2 else None)
