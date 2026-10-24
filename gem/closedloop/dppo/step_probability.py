"""逐去噪步骤 DPPO 的统计和诊断，不构造跨步骤概率乘积。

输入是同一物理上层转移的 K 个可训练高斯步骤，形状为 [N,K]。每步的
log-ratio 仍严格求和该步所有自由动作坐标；已承诺前缀、padding、冻结步骤和
确定性收尾不参与密度。共享同一上层 GAE，只在步骤维广播，不重复标准化优势。
PPO 在每步独立裁剪，最终先对 K 取平均、再对 N 个物理转移取平均。
诊断中的样本量始终为 N，N*K 仅是内部转移访问数。KL 门禁统计单步条件
高斯 KL，跨步骤 KL 之和只作为带明确名称的附加诊断，绝不用于 PPO 比值。
本模块不改变采样核、奖励、物理终止、优化器或行为数据。
"""
from __future__ import annotations

import torch

from .gaussian_probability import distribution, ppo_surrogate


def step_statistics(log_ratios, kls, counts):
    if log_ratios.ndim != 2 or kls.shape != log_ratios.shape or counts.shape != log_ratios.shape:
        raise ValueError('Denoising statistics require equal [chains,steps] tensors')
    if log_ratios.shape[1] < 1:
        raise ValueError('At least one trainable Gaussian step required')
    # 一个物理动作的mask在各去噪步相同；不把缺步骤误当可变样本数。
    if not torch.equal(counts, counts[:, :1].expand_as(counts)):
        raise ValueError('Every denoising step must share the same free-coordinate mask count')
    return dict(joint_log_ratio=log_ratios.double(), joint_kl=kls.double(), free_count=counts,
        mean_coordinate_log_ratio=log_ratios.double()/counts.clamp_min(1),
        mean_coordinate_kl=kls.double()/counts.clamp_min(1))


def step_surrogate(statistics, advantages, *, clip):
    if advantages.ndim != 1 or len(advantages) != statistics['joint_log_ratio'].shape[0]:
        raise ValueError('One upper GAE advantage per complete physical transition required')
    return ppo_surrogate(statistics, advantages[:, None], clip=clip, reduction='joint_sum')


def step_diagnostics(statistics, advantages, *, clip):
    surrogate = step_surrogate(statistics, advantages, clip=clip)
    valid = statistics['free_count'] > 0
    valid_chains = valid[:, 0]
    advantage = advantages.detach().to(surrogate['objective'])[:, None].expand_as(valid)
    log_ratio = surrogate['log_ratio'][valid].detach()
    return dict(denoising_step_log_ratio=distribution(log_ratio),
        denoising_step_ratio=distribution(surrogate['ratio'][valid]),
        denoising_step_kl=distribution(statistics['joint_kl'][valid]),
        chain_kl_sum_diagnostic=distribution(statistics['joint_kl'].sum(1)[valid_chains]),
        objective_mean=float(surrogate['objective'][valid].mean()),
        behavior_objective_mean=float(advantages[valid_chains].double().mean()),
        objective_gain_mean=float((surrogate['objective']-advantage)[valid].mean()),
        joint_score_gain_mean=float((advantage*surrogate['log_ratio'])[valid].mean()),
        clip_fraction=float(surrogate['clipped'][valid].double().mean()),
        active_clip_fraction=float(surrogate['objective_clipped'][valid].double().mean()),
        nonfinite_fraction=0., ratio_underflow_fraction=0., ratio_is_joint_probability=False,
        ratio_scope='one_denoising_step_joint_free_coordinates',
        complete_chain_count=int(valid_chains.sum()), internal_transition_count=int(valid.sum()),
        denominator='mean_over_K_steps_then_mean_over_N_physical_transitions')
