"""RoboFFT-D 前向腐蚀代理与一次上层动作一次裁剪的纯张量数学。

输入为最终解码前 normalized action_x0、独立固定的标准正态 eps、原始扩散时间和
同 CFG 的 START_X 预测。前向腐蚀使用原始 alpha_bar，已知坐标由调用者保持干净、
padding 保持零；仅有效未知 qpos30 进入 epsilon MSE，特征维平均、时间帧求和。
Delta log-SNR 的 1e-3 裁剪只用于权重，不改变 q_sample 的实际方差。

参考 shshsunny/RoboFFT 的 diffusion_fpo.py：先对每个 MC 的 logp 差 clamp，再乘
alpha、平均 MC 对数比、exp，最后按动作做一次 PPO clip 和负优势折扣。本模块不做
优势标准化，也不除以动作数/world_size；调用者使用真实全局有效动作数统一归一化。
surrogate_kl 是该代理比值的诊断量，不是真实策略 KL。contact2 不参与任何概率。
"""
from __future__ import annotations

import torch


def delta_log_snr(alpha, previous_alpha):
    def clipped_log_snr(value):
        variance = (1. - value).clamp_min(1e-3)
        return (1. - variance).clamp_min(1e-3).log() - variance.log()
    return clipped_log_snr(previous_alpha) - clipped_log_snr(alpha)


def q_sample(x0, eps, alpha, known_x, known_mask, valid):
    noisy = alpha.sqrt() * x0 + (1. - alpha).sqrt() * eps
    return torch.where(valid[..., None], torch.where(known_mask, known_x, noisy), 0.)


def x0_to_epsilon(x_t, pred_x0, alpha):
    return (x_t - alpha.sqrt() * pred_x0) / (1. - alpha).sqrt()


def epsilon_loss(x_t, pred_x0, eps, alpha, weight, free_mask, time_weights=None):
    residual = x0_to_epsilon(x_t, pred_x0, alpha) - eps
    error = torch.where(free_mask, residual, 0.).double().square()
    if time_weights is not None:
        if time_weights.shape != error.shape[:-1] or time_weights.requires_grad:
            raise ValueError('Fixed execution time weights must align with frames')
        error = error * time_weights.to(error)[..., None]
    return .5 * weight * error.mean(dim=-1).sum(dim=-1)


def action_objective(new_loss, old_loss, advantages, *, clip=.01, beta_neg=.125,
                     logprob_normalizer=1., logratio_alpha=.1):
    """返回逐动作 loss/ratio/诊断；old_loss/advantages 永久脱离计算图。"""
    if new_loss.ndim != 2 or old_loss.shape != new_loss.shape or advantages.shape != new_loss.shape[:1]:
        raise ValueError('RoboFFT requires loss [actions, MC] and one advantage per action')
    if not (0 < clip < 1 and 0 < beta_neg <= 1 and logprob_normalizer > 0 and logratio_alpha > 0):
        raise ValueError('Invalid action-level PPO parameters')
    new_logp = -new_loss * logprob_normalizer
    old_logp = -old_loss.detach() * logprob_normalizer
    raw = new_logp - old_logp
    logratio = (raw.clamp(-10., 10.) * logratio_alpha).mean(dim=-1)
    ratio = logratio.exp()
    advantage = advantages.detach().to(ratio)
    loss = torch.maximum(-advantage * ratio, -advantage * ratio.clamp(1. - clip, 1. + clip))
    loss = loss * torch.where(advantage < 0, beta_neg, 1.)
    return dict(loss=loss, ratio=ratio, logratio=logratio,
        surrogate_kl=(ratio - 1.) - logratio,
        ratio_clip_fraction=((ratio - 1.).abs() > clip).double(),
        logratio_clamp_fraction=((raw < -10.) | (raw > 10.)).double().mean(-1),
        new_logp_range_fraction=((new_logp < -200.) | (new_logp > 2.)).double().mean(-1),
        old_logp_range_fraction=((old_logp < -200.) | (old_logp > 2.)).double().mean(-1))
