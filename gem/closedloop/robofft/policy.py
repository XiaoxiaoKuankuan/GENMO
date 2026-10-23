"""GENMO START_X 的 RoboFFT-D 适配器：紧凑采样与同 CFG 前向代理。

继承已验收的 9 冻结/10 可训练/1 冻结确定性采样，只关闭整链和 Gaussian 旧概率
存储；最终 normalized action_x0 在物理解码前形成，输出快照不引用参考计划缓存。
代理直接调用采样共用的 x0/CFG 前向，绝不调用 Stage1 的随机音乐 dropout 训练损失。
新旧 Actor 均 eval，但当前 Actor 的 Transformer、音乐、历史和前缀编码器保留梯度。

MC 编号、采样索引、original timestep 分开保存。均匀选择采样索引 9..18，腐蚀
alpha 来自原 1000 步日程；Delta log-SNR 使用对应相邻采样时刻的 alpha。MC 不使用
rollout std_schedule，不训练末步冻结网络，不给 contact2 构造代理概率。
"""
from __future__ import annotations

import torch

from . import ALGORITHM
from .surrogate import delta_log_snr, epsilon_loss, q_sample
from ..dppo.frozen_prefix import FrozenPrefixPolicy
from ..dppo.execution_checks import checked_policy_phase


class RoboFFTDiffusionPolicy(FrozenPrefixPolicy):
    algorithm = ALGORITHM

    def __init__(self, actor, *, robofft, **kwargs):
        super().__init__(actor, **kwargs)
        if self.frozen_prefix_steps != 9 or self.actor.diffusion_steps != 1000:
            raise ValueError('RoboFFT requires original 1000-step diffusion and sampling indices 9..18')
        self.robofft = dict(robofft)
        # 不修改行为核身份；同 RNG 的完整链和紧凑采样仍是同一动作分布。
        self.original_alpha = torch.tensor(self.actor._diffusion(1000).alphas_cumprod, dtype=torch.float64)
        current = self.original_alpha[torch.tensor(self.timestep_map[9:19])]
        previous = self.original_alpha[torch.tensor(self.timestep_map[10:20])]
        self.delta_weights = delta_log_snr(current, previous)

    def sample_rollout(self, conditions, **kwargs):
        kwargs.setdefault('store_chain', False)
        trace = super().sample_rollout(conditions, **kwargs)
        trace.update(algorithm=ALGORITHM, action_x0=trace['normalized'].detach().clone())
        return trace

    def draw_mc(self, action_x0, generator):
        """独立 CPU Generator，每动作固定抽样次序，不受微批、设备或环境 RNG 影响。"""
        if generator.device.type != 'cpu':
            raise ValueError('Recoverable RoboFFT MC generator must be an independent CPU generator')
        m = self.robofft['mc_samples']
        indices, noises = [], []
        for action in action_x0:
            indices.append(torch.randint(9, 19, (m,), generator=generator))
            noises.append(torch.randn((m, *action.shape), generator=generator))
        sampling_indices = torch.stack(indices)
        original_timesteps = torch.tensor(self.timestep_map)[sampling_indices]
        return dict(eps=torch.stack(noises), sampling_indices=sampling_indices,
                    original_timesteps=original_timesteps)

    @checked_policy_phase
    def losses(self, conditions, action_x0, eps, sampling_indices, original_timesteps):
        """计算 [动作, MC] 旧/新代理损失；不对 MC 求均值、不做 PPO clip。"""
        self._prepare_actor()
        b, m = eps.shape[:2]
        if (action_x0.shape != (b, self.generation_frames, 30) or
                eps.shape != (b, m, self.generation_frames, 30) or
                sampling_indices.shape != (b, m) or original_timesteps.shape != (b, m)):
            raise ValueError('MC tensors must align with complete normalized qpos30 actions')
        device = action_x0.device
        indices = sampling_indices.to(device)
        timesteps = original_timesteps.to(device)
        if not bool(((indices >= 9) & (indices <= 18)).all()) or not torch.equal(
                torch.tensor(self.timestep_map, device=device)[indices], timesteps):
            raise ValueError('Sampling index, original timestep and MC number must not be conflated')
        expanded = {k: v[:, None].expand(-1, m, *v.shape[1:]).flatten(0, 1) for k, v in conditions.items()}
        with torch.autocast(device_type=device.type, enabled=False):
            prepared = self.prepare_conditions(expanded, include_frozen=False)
            adapted = prepared['adapted']
            x0 = action_x0.detach()[:, None].expand(-1, m, -1, -1).flatten(0, 1)
            noise = eps.detach().to(device).flatten(0, 1)
            alpha = self.original_alpha.to(device)[timesteps.flatten()].float()[:, None, None]
            state = q_sample(x0, noise, alpha, adapted['known_x'], adapted['known_mask'], adapted['future_valid'])
            prediction = self._predict_prepared(prepared, state, timesteps.flatten())['pred_x_start']
            free = adapted['future_valid'][..., None] & ~adapted['known_mask']
            weights = self.delta_weights.to(device)[indices.flatten() - 9]
            loss = epsilon_loss(state, prediction, noise, alpha, weights, free).reshape(b, m)
            if not bool(torch.isfinite(loss).all()):
                raise FloatingPointError('Nonfinite RoboFFT surrogate loss')
            return loss
