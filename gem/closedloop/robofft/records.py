"""单轮不可变动作、MC 噪声/时间和旧代理损失的记录与校验。

每个真实上层动作持有独立 CPU MC 记录；先完成全 rollout 的旧损失，再允许第一步
optimizer 更新。记录的 eps、采样索引、original timestep、MC 编号和 old_loss 均 detach
复制，微批/重排仅选取，不重采、不重算旧损失。张量版本与内容边界阻止原地污染。
动作仍来自最终采样，不接受 Gaussian 链伪零占位。记录只随本轮诊断持久化，完整训练
恢复保存独立 MC RNG、重建物理世界并重新采样，不复用旧 rollout。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import torch

from . import ALGORITHM


@dataclass(frozen=True)
class MCRecord:
    eps: torch.Tensor
    sampling_indices: torch.Tensor
    original_timesteps: torch.Tensor
    old_loss: torch.Tensor
    mc_numbers: torch.Tensor
    algorithm: str = ALGORITHM
    _versions: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        for name in ('eps', 'sampling_indices', 'original_timesteps', 'old_loss', 'mc_numbers'):
            object.__setattr__(self, name, getattr(self, name).detach().cpu().clone())
        object.__setattr__(self, '_versions', self.signature())

    def signature(self):
        return tuple((id(v), v._version) for v in
            (self.eps, self.sampling_indices, self.original_timesteps, self.old_loss, self.mc_numbers))

    def __getstate__(self):
        self.validate()
        return {key: value for key, value in vars(self).items() if key != '_versions'}

    def __setstate__(self, state):
        for key, value in state.items():
            object.__setattr__(self, key, value)
        object.__setattr__(self, '_versions', self.signature())
        self.validate()

    def validate(self, *, check_versions=True):
        m = len(self.old_loss)
        if check_versions and self.signature() != self._versions:
            raise ValueError('Immutable RoboFFT MC record was modified')
        if (self.algorithm != ALGORITHM or m != 16 or self.old_loss.shape != (m,) or
                self.eps.ndim != 3 or self.eps.shape[0] != m or self.eps.shape[-1] != 30 or
                self.sampling_indices.shape != (m,) or self.original_timesteps.shape != (m,) or
                self.sampling_indices.dtype != torch.int64 or self.original_timesteps.dtype != torch.int64 or
                not torch.equal(self.mc_numbers, torch.arange(m)) or
                not bool(((self.sampling_indices >= 9) & (self.sampling_indices <= 18)).all()) or
                not bool(torch.isfinite(self.eps).all() & torch.isfinite(self.old_loss).all()) or
                self.eps.requires_grad or self.old_loss.requires_grad):
            raise ValueError('Invalid fixed RoboFFT MC record')


def validate_action(row, policy=None):
    trace = row.metadata['sampler_trace']
    value = row.action_x0
    expected = row.context['future_valid'][0, :, None] & ~row.context['known_qpos30_mask'][0]
    if (row.algorithm != ALGORITHM or trace.get('algorithm') != ALGORITHM or
            row.chain is not None or row.old_log_prob is not None or
            any(k in trace for k in ('chain', 'old_log_probs', 'old_means', 'old_stds', 'sampling_noise'))):
        raise ValueError('RoboFFT requires an explicit compact action, never a fabricated Gaussian chain')
    if (value is None or value.shape != expected.shape or value.dtype != torch.float32 or
            value.requires_grad or not bool(torch.isfinite(value).all()) or
            not torch.equal(row.free_mask, expected) or
            not torch.equal(value, trace['action_x0'][0]) or
            not torch.equal(value, trace['normalized'][0]) or
            bool((value[~row.context['future_valid'][0]] != 0).any())):
        raise ValueError('Invalid final normalized action_x0 or free mask')
    if policy is not None:
        if trace['kernel_config'] != policy.kernel_config or trace['timestep_map'].tolist() != list(policy.timestep_map):
            raise ValueError('RoboFFT behavior sampling kernel differs')
        with torch.no_grad():
            adapted = policy.actor.adapt_conditions(row.context)
            known = adapted['known_mask'][0] & adapted['future_valid'][0, :, None]
            if not torch.equal(value[known], adapted['known_x'][0][known]):
                raise ValueError('Known prefix must remain clean normalized x0')
    if row.mc_record is not None:
        row.mc_record.validate()
