"""第九步独立上层价值网络：音乐、真实 proprio48 历史、已承诺前缀及任务时间。

Critic 不引用 Actor 或 GMT 参数，不读取本轮刚采样动作、去噪链、执行奖励、未来真实
观测或特权机器人状态。历史使用带决策相对时间的 49 维 GRUCell，音乐及逐坐标已知
前缀分别使用轻量 MLP 和有效池化；三个 128 维编码加真实音乐剩余秒数得到 385 维，
再通过 256→128→1 回归未来累计回报。行政采集额度不能代替音乐剩余时长。

统计量由调用方按已核验 Stage1 数值传入并复制为本网络只读 buffer；不共享可训练对象。
所有输入先屏蔽无效占位，P=0、空历史和无音乐均显式输出零编码，避免 padding 或
未知占位影响价值。损失和固定回报目标由训练器计算，本模块仅提供可微 value。
"""
from __future__ import annotations

import torch
from torch import nn

from gem.closedloop.contracts import STAGE1_CONDITION_KEYS, validate_stage1_condition_batch


class UpperCritic(nn.Module):
    def __init__(self, *, qpos_mean=None, qpos_std=None, proprio_scales=(1., 1., 1., 1.),
                 time_scale_seconds=30.):
        super().__init__()
        mean = torch.zeros(30) if qpos_mean is None else torch.as_tensor(qpos_mean, dtype=torch.float32).detach().clone().reshape(-1)
        std = torch.ones(30) if qpos_std is None else torch.as_tensor(qpos_std, dtype=torch.float32).detach().clone().reshape(-1)
        if mean.shape != (30,) or std.shape != (30,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("critic requires finite qpos30 mean and positive std")
        scales = torch.as_tensor(proprio_scales, dtype=torch.float32)
        if scales.shape != (4,) or not torch.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError("critic requires four positive physical scales")
        if not torch.isfinite(torch.tensor(time_scale_seconds)) or time_scale_seconds <= 0:
            raise ValueError("time scale must be finite and positive")
        self.register_buffer("qpos_mean", mean)
        self.register_buffer("qpos_std", std)
        self.register_buffer("proprio_scale", torch.repeat_interleave(scales, torch.tensor([3, 3, 21, 21])))
        self.time_scale_seconds = float(time_scale_seconds)
        self.history_encoder = nn.GRUCell(49, 128)
        self.music_encoder = nn.Sequential(nn.Linear(35, 128), nn.SiLU(), nn.Linear(128, 128))
        self.prefix_encoder = nn.Sequential(nn.Linear(60, 128), nn.SiLU(), nn.Linear(128, 128))
        self.value_head = nn.Sequential(nn.Linear(385, 256), nn.SiLU(), nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1))

    @staticmethod
    def _pool(values, valid):
        masked = torch.where(valid[..., None], values, 0.)
        return masked.sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1).to(values)

    def forward(self, context, remaining_music_seconds):
        if set(context) != set(STAGE1_CONDITION_KEYS):
            raise ValueError("Critic context must contain only the original ten Actor condition fields")
        validate_stage1_condition_batch(context, history_steps=50)
        device, dtype = self.qpos_mean.device, self.qpos_mean.dtype
        values = {key: value.to(device=device) for key, value in context.items()}
        batch = values["music_features"].shape[0]
        remaining = torch.as_tensor(remaining_music_seconds, device=device, dtype=dtype).reshape(-1)
        if remaining.numel() == 1 and batch == 1:
            remaining = remaining.reshape(1)
        if remaining.shape != (batch,) or not torch.isfinite(remaining).all() or (remaining < 0).any():
            raise ValueError("remaining_music_seconds must be finite nonnegative [B]")
        h_valid = values["proprio_history_valid"]
        history = torch.where(h_valid[..., None], values["proprio_history"].to(dtype), 0.) / self.proprio_scale
        relative = (values["proprio_history_times"] - values["decision_time"][:, None]).to(dtype)
        relative = torch.where(h_valid, relative, 0.)
        features = torch.cat((history, relative[..., None]), dim=-1)
        h = history.new_zeros((batch, 128))
        for index in range(50):
            proposed = self.history_encoder(features[:, index], h)
            h = torch.where(h_valid[:, index, None], proposed, h)
        m_valid = values["music_valid"] & values["future_valid"]
        music = torch.where(m_valid[..., None], values["music_features"].to(dtype), 0.)
        music_code = self._pool(self.music_encoder(music), m_valid)
        mask = values["known_qpos30_mask"] & values["future_valid"][..., None]
        physical = torch.where(mask, values["known_qpos30"].to(dtype), 0.)
        known = torch.where(mask, (physical - self.qpos_mean) / self.qpos_std, 0.)
        prefix_code = self._pool(self.prefix_encoder(torch.cat((known, mask.to(dtype)), -1)), mask.any(-1))
        combined = torch.cat((music_code, h, prefix_code, remaining[:, None] / self.time_scale_seconds), -1)
        result = self.value_head(combined).squeeze(-1)
        if not torch.isfinite(result).all():
            raise FloatingPointError("Critic produced a nonfinite value")
        return result
