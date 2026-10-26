"""固定音频目标与因果、纯Torch的执行节拍奖励。

EDGE35第34列只在音频侧离线取拍点；运动侧使用执行关节速度的连续能量谷，不调用
布尔节拍指标。推进到n时只结算n-2，用n-4/n-2/n；尾随20点活动统计随增强状态保存。
无beat、无效音频和不足五点各自mask，不复制未来状态补分；s0与音频均值始终固定。
"""

from dataclasses import dataclass, replace

import torch

from .contracts import CONTROL_TICKS, CLOCK_HZ, MUSIC_TICKS


@dataclass(frozen=True)
class AudioTrack:
    features: torch.Tensor
    valid: torch.Tensor
    start_tick: int
    beat_ticks: tuple[torch.Tensor, ...]
    mean_target: torch.Tensor
    has_beats: torch.Tensor
    sigma_s: float = .04

    @property
    def end_tick(self):
        return self.start_tick + (self.features.shape[1] - 1) * MUSIC_TICKS

    @classmethod
    def from_features(cls, features, *, start_tick=0, valid=None, sigma_s=.04):
        if features.ndim != 3 or features.shape[-1] != 35 or features.shape[1] < 2:
            raise ValueError("音乐必须是[B,N>=2,35]原30Hz特征")
        if not torch.isfinite(features).all() or sigma_s <= 0:
            raise ValueError("音乐必须有限且sigma为正")
        features = features.detach().clone()
        valid = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device) if valid is None else valid.bool().clone()
        if valid.shape != features.shape[:2]:
            raise ValueError("音频有效mask形状不匹配")
        beats = tuple(start_tick + torch.nonzero((row[:, 34] > .5) & mask).flatten() * MUSIC_TICKS
                      for row, mask in zip(features, valid))
        has = torch.tensor([len(b) > 0 for b in beats], device=features.device)
        track = cls(features, valid, start_tick, beats, features.new_zeros(len(features)), has, sigma_s)
        # 对音频自身时间网格取固定均值，平移绝对时钟不会改变mu。
        grid = torch.arange(start_tick, track.end_tick + 1, CONTROL_TICKS, device=features.device)
        b, mask = track.target(grid)
        mu = (b * mask).sum(-1) / mask.sum(-1).clamp_min(1)
        return replace(track, mean_target=mu.detach())

    def target(self, ticks):
        ticks = torch.as_tensor(ticks, device=self.features.device, dtype=torch.long).reshape(-1)
        idx = (ticks - self.start_tick) // MUSIC_TICKS
        inside = (ticks >= self.start_tick) & (ticks <= self.end_tick)
        mask = self.valid[:, idx.clamp(0, self.features.shape[1] - 1)] & inside[None] & self.has_beats[:, None]
        rows = []
        for beats in self.beat_ticks:
            if len(beats):
                # 最近固定拍点等价于高斯包络的max；searchsorted避免全曲二次内存。
                right = torch.searchsorted(beats, ticks).clamp_max(len(beats)-1)
                left = (torch.searchsorted(beats, ticks)-1).clamp_min(0)
                distance = torch.minimum((ticks-beats[left]).abs(), (ticks-beats[right]).abs()).to(self.features)
                rows.append(torch.exp(-.5 * (distance / (CLOCK_HZ * self.sigma_s))**2))
            else:
                rows.append(self.features.new_zeros(len(ticks)))
        return torch.stack(rows) * mask, mask

    def preview(self, tick):
        idx = (tick - self.start_tick) // MUSIC_TICKS + torch.arange(16, device=self.features.device)
        mask = (idx >= 0) & (idx < self.features.shape[1])
        clipped = idx.clamp(0, self.features.shape[1] - 1)
        valid = self.valid[:, clipped] & mask[None]
        values = self.features[:, clipped] * valid[..., None]
        times = (self.start_tick + idx * MUSIC_TICKS - tick).to(values) / CLOCK_HZ
        return values, valid, times[None].expand(len(values), -1)


@dataclass(frozen=True)
class MusicState:
    energy: torch.Tensor
    activity: torch.Tensor
    nominal_activity: torch.Tensor
    valid: torch.Tensor
    ticks: torch.Tensor
    last_paid_tick: int | None = None

    @classmethod
    def empty(cls, like):
        values = like.new_zeros(len(like), 20)
        return cls(values, values.clone(), values.clone(), torch.zeros_like(values, dtype=torch.bool),
                   torch.full((20,), -10**12, device=like.device, dtype=torch.long))

    def features(self):
        return torch.stack((self.energy, self.activity, self.nominal_activity, self.valid.to(self.energy)), -1)


class MusicReward:
    def __init__(self, s0, *, tau=.05, margin=.01, epsilon=1e-6, activity_fraction=.7, activity_floor=.05):
        self.s0 = torch.as_tensor(s0).detach().clone()
        if self.s0.numel() not in (1, 21) or not torch.isfinite(self.s0).all() or (self.s0 < 1e-4).any():
            raise ValueError("s0必须来自固定训练基线，形状为标量或21维且>=1e-4")
        if tau <= 0 or not 0 < epsilon < .1 or activity_floor <= 0:
            raise ValueError("音乐数值参数无效")
        self.tau, self.margin, self.epsilon = tau, margin, epsilon
        self.activity_fraction, self.activity_floor = activity_fraction, activity_floor

    def energy(self, qd):
        return (torch.sqrt((qd / self.s0.to(qd))**2 + self.epsilon**2) - self.epsilon).mean(-1)

    def push(self, memory, qd, nominal_qd, tick):
        if memory.valid.any() and tick != int(memory.ticks[-1]) + CONTROL_TICKS:
            raise ValueError("音乐运动历史不连续；reset需要新建MusicState")
        def shift(old, new):
            return torch.cat((old[:, 1:], new[:, None]), 1)
        return replace(memory, energy=shift(memory.energy, self.energy(qd)),
                       activity=shift(memory.activity, qd.abs().mean(-1)),
                       nominal_activity=shift(memory.nominal_activity, nominal_qd.abs().mean(-1)),
                       valid=shift(memory.valid, torch.ones(len(qd), dtype=torch.bool, device=qd.device)),
                       ticks=torch.cat((memory.ticks[1:], memory.ticks.new_tensor([tick]))))

    def settle(self, memory, audio):
        center = int(memory.ticks[-1]) - 2 * CONTROL_TICKS
        if memory.last_paid_tick is not None and center <= memory.last_paid_tick:
            raise ValueError("同一音乐中心不得重复结算")
        valid = memory.valid[:, -5:].all(-1)
        b, audio_valid = audio.target([center])
        valid = valid & audio_valid[:, 0]
        e = memory.energy
        p = torch.sigmoid((e[:, -5] - e[:, -3] - self.margin) / self.tau) * torch.sigmoid((e[:, -1] - e[:, -3] - self.margin) / self.tau)
        p_eps = self.epsilon + (1 - 2 * self.epsilon) * p
        mu = audio.mean_target.clamp(1e-4, 1-1e-4)
        loss = -.5 * b[:, 0] / mu * p_eps.log() - .5 * (1-b[:, 0]) / (1-mu) * torch.log1p(-p_eps)
        reward = torch.where(valid, -loss, torch.zeros_like(loss))
        count = memory.valid.sum(-1).clamp_min(1)
        actual = (memory.activity * memory.valid).sum(-1) / count
        nominal = (memory.nominal_activity * memory.valid).sum(-1) / count
        activity = (torch.relu(self.activity_fraction * nominal - actual) / nominal.clamp_min(self.activity_floor))**2
        activity = activity * memory.valid.all(-1) * (nominal > self.activity_floor)
        return replace(memory, last_paid_tick=center), reward, activity, {
            "music_probability": p, "music_target": b[:, 0], "music_valid": valid,
            "music_center_tick": center, "activity_actual": actual, "activity_nominal": nominal}


def calibrate_s0(baseline_joint_velocities):
    """只供零Adapter真实训练集基线离线标定；调用方保存来源并冻结结果。"""
    qd = baseline_joint_velocities.detach().reshape(-1, 21)
    if not len(qd) or not torch.isfinite(qd).all():
        raise ValueError("s0标定缺少有限真实基线速度")
    return qd.square().mean(0).sqrt().clamp_min(.05)
