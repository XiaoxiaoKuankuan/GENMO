"""固定音频目标与因果、纯Torch的执行节拍奖励。

EDGE35第34列只在音频侧离线取拍点；运动侧使用执行关节速度的连续能量谷，不调用
布尔节拍指标。n时提取n-2的事件，立即结算该事件的非拍点成本；拍点的覆盖项在
固定容差窗关闭后结算一次，允许音乐预知，不允许未来执行状态。用soft union匹配
事件而不要求高斯肩部逐点出现速度谷；所有记忆仍来自尾随20点执行速度并保留图。
无beat/无效/不完整窗口不补分，s0固定；此v2奖励不能等价恢复旧逐点BCE训练。
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
    task_end_tick: int | None = None
    condition_kind: str = "music"

    @property
    def end_tick(self):
        return self.start_tick + (self.features.shape[1] - 1) * MUSIC_TICKS

    @property
    def control_end_tick(self):
        return self.end_tick if self.task_end_tick is None else self.task_end_tick

    @classmethod
    def without_audio(cls, end_tick, *, like, sigma_s=.04):
        """仅在内存表示无音频mask；任务期限独立保存，不写伪EDGE文件。"""
        count = max(2, (int(end_tick)+MUSIC_TICKS-1)//MUSIC_TICKS+1)
        values = like.new_zeros(len(like), count, 35)
        valid = torch.zeros(values.shape[:2], device=like.device, dtype=torch.bool)
        return replace(cls.from_features(values, valid=valid, sigma_s=sigma_s),
                       task_end_tick=int(end_tick), condition_kind="text")

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
        times = times[None].expand(len(values), -1)
        return values, valid, times if self.condition_kind == "music" else torch.zeros_like(times)


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
    def __init__(self, s0, *, tau=.05, margin=.01, epsilon=1e-6, activity_fraction=.7, activity_floor=.05,
                 match_tolerance_s=.06, event_activity_floor=.1, extra_event_weight=1.):
        self.s0 = torch.as_tensor(s0).detach().clone()
        if self.s0.numel() not in (1, 21) or not torch.isfinite(self.s0).all() or (self.s0 < 1e-4).any():
            raise ValueError("s0必须来自固定训练基线，形状为标量或21维且>=1e-4")
        if tau <= 0 or not 0 < epsilon < .1 or activity_floor <= 0:
            raise ValueError("音乐数值参数无效")
        self.tau, self.margin, self.epsilon = tau, margin, epsilon
        self.activity_fraction, self.activity_floor = activity_fraction, activity_floor
        self.tolerance_ticks = round(match_tolerance_s * CLOCK_HZ)
        if not 12 <= self.tolerance_ticks <= 72 or event_activity_floor <= 0 or margin <= 0:
            raise ValueError("事件匹配窗口必须落在20点因果历史支持内，活动/显著性尺度须为正")
        self.event_activity_floor, self.extra_event_weight = event_activity_floor, extra_event_weight

    def events(self, energy):
        """连续双侧谷响应：常值严格零，幅度门槛拒绝微小噪声伪事件。"""
        left, center, right = energy[:, :-4], energy[:, 2:-2], energy[:, 4:]
        dl, dr = left-center, right-center
        valley = torch.sigmoid((dl-self.margin)/self.tau) * torch.sigmoid((dr-self.margin)/self.tau)
        # 减去恒速响应，禁止斜坡两侧仅靠速度变化幅度产生伪事件。
        baseline = torch.sigmoid(energy.new_tensor(-self.margin/self.tau)).square()
        valley = torch.relu(valley-baseline)/(1-baseline)
        flank = (left+right)*.5
        activity = flank.square()/(flank.square()+self.event_activity_floor**2)
        return valley * activity

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
        events = self.events(memory.energy)
        event_ticks = memory.ticks[2:-2]
        event_valid = memory.valid.unfold(1, 5, 1).all(-1) & audio.target(event_ticks)[1]
        events = events * event_valid
        p = events[:, -1]
        # 每事件按最近音频拍点容差匹配；音频固定密度用于跨歌曲量纲平衡。
        density = memory.energy.new_tensor([len(bs)/max(1, (audio.end_tick-audio.start_tick)//12+1)
                                           for bs in audio.beat_ticks]).clamp_min(1e-4)
        precision_cost = p * (1-b[:, 0]) / density
        coverage_cost, duplicate_cost, paid_beats = [], [], []
        for i, beats in enumerate(audio.beat_ticks):
            closed = beats[(beats+self.tolerance_ticks > center-CONTROL_TICKS) &
                           (beats+self.tolerance_ticks <= center)]
            loss, duplicates = p[i]*0, p[i]*0
            count_beats = 0
            for beat in closed:
                inside = (event_ticks-beat).abs() <= self.tolerance_ticks
                # 首/尾缺观测或音频洞不能用零状态补完整beat窗。
                if not bool(inside.any()) or not bool(event_valid[i, inside].all()):
                    continue
                if int(event_ticks[0]) > int(beat)-self.tolerance_ticks:
                    continue
                kernel = torch.exp(-.5*((event_ticks[inside]-beat)/(CLOCK_HZ*audio.sigma_s))**2)
                matched = (events[i, inside]*kernel).clamp(max=1-self.epsilon)
                coverage = -torch.expm1(torch.log1p(-matched).sum())
                loss = loss + 1-coverage
                # 同一拍点容差内大量事件不能通过coverage饱和获利。
                duplicates = duplicates + torch.relu(matched.sum()-1).square()
                count_beats += 1
            coverage_cost.append(loss/density[i])
            duplicate_cost.append(duplicates/density[i])
            paid_beats.append(count_beats)
        coverage_cost, duplicate_cost = torch.stack(coverage_cost), torch.stack(duplicate_cost)
        reward = -(precision_cost+self.extra_event_weight*duplicate_cost+coverage_cost)
        reward = torch.where(valid, reward, torch.zeros_like(reward))
        count = memory.valid.sum(-1).clamp_min(1)
        actual = (memory.activity * memory.valid).sum(-1) / count
        nominal = (memory.nominal_activity * memory.valid).sum(-1) / count
        activity = (torch.relu(self.activity_fraction * nominal - actual) / nominal.clamp_min(self.activity_floor))**2
        activity = activity * memory.valid.all(-1) * (nominal > self.activity_floor)
        return replace(memory, last_paid_tick=center), reward, activity, {
            "music_probability": p, "music_target": b[:, 0], "music_valid": valid,
            "music_center_tick": center, "music_precision_cost": precision_cost,
            "music_coverage_cost": coverage_cost, "music_duplicate_cost": duplicate_cost,
            "music_settled_beats": memory.energy.new_tensor(paid_beats),
            "music_coverage_latest_tick": center-self.tolerance_ticks,
            "activity_actual": actual, "activity_nominal": nominal}


def calibrate_s0(baseline_joint_velocities):
    """只供零Adapter真实训练集基线离线标定；调用方保存来源并冻结结果。"""
    qd = baseline_joint_velocities.detach().reshape(-1, 21)
    if not len(qd) or not torch.isfinite(qd).all():
        raise ValueError("s0标定缺少有限真实基线速度")
    return qd.square().mean(0).sqrt().clamp_min(.05)
