"""独立PlanAwareCritic：对状态、历史、有效参考和调度保留输入梯度。

每类输入形成独立投影token，256宽两层四头Transformer和Value Token输出标量。
缓存包含仍可能被读取的全部源点、halo、contact、相对时间与来源版本；音乐和剩余
时间、延迟历史共同进入状态。支持任意合法50Hz时刻；在途请求额外编码原请求完整状态与噪声。未知完成时间不入网，
不加载旧价值或no-grad编码。
"""

import torch
from torch import nn


class PlanAwareCritic(nn.Module):
    def __init__(self, scheduler, width=256, layers=2, heads=4):
        super().__init__()
        self.scheduler = scheduler
        self.state_proj = nn.Linear(101, width)
        self.sm_proj = nn.Linear(76, width)
        self.history_proj = nn.Linear(50, width)
        self.reference_proj = nn.Linear(33, width)
        self.music_proj = nn.Linear(37, width)
        self.schedule_proj = nn.Linear(50, width)
        self.noise_proj = nn.Linear(30, width)
        self.pending_type = nn.Parameter(torch.zeros(1, 1, width))
        self.value_token = nn.Parameter(torch.zeros(1, 1, width))
        self.types = nn.Parameter(torch.zeros(7, width))
        layer = nn.TransformerEncoderLayer(width, heads, width * 4, dropout=0., batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))

    def tokens(self, z):
        z.validate()
        b, dev = len(z.state), z.state.device
        ones = lambda n: torch.ones(b, n, dtype=torch.bool, device=dev)
        current = self.state_proj(torch.cat((z.state, z.xi, z.previous_action), -1))[:, None]
        sm_time = torch.arange(-9, 1, device=dev).to(z.state)[None, :, None].expand(b, -1, -1) * .02
        sm = self.sm_proj(torch.cat((z.sm_history, sm_time), -1))
        hist_time = torch.arange(-49, 1, device=dev).to(z.state)[None, :, None].expand(b, -1, -1) * .02
        hist = self.history_proj(torch.cat((torch.where(z.genmo_valid[..., None], z.genmo_history, 0.),
                                            z.genmo_valid[..., None].to(z.state), hist_time), -1))
        cache = z.reference
        rt = ((cache.ticks - z.tick).to(z.state) / 600)[None, :, None].expand(b, -1, -1)
        versions = (cache.source_versions - cache.version).to(z.state)[None, :, None].expand(b, -1, -1)
        ref = self.reference_proj(torch.cat((cache.world_qpos, cache.contact, rt,
                                             torch.ones_like(rt), versions), -1))
        future_ticks = z.tick + torch.arange(60, device=dev) * 20
        indices = (future_ticks - z.music_start_tick) // 20
        valid_music = (indices >= 0) & (indices < z.music.shape[1])
        music = z.music[:, indices.clamp(0, z.music.shape[1] - 1)]
        music = torch.where(valid_music[None, :, None], music, 0.)
        mt = ((future_ticks - z.tick).to(z.state) / 600)[None, :, None].expand(b, -1, -1)
        music = self.music_proj(torch.cat((music, mt, valid_music[None, :, None].expand(b, -1, -1).to(music)), -1))
        old = z.latency_history[-16:]
        scalar = [(z.tick - z.music_start_tick) / 600, (z.music_end_tick - z.tick) / 600,
                  (z.protected_until - z.tick) / 600, (cache.lower - z.tick) / 600,
                  (cache.upper - z.tick) / 600, z.consecutive_timeouts, z.decision_id,
                  self.scheduler.estimated_latency(z) / 600]
        pending = z.pending_request
        if pending is None:
            extra = [0., cache.version] + [0.]*8
        else:
            t = pending.ticket
            extra = [1., cache.version, (z.tick-t.request_tick)/600, (t.predicted_ready_tick-z.tick)/600,
                     (t.first_mutable_source_tick-z.tick)/600, (t.first_influence_tick-z.tick)/600,
                     t.prefix_frames/60, t.base_version-cache.version, pending.actor_version, t.decision_id]
        timing = z.state.new_tensor(scalar + extra + [0.] * (16 - len(old)) + [x / 600 for x in old] +
                                    [0.] * (16 - len(old)) + [1.] * len(old))
        schedule = self.schedule_proj(timing[None].expand(b, -1))[:, None]
        tokens = [self.value_token.expand(b, -1, -1), current, sm, hist, ref, music, schedule]
        mask = torch.cat((ones(1), ones(1), ones(10), z.genmo_valid, ones(ref.shape[1]),
                          valid_music[None].expand(b, -1), ones(1)), 1)
        return torch.cat([x + self.types[i] for i, x in enumerate(tokens)], 1), mask

    def forward(self, z):
        tokens, mask = self.tokens(z)
        if z.pending_request is not None:
            # 条件仍为请求当时的快照；没有当次实测延迟，也没有未来真实状态。
            pending, visible = self.tokens(z.pending_request.snapshot)
            noise = self.noise_proj(z.pending_request.noise)
            tokens = torch.cat((tokens, pending+self.pending_type, noise+self.pending_type), 1)
            mask = torch.cat((mask, visible, torch.ones(noise.shape[:2], device=mask.device, dtype=torch.bool)), 1)
        encoded = self.encoder(tokens, src_key_padding_mask=~mask)
        return self.head(encoded[:, 0]).squeeze(-1)
