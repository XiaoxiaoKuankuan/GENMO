"""独立PlanAwareCritic：对状态、历史、有效参考和调度保留输入梯度。

每类输入形成独立投影token，256宽两层四头Transformer和Value Token输出标量。
缓存包含仍可能被读取的全部源点、halo、contact、相对时间与来源版本；音乐和剩余
时间、延迟历史共同进入状态。只定义在无pending请求边界，不加载旧价值或no-grad编码。
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
        self.schedule_proj = nn.Linear(40, width)
        self.value_token = nn.Parameter(torch.zeros(1, 1, width))
        self.types = nn.Parameter(torch.zeros(7, width))
        layer = nn.TransformerEncoderLayer(width, heads, width * 4, dropout=0., batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))

    def forward(self, z):
        z.validate(decision=True)
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
        timing = z.state.new_tensor(scalar + [0.] * (16 - len(old)) + [x / 600 for x in old] +
                                    [0.] * (16 - len(old)) + [1.] * len(old))
        schedule = self.schedule_proj(timing[None].expand(b, -1))[:, None]
        tokens = [self.value_token.expand(b, -1, -1), current, sm, hist, ref, music, schedule]
        mask = torch.cat((ones(1), ones(1), ones(10), z.genmo_valid, ones(ref.shape[1]),
                          valid_music[None].expand(b, -1), ones(1)), 1)
        encoded = self.encoder(torch.cat([x + self.types[i] for i, x in enumerate(tokens)], 1), src_key_padding_mask=~mask)
        return self.head(encoded[:, 0]).squeeze(-1)
