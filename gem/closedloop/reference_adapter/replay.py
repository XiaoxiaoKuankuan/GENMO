"""真实连续转移回放与歌曲隔离的窗口采样。

入库检查来源、episode、tick和状态连续性；WM窗口绝不跨reset，尾部不足直接不用，
不把想象结果当监督标签。按歌曲固定train/validation，混合零Adapter基线与近期真实
分布；所有游标、采样RNG及窗口内容进入完整checkpoint。
"""

from collections import OrderedDict
import random

import torch


class RealReplay:
    def __init__(self, capacity=20000, seed=42):
        self.capacity, self.episodes = capacity, OrderedDict()
        self.song_splits = {}
        self.rng = random.Random(seed)

    def add(self, row):
        z = row.start
        if z.origin != "real" or row.end.origin != "real" or row.end.tick != z.tick+12:
            raise ValueError("回放只接收真实连续单步转移")
        if z.episode_id != row.end.episode_id or z.song_id != row.end.song_id:
            raise ValueError("转移不得跨reset或歌曲")
        if self.song_splits.setdefault(z.song_id, z.split) != z.split:
            raise ValueError("同歌泄漏到训练和验证集")
        episode = self.episodes.setdefault(z.episode_id, [])
        if episode:
            prev = episode[-1]
            if prev.terminated.any() or prev.end.tick != z.tick or not torch.equal(prev.end.executed_state, z.executed_state):
                raise ValueError("episode重复/不连续或终止后继续写入")
        episode.append(row)
        while len(self) > self.capacity:
            first = next(iter(self.episodes))
            if first == z.episode_id:
                del self.episodes[first][0]
            else:
                del self.episodes[first]

    def __len__(self):
        return sum(len(e) for e in self.episodes.values())

    def candidates(self, window, split="train"):
        return [episode[i:i+window] for episode in self.episodes.values()
                if episode and episode[0].start.split == split for i in range(len(episode)-window+1)
                if not any(bool(r.terminated.any()) for r in episode[i:i+window-1])]

    def sample_windows(self, batch, window=8, split="train"):
        candidates = self.candidates(window, split)
        if not candidates:
            raise ValueError(f"没有{split}真实连续{window}步窗口")
        baseline = [w for w in candidates if all(r.policy_version == 0 for r in w)]
        recent = [w for w in candidates if any(r.policy_version > 0 for r in w)]
        chosen = [self.rng.choice((baseline if i % 2 == 0 else recent) or candidates) for i in range(batch)]
        def stack(getter):
            return torch.cat([torch.stack([getter(row) for row in w], 1) for w in chosen], 0)
        return {"origin": "real", "state": stack(lambda r: r.start.executed_state),
                "action": stack(lambda r: r.action), "xi": stack(lambda r: r.start.xi),
                "next_state": stack(lambda r: r.end.executed_state),
                "mask": torch.ones(batch, window, device=chosen[0][0].action.device),
                "episode_ids": [w[0].start.episode_id for w in chosen]}

    def sample_starts(self, batch, split="train"):
        candidates = [r.start for episode in self.episodes.values() for r in episode
                      if r.start.split == split and r.start.tick+144 <= r.start.nominal_reference.upper
                      and (r.start.plan_boundary_tick is None or r.start.tick+12 <= r.start.plan_boundary_tick)]
        if not candidates:
            raise ValueError("缺少合法真实BPTT起点")
        return [self.rng.choice(candidates) for _ in range(batch)]

    def complete_episodes(self, split):
        return [e for e in self.episodes.values() if e and e[0].start.split == split and bool(e[-1].terminated.all())]

    def state_dict(self):
        return dict(capacity=self.capacity, episodes=dict(self.episodes), song_splits=self.song_splits, rng=self.rng.getstate())

    def load_state_dict(self, payload):
        self.capacity, self.episodes = payload["capacity"], OrderedDict(payload["episodes"])
        self.song_splits = payload["song_splits"]
        self.rng.setstate(payload["rng"])
