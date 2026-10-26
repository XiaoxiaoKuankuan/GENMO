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
        self.episode_metadata = {}
        self._windows = {}

    def add(self, row):
        z = row.start
        if z.origin != "real" or row.end.origin != "real" or row.end.tick != z.tick+12:
            raise ValueError("回放只接收真实连续单步转移")
        if z.episode_id != row.end.episode_id or z.song_id != row.end.song_id:
            raise ValueError("转移不得跨reset或歌曲")
        if self.song_splits.setdefault(z.song_id, z.split) != z.split:
            raise ValueError("同歌泄漏到训练和验证集")
        episode = self.episodes.setdefault(z.episode_id, [])
        meta = self.episode_metadata.setdefault(z.episode_id, dict(
            true_start=z.tick == z.episode_start_tick, head_truncated=z.tick != z.episode_start_tick,
            true_terminal=False, terminal_reason=None))
        if episode:
            prev = episode[-1]
            if prev.terminated.any() or prev.end.tick != z.tick or not torch.equal(prev.end.executed_state, z.executed_state):
                raise ValueError("episode重复/不连续或终止后继续写入")
        episode.append(row)
        meta.update(true_terminal=bool(row.terminated.all()) and row.reason in ("music_end", "physical_failure"),
                    terminal_reason=row.reason)
        self._windows.clear()
        while len(self) > self.capacity:
            first = next(iter(self.episodes))
            if first == z.episode_id:
                del self.episodes[first][0]
                self.episode_metadata[first].update(head_truncated=True, true_start=False)
            else:
                del self.episodes[first]
                del self.episode_metadata[first]

    def __len__(self):
        return sum(len(e) for e in self.episodes.values())

    def candidates(self, window, split="train"):
        key = (window, split)
        if key not in self._windows:
            self._windows[key] = [episode[i:i+window] for episode in self.episodes.values()
                if episode and episode[0].start.split == split for i in range(len(episode)-window+1)
                if not any(bool(r.terminated.any()) or r.truncated for r in episode[i:i+window-1])
                and len({r.start.nominal_reference.version for r in episode[i:i+window]}) == 1
                and len({r.policy_version for r in episode[i:i+window]}) == 1]
        return self._windows[key]

    def sample_windows(self, batch, window=8, split="train", *, latest_only=False, deterministic=False):
        candidates = self.candidates(window, split)
        if candidates and latest_only:
            latest = max(r.policy_version for e in self.episodes.values() for r in e if r.start.split == split)
            candidates = [w for w in candidates if w[0].policy_version == latest]
        if not candidates:
            raise ValueError(f"没有{split}真实连续{window}步窗口")
        baseline = [w for w in candidates if all(r.policy_version == 0 for r in w)]
        recent = [w for w in candidates if any(r.policy_version > 0 for r in w)]
        if deterministic:
            # 验证固定覆盖当前策略各首歌及首尾时间，不能反复抽样直到幸运批次通过。
            groups = {}
            for w in candidates:
                groups.setdefault(w[0].start.song_id, []).append(w)
            chosen = []
            batch = max(batch, len(groups))
            for i, song in enumerate(sorted(groups)):
                group = groups[song]
                count = batch//len(groups) + int(i < batch % len(groups))
                chosen.extend(group[round(j*(len(group)-1)/max(1, count-1))] for j in range(count))
        else:
            chosen = [self.rng.choice((baseline if i % 2 == 0 else recent) or candidates) for i in range(batch)]
        def stack(getter):
            return torch.cat([torch.stack([getter(row) for row in w], 1) for w in chosen], 0)
        return {"origin": "real", "state": stack(lambda r: r.start.executed_state),
                "action": stack(lambda r: r.action), "xi": stack(lambda r: r.start.xi),
                "next_state": stack(lambda r: r.end.executed_state),
                "mask": torch.ones(batch, window, device=chosen[0][0].action.device),
                "episode_ids": [w[0].start.episode_id for w in chosen]}

    def sample_starts(self, batch, split="train"):
        candidates = [r for episode in self.episodes.values() for r in episode
                      if r.start.split == split and r.start.tick+144 <= r.start.nominal_reference.upper
                      and (r.start.plan_boundary_tick is None or r.start.tick+12 <= r.start.plan_boundary_tick)]
        if not candidates:
            raise ValueError("缺少合法真实BPTT起点")
        latest = max(r.policy_version for r in candidates)
        baseline = [r for r in candidates if r.policy_version == 0]
        recent = [r for r in candidates if r.policy_version == latest]
        return [self.rng.choice((baseline if i % 2 == 0 else recent) or candidates).start for i in range(batch)]

    def complete_episodes(self, split):
        return [e for key, e in self.episodes.items() if e and e[0].start.split == split
                and self.episode_metadata[key]["true_start"] and not self.episode_metadata[key]["head_truncated"]
                and self.episode_metadata[key]["true_terminal"]]

    def state_dict(self):
        return dict(capacity=self.capacity, episodes=dict(self.episodes), episode_metadata=self.episode_metadata,
                    song_splits=self.song_splits, rng=self.rng.getstate())

    def load_state_dict(self, payload):
        self.capacity, self.episodes = payload["capacity"], OrderedDict(payload["episodes"])
        self.song_splits = payload["song_splits"]
        if "episode_metadata" not in payload:
            raise ValueError("旧回放未证明完整MC起点，不自动迁移为完整episode")
        self.episode_metadata = payload["episode_metadata"]
        self._windows.clear()
        self.rng.setstate(payload["rng"])
