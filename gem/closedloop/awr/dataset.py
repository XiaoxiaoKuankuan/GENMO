"""固定Replay快照到Actor和Critic训练Batch的适配。

读取已经完整校验的Episode；Actor只遍历实际影响执行且目标一致的记录，Critic
保留完整回报链上的全部状态。原始条件直接拼接，不重算音乐时间、不从真实机器人
轨迹构造目标、不补Contact/ground标签。权重通过唯一sample_id严格对齐且整轮固定。
"""
from __future__ import annotations

import time
import torch


class ReplayDataset:
    def __init__(self, episodes, labels=None, *, unit_weights=False):
        self.episodes = list(episodes)
        self.rows, self.returns, self.sources = [], [], []
        for episode in self.episodes:
            episode.validate()
            if episode.returns is None:
                raise ValueError("Replay needs audited complete returns")
            for index, row in enumerate(episode.rows):
                if row.actor_valid:
                    self.rows.append(row)
                    self.returns.append(episode.returns[index])
                    self.sources.append(episode.source)
        if not self.rows:
            raise ValueError("No actually executed AWR references")
        self.sample_ids = [r.sample_id for r in self.rows]
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("Repeated Replay Actor sample")
        self.returns = torch.stack(self.returns).double()
        self.weights = torch.ones(len(self.rows), dtype=torch.float64)
        if labels is not None:
            lookup = {key: i for i, key in enumerate(labels.sample_ids)}
            if set(lookup) != set(self.sample_ids):
                raise ValueError("Frozen labels differ from Replay snapshot")
            order = torch.tensor([lookup[key] for key in self.sample_ids])
            if not torch.equal(labels.returns[order], self.returns):
                raise ValueError("Frozen MC labels changed")
            if not unit_weights:
                self.weights = labels.weights[order].clone()
        elif not unit_weights:
            raise ValueError("AWR Actor dataset requires frozen cross-fit weights")
        self.read_seconds = 0.

    def __len__(self):
        return len(self.rows)

    def batch(self, indices, device):
        started = time.perf_counter()
        rows = [self.rows[int(i)] for i in indices]
        if not rows:
            raise ValueError("Empty local shards must skip the forward")
        result = dict(
            conditions={k: torch.cat([r.context[k] for r in rows]).to(device) for k in rows[0].context},
            target=torch.stack([r.target_qpos30 for r in rows]).to(device),
            target_valid=torch.stack([r.target_valid for r in rows]).to(device),
            free_mask=torch.stack([r.free_mask for r in rows]).to(device),
            weights=self.weights[torch.as_tensor(indices)].to(device),
        )
        self.read_seconds += time.perf_counter()-started
        return result


def critic_batch(rows, device):
    contexts = [r.metadata["critic_context"] for r in rows]
    return ({k: torch.cat([c[k] for c in contexts]).to(device) for k in contexts[0]},
            torch.tensor([r.metadata["remaining_music_seconds"] for r in rows], device=device))
