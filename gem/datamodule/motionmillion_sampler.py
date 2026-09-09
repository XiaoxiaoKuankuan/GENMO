# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""MotionMillion 分片感知、无重复的确定性 DDP sampler。

普通 DistributedSampler 会在百万级全局索引上随机打散，使相邻 batch 频繁跨越 motion
shard。该 sampler 先打乱 shard，再打乱 shard 内样本，保持磁盘局部性；完整全局序列
随后按 rank stride 切分。训练固定 ``drop_last=True``，因此不同 rank 没有为补齐而
复制的样本，丢弃数量可以精确报告并在每个 epoch 通过 ``seed + epoch`` 复现。
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler


class ShardAwareDistributedSampler(Sampler[int]):
    """按 shard 分组打乱并分发到 DDP rank。"""

    def __init__(
        self,
        dataset: Dataset,
        *,
        seed: int = 20260909,
        shuffle: bool = True,
        drop_last: bool = True,
        rank: int | None = None,
        num_replicas: int | None = None,
    ) -> None:
        getter = getattr(dataset, "sample_shard_ids", None)
        if getter is None:
            raise TypeError(f"{type(dataset).__name__} 未实现 sample_shard_ids()")
        shard_ids = np.asarray(getter(), dtype=np.int64)
        if shard_ids.ndim != 1 or len(shard_ids) != len(dataset):
            raise ValueError("sample_shard_ids 必须是一维且长度等于 Dataset")
        if len(shard_ids) == 0 or (shard_ids < 0).any():
            raise ValueError("sample_shard_ids 不能为空或包含负数")
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        self.dataset = dataset
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.epoch = 0
        if self.num_replicas <= 0 or not 0 <= self.rank < self.num_replicas:
            raise ValueError("rank/num_replicas 无效")

        self.groups: list[torch.Tensor] = []
        for shard_id in sorted(np.unique(shard_ids).tolist()):
            indices = np.flatnonzero(shard_ids == shard_id)
            self.groups.append(torch.from_numpy(indices.astype(np.int64, copy=False)))
        if self.drop_last:
            self.total_size = (len(dataset) // self.num_replicas) * self.num_replicas
        else:
            self.total_size = math.ceil(len(dataset) / self.num_replicas) * self.num_replicas
        self.num_samples = self.total_size // self.num_replicas
        self.dropped_samples = len(dataset) - self.total_size if self.drop_last else 0

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _global_indices(self) -> torch.Tensor:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        if self.shuffle:
            group_order = torch.randperm(len(self.groups), generator=generator).tolist()
        else:
            group_order = list(range(len(self.groups)))
        chunks = []
        for group_index in group_order:
            group = self.groups[group_index]
            if self.shuffle and len(group) > 1:
                group = group[torch.randperm(len(group), generator=generator)]
            chunks.append(group)
        indices = torch.cat(chunks)
        if self.drop_last:
            return indices[: self.total_size]
        if len(indices) < self.total_size:
            # 非训练用途显式允许确定性补齐；训练配置固定 drop_last=True。
            missing = self.total_size - len(indices)
            repeats = math.ceil(missing / len(indices))
            padding = indices.repeat(repeats)[:missing]
            indices = torch.cat([indices, padding])
        return indices

    def __iter__(self) -> Iterator[int]:
        global_indices = self._global_indices()
        rank_indices = global_indices[self.rank : self.total_size : self.num_replicas]
        if len(rank_indices) != self.num_samples:
            raise RuntimeError("DDP shard sampler rank 长度不一致")
        return iter(rank_indices.tolist())


__all__ = ["ShardAwareDistributedSampler"]
