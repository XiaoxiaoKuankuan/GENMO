"""Stage 1 多卡加权采样与可恢复的数据随机性管理。

本模块继续使用训练入口按四库比例分配的逐索引权重：上游 Dataset 已把动作时长展开为
duration-aware 索引，因此这里不重新计算来源权重，也不再套一层 DistributedSampler。
所有 rank 用相同 seed 和 epoch 生成一个全局有放回抽样序列，再按 rank 分片；
num_samples 始终表示每个 rank 每个 epoch 的样本数，而非全局样本数。相同动作索引可因
有放回抽样重复出现，但各 rank 消费的全局抽样位置互不重叠。

checkpoint 由训练入口保存 epoch 和每个 rank 已实际消费的样本数，恢复时调用
set_epoch(epoch, start_index)。采样器自身不会根据 DataLoader 的预取进度推进 offset。
正式训练可启用 emit_draw_keys，并配合 DeterministicDrawDataset：每次抽样根据稳定的
位置标识临时设置 Python、NumPy、PyTorch CPU 随机状态，读取原 Dataset 后原样恢复。
这样随机 decision frame 不依赖 worker 数量、预取或恢复时的环境随机状态；原第 3 步
样本字段和 provenance 保持不变，也不会消耗训练扩散噪声使用的 CPU 随机状态。
本模块不访问 CUDA RNG，不要求或初始化分布式进程组，不管理 optimizer 或 checkpoint。
"""

from __future__ import annotations

import hashlib
import numbers
import random
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

DrawKey = tuple[int, int, int, int]


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    """拒绝隐式取整和 bool，避免恢复位置或分片参数被悄悄修改。"""

    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


class ResumableDistributedWeightedSampler(Sampler[int | DrawKey]):
    """从相同全局加权抽样序列分片，支持每 rank 的已消费位置恢复。"""

    def __init__(
        self,
        weights: Sequence[float] | torch.Tensor,
        num_samples: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        *,
        emit_draw_keys: bool = False,
    ) -> None:
        self.num_samples = _integer(num_samples, "num_samples", minimum=1)
        self.seed = _integer(seed, "seed")
        self.rank = _integer(rank, "rank")
        self.world_size = _integer(world_size, "world_size", minimum=1)
        if self.rank >= self.world_size:
            raise ValueError("rank must be smaller than world_size")
        if self.seed > 2**63 - 1:
            raise ValueError("seed must be <= 2**63 - 1")
        if not isinstance(emit_draw_keys, bool):
            raise ValueError("emit_draw_keys must be a bool")
        self.emit_draw_keys = emit_draw_keys
        self.weights = torch.as_tensor(weights, dtype=torch.double, device="cpu").detach().clone()
        if self.weights.ndim != 1 or self.weights.numel() == 0:
            raise ValueError("weights must be a nonempty one-dimensional sequence")
        if not torch.isfinite(self.weights).all() or (self.weights < 0).any():
            raise ValueError("weights must be finite and nonnegative")
        total = self.weights.sum()
        if not torch.isfinite(total) or total <= 0:
            raise ValueError("weights must have a finite positive sum")
        self.epoch = 0
        self.start_index = 0

    def set_epoch(self, epoch: int, start_index: int = 0) -> None:
        """设置下一次迭代的 epoch 和每 rank 的已消费样本数。"""

        epoch = _integer(epoch, "epoch")
        start_index = _integer(start_index, "start_index")
        if epoch > 2**63 - 1:
            raise ValueError("epoch must be <= 2**63 - 1")
        if start_index > self.num_samples:
            raise ValueError("start_index cannot exceed per-rank num_samples")
        self.epoch = epoch
        self.start_index = start_index

    def __iter__(self) -> Iterator[int | DrawKey]:
        epoch, start_index = self.epoch, self.start_index
        if start_index == self.num_samples:
            return
        generator = torch.Generator(device="cpu").manual_seed(self.seed + epoch)
        global_indices = torch.multinomial(
            self.weights,
            self.num_samples * self.world_size,
            replacement=True,
            generator=generator,
        )
        local_indices = global_indices[self.rank :: self.world_size]
        for position in range(start_index, self.num_samples):
            index = int(local_indices[position])
            if self.emit_draw_keys:
                yield (index, epoch, self.rank, position)
            else:
                yield index

    def __len__(self) -> int:
        return self.num_samples - self.start_index


class DeterministicDrawDataset(Dataset):
    """为原 Dataset 的单次读取绑定稳定随机状态，不增加或重写样本字段。"""

    def __init__(self, inner: Dataset, seed: int) -> None:
        self.inner = inner
        self.seed = _integer(seed, "seed")

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, draw_key: DrawKey) -> Any:
        if not isinstance(draw_key, tuple) or len(draw_key) != 4:
            raise ValueError(
                "DeterministicDrawDataset needs an (index, epoch, rank, position) draw key"
            )
        index, epoch, rank, position = (
            _integer(value, name)
            for value, name in zip(draw_key, ("index", "epoch", "rank", "position"))
        )
        if index >= len(self.inner):
            raise IndexError(f"draw index {index} outside dataset of length {len(self.inner)}")
        identity = f"{self.seed}:{index}:{epoch}:{rank}:{position}".encode("ascii")
        draw_seed = int.from_bytes(hashlib.blake2b(identity, digest_size=8).digest(), "little")
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        try:
            random.seed(draw_seed)
            np.random.seed(draw_seed % 2**32)
            cpu_generator = torch.Generator(device="cpu").manual_seed(draw_seed)
            torch.set_rng_state(cpu_generator.get_state())
            return self.inner[index]
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)


__all__ = ["DeterministicDrawDataset", "ResumableDistributedWeightedSampler"]
