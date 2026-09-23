"""Stage 1 分布式加权采样与数据恢复行为的独立回归测试。

这些测试不连接训练服务器，不创建进程组，不使用 GPU 或正式数据，也不代表真实模型
训练验收。它们验证全局加权抽样按 rank 正确分片、四来源概率不会被分片或时长展开破坏、
epoch/已消费位置恢复以及非法配置拒绝。确定性数据包装测试覆盖 worker 数量、外部随机
状态、异常退出和断点位置变化，确保原 Dataset 的随机裁剪不依赖预取时序；测试不修改
第 3 步 batch 契约，不以样本值不重复冒充有放回抽样的抽样位置互斥。
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from gem.closedloop.sampling import (
    DeterministicDrawDataset,
    ResumableDistributedWeightedSampler,
)


class _RandomDataset(Dataset):
    def __len__(self):
        return 20

    def __getitem__(self, index):
        return torch.tensor([index, random.random(), np.random.random(), torch.rand(()).item()])


class _FailingDataset(_RandomDataset):
    def __getitem__(self, index):
        super().__getitem__(index)
        raise RuntimeError("expected dataset error")


def _collect_random_draws(*, workers: int, start_index: int = 0):
    sampler = ResumableDistributedWeightedSampler(
        [1.0] * 20, 24, seed=87, rank=1, world_size=2, emit_draw_keys=True
    )
    sampler.set_epoch(3, start_index=start_index)
    loader = DataLoader(
        DeterministicDrawDataset(_RandomDataset(), seed=45),
        sampler=sampler,
        batch_size=4,
        num_workers=workers,
        generator=torch.Generator().manual_seed(17),
    )
    return torch.cat(list(loader))


def test_all_ranks_reconstruct_exact_global_weighted_stream():
    weights = [0.2, 0.1, 0.4, 0.3]
    draws, world_size, seed, epoch = 23, 8, 42, 3
    expected = torch.multinomial(
        torch.tensor(weights, dtype=torch.double),
        draws * world_size,
        replacement=True,
        generator=torch.Generator().manual_seed(seed + epoch),
    ).tolist()
    reconstructed = [None] * len(expected)
    for rank in range(world_size):
        sampler = ResumableDistributedWeightedSampler(weights, draws, seed, rank, world_size)
        sampler.set_epoch(epoch)
        stream = list(sampler)
        assert len(stream) == draws
        assert stream == expected[rank::world_size]
        reconstructed[rank::world_size] = stream
    assert reconstructed == expected


def test_fixed_seed_repeats_and_epoch_changes_without_mutating_global_rng():
    sampler = ResumableDistributedWeightedSampler([1.0] * 10, 50, seed=12)
    before = torch.get_rng_state().clone()
    first = list(sampler)
    assert list(sampler) == first
    assert torch.equal(before, torch.get_rng_state())
    sampler.set_epoch(1)
    assert list(sampler) != first
    sampler.set_epoch(0)
    assert list(sampler) == first


def test_four_source_weights_survive_duration_aware_index_lengths():
    lengths = [10, 70, 15, 5]
    probabilities = [0.20, 0.35, 0.25, 0.20]
    weights = [probability / size for size, probability in zip(lengths, probabilities) for _ in range(size)]
    sampler = ResumableDistributedWeightedSampler(weights, 50000, seed=182, rank=3, world_size=8)
    samples = np.asarray(list(sampler))
    boundaries = np.cumsum([0, *lengths])
    for source, probability in enumerate(probabilities):
        observed = ((samples >= boundaries[source]) & (samples < boundaries[source + 1])).mean()
        assert observed == pytest.approx(probability, abs=0.008)


def test_resume_offset_is_per_rank_and_independent_of_prefetch():
    sampler = ResumableDistributedWeightedSampler([0.1, 0.3, 0.6], 17, 40, rank=2, world_size=8)
    sampler.set_epoch(5)
    full = list(sampler)
    sampler.set_epoch(5, start_index=6)
    assert len(sampler) == 11
    assert list(sampler) == full[6:]
    assert list(sampler) == full[6:]
    sampler.set_epoch(5, start_index=17)
    assert len(sampler) == 0 and list(sampler) == []
    sampler.set_epoch(6)
    assert len(sampler) == 17


def test_draw_positions_are_unique_across_ranks_even_with_repeated_values():
    keys = []
    for rank in range(8):
        sampler = ResumableDistributedWeightedSampler(
            [1.0], 7, seed=12, rank=rank, world_size=8, emit_draw_keys=True
        )
        sampler.set_epoch(2, start_index=3)
        keys.extend(sampler)
    assert len(keys) == 32 and len(set(keys)) == 32
    assert {key[0] for key in keys} == {0}
    assert {key[1] for key in keys} == {2}
    assert {key[3] for key in keys} == {3, 4, 5, 6}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"weights": []}, {"weights": [[1.0]]}, {"weights": [0.0, 0.0]},
        {"weights": [-1.0, 2.0]}, {"weights": [float("nan")]},
        {"weights": [float("inf")]}, {"weights": [1e308, 1e308]},
        {"num_samples": 0}, {"num_samples": 1.5}, {"num_samples": True},
        {"seed": -1}, {"seed": 2**63}, {"rank": -1}, {"rank": 1},
        {"world_size": 0}, {"world_size": 1.5}, {"emit_draw_keys": "true"},
    ],
)
def test_invalid_sampler_arguments_are_rejected(kwargs):
    arguments = {"weights": [1.0], "num_samples": 3, "seed": 42}
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        ResumableDistributedWeightedSampler(**arguments)


@pytest.mark.parametrize("epoch,offset", [(-1, 0), (1.5, 0), (2**63, 0), (0, -1), (0, 4), (0, 1.5)])
def test_invalid_resume_state_is_rejected(epoch, offset):
    sampler = ResumableDistributedWeightedSampler([1.0], 3, seed=1)
    with pytest.raises(ValueError):
        sampler.set_epoch(epoch, offset)
    assert sampler.epoch == 0 and sampler.start_index == 0


def test_zero_weight_indices_are_never_selected_and_input_weights_are_copied():
    weights = torch.tensor([0.0, 1.0, 0.0])
    sampler = ResumableDistributedWeightedSampler(weights, 15, seed=12)
    weights[1] = 0
    assert list(sampler) == [1] * 15


def test_deterministic_draws_do_not_depend_on_worker_count_or_resume():
    full = _collect_random_draws(workers=0)
    assert torch.equal(full, _collect_random_draws(workers=2))
    assert torch.equal(full[8:], _collect_random_draws(workers=2, start_index=8))


def test_deterministic_draws_ignore_ambient_rng_and_restore_it():
    dataset = DeterministicDrawDataset(_RandomDataset(), seed=42)
    key = (4, 2, 1, 7)
    first = dataset[key]
    random.seed(872)
    np.random.seed(18)
    torch.manual_seed(918)
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    assert torch.equal(first, dataset[key])
    assert random.getstate() == python_state
    actual_numpy_state = np.random.get_state()
    assert actual_numpy_state[0] == numpy_state[0]
    assert np.array_equal(actual_numpy_state[1], numpy_state[1])
    assert actual_numpy_state[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert not torch.equal(first, dataset[(4, 2, 1, 8)])


def test_dataset_error_also_restores_rng():
    dataset = DeterministicDrawDataset(_FailingDataset(), seed=42)
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    with pytest.raises(RuntimeError, match="expected dataset error"):
        dataset[(0, 0, 0, 0)]
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)


@pytest.mark.parametrize("key", [0, (1, 0, 0), (1, -1, 0, 0), (1, 0, 0, 1.5)])
def test_invalid_draw_keys_are_rejected(key):
    dataset = DeterministicDrawDataset(_RandomDataset(), seed=42)
    with pytest.raises(ValueError):
        dataset[key]
    assert len(dataset) == len(dataset.inner)


def test_out_of_bounds_draw_key_is_rejected():
    dataset = DeterministicDrawDataset(_RandomDataset(), seed=42)
    with pytest.raises(IndexError):
        dataset[(len(dataset), 0, 0, 0)]
