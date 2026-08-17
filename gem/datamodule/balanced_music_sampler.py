# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Deterministic hierarchical DDP sampling for de-duplicated music datasets."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterator, Sequence
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler, Subset


def project_bounded_probabilities(
    weights: Sequence[float], *, minimum: float, maximum: float
) -> list[float]:
    """Project positive relative weights onto a bounded probability simplex.

    The solution preserves relative weights for every component that is not at
    a bound: ``p_i = clamp(lambda * weights_i, minimum, maximum)``.
    """
    values = torch.as_tensor(weights, dtype=torch.float64)
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("weights must be a non-empty one-dimensional sequence")
    if not torch.isfinite(values).all() or (values <= 0).any():
        raise ValueError("all sampling weights must be finite and positive")
    count = int(values.numel())
    if minimum < 0 or maximum <= 0 or minimum > maximum:
        raise ValueError("invalid probability bounds")
    if count * minimum > 1.0 + 1e-12 or count * maximum < 1.0 - 1e-12:
        raise ValueError("probability bounds do not intersect the simplex")

    low = 0.0
    high = 1.0 / float(values.min())
    for _ in range(100):
        middle = (low + high) / 2.0
        total = torch.clamp(values * middle, minimum, maximum).sum().item()
        if total < 1.0:
            low = middle
        else:
            high = middle
    probabilities = torch.clamp(values * high, minimum, maximum)
    probabilities /= probabilities.sum()
    return [float(value) for value in probabilities]


def _dataset_sampling_records(dataset: Dataset) -> list[dict[str, Any]]:
    if isinstance(dataset, Subset):
        base_records = {
            int(record["dataset_index"]): record
            for record in _dataset_sampling_records(dataset.dataset)
        }
        records = []
        for subset_index, base_index in enumerate(dataset.indices):
            record = base_records.get(int(base_index))
            if record is not None:
                records.append({**record, "dataset_index": subset_index})
        return records
    getter = getattr(dataset, "get_music_sampling_records", None)
    if getter is None:
        raise TypeError(
            f"{type(dataset).__name__} does not implement get_music_sampling_records()"
        )
    records = list(getter())
    if not records:
        raise ValueError(f"{type(dataset).__name__} has no music sampling records")
    return records


class HierarchicalMusicDistributedSampler(Sampler[int]):
    """Sample dataset -> duration-weighted music group -> uniform variant.

    A complete global epoch is generated from ``seed + epoch`` on every rank,
    shuffled once, and then stride-sharded by rank.  This gives the same global
    sample multiset and deterministic disjoint rank partitions after resume.
    """

    def __init__(
        self,
        datasets: Sequence[Dataset],
        *,
        dataset_names: Sequence[str] | None = None,
        samples_per_epoch: int = 52224,
        fps: float = 30.0,
        temperature: float = 0.5,
        minimum_dataset_probability: float = 0.05,
        maximum_dataset_probability: float = 0.50,
        seed: int = 42,
        rank: int | None = None,
        num_replicas: int | None = None,
    ) -> None:
        if not datasets:
            raise ValueError("at least one dataset is required")
        self.datasets = list(datasets)
        self.dataset_names = (
            [str(value) for value in dataset_names]
            if dataset_names is not None
            else [f"dataset_{index}" for index in range(len(datasets))]
        )
        if len(self.dataset_names) != len(self.datasets):
            raise ValueError("dataset_names and datasets must have equal length")
        self.samples_per_epoch = int(samples_per_epoch)
        self.fps = float(fps)
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.epoch = 0
        if self.samples_per_epoch <= 0 or self.fps <= 0:
            raise ValueError("samples_per_epoch and fps must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")

        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        if self.num_replicas <= 0 or not 0 <= self.rank < self.num_replicas:
            raise ValueError("invalid rank/num_replicas")
        if self.samples_per_epoch % self.num_replicas != 0:
            raise ValueError(
                "samples_per_epoch must be divisible by num_replicas; padding would "
                "duplicate samples across DDP ranks"
            )
        self.num_samples = self.samples_per_epoch // self.num_replicas

        self.dataset_offsets: list[int] = []
        offset = 0
        self.groups: list[dict[str, list[int]]] = []
        self.group_frames: list[dict[str, int]] = []
        self.unique_music_hours: list[float] = []
        for dataset in self.datasets:
            self.dataset_offsets.append(offset)
            offset += len(dataset)
            variants: dict[str, list[int]] = defaultdict(list)
            durations: dict[str, int] = {}
            for record in _dataset_sampling_records(dataset):
                local_index = int(record["dataset_index"])
                group_id = str(record["group_id"])
                num_frames = int(record["num_frames"])
                if not 0 <= local_index < len(dataset):
                    raise ValueError(f"sampling record index out of bounds: {local_index}")
                if not group_id or num_frames <= 0:
                    raise ValueError("group_id and num_frames must be valid")
                variants[group_id].append(local_index)
                durations[group_id] = max(durations.get(group_id, 0), num_frames)
            if not variants:
                raise ValueError("dataset contains no music groups")
            self.groups.append(dict(variants))
            self.group_frames.append(durations)
            unique_frames = sum(durations.values())
            self.unique_music_hours.append(unique_frames / self.fps / 3600.0)

        relative_weights = [
            hours**self.temperature for hours in self.unique_music_hours
        ]
        self.dataset_probabilities = project_bounded_probabilities(
            relative_weights,
            minimum=float(minimum_dataset_probability),
            maximum=float(maximum_dataset_probability),
        )

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "seed": self.seed}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        if int(state_dict.get("seed", self.seed)) != self.seed:
            raise ValueError("sampler state seed differs from configured seed")
        self.set_epoch(int(state_dict["epoch"]))

    def _dataset_counts(self) -> list[int]:
        expected = [value * self.samples_per_epoch for value in self.dataset_probabilities]
        counts = [math.floor(value) for value in expected]
        remainder = self.samples_per_epoch - sum(counts)
        order = sorted(
            range(len(counts)),
            key=lambda index: (expected[index] - counts[index], -index),
            reverse=True,
        )
        for index in order[:remainder]:
            counts[index] += 1
        return counts

    def _global_indices(self) -> torch.Tensor:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        global_indices: list[int] = []
        for dataset_index, count in enumerate(self._dataset_counts()):
            group_ids = sorted(self.groups[dataset_index])
            group_weights = torch.tensor(
                [self.group_frames[dataset_index][key] for key in group_ids],
                dtype=torch.float64,
            )
            selected_groups = torch.multinomial(
                group_weights,
                num_samples=count,
                replacement=True,
                generator=generator,
            )
            offset = self.dataset_offsets[dataset_index]
            for group_index in selected_groups.tolist():
                variants = self.groups[dataset_index][group_ids[group_index]]
                variant_index = int(
                    torch.randint(len(variants), (1,), generator=generator).item()
                )
                global_indices.append(offset + variants[variant_index])
        indices = torch.tensor(global_indices, dtype=torch.int64)
        permutation = torch.randperm(len(indices), generator=generator)
        return indices[permutation]

    def __iter__(self) -> Iterator[int]:
        indices = self._global_indices()
        rank_indices = indices[self.rank :: self.num_replicas]
        if len(rank_indices) != self.num_samples:
            raise RuntimeError("internal DDP sampler partition length mismatch")
        return iter(rank_indices.tolist())


__all__ = [
    "HierarchicalMusicDistributedSampler",
    "project_bounded_probabilities",
]
