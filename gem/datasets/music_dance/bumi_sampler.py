"""Deterministic de-duplicated hierarchical sampling for BUMI music motion."""

from __future__ import annotations

import math
import os
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler

from gem.utils.pylogger import Log


@dataclass(frozen=True)
class BumiWindowIndex:
    """One globally numbered dataset/group/variant/window draw."""

    dataset_index: int
    row_index: int
    start_frame: int
    draw_index: int


def project_probabilities_with_bounds(
    weights: Sequence[float], minimum: float, maximum: float
) -> list[float]:
    """Project positive relative weights to a simplex with box constraints.

    Free entries preserve their relative input weights.  The iterative
    water-filling formulation is deterministic and is sufficient for the four
    top-level datasets without introducing an optimisation dependency.
    """

    values = [float(value) for value in weights]
    count = len(values)
    if count == 0 or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("sampling weights must be a non-empty sequence of positive values")
    if not 0.0 <= minimum <= maximum <= 1.0:
        raise ValueError("probability bounds must satisfy 0 <= minimum <= maximum <= 1")
    if count * minimum > 1.0 + 1.0e-12 or count * maximum < 1.0 - 1.0e-12:
        raise ValueError("probability bounds cannot contain a unit simplex")
    result = [0.0] * count
    free = set(range(count))
    remaining = 1.0
    while free:
        weight_sum = sum(values[index] for index in free)
        proposal = {
            index: remaining * values[index] / weight_sum for index in free
        }
        low = sorted(index for index, value in proposal.items() if value < minimum)
        high = sorted(index for index, value in proposal.items() if value > maximum)
        if not low and not high:
            for index, value in proposal.items():
                result[index] = value
            break
        # Fix every currently violating entry. Feasibility guarantees progress.
        for index in low:
            result[index] = minimum
            remaining -= minimum
            free.remove(index)
        for index in high:
            if index not in free:
                continue
            result[index] = maximum
            remaining -= maximum
            free.remove(index)
        if remaining < -1.0e-10:
            raise RuntimeError("internal bounded-simplex projection failure")
    correction = 1.0 - sum(result)
    if abs(correction) > 1.0e-10:
        candidates = [
            index
            for index, value in enumerate(result)
            if minimum - 1.0e-12 <= value + correction <= maximum + 1.0e-12
        ]
        if not candidates:
            raise RuntimeError("could not correct bounded sampling probabilities")
        result[candidates[0]] += correction
    return result


class BumiBalancedMultiDataset(Dataset):
    """Delegate sampler-selected indices to their source dataset."""

    def __init__(self, datasets: Sequence[Dataset]) -> None:
        self.datasets = tuple(datasets)
        if not self.datasets:
            raise ValueError("BumiBalancedMultiDataset requires at least one dataset")

    def __len__(self) -> int:
        return sum(len(getattr(dataset, "rows", dataset)) for dataset in self.datasets)

    def __getitem__(self, index: BumiWindowIndex) -> dict[str, Any]:
        if not isinstance(index, BumiWindowIndex):
            raise TypeError(
                "hierarchical BUMI dataset requires BumiWindowIndex values from its sampler"
            )
        dataset = self.datasets[index.dataset_index]
        result = dataset.get_window(index.row_index, index.start_frame)
        result["meta"]["sampler_draw_index"] = index.draw_index
        return result


class DeduplicatedBumiSampler(Sampler[BumiWindowIndex]):
    """Sample dataset -> duration-weighted music group -> variant -> window.

    ``samples_per_epoch`` is global across all ranks.  Each rank receives a
    strided, disjoint set of global draw numbers.  A draw is a pure function of
    ``seed`` and ``epoch``, so restoring an epoch plus Lightning's batch-loop
    cursor reproduces all remaining samples.
    """

    def __init__(
        self,
        datasets: Sequence[Dataset],
        *,
        samples_per_epoch: int,
        seed: int = 42,
        probability_min: float = 0.05,
        probability_max: float = 0.50,
        rank: int | None = None,
        world_size: int | None = None,
    ) -> None:
        super().__init__()
        self.datasets = tuple(datasets)
        if not self.datasets:
            raise ValueError("hierarchical sampler requires at least one dataset")
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)
        self.epoch = 0
        if self.samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive")
        # ``scripts/train.py`` constructs the DataLoader before Lightning calls
        # ``init_process_group``.  Pinning rank/world-size here would therefore
        # make every DDP process rank 0 of 1.  Keep automatic values dynamic and
        # resolve them at iteration/length time, when either torch.distributed or
        # Lightning's RANK/WORLD_SIZE environment is available.
        self._rank_override = None if rank is None else int(rank)
        self._world_size_override = None if world_size is None else int(world_size)
        self._logged_distributed_contexts: set[tuple[int, int]] = set()
        self._validate_distributed_context(*self._distributed_context())

        self.dataset_names: list[str] = []
        self.group_ids: list[list[str]] = []
        self.group_rows: list[list[list[int]]] = []
        self.group_weights: list[torch.Tensor] = []
        self.deduplicated_hours: list[float] = []
        for dataset in self.datasets:
            if bool(getattr(dataset, "duration_aware_sampling", False)):
                raise ValueError(
                    "deduplicated hierarchical sampling requires duration_aware_sampling=false"
                )
            rows = getattr(dataset, "rows", None)
            if not isinstance(rows, list) or not rows:
                raise ValueError("every hierarchical BUMI dataset must expose non-empty rows")
            grouped: dict[str, list[int]] = defaultdict(list)
            for row_index, row in enumerate(rows):
                group_id = str(row.get("music_group_id", ""))
                if not group_id:
                    raise ValueError(f"row {row_index} is missing music_group_id")
                grouped[group_id].append(row_index)
            ids = sorted(grouped)
            variants = [grouped[group_id] for group_id in ids]
            durations = [
                max(int(rows[row_index]["num_frames"]) for row_index in row_indices)
                for row_indices in variants
            ]
            if any(value <= 0 for value in durations):
                raise ValueError("music group durations must be positive")
            self.dataset_names.append(str(getattr(dataset, "dataset_name", "unknown")))
            self.group_ids.append(ids)
            self.group_rows.append(variants)
            self.group_weights.append(torch.tensor(durations, dtype=torch.double))
            self.deduplicated_hours.append(sum(durations) / 30.0 / 3600.0)
        top_weights = [math.sqrt(value) for value in self.deduplicated_hours]
        self.dataset_probabilities = project_probabilities_with_bounds(
            top_weights, float(probability_min), float(probability_max)
        )
        self.dataset_probability_min = float(probability_min)
        self.dataset_probability_max = float(probability_max)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _environment_rank_world_size() -> tuple[int, int]:
        world_size_text = os.environ.get("WORLD_SIZE")
        rank_text = os.environ.get("RANK")
        if world_size_text is None:
            return 0, 1
        world_size = int(world_size_text)
        if rank_text is not None:
            return int(rank_text), world_size
        # LOCAL_RANK is sufficient for the supported single-node launch and is
        # preferable to silently treating every spawned worker as rank zero.
        local_rank_text = os.environ.get("LOCAL_RANK")
        return (int(local_rank_text) if local_rank_text is not None else 0), world_size

    def _distributed_context(self) -> tuple[int, int]:
        if dist.is_available() and dist.is_initialized():
            automatic_rank = int(dist.get_rank())
            automatic_world_size = int(dist.get_world_size())
        else:
            automatic_rank, automatic_world_size = self._environment_rank_world_size()
        rank = automatic_rank if self._rank_override is None else self._rank_override
        world_size = (
            automatic_world_size
            if self._world_size_override is None
            else self._world_size_override
        )
        return int(rank), int(world_size)

    def _validate_distributed_context(self, rank: int, world_size: int) -> None:
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError(f"invalid rank/world_size={rank}/{world_size}")
        if self.samples_per_epoch % world_size != 0:
            raise ValueError(
                "samples_per_epoch must be divisible by world_size; padding would duplicate draws"
            )

    @property
    def rank(self) -> int:
        rank, world_size = self._distributed_context()
        self._validate_distributed_context(rank, world_size)
        return rank

    @property
    def world_size(self) -> int:
        rank, world_size = self._distributed_context()
        self._validate_distributed_context(rank, world_size)
        return world_size

    def state_dict(self) -> dict[str, int]:
        rank, world_size = self._distributed_context()
        self._validate_distributed_context(rank, world_size)
        return {
            "epoch": self.epoch,
            "seed": self.seed,
            "samples_per_epoch": self.samples_per_epoch,
            "world_size": world_size,
        }

    def load_state_dict(self, state: dict[str, int]) -> None:
        for key, expected in (
            ("seed", self.seed),
            ("samples_per_epoch", self.samples_per_epoch),
            ("world_size", self.world_size),
        ):
            if int(state.get(key, -1)) != expected:
                raise ValueError(
                    f"cannot restore BUMI sampler: {key}={state.get(key)!r}, expected={expected}"
                )
        self.set_epoch(int(state["epoch"]))

    def summary(self) -> dict[str, Any]:
        rank, world_size = self._distributed_context()
        self._validate_distributed_context(rank, world_size)
        return {
            "strategy": "deduplicated_hierarchical_sqrt_duration_v1",
            "samples_per_epoch_global": self.samples_per_epoch,
            "samples_per_epoch_rank": self.samples_per_epoch // world_size,
            "seed": self.seed,
            "rank": rank,
            "world_size": world_size,
            "datasets": [
                {
                    "name": name,
                    "groups": len(groups),
                    "deduplicated_hours": hours,
                    "probability": probability,
                }
                for name, groups, hours, probability in zip(
                    self.dataset_names,
                    self.group_ids,
                    self.deduplicated_hours,
                    self.dataset_probabilities,
                    strict=True,
                )
            ],
        }

    def _global_draws(self) -> list[BumiWindowIndex]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch * 1_000_003)
        dataset_choices = torch.multinomial(
            torch.tensor(self.dataset_probabilities, dtype=torch.double),
            self.samples_per_epoch,
            replacement=True,
            generator=generator,
        ).tolist()
        output: list[BumiWindowIndex] = []
        for draw_index, dataset_index in enumerate(dataset_choices):
            group_index = int(
                torch.multinomial(
                    self.group_weights[dataset_index],
                    1,
                    replacement=True,
                    generator=generator,
                ).item()
            )
            variants = self.group_rows[dataset_index][group_index]
            variant_offset = int(
                torch.randint(len(variants), (1,), generator=generator).item()
            )
            row_index = variants[variant_offset]
            row = self.datasets[dataset_index].rows[row_index]
            last_start = max(
                int(row["num_frames"]) - int(self.datasets[dataset_index].motion_frames), 0
            )
            start = int(torch.randint(last_start + 1, (1,), generator=generator).item())
            output.append(
                BumiWindowIndex(dataset_index, row_index, start, draw_index)
            )
        return output

    def __iter__(self) -> Iterator[BumiWindowIndex]:
        rank, world_size = self._distributed_context()
        self._validate_distributed_context(rank, world_size)
        context = (rank, world_size)
        if context not in self._logged_distributed_contexts:
            Log.info(
                "[BUMI Train Sampler Effective DDP] "
                f"rank={rank}, world_size={world_size}, "
                f"samples_per_epoch_rank={self.samples_per_epoch // world_size}"
            )
            self._logged_distributed_contexts.add(context)
        draws = self._global_draws()
        return iter(draws[rank::world_size])

    def __len__(self) -> int:
        rank, world_size = self._distributed_context()
        self._validate_distributed_context(rank, world_size)
        return self.samples_per_epoch // world_size


__all__ = [
    "BumiBalancedMultiDataset",
    "BumiWindowIndex",
    "DeduplicatedBumiSampler",
    "project_probabilities_with_bounds",
]
