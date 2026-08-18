from __future__ import annotations

from collections import Counter

from gem.datasets.music_dance.bumi_sampler import (
    DeduplicatedBumiSampler,
    project_probabilities_with_bounds,
)


class FakeDataset:
    def __init__(self, name: str, group_frames: list[int]) -> None:
        self.dataset_name = name
        self.duration_aware_sampling = False
        self.motion_frames = 120
        self.rows = [
            {
                "sample_id": f"{name}_{index}",
                "music_group_id": f"group_{index}",
                "num_frames": frames,
            }
            for index, frames in enumerate(group_frames)
        ]


def test_bounded_probability_projection_matches_formal_ratios() -> None:
    hours = [0.407620, 13.149139, 3.541583, 0.083843]
    probabilities = project_probabilities_with_bounds(
        [value**0.5 for value in hours], 0.05, 0.50
    )
    expected = [0.113993, 0.50, 0.336007, 0.05]
    for actual, target in zip(probabilities, expected, strict=True):
        assert abs(actual - target) < 2.0e-5
    assert abs(sum(probabilities) - 1.0) < 1.0e-12


def test_sampler_ddp_ratio_reproducibility_and_restore() -> None:
    hours = [0.407620, 13.149139, 3.541583, 0.083843]
    datasets = [
        FakeDataset(f"dataset_{index}", [max(120, round(hours_value * 108000))])
        for index, hours_value in enumerate(hours)
    ]
    sampler = DeduplicatedBumiSampler(
        datasets, samples_per_epoch=52224, seed=17, rank=0, world_size=1
    )
    first = list(sampler)
    second = list(sampler)
    assert first == second
    counts = Counter(item.dataset_index for item in first)
    for index, probability in enumerate(sampler.dataset_probabilities):
        assert abs(counts[index] / len(first) - probability) < 0.005

    sampler.set_epoch(9)
    state = sampler.state_dict()
    restored = DeduplicatedBumiSampler(
        datasets, samples_per_epoch=52224, seed=17, rank=0, world_size=1
    )
    restored.load_state_dict(state)
    assert list(sampler) == list(restored)

    rank0 = DeduplicatedBumiSampler(
        datasets, samples_per_epoch=1024, seed=23, rank=0, world_size=8
    )
    rank1 = DeduplicatedBumiSampler(
        datasets, samples_per_epoch=1024, seed=23, rank=1, world_size=8
    )
    draw0 = {item.draw_index for item in rank0}
    draw1 = {item.draw_index for item in rank1}
    assert len(draw0) == len(draw1) == 128
    assert draw0.isdisjoint(draw1)
