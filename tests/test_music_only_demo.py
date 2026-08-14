"""CPU contracts for the specialist music-only demo and physical sanity gate."""

from __future__ import annotations

import pytest
import torch

from gem.runtime.motion_sanity import evaluate_global_motion_sanity
from scripts.demo_music_only import select_music_window


def _music(frames: int) -> torch.Tensor:
    value = torch.zeros(frames, 35)
    value[::2, 33] = 1.0
    value[1::2, 34] = 1.0
    return value


def test_demo_allows_more_than_training_window_without_truncation() -> None:
    selected = select_music_window(
        _music(240), start_frame=0, num_frames=None, max_frames=600, source="test"
    )
    assert selected.shape == (240, 35)


def test_demo_selects_explicit_long_window_and_enforces_safety_limit() -> None:
    selected = select_music_window(
        _music(400), start_frame=50, num_frames=200, max_frames=300, source="test"
    )
    assert selected.shape == (200, 35)
    with pytest.raises(ValueError, match="exceeds --max-frames"):
        select_music_window(
            _music(400), start_frame=0, num_frames=None, max_frames=120, source="test"
        )


def test_motion_sanity_accepts_metric_upright_motion() -> None:
    frames = 120
    transl = torch.zeros(frames, 3)
    transl[:, 0] = torch.arange(frames) * 0.01
    report = evaluate_global_motion_sanity(
        {"transl": transl, "global_orient": torch.zeros(frames, 3)}
    )
    assert report["physical_sanity_pass"] is True
    assert report["mean_body_up_y_dot"] == pytest.approx(1.0)


def test_motion_sanity_rejects_flying_and_horizontal_body() -> None:
    frames = 120
    transl = torch.zeros(frames, 3)
    transl[:, 1] = torch.arange(frames) * 1.0
    # +90 degrees around X maps the body's local +Y away from world +Y.
    orient = torch.zeros(frames, 3)
    orient[:, 0] = torch.pi / 2
    report = evaluate_global_motion_sanity(
        {"transl": transl, "global_orient": orient}
    )
    assert report["physical_sanity_pass"] is False
    assert len(report["issues"]) == 2
