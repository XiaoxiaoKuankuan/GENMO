"""Unit tests for the FineDance canonical conversion contract."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gem.utils.rotation_conversions import axis_angle_to_matrix, matrix_to_rotation_6d
from tools.data.music_dance.finedance.common import (
    convert_motion_array,
    inspect_motion_array,
    official_genre_split,
    validate_canonical_motion,
)


def _motion(frames: int = 120) -> tuple[np.ndarray, torch.Tensor]:
    generator = torch.Generator().manual_seed(42)
    axis_angle = torch.randn(frames, 52, 3, generator=generator) * 0.3
    rotation_6d = matrix_to_rotation_6d(axis_angle_to_matrix(axis_angle)).reshape(frames, 312)
    translation = torch.randn(frames, 3, generator=generator)
    data = torch.cat([translation, rotation_6d], dim=-1).numpy().astype(np.float32)
    return data, axis_angle


def test_real_layout_is_translation_plus_52_rotation6d() -> None:
    data, axis_angle = _motion()
    info = inspect_motion_array(data, "001")
    assert info["shape"] == [120, 315]
    assert info["rotation_6d_shape"] == [120, 52, 6]

    motion = convert_motion_array(data, sample_id="001", source_fps=30, target_fps=30)
    assert validate_canonical_motion(motion) == 120
    assert motion["pose"].shape == (120, 66)
    assert motion["transl"].shape == (120, 3)
    assert motion["betas"].shape == (120, 10)
    # Axis-angle is non-unique; matrices must be identical.
    expected = axis_angle[:, :22]
    actual = motion["pose"].reshape(120, 22, 3)
    torch.testing.assert_close(axis_angle_to_matrix(actual), axis_angle_to_matrix(expected))
    torch.testing.assert_close(motion["transl"], torch.from_numpy(data[:, :3]))
    assert torch.equal(motion["betas"], torch.zeros_like(motion["betas"]))


def test_wrong_shape_and_nonfinite_are_rejected() -> None:
    data, _ = _motion()
    with pytest.raises(ValueError, match=r"\[T,315\]"):
        inspect_motion_array(data[:, :-1], "001")
    data[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN/Inf"):
        inspect_motion_array(data, "001")


def test_non_30fps_motion_resampling_uses_target_clock() -> None:
    data, _ = _motion(120)
    motion = convert_motion_array(data, sample_id="001", source_fps=60, target_fps=30)
    assert validate_canonical_motion(motion) == 60
    torch.testing.assert_close(motion["transl"][0], torch.from_numpy(data[0, :3]))


def test_official_genre_split_has_no_leakage() -> None:
    available = {f"{value:03d}" for value in range(1, 212)} - set(
        ["116", "117", "118", "119", "120", "121", "122", "123"]
    )
    splits, metadata = official_genre_split(available)
    assert {key: len(value) for key, value in splits.items()} == {
        "train": 183,
        "val": 2,
        "test": 18,
    }
    assert splits["val"] == ["130", "202"]
    assert "FineDance@Genre" == metadata["name"]
    assert not (set(splits["train"]) & set(splits["val"]))
    assert not (set(splits["train"]) & set(splits["test"]))
    assert not (set(splits["val"]) & set(splits["test"]))


def test_canonical_validator_requires_pose_66() -> None:
    bad = {
        "pose": torch.zeros(10, 65),
        "transl": torch.zeros(10, 3),
        "betas": torch.zeros(10, 10),
    }
    with pytest.raises(ValueError, match=r"pose must be a Tensor \[T,66\]"):
        validate_canonical_motion(bad)
