"""Unit tests for the AIOZ-GDANCE canonical conversion contract."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tools.data.music_dance.aioz.common import (
    SequenceLabel,
    convert_person_motion,
    filename_frame_span,
    inspect_motion_payload,
    select_group_labels,
    validate_canonical_smpl,
)


def _payload(persons: int = 3, frames: int = 120) -> dict:
    rng = np.random.default_rng(42)
    return {
        "smpl_poses": rng.normal(size=(persons, frames, 72)).astype(np.float64),
        "root_trans": rng.normal(size=(persons, frames, 3)).astype(np.float32),
        "smpl_betas": rng.normal(size=(persons, frames, 10)).astype(np.float32),
        "meta": {
            "vid_name": "group",
            "orig_start": 0,
            "orig_end": frames,
            "n_persons": persons,
        },
    }


def test_real_aioz_shape_contract_and_exact_pose_slice() -> None:
    payload = _payload()
    info = inspect_motion_payload(payload, "group_0_120")
    assert info["num_persons"] == 3
    assert info["num_frames"] == 120

    motion = convert_person_motion(
        payload, group_id="group_0_120", person_id=1, source_fps=30, target_fps=30
    )
    assert validate_canonical_smpl(motion) == 120
    np.testing.assert_array_equal(
        motion["global_orient"].numpy(), payload["smpl_poses"][1, :, :3].astype(np.float32)
    )
    np.testing.assert_array_equal(
        motion["body_pose"].numpy(), payload["smpl_poses"][1, :, 3:66].astype(np.float32)
    )
    np.testing.assert_array_equal(
        motion["transl"].numpy(), payload["root_trans"][1].astype(np.float32)
    )


def test_invalid_pose_dimension_and_nonfinite_fail() -> None:
    payload = _payload()
    payload["smpl_poses"] = payload["smpl_poses"][..., :71]
    with pytest.raises(ValueError, match=r"\[P,T,72\]"):
        inspect_motion_payload(payload, "group_0_120")

    payload = _payload()
    payload["root_trans"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        inspect_motion_payload(payload, "group_0_120")


def test_metadata_and_filename_frame_spans_must_match() -> None:
    assert filename_frame_span("youtube_03_114_1080") == 966
    payload = _payload(frames=119)
    with pytest.raises(ValueError, match="filename frame span"):
        inspect_motion_payload(payload, "group_0_120")


def test_non_30fps_resampling_uses_target_clock() -> None:
    payload = _payload(frames=120)
    motion = convert_person_motion(
        payload, group_id="group_0_120", person_id=0, source_fps=60, target_fps=30
    )
    assert validate_canonical_smpl(motion) == 60
    torch.testing.assert_close(motion["transl"][0], torch.from_numpy(payload["root_trans"][0, 0]))


def test_small_group_selection_is_group_level_and_covers_splits() -> None:
    labels = {
        split: [
            SequenceLabel(
                group_id=f"{split}_group_{index}_0_120",
                split=split,
                music_genre="Pop",
                dance_style="Dance",
            )
            for index in range(10)
        ]
        for split in ("train", "val", "test")
    }
    selected = select_group_labels(labels, sample_groups=10, seed=1)
    assert len(selected) == 10
    assert len({row.group_id for row in selected}) == 10
    assert {row.split for row in selected} == {"train", "val", "test"}


def test_canonical_validator_rejects_wrong_body_dimension() -> None:
    motion = {
        "global_orient": torch.zeros(120, 3),
        "body_pose": torch.zeros(120, 66),
        "transl": torch.zeros(120, 3),
        "betas": torch.zeros(120, 10),
    }
    with pytest.raises(ValueError, match=r"body_pose must be \[T,63\]"):
        validate_canonical_smpl(motion)
