"""Unit tests for the CoMPAS3D canonical conversion contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from gem.utils.rotation_conversions import axis_angle_to_matrix
from tools.data.music_dance.compas3d.common import (
    SOURCE_Z_UP_TO_GENMO_Y_UP,
    SequenceFiles,
    build_splits,
    convert_source_motion,
    discover_local_sequences,
    parse_sequence_id,
    temporal_sample,
    validate_canonical_motion,
    validate_source_motion,
)


def _source_fields(frames: int = 120, fps: float = 30.0) -> dict:
    rng = np.random.default_rng(42)
    poses = rng.normal(scale=0.15, size=(frames, 165)).astype(np.float64)
    poses[:, 66:75] = 0.0
    markers_obs = rng.normal(size=(frames, 53, 3)).astype(np.float64).astype(object)
    markers_sim = rng.normal(size=(frames, 53, 3)).astype(np.float64).astype(object)
    return {
        "gender": np.asarray("male"),
        "surface_model_type": np.asarray("smplx_locked_head"),
        "mocap_frame_rate": np.asarray(fps),
        "betas": rng.normal(size=300).astype(np.float64),
        "poses": poses,
        "trans": rng.normal(size=(frames, 3)).astype(np.float64),
        "markers_obs": markers_obs,
        "markers_sim": markers_sim,
        "v_template": np.asarray(None, dtype=object),
    }


def _sequence(tmp_path: Path, sequence_id: str) -> SequenceFiles:
    parts = parse_sequence_id(sequence_id)
    directory = tmp_path / parts.pair_id / sequence_id
    return SequenceFiles(
        parts=parts,
        directory=directory,
        mp4=directory / f"{sequence_id}.mp4",
        leader=directory / f"{sequence_id}_leader.npz",
        follower=directory / f"{sequence_id}_follower.npz",
    )


def test_real_npz_semantic_contract_and_pose_segments() -> None:
    fields = _source_fields()
    info = validate_source_motion(fields)
    assert info["num_frames"] == 120
    assert info["fps"] == 30
    assert info["pose_shape"] == [120, 165]
    assert info["betas_shape"] == [300]

    motion, temporal = convert_source_motion(
        fields,
        source_pelvis=torch.tensor([0.1, -0.2, 0.3]),
        target_pelvis=torch.tensor([-0.1, 0.2, -0.3]),
    )
    assert validate_canonical_motion(motion) == 120
    assert temporal["method"] == "identity_already_30fps"
    torch.testing.assert_close(motion["body_pose"], torch.from_numpy(fields["poses"][:, 3:66]).float())
    expected_root_matrix = SOURCE_Z_UP_TO_GENMO_Y_UP @ axis_angle_to_matrix(
        torch.from_numpy(fields["poses"][:, :3]).float()
    )
    torch.testing.assert_close(axis_angle_to_matrix(motion["global_orient"]), expected_root_matrix)
    source_pelvis = torch.tensor([0.1, -0.2, 0.3])
    target_pelvis = torch.tensor([-0.1, 0.2, -0.3])
    expected_translation = (
        SOURCE_Z_UP_TO_GENMO_Y_UP
        @ (torch.from_numpy(fields["trans"]).float() + source_pelvis).unsqueeze(-1)
    ).squeeze(-1) - target_pelvis
    torch.testing.assert_close(motion["transl"], expected_translation)
    assert torch.equal(motion["betas"], torch.zeros_like(motion["betas"]))


def test_wrong_npz_shape_and_nonfinite_fail() -> None:
    fields = _source_fields()
    fields["poses"] = fields["poses"][:, :-1]
    with pytest.raises(ValueError, match=r"\[T,165\]"):
        validate_source_motion(fields)
    fields = _source_fields()
    fields["trans"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_source_motion(fields)


def test_120_to_30_uses_exact_shared_four_to_one_indices() -> None:
    frames = 121
    poses = np.zeros((frames, 165), dtype=np.float64)
    trans = np.arange(frames * 3, dtype=np.float64).reshape(frames, 3)
    poses[:, 0] = np.arange(frames)
    sampled_pose, sampled_trans, metadata = temporal_sample(
        poses, trans, source_fps=120, target_fps=30
    )
    indices = np.arange(0, frames, 4)
    np.testing.assert_array_equal(sampled_pose, poses[indices])
    np.testing.assert_array_equal(sampled_trans, trans[indices])
    assert metadata["method"] == "deterministic_4_to_1_frame_selection"
    assert metadata["source_index_stride"] == 4


def test_music_identity_split_prevents_song_leakage(tmp_path: Path) -> None:
    sequences = [
        _sequence(tmp_path, f"Pair{pair}_song{song}_take1")
        for pair in (1, 2)
        for song in (1, 2, 3, 4)
    ]
    splits, report = build_splits(sequences, "music_identity")
    assert {row.split("_")[1] for row in splits["train"]} == {"song1", "song2"}
    assert {row.split("_")[1] for row in splits["val"]} == {"song3"}
    assert {row.split("_")[1] for row in splits["test"]} == {"song4"}
    assert report["music_identity_leakage_count"] == 0

    _, official = build_splits(sequences, "official_interaction")
    assert official["music_identity_leakage_count"] > 0


def test_discovery_matches_leaderi_typo_by_substring(tmp_path: Path) -> None:
    sequence_id = "Pair7_song2_take1"
    directory = tmp_path / "Pair7" / sequence_id
    directory.mkdir(parents=True)
    for name in (
        f"{sequence_id}.mp4",
        f"{sequence_id}_leaderi.npz",
        f"{sequence_id}_follower.npz",
    ):
        (directory / name).write_bytes(b"0" * 2048)
    complete, incomplete = discover_local_sequences(tmp_path)
    assert not incomplete
    assert len(complete) == 1
    assert complete[0].leader.name.endswith("_leaderi.npz")


def test_canonical_validator_rejects_wrong_pose_dimension() -> None:
    bad = {
        "pose": torch.zeros(10, 65),
        "global_orient": torch.zeros(10, 3),
        "body_pose": torch.zeros(10, 63),
        "transl": torch.zeros(10, 3),
        "betas": torch.zeros(10, 10),
    }
    with pytest.raises(ValueError, match=r"pose must be Tensor \[T,66\]"):
        validate_canonical_motion(bad)
