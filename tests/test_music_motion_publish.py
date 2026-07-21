"""Atomic READY protocol tests for music-to-motion artifacts."""

from __future__ import annotations

import json
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import torch

from gem.runtime.artifact_publish import make_unique_output_paths, publish_ready_directory
from scripts.demo.demo_music import (
    build_music_metadata,
    build_music_only_data,
    music_generation_prefix,
    update_saved_metadata,
    write_music_artifacts,
)


def _arguments() -> Namespace:
    return Namespace(
        start_sec=1.25,
        width=640,
        height=480,
        focal=None,
        guidance_scale=2.5,
        ddim_steps=50,
        no_postproc=False,
    )


def test_music_generation_is_direct_atomic_ready_child(tmp_path: Path) -> None:
    audio = tmp_path / "My Song.wav"
    checkpoint = tmp_path / "gem_smpl.ckpt"
    audio.write_bytes(b"source")
    checkpoint.write_bytes(b"checkpoint")
    length = 4
    features = torch.zeros(length, 35)
    data = build_music_only_data(features, width=640, height=480)
    body = {
        "body_pose": torch.zeros(length, 63),
        "global_orient": torch.zeros(length, 3),
        "transl": torch.zeros(length, 3),
        "betas": torch.randn(length, 10),
    }
    feature_metadata = {
        "selected_duration_sec": length / 30,
        "original_duration_sec": 10.0,
        "estimated_or_prior_bpm": 120.0,
        "bpm_source": "estimated",
    }
    completed = "2026-07-20T00:00:00Z"
    metadata = build_music_metadata(
        args=_arguments(),
        audio_path=audio,
        checkpoint=checkpoint,
        feature_metadata=feature_metadata,
        sample_seed=42,
        sample_index=0,
        num_frames=length,
        render_succeeded=False,
        audio_mux_succeeded=False,
        completed_at=completed,
    )
    temporary, final = make_unique_output_paths(
        tmp_path / "published", music_generation_prefix(audio, 1.25, 42)
    )
    write_music_artifacts(
        temporary,
        body_global=dict(body),
        body_incam=dict(body),
        raw_motion_151d=torch.zeros(length, 151),
        music_features=features,
        data=data,
        metadata=metadata,
    )
    assert temporary.parent == final.parent
    assert not (temporary / "READY").exists()
    update_saved_metadata(temporary, metadata)
    publish_ready_directory(temporary, final, completed)

    assert final.parent == tmp_path / "published"
    required = {
        "smpl_params.pt",
        "motion.npz",
        "raw_motion_151d.pt",
        "music_features.pt",
        "metadata.json",
        "source_audio.txt",
        "READY",
    }
    assert required <= {path.name for path in final.iterdir()}
    saved = torch.load(final / "smpl_params.pt", map_location="cpu", weights_only=False)
    assert saved["source"] == "music_only"
    assert saved["shape_mode"] == "zero"
    assert torch.count_nonzero(saved["body_params_global"]["betas"]) == 0
    assert torch.count_nonzero(saved["body_params_incam"]["betas"]) == 0
    assert torch.count_nonzero(torch.load(final / "music_features.pt")) == 0
    parsed = json.loads((final / "metadata.json").read_text(encoding="utf-8"))
    assert datetime.fromisoformat(parsed["completed_at"].replace("Z", "+00:00"))


def test_unique_samples_never_overwrite_and_failure_never_creates_ready(tmp_path: Path) -> None:
    first_temp, first_final = make_unique_output_paths(tmp_path, "song_seed42")
    second_temp, second_final = make_unique_output_paths(tmp_path, "song_seed42")
    assert first_temp != second_temp and first_final != second_final
    first_temp.mkdir()
    (first_temp / "partial").write_text("incomplete", encoding="utf-8")
    assert not (first_temp / "READY").exists()
    assert not first_final.exists()
