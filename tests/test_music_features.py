"""Tests for EDGE baseline35 extraction and AIST++ feature compatibility."""

from __future__ import annotations

import importlib.util
import wave
from pathlib import Path

import numpy as np
import pytest
import torch

from gem.datasets.aistpp.aistplusplus import (
    load_music_beats,
    load_music_feature_tensor,
    validate_musicfeat_v2,
)
from gem.utils.music_features import (
    EDGE_BASELINE_FEATURE_NAMES,
    EDGE_SAMPLE_RATE,
    align_features_to_length,
    extract_edge_baseline35,
    get_aist_tempo_prior,
)


def _write_test_wav(path: Path, seconds: float = 1.0) -> None:
    frame_count = int(round(seconds * EDGE_SAMPLE_RATE))
    times = np.arange(frame_count, dtype=np.float32) / EDGE_SAMPLE_RATE
    signal = 0.15 * np.sin(2 * np.pi * 220.0 * times)
    for click_start in range(0, frame_count, EDGE_SAMPLE_RATE // 4):
        click_end = min(click_start + 96, frame_count)
        signal[click_start:click_end] += (
            np.hanning((click_end - click_start) * 2)[: click_end - click_start] * 0.7
        )
    pcm = np.clip(signal, -1.0, 1.0)
    pcm = (pcm * np.iinfo(np.int16).max).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(EDGE_SAMPLE_RATE)
        output.writeframes(pcm.tobytes())


@pytest.mark.skipif(importlib.util.find_spec("librosa") is None, reason="librosa not installed")
def test_synthetic_audio_produces_edge_baseline35(tmp_path: Path) -> None:
    audio = tmp_path / "synthetic.wav"
    _write_test_wav(audio)
    features, metadata = extract_edge_baseline35(audio)
    assert features.ndim == 2
    assert features.shape[1] == 35
    assert features.dtype == torch.float32
    assert 29 <= features.shape[0] <= 32
    assert torch.isfinite(features).all()
    assert set(torch.unique(features[:, 33]).tolist()) <= {0.0, 1.0}
    assert set(torch.unique(features[:, 34]).tolist()) <= {0.0, 1.0}
    assert metadata["sample_rate"] == 15360
    assert metadata["hop_length"] == 512
    assert metadata["target_fps"] == 30
    assert metadata["feature_dim"] == 35


def test_feature_channel_order_is_exact() -> None:
    assert len(EDGE_BASELINE_FEATURE_NAMES) == 35
    assert EDGE_BASELINE_FEATURE_NAMES[0] == "onset_strength"
    assert EDGE_BASELINE_FEATURE_NAMES[1:21] == tuple(f"mfcc_{index:02d}" for index in range(1, 21))
    assert EDGE_BASELINE_FEATURE_NAMES[21:33] == tuple(
        f"chroma_cens_{index:02d}" for index in range(1, 13)
    )
    assert EDGE_BASELINE_FEATURE_NAMES[33:] == ("onset_peak", "beat_peak")


def test_align_features_to_length_policies() -> None:
    features = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    assert torch.equal(align_features_to_length(features, 3, "trim"), features[:3])
    padded = align_features_to_length(features, 7, "pad_last")
    assert padded.shape == (7, 4)
    assert torch.equal(padded[-2:], features[-1:].expand(2, -1))
    assert align_features_to_length(features, 3, "trim_or_pad_last").shape[0] == 3
    assert align_features_to_length(features, 7, "trim_or_pad_last").shape[0] == 7
    with pytest.raises(ValueError, match="cannot shorten"):
        align_features_to_length(features, 3, "pad_last")
    with pytest.raises(ValueError, match="cannot extend"):
        align_features_to_length(features, 7, "trim")


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("mBR0", 80.0),
        ("gBR_sBM_cAll_d04_mBR0_ch01", 80.0),
        ("mHO4.wav", 130.0),
        ("my_arbitrary_song.wav", None),
    ],
)
def test_aist_tempo_prior_parsing(name: str, expected: float | None) -> None:
    assert get_aist_tempo_prior(name) == expected


@pytest.mark.parametrize("as_numpy", [True, False])
def test_loader_accepts_numpy_and_tensor_pt(tmp_path: Path, as_numpy: bool) -> None:
    path = tmp_path / "features.pt"
    array = np.zeros((4, 35), dtype=np.float64)
    torch.save(array if as_numpy else torch.from_numpy(array), path)
    loaded = load_music_feature_tensor(path)
    assert loaded.shape == (4, 35)
    assert loaded.dtype == torch.float32


def test_missing_legacy_musicfeat_uses_v2_beat_channel(tmp_path: Path) -> None:
    features = torch.zeros(6, 35)
    features[[1, 4], 34] = 1.0
    validate_musicfeat_v2(features)
    beats = load_music_beats(tmp_path, "sequence", features)
    assert torch.equal(beats, features[:, 34])


def test_legacy_beat_channel_is_preserved(tmp_path: Path) -> None:
    legacy_dir = tmp_path / "musicfeat"
    legacy_dir.mkdir()
    legacy = torch.zeros(5, 54)
    legacy[2, 53] = 1.0
    torch.save(legacy, legacy_dir / "sequence_musicfeat_fps30.pt")
    v2 = torch.zeros(5, 35)
    v2[1, 34] = 1.0
    assert torch.equal(load_music_beats(tmp_path, "sequence", v2), legacy[:, 53])
