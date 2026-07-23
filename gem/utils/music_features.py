# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""EDGE-compatible baseline music features for GEM music conditioning."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

EDGE_TARGET_FPS = 30
EDGE_HOP_LENGTH = 512
EDGE_SAMPLE_RATE = EDGE_TARGET_FPS * EDGE_HOP_LENGTH
EDGE_BASELINE_FEATURE_NAMES = (
    "onset_strength",
    *(f"mfcc_{index:02d}" for index in range(1, 21)),
    *(f"chroma_cens_{index:02d}" for index in range(1, 13)),
    "onset_peak",
    "beat_peak",
)
EDGE_FEATURE_DIM = len(EDGE_BASELINE_FEATURE_NAMES)

_AIST_TEMPO_FAMILIES_10 = {"BR", "PO", "LO", "MH", "LH", "WA", "KR", "JS", "JB"}
_AIST_MUSIC_ID = re.compile(r"^m([A-Z]{2})(\d)$")


def _import_librosa():
    """Import librosa with an actionable error instead of fabricating features."""
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError(
            "EDGE baseline35 extraction requires librosa. Install the project again "
            "or run `python -m pip install 'librosa>=0.10,<0.11'`."
        ) from exc
    return librosa


def get_aist_tempo_prior(name: str | None) -> float | None:
    """Return EDGE's BPM prior for an AIST music/sequence name, if recognized."""
    if not name:
        return None
    stem = Path(name).stem
    candidates = [stem, *stem.split("_")]
    for candidate in candidates:
        match = _AIST_MUSIC_ID.fullmatch(candidate)
        if match is None:
            continue
        family, tempo_index_text = match.groups()
        tempo_index = int(tempo_index_text)
        if family in _AIST_TEMPO_FAMILIES_10:
            return float(tempo_index * 10 + 80)
        if family == "HO":
            return float(tempo_index * 5 + 110)
    return None


def _looks_like_aist_name(name: str) -> bool:
    """Distinguish AIST identifiers from arbitrary names containing similar text."""
    stem = Path(name).stem
    if _AIST_MUSIC_ID.fullmatch(stem):
        return True
    fields = stem.split("_")
    return (
        len(fields) >= 6
        and fields[0].startswith("g")
        and fields[3].startswith("d")
        and _AIST_MUSIC_ID.fullmatch(fields[4]) is not None
        and fields[5].startswith("ch")
    )


def _tempo_scalar(value: Any) -> float:
    """Normalize scalar/array tempo return values across librosa versions."""
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return 0.0
    return float(finite[0])


def _estimate_tempo(librosa: Any, waveform: np.ndarray, onset_envelope: np.ndarray) -> float:
    """Estimate BPM using librosa while passing EDGE timing parameters explicitly."""
    try:
        from librosa.feature.rhythm import tempo as tempo_function
    except ImportError:
        tempo_function = librosa.beat.tempo
    tempo = tempo_function(
        y=waveform,
        onset_envelope=onset_envelope,
        sr=EDGE_SAMPLE_RATE,
        hop_length=EDGE_HOP_LENGTH,
    )
    return _tempo_scalar(tempo)


def _audio_duration_seconds(librosa: Any, path: Path) -> float:
    """Read source duration without decoding the full waveform when supported."""
    try:
        duration = librosa.get_duration(path=str(path))
    except TypeError:
        # librosa < 0.10 used ``filename`` instead of ``path``.
        duration = librosa.get_duration(filename=str(path))
    duration = float(duration)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError(f"Unable to determine a positive duration for audio file: {path}")
    return duration


def _load_selected_waveform(
    librosa: Any,
    path: Path,
    *,
    start_sec: float,
    duration_sec: float | None,
    original_duration_sec: float,
) -> tuple[np.ndarray, str]:
    """Decode only the selected range, with a compatible full-decode fallback."""
    try:
        waveform, _ = librosa.load(
            str(path),
            sr=EDGE_SAMPLE_RATE,
            mono=True,
            offset=float(start_sec),
            duration=None if duration_sec is None else float(duration_sec),
        )
        decode_mode = "range"
    except (OSError, RuntimeError, TypeError, ValueError):
        full_waveform, _ = librosa.load(str(path), sr=EDGE_SAMPLE_RATE, mono=True)
        full_waveform = np.asarray(full_waveform, dtype=np.float32)
        start_sample = int(round(start_sec * EDGE_SAMPLE_RATE))
        if duration_sec is None:
            end_sample = full_waveform.shape[0]
        else:
            end_sample = min(
                full_waveform.shape[0],
                start_sample + int(round(duration_sec * EDGE_SAMPLE_RATE)),
            )
        waveform = full_waveform[start_sample:end_sample]
        decode_mode = "full_fallback"

    waveform = np.ascontiguousarray(np.asarray(waveform, dtype=np.float32))
    if waveform.size == 0:
        raise ValueError(
            f"The selected audio range is empty: start_sec={start_sec}, "
            f"source_duration={original_duration_sec:.6f}s"
        )
    return waveform, decode_mode


def extract_edge_baseline35(
    audio_path: str | Path,
    start_sec: float = 0.0,
    duration_sec: float | None = None,
    target_fps: int = EDGE_TARGET_FPS,
    aist_sequence_name: str | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Extract EDGE baseline features in the exact 35-channel training order.

    Waveform selection happens before feature extraction. Unlike EDGE's dataset
    preparation script, this function does not apply a fixed five-second crop;
    callers may process a full arbitrary song or request their own time range.
    """
    path = Path(audio_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {path}")
    if target_fps != EDGE_TARGET_FPS:
        raise ValueError(
            f"edge_baseline35 is defined at {EDGE_TARGET_FPS} FPS; got target_fps={target_fps}"
        )
    if not math.isfinite(start_sec) or start_sec < 0:
        raise ValueError(f"start_sec must be finite and >= 0; got {start_sec}")
    if duration_sec is not None and (not math.isfinite(duration_sec) or duration_sec <= 0):
        raise ValueError(f"duration_sec must be finite and > 0 when provided; got {duration_sec}")

    librosa = _import_librosa()
    original_duration_sec = _audio_duration_seconds(librosa, path)
    if start_sec >= original_duration_sec:
        raise ValueError(
            f"start_sec={start_sec} is outside the {original_duration_sec:.3f}s audio file"
        )
    waveform, audio_decode_mode = _load_selected_waveform(
        librosa,
        path,
        start_sec=float(start_sec),
        duration_sec=None if duration_sec is None else float(duration_sec),
        original_duration_sec=original_duration_sec,
    )

    onset_envelope = librosa.onset.onset_strength(
        y=waveform,
        sr=EDGE_SAMPLE_RATE,
        hop_length=EDGE_HOP_LENGTH,
    )
    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=EDGE_SAMPLE_RATE,
        n_mfcc=20,
        hop_length=EDGE_HOP_LENGTH,
    ).T
    chroma = librosa.feature.chroma_cens(
        y=waveform,
        sr=EDGE_SAMPLE_RATE,
        n_chroma=12,
        hop_length=EDGE_HOP_LENGTH,
    ).T
    frame_count = int(onset_envelope.shape[0])
    if mfcc.shape[0] != frame_count or chroma.shape[0] != frame_count:
        raise RuntimeError(
            "librosa returned inconsistent EDGE feature frame counts: "
            f"onset={frame_count}, mfcc={mfcc.shape[0]}, chroma={chroma.shape[0]}"
        )

    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_envelope,
        sr=EDGE_SAMPLE_RATE,
        hop_length=EDGE_HOP_LENGTH,
        units="frames",
    )
    onset_one_hot = np.zeros(frame_count, dtype=np.float32)
    onset_frames = np.asarray(onset_frames, dtype=np.int64)
    onset_frames = onset_frames[(onset_frames >= 0) & (onset_frames < frame_count)]
    onset_one_hot[onset_frames] = 1.0

    prior_name = aist_sequence_name if aist_sequence_name is not None else path.stem
    tempo_prior = (
        get_aist_tempo_prior(prior_name)
        if aist_sequence_name is not None or _looks_like_aist_name(path.stem)
        else None
    )
    estimated_tempo = _estimate_tempo(librosa, waveform, onset_envelope)
    start_bpm = tempo_prior if tempo_prior is not None else estimated_tempo
    if not np.isfinite(start_bpm) or start_bpm <= 0:
        start_bpm = 120.0
    _, beat_frames = librosa.beat.beat_track(
        y=waveform,
        sr=EDGE_SAMPLE_RATE,
        hop_length=EDGE_HOP_LENGTH,
        start_bpm=float(start_bpm),
        tightness=100,
        units="frames",
    )
    beat_one_hot = np.zeros(frame_count, dtype=np.float32)
    beat_frames = np.asarray(beat_frames, dtype=np.int64)
    beat_frames = beat_frames[(beat_frames >= 0) & (beat_frames < frame_count)]
    beat_one_hot[beat_frames] = 1.0

    feature_array = np.concatenate(
        [
            onset_envelope[:, None],
            mfcc,
            chroma,
            onset_one_hot[:, None],
            beat_one_hot[:, None],
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    if feature_array.shape != (frame_count, 35):
        raise RuntimeError(f"Unexpected EDGE feature shape: {feature_array.shape}")
    if not np.isfinite(feature_array).all():
        raise RuntimeError(f"EDGE feature extraction produced NaN or Inf for: {path}")

    selected_duration_sec = waveform.shape[0] / EDGE_SAMPLE_RATE
    metadata: dict[str, Any] = {
        "source_path": str(path.resolve()),
        "original_duration_sec": float(original_duration_sec),
        "selected_start_sec": float(start_sec),
        "selected_duration_sec": float(selected_duration_sec),
        "sample_rate": EDGE_SAMPLE_RATE,
        "hop_length": EDGE_HOP_LENGTH,
        "target_fps": EDGE_TARGET_FPS,
        "feature_dim": 35,
        "feature_names": list(EDGE_BASELINE_FEATURE_NAMES),
        "estimated_or_prior_bpm": float(
            tempo_prior if tempo_prior is not None else estimated_tempo
        ),
        "bpm_source": "aist_prior" if tempo_prior is not None else "librosa_estimate",
        "librosa_estimated_bpm": float(estimated_tempo),
        "feature_frames": frame_count,
        "audio_decode_mode": audio_decode_mode,
    }
    return torch.from_numpy(feature_array), metadata


AlignmentPolicy = Literal["trim", "pad_last", "trim_or_pad_last"]


def align_features_to_length(
    features: torch.Tensor,
    target_length: int,
    policy: AlignmentPolicy,
) -> torch.Tensor:
    """Align temporal length by explicit trimming and/or last-frame padding."""
    if not isinstance(features, torch.Tensor) or features.ndim != 2:
        raise ValueError("features must be a 2D torch.Tensor shaped [L, C]")
    if target_length <= 0:
        raise ValueError(f"target_length must be > 0; got {target_length}")
    if policy not in {"trim", "pad_last", "trim_or_pad_last"}:
        raise ValueError(f"Unknown feature alignment policy: {policy}")

    current_length = int(features.shape[0])
    if current_length == target_length:
        return features
    if current_length > target_length:
        if policy == "pad_last":
            raise ValueError(
                f"pad_last cannot shorten {current_length} frames to {target_length}; use trim"
            )
        return features[:target_length]
    if policy == "trim":
        raise ValueError(
            f"trim cannot extend {current_length} frames to {target_length}; use pad_last"
        )
    if current_length == 0:
        raise ValueError("Cannot pad an empty feature sequence")
    padding = features[-1:].expand(target_length - current_length, -1)
    return torch.cat([features, padding], dim=0)
