#!/usr/bin/env python3
"""Data-contract helpers for converting the downloaded FineDance release."""

from __future__ import annotations

import json
import math
import wave
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from gem.utils.rotation_conversions import matrix_to_axis_angle, rotation_6d_to_matrix
from tools.data.music_dance.aioz.common import (
    WavInfo,
    _resample_axis_angle,
    _resample_linear,
    write_json,
    write_jsonl,
)

SPLITS = ("train", "val", "test")
TARGET_FPS = 30.0
SOURCE_FEATURE_DIM = 315
SOURCE_JOINTS = 52
SOURCE_ROTATION_DIM = SOURCE_JOINTS * 6
BODY_JOINTS = 22
BODY_POSE_DIM = BODY_JOINTS * 3
HAND_POSE_DIM = (SOURCE_JOINTS - BODY_JOINTS) * 3
FORMAT_VERSION = 1

# FineDance@Genre, copied verbatim from the official repository's
# dataset/FineDance_dataset.py (get_train_test_list("cross_genre")), commit
# 0476cd4. The code applies the ignore list after selecting test items.
OFFICIAL_GENRE_TEST_RAW = (
    "063", "132", "143", "036", "098", "198", "130", "012", "211", "193",
    "179", "065", "137", "161", "092", "120", "037", "109", "204", "144",
)
OFFICIAL_GENRE_IGNORE_RAW = (
    "116", "117", "118", "119", "120", "121", "122", "123", "202", "130",
)
# The official release claims train/val/test but its public loader calls all ten
# entries above "ignore". Eight are in the release's missing/corrupt 116--123
# block. The remaining complete IDs are the only reproducible validation
# holdout. This inference is emitted explicitly in every report.
OFFICIAL_GENRE_VALIDATION_HOLDOUT = ("130", "202")
OFFICIAL_SPLIT_SOURCE = (
    "https://github.com/li-ronghui/FineDance/blob/"
    "0476cd4/dataset/FineDance_dataset.py"
)


def validate_sample_id(sample_id: str) -> None:
    if len(sample_id) != 3 or not sample_id.isdigit():
        raise ValueError(f"FineDance sample ID must be exactly three digits: {sample_id!r}")


def list_ids(root: str | Path) -> dict[str, set[str]]:
    root = Path(root).expanduser().resolve()
    layout = {
        "motion": ("motion", ".npy"),
        "music_wav": ("music_wav", ".wav"),
        "music_npy": ("music_npy", ".npy"),
        "label_json": ("label_json", ".json"),
    }
    result: dict[str, set[str]] = {}
    for name, (directory, suffix) in layout.items():
        path = root / directory
        if not path.is_dir():
            raise FileNotFoundError(f"missing FineDance directory: {path}")
        values = {item.stem for item in path.iterdir() if item.is_file() and item.suffix == suffix}
        for sample_id in values:
            validate_sample_id(sample_id)
        result[name] = values
    return result


def inventory_audit(root: str | Path) -> dict[str, Any]:
    ids = list_ids(root)
    union = set().union(*ids.values())
    intersection = set.intersection(*ids.values())
    motion_wav = ids["motion"] & ids["music_wav"]
    return {
        "ids": {name: sorted(values) for name, values in ids.items()},
        "counts": {name: len(values) for name, values in ids.items()},
        "union_count": len(union),
        "four_way_intersection": sorted(intersection),
        "four_way_intersection_count": len(intersection),
        "motion_wav_intersection": sorted(motion_wav),
        "motion_wav_intersection_count": len(motion_wav),
        "missing_vs_union": {name: sorted(union - values) for name, values in ids.items()},
        "extra_vs_four_way_intersection": {
            name: sorted(values - intersection) for name, values in ids.items()
        },
    }


def read_label(root: str | Path, sample_id: str) -> dict[str, Any]:
    validate_sample_id(sample_id)
    path = Path(root) / "label_json" / f"{sample_id}.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {"name", "style1", "style2", "frames"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{path}: label keys must be exactly {sorted(expected)}")
    # One real release file (187.json) stores the song name as integer 711.
    # Preserve that observed value instead of fabricating a string. Style fields
    # remain categorical strings.
    if not isinstance(value["name"], (str, int, float)) or value["name"] == "":
        raise ValueError(f"{path}: name must be a non-empty string or numeric release value")
    if not all(isinstance(value[key], str) and value[key] for key in ("style1", "style2")):
        raise ValueError(f"{path}: style1/style2 must be non-empty strings")
    if not isinstance(value["frames"], int) or value["frames"] <= 0:
        raise ValueError(f"{path}: frames must be a positive integer")
    return value


def read_wav_info(root: str | Path, sample_id: str) -> WavInfo:
    validate_sample_id(sample_id)
    path = Path(root) / "music_wav" / f"{sample_id}.wav"
    with wave.open(str(path), "rb") as wav:
        return WavInfo(
            sample_rate=int(wav.getframerate()),
            channels=int(wav.getnchannels()),
            sample_width_bytes=int(wav.getsampwidth()),
            audio_frames=int(wav.getnframes()),
        )


def inspect_motion_array(data: Any, sample_id: str) -> dict[str, Any]:
    validate_sample_id(sample_id)
    if not isinstance(data, np.ndarray):
        raise ValueError(f"{sample_id}: motion must be a numpy array")
    if data.ndim != 2 or data.shape[1] != SOURCE_FEATURE_DIM:
        raise ValueError(f"{sample_id}: motion must be [T,315], got {data.shape}")
    if data.shape[0] <= 0 or not np.issubdtype(data.dtype, np.floating):
        raise ValueError(f"{sample_id}: motion must contain positive-length floating data")
    finite = np.isfinite(data)
    if not finite.all():
        raise ValueError(
            f"{sample_id}: motion contains NaN/Inf "
            f"(nan={int(np.isnan(data).sum())}, inf={int(np.isinf(data).sum())})"
        )
    rotations = data[:, 3:].reshape(data.shape[0], SOURCE_JOINTS, 6)
    first_norm = np.linalg.norm(rotations[..., :3], axis=-1)
    second_residual = rotations[..., 3:] - (
        (rotations[..., :3] * rotations[..., 3:]).sum(-1, keepdims=True)
        / np.maximum((rotations[..., :3] ** 2).sum(-1, keepdims=True), 1e-12)
    ) * rotations[..., :3]
    if float(first_norm.min()) <= 1e-6 or float(np.linalg.norm(second_residual, axis=-1).min()) <= 1e-6:
        raise ValueError(f"{sample_id}: degenerate rotation-6D vector")
    return {
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "num_frames": int(data.shape[0]),
        "translation_shape": [int(data.shape[0]), 3],
        "rotation_6d_shape": [int(data.shape[0]), SOURCE_JOINTS, 6],
        "nan_count": int(np.isnan(data).sum()),
        "inf_count": int(np.isinf(data).sum()),
        "finite": True,
        "min": float(data.min()),
        "max": float(data.max()),
    }


def load_motion(root: str | Path, sample_id: str, *, mmap: bool = False) -> np.ndarray:
    validate_sample_id(sample_id)
    data = np.load(
        Path(root) / "motion" / f"{sample_id}.npy",
        mmap_mode="r" if mmap else None,
        allow_pickle=False,
    )
    inspect_motion_array(data, sample_id)
    return data


def official_genre_split(available_ids: Iterable[str]) -> tuple[dict[str, list[str]], dict[str, Any]]:
    available = set(available_ids)
    test = (set(OFFICIAL_GENRE_TEST_RAW) - set(OFFICIAL_GENRE_IGNORE_RAW)) & available
    val = set(OFFICIAL_GENRE_VALIDATION_HOLDOUT) & available
    train = available - test - val
    splits = {"train": sorted(train), "val": sorted(val), "test": sorted(test)}
    metadata = {
        "name": "FineDance@Genre",
        "source": OFFICIAL_SPLIT_SOURCE,
        "official_test_raw": list(OFFICIAL_GENRE_TEST_RAW),
        "official_ignore_raw": list(OFFICIAL_GENRE_IGNORE_RAW),
        "validation_interpretation": (
            "The public loader does not expose a separate Genre val list. IDs 130 and 202 "
            "are the only complete files in its ignore list outside the missing 116--123 block, "
            "so they are retained as the reproducible validation holdout."
        ),
        "counts": {split: len(values) for split, values in splits.items()},
    }
    if set(splits["train"]) & set(splits["val"]) or set(splits["train"]) & set(splits["test"]):
        raise RuntimeError("FineDance official split construction leaked IDs")
    return splits, metadata


def split_lookup(splits: dict[str, list[str]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for split in SPLITS:
        for sample_id in splits[split]:
            if sample_id in result:
                raise ValueError(f"split leakage: {sample_id} in {result[sample_id]} and {split}")
            result[sample_id] = split
    return result


def convert_motion_array(
    data: np.ndarray,
    *,
    sample_id: str,
    source_fps: float,
    target_fps: float = TARGET_FPS,
) -> dict[str, torch.Tensor]:
    """Convert FineDance translation + 52x6D rotations to body axis-angle."""
    inspect_motion_array(data, sample_id)
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("source_fps and target_fps must be positive")
    source = torch.from_numpy(np.asarray(data, dtype=np.float32).copy())
    rotations = source[:, 3:].reshape(-1, SOURCE_JOINTS, 6)
    axis_angle_52 = matrix_to_axis_angle(rotation_6d_to_matrix(rotations)).cpu().numpy()
    translation = source[:, :3].cpu().numpy()
    if not math.isclose(source_fps, target_fps, rel_tol=0.0, abs_tol=1e-9):
        axis_angle_52 = _resample_axis_angle(axis_angle_52, source_fps, target_fps)
        translation = _resample_linear(translation, source_fps, target_fps)
    pose = torch.from_numpy(np.ascontiguousarray(axis_angle_52[:, :BODY_JOINTS])).reshape(-1, 66)
    transl = torch.from_numpy(np.ascontiguousarray(translation, dtype=np.float32))
    result = {
        "pose": pose.float().contiguous(),
        "transl": transl.float().contiguous(),
        # FineDance does not provide subject shape. Neutral zero shape is explicit,
        # not inferred from the motion.
        "betas": torch.zeros(len(pose), 10, dtype=torch.float32),
    }
    validate_canonical_motion(result, source=sample_id)
    return result


def validate_canonical_motion(motion: Any, source: str = "motion") -> int:
    if not isinstance(motion, dict):
        raise ValueError(f"{source}: canonical motion must be a dict")
    expected = {"pose": 66, "transl": 3, "betas": 10}
    if set(motion) != set(expected):
        raise ValueError(f"{source}: canonical tensor keys must be {sorted(expected)}")
    lengths: set[int] = set()
    for field, width in expected.items():
        value = motion[field]
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] != width:
            raise ValueError(f"{source}: {field} must be a Tensor [T,{width}]")
        if value.dtype != torch.float32 or value.device.type != "cpu":
            raise ValueError(f"{source}: {field} must be CPU float32")
        if not value.is_contiguous() or not torch.isfinite(value).all():
            raise ValueError(f"{source}: {field} must be contiguous and finite")
        lengths.add(int(value.shape[0]))
    if len(lengths) != 1 or next(iter(lengths)) <= 0:
        raise ValueError(f"{source}: motion fields must share one positive T")
    return next(iter(lengths))


def summarize_counts(values: Iterable[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(Counter(values).items())}


__all__ = [
    "FORMAT_VERSION",
    "OFFICIAL_GENRE_IGNORE_RAW",
    "OFFICIAL_GENRE_TEST_RAW",
    "SPLITS",
    "TARGET_FPS",
    "convert_motion_array",
    "inspect_motion_array",
    "inventory_audit",
    "list_ids",
    "load_motion",
    "official_genre_split",
    "read_label",
    "read_wav_info",
    "split_lookup",
    "summarize_counts",
    "validate_canonical_motion",
    "write_json",
    "write_jsonl",
]
