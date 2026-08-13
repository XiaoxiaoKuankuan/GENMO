#!/usr/bin/env python3
"""Shared, data-contract-focused helpers for CoMPAS3D conversion."""

from __future__ import annotations

import json
import math
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from gem.utils.music_features import EDGE_TARGET_FPS
from gem.utils.rotation_conversions import axis_angle_to_matrix, matrix_to_axis_angle
from tools.data.music_dance.aioz.common import (
    _resample_axis_angle,
    _resample_linear,
    write_json,
    write_jsonl,
)

SPLITS = ("train", "val", "test")
FORMAT_VERSION = 1
TARGET_FPS = float(EDGE_TARGET_FPS)
SOURCE_POSE_DIM = 165
SOURCE_BETAS_DIM = 300
SOURCE_JOINTS = 55
GENMO_POSE_DIM = 66

SEQUENCE_PATTERN = re.compile(
    r"^(?P<pair>Pair(?P<pair_number>\d+))_"
    r"(?P<song>song(?P<song_number>\d+))_"
    r"(?P<take>take(?P<take_number>\d+))$",
    re.IGNORECASE,
)

# The downloaded NPZ is standard SMPL-X axis-angle in this order. The first 66
# values are the exact GENMO body contract. Face and hands remain source-only.
SMPLX_POSE_LAYOUT = {
    "global_orient": (0, 3),
    "body_pose": (3, 66),
    "jaw_pose": (66, 69),
    "left_eye_pose": (69, 72),
    "right_eye_pose": (72, 75),
    "left_hand_pose": (75, 120),
    "right_hand_pose": (120, 165),
}

# Source MoSh world coordinates are Z-up. GENMO and its SMPL-X body model use
# Y-up. This maps (x, y, z)_source -> (x, z, -y)_target.
SOURCE_Z_UP_TO_GENMO_Y_UP = torch.tensor(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
    dtype=torch.float32,
)

# CoMPAS3D README official interaction benchmark split. Entries not listed are
# training data. This is reported for reproducibility but is not the default for
# music-only training because the four song identities leak across its splits.
OFFICIAL_VALIDATION = {
    "Pair1_song2_take2",
    "Pair2_song3_take1",
    "Pair5_song1_take1",
}
OFFICIAL_TEST = {
    "Pair1_song1_take1",
    "Pair2_song1_take2",
    "Pair3_song2_take1",
    "Pair4_song2_take2",
    "Pair5_song3_take1",
    "Pair6_song3_take2",
    "Pair7_song4_take1",
    "Pair8_song4_take2",
    "Pair9_song1_take1",
}


@dataclass(frozen=True)
class SequenceParts:
    sequence_id: str
    pair_id: str
    song_id: str
    take_id: str


@dataclass(frozen=True)
class SequenceFiles:
    parts: SequenceParts
    directory: Path
    mp4: Path
    leader: Path
    follower: Path

    @property
    def sequence_id(self) -> str:
        return self.parts.sequence_id

    def role_path(self, role: str) -> Path:
        if role == "leader":
            return self.leader
        if role == "follower":
            return self.follower
        raise ValueError(f"unknown CoMPAS3D role: {role}")


def parse_sequence_id(sequence_id: str) -> SequenceParts:
    match = SEQUENCE_PATTERN.fullmatch(sequence_id)
    if match is None:
        raise ValueError(f"invalid CoMPAS3D sequence id: {sequence_id!r}")
    return SequenceParts(
        sequence_id=sequence_id,
        pair_id=f"Pair{int(match.group('pair_number'))}",
        song_id=f"song{int(match.group('song_number'))}",
        take_id=f"take{int(match.group('take_number'))}",
    )


def _is_real_file(path: Path, *, minimum_bytes: int = 1024) -> bool:
    return path.is_file() and path.stat().st_size >= minimum_bytes


def _role_candidates(directory: Path, role: str) -> list[Path]:
    # Deliberately match by substring. The official release contains
    # Pair7_song2_take1_leaderi.npz and must not be lost to a strict suffix rule.
    return sorted(
        path
        for path in directory.glob("*.npz")
        if role in path.stem.lower() and _is_real_file(path)
    )


def discover_local_sequences(root: str | Path) -> tuple[list[SequenceFiles], list[dict[str, Any]]]:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"CoMPAS3D root does not exist: {root}")
    complete: list[SequenceFiles] = []
    incomplete: list[dict[str, Any]] = []
    for directory in sorted(root.glob("Pair*/Pair*_song*_take*")):
        if not directory.is_dir():
            continue
        try:
            parts = parse_sequence_id(directory.name)
        except ValueError:
            continue
        mp4 = sorted(path for path in directory.glob("*.mp4") if _is_real_file(path))
        leader = _role_candidates(directory, "leader")
        follower = _role_candidates(directory, "follower")
        reasons: list[str] = []
        if len(mp4) != 1:
            reasons.append(f"expected_one_real_mp4_found_{len(mp4)}")
        if len(leader) != 1:
            reasons.append(f"expected_one_real_leader_npz_found_{len(leader)}")
        if len(follower) != 1:
            reasons.append(f"expected_one_real_follower_npz_found_{len(follower)}")
        record = {
            "sequence_id": directory.name,
            "pair_id": parts.pair_id,
            "song_id": parts.song_id,
            "take_id": parts.take_id,
            "directory": str(directory),
            "mp4_candidates": [path.name for path in mp4],
            "leader_candidates": [path.name for path in leader],
            "follower_candidates": [path.name for path in follower],
            "reasons": reasons,
        }
        if reasons:
            incomplete.append(record)
        else:
            complete.append(SequenceFiles(parts, directory, mp4[0], leader[0], follower[0]))
    return complete, incomplete


def reference_inventory(reference_root: str | Path) -> dict[str, Any]:
    reference_root = Path(reference_root).expanduser().resolve()
    if not reference_root.is_dir():
        raise FileNotFoundError(f"CoMPAS3D reference root does not exist: {reference_root}")
    records: dict[str, dict[str, Any]] = {}
    for directory in sorted(reference_root.glob("Pair*/Pair*_song*_take*")):
        if not directory.is_dir():
            continue
        try:
            parts = parse_sequence_id(directory.name)
        except ValueError:
            continue
        npz = sorted(path.name for path in directory.glob("*.npz"))
        records[directory.name] = {
            "sequence_id": directory.name,
            "pair_id": parts.pair_id,
            "song_id": parts.song_id,
            "take_id": parts.take_id,
            "mp4_files": sorted(path.name for path in directory.glob("*.mp4")),
            "npz_files": npz,
            "leader_files": [name for name in npz if "leader" in Path(name).stem.lower()],
            "follower_files": [name for name in npz if "follower" in Path(name).stem.lower()],
        }
    all_mp4 = [path for path in reference_root.glob("Pair*/*/*.mp4")]
    all_npz = [path for path in reference_root.glob("Pair*/*/*.npz")]
    return {
        "root": str(reference_root),
        "sequence_count": len(records),
        "mp4_file_count": sum(len(row["mp4_files"]) for row in records.values()),
        "npz_file_count": sum(len(row["npz_files"]) for row in records.values()),
        "mp4_lfs_pointer_or_tiny_count": sum(not _is_real_file(path) for path in all_mp4),
        "mp4_materialized_count": sum(_is_real_file(path) for path in all_mp4),
        "npz_lfs_pointer_or_tiny_count": sum(not _is_real_file(path) for path in all_npz),
        "npz_materialized_count": sum(_is_real_file(path) for path in all_npz),
        "records": records,
    }


def inventory_audit(root: str | Path, reference_root: str | Path) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    complete, local_incomplete = discover_local_sequences(root)
    reference = reference_inventory(reference_root)
    local_dirs = {
        path.name: path
        for path in root.glob("Pair*/Pair*_song*_take*")
        if path.is_dir() and SEQUENCE_PATTERN.fullmatch(path.name)
    }
    incomplete_by_id = {row["sequence_id"]: row for row in local_incomplete}
    missing_or_incomplete: list[dict[str, Any]] = []
    complete_ids = {row.sequence_id for row in complete}
    for sequence_id, expected in sorted(reference["records"].items()):
        if sequence_id in complete_ids:
            continue
        directory = local_dirs.get(sequence_id)
        local_mp4 = sorted(path.name for path in directory.glob("*.mp4")) if directory else []
        local_npz = sorted(path.name for path in directory.glob("*.npz")) if directory else []
        reasons = list(incomplete_by_id.get(sequence_id, {}).get("reasons", []))
        if directory is None:
            reasons.append("local_sequence_directory_missing")
        if not expected["mp4_files"]:
            reasons.append("official_reference_has_no_mp4")
        missing_or_incomplete.append(
            {
                **{key: expected[key] for key in ("sequence_id", "pair_id", "song_id", "take_id")},
                "expected_mp4_files": expected["mp4_files"],
                "expected_npz_files": expected["npz_files"],
                "local_mp4_files": local_mp4,
                "local_npz_files": local_npz,
                "reasons": sorted(set(reasons)),
            }
        )
    real_mp4 = [path for path in root.glob("Pair*/*/*.mp4") if _is_real_file(path)]
    real_npz = [path for path in root.glob("Pair*/*/*.npz") if _is_real_file(path)]
    return {
        "raw_root": str(root),
        "reference": {key: value for key, value in reference.items() if key != "records"},
        "local_sequence_directory_count": len(local_dirs),
        "local_real_mp4_count": len(real_mp4),
        "local_real_npz_count": len(real_npz),
        "complete_sequence_count": len(complete),
        "complete_sequence_ids": [row.sequence_id for row in complete],
        "incomplete_sequence_count": len(missing_or_incomplete),
        "incomplete_sequences": missing_or_incomplete,
        "role_filename_anomalies": [
            {
                "sequence_id": path.parent.name,
                "filename": path.name,
                "recognized_role": "leader",
                "reason": "contains leader but does not end with _leader.npz",
            }
            for path in Path(reference_root).glob("Pair*/*/*.npz")
            if "leader" in path.stem.lower() and not path.stem.lower().endswith("_leader")
        ],
    }


def load_npz_fields(path: str | Path) -> dict[str, Any]:
    """Load the trusted local MoSh artifact, including its object marker arrays."""
    path = Path(path)
    if not _is_real_file(path):
        raise ValueError(f"missing, tiny, or LFS-pointer NPZ: {path}")
    try:
        with np.load(path, allow_pickle=True) as archive:
            return {key: np.asarray(archive[key]) for key in archive.files}
    except Exception as exc:
        raise ValueError(f"cannot load real CoMPAS3D NPZ {path}: {exc}") from exc


def validate_source_motion(fields: dict[str, Any], source: str = "CoMPAS3D NPZ") -> dict[str, Any]:
    required = {
        "gender", "surface_model_type", "mocap_frame_rate", "betas", "poses", "trans",
        "markers_obs", "markers_sim", "v_template",
    }
    missing = required - set(fields)
    if missing:
        raise ValueError(f"{source}: missing NPZ keys {sorted(missing)}")
    poses = np.asarray(fields["poses"])
    trans = np.asarray(fields["trans"])
    betas = np.asarray(fields["betas"])
    if poses.ndim != 2 or poses.shape[1] != SOURCE_POSE_DIM or poses.shape[0] <= 0:
        raise ValueError(f"{source}: poses must be [T,165], got {poses.shape}")
    if trans.shape != (poses.shape[0], 3):
        raise ValueError(f"{source}: trans must be [T,3], got {trans.shape}")
    if betas.shape != (SOURCE_BETAS_DIM,):
        raise ValueError(f"{source}: betas must be [300], got {betas.shape}")
    for name, value in (("poses", poses), ("trans", trans), ("betas", betas)):
        if not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
            raise ValueError(f"{source}: {name} must be finite floating data")
    fps = float(np.asarray(fields["mocap_frame_rate"]).item())
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"{source}: invalid mocap_frame_rate={fps}")
    gender = str(np.asarray(fields["gender"]).item()).lower()
    if gender not in {"male", "female", "neutral"}:
        raise ValueError(f"{source}: unsupported gender={gender!r}")
    model_type = str(np.asarray(fields["surface_model_type"]).item())
    for marker_key in ("markers_obs", "markers_sim"):
        marker_object = np.asarray(fields[marker_key])
        if marker_object.shape != (poses.shape[0], 53, 3):
            raise ValueError(
                f"{source}: {marker_key} must be [T,53,3], got {marker_object.shape}"
            )
        try:
            marker_numeric = np.asarray(marker_object, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{source}: {marker_key} cannot be converted to numeric data") from exc
        if not np.isfinite(marker_numeric).all():
            raise ValueError(f"{source}: {marker_key} contains NaN/Inf")
    v_template = np.asarray(fields["v_template"])
    if v_template.ndim != 0:
        raise ValueError(f"{source}: v_template must be an object scalar, got {v_template.shape}")
    return {
        "num_frames": int(poses.shape[0]),
        "fps": fps,
        "duration_sec": int(poses.shape[0]) / fps,
        "gender": gender,
        "surface_model_type": model_type,
        "pose_shape": list(poses.shape),
        "trans_shape": list(trans.shape),
        "betas_shape": list(betas.shape),
    }


def temporal_sample(
    poses: np.ndarray,
    trans: np.ndarray,
    *,
    source_fps: float,
    target_fps: float = TARGET_FPS,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if poses.ndim != 2 or poses.shape[1] != SOURCE_POSE_DIM or trans.shape != (len(poses), 3):
        raise ValueError("temporal_sample expects poses [T,165] and trans [T,3]")
    if math.isclose(source_fps, target_fps, rel_tol=0.0, abs_tol=1e-9):
        indices = np.arange(len(poses), dtype=np.int64)
        return poses.copy(), trans.copy(), {
            "method": "identity_already_30fps",
            "source_index_stride": 1,
            "source_indices_first_last": [0, int(indices[-1])],
        }
    if math.isclose(source_fps, 120.0, rel_tol=0.0, abs_tol=1e-9) and math.isclose(
        target_fps, 30.0, rel_tol=0.0, abs_tol=1e-9
    ):
        indices = np.arange(0, len(poses), 4, dtype=np.int64)
        return poses[indices].copy(), trans[indices].copy(), {
            "method": "deterministic_4_to_1_frame_selection",
            "source_index_stride": 4,
            "source_indices_first_last": [0, int(indices[-1])],
        }
    # This branch is not used by the currently downloaded release. It performs
    # quaternion/spherical interpolation through the existing shared helper,
    # never ordinary linear interpolation of axis-angle values.
    pose_joints = poses.reshape(len(poses), SOURCE_JOINTS, 3)
    sampled_pose = _resample_axis_angle(pose_joints, source_fps, target_fps).reshape(-1, 165)
    sampled_trans = _resample_linear(trans, source_fps, target_fps)
    return sampled_pose, sampled_trans, {
        "method": "quaternion_slerp_pose_and_linear_translation",
        "source_index_stride": None,
        "source_indices_first_last": None,
    }


def convert_source_motion(
    fields: dict[str, Any],
    *,
    source_pelvis: torch.Tensor,
    target_pelvis: torch.Tensor,
    target_fps: float = TARGET_FPS,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    info = validate_source_motion(fields)
    sampled_pose, sampled_trans, temporal = temporal_sample(
        np.asarray(fields["poses"], dtype=np.float64),
        np.asarray(fields["trans"], dtype=np.float64),
        source_fps=info["fps"],
        target_fps=target_fps,
    )
    pose = torch.from_numpy(np.ascontiguousarray(sampled_pose, dtype=np.float32))
    trans = torch.from_numpy(np.ascontiguousarray(sampled_trans, dtype=np.float32))
    source_pelvis = torch.as_tensor(source_pelvis, dtype=torch.float32).reshape(3)
    target_pelvis = torch.as_tensor(target_pelvis, dtype=torch.float32).reshape(3)
    Q = SOURCE_Z_UP_TO_GENMO_Y_UP
    global_orient = matrix_to_axis_angle(Q @ axis_angle_to_matrix(pose[:, :3]))
    # SMPL-X rotates around its shaped pelvis, not the world origin. Preserve the
    # physical pelvis path while moving from the source gender/300D shape to the
    # neutral zero-beta GENMO body.
    transl = (Q @ (trans + source_pelvis).unsqueeze(-1)).squeeze(-1) - target_pelvis
    body_pose = pose[:, 3:66].contiguous()
    canonical = {
        "pose": torch.cat((global_orient, body_pose), dim=-1).float().contiguous(),
        "global_orient": global_orient.float().contiguous(),
        "body_pose": body_pose.float().contiguous(),
        "transl": transl.float().contiguous(),
        "betas": torch.zeros(len(pose), 10, dtype=torch.float32),
    }
    validate_canonical_motion(canonical)
    metadata = {
        **temporal,
        "source_fps": info["fps"],
        "target_fps": float(target_fps),
        "source_num_frames": info["num_frames"],
        "target_num_frames": len(pose),
        "duration_before_sec": info["duration_sec"],
        "duration_after_sec": len(pose) / target_fps,
        "source_pelvis_offset": source_pelvis.tolist(),
        "target_pelvis_offset": target_pelvis.tolist(),
    }
    return canonical, metadata


def validate_canonical_motion(motion: dict[str, Any], source: str = "motion") -> int:
    expected = {
        "pose": 66,
        "global_orient": 3,
        "body_pose": 63,
        "transl": 3,
        "betas": 10,
    }
    if not isinstance(motion, dict) or set(motion) != set(expected):
        raise ValueError(f"{source}: canonical keys must be exactly {sorted(expected)}")
    lengths: set[int] = set()
    for key, width in expected.items():
        value = motion[key]
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] != width:
            raise ValueError(f"{source}: {key} must be Tensor [T,{width}]")
        if value.dtype != torch.float32 or value.device.type != "cpu":
            raise ValueError(f"{source}: {key} must be CPU float32")
        if not value.is_contiguous() or not torch.isfinite(value).all():
            raise ValueError(f"{source}: {key} must be contiguous and finite")
        lengths.add(int(value.shape[0]))
    if len(lengths) != 1 or next(iter(lengths)) <= 0:
        raise ValueError(f"{source}: all fields must share one positive T")
    torch.testing.assert_close(motion["pose"][:, :3], motion["global_orient"])
    torch.testing.assert_close(motion["pose"][:, 3:66], motion["body_pose"])
    return next(iter(lengths))


def build_splits(
    sequences: Iterable[SequenceFiles], strategy: str = "music_identity"
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    sequences = list(sequences)
    ids = {row.sequence_id for row in sequences}
    if strategy == "music_identity":
        song_to_split = {"song1": "train", "song2": "train", "song3": "val", "song4": "test"}
        unknown = sorted({row.parts.song_id for row in sequences} - set(song_to_split))
        if unknown:
            raise ValueError(f"music_identity split has no assignment for songs: {unknown}")
        splits = {
            split: sorted(row.sequence_id for row in sequences if song_to_split[row.parts.song_id] == split)
            for split in SPLITS
        }
        metadata = {
            "strategy": strategy,
            "default_for_music_generation": True,
            "song_identity_assignment": song_to_split,
            "rationale": (
                "All Pair/take performances of one original song stay in one split. "
                "This prevents music leakage for music-conditioned generation."
            ),
        }
    elif strategy == "official_interaction":
        splits = {
            "val": sorted(ids & OFFICIAL_VALIDATION),
            "test": sorted(ids & OFFICIAL_TEST),
            "train": sorted(ids - OFFICIAL_VALIDATION - OFFICIAL_TEST),
        }
        metadata = {
            "strategy": strategy,
            "default_for_music_generation": False,
            "source": "CoMPAS3D official README validation/test table",
            "warning": (
                "The same song identities occur across train/val/test. This reproduces the "
                "interaction benchmark but leaks music for music-conditioned generation."
            ),
        }
    else:
        raise ValueError("split strategy must be music_identity or official_interaction")
    lookup = {sequence_id: split for split, values in splits.items() for sequence_id in values}
    if len(lookup) != len(ids) or set(lookup) != ids:
        raise RuntimeError("split construction omitted or duplicated sequences")
    music_to_splits: dict[str, set[str]] = {}
    by_id = {row.sequence_id: row for row in sequences}
    for sequence_id, split in lookup.items():
        music_to_splits.setdefault(by_id[sequence_id].parts.song_id, set()).add(split)
    leakage = {
        song: sorted(values) for song, values in music_to_splits.items() if len(values) > 1
    }
    metadata.update(
        {
            "sequence_counts": {split: len(splits[split]) for split in SPLITS},
            "sequence_ids": splits,
            "music_identity_leakage": leakage,
            "music_identity_leakage_count": len(leakage),
        }
    )
    return splits, metadata


def split_lookup(splits: dict[str, list[str]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for split in SPLITS:
        for sequence_id in splits[split]:
            if sequence_id in result:
                raise ValueError(f"sequence split leakage: {sequence_id}")
            result[sequence_id] = split
    return result


def ffprobe_media(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    command = [
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)
    ]
    try:
        payload = json.loads(subprocess.check_output(command, text=True))
    except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise ValueError(f"ffprobe failed for {path}: {exc}") from exc
    video = next((row for row in payload.get("streams", []) if row.get("codec_type") == "video"), None)
    audio = next((row for row in payload.get("streams", []) if row.get("codec_type") == "audio"), None)
    if video is None or audio is None:
        raise ValueError(f"{path}: expected one video and one audio stream")

    def number(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def fraction(value: str | None) -> float | None:
        if not value or "/" not in value:
            return number(value)
        numerator, denominator = value.split("/", 1)
        denominator_value = float(denominator)
        return float(numerator) / denominator_value if denominator_value else None

    format_duration = number(payload.get("format", {}).get("duration"))
    return {
        "path": str(path.resolve()),
        "file_size_bytes": path.stat().st_size,
        "video": {
            "codec": video.get("codec_name"),
            "fps_fraction": video.get("avg_frame_rate"),
            "fps": fraction(video.get("avg_frame_rate")),
            "frame_count": int(video["nb_frames"]) if video.get("nb_frames") else None,
            "duration_sec": number(video.get("duration")) or format_duration,
            "width": video.get("width"),
            "height": video.get("height"),
        },
        "audio": {
            "codec": audio.get("codec_name"),
            "sample_format": audio.get("sample_fmt"),
            "sample_rate": int(audio["sample_rate"]) if audio.get("sample_rate") else None,
            "channels": audio.get("channels"),
            "channel_layout": audio.get("channel_layout"),
            "duration_sec": number(audio.get("duration")) or format_duration,
        },
        "container_duration_sec": format_duration,
    }


def summarize_numeric(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": int(len(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "within_1_count": int((np.abs(array) <= 1.000001).sum()),
        "within_2_count": int((np.abs(array) <= 2.000001).sum()),
    }


def summarize_counts(values: Iterable[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(Counter(values).items())}


__all__ = [
    "FORMAT_VERSION", "GENMO_POSE_DIM", "OFFICIAL_TEST", "OFFICIAL_VALIDATION",
    "SEQUENCE_PATTERN", "SMPLX_POSE_LAYOUT", "SOURCE_BETAS_DIM", "SOURCE_JOINTS",
    "SOURCE_POSE_DIM", "SOURCE_Z_UP_TO_GENMO_Y_UP", "SPLITS", "SequenceFiles",
    "SequenceParts", "TARGET_FPS", "build_splits", "convert_source_motion",
    "discover_local_sequences", "ffprobe_media", "inventory_audit", "load_npz_fields",
    "parse_sequence_id", "reference_inventory", "split_lookup", "summarize_counts",
    "summarize_numeric", "temporal_sample", "validate_canonical_motion",
    "validate_source_motion", "write_json", "write_jsonl",
]
