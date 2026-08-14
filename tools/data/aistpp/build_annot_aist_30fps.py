#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Build a custom partial AIST++ 30 FPS annotation set from local files only.

The resulting split is an engineering dataset made from all locally available
motions except official crossmodal validation/test sequences.  It is not the
official crossmodal training split and must not be used to claim benchmark
reproduction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gem.datasets.aistpp.aistplusplus import (  # noqa: E402
    load_music_feature_tensor,
    validate_aist_metric_translation,
    validate_musicfeat_v2,
)
from gem.utils.motion_utils import get_c_rootparam  # noqa: E402
from gem.utils.smplx_utils import make_smplx  # noqa: E402

DEFAULT_ANNOTATIONS_ROOT = Path("/home/weili/datasets/AISTPP_official/annotations")
DEFAULT_MUSICFEAT_DIR = Path("/home/weili/GENMO/inputs/AIST++/musicfeat_v2")
DEFAULT_OUTPUT_ROOT = Path("/home/weili/GENMO/inputs/AIST++")
DEFAULT_REPORT_DIR = Path("outputs/aistpp_partial_build_report")

DEFAULT_FILENAMES = {
    "annot": "annot_aist_30fps_partial.pt",
    "train": "train_partial.pt",
    "val": "val_partial.pt",
    "test": "test_partial.pt",
    "minitrain": "minitrain_partial.pt",
}

ANNOT_RECORD_KEYS = {
    "smpl_pose_global",
    "smpl_trans_global",
    "smpl_pose",
    "smpl_trans",
    "bbox_xyxy",
    "intrinsics",
    "T_w2c",
    "contact_supervision_valid",
    "height",
    "width",
}

PARTIAL_SPLIT_TYPE = "custom_partial_all_available_excluding_official_crossmodal_eval"
PARTIAL_DISCLAIMER = (
    "This is a custom partial AIST++ split. "
    "It is not the official 980-sequence crossmodal training split. "
    "It must not be used to claim official AIST++ benchmark reproduction."
)


class AISTPartialBuildError(RuntimeError):
    """Raised when a source record violates the requested data contract."""


@dataclass
class BuildReports:
    """Mutable report payloads populated by the partial builder."""

    manifest: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    bbox_fill: list[dict[str, Any]] = field(default_factory=list)
    invalid_records: list[dict[str, Any]] = field(default_factory=list)
    camera_environments: Counter[str] = field(default_factory=Counter)
    camera_dimensions: Counter[str] = field(default_factory=Counter)
    scalings: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    split_summary: dict[str, Any] = field(default_factory=dict)


def safe_torch_load(path: str | Path) -> Any:
    """Load a Torch artifact across versions with and without ``weights_only``."""
    path = Path(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def normalize_smpl_poses(poses: Any) -> np.ndarray:
    """Normalize official ``[N,24,3]`` or ``[N,72]`` poses to float32 ``[N,72]``."""
    array = np.asarray(poses)
    if array.ndim == 3 and array.shape[1:] == (24, 3):
        array = array.reshape(array.shape[0], 72)
    elif array.ndim != 2 or array.shape[1] != 72:
        raise ValueError(
            f"smpl_poses must have shape [N,24,3] or [N,72], got {array.shape}"
        )
    if array.shape[0] == 0:
        raise ValueError("smpl_poses is empty")
    result = np.ascontiguousarray(array, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("smpl_poses contains NaN or Inf")
    return result


def downsample_motion_indices(
    frame_count: int, source_fps: int, target_fps: int
) -> np.ndarray:
    """Return deterministic source indices for integer-ratio FPS downsampling."""
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("source_fps and target_fps must be positive")
    if source_fps % target_fps != 0:
        raise ValueError(
            "Only integer source_fps/target_fps ratios are supported; "
            f"got {source_fps}/{target_fps}"
        )
    stride = source_fps // target_fps
    return np.arange(0, frame_count, stride, dtype=np.int64)


def view_to_index(view: str) -> int:
    """Map an exact AIST++ camera name ``c01``..``c09`` to its array index."""
    valid = tuple(f"c{index:02d}" for index in range(1, 10))
    if view not in valid:
        raise ValueError(f"view must be one of {valid}, got {view!r}")
    return int(view[1:]) - 1


def select_keypoint_frames(
    keypoints: np.ndarray,
    view: str,
    motion_indices: np.ndarray,
    motion_frame_count: int,
    max_frame_difference: int,
) -> tuple[np.ndarray, int, int]:
    """Select one view and clamp only the final one or two missing KP frames."""
    array = np.asarray(keypoints)
    if array.ndim != 4 or array.shape[0] != 9 or array.shape[2:] != (17, 3):
        raise ValueError(f"keypoints2d must have shape [9,N,17,3], got {array.shape}")
    kp_frames = int(array.shape[1])
    if kp_frames <= 0:
        raise ValueError("keypoints2d has no frames")
    if abs(motion_frame_count - kp_frames) > max_frame_difference:
        raise ValueError(
            "motion/keypoint frame difference exceeds threshold: "
            f"motion={motion_frame_count}, keypoints={kp_frames}, "
            f"max={max_frame_difference}"
        )
    clamped = np.minimum(motion_indices, kp_frames - 1)
    clamped_count = int(np.count_nonzero(clamped != motion_indices))
    selected = np.ascontiguousarray(array[view_to_index(view), clamped], dtype=np.float32)
    return selected, clamped_count, kp_frames


def compute_tight_bboxes(
    keypoints: np.ndarray,
    width: int,
    height: int,
    confidence_threshold: float,
    min_valid_joints: int,
) -> tuple[np.ndarray, int]:
    """Compute tight per-frame boxes and interpolate/fill invalid frames."""
    array = np.asarray(keypoints, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (17, 3):
        raise ValueError(f"selected keypoints must have shape [L,17,3], got {array.shape}")
    if width <= 0 or height <= 0:
        raise ValueError("camera width and height must be positive")
    if min_valid_joints <= 0 or min_valid_joints > 17:
        raise ValueError("min_valid_joints must be in [1,17]")

    boxes = np.full((array.shape[0], 4), np.nan, dtype=np.float32)
    for frame_index, frame in enumerate(array):
        finite = np.isfinite(frame).all(axis=-1)
        valid = (
            finite
            & (frame[:, 2] >= confidence_threshold)
            & (frame[:, 0] >= 0.0)
            & (frame[:, 0] <= width - 1)
            & (frame[:, 1] >= 0.0)
            & (frame[:, 1] <= height - 1)
        )
        if int(valid.sum()) < min_valid_joints:
            continue
        xy = frame[valid, :2]
        x1, y1 = xy.min(axis=0)
        x2, y2 = xy.max(axis=0)
        x1 = float(np.clip(x1, 0, width - 1))
        x2 = float(np.clip(x2, 0, width - 1))
        y1 = float(np.clip(y1, 0, height - 1))
        y2 = float(np.clip(y2, 0, height - 1))
        if x2 > x1 and y2 > y1:
            boxes[frame_index] = (x1, y1, x2, y2)

    valid_frames = np.isfinite(boxes).all(axis=1)
    invalid_count = int((~valid_frames).sum())
    if not valid_frames.any():
        raise ValueError("sequence has no valid bbox frame")
    timeline = np.arange(len(boxes), dtype=np.float64)
    valid_timeline = timeline[valid_frames]
    for coordinate in range(4):
        boxes[:, coordinate] = np.interp(
            timeline, valid_timeline, boxes[valid_frames, coordinate]
        )
    if not np.isfinite(boxes).all() or np.any(boxes[:, 2] <= boxes[:, 0]) or np.any(
        boxes[:, 3] <= boxes[:, 1]
    ):
        raise ValueError("bbox interpolation produced an invalid box")
    return np.ascontiguousarray(boxes, dtype=np.float32), invalid_count


def parse_camera_mapping(path: str | Path) -> dict[str, str]:
    """Parse official ``sequence environment`` camera mapping lines."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"camera mapping does not exist: {path}")
    mapping: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"invalid camera mapping line {line_number}: {line!r}")
        sequence, environment = parts
        if sequence in mapping and mapping[sequence] != environment:
            raise ValueError(f"duplicate camera mapping for {sequence}")
        mapping[sequence] = environment
    return mapping


def select_camera(camera_payload: Any, view: str) -> dict[str, Any]:
    """Select a camera by exact ``name`` without relying on list order."""
    if not isinstance(camera_payload, list):
        raise ValueError("camera environment JSON must contain a list")
    matches = [camera for camera in camera_payload if camera.get("name") == view]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one camera named {view}, found {len(matches)}")
    return matches[0]


def build_camera_tensors(camera: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Build validated intrinsics and world-to-camera transform from official JSON."""
    required = {"name", "size", "matrix", "rotation", "translation", "distortions"}
    missing = required - set(camera)
    if missing:
        raise ValueError(f"camera is missing fields: {sorted(missing)}")
    size = np.asarray(camera["size"])
    matrix = np.asarray(camera["matrix"], dtype=np.float64)
    rotation = np.asarray(camera["rotation"], dtype=np.float64).reshape(-1)
    translation = np.asarray(camera["translation"], dtype=np.float64).reshape(-1)
    distortions = np.asarray(camera["distortions"], dtype=np.float64).reshape(-1)
    if size.shape != (2,) or matrix.shape != (3, 3):
        raise ValueError(f"invalid camera size/matrix shape: {size.shape}, {matrix.shape}")
    if rotation.shape != (3,) or translation.shape != (3,):
        raise ValueError(
            f"rotation/translation must each contain 3 values: {rotation.shape}, {translation.shape}"
        )
    if not all(
        np.isfinite(value).all()
        for value in (size, matrix, rotation, translation, distortions)
    ):
        raise ValueError("camera contains NaN or Inf")
    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0 or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError("camera dimensions and focal lengths must be positive")
    rotation_matrix, _ = cv2.Rodrigues(rotation)
    identity = np.eye(3, dtype=np.float64)
    if not np.allclose(rotation_matrix.T @ rotation_matrix, identity, atol=1e-6):
        raise ValueError("camera rotation is not orthogonal")
    if not math.isclose(float(np.linalg.det(rotation_matrix)), 1.0, abs_tol=1e-6):
        raise ValueError("camera rotation determinant is not 1")
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation_matrix.astype(np.float32)
    transform[:3, 3] = translation.astype(np.float32)
    intrinsics = torch.from_numpy(np.ascontiguousarray(matrix, dtype=np.float32)).contiguous()
    t_w2c = torch.from_numpy(transform).contiguous()
    if not torch.isfinite(intrinsics).all() or not torch.isfinite(t_w2c).all():
        raise ValueError("camera tensors contain NaN or Inf")
    if not torch.equal(t_w2c[3], torch.tensor([0.0, 0.0, 0.0, 1.0])):
        raise ValueError("T_w2c last row is invalid")
    return intrinsics.float().cpu(), t_w2c.float().cpu(), width, height


def camera_space_smpl(
    smpl_pose_global: np.ndarray,
    smpl_trans_global: np.ndarray,
    t_w2c: torch.Tensor,
    offset: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert only SMPL root parameters with GENMO's ``get_c_rootparam``."""
    pose_global = np.asarray(smpl_pose_global, dtype=np.float32)
    trans_global = np.asarray(smpl_trans_global, dtype=np.float32)
    global_orient_w = torch.from_numpy(pose_global[:, :3]).float()
    transl_w = torch.from_numpy(trans_global).float()
    offset = torch.as_tensor(offset).detach().cpu().float().reshape(3)
    global_orient_c, transl_c = get_c_rootparam(
        global_orient_w, transl_w, t_w2c, offset
    )
    pose_camera = np.ascontiguousarray(pose_global.copy(), dtype=np.float32)
    pose_camera[:, :3] = global_orient_c.detach().cpu().numpy()
    trans_camera = np.ascontiguousarray(transl_c.detach().cpu().numpy(), dtype=np.float32)

    # Recompute through the same public function used by the Dataset assertion.
    verify_orient, verify_trans = get_c_rootparam(
        global_orient_w, transl_w, t_w2c, offset
    )
    orient_error = float(
        np.max(np.abs(pose_camera[:, :3] - verify_orient.detach().cpu().numpy()))
    )
    trans_error = float(np.max(np.abs(trans_camera - verify_trans.detach().cpu().numpy())))
    if orient_error >= 1e-4 or trans_error >= 1e-4:
        raise ValueError(
            "camera root verification failed: "
            f"orient_error={orient_error:.3e}, trans_error={trans_error:.3e}"
        )
    return pose_camera, trans_camera


def normalize_aist_translation(
    smpl_trans: np.ndarray, smpl_scaling: float
) -> np.ndarray:
    """Convert fitted AIST++ translation to generic-SMPL metric scale.

    AIST++ fits a per-sequence scaled SMPL body. Its official motion-feature
    example divides ``smpl_trans`` by ``smpl_scaling`` before passing the motion
    to an unscaled generic SMPL model. GEM uses that generic model and metric
    velocity thresholds, so retaining fitted-scene units corrupts both the 151D
    root-velocity target and static-contact labels.
    """
    if not np.isfinite(smpl_scaling) or smpl_scaling == 0:
        raise ValueError(f"smpl_scaling must be finite and non-zero, got {smpl_scaling}")
    # Scale is a geometric magnitude. A small number of upstream ignored fits
    # contain a negative signed value; callers retain them only when explicitly
    # allowing ignored official IDs and mark their contact supervision invalid.
    scale_magnitude = abs(float(smpl_scaling))
    translation = np.asarray(smpl_trans, dtype=np.float32)
    return np.ascontiguousarray(
        translation / np.float32(scale_magnitude), dtype=np.float32
    )


def normalize_aist_camera_extrinsics(
    t_w2c: torch.Tensor, smpl_scaling: float
) -> torch.Tensor:
    """Put camera translation in the same generic-SMPL scale as the motion."""
    if not np.isfinite(smpl_scaling) or smpl_scaling == 0:
        raise ValueError(f"smpl_scaling must be finite and non-zero, got {smpl_scaling}")
    normalized = torch.as_tensor(t_w2c).detach().clone().float()
    normalized[:3, 3] /= abs(float(smpl_scaling))
    return normalized.contiguous()


def validate_music_features(path: str | Path, expected_length: int) -> torch.Tensor:
    """Load and validate aligned EDGE baseline35 features without changing length."""
    features = load_music_feature_tensor(path).contiguous().cpu().float()
    validate_musicfeat_v2(features, source=path)
    if int(features.shape[0]) != expected_length:
        raise ValueError(
            f"music feature length mismatch: features={features.shape[0]}, motion={expected_length}"
        )
    return features


def validate_annot_record(record: dict[str, Any], sequence: str = "<record>") -> int:
    """Validate one record against ``AISTPlusPlusSmplDataset``'s exact contract."""
    if not isinstance(record, dict) or set(record) != ANNOT_RECORD_KEYS:
        raise ValueError(f"{sequence}: record fields must be exactly {sorted(ANNOT_RECORD_KEYS)}")
    if not isinstance(record["contact_supervision_valid"], (bool, np.bool_)):
        raise ValueError(f"{sequence}: contact_supervision_valid must be boolean")
    # A stale pre-normalization artifact is finite and shape-correct, so shape
    # checks alone cannot catch its 80--100x root-velocity error.
    validate_aist_metric_translation(
        record["smpl_trans_global"], sequence_id=sequence
    )
    arrays = {
        name: record[name]
        for name in (
            "smpl_pose_global",
            "smpl_trans_global",
            "smpl_pose",
            "smpl_trans",
            "bbox_xyxy",
        )
    }
    if arrays["smpl_pose_global"].ndim != 2 or arrays["smpl_pose_global"].shape[1] != 72:
        raise ValueError(f"{sequence}: smpl_pose_global must be [L,72]")
    length = int(arrays["smpl_pose_global"].shape[0])
    expected_shapes = {
        "smpl_trans_global": (length, 3),
        "smpl_pose": (length, 72),
        "smpl_trans": (length, 3),
        "bbox_xyxy": (length, 4),
    }
    for name, array in arrays.items():
        if not isinstance(array, np.ndarray) or array.dtype != np.float32:
            raise ValueError(f"{sequence}: {name} must be a float32 NumPy array")
        if not array.flags.c_contiguous or not np.isfinite(array).all():
            raise ValueError(f"{sequence}: {name} must be contiguous and finite")
        if name in expected_shapes and array.shape != expected_shapes[name]:
            raise ValueError(
                f"{sequence}: {name} shape is {array.shape}, expected {expected_shapes[name]}"
            )
    if length <= 0:
        raise ValueError(f"{sequence}: motion is empty")
    for name, shape in (("intrinsics", (3, 3)), ("T_w2c", (4, 4))):
        tensor = record[name]
        if not isinstance(tensor, torch.Tensor) or tensor.shape != shape:
            raise ValueError(f"{sequence}: {name} must be Tensor{shape}")
        if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
            raise ValueError(f"{sequence}: {name} must be CPU float32")
        if not tensor.is_contiguous() or not torch.isfinite(tensor).all():
            raise ValueError(f"{sequence}: {name} must be contiguous and finite")
    if record["intrinsics"][0, 0] <= 0 or record["intrinsics"][1, 1] <= 0:
        raise ValueError(f"{sequence}: intrinsics focal lengths must be positive")
    if not isinstance(record["height"], int) or not isinstance(record["width"], int):
        raise ValueError(f"{sequence}: height and width must be integers")
    if record["height"] <= 0 or record["width"] <= 0:
        raise ValueError(f"{sequence}: height and width must be positive")
    return length


def build_partial_splits(
    built_sequences: set[str],
    crossmodal_val_ids: set[str],
    crossmodal_test_ids: set[str],
) -> tuple[list[str], list[str], list[str]]:
    """Build deterministic disjoint custom partial train/val/test lists."""
    val = built_sequences & crossmodal_val_ids
    test = built_sequences & crossmodal_test_ids
    if val & test:
        raise ValueError(f"official val/test overlap: {sorted(val & test)[:10]}")
    train = built_sequences - val - test
    return sorted(train), sorted(val), sorted(test)


def choose_minitrain(
    train: list[str], annot: dict[str, dict[str, Any]], size: int, min_frames: int
) -> list[str]:
    """Choose the first deterministic sufficiently long partial-train records."""
    if size <= 0 or min_frames <= 0:
        raise ValueError("minitrain size and minimum frames must be positive")
    return [
        sequence
        for sequence in sorted(train)
        if annot[sequence]["smpl_pose_global"].shape[0] >= min_frames
    ][:size]


def validate_outputs(
    annot: Any,
    splits: dict[str, Any],
    music_lengths: dict[str, int] | None = None,
) -> tuple[int, int]:
    """Validate complete annot/split objects and return records and total frames."""
    if not isinstance(annot, dict):
        raise ValueError("annot output must be a dict")
    total_frames = 0
    for sequence, record in annot.items():
        if not isinstance(sequence, str) or not sequence or sequence.startswith("__"):
            raise ValueError(f"invalid annot sequence key: {sequence!r}")
        length = validate_annot_record(record, sequence)
        if music_lengths is not None and music_lengths.get(sequence) != length:
            raise ValueError(f"{sequence}: saved record length disagrees with music feature")
        total_frames += length
    normalized: dict[str, set[str]] = {}
    for name in ("train", "val", "test", "minitrain"):
        split = splits.get(name)
        if not isinstance(split, list) or split != sorted(split):
            raise ValueError(f"{name} split must be a sorted Python list")
        if len(split) != len(set(split)):
            raise ValueError(f"{name} split contains duplicates")
        unknown = set(split) - set(annot)
        if unknown:
            raise ValueError(f"{name} split references missing annot records: {sorted(unknown)[:10]}")
        normalized[name] = set(split)
    if not normalized["train"]:
        raise ValueError("train_partial must be non-empty")
    if normalized["train"] & normalized["val"]:
        raise ValueError("train and val splits overlap")
    if normalized["train"] & normalized["test"]:
        raise ValueError("train and test splits overlap")
    if normalized["val"] & normalized["test"]:
        raise ValueError("val and test splits overlap")
    if not normalized["minitrain"] <= normalized["train"]:
        raise ValueError("minitrain must be a subset of train")
    return len(annot), total_frames


def _read_id_file(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"required ID list does not exist: {path}")
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _read_pickle(path: Path) -> Any:
    with path.open("rb") as file:
        return pickle.load(file)


def _load_motion(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    payload = _read_pickle(path)
    if not isinstance(payload, dict):
        raise ValueError("motion pickle must contain a dict")
    missing = {"smpl_poses", "smpl_scaling", "smpl_trans"} - set(payload)
    if missing:
        raise ValueError(f"motion is missing fields: {sorted(missing)}")
    poses = normalize_smpl_poses(payload["smpl_poses"])
    trans = np.ascontiguousarray(np.asarray(payload["smpl_trans"]), dtype=np.float32)
    if trans.shape != (poses.shape[0], 3):
        raise ValueError(f"smpl_trans shape is {trans.shape}, expected {(poses.shape[0], 3)}")
    if not np.isfinite(trans).all():
        raise ValueError("smpl_trans contains NaN or Inf")
    scaling_array = np.asarray(payload["smpl_scaling"], dtype=np.float64).reshape(-1)
    if scaling_array.size == 0 or not np.isfinite(scaling_array).all():
        raise ValueError("smpl_scaling is empty or non-finite")
    if scaling_array.size != 1:
        raise ValueError(f"smpl_scaling must contain one value, got {scaling_array.size}")
    scaling = float(scaling_array[0])
    return poses, normalize_aist_translation(trans, scaling), scaling


def _load_keypoints(path: Path) -> np.ndarray:
    payload = _read_pickle(path)
    if isinstance(payload, dict):
        if "keypoints2d" not in payload:
            raise ValueError("keypoint pickle dict is missing 'keypoints2d'")
        payload = payload["keypoints2d"]
    if not isinstance(payload, np.ndarray):
        raise ValueError(f"keypoints2d must be a dense NumPy array, got {type(payload).__name__}")
    if payload.ndim != 4 or payload.shape[0] != 9 or payload.shape[2:] != (17, 3):
        raise ValueError(f"keypoints2d must have shape [9,N,17,3], got {payload.shape}")
    return payload


def _record_skip(
    reports: BuildReports,
    sequence: str,
    reason: str,
    error: str = "",
    **manifest_values: Any,
) -> None:
    row = {"sequence": sequence, "status": "skipped", "reason": reason, "error": error}
    row.update(manifest_values)
    reports.skipped.append(row.copy())
    reports.manifest.append(row)


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        if not fieldnames:
            return
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_reports(report_dir: Path, reports: BuildReports) -> None:
    """Write all JSON/CSV reports outside the annot dictionary."""
    report_dir.mkdir(parents=True, exist_ok=True)
    _save_json(report_dir / "build_summary.json", reports.summary)
    _save_csv(report_dir / "sequence_manifest.csv", reports.manifest)
    _save_csv(report_dir / "skipped_sequences.csv", reports.skipped)
    _save_json(report_dir / "split_summary.json", reports.split_summary)
    _save_csv(report_dir / "bbox_fill_report.csv", reports.bbox_fill)

    camera_summary = {
        "selected_view": reports.summary.get("selected_view"),
        "environment_counts": dict(sorted(reports.camera_environments.items())),
        "image_dimension_counts": dict(sorted(reports.camera_dimensions.items())),
        "camera_convention": "X_camera = R_w2c @ X_world + tvec",
    }
    _save_json(report_dir / "camera_summary.json", camera_summary)
    scaling_values = [row["scaling"] for row in reports.scalings]
    scaling_summary: dict[str, Any] = {
        "note": (
            "smpl_trans and camera translation are divided by per-sequence "
            "smpl_scaling; scaling is not converted to betas"
        ),
        "count": len(scaling_values),
        "per_sequence": reports.scalings,
    }
    if scaling_values:
        values = np.asarray(scaling_values, dtype=np.float64)
        scaling_summary.update(
            {
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "std": float(values.std()),
            }
        )
    _save_json(report_dir / "scaling_statistics.json", scaling_summary)
    _save_json(report_dir / "invalid_records.json", reports.invalid_records)


def output_paths(args: argparse.Namespace) -> dict[str, Path]:
    """Resolve all partial artifact paths under the selected output root."""
    return {
        "annot": args.output_root / args.annot_filename,
        "train": args.output_root / args.train_split_filename,
        "val": args.output_root / args.val_split_filename,
        "test": args.output_root / args.test_split_filename,
        "minitrain": args.output_root / args.minitrain_split_filename,
    }


def atomic_save_outputs(
    annot: dict[str, dict[str, Any]],
    splits: dict[str, list[str]],
    paths: dict[str, Path],
    overwrite: bool,
) -> None:
    """Save all temporary files, reload/validate, then atomically replace finals."""
    for path in paths.values():
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing partial output: {path}")
    temporary = {name: path.with_name(path.name + ".tmp") for name, path in paths.items()}
    for path in temporary.values():
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite temporary output: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
    try:
        torch.save(annot, temporary["annot"])
        for name in ("train", "val", "test", "minitrain"):
            torch.save(splits[name], temporary[name])
        reloaded_annot = safe_torch_load(temporary["annot"])
        reloaded_splits = {
            name: safe_torch_load(temporary[name])
            for name in ("train", "val", "test", "minitrain")
        }
        validate_outputs(reloaded_annot, reloaded_splits)
        for name, final_path in paths.items():
            os.replace(temporary[name], final_path)
    except Exception:
        for path in temporary.values():
            path.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    """Create the partial AIST++ builder CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-root", type=Path, default=DEFAULT_ANNOTATIONS_ROOT)
    parser.add_argument("--musicfeat-dir", type=Path, default=DEFAULT_MUSICFEAT_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--annot-filename", default=DEFAULT_FILENAMES["annot"])
    parser.add_argument("--train-split-filename", default=DEFAULT_FILENAMES["train"])
    parser.add_argument("--val-split-filename", default=DEFAULT_FILENAMES["val"])
    parser.add_argument("--test-split-filename", default=DEFAULT_FILENAMES["test"])
    parser.add_argument(
        "--minitrain-split-filename", default=DEFAULT_FILENAMES["minitrain"]
    )
    parser.add_argument(
        "--view", choices=tuple(f"c{index:02d}" for index in range(1, 10)), default="c01"
    )
    parser.add_argument("--source-fps", type=int, default=60)
    parser.add_argument("--target-fps", type=int, default=30)
    parser.add_argument("--kp-confidence-threshold", type=float, default=0.1)
    parser.add_argument("--min-valid-joints", type=int, default=4)
    parser.add_argument("--max-motion-kp-frame-difference", type=int, default=2)
    parser.add_argument("--min-sequence-frames", type=int, default=120)
    parser.add_argument("--minitrain-size", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    required_paths = {
        "motions": args.annotations_root / "motions",
        "keypoints2d": args.annotations_root / "keypoints2d",
        "cameras": args.annotations_root / "cameras",
        "camera mapping": args.annotations_root / "cameras" / "mapping.txt",
        "splits": args.annotations_root / "splits",
        "ignore list": args.annotations_root / "ignore_list.txt",
        "music features": args.musicfeat_dir,
    }
    for label, path in required_paths.items():
        expected = path.is_file() if path.suffix else path.is_dir()
        if not expected:
            raise FileNotFoundError(f"required {label} path does not exist: {path}")
    downsample_motion_indices(1, args.source_fps, args.target_fps)
    if not 0 <= args.kp_confidence_threshold <= 1:
        raise ValueError("--kp-confidence-threshold must be in [0,1]")
    if not 1 <= args.min_valid_joints <= 17:
        raise ValueError("--min-valid-joints must be in [1,17]")
    if args.max_motion_kp_frame_difference < 0:
        raise ValueError("--max-motion-kp-frame-difference must be non-negative")
    if args.min_sequence_frames <= 0 or args.minitrain_size <= 0:
        raise ValueError("--min-sequence-frames and --minitrain-size must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if not args.dry_run:
        for path in output_paths(args).values():
            if path.exists() and not args.overwrite:
                raise FileExistsError(f"Refusing to overwrite existing partial output: {path}")


def _load_camera_environment(
    cameras_root: Path, environment: str, view: str
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, int, int]:
    path = cameras_root / f"{environment}.json"
    if not path.is_file():
        raise FileNotFoundError(f"camera environment JSON does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    camera = select_camera(payload, view)
    intrinsics, t_w2c, width, height = build_camera_tensors(camera)
    return camera, intrinsics, t_w2c, width, height


def build_partial_dataset(
    args: argparse.Namespace, reports: BuildReports
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]], dict[str, int]]:
    """Build all valid local records and custom partial split lists."""
    motions_root = args.annotations_root / "motions"
    keypoints_root = args.annotations_root / "keypoints2d"
    cameras_root = args.annotations_root / "cameras"
    splits_root = args.annotations_root / "splits"

    motion_paths = sorted(motions_root.glob("*.pkl"))
    keypoint_paths = sorted(keypoints_root.glob("*.pkl"))
    music_paths = sorted(args.musicfeat_dir.glob("*_musicfeat_fps30.pt"))
    camera_paths = sorted(cameras_root.glob("*.json"))
    motion_ids = {path.stem for path in motion_paths}
    keypoint_ids = {path.stem for path in keypoint_paths}
    music_ids = {
        path.name.removesuffix("_musicfeat_fps30.pt") for path in music_paths
    }
    camera_mapping = parse_camera_mapping(cameras_root / "mapping.txt")
    ignore_ids = _read_id_file(args.annotations_root / "ignore_list.txt")
    crossmodal_train = _read_id_file(splits_root / "crossmodal_train.txt")
    crossmodal_val = _read_id_file(splits_root / "crossmodal_val.txt")
    crossmodal_test = _read_id_file(splits_root / "crossmodal_test.txt")

    print(f"[Audit] motion files: {len(motion_paths)}")
    print(f"[Audit] keypoints2d files: {len(keypoint_paths)}")
    print(f"[Audit] camera JSON files: {len(camera_paths)}")
    print(f"[Audit] music feature files: {len(music_paths)}")
    print(f"[Audit] ignore list IDs: {len(ignore_ids)}")
    print(f"[Audit] crossmodal_val IDs: {len(crossmodal_val)}")
    print(f"[Audit] crossmodal_test IDs: {len(crossmodal_test)}")
    print(
        "[Audit] crossmodal_train/local motion intersection: "
        f"{len(crossmodal_train & motion_ids)}"
    )

    for sequence in sorted(motion_ids & ignore_ids):
        _record_skip(reports, sequence, "ignored_by_official_list")
    candidates = sorted(motion_ids - ignore_ids)
    if args.limit is not None:
        candidates = candidates[: args.limit]

    smpl_model = make_smplx("supermotion")
    offset = smpl_model.get_skeleton(torch.zeros(10))[0].detach().cpu().float()
    annot: dict[str, dict[str, Any]] = {}
    music_lengths: dict[str, int] = {}

    for index, sequence in enumerate(candidates, start=1):
        common_manifest = {
            "source_motion_frames": "",
            "source_keypoint_frames": "",
            "output_frames": "",
            "music_feature_frames": "",
            "selected_view": args.view,
            "environment": camera_mapping.get(sequence, ""),
            "invalid_bbox_frames_before_fill": "",
            "clamped_keypoint_frames": "",
            "scaling": "",
            "split_membership": "",
        }
        music_path = args.musicfeat_dir / f"{sequence}_musicfeat_fps30.pt"
        if sequence not in music_ids or not music_path.is_file():
            _record_skip(reports, sequence, "missing_music_feature", **common_manifest)
            continue
        if sequence not in keypoint_ids:
            message = f"keypoints2d file is missing for {sequence}"
            _record_skip(reports, sequence, "missing_keypoints", message, **common_manifest)
            if args.strict:
                raise AISTPartialBuildError(message)
            continue
        if sequence not in camera_mapping:
            message = f"camera mapping is missing for {sequence}"
            _record_skip(reports, sequence, "missing_camera_mapping", message, **common_manifest)
            if args.strict:
                raise AISTPartialBuildError(message)
            continue

        try:
            poses, trans, scaling = _load_motion(motions_root / f"{sequence}.pkl")
            motion_indices = downsample_motion_indices(
                len(poses), args.source_fps, args.target_fps
            )
            pose_global = np.ascontiguousarray(poses[motion_indices], dtype=np.float32)
            trans_global = np.ascontiguousarray(trans[motion_indices], dtype=np.float32)
            common_manifest["source_motion_frames"] = len(poses)
            common_manifest["output_frames"] = len(motion_indices)
            common_manifest["scaling"] = scaling

            keypoints = _load_keypoints(keypoints_root / f"{sequence}.pkl")
            selected_keypoints, clamped_count, kp_frames = select_keypoint_frames(
                keypoints,
                args.view,
                motion_indices,
                len(poses),
                args.max_motion_kp_frame_difference,
            )
            common_manifest["source_keypoint_frames"] = kp_frames
            common_manifest["clamped_keypoint_frames"] = clamped_count

            environment = camera_mapping[sequence]
            camera, intrinsics, t_w2c, width, height = _load_camera_environment(
                cameras_root, environment, args.view
            )
            t_w2c = normalize_aist_camera_extrinsics(t_w2c, scaling)
            common_manifest["environment"] = environment
            bboxes, invalid_bbox_count = compute_tight_bboxes(
                selected_keypoints,
                width,
                height,
                args.kp_confidence_threshold,
                args.min_valid_joints,
            )
            common_manifest["invalid_bbox_frames_before_fill"] = invalid_bbox_count

            music_features = validate_music_features(music_path, len(motion_indices))
            common_manifest["music_feature_frames"] = int(music_features.shape[0])
            pose_camera, trans_camera = camera_space_smpl(
                pose_global, trans_global, t_w2c, offset
            )
            record = {
                "smpl_pose_global": np.ascontiguousarray(pose_global, dtype=np.float32),
                "smpl_trans_global": np.ascontiguousarray(trans_global, dtype=np.float32),
                "smpl_pose": np.ascontiguousarray(pose_camera, dtype=np.float32),
                "smpl_trans": np.ascontiguousarray(trans_camera, dtype=np.float32),
                "bbox_xyxy": np.ascontiguousarray(bboxes, dtype=np.float32),
                "intrinsics": intrinsics.detach().cpu().float().contiguous(),
                "T_w2c": t_w2c.detach().cpu().float().contiguous(),
                "contact_supervision_valid": True,
                "height": int(height),
                "width": int(width),
            }
            length = validate_annot_record(record, sequence)
            if length != len(music_features):
                raise ValueError(
                    f"all record time dimensions must equal music length: {length} != {len(music_features)}"
                )
            annot[sequence] = record
            music_lengths[sequence] = len(music_features)
            reports.bbox_fill.append(
                {
                    "sequence": sequence,
                    "frames": length,
                    "invalid_bbox_frames_before_fill": invalid_bbox_count,
                    "filled_bbox_frames": invalid_bbox_count,
                }
            )
            reports.camera_environments[environment] += 1
            reports.camera_dimensions[f"{width}x{height}"] += 1
            reports.scalings.append({"sequence": sequence, "scaling": scaling})
            reports.manifest.append(
                {"sequence": sequence, "status": "built", "reason": "", **common_manifest}
            )
            _ = camera  # camera distortions are validated and summarized by environment.
        except Exception as exc:
            text = str(exc)
            if "music feature length mismatch" in text:
                reason = "music_length_mismatch"
            elif "musicfeat" in text or "music feature" in text:
                reason = "invalid_music_feature"
            elif "keypoint" in text:
                reason = "invalid_keypoints"
            elif "bbox" in text:
                reason = "invalid_bbox"
            elif "camera" in text or "T_w2c" in text or "intrinsics" in text:
                reason = "invalid_camera"
            else:
                reason = "invalid_motion"
            issue = {"sequence": sequence, "reason": reason, "error": text}
            reports.invalid_records.append(issue)
            _record_skip(reports, sequence, reason, text, **common_manifest)
            if args.strict:
                raise AISTPartialBuildError(f"{sequence}: {text}") from exc

        if index % 50 == 0 or index == len(candidates):
            print(f"[Build] {index}/{len(candidates)} candidates, built={len(annot)}")

    built_sequences = set(annot)
    train, val, test = build_partial_splits(
        built_sequences, crossmodal_val, crossmodal_test
    )
    minitrain = choose_minitrain(
        train, annot, args.minitrain_size, args.min_sequence_frames
    )
    splits = {"train": train, "val": val, "test": test, "minitrain": minitrain}
    membership = {
        sequence: name
        for name, split in (("train", train), ("val", val), ("test", test))
        for sequence in split
    }
    for row in reports.manifest:
        if row["status"] == "built":
            row["split_membership"] = membership[row["sequence"]]
    validate_outputs(annot, splits, music_lengths)

    skip_counts = Counter(row["reason"] for row in reports.skipped)
    total_frames = sum(record["smpl_pose_global"].shape[0] for record in annot.values())
    paths = output_paths(args)
    reports.summary.update(
        {
            "status": "dry_run_complete" if args.dry_run else "records_built",
            "mode": "partial_existing_411",
            "split_type": PARTIAL_SPLIT_TYPE,
            "disclaimer": PARTIAL_DISCLAIMER,
            "source_motion_count": len(motion_paths),
            "source_keypoints2d_count": len(keypoint_paths),
            "source_camera_json_count": len(camera_paths),
            "source_musicfeat_count": len(music_paths),
            "ignored_count": len(motion_ids & ignore_ids),
            "selected_candidate_count": len(candidates),
            "crossmodal_train_local_intersection": len(crossmodal_train & motion_ids),
            "successfully_built_count": len(annot),
            "skipped_missing_musicfeat": skip_counts["missing_music_feature"],
            "skipped_missing_keypoints": skip_counts["missing_keypoints"],
            "skipped_missing_camera": skip_counts["missing_camera_mapping"]
            + skip_counts["invalid_camera"],
            "skipped_length_mismatch": skip_counts["music_length_mismatch"],
            "skipped_invalid_bbox": skip_counts["invalid_bbox"],
            "skipped_invalid_keypoints": skip_counts["invalid_keypoints"],
            "skipped_invalid_motion": skip_counts["invalid_motion"],
            "skipped_invalid_music_feature": skip_counts["invalid_music_feature"],
            "train_partial_count": len(train),
            "val_partial_count": len(val),
            "test_partial_count": len(test),
            "minitrain_partial_count": len(minitrain),
            "total_frames": total_frames,
            "total_hours_at_30fps": total_frames / args.target_fps / 3600.0,
            "bbox_frames_filled": sum(
                row["filled_bbox_frames"] for row in reports.bbox_fill
            ),
            "bbox_sequences_with_fill": sum(
                row["filled_bbox_frames"] > 0 for row in reports.bbox_fill
            ),
            "clamped_keypoint_frame_count": sum(
                int(row.get("clamped_keypoint_frames") or 0)
                for row in reports.manifest
                if row["status"] == "built"
            ),
            "selected_view": args.view,
            "source_fps": args.source_fps,
            "target_fps": args.target_fps,
            "dry_run": args.dry_run,
            "output_files": {name: str(path.resolve()) for name, path in paths.items()},
        }
    )
    reports.split_summary.update(
        {
            "split_type": PARTIAL_SPLIT_TYPE,
            "disclaimer": PARTIAL_DISCLAIMER,
            "train_partial_count": len(train),
            "val_partial_count": len(val),
            "test_partial_count": len(test),
            "minitrain_partial_count": len(minitrain),
            "train_partial": train,
            "val_partial": val,
            "test_partial": test,
            "minitrain_partial": minitrain,
        }
    )
    return annot, splits, music_lengths


def _print_summary(summary: dict[str, Any]) -> None:
    print("=" * 72)
    print("AIST++ partial 工程数据集构建摘要")
    print(f"  成功构建:       {summary.get('successfully_built_count', 0)}")
    print(f"  缺音乐跳过:     {summary.get('skipped_missing_musicfeat', 0)}")
    print(f"  train/val/test:  {summary.get('train_partial_count', 0)}/"
          f"{summary.get('val_partial_count', 0)}/{summary.get('test_partial_count', 0)}")
    print(f"  minitrain:       {summary.get('minitrain_partial_count', 0)}")
    print(f"  总帧数:         {summary.get('total_frames', 0)}")
    print(f"  总时长:         {summary.get('total_hours_at_30fps', 0.0):.3f} 小时")
    print(f"  bbox 填充帧:    {summary.get('bbox_frames_filled', 0)}")
    print(PARTIAL_DISCLAIMER)
    print("=" * 72)


def main(argv: list[str] | None = None) -> int:
    """Run the partial AIST++ builder."""
    args = build_parser().parse_args(argv)
    _validate_args(args)
    reports = BuildReports()
    try:
        annot, splits, _ = build_partial_dataset(args, reports)
        if not args.dry_run:
            atomic_save_outputs(annot, splits, output_paths(args), args.overwrite)
            reports.summary["status"] = "complete"
            reports.summary["output_size_bytes"] = {
                name: path.stat().st_size for name, path in output_paths(args).items()
            }
        write_reports(args.report_dir, reports)
    except Exception as exc:
        reports.summary.setdefault("status", "failed")
        reports.summary["error"] = f"{type(exc).__name__}: {exc}"
        reports.summary.setdefault("disclaimer", PARTIAL_DISCLAIMER)
        try:
            write_reports(args.report_dir, reports)
        except Exception as report_exc:
            print(f"WARNING: failed to write build reports: {report_exc}", file=sys.stderr)
        raise
    _print_summary(reports.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
