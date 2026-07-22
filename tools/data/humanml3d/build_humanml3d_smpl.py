#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Build GENMO's HumanML3D SMPL-X motion/text metadata file.

This tool combines the official HumanML3D split, text annotations and index
mapping with GENMO's already-preprocessed AMASS SMPL-X tensors.  It deliberately
does not create T5 embeddings and does not transform AZ coordinates to AY;
``Humanml3dDataset`` performs that coordinate conversion while loading samples.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

# Running ``python tools/data/.../build_humanml3d_smpl.py`` puts the script
# directory, rather than the repository root, at sys.path[0].
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gem.utils.rotation_conversions import (  # noqa: E402
    axis_angle_to_matrix,
    matrix_to_axis_angle,
)

DEFAULT_HUMANML_ROOT = Path("/home/weili/datasets/HumanML3D_official")
DEFAULT_AMASS_FILE = Path("inputs/AMASS/hmr4d_support/smplxpose_v2.pth")
DEFAULT_MAPPING_CSV = Path("outputs/humanml3d_amass_exact_coverage.csv")
DEFAULT_OUTPUT = Path(
    "inputs/HumanML3D_SMPL/hmr4d_support/humanml3d_smplhpose_train.pth"
)
DEFAULT_REPORT_DIR = Path("outputs/humanml3d_build_report")

PREFIX_TRIM_SECONDS = {
    "eyes_japan_dataset": 3.0,
    "mpi_hdm05": 3.0,
    "totalcapture": 1.0,
    "mpi_limits": 1.0,
    "transitions_mocap": 0.5,
}

# The generated exact-coverage report uses collapsed family names.  Keep the
# official spellings above and resolve both forms explicitly.
_FAMILY_ALIASES = {
    "eyesjapandataset": "eyes_japan_dataset",
    "mpihdm05": "mpi_hdm05",
    "totalcapture": "totalcapture",
    "mpilimits": "mpi_limits",
    "transitionsmocap": "transitions_mocap",
}

SMPL_LEFT_RIGHT_PAIRS = (
    (1, 2),
    (4, 5),
    (7, 8),
    (10, 11),
    (13, 14),
    (16, 17),
    (18, 19),
    (20, 21),
)

MAPPING_REQUIRED_COLUMNS = {
    "new_name",
    "source_path",
    "start_frame",
    "end_frame",
    "in_train",
    "match_status",
    "amass_key",
    "normalized_family",
}

INDEX_REQUIRED_COLUMNS = {"source_path", "start_frame", "end_frame", "new_name"}

REPORT_JSON_FILES = (
    "invalid_text_lines.json",
    "dropped_subclips.json",
    "too_short_segments.json",
    "duration_mismatches.json",
    "negative_end_frames.json",
)


class HumanML3DBuildError(RuntimeError):
    """Raised when an input cannot satisfy the HumanML3D output contract."""


@dataclass(frozen=True)
class TextAnnotation:
    """One parsed line from an official HumanML3D text file."""

    caption: str
    tokens: list[str]
    f_tag: float
    to_tag: float


@dataclass(frozen=True)
class CropResult:
    """Resolved target-FPS crop bounds and their duration audit values."""

    start: int
    end: int
    expected_frames: int | None
    actual_frames: int
    negative_end_fallback: bool


@dataclass
class BuildReports:
    """Mutable report payloads populated while building the dataset."""

    built_records: list[dict[str, Any]] = field(default_factory=list)
    skipped_records: list[dict[str, Any]] = field(default_factory=list)
    invalid_text_lines: list[dict[str, Any]] = field(default_factory=list)
    dropped_subclips: list[dict[str, Any]] = field(default_factory=list)
    too_short_segments: list[dict[str, Any]] = field(default_factory=list)
    duration_mismatches: list[dict[str, Any]] = field(default_factory=list)
    negative_end_frames: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def safe_torch_load(path: str | Path) -> Any:
    """Load a Torch artifact across versions with and without ``weights_only``."""
    path = Path(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def normalize_motion_id(value: str) -> str:
    """Return a HumanML3D ID without its filename extension."""
    return Path(value.strip()).stem


def base_motion_id(motion_id: str) -> str:
    """Strip the official mirror prefix from a HumanML3D motion ID."""
    return motion_id[1:] if motion_id.startswith("M") else motion_id


def prefix_trim_seconds(normalized_family: str) -> float:
    """Resolve the official HumanML3D AMASS prefix trim for a family."""
    value = normalized_family.strip().lower().replace("-", "_")
    canonical = _FAMILY_ALIASES.get(value.replace("_", ""), value)
    return PREFIX_TRIM_SECONDS.get(canonical, 0.0)


def convert_crop_bounds(
    start_frame: int,
    end_frame: int,
    normalized_family: str,
    source_fps: float,
    target_fps: float,
    total_frames: int,
) -> CropResult:
    """Convert HumanML3D 20 FPS index bounds to clamped target-FPS bounds."""
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("source_fps and target_fps must be positive")
    if total_frames < 0:
        raise ValueError("total_frames must be non-negative")
    if start_frame < 0:
        raise ValueError(f"start_frame must be non-negative, got {start_frame}")

    prefix = prefix_trim_seconds(normalized_family)
    start = round((prefix + start_frame / source_fps) * target_fps)
    negative_fallback = end_frame < 0
    if negative_fallback:
        # Official numpy slicing ``[:-1]`` excludes one 20 FPS sample.  At a
        # 30 FPS target grid this corresponds to approximately 1.5 frames.
        excluded_target_frames = max(1, round(target_fps / source_fps))
        end = total_frames - excluded_target_frames
        expected_frames = None
    else:
        end = round((prefix + end_frame / source_fps) * target_fps)
        expected_frames = round(((end_frame - start_frame) / source_fps) * target_fps)

    start = min(max(start, 0), total_frames)
    end = min(max(end, 0), total_frames)
    return CropResult(
        start=start,
        end=end,
        expected_frames=expected_frames,
        actual_frames=end - start,
        negative_end_fallback=negative_fallback,
    )


def parse_text_line(line: str) -> TextAnnotation:
    """Parse ``caption#processed_tokens#start_time#end_time``."""
    parts = line.rstrip("\r\n").split("#")
    if len(parts) != 4:
        raise ValueError(f"expected 4 '#' separated fields, got {len(parts)}")
    caption = parts[0].strip()
    tokens = parts[1].strip().split()
    if not caption:
        raise ValueError("caption is empty")
    if not tokens:
        raise ValueError("processed token list is empty")
    try:
        f_tag = float(parts[2])
        to_tag = float(parts[3])
    except ValueError as exc:
        raise ValueError("start_time and end_time must be numeric") from exc
    if not math.isfinite(f_tag) or not math.isfinite(to_tag):
        raise ValueError("start_time and end_time must be finite")
    if not ((f_tag == 0.0 and to_tag == 0.0) or (f_tag >= 0.0 and to_tag > f_tag)):
        raise ValueError(
            "time tags must be (0, 0) for full motion or satisfy 0 <= start < end"
        )
    return TextAnnotation(caption=caption, tokens=tokens, f_tag=f_tag, to_tag=to_tag)


def group_text_annotations(
    annotations: Iterable[TextAnnotation],
) -> dict[tuple[float, float], list[TextAnnotation]]:
    """Group captions which describe the same deterministic motion interval."""
    grouped: dict[tuple[float, float], list[TextAnnotation]] = defaultdict(list)
    for annotation in annotations:
        grouped[(annotation.f_tag, annotation.to_tag)].append(annotation)
    return dict(grouped)


def make_segment_key(motion_id: str, start_sec: float, end_sec: float) -> str:
    """Create a deterministic HumanML3D subclip record key."""
    start_ms = round(start_sec * 1000.0)
    end_ms = round(end_sec * 1000.0)
    return f"{motion_id}__seg_{start_ms}_{end_ms}"


def slice_motion_tensor(tensor: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Slice and detach storage from the potentially very large source tensor."""
    return tensor[start:end].contiguous().clone().float().cpu()


def mirror_smpl_pose(pose: torch.Tensor) -> torch.Tensor:
    """Mirror 22-joint SMPL axis-angle poses across X and exchange left/right."""
    pose_cpu = torch.as_tensor(pose).detach().float().cpu()
    if pose_cpu.ndim != 2 or pose_cpu.shape[1] != 66:
        raise ValueError(f"pose must have shape [F, 66], got {tuple(pose_cpu.shape)}")
    rotations = axis_angle_to_matrix(pose_cpu.reshape(-1, 22, 3))
    reflection = torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=rotations.dtype))
    reflected = reflection @ rotations @ reflection
    order = list(range(22))
    for left, right in SMPL_LEFT_RIGHT_PAIRS:
        order[left], order[right] = order[right], order[left]
    reflected = reflected[:, order]
    mirrored = matrix_to_axis_angle(reflected).reshape(-1, 66)
    return mirrored.contiguous().float().cpu()


def mirror_translation(trans: torch.Tensor) -> torch.Tensor:
    """Mirror an SMPL root translation sequence across its X axis."""
    mirrored = torch.as_tensor(trans).detach().float().cpu().contiguous().clone()
    if mirrored.ndim != 2 or mirrored.shape[1] != 3:
        raise ValueError(f"trans must have shape [F, 3], got {tuple(mirrored.shape)}")
    mirrored[:, 0] *= -1.0
    return mirrored


def validate_mirror_transform(
    original_pose: torch.Tensor,
    original_trans: torch.Tensor,
    mirrored_pose: torch.Tensor,
    mirrored_trans: torch.Tensor,
    tolerance: float = 1e-4,
) -> None:
    """Validate proper rotations and numerical double-mirror recovery."""
    mirrored_rot = axis_angle_to_matrix(mirrored_pose.reshape(-1, 22, 3))
    identity = torch.eye(3, dtype=mirrored_rot.dtype)
    orthogonal_error = (mirrored_rot.transpose(-1, -2) @ mirrored_rot - identity).abs().max()
    determinant_error = (torch.linalg.det(mirrored_rot) - 1.0).abs().max()
    if float(orthogonal_error) >= tolerance or float(determinant_error) >= tolerance:
        raise ValueError(
            "mirrored rotations are invalid: "
            f"orthogonal_error={float(orthogonal_error):.3e}, "
            f"determinant_error={float(determinant_error):.3e}"
        )

    restored_pose = mirror_smpl_pose(mirrored_pose)
    restored_rot = axis_angle_to_matrix(restored_pose.reshape(-1, 22, 3))
    original_rot = axis_angle_to_matrix(original_pose.reshape(-1, 22, 3))
    rotation_error = (restored_rot - original_rot).abs().max()
    restored_trans = mirror_translation(mirrored_trans)
    translation_error = (restored_trans - original_trans).abs().max()
    if float(rotation_error) >= tolerance or float(translation_error) >= tolerance:
        raise ValueError(
            "double mirror did not recover the source motion: "
            f"rotation_error={float(rotation_error):.3e}, "
            f"translation_error={float(translation_error):.3e}"
        )


def _text_data(annotations: Sequence[TextAnnotation]) -> list[dict[str, Any]]:
    return [
        {"caption": annotation.caption, "tokens": list(annotation.tokens)}
        for annotation in annotations
    ]


def make_output_record(
    pose: torch.Tensor,
    trans: torch.Tensor,
    beta: torch.Tensor,
    gender: str,
    annotations: Sequence[TextAnnotation],
) -> dict[str, Any]:
    """Create and validate one record matching ``Humanml3dDataset`` exactly."""
    record = {
        "pose": torch.as_tensor(pose).detach().contiguous().clone().float().cpu(),
        "trans": torch.as_tensor(trans).detach().contiguous().clone().float().cpu(),
        "beta": torch.as_tensor(beta).detach().reshape(-1)[:10].contiguous().clone().float().cpu(),
        "gender": str(gender),
        "text_data": _text_data(annotations),
    }
    validate_output_record(record)
    return record


def validate_output_record(record: dict[str, Any], key: str = "<record>") -> None:
    """Validate one serialized record against the current dataset loader contract."""
    expected_keys = {"pose", "trans", "beta", "gender", "text_data"}
    if not isinstance(record, dict) or set(record) != expected_keys:
        raise ValueError(f"{key}: record fields must be exactly {sorted(expected_keys)}")
    pose, trans, beta = record["pose"], record["trans"], record["beta"]
    if not isinstance(pose, torch.Tensor) or pose.ndim != 2 or pose.shape[1] != 66:
        raise ValueError(f"{key}: pose must be Tensor[F, 66]")
    if not isinstance(trans, torch.Tensor) or trans.shape != (pose.shape[0], 3):
        raise ValueError(f"{key}: trans must be Tensor[F, 3] matching pose")
    if not isinstance(beta, torch.Tensor) or beta.shape != (10,):
        raise ValueError(f"{key}: beta must be Tensor[10]")
    for name, tensor in (("pose", pose), ("trans", trans), ("beta", beta)):
        if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
            raise ValueError(f"{key}: {name} must be CPU float32")
        if not tensor.is_contiguous():
            raise ValueError(f"{key}: {name} must be contiguous")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{key}: {name} contains NaN or Inf")
    if not isinstance(record["gender"], str):
        raise ValueError(f"{key}: gender must be a string")
    text_data = record["text_data"]
    if not isinstance(text_data, list) or not text_data:
        raise ValueError(f"{key}: text_data must be a non-empty list")
    for index, text in enumerate(text_data):
        if not isinstance(text, dict) or set(text) != {"caption", "tokens"}:
            raise ValueError(f"{key}: text_data[{index}] has an invalid structure")
        if not isinstance(text["caption"], str) or not text["caption"].strip():
            raise ValueError(f"{key}: text_data[{index}].caption is empty")
        if not isinstance(text["tokens"], list) or not text["tokens"]:
            raise ValueError(f"{key}: text_data[{index}].tokens is empty")
        if not all(isinstance(token, str) and token for token in text["tokens"]):
            raise ValueError(f"{key}: text_data[{index}].tokens contains an invalid token")


def validate_output_dataset(dataset: dict[str, dict[str, Any]]) -> tuple[int, int]:
    """Validate every top-level entry and return ``(records, frames)``."""
    if not isinstance(dataset, dict):
        raise ValueError("saved HumanML3D object must be a dict")
    total_frames = 0
    for key, record in dataset.items():
        if not isinstance(key, str) or not key or key.startswith("__"):
            raise ValueError(f"invalid top-level motion key: {key!r}")
        validate_output_record(record, key)
        total_frames += int(record["pose"].shape[0])
    return len(dataset), total_frames


class TextRepository:
    """Read official text files from ``texts/`` or a read-only ``texts.zip``."""

    def __init__(self, humanml_root: Path):
        dataset_root = humanml_root / "HumanML3D"
        self.text_dir = dataset_root / "texts"
        self.zip_path = dataset_root / "texts.zip"
        self._zip: zipfile.ZipFile | None = None
        self._zip_members: dict[str, str] = {}
        if self.text_dir.is_dir():
            self.source = str(self.text_dir.resolve())
        elif self.zip_path.is_file():
            self._zip = zipfile.ZipFile(self.zip_path, "r")
            self._zip_members = {
                Path(name).name: name
                for name in self._zip.namelist()
                if name.lower().endswith(".txt")
            }
            self.source = f"{self.zip_path.resolve()} (read-only ZIP fallback)"
        else:
            raise FileNotFoundError(
                "HumanML3D text annotations were not found. Expected either "
                f"{self.text_dir} or {self.zip_path}."
            )

    def read(self, motion_id: str) -> list[str] | None:
        """Read one official text file, returning ``None`` when absent."""
        filename = f"{motion_id}.txt"
        if self.text_dir.is_dir():
            path = self.text_dir / filename
            if not path.is_file():
                return None
            return path.read_text(encoding="utf-8").splitlines()
        member = self._zip_members.get(filename)
        if member is None or self._zip is None:
            return None
        with self._zip.open(member, "r") as file:
            return file.read().decode("utf-8").splitlines()

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()


def _parse_text_file(
    repository: TextRepository,
    motion_id: str,
    reports: BuildReports,
    strict: bool,
) -> list[TextAnnotation] | None:
    lines = repository.read(motion_id)
    if lines is None:
        reports.skipped_records.append(
            {"motion_id": motion_id, "reason": "missing_text_file"}
        )
        if strict:
            raise HumanML3DBuildError(f"Missing official text file for {motion_id}")
        return None

    annotations: list[TextAnnotation] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            annotations.append(parse_text_line(line))
        except ValueError as exc:
            issue = {
                "motion_id": motion_id,
                "line_number": line_number,
                "line": line,
                "error": str(exc),
            }
            reports.invalid_text_lines.append(issue)
            if strict:
                raise HumanML3DBuildError(
                    f"Invalid text line in {motion_id}.txt:{line_number}: {exc}"
                ) from exc
    return annotations


def _validate_amass_record(record: Any, amass_key: str) -> tuple[torch.Tensor, ...]:
    if not isinstance(record, dict):
        raise ValueError(f"AMASS record {amass_key!r} is not a dict")
    missing = [name for name in ("pose", "trans", "beta", "gender") if name not in record]
    if missing:
        raise ValueError(f"AMASS record {amass_key!r} is missing fields: {missing}")
    pose = torch.as_tensor(record["pose"])
    trans = torch.as_tensor(record["trans"])
    beta = torch.as_tensor(record["beta"])
    if pose.ndim != 2 or pose.shape[1] != 66:
        raise ValueError(f"AMASS {amass_key!r} pose shape is {tuple(pose.shape)}, expected [F,66]")
    if trans.shape != (pose.shape[0], 3):
        raise ValueError(
            f"AMASS {amass_key!r} trans shape is {tuple(trans.shape)}, expected {(pose.shape[0], 3)}"
        )
    if beta.numel() < 10:
        raise ValueError(f"AMASS {amass_key!r} beta has {beta.numel()} values, expected >=10")
    for name, tensor in (("pose", pose), ("trans", trans), ("beta", beta)):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"AMASS {amass_key!r} {name} contains NaN or Inf")
    return pose, trans, beta.reshape(-1)[:10], str(record["gender"])


def _read_mapping(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Mapping CSV does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        columns = set(reader.fieldnames or [])
        missing = MAPPING_REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"Mapping CSV is missing required columns: {sorted(missing)}")
        return list(reader)


def _read_index(path: Path) -> list[dict[str, str]]:
    """Read and validate the official HumanML3D ``index.csv``."""
    if not path.is_file():
        raise FileNotFoundError(f"HumanML3D index.csv does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        columns = set(reader.fieldnames or [])
        missing = INDEX_REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"HumanML3D index.csv is missing columns: {sorted(missing)}")
        return list(reader)


def _mapping_matches_index(mapping: dict[str, str], index_row: dict[str, str]) -> bool:
    """Check that a coverage row preserves its official index.csv identity and crop."""
    try:
        frame_match = int(float(mapping["start_frame"])) == int(
            float(index_row["start_frame"])
        ) and int(float(mapping["end_frame"])) == int(float(index_row["end_frame"]))
    except ValueError:
        return False
    return (
        normalize_motion_id(mapping["new_name"])
        == normalize_motion_id(index_row["new_name"])
        and mapping["source_path"].strip() == index_row["source_path"].strip()
        and frame_match
    )


def _read_train_ids(path: Path) -> tuple[set[str], set[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"HumanML3D train split does not exist: {path}")
    train_ids = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    return train_ids, {base_motion_id(motion_id) for motion_id in train_ids}


def _add_record(
    output: dict[str, dict[str, Any]] | None,
    record_key: str,
    pose: torch.Tensor,
    trans: torch.Tensor,
    beta: torch.Tensor,
    gender: str,
    annotations: Sequence[TextAnnotation],
    reports: BuildReports,
    *,
    source_motion_id: str,
    amass_key: str,
    mirrored: bool,
    subclip: bool,
    interval: tuple[float, float],
) -> int:
    if output is not None and record_key in output:
        raise HumanML3DBuildError(f"Duplicate output motion key: {record_key}")
    record = make_output_record(pose, trans, beta, gender, annotations)
    frames = int(record["pose"].shape[0])
    if output is not None:
        output[record_key] = record
    reports.built_records.append(
        {
            "motion_id": record_key,
            "source_motion_id": source_motion_id,
            "amass_key": amass_key,
            "mirrored": mirrored,
            "subclip": subclip,
            "start_sec": interval[0],
            "end_sec": interval[1],
            "frames": frames,
            "captions": len(annotations),
        }
    )
    return frames


def _build_variant_records(
    output: dict[str, dict[str, Any]] | None,
    motion_id: str,
    pose: torch.Tensor,
    trans: torch.Tensor,
    beta: torch.Tensor,
    gender: str,
    annotations: Sequence[TextAnnotation],
    amass_key: str,
    min_frames: int,
    target_fps: float,
    subclip_policy: str,
    mirrored: bool,
    reports: BuildReports,
    strict: bool,
) -> tuple[int, int, int]:
    """Build full and grouped subclip records for one original/mirror variant."""
    grouped = group_text_annotations(annotations)
    records = 0
    frames = 0
    subclips = 0

    full_annotations = grouped.get((0.0, 0.0), [])
    if full_annotations:
        frames += _add_record(
            output,
            motion_id,
            pose,
            trans,
            beta,
            gender,
            full_annotations,
            reports,
            source_motion_id=motion_id,
            amass_key=amass_key,
            mirrored=mirrored,
            subclip=False,
            interval=(0.0, 0.0),
        )
        records += 1

    for (start_sec, end_sec), interval_annotations in grouped.items():
        if (start_sec, end_sec) == (0.0, 0.0):
            continue
        segment_key = make_segment_key(motion_id, start_sec, end_sec)
        if subclip_policy == "full_only":
            reports.dropped_subclips.append(
                {
                    "motion_id": motion_id,
                    "segment_key": segment_key,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "captions": len(interval_annotations),
                    "reason": "subclip_policy=full_only",
                }
            )
            continue

        raw_start = round(start_sec * target_fps)
        raw_end = round(end_sec * target_fps)
        length = int(pose.shape[0])
        if raw_start < -2 or raw_end > length + 2:
            issue = {
                "motion_id": motion_id,
                "segment_key": segment_key,
                "start_frame": raw_start,
                "end_frame": raw_end,
                "base_frames": length,
                "reason": "subclip_out_of_bounds",
            }
            reports.skipped_records.append(issue)
            if strict:
                raise HumanML3DBuildError(
                    f"Subclip {segment_key} is outside its {length}-frame base motion"
                )
            continue
        start = min(max(raw_start, 0), length)
        end = min(max(raw_end, 0), length)
        segment_frames = end - start
        if segment_frames < min_frames:
            issue = {
                "motion_id": motion_id,
                "segment_key": segment_key,
                "start_frame": start,
                "end_frame": end,
                "frames": segment_frames,
                "min_frames": min_frames,
            }
            reports.too_short_segments.append(issue)
            continue
        frames += _add_record(
            output,
            segment_key,
            slice_motion_tensor(pose, start, end),
            slice_motion_tensor(trans, start, end),
            beta,
            gender,
            interval_annotations,
            reports,
            source_motion_id=motion_id,
            amass_key=amass_key,
            mirrored=mirrored,
            subclip=True,
            interval=(start_sec, end_sec),
        )
        records += 1
        subclips += 1

    if records == 0:
        reports.skipped_records.append(
            {"motion_id": motion_id, "amass_key": amass_key, "reason": "no_valid_text_groups"}
        )
        if strict:
            raise HumanML3DBuildError(f"{motion_id} has no valid full-motion or subclip text")
    return records, frames, subclips


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        if not fieldnames:
            file.write("")
            return
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_reports(report_dir: Path, reports: BuildReports) -> None:
    """Write all audit reports without modifying any source dataset."""
    report_dir.mkdir(parents=True, exist_ok=True)
    _save_json(report_dir / "build_summary.json", reports.summary)
    _save_csv(report_dir / "built_records.csv", reports.built_records)
    _save_csv(report_dir / "skipped_records.csv", reports.skipped_records)
    for filename in REPORT_JSON_FILES:
        _save_json(report_dir / filename, getattr(reports, Path(filename).stem))


def atomic_save_dataset(dataset: dict[str, dict[str, Any]], output_path: Path, overwrite: bool) -> int:
    """Save, reload, fully validate, then atomically publish the output PTH."""
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    if temporary_path.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing temporary file: {temporary_path}")
        temporary_path.unlink()

    try:
        torch.save(dataset, temporary_path)
        del dataset
        gc.collect()
        reloaded = safe_torch_load(temporary_path)
        validate_output_dataset(reloaded)
        del reloaded
        gc.collect()
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return output_path.stat().st_size


def build_parser() -> argparse.ArgumentParser:
    """Create the HumanML3D-to-GENMO conversion CLI parser."""
    parser = argparse.ArgumentParser(
        description="Build HumanML3D SMPL-X training motion/text metadata from preprocessed AMASS."
    )
    parser.add_argument("--humanml-root", type=Path, default=DEFAULT_HUMANML_ROOT)
    parser.add_argument("--amass-file", type=Path, default=DEFAULT_AMASS_FILE)
    parser.add_argument("--mapping-csv", type=Path, default=DEFAULT_MAPPING_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--source-fps", type=float, default=20.0)
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--min-frames", type=int, default=25)
    parser.add_argument(
        "--subclip-policy", choices=("group", "full_only"), default="group"
    )
    parser.add_argument("--no-mirror", action="store_true", help="Do not create official M records.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser


def _validate_arguments(args: argparse.Namespace) -> None:
    if args.source_fps <= 0 or args.target_fps <= 0:
        raise ValueError("--source-fps and --target-fps must be positive")
    if args.min_frames <= 0:
        raise ValueError("--min-frames must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    for path, label in (
        (args.humanml_root / "index.csv", "HumanML3D index.csv"),
        (args.humanml_root / "HumanML3D" / "train.txt", "HumanML3D train.txt"),
        (args.amass_file, "AMASS file"),
        (args.mapping_csv, "mapping CSV"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if not args.dry_run and args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")


def _mapping_stats(rows: Sequence[dict[str, str]]) -> tuple[list[dict[str, str]], Counter[str]]:
    train_rows = [row for row in rows if _parse_bool(row["in_train"])]
    statuses = Counter(row["match_status"].strip() for row in train_rows)
    exact = [
        row
        for row in train_rows
        if row["match_status"].strip() == "exact_family_path" and row["amass_key"].strip()
    ]
    return exact, statuses


def build_dataset(args: argparse.Namespace, reports: BuildReports) -> dict[str, dict[str, Any]] | None:
    """Execute input validation and build records; return ``None`` for dry-run."""
    index_rows = _read_index(args.humanml_root / "index.csv")
    mapping_rows = _read_mapping(args.mapping_csv)
    exact_rows, statuses = _mapping_stats(mapping_rows)
    exact_ids = [normalize_motion_id(row["new_name"]) for row in exact_rows]
    duplicate_ids = sorted(key for key, count in Counter(exact_ids).items() if count > 1)

    train_path = args.humanml_root / "HumanML3D" / "train.txt"
    train_ids, train_base_ids = _read_train_ids(train_path)
    exact_train_rows = [
        row for row in exact_rows if normalize_motion_id(row["new_name"]) in train_base_ids
    ]
    index_by_id = {
        normalize_motion_id(row["new_name"]): row
        for row in index_rows
        if normalize_motion_id(row["new_name"])
    }
    mapping_index_mismatches = [
        row
        for row in exact_train_rows
        if normalize_motion_id(row["new_name"]) not in index_by_id
        or not _mapping_matches_index(
            row, index_by_id[normalize_motion_id(row["new_name"])]
        )
    ]
    negative_rows = [row for row in exact_train_rows if int(float(row["end_frame"])) < 0]

    print(f"[Mapping] exact train base motions: {len(exact_train_rows)}")
    print(f"[Mapping] skipped HumanAct12: {statuses.get('humanact12_not_in_amass', 0)}")
    print(f"[Mapping] skipped unmatched: {statuses.get('unmatched', 0)}")
    print(f"[Mapping] duplicate motion IDs: {len(duplicate_ids)}")
    print(f"[Mapping] index.csv mismatches: {len(mapping_index_mismatches)}")
    print(f"[Mapping] negative exact end frames: {len(negative_rows)}")

    if duplicate_ids and args.strict:
        raise HumanML3DBuildError(f"Duplicate exact mapping motion IDs: {duplicate_ids[:20]}")
    if mapping_index_mismatches and args.strict:
        ids = [normalize_motion_id(row["new_name"]) for row in mapping_index_mismatches]
        raise HumanML3DBuildError(
            f"Coverage CSV rows disagree with official index.csv: {ids[:20]}"
        )
    if negative_rows and args.strict:
        ids = [normalize_motion_id(row["new_name"]) for row in negative_rows]
        raise HumanML3DBuildError(
            f"Exact mappings contain negative end_frame values: {ids[:20]}"
        )

    print(f"[AMASS] Loading {args.amass_file} ...")
    amass = safe_torch_load(args.amass_file)
    if not isinstance(amass, dict):
        raise HumanML3DBuildError("AMASS artifact must contain a dict")
    missing_amass_keys = sorted(
        {row["amass_key"].strip() for row in exact_train_rows if row["amass_key"].strip() not in amass}
    )
    print(f"[AMASS] records: {len(amass)}, missing mapped keys: {len(missing_amass_keys)}")
    if missing_amass_keys and args.strict:
        raise HumanML3DBuildError(f"Mapped AMASS keys are missing: {missing_amass_keys[:20]}")

    mismatched_ids = {
        normalize_motion_id(row["new_name"]) for row in mapping_index_mismatches
    }
    usable_rows = [
        row
        for row in exact_train_rows
        if normalize_motion_id(row["new_name"]) not in mismatched_ids
    ]
    selected_rows = usable_rows[: args.limit] if args.limit is not None else usable_rows
    output: dict[str, dict[str, Any]] | None = {} if not args.dry_run else None
    text_repository = TextRepository(args.humanml_root)
    print(f"[Text] source: {text_repository.source}")

    original_full = mirrored_full = subclip_records = 0
    original_subclips = mirrored_subclips = 0
    total_frames = 0
    invalid_motion_shapes = 0
    missing_text_files = 0
    processed_mapping_ids: set[str] = set()

    try:
        for row_index, row in enumerate(selected_rows, start=1):
            motion_id = normalize_motion_id(row["new_name"])
            amass_key = row["amass_key"].strip()
            if motion_id in processed_mapping_ids:
                reports.skipped_records.append(
                    {"motion_id": motion_id, "amass_key": amass_key, "reason": "duplicate_mapping_id"}
                )
                continue
            processed_mapping_ids.add(motion_id)
            if amass_key not in amass:
                reports.skipped_records.append(
                    {"motion_id": motion_id, "amass_key": amass_key, "reason": "missing_amass_key"}
                )
                continue

            try:
                source_pose, source_trans, beta, gender = _validate_amass_record(
                    amass[amass_key], amass_key
                )
                start_frame = int(float(row["start_frame"]))
                end_frame = int(float(row["end_frame"]))
                crop = convert_crop_bounds(
                    start_frame,
                    end_frame,
                    row["normalized_family"],
                    args.source_fps,
                    args.target_fps,
                    int(source_pose.shape[0]),
                )
            except (TypeError, ValueError) as exc:
                invalid_motion_shapes += 1
                reports.skipped_records.append(
                    {
                        "motion_id": motion_id,
                        "amass_key": amass_key,
                        "reason": "invalid_motion_or_frame_range",
                        "error": str(exc),
                    }
                )
                if args.strict:
                    raise HumanML3DBuildError(f"Invalid motion {motion_id}: {exc}") from exc
                continue

            if crop.negative_end_fallback:
                reports.negative_end_frames.append(
                    {
                        "motion_id": motion_id,
                        "amass_key": amass_key,
                        "end_frame": end_frame,
                        "resolved_end_target": crop.end,
                        "negative_end_fallback": True,
                    }
                )
            if crop.end <= crop.start:
                issue = {
                    "motion_id": motion_id,
                    "amass_key": amass_key,
                    "start_target": crop.start,
                    "end_target": crop.end,
                    "reason": "empty_crop",
                }
                reports.skipped_records.append(issue)
                if args.strict:
                    raise HumanML3DBuildError(f"Empty crop for {motion_id}: {issue}")
                continue
            if crop.actual_frames < args.min_frames:
                reports.skipped_records.append(
                    {
                        "motion_id": motion_id,
                        "amass_key": amass_key,
                        "frames": crop.actual_frames,
                        "min_frames": args.min_frames,
                        "reason": "base_motion_too_short",
                    }
                )
                continue
            if (
                crop.expected_frames is not None
                and abs(crop.actual_frames - crop.expected_frames) > 2
            ):
                issue = {
                    "motion_id": motion_id,
                    "amass_key": amass_key,
                    "expected_target_frames": crop.expected_frames,
                    "actual_frames": crop.actual_frames,
                    "difference": crop.actual_frames - crop.expected_frames,
                    "start_target": crop.start,
                    "end_target": crop.end,
                }
                reports.duration_mismatches.append(issue)
                if args.strict:
                    raise HumanML3DBuildError(f"Duration mismatch for {motion_id}: {issue}")

            base_pose = slice_motion_tensor(source_pose, crop.start, crop.end)
            base_trans = slice_motion_tensor(source_trans, crop.start, crop.end)

            if motion_id in train_ids:
                annotations = _parse_text_file(
                    text_repository, motion_id, reports, args.strict
                )
                if annotations is None:
                    missing_text_files += 1
                else:
                    before = len(reports.built_records)
                    record_count, frame_count, subclips = _build_variant_records(
                        output,
                        motion_id,
                        base_pose,
                        base_trans,
                        beta,
                        gender,
                        annotations,
                        amass_key,
                        args.min_frames,
                        args.target_fps,
                        args.subclip_policy,
                        False,
                        reports,
                        args.strict,
                    )
                    total_frames += frame_count
                    subclip_records += subclips
                    original_subclips += subclips
                    original_full += record_count - subclips
                    assert len(reports.built_records) - before == record_count

            mirror_id = f"M{motion_id}"
            if not args.no_mirror and mirror_id in train_ids:
                mirror_annotations = _parse_text_file(
                    text_repository, mirror_id, reports, args.strict
                )
                if mirror_annotations is None:
                    missing_text_files += 1
                else:
                    mirrored_pose = mirror_smpl_pose(base_pose)
                    mirrored_trans = mirror_translation(base_trans)
                    try:
                        validate_mirror_transform(
                            base_pose, base_trans, mirrored_pose, mirrored_trans
                        )
                    except ValueError as exc:
                        reports.skipped_records.append(
                            {
                                "motion_id": mirror_id,
                                "amass_key": amass_key,
                                "reason": "mirror_validation_failed",
                                "error": str(exc),
                            }
                        )
                        if args.strict:
                            raise HumanML3DBuildError(
                                f"Mirror validation failed for {mirror_id}: {exc}"
                            ) from exc
                        continue
                    record_count, frame_count, subclips = _build_variant_records(
                        output,
                        mirror_id,
                        mirrored_pose,
                        mirrored_trans,
                        beta,
                        gender,
                        mirror_annotations,
                        amass_key,
                        args.min_frames,
                        args.target_fps,
                        args.subclip_policy,
                        True,
                        reports,
                        args.strict,
                    )
                    total_frames += frame_count
                    subclip_records += subclips
                    mirrored_subclips += subclips
                    mirrored_full += record_count - subclips

            if row_index % 500 == 0 or row_index == len(selected_rows):
                print(
                    f"[Build] {row_index}/{len(selected_rows)} base motions, "
                    f"records={len(reports.built_records)}, frames={total_frames}"
                )
    finally:
        text_repository.close()

    reports.summary.update(
        {
            "status": "dry_run_complete" if args.dry_run else "records_built",
            "source_index_rows": len(index_rows),
            "exact_train_base_motions": len(exact_train_rows),
            "selected_exact_base_motions": len(selected_rows),
            "original_records_built": original_full,
            "mirrored_records_built": mirrored_full,
            "subclip_records_built": subclip_records,
            "original_subclip_records_built": original_subclips,
            "mirrored_subclip_records_built": mirrored_subclips,
            "total_records": len(reports.built_records),
            "total_frames": total_frames,
            "total_hours_at_30fps": total_frames / args.target_fps / 3600.0,
            "skipped_humanact12": statuses.get("humanact12_not_in_amass", 0),
            "skipped_unmatched": statuses.get("unmatched", 0),
            "mapping_duplicate_motion_ids": len(duplicate_ids),
            "mapping_index_mismatches": len(mapping_index_mismatches),
            "missing_amass_keys": len(missing_amass_keys),
            "missing_text_files": missing_text_files,
            "invalid_motion_shapes": invalid_motion_shapes,
            "duration_mismatches": len(reports.duration_mismatches),
            "negative_end_frames": len(reports.negative_end_frames),
            "invalid_text_lines": len(reports.invalid_text_lines),
            "dropped_subclips": len(reports.dropped_subclips),
            "too_short_segments": len(reports.too_short_segments),
            "text_source": text_repository.source,
            "output_path": str(args.output.resolve()),
            "output_size_bytes": 0,
            "source_fps": args.source_fps,
            "target_fps": args.target_fps,
            "min_frames": args.min_frames,
            "subclip_policy": args.subclip_policy,
            "mirror_enabled": not args.no_mirror,
            "dry_run": args.dry_run,
            "t5_embeddings_generated": False,
        }
    )
    return output


def _print_summary(summary: dict[str, Any]) -> None:
    print("=" * 68)
    print("HumanML3D SMPL-X 构建摘要")
    print(f"  exact 基础动作:   {summary.get('exact_train_base_motions', 0)}")
    print(f"  本次处理基础动作: {summary.get('selected_exact_base_motions', 0)}")
    print(f"  原动作记录:       {summary.get('original_records_built', 0)}")
    print(f"  镜像动作记录:     {summary.get('mirrored_records_built', 0)}")
    print(f"  子片段记录:       {summary.get('subclip_records_built', 0)}")
    print(f"  总记录:           {summary.get('total_records', 0)}")
    print(f"  总帧数:           {summary.get('total_frames', 0)}")
    print(f"  总时长:           {summary.get('total_hours_at_30fps', 0.0):.3f} 小时")
    print(f"  HumanAct12 跳过:  {summary.get('skipped_humanact12', 0)}")
    print(f"  unmatched 跳过:   {summary.get('skipped_unmatched', 0)}")
    print(f"  duration mismatch:{summary.get('duration_mismatches', 0)}")
    print(f"  输出:              {summary.get('output_path')}")
    print(f"  输出字节数:        {summary.get('output_size_bytes', 0)}")
    print("  T5 embedding:      未生成（由下一阶段单独生成）")
    print("=" * 68)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_arguments(args)
    reports = BuildReports()
    try:
        dataset = build_dataset(args, reports)
        if not args.dry_run:
            if dataset is None:
                raise AssertionError("non-dry-run build did not produce a dataset")
            records, frames = validate_output_dataset(dataset)
            if records != reports.summary["total_records"] or frames != reports.summary["total_frames"]:
                raise HumanML3DBuildError(
                    "In-memory output counts disagree with the build report: "
                    f"records={records}/{reports.summary['total_records']}, "
                    f"frames={frames}/{reports.summary['total_frames']}"
                )
            output_size = atomic_save_dataset(dataset, args.output, args.overwrite)
            reports.summary["output_size_bytes"] = output_size
            reports.summary["status"] = "complete"
        write_reports(args.report_dir, reports)
    except Exception as exc:
        reports.summary.setdefault("status", "failed")
        reports.summary["error"] = f"{type(exc).__name__}: {exc}"
        try:
            write_reports(args.report_dir, reports)
        except Exception as report_exc:
            print(f"WARNING: unable to write failure reports: {report_exc}", file=sys.stderr)
        raise
    _print_summary(reports.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
