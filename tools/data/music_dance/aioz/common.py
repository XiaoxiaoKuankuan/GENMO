#!/usr/bin/env python3
"""Shared, data-contract-focused helpers for AIOZ-GDANCE conversion."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import pickle
import random
import shutil
import tempfile
import wave
import zipfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

SPLITS = ("train", "val", "test")
TARGET_FPS = 30.0
POSE_DIM_SOURCE = 72
POSE_DIM_GENMO = 66
BODY_POSE_DIM_GENMO = 63
BETAS_DIM = 10
FORMAT_VERSION = 1


@dataclass(frozen=True)
class SequenceLabel:
    """One official group-sequence row and its group-level split."""

    group_id: str
    split: str
    music_genre: str
    dance_style: str

    def as_dict(self) -> dict[str, str]:
        return {
            "group_id": self.group_id,
            "split": self.split,
            "music_genre": self.music_genre,
            "dance_style": self.dance_style,
        }


@dataclass(frozen=True)
class WavInfo:
    sample_rate: int
    channels: int
    sample_width_bytes: int
    audio_frames: int

    @property
    def duration_sec(self) -> float:
        return self.audio_frames / self.sample_rate

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "sample_width_bytes": self.sample_width_bytes,
            "audio_frames": self.audio_frames,
            "duration_sec": self.duration_sec,
        }


def validate_group_id(group_id: str) -> None:
    """Reject IDs that cannot safely be used as one output filename component."""
    if not group_id or group_id in {".", ".."}:
        raise ValueError(f"invalid empty/reserved AIOZ group id: {group_id!r}")
    if any(character in group_id for character in ("/", "\\", "\x00")):
        raise ValueError(f"AIOZ group id contains a path separator: {group_id!r}")


def filename_frame_span(group_id: str) -> int | None:
    """Read the final ``start_end`` span used by official AIOZ clip IDs."""
    fields = group_id.rsplit("_", 2)
    if len(fields) != 3:
        return None
    try:
        start, end = int(fields[-2]), int(fields[-1])
    except ValueError:
        return None
    return end - start if end > start else None


class AiozRawDataset:
    """Read either the downloaded ZIP pair or an extracted AIOZ tree."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.dataset_zip_path = self.root / "GDANCE_Dataset.zip"
        self.labels_zip_path = self.root / "GDANCE_Labels.zip"
        self.motion_dir = self.root / "motions_smpl"
        self.music_dir = self.root / "musics"
        self._dataset_zip: zipfile.ZipFile | None = None
        self._labels_zip: zipfile.ZipFile | None = None

        zip_mode = self.dataset_zip_path.is_file() and self.labels_zip_path.is_file()
        directory_mode = self.motion_dir.is_dir() and self.music_dir.is_dir()
        if not zip_mode and not directory_mode:
            raise FileNotFoundError(
                "AIOZ root must contain GDANCE_Dataset.zip + GDANCE_Labels.zip, "
                f"or extracted motions_smpl/ + musics/: {self.root}"
            )
        self.mode = "zip" if zip_mode else "directory"

    def __enter__(self) -> AiozRawDataset:
        if self.mode == "zip":
            self._dataset_zip = zipfile.ZipFile(self.dataset_zip_path)
            self._labels_zip = zipfile.ZipFile(self.labels_zip_path)
        return self

    def __exit__(self, *_: object) -> None:
        if self._dataset_zip is not None:
            self._dataset_zip.close()
        if self._labels_zip is not None:
            self._labels_zip.close()
        self._dataset_zip = None
        self._labels_zip = None

    @property
    def dataset_zip(self) -> zipfile.ZipFile:
        if self._dataset_zip is None:
            raise RuntimeError("AiozRawDataset must be used as a context manager")
        return self._dataset_zip

    @property
    def labels_zip(self) -> zipfile.ZipFile:
        if self._labels_zip is None:
            raise RuntimeError("AiozRawDataset must be used as a context manager")
        return self._labels_zip

    def _read_text(self, relative_path: str, *, labels: bool = False) -> str:
        if self.mode == "zip":
            archive = self.labels_zip if labels else self.dataset_zip
            return archive.read(relative_path).decode("utf-8-sig")
        path = self.root / relative_path
        if not path.is_file() and labels:
            path = self.root / "labels" / relative_path
        return path.read_text(encoding="utf-8-sig")

    def read_labels(self) -> dict[str, list[SequenceLabel]]:
        result: dict[str, list[SequenceLabel]] = {}
        seen_global: dict[str, str] = {}
        for split in SPLITS:
            text = self._read_text(f"{split}_labels.csv", labels=True)
            rows = list(csv.DictReader(io.StringIO(text)))
            if not rows:
                raise ValueError(f"AIOZ {split}_labels.csv is empty")
            if set(rows[0]) != {"id", "music_genre", "dance_style"}:
                raise ValueError(f"unexpected {split}_labels.csv columns: {list(rows[0])}")
            labels: list[SequenceLabel] = []
            for row in rows:
                group_id = row["id"].strip()
                validate_group_id(group_id)
                if group_id in seen_global:
                    raise ValueError(
                        f"group split leakage/duplication: {group_id} is in both "
                        f"{seen_global[group_id]} and {split}"
                    )
                seen_global[group_id] = split
                labels.append(
                    SequenceLabel(
                        group_id=group_id,
                        split=split,
                        music_genre=row["music_genre"].strip(),
                        dance_style=row["dance_style"].strip(),
                    )
                )
            result[split] = labels
        return result

    def read_internal_split_ids(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for split in SPLITS:
            text = self._read_text(f"{split}_split_sequence_names.txt")
            values = [line.strip() for line in text.splitlines() if line.strip()]
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate IDs in {split}_split_sequence_names.txt")
            result[split] = values
        return result

    def inventory(self) -> dict[str, set[str]]:
        if self.mode == "zip":
            motion_ids = {
                Path(name).stem
                for name in self.dataset_zip.namelist()
                if name.startswith("motions_smpl/") and name.endswith(".pkl")
            }
            music_ids = {
                Path(name).stem
                for name in self.dataset_zip.namelist()
                if name.startswith("musics/") and name.endswith(".wav")
            }
        else:
            motion_ids = {path.stem for path in self.motion_dir.glob("*.pkl")}
            music_ids = {path.stem for path in self.music_dir.glob("*.wav")}
        return {"motion_ids": motion_ids, "music_ids": music_ids}

    def audit_splits_and_inventory(self) -> dict[str, Any]:
        labels = self.read_labels()
        internal = self.read_internal_split_ids()
        inventory = self.inventory()
        label_ids = {label.group_id for split_rows in labels.values() for label in split_rows}
        split_agreement = {
            split: set(internal[split]) == {row.group_id for row in labels[split]}
            for split in SPLITS
        }
        missing_motion = sorted(label_ids - inventory["motion_ids"])
        missing_music = sorted(label_ids - inventory["music_ids"])
        extra_motion = sorted(inventory["motion_ids"] - label_ids)
        extra_music = sorted(inventory["music_ids"] - label_ids)
        return {
            "mode": self.mode,
            "split_counts": {split: len(labels[split]) for split in SPLITS},
            "internal_split_counts": {split: len(internal[split]) for split in SPLITS},
            "split_csv_matches_internal_txt": split_agreement,
            "motion_file_count": len(inventory["motion_ids"]),
            "wav_file_count": len(inventory["music_ids"]),
            "missing_motion": missing_motion,
            "missing_music": missing_music,
            "extra_motion": extra_motion,
            "extra_music": extra_music,
        }

    def load_motion(self, group_id: str) -> Any:
        validate_group_id(group_id)
        if self.mode == "zip":
            # The downloaded dataset is trusted local input. Pickle is never
            # accepted from a network request at runtime.
            return pickle.loads(self.dataset_zip.read(f"motions_smpl/{group_id}.pkl"))
        with (self.motion_dir / f"{group_id}.pkl").open("rb") as file:
            return pickle.load(file)

    def wav_info(self, group_id: str) -> WavInfo:
        validate_group_id(group_id)
        if self.mode == "zip":
            stream: Any = self.dataset_zip.open(f"musics/{group_id}.wav")
        else:
            stream = (self.music_dir / f"{group_id}.wav").open("rb")
        with stream:
            with wave.open(stream, "rb") as wav:
                return WavInfo(
                    sample_rate=int(wav.getframerate()),
                    channels=int(wav.getnchannels()),
                    sample_width_bytes=int(wav.getsampwidth()),
                    audio_frames=int(wav.getnframes()),
                )

    def copy_wav(self, group_id: str, destination: str | Path) -> Path:
        validate_group_id(group_id)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if self.mode == "zip":
            with self.dataset_zip.open(f"musics/{group_id}.wav") as source:
                with destination.open("wb") as target:
                    shutil.copyfileobj(source, target)
        else:
            shutil.copyfile(self.music_dir / f"{group_id}.wav", destination)
        return destination


def flatten_labels(labels: dict[str, list[SequenceLabel]]) -> list[SequenceLabel]:
    return [row for split in SPLITS for row in labels[split]]


def select_group_labels(
    labels: dict[str, list[SequenceLabel]],
    *,
    sample_groups: int | None,
    seed: int,
    requested_group_ids: Iterable[str] | None = None,
) -> list[SequenceLabel]:
    """Select complete group records; small samples cover every split when possible."""
    all_rows = flatten_labels(labels)
    by_id = {row.group_id: row for row in all_rows}
    requested = list(requested_group_ids or [])
    if requested:
        missing = [group_id for group_id in requested if group_id not in by_id]
        if missing:
            raise KeyError(f"requested AIOZ group IDs are unknown: {missing}")
        if len(requested) != len(set(requested)):
            raise ValueError("--group-id contains duplicates")
        return [by_id[group_id] for group_id in requested]
    if sample_groups is None:
        return all_rows
    if sample_groups <= 0 or sample_groups > len(all_rows):
        raise ValueError(f"sample_groups must be in [1,{len(all_rows)}]")
    rng = random.Random(seed)
    selected: list[SequenceLabel] = []
    if sample_groups >= len(SPLITS):
        selected.extend(rng.choice(labels[split]) for split in SPLITS)
    selected_ids = {row.group_id for row in selected}
    remaining = [row for row in all_rows if row.group_id not in selected_ids]
    selected.extend(rng.sample(remaining, sample_groups - len(selected)))
    split_order = {split: index for index, split in enumerate(SPLITS)}
    return sorted(selected, key=lambda row: (split_order[row.split], row.group_id))


def inspect_motion_payload(payload: Any, group_id: str) -> dict[str, Any]:
    """Validate the exact downloaded PKL fields without silently coercing shapes."""
    if not isinstance(payload, dict):
        raise ValueError(f"{group_id}: motion pickle must contain a dict")
    required = {"smpl_poses", "root_trans", "smpl_betas", "meta"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"{group_id}: motion pickle is missing {sorted(missing)}")
    poses = np.asarray(payload["smpl_poses"])
    trans = np.asarray(payload["root_trans"])
    betas = np.asarray(payload["smpl_betas"])
    meta = payload["meta"]
    if poses.ndim != 3 or poses.shape[-1] != POSE_DIM_SOURCE:
        raise ValueError(f"{group_id}: smpl_poses must be [P,T,72], got {poses.shape}")
    persons, frames, _ = poses.shape
    if persons <= 0 or frames <= 0:
        raise ValueError(f"{group_id}: empty person/time dimension in {poses.shape}")
    if trans.shape != (persons, frames, 3):
        raise ValueError(
            f"{group_id}: root_trans must be {(persons, frames, 3)}, got {trans.shape}"
        )
    if betas.shape != (persons, frames, BETAS_DIM):
        raise ValueError(
            f"{group_id}: smpl_betas must be {(persons, frames, BETAS_DIM)}, got {betas.shape}"
        )
    for name, array in (("smpl_poses", poses), ("root_trans", trans), ("smpl_betas", betas)):
        if not np.issubdtype(array.dtype, np.floating):
            raise ValueError(f"{group_id}: {name} must be floating point, got {array.dtype}")
        if not np.isfinite(array).all():
            raise ValueError(f"{group_id}: {name} contains NaN or Inf")
    if not isinstance(meta, dict):
        raise ValueError(f"{group_id}: meta must be a dict, got {type(meta).__name__}")
    if "n_persons" in meta and int(meta["n_persons"]) != persons:
        raise ValueError(f"{group_id}: meta.n_persons={meta['n_persons']} but tensor P={persons}")
    if "orig_start" in meta and "orig_end" in meta:
        span = int(meta["orig_end"]) - int(meta["orig_start"])
        if span != frames:
            raise ValueError(f"{group_id}: meta frame span={span} but tensor T={frames}")
    filename_span = filename_frame_span(group_id)
    if filename_span is not None and filename_span != frames:
        raise ValueError(f"{group_id}: filename frame span={filename_span} but tensor T={frames}")
    return {
        "keys": list(payload.keys()),
        "num_persons": persons,
        "num_frames": frames,
        "smpl_poses_shape": list(poses.shape),
        "root_trans_shape": list(trans.shape),
        "smpl_betas_shape": list(betas.shape),
        "smpl_poses_dtype": str(poses.dtype),
        "root_trans_dtype": str(trans.dtype),
        "smpl_betas_dtype": str(betas.dtype),
        "meta": {key: _json_scalar(value) for key, value in meta.items()},
        "filename_frame_span": filename_span,
    }


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _resample_axis_angle(values: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    """Slerp ``[T,J,3]`` axis-angle rotations on the target frame clock."""
    from scipy.spatial.transform import Rotation, Slerp

    frames, joints, _ = values.shape
    target_frames = max(1, int(round(frames * target_fps / source_fps)))
    if frames == 1:
        return np.repeat(values, target_frames, axis=0).astype(np.float32)
    source_times = np.arange(frames, dtype=np.float64) / source_fps
    target_times = np.arange(target_frames, dtype=np.float64) / target_fps
    target_times = np.minimum(target_times, source_times[-1])
    result = np.empty((target_frames, joints, 3), dtype=np.float32)
    for joint in range(joints):
        rotations = Rotation.from_rotvec(values[:, joint].astype(np.float64))
        result[:, joint] = (
            Slerp(source_times, rotations)(target_times).as_rotvec().astype(np.float32)
        )
    return result


def _resample_linear(values: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    frames = values.shape[0]
    target_frames = max(1, int(round(frames * target_fps / source_fps)))
    if frames == 1:
        return np.repeat(values, target_frames, axis=0).astype(np.float32)
    source_times = np.arange(frames, dtype=np.float64) / source_fps
    target_times = np.arange(target_frames, dtype=np.float64) / target_fps
    target_times = np.minimum(target_times, source_times[-1])
    flattened = values.reshape(frames, -1)
    result = np.stack(
        [
            np.interp(target_times, source_times, flattened[:, index])
            for index in range(flattened.shape[1])
        ],
        axis=-1,
    )
    return result.reshape((target_frames, *values.shape[1:])).astype(np.float32)


def convert_person_motion(
    payload: dict[str, Any],
    *,
    group_id: str,
    person_id: int,
    source_fps: float,
    target_fps: float = TARGET_FPS,
) -> dict[str, torch.Tensor]:
    """Create the canonical GENMO SMPL body fields for one AIOZ dancer."""
    info = inspect_motion_payload(payload, group_id)
    if not 0 <= person_id < info["num_persons"]:
        raise IndexError(f"{group_id}: person_id={person_id} is outside P={info['num_persons']}")
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("source_fps and target_fps must be positive")

    # Exact source order: root, the next 21 SMPL body joints, then two joints
    # unused by GENMO. No coordinate transform or joint permutation is applied.
    poses_66 = np.asarray(payload["smpl_poses"][person_id, :, :POSE_DIM_GENMO])
    rotations = poses_66.reshape(poses_66.shape[0], 22, 3)
    transl = np.asarray(payload["root_trans"][person_id])
    betas = np.asarray(payload["smpl_betas"][person_id])
    if math.isclose(source_fps, target_fps, rel_tol=0.0, abs_tol=1e-9):
        rotations_out = np.ascontiguousarray(rotations, dtype=np.float32)
        transl_out = np.ascontiguousarray(transl, dtype=np.float32)
        betas_out = np.ascontiguousarray(betas, dtype=np.float32)
    else:
        rotations_out = _resample_axis_angle(rotations, source_fps, target_fps)
        transl_out = _resample_linear(transl, source_fps, target_fps)
        betas_out = _resample_linear(betas, source_fps, target_fps)
    if not (len(rotations_out) == len(transl_out) == len(betas_out)):
        raise RuntimeError(f"{group_id}: resampled pose/trans/betas lengths disagree")
    result = {
        "global_orient": torch.from_numpy(rotations_out[:, 0]).float().contiguous(),
        "body_pose": torch.from_numpy(rotations_out[:, 1:].reshape(-1, 63)).float().contiguous(),
        "transl": torch.from_numpy(transl_out).float().contiguous(),
        "betas": torch.from_numpy(betas_out).float().contiguous(),
    }
    validate_canonical_smpl(result, source=f"{group_id}/person_{person_id}")
    return result


def validate_canonical_smpl(motion: Any, source: str = "motion") -> int:
    if not isinstance(motion, dict):
        raise ValueError(f"{source}: canonical motion must be a dict")
    expected_dims = {"global_orient": 3, "body_pose": 63, "transl": 3, "betas": 10}
    if set(motion) != set(expected_dims):
        raise ValueError(f"{source}: canonical keys must be exactly {sorted(expected_dims)}")
    lengths: set[int] = set()
    for name, dimension in expected_dims.items():
        tensor = motion[name]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{source}: {name} must be a torch.Tensor")
        if tensor.ndim != 2 or tensor.shape[1] != dimension:
            raise ValueError(f"{source}: {name} must be [T,{dimension}], got {tuple(tensor.shape)}")
        if tensor.dtype != torch.float32 or tensor.device.type != "cpu":
            raise ValueError(f"{source}: {name} must be CPU float32")
        if not tensor.is_contiguous() or not torch.isfinite(tensor).all():
            raise ValueError(f"{source}: {name} must be contiguous and finite")
        lengths.add(int(tensor.shape[0]))
    if len(lengths) != 1 or next(iter(lengths)) <= 0:
        raise ValueError(f"{source}: canonical fields must share one positive T")
    return next(iter(lengths))


def safe_torch_load(path: str | Path) -> Any:
    try:
        return torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(Path(path), map_location="cpu")


def atomic_torch_save(value: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as file:
        temporary = Path(file.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def summarize_counter(counter: Counter[Any]) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in sorted(counter.items(), key=lambda item: str(item[0]))
    }
