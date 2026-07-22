#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Build the deterministic BEAT2 split index consumed by GENMO."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import wave
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_ROOT = Path("/home/weili/datasets/BEAT2_official")
DEFAULT_REPORT_DIR = Path("outputs/beat2_build_report")
SPLITS = ("train", "val", "test", "minitrain", "additional")
SOURCE_SPLITS = ("train", "val", "test", "additional")
FORMAL_SPLITS = frozenset(("train", "val", "test"))
VALID_GENDERS = frozenset(("male", "female", "neutral"))
LFS_PREFIX = b"version https://git-lfs.github.com/spec/v1"


class BEAT2BuildError(RuntimeError):
    """Raised when a safe BEAT2 index cannot be published."""


@dataclass
class Reports:
    """Report rows accumulated while scanning BEAT2."""

    valid: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    missing_npz: list[dict[str, Any]] = field(default_factory=list)
    missing_wav: list[dict[str, Any]] = field(default_factory=list)
    orphan_npz: list[dict[str, Any]] = field(default_factory=list)
    orphan_wav: list[dict[str, Any]] = field(default_factory=list)
    invalid_npz: list[dict[str, Any]] = field(default_factory=list)
    short_audio: list[dict[str, Any]] = field(default_factory=list)
    too_short: list[dict[str, Any]] = field(default_factory=list)
    unknown_split: list[dict[str, Any]] = field(default_factory=list)
    duplicates: list[dict[str, Any]] = field(default_factory=list)
    subset_summary: dict[str, dict[str, Any]] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CSVEntry:
    """One normalized selected row from a BEAT2 split CSV."""

    video_id: str
    split: str
    row_number: int


@dataclass
class CSVParseResult:
    """Parsed CSV records plus full-file IDs used for orphan auditing."""

    entries: list[CSVEntry]
    all_ids: set[str]
    all_types: dict[str, str]
    full_row_count: int
    processed_row_count: int
    split_counts: Counter[str]
    duplicate_ids: set[str]


def safe_torch_load(path: str | Path) -> Any:
    """Load a trusted local Torch artifact across PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def normalize_split_type(value: str) -> str:
    """Normalize one supported BEAT2 CSV split label."""
    normalized = value.strip().lower()
    if normalized == "validation":
        return "val"
    if normalized not in SOURCE_SPLITS:
        raise ValueError(f"unknown split type: {value!r}")
    return normalized


def _canonical_csv_row(row: dict[str, Any]) -> dict[str, Any]:
    return {str(key).strip().lower(): value for key, value in row.items() if key is not None}


def parse_split_csv(
    path: str | Path,
    *,
    subset: str,
    reports: Reports,
    limit: int | None = None,
    strict: bool = False,
) -> CSVParseResult:
    """Parse and normalize a BEAT2 split CSV without guessing unknown labels."""
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = [str(name).strip().lower() for name in (reader.fieldnames or [])]
        if "id" not in fieldnames or "type" not in fieldnames:
            raise BEAT2BuildError(f"{path} must contain id and type columns")
        raw_rows = list(reader)

    all_ids: set[str] = set()
    all_types: dict[str, str] = {}
    for raw in raw_rows:
        row = _canonical_csv_row(raw)
        video_id = str(row.get("id") or "").strip()
        split_raw = str(row.get("type") or "").strip()
        if video_id:
            all_ids.add(video_id)
            try:
                all_types.setdefault(video_id, normalize_split_type(split_raw))
            except ValueError:
                all_types.setdefault(video_id, split_raw.lower())

    selected = raw_rows if limit is None else raw_rows[:limit]
    candidates: list[CSVEntry] = []
    seen: dict[str, CSVEntry] = {}
    duplicate_ids: set[str] = set()
    split_counts: Counter[str] = Counter()
    for row_number, raw in enumerate(selected, start=2):
        row = _canonical_csv_row(raw)
        video_id = str(row.get("id") or "").strip()
        split_raw = str(row.get("type") or "").strip()
        if not video_id or not split_raw:
            issue = "empty_id" if not video_id else "empty_type"
            detail = {
                "subset": subset,
                "row_number": row_number,
                "video_id": video_id,
                "split": split_raw,
                "reason": issue,
            }
            reports.skipped.append(detail)
            if strict:
                raise BEAT2BuildError(f"{path}:{row_number}: {issue}")
            continue
        try:
            split = normalize_split_type(split_raw)
        except ValueError:
            detail = {
                "subset": subset,
                "row_number": row_number,
                "video_id": video_id,
                "split": split_raw,
                "reason": "unknown_split",
            }
            reports.unknown_split.append(detail)
            reports.skipped.append(detail)
            if strict:
                raise BEAT2BuildError(
                    f"{path}:{row_number}: unknown split {split_raw!r}"
                ) from None
            continue
        split_counts[split] += 1
        entry = CSVEntry(video_id=video_id, split=split, row_number=row_number)
        previous = seen.get(video_id)
        if previous is not None:
            duplicate_ids.add(video_id)
            reports.duplicates.append(
                {
                    "subset": subset,
                    "video_id": video_id,
                    "first_row": previous.row_number,
                    "duplicate_row": row_number,
                    "first_split": previous.split,
                    "duplicate_split": split,
                    "split_conflict": previous.split != split,
                }
            )
            continue
        seen[video_id] = entry
        candidates.append(entry)

    entries = [entry for entry in candidates if entry.video_id not in duplicate_ids]
    return CSVParseResult(
        entries=entries,
        all_ids=all_ids,
        all_types=all_types,
        full_row_count=len(raw_rows),
        processed_row_count=len(selected),
        split_counts=split_counts,
        duplicate_ids=duplicate_ids,
    )


def is_lfs_pointer(path: str | Path) -> bool:
    """Return whether a file is an unresolved Git LFS pointer."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    with path.open("rb") as file:
        return file.read(len(LFS_PREFIX)) == LFS_PREFIX


def normalize_gender(value: Any) -> str:
    """Normalize supported scalar or one-element NumPy gender values."""
    array = np.asarray(value)
    if array.ndim == 0:
        item = array.item()
    elif array.size == 1:
        item = array.reshape(-1)[0].item()
    else:
        raise ValueError(f"gender must be scalar or one element, got shape {array.shape}")
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    gender = str(item).strip().lower()
    if gender not in VALID_GENDERS:
        raise ValueError(f"unsupported gender {gender!r}")
    return gender


def validate_npz(path: str | Path) -> dict[str, Any]:
    """Validate one motion NPZ against the current GENMO Loader contract."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError("missing_or_empty_npz")
    if is_lfs_pointer(path):
        raise ValueError("lfs_pointer_not_downloaded")
    try:
        motion = np.load(path, allow_pickle=True)
    except Exception as exc:
        raise ValueError(f"npz_load_error: {exc}") from exc
    try:
        required = ("poses", "betas", "trans", "gender")
        missing = [name for name in required if name not in motion]
        if missing:
            raise ValueError(f"missing_fields:{','.join(missing)}")
        poses = motion["poses"]
        if not isinstance(poses, np.ndarray) or poses.ndim != 2:
            raise ValueError(f"invalid_poses_shape:{getattr(poses, 'shape', None)}")
        if poses.shape[0] <= 0 or poses.shape[1] < 66:
            raise ValueError(f"invalid_poses_shape:{poses.shape}")
        if not np.issubdtype(poses.dtype, np.number):
            raise ValueError(f"nonnumeric_poses:{poses.dtype}")
        if not np.isfinite(poses).all():
            raise ValueError("nonfinite_poses")

        trans = np.asarray(motion["trans"])
        if trans.ndim != 2 or trans.shape != (poses.shape[0], 3):
            raise ValueError(
                f"invalid_trans_shape:{trans.shape};expected=({poses.shape[0]},3)"
            )
        if not np.issubdtype(trans.dtype, np.number):
            raise ValueError(f"nonnumeric_trans:{trans.dtype}")
        if not np.isfinite(trans).all():
            raise ValueError("nonfinite_trans")

        betas = np.asarray(motion["betas"])
        if betas.ndim != 1 or betas.shape[0] < 10:
            raise ValueError(f"unsupported_betas_shape:{betas.shape}")
        if not np.issubdtype(betas.dtype, np.number):
            raise ValueError(f"nonnumeric_betas:{betas.dtype}")
        if not np.isfinite(betas).all():
            raise ValueError("nonfinite_betas")

        raw_gender = motion["gender"]
        gender = normalize_gender(raw_gender)
        loader_gender = str(raw_gender).strip().lower()
        if loader_gender != gender:
            raise ValueError(
                "unsupported_gender_loader_repr:"
                f"shape={np.asarray(raw_gender).shape},str={str(raw_gender)!r}"
            )

        fps: float | None = None
        fps_warning = ""
        if "mocap_frame_rate" in motion:
            fps_array = np.asarray(motion["mocap_frame_rate"])
            if fps_array.size != 1 or not np.isfinite(fps_array).all():
                raise ValueError("invalid_mocap_frame_rate")
            fps = float(fps_array.reshape(-1)[0])
            if abs(fps - 30.0) > 0.5:
                raise ValueError(f"mocap_frame_rate_not_30:{fps}")
        else:
            fps_warning = "missing_mocap_frame_rate"
        return {
            "length": int(poses.shape[0]),
            "pose_dim": int(poses.shape[1]),
            "betas_shape": list(betas.shape),
            "gender": gender,
            "mocap_frame_rate": fps,
            "fps_warning": fps_warning,
            "has_expressions": "expressions" in motion,
            "model": str(motion["model"]) if "model" in motion else "",
        }
    finally:
        if hasattr(motion, "close"):
            motion.close()


def read_audio_info(path: str | Path) -> dict[str, Any]:
    """Read WAV metadata without loading the full waveform."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError("missing_or_empty_wav")
    if is_lfs_pointer(path):
        raise ValueError("lfs_pointer_not_downloaded")
    try:
        import soundfile as sf

        info = sf.info(path)
        samplerate = int(info.samplerate)
        frames = int(info.frames)
        channels = int(info.channels)
    except ImportError:
        try:
            with wave.open(str(path), "rb") as file:
                samplerate = int(file.getframerate())
                frames = int(file.getnframes())
                channels = int(file.getnchannels())
        except Exception as exc:
            raise ValueError(f"wav_metadata_error: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"wav_metadata_error: {exc}") from exc
    if samplerate <= 0 or frames <= 0 or channels <= 0:
        raise ValueError(
            f"invalid_wav_metadata:sr={samplerate},frames={frames},channels={channels}"
        )
    return {
        "samplerate": samplerate,
        "frames": frames,
        "channels": channels,
        "duration_sec": frames / samplerate,
    }


def discover_subsets(root: str | Path, requested: list[str] | None = None) -> list[Path]:
    """Resolve explicit subsets or discover sorted beat_*_v2.0.0 directories."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"BEAT2 root does not exist: {root}")
    if requested:
        subsets = [root / name for name in requested]
    else:
        subsets = sorted(
            path.parent
            for path in root.glob("beat_*_v2.0.0/train_test_split.csv")
            if path.is_file()
        )
    missing = [path for path in subsets if not (path / "train_test_split.csv").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing subset CSV: {missing}")
    if not subsets:
        raise BEAT2BuildError(f"No beat_*_v2.0.0 subsets found under {root}")
    return sorted(subsets, key=lambda path: path.name)


def make_minitrain(train: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    """Select the first deterministic records from the sorted final train split."""
    if size < 0:
        raise ValueError("minitrain size cannot be negative")
    return [dict(item) for item in train[:size]]


def _item_key(item: dict[str, Any]) -> tuple[str, str]:
    return item["subset"], item["video_id"]


def validate_split_artifact(
    value: Any,
    root: str | Path,
    *,
    include_additional_as_train: bool,
) -> None:
    """Fully validate the serialized split structure and referenced files."""
    root = Path(root)
    if not isinstance(value, dict) or set(value) != set(SPLITS):
        raise ValueError(f"all_splits must contain exactly {SPLITS}")
    keys_by_split: dict[str, set[tuple[str, str]]] = {}
    for split in SPLITS:
        items = value[split]
        if not isinstance(items, list):
            raise ValueError(f"split {split} must be a list")
        if items != sorted(items, key=_item_key):
            raise ValueError(f"split {split} is not deterministically sorted")
        keys: set[tuple[str, str]] = set()
        for item in items:
            if not isinstance(item, dict) or set(item) != {"video_id", "subset", "length"}:
                raise ValueError(f"invalid {split} item: {item!r}")
            if not isinstance(item["video_id"], str) or not item["video_id"]:
                raise ValueError(f"invalid video_id in {split}")
            if not isinstance(item["subset"], str) or not item["subset"]:
                raise ValueError(f"invalid subset in {split}")
            if type(item["length"]) is not int or item["length"] <= 0:
                raise ValueError(f"invalid length in {split}: {item['length']!r}")
            key = _item_key(item)
            if key in keys:
                raise ValueError(f"duplicate item in {split}: {key}")
            keys.add(key)
            npz = root / item["subset"] / "smplxflame_30" / f"{item['video_id']}.npz"
            wav = root / item["subset"] / "wave16k" / f"{item['video_id']}.wav"
            if not npz.is_file() or npz.stat().st_size <= 0:
                raise FileNotFoundError(f"referenced NPZ is missing: {npz}")
            if not wav.is_file() or wav.stat().st_size <= 0:
                raise FileNotFoundError(f"referenced WAV is missing: {wav}")
            if is_lfs_pointer(npz) or is_lfs_pointer(wav):
                raise ValueError(f"referenced source is a Git LFS pointer: {key}")
        keys_by_split[split] = keys

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = keys_by_split[left] & keys_by_split[right]
        if overlap:
            raise ValueError(f"{left}/{right} overlap: {sorted(overlap)[:10]}")
    for split in ("val", "test"):
        overlap = keys_by_split[split] & keys_by_split["additional"]
        if overlap:
            raise ValueError(f"{split}/additional overlap: {sorted(overlap)[:10]}")
    train_additional = keys_by_split["train"] & keys_by_split["additional"]
    if include_additional_as_train:
        if train_additional != keys_by_split["additional"]:
            raise ValueError("merged train must contain every additional item")
    elif train_additional:
        raise ValueError("train/additional overlap without merge option")
    expected_minitrain = [_item_key(item) for item in value["train"]][
        : len(value["minitrain"])
    ]
    actual_minitrain = [_item_key(item) for item in value["minitrain"]]
    if actual_minitrain != expected_minitrain:
        raise ValueError("minitrain must be the deterministic sorted train prefix")


def atomic_save_splits(
    value: dict[str, list[dict[str, Any]]],
    output: str | Path,
    root: str | Path,
    *,
    overwrite: bool,
    include_additional_as_train: bool,
) -> int:
    """Save, reload, validate, and atomically publish all_splits.pth."""
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        torch.save(value, temporary)
        reloaded = safe_torch_load(temporary)
        validate_split_artifact(
            reloaded,
            root,
            include_additional_as_train=include_additional_as_train,
        )
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return output.stat().st_size


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        if not fieldnames:
            return
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_reports(report_dir: str | Path, reports: Reports) -> None:
    """Write all requested JSON/CSV audit artifacts outside the PTH."""
    report_dir = Path(report_dir)
    _write_json(report_dir / "build_summary.json", reports.summary)
    _write_json(report_dir / "subset_summary.json", reports.subset_summary)
    for filename, rows in (
        ("valid_records.csv", reports.valid),
        ("skipped_records.csv", reports.skipped),
        ("missing_npz.csv", reports.missing_npz),
        ("missing_wav.csv", reports.missing_wav),
        ("orphan_npz.csv", reports.orphan_npz),
        ("orphan_wav.csv", reports.orphan_wav),
        ("invalid_npz.csv", reports.invalid_npz),
        ("short_audio.csv", reports.short_audio),
        ("too_short.csv", reports.too_short),
        ("unknown_split_rows.csv", reports.unknown_split),
        ("duplicate_rows.csv", reports.duplicates),
    ):
        _write_csv(report_dir / filename, rows)


def _deduplicate_report(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    seen: set[tuple[Any, ...]] = set()
    unique = []
    for row in rows:
        key = tuple(row.get(field) for field in fields)
        if key not in seen:
            seen.add(key)
            unique.append(row)
    rows[:] = unique


def build_splits(args: argparse.Namespace, reports: Reports) -> dict[str, list[dict[str, Any]]]:
    """Scan all selected subsets and build only fully validated split items."""
    subsets = discover_subsets(args.root, args.subsets)
    print("Discovered subsets:", ", ".join(path.name for path in subsets))
    split_lists: dict[str, list[dict[str, Any]]] = {name: [] for name in SOURCE_SPLITS}
    csv_split_counts: Counter[str] = Counter()
    fatal_missing: list[str] = []
    total_csv_rows = 0
    total_processed_rows = 0
    source_npz_count = 0
    source_wav_count = 0
    lfs_paths: set[str] = set()

    for subset_path in subsets:
        subset = subset_path.name
        npz_dir = subset_path / "smplxflame_30"
        wav_dir = subset_path / "wave16k"
        npz_files = {path.stem: path for path in sorted(npz_dir.glob("*.npz"))}
        wav_files = {path.stem: path for path in sorted(wav_dir.glob("*.wav"))}
        source_npz_count += len(npz_files)
        source_wav_count += len(wav_files)
        parsed = parse_split_csv(
            subset_path / "train_test_split.csv",
            subset=subset,
            reports=reports,
            limit=args.limit,
            strict=args.strict,
        )
        total_csv_rows += parsed.full_row_count
        total_processed_rows += parsed.processed_row_count
        csv_split_counts.update(parsed.split_counts)
        if parsed.duplicate_ids:
            raise BEAT2BuildError(
                f"Duplicate/conflicting CSV IDs in {subset}: {sorted(parsed.duplicate_ids)[:10]}"
            )

        missing_npz_ids = sorted(parsed.all_ids - set(npz_files))
        missing_wav_ids = sorted(parsed.all_ids - set(wav_files))
        orphan_npz_ids = sorted(set(npz_files) - parsed.all_ids)
        orphan_wav_ids = sorted(set(wav_files) - parsed.all_ids)
        reports.missing_npz.extend(
            {
                "subset": subset,
                "video_id": video_id,
                "split": parsed.all_types.get(video_id, ""),
                "path": str(npz_dir / f"{video_id}.npz"),
            }
            for video_id in missing_npz_ids
        )
        reports.missing_wav.extend(
            {
                "subset": subset,
                "video_id": video_id,
                "split": parsed.all_types.get(video_id, ""),
                "path": str(wav_dir / f"{video_id}.wav"),
            }
            for video_id in missing_wav_ids
        )
        reports.orphan_npz.extend(
            {"subset": subset, "video_id": video_id, "path": str(npz_files[video_id])}
            for video_id in orphan_npz_ids
        )
        reports.orphan_wav.extend(
            {"subset": subset, "video_id": video_id, "path": str(wav_files[video_id])}
            for video_id in orphan_wav_ids
        )

        subset_before = {name: len(split_lists[name]) for name in SOURCE_SPLITS}
        subset_valid = 0
        subset_short_audio = 0
        subset_too_short = 0
        subset_invalid_npz = 0
        subset_valid_frames = 0
        subset_audio_rate_mismatch = 0
        subset_missing_mocap_fps = 0
        for record_index, entry in enumerate(parsed.entries, start=1):
            if record_index == 1 or record_index % 100 == 0:
                print(
                    f"[{subset}] validating {record_index}/{len(parsed.entries)} "
                    f"(NPZ={len(npz_files)}, WAV={len(wav_files)})"
                )
            npz_path = npz_files.get(entry.video_id)
            wav_path = wav_files.get(entry.video_id)
            if npz_path is None or wav_path is None:
                missing_kind = []
                if npz_path is None:
                    missing_kind.append("npz")
                if wav_path is None:
                    missing_kind.append("wav")
                reason = "missing_" + "_and_".join(missing_kind)
                reports.skipped.append(
                    {
                        "subset": subset,
                        "video_id": entry.video_id,
                        "split": entry.split,
                        "reason": reason,
                    }
                )
                if entry.split in FORMAL_SPLITS and not args.allow_missing_pairs:
                    fatal_missing.append(f"{subset}/{entry.video_id}:{reason}")
                continue

            if is_lfs_pointer(npz_path) or is_lfs_pointer(wav_path):
                pointer_paths = [
                    str(path)
                    for path in (npz_path, wav_path)
                    if is_lfs_pointer(path)
                ]
                lfs_paths.update(pointer_paths)
                reports.skipped.append(
                    {
                        "subset": subset,
                        "video_id": entry.video_id,
                        "split": entry.split,
                        "reason": "lfs_pointer_not_downloaded",
                        "paths": ";".join(pointer_paths),
                    }
                )
                if entry.split in FORMAL_SPLITS and not args.allow_missing_pairs:
                    fatal_missing.append(
                        f"{subset}/{entry.video_id}:lfs_pointer_not_downloaded"
                    )
                continue

            try:
                motion_info = validate_npz(npz_path)
            except Exception as exc:
                subset_invalid_npz += 1
                reason = str(exc)
                reports.invalid_npz.append(
                    {
                        "subset": subset,
                        "video_id": entry.video_id,
                        "split": entry.split,
                        "path": str(npz_path),
                        "reason": reason,
                    }
                )
                reports.skipped.append(
                    {
                        "subset": subset,
                        "video_id": entry.video_id,
                        "split": entry.split,
                        "reason": f"invalid_npz:{reason}",
                    }
                )
                if "lfs_pointer_not_downloaded" in reason:
                    lfs_paths.add(str(npz_path))
                if args.strict:
                    raise BEAT2BuildError(
                        f"Invalid NPZ {npz_path}: {reason}"
                    ) from exc
                continue

            length = motion_info["length"]
            if length < args.min_frames:
                subset_too_short += 1
                detail = {
                    "subset": subset,
                    "video_id": entry.video_id,
                    "split": entry.split,
                    "length": length,
                    "min_frames": args.min_frames,
                }
                reports.too_short.append(detail)
                reports.skipped.append({**detail, "reason": "too_short"})
                continue

            try:
                audio_info = read_audio_info(wav_path)
            except Exception as exc:
                reason = str(exc)
                reports.skipped.append(
                    {
                        "subset": subset,
                        "video_id": entry.video_id,
                        "split": entry.split,
                        "reason": f"invalid_wav:{reason}",
                    }
                )
                if "lfs_pointer_not_downloaded" in reason:
                    lfs_paths.add(str(wav_path))
                if entry.split in FORMAL_SPLITS and not args.allow_missing_pairs:
                    fatal_missing.append(f"{subset}/{entry.video_id}:invalid_wav:{reason}")
                if args.strict:
                    raise BEAT2BuildError(f"Invalid WAV {wav_path}: {reason}") from exc
                continue

            motion_duration = length / 30.0
            audio_duration = audio_info["duration_sec"]
            is_short_audio = (
                audio_duration + args.audio_short_tolerance_sec < motion_duration
            )
            if is_short_audio:
                subset_short_audio += 1
                short_detail = {
                    "subset": subset,
                    "video_id": entry.video_id,
                    "split": entry.split,
                    "motion_frames": length,
                    "motion_duration_sec": motion_duration,
                    "audio_duration_sec": audio_duration,
                    "short_by_sec": motion_duration - audio_duration,
                    "tolerance_sec": args.audio_short_tolerance_sec,
                }
                reports.short_audio.append(short_detail)
                if not args.allow_short_audio:
                    reports.skipped.append({**short_detail, "reason": "short_audio"})
                    if args.strict:
                        raise BEAT2BuildError(
                            f"Audio is shorter than motion: {wav_path}"
                        )
                    continue

            item = {"video_id": entry.video_id, "subset": subset, "length": length}
            split_lists[entry.split].append(item)
            subset_valid += 1
            subset_valid_frames += length
            subset_audio_rate_mismatch += int(
                audio_info["samplerate"] != args.audio_sample_rate
            )
            subset_missing_mocap_fps += int(
                motion_info["mocap_frame_rate"] is None
            )
            reports.valid.append(
                {
                    **item,
                    "split": entry.split,
                    "npz_path": str(npz_path),
                    "wav_path": str(wav_path),
                    "motion_duration_sec": motion_duration,
                    "audio_duration_sec": audio_duration,
                    "audio_samplerate": audio_info["samplerate"],
                    "audio_channels": audio_info["channels"],
                    "audio_expected_samplerate": args.audio_sample_rate,
                    "audio_samplerate_matches": audio_info["samplerate"]
                    == args.audio_sample_rate,
                    "gender": motion_info["gender"],
                    "mocap_frame_rate": motion_info["mocap_frame_rate"],
                    "fps_warning": motion_info["fps_warning"],
                    "short_audio_allowed": is_short_audio,
                }
            )

        for path in [*npz_files.values(), *wav_files.values()]:
            if is_lfs_pointer(path):
                lfs_paths.add(str(path))
        subset_after = {name: len(split_lists[name]) for name in SOURCE_SPLITS}
        reports.subset_summary[subset] = {
            "csv_row_count": parsed.full_row_count,
            "processed_csv_row_count": parsed.processed_row_count,
            "csv_split_counts": dict(sorted(parsed.split_counts.items())),
            "source_npz_count": len(npz_files),
            "source_wav_count": len(wav_files),
            "valid_record_count": subset_valid,
            "train_count": subset_after["train"] - subset_before["train"],
            "val_count": subset_after["val"] - subset_before["val"],
            "test_count": subset_after["test"] - subset_before["test"],
            "additional_count": subset_after["additional"] - subset_before["additional"],
            "missing_npz_count": len(missing_npz_ids),
            "missing_wav_count": len(missing_wav_ids),
            "orphan_npz_count": len(orphan_npz_ids),
            "orphan_wav_count": len(orphan_wav_ids),
            "invalid_npz_count": subset_invalid_npz,
            "short_audio_count": subset_short_audio,
            "too_short_count": subset_too_short,
            "lfs_pointer_count": sum(
                str(subset_path) in path for path in lfs_paths
            ),
            "audio_sample_rate_mismatch_count": subset_audio_rate_mismatch,
            "missing_mocap_frame_rate_count": subset_missing_mocap_fps,
            "total_motion_frames": subset_valid_frames,
            "total_motion_hours": subset_valid_frames / 30.0 / 3600.0,
            "unknown_split_count": sum(
                row["subset"] == subset for row in reports.unknown_split
            ),
            "duplicate_count": sum(row["subset"] == subset for row in reports.duplicates),
        }
        print(
            f"[{subset}] CSV={parsed.full_row_count}, NPZ={len(npz_files)}, "
            f"WAV={len(wav_files)}, valid={subset_valid}"
        )

    _deduplicate_report(reports.missing_npz, ("subset", "video_id"))
    _deduplicate_report(reports.missing_wav, ("subset", "video_id"))
    for name in SOURCE_SPLITS:
        split_lists[name].sort(key=_item_key)
    if args.include_additional_as_train:
        split_lists["train"] = sorted(
            [*split_lists["train"], *split_lists["additional"]], key=_item_key
        )
    minitrain = make_minitrain(split_lists["train"], args.minitrain_size)
    if len(minitrain) < args.minitrain_size:
        print(
            f"WARNING: only {len(minitrain)} train records available for "
            f"--minitrain-size={args.minitrain_size}"
        )
    output = {
        "train": split_lists["train"],
        "val": split_lists["val"],
        "test": split_lists["test"],
        "minitrain": minitrain,
        "additional": split_lists["additional"],
    }

    # Count every validated source record once even when additional is also merged into train.
    total_frames = sum(int(row["length"]) for row in reports.valid)
    reports.summary.update(
        {
            "status": "scanned",
            "root": str(Path(args.root).resolve()),
            "output": str(Path(args.output).resolve()),
            "discovered_subsets": [path.name for path in subsets],
            "csv_row_count": total_csv_rows,
            "processed_csv_row_count": total_processed_rows,
            "csv_split_counts": dict(sorted(csv_split_counts.items())),
            "source_npz_count": source_npz_count,
            "source_wav_count": source_wav_count,
            "valid_record_count": len(reports.valid),
            "train_count": len(output["train"]),
            "val_count": len(output["val"]),
            "test_count": len(output["test"]),
            "additional_count": len(output["additional"]),
            "minitrain_count": len(output["minitrain"]),
            "missing_npz_count": len(reports.missing_npz),
            "missing_wav_count": len(reports.missing_wav),
            "orphan_npz_count": len(reports.orphan_npz),
            "orphan_wav_count": len(reports.orphan_wav),
            "invalid_npz_count": len(reports.invalid_npz),
            "lfs_pointer_count": len(lfs_paths),
            "lfs_pointer_paths": sorted(lfs_paths),
            "short_audio_count": len(reports.short_audio),
            "too_short_count": len(reports.too_short),
            "unknown_split_count": len(reports.unknown_split),
            "duplicate_count": len(reports.duplicates),
            "total_motion_frames": total_frames,
            "total_motion_hours": total_frames / 30.0 / 3600.0,
            "include_additional_as_train": args.include_additional_as_train,
            "additional_merged_into_train": args.include_additional_as_train,
            "allow_missing_pairs": args.allow_missing_pairs,
            "allow_short_audio": args.allow_short_audio,
            "output_size_bytes": 0,
        }
    )
    if fatal_missing:
        raise BEAT2BuildError(
            f"Found {len(fatal_missing)} formal train/val/test records with missing or "
            "invalid NPZ/WAV pairs. Re-download the files or explicitly use "
            f"--allow-missing-pairs to skip them. Examples: {fatal_missing[:10]}"
        )
    return output


def build_parser() -> argparse.ArgumentParser:
    """Create the BEAT2 index builder CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report-dir", type=Path, default=Path("outputs/beat2_build_report"))
    parser.add_argument("--subsets", nargs="*")
    parser.add_argument("--min-frames", type=int, default=120)
    parser.add_argument("--minitrain-size", type=int, default=16)
    parser.add_argument("--audio-sample-rate", type=int, default=16000)
    parser.add_argument("--audio-short-tolerance-sec", type=float, default=0.05)
    parser.add_argument("--include-additional-as-train", action="store_true")
    parser.add_argument("--allow-missing-pairs", action="store_true")
    parser.add_argument("--allow-short-audio", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.output is None:
        args.output = args.root / "all_splits.pth"
    if args.min_frames <= 0:
        raise ValueError("--min-frames must be positive")
    if args.minitrain_size < 0:
        raise ValueError("--minitrain-size cannot be negative")
    if args.audio_sample_rate <= 0:
        raise ValueError("--audio-sample-rate must be positive")
    if args.audio_short_tolerance_sec < 0:
        raise ValueError("--audio-short-tolerance-sec cannot be negative")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if not args.dry_run and args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")


def main(argv: list[str] | None = None) -> int:
    """Run a full BEAT2 scan, report it, and optionally publish all_splits.pth."""
    started = time.monotonic()
    args = build_parser().parse_args(argv)
    _validate_args(args)
    reports = Reports()
    try:
        splits = build_splits(args, reports)
        validate_split_artifact(
            splits,
            args.root,
            include_additional_as_train=args.include_additional_as_train,
        )
        if args.dry_run:
            reports.summary["status"] = "dry_run_complete"
        else:
            reports.summary["output_size_bytes"] = atomic_save_splits(
                splits,
                args.output,
                args.root,
                overwrite=args.overwrite,
                include_additional_as_train=args.include_additional_as_train,
            )
            reports.summary["status"] = "complete"
        reports.summary["elapsed_seconds"] = time.monotonic() - started
        write_reports(args.report_dir, reports)
    except Exception as exc:
        reports.summary.setdefault("root", str(Path(args.root).resolve()))
        reports.summary.setdefault("output", str(Path(args.output).resolve()))
        reports.summary["status"] = "failed"
        reports.summary["error"] = f"{type(exc).__name__}: {exc}"
        reports.summary["elapsed_seconds"] = time.monotonic() - started
        write_reports(args.report_dir, reports)
        raise

    print("=" * 72)
    print("BEAT2 all_splits 构建完成" if not args.dry_run else "BEAT2 dry-run 审计完成")
    for name in ("train", "val", "test", "additional", "minitrain"):
        print(f"  {name:10s}: {reports.summary[f'{name}_count']}")
    print(f"  valid:      {reports.summary['valid_record_count']}")
    print(f"  frames:     {reports.summary['total_motion_frames']}")
    print(f"  hours:      {reports.summary['total_motion_hours']:.4f}")
    print(f"  output:     {args.output if not args.dry_run else '(dry-run; not saved)'}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
