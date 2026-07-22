#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Build the official AIST++ crossmodal 30 FPS GENMO-SMPL artifacts.

Unlike the partial engineering builder, this command uses only the ordered
official crossmodal train/validation/test lists.  Every listed sequence must
have motion, keypoints, camera calibration and aligned EDGE baseline35 music
features before any standard artifact is published.
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gem.utils.smplx_utils import make_smplx  # noqa: E402
from tools.data.aistpp.build_annot_aist_30fps import (  # noqa: E402
    _load_camera_environment,
    _load_keypoints,
    _load_motion,
    _save_csv,
    _save_json,
    camera_space_smpl,
    compute_tight_bboxes,
    downsample_motion_indices,
    parse_camera_mapping,
    safe_torch_load,
    select_keypoint_frames,
    validate_annot_record,
    validate_music_features,
)

DEFAULT_ANNOTATIONS_ROOT = Path(
    "/home/weili/datasets/AISTPP_fullset/aist_plusplus_final"
)
DEFAULT_MUSICFEAT_DIR = Path("/home/weili/GENMO/inputs/AIST++/musicfeat_v2")
DEFAULT_OUTPUT_ROOT = Path("/home/weili/GENMO/inputs/AIST++")
DEFAULT_REPORT_DIR = Path("outputs/aistpp_official_build_report")
DEFAULT_FILENAMES = {
    "annot": "annot_aist_30fps.pt",
    "train": "train.pt",
    "val": "val.pt",
    "test": "test.pt",
    "minitrain": "minitrain.pt",
}
SPLIT_FILENAMES = {
    "train": "crossmodal_train.txt",
    "val": "crossmodal_val.txt",
    "test": "crossmodal_test.txt",
}


class AISTOfficialBuildError(RuntimeError):
    """Raised when official AIST++ inputs violate the publishing contract."""


@dataclass
class OfficialBuildReports:
    """Report payloads accumulated while auditing and building official data."""

    manifest: list[dict[str, Any]] = field(default_factory=list)
    bbox_fill: list[dict[str, Any]] = field(default_factory=list)
    invalid_records: list[dict[str, Any]] = field(default_factory=list)
    camera_environments: Counter[str] = field(default_factory=Counter)
    camera_dimensions: Counter[str] = field(default_factory=Counter)
    scalings: list[dict[str, Any]] = field(default_factory=list)
    missing_required: dict[str, Any] = field(default_factory=dict)
    extra_data: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    split_summary: dict[str, Any] = field(default_factory=dict)
    validation_summary: dict[str, Any] = field(default_factory=dict)


def read_ordered_unique_ids(path: str | Path) -> list[str]:
    """Read non-empty IDs from a split file while preserving source order."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"required split file does not exist: {path}")
    result: list[str] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        sequence = line.strip()
        if not sequence:
            continue
        if sequence in seen:
            raise ValueError(
                f"duplicate sequence ID {sequence!r} in {path} at line {line_number}"
            )
        seen.add(sequence)
        result.append(sequence)
    if not result:
        raise ValueError(f"split file contains no sequence IDs: {path}")
    return result


def validate_official_splits(
    train: list[str],
    val: list[str],
    test: list[str],
    *,
    expected_train: int = 980,
    expected_val: int = 20,
    expected_test: int = 20,
) -> set[str]:
    """Validate exact split counts, uniqueness and pairwise disjointness."""

    expected = {"train": expected_train, "val": expected_val, "test": expected_test}
    splits = {"train": train, "val": val, "test": test}
    for name, values in splits.items():
        if not isinstance(values, list) or any(not isinstance(item, str) or not item for item in values):
            raise ValueError(f"{name} split must be a list of non-empty strings")
        if len(values) != len(set(values)):
            raise ValueError(f"{name} split contains duplicate IDs")
        if len(values) != expected[name]:
            raise ValueError(
                f"official {name} split count is {len(values)}, expected {expected[name]}"
            )
    overlap_tv = set(train) & set(val)
    overlap_tt = set(train) & set(test)
    overlap_vt = set(val) & set(test)
    if overlap_tv or overlap_tt or overlap_vt:
        raise ValueError(
            "official crossmodal splits overlap: "
            f"train/val={sorted(overlap_tv)[:10]}, "
            f"train/test={sorted(overlap_tt)[:10]}, "
            f"val/test={sorted(overlap_vt)[:10]}"
        )
    return set(train) | set(val) | set(test)


def choose_ordered_minitrain(
    train: list[str],
    annot: dict[str, dict[str, Any]],
    size: int,
    min_frames: int,
) -> list[str]:
    """Select sufficiently long records in official train-file order."""

    if size <= 0 or min_frames <= 0:
        raise ValueError("minitrain size and minimum frames must be positive")
    result = [
        sequence
        for sequence in train
        if sequence in annot
        and int(annot[sequence]["smpl_pose_global"].shape[0]) >= min_frames
    ][:size]
    if len(result) != size:
        raise ValueError(
            f"only {len(result)} official train records have at least {min_frames} frames; "
            f"minitrain requires {size}"
        )
    return result


def output_paths(args: argparse.Namespace) -> dict[str, Path]:
    """Resolve the five standard official artifacts under ``output_root``."""

    return {
        "annot": args.output_root / args.annot_filename,
        "train": args.output_root / args.train_split_filename,
        "val": args.output_root / args.val_split_filename,
        "test": args.output_root / args.test_split_filename,
        "minitrain": args.output_root / args.minitrain_split_filename,
    }


def _music_ids(directory: Path) -> set[str]:
    suffix = "_musicfeat_fps30.pt"
    return {path.name.removesuffix(suffix) for path in directory.glob(f"*{suffix}")}


def _read_ignore_ids(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"required ignore list does not exist: {path}")
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def preflight_official_inputs(
    args: argparse.Namespace,
    official_ids: set[str],
    reports: OfficialBuildReports,
) -> tuple[set[str], set[str], set[str], dict[str, str]]:
    """Audit all mandatory sources and fail without silently dropping official IDs."""

    motions_root = args.annotations_root / "motions"
    keypoints_root = args.annotations_root / "keypoints2d"
    cameras_root = args.annotations_root / "cameras"
    motion_ids = {path.stem for path in motions_root.glob("*.pkl")}
    keypoint_ids = {path.stem for path in keypoints_root.glob("*.pkl")}
    music_ids = _music_ids(args.musicfeat_dir)
    camera_mapping = parse_camera_mapping(cameras_root / "mapping.txt")
    camera_environments = {path.stem for path in cameras_root.glob("*.json")}
    ignore_ids = _read_ignore_ids(args.annotations_root / "ignore_list.txt")

    missing_motion = sorted(official_ids - motion_ids)
    missing_keypoints = sorted(official_ids - keypoint_ids)
    missing_music = sorted(official_ids - music_ids)
    missing_mapping = sorted(official_ids - set(camera_mapping))
    missing_environment = sorted(
        sequence
        for sequence in official_ids & set(camera_mapping)
        if camera_mapping[sequence] not in camera_environments
    )
    missing_camera = sorted(set(missing_mapping) | set(missing_environment))
    ignored_official = sorted(official_ids & ignore_ids)
    reports.missing_required = {
        "missing_motion": missing_motion,
        "missing_keypoints2d": missing_keypoints,
        "missing_musicfeat": missing_music,
        "missing_camera_mapping": missing_mapping,
        "missing_camera_environment_json": missing_environment,
        "ignored_official_ids": ignored_official,
        "missing_required_count": len(
            set(missing_motion)
            | set(missing_keypoints)
            | set(missing_music)
            | set(missing_camera)
        ),
    }
    reports.extra_data = {
        "note": (
            "extra music features are ignored and are not part of the official "
            "crossmodal train/val/test split."
        ),
        "extra_motion_ids": sorted(motion_ids - official_ids),
        "extra_keypoints2d_ids": sorted(keypoint_ids - official_ids),
        "extra_musicfeat_ids": sorted(music_ids - official_ids),
    }
    reports.summary.update(
        {
            "source_motion_count": len(motion_ids),
            "source_keypoints2d_count": len(keypoint_ids),
            "source_musicfeat_count": len(music_ids),
            "extra_motion_count": len(motion_ids - official_ids),
            "extra_keypoints2d_count": len(keypoint_ids - official_ids),
            "extra_musicfeat_count": len(music_ids - official_ids),
            "ignored_official_count": len(ignored_official),
            "missing_motion_count": len(missing_motion),
            "missing_keypoints_count": len(missing_keypoints),
            "missing_musicfeat_count": len(missing_music),
            "missing_camera_count": len(missing_camera),
        }
    )

    problems: list[str] = []
    for label, values in (
        ("motion", missing_motion),
        ("keypoints2d", missing_keypoints),
        ("musicfeat", missing_music),
        ("camera", missing_camera),
    ):
        if values:
            problems.append(f"missing {label} for {len(values)} official IDs")
    if ignored_official:
        problems.append(
            f"{len(ignored_official)} official IDs also occur in ignore_list.txt"
        )
    if problems:
        raise AISTOfficialBuildError(
            "Official AIST++ preflight failed: " + "; ".join(problems)
        )
    return motion_ids, keypoint_ids, music_ids, camera_mapping


def validate_official_outputs(
    annot: Any,
    splits: dict[str, Any],
    expected_splits: dict[str, list[str]],
    music_lengths: dict[str, int] | None = None,
) -> tuple[int, int]:
    """Validate official artifacts without sorting away source split order."""

    if not isinstance(annot, dict):
        raise ValueError("annot output must be a dict")
    official_order = (
        expected_splits["train"] + expected_splits["val"] + expected_splits["test"]
    )
    if list(annot) != official_order:
        raise ValueError("annot insertion order must be train, then val, then test source order")
    if set(annot) != set(official_order):
        raise ValueError("annot keys do not exactly equal the official split union")
    total_frames = 0
    for sequence, record in annot.items():
        length = validate_annot_record(record, sequence)
        if music_lengths is not None and music_lengths.get(sequence) != length:
            raise ValueError(f"{sequence}: annot length disagrees with music feature length")
        total_frames += length

    for name in ("train", "val", "test"):
        split = splits.get(name)
        if split != expected_splits[name]:
            raise ValueError(f"{name} output does not preserve the official split file order")
        if len(split) != len(set(split)):
            raise ValueError(f"{name} output contains duplicate IDs")
    train_set, val_set, test_set = map(
        set, (splits["train"], splits["val"], splits["test"])
    )
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise ValueError("published train/val/test splits overlap")
    if train_set | val_set | test_set != set(annot):
        raise ValueError("published split union does not equal annot keys")
    minitrain = splits.get("minitrain")
    if not isinstance(minitrain, list) or not set(minitrain) <= train_set:
        raise ValueError("minitrain must be an ordered subset of train")
    expected_positions = [splits["train"].index(item) for item in minitrain]
    if expected_positions != sorted(expected_positions):
        raise ValueError("minitrain does not preserve official train order")
    return len(annot), total_frames


def atomic_save_official_outputs(
    annot: dict[str, dict[str, Any]],
    splits: dict[str, list[str]],
    expected_splits: dict[str, list[str]],
    paths: dict[str, Path],
    overwrite: bool,
) -> None:
    """Save, reload and validate all artifacts, with rollback on publish failure."""

    for path in paths.values():
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing official output: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary = {name: path.with_name(path.name + ".tmp") for name, path in paths.items()}
    for path in temporary.values():
        if path.exists():
            path.unlink()

    backups: dict[str, Path] = {}
    published: list[Path] = []
    token = uuid.uuid4().hex
    try:
        torch.save(annot, temporary["annot"])
        for name in ("train", "val", "test", "minitrain"):
            torch.save(splits[name], temporary[name])
        reloaded_annot = safe_torch_load(temporary["annot"])
        reloaded_splits = {
            name: safe_torch_load(temporary[name])
            for name in ("train", "val", "test", "minitrain")
        }
        validate_official_outputs(reloaded_annot, reloaded_splits, expected_splits)

        for name, final_path in paths.items():
            if final_path.exists():
                backup = final_path.with_name(f".{final_path.name}.bak_{token}")
                os.replace(final_path, backup)
                backups[name] = backup
        for name, final_path in paths.items():
            os.replace(temporary[name], final_path)
            published.append(final_path)
    except Exception:
        for path in published:
            path.unlink(missing_ok=True)
        for name, backup in backups.items():
            if backup.exists():
                os.replace(backup, paths[name])
        raise
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)
        for backup in backups.values():
            backup.unlink(missing_ok=True)


def _base_summary(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "status": "starting",
        "mode": "official_crossmodal",
        "annotations_root": str(args.annotations_root.resolve()),
        "musicfeat_dir": str(args.musicfeat_dir.resolve()),
        "selected_view": args.view,
        "source_fps": args.source_fps,
        "target_fps": args.target_fps,
        "dry_run": args.dry_run,
        "output_files": {
            name: str(path.resolve()) for name, path in output_paths(args).items()
        },
        "output_size_bytes": {},
    }


def _manifest_base(sequence: str, split: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "sequence": sequence,
        "split": split,
        "status": "pending",
        "reason": "",
        "source_motion_frames": "",
        "source_keypoint_frames": "",
        "output_frames": "",
        "music_feature_frames": "",
        "selected_view": args.view,
        "environment": "",
        "invalid_bbox_frames_before_fill": "",
        "clamped_keypoint_frames": "",
        "scaling": "",
    }


def build_official_dataset(
    args: argparse.Namespace,
    reports: OfficialBuildReports,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, list[str]],
    dict[str, list[str]],
    dict[str, int],
]:
    """Build all selected official split records in source-file order."""

    splits_root = args.annotations_root / "splits"
    full_splits = {
        name: read_ordered_unique_ids(splits_root / filename)
        for name, filename in SPLIT_FILENAMES.items()
    }
    official_ids = validate_official_splits(
        full_splits["train"],
        full_splits["val"],
        full_splits["test"],
        expected_train=args.expected_train_count,
        expected_val=args.expected_val_count,
        expected_test=args.expected_test_count,
    )
    selected_splits = {
        name: values[: args.limit] if args.limit is not None else list(values)
        for name, values in full_splits.items()
    }
    selected_ids = set().union(*map(set, selected_splits.values()))
    reports.summary.update(
        {
            "official_train_count": len(full_splits["train"]),
            "official_val_count": len(full_splits["val"]),
            "official_test_count": len(full_splits["test"]),
            "official_union_count": len(official_ids),
            "selected_train_count": len(selected_splits["train"]),
            "selected_val_count": len(selected_splits["val"]),
            "selected_test_count": len(selected_splits["test"]),
            "selected_union_count": len(selected_ids),
        }
    )
    reports.split_summary = {
        "source_order_preserved": True,
        "limited_debug_build": args.limit is not None,
        "official_train": full_splits["train"],
        "official_val": full_splits["val"],
        "official_test": full_splits["test"],
        "selected_train": selected_splits["train"],
        "selected_val": selected_splits["val"],
        "selected_test": selected_splits["test"],
    }

    # Ignore-list safety is checked against the complete official union even in
    # debug-limit mode; a limit must never hide a split-level contract conflict.
    preflight_official_inputs(args, official_ids, reports)

    smpl_model = make_smplx("supermotion")
    offset = smpl_model.get_skeleton(torch.zeros(10))[0].detach().cpu().float()
    motions_root = args.annotations_root / "motions"
    keypoints_root = args.annotations_root / "keypoints2d"
    cameras_root = args.annotations_root / "cameras"
    camera_mapping = parse_camera_mapping(cameras_root / "mapping.txt")
    annot: dict[str, dict[str, Any]] = {}
    music_lengths: dict[str, int] = {}
    ordered_with_split = [
        (sequence, name)
        for name in ("train", "val", "test")
        for sequence in selected_splits[name]
    ]

    for index, (sequence, split_name) in enumerate(ordered_with_split, 1):
        row = _manifest_base(sequence, split_name, args)
        try:
            poses, trans, scaling = _load_motion(motions_root / f"{sequence}.pkl")
            indices = downsample_motion_indices(
                len(poses), args.source_fps, args.target_fps
            )
            pose_global = np.ascontiguousarray(poses[indices], dtype=np.float32)
            trans_global = np.ascontiguousarray(trans[indices], dtype=np.float32)
            row.update(
                source_motion_frames=len(poses),
                output_frames=len(indices),
                scaling=scaling,
            )

            keypoints = _load_keypoints(keypoints_root / f"{sequence}.pkl")
            selected_keypoints, clamped_count, kp_frames = select_keypoint_frames(
                keypoints,
                args.view,
                indices,
                len(poses),
                args.max_motion_kp_frame_difference,
            )
            row.update(
                source_keypoint_frames=kp_frames,
                clamped_keypoint_frames=clamped_count,
            )
            environment = camera_mapping[sequence]
            _, intrinsics, t_w2c, width, height = _load_camera_environment(
                cameras_root, environment, args.view
            )
            row["environment"] = environment
            bboxes, invalid_bbox_count = compute_tight_bboxes(
                selected_keypoints,
                width,
                height,
                args.kp_confidence_threshold,
                args.min_valid_joints,
            )
            row["invalid_bbox_frames_before_fill"] = invalid_bbox_count
            if args.strict and (clamped_count or invalid_bbox_count):
                raise ValueError(
                    "strict mode rejects non-critical repair: "
                    f"clamped_keypoints={clamped_count}, filled_bboxes={invalid_bbox_count}"
                )

            music_path = args.musicfeat_dir / f"{sequence}_musicfeat_fps30.pt"
            music = validate_music_features(music_path, len(indices))
            row["music_feature_frames"] = len(music)
            pose_camera, trans_camera = camera_space_smpl(
                pose_global, trans_global, t_w2c, offset
            )
            record = {
                "smpl_pose_global": pose_global,
                "smpl_trans_global": trans_global,
                "smpl_pose": np.ascontiguousarray(pose_camera, dtype=np.float32),
                "smpl_trans": np.ascontiguousarray(trans_camera, dtype=np.float32),
                "bbox_xyxy": np.ascontiguousarray(bboxes, dtype=np.float32),
                "intrinsics": intrinsics.detach().cpu().float().contiguous(),
                "T_w2c": t_w2c.detach().cpu().float().contiguous(),
                "height": int(height),
                "width": int(width),
            }
            length = validate_annot_record(record, sequence)
            if length != len(music):
                raise ValueError(
                    f"record/music length mismatch after construction: {length} != {len(music)}"
                )
            annot[sequence] = record
            music_lengths[sequence] = length
            row["status"] = "built"
            reports.bbox_fill.append(
                {
                    "sequence": sequence,
                    "split": split_name,
                    "frames": length,
                    "invalid_bbox_frames_before_fill": invalid_bbox_count,
                    "filled_bbox_frames": invalid_bbox_count,
                }
            )
            reports.camera_environments[environment] += 1
            reports.camera_dimensions[f"{width}x{height}"] += 1
            reports.scalings.append({"sequence": sequence, "scaling": scaling})
        except Exception as exc:
            row.update(status="invalid", reason=type(exc).__name__)
            reports.invalid_records.append(
                {"sequence": sequence, "split": split_name, "error": str(exc)}
            )
        reports.manifest.append(row)
        if index % 50 == 0 or index == len(ordered_with_split):
            print(
                f"[Build] {index}/{len(ordered_with_split)} official sequences, "
                f"built={len(annot)}, invalid={len(reports.invalid_records)}"
            )

    if reports.invalid_records:
        raise AISTOfficialBuildError(
            f"{len(reports.invalid_records)} official records failed validation; no outputs published"
        )
    minitrain = choose_ordered_minitrain(
        selected_splits["train"], annot, args.minitrain_size, args.min_sequence_frames
    )
    splits = {**selected_splits, "minitrain": minitrain}
    validate_official_outputs(annot, splits, selected_splits, music_lengths)

    total_frames = sum(validate_annot_record(record, sequence) for sequence, record in annot.items())
    reports.summary.update(
        {
            "successfully_built_count": len(annot),
            "invalid_record_count": 0,
            "train_output_count": len(splits["train"]),
            "val_output_count": len(splits["val"]),
            "test_output_count": len(splits["test"]),
            "minitrain_output_count": len(minitrain),
            "total_frames": total_frames,
            "total_hours_at_30fps": total_frames / args.target_fps / 3600.0,
            "bbox_frames_filled": sum(row["filled_bbox_frames"] for row in reports.bbox_fill),
            "bbox_sequences_with_fill": sum(
                row["filled_bbox_frames"] > 0 for row in reports.bbox_fill
            ),
            "clamped_keypoint_frame_count": sum(
                int(row["clamped_keypoint_frames"] or 0) for row in reports.manifest
            ),
        }
    )
    reports.validation_summary = {
        "status": "passed",
        "annot_key_order_matches_train_val_test": list(annot)
        == selected_splits["train"] + selected_splits["val"] + selected_splits["test"],
        "annot_keys_equal_official_union": set(annot) == selected_ids,
        "split_source_order_preserved": all(
            splits[name] == selected_splits[name] for name in ("train", "val", "test")
        ),
        "split_pairwise_disjoint": True,
        "all_record_contracts_valid": True,
        "all_music_lengths_exact": True,
        "record_count": len(annot),
        "total_frames": total_frames,
    }
    return annot, splits, selected_splits, music_lengths


def write_reports(report_dir: Path, reports: OfficialBuildReports) -> None:
    """Write every required official-build audit report."""

    report_dir.mkdir(parents=True, exist_ok=True)
    _save_json(report_dir / "build_summary.json", reports.summary)
    _save_json(report_dir / "split_summary.json", reports.split_summary)
    _save_csv(report_dir / "sequence_manifest.csv", reports.manifest)
    _save_json(report_dir / "missing_required_data.json", reports.missing_required)
    _save_json(report_dir / "extra_data_summary.json", reports.extra_data)
    _save_json(report_dir / "invalid_records.json", reports.invalid_records)
    _save_csv(report_dir / "bbox_fill_report.csv", reports.bbox_fill)
    _save_json(
        report_dir / "camera_summary.json",
        {
            "selected_view": reports.summary.get("selected_view"),
            "environment_counts": dict(sorted(reports.camera_environments.items())),
            "image_dimension_counts": dict(sorted(reports.camera_dimensions.items())),
            "camera_convention": "X_camera = R_w2c @ X_world + tvec",
        },
    )
    values = np.asarray([row["scaling"] for row in reports.scalings], dtype=np.float64)
    scaling_report: dict[str, Any] = {
        "note": "smpl_scaling is audited only; it is not applied to translation or betas",
        "count": len(values),
        "per_sequence": reports.scalings,
    }
    if len(values):
        scaling_report.update(
            min=float(values.min()),
            max=float(values.max()),
            mean=float(values.mean()),
            median=float(np.median(values)),
            std=float(values.std()),
        )
    _save_json(report_dir / "scaling_statistics.json", scaling_report)
    _save_json(report_dir / "validation_summary.json", reports.validation_summary)


def build_parser() -> argparse.ArgumentParser:
    """Create the official crossmodal builder argument parser."""

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
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--expected-train-count", type=int, default=980)
    parser.add_argument("--expected-val-count", type=int, default=20)
    parser.add_argument("--expected-test-count", type=int, default=20)
    return parser


def _uses_standard_outputs(args: argparse.Namespace) -> bool:
    return (
        args.output_root.resolve() == DEFAULT_OUTPUT_ROOT.resolve()
        and all(
            getattr(args, f"{name}_filename") == filename
            for name, filename in (
                ("annot", DEFAULT_FILENAMES["annot"]),
                ("train_split", DEFAULT_FILENAMES["train"]),
                ("val_split", DEFAULT_FILENAMES["val"]),
                ("test_split", DEFAULT_FILENAMES["test"]),
                ("minitrain_split", DEFAULT_FILENAMES["minitrain"]),
            )
        )
    )


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI values and protect standard outputs from debug-limit builds."""

    required = {
        "motions": args.annotations_root / "motions",
        "keypoints2d": args.annotations_root / "keypoints2d",
        "cameras": args.annotations_root / "cameras",
        "camera mapping": args.annotations_root / "cameras" / "mapping.txt",
        "splits": args.annotations_root / "splits",
        "ignore list": args.annotations_root / "ignore_list.txt",
        "music features": args.musicfeat_dir,
    }
    for label, path in required.items():
        exists = path.is_file() if path.suffix else path.is_dir()
        if not exists:
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
    for name in ("expected_train_count", "expected_val_count", "expected_test_count"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        if not args.dry_run and _uses_standard_outputs(args):
            raise ValueError(
                "--limit cannot publish the standard official outputs; use --dry-run "
                "or select a dedicated test --output-root"
            )
    if not args.dry_run:
        for path in output_paths(args).values():
            if path.exists() and not args.overwrite:
                raise FileExistsError(f"Refusing to overwrite existing official output: {path}")


def _print_summary(summary: dict[str, Any]) -> None:
    print("=" * 72)
    print("AIST++ 官方 crossmodal 30 FPS 构建摘要")
    print(
        "  官方 train/val/test: "
        f"{summary.get('official_train_count', 0)}/"
        f"{summary.get('official_val_count', 0)}/"
        f"{summary.get('official_test_count', 0)}"
    )
    print(f"  官方并集:             {summary.get('official_union_count', 0)}")
    print(f"  成功构建:             {summary.get('successfully_built_count', 0)}")
    print(f"  额外音乐特征（忽略）: {summary.get('extra_musicfeat_count', 0)}")
    print(f"  ignore 交集:          {summary.get('ignored_official_count', 0)}")
    print(f"  bbox 填充帧:          {summary.get('bbox_frames_filled', 0)}")
    print(f"  总帧数:               {summary.get('total_frames', 0)}")
    print(f"  总时长:               {summary.get('total_hours_at_30fps', 0.0):.3f} 小时")
    print(f"  状态:                 {summary.get('status', 'unknown')}")
    print("=" * 72)


def main(argv: list[str] | None = None) -> int:
    """Run the official AIST++ crossmodal builder."""

    args = build_parser().parse_args(argv)
    validate_args(args)
    reports = OfficialBuildReports(summary=_base_summary(args))
    try:
        annot, splits, expected_splits, _ = build_official_dataset(args, reports)
        if args.dry_run:
            reports.summary["status"] = "dry_run_complete"
        else:
            atomic_save_official_outputs(
                annot,
                splits,
                expected_splits,
                output_paths(args),
                args.overwrite,
            )
            reports.summary["status"] = "complete"
            reports.summary["output_size_bytes"] = {
                name: path.stat().st_size for name, path in output_paths(args).items()
            }
        write_reports(args.report_dir, reports)
    except Exception as exc:
        reports.summary["status"] = "failed"
        reports.summary["invalid_record_count"] = len(reports.invalid_records)
        reports.summary["error"] = f"{type(exc).__name__}: {exc}"
        reports.validation_summary.setdefault("status", "failed")
        reports.validation_summary["error"] = reports.summary["error"]
        try:
            write_reports(args.report_dir, reports)
        except Exception as report_exc:
            print(f"WARNING: failed to write official build reports: {report_exc}", file=sys.stderr)
        _print_summary(reports.summary)
        raise
    _print_summary(reports.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
