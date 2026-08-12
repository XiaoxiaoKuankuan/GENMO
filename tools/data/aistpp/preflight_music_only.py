#!/usr/bin/env python3
"""Preflight official AIST++ artifacts for music-only GEM-SMPL training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gem.datasets.aistpp.aistplusplus import (  # noqa: E402
    load_aist_artifact,
    load_music_feature_tensor,
    validate_musicfeat_v2,
)

EXPECTED_COUNTS = {"train": 980, "val": 20, "test": 20}
REQUIRED_MOTION_FIELDS = (
    "smpl_pose_global",
    "smpl_trans_global",
    "smpl_pose",
    "smpl_trans",
    "bbox_xyxy",
    "intrinsics",
    "T_w2c",
)


def _finite_array(value: Any, *, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be numeric; got dtype={array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def validate_motion_record(sequence: str, record: Any) -> int:
    """Validate the motion fields consumed by AISTPlusPlusSmplDataset."""
    if not isinstance(record, dict):
        raise ValueError(f"{sequence}: annotation record must be a dict")
    missing = [field for field in REQUIRED_MOTION_FIELDS if field not in record]
    if missing:
        raise ValueError(f"{sequence}: missing motion fields {missing}")

    pose_global = _finite_array(record["smpl_pose_global"], name="smpl_pose_global")
    if pose_global.ndim != 2 or pose_global.shape[1] != 72 or pose_global.shape[0] <= 0:
        raise ValueError(
            f"{sequence}: smpl_pose_global must be non-empty [F,72], got {pose_global.shape}"
        )
    frames = int(pose_global.shape[0])
    expected = {
        "smpl_trans_global": (frames, 3),
        "smpl_pose": (frames, 72),
        "smpl_trans": (frames, 3),
        "bbox_xyxy": (frames, 4),
        "intrinsics": (3, 3),
        "T_w2c": (4, 4),
    }
    for field, shape in expected.items():
        array = _finite_array(record[field], name=field)
        if tuple(array.shape) != shape:
            raise ValueError(
                f"{sequence}: {field} must have shape {shape}, got {tuple(array.shape)}"
            )
    intrinsics = _finite_array(record["intrinsics"], name="intrinsics")
    if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
        raise ValueError(f"{sequence}: intrinsics focal lengths must be positive")
    if not np.allclose(
        _finite_array(record["T_w2c"], name="T_w2c")[3],
        np.array([0.0, 0.0, 0.0, 1.0]),
        atol=1e-6,
    ):
        raise ValueError(f"{sequence}: T_w2c has an invalid homogeneous last row")
    return frames


def validate_music_file(path: Path) -> int:
    """Validate dtype plus the shared EDGE baseline35 feature contract."""
    raw = load_aist_artifact(path)
    if isinstance(raw, torch.Tensor):
        is_float = raw.dtype.is_floating_point
    else:
        is_float = np.issubdtype(np.asarray(raw).dtype, np.floating)
    if not is_float:
        raise ValueError(f"music feature must use a floating dtype: {path}")
    feature = load_music_feature_tensor(path)
    validate_musicfeat_v2(feature, source=path)
    return int(feature.shape[0])


def _empty_alignment_stats() -> dict[str, Any]:
    return {
        "exact_match_count": 0,
        "within_1_count": 0,
        "within_2_count": 0,
        "mismatch_gt_2_count": 0,
        "max_abs_mismatch": 0,
        "sequences": [],
    }


def _update_alignment(
    stats: dict[str, Any], sequence: str, motion_frames: int, music_frames: int
) -> None:
    difference = abs(motion_frames - music_frames)
    stats["exact_match_count"] += int(difference == 0)
    stats["within_1_count"] += int(difference <= 1)
    stats["within_2_count"] += int(difference <= 2)
    stats["mismatch_gt_2_count"] += int(difference > 2)
    stats["max_abs_mismatch"] = max(stats["max_abs_mismatch"], difference)
    stats["sequences"].append(
        {
            "sequence": sequence,
            "motion_frames": motion_frames,
            "music_frames": music_frames,
            "difference": difference,
        }
    )


def run_preflight(root: Path, *, strict: bool, allow_subset: bool) -> dict[str, Any]:
    root = root.expanduser()
    required = {
        "annot": root / "annot_aist_30fps.pt",
        "train": root / "train.pt",
        "val": root / "val.pt",
        "test": root / "test.pt",
    }
    report: dict[str, Any] = {
        "root": str(root.resolve()),
        "strict": strict,
        "allow_subset": allow_subset,
        "required_files": {
            name: {"path": str(path.resolve()), "exists": path.is_file()}
            for name, path in required.items()
        },
        "split_counts": {},
        "split_overlaps": {},
        "missing_motion": [],
        "missing_music": [],
        "invalid_music": [],
        "invalid_motion": [],
        "alignment_stats": {},
        "hours": {},
        "blocking_issues": [],
        "final_pass": False,
    }
    missing_files = [name for name, path in required.items() if not path.is_file()]
    if missing_files:
        report["blocking_issues"].append(f"missing required artifacts: {missing_files}")
        return report

    try:
        annot = load_aist_artifact(required["annot"])
    except Exception as exc:
        report["blocking_issues"].append(f"cannot load annot_aist_30fps.pt: {exc}")
        return report
    if not isinstance(annot, dict):
        report["blocking_issues"].append("annot_aist_30fps.pt must contain a dict")
        return report

    splits: dict[str, list[str]] = {}
    for name in ("train", "val", "test"):
        try:
            value = load_aist_artifact(required[name])
            if not isinstance(value, (list, tuple)) or any(
                not isinstance(item, str) or not item for item in value
            ):
                raise ValueError("split must be a list of non-empty sequence IDs")
            splits[name] = list(value)
        except Exception as exc:
            report["blocking_issues"].append(f"cannot load {name}.pt: {exc}")
            splits[name] = []
        report["split_counts"][name] = len(splits[name])
        if len(splits[name]) != len(set(splits[name])):
            report["blocking_issues"].append(f"{name} split contains duplicate IDs")
        if strict and not allow_subset and len(splits[name]) != EXPECTED_COUNTS[name]:
            report["blocking_issues"].append(
                f"official {name} count is {len(splits[name])}, expected {EXPECTED_COUNTS[name]}"
            )

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sorted(set(splits[left]) & set(splits[right]))
        report["split_overlaps"][f"{left}_{right}"] = overlap
        if overlap:
            report["blocking_issues"].append(
                f"{left}/{right} overlap contains {len(overlap)} sequences"
            )

    total_stats = _empty_alignment_stats()
    total_frames = 0
    for split_name, sequences in splits.items():
        stats = _empty_alignment_stats()
        split_frames = 0
        for sequence in sequences:
            if sequence not in annot:
                report["missing_motion"].append(
                    {"split": split_name, "sequence": sequence}
                )
                continue
            try:
                motion_frames = validate_motion_record(sequence, annot[sequence])
            except Exception as exc:
                report["invalid_motion"].append(
                    {"split": split_name, "sequence": sequence, "error": str(exc)}
                )
                continue
            music_path = root / "musicfeat_v2" / f"{sequence}_musicfeat_fps30.pt"
            if not music_path.is_file():
                report["missing_music"].append(
                    {
                        "split": split_name,
                        "sequence": sequence,
                        "path": str(music_path.resolve()),
                    }
                )
                continue
            try:
                music_frames = validate_music_file(music_path)
            except Exception as exc:
                report["invalid_music"].append(
                    {
                        "split": split_name,
                        "sequence": sequence,
                        "path": str(music_path.resolve()),
                        "error": str(exc),
                    }
                )
                continue
            _update_alignment(stats, sequence, motion_frames, music_frames)
            _update_alignment(total_stats, sequence, motion_frames, music_frames)
            split_frames += motion_frames
            total_frames += motion_frames
        report["alignment_stats"][split_name] = stats
        report["hours"][split_name] = split_frames / 30.0 / 3600.0

    report["alignment_stats"]["total"] = total_stats
    report["hours"]["total"] = total_frames / 30.0 / 3600.0
    report["total_frames"] = total_frames

    for label in ("missing_motion", "missing_music", "invalid_music", "invalid_motion"):
        if report[label]:
            report["blocking_issues"].append(f"{label}: {len(report[label])}")
    if total_stats["mismatch_gt_2_count"]:
        report["blocking_issues"].append(
            f"music-motion frame mismatch > 2: {total_stats['mismatch_gt_2_count']}"
        )
    report["final_pass"] = not report["blocking_issues"]
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("inputs/AIST++"))
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help="Allow non-980/20/20 split sizes while retaining all other checks",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/aistpp_music_only_preflight.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_preflight(args.root, strict=args.strict, allow_subset=args.allow_subset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    alignment_summary = dict(report["alignment_stats"].get("total", {}))
    alignment_summary.pop("sequences", None)
    print(json.dumps({
        "split_counts": report["split_counts"],
        "hours": report["hours"],
        "alignment": alignment_summary,
        "missing_motion": len(report["missing_motion"]),
        "missing_music": len(report["missing_music"]),
        "invalid_motion": len(report["invalid_motion"]),
        "invalid_music": len(report["invalid_music"]),
        "final_pass": report["final_pass"],
        "output": str(args.output),
    }, indent=2, ensure_ascii=False))
    if args.strict and not report["final_pass"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
