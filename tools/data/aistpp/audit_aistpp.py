#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Audit official AIST++ annotations without modifying source data."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_ANNOTATIONS = Path("/home/weili/datasets/AISTPP_official/annotations")


def _path_report(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "exists": path.exists(), "is_dir": path.is_dir()}


def audit_annotations(
    annotations_root: str | Path,
    aligned_wav_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Return a JSON-serializable AIST++ structure, motion, and WAV coverage audit."""
    root = Path(annotations_root).expanduser()
    required = {
        "motions": root / "motions",
        "cameras": root / "cameras",
        "keypoints2d": root / "keypoints2d",
        "splits": root / "splits",
        "ignore_list": root / "ignore_list.txt",
    }
    report: dict[str, Any] = {
        "annotations_root": str(root.resolve()),
        "required_paths": {name: _path_report(path) for name, path in required.items()},
    }

    motion_paths = sorted(required["motions"].glob("*.pkl")) if required["motions"].is_dir() else []
    frame_lengths: list[int] = []
    invalid_motions: list[dict[str, str]] = []
    for path in motion_paths:
        try:
            with path.open("rb") as file:
                motion = pickle.load(file)
            poses = motion["smpl_poses"]
            if getattr(poses, "ndim", 0) != 2:
                raise ValueError(f"smpl_poses shape={getattr(poses, 'shape', None)}")
            frame_lengths.append(int(len(poses)))
        except Exception as exc:
            invalid_motions.append({"sequence": path.stem, "error": str(exc)})

    if frame_lengths:
        length_array = np.asarray(frame_lengths)
        length_stats: dict[str, Any] = {
            "min": int(length_array.min()),
            "max": int(length_array.max()),
            "mean": float(length_array.mean()),
            "median": float(np.median(length_array)),
            "downsampled_30fps_min": int((length_array.min() + 1) // 2),
            "downsampled_30fps_max": int((length_array.max() + 1) // 2),
        }
    else:
        length_stats = {}
    report["motions"] = {
        "sequence_count": len(motion_paths),
        "valid_sequence_count": len(frame_lengths),
        "invalid": invalid_motions,
        "frame_length_stats": length_stats,
        "official_fps_assumption": 60,
        "fps_evidence": (
            "Official AIST++ motions are defined at 60 FPS; these pickle files do not store "
            "an explicit FPS field, so structural compatibility is checked and 30 FPS length "
            "is computed exactly as len(smpl_poses[::2])."
        ),
        "compatible_with_sync_downsample_by_2": not invalid_motions and bool(frame_lengths),
    }
    report["file_counts"] = {
        "cameras": len(list(required["cameras"].glob("*"))) if required["cameras"].is_dir() else 0,
        "keypoints2d": len(list(required["keypoints2d"].glob("*")))
        if required["keypoints2d"].is_dir()
        else 0,
        "splits": len(list(required["splits"].glob("*"))) if required["splits"].is_dir() else 0,
    }

    annotation_audio = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".wav", ".mp3", ".flac"}
    )
    report["annotation_audio"] = {
        "count": len(annotation_audio),
        "files": [str(path.resolve()) for path in annotation_audio[:20]],
    }
    if not annotation_audio:
        report["annotation_audio"]["note"] = (
            "annotations only; music inference can still run with --audio. Batch AIST++ "
            "feature preparation additionally requires aligned per-sequence WAVs."
        )

    if aligned_wav_dir is not None:
        wav_root = Path(aligned_wav_dir).expanduser()
        covered = [path.stem for path in motion_paths if (wav_root / f"{path.stem}.wav").is_file()]
        missing = [path.stem for path in motion_paths if path.stem not in set(covered)]
        report["aligned_wav_coverage"] = {
            "path": str(wav_root.resolve()),
            "exists": wav_root.is_dir(),
            "covered": len(covered),
            "total": len(motion_paths),
            "coverage_ratio": len(covered) / len(motion_paths) if motion_paths else 0.0,
            "missing_sequences": missing,
        }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-root", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--aligned-wav-dir", type=Path)
    parser.add_argument(
        "--output", type=Path, help="Optional JSON output; stdout is always printed"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_annotations(args.annotations_root, args.aligned_wav_dir)
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    print(rendered, end="")
    if args.output is not None:
        if args.output.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing report: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    missing_required = [
        name for name, info in report["required_paths"].items() if not info["exists"]
    ]
    return 1 if missing_required else 0


if __name__ == "__main__":
    raise SystemExit(main())
