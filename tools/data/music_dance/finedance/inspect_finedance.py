#!/usr/bin/env python3
"""Inspect the real FineDance release and write a reproducible JSON audit."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.data.music_dance.finedance.common import (  # noqa: E402
    inspect_motion_array,
    inventory_audit,
    list_ids,
    load_motion,
    official_genre_split,
    read_label,
    read_wav_info,
    summarize_counts,
    write_json,
)


def inspect_dataset(root: Path, *, sample_count: int, seed: int, source_fps: float) -> dict[str, Any]:
    inventory = inventory_audit(root)
    ids = list_ids(root)
    paired = sorted(ids["motion"] & ids["music_wav"])
    splits, split_metadata = official_genre_split(paired)
    rng = random.Random(seed)
    selected = rng.sample(paired, min(sample_count, len(paired)))
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    audio_differences: list[float] = []
    label_differences: list[int] = []
    music_npy_differences: list[int] = []
    wav_rates: Counter[int] = Counter()
    wav_channels: Counter[int] = Counter()
    frame_counts: dict[str, int] = {}
    temporal_records: list[dict[str, Any]] = []

    for index, sample_id in enumerate(paired, start=1):
        try:
            motion = load_motion(root, sample_id, mmap=True)
            frame_count = int(motion.shape[0])
            frame_counts[sample_id] = frame_count
            label = read_label(root, sample_id)
            wav = read_wav_info(root, sample_id)
            music_npy = np.load(root / "music_npy" / f"{sample_id}.npy", mmap_mode="r", allow_pickle=False)
            if music_npy.ndim != 2 or music_npy.shape[1] != 35:
                raise ValueError(f"{sample_id}: official music_npy must be [T,35], got {music_npy.shape}")
            if not np.issubdtype(music_npy.dtype, np.floating) or not np.isfinite(music_npy).all():
                raise ValueError(f"{sample_id}: official music_npy must be finite floating data")
            audio_difference = wav.duration_sec * source_fps - frame_count
            label_difference = int(label["frames"]) - frame_count
            music_npy_difference = int(music_npy.shape[0]) - frame_count
            audio_differences.append(audio_difference)
            label_differences.append(label_difference)
            music_npy_differences.append(music_npy_difference)
            wav_rates[wav.sample_rate] += 1
            wav_channels[wav.channels] += 1
            temporal_records.append(
                {
                    "sample_id": sample_id,
                    "motion_frames": frame_count,
                    "label_frames": int(label["frames"]),
                    "official_music_npy_frames": int(music_npy.shape[0]),
                    "wav_duration_sec": wav.duration_sec,
                    "wav_frames_at_30hz": wav.duration_sec * source_fps,
                    "audio_minus_motion_frames": audio_difference,
                    "label_minus_motion_frames": label_difference,
                    "music_npy_minus_motion_frames": music_npy_difference,
                }
            )
            if sample_id in selected:
                records.append(
                    {
                        "sample_id": sample_id,
                        "motion": inspect_motion_array(motion, sample_id),
                        "label": label,
                        "official_music_npy": {
                            "shape": list(music_npy.shape),
                            "dtype": str(music_npy.dtype),
                            "finite": True,
                        },
                        "wav": wav.as_dict(),
                        "audio_minus_motion_frames": audio_difference,
                    }
                )
        except Exception as exc:
            failures.append({"sample_id": sample_id, "error": f"{type(exc).__name__}: {exc}"})
        if index % 25 == 0 or index == len(paired):
            print(f"[inspect] {index}/{len(paired)}; failures={len(failures)}")

    selected_order = {sample_id: index for index, sample_id in enumerate(selected)}
    records.sort(key=lambda item: selected_order[item["sample_id"]])
    for record in records:
        print(
            f"{record['sample_id']}: motion={record['motion']['shape']} "
            f"{record['motion']['dtype']}, finite={record['motion']['finite']}, "
            f"wav={record['wav']['duration_sec']:.6f}s, "
            f"audio-motion={record['audio_minus_motion_frames']:.6f} frames, "
            f"label={record['label']}"
        )

    differences = np.asarray(audio_differences, dtype=np.float64)
    split_lookup = {sample_id: split for split, values in splits.items() for sample_id in values}
    split_hours = {
        split: sum(frame_counts[sample_id] for sample_id in values if sample_id in frame_counts)
        / source_fps
        / 3600
        for split, values in splits.items()
    }
    severe = [
        record for record in temporal_records if abs(record["audio_minus_motion_frames"]) > 2.000001
    ]
    return {
        "format": "finedance_inspection_v1",
        "root": str(root.resolve()),
        "inventory": inventory,
        "format_finding": {
            "motion": "[T,315] = translation [T,3] + 52 local rotations [T,52,6]",
            "rotation_representation": "continuous rotation-6D (not axis-angle)",
            "official_loader_evidence": "FineDance data/code/pre_motion.py and render.py",
            "official_music_npy_analysis_only": True,
        },
        "fps": {
            "value": source_fps,
            "explicit_per_file_fps": False,
            "evidence": (
                "Official pre_motion.py declares raw_fps=data_fps=30; all label frame counts "
                "equal motion_T-1, and WAV durations are audited against a 30 Hz clock."
            ),
        },
        "official_split": split_metadata,
        "split_ids": splits,
        "split_hours_before_alignment_filter": split_hours,
        "random_motion_records": records,
        "paired_sample_count": len(paired),
        "total_motion_frames": sum(frame_counts.values()),
        "total_motion_hours": sum(frame_counts.values()) / source_fps / 3600,
        "label_minus_motion_histogram": summarize_counts(label_differences),
        "official_music_npy_minus_motion_histogram": summarize_counts(music_npy_differences),
        "wav_sample_rates": summarize_counts(wav_rates.elements()),
        "wav_channels": summarize_counts(wav_channels.elements()),
        "audio_minus_motion_frames": {
            "min": float(differences.min()) if len(differences) else None,
            "max": float(differences.max()) if len(differences) else None,
            "max_abs": float(np.abs(differences).max()) if len(differences) else None,
            "within_1_count": int((np.abs(differences) <= 1.000001).sum()),
            "within_2_count": int((np.abs(differences) <= 2.000001).sum()),
            "mismatch_gt_2_count": len(severe),
        },
        "severe_audio_motion_mismatches": severe,
        "split_for_each_severe_mismatch": {
            item["sample_id"]: split_lookup[item["sample_id"]] for item in severe
        },
        "failures": failures,
        "final_pass": not failures,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.sample_count < 10:
        raise ValueError("--sample-count must be at least 10")
    if args.source_fps <= 0:
        raise ValueError("--source-fps must be positive")
    report = inspect_dataset(
        args.root.expanduser().resolve(),
        sample_count=args.sample_count,
        seed=args.seed,
        source_fps=args.source_fps,
    )
    write_json(args.output, report)
    summary_keys = (
        "paired_sample_count",
        "total_motion_hours",
        "label_minus_motion_histogram",
        "audio_minus_motion_frames",
        "failures",
        "final_pass",
    )
    print(json.dumps({key: report[key] for key in summary_keys}, indent=2, ensure_ascii=False))
    print(f"[inspect] report: {args.output.expanduser().resolve()}")
    return 0 if report["final_pass"] or not args.strict else 2


if __name__ == "__main__":
    raise SystemExit(main())
