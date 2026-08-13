#!/usr/bin/env python3
"""Inspect real AIOZ-GDANCE ZIP/directory contents and emit a JSON audit."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.data.music_dance.aioz.common import (  # noqa: E402
    AiozRawDataset,
    SequenceLabel,
    inspect_motion_payload,
    select_group_labels,
    summarize_counter,
    write_json,
)


def inspect_one(source: AiozRawDataset, label: SequenceLabel, source_fps: float) -> dict[str, Any]:
    payload = source.load_motion(label.group_id)
    motion = inspect_motion_payload(payload, label.group_id)
    wav = source.wav_info(label.group_id)
    audio_frames_at_motion_fps = wav.duration_sec * source_fps
    motion_audio_difference = audio_frames_at_motion_fps - motion["num_frames"]
    return {
        **label.as_dict(),
        "motion_member": f"motions_smpl/{label.group_id}.pkl",
        "audio_member": f"musics/{label.group_id}.wav",
        "motion": motion,
        "wav": wav.as_dict(),
        "audio_frames_at_declared_motion_fps": audio_frames_at_motion_fps,
        "audio_minus_motion_frames": motion_audio_difference,
        "abs_audio_motion_frame_difference": abs(motion_audio_difference),
    }


def print_sample(record: dict[str, Any]) -> None:
    motion = record["motion"]
    wav = record["wav"]
    print(
        f"[{record['split']}] {record['group_id']} -> "
        f"motions_smpl/{record['group_id']}.pkl + musics/{record['group_id']}.wav"
    )
    print(f"  keys={motion['keys']}")
    print(
        "  shapes: "
        f"smpl_poses={motion['smpl_poses_shape']} ({motion['smpl_poses_dtype']}), "
        f"root_trans={motion['root_trans_shape']} ({motion['root_trans_dtype']}), "
        f"smpl_betas={motion['smpl_betas_shape']} ({motion['smpl_betas_dtype']})"
    )
    print(
        f"  persons={motion['num_persons']}, frames={motion['num_frames']}, meta={motion['meta']}"
    )
    print(
        f"  wav: sr={wav['sample_rate']}, channels={wav['channels']}, "
        f"frames={wav['audio_frames']}, duration={wav['duration_sec']:.6f}s; "
        f"audio-motion diff@30Hz={record['audio_minus_motion_frames']:.6f} frames"
    )


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    with AiozRawDataset(args.root) as source:
        audit = source.audit_splits_and_inventory()
        labels = source.read_labels()
        selected = select_group_labels(
            labels,
            sample_groups=None if args.all else args.num_groups,
            seed=args.seed,
        )
        records: list[dict[str, Any]] = []
        failures: list[dict[str, str]] = []
        for index, label in enumerate(selected, start=1):
            try:
                records.append(inspect_one(source, label, args.source_fps))
            except Exception as exc:
                failures.append(
                    {
                        "group_id": label.group_id,
                        "split": label.split,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                if args.strict:
                    raise
            if index % args.report_every == 0 or index == len(selected):
                print(f"[inspect] {index}/{len(selected)} groups; failures={len(failures)}")

    for record in records[: args.print_groups]:
        print_sample(record)
    people = np.asarray([record["motion"]["num_persons"] for record in records], dtype=np.int64)
    frames = np.asarray([record["motion"]["num_frames"] for record in records], dtype=np.int64)
    differences = np.asarray(
        [record["audio_minus_motion_frames"] for record in records], dtype=np.float64
    )
    split_group_counts = Counter(record["split"] for record in records)
    split_person_counts = Counter()
    for record in records:
        split_person_counts[record["split"]] += record["motion"]["num_persons"]
    pose_shapes = Counter(tuple(record["motion"]["smpl_poses_shape"][2:]) for record in records)
    key_layouts = Counter(tuple(record["motion"]["keys"]) for record in records)
    wav_rates = Counter(record["wav"]["sample_rate"] for record in records)
    wav_channels = Counter(record["wav"]["channels"] for record in records)
    filename_matches = sum(
        record["motion"]["filename_frame_span"] == record["motion"]["num_frames"]
        for record in records
    )
    total_person_frames = sum(
        record["motion"]["num_persons"] * record["motion"]["num_frames"] for record in records
    )
    report: dict[str, Any] = {
        "root": str(args.root.expanduser().resolve()),
        "source_fps": {
            "value": args.source_fps,
            "explicit_fps_key_in_pkl": False,
            "evidence": (
                "The downloaded PKL has no fps key. FPS is verified from tensor T, "
                "meta/filename start-end spans, and WAV duration before conversion."
            ),
            "filename_span_matches_motion_count": filename_matches,
            "audio_minus_motion_frames": {
                "min": float(differences.min()) if len(differences) else None,
                "max": float(differences.max()) if len(differences) else None,
                "mean": float(differences.mean()) if len(differences) else None,
                "median": float(np.median(differences)) if len(differences) else None,
                "max_abs": float(np.abs(differences).max()) if len(differences) else None,
                "within_1_count": int((np.abs(differences) <= 1).sum()),
                "within_2_count": int((np.abs(differences) <= 2).sum()),
            },
        },
        "inventory_audit": audit,
        "inspected_all_groups": args.all,
        "selected_group_count": len(selected),
        "valid_group_count": len(records),
        "failure_count": len(failures),
        "failures": failures,
        "group_counts_by_split": summarize_counter(split_group_counts),
        "person_sample_counts_by_split": summarize_counter(split_person_counts),
        "person_count": {
            "min": int(people.min()) if len(people) else None,
            "max": int(people.max()) if len(people) else None,
            "mean": float(people.mean()) if len(people) else None,
            "histogram": summarize_counter(Counter(people.tolist())),
        },
        "motion_frames": {
            "min": int(frames.min()) if len(frames) else None,
            "max": int(frames.max()) if len(frames) else None,
            "sum_group_frames": int(frames.sum()) if len(frames) else 0,
            "sum_person_frames": total_person_frames,
            "person_motion_hours_at_source_fps": total_person_frames / args.source_fps / 3600,
        },
        "pose_tail_shapes": {str(key): value for key, value in sorted(pose_shapes.items())},
        "pkl_key_layouts": {str(key): value for key, value in sorted(key_layouts.items())},
        "wav_sample_rates": summarize_counter(wav_rates),
        "wav_channels": summarize_counter(wav_channels),
        "records": records,
        "final_pass": not failures
        and not audit["missing_motion"]
        and not audit["missing_music"]
        and not audit["extra_motion"]
        and not audit["extra_music"]
        and all(audit["split_csv_matches_internal_txt"].values()),
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--num-groups", type=int, default=10)
    parser.add_argument("--all", action="store_true", help="Inspect all official split groups")
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--print-groups", type=int, default=10)
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.num_groups <= 0 or args.print_groups < 0 or args.report_every <= 0:
        raise ValueError("group/report counts must be positive (print-groups may be zero)")
    if args.source_fps <= 0:
        raise ValueError("--source-fps must be positive")
    report = build_report(args)
    if args.output is not None:
        write_json(args.output, report)
        print(f"[inspect] report: {args.output.resolve()}")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "selected_group_count",
                    "valid_group_count",
                    "failure_count",
                    "group_counts_by_split",
                    "person_sample_counts_by_split",
                    "person_count",
                    "motion_frames",
                    "wav_sample_rates",
                    "wav_channels",
                    "final_pass",
                )
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if report["final_pass"] or not args.strict else 2


if __name__ == "__main__":
    raise SystemExit(main())
