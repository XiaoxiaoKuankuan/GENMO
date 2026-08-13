#!/usr/bin/env python3
"""Convert safely aligned FineDance motions and WAVs to canonical GENMO data."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gem.datasets.aistpp.aistplusplus import validate_musicfeat_v2  # noqa: E402
from gem.utils.music_features import (  # noqa: E402
    EDGE_TARGET_FPS,
    align_features_to_length,
    extract_edge_baseline35,
)
from tools.data.music_dance.finedance.common import (  # noqa: E402
    FORMAT_VERSION,
    SPLITS,
    convert_motion_array,
    inventory_audit,
    list_ids,
    load_motion,
    official_genre_split,
    read_label,
    read_wav_info,
    split_lookup,
    validate_canonical_motion,
    write_json,
    write_jsonl,
)


def _prepare_output(root: Path, overwrite: bool) -> None:
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise FileExistsError(f"output is not empty; use a fresh path or --overwrite: {root}")
    for name in ("motions", "musicfeat_v2", "manifests", "reports", ".staging"):
        (root / name).mkdir(parents=True, exist_ok=True)


def _save_group(
    output_root: Path,
    sample_id: str,
    payload: dict[str, Any],
    music: torch.Tensor,
    *,
    overwrite: bool,
) -> tuple[str, str]:
    motion_relative = f"motions/{sample_id}.pt"
    music_relative = f"musicfeat_v2/{sample_id}_musicfeat_fps30.pt"
    targets = [output_root / motion_relative, output_root / music_relative]
    if any(path.exists() for path in targets) and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing output for {sample_id}")
    staging = output_root / ".staging" / f"{sample_id}.{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        motion_staged = staging / f"{sample_id}.pt"
        music_staged = staging / f"{sample_id}_musicfeat_fps30.pt"
        torch.save(payload, motion_staged)
        torch.save(music, music_staged)
        os.replace(motion_staged, targets[0])
        os.replace(music_staged, targets[1])
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return motion_relative, music_relative


def convert_one(
    *,
    raw_root: Path,
    output_root: Path,
    sample_id: str,
    split: str,
    source_fps: float,
    target_fps: float,
    max_audio_motion_frame_mismatch: float,
    max_feature_frame_mismatch: int,
    overwrite: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = load_motion(raw_root, sample_id)
    source_frames = int(source.shape[0])
    wav = read_wav_info(raw_root, sample_id)
    audio_difference = wav.duration_sec * source_fps - source_frames
    if abs(audio_difference) > max_audio_motion_frame_mismatch:
        raise ValueError(
            f"{sample_id}: WAV/motion mismatch={audio_difference:.6f} frames at {source_fps:g}Hz "
            f"exceeds limit={max_audio_motion_frame_mismatch:g}; refusing large trim/pad"
        )
    tensors = convert_motion_array(
        source,
        sample_id=sample_id,
        source_fps=source_fps,
        target_fps=target_fps,
    )
    target_frames = validate_canonical_motion(tensors, source=sample_id)
    audio_path = raw_root / "music_wav" / f"{sample_id}.wav"
    features, feature_metadata = extract_edge_baseline35(audio_path)
    raw_feature_frames = int(features.shape[0])
    feature_difference = raw_feature_frames - target_frames
    if abs(feature_difference) > max_feature_frame_mismatch:
        raise ValueError(
            f"{sample_id}: newly extracted EDGE35/motion mismatch={feature_difference} frames "
            f"(music={raw_feature_frames}, motion={target_frames}), "
            f"limit={max_feature_frame_mismatch}"
        )
    features = align_features_to_length(features, target_frames, "trim_or_pad_last")
    features = features.detach().cpu().float().contiguous()
    validate_musicfeat_v2(features, source=f"FineDance EDGE35 {sample_id}")
    if len(features) != target_frames:
        raise RuntimeError(f"{sample_id}: aligned music_T != motion_T")
    label = read_label(raw_root, sample_id)
    payload: dict[str, Any] = {
        "format": "finedance_genmo_smpl_v1",
        "format_version": FORMAT_VERSION,
        "sample_id": sample_id,
        "dataset": "finedance",
        "fps": target_fps,
        "num_frames": target_frames,
        "pose": tensors["pose"],
        "global_orient": tensors["pose"][:, :3],
        "body_pose": tensors["pose"][:, 3:66],
        "transl": tensors["transl"],
        "betas": tensors["betas"],
        "label": label,
        "source": {
            "motion_path": f"motion/{sample_id}.npy",
            "audio_path": f"music_wav/{sample_id}.wav",
            "source_fps": source_fps,
            "source_num_frames": source_frames,
            "source_feature_dim": 315,
            "source_layout": "translation_3_plus_52x_rotation6d",
            "rotation_conversion": "rotation6d_to_matrix_to_axis_angle",
            "kept_joint_slice": "0:22",
            "discarded_joint_slice": "22:52 (SMPL-H hands)",
            "joint_reordering_applied": False,
            "coordinate_transform_applied": False,
            "translation_scale_applied": False,
            "betas_source": "neutral_zeros_finedance_has_no_shape",
        },
    }
    motion_relative, music_relative = _save_group(
        output_root, sample_id, payload, features, overwrite=overwrite
    )
    manifest = {
        "sample_id": sample_id,
        "dataset": "finedance",
        "motion_path": motion_relative,
        "music_feature_path": music_relative,
        "fps": target_fps,
        "num_frames": target_frames,
        "split": split,
        "song_name": label["name"],
        "coarse_style": label["style1"],
        "fine_style": label["style2"],
    }
    if feature_difference == 0:
        alignment_action = "exact"
    elif feature_difference > 0:
        alignment_action = "trim_music_tail"
    else:
        alignment_action = "pad_music_last_frame"
    detail = {
        **manifest,
        "source_motion_frames": source_frames,
        "source_wav": wav.as_dict(),
        "audio_minus_motion_frames": audio_difference,
        "raw_music_feature_frames": raw_feature_frames,
        "music_minus_motion_before_feature_alignment": feature_difference,
        "alignment_action": alignment_action,
        "feature_metadata": {
            key: value for key, value in feature_metadata.items() if key not in {"source_path", "feature_names"}
        },
    }
    return manifest, detail


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--max-audio-motion-frame-mismatch", type=float, default=2.0)
    parser.add_argument("--max-feature-frame-mismatch", type=int, default=2)
    parser.add_argument("--report-every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not math.isclose(args.target_fps, EDGE_TARGET_FPS):
        raise ValueError(f"EDGE baseline35 requires target FPS={EDGE_TARGET_FPS}")
    if args.source_fps <= 0 or args.report_every <= 0:
        raise ValueError("source-fps/report-every must be positive")
    raw_root = args.root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    _prepare_output(output_root, args.overwrite)
    inventory = inventory_audit(raw_root)
    ids = list_ids(raw_root)
    paired = sorted(ids["motion"] & ids["music_wav"])
    splits, split_metadata = official_genre_split(paired)
    lookup = split_lookup(splits)
    if args.sample_id:
        unknown = sorted(set(args.sample_id) - set(paired))
        if unknown:
            raise KeyError(f"requested sample IDs are not complete motion+WAV pairs: {unknown}")
        selected = list(dict.fromkeys(args.sample_id))
    else:
        selected = paired[: args.sample_count] if args.sample_count is not None else paired
    if not selected:
        raise ValueError("no FineDance samples selected")
    manifests: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, sample_id in enumerate(selected, start=1):
        try:
            manifest, detail = convert_one(
                raw_root=raw_root,
                output_root=output_root,
                sample_id=sample_id,
                split=lookup[sample_id],
                source_fps=args.source_fps,
                target_fps=args.target_fps,
                max_audio_motion_frame_mismatch=args.max_audio_motion_frame_mismatch,
                max_feature_frame_mismatch=args.max_feature_frame_mismatch,
                overwrite=args.overwrite,
            )
            manifests.append(manifest)
            details.append(detail)
        except Exception as exc:
            failures.append(
                {"sample_id": sample_id, "split": lookup[sample_id], "error": f"{type(exc).__name__}: {exc}"}
            )
        if index % args.report_every == 0 or index == len(selected):
            print(f"[convert] {index}/{len(selected)}; ok={len(manifests)}, failed={len(failures)}")
    manifests.sort(key=lambda row: row["sample_id"])
    details.sort(key=lambda row: row["sample_id"])
    for split in SPLITS:
        write_jsonl(
            output_root / "manifests" / f"{split}.jsonl",
            (row for row in manifests if row["split"] == split),
        )
    write_jsonl(output_root / "manifests" / "samples.jsonl", details)
    converted_frames = sum(row["num_frames"] for row in manifests)
    report = {
        "format": "finedance_genmo_conversion_report_v1",
        "source_root": str(raw_root),
        "output_root": str(output_root),
        "settings": {
            "root": str(raw_root),
            "output_root": str(output_root),
            "source_fps": args.source_fps,
            "target_fps": args.target_fps,
            "sample_count": args.sample_count,
            "sample_ids": list(args.sample_id),
            "max_audio_motion_frame_mismatch": args.max_audio_motion_frame_mismatch,
            "max_feature_frame_mismatch": args.max_feature_frame_mismatch,
            "overwrite": args.overwrite,
            "strict": args.strict,
        },
        "inventory": inventory,
        "official_split": split_metadata,
        "selected_sample_count": len(selected),
        "converted_sample_count": len(manifests),
        "sample_counts_by_split": {
            split: sum(row["split"] == split for row in manifests) for split in SPLITS
        },
        "total_frames": converted_frames,
        "total_hours": converted_frames / args.target_fps / 3600,
        "trimmed_or_padded_samples": [
            {
                "sample_id": row["sample_id"],
                "difference": row["music_minus_motion_before_feature_alignment"],
                "action": row["alignment_action"],
            }
            for row in details
            if row["alignment_action"] != "exact"
        ],
        "failed_or_skipped": failures,
        "failed_or_skipped_count": len(failures),
        "final_pass": len(manifests) == len(selected) and not failures,
    }
    write_json(output_root / "reports" / "conversion_report.json", report)
    print(
        json.dumps(
            {key: report[key] for key in (
                "selected_sample_count", "converted_sample_count", "sample_counts_by_split",
                "total_hours", "failed_or_skipped_count", "final_pass"
            )},
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"[convert] report: {output_root / 'reports' / 'conversion_report.json'}")
    return 0 if report["final_pass"] or not args.strict else 2


if __name__ == "__main__":
    raise SystemExit(main())
