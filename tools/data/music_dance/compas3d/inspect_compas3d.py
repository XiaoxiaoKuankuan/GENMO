#!/usr/bin/env python3
"""Inspect local CoMPAS3D files without downloading missing LFS objects."""

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

from tools.data.music_dance.compas3d.common import (  # noqa: E402
    SMPLX_POSE_LAYOUT,
    build_splits,
    discover_local_sequences,
    ffprobe_media,
    inventory_audit,
    load_npz_fields,
    summarize_counts,
    summarize_numeric,
    validate_source_motion,
    write_json,
)


def _array_summary(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    result: dict[str, Any] = {"shape": list(array.shape), "dtype": str(array.dtype)}
    if array.ndim == 0:
        item = array.item()
        result["value"] = item if isinstance(item, (str, int, float, bool)) or item is None else repr(item)
        return result
    numeric: np.ndarray | None
    if np.issubdtype(array.dtype, np.number):
        numeric = array
    elif array.dtype == object:
        try:
            numeric = np.asarray(array, dtype=np.float64)
        except (TypeError, ValueError):
            numeric = None
    else:
        numeric = None
    if numeric is None:
        result["numeric"] = False
        return result
    result.update(
        {
            "numeric": True,
            "min": float(np.nanmin(numeric)),
            "max": float(np.nanmax(numeric)),
            "nan_count": int(np.isnan(numeric).sum()),
            "inf_count": int(np.isinf(numeric).sum()),
            "finite": bool(np.isfinite(numeric).all()),
        }
    )
    return result


def _motion_record(root: Path, path: Path) -> dict[str, Any]:
    fields = load_npz_fields(path)
    info = validate_source_motion(fields, source=str(path))
    poses = np.asarray(fields["poses"])
    components = {
        name: _array_summary(poses[:, start:end])
        for name, (start, end) in SMPLX_POSE_LAYOUT.items()
    }
    return {
        "path": str(path.relative_to(root)),
        "filename": path.name,
        "role": "leader" if "leader" in path.stem.lower() else "follower",
        "keys": list(fields),
        "fields": {key: _array_summary(value) for key, value in fields.items()},
        "pose_components": components,
        "num_frames": info["num_frames"],
        "fps": info["fps"],
        "fps_source": "NPZ scalar key mocap_frame_rate",
        "duration_sec": info["duration_sec"],
        "gender": info["gender"],
        "surface_model_type": info["surface_model_type"],
    }


def inspect_dataset(
    root: Path,
    reference_root: Path,
    *,
    npz_sample_count: int,
    media_sample_count: int,
    seed: int,
) -> dict[str, Any]:
    complete, _ = discover_local_sequences(root)
    if not complete:
        raise ValueError("no locally complete CoMPAS3D sequence found")
    inventory = inventory_audit(root, reference_root)
    all_npz = [path for sequence in complete for path in (sequence.leader, sequence.follower)]
    rng = random.Random(seed)

    # Ensure coverage across available Pairs and roles before filling randomly.
    detailed_npz: list[Path] = []
    for pair_id in sorted({sequence.parts.pair_id for sequence in complete}):
        sequence = next(row for row in complete if row.parts.pair_id == pair_id)
        detailed_npz.extend((sequence.leader, sequence.follower))
    for path in rng.sample(all_npz, len(all_npz)):
        if path not in detailed_npz:
            detailed_npz.append(path)
        if len(detailed_npz) >= npz_sample_count:
            break
    detailed_npz = detailed_npz[:npz_sample_count]

    motion_records: list[dict[str, Any]] = []
    motion_failures: list[dict[str, str]] = []
    schema_counts: Counter[str] = Counter()
    fps_values: list[float] = []
    gender_values: list[str] = []
    model_values: list[str] = []
    leader_follower_differences: list[int] = []
    all_motion_info: dict[str, dict[str, Any]] = {}
    coordinate_extent_samples: list[dict[str, Any]] = []
    selected_set = set(detailed_npz)
    for sequence in complete:
        lengths: list[int] = []
        for role, path in (("leader", sequence.leader), ("follower", sequence.follower)):
            try:
                fields = load_npz_fields(path)
                info = validate_source_motion(fields, source=str(path))
                schema = tuple(
                    (key, tuple(np.asarray(value).shape), str(np.asarray(value).dtype))
                    for key, value in fields.items()
                )
                schema_counts[repr(schema)] += 1
                fps_values.append(info["fps"])
                gender_values.append(info["gender"])
                model_values.append(info["surface_model_type"])
                lengths.append(info["num_frames"])
                all_motion_info[f"{sequence.sequence_id}_{role}"] = info
                if path in selected_set:
                    motion_records.append(_motion_record(root, path))
                if len(coordinate_extent_samples) < 10:
                    markers = np.asarray(fields["markers_obs"], dtype=np.float64)
                    index = np.linspace(0, len(markers) - 1, min(32, len(markers)), dtype=np.int64)
                    selected = markers[index]
                    extents = np.nanmax(selected, axis=1) - np.nanmin(selected, axis=1)
                    minima = np.nanmin(selected, axis=1)
                    coordinate_extent_samples.append(
                        {
                            "sample_id": f"{sequence.sequence_id}_{role}",
                            "median_marker_extent_xyz_m": np.nanmedian(extents, axis=0).tolist(),
                            "marker_minimum_std_xyz_m": np.nanstd(minima, axis=0).tolist(),
                            "evidence": "largest body extent and most stable floor minimum are on Z",
                        }
                    )
            except Exception as exc:
                motion_failures.append(
                    {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}
                )
        if len(lengths) == 2:
            leader_follower_differences.append(lengths[0] - lengths[1])

    media_records_all: list[dict[str, Any]] = []
    media_failures: list[dict[str, str]] = []
    audio_motion_differences: list[float] = []
    for sequence in complete:
        try:
            media = ffprobe_media(sequence.mp4)
            frames = all_motion_info[f"{sequence.sequence_id}_leader"]["num_frames"]
            fps = all_motion_info[f"{sequence.sequence_id}_leader"]["fps"]
            audio_duration = float(media["audio"]["duration_sec"])
            difference = audio_duration * fps - frames
            media["sequence_id"] = sequence.sequence_id
            media["motion_frames"] = frames
            media["motion_fps"] = fps
            media["motion_duration_sec"] = frames / fps
            media["audio_minus_motion_frames"] = difference
            media_records_all.append(media)
            audio_motion_differences.append(difference)
        except Exception as exc:
            media_failures.append(
                {"sequence_id": sequence.sequence_id, "error": f"{type(exc).__name__}: {exc}"}
            )
    media_detailed_ids = {
        row.sequence_id
        for row in rng.sample(complete, min(media_sample_count, len(complete)))
    }
    media_records = [row for row in media_records_all if row["sequence_id"] in media_detailed_ids]
    music_splits, split_report = build_splits(complete, "music_identity")
    _, official_split_report = build_splits(complete, "official_interaction")
    fps_counter = Counter(round(value, 8) for value in fps_values)
    video_fps_counter = Counter(round(float(row["video"]["fps"]), 6) for row in media_records_all)
    report = {
        "format": "compas3d_inspection_v1",
        "root": str(root),
        "reference_root": str(reference_root),
        "inventory": inventory,
        "complete_sequence_count": len(complete),
        "complete_sequence_ids": [row.sequence_id for row in complete],
        "inspected_npz_count": len(all_npz),
        "detailed_npz_count": len(motion_records),
        "detailed_npz_records": motion_records,
        "npz_schema_variant_count_including_temporal_shape": len(schema_counts),
        "npz_schema_counts": dict(schema_counts),
        "semantic_schema_consistent": not motion_failures
        and len(set(fps_values)) == 1
        and all(value == 0 for value in leader_follower_differences),
        "source_format": {
            "keys": [
                "gender", "surface_model_type", "mocap_frame_rate", "betas", "poses",
                "trans", "markers_obs", "markers_sim", "v_template",
            ],
            "poses": "[T,165] SMPL-X axis-angle = 55 joints x 3",
            "pose_layout": SMPLX_POSE_LAYOUT,
            "trans": "[T,3] root translation in metres",
            "betas": "[300] subject-specific SMPL-X shape coefficients",
            "markers": "[T,53,3] object-dtype arrays containing finite numeric marker positions",
            "v_template": "object scalar None in all inspected local files",
        },
        "fps": {
            "values": summarize_counts(fps_values),
            "source": "mocap_frame_rate stored in every NPZ",
            "finding": "The Vicon capture was 120 FPS, but released MoSh NPZ files are already 30 FPS.",
            "must_not_downsample_current_release_again": all(abs(value - 30.0) < 1e-9 for value in fps_values),
        },
        "genders": summarize_counts(gender_values),
        "surface_model_types": summarize_counts(model_values),
        "leader_minus_follower_frame_histogram": summarize_counts(leader_follower_differences),
        "coordinate_system": {
            "source": "right-handed Z-up SMPL-X/MoSh world coordinates",
            "target": "right-handed Y-up GENMO world coordinates",
            "unit": "metres",
            "transform": "Rx(-90deg): source (x,y,z) -> target (x,z,-y)",
            "pelvis_offset_compensation_required": True,
            "evidence_samples": coordinate_extent_samples,
        },
        "media": {
            "scanned_count": len(media_records_all),
            "detailed_count": len(media_records),
            "detailed_records": media_records,
            "video_fps_histogram": summarize_counts(video_fps_counter.elements()),
            "video_codecs": summarize_counts(row["video"]["codec"] for row in media_records_all),
            "audio_codecs": summarize_counts(row["audio"]["codec"] for row in media_records_all),
            "audio_sample_rates": summarize_counts(row["audio"]["sample_rate"] for row in media_records_all),
            "audio_channels": summarize_counts(row["audio"]["channels"] for row in media_records_all),
            "audio_minus_motion_frames": summarize_numeric(audio_motion_differences),
            "alignment_finding": (
                "All local complete MP4 audio tracks end about 3.5-5.6 frames after motion. "
                "This consistent positive tail is explicitly trimmed to the motion duration."
            ),
        },
        "split_analysis": {
            "default_music_identity": split_report,
            "default_split_ids": music_splits,
            "official_interaction": official_split_report,
        },
        "motion_failures": motion_failures,
        "media_failures": media_failures,
        "final_pass": not motion_failures and not media_failures,
    }
    return report


def _write_summary(path: Path, report: dict[str, Any]) -> None:
    inventory = report["inventory"]
    media = report["media"]
    lines = [
        "CoMPAS3D 本地数据检查摘要",
        "=========================",
        f"本地真实 MP4: {inventory['local_real_mp4_count']}",
        f"本地真实 NPZ: {inventory['local_real_npz_count']}",
        f"当前完整 sequence: {report['complete_sequence_count']}",
        f"缺失/不完整 sequence: {inventory['incomplete_sequence_count']}",
        "NPZ: poses [T,165] axis-angle, trans [T,3], betas [300]",
        f"NPZ FPS: {report['fps']['values']}（发布文件已经是 30 FPS）",
        f"性别: {report['genders']}",
        f"MP4 video FPS: {media['video_fps_histogram']}",
        f"MP4 audio: {media['audio_codecs']}, sample rates={media['audio_sample_rates']}, channels={media['audio_channels']}",
        f"audio-motion 帧差: {media['audio_minus_motion_frames']}",
        "坐标: 源 Z-up 米制；GENMO Y-up 米制；使用 Rx(-90°) 并补偿 pelvis offset。",
        "默认 split: song1/song2=train, song3=val, song4=test，防止音乐 identity 泄漏。",
        f"最终检查通过: {report['final_pass']}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--npz-sample-count", type=int, default=20)
    parser.add_argument("--media-sample-count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.npz_sample_count < 10 or args.media_sample_count < 10:
        raise ValueError("sample counts must each be at least 10")
    report = inspect_dataset(
        args.root.expanduser().resolve(),
        args.reference_root.expanduser().resolve(),
        npz_sample_count=args.npz_sample_count,
        media_sample_count=args.media_sample_count,
        seed=args.seed,
    )
    output = args.output.expanduser().resolve()
    write_json(output, report)
    summary = (
        args.summary_output.expanduser().resolve()
        if args.summary_output is not None
        else output.with_name("inspection_summary.txt")
    )
    _write_summary(summary, report)
    console = {
        "complete_sequence_count": report["complete_sequence_count"],
        "real_npz_count": report["inventory"]["local_real_npz_count"],
        "real_mp4_count": report["inventory"]["local_real_mp4_count"],
        "source_fps": report["fps"]["values"],
        "audio_minus_motion_frames": report["media"]["audio_minus_motion_frames"],
        "final_pass": report["final_pass"],
    }
    print(json.dumps(console, ensure_ascii=False, indent=2))
    print(f"[inspect] JSON: {output}")
    print(f"[inspect] summary: {summary}")
    return 0 if report["final_pass"] or not args.strict else 2


if __name__ == "__main__":
    raise SystemExit(main())
