#!/usr/bin/env python3
"""Convert AIOZ group motions into independent GENMO SMPL dancer samples."""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from tools.data.music_dance.aioz.common import (  # noqa: E402
    FORMAT_VERSION,
    SPLITS,
    AiozRawDataset,
    SequenceLabel,
    convert_person_motion,
    inspect_motion_payload,
    select_group_labels,
    validate_canonical_smpl,
    write_json,
    write_jsonl,
)

_WORKER_SOURCE: AiozRawDataset | None = None


def _close_worker_source() -> None:
    global _WORKER_SOURCE
    if _WORKER_SOURCE is not None:
        _WORKER_SOURCE.__exit__(None, None, None)
        _WORKER_SOURCE = None


def _configure_numba_cache(cache_root: str | Path) -> str:
    """Isolate librosa/Numba caches from stale shared-environment artifacts."""
    worker_cache = Path(cache_root).expanduser().resolve() / f"worker_{os.getpid()}"
    worker_cache.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(worker_cache)
    return str(worker_cache)


def _init_worker(raw_root: str, numba_cache_root: str) -> None:
    global _WORKER_SOURCE
    _configure_numba_cache(numba_cache_root)
    _WORKER_SOURCE = AiozRawDataset(raw_root)
    _WORKER_SOURCE.__enter__()
    atexit.register(_close_worker_source)


def _save_group_outputs(
    *,
    output_root: Path,
    label: SequenceLabel,
    motions: list[dict[str, Any]],
    music_features: torch.Tensor,
    overwrite: bool,
) -> tuple[str, list[str]]:
    music_relative = f"musicfeat_v2/{label.group_id}_musicfeat_fps30.pt"
    sample_relatives = [f"motions/{item['sample_id']}.pt" for item in motions]
    final_paths = [output_root / music_relative]
    final_paths.extend(output_root / relative for relative in sample_relatives)
    existing = [str(path) for path in final_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing group output: {existing[0]}")

    staging = output_root / ".staging" / f"{label.group_id}.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        staged_music = staging / Path(music_relative).name
        torch.save(music_features.detach().cpu().float().contiguous(), staged_music)
        staged_motions: list[Path] = []
        for item, relative in zip(motions, sample_relatives):
            staged = staging / Path(relative).name
            torch.save(item, staged)
            staged_motions.append(staged)
        for final in final_paths:
            final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged_music, final_paths[0])
        for staged, final in zip(staged_motions, final_paths[1:]):
            os.replace(staged, final)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return music_relative, sample_relatives


def convert_group(
    source: AiozRawDataset,
    label: SequenceLabel,
    settings: dict[str, Any],
) -> dict[str, Any]:
    group_id = label.group_id
    payload = source.load_motion(group_id)
    source_motion = inspect_motion_payload(payload, group_id)
    wav = source.wav_info(group_id)
    source_fps = float(settings["source_fps"])
    target_fps = float(settings["target_fps"])
    source_frames = int(source_motion["num_frames"])
    audio_minus_motion = wav.duration_sec * source_fps - source_frames
    if abs(audio_minus_motion) > float(settings["max_audio_motion_frame_mismatch"]):
        raise ValueError(
            f"{group_id}: WAV/motion mismatch is {audio_minus_motion:.6f} source frames, "
            f"limit={settings['max_audio_motion_frame_mismatch']}"
        )

    canonical_motions = [
        convert_person_motion(
            payload,
            group_id=group_id,
            person_id=person_id,
            source_fps=source_fps,
            target_fps=target_fps,
        )
        for person_id in range(source_motion["num_persons"])
    ]
    target_frames = validate_canonical_smpl(canonical_motions[0], source=group_id)
    if any(
        validate_canonical_smpl(motion, source=group_id) != target_frames
        for motion in canonical_motions
    ):
        raise RuntimeError(f"{group_id}: dancers do not share one output length")

    with tempfile.TemporaryDirectory(prefix="aioz_audio_") as temporary_dir:
        wav_path = source.copy_wav(group_id, Path(temporary_dir) / f"{group_id}.wav")
        raw_features, feature_metadata = extract_edge_baseline35(wav_path)
    raw_feature_frames = int(raw_features.shape[0])
    feature_difference = raw_feature_frames - target_frames
    if abs(feature_difference) > int(settings["max_feature_frame_mismatch"]):
        raise ValueError(
            f"{group_id}: EDGE35/motion mismatch is {feature_difference} frames "
            f"(music={raw_feature_frames}, motion={target_frames}), "
            f"limit={settings['max_feature_frame_mismatch']}"
        )
    features = align_features_to_length(raw_features, target_frames, "trim_or_pad_last")
    features = features.detach().cpu().float().contiguous()
    validate_musicfeat_v2(features, source=f"AIOZ EDGE35 {group_id}")
    if len(features) != target_frames:
        raise RuntimeError(f"{group_id}: aligned music length still disagrees with motion")

    width = max(2, len(str(source_motion["num_persons"] - 1)))
    motion_payloads: list[dict[str, Any]] = []
    for person_id, smpl_params in enumerate(canonical_motions):
        sample_id = f"{group_id}_dancer_{person_id:0{width}d}"
        motion_payloads.append(
            {
                "format": "aioz_genmo_smpl_v1",
                "format_version": FORMAT_VERSION,
                "sample_id": sample_id,
                "group_id": group_id,
                "person_id": person_id,
                "fps": target_fps,
                "num_frames": target_frames,
                "global_orient": smpl_params["global_orient"],
                "body_pose": smpl_params["body_pose"],
                "transl": smpl_params["transl"],
                "betas": smpl_params["betas"],
                "source": {
                    "motion_path": f"motions_smpl/{group_id}.pkl",
                    "source_fps": source_fps,
                    "source_num_frames": source_frames,
                    "source_pose_dim": 72,
                    "kept_pose_slice": "0:66",
                    "discarded_pose_slice": "66:72",
                    "joint_reordering_applied": False,
                    "coordinate_transform_applied": False,
                    "translation_scale_applied": False,
                },
            }
        )

    music_relative, motion_relatives = _save_group_outputs(
        output_root=Path(settings["output_root"]),
        label=label,
        motions=motion_payloads,
        music_features=features,
        overwrite=bool(settings["overwrite"]),
    )
    manifests: list[dict[str, Any]] = []
    for payload_out, motion_relative in zip(motion_payloads, motion_relatives):
        manifests.append(
            {
                "sample_id": payload_out["sample_id"],
                "group_id": group_id,
                "person_id": payload_out["person_id"],
                "motion_path": motion_relative,
                "music_feature_path": music_relative,
                "num_frames": target_frames,
                "fps": target_fps,
                "split": label.split,
                "music_genre": label.music_genre,
                "dance_style": label.dance_style,
            }
        )
    if feature_difference == 0:
        alignment_action = "exact"
    elif feature_difference > 0:
        alignment_action = "trim_music_tail"
    else:
        alignment_action = "pad_music_last_frame"
    return {
        "group_id": group_id,
        "split": label.split,
        "music_genre": label.music_genre,
        "dance_style": label.dance_style,
        "num_persons": source_motion["num_persons"],
        "source_motion_frames": source_frames,
        "source_fps": source_fps,
        "target_motion_frames": target_frames,
        "target_fps": target_fps,
        "source_motion_member": f"motions_smpl/{group_id}.pkl",
        "source_audio_member": f"musics/{group_id}.wav",
        "wav": wav.as_dict(),
        "audio_minus_motion_source_frames": audio_minus_motion,
        "raw_music_feature_frames": raw_feature_frames,
        "final_music_feature_frames": len(features),
        "music_minus_motion_before_alignment": feature_difference,
        "alignment_action": alignment_action,
        "feature_metadata": {
            key: value
            for key, value in feature_metadata.items()
            if key not in {"source_path", "feature_names"}
        },
        "music_feature_path": music_relative,
        "samples": manifests,
    }


def _convert_worker(task: tuple[dict[str, str], dict[str, Any]]) -> dict[str, Any]:
    if _WORKER_SOURCE is None:
        raise RuntimeError("AIOZ conversion worker was not initialized")
    row, settings = task
    return convert_group(_WORKER_SOURCE, SequenceLabel(**row), settings)


def _prepare_output(output_root: Path, overwrite: bool) -> None:
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(
            f"output directory is not empty; choose a fresh path or pass --overwrite: {output_root}"
        )
    for relative in ("motions", "musicfeat_v2", "manifests", "reports", ".staging"):
        (output_root / relative).mkdir(parents=True, exist_ok=True)


def _summary(
    *,
    source_root: Path,
    output_root: Path,
    selected: list[SequenceLabel],
    groups: list[dict[str, Any]],
    failures: list[dict[str, str]],
    settings: dict[str, Any],
) -> dict[str, Any]:
    samples = [sample for group in groups for sample in group["samples"]]
    split_groups = Counter(group["split"] for group in groups)
    split_samples = Counter(sample["split"] for sample in samples)
    adjusted = [
        {
            "group_id": group["group_id"],
            "raw_music_feature_frames": group["raw_music_feature_frames"],
            "target_motion_frames": group["target_motion_frames"],
            "difference": group["music_minus_motion_before_alignment"],
            "action": group["alignment_action"],
        }
        for group in groups
        if group["alignment_action"] != "exact"
    ]
    person_frames = sum(sample["num_frames"] for sample in samples)
    music_frames = sum(group["target_motion_frames"] for group in groups)
    return {
        "format": "aioz_genmo_conversion_report_v1",
        "source_root": str(source_root.resolve()),
        "output_root": str(output_root.resolve()),
        "settings": settings,
        "selected_group_count": len(selected),
        "converted_group_count": len(groups),
        "dancer_sample_count": len(samples),
        "group_counts_by_split": {split: split_groups[split] for split in SPLITS},
        "sample_counts_by_split": {split: split_samples[split] for split in SPLITS},
        "person_motion_frames": person_frames,
        "person_motion_hours": person_frames / settings["target_fps"] / 3600,
        "unique_music_count": len(groups),
        "unique_music_frames": music_frames,
        "unique_music_hours": music_frames / settings["target_fps"] / 3600,
        "motion_music_length_mismatch_after_alignment_count": sum(
            group["target_motion_frames"] != group["final_music_feature_frames"] for group in groups
        ),
        "trimmed_or_padded_group_count": len(adjusted),
        "trimmed_or_padded_groups": adjusted,
        "failed_or_skipped_count": len(failures),
        "failed_or_skipped": failures,
        "final_pass": len(groups) == len(selected) and not failures,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--sample-groups", type=int)
    parser.add_argument("--group-id", action="append", default=[])
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-audio-motion-frame-mismatch", type=float, default=2.0)
    parser.add_argument("--max-feature-frame-mismatch", type=int, default=2)
    parser.add_argument("--report-every", type=int, default=10)
    parser.add_argument(
        "--numba-cache-root",
        type=Path,
        default=Path("/tmp/genmo_aioz_numba_cache"),
        help="Per-process librosa/Numba cache root (avoids stale shared cache crashes)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not math.isclose(args.target_fps, EDGE_TARGET_FPS):
        raise ValueError(f"EDGE baseline35 is fixed at {EDGE_TARGET_FPS} FPS")
    if args.source_fps <= 0 or args.workers <= 0 or args.report_every <= 0:
        raise ValueError("source-fps/workers/report-every must be positive")
    if args.max_audio_motion_frame_mismatch < 0 or args.max_feature_frame_mismatch < 0:
        raise ValueError("alignment mismatch limits must be non-negative")
    output_root = args.output_root.expanduser().resolve()
    _prepare_output(output_root, args.overwrite)
    cache_namespace = hashlib.sha256(str(output_root).encode()).hexdigest()[:12]
    numba_cache_root = args.numba_cache_root.expanduser().resolve() / cache_namespace
    with AiozRawDataset(args.root) as source:
        audit = source.audit_splits_and_inventory()
        if (
            audit["missing_motion"]
            or audit["missing_music"]
            or audit["extra_motion"]
            or audit["extra_music"]
            or not all(audit["split_csv_matches_internal_txt"].values())
        ):
            raise ValueError(f"raw AIOZ inventory/split audit failed: {audit}")
        selected = select_group_labels(
            source.read_labels(),
            sample_groups=args.sample_groups,
            seed=args.seed,
            requested_group_ids=args.group_id,
        )

    settings = {
        "output_root": str(output_root),
        "source_fps": float(args.source_fps),
        "target_fps": float(args.target_fps),
        "max_audio_motion_frame_mismatch": float(args.max_audio_motion_frame_mismatch),
        "max_feature_frame_mismatch": int(args.max_feature_frame_mismatch),
        "overwrite": bool(args.overwrite),
        "workers": int(args.workers),
        "seed": int(args.seed),
        "numba_cache_root": str(numba_cache_root),
    }
    tasks = [(row.as_dict(), settings) for row in selected]
    groups: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    if args.workers == 1:
        _configure_numba_cache(numba_cache_root)
        with AiozRawDataset(args.root) as source:
            for index, row in enumerate(selected, start=1):
                try:
                    groups.append(convert_group(source, row, settings))
                except Exception as exc:
                    failures.append(
                        {
                            "group_id": row.group_id,
                            "split": row.split,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                if index % args.report_every == 0 or index == len(selected):
                    print(
                        f"[convert] {index}/{len(selected)}; ok={len(groups)}, failed={len(failures)}"
                    )
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(str(args.root.expanduser().resolve()), str(numba_cache_root)),
        ) as executor:
            future_to_row = {
                executor.submit(_convert_worker, task): row for task, row in zip(tasks, selected)
            }
            for index, future in enumerate(as_completed(future_to_row), start=1):
                row = future_to_row[future]
                try:
                    groups.append(future.result())
                except Exception as exc:
                    failures.append(
                        {
                            "group_id": row.group_id,
                            "split": row.split,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                if index % args.report_every == 0 or index == len(selected):
                    print(
                        f"[convert] {index}/{len(selected)}; ok={len(groups)}, failed={len(failures)}"
                    )

    split_order = {split: index for index, split in enumerate(SPLITS)}
    groups.sort(key=lambda group: (split_order[group["split"]], group["group_id"]))
    failures.sort(key=lambda row: (split_order[row["split"]], row["group_id"]))
    manifests = [sample for group in groups for sample in group["samples"]]
    for split in SPLITS:
        write_jsonl(
            output_root / "manifests" / f"{split}.jsonl",
            (row for row in manifests if row["split"] == split),
        )
    write_jsonl(output_root / "manifests" / "groups.jsonl", groups)
    report = _summary(
        source_root=args.root,
        output_root=output_root,
        selected=selected,
        groups=groups,
        failures=failures,
        settings=settings,
    )
    report["raw_inventory_audit"] = audit
    write_json(output_root / "reports" / "conversion_report.json", report)
    console_summary = {
        key: report[key]
        for key in (
            "selected_group_count",
            "converted_group_count",
            "dancer_sample_count",
            "group_counts_by_split",
            "sample_counts_by_split",
            "person_motion_hours",
            "unique_music_count",
            "unique_music_hours",
            "motion_music_length_mismatch_after_alignment_count",
            "trimmed_or_padded_group_count",
            "failed_or_skipped_count",
            "final_pass",
        )
    }
    print(json.dumps(console_summary, indent=2, ensure_ascii=False))
    print(f"[convert] full report: {output_root / 'reports' / 'conversion_report.json'}")
    if args.strict and not report["final_pass"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
