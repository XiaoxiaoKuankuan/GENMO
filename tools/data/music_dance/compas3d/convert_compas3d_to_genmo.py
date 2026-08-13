#!/usr/bin/env python3
"""Incrementally convert locally complete CoMPAS3D sequences to GENMO data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
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
from gem.utils.smplx_utils import make_smplx  # noqa: E402
from tools.data.music_dance.aioz.common import read_jsonl, safe_torch_load  # noqa: E402
from tools.data.music_dance.compas3d.common import (  # noqa: E402
    FORMAT_VERSION,
    SMPLX_POSE_LAYOUT,
    SOURCE_Z_UP_TO_GENMO_Y_UP,
    SPLITS,
    TARGET_FPS,
    SequenceFiles,
    build_splits,
    convert_source_motion,
    discover_local_sequences,
    ffprobe_media,
    inventory_audit,
    load_npz_fields,
    split_lookup,
    validate_canonical_motion,
    validate_source_motion,
    write_json,
    write_jsonl,
)


class PelvisResolver:
    """Compute shaped SMPL-X pelvis offsets and cache repeated participants."""

    def __init__(self) -> None:
        self.models: dict[str, Any] = {}
        self.cache: dict[str, torch.Tensor] = {}
        target_model = make_smplx(
            "supermotion", gender="neutral", num_betas=10, use_pca=False, flat_hand_mean=True
        ).eval()
        with torch.no_grad():
            self.target_pelvis = target_model.get_skeleton(torch.zeros(1, 10))[0, 0].cpu().float()

    def source_pelvis(self, gender: str, betas: np.ndarray) -> torch.Tensor:
        betas = np.asarray(betas, dtype=np.float32)
        digest = hashlib.sha256(betas.tobytes()).hexdigest()
        key = f"{gender}:{digest}"
        if key in self.cache:
            return self.cache[key]
        if gender not in self.models:
            self.models[gender] = make_smplx(
                "supermotion",
                gender=gender,
                num_betas=300,
                use_pca=False,
                flat_hand_mean=True,
            ).eval()
        with torch.no_grad():
            pelvis = self.models[gender].get_skeleton(torch.from_numpy(betas)[None])[0, 0]
        self.cache[key] = pelvis.detach().cpu().float()
        return self.cache[key]


def _configure_numba_cache(cache_root: Path, output_root: Path) -> Path:
    namespace = hashlib.sha256(str(output_root).encode()).hexdigest()[:12]
    cache = cache_root.expanduser().resolve() / namespace / f"process_{os.getpid()}"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(cache)
    return cache


def _prepare_output(output_root: Path) -> None:
    for relative in (
        "motions", "musicfeat_v2", "manifests", "reports", "renders", "audio_cache", ".staging"
    ):
        (output_root / relative).mkdir(parents=True, exist_ok=True)


def _extract_audio(mp4: Path, destination: Path, duration_sec: float, overwrite: bool) -> Path:
    if destination.is_file() and not overwrite:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.wav")
    temporary.unlink(missing_ok=True)
    command = [
        "ffmpeg", "-v", "error", "-y", "-i", str(mp4), "-map", "0:a:0", "-vn",
        "-t", f"{duration_sec:.9f}", "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _save_sequence_outputs(
    *,
    output_root: Path,
    sequence_id: str,
    payloads: list[dict[str, Any]],
    music: torch.Tensor,
    overwrite: bool,
) -> tuple[str, list[str]]:
    music_relative = f"musicfeat_v2/{sequence_id}_musicfeat_fps30.pt"
    motion_relatives = [f"motions/{payload['sample_id']}.pt" for payload in payloads]
    final_paths = [output_root / music_relative] + [output_root / path for path in motion_relatives]
    existing = [path for path in final_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"partial or unvalidated output exists for {sequence_id}; rerun with --overwrite: {existing[0]}"
        )
    staging = output_root / ".staging" / f"{sequence_id}.{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        staged_music = staging / Path(music_relative).name
        torch.save(music.detach().cpu().float().contiguous(), staged_music)
        staged_motion: list[Path] = []
        for payload, relative in zip(payloads, motion_relatives):
            path = staging / Path(relative).name
            torch.save(payload, path)
            staged_motion.append(path)
        for final in final_paths:
            final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged_music, final_paths[0])
        for staged, final in zip(staged_motion, final_paths[1:]):
            os.replace(staged, final)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return music_relative, motion_relatives


def _valid_existing_group(output_root: Path, sequence: SequenceFiles) -> dict[str, Any] | None:
    groups_path = output_root / "manifests" / "groups.jsonl"
    if not groups_path.is_file():
        return None
    rows = {str(row.get("sequence_id")): row for row in read_jsonl(groups_path)}
    group = rows.get(sequence.sequence_id)
    if group is None:
        return None
    try:
        samples = group["samples"]
        if len(samples) != 2 or {row["role"] for row in samples} != {"leader", "follower"}:
            return None
        music_paths = {row["music_feature_path"] for row in samples}
        if len(music_paths) != 1:
            return None
        music_path = output_root / next(iter(music_paths))
        music = safe_torch_load(music_path)
        validate_musicfeat_v2(music, source=music_path)
        for row in samples:
            payload = safe_torch_load(output_root / row["motion_path"])
            if payload.get("sequence_id") != sequence.sequence_id or payload.get("role") != row["role"]:
                return None
            canonical = {
                key: payload[key]
                for key in ("pose", "global_orient", "body_pose", "transl", "betas")
            }
            length = validate_canonical_motion(canonical, source=row["sample_id"])
            if length != len(music) or length != row["num_frames"]:
                return None
        return group
    except Exception:
        return None


def _audio_alignment(media: dict[str, Any], motion_frames: int, source_fps: float) -> float:
    return float(media["audio"]["duration_sec"]) * source_fps - motion_frames


def convert_sequence(
    *,
    sequence: SequenceFiles,
    split: str,
    output_root: Path,
    pelvis: PelvisResolver,
    target_fps: float,
    max_audio_leading_shortfall_frames: float,
    max_audio_tail_frames: float,
    max_feature_frame_mismatch: int,
    overwrite: bool,
) -> dict[str, Any]:
    source_fields: dict[str, dict[str, Any]] = {}
    source_info: dict[str, dict[str, Any]] = {}
    for role in ("leader", "follower"):
        path = sequence.role_path(role)
        fields = load_npz_fields(path)
        info = validate_source_motion(fields, source=str(path))
        source_fields[role] = fields
        source_info[role] = info
    if source_info["leader"]["num_frames"] != source_info["follower"]["num_frames"]:
        raise ValueError(
            f"{sequence.sequence_id}: leader/follower source lengths differ: "
            f"{source_info['leader']['num_frames']} vs {source_info['follower']['num_frames']}"
        )
    if not math.isclose(source_info["leader"]["fps"], source_info["follower"]["fps"]):
        raise ValueError(f"{sequence.sequence_id}: leader/follower FPS differ")
    source_fps = source_info["leader"]["fps"]
    source_frames = source_info["leader"]["num_frames"]
    media = ffprobe_media(sequence.mp4)
    audio_minus_motion = _audio_alignment(media, source_frames, source_fps)
    if audio_minus_motion < -max_audio_leading_shortfall_frames or audio_minus_motion > max_audio_tail_frames:
        raise ValueError(
            f"{sequence.sequence_id}: audio-motion mismatch={audio_minus_motion:.6f} source frames; "
            f"allowed=[-{max_audio_leading_shortfall_frames:g}, +{max_audio_tail_frames:g}]"
        )

    canonical_by_role: dict[str, dict[str, torch.Tensor]] = {}
    temporal_by_role: dict[str, dict[str, Any]] = {}
    for role in ("leader", "follower"):
        fields = source_fields[role]
        source_pelvis = pelvis.source_pelvis(source_info[role]["gender"], fields["betas"])
        canonical, temporal = convert_source_motion(
            fields,
            source_pelvis=source_pelvis,
            target_pelvis=pelvis.target_pelvis,
            target_fps=target_fps,
        )
        canonical_by_role[role] = canonical
        temporal_by_role[role] = temporal
    target_frames = validate_canonical_motion(
        canonical_by_role["leader"], source=f"{sequence.sequence_id}_leader"
    )
    if validate_canonical_motion(
        canonical_by_role["follower"], source=f"{sequence.sequence_id}_follower"
    ) != target_frames:
        raise RuntimeError(f"{sequence.sequence_id}: converted role lengths differ")

    audio_relative = f"audio_cache/{sequence.sequence_id}.wav"
    audio_path = _extract_audio(
        sequence.mp4,
        output_root / audio_relative,
        duration_sec=target_frames / target_fps,
        overwrite=overwrite,
    )
    raw_features, feature_metadata = extract_edge_baseline35(audio_path)
    raw_feature_frames = int(raw_features.shape[0])
    feature_difference = raw_feature_frames - target_frames
    if abs(feature_difference) > max_feature_frame_mismatch:
        raise ValueError(
            f"{sequence.sequence_id}: EDGE35-motion mismatch={feature_difference} frames "
            f"(music={raw_feature_frames}, motion={target_frames}), "
            f"limit={max_feature_frame_mismatch}"
        )
    features = align_features_to_length(raw_features, target_frames, "trim_or_pad_last")
    features = features.detach().cpu().float().contiguous()
    validate_musicfeat_v2(features, source=f"CoMPAS3D EDGE35 {sequence.sequence_id}")

    payloads: list[dict[str, Any]] = []
    for role in ("leader", "follower"):
        canonical = canonical_by_role[role]
        fields = source_fields[role]
        info = source_info[role]
        path = sequence.role_path(role)
        payloads.append(
            {
                "format": "compas3d_genmo_smpl_v1",
                "format_version": FORMAT_VERSION,
                "sample_id": f"{sequence.sequence_id}_{role}",
                "dataset": "compas3d",
                "sequence_id": sequence.sequence_id,
                "pair_id": sequence.parts.pair_id,
                "song_id": sequence.parts.song_id,
                "take_id": sequence.parts.take_id,
                "role": role,
                "fps": float(target_fps),
                "num_frames": target_frames,
                **canonical,
                "source_smplx": {
                    "source_npz": str(path.resolve()),
                    "source_relative_path": str(path.relative_to(sequence.directory.parents[1])),
                    "source_fps": info["fps"],
                    "source_num_frames": info["num_frames"],
                    "gender": info["gender"],
                    "surface_model_type": info["surface_model_type"],
                    "betas_300": torch.from_numpy(
                        np.ascontiguousarray(fields["betas"], dtype=np.float32)
                    ),
                    "pose_layout": SMPLX_POSE_LAYOUT,
                    "kept_pose_slice": "0:66 (global_orient 0:3 + body_pose 3:66)",
                    "discarded_pose_slices": "66:75 face/eyes, 75:165 left/right hands",
                    "joint_reordering_applied": False,
                    "translation_scale_applied": False,
                    "source_coordinate_system": "right_handed_z_up_metres",
                    "target_coordinate_system": "right_handed_y_up_metres",
                    "coordinate_rotation_matrix": SOURCE_Z_UP_TO_GENMO_Y_UP.clone(),
                    "pelvis_offset_compensation_applied": True,
                    "genmo_betas": "neutral zeros [T,10]; source 300D betas preserved above",
                    "temporal": temporal_by_role[role],
                },
            }
        )

    music_relative, motion_relatives = _save_sequence_outputs(
        output_root=output_root,
        sequence_id=sequence.sequence_id,
        payloads=payloads,
        music=features,
        overwrite=overwrite,
    )
    manifests: list[dict[str, Any]] = []
    for payload, relative in zip(payloads, motion_relatives):
        manifests.append(
            {
                "sample_id": payload["sample_id"],
                "dataset": "compas3d",
                "sequence_id": sequence.sequence_id,
                "pair_id": sequence.parts.pair_id,
                "song_id": sequence.parts.song_id,
                "take_id": sequence.parts.take_id,
                "role": payload["role"],
                "motion_path": relative,
                "music_feature_path": music_relative,
                "fps": float(target_fps),
                "num_frames": target_frames,
                "split": split,
            }
        )
    action = "exact" if feature_difference == 0 else (
        "trim_music_tail" if feature_difference > 0 else "pad_music_last_frame"
    )
    return {
        "sequence_id": sequence.sequence_id,
        "pair_id": sequence.parts.pair_id,
        "song_id": sequence.parts.song_id,
        "take_id": sequence.parts.take_id,
        "split": split,
        "source_mp4": str(sequence.mp4.resolve()),
        "source_npz": {
            "leader": str(sequence.leader.resolve()),
            "follower": str(sequence.follower.resolve()),
        },
        "source_fps": source_fps,
        "target_fps": float(target_fps),
        "source_motion_frames": source_frames,
        "target_motion_frames": target_frames,
        "duration_before_sec": source_frames / source_fps,
        "duration_after_sec": target_frames / target_fps,
        "leader_minus_follower_source_frames": 0,
        "media": media,
        "audio_minus_motion_source_frames": audio_minus_motion,
        "audio_alignment_action": "trim_positive_mp4_aac_tail_to_motion_duration",
        "audio_cache_path": audio_relative,
        "raw_music_feature_frames": raw_feature_frames,
        "music_minus_motion_before_feature_alignment": feature_difference,
        "feature_alignment_action": action,
        "final_music_feature_frames": len(features),
        "music_feature_path": music_relative,
        "feature_metadata": {
            key: value
            for key, value in feature_metadata.items()
            if key not in {"source_path", "feature_names"}
        },
        "samples": manifests,
    }


def _select_sequences(
    complete: list[SequenceFiles],
    requested: list[str],
    sample_sequences: int | None,
    seed: int,
) -> list[SequenceFiles]:
    by_id = {row.sequence_id: row for row in complete}
    if requested:
        unknown = sorted(set(requested) - set(by_id))
        if unknown:
            raise KeyError(f"requested sequences are not locally complete: {unknown}")
        return [by_id[value] for value in dict.fromkeys(requested)]
    if sample_sequences is None or sample_sequences >= len(complete):
        return complete
    if sample_sequences <= 0:
        raise ValueError("--sample-sequences must be positive")
    # Deterministic broad-coverage smoke selection: round-robin songs then Pairs.
    ordered = sorted(complete, key=lambda row: (row.parts.song_id, row.parts.pair_id, row.parts.take_id))
    chosen: list[SequenceFiles] = []
    for song in sorted({row.parts.song_id for row in ordered}):
        candidate = next(row for row in ordered if row.parts.song_id == song)
        if candidate not in chosen:
            chosen.append(candidate)
        if len(chosen) == sample_sequences:
            return chosen
    for row in sorted(ordered, key=lambda item: (item.parts.pair_id, item.parts.song_id, item.parts.take_id)):
        if row not in chosen and row.parts.pair_id not in {item.parts.pair_id for item in chosen}:
            chosen.append(row)
        if len(chosen) == sample_sequences:
            return chosen
    return chosen[:sample_sequences]


def _write_outputs(
    output_root: Path,
    groups: list[dict[str, Any]],
    split_report: dict[str, Any],
    inventory: dict[str, Any],
    settings: dict[str, Any],
    selected_count: int,
    failures: list[dict[str, str]],
    skipped_valid: list[str],
) -> dict[str, Any]:
    split_order = {name: index for index, name in enumerate(SPLITS)}
    groups.sort(key=lambda row: (split_order[row["split"]], row["sequence_id"]))
    manifests = [sample for group in groups for sample in group["samples"]]
    for split in SPLITS:
        write_jsonl(
            output_root / "manifests" / f"{split}.jsonl",
            (row for row in manifests if row["split"] == split),
        )
    write_jsonl(output_root / "manifests" / "groups.jsonl", groups)
    write_json(output_root / "reports" / "split_report.json", split_report)
    write_json(
        output_root / "reports" / "incomplete_sequences.json",
        {
            "format": "compas3d_incomplete_sequences_v1",
            "count": inventory["incomplete_sequence_count"],
            "sequences": inventory["incomplete_sequences"],
            "blocking_for_current_complete_subset": False,
        },
    )
    role_frames = Counter()
    for sample in manifests:
        role_frames[sample["role"]] += int(sample["num_frames"])
    unique_music_frames = sum(int(group["target_motion_frames"]) for group in groups)
    person_frames = sum(int(sample["num_frames"]) for sample in manifests)
    alignment = [float(group["audio_minus_motion_source_frames"]) for group in groups]
    feature_alignment = [int(group["music_minus_motion_before_feature_alignment"]) for group in groups]
    report = {
        "format": "compas3d_genmo_conversion_report_v1",
        "source_root": inventory["raw_root"],
        "output_root": str(output_root.resolve()),
        "settings": settings,
        "raw_complete_sequence_count": inventory["complete_sequence_count"],
        "raw_incomplete_sequence_count": inventory["incomplete_sequence_count"],
        "selected_sequence_count": selected_count,
        "converted_sequence_count": len(groups),
        "newly_converted_sequence_count": len(groups) - len(skipped_valid),
        "skipped_already_valid_sequence_count": len(skipped_valid),
        "skipped_already_valid_sequences": sorted(skipped_valid),
        "dancer_sample_count": len(manifests),
        "sequence_counts_by_split": {
            split: sum(group["split"] == split for group in groups) for split in SPLITS
        },
        "sample_counts_by_split": {
            split: sum(sample["split"] == split for sample in manifests) for split in SPLITS
        },
        "person_motion_frames": person_frames,
        "person_motion_hours": person_frames / TARGET_FPS / 3600,
        "leader_frames": role_frames["leader"],
        "leader_hours": role_frames["leader"] / TARGET_FPS / 3600,
        "follower_frames": role_frames["follower"],
        "follower_hours": role_frames["follower"] / TARGET_FPS / 3600,
        "unique_music_count": len(groups),
        "unique_music_frames": unique_music_frames,
        "unique_music_hours": unique_music_frames / TARGET_FPS / 3600,
        "audio_minus_motion_source_frames": {
            "min": min(alignment) if alignment else None,
            "max": max(alignment) if alignment else None,
            "mean": sum(alignment) / len(alignment) if alignment else None,
        },
        "feature_minus_motion_before_alignment_histogram": {
            str(value): feature_alignment.count(value) for value in sorted(set(feature_alignment))
        },
        "motion_music_length_mismatch_after_alignment_count": sum(
            group["target_motion_frames"] != group["final_music_feature_frames"] for group in groups
        ),
        "leader_follower_length_mismatch_count": sum(
            group["leader_minus_follower_source_frames"] != 0 for group in groups
        ),
        "failed_or_skipped_count": len(failures),
        "failed_or_skipped": failures,
        "incomplete_sequences_report": "reports/incomplete_sequences.json",
        "final_pass_for_selected_complete_subset": len(groups) == selected_count and not failures,
    }
    write_json(output_root / "reports" / "conversion_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sequence-id", action="append", default=[])
    parser.add_argument("--sample-sequences", type=int)
    parser.add_argument("--split-strategy", choices=("music_identity", "official_interaction"), default="music_identity")
    parser.add_argument("--target-fps", type=float, default=TARGET_FPS)
    parser.add_argument("--max-audio-leading-shortfall-frames", type=float, default=2.0)
    parser.add_argument("--max-audio-tail-frames", type=float, default=6.0)
    parser.add_argument("--max-feature-frame-mismatch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--report-every", type=int, default=5)
    parser.add_argument("--numba-cache-root", type=Path, default=Path("/tmp/genmo_compas3d_numba_cache"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not math.isclose(args.target_fps, EDGE_TARGET_FPS):
        raise ValueError(f"EDGE baseline35 is fixed at {EDGE_TARGET_FPS} FPS")
    if args.report_every <= 0 or min(
        args.max_audio_leading_shortfall_frames,
        args.max_audio_tail_frames,
        args.max_feature_frame_mismatch,
    ) < 0:
        raise ValueError("report interval and alignment limits must be non-negative/positive")
    raw_root = args.root.expanduser().resolve()
    reference_root = args.reference_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    _prepare_output(output_root)
    numba_cache = _configure_numba_cache(args.numba_cache_root, output_root)
    complete, _ = discover_local_sequences(raw_root)
    inventory = inventory_audit(raw_root, reference_root)
    splits, split_report = build_splits(complete, args.split_strategy)
    lookup = split_lookup(splits)
    selected = _select_sequences(complete, args.sequence_id, args.sample_sequences, args.seed)
    selected_ids = {row.sequence_id for row in selected}
    # Incremental runs preserve only valid existing rows that remain selected.
    groups: list[dict[str, Any]] = []
    skipped_valid: list[str] = []
    pending: list[SequenceFiles] = []
    if not args.overwrite:
        for sequence in selected:
            existing = _valid_existing_group(output_root, sequence)
            if existing is None:
                pending.append(sequence)
            else:
                existing["split"] = lookup[sequence.sequence_id]
                for sample in existing["samples"]:
                    sample["split"] = lookup[sequence.sequence_id]
                groups.append(existing)
                skipped_valid.append(sequence.sequence_id)
    else:
        pending = selected
    pelvis = PelvisResolver() if pending else None
    failures: list[dict[str, str]] = []
    for index, sequence in enumerate(pending, start=1):
        try:
            assert pelvis is not None
            groups.append(
                convert_sequence(
                    sequence=sequence,
                    split=lookup[sequence.sequence_id],
                    output_root=output_root,
                    pelvis=pelvis,
                    target_fps=float(args.target_fps),
                    max_audio_leading_shortfall_frames=float(args.max_audio_leading_shortfall_frames),
                    max_audio_tail_frames=float(args.max_audio_tail_frames),
                    max_feature_frame_mismatch=int(args.max_feature_frame_mismatch),
                    overwrite=bool(args.overwrite),
                )
            )
        except Exception as exc:
            failures.append(
                {
                    "sequence_id": sequence.sequence_id,
                    "split": lookup[sequence.sequence_id],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if index % args.report_every == 0 or index == len(pending):
            print(
                f"[convert] pending {index}/{len(pending)}; total_ok={len(groups)}, "
                f"already_valid={len(skipped_valid)}, failed={len(failures)}"
            )
    settings = {
        "root": str(raw_root),
        "reference_root": str(reference_root),
        "output_root": str(output_root),
        "selected_sequence_ids": sorted(selected_ids),
        "sample_sequences": args.sample_sequences,
        "split_strategy": args.split_strategy,
        "target_fps": float(args.target_fps),
        "max_audio_leading_shortfall_frames": args.max_audio_leading_shortfall_frames,
        "max_audio_tail_frames": args.max_audio_tail_frames,
        "max_feature_frame_mismatch": args.max_feature_frame_mismatch,
        "overwrite": bool(args.overwrite),
        "numba_cache": str(numba_cache),
    }
    report = _write_outputs(
        output_root,
        groups,
        split_report,
        inventory,
        settings,
        selected_count=len(selected),
        failures=failures,
        skipped_valid=skipped_valid,
    )
    summary_keys = (
        "raw_complete_sequence_count", "raw_incomplete_sequence_count", "selected_sequence_count",
        "converted_sequence_count", "newly_converted_sequence_count",
        "skipped_already_valid_sequence_count", "dancer_sample_count", "sequence_counts_by_split",
        "person_motion_hours", "unique_music_hours", "failed_or_skipped_count",
        "final_pass_for_selected_complete_subset",
    )
    print(json.dumps({key: report[key] for key in summary_keys}, ensure_ascii=False, indent=2))
    print(f"[convert] report: {output_root / 'reports' / 'conversion_report.json'}")
    return 0 if report["final_pass_for_selected_complete_subset"] or not args.strict else 2


if __name__ == "__main__":
    raise SystemExit(main())
