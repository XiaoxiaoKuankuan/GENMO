#!/usr/bin/env python3
"""Validate canonical AIOZ-to-GENMO outputs and optionally render samples."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gem.datasets.aistpp.aistplusplus import validate_musicfeat_v2  # noqa: E402
from tools.data.music_dance.aioz.common import (  # noqa: E402
    FORMAT_VERSION,
    SPLITS,
    read_jsonl,
    safe_torch_load,
    validate_canonical_smpl,
    write_json,
)


def _resolve_relative(root: Path, relative: Any, field: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{field} must be a non-empty relative path")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field} escapes output root: {relative}") from exc
    return candidate


def validate_motion_file(path: Path, manifest: dict[str, Any]) -> int:
    payload = safe_torch_load(path)
    if not isinstance(payload, dict) or payload.get("format") != "aioz_genmo_smpl_v1":
        raise ValueError(f"{path}: unsupported canonical payload")
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{path}: format_version is not {FORMAT_VERSION}")
    for field in ("sample_id", "group_id", "person_id", "num_frames", "fps"):
        if payload.get(field) != manifest[field]:
            raise ValueError(
                f"{path}: payload {field}={payload.get(field)!r} != manifest {manifest[field]!r}"
            )
    motion = {key: payload[key] for key in ("global_orient", "body_pose", "transl", "betas")}
    length = validate_canonical_smpl(motion, source=path)
    if length != manifest["num_frames"]:
        raise ValueError(f"{path}: tensor T={length} != manifest num_frames")
    source = payload.get("source")
    if not isinstance(source, dict):
        raise ValueError(f"{path}: missing source provenance")
    expected_provenance = {
        "source_pose_dim": 72,
        "kept_pose_slice": "0:66",
        "discarded_pose_slice": "66:72",
        "joint_reordering_applied": False,
        "coordinate_transform_applied": False,
        "translation_scale_applied": False,
    }
    for key, value in expected_provenance.items():
        if source.get(key) != value:
            raise ValueError(f"{path}: invalid source provenance {key}={source.get(key)!r}")
    return length


def _load_manifests(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        path = root / "manifests" / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"missing split manifest: {path}")
        split_rows = read_jsonl(path)
        for row in split_rows:
            if row.get("split") != split:
                raise ValueError(f"{path}: row declares split={row.get('split')!r}")
        rows.extend(split_rows)
    if not rows:
        raise ValueError("all AIOZ sample manifests are empty")
    return rows


def validate_dataset(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    required_fields = {
        "sample_id",
        "group_id",
        "person_id",
        "motion_path",
        "music_feature_path",
        "num_frames",
        "fps",
        "split",
        "music_genre",
        "dance_style",
    }
    manifests = _load_manifests(root)
    sample_ids: set[str] = set()
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    errors: list[dict[str, str]] = []
    music_cache: dict[str, torch.Tensor] = {}
    mismatch_count = 0
    for row in manifests:
        try:
            missing = required_fields - set(row)
            if missing:
                raise ValueError(f"manifest row is missing {sorted(missing)}")
            sample_id = row["sample_id"]
            if sample_id in sample_ids:
                raise ValueError(f"duplicate sample_id: {sample_id}")
            sample_ids.add(sample_id)
            if row["split"] not in SPLITS:
                raise ValueError(f"invalid split: {row['split']}")
            if not isinstance(row["person_id"], int) or row["person_id"] < 0:
                raise ValueError(f"invalid person_id: {row['person_id']}")
            if not isinstance(row["num_frames"], int) or row["num_frames"] <= 0:
                raise ValueError(f"invalid num_frames: {row['num_frames']}")
            if not math.isclose(float(row["fps"]), 30.0):
                raise ValueError(f"canonical fps must be 30, got {row['fps']}")
            motion_path = _resolve_relative(root, row["motion_path"], "motion_path")
            music_path = _resolve_relative(root, row["music_feature_path"], "music_feature_path")
            if not motion_path.is_file():
                raise FileNotFoundError(f"missing motion file: {motion_path}")
            if not music_path.is_file():
                raise FileNotFoundError(f"missing music feature file: {music_path}")
            motion_length = validate_motion_file(motion_path, row)
            if row["music_feature_path"] not in music_cache:
                features = safe_torch_load(music_path)
                if not isinstance(features, torch.Tensor):
                    raise ValueError(f"{music_path}: music feature payload must be a Tensor")
                features = features.detach().cpu().float()
                validate_musicfeat_v2(features, source=music_path)
                music_cache[row["music_feature_path"]] = features
            music_length = int(music_cache[row["music_feature_path"]].shape[0])
            if motion_length != music_length:
                mismatch_count += 1
                raise ValueError(f"{sample_id}: motion_T={motion_length} != music_T={music_length}")
            groups[row["group_id"]].append(row)
        except Exception as exc:
            errors.append(
                {
                    "sample_id": str(row.get("sample_id", "<unknown>")),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    group_manifest_path = root / "manifests" / "groups.jsonl"
    group_rows = read_jsonl(group_manifest_path) if group_manifest_path.is_file() else []
    group_metadata = {row["group_id"]: row for row in group_rows}
    split_group_counts = Counter()
    split_sample_counts = Counter(row["split"] for row in manifests)
    music_reference_counts = Counter(row["music_feature_path"] for row in manifests)
    shared_feature_groups = 0
    for group_id, rows in groups.items():
        try:
            splits = {row["split"] for row in rows}
            if len(splits) != 1:
                raise ValueError(f"group split leakage: {group_id} -> {sorted(splits)}")
            split_group_counts[next(iter(splits))] += 1
            paths = {row["music_feature_path"] for row in rows}
            if len(paths) != 1:
                raise ValueError(f"group has multiple music feature files: {group_id} -> {paths}")
            person_ids = sorted(row["person_id"] for row in rows)
            if person_ids != list(range(len(rows))):
                raise ValueError(f"group dancer IDs are not contiguous: {group_id} -> {person_ids}")
            if len(rows) > 1:
                shared_feature_groups += 1
                only_path = next(iter(paths))
                if music_reference_counts[only_path] != len(rows):
                    raise ValueError(f"music feature is not shared by all dancers: {group_id}")
            if group_id in group_metadata:
                expected_people = group_metadata[group_id].get("num_persons")
                if expected_people != len(rows):
                    raise ValueError(
                        f"group manifest num_persons={expected_people}, samples={len(rows)}: {group_id}"
                    )
        except Exception as exc:
            errors.append({"sample_id": group_id, "error": f"{type(exc).__name__}: {exc}"})

    manifest_motion_paths = {row["motion_path"] for row in manifests}
    manifest_music_paths = {row["music_feature_path"] for row in manifests}
    disk_motion_paths = {
        path.relative_to(root).as_posix() for path in (root / "motions").glob("*.pt")
    }
    disk_music_paths = {
        path.relative_to(root).as_posix() for path in (root / "musicfeat_v2").glob("*.pt")
    }
    orphan_motion = sorted(disk_motion_paths - manifest_motion_paths)
    missing_manifest_motion = sorted(manifest_motion_paths - disk_motion_paths)
    orphan_music = sorted(disk_music_paths - manifest_music_paths)
    missing_manifest_music = sorted(manifest_music_paths - disk_music_paths)
    if orphan_motion or missing_manifest_motion or orphan_music or missing_manifest_music:
        errors.append(
            {
                "sample_id": "<inventory>",
                "error": (
                    f"motion orphan/missing={len(orphan_motion)}/{len(missing_manifest_motion)}, "
                    f"music orphan/missing={len(orphan_music)}/{len(missing_manifest_music)}"
                ),
            }
        )

    person_frames = sum(int(row["num_frames"]) for row in manifests)
    unique_music_frames = sum(int(features.shape[0]) for features in music_cache.values())
    conversion_report_path = root / "reports" / "conversion_report.json"
    conversion_report = (
        json.loads(conversion_report_path.read_text(encoding="utf-8"))
        if conversion_report_path.is_file()
        else {}
    )
    report: dict[str, Any] = {
        "format": "aioz_genmo_validation_report_v1",
        "output_root": str(root),
        "group_sequence_count": len(groups),
        "dancer_sample_count": len(manifests),
        "group_counts_by_split": {split: split_group_counts[split] for split in SPLITS},
        "sample_counts_by_split": {split: split_sample_counts[split] for split in SPLITS},
        "person_motion_frames": person_frames,
        "person_motion_hours": person_frames / 30 / 3600,
        "unique_music_count": len(music_cache),
        "unique_music_frames": unique_music_frames,
        "unique_music_hours": unique_music_frames / 30 / 3600,
        "motion_music_length_mismatch_count": mismatch_count,
        "groups_with_shared_music_reference": shared_feature_groups,
        "trimmed_or_padded_groups": conversion_report.get("trimmed_or_padded_groups", []),
        "failed_or_skipped": conversion_report.get("failed_or_skipped", []),
        "orphan_motion_files": orphan_motion,
        "missing_manifest_motion_files": missing_manifest_motion,
        "orphan_music_files": orphan_music,
        "missing_manifest_music_files": missing_manifest_music,
        "validation_error_count": len(errors),
        "validation_errors": errors,
        "final_pass": not errors and mismatch_count == 0,
    }
    return report, manifests


def _choose_render_samples(
    manifests: list[dict[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    by_group: dict[str, dict[str, Any]] = {}
    for row in manifests:
        by_group.setdefault(row["group_id"], row)
    values = list(by_group.values())
    if count > len(values):
        raise ValueError(f"requested {count} renders but only {len(values)} groups exist")
    return random.Random(seed).sample(values, count)


def render_sample(
    root: Path,
    manifest: dict[str, Any],
    render_dir: Path,
    *,
    width: int,
    height: int,
    max_frames: int,
) -> dict[str, Any]:
    from gem.utils.smplx_utils import make_smplx
    from gem.utils.video_io_utils import save_video
    from scripts.demo.demo_utils import render_global_frames

    payload = safe_torch_load(_resolve_relative(root, manifest["motion_path"], "motion_path"))
    total_frames = int(payload["num_frames"])
    stride = max(1, math.ceil(total_frames / max_frames))
    indices = torch.arange(0, total_frames, stride)
    body_model = make_smplx("supermotion").eval()
    vertices: list[torch.Tensor] = []
    with torch.no_grad():
        for chunk in indices.split(64):
            output = body_model(
                global_orient=payload["global_orient"][chunk],
                body_pose=payload["body_pose"][chunk],
                transl=payload["transl"][chunk],
                betas=payload["betas"][chunk],
            )
            vertices.append(output.vertices.detach().cpu().float())
    verts = torch.cat(vertices)
    if not torch.isfinite(verts).all():
        raise ValueError(f"{manifest['sample_id']}: SMPL forward produced NaN or Inf")
    # AIOZ root translations are preserved exactly in the saved motion. For
    # visualization only, apply one constant vertical offset so the lowest
    # vertex touches the renderer's Y=0 ground; X/Z trajectory is untouched.
    vertical_offset = float(verts[..., 1].min())
    verts[..., 1] -= vertical_offset
    faces = torch.from_numpy(np.asarray(body_model.faces, dtype=np.int64)).long()
    frames = render_global_frames(verts, faces, width, height)
    render_dir.mkdir(parents=True, exist_ok=True)
    output_path = render_dir / f"{manifest['sample_id']}.mp4"
    save_video(frames, str(output_path), fps=Fraction(30, stride))
    return {
        "sample_id": manifest["sample_id"],
        "group_id": manifest["group_id"],
        "source_motion": manifest["motion_path"],
        "render_path": str(output_path.resolve()),
        "source_frames": total_frames,
        "rendered_frames": len(indices),
        "frame_stride": stride,
        "render_fps": 30 / stride,
        "visualization_only_vertical_offset": vertical_offset,
        "root_translation_min": payload["transl"].min(dim=0).values.tolist(),
        "root_translation_max": payload["transl"].max(dim=0).values.tolist(),
        "finite": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Converted output root")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--render-samples", type=int, default=0)
    parser.add_argument("--render-dir", type=Path)
    parser.add_argument("--render-width", type=int, default=512)
    parser.add_argument("--render-height", type=int, default=512)
    parser.add_argument("--max-render-frames", type=int, default=180)
    parser.add_argument("--seed", type=int, default=20260813)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.render_samples < 0:
        raise ValueError("--render-samples must be non-negative")
    if args.render_width <= 0 or args.render_height <= 0 or args.max_render_frames <= 0:
        raise ValueError("render dimensions/frame limit must be positive")
    root = args.root.expanduser().resolve()
    report, manifests = validate_dataset(root)
    render_results: list[dict[str, Any]] = []
    render_errors: list[dict[str, str]] = []
    if args.render_samples:
        render_dir = (
            args.render_dir.expanduser().resolve()
            if args.render_dir is not None
            else root / "reports" / "renders"
        )
        for manifest in _choose_render_samples(manifests, args.render_samples, args.seed):
            try:
                result = render_sample(
                    root,
                    manifest,
                    render_dir,
                    width=args.render_width,
                    height=args.render_height,
                    max_frames=args.max_render_frames,
                )
                render_results.append(result)
                print(f"[render] {result['sample_id']} -> {result['render_path']}")
            except Exception as exc:
                render_errors.append(
                    {
                        "sample_id": manifest["sample_id"],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    report["requested_render_count"] = args.render_samples
    report["successful_render_count"] = len(render_results)
    report["renders"] = render_results
    report["render_errors"] = render_errors
    report["final_pass"] = report["final_pass"] and not render_errors
    report_path = root / "reports" / "validation_report.json"
    write_json(report_path, report)
    console_summary = {
        key: report[key]
        for key in (
            "group_sequence_count",
            "dancer_sample_count",
            "group_counts_by_split",
            "sample_counts_by_split",
            "person_motion_hours",
            "unique_music_count",
            "unique_music_hours",
            "motion_music_length_mismatch_count",
            "groups_with_shared_music_reference",
            "validation_error_count",
            "successful_render_count",
            "final_pass",
        )
    }
    print(json.dumps(console_summary, indent=2, ensure_ascii=False))
    print(f"[validate] report: {report_path}")
    return 0 if report["final_pass"] or not args.strict else 2


if __name__ == "__main__":
    raise SystemExit(main())
