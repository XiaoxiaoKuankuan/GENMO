#!/usr/bin/env python3
"""Strictly validate canonical FineDance GENMO output and optionally render it."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gem.datasets.aistpp.aistplusplus import validate_musicfeat_v2  # noqa: E402
from tools.data.music_dance.aioz.common import read_jsonl, safe_torch_load  # noqa: E402
from tools.data.music_dance.finedance.common import (  # noqa: E402
    FORMAT_VERSION,
    SPLITS,
    validate_canonical_motion,
    write_json,
)


def _resolve(root: Path, relative: Any, field: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{field} must be a non-empty relative path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field} escapes output root: {relative}") from exc
    return path


def _load_manifests(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        path = root / "manifests" / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"missing manifest: {path}")
        values = read_jsonl(path)
        if any(row.get("split") != split for row in values):
            raise ValueError(f"{path}: row split differs from manifest name")
        rows.extend(values)
    if not rows:
        raise ValueError("all FineDance manifests are empty")
    return rows


def validate_dataset(root: Path, *, smpl_forward_samples: int, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifests = _load_manifests(root)
    errors: list[dict[str, str]] = []
    seen: set[str] = set()
    split_ids = {split: set() for split in SPLITS}
    motion_paths: set[str] = set()
    music_paths: set[str] = set()
    lengths: dict[str, int] = {}
    payload_cache: dict[str, dict[str, Any]] = {}
    for row in manifests:
        sample_id = str(row.get("sample_id", "<unknown>"))
        try:
            required = {
                "sample_id", "dataset", "motion_path", "music_feature_path", "fps",
                "num_frames", "split", "song_name", "coarse_style", "fine_style",
            }
            missing = required - set(row)
            if missing:
                raise ValueError(f"manifest fields missing: {sorted(missing)}")
            if sample_id in seen:
                raise ValueError(f"duplicate sample_id: {sample_id}")
            seen.add(sample_id)
            if row["dataset"] != "finedance" or row["split"] not in SPLITS:
                raise ValueError("invalid dataset/split")
            if not math.isclose(float(row["fps"]), 30.0):
                raise ValueError(f"fps must be 30, got {row['fps']}")
            split_ids[row["split"]].add(sample_id)
            motion_path = _resolve(root, row["motion_path"], "motion_path")
            music_path = _resolve(root, row["music_feature_path"], "music_feature_path")
            if not motion_path.is_file() or not music_path.is_file():
                raise FileNotFoundError(f"missing motion/music for {sample_id}")
            payload = safe_torch_load(motion_path)
            if not isinstance(payload, dict) or payload.get("format") != "finedance_genmo_smpl_v1":
                raise ValueError("unsupported motion payload")
            if payload.get("format_version") != FORMAT_VERSION:
                raise ValueError("wrong format_version")
            if payload.get("sample_id") != sample_id or payload.get("num_frames") != row["num_frames"]:
                raise ValueError("payload/manifest identity or length mismatch")
            tensors = {key: payload[key] for key in ("pose", "transl", "betas")}
            length = validate_canonical_motion(tensors, source=sample_id)
            if payload["global_orient"].shape != (length, 3) or payload["body_pose"].shape != (length, 63):
                raise ValueError("global_orient/body_pose has wrong shape")
            torch.testing.assert_close(payload["global_orient"], payload["pose"][:, :3])
            torch.testing.assert_close(payload["body_pose"], payload["pose"][:, 3:66])
            if not torch.equal(payload["betas"], torch.zeros_like(payload["betas"])):
                raise ValueError("FineDance neutral betas must be exact zeros")
            source = payload.get("source", {})
            expected_source = {
                "source_feature_dim": 315,
                "source_layout": "translation_3_plus_52x_rotation6d",
                "rotation_conversion": "rotation6d_to_matrix_to_axis_angle",
                "kept_joint_slice": "0:22",
                "joint_reordering_applied": False,
                "coordinate_transform_applied": False,
                "translation_scale_applied": False,
            }
            if any(source.get(key) != value for key, value in expected_source.items()):
                raise ValueError(f"invalid source provenance: {source}")
            music = safe_torch_load(music_path)
            if not isinstance(music, torch.Tensor):
                raise ValueError("music feature must be a tensor")
            music = music.detach().cpu().float()
            validate_musicfeat_v2(music, source=music_path)
            if length != int(music.shape[0]) or length != int(row["num_frames"]):
                raise ValueError(f"motion_T={length}, music_T={len(music)}, manifest={row['num_frames']}")
            lengths[sample_id] = length
            payload_cache[sample_id] = payload
            motion_paths.add(row["motion_path"])
            music_paths.add(row["music_feature_path"])
        except Exception as exc:
            errors.append({"sample_id": sample_id, "error": f"{type(exc).__name__}: {exc}"})

    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            leakage = sorted(split_ids[left] & split_ids[right])
            if leakage:
                errors.append({"sample_id": "<split>", "error": f"{left}/{right} leakage: {leakage}"})
    disk_motion = {path.relative_to(root).as_posix() for path in (root / "motions").glob("*.pt")}
    disk_music = {path.relative_to(root).as_posix() for path in (root / "musicfeat_v2").glob("*.pt")}
    if disk_motion != motion_paths or disk_music != music_paths:
        errors.append(
            {
                "sample_id": "<inventory>",
                "error": (
                    f"motion disk/manifest={len(disk_motion)}/{len(motion_paths)}, "
                    f"music disk/manifest={len(disk_music)}/{len(music_paths)}"
                ),
            }
        )

    forward_results: list[dict[str, Any]] = []
    if smpl_forward_samples:
        from gem.utils.smplx_utils import make_smplx

        candidates = [row for row in manifests if row["sample_id"] in payload_cache]
        chosen = random.Random(seed).sample(candidates, min(smpl_forward_samples, len(candidates)))
        model = make_smplx("supermotion", use_pca=False, flat_hand_mean=True).eval()
        with torch.no_grad():
            for row in chosen:
                payload = payload_cache[row["sample_id"]]
                frame_count = int(payload["num_frames"])
                indices = torch.linspace(0, frame_count - 1, min(16, frame_count)).long()
                output = model(
                    global_orient=payload["global_orient"][indices],
                    body_pose=payload["body_pose"][indices],
                    transl=payload["transl"][indices],
                    betas=payload["betas"][indices],
                )
                finite = bool(torch.isfinite(output.vertices).all() and torch.isfinite(output.joints).all())
                if not finite:
                    errors.append({"sample_id": row["sample_id"], "error": "SMPL forward is non-finite"})
                forward_results.append(
                    {
                        "sample_id": row["sample_id"],
                        "checked_frames": len(indices),
                        "vertices_shape": list(output.vertices.shape),
                        "joints_shape": list(output.joints.shape),
                        "finite": finite,
                    }
                )
    conversion_report_path = root / "reports" / "conversion_report.json"
    conversion_report = json.loads(conversion_report_path.read_text()) if conversion_report_path.is_file() else {}
    total_frames = sum(lengths.values())
    report = {
        "format": "finedance_genmo_validation_report_v1",
        "root": str(root.resolve()),
        "sample_count": len(manifests),
        "sample_counts_by_split": {split: len(split_ids[split]) for split in SPLITS},
        "total_frames": total_frames,
        "total_hours": total_frames / 30 / 3600,
        "motion_music_length_mismatch_count": sum(
            "motion_T=" in error["error"] for error in errors
        ),
        "smpl_forward_results": forward_results,
        "trimmed_or_padded_samples": conversion_report.get("trimmed_or_padded_samples", []),
        "failed_or_skipped": conversion_report.get("failed_or_skipped", []),
        "validation_error_count": len(errors),
        "validation_errors": errors,
        "final_pass": not errors,
    }
    return report, manifests


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

    payload = safe_torch_load(_resolve(root, manifest["motion_path"], "motion_path"))
    total = int(payload["num_frames"])
    stride = max(1, math.ceil(total / max_frames))
    indices = torch.arange(0, total, stride)
    model = make_smplx("supermotion", use_pca=False, flat_hand_mean=True).eval()
    vertices: list[torch.Tensor] = []
    with torch.no_grad():
        for chunk in indices.split(64):
            out = model(
                global_orient=payload["global_orient"][chunk],
                body_pose=payload["body_pose"][chunk],
                transl=payload["transl"][chunk],
                betas=payload["betas"][chunk],
            )
            vertices.append(out.vertices.detach().cpu().float())
    verts = torch.cat(vertices)
    if not torch.isfinite(verts).all():
        raise ValueError("SMPL render vertices contain NaN/Inf")
    # Visualization-only height alignment; saved translation is untouched.
    vertical_offset = float(verts[..., 1].min())
    verts[..., 1] -= vertical_offset
    faces = torch.from_numpy(np.asarray(model.faces, dtype=np.int64)).long()
    frames = render_global_frames(verts, faces, width, height)
    render_dir.mkdir(parents=True, exist_ok=True)
    output_path = render_dir / f"{manifest['sample_id']}.mp4"
    save_video(frames, str(output_path), fps=Fraction(30, stride))
    return {
        "sample_id": manifest["sample_id"],
        "render_path": str(output_path.resolve()),
        "source_frames": total,
        "rendered_frames": len(indices),
        "stride": stride,
        "render_fps": 30 / stride,
        "visualization_only_vertical_offset": vertical_offset,
        "finite": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--smpl-forward-samples", type=int, default=3)
    parser.add_argument("--render-samples", type=int, default=0)
    parser.add_argument("--render-dir", type=Path)
    parser.add_argument("--render-width", type=int, default=512)
    parser.add_argument("--render-height", type=int, default=512)
    parser.add_argument("--max-render-frames", type=int, default=180)
    parser.add_argument("--seed", type=int, default=20260813)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if min(args.smpl_forward_samples, args.render_samples) < 0:
        raise ValueError("sample counts must be non-negative")
    root = args.root.expanduser().resolve()
    report, manifests = validate_dataset(
        root, smpl_forward_samples=args.smpl_forward_samples, seed=args.seed
    )
    render_errors: list[dict[str, str]] = []
    renders: list[dict[str, Any]] = []
    if args.render_samples:
        render_dir = (
            args.render_dir.expanduser().resolve()
            if args.render_dir is not None
            else root / "reports" / "renders"
        )
        chosen = random.Random(args.seed).sample(manifests, min(args.render_samples, len(manifests)))
        for row in chosen:
            try:
                result = render_sample(
                    root,
                    row,
                    render_dir,
                    width=args.render_width,
                    height=args.render_height,
                    max_frames=args.max_render_frames,
                )
                renders.append(result)
                print(f"[render] {result['sample_id']} -> {result['render_path']}")
            except Exception as exc:
                render_errors.append(
                    {"sample_id": row["sample_id"], "error": f"{type(exc).__name__}: {exc}"}
                )
    report["renders"] = renders
    report["render_errors"] = render_errors
    report["final_pass"] = report["final_pass"] and not render_errors
    report_path = root / "reports" / "validation_report.json"
    write_json(report_path, report)
    print(
        json.dumps(
            {key: report[key] for key in (
                "sample_count", "sample_counts_by_split", "total_hours",
                "motion_music_length_mismatch_count", "validation_error_count", "final_pass"
            )},
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"[validate] report: {report_path}")
    return 0 if report["final_pass"] or not args.strict else 2


if __name__ == "__main__":
    raise SystemExit(main())
