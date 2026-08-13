#!/usr/bin/env python3
"""Strictly validate canonical CoMPAS3D GENMO output and optionally render clips."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gem.datasets.aistpp.aistplusplus import validate_musicfeat_v2  # noqa: E402
from gem.utils.smplx_utils import make_smplx  # noqa: E402
from tools.data.music_dance.aioz.common import read_jsonl, safe_torch_load  # noqa: E402
from tools.data.music_dance.compas3d.common import (  # noqa: E402
    FORMAT_VERSION,
    SMPLX_POSE_LAYOUT,
    SOURCE_Z_UP_TO_GENMO_Y_UP,
    SPLITS,
    TARGET_FPS,
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
            raise ValueError(f"{path}: manifest row split differs from filename")
        rows.extend(values)
    if not rows:
        raise ValueError("all CoMPAS3D manifests are empty")
    return rows


def _source_exists(path: Any) -> bool:
    return isinstance(path, str) and Path(path).is_file()


def validate_dataset(
    root: Path,
    *,
    smpl_forward_samples: int,
    seed: int,
    require_source_files: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifests = _load_manifests(root)
    conversion_path = root / "reports" / "conversion_report.json"
    conversion = json.loads(conversion_path.read_text()) if conversion_path.is_file() else {}
    split_path = root / "reports" / "split_report.json"
    split_report = json.loads(split_path.read_text()) if split_path.is_file() else {}
    errors: list[dict[str, str]] = []
    seen: set[str] = set()
    split_samples = {split: set() for split in SPLITS}
    split_sequences = {split: set() for split in SPLITS}
    sequence_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    motion_paths: set[str] = set()
    music_paths: set[str] = set()
    payload_cache: dict[str, dict[str, Any]] = {}
    music_cache: dict[str, torch.Tensor] = {}
    lengths: dict[str, int] = {}
    roles = Counter()

    required = {
        "sample_id", "dataset", "sequence_id", "pair_id", "song_id", "take_id", "role",
        "motion_path", "music_feature_path", "fps", "num_frames", "split",
    }
    for row in manifests:
        sample_id = str(row.get("sample_id", "<unknown>"))
        try:
            missing = required - set(row)
            if missing:
                raise ValueError(f"manifest fields missing: {sorted(missing)}")
            if sample_id in seen:
                raise ValueError(f"duplicate sample_id={sample_id}")
            seen.add(sample_id)
            if row["dataset"] != "compas3d" or row["role"] not in {"leader", "follower"}:
                raise ValueError("invalid dataset or role")
            if row["split"] not in SPLITS or not math.isclose(float(row["fps"]), TARGET_FPS):
                raise ValueError("invalid split or FPS")
            expected_sample = f"{row['sequence_id']}_{row['role']}"
            if sample_id != expected_sample:
                raise ValueError(f"sample_id must equal {expected_sample}")
            split_samples[row["split"]].add(sample_id)
            split_sequences[row["split"]].add(row["sequence_id"])
            sequence_rows[row["sequence_id"]].append(row)
            roles[row["role"]] += 1

            motion_path = _resolve(root, row["motion_path"], "motion_path")
            music_path = _resolve(root, row["music_feature_path"], "music_feature_path")
            if not motion_path.is_file() or not music_path.is_file():
                raise FileNotFoundError(f"missing motion/music output for {sample_id}")
            payload = safe_torch_load(motion_path)
            if not isinstance(payload, dict) or payload.get("format") != "compas3d_genmo_smpl_v1":
                raise ValueError("unsupported motion payload format")
            if payload.get("format_version") != FORMAT_VERSION:
                raise ValueError("wrong motion format_version")
            for field in ("sample_id", "sequence_id", "pair_id", "song_id", "take_id", "role"):
                if payload.get(field) != row[field]:
                    raise ValueError(f"payload/manifest {field} mismatch")
            canonical = {
                key: payload[key]
                for key in ("pose", "global_orient", "body_pose", "transl", "betas")
            }
            length = validate_canonical_motion(canonical, source=sample_id)
            if length != int(row["num_frames"]) or payload.get("num_frames") != length:
                raise ValueError("payload/manifest motion length mismatch")
            if not torch.equal(payload["betas"], torch.zeros_like(payload["betas"])):
                raise ValueError("GENMO training betas must remain exact neutral zeros [T,10]")
            source = payload.get("source_smplx", {})
            if source.get("pose_layout") != SMPLX_POSE_LAYOUT:
                raise ValueError("wrong recorded source SMPL-X pose layout")
            if not isinstance(source.get("betas_300"), torch.Tensor) or source["betas_300"].shape != (300,):
                raise ValueError("source betas_300 must be retained as Tensor[300]")
            if not torch.isfinite(source["betas_300"]).all():
                raise ValueError("source betas_300 contains NaN/Inf")
            expected_matrix = SOURCE_Z_UP_TO_GENMO_Y_UP
            torch.testing.assert_close(source["coordinate_rotation_matrix"], expected_matrix)
            if source.get("pelvis_offset_compensation_applied") is not True:
                raise ValueError("coordinate transform lacks pelvis offset compensation")
            if source.get("joint_reordering_applied") is not False:
                raise ValueError("unexpected body joint reordering")
            if source.get("translation_scale_applied") is not False:
                raise ValueError("unexpected translation scale conversion")
            if require_source_files and not _source_exists(source.get("source_npz")):
                raise FileNotFoundError(f"source NPZ no longer exists: {source.get('source_npz')}")

            relative_music = row["music_feature_path"]
            if relative_music not in music_cache:
                music = safe_torch_load(music_path)
                if not isinstance(music, torch.Tensor):
                    raise ValueError("music feature must be Tensor")
                music = music.detach().cpu().float().contiguous()
                validate_musicfeat_v2(music, source=music_path)
                music_cache[relative_music] = music
            music = music_cache[relative_music]
            if length != len(music):
                raise ValueError(f"motion_T={length}, music_T={len(music)}")
            lengths[sample_id] = length
            payload_cache[sample_id] = payload
            motion_paths.add(row["motion_path"])
            music_paths.add(relative_music)
        except Exception as exc:
            errors.append({"sample_id": sample_id, "error": f"{type(exc).__name__}: {exc}"})

    for sequence_id, rows in sorted(sequence_rows.items()):
        try:
            if len(rows) != 2 or {row["role"] for row in rows} != {"leader", "follower"}:
                raise ValueError("must contain exactly one leader and one follower")
            if len({row["split"] for row in rows}) != 1:
                raise ValueError("leader/follower cross-split leakage")
            if len({row["music_feature_path"] for row in rows}) != 1:
                raise ValueError("leader/follower must reference the exact same music feature")
            if len({row["num_frames"] for row in rows}) != 1:
                raise ValueError("leader/follower output lengths differ")
            leader = payload_cache.get(f"{sequence_id}_leader")
            follower = payload_cache.get(f"{sequence_id}_follower")
            if leader is not None and follower is not None and torch.equal(leader["pose"], follower["pose"]):
                raise ValueError("leader and follower pose tensors are incorrectly identical")
        except Exception as exc:
            errors.append({"sample_id": sequence_id, "error": f"{type(exc).__name__}: {exc}"})

    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            leakage = sorted(split_sequences[left] & split_sequences[right])
            if leakage:
                errors.append({"sample_id": "<split>", "error": f"sequence leakage {left}/{right}: {leakage}"})
    song_splits: dict[str, set[str]] = defaultdict(set)
    for row in manifests:
        song_splits[row["song_id"]].add(row["split"])
    music_leakage = {
        song: sorted(values) for song, values in song_splits.items() if len(values) > 1
    }
    if split_report.get("strategy") == "music_identity" and music_leakage:
        errors.append({"sample_id": "<split>", "error": f"music identity leakage: {music_leakage}"})

    disk_motion = {path.relative_to(root).as_posix() for path in (root / "motions").glob("*.pt")}
    disk_music = {path.relative_to(root).as_posix() for path in (root / "musicfeat_v2").glob("*.pt")}
    if disk_motion != motion_paths or disk_music != music_paths:
        errors.append(
            {
                "sample_id": "<inventory>",
                "error": (
                    f"disk/manifest mismatch: motion={len(disk_motion)}/{len(motion_paths)}, "
                    f"music={len(disk_music)}/{len(music_paths)}"
                ),
            }
        )

    forward_results: list[dict[str, Any]] = []
    candidates = [row for row in manifests if row["sample_id"] in payload_cache]
    if smpl_forward_samples and candidates:
        model = make_smplx("supermotion", use_pca=False, flat_hand_mean=True).eval()
        selected = random.Random(seed).sample(candidates, min(smpl_forward_samples, len(candidates)))
        with torch.no_grad():
            for row in selected:
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
                height = output.vertices[..., 1].max(1).values - output.vertices[..., 1].min(1).values
                result = {
                    "sample_id": row["sample_id"],
                    "checked_frames": len(indices),
                    "vertices_shape": list(output.vertices.shape),
                    "joints_shape": list(output.joints.shape),
                    "finite": finite,
                    "body_height_m_min": float(height.min()),
                    "body_height_m_max": float(height.max()),
                    "root_translation_min": payload["transl"].min(0).values.tolist(),
                    "root_translation_max": payload["transl"].max(0).values.tolist(),
                }
                if not finite or not (0.8 <= float(height.median()) <= 2.5):
                    errors.append({"sample_id": row["sample_id"], "error": "invalid SMPL forward/height"})
                forward_results.append(result)

    person_frames = sum(lengths.values())
    group_frames = {
        sequence_id: int(rows[0]["num_frames"])
        for sequence_id, rows in sequence_rows.items()
        if rows
    }
    leader_frames = sum(lengths.get(f"{sequence_id}_leader", 0) for sequence_id in sequence_rows)
    follower_frames = sum(lengths.get(f"{sequence_id}_follower", 0) for sequence_id in sequence_rows)
    report = {
        "format": "compas3d_genmo_validation_report_v1",
        "root": str(root.resolve()),
        "raw_complete_sequence_count": conversion.get("raw_complete_sequence_count"),
        "converted_sequence_count": len(sequence_rows),
        "dancer_sample_count": len(manifests),
        "roles": dict(roles),
        "sequence_counts_by_split": {split: len(split_sequences[split]) for split in SPLITS},
        "sample_counts_by_split": {split: len(split_samples[split]) for split in SPLITS},
        "person_motion_frames": person_frames,
        "person_motion_hours": person_frames / TARGET_FPS / 3600,
        "leader_hours": leader_frames / TARGET_FPS / 3600,
        "follower_hours": follower_frames / TARGET_FPS / 3600,
        "unique_music_count": len(music_paths),
        "unique_music_frames": sum(group_frames.values()),
        "unique_music_hours": sum(group_frames.values()) / TARGET_FPS / 3600,
        "leader_follower_shared_music_sequence_count": sum(
            len(rows) == 2 and len({row["music_feature_path"] for row in rows}) == 1
            for rows in sequence_rows.values()
        ),
        "motion_music_length_mismatch_count": sum("motion_T=" in error["error"] for error in errors),
        "leader_follower_length_mismatch_count": sum(
            "lengths differ" in error["error"] for error in errors
        ),
        "music_identity_leakage": music_leakage,
        "smpl_forward_results": forward_results,
        "conversion_alignment": {
            "audio_minus_motion_source_frames": conversion.get("audio_minus_motion_source_frames"),
            "feature_minus_motion_before_alignment_histogram": conversion.get(
                "feature_minus_motion_before_alignment_histogram"
            ),
            "failed_or_skipped": conversion.get("failed_or_skipped", []),
        },
        "validation_error_count": len(errors),
        "validation_errors": errors,
        "final_pass": not errors,
    }
    return report, manifests


def _choose_render_samples(manifests: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    shuffled = manifests.copy()
    rng.shuffle(shuffled)
    chosen: list[dict[str, Any]] = []
    roles: set[str] = set()
    pairs: set[str] = set()
    for row in shuffled:
        if row["pair_id"] in pairs:
            continue
        if len(chosen) == 1 and row["role"] in roles:
            continue
        chosen.append(row)
        roles.add(row["role"])
        pairs.add(row["pair_id"])
        if len(chosen) == count:
            break
    for row in shuffled:
        if row not in chosen and row["pair_id"] not in pairs:
            chosen.append(row)
            pairs.add(row["pair_id"])
        if len(chosen) == count:
            break
    return chosen[:count]


def render_sample(
    root: Path,
    manifest: dict[str, Any],
    render_dir: Path,
    *,
    width: int,
    height: int,
    clip_frames: int,
) -> dict[str, Any]:
    from gem.utils.video_io_utils import save_video
    from scripts.demo.demo_utils import render_global_frames

    payload = safe_torch_load(_resolve(root, manifest["motion_path"], "motion_path"))
    total = int(payload["num_frames"])
    length = min(total, clip_frames)
    start = max(0, (total - length) // 2)
    indices = torch.arange(start, start + length)
    model = make_smplx("supermotion", use_pca=False, flat_hand_mean=True).eval()
    vertices: list[torch.Tensor] = []
    with torch.no_grad():
        for chunk in indices.split(64):
            output = model(
                global_orient=payload["global_orient"][chunk],
                body_pose=payload["body_pose"][chunk],
                transl=payload["transl"][chunk],
                betas=payload["betas"][chunk],
            )
            vertices.append(output.vertices.detach().cpu().float())
    verts = torch.cat(vertices)
    if not torch.isfinite(verts).all():
        raise ValueError("SMPL render vertices contain NaN/Inf")
    # Constant visualization-only floor alignment; saved root translation is untouched.
    vertical_offset = float(verts[..., 1].min())
    verts[..., 1] -= vertical_offset
    faces = torch.from_numpy(np.asarray(model.faces, dtype=np.int64)).long()
    frames = render_global_frames(verts, faces, width, height)
    render_dir.mkdir(parents=True, exist_ok=True)
    output_path = render_dir / f"{manifest['sample_id']}.mp4"
    save_video(frames, str(output_path), fps=30)
    return {
        "sample_id": manifest["sample_id"],
        "pair_id": manifest["pair_id"],
        "role": manifest["role"],
        "render_path": str(output_path.resolve()),
        "source_frames": total,
        "clip_start_frame": start,
        "rendered_frames": length,
        "stride": 1,
        "render_fps": 30,
        "visualization_only_vertical_offset": vertical_offset,
        "finite": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--smpl-forward-samples", type=int, default=10)
    parser.add_argument("--require-source-files", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render-samples", type=int, default=0)
    parser.add_argument("--render-dir", type=Path)
    parser.add_argument("--render-width", type=int, default=512)
    parser.add_argument("--render-height", type=int, default=512)
    parser.add_argument("--render-clip-frames", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260813)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if min(args.smpl_forward_samples, args.render_samples) < 0:
        raise ValueError("sample counts must be non-negative")
    if min(args.render_width, args.render_height, args.render_clip_frames) <= 0:
        raise ValueError("render dimensions and clip length must be positive")
    root = args.root.expanduser().resolve()
    report, manifests = validate_dataset(
        root,
        smpl_forward_samples=args.smpl_forward_samples,
        seed=args.seed,
        require_source_files=args.require_source_files,
    )
    renders: list[dict[str, Any]] = []
    render_errors: list[dict[str, str]] = []
    if args.render_samples:
        render_dir = (
            args.render_dir.expanduser().resolve()
            if args.render_dir is not None
            else root / "renders"
        )
        for row in _choose_render_samples(manifests, args.render_samples, args.seed):
            try:
                result = render_sample(
                    root,
                    row,
                    render_dir,
                    width=args.render_width,
                    height=args.render_height,
                    clip_frames=args.render_clip_frames,
                )
                renders.append(result)
                print(f"[render] {result['sample_id']} -> {result['render_path']}")
            except Exception as exc:
                render_errors.append(
                    {"sample_id": row["sample_id"], "error": f"{type(exc).__name__}: {exc}"}
                )
    report["requested_render_count"] = args.render_samples
    report["successful_render_count"] = len(renders)
    report["renders"] = renders
    report["render_errors"] = render_errors
    report["final_pass"] = report["final_pass"] and not render_errors
    report_path = root / "reports" / "validation_report.json"
    write_json(report_path, report)
    keys = (
        "raw_complete_sequence_count", "converted_sequence_count", "dancer_sample_count",
        "sequence_counts_by_split", "person_motion_hours", "unique_music_count",
        "unique_music_hours", "motion_music_length_mismatch_count",
        "leader_follower_shared_music_sequence_count", "validation_error_count",
        "successful_render_count", "final_pass",
    )
    print(json.dumps({key: report[key] for key in keys}, ensure_ascii=False, indent=2))
    print(f"[validate] report: {report_path}")
    return 0 if report["final_pass"] or not args.strict else 2


if __name__ == "__main__":
    raise SystemExit(main())
