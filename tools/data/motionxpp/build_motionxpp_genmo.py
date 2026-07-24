#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""把 Motion-X++ SMPL-X 3D 动作和 semantic text 构建成 GENMO 分片。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.data.motionxpp.common import (  # noqa: E402
    IGNORED_RANGES,
    MIN_FRAMES,
    OFFICIAL_FPS,
    FilteredMotionError,
    MotionXppError,
    anomaly_statistics,
    atomic_torch_save,
    atomic_write_json,
    atomic_write_jsonl,
    content_hash,
    convert_coordinate_system,
    deterministic_split,
    discover_subsets,
    motion_group,
    paired_asset_indices,
    parse_motion_asset,
    parse_text_asset,
    resample_motion,
    safe_torch_load,
    validate_record,
)

DEFAULT_ROOT = Path("inputs/Motion-Xplusplus")
DEFAULT_OUTPUT_ROOT = DEFAULT_ROOT / "genmo_support"
DEFAULT_SUBSETS_FILE = Path("outputs/motionxpp_inspect/recommended_subsets.txt")
BUILD_VERSION = 1


def _read_subset_file(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Subset file does not exist: {path}")
    result: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.split("#", 1)[0].strip()
        if value:
            result.append(value)
    return result


def select_subsets(args: argparse.Namespace) -> list[str]:
    """解析显式列表、推荐列表和排除列表。"""
    discovered = discover_subsets(args.root)
    pairable = set(discovered["motion"]) & set(discovered["text"])
    if args.subsets:
        selected = list(args.subsets)
    elif args.subsets_file is not None:
        selected = _read_subset_file(args.subsets_file)
    elif DEFAULT_SUBSETS_FILE.is_file():
        selected = _read_subset_file(DEFAULT_SUBSETS_FILE)
    else:
        selected = sorted(pairable)
    excluded = set(args.exclude_subsets or [])
    selected = [subset for subset in selected if subset not in excluded]
    unknown = sorted(set(selected) - pairable)
    if unknown:
        raise MotionXppError(
            f"Selected subsets do not have both motion and semantic text: {unknown}"
        )
    # 稳定去重，保留用户顺序。
    selected = list(dict.fromkeys(selected))
    if not selected:
        raise MotionXppError("No Motion-X++ subsets remain after filtering")
    return selected


def _build_fingerprint(args: argparse.Namespace, subsets: list[str]) -> str:
    contract = {
        "build_version": BUILD_VERSION,
        "root": str(Path(args.root).expanduser().resolve()),
        "subsets": subsets,
        "source_up_axis": args.source_up_axis,
        "source_fps": args.source_fps,
        "official_fps_fallback": OFFICIAL_FPS,
        "target_fps": args.target_fps,
        "split_seed": args.split_seed,
        "records_per_shard": args.records_per_shard,
        "limit": args.limit,
        "minimum_frames": MIN_FRAMES,
        "ignored_ranges": IGNORED_RANGES,
    }
    return hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _shard_fingerprint(
    build_fingerprint: str, split: str, shard_index: int, buffer: dict[str, dict]
) -> str:
    digest = hashlib.sha256()
    digest.update(build_fingerprint.encode())
    digest.update(split.encode())
    digest.update(str(shard_index).encode())
    for motion_id, record in buffer.items():
        digest.update(motion_id.encode())
        digest.update(str(record["content_hash"]).encode())
    return digest.hexdigest()


def _validate_existing_shard(
    path: Path,
    expected_ids: list[str],
    expected_records: dict[str, dict[str, Any]],
) -> int:
    shard = safe_torch_load(path)
    if not isinstance(shard, dict) or list(shard) != expected_ids:
        raise MotionXppError(
            f"Resume shard key mismatch for {path}: "
            f"expected {expected_ids[:3]}..., got {list(shard)[:3] if isinstance(shard, dict) else type(shard)}"
        )
    for motion_id, record in shard.items():
        validate_record(record, motion_id)
        if record.get("content_hash") != expected_records[motion_id].get("content_hash"):
            raise MotionXppError(f"Resume shard content hash mismatch for {motion_id} in {path}")
    return path.stat().st_size


def _flush_shard(
    *,
    split: str,
    shard_index: int,
    buffer: dict[str, dict[str, Any]],
    output_root: Path,
    build_fingerprint: str,
    resume: bool,
    dry_run: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    shard_rel = Path("shards") / split / f"motionxpp_{split}_{shard_index:05d}.pth"
    meta_rel = shard_rel.with_suffix(".meta.json")
    shard_path = output_root / shard_rel
    meta_path = output_root / meta_rel
    fingerprint = _shard_fingerprint(build_fingerprint, split, shard_index, buffer)
    expected_ids = list(buffer)
    resumed = False
    output_size = 0
    if not dry_run:
        if shard_path.exists() or meta_path.exists():
            if not resume:
                raise FileExistsError(f"Shard state already exists: {shard_path}; use --resume")
            if shard_path.is_file():
                output_size = _validate_existing_shard(shard_path, expected_ids, buffer)
                if meta_path.is_file():
                    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                    if metadata.get("fingerprint") != fingerprint:
                        raise MotionXppError(
                            f"Existing shard fingerprint differs: {shard_path}. "
                            "Do not mix outputs from different build settings."
                        )
                else:
                    # 分片已原子发布但进程在 meta 发布前退出：验证后补写 meta。
                    atomic_write_json(
                        meta_path,
                        {
                            "build_version": BUILD_VERSION,
                            "fingerprint": fingerprint,
                            "build_fingerprint": build_fingerprint,
                            "split": split,
                            "shard_index": shard_index,
                            "motion_ids": expected_ids,
                            "record_count": len(expected_ids),
                            "output_size_bytes": output_size,
                        },
                    )
                resumed = True
            else:
                # 只有 meta 表明上次尚未成功发布 PTH；丢弃本工具自己的不完整 meta。
                meta_path.unlink(missing_ok=True)
        else:
            pass
        if not resumed:
            for motion_id, record in buffer.items():
                validate_record(record, motion_id)
            atomic_torch_save(buffer, shard_path)
            output_size = _validate_existing_shard(shard_path, expected_ids, buffer)
            atomic_write_json(
                meta_path,
                {
                    "build_version": BUILD_VERSION,
                    "fingerprint": fingerprint,
                    "build_fingerprint": build_fingerprint,
                    "split": split,
                    "shard_index": shard_index,
                    "motion_ids": expected_ids,
                    "record_count": len(expected_ids),
                    "output_size_bytes": output_size,
                },
            )

    rows = []
    for motion_id, record in buffer.items():
        rows.append(
            {
                "motion_id": motion_id,
                "shard_path": shard_rel.as_posix(),
                "record_key": motion_id,
                "frames": int(record["pose"].shape[0]),
                "caption_count": len(record["text_data"]),
                "subset": record["source_subset"],
                "source_group": record["source_group"],
                "content_hash": record["content_hash"],
                "motion_hash": record["motion_hash"],
                "fps": float(record["fps"]),
            }
        )
    return rows, {
        "split": split,
        "shard_index": shard_index,
        "path": shard_rel.as_posix(),
        "record_count": len(buffer),
        "output_size_bytes": output_size,
        "fingerprint": fingerprint,
        "resumed": resumed,
        "dry_run": dry_run,
    }


def _motion_id(subset: str, stem: str) -> str:
    """构造跨 subset 唯一且确定的 record key。"""
    return f"{subset}__{stem}"


def _source_fps(parsed: dict[str, Any], args: argparse.Namespace) -> tuple[float, str]:
    if parsed.get("fps") is not None:
        return float(parsed["fps"]), "motion_metadata"
    if args.source_fps is not None:
        return float(args.source_fps), "cli_default"
    # Motion-X 官方 README 明确所有发布动作已统一为 30 FPS。
    return OFFICIAL_FPS, "official_motionx_unified_30fps"


def _build_record(
    subset: str,
    stem: str,
    motion_ref: Any,
    text_ref: Any,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    parsed = parse_motion_asset(motion_ref)
    source_fps, fps_source = _source_fps(parsed, args)
    root, transl, _, rotation = convert_coordinate_system(
        parsed["global_orient"],
        parsed["transl"],
        args.source_up_axis,
    )
    parsed["global_orient"] = root
    parsed["transl"] = transl
    parsed = resample_motion(parsed, source_fps, args.target_fps)
    frame_count = parsed["body_pose"].shape[0]
    if frame_count < MIN_FRAMES:
        raise FilteredMotionError(
            f"resampled motion has {frame_count} frames, minimum is {MIN_FRAMES}"
        )
    text_data, text_metadata = parse_text_asset(text_ref)
    for item in text_data:
        item["source"] = subset
    pose = (
        torch.cat([parsed["global_orient"], parsed["body_pose"]], dim=-1)
        .float()
        .cpu()
        .contiguous()
        .clone()
    )
    trans = parsed["transl"].float().cpu().contiguous().clone()
    beta = parsed["betas"].float().cpu()
    if beta.ndim == 2 and torch.allclose(beta, beta[:1].expand_as(beta), atol=0.0, rtol=0.0):
        beta = beta[0]
    beta = beta.contiguous().clone()
    motion_digest, source_digest = content_hash(
        pose,
        trans,
        beta,
        subset=subset,
        source_path=motion_ref.source_path,
    )
    record = {
        "pose": pose,
        "trans": trans,
        "beta": beta,
        "gender": parsed["gender"],
        "fps": float(args.target_fps),
        "text_data": text_data,
        "source_subset": subset,
        "source_path": motion_ref.source_path,
        "source_text_path": text_ref.source_path,
        "source_group": f"{subset}:{motion_group(stem)}",
        "content_hash": source_digest,
        "motion_hash": motion_digest,
    }
    motion_id = _motion_id(subset, stem)
    validate_record(record, motion_id)
    audit = {
        "motion_id": motion_id,
        "source_frames": int(parsed.get("raw_shape", [0])[0]) if parsed.get("raw_shape") else None,
        "output_frames": frame_count,
        "source_fps": source_fps,
        "fps_source": fps_source,
        "target_fps": float(args.target_fps),
        "source_up_axis": args.source_up_axis,
        "coordinate_rotation": rotation.tolist(),
        "caption_count": len(text_data),
        "text_metadata": text_metadata,
        "anomalies": anomaly_statistics(pose, trans, args.target_fps),
    }
    return record, audit


def _prepare_output(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    if args.dry_run:
        return
    manifests = output_root / "manifests"
    existing_manifests = list(manifests.glob("*.jsonl")) if manifests.is_dir() else []
    existing_shards = (
        list((output_root / "shards").rglob("*.pth")) if (output_root / "shards").is_dir() else []
    )
    if (existing_manifests or existing_shards) and not args.resume:
        raise FileExistsError(
            f"Build output already exists under {output_root}; use --resume or a new output root"
        )


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    """流式构建 Motion-X++ motion shards、manifest 和审计报告。"""
    started = time.monotonic()
    args.root = Path(args.root).expanduser().resolve()
    args.output_root = Path(args.output_root)
    if not args.root.is_dir():
        raise FileNotFoundError(f"Motion-X++ root does not exist: {args.root}")
    if not args.source_up_axis:
        raise MotionXppError(
            "Motion-X++ build requires an explicit --source-up-axis. "
            "The audited Motion-X++ smplx322 release is Y-up; pass --source-up-axis y."
        )
    _prepare_output(args)
    subsets = select_subsets(args)
    build_fingerprint = _build_fingerprint(args, subsets)

    pairs: list[tuple[str, str, Any, Any]] = []
    pairing_summary: dict[str, Any] = {}
    for subset in subsets:
        motion_index, text_index, _ = paired_asset_indices(args.root, subset)
        paired = sorted(set(motion_index.assets) & set(text_index.assets))
        pairing_summary[subset] = {
            "motion_count": len(motion_index.assets),
            "text_count": len(text_index.assets),
            "paired_count": len(paired),
            "motion_without_text": len(set(motion_index.assets) - set(text_index.assets)),
            "text_without_motion": len(set(text_index.assets) - set(motion_index.assets)),
            "stem_collisions": {
                "motion": motion_index.collisions,
                "text": text_index.collisions,
            },
        }
        for stem in paired:
            pairs.append(
                (
                    subset,
                    stem,
                    motion_index.assets[stem],
                    text_index.assets[stem],
                )
            )
    pairs.sort(key=lambda item: (item[0], item[1]))

    buffers: dict[str, dict[str, dict[str, Any]]] = {
        "train": {},
        "val": {},
        "test": {},
    }
    manifests: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    shard_indices = {"train": 0, "val": 0, "test": 0}
    shard_reports: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    coordinate_rows: list[dict[str, Any]] = []
    seen_motion_hash: dict[str, str] = {}
    accepted_count = 0

    def flush(split: str) -> None:
        if not buffers[split]:
            return
        rows, report = _flush_shard(
            split=split,
            shard_index=shard_indices[split],
            buffer=buffers[split],
            output_root=args.output_root,
            build_fingerprint=build_fingerprint,
            resume=args.resume,
            dry_run=args.dry_run,
        )
        manifests[split].extend(rows)
        shard_reports.append(report)
        shard_indices[split] += 1
        buffers[split] = {}

    for subset, stem, motion_ref, text_ref in pairs:
        if args.limit is not None and accepted_count >= args.limit:
            break
        motion_id = _motion_id(subset, stem)
        try:
            record, audit = _build_record(subset, stem, motion_ref, text_ref, args)
            duplicate_of = seen_motion_hash.get(record["motion_hash"])
            if duplicate_of is not None:
                duplicates.append(
                    {
                        "motion_id": motion_id,
                        "duplicate_of": duplicate_of,
                        "motion_hash": record["motion_hash"],
                        "source_path": motion_ref.source_path,
                    }
                )
                continue
            seen_motion_hash[record["motion_hash"]] = motion_id
            split = deterministic_split(subset, stem, args.split_seed)
            buffers[split][motion_id] = record
            coordinate_rows.append(audit)
            accepted_count += 1
            if len(buffers[split]) >= args.records_per_shard:
                flush(split)
        except Exception as exc:
            row = {
                "motion_id": motion_id,
                "subset": subset,
                "motion_path": motion_ref.source_path,
                "text_path": text_ref.source_path,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            rejected.append(row)
            if args.strict and not isinstance(exc, FilteredMotionError):
                raise MotionXppError(
                    f"Strict build rejected {motion_id}: {type(exc).__name__}: {exc}"
                ) from exc

    for split in ("train", "val", "test"):
        flush(split)

    if not args.dry_run:
        for split, rows in manifests.items():
            atomic_write_jsonl(args.output_root / "manifests" / f"{split}.jsonl", rows)
    reports_root = args.output_root / "reports"
    split_summary = {
        split: {
            "records": len(rows),
            "frames": sum(int(row["frames"]) for row in rows),
            "captions": sum(int(row["caption_count"]) for row in rows),
            "shards": shard_indices[split],
            "subsets": dict(Counter(row["subset"] for row in rows)),
        }
        for split, rows in manifests.items()
    }
    output_size = sum(item["output_size_bytes"] for item in shard_reports)
    summary = {
        "status": "dry_run_complete" if args.dry_run else "complete",
        "build_version": BUILD_VERSION,
        "build_fingerprint": build_fingerprint,
        "root": str(args.root),
        "output_root": str(args.output_root.resolve()),
        "subsets": subsets,
        "candidate_pairs": len(pairs),
        "accepted_records": accepted_count,
        "rejected_records": len(rejected),
        "filtered_short_records": sum(
            row["error_type"] == "FilteredMotionError" for row in rejected
        ),
        "duplicate_records": len(duplicates),
        "split_records": {key: value["records"] for key, value in split_summary.items()},
        "total_frames": sum(value["frames"] for value in split_summary.values()),
        "total_captions": sum(value["captions"] for value in split_summary.values()),
        "source_up_axis": args.source_up_axis,
        "target_coordinate": "AY / Y-up",
        "source_fps_override": args.source_fps,
        "official_fps_fallback": OFFICIAL_FPS,
        "target_fps": float(args.target_fps),
        "minimum_frames": MIN_FRAMES,
        "ignored_smplx_dimensions": IGNORED_RANGES,
        "keypoints_included": False,
        "keypoint_reason": (
            "本版不写入 2D 条件：真实归档缺 image width/height 和 calibrated camera intrinsics。"
        ),
        "records_per_shard": args.records_per_shard,
        "shard_count": sum(shard_indices.values()),
        "resumed_shards": sum(bool(item["resumed"]) for item in shard_reports),
        "output_size_bytes": output_size,
        "pairing": pairing_summary,
        "limit": args.limit,
        "elapsed_seconds": time.monotonic() - started,
    }
    # dry-run 仍写独立审计报告，但绝不写 motion shard/manifest。
    reports_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(reports_root / "build_summary.json", summary)
    atomic_write_jsonl(reports_root / "rejected_samples.jsonl", rejected)
    atomic_write_jsonl(reports_root / "duplicates.jsonl", duplicates)
    atomic_write_json(
        reports_root / "coordinate_audit.json",
        {
            "source_up_axis": args.source_up_axis,
            "target": "AY / Y-up",
            "records": coordinate_rows,
        },
    )
    atomic_write_json(reports_root / "split_summary.json", split_summary)
    return {
        "summary": summary,
        "split_summary": split_summary,
        "manifests": manifests,
        "rejected": rejected,
        "duplicates": duplicates,
    }


def build_parser() -> argparse.ArgumentParser:
    """创建 Motion-X++ builder CLI。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--subsets", nargs="+")
    parser.add_argument("--subsets-file", type=Path)
    parser.add_argument("--exclude-subsets", nargs="*", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--records-per-shard", type=int, default=512)
    parser.add_argument("--source-up-axis", choices=("x", "y", "z"))
    parser.add_argument(
        "--source-fps",
        type=float,
        help=(
            "Fallback only when a sample has no FPS. Omit for the official Motion-X "
            "release contract (30 FPS)."
        ),
    )
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--split-seed", type=int, default=20260724)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.records_per_shard <= 0:
        raise ValueError("--records-per-shard must be positive")
    if args.target_fps <= 0:
        raise ValueError("--target-fps must be positive")
    if args.source_fps is not None and args.source_fps <= 0:
        raise ValueError("--source-fps must be positive")
    if args.subsets and args.subsets_file:
        raise ValueError("--subsets and --subsets-file are mutually exclusive")


def main(argv: list[str] | None = None) -> int:
    """构建并打印摘要。"""
    args = build_parser().parse_args(argv)
    _validate_args(args)
    result = build_dataset(args)
    summary = result["summary"]
    print("=" * 72)
    print("Motion-X++ -> GENMO 构建完成")
    print(f"  mode:          {summary['status']}")
    print(f"  subsets:       {', '.join(summary['subsets'])}")
    print(f"  candidate:     {summary['candidate_pairs']}")
    print(f"  accepted:      {summary['accepted_records']}")
    print(f"  rejected:      {summary['rejected_records']}")
    print(f"  duplicates:    {summary['duplicate_records']}")
    print(f"  split:         {summary['split_records']}")
    print(f"  total frames:  {summary['total_frames']}")
    print(f"  shards:        {summary['shard_count']}")
    print(f"  output:        {args.output_root}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
