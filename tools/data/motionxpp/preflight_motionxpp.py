#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""训练前抽样校验 Motion-X++ motion/embedding shards、Dataset 和 collate。"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gem.datamodule.mocap_trainX_testY import collate_fn  # noqa: E402
from gem.datasets.pure_motion.motionxpp import MotionXppDataset  # noqa: E402
from tools.data.motionxpp.common import (  # noqa: E402
    MotionXppError,
    atomic_write_json,
    read_jsonl,
    safe_torch_load,
    validate_record,
)

DEFAULT_ROOT = Path("inputs/Motion-Xplusplus")
DEFAULT_MOTION_MANIFEST = DEFAULT_ROOT / "genmo_support/manifests/train.jsonl"
DEFAULT_EMBEDDING_MANIFEST = DEFAULT_ROOT / "t5_embeddings_v1_half/manifests/train.json"
DEFAULT_REPORT = Path("outputs/motionxpp_preflight/report.json")


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _cached_load(path: Path, cache: OrderedDict[str, Any], max_size: int = 2) -> dict[str, Any]:
    key = str(path.resolve())
    if key in cache:
        value = cache.pop(key)
        cache[key] = value
        return value
    value = safe_torch_load(path)
    if not isinstance(value, dict):
        raise MotionXppError(f"Shard must contain a dict: {path}")
    cache[key] = value
    while len(cache) > max_size:
        cache.popitem(last=False)
    return value


def _shape_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "finite": bool(torch.isfinite(value).all()) if value.is_floating_point() else True,
        }
    if isinstance(value, dict):
        return {key: _shape_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_shape_tree(item) for item in value[:2]]
    return type(value).__name__


def preflight(
    *,
    root: Path,
    motion_manifest_path: Path,
    embedding_manifest_path: Path,
    sample_records: int,
    dataset_samples: int,
    seed: int,
) -> dict[str, Any]:
    """校验两个 manifest 的键、分片内容、Dataset 输出和 batch。"""
    root = root.expanduser()
    rows = read_jsonl(motion_manifest_path)
    if not rows:
        raise MotionXppError("Motion manifest is empty")
    embedding_manifest = json.loads(embedding_manifest_path.read_text(encoding="utf-8"))
    embedding_map = embedding_manifest.get("motion_to_shard")
    if not isinstance(embedding_map, dict):
        raise MotionXppError("Embedding manifest is missing motion_to_shard")
    motion_ids = [str(row["motion_id"]) for row in rows]
    if set(motion_ids) != set(embedding_map):
        missing = sorted(set(motion_ids) - set(embedding_map))
        extra = sorted(set(embedding_map) - set(motion_ids))
        raise MotionXppError(
            f"Motion/embedding key mismatch: missing={missing[:10]}, extra={extra[:10]}"
        )
    motion_root = motion_manifest_path.parent.parent
    embedding_root = embedding_manifest_path.parent.parent
    rng = random.Random(seed)
    selected = rng.sample(rows, min(sample_records, len(rows)))
    motion_cache: OrderedDict[str, Any] = OrderedDict()
    embedding_cache: OrderedDict[str, Any] = OrderedDict()
    validated: list[dict[str, Any]] = []
    for row in selected:
        motion_id = str(row["motion_id"])
        motion_path = _resolve(motion_root, str(row["shard_path"]))
        if not motion_path.is_file():
            raise FileNotFoundError(f"Motion shard does not exist: {motion_path}")
        motion_shard = _cached_load(motion_path, motion_cache)
        record_key = str(row.get("record_key", motion_id))
        record = motion_shard[record_key]
        validate_record(record, motion_id)
        if int(row["frames"]) != record["pose"].shape[0]:
            raise MotionXppError(f"{motion_id}: manifest frame count mismatch")
        if int(row["caption_count"]) != len(record["text_data"]):
            raise MotionXppError(f"{motion_id}: manifest caption count mismatch")

        embed_meta = embedding_map[motion_id]
        embed_path = _resolve(embedding_root, str(embed_meta["shard_path"]))
        if not embed_path.is_file():
            raise FileNotFoundError(f"Embedding shard does not exist: {embed_path}")
        embed_shard = _cached_load(embed_path, embedding_cache)
        embed_key = str(embed_meta.get("record_key", motion_id))
        embedding = embed_shard[embed_key]
        expected = (len(record["text_data"]), 50, 1024)
        if (
            not isinstance(embedding, torch.Tensor)
            or tuple(embedding.shape) != expected
            or embedding.dtype != torch.float16
            or not torch.isfinite(embedding).all()
        ):
            raise MotionXppError(f"{motion_id}: embedding must be finite FP16 {expected}")
        validated.append(
            {
                "motion_id": motion_id,
                "frames": int(record["pose"].shape[0]),
                "captions": len(record["text_data"]),
                "motion_shard": str(motion_path),
                "embedding_shard": str(embed_path),
            }
        )

    dataset = MotionXppDataset(
        root=root,
        manifest_path=motion_manifest_path,
        embedding_manifest_path=embedding_manifest_path,
        split=motion_manifest_path.stem,
        motion_frames=120,
        cam_augmentation="static",
        condition_on_keypoints=False,
        limit_size=max(dataset_samples, 2),
        shard_cache_size=2,
        random_seed=seed,
    )
    samples = [dataset[index] for index in range(min(dataset_samples, len(dataset)))]
    if len(samples) < 2:
        raise MotionXppError("Preflight needs at least two Dataset samples")
    for index, sample in enumerate(samples):
        if sample["text_embed"].shape != (50, 1024):
            raise MotionXppError(f"Dataset sample {index} has invalid text_embed")
        if not sample["has_text"] or not sample["caption"]:
            raise MotionXppError(f"Dataset sample {index} lost semantic text")
        if bool(sample["mask"]["2d_only"]):
            raise MotionXppError(f"Dataset sample {index} was incorrectly marked 2d_only")
        if sample["mask"]["has_2d_mask"].any():
            raise MotionXppError(f"Dataset sample {index} incorrectly enabled uncalibrated 2D")
    collate_cfg = OmegaConf.create(
        {
            "max_motion_frames": 120,
            "default_frame_feature_dim": {
                "music_array": [1024],
                "music_embed": [35],
                "music_beats": [],
                "audio_array": [],
                "use_det_kp": [],
            },
            "default_seq_feature_dim": {"text_embed": [50, 1024]},
            "default_seq_feature_length_multiplier": {"audio_array": 600},
            "default_feature_val": {
                "caption": "",
                "music_fps": 30,
                "audio_fps": 30,
                "has_text": False,
            },
            "default_feature_type": {},
        }
    )
    batch = collate_fn(samples[:2], mode="train", collate_cfg=collate_cfg)
    if batch["text_embed"].shape != (2, 50, 1024):
        raise MotionXppError(
            f"Collated text_embed must be [2,50,1024], got {batch['text_embed'].shape}"
        )
    if batch["smpl_params_w"]["body_pose"].shape != (2, 120, 63):
        raise MotionXppError("Collated body_pose must be [2,120,63]")
    return {
        "status": "passed",
        "root": str(root.resolve()),
        "motion_manifest": str(motion_manifest_path.resolve()),
        "embedding_manifest": str(embedding_manifest_path.resolve()),
        "manifest_records": len(rows),
        "embedding_records": len(embedding_map),
        "validated_record_count": len(validated),
        "validated_records": validated,
        "dataset_length": len(dataset),
        "dataset_sample_count": len(samples),
        "dataset_sample_shapes": [_shape_tree(sample) for sample in samples[:2]],
        "batch_size": 2,
        "batch_shapes": _shape_tree(batch),
        "all_finite": True,
        "motion_embedding_keys_exact": True,
        "condition_on_keypoints": False,
    }


def build_parser() -> argparse.ArgumentParser:
    """创建预检 CLI。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--motion-manifest", type=Path, default=DEFAULT_MOTION_MANIFEST)
    parser.add_argument("--embedding-manifest", type=Path, default=DEFAULT_EMBEDDING_MANIFEST)
    parser.add_argument("--sample-records", type=int, default=64)
    parser.add_argument("--dataset-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main(argv: list[str] | None = None) -> int:
    """运行预检并原子写 report.json。"""
    args = build_parser().parse_args(argv)
    if args.sample_records <= 0 or args.dataset_samples < 2:
        raise ValueError("--sample-records must be >0 and --dataset-samples must be >=2")
    report = preflight(
        root=args.root,
        motion_manifest_path=args.motion_manifest,
        embedding_manifest_path=args.embedding_manifest,
        sample_records=args.sample_records,
        dataset_samples=args.dataset_samples,
        seed=args.seed,
    )
    atomic_write_json(args.report, report)
    print("=" * 72)
    print("Motion-X++ 训练前预检通过")
    print(f"  manifest records: {report['manifest_records']}")
    print(f"  checked records:  {report['validated_record_count']}")
    print(f"  dataset samples:  {report['dataset_sample_count']}")
    print(f"  batch size:       {report['batch_size']}")
    print(f"  report:           {args.report}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
