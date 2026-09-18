#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""训练前审计 MotionMillion motion/T5 release 与 GENMO 151D 统计分布。

该工具只读检查三个官方 split 的 manifest、sample index、motion/embedding shard
一一对齐关系、SHA256、record 顺序、文本有效 token、ID/镜像 base ID 泄漏和物理
尺度。可选的 ``--normalized-stats-samples`` 会实际走 MotionMillionDataset 与现有
``MM_V1_AMASS_LOCAL_BEDLAM_CAM`` EnDecoder，报告 151 个归一化通道的均值、标准差
及超阈比例；大范围离群会阻断后续 smoke，而不会临时更换统计量。

预检不会下载数据、加载 T5、启动训练或修改 release。报告以原子 JSON 写入用户指定
路径，默认位于转换数据的 reports 目录，便于和 dataset fingerprint 一起归档。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import default_collate

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.data.motionmillion.common import (  # noqa: E402
    SAMPLE_INDEX_DTYPE,
    SCHEMA_VERSION,
    SPLITS,
    MotionMillionError,
    atomic_write_json,
    mirror_base_id,
    read_json,
    safe_torch_load,
    sha256_file,
    validate_embedding_record,
    validate_motion_record,
)

DEFAULT_MOTION_ROOT = Path("/data0/user/liwei/datasets/MotionMillion/genmo_smpl_v1")
DEFAULT_EMBEDDING_ROOT = Path("/data0/user/liwei/datasets/MotionMillion/t5_3b_v1_fp16")


def _audit_release(
    motion_root: Path,
    embedding_root: Path,
    *,
    verify_sha256: bool,
    max_shards: int | None,
    sequence_mode: str = "crop",
    pad_to_frames: int = 120,
) -> dict[str, Any]:
    """审计 release 闭环并返回 split/尺度/文本统计。"""
    motion_release = read_json(motion_root / "dataset_release.json")
    embedding_release = read_json(embedding_root / "embedding_release.json")
    if int(motion_release.get("schema_version", -1)) != SCHEMA_VERSION:
        raise MotionMillionError("motion release schema 不兼容")
    if int(embedding_release.get("schema_version", -1)) != SCHEMA_VERSION:
        raise MotionMillionError("embedding release schema 不兼容")
    if motion_release.get("build_fingerprint") != embedding_release.get(
        "source_build_fingerprint"
    ):
        raise MotionMillionError("motion/embedding release fingerprint 不一致")

    split_ids: dict[str, set[str]] = {split: set() for split in SPLITS}
    base_to_splits: dict[str, set[str]] = defaultdict(set)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "build_fingerprint": motion_release["build_fingerprint"],
        "embedding_revision": embedding_release.get("resolved_revision"),
        "verified_sha256": verify_sha256,
        "partial_max_shards": max_shards,
        "splits": {},
    }
    all_root_heights: list[torch.Tensor] = []
    all_root_speeds: list[torch.Tensor] = []
    all_pose_norms: list[torch.Tensor] = []

    for split in SPLITS:
        motion_manifest = read_json(motion_root / "manifests" / f"{split}.json")
        embedding_manifest = read_json(
            embedding_root / "manifests" / f"{split}.json"
        )
        if sequence_mode == "full":
            for name, manifest in (("motion", motion_manifest), ("embedding", embedding_manifest)):
                if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("split") != split:
                    raise MotionMillionError(f"{split}: {name} manifest schema/split不一致")
            if (motion_manifest.get("build_fingerprint") != motion_release["build_fingerprint"]
                    or embedding_manifest.get("source_build_fingerprint") != motion_release["build_fingerprint"]):
                raise MotionMillionError(f"{split}: manifest/release数据身份不一致")
            if (motion_manifest.get("fps"), motion_manifest.get("source_up_axis"), motion_manifest.get("motion_frames")) != (30, "y", 120):
                raise MotionMillionError("A0预检要求原v1 release的30 FPS / Y-up / motion_frames=120")
            if (embedding_manifest.get("max_text_tokens"), embedding_manifest.get("hidden_dim")) != (150, 1024):
                raise MotionMillionError("A0预检要求150-token/1024D文本契约")
            if len(motion_manifest["shards"]) != len(embedding_manifest["shards"]):
                raise MotionMillionError(f"{split}:完整motion/T5 shard数量不一致")
        motion_shards = list(motion_manifest["shards"])
        embedding_shards = list(embedding_manifest["shards"])
        if max_shards is not None:
            motion_shards = motion_shards[:max_shards]
            embedding_shards = embedding_shards[:max_shards]
        if len(motion_shards) != len(embedding_shards):
            raise MotionMillionError(f"{split}: motion/embedding shard 数不一致")

        index = np.load(
            motion_root / str(motion_manifest["sample_index_path"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        if index.dtype != SAMPLE_INDEX_DTYPE or index.ndim != 1:
            raise MotionMillionError(f"{split}: sample index dtype/shape 不符合契约")
        if max_shards is None and len(index) != int(motion_manifest["sample_count"]):
            raise MotionMillionError(f"{split}: sample index 数量与 manifest 不一致")

        counters = defaultdict(int)
        checked_lengths = []
        for expected_shard, (motion_meta, embedding_meta) in enumerate(
            zip(motion_shards, embedding_shards)
        ):
            if int(motion_meta["shard_id"]) != expected_shard or int(
                embedding_meta["shard_id"]
            ) != expected_shard:
                raise MotionMillionError(f"{split}: shard_id 不连续")
            motion_path = motion_root / str(motion_meta["path"])
            embedding_path = embedding_root / str(embedding_meta["path"])
            if (embedding_meta.get("source_motion_path") != motion_meta["path"]
                    or embedding_meta.get("source_motion_sha256") != motion_meta["sha256"]):
                raise MotionMillionError(f"{split}: motion/T5 shard SHA绑定不一致")
            if verify_sha256:
                if sha256_file(motion_path) != motion_meta["sha256"]:
                    raise MotionMillionError(f"motion shard SHA256 不一致: {motion_path}")
                if sha256_file(embedding_path) != embedding_meta["sha256"]:
                    raise MotionMillionError(
                        f"embedding shard SHA256 不一致: {embedding_path}"
                    )
            motions = safe_torch_load(motion_path)
            embeddings = safe_torch_load(embedding_path)
            if not isinstance(motions, list) or not isinstance(embeddings, list):
                raise MotionMillionError(f"{split}:{expected_shard} shard 必须为 list")
            if len(motions) != len(embeddings):
                raise MotionMillionError(f"{split}:{expected_shard} record 数不一致")
            if sequence_mode == "full" and (
                len(motions) != int(motion_meta["record_count"])
                or len(embeddings) != int(embedding_meta["record_count"])
            ):
                raise MotionMillionError("实际shard记录数与manifest不一致")
            shard_index = index[index["shard_id"] == expected_shard]
            if not np.array_equal(np.sort(shard_index["record_index"]), np.arange(len(motions))):
                raise MotionMillionError("sample index 未恰好覆盖每条record一次")
            indexed_lengths = {int(row["record_index"]): int(row["frames"]) for row in shard_index}
            if (shard_index["window_index"] != 0).any():
                raise MotionMillionError("v1不接受分窗重复索引")
            for record_index, (motion, embedding) in enumerate(zip(motions, embeddings)):
                validate_motion_record(motion)
                validate_embedding_record(
                    embedding, caption_count=len(motion["captions"])
                )
                motion_id = str(motion["motion_id"])
                if motion["split"] != split or indexed_lengths[record_index] != len(motion["pose"]):
                    raise MotionMillionError("sample index frames / split 与实际pose/trans不一致")
                checked_lengths.append(len(motion["pose"]))
                if motion_id != str(embedding.get("motion_id")):
                    raise MotionMillionError(f"{split}: motion/embedding ID 顺序不一致")
                if motion_id in split_ids[split]:
                    raise MotionMillionError(f"{split}: 重复 motion ID {motion_id}")
                split_ids[split].add(motion_id)
                base_to_splits[mirror_base_id(motion_id)].add(split)
                counters["records"] += 1
                counters["captions"] += len(motion["captions"])
                counters["valid_tokens"] += int(embedding["embeddings"].shape[0])
                all_root_heights.append(motion["trans"][:, 1])
                all_root_speeds.append(
                    torch.linalg.vector_norm(motion["trans"][1:] - motion["trans"][:-1], dim=-1)
                    * 30.0
                )
                all_pose_norms.append(
                    torch.linalg.vector_norm(motion["pose"].reshape(-1, 22, 3), dim=-1).flatten()
                )
        report["splits"][split] = {
            "records_checked": counters["records"],
            "captions_checked": counters["captions"],
            "valid_tokens_checked": counters["valid_tokens"],
            "shards_checked": len(motion_shards),
            "manifest_records": int(motion_manifest["record_count"]),
            "manifest_samples": int(motion_manifest["sample_count"]),
            "sequence_statistics": sequence_statistics(checked_lengths, sequence_mode, pad_to_frames),
        }

    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlap = split_ids[left] & split_ids[right]
            if overlap:
                raise MotionMillionError(
                    f"split ID 泄漏 {left}/{right}: {sorted(overlap)[:20]}"
                )
    mirror_leaks = [base for base, splits in base_to_splits.items() if len(splits) > 1]
    if mirror_leaks:
        raise MotionMillionError(f"镜像 base ID 跨 split: {sorted(mirror_leaks)[:20]}")

    def tensor_stats(values: list[torch.Tensor]) -> dict[str, float]:
        if not values:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
        value = torch.cat(values).float()
        return {
            "min": float(value.min()),
            "max": float(value.max()),
            "mean": float(value.mean()),
            "std": float(value.std(unbiased=False)),
        }

    report["physical_statistics"] = {
        "root_height_m": tensor_stats(all_root_heights),
        "root_speed_mps": tensor_stats(all_root_speeds),
        "joint_axis_angle_norm_rad": tensor_stats(all_pose_norms),
    }
    return report


def sequence_statistics(lengths, sequence_mode, pad_to_frames):
    """按已实际读取的record统计有效帧、padding及裁剪；不把索引扫描说成制品核验。"""
    if sequence_mode not in {"crop", "full"} or pad_to_frames <= 0:
        raise ValueError("无效序列预检配置")
    values = np.asarray(lengths, dtype=np.int64)
    if len(values) == 0 or ((values < 60) | (values > 300)).any():
        raise MotionMillionError("实际动作长度不在60—300帧")
    clipped = int((values > pad_to_frames).sum())
    if sequence_mode == "full" and (pad_to_frames != 300 or clipped):
        raise MotionMillionError("full 模式必须pad300且裁剪计数为零")
    effective = values if sequence_mode == "full" else np.minimum(values, pad_to_frames)
    return {
        "sequence_mode": sequence_mode, "pad_to_frames": pad_to_frames,
        "source_frames": int(values.sum()), "valid_frames": int(effective.sum()),
        "padding_frames": int(len(values) * pad_to_frames - effective.sum()),
        "padding_ratio": float(1 - effective.sum() / (len(values) * pad_to_frames)),
        "crop_count": clipped,
        "min": int(values.min()), "max": int(values.max()), "mean": float(values.mean()),
        "percentiles_0_25_50_75_100": np.percentile(values, [0, 25, 50, 75, 100]).tolist(),
    }


def _normalized_151d_statistics(
    motion_root: Path,
    embedding_root: Path,
    *,
    sample_count: int,
    z_threshold: float,
    sequence_mode: str = "crop",
    pad_to_frames: int = 120,
) -> dict[str, Any]:
    """实际执行 Dataset→EnDecoder，并统计现有 151D 归一化契约。"""
    from gem.datasets.pure_motion.motionmillion import MotionMillionDataset
    from gem.network.endecoder import EnDecoder

    torch.manual_seed(20260909)
    np.random.seed(20260909)
    dataset = MotionMillionDataset(
        root=motion_root.parent,
        split="train",
        motion_frames=120,
        motion_manifest_path=motion_root / "manifests" / "train.json",
        embedding_manifest_path=embedding_root / "manifests" / "train.json",
        shard_cache_size=2,
        random_seed=20260909,
        source_up_axis="y",
        cam_augmentation="static",
        random_crop=False if sequence_mode == "crop" else None,
        sequence_mode=sequence_mode,
        pad_to_frames=pad_to_frames,
        caption_sampling="first",
    )
    count = min(int(sample_count), len(dataset))
    indices = np.linspace(0, len(dataset) - 1, count, dtype=np.int64)
    endecoder = EnDecoder(
        stats_name="MM_V1_AMASS_LOCAL_BEDLAM_CAM",
        feat_dim=151,
        encode_type="gvhmr",
        clip_std=True,
    ).eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, count, 8):
            batch = default_collate([dataset[int(index)] for index in indices[start : start + 8]])
            batch["valid_length_processing"] = sequence_mode == "full"
            encoded = endecoder.encode(batch)
            valid = batch["mask"]["valid"].bool()
            chunks.append(encoded[valid].cpu())
    values = torch.cat(chunks, dim=0)
    means = values.mean(dim=0)
    stds = values.std(dim=0, unbiased=False)
    fractions = (values.abs() > z_threshold).float().mean(dim=0)
    return {
        "stats_name": "MM_V1_AMASS_LOCAL_BEDLAM_CAM",
        "clip_std": True,
        "sample_count": count,
        "valid_frame_count": len(values),
        "z_threshold": z_threshold,
        "max_abs_z": float(values.abs().max()),
        "max_channel_outlier_fraction": float(fractions.max()),
        "channels": [
            {
                "index": index,
                "mean_z": float(means[index]),
                "std_z": float(stds[index]),
                "outlier_fraction": float(fractions[index]),
            }
            for index in range(values.shape[1])
        ],
    }


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    motion_root = Path(args.motion_root).expanduser().resolve()
    embedding_root = Path(args.embedding_root).expanduser().resolve()
    report = _audit_release(
        motion_root,
        embedding_root,
        verify_sha256=args.verify_sha256,
        max_shards=args.max_shards,
        sequence_mode=getattr(args, "sequence_mode", "crop"),
        pad_to_frames=getattr(args, "pad_to_frames", 120),
    )
    if args.normalized_stats_samples > 0:
        normalized = _normalized_151d_statistics(
            motion_root,
            embedding_root,
            sample_count=args.normalized_stats_samples,
            z_threshold=args.z_threshold,
            sequence_mode=getattr(args, "sequence_mode", "crop"),
            pad_to_frames=getattr(args, "pad_to_frames", 120),
        )
        report["normalized_151d"] = normalized
        if normalized["max_channel_outlier_fraction"] > args.max_outlier_fraction:
            raise MotionMillionError(
                "151D 统计预检失败：最大通道离群比例 "
                f"{normalized['max_channel_outlier_fraction']:.6f} > "
                f"{args.max_outlier_fraction:.6f}"
            )
    report["elapsed_seconds"] = time.monotonic() - started
    report["status"] = "PASS"
    report_path = (
        Path(args.report).expanduser().resolve()
        if args.report is not None
        else motion_root / "reports" / "preflight_motionmillion.json"
    )
    atomic_write_json(report_path, report)
    report["report_path"] = str(report_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-root", type=Path, default=DEFAULT_MOTION_ROOT)
    parser.add_argument("--embedding-root", type=Path, default=DEFAULT_EMBEDDING_ROOT)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--verify-sha256", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--sequence-mode", choices=("crop", "full"), default="crop")
    parser.add_argument("--pad-to-frames", type=int, default=120)
    parser.add_argument("--normalized-stats-samples", type=int, default=1000)
    parser.add_argument("--z-threshold", type=float, default=8.0)
    parser.add_argument("--max-outlier-fraction", type=float, default=0.25)
    parser.add_argument("--config-only", action="store_true", help="只解析A0配置，不读真实分片、不建立模型、不访问GPU")
    parser.add_argument("--config-override", action="append", default=[], help="Hydra配置覆盖，可重复传入")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.config_only:
        import builtins
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        from gem.utils.sequence_contract import validate_sequence_experiment

        OmegaConf.register_new_resolver("eval", builtins.eval, replace=True)
        with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base="1.3"):
            cfg = compose(config_name="train", overrides=["exp=gem_smpl_motionmillion_text_fullseq", *args.config_override])
        contract = validate_sequence_experiment(cfg)
        print(json.dumps({"status": "CONFIG_PASS", "sequence_contract": contract,
                          "actual_release_verified": False,
                          "effective_global_batch_candidate": cfg.data.loader_opts.train.batch_size * cfg.pl_trainer.devices * cfg.pl_trainer.accumulate_grad_batches,
                          "gpu_memory_verified": False}, ensure_ascii=False, indent=2))
        if args.report is not None:
            # 可选保存完整解析配置，后续指标 --experiment-config 使用同一快照。
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(cfg, args.report, resolve=True)
        return
    if args.max_shards is not None and args.max_shards <= 0:
        raise SystemExit("max-shards 必须为正数")
    if args.normalized_stats_samples < 0 or args.z_threshold <= 0:
        raise SystemExit("normalized-stats-samples 不能为负，z-threshold 必须为正")
    if not 0 <= args.max_outlier_fraction <= 1:
        raise SystemExit("max-outlier-fraction 必须位于 [0,1]")
    report = run_preflight(args)
    print(
        f"MotionMillion preflight PASS: report={report['report_path']}, "
        f"elapsed={report['elapsed_seconds']:.2f}s"
    )


if __name__ == "__main__":
    main()
