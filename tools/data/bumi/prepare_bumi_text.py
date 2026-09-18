#!/usr/bin/env python3
"""BUMI 文本数据接入、预检和 train-only 统计量工具。

build 读取 conversion.json 中的原生 NPZ 与文本特征引用，验证后形成完整动作分片；
按来源区间排除 train/held-out 重叠及精确重复，原 NPZ/T5 文件不改写。不执行重定向、
人体转换或自动地面修正。stats 只遍历 train，有效XY差分不包含每条动作最后一帧。
preflight 报告真实长度、padding和裁剪计数。所有写入要求显式输出路径且拒绝覆盖。
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from gem.datasets.pure_motion.bumi_text import (
    AssetCache,
    BumiTextDataset,
    SCHEMA,
    read_embedding,
    resolve_reference,
    validate_record,
)
from gem.runtime.bumi_text_contract import MJCF_SHA256, sha256_file
from gem.robots.bumi.kinematics import BumiKinematics
from gem.robots.bumi.feature_codec import (
    BumiMotionFeatureCodec,
    BUMI_FEATURE_SLICES,
    BUMI_ANCHOR_MODE,
    BUMI_REPRESENTATION_CONTRACT_VERSION,
)
from gem.robots.bumi.endecoder import STATS_CONTRACT_VERSION


def write_json(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"拒绝覆盖已有产物: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    pending.replace(path)


def select_records(records):
    """保留原split；有来源重叠时优先保留held-out，未确认跨库来源单独报告。"""
    groups = defaultdict(list)
    rejected, accepted, unknown = [], [], []
    for item in records:
        p = item["provenance"]
        key = p.get("canonical_source_id")
        if not key:
            key = item["dataset"] + ":" + p["source_id"]
            unknown.append([item["dataset"], item["motion_id"]])
        groups[key].append(item)
    for group in groups.values():
        held = [r["provenance"]["interval_seconds"] for r in group if r["split"] != "train"]
        seen = set()
        for item in sorted(
            group, key=lambda r: (r["split"] == "train", r["dataset"], r["motion_id"])
        ):
            a, b = item["provenance"]["interval_seconds"]
            reason = None
            if item["split"] == "train" and any(max(a, c) < min(b, d) for c, d in held):
                reason = "train_overlaps_held_out"
            identity = (item["split"], a, b)
            if identity in seen:
                reason = "duplicate_source_interval"
            if reason:
                rejected.append(
                    dict(dataset=item["dataset"], motion_id=item["motion_id"], reason=reason)
                )
            else:
                accepted.append(item)
                seen.add(identity)
    return accepted, {"excluded": rejected, "unverified_cross_dataset_lineage": unknown}


def build(source, output, *, records_per_shard=512):
    source, output = Path(source).resolve(), Path(output).resolve()
    if records_per_shard < 1:
        raise ValueError("records_per_shard必须为正")
    if output.exists():
        raise FileExistsError("使用新的release目录，不能覆盖原数据")
    payload = json.loads(source.read_text())
    if payload.get("schema") != "genmo.bumi_text_conversion.v1":
        raise ValueError("需要 conversion.v1 清单")
    kin_path = resolve_reference(payload["kinematics"]["path"], source.parent)
    if sha256_file(kin_path) != payload["kinematics"]["sha256"]:
        raise ValueError("转换清单 kinematics SHA错误")
    kin = BumiKinematics(kin_path)
    if kin.source_mjcf_sha256 != MJCF_SHA256:
        raise ValueError("仅接受fe934 BUMI")
    rows, report = select_records(payload["records"])
    output.mkdir(parents=True)
    (output / "shards").mkdir()
    # 固定资产复制一份到新release，源文件与原fingerprint保持不变。
    (output / "kinematics.json").write_bytes(kin_path.read_bytes())
    cache = AssetCache(4)
    counts = Counter()
    for split in ("train", "val", "test"):
        shards, chunk = [], []

        def flush():
            if not chunk:
                return
            sid = len(shards)
            path = output / "shards" / f"{split}_{sid:05d}.pt"
            torch.save(chunk, path)
            shards.append(
                dict(
                    shard_id=sid,
                    path=str(path.relative_to(output)),
                    sha256=sha256_file(path),
                    record_count=len(chunk),
                    records=[
                        dict(
                            record_index=i,
                            frames=r["frames"],
                            dataset=r["dataset"],
                            motion_id=r["motion_id"],
                        )
                        for i, r in enumerate(chunk)
                    ],
                )
            )
            chunk.clear()

        for original in rows:
            if original["split"] != split:
                continue
            record = copy.deepcopy(original)
            path = resolve_reference(record.pop("qpos_path"), source.parent)
            with np.load(path, allow_pickle=False) as npz:
                if float(npz["fps"]) != 30 or list(npz["joint_names"].astype(str)) != list(
                    kin.joint_order
                ):
                    raise ValueError(f"NPZ FPS/关节顺序错误: {path}")
                record["qpos"] = torch.from_numpy(np.asarray(npz["qpos"], dtype=np.float32).copy())
                for key in ("foot_contact", "foot_contact_available"):
                    if key in npz:
                        record[key] = torch.from_numpy(npz[key].copy())
            record["source_qpos_sha256"] = sha256_file(path)
            record["frames"] = len(record["qpos"])
            if not 60 <= record["frames"] <= 300:
                report["excluded"].append(
                    dict(
                        dataset=record["dataset"],
                        motion_id=record["motion_id"],
                        reason="length_outside_60_300",
                        frames=record["frames"],
                    )
                )
                continue
            validate_record(record, split=split)
            if record.get("crop_start", 0) != 0 or record.get("sequence_mode", "full") != "full":
                raise ValueError("full release不接受训练时裁剪记录")
            for caption, ref in zip(record["captions"], record["embeddings"]):
                for key in ("path", "motion_manifest", "embedding_manifest"):
                    if key in ref:
                        ref[key] = str(resolve_reference(ref[key], source.parent))
                read_embedding(ref, caption, cache, output, expected_frames=record["frames"])
            chunk.append(record)
            counts[f"{split}/{record['dataset']}"] += 1
            if len(chunk) >= records_per_shard:
                flush()
        flush()
        manifest = dict(
            schema=SCHEMA,
            split=split,
            fps=30,
            qpos_dim=28,
            quaternion_convention="wxyz",
            coordinate_system="z_up",
            source_mjcf_sha256=MJCF_SHA256,
            joint_names=list(kin.joint_order),
            kinematics={"path": "kinematics.json", "sha256": kin.kinematics_sha256},
            source_conversion_sha256=sha256_file(source),
            shards=shards,
            full_sequence=True,
            crop_count=0,
        )
        write_json(output / "manifests" / f"{split}.json", manifest)
    report.update(counts=dict(counts), crop_count=0)
    write_json(output / "build_report.json", report)
    return report


def statistics(root, output, dataset=None):
    ds = BumiTextDataset(root, "train", dataset=dataset, caption_sampling="first")
    kin_path = resolve_reference(ds.manifest["kinematics"]["path"], ds.root)
    kin = BumiKinematics(kin_path)
    codec = BumiMotionFeatureCodec(kin)
    sums, squares, counts = (torch.zeros(30, dtype=torch.float64) for _ in range(3))
    frames_total = 0
    for i in range(len(ds)):
        record = ds.read_record(i)
        features = codec.encode(record["qpos"]).physical_features.double()
        mask = torch.ones_like(features, dtype=torch.bool)
        mask[-1, :2] = False
        sums += torch.where(mask, features, 0).sum(0)
        squares += torch.where(mask, features.square(), 0).sum(0)
        counts += mask.sum(0)
        frames_total += len(features)
    mean = sums / counts
    std = (squares / counts - mean.square()).clamp_min(0).sqrt()
    value = dict(
        contract_version=STATS_CONTRACT_VERSION,
        representation_contract_version=BUMI_REPRESENTATION_CONTRACT_VERSION,
        robot_name="bumi",
        feature_dim=30,
        anchor_mode=BUMI_ANCHOR_MODE,
        quaternion_convention="wxyz",
        training_clip_std_min=0.01,
        feature_slices=dict(BUMI_FEATURE_SLICES),
        joint_names=list(kin.joint_order),
        kinematics_sha256=kin.kinematics_sha256,
        data_kind="bumi_text_fullseq",
        split="train",
        dataset=dataset,
        data_identity=ds.data_identity,
        mean=mean.tolist(),
        std=std.tolist(),
        valid_element_counts=counts.int().tolist(),
        records=len(ds),
        valid_frames=frames_total,
        is_placeholder=False,
    )
    write_json(output, value)
    return value


def preflight(root, split="train", dataset=None, limit=128):
    if limit < 0:
        raise ValueError("limit不能为负；0表示全量")
    ds = BumiTextDataset(root, split, dataset=dataset, caption_sampling="first")
    lengths = []
    posture_diagnostics = []
    for i in range(min(limit, len(ds)) if limit else len(ds)):
        record = ds.read_record(i)
        for text, ref in zip(record["captions"], record["embeddings"]):
            read_embedding(ref, text, ds.cache, ds.root, expected_frames=record["frames"])
        sample = ds[i]
        if sample["meta"]["crop_start"] != 0 or sample["mask"]["valid"].sum() != sample["length"]:
            raise ValueError("发现裁剪或真实长度错误")
        lengths.append(sample["length"])
        from gem.utils.rotation_conversions import quaternion_to_matrix

        tilt = torch.acos(quaternion_to_matrix(record["qpos"][:, 3:7])[:, 2, 2].clamp(-1, 1))
        minimum = float(record["qpos"][:, 2].min())
        if float(tilt.max()) > 0.8 or minimum < 0.3:
            posture_diagnostics.append(
                dict(
                    dataset=record["dataset"],
                    motion_id=record["motion_id"],
                    min_root_height_m=minimum,
                    max_root_tilt_rad=float(tilt.max()),
                    action="report_only_no_standing_filter",
                )
            )
    if ds.manifest.get("crop_count") != 0 or not ds.manifest.get("full_sequence"):
        raise ValueError("manifest裁剪计数非零或缺少完整动作声明")
    return dict(
        records_checked=len(lengths),
        records_total=len(ds),
        length_distribution=dict(Counter(lengths)),
        valid_frames=sum(lengths),
        padding_frames=300 * len(lengths) - sum(lengths),
        padding_fraction=1 - sum(lengths) / (300 * len(lengths)),
        crop_count=0,
        data_identity=ds.data_identity,
        low_or_tilted_postures=posture_diagnostics,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--records-per-shard", type=int, default=512)
    for command in ("stats", "preflight"):
        p = sub.add_parser(command)
        p.add_argument("--root", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--dataset", choices=["motionmillion", "humanml3d"])
        if command == "preflight":
            p.add_argument("--split", default="train", choices=["train", "val", "test"])
            p.add_argument("--limit", type=int, default=128)
    args = parser.parse_args()
    if args.command == "build":
        result = build(args.source, args.output, records_per_shard=args.records_per_shard)
    elif args.command == "stats":
        result = statistics(args.root, args.output, args.dataset)
    else:
        result = preflight(args.root, args.split, args.dataset, args.limit)
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
