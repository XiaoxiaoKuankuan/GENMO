#!/usr/bin/env python3
"""BUMI 文本数据接入、预检和 train-only 统计量工具。

build 读取 conversion.json 中的原生 NPZ 与文本特征引用，验证后形成完整动作分片；
按来源区间排除 train/held-out 重叠及精确重复，原 NPZ/T5 文件不改写。不执行重定向、
人体转换或自动地面修正。stats 只遍历 train，有效XY差分不包含每条动作最后一帧。
preflight 报告真实长度、padding和裁剪计数。所有写入要求显式输出路径且拒绝覆盖。
filter-umr 接入MotionMillion/HumanML3D原生UMR的全量数值/质量筛选与可恢复报告。
humanml-conversion仅从完整PASS报告和逐字匹配的原文本生成训练派生清单，不创造验证集。
build可通过
--quality-report只接收完整报告中的PASS，并核对UMR源身份、双输入SHA和完整文本对应。
此模式不裁剪长动作、不自动生成caption或split，构建成功后同盘原子发布release。
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from gem.datasets.pure_motion.bumi_text import (
    SCHEMA,
    AssetCache,
    BumiTextDataset,
    read_embedding,
    resolve_reference,
    validate_record,
)
from gem.robots.bumi.endecoder import STATS_CONTRACT_VERSION
from gem.robots.bumi.feature_codec import (
    BUMI_ANCHOR_MODE,
    BUMI_FEATURE_SLICES,
    BUMI_REPRESENTATION_CONTRACT_VERSION,
    BumiMotionFeatureCodec,
)
from gem.robots.bumi.kinematics import BumiKinematics
from gem.runtime.bumi_text_contract import MJCF_SHA256, sha256_file


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
            # 镜像共享防泄漏分组，但属于不同训练样本，不能按时间区间误删镜像。
            identity = (item["split"], item["provenance"]["source_id"], a, b)
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


def humanml_conversion(quality_report, output):
    """把完整HumanML3D PASS报告与原caption绑定，供T5编码及现有build入口使用。"""
    from tools.data.bumi.umr_text_preprocess import QualityGate, humanml_catalog

    gate = QualityGate(quality_report)
    try:
        if gate.paths.get("dataset") != "humanml3d":
            raise ValueError("需要HumanML3D的完整筛选报告")
        _, catalog = humanml_catalog(gate.paths)
        records = []
        for line in (gate.root / "train_candidates.jsonl").read_text().splitlines():
            candidate = json.loads(line)
            path = Path(gate.paths["input_root"]) / candidate["relative_path"]
            row = gate.lookup(path)
            if row["status"] != "PASS" or not row["training_eligible"]:
                raise ValueError("候选与质量数据库不一致")
            key = row["source_motion_id"]
            texts = catalog[key]["captions"]
            records.append(
                dict(
                    dataset="humanml3d",
                    motion_id=key,
                    text_source_motion_id=key,
                    split="train",
                    qpos_path=str(path),
                    fps=30,
                    captions=[c["caption"] for c in texts],
                    caption_ids=[f"{key}:{i}" for i in range(len(texts))],
                    embeddings=[],
                    provenance=dict(
                        source_id=key,
                        canonical_source_id=row["canonical_source_id"],
                        interval_seconds=row["interval_seconds"],
                        mirrored=row["mirrored"],
                        annotation_interval_seconds=row["annotation_interval_seconds"],
                        source_end_clipped=row["source_end_clipped"],
                        retargeter="UMR",
                        retarget_version=gate.run["fingerprint"],
                        source_dataset="GENMO HumanML3D training derivative",
                    ),
                )
            )
        if not records:
            raise ValueError("没有满足训练要求的PASS动作，拒绝发布空训练数据")
        payload = dict(
            schema="genmo.bumi_text_conversion.v1",
            kinematics=dict(
                path=gate.paths["kinematics"], sha256=sha256_file(gate.paths["kinematics"])
            ),
            quality_report=str(gate.root),
            split_policy="source_training_derivative_only_no_invented_heldout",
            records=records,
        )
        write_json(output, payload)
        return dict(
            records=len(records),
            captions=sum(len(r["captions"]) for r in records),
            output=str(Path(output).resolve()),
        )
    finally:
        gate.close()


def build(source, output, *, records_per_shard=512, quality_report=None):
    """所有模式在隔离staging构建，失败自动清理，完成后才发布正式release。"""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("使用新的release目录，不能覆盖原数据")
    output.parent.mkdir(parents=True, exist_ok=True)
    gate = None
    if quality_report is not None:
        from tools.data.bumi.umr_text_preprocess import QualityGate

        gate = QualityGate(quality_report)
    try:
        with tempfile.TemporaryDirectory(
            prefix=f".{output.name}.staging-", dir=output.parent
        ) as temp:
            staged = Path(temp) / "release"
            result = _build(source, staged, records_per_shard=records_per_shard, quality_gate=gate)
            if output.exists():
                raise FileExistsError("构建期间目标目录被创建，拒绝覆盖")
            staged.rename(output)
            return result
    finally:
        if gate is not None:
            gate.close()


def _build(source, output, *, records_per_shard=512, quality_gate=None):
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
    quality_excluded = []
    if quality_gate is not None:
        if kin.kinematics_sha256 != quality_gate.engine.kin.kinematics_sha256:
            raise ValueError("构建运动学与质量报告指纹不同")
        # 在来源分组之前绑定canonical ID，防止镜像跨split时沿用不完整的调用者ID。
        eligible = []
        for record in payload["records"]:
            path = resolve_reference(record["qpos_path"], source.parent)
            row = quality_gate.lookup(path)
            if not row["training_eligible"] or row["status"] != "PASS":
                quality_excluded.append(
                    dict(
                        dataset=record["dataset"],
                        motion_id=record["motion_id"],
                        reason="umr_quality_or_length",
                        status=row["status"],
                    )
                )
                continue
            quality_gate.validate_identity(row, record)
            record["provenance"]["canonical_source_id"] = row["canonical_source_id"]
            eligible.append(record)
        payload["records"] = eligible
    rows, report = select_records(payload["records"])
    report["excluded"].extend(quality_excluded)
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
            if quality_gate is not None:
                qpos, quality = quality_gate.read_candidate(path, record)
                if qpos is None:
                    report["excluded"].append(
                        dict(
                            dataset=record["dataset"],
                            motion_id=record["motion_id"],
                            reason="umr_quality_or_length",
                            status=quality["status"],
                        )
                    )
                    continue
                record["qpos"] = qpos
                record["ground_alignment"] = dict(
                    applied=False,
                    offset_z=0.0,
                    reference="UMR world-Z=0; source_ground_z is preprocessing metadata",
                )
                from gem.datasets.pure_motion.bumi_text import GROUND
                from gem.robots.bumi.contacts import derive_bumi_foot_contact

                record["ground_semantics"] = GROUND
                contact = derive_bumi_foot_contact(qpos, kin, ground_height=0.0)
                record["foot_contact"] = contact.contact
                record["foot_contact_available"] = contact.valid_mask
                record["quality_provenance"] = dict(
                    run_fingerprint=quality_gate.run["fingerprint"],
                    human_sha256=quality["human_sha256"],
                    status="PASS",
                )
            else:
                with np.load(path, allow_pickle=False) as npz:
                    if float(npz["fps"]) != 30 or list(npz["joint_names"].astype(str)) != list(
                        kin.joint_order
                    ):
                        raise ValueError(f"NPZ FPS/关节顺序错误: {path}")
                    record["qpos"] = torch.from_numpy(
                        np.asarray(npz["qpos"], dtype=np.float32).copy()
                    )
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
            quality_run_fingerprint=quality_gate.run["fingerprint"] if quality_gate else None,
        )
        write_json(output / "manifests" / f"{split}.json", manifest)
    report.update(counts=dict(counts), crop_count=0)
    write_json(output / "build_report.json", report)
    return report


def _report_dataset(root, split, dataset):
    """单来源release自动绑定单集身份，避免统计和预检报告误标为联合数据。"""
    ds = BumiTextDataset(root, split, dataset=dataset, caption_sampling="first")
    if dataset is None:
        available = {row[2]["dataset"] for row in ds.index}
        if len(available) == 1:
            # 单来源release必须写单集统计身份，否则BumiTextGEM会按联合实验拒绝加载。
            dataset = next(iter(available))
            ds = BumiTextDataset(root, split, dataset=dataset, caption_sampling="first")
    return ds


def statistics(root, output, dataset=None):
    ds = _report_dataset(root, "train", dataset)
    dataset = ds.dataset
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
        root_height_reference_m=float(codec.default_root_height),
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
    ds = _report_dataset(root, split, dataset)
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
    p.add_argument(
        "--quality-report", type=Path, help="完整UMR筛选报告目录；该模式读取原生UMR qpos"
    )
    # 只在执行filter时导入MuJoCo，原有build/stats/preflight保持原依赖边界。
    p = sub.add_parser("humanml-conversion", help="由完整HumanML3D PASS报告构建文本转换清单")
    p.add_argument("--quality-report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("filter-umr", help="全量筛选MotionMillion/HumanML3D UMR，支持断点续跑")
    p.add_argument("--dataset", choices=["motionmillion", "humanml3d"], default="motionmillion")
    p.add_argument("--recorded-output-root", type=Path, help="HumanML3D迁移前输出目录的显式映射")
    p.add_argument("--recorded-robot-xml", type=Path, help="经资产SHA核验后允许的旧XML绝对路径")
    for name in (
        "input-root",
        "source-root",
        "robot-xml",
        "retarget-config",
        "asset-manifest",
        "output",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--batch-config", type=Path)
    p.add_argument(
        "--config", type=Path, default=ROOT / "configs/bumi/quality_filter_umr_text_30hz_v1.yaml"
    )
    p.add_argument(
        "--kinematics",
        type=Path,
        default=ROOT / "configs/bumi/bumi_kinematics_robot_retargeter_fe934_v1.json",
    )
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--expected-records", type=int, help="校验全量清单条数，防止漏目录")
    p.add_argument("--folders", nargs="+", help="只检查指定folder；报告标记为partial")
    p.add_argument("--limit", type=int, help="有界验证；报告标记为partial")
    p.add_argument("--resume", action="store_true")
    for command in ("stats", "preflight"):
        p = sub.add_parser(command)
        p.add_argument("--root", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--dataset", choices=["motionmillion", "humanml3d"])
        if command == "preflight":
            p.add_argument("--split", default="train", choices=["train", "val", "test"])
            p.add_argument("--limit", type=int, default=128)
    args = parser.parse_args()
    if args.command == "filter-umr":
        from tools.data.bumi.umr_text_preprocess import run_filter

        raise SystemExit(run_filter(args))
    if args.command == "humanml-conversion":
        result = humanml_conversion(args.quality_report, args.output)
    elif args.command == "build":
        result = build(
            args.source,
            args.output,
            records_per_shard=args.records_per_shard,
            quality_report=args.quality_report,
        )
    elif args.command == "stats":
        result = statistics(args.root, args.output, args.dataset)
    else:
        result = preflight(args.root, args.split, args.dataset, args.limit)
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
