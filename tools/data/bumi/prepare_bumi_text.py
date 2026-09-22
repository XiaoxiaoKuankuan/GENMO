#!/usr/bin/env python3
"""BUMI 文本数据接入、预检和 train-only 统计量工具。

build 读取 conversion.json 中的原生 NPZ 与文本特征引用，验证后形成完整动作分片；
按来源区间排除 train/held-out 重叠及精确重复，原 NPZ/T5 文件不改写。不执行重定向、
人体转换或自动地面修正。stats 只遍历 train，有效XY差分不包含每条动作最后一帧。
four-conversion接入四库完整PASS源动作和BONES真实事件时间；crop统计按确定性区间窗口计算。
preflight 报告真实长度、padding和源文件裁剪计数（训练随机窗口不改源文件）。所有写入要求显式输出路径且拒绝覆盖。
filter-umr 接入MotionMillion/HumanML3D原生UMR的全量数值/质量筛选与可恢复报告。
umr-pass统一发布四库所有长度的PASS和原文本，支持只引用已校验的原始NPZ；
bones-pass/kitml-pass保持兼容，原始文本索引不随质量规则改变而重新生成。
humanml-conversion仅从完整PASS报告和逐字匹配的原文本生成训练派生清单，不创造验证集。
build可通过
--quality-report只接收完整报告中的PASS，并核对UMR源身份、双输入SHA和完整文本对应。
此模式不裁剪长动作、不自动生成caption或split，构建成功后同盘原子发布release。
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing
import sqlite3
import sys
import tempfile
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from gem.datasets.pure_motion.bumi_text import (
    SCHEMA,
    AssetCache,
    BumiTextDataset,
    caption_intervals,
    read_embedding,
    resolve_reference,
    validate_record,
    window_bounds,
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
        if not key or p.get("cross_dataset_lineage_verified") is False:
            unknown.append([item["dataset"], item["motion_id"]])
        if not key:
            key = item["dataset"] + ":" + p["source_id"]
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


def build(
    source,
    output,
    *,
    records_per_shard=512,
    quality_report=None,
    workers=1,
    text_feature_mode="precomputed",
):
    """所有模式在隔离staging构建，失败自动清理，完成后才发布正式release。"""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("使用新的release目录，不能覆盖原数据")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(Path(source).read_text())
    reports = payload.get("quality_reports", {})
    if quality_report is not None and reports:
        raise ValueError("联合转换已绑定四库质量报告，不能再覆盖单个report")
    with ExitStack() as resources:
        gates = {}
        for dataset, report in (
            reports or ({"*": quality_report} if quality_report else {})
        ).items():
            from tools.data.bumi.umr_text_preprocess import QualityGate

            gate = QualityGate(report)
            resources.callback(gate.close)
            gates[dataset] = gate
        with tempfile.TemporaryDirectory(
            prefix=f".{output.name}.staging-", dir=output.parent
        ) as temp:
            staged = Path(temp) / "release"
            result = _build(
                source,
                staged,
                records_per_shard=records_per_shard,
                quality_gates=gates,
                workers=workers,
                text_feature_mode=text_feature_mode,
            )
            if output.exists():
                raise FileExistsError("构建期间目标目录被创建，拒绝覆盖")
            staged.rename(output)
            return result


_BUILD_STATE = None


def _init_build_worker(gates, kin, base, all_lengths, text_mode):
    """每个子进程独立打开只读 SQLite；不跨进程复用父进程连接。"""
    global _BUILD_STATE
    torch.set_num_threads(1)
    for gate in gates.values():
        gate.db.close()
        path = quote(str(gate.root / "quality.sqlite"), safe="/")
        gate.db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    _BUILD_STATE = (gates, kin, base, all_lengths, text_mode)


def _prepare_record(original):
    """并行读取完整源动作并重复校验输入身份，接触标签仍由同一 FK 实现生成。"""
    gates, kin, base, all_lengths, text_mode = _BUILD_STATE
    record = copy.deepcopy(original)
    path = resolve_reference(record.pop("qpos_path"), base)
    gate = gates.get(record["dataset"], gates.get("*"))
    if gate is not None:
        qpos, quality = gate.read_candidate(path, record, allow_all_pass=all_lengths)
        if qpos is None:
            raise ValueError("分组后动作不再满足 PASS，禁止发布不完整 release")
        from gem.datasets.pure_motion.bumi_text import GROUND
        from gem.robots.bumi.contacts import derive_bumi_foot_contact

        record["qpos"] = qpos
        record["ground_alignment"] = dict(
            applied=False,
            offset_z=0.0,
            reference="UMR world-Z=0; source_ground_z is preprocessing metadata",
        )
        record["ground_semantics"] = GROUND
        contact = derive_bumi_foot_contact(qpos, kin, ground_height=0.0)
        record["foot_contact"], record["foot_contact_available"] = (
            contact.contact,
            contact.valid_mask,
        )
        record["quality_provenance"] = dict(
            run_fingerprint=gate.run["fingerprint"],
            human_sha256=quality["human_sha256"],
            status="PASS",
        )
    else:
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
    if text_mode == "online_t5":
        if record.get("embeddings"):
            raise ValueError("在线文本构建不接受未使用的预计算引用")
        record["text_feature_mode"] = text_mode
    validate_record(record, split=record["split"])
    # Pool 的 Tensor reducer 会为每条完整动作传递共享内存文件描述符；大规模流水线
    # 会耗尽默认 1024 个 FD。进程边界只传普通 NumPy，父进程写分片前再恢复 Tensor。
    return {
        key: value.numpy() if isinstance(value, torch.Tensor) else value
        for key, value in record.items()
    }


def _build(
    source,
    output,
    *,
    records_per_shard=512,
    quality_gates=None,
    workers=1,
    text_feature_mode="precomputed",
):
    source, output = Path(source).resolve(), Path(output).resolve()
    if records_per_shard < 1:
        raise ValueError("records_per_shard必须为正")
    if workers < 1 or text_feature_mode not in {"precomputed", "online_t5"}:
        raise ValueError("非法构建 workers 或文本模式")
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
    quality_gates = quality_gates or {}
    all_lengths = payload.get("source_storage") == "full"
    quality_excluded = []
    if quality_gates:
        if any(
            kin.kinematics_sha256 != gate.engine.kin.kinematics_sha256
            for gate in quality_gates.values()
        ):
            raise ValueError("构建运动学与质量报告指纹不同")
        # 在来源分组之前绑定canonical ID，防止镜像跨split时沿用不完整的调用者ID。
        eligible = []
        for record in payload["records"]:
            quality_gate = quality_gates.get(record["dataset"], quality_gates.get("*"))
            if quality_gate is None:
                raise ValueError("数据集缺少对应的质量报告")
            path = resolve_reference(record["qpos_path"], source.parent)
            row = quality_gate.lookup(path, verify_inputs=False)
            if row["status"] != "PASS" or (not all_lengths and not row["training_eligible"]):
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
    global _BUILD_STATE
    _BUILD_STATE = (quality_gates, kin, source.parent, all_lengths, text_feature_mode)
    with ExitStack() as resources:
        pool = None
        if workers > 1:
            pool = resources.enter_context(
                multiprocessing.get_context("fork").Pool(
                    workers,
                    initializer=_init_build_worker,
                    initargs=_BUILD_STATE,
                )
            )
        return _write_release(
            rows,
            report,
            counts,
            pool,
            output,
            source,
            kin,
            quality_gates,
            all_lengths,
            records_per_shard,
            cache,
            text_feature_mode,
        )


def _write_release(
    rows,
    report,
    counts,
    pool,
    output,
    source,
    kin,
    quality_gates,
    all_lengths,
    records_per_shard,
    cache,
    text_feature_mode,
):
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
                            canonical_source_id=r["provenance"].get("canonical_source_id"),
                        )
                        for i, r in enumerate(chunk)
                    ],
                )
            )
            chunk.clear()

        originals = (r for r in rows if r["split"] == split)
        prepared = (
            pool.imap(_prepare_record, originals, chunksize=8)
            if pool
            else map(_prepare_record, originals)
        )
        for record in prepared:
            for key in ("qpos", "foot_contact", "foot_contact_available"):
                if key in record and isinstance(record[key], np.ndarray):
                    record[key] = torch.from_numpy(record[key])
            if record["frames"] < 4 or (not all_lengths and not 60 <= record["frames"] <= 300):
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
            if sum(counts.values()) % 5000 == 0:
                print(
                    json.dumps(
                        dict(stage="build", records=sum(counts.values()), counts=dict(counts))
                    ),
                    flush=True,
                )
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
            text_feature_mode=text_feature_mode,
            quality_run_fingerprints={
                key: gate.run["fingerprint"] for key, gate in quality_gates.items()
            },
        )
        write_json(output / "manifests" / f"{split}.json", manifest)
    report.update(counts=dict(counts), crop_count=0)
    write_json(output / "build_report.json", report)
    return report


def _report_dataset(root, split, dataset, sequence_mode="full"):
    opts = dict(sequence_mode=sequence_mode, pad_to_frames=120 if sequence_mode == "crop" else 300)
    """单来源release自动绑定单集身份，避免统计和预检报告误标为联合数据。"""
    ds = BumiTextDataset(root, split, dataset=dataset, caption_sampling="first", **opts)
    if dataset is None:
        available = {row[2]["dataset"] for row in ds.index}
        if len(available) == 1:
            # 单来源release必须写单集统计身份，否则BumiTextGEM会按联合实验拒绝加载。
            dataset = next(iter(available))
            ds = BumiTextDataset(root, split, dataset=dataset, caption_sampling="first", **opts)
    return ds


_STATS_STATE = None


def _statistics_chunk(indices):
    ds, codec, sequence_mode = _STATS_STATE
    torch.set_num_threads(1)
    sums, squares, counts = (torch.zeros(30, dtype=torch.float64) for _ in range(3))
    frames_total = 0
    for i in indices:
        record = ds.read_record(i)
        intervals = (
            caption_intervals(record) if sequence_mode == "crop" else [[0, record["frames"]]]
        )
        seen = set()
        for tid, interval in enumerate(intervals):
            if tuple(interval) in seen:
                continue
            seen.add(tuple(interval))
            a, b = window_bounds(record, tid, sequence_mode, ds.pad_to_frames)
            features = codec.encode(record["qpos"][a:b]).physical_features.double()
            mask = torch.ones_like(features, dtype=torch.bool)
            mask[-1, :2] = False
            sums += torch.where(mask, features, 0).sum(0)
            squares += torch.where(mask, features.square(), 0).sum(0)
            counts += mask.sum(0)
            frames_total += len(features)
    return sums.numpy(), squares.numpy(), counts.numpy(), frames_total


def statistics(root, output, dataset=None, sequence_mode="full", workers=1):
    ds = _report_dataset(root, "train", dataset, sequence_mode)
    dataset = ds.dataset
    kin_path = resolve_reference(ds.manifest["kinematics"]["path"], ds.root)
    kin = BumiKinematics(kin_path)
    codec = BumiMotionFeatureCodec(kin)
    sums, squares, counts = (torch.zeros(30, dtype=torch.float64) for _ in range(3))
    frames_total = 0
    if workers < 1:
        raise ValueError("stats workers 必须为正")
    global _STATS_STATE
    _STATS_STATE = (ds, codec, sequence_mode)
    chunks = (range(i, min(i + 256, len(ds))) for i in range(0, len(ds), 256))
    with ExitStack() as resources:
        pool = (
            resources.enter_context(multiprocessing.get_context("fork").Pool(workers))
            if workers > 1
            else None
        )
        results = pool.imap(_statistics_chunk, chunks) if pool else map(_statistics_chunk, chunks)
        for index, (s, sq, c, frames) in enumerate(results):
            sums += torch.from_numpy(s)
            squares += torch.from_numpy(sq)
            counts += torch.from_numpy(c)
            frames_total += frames
            if (index + 1) % 40 == 0:
                print(
                    json.dumps(dict(stage="stats", records=min((index + 1) * 256, len(ds)))),
                    flush=True,
                )
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
        data_kind="bumi_text_crop120" if sequence_mode == "crop" else "bumi_text_fullseq",
        datasets=sorted({row[2]["dataset"] for row in ds.index}),
        window_policy="unique_annotation_intervals_center_window"
        if sequence_mode == "crop"
        else "full",
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


def preflight(root, split="train", dataset=None, limit=128, sequence_mode="full"):
    if limit < 0:
        raise ValueError("limit不能为负；0表示全量")
    ds = _report_dataset(root, split, dataset, sequence_mode)
    lengths = []
    posture_diagnostics = []
    for i in range(min(limit, len(ds)) if limit else len(ds)):
        record = ds.read_record(i)
        for text, ref in zip(record["captions"], record["embeddings"]):
            read_embedding(ref, text, ds.cache, ds.root, expected_frames=record["frames"])
        sample = ds[i]
        if (sequence_mode == "full" and sample["meta"]["crop_start"] != 0) or sample["mask"][
            "valid"
        ].sum() != sample["length"]:
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
        padding_frames=ds.pad_to_frames * len(lengths) - sum(lengths),
        padding_fraction=1 - sum(lengths) / (ds.pad_to_frames * len(lengths)),
        crop_count=0,
        data_identity=ds.data_identity,
        low_or_tilted_postures=posture_diagnostics,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("four-conversion", help="保留四库完整源动作并接入BONES事件时间")
    p.add_argument("--releases", type=Path, nargs=4, required=True)
    p.add_argument("--bones-temporal", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split-seed", type=int, default=20260922)
    p = sub.add_parser("build")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--records-per-shard", type=int, default=512)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument(
        "--text-feature-mode", choices=["precomputed", "online_t5"], default="precomputed"
    )
    p.add_argument(
        "--quality-report", type=Path, help="完整UMR筛选报告目录；该模式读取原生UMR qpos"
    )
    p = sub.add_parser("motionmillion-texts", help="从官方归档构建原生UMR动作文本索引")
    p.add_argument("--quality-report", type=Path, required=True)
    p.add_argument("--texts-archive", type=Path, required=True)
    p.add_argument("--split-archive", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("bind-motionmillion-texts", help="核验动作来源并绑定新筛选报告文本")
    p.add_argument("--quality-report", type=Path, required=True)
    p.add_argument("--text-catalog", type=Path, required=True)
    # 只在执行filter时导入MuJoCo，原有build/stats/preflight保持原依赖边界。
    p = sub.add_parser("humanml-conversion", help="由完整HumanML3D PASS报告构建文本转换清单")
    p.add_argument("--quality-report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("bones-pass", help="发布BONES-SEED全部PASS原生轨迹并精确核验官方文本")
    p.add_argument("--quality-report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("kitml-pass", help="发布KIT-ML全部PASS原生轨迹和白名单原文本")
    p.add_argument("--quality-report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("umr-pass", help="统一发布四个UMR数据集的全部PASS与原文本")
    p.add_argument("--quality-report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--dataset", required=True, choices=["motionmillion", "humanml3d", "bones_seed", "kitml"]
    )
    p.add_argument("--text-catalog", type=Path, help="MotionMillion原始文本SQLite索引")
    p.add_argument(
        "--reference-only", action="store_true", help="清单引用已核验原始NPZ，避免跨盘复制"
    )
    p = sub.add_parser("filter-umr", help="全量筛选MotionMillion/HumanML3D/BONES-SEED/KIT-ML UMR")
    p.add_argument(
        "--dataset",
        choices=["motionmillion", "humanml3d", "bones_seed", "kitml"],
        default="motionmillion",
    )
    p.add_argument("--metadata-json", type=Path, help="KIT-ML metadata_ready.json完整动作白名单")
    p.add_argument("--metadata-csv", type=Path, help="BONES-SEED官方完整动作文本CSV")
    p.add_argument("--original-source-root", type=Path, help="BONES-SEED原始SMPL pickle目录")
    p.add_argument(
        "--recorded-output-root", type=Path, help="HumanML3D/BONES-SEED迁移前输出目录的显式映射"
    )
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
        p.add_argument("--dataset", choices=["motionmillion", "humanml3d", "kitml", "bones_seed"])
        p.add_argument("--sequence-mode", choices=["crop", "full"], default="crop")
        if command == "stats":
            p.add_argument("--workers", type=int, default=1)
        if command == "preflight":
            p.add_argument("--split", default="train", choices=["train", "val", "test"])
            p.add_argument("--limit", type=int, default=128)
    args = parser.parse_args()
    if args.command == "filter-umr":
        from tools.data.bumi.umr_text_preprocess import run_filter

        raise SystemExit(run_filter(args))
    if args.command in {"bones-pass", "kitml-pass", "umr-pass"}:
        from tools.data.bumi.umr_text_preprocess import publish_umr_text_pass

        result = publish_umr_text_pass(
            args.quality_report,
            args.output,
            dataset=args.dataset
            if args.command == "umr-pass"
            else "bones_seed"
            if args.command == "bones-pass"
            else "kitml",
            text_catalog=getattr(args, "text_catalog", None),
            reference_only=getattr(args, "reference_only", False),
        )
    elif args.command == "motionmillion-texts":
        from tools.data.bumi.motionmillion_text import build_catalog

        result = build_catalog(
            args.quality_report, args.texts_archive, args.split_archive, args.output
        )
    elif args.command == "bind-motionmillion-texts":
        from tools.data.bumi.motionmillion_text import bind_report

        result = bind_report(args.quality_report, args.text_catalog)
    elif args.command == "four-conversion":
        from tools.data.bumi.text_windows import four_dataset_conversion

        result = four_dataset_conversion(
            args.releases, args.bones_temporal, args.output, split_seed=args.split_seed
        )
    elif args.command == "humanml-conversion":
        result = humanml_conversion(args.quality_report, args.output)
    elif args.command == "build":
        result = build(
            args.source,
            args.output,
            records_per_shard=args.records_per_shard,
            quality_report=args.quality_report,
            workers=args.workers,
            text_feature_mode=args.text_feature_mode,
        )
    elif args.command == "stats":
        result = statistics(
            args.root, args.output, args.dataset, args.sequence_mode, workers=args.workers
        )
    else:
        result = preflight(args.root, args.split, args.dataset, args.limit, args.sequence_mode)
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
