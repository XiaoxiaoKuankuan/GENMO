"""四库完整源动作的文本与训练窗口接入。

读取经过指纹绑定的UMR PASS发布清单，保留完整源NPZ引用；BONES只采用官方事件
description及秒级时间范围，将时间映射到30Hz半开帧区间，不复制裁剪动作。
其他三库使用现有完整记录文本，HumanML3D已裁好的子片段保持原来源区间。
官方split优先；没有官方split及HumanML3D训练派生数据按母来源做90/5/5内部划分。
内部验证不能作为官方测试成绩，跨库母来源尚未建立统一映射的情况明确写入报告。
本工具只生成转换索引，实际构建仍通过QualityGate核验机器人/人体SHA及PASS身份，
并要求后续编码真实T5特征。所有输出拒绝覆盖既有数据。
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from gem.datasets.pure_motion.bumi_text import DATASETS
from gem.runtime.bumi_text_contract import sha256_file


def internal_split(canonical_source_id, seed=20260922):
    value = hashlib.sha256(f"{seed}:{canonical_source_id}".encode()).digest()
    fraction = int.from_bytes(value[:8], "big") / 2**64
    return "train" if fraction < 0.90 else "val" if fraction < 0.95 else "test"


def temporal_captions(events, frames, *, tolerance_seconds=0.1):
    """只取事件内部完整采样点；最多容许重采样尾端0.1秒差异并显式记录。"""
    captions, intervals, provenance = [], [], []
    for index, event in enumerate(events):
        start, end = float(event["start_time"]), float(event["end_time"])
        text = event["description"].strip()
        if not all(map(math.isfinite, [start, end])) or start < 0 or end <= start or not text:
            raise ValueError("BONES事件必须有非空原文及合法秒级区间")
        if end > frames / 30 + tolerance_seconds + 1e-8:
            raise ValueError("BONES事件超出完整源动作时间线，禁止静默裁掉大幅错位")
        a = math.ceil(start * 30 - 1e-8)
        b = min(frames, math.floor(end * 30 + 1e-8))
        if b - a < 4:
            continue
        captions.append(text)
        intervals.append([a, b])
        provenance.append(
            dict(event_index=index, interval_seconds=[start, end], end_clipped=end > frames / 30)
        )
    return captions, intervals, provenance


def four_dataset_conversion(releases, bones_temporal, output, *, split_seed=20260922):
    from tools.data.bumi.prepare_bumi_text import write_json

    bones_temporal = Path(bones_temporal).resolve(strict=True)
    temporal_sha = sha256_file(bones_temporal)
    events = {}
    with bones_temporal.open() as stream:
        for line in stream:
            row = json.loads(line)
            key = row["filename"]
            if key in events or row["num_events"] != len(row["events"]):
                raise ValueError("BONES时间标注ID重复或事件数错误")
            events[key] = row["events"]
    records, reports, counts, manifests = [], {}, Counter(), {}
    kin = None
    for release in releases:
        root = Path(release).resolve(strict=True)
        info = json.loads((root / "dataset_info.json").read_text())
        dataset = info["schema"].removeprefix("genmo.").removesuffix("_umr_pass.v1")
        if dataset not in DATASETS or dataset in reports:
            raise ValueError("每个数据集只能提供一个有效PASS release")
        manifest = root / "manifests/pass.jsonl"
        if sha256_file(manifest) != info["manifests"]["pass.jsonl"]:
            raise ValueError("PASS发布清单发生改变")
        report = Path(info["quality_report"]).resolve(strict=True)
        run = json.loads((report / "run.json").read_text())
        if (
            run["state"] != "complete"
            or run["partial_scan"]
            or run["fingerprint"] != info["quality_fingerprint"]
        ):
            raise ValueError("PASS发布与完整质量报告不一致")
        paths = run["identity"]["paths"]
        selected_kin = dict(path=paths["kinematics"], sha256=sha256_file(paths["kinematics"]))
        if kin is not None and kin["sha256"] != selected_kin["sha256"]:
            raise ValueError("四库运动学资产不一致")
        kin = selected_kin
        reports[dataset] = str(report)
        manifests[dataset] = dict(path=str(manifest), sha256=sha256_file(manifest))
        official_groups = {}
        if dataset == "motionmillion":
            with manifest.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    if row["split"] in {"train", "val", "test"}:
                        key = row["canonical_source_id"]
                        if key in official_groups and official_groups[key] != row["split"]:
                            raise ValueError("MotionMillion同一母来源出现冲突的官方split")
                        official_groups[key] = row["split"]
        with manifest.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["status"] != "PASS" or row["fps"] != 30 or row["frames"] < 4:
                    raise ValueError("PASS release中存在非法动作")
                key, frames = row["motion_id"], row["frames"]
                canonical = row["canonical_source_id"]
                if not canonical:
                    raise ValueError("缺少母来源ID，无法防止镜像跨split")
                annotations = None
                if dataset == "bones_seed":
                    captions, intervals, annotations = temporal_captions(
                        events.get(key, []), frames
                    )
                    counts["bones_events_excluded_too_short"] += len(events.get(key, [])) - len(
                        captions
                    )
                else:
                    captions = [c if isinstance(c, str) else c["caption"] for c in row["captions"]]
                    intervals = [[0, frames] for _ in captions]
                if not captions:
                    counts[f"{dataset}/excluded_without_usable_text"] += 1
                    continue
                source_split = row["split"]
                official = canonical in official_groups
                split = (
                    official_groups[canonical]
                    if official
                    else internal_split(canonical, split_seed)
                )
                provenance = dict(
                    source_id=key,
                    canonical_source_id=canonical,
                    interval_seconds=row.get("interval_seconds") or [0, frames / 30],
                    annotation_interval_seconds=row.get("annotation_interval_seconds"),
                    mirrored=row.get("mirrored"),
                    source_split=source_split,
                    split_origin="official" if official else "internal_group_holdout",
                    split_seed=split_seed,
                    cross_dataset_lineage_verified=False,
                    retargeter="UMR",
                    retarget_version=run["fingerprint"],
                )
                if annotations is not None:
                    provenance["temporal_annotations"] = dict(
                        path=str(bones_temporal), sha256=temporal_sha, events=annotations
                    )
                    counts["bones_events_end_clipped"] += sum(a["end_clipped"] for a in annotations)
                records.append(
                    dict(
                        dataset=dataset,
                        motion_id=key,
                        text_source_motion_id=key,
                        split=split,
                        qpos_path=row["source_path"],
                        fps=30,
                        frames=frames,
                        captions=captions,
                        caption_ids=[f"{dataset}/{key}:{i}" for i in range(len(captions))],
                        caption_intervals=intervals,
                        text_annotation_scope="temporal"
                        if annotations is not None
                        else "whole_record",
                        embeddings=[],
                        provenance=provenance,
                    )
                )
                counts[f"{split}/{dataset}"] += 1
    if set(reports) != DATASETS:
        raise ValueError("联合转换必须完整提供四库PASS release")
    payload = dict(
        schema="genmo.bumi_text_conversion.v1",
        kinematics=kin,
        quality_reports=reports,
        source_storage="full",
        source_manifests=manifests,
        records=records,
        split_policy="official_if_available_else_internal_canonical_90_5_5",
        cross_dataset_lineage_verified=False,
        conversion_counts=dict(counts),
    )
    write_json(output, payload)
    return dict(
        records=len(records),
        counts=dict(counts),
        output=str(Path(output).resolve()),
        split_policy=payload["split_policy"],
        cross_dataset_lineage_verified=False,
    )
