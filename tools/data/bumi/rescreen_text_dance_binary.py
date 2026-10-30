#!/usr/bin/env python3
"""用保留舞蹈的新二分类规则重新检查四库原生UMR动作，不复用旧PASS白名单。

输入是原完整筛选任务的run.json及SQLite输入索引；索引的用途仅是定位所有原始
机器人/人体文件和已核验的坐标、文本来源，旧PASS/REJECT不参与新决策。每个worker
重新读取原NPZ、检查完整30Hz时间线和具名关节，并核对机器人及人体文件内容身份，
随后调用音乐与文本共用的新evaluate_motion。低姿态、贴地和高动态只作诊断。

输出写入全新目录：压缩逐条质量报告、PASS源文件清单、REJECT清单、配置及来源
快照、分库计数和从旧拒绝中恢复的数量。PASS保留全部帧，不套用60..300帧或旧环境
筛选白名单。文本保留准确原ID、索引路径和摘要，供迁回时保存条件资料；本工具不
生成caption、不复制音频、不修改源数据、不构建GENMO分片，也不启动任何训练。
逐条文件/数值错误记REJECT，程序错误向上抛出；CPU进程只处理有限批量，避免把
四库所有指标或轨迹放进内存。新结果只有PASS/REJECT，没有第三种分类。
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from gem.robots.bumi.kinematics import BumiKinematics
from gem.robots.bumi.motion_utils import sha256_file
from tools.data.bumi.filter_robot_retargeter_npz_motions import DANCE_CONFIG, load_config
from tools.data.bumi.npz_quality_utils import evaluate_motion
from tools.data.bumi.umr_qpos_adapter import load_umr_qpos, qpos_arrays

LABELS = {
    "bones_seed": "SOMA-BONES-SEED",
    "humanml3d": "HumanML3D",
    "kitml": "KIT-ML",
    "motionmillion": "MotionMillion",
}


def initialize(config, kinematics):
    global CONFIG, KIN
    torch.set_num_threads(1)
    CONFIG, KIN = load_config(config), BumiKinematics(kinematics)


def evaluate(row):
    result = dict(row)
    try:
        path = Path(row["source_path"])
        digest = sha256_file(path)
        if digest != row["source_sha256"]:
            raise ValueError("原机器人文件与完整生产索引的内容不一致")
        if row.get("human_sha256") and sha256_file(row["human_path"]) != row["human_sha256"]:
            raise ValueError("原人体文件与生产身份不一致")
        qpos, meta = load_umr_qpos(path, KIN, Path(row["recorded_robot_xml"]))
        if (
            row.get("source_sequence_key")
            and meta["source_sequence_key"] != row["source_sequence_key"]
        ):
            raise ValueError("原始序列身份不一致")
        decision = evaluate_motion(qpos_arrays(qpos, KIN, CONFIG), CONFIG)
        result.update(decision, frames=len(qpos), fps=30, source_bytes=path.stat().st_size)
    except (ValueError, OSError, KeyError) as error:
        result.update(
            status="REJECT",
            quality_accepted=False,
            reason_codes=["MOTION_CONTRACT_ERROR"],
            error_message=str(error),
            valid_intervals=[],
        )
    return result


def inputs(report):
    run = json.loads((report / "run.json").read_text())
    identity, paths = run["identity"], run["identity"]["paths"]
    dataset = paths["dataset"]
    db = sqlite3.connect(f"file:{report / 'quality.sqlite'}?mode=ro", uri=True)
    count = db.execute("select count(*) from inputs").fetchone()[0]
    summary = json.loads((report / "quality_summary.json").read_text())
    if count != summary["indexed_records"] or count != summary["processed_records"]:
        raise ValueError(f"{report}: 原生产索引不完整")
    # 核验资产和原坐标合同；不把旧质量状态或源码身份当新规则门禁。
    if identity["assets"]["robot_xml_sha256"] != CONFIG.robot_xml_sha256:
        raise ValueError(f"{report}: 机器人资产不匹配")
    metadata_paths = identity.get("source_contract_hashes", {})
    for path, digest in metadata_paths.items():
        if sha256_file(path) != digest:
            raise ValueError(f"原文本/来源索引已改变: {path}")
    expected_xml = paths.get("recorded_robot_xml", paths["robot_xml"])
    text_metadata = paths.get(
        "metadata_csv", paths.get("metadata_json", str(Path(paths["source_root"]) / "texts.json"))
    )
    if dataset == "motionmillion":
        binding = json.loads((report / "text_binding.json").read_text())
        text_metadata = str(Path(binding["text_catalog"]) / "texts.sqlite")
    query = "select i.relative_path, i.human_path, r.payload from inputs i left join results r using(relative_path) order by i.relative_path"
    try:
        for relative, human, payload in db.execute(query):
            if payload is None:
                raise ValueError(f"{report}: 缺少生产来源身份 {relative}")
            old = json.loads(payload)
            key = old.get("source_motion_id") or Path(relative).stem.removesuffix("_bumi3")
            yield dict(
                dataset=LABELS[dataset],
                source_kind="text",
                motion_id=f"{LABELS[dataset]}::{key}",
                source_path=str(Path(paths["input_root"]) / relative),
                source_sha256=old.get("source_sha256"),
                human_path=human,
                human_sha256=old.get("human_sha256"),
                source_sequence_key=old.get("source_sequence_key"),
                recorded_robot_xml=expected_xml,
                text_id=key,
                text_metadata=text_metadata,
                original_status=old["status"],
                original_reason_codes=old.get("reason_codes", []),
                original_relative_path=relative,
            )
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DANCE_CONFIG)
    parser.add_argument(
        "--kinematics",
        type=Path,
        default=ROOT / "configs/bumi/bumi_kinematics_robot_retargeter_fe934_v1.json",
    )
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    initialize(args.config, args.kinematics)
    if CONFIG.decision_policy != "preserve_dance_binary":
        raise ValueError("本入口只执行保留舞蹈的二分类规则")
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.config, args.output / "quality_config.snapshot.yaml")
    sources = []
    for i, report in enumerate(args.reports):
        raw = json.loads((report / "run.json").read_text())
        shutil.copyfile(report / "run.json", args.output / f"source_run_{i}.json")
        sources.append(
            dict(
                report=str(report),
                dataset=raw["identity"]["paths"]["dataset"],
                paths=raw["identity"]["paths"],
            )
        )
    (args.output / "sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2))
    started = time.monotonic()
    last_progress = 0
    counts, frames, reasons, restored = defaultdict(Counter), Counter(), Counter(), Counter()
    config_sha = sha256_file(args.config)
    with (
        gzip.open(
            args.output / "quality_report.jsonl.gz", "wt", encoding="utf-8", compresslevel=1
        ) as report_out,
        (args.output / "pass.jsonl").open("w") as pass_out,
        (args.output / "reject.jsonl").open("w") as reject_out,
        ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=initialize,
            initargs=(str(args.config), str(args.kinematics)),
        ) as pool,
    ):
        for report in args.reports:
            iterator = iter(inputs(report))
            while batch := list(islice(iterator, args.workers * 16)):
                for result in pool.map(evaluate, batch, chunksize=8):
                    ds, status = result["dataset"], result["status"]
                    counts[ds][status] += 1
                    result["quality_config_sha256"] = config_sha
                    result["decision_policy"] = CONFIG.decision_policy
                    report_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    reasons.update(result["reason_codes"])
                    if status == "PASS":
                        frames[ds] += result["frames"]
                        compact = {
                            k: v
                            for k, v in result.items()
                            if k
                            not in (
                                "metrics",
                                "floor_intervals",
                                "reason_statuses",
                                "report_contract_version",
                            )
                        }
                        pass_out.write(json.dumps(compact, ensure_ascii=False) + "\n")
                        if result["original_status"] != "PASS":
                            restored[ds] += 1
                    else:
                        reject_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                processed = sum(sum(c.values()) for c in counts.values())
                if processed - last_progress >= args.workers * 160:
                    print(
                        json.dumps(
                            dict(
                                processed=processed,
                                by_dataset=counts,
                                seconds=round(time.monotonic() - started, 1),
                            ),
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    last_progress = processed
    summary = dict(
        decision_policy=CONFIG.decision_policy,
        result_states=["PASS", "REJECT"],
        complete=True,
        counts={k: {"PASS": c["PASS"], "REJECT": c["REJECT"]} for k, c in counts.items()},
        frames30=dict(frames),
        restored_from_old_nonpass=dict(restored),
        reason_counts=dict(reasons),
        quality_config_sha256=config_sha,
        kinematics_sha256=sha256_file(args.kinematics),
        pass_manifest_sha256=sha256_file(args.output / "pass.jsonl"),
        elapsed_seconds=time.monotonic() - started,
        full_sequences=True,
        source_files_modified=False,
        crop_count=0,
    )
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
