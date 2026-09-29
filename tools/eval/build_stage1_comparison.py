#!/usr/bin/env python3
"""把Stage1训练证据、固定窗口指标和完整舞蹈视频汇总成本地静态对比报告。

读取验证工具已经完成的JSON及MP4，计算分数据集/划分/P的描述性统计；不改变选样、
模型或动作结果。网页提供原始BUMI重定向舞蹈和ONNX自主滚动生成的同步播放器、筛选、
播放速度、逐样本指标、训练曲线与原始报告下载。所有依赖均使用本地相对路径，无外部
分析服务；明确区分本次留出与预训练未见、4秒带示范条件评测与完整自主运动学滚动。
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
from pathlib import Path


def build(root: Path, analysis: Path, selection: Path) -> None:
    records = json.loads((root / "results.json").read_text())
    train = json.loads((analysis / "training_analysis.json").read_text())
    parity = json.loads((root / "onnx_parity.json").read_text())
    summary = {
        "samples": len(records),
        "videos": sum(
            (root / "samples" / f"{r['number']:02d}" / f"{kind}.mp4").is_file()
            for r in records
            for kind in ("original", "generated")
        ),
        "total_source_seconds": sum(r["duration_seconds"] for r in records),
        "by_source": {},
    }
    for source in dict.fromkeys(r["source"] for r in records):
        sub = [r for r in records if r["source"] == source]
        item = {"count": len(sub), "splits": {}}
        for split in ("train", "test"):
            rows = [r for r in sub if r["split"] == split]
            if not rows:
                continue
            item["splits"][split] = {"count": len(rows)}
            for prefix in (0, 12):
                windows = [
                    w for r in rows for w in r["fixed_window"] if w["prefix_frames"] == prefix
                ]
                item["splits"][split][f"P{prefix}"] = {
                    "loss_mean": statistics.mean(w["loss"] for w in windows),
                    "loss_median": statistics.median(w["loss"] for w in windows),
                    "joint_angle_mae_rad": statistics.mean(
                        w["metrics"]["joint_angle_mae_rad"] for w in windows
                    ),
                    "contact_accuracy": statistics.mean(
                        w["metrics"].get("contact_accuracy", 0) for w in windows
                    ),
                }
        item["full_rollout"] = {
            "joint_violation_sample_count": sum(
                r["generated_metrics"]["joint_limit_violation_rate"] > 0 for r in sub
            ),
            "penetration_over_5cm_sample_count": sum(
                r["generated_metrics"]["foot_penetration_max_m"] > 0.05 for r in sub
            ),
            "tilt_over_0_5rad_sample_count": sum(
                r["generated_metrics"]["root_tilt_max_rad"] > 0.5 for r in sub
            ),
            "mean_joint_violation_rate": statistics.mean(
                r["generated_metrics"]["joint_limit_violation_rate"] for r in sub
            ),
            "worst_penetration_m": max(
                r["generated_metrics"]["foot_penetration_max_m"] for r in sub
            ),
        }
        summary["by_source"][source] = item
    for name in ("training_analysis.json", "training_curves.png"):
        shutil.copy2(analysis / name, root / name)
    shutil.copy2(selection, root / "selection.json")
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    identity_path = root / "data_identity.json"
    identity = json.loads(identity_path.read_text()) if identity_path.exists() else {}
    payload = {
        "records": records,
        "training": train,
        "summary": summary,
        "parity": parity,
        "data_identity": identity,
    }
    template = Path(__file__).with_name("templates") / "stage1_comparison.html"
    encoded = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    (root / "index.html").write_text(template.read_text().replace("__REPORT_DATA__", encoded))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()
    build(args.root, args.analysis, args.selection)
