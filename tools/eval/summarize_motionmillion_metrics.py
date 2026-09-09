#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""汇总 MotionMillion 官方 evaluator 的 20 个固定 seed 指标并选择候选 checkpoint。

每个输入 JSON 必须包含相同 checkpoint SHA256、实验配置指纹、数据 release 指纹、
evaluator 指纹、DDIM/CFG，以及单次 FID、Diversity、R@1/@2/@3、Matching Score。
工具拒绝身份链混用和重复 seed，按样本标准差计算 95% 置信区间。多个 checkpoint
summary 可再用 ``--candidate-summary`` 比较：先最低 FID；FID 差不超过指定阈值时，
再按更高 R@1、最后更低 Matching Score 决胜。

本工具不伪造或重算官方 embedding；其输入必须由官方 eligibility、mean/std 和
evaluator checkpoint 实际运行得到。MotionMillion-Eval 126 条人工提示不属于这里。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

IDENTITY_KEYS = (
    "checkpoint_sha256",
    "experiment_config_sha256",
    "dataset_release_fingerprint",
    "evaluator_fingerprint",
    "ddim_steps",
    "cfg_scale",
)
METRIC_KEYS = ("fid", "diversity", "r_precision_1", "r_precision_2", "r_precision_3", "matching_score")


def _mean_ci(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"mean": mean, "ci95": 1.96 * std / math.sqrt(len(values))}


def summarize(paths: list[Path], *, required_runs: int = 20) -> dict[str, Any]:
    if len(paths) != required_runs:
        raise ValueError(f"正式评测要求 {required_runs} 个 seed，实际 {len(paths)}")
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    identity = {key: rows[0].get(key) for key in IDENTITY_KEYS}
    if any(value is None for value in identity.values()):
        raise ValueError("首份评测 JSON 缺少身份链")
    seeds = []
    for path, row in zip(paths, rows):
        if any(row.get(key) != value for key, value in identity.items()):
            raise ValueError(f"评测身份链不一致: {path}")
        if any(key not in row for key in METRIC_KEYS) or "seed" not in row:
            raise ValueError(f"评测 JSON 缺少 seed/metrics: {path}")
        seeds.append(int(row["seed"]))
    if len(seeds) != len(set(seeds)):
        raise ValueError("评测 seed 重复")
    metrics = {
        key: _mean_ci([float(row[key]) for row in rows]) for key in METRIC_KEYS
    }
    return {"schema_version": 1, **identity, "seeds": sorted(seeds), "metrics": metrics}


def choose_candidate(summaries: list[dict[str, Any]], fid_tie: float) -> dict[str, Any]:
    if not summaries:
        raise ValueError("没有候选 summary")
    best_fid = min(row["metrics"]["fid"]["mean"] for row in summaries)
    close = [
        row
        for row in summaries
        if row["metrics"]["fid"]["mean"] <= best_fid + fid_tie
    ]
    return min(
        close,
        key=lambda row: (
            -row["metrics"]["r_precision_1"]["mean"],
            row["metrics"]["matching_score"]["mean"],
            row["metrics"]["fid"]["mean"],
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="*", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--required-runs", type=int, default=20)
    parser.add_argument("--candidate-summary", action="append", type=Path, default=[])
    parser.add_argument("--fid-tie", type=float, default=0.01)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.candidate_summary:
        candidates = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in args.candidate_summary
        ]
        result = {
            "schema_version": 1,
            "fid_tie": args.fid_tie,
            "selected": choose_candidate(candidates, args.fid_tie),
            "candidates": candidates,
        }
    else:
        result = summarize(args.reports, required_runs=args.required_runs)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(args.output)
        print(f"MotionMillion metric summary complete: {args.output}")


if __name__ == "__main__":
    main()
