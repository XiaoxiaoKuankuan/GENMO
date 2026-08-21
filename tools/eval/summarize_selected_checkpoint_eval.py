#!/usr/bin/env python3
"""汇总 GENMO 候选 checkpoint 的多数据集评测结果。

该工具读取顺序评测脚本为 EMDB_1、EMDB_2、3DPW 和 RICH 生成的日志，严格检查每个
候选 checkpoint 的成功标记与必需指标是否齐全，再分别计算局部几何、全局运动和时序
稳定性得分。输出包含原始指标长/宽表、逐指标获胜者、综合排名、JSON 摘要和可读报告，
并为局部、全局及综合最佳 checkpoint 建立相对软链接，便于后续部署和复现实验选择。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from statistics import mean

MetricKey = tuple[str, str]

# ============================================================
# 指标分组
# ============================================================

# 局部几何精度：相机坐标或根对齐后的姿态/表面误差
LOCAL_KEYS: list[MetricKey] = [
    ("EMDB_1", "pa_mpjpe"),
    ("EMDB_1", "mpjpe"),
    ("EMDB_1", "pve"),
    ("3DPW", "pa_mpjpe"),
    ("3DPW", "mpjpe"),
    ("3DPW", "pve"),
    ("RICH", "pa_mpjpe"),
    ("RICH", "mpjpe"),
    ("RICH", "pve"),
]

# 全局运动精度：世界坐标轨迹、方向和相对轨迹误差
GLOBAL_KEYS: list[MetricKey] = [
    ("EMDB_2", "wa2_mpjpe"),
    ("EMDB_2", "waa_mpjpe"),
    ("EMDB_2", "rte"),
    ("RICH", "wa2_mpjpe"),
    ("RICH", "waa_mpjpe"),
    ("RICH", "rte"),
]

# 时序/物理稳定性：动作加速度、抖动和脚滑
STABILITY_KEYS: list[MetricKey] = [
    ("EMDB_1", "accel"),
    ("3DPW", "accel"),
    ("RICH", "accel"),
    ("EMDB_2", "jitter"),
    ("EMDB_2", "fs"),
    ("RICH", "jitter"),
    ("RICH", "fs"),
]

ALL_KEYS: list[MetricKey] = LOCAL_KEYS + GLOBAL_KEYS + STABILITY_KEYS

# 综合得分权重。需要调整时修改这里。
OVERALL_WEIGHTS = {
    "local": 0.45,
    "global": 0.40,
    "stability": 0.15,
}

# 所有指标均为越低越好
LOWER_IS_BETTER = set(ALL_KEYS)

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
DATASET_RE = re.compile(r"\[Metrics\]\s+([A-Za-z0-9_-]+):")

METRIC_NAMES = sorted(
    {metric for _, metric in ALL_KEYS},
    key=len,
    reverse=True,
)

METRIC_PATTERN = "|".join(re.escape(name) for name in METRIC_NAMES)

NUMBER_PATTERN = (
    r"[-+]?"
    r"(?:"
    r"(?:\d+(?:\.\d*)?)"
    r"|"
    r"(?:\.\d+)"
    r")"
    r"(?:[eE][-+]?\d+)?"
)

METRIC_RE = re.compile(
    rf"(?<![A-Za-z0-9_])"
    rf"({METRIC_PATTERN})"
    rf":\s*"
    rf"({NUMBER_PATTERN})"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate GENMO checkpoint evaluation logs and select best checkpoints."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Training run directory containing checkpoints/.",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        required=True,
        help="Evaluation root containing sXXXXXX/eval.log.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        required=True,
        help="Checkpoint steps to aggregate.",
    )
    return parser.parse_args()


def parse_metric_log(log_path: Path) -> dict[MetricKey, float]:
    if not log_path.is_file():
        raise FileNotFoundError(f"Evaluation log does not exist: {log_path}")

    metrics: dict[MetricKey, float] = {}
    current_dataset: str | None = None

    text = log_path.read_text(encoding="utf-8", errors="replace")

    for raw_line in text.splitlines():
        line = ANSI_RE.sub("", raw_line).strip()

        dataset_match = DATASET_RE.search(line)
        if dataset_match:
            current_dataset = dataset_match.group(1)
            continue

        if current_dataset is None:
            continue

        if line.startswith("------"):
            current_dataset = None
            continue

        metric_match = METRIC_RE.search(line)
        if metric_match:
            metric_name = metric_match.group(1)
            value = float(metric_match.group(2))

            if not math.isfinite(value):
                raise ValueError(
                    f"Non-finite metric in {log_path}: {current_dataset}/{metric_name}={value}"
                )

            metrics[(current_dataset, metric_name)] = value

    return metrics


def average_tie_ranks(values: dict[int, float]) -> dict[int, float]:
    """Return 1-based ranks; tied values receive their average rank."""
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    result: dict[int, float] = {}

    i = 0
    while i < len(ordered):
        j = i + 1
        value = ordered[i][1]

        while j < len(ordered) and math.isclose(
            ordered[j][1],
            value,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            j += 1

        # positions are i+1 through j
        average_rank = ((i + 1) + j) / 2.0

        for k in range(i, j):
            result[ordered[k][0]] = average_rank

        i = j

    return result


def metric_group(key: MetricKey) -> str:
    if key in LOCAL_KEYS:
        return "local"
    if key in GLOBAL_KEYS:
        return "global"
    if key in STABILITY_KEYS:
        return "stability"
    raise KeyError(key)


def safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("Cannot average an empty sequence")
    return mean(values)


def replace_symlink(link_path: Path, target_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)

    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()

    relative_target = os.path.relpath(target_path, start=link_path.parent)
    link_path.symlink_to(relative_target)


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    run_dir = args.run_dir.resolve()
    eval_dir = args.eval_dir.resolve()
    steps = sorted(set(args.steps))

    checkpoints_dir = run_dir / "checkpoints"

    if not checkpoints_dir.is_dir():
        raise SystemExit(f"Checkpoint directory not found: {checkpoints_dir}")

    eval_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 1. 读取全部checkpoint日志
    # ============================================================

    parsed: dict[int, dict[MetricKey, float]] = {}
    incomplete: dict[int, list[str]] = {}

    for step in steps:
        name = f"s{step:06d}"
        eval_subdir = eval_dir / name
        log_path = eval_subdir / "eval.log"
        success_path = eval_subdir / "SUCCESS"

        if not success_path.is_file():
            incomplete.setdefault(step, []).append("SUCCESS marker missing")

        try:
            metrics = parse_metric_log(log_path)
        except Exception as exc:
            incomplete.setdefault(step, []).append(str(exc))
            continue

        missing = [
            f"{dataset}/{metric}"
            for dataset, metric in ALL_KEYS
            if (dataset, metric) not in metrics
        ]

        if missing:
            incomplete.setdefault(step, []).append("missing metrics: " + ", ".join(missing))
            continue

        parsed[step] = metrics

    if incomplete:
        print("\n[ERROR] Incomplete checkpoint evaluations:")
        for step, reasons in sorted(incomplete.items()):
            print(f"  s{step:06d}:")
            for reason in reasons:
                print(f"    - {reason}")

        raise SystemExit("\nPlease fix failed evaluations before generating final ranking.")

    if len(parsed) != len(steps):
        raise SystemExit(f"Expected {len(steps)} complete checkpoints, got {len(parsed)}")

    # ============================================================
    # 2. 写原始指标长表/宽表
    # ============================================================

    long_rows: list[dict] = []

    for step in steps:
        name = f"s{step:06d}"

        for dataset, metric in ALL_KEYS:
            long_rows.append(
                {
                    "step": step,
                    "checkpoint": name,
                    "group": metric_group((dataset, metric)),
                    "dataset": dataset,
                    "metric": metric,
                    "value": parsed[step][(dataset, metric)],
                }
            )

    write_csv(
        eval_dir / "metrics_long.csv",
        [
            "step",
            "checkpoint",
            "group",
            "dataset",
            "metric",
            "value",
        ],
        long_rows,
    )

    wide_metric_columns = [f"{dataset}/{metric}" for dataset, metric in ALL_KEYS]

    wide_rows: list[dict] = []

    for step in steps:
        row = {
            "step": step,
            "checkpoint": f"s{step:06d}",
        }

        for dataset, metric in ALL_KEYS:
            row[f"{dataset}/{metric}"] = parsed[step][(dataset, metric)]

        wide_rows.append(row)

    write_csv(
        eval_dir / "metrics_wide.csv",
        ["step", "checkpoint", *wide_metric_columns],
        wide_rows,
    )

    # ============================================================
    # 3. 每个指标在候选checkpoint间做归一化和排名
    # ============================================================

    normalized: dict[int, dict[MetricKey, float]] = defaultdict(dict)
    ranks: dict[int, dict[MetricKey, float]] = defaultdict(dict)

    winner_rows: list[dict] = []

    for key in ALL_KEYS:
        dataset, metric = key
        values = {step: parsed[step][key] for step in steps}

        minimum = min(values.values())
        maximum = max(values.values())
        span = maximum - minimum

        if span <= 1e-12:
            for step in steps:
                normalized[step][key] = 0.0
        else:
            for step in steps:
                # 所有当前指标越低越好
                normalized[step][key] = (values[step] - minimum) / span

        rank_map = average_tie_ranks(values)

        for step in steps:
            ranks[step][key] = rank_map[step]

        winners = [
            step
            for step, value in values.items()
            if math.isclose(
                value,
                minimum,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ]

        winner_rows.append(
            {
                "group": metric_group(key),
                "dataset": dataset,
                "metric": metric,
                "best_value": minimum,
                "winner_steps": ";".join(str(step) for step in winners),
                "winner_checkpoints": ";".join(f"s{step:06d}" for step in winners),
            }
        )

    write_csv(
        eval_dir / "metric_winners.csv",
        [
            "group",
            "dataset",
            "metric",
            "best_value",
            "winner_steps",
            "winner_checkpoints",
        ],
        winner_rows,
    )

    # ============================================================
    # 4. 计算局部、全局、稳定性、综合得分
    # ============================================================

    ranking_rows: list[dict] = []

    for step in steps:
        local_score = safe_mean(normalized[step][key] for key in LOCAL_KEYS)
        global_score = safe_mean(normalized[step][key] for key in GLOBAL_KEYS)
        stability_score = safe_mean(normalized[step][key] for key in STABILITY_KEYS)

        local_mean_rank = safe_mean(ranks[step][key] for key in LOCAL_KEYS)
        global_mean_rank = safe_mean(ranks[step][key] for key in GLOBAL_KEYS)
        stability_mean_rank = safe_mean(ranks[step][key] for key in STABILITY_KEYS)

        overall_score = (
            OVERALL_WEIGHTS["local"] * local_score
            + OVERALL_WEIGHTS["global"] * global_score
            + OVERALL_WEIGHTS["stability"] * stability_score
        )

        overall_mean_rank = (
            OVERALL_WEIGHTS["local"] * local_mean_rank
            + OVERALL_WEIGHTS["global"] * global_mean_rank
            + OVERALL_WEIGHTS["stability"] * stability_mean_rank
        )

        local_wins = sum(
            math.isclose(
                parsed[step][key],
                min(parsed[s][key] for s in steps),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for key in LOCAL_KEYS
        )

        global_wins = sum(
            math.isclose(
                parsed[step][key],
                min(parsed[s][key] for s in steps),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for key in GLOBAL_KEYS
        )

        stability_wins = sum(
            math.isclose(
                parsed[step][key],
                min(parsed[s][key] for s in steps),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for key in STABILITY_KEYS
        )

        ranking_rows.append(
            {
                "step": step,
                "checkpoint": f"s{step:06d}",
                "local_score": local_score,
                "global_score": global_score,
                "stability_score": stability_score,
                "overall_score": overall_score,
                "local_mean_rank": local_mean_rank,
                "global_mean_rank": global_mean_rank,
                "stability_mean_rank": stability_mean_rank,
                "overall_mean_rank": overall_mean_rank,
                "local_wins": local_wins,
                "global_wins": global_wins,
                "stability_wins": stability_wins,
            }
        )

    # 综合排序，越低越好
    ranking_rows.sort(
        key=lambda row: (
            row["overall_score"],
            row["overall_mean_rank"],
            row["step"],
        )
    )

    write_csv(
        eval_dir / "checkpoint_ranking.csv",
        [
            "step",
            "checkpoint",
            "local_score",
            "global_score",
            "stability_score",
            "overall_score",
            "local_mean_rank",
            "global_mean_rank",
            "stability_mean_rank",
            "overall_mean_rank",
            "local_wins",
            "global_wins",
            "stability_wins",
        ],
        ranking_rows,
    )

    # ============================================================
    # 5. 选择三类最佳checkpoint
    # ============================================================

    best_local = min(
        ranking_rows,
        key=lambda row: (
            row["local_score"],
            row["local_mean_rank"],
            row["step"],
        ),
    )

    best_global = min(
        ranking_rows,
        key=lambda row: (
            row["global_score"],
            row["global_mean_rank"],
            row["step"],
        ),
    )

    best_overall = min(
        ranking_rows,
        key=lambda row: (
            row["overall_score"],
            row["overall_mean_rank"],
            row["step"],
        ),
    )

    best_results = {
        "scoring": {
            "lower_is_better": True,
            "normalization": "per-metric min-max across candidate checkpoints",
            "overall_weights": OVERALL_WEIGHTS,
            "local_metrics": [f"{dataset}/{metric}" for dataset, metric in LOCAL_KEYS],
            "global_metrics": [f"{dataset}/{metric}" for dataset, metric in GLOBAL_KEYS],
            "stability_metrics": [f"{dataset}/{metric}" for dataset, metric in STABILITY_KEYS],
        },
        "best_local": best_local,
        "best_global": best_global,
        "best_overall": best_overall,
    }

    with (eval_dir / "best_checkpoints.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            best_results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # ============================================================
    # 6. 创建最佳checkpoint软链接
    # ============================================================

    best_dir = eval_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)

    best_links = {
        "best_local.ckpt": best_local,
        "best_global.ckpt": best_global,
        "best_overall.ckpt": best_overall,
    }

    for link_name, result in best_links.items():
        target = checkpoints_dir / f"{result['checkpoint']}.ckpt"

        if not target.is_file():
            raise FileNotFoundError(f"Selected checkpoint does not exist: {target}")

        replace_symlink(
            best_dir / link_name,
            target,
        )

    # ============================================================
    # 7. 输出可读报告
    # ============================================================

    report_lines = [
        "GENMO selected checkpoint evaluation report",
        "=" * 78,
        "",
        "Candidates:",
        "  " + ", ".join(f"s{step:06d}" for step in steps),
        "",
        "Score definition:",
        "  Per-metric min-max normalization; lower is better.",
        (
            "  Overall = "
            f"{OVERALL_WEIGHTS['local']:.2f} * local"
            " + "
            f"{OVERALL_WEIGHTS['global']:.2f} * global"
            " + "
            f"{OVERALL_WEIGHTS['stability']:.2f} * stability"
        ),
        "",
        "Ranking:",
        (
            f"{'rank':>4} "
            f"{'checkpoint':>12} "
            f"{'local':>10} "
            f"{'global':>10} "
            f"{'stable':>10} "
            f"{'overall':>10} "
            f"{'L-wins':>7} "
            f"{'G-wins':>7} "
            f"{'S-wins':>7}"
        ),
        "-" * 92,
    ]

    for rank_index, row in enumerate(ranking_rows, start=1):
        report_lines.append(
            f"{rank_index:>4} "
            f"{row['checkpoint']:>12} "
            f"{row['local_score']:>10.6f} "
            f"{row['global_score']:>10.6f} "
            f"{row['stability_score']:>10.6f} "
            f"{row['overall_score']:>10.6f} "
            f"{row['local_wins']:>7} "
            f"{row['global_wins']:>7} "
            f"{row['stability_wins']:>7}"
        )

    report_lines.extend(
        [
            "",
            "Selected checkpoints:",
            (
                "  Best local   : "
                f"{best_local['checkpoint']} "
                f"(score={best_local['local_score']:.6f})"
            ),
            (
                "  Best global  : "
                f"{best_global['checkpoint']} "
                f"(score={best_global['global_score']:.6f})"
            ),
            (
                "  Best overall : "
                f"{best_overall['checkpoint']} "
                f"(score={best_overall['overall_score']:.6f})"
            ),
            "",
            "Links:",
            f"  {best_dir / 'best_local.ckpt'}",
            f"  {best_dir / 'best_global.ckpt'}",
            f"  {best_dir / 'best_overall.ckpt'}",
            "",
            "Generated CSV files:",
            f"  {eval_dir / 'metrics_long.csv'}",
            f"  {eval_dir / 'metrics_wide.csv'}",
            f"  {eval_dir / 'checkpoint_ranking.csv'}",
            f"  {eval_dir / 'metric_winners.csv'}",
        ]
    )

    report_text = "\n".join(report_lines) + "\n"

    (eval_dir / "report.txt").write_text(
        report_text,
        encoding="utf-8",
    )

    print()
    print(report_text)


if __name__ == "__main__":
    main()
