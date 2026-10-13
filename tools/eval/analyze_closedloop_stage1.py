#!/usr/bin/env python3
"""分析 Stage 1 训练的 TensorBoard 与固定生成验证记录。

本工具只读取从训练服务器捕获的证据，不改变训练配置、checkpoint 或数据划分。
训练曲线使用完整 scalar 序列计算分段均值、末期趋势和非有限值；绘图仅作分箱降采样。
验证曲线保持四个来源独立，并区分 P=0 与有前缀样本，保留损失分量以避免把监督去噪
收敛误报成生成质量收敛。支持不同目标步数和最终报告未发布的中断训练，逐项列出实际
存在的 TensorBoard 标签、点数与步数范围；没有完成报告时不会推断训练成功。
输出 JSON 和 PNG，可被本地对比网页直接引用，不修改或补写原始训练事件。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def analyze(evidence: Path, output: Path) -> dict:
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
        plt.rcParams["axes.unicode_minus"] = False
    output.mkdir(parents=True, exist_ok=True)
    events = EventAccumulator(str(evidence / "tensorboard"), size_guidance={"scalars": 0}).Reload()
    series = {}
    for tag in events.Tags()["scalars"]:
        rows = events.Scalars(tag)
        series[tag] = np.asarray([(r.step, r.value) for r in rows], dtype=float)
    reports = list((evidence / "reports").glob("from_*.json"))
    run = json.loads(reports[-1].read_text()) if reports else {"status": "completion_report_absent"}
    result = {k: v for k, v in run.items() if k not in {"steps", "validation_steps"}}
    result["tensorboard_inventory"] = {
        tag: {
            "count": len(values),
            "first_step": int(values[0, 0]),
            "last_step": int(values[-1, 0]),
            "nonfinite_count": int((~np.isfinite(values[:, 1])).sum()),
        }
        for tag, values in series.items()
    }
    result["tensorboard_other_plugins"] = {
        key: value for key, value in events.Tags().items() if key != "scalars"
    }
    result["training_windows"] = []
    x, y = series["train/loss"].T
    final_step = int(x.max())
    windows = [(1, 10000), (40001, 50000)]
    windows += [(end - 9999, end) for end in range(100000, final_step + 1, 100000)]
    windows += [(max(1, final_step - size + 1), final_step) for size in (50000, 20000, 10000)]
    for lo, hi in dict.fromkeys(windows):
        mask = (x >= lo) & (x <= hi)
        if not mask.any():
            continue
        result["training_windows"].append(
            {
                "start": lo,
                "end": hi,
                "mean": float(y[mask].mean()),
                "std": float(y[mask].std()),
                "count": int(mask.sum()),
            }
        )
    tail = x >= max(1, final_step - 49999)
    result["tail_slope_per_10000_steps"] = float(np.polyfit(x[tail], y[tail], 1)[0] * 10000)
    result["scalar_nonfinite_count"] = sum(int((~np.isfinite(v)).sum()) for v in series.values())
    result["train_scalar_count"] = len(x)
    result["last_logged_step"] = final_step
    latest = evidence / "latest.json"
    if latest.exists():
        result["latest_checkpoint"] = json.loads(latest.read_text())
    initialization = evidence / "weight_loading_report.json"
    if initialization.exists():
        result["initialization"] = json.loads(initialization.read_text())
    result["learning_rate_final"] = float(series["train/learning_rate"][-1, 1])
    result["gradient_norm"] = {
        "p50": float(np.median(series["train/gradient_norm"][:, 1])),
        "p99": float(np.quantile(series["train/gradient_norm"][:, 1], 0.99)),
        "max": float(series["train/gradient_norm"][:, 1].max()),
    }
    validations = [
        json.loads(p.read_text()) for p in sorted((evidence / "validation").rglob("s*.json"))
    ]
    result["validation_count"] = len(validations)
    result["validation"] = {}
    for source in validations[0]["sources"]:
        arrays = {}
        for field in (
            "loss",
            "weighted_contact_bce_loss",
            "weighted_repr_root_rot_loss",
            "raw_joint_dof_loss",
            "raw_foot_slide_loss",
            "raw_penetration_max_loss",
        ):
            arrays[field] = np.asarray(
                [np.mean([r[field] for r in v["sources"][source]]) for v in validations]
            )
        steps = np.asarray([v["step"] for v in validations])
        loss = arrays["loss"]
        best = int(np.argmin(loss))
        at_latest = int(
            np.argmin(
                np.abs(steps - result.get("latest_checkpoint", {}).get("global_step", final_step))
            )
        )
        result["validation"][source] = {
            "first": float(loss[0]),
            "final": float(loss[-1]),
            "best": float(loss[best]),
            "best_step": int(steps[best]),
            "at_latest_checkpoint": float(loss[at_latest]),
            "at_latest_checkpoint_step": int(steps[at_latest]),
            "first_10_mean": float(loss[:10].mean()),
            "last_20_mean": float(loss[-20:].mean()),
            "previous_20_mean": float(loss[-40:-20].mean()),
            "last_100_mean": float(loss[-100:].mean()),
            "previous_100_mean": float(loss[-200:-100].mean()),
            "tail_slope_per_10000_steps": float(
                np.polyfit(steps[-100:], loss[-100:], 1)[0] * 10000
            ),
            "first_to_final_percent": float((loss[-1] / loss[0] - 1) * 100),
            "samples": [
                {
                    "index": i,
                    "known_coordinates": r["known_coordinates"],
                    "first_loss": validations[0]["sources"][source][i]["loss"],
                    "final_loss": r["loss"],
                }
                for i, r in enumerate(validations[-1]["sources"][source])
            ],
            "final_components": {key: float(v[-1]) for key, v in arrays.items()},
        }
    result["known_prefix_max_error"] = max(
        r["known_max_abs_error"]
        for v in validations
        for rows in v["sources"].values()
        for r in rows
    )
    result["validation_nonfinite_count"] = sum(
        not np.isfinite(value)
        for v in validations
        for rows in v["sources"].values()
        for row in rows
        for value in row.values()
        if isinstance(value, (int, float))
    )
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    edges = np.arange(0, len(y), 1000)
    axes[0, 0].plot(x[edges], [y[i : i + 1000].mean() for i in edges], label="每1000步均值")
    axes[0, 0].set(title="监督去噪训练损失", xlabel="优化步数", ylabel="损失")
    for tag, values in series.items():
        if tag.startswith("val/"):
            axes[0, 1].plot(*values.T, label=tag.split("/")[1], linewidth=1)
    axes[0, 1].set(
        title="固定生成验证损失（每库2个窗口，DDIM20 / CFG2.5）",
        xlabel="优化步数",
        ylabel="损失",
    )
    axes[0, 1].legend()
    for prefix_zero in (True, False):
        group_count = sum(
            (r["known_coordinates"] == 0) == prefix_zero
            for rows in validations[0]["sources"].values()
            for r in rows
        )
        values = [
            np.mean(
                [
                    r["loss"]
                    for rows in v["sources"].values()
                    for r in rows
                    if (r["known_coordinates"] == 0) == prefix_zero
                ]
            )
            for v in validations
        ]
        axes[1, 0].plot(
            [v["step"] for v in validations],
            values,
            label=f"P=0（{group_count}个窗口）"
            if prefix_zero
            else f"P>0（{group_count}个不同窗口）",
        )
    axes[1, 0].set(
        title="前缀分组验证（不同样本，不能比较前缀优劣）",
        xlabel="优化步数",
        ylabel="损失",
    )
    axes[1, 0].legend()
    axes[1, 1].plot(*series["train/learning_rate"][::100].T, color="#e79d3c")
    axes[1, 1].set(title="学习率", xlabel="优化步数", ylabel="学习率")
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.savefig(output / "training_curves.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    for source in validations[0]["sources"]:
        values = [
            np.mean([r["weighted_contact_bce_loss"] for r in v["sources"][source]])
            for v in validations
        ]
        axes[0].plot([v["step"] for v in validations], values, label=source)
    axes[0].set(title="固定生成验证：接触 BCE 分量", xlabel="优化步数", ylabel="加权损失")
    axes[0].legend()
    gradient = series["train/gradient_norm"]
    axes[1].plot(gradient[::100, 0], gradient[::100, 1], linewidth=0.5)
    axes[1].set(title="梯度范数（裁剪前）", xlabel="优化步数", ylabel="范数")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(output / "validation_components.png", dpi=150)
    plt.close(fig)
    (output / "training_analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("training_windows", "tail_slope_per_10000_steps", "validation")
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.evidence, args.output)
