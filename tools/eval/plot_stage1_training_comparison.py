#!/usr/bin/env python3
"""整理共同训练集上的Stage1模型比较，并拆解固定验证曲线以解释各指标的含义。

本工具只读取已经完成的配对采样结果、选样清单、旧Stage1完整train清单和本次捕获的
新数据成员核验，不重新采样、不加载模型、不改变训练或数据。先逐条确认选中的32条
训练动作都属于两版Stage1的train，再输出P=0/P=12训练集结果及分来源图，避免把旧音乐
预训练见过的当前test样本作为严格泛化比较。两个模型仍各自使用训练绑定的stats，
评分使用相同音乐/历史/前缀/噪声和统一第12帧后的物理指标，保留Mine高度/接触版本差异。
另将服务器2自己的固定val总loss拆成接触BCE及其余项，并单列关节Smooth L1和脚滑分量，
防止把总loss增长解释成所有动作误差同步增长。图中明确人数、窗口数、单位和改善方向；
接触误判率按1-准确率显示，根高与限位改善仍需要结合动作还原误差判断。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from tools.eval.compare_closedloop_stage1_checkpoints import summarize  # noqa: E402

SOURCES = ("AIST++", "AIOZ-GDANCE", "FineDance", "Mine")
COLORS = ("#4682b4", "#e78b47")


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def configure_fonts():
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    configure_fonts()
    comparison = load(args.comparison / "comparison.json")
    selection = [r for r in load(args.selection)["samples"] if r["split"] == "train"]
    current = load(args.current_membership)
    current_ids = {
        (r["source"], r["sample_id"])
        for r in current["checked_selection"]
        if r["split"] == "train" and r["in_current_split"]
    }
    old_ids, manifests = {}, {}
    for source in SOURCES:
        path = args.old_metadata / source / "manifests/train.jsonl"
        old_ids[source] = {json.loads(line)["sample_id"] for line in path.read_text().splitlines()}
        manifests[source] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    membership = [
        {
            "source": r["source"],
            "sample_id": r["row"]["sample_id"],
            "old_stage1_train": r["row"]["sample_id"] in old_ids[r["source"]],
            "new_stage1_train": (r["source"], r["row"]["sample_id"]) in current_ids,
        }
        for r in selection
    ]
    if len(membership) != 32 or not all(
        r["old_stage1_train"] and r["new_stage1_train"] for r in membership
    ):
        raise ValueError("共同训练动作数量或成员核验失败")
    old = [r for r in load(args.comparison / "old_fixed.json") if r["split"] == "train"]
    new = [r for r in load(args.comparison / "new_fixed.json") if r["split"] == "train"]
    identity = (
        "source",
        "sample_id",
        "window",
        "prefix_frames",
        "decision_frame",
        "seed",
        "scored_frames",
    )
    if [tuple(r[k] for k in identity) for r in old] != [tuple(r[k] for k in identity) for r in new]:
        raise ValueError("新旧训练集结果没有逐窗口配对")
    if len(old) != 192 or not all(r["known_max_abs_error"] == 0 for r in old + new):
        raise ValueError("窗口数量或前缀校验失败")
    groups = {}
    for source in ("all", *SOURCES):
        for prefix in (0, 12):
            a = [
                r
                for r in old
                if r["prefix_frames"] == prefix and (source == "all" or r["source"] == source)
            ]
            b = [
                r
                for r in new
                if r["prefix_frames"] == prefix and (source == "all" or r["source"] == source)
            ]
            groups[f"{source}/P{prefix}"] = summarize(a, b)
    report = {
        "scope": "common_stage1_train_fitting_and_generation_not_unseen_generalization",
        "actions": 32,
        "windows_per_prefix": 96,
        "scoring_start_frame": 12,
        "protocol": comparison["protocol"],
        "old_train_manifests": manifests,
        "membership": membership,
        "fixed": groups,
    }
    write(args.output / "train_comparison.json", report)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    specifications = [
        ("joint_angle_mae_rad", "关节平均绝对误差（rad）", 1),
        ("root_trajectory_error_m", "根部轨迹误差（cm）", 100),
        ("fk_body_position_error_m", "FK身体位置误差（cm）", 100),
        ("contact_accuracy", "接触误判率（%）", -100),
        ("foot_sliding_mean_mps", "支撑脚滑动速度（cm/s）", 100),
        ("foot_penetration_mean_m", "平均穿地深度（mm）", 1000),
    ]
    for ax, (key, title, scale) in zip(axes.flat, specifications):
        r = groups["all/P12"]["metrics"][key]
        values = [r["old_mean"] * scale, r["new_mean"] * scale]
        if scale < 0:
            values = [(1 - r["old_mean"]) * 100, (1 - r["new_mean"]) * 100]
        bars = ax.bar(["旧Stage1 350k", "新Stage1 595k"], values, color=COLORS, width=0.55)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom"
            )
        ax.set(title=title + "，越低越好", ylim=(0, max(values) * 1.2))
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "共同训练集上的完整生成比较：四库各8条动作 × 首/中/尾3个窗口，共96对；P=12\nDDIM20 / CFG2.5；相同条件与噪声；统一只评分第12帧之后；此图衡量训练动作还原，不衡量未见泛化",
        fontsize=12,
    )
    fig.savefig(args.output / "checkpoint_comparison_train.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    x = np.arange(4)
    for ax, (key, title, scale) in zip(axes.flat, specifications[:4]):
        for label, color, offset in (("old", COLORS[0], -0.18), ("new", COLORS[1], 0.18)):
            values = [
                groups[f"{source}/P12"]["metrics"][key][f"{label}_mean"] for source in SOURCES
            ]
            values = [(1 - value) * 100 if scale < 0 else value * scale for value in values]
            bars = ax.bar(
                x + offset,
                values,
                width=0.36,
                color=color,
                label="旧350k" if label == "old" else "新595k",
            )
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        ax.set(
            title=title + "，越低越好",
            xticks=x,
            xticklabels=SOURCES,
            ylim=(0, ax.get_ylim()[1] * 1.12),
        )
        ax.grid(axis="y", alpha=0.2)
        ax.legend()
    fig.suptitle(
        "按来源比较共同train：每库8条动作 × 3窗口 = 24对；P=12\nMine使用当前恢复高度及接触标签；旧模型学习过的Mine数据版本不同，根高和穿地需结合这一变化解释",
        fontsize=12,
    )
    fig.savefig(args.output / "checkpoint_comparison_train_by_source.png", dpi=150)
    plt.close(fig)

    validations = [load(path) for path in sorted(args.validation.glob("s*.json"))]
    steps = [row["step"] for row in validations]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    series = {}
    for source in validations[0]["sources"]:
        total = np.asarray(
            [np.mean([r["loss"] for r in v["sources"][source]]) for v in validations]
        )
        contact = np.asarray(
            [
                np.mean([r["weighted_contact_bce_loss"] for r in v["sources"][source]])
                for v in validations
            ]
        )
        joint = np.asarray(
            [np.mean([r["raw_joint_dof_loss"] for r in v["sources"][source]]) for v in validations]
        )
        slide = np.asarray(
            [np.mean([r["raw_foot_slide_loss"] for r in v["sources"][source]]) for v in validations]
        )
        series[source] = {
            "total": total.tolist(),
            "weighted_contact_bce": contact.tolist(),
            "total_without_contact": (total - contact).tolist(),
            "raw_joint_smooth_l1": joint.tolist(),
            "raw_foot_slide_mps": slide.tolist(),
        }
        for ax, values in zip(axes.flat, (contact, total - contact, joint, slide)):
            ax.plot(steps, values, label=source, linewidth=0.8)
    for ax, title, unit in zip(
        axes.flat,
        (
            "加权接触BCE：预测是否接触的惩罚",
            "总loss扣除接触BCE：其余损失之和",
            "关节Smooth L1分量：与MAE柱状图定义不同",
            "脚滑分量：按GT接触区域计算",
        ),
        ("加权损失", "加权损失", "原始Smooth L1损失", "m/s"),
    ):
        ax.set(title=title, xlabel="服务器2优化步数", ylabel=unit)
        ax.legend()
        ax.grid(alpha=0.2)
    fig.suptitle(
        "服务器2自身的固定val分量随训练变化：四库各2个固定窗口\n此图只含新版训练过程；旧模型的预训练曝光不影响本图；不同分量的纵轴不能直接比较",
        fontsize=12,
    )
    fig.savefig(args.output / "validation_components_explained.png", dpi=150)
    plt.close(fig)
    write(args.output / "validation_component_series.json", {"steps": steps, "sources": series})
    print(
        json.dumps(
            {
                "actions": 32,
                "windows_per_prefix": 96,
                "membership_verified": True,
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "comparison",
        "selection",
        "old-metadata",
        "current-membership",
        "validation",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(parser.parse_args())
