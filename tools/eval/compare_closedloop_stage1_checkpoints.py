#!/usr/bin/env python3
"""在相同当前数据、条件和随机噪声上比较两个 Stage1 checkpoint 的运动学表现。

本工具复用现有 Stage1 Dataset、严格 checkpoint 加载、完整 DDIM 采样及 BUMI 运动学
指标，不另造训练数据格式或替换模型的归一化统计量。两个模型各自使用原始训练绑定的
stats，评估则统一在物理单位下进行，禁止用不同归一化尺度的总训练 loss 判断优劣。
每条预先固定的动作取首部、中央、尾部三个完整120帧窗口，分别提供P=0/P=12条件，
两种条件均只在第12帧之后评分；模型共享音乐、因果示范历史、前缀值和固定噪声。
当前Stage1 test中的八条动作另执行完整自主滚动，仅首12帧来自示范，随后历史和前缀
全部来自自身输出。保留每个窗口及完整滚动的数值动作、逐样本指标和按来源/划分/P汇总，
既报告改善率也保留退化样本。测试划分仅说明未参加当前Stage1梯度更新，旧热启动模型
可能在音乐预训练阶段见过这些样本。本工具不运行GMT、物理仿真或真实机器人。
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from gem.closedloop.contracts import STAGE1_CONDITION_KEYS  # noqa: E402
from gem.closedloop.stage1_dataset import collate_stage1_training_samples  # noqa: E402
from gem.closedloop.training import batch_to_device  # noqa: E402
from tools.eval.validate_closedloop_stage1 import (  # noqa: E402
    dataset_for,
    generate_full,
    load_actor,
    metric_values,
    write_json,
)


def summarize(old, new):
    """按同一组配对记录统计，不混合来源或改变各模型的样本权重。"""
    result = {}
    for key in old[0]["metrics"]:
        a = np.asarray([r["metrics"][key] for r in old])
        b = np.asarray([r["metrics"][key] for r in new])
        higher = key in {"contact_accuracy", "beat_alignment_score", "minimum_joint_margin_rad"}
        descriptive = (
            key.startswith("root_height_")
            or key.startswith("root_linear_velocity_")
            or key.startswith("root_angular_velocity_")
            or key.startswith("joint_velocity_")
            or key.startswith("diversity_")
        )
        result[key] = {
            "old_mean": float(a.mean()),
            "new_mean": float(b.mean()),
            "old_median": float(np.median(a)),
            "new_median": float(np.median(b)),
            "change_percent": float((b.mean() / a.mean() - 1) * 100) if a.mean() != 0 else None,
            "preferred_direction": None if descriptive else "higher" if higher else "lower",
            "paired_improvement_fraction": None
            if descriptive
            else float(np.mean(b > a if higher else b < a)),
        }
    return {"paired_count": len(old), "metrics": result}


@torch.inference_mode()
def evaluate(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42)
    entries = json.loads(args.selection.read_text())["samples"]
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "selection": str(args.selection),
        "data": str(args.data),
        "samples": len(entries),
        "windows": ["start", "middle", "end"],
        "prefix_frames": [0, 12],
        "scoring_start_frame_for_both_prefixes": 12,
        "ddim_steps": 20,
        "cfg": 2.5,
        "seeds_by_window": [42, 43, 44],
        "backend": "PyTorch CUDA FP32",
        "history": "causal_demo_proxy_for_fixed_windows",
        "full_rollout": "test_samples_only_initial12_demo_then_own_history_and_prefix",
        "aggregate_loss_comparison": False,
        "scope": "offline_kinematic_no_gmt_no_dynamics",
        "checkpoint_summary": {},
    }
    write_json(args.output / "protocol.json", protocol)
    for label in ("old", "new"):
        checkpoint = getattr(args, f"{label}_checkpoint")
        stats = getattr(args, f"{label}_stats")
        actor, _, capture = load_actor(checkpoint, stats, args.kinematics, "cuda")
        protocol["checkpoint_summary"][label] = capture
        write_json(args.output / "protocol.json", protocol)
        datasets, fixed_rows, rollout_rows = {}, [], []
        for number, entry in enumerate(entries):
            key = (entry["source"], entry["split"])
            if key not in datasets:
                datasets[key] = dataset_for(args.data, entry, args.kinematics)
            dataset = datasets[key]
            index = next(
                i
                for i, row in enumerate(dataset.rows)
                if row["sample_id"] == entry["row"]["sample_id"]
            )
            frames = int(dataset.rows[index]["num_frames"])
            decisions = [0, max(0, (frames - 120) // 2), max(0, frames - 120)]
            directory = args.output / "samples" / f"{number + 1:02d}"
            directory.mkdir(parents=True, exist_ok=True)
            generated_windows = []
            for window, decision in enumerate(decisions):
                for prefix in (0, 12):
                    dataset.prefix_min_frames = dataset.prefix_max_frames = prefix
                    sample = dataset.get_window(index, start_frame=decision)
                    batch = batch_to_device(collate_stage1_training_samples([sample]), "cuda")
                    conditions = {k: batch[k] for k in STAGE1_CONDITION_KEYS}
                    noise = torch.randn(
                        (1, 120, 30), generator=torch.Generator().manual_seed(42 + window)
                    ).cuda()
                    output = actor.sample(conditions, noise=noise, steps=20, guidance_scale=2.5)
                    if not all(torch.isfinite(output[k]).all() for k in ("qpos", "contact_logits")):
                        raise FloatingPointError(f"{label}/{number}/{window}/{prefix}")
                    known = conditions["known_qpos30_mask"]
                    error = (
                        float(
                            (output["qpos30"][known] - conditions["known_qpos30"][known])
                            .abs()
                            .max()
                        )
                        if known.any()
                        else 0.0
                    )
                    if error != 0:
                        raise AssertionError("已知动作前缀发生变化")
                    valid = batch["future_valid"].clone()
                    valid[:, :12] = False
                    gt = actor.endecoder.codec.decode_to_canonical_qpos(batch["target_qpos30"])
                    ground = (
                        sample["meta"]["ground_supervision"]["ground_height_world_m"]
                        - actor.endecoder.codec.default_root_height
                    )
                    metrics = metric_values(
                        output["qpos"],
                        actor.endecoder.kinematics,
                        target_qpos=gt,
                        valid_mask=valid,
                        pred_contact_logits=output["contact_logits"],
                        target_contact=batch["target_contact"],
                        music_beats=batch["music_features"][..., 34],
                        ground_height=ground,
                    )
                    contact_valid = batch["target_contact_valid"] & valid[..., None]
                    if not contact_valid.any():
                        raise ValueError("窗口没有有效接触监督")
                    metrics["contact_accuracy"] = float(
                        ((output["contact_logits"] >= 0) == batch["target_contact"].bool())[
                            contact_valid
                        ]
                        .float()
                        .mean()
                    )
                    record = {
                        "sample_number": number + 1,
                        "source": entry["source"],
                        "split": entry["split"],
                        "sample_id": entry["row"]["sample_id"],
                        "window": window,
                        "decision_frame": decision,
                        "prefix_frames": prefix,
                        "seed": 42 + window,
                        "known_max_abs_error": error,
                        "scored_frames": int(valid.sum()),
                        "metrics": metrics,
                    }
                    fixed_rows.append(record)
                    generated_windows.append(
                        {
                            "record": record,
                            "qpos": output["qpos"].cpu(),
                            "contact_logits": output["contact_logits"].cpu(),
                        }
                    )
            torch.save(generated_windows, directory / f"{label}_fixed_windows.pt")
            if entry["split"] == "test":
                original, generated, contact, rollout = generate_full(
                    actor, actor, dataset, index, 42
                )
                valid = torch.ones(len(generated), dtype=torch.bool)
                valid[:12] = False
                metrics = metric_values(
                    generated,
                    actor.endecoder.kinematics.cpu(),
                    valid_mask=valid,
                    pred_contact_logits=contact,
                    ground_height=rollout["ground_height_world_m"],
                )
                actor.endecoder.kinematics.cuda()
                torch.save(
                    {
                        "original_qpos": original,
                        "generated_qpos": generated,
                        "contact_logits": contact,
                        "fps": 30,
                    },
                    directory / f"{label}_rollout.pt",
                )
                rollout_rows.append(
                    {
                        "sample_number": number + 1,
                        "source": entry["source"],
                        "split": "test",
                        "sample_id": entry["row"]["sample_id"],
                        "metrics": metrics,
                        "rollout": rollout,
                    }
                )
            write_json(args.output / f"{label}_fixed.json", fixed_rows)
            write_json(args.output / f"{label}_rollout.json", rollout_rows)
            print(
                f"{label}: {number + 1}/{len(entries)} {entry['source']} {entry['split']}",
                flush=True,
            )
        del actor, datasets, output, batch, conditions, noise
        gc.collect()
        torch.cuda.empty_cache()
    old = json.loads((args.output / "old_fixed.json").read_text())
    new = json.loads((args.output / "new_fixed.json").read_text())

    def identity(row):
        return (
            row["sample_number"],
            row["window"],
            row["prefix_frames"],
            row["decision_frame"],
            row["seed"],
        )

    if [identity(r) for r in old] != [identity(r) for r in new]:
        raise AssertionError("两个模型的配对样本不同")
    groups = defaultdict(lambda: ([], []))
    for a, b in zip(old, new):
        for source in ("all", a["source"]):
            key = f"{source}/{a['split']}/P{a['prefix_frames']}"
            groups[key][0].append(a)
            groups[key][1].append(b)
    roll_old = json.loads((args.output / "old_rollout.json").read_text())
    roll_new = json.loads((args.output / "new_rollout.json").read_text())
    if [r["sample_id"] for r in roll_old] != [r["sample_id"] for r in roll_new]:
        raise AssertionError("完整滚动配对样本不同")
    result = {
        "protocol": protocol,
        "fixed": {key: summarize(*pair) for key, pair in groups.items()},
        "rollout": summarize(roll_old, roll_new),
        "rollout_by_source": {
            source: summarize(
                [r for r in roll_old if r["source"] == source],
                [r for r in roll_new if r["source"] == source],
            )
            for source in sorted({r["source"] for r in roll_old})
        },
    }
    write_json(args.output / "comparison.json", result)
    print("COMPARISON_COMPLETE", args.output / "comparison.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "old-checkpoint",
        "new-checkpoint",
        "old-stats",
        "new-stats",
        "kinematics",
        "selection",
        "data",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    for name, value in vars(args).items():
        setattr(args, name, value.resolve())
    evaluate(args)
