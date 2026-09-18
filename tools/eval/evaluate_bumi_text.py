#!/usr/bin/env python3
"""固定BUMI文本验证清单、按GT长度生成及机器人运动学诊断。

cohort默认各数据集64条，稳定排序后固定种子抽样，caption取第一条；生成严格使用真实F。
run同时记录GT、原始与足锁结果的运动学指标，另给人工1–5分表。没有人体evaluator，
不输出R-Precision/FID，不以运动学指标证明语义匹配或闭环可跟踪。
"""

from pathlib import Path
import argparse
import csv
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gem.datasets.pure_motion.bumi_text import BumiTextDataset
from gem.runtime.bumi_text_contract import sha256_file
from tools.data.bumi.prepare_bumi_text import write_json


def cohort(root, output, per_dataset=64, seed=42):
    import random

    if per_dataset < 1:
        raise ValueError("per_dataset必须为正")
    rows, identities = [], {}
    for name in ("motionmillion", "humanml3d"):
        ds = BumiTextDataset(root, "val", dataset=name, caption_sampling="first")
        ordered = sorted(range(len(ds)), key=lambda i: ds.index[i][2]["motion_id"])
        if len(ordered) < per_dataset:
            raise ValueError(f"{name}验证记录不足{per_dataset}条")
        indices = random.Random(seed).sample(ordered, per_dataset)
        identities[name] = ds.data_identity
        for i in indices:
            record = ds.read_record(i)
            rows.append(
                dict(
                    dataset=name,
                    motion_id=record["motion_id"],
                    frames=record["frames"],
                    caption=record["captions"][0],
                    caption_id=record["caption_ids"][0],
                    text_index=0,
                    seed=seed,
                )
            )
    payload = dict(
        schema="genmo.bumi_text_eval.v1",
        protocol="full_sequence_matched_length",
        split="val",
        data_identity=identities,
        ddim_steps=50,
        guidance_scale=2.5,
        fps=30,
        records=rows,
        official_human_evaluator=False,
    )
    write_json(output, payload)
    return payload


def run(root, cohort_path, checkpoint, output, device="cuda:0", render=False):
    import numpy as np
    import torch
    from gem.runtime.bumi_text_runtime import ResidentBumiTextEngine
    from gem.robots.bumi.metrics import compute_bumi_kinematic_metrics

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("诊断报告必须使用新目录")
    payload = json.loads(Path(cohort_path).read_text())
    if (
        payload["schema"] != "genmo.bumi_text_eval.v1"
        or payload["protocol"] != "full_sequence_matched_length"
    ):
        raise ValueError("验证协议不匹配")
    if (payload["ddim_steps"], payload["guidance_scale"], payload["fps"], payload["split"]) != (
        50,
        2.5,
        30,
        "val",
    ):
        raise ValueError("首批固定验证使用DDIM50/CFG2.5/30FPS/val")
    datasets = {
        name: BumiTextDataset(root, "val", dataset=name, caption_sampling="first")
        for name in ("motionmillion", "humanml3d")
    }
    for name, ds in datasets.items():
        if ds.data_identity != payload["data_identity"][name]:
            raise ValueError("验证release身份变化，不能复用原cohort")
    lookups = {
        name: {row[2]["motion_id"]: i for i, row in enumerate(ds.index)}
        for name, ds in datasets.items()
    }
    output.mkdir(parents=True)
    engine = ResidentBumiTextEngine(checkpoint, device=device, ddim_steps=50, output_root=output)
    records = []
    try:
        engine.initialize()
        for number, row in enumerate(payload["records"]):
            ds = datasets[row["dataset"]]
            index = lookups[row["dataset"]][row["motion_id"]]
            sample = ds[index]
            source = ds.read_record(index)
            if (
                sample["length"],
                sample["caption"],
                sample["meta"]["caption_id"],
                row["text_index"],
            ) != (row["frames"], row["caption"], row["caption_id"], 0):
                raise ValueError("cohort的长度或caption映射发生变化")
            arrays = engine.generate_arrays(
                sample["text_embed"][None].to(device),
                sample["text_attention_mask"][None].to(device),
                row["frames"],
                seed=row["seed"],
            )
            job = output / f"{number:03d}"
            job.mkdir()
            np.savez_compressed(job / "motion.npz", **arrays)
            np.savez_compressed(job / "gt.npz", qpos=source["qpos"].numpy(), fps=30)
            write_json(
                job / "metadata.json",
                dict(
                    **row,
                    motion_backend="bumi",
                    robot_manifest=str(engine.robot_manifest),
                    kinematics_path=str(engine.asset_paths["kinematics"]),
                ),
            )
            metrics = {}
            for key, qpos in [
                ("gt", source["qpos"]),
                ("raw", torch.from_numpy(arrays["qpos_raw"])),
                ("postprocessed", torch.from_numpy(arrays["qpos"])),
            ]:
                kwargs = (
                    {}
                    if key == "gt"
                    else {
                        "pred_contact_logits": torch.from_numpy(arrays["foot_contact_logits"]).to(
                            device
                        )
                    }
                )
                values = compute_bumi_kinematic_metrics(
                    qpos.to(device), engine.endecoder.kinematics, **kwargs
                )
                metrics[key] = {k: float(v) for k, v in values.items()}
            write_json(job / "kinematics.json", metrics)
            if render:
                from gem.runtime.text_motion_web.worker import run_renderer

                run_renderer(job, row["frames"])
            records.append(dict(**row, output_dir=job.name, metrics=metrics))
        report = dict(
            **{k: v for k, v in payload.items() if k != "records"},
            records=records,
            checkpoint_sha256=engine.source_checkpoint_sha256,
            contract=engine.contract,
            cohort_sha256=sha256_file(cohort_path),
            semantic_metrics="manual_only",
        )
        write_json(output / "report.json", report)
        with (output / "human_ratings.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "dataset",
                    "motion_id",
                    "caption",
                    "rater_id",
                    "text_match_1_5",
                    "coherence_1_5",
                    "pose_plausibility_1_5",
                    "notes",
                ]
            )
            for row in records:
                writer.writerow(
                    [row["dataset"], row["motion_id"], row["caption"], "", "", "", "", ""]
                )
        return report
    finally:
        engine.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("cohort", "run"):
        c = sub.add_parser(name)
        c.add_argument("--root", type=Path, required=True)
        c.add_argument("--output", type=Path, required=True)
        if name == "cohort":
            c.add_argument("--per-dataset", type=int, default=64)
            c.add_argument("--seed", type=int, default=42)
        else:
            c.add_argument("--cohort", type=Path, required=True)
            c.add_argument("--checkpoint", type=Path, required=True)
            c.add_argument("--device", default="cuda:0")
            c.add_argument("--render", action="store_true")
    a = p.parse_args()
    if a.command == "cohort":
        cohort(a.root, a.output, a.per_dataset, a.seed)
    else:
        run(a.root, a.cohort, a.checkpoint, a.output, a.device, a.render)
