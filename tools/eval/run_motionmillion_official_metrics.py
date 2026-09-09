#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""用固定官方 MotionMillion evaluator 计算一轮 GENMO 验证集指标。

输入 prediction manifest 是 JSONL，每行包含 ``motion_id/caption/path/length``，其中
path 指向 GENMO SMPL 已转换回的原始尺度 272D ``.npy``。工具重验 evaluator 身份与
完整官方 val 资格集合，按固定 seed 模拟官方 ``shuffle=True, batch=32, drop_last``，
调用官方 ``EvaluatorModelWrapper272RPR`` 提取文本/动作 embedding，再按官方公式计算
FID、Diversity、R-Precision@1/2/3 和 Matching Score。

本工具不会生成动作，也不会选择 checkpoint；每个 seed 产出一份身份绑定 JSON，随后
由 ``summarize_motionmillion_metrics.py`` 对恰好 20 个不同 seed 汇总 95% CI。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import linalg

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.data.motionmillion.common import (  # noqa: E402
    atomic_write_json,
    read_json,
    sha256_file,
)


@contextlib.contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"prediction manifest 第 {line_number} 行不是对象")
            rows.append(row)
    return rows


def calculate_r_precision(
    text_embedding: np.ndarray, motion_embedding: np.ndarray
) -> tuple[np.ndarray, float]:
    squared = (
        -2.0 * text_embedding @ motion_embedding.T
        + np.square(text_embedding).sum(axis=1, keepdims=True)
        + np.square(motion_embedding).sum(axis=1)
    )
    distances = np.sqrt(np.maximum(squared, 0.0))
    matching = float(np.trace(distances))
    ranking = np.argsort(distances, axis=1)
    target = np.arange(len(ranking))[:, None]
    matches = ranking[:, :3] == target
    cumulative = np.maximum.accumulate(matches, axis=1)
    return cumulative.sum(axis=0), matching


def calculate_fid(real: np.ndarray, generated: np.ndarray) -> float:
    real_mean, generated_mean = real.mean(axis=0), generated.mean(axis=0)
    real_cov = np.cov(real, rowvar=False)
    generated_cov = np.cov(generated, rowvar=False)
    covariance_mean, _ = linalg.sqrtm(real_cov @ generated_cov, disp=False)
    if not np.isfinite(covariance_mean).all():
        offset = np.eye(real_cov.shape[0]) * 1.0e-6
        covariance_mean = linalg.sqrtm((real_cov + offset) @ (generated_cov + offset))
    if np.iscomplexobj(covariance_mean):
        if not np.allclose(np.diag(covariance_mean).imag, 0, atol=1.0e-3):
            raise ValueError("FID covariance sqrt 包含不可接受的虚部")
        covariance_mean = covariance_mean.real
    difference = real_mean - generated_mean
    value = float(
        difference @ difference
        + np.trace(real_cov)
        + np.trace(generated_cov)
        - 2.0 * np.trace(covariance_mean)
    )
    # 理论 FID 非负；完全相同输入在 scipy sqrtm 下可能出现约 1e-9 的负舍入误差。
    return max(value, 0.0)


def calculate_diversity(embedding: np.ndarray, seed: int) -> float:
    count = 300 if len(embedding) > 300 else 100
    if len(embedding) <= count:
        raise ValueError(f"Diversity 至少需要 {count + 1} 条有效样本")
    rng = np.random.RandomState(seed)
    first = rng.choice(len(embedding), count, replace=False)
    second = rng.choice(len(embedding), count, replace=False)
    return float(linalg.norm(embedding[first] - embedding[second], axis=1).mean())


def _load_motion(path: Path, length: int, *, max_length: int = 300) -> torch.Tensor:
    motion = np.load(path, allow_pickle=False)
    if motion.ndim != 2 or motion.shape[1] != 272 or len(motion) != length:
        raise ValueError(f"272D prediction/GT shape 与 length 不一致: {path}: {motion.shape}/{length}")
    if not 1 <= length <= max_length or not np.isfinite(motion).all():
        raise ValueError(f"272D prediction/GT length/finite 异常: {path}")
    padded = np.zeros((max_length, 272), dtype=np.float32)
    padded[:length] = motion.astype(np.float32, copy=False)
    return torch.from_numpy(padded)


def run_metrics(args: argparse.Namespace) -> dict[str, Any]:
    identity_path = Path(args.evaluator_identity).expanduser().resolve()
    identity = read_json(identity_path)
    if identity.get("status") != "PASS" or not identity.get("evaluator_fingerprint"):
        raise ValueError("evaluator identity 未通过 fingerprint 预检")
    for row in identity["files"]:
        path = Path(row["path"])
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise ValueError(f"evaluator 身份文件漂移: {path}")

    eligibility = read_json(args.eligibility)
    dataset_root = Path(eligibility["dataset_root"])
    eligible = {row["motion_id"]: row for row in eligibility["records"]}
    predictions = _read_jsonl(Path(args.predictions))
    generation_progress_path = Path(args.predictions).parent / "generation_progress.json"
    generation_progress = read_json(generation_progress_path)
    generation_identity = generation_progress.get("identity", {})
    if generation_progress.get("status") != "complete":
        raise ValueError("验证集 prediction generation_progress 尚未 complete")
    expected_generation_identity = {
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "eligibility_sha256": sha256_file(args.eligibility),
        "global_seed": int(args.seed),
        "num_frames": 120,
        "ddim_steps": int(args.ddim_steps),
        "cfg_scale": float(args.cfg_scale),
    }
    actual_generation_identity = {
        key: generation_identity.get(key) for key in expected_generation_identity
    }
    if actual_generation_identity != expected_generation_identity:
        raise ValueError(
            "prediction 生成身份与本轮指标参数不一致: "
            f"expected={expected_generation_identity}, actual={actual_generation_identity}"
        )
    prediction_by_id = {str(row.get("motion_id")): row for row in predictions}
    if len(prediction_by_id) != len(predictions):
        raise ValueError("prediction manifest motion_id 重复")
    if set(prediction_by_id) != set(eligible):
        missing = sorted(set(eligible) - set(prediction_by_id))[:20]
        extra = sorted(set(prediction_by_id) - set(eligible))[:20]
        raise ValueError(f"prediction 与完整 eligibility 不闭环: missing={missing}, extra={extra}")

    generator = torch.Generator().manual_seed(int(args.seed))
    order = torch.randperm(len(eligible), generator=generator).tolist()
    ordered_ids = sorted(eligible)
    ordered_ids = [ordered_ids[index] for index in order]
    usable_count = len(ordered_ids) // args.batch_size * args.batch_size
    ordered_ids = ordered_ids[:usable_count]
    if usable_count == 0:
        raise ValueError("drop_last 后没有 evaluator 样本")

    code_root = Path(identity["code_root"])
    evaluator_root = Path(identity["root"])
    sys.path.insert(0, str(code_root))
    try:
        from models.evaluator_wrapper_motionmillion_rpr272 import (  # type: ignore
            EvaluatorModelWrapper272RPR,
        )

        with _working_directory(evaluator_root):
            wrapper = EvaluatorModelWrapper272RPR(
                argparse.Namespace(dataname="motionmillion"), torch.device(args.device)
            )
    finally:
        if sys.path[0] == str(code_root):
            sys.path.pop(0)

    real_embeddings = []
    generated_embeddings = []
    r_precision = np.zeros(3, dtype=np.float64)
    matching_score = 0.0
    with torch.inference_mode(), _working_directory(evaluator_root):
        for start in range(0, usable_count, args.batch_size):
            ids = ordered_ids[start : start + args.batch_size]
            captions = []
            gt_motions = []
            gt_lengths = []
            pred_motions = []
            pred_lengths = []
            for motion_id in ids:
                source = eligible[motion_id]
                prediction = prediction_by_id[motion_id]
                caption = str(prediction.get("caption", "")).strip()
                if caption not in source["captions"]:
                    raise ValueError(f"prediction caption 不属于官方文本: {motion_id}")
                captions.append(caption)
                gt_lengths.append(int(source["frames"]))
                gt_motions.append(
                    _load_motion(dataset_root / source["motion_path"], int(source["frames"]))
                )
                prediction_path = Path(prediction["path"]).expanduser().resolve()
                if sha256_file(prediction_path) != prediction.get("sha256"):
                    raise ValueError(f"prediction 272D SHA256 漂移: {motion_id}")
                pred_lengths.append(int(prediction["length"]))
                pred_motions.append(_load_motion(prediction_path, int(prediction["length"])))
            gt_batch = torch.stack(gt_motions).to(args.device)
            pred_batch = torch.stack(pred_motions).to(args.device)
            gt_length_tensor = torch.tensor(gt_lengths, dtype=torch.long, device=args.device)
            pred_length_tensor = torch.tensor(pred_lengths, dtype=torch.long, device=args.device)
            text_embedding, real_embedding = wrapper.get_co_embeddings(
                captions, gt_batch, gt_length_tensor
            )
            pred_text_embedding, generated_embedding = wrapper.get_co_embeddings(
                captions, pred_batch, pred_length_tensor
            )
            real_embeddings.append(real_embedding.cpu().numpy())
            generated_embeddings.append(generated_embedding.cpu().numpy())
            batch_r, batch_matching = calculate_r_precision(
                pred_text_embedding.cpu().numpy(), generated_embedding.cpu().numpy()
            )
            r_precision += batch_r
            matching_score += batch_matching

    real = np.concatenate(real_embeddings)
    generated = np.concatenate(generated_embeddings)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    config = Path(args.experiment_config).expanduser().resolve()
    dataset_release = read_json(args.dataset_release)
    result = {
        "schema_version": 1,
        "seed": int(args.seed),
        "fid": calculate_fid(real, generated),
        "diversity": calculate_diversity(generated, int(args.seed)),
        "r_precision_1": float(r_precision[0] / usable_count),
        "r_precision_2": float(r_precision[1] / usable_count),
        "r_precision_3": float(r_precision[2] / usable_count),
        "matching_score": float(matching_score / usable_count),
        "eligible_count": len(eligible),
        "evaluated_count": usable_count,
        "dropped_by_official_batch_rule": len(eligible) - usable_count,
        "batch_size": int(args.batch_size),
        "checkpoint_sha256": sha256_file(checkpoint),
        "experiment_config_sha256": sha256_file(config),
        "dataset_release_fingerprint": dataset_release["build_fingerprint"],
        "evaluator_fingerprint": identity["evaluator_fingerprint"],
        "ddim_steps": int(args.ddim_steps),
        "cfg_scale": float(args.cfg_scale),
        "prediction_manifest_sha256": sha256_file(args.predictions),
        "generation_progress_sha256": sha256_file(generation_progress_path),
        "evaluator_identity_sha256": sha256_file(identity_path),
    }
    if not all(
        math.isfinite(float(result[key]))
        for key in (
            "fid",
            "diversity",
            "r_precision_1",
            "r_precision_2",
            "r_precision_3",
            "matching_score",
        )
    ):
        raise ValueError("官方 evaluator 输出包含 NaN/Inf")
    atomic_write_json(Path(args.output), result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--eligibility", type=Path, required=True)
    parser.add_argument("--evaluator-identity", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--dataset-release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=2.5)
    return parser


def main() -> None:
    result = run_metrics(build_parser().parse_args())
    print(
        f"MotionMillion official metrics seed={result['seed']}: "
        f"FID={result['fid']:.6f}, R@1={result['r_precision_1']:.6f}"
    )


if __name__ == "__main__":
    main()
