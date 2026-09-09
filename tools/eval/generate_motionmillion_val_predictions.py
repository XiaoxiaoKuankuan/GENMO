#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""常驻加载 T5/GEM，为完整 MotionMillion val eligibility 生成一轮 272D 预测。

工具按 ``eligibility.json`` 的完整动作集合，为每个 motion 用全局 seed 和 motion ID
确定性选择一条官方 caption，并派生独立 DDIM seed。T5-3B、GEM checkpoint 与 SMPL FK
各只加载一次；每条成功结果先按 READY 协议保存 SMPL，再转换成官方 272D，最后原子
更新进度和 ``predictions_272.jsonl``。恢复只接受 checkpoint、eligibility、T5 路径、
DDIM/CFG/帧数/后处理完全一致的身份。

运行本工具会实际占用 GPU 并执行完整验证集推理，必须在服务器 1 GPU 空闲且用户单独
授权后运行；代码实现本身不构成启动授权。
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gem.network.endecoder import EnDecoder  # noqa: E402
from gem.runtime.resident_text_motion import (  # noqa: E402
    ResidentTextMotionEngine,
    TextMotionRequest,
)
from tools.data.motionmillion.common import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
    read_json,
    sha256_file,
)
from tools.eval.motionmillion_smpl_to_272 import export_motion  # noqa: E402


def select_caption_and_seed(
    motion_id: str, captions: list[str], global_seed: int
) -> tuple[int, str, int]:
    """不依赖 Python hash 随机化地选择 caption 并派生逐动作 seed。"""
    if not captions or any(not str(value).strip() for value in captions):
        raise ValueError(f"官方 caption 为空: {motion_id}")
    digest = hashlib.sha256(f"{global_seed}:{motion_id}".encode()).digest()
    text_index = int.from_bytes(digest[:8], "big") % len(captions)
    sample_seed = int.from_bytes(digest[8:12], "big") % (2**31 - 1)
    return text_index, str(captions[text_index]), sample_seed


def _identity(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    eligibility = Path(args.eligibility).expanduser().resolve()
    t5_model = Path(args.t5_model).expanduser().resolve()
    t5_release_path = Path(args.t5_release).expanduser().resolve()
    if not checkpoint.is_file() or not eligibility.is_file():
        raise FileNotFoundError("checkpoint 或 eligibility 不存在")
    if not t5_model.is_dir() or not t5_release_path.is_file():
        raise FileNotFoundError(f"T5 必须是已固定 revision 的本地快照路径: {t5_model}")
    t5_release = read_json(t5_release_path)
    model_files = list(t5_release.get("model_files", []))
    if not model_files:
        raise ValueError("T5 embedding release 没有模型文件指纹")
    for row in model_files:
        path = t5_model / str(row["path"])
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise ValueError(f"T5 本地快照与训练 embedding 身份不一致: {path}")
    return {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "eligibility": str(eligibility),
        "eligibility_sha256": sha256_file(eligibility),
        "t5_model": str(t5_model),
        "t5_release": str(t5_release_path),
        "t5_release_sha256": sha256_file(t5_release_path),
        "t5_resolved_revision": t5_release["resolved_revision"],
        "global_seed": int(args.seed),
        "num_frames": int(args.num_frames),
        "fps": 30.0,
        "ddim_steps": int(args.ddim_steps),
        "cfg_scale": float(args.cfg_scale),
        "postprocess": not args.no_postprocess,
    }


def generate_predictions(args: argparse.Namespace) -> dict[str, Any]:
    if args.num_frames != 120:
        raise ValueError("MotionMillion v1 正式评测只承诺 120 帧")
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / "generation_progress.json"
    identity = _identity(args)
    completed: dict[str, dict[str, Any]] = {}
    if progress_path.is_file():
        if not args.resume:
            raise FileExistsError(f"已有生成进度: {progress_path}；请使用 --resume")
        previous = read_json(progress_path)
        if previous.get("identity") != identity:
            raise ValueError("已有验证集生成进度与本次身份不一致")
        for row in previous.get("records", []):
            path = Path(row["path"])
            if not path.is_file() or sha256_file(path) != row["sha256"]:
                raise ValueError(f"已完成 272D 预测漂移: {path}")
            completed[str(row["motion_id"])] = row

    eligibility = read_json(identity["eligibility"])
    source_rows = list(eligibility["records"])
    engine = ResidentTextMotionEngine(
        ckpt_path=identity["checkpoint"],
        t5_model=identity["t5_model"],
        device=args.device,
        text_dtype=args.text_dtype,
        local_files_only=True,
        ddim_steps=args.ddim_steps,
        guidance_scale=args.cfg_scale,
        output_root=output_root / "smpl",
        postproc=not args.no_postprocess,
        shape_mode="zero",
        warmup_enabled=args.warmup,
        max_frames=120,
    )
    endecoder = EnDecoder(
        stats_name="MM_V1_AMASS_LOCAL_BEDLAM_CAM",
        feat_dim=151,
        encode_type="gvhmr",
        clip_std=True,
    ).eval()
    engine.initialize()
    try:
        for index, source in enumerate(source_rows, start=1):
            motion_id = str(source["motion_id"])
            if motion_id in completed:
                continue
            text_index, caption, sample_seed = select_caption_and_seed(
                motion_id, list(source["captions"]), args.seed
            )
            response = engine.generate(
                TextMotionRequest(
                    prompt=caption,
                    num_frames=120,
                    fps=30.0,
                    seed=sample_seed,
                    request_id=f"motionmillion-val-{motion_id}",
                    metadata={
                        "motion_id": motion_id,
                        "text_index": text_index,
                        "global_seed": args.seed,
                    },
                )
            )
            if not response.get("ok"):
                raise RuntimeError(f"生成失败: {motion_id}: {response}")
            output = output_root / "272" / f"{motion_id}.npy"
            exported = export_motion(
                argparse.Namespace(
                    input=Path(response["motion_npz"]),
                    output=output,
                    joint_positions=None,
                ),
                endecoder=endecoder,
            )
            row = {
                "motion_id": motion_id,
                "caption": caption,
                "text_index": text_index,
                "sample_seed": sample_seed,
                "path": exported["output"],
                "sha256": exported["output_sha256"],
                "length": 120,
                "source_smpl": response["motion_npz"],
            }
            completed[motion_id] = row
            ordered_completed = [
                completed[str(item["motion_id"])]
                for item in source_rows
                if str(item["motion_id"]) in completed
            ]
            atomic_write_json(
                progress_path,
                {
                    "schema_version": 1,
                    "status": "in_progress",
                    "identity": identity,
                    "completed": len(completed),
                    "total": len(source_rows),
                    "records": ordered_completed,
                },
            )
            print(
                f"[MotionMillion val] generated={index}/{len(source_rows)}, "
                f"motion_id={motion_id}",
                flush=True,
            )
    finally:
        engine.close()

    if len(completed) != len(source_rows):
        raise RuntimeError("验证集生成未闭环")
    ordered = [completed[str(row["motion_id"])] for row in source_rows]
    manifest_path = output_root / "predictions_272.jsonl"
    atomic_write_jsonl(manifest_path, ordered)
    report = {
        "schema_version": 1,
        "status": "complete",
        "identity": identity,
        "record_count": len(ordered),
        "prediction_manifest": str(manifest_path),
        "prediction_manifest_sha256": sha256_file(manifest_path),
        "records": ordered,
    }
    atomic_write_json(progress_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eligibility", type=Path, required=True)
    parser.add_argument("--t5-model", type=Path, required=True)
    parser.add_argument("--t5-release", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-frames", type=int, default=120)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=2.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--text-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--no-postprocess", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    report = generate_predictions(build_parser().parse_args())
    print(
        "MotionMillion val generation complete: "
        f"records={report['record_count']}, manifest={report['prediction_manifest']}"
    )


if __name__ == "__main__":
    main()
