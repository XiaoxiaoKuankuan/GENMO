#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""把 GENMO 生成的 SMPL 序列导出为 MotionMillion 官方 272D 评测输入。

输入支持文本 demo 发布的 ``motion.npz`` 或 ``smpl_params.pt``。工具默认用 GENMO
现有 SMPL FK 得到 22 个世界关节，再由 MotionMillion heading/局部速度契约编码为
``[F,272]``；也可通过 ``--joint-positions`` 提供已审计的 ``[F,22,3]`` NPY，便于
独立 parity 测试。输出旁的 JSON 绑定源文件与输出 SHA256、FPS、帧数、checkpoint、
DDIM/CFG/seed 等身份信息，供官方 evaluator 的 20-seed 报告引用。

本工具只做表示适配，不修改官方 mean/std、长度 eligibility 或 evaluator 代码，且
不会把 126 条无 GT 的 MotionMillion-Eval 提示混入 FID。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.data.motionmillion.common import (  # noqa: E402
    OFFICIAL_FPS,
    MotionMillionError,
    atomic_save_npy,
    atomic_write_json,
    sha256_file,
    smpl_to_272,
)


def load_generated_smpl(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """读取 demo 的两种公开制品格式并返回统一的世界 SMPL 参数。"""
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            required = {"body_pose", "global_orient", "transl", "betas"}
            missing = sorted(required - set(payload.files))
            if missing:
                raise MotionMillionError(f"NPZ 缺少字段: {missing}")
            params = {key: torch.from_numpy(np.asarray(payload[key])).float() for key in required}
            metadata = {
                "fps": float(np.asarray(payload["fps"]).item())
                if "fps" in payload.files
                else OFFICIAL_FPS
            }
    elif path.suffix in {".pt", ".pth", ".ckpt"}:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict) or not isinstance(
            payload.get("body_params_global"), dict
        ):
            raise MotionMillionError("PT 制品缺少 body_params_global")
        params = {
            key: torch.as_tensor(payload["body_params_global"][key]).float().cpu()
            for key in ("body_pose", "global_orient", "transl", "betas")
        }
        metadata = {
            key: payload[key]
            for key in ("fps", "prompt", "seed", "guidance_scale", "ddim_steps", "checkpoint")
            if key in payload
        }
    else:
        raise MotionMillionError(f"不支持的输入格式: {path}")

    frames = int(params["body_pose"].shape[0])
    expected = {
        "body_pose": (frames, 63),
        "global_orient": (frames, 3),
        "transl": (frames, 3),
        "betas": (frames, 10),
    }
    for key, shape in expected.items():
        if tuple(params[key].shape) != shape or not torch.isfinite(params[key]).all():
            raise MotionMillionError(f"{key} shape/finite 不符合 {shape}")
    if abs(float(metadata.get("fps", OFFICIAL_FPS)) - OFFICIAL_FPS) > 1.0e-6:
        raise MotionMillionError("MotionMillion v1 evaluator adapter 只接受 30 FPS")
    return params, metadata


def compute_world_joints(
    params: dict[str, torch.Tensor], endecoder: Any | None = None
) -> torch.Tensor:
    """使用 GENMO 当前 SMPL skeleton 做 22 关节 FK。"""
    if endecoder is None:
        from gem.network.endecoder import EnDecoder

        endecoder = EnDecoder(
            stats_name="MM_V1_AMASS_LOCAL_BEDLAM_CAM",
            feat_dim=151,
            encode_type="gvhmr",
            clip_std=True,
        ).eval()
    with torch.no_grad():
        joints = endecoder.fk_v2(
            body_pose=params["body_pose"][None],
            betas=params["betas"][None],
            global_orient=params["global_orient"][None],
            transl=params["transl"][None],
        )[0]
    if tuple(joints.shape) != (len(params["body_pose"]), 22, 3):
        raise MotionMillionError(f"SMPL FK 输出 shape 异常: {tuple(joints.shape)}")
    return joints.cpu()


def export_motion(
    args: argparse.Namespace, *, endecoder: Any | None = None
) -> dict[str, Any]:
    source = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    params, metadata = load_generated_smpl(source)
    if args.joint_positions is None:
        joints = compute_world_joints(params, endecoder=endecoder)
        joint_source = "genmo_smpl_fk"
    else:
        joint_path = Path(args.joint_positions).expanduser().resolve()
        joints = torch.from_numpy(np.load(joint_path, allow_pickle=False)).float()
        joint_source = str(joint_path)
    pose = torch.cat([params["global_orient"], params["body_pose"]], dim=-1)
    encoded = smpl_to_272(pose, params["transl"], joints)
    atomic_save_npy(output, encoded.numpy())
    report = {
        "schema_version": 1,
        "representation": "motionmillion_272rpr",
        "input": str(source),
        "input_sha256": sha256_file(source),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "frames": int(encoded.shape[0]),
        "fps": OFFICIAL_FPS,
        "source_up_axis": "y",
        "joint_source": joint_source,
        "metadata": metadata,
    }
    atomic_write_json(output.with_suffix(output.suffix + ".json"), report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--joint-positions", type=Path)
    return parser


def main() -> None:
    report = export_motion(build_parser().parse_args())
    print(
        f"MotionMillion 272D export complete: frames={report['frames']}, "
        f"sha256={report['output_sha256']}"
    )


if __name__ == "__main__":
    main()
