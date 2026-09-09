#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""从 10,000 条 MotionMillion pilot 中确定性选择并渲染 32 条转换动作。

选择器按 caption/ID 将样本分为走跑、舞蹈、武术、地面动作、官方镜像和其他动作，
默认配额为 6/6/6/6/2/6；任一关键类别不足都会失败，避免用 32 条同质动作冒充转换
验收。渲染读取已转换 ``pose[66]+trans[3]+beta[10]`` shard，不再访问 272D 原包；
仅为画面把全局最低顶点平移到地面，保存的动作参数不被修改。

输出视频与 ``pilot_render_manifest.json`` 属于数据侧验收制品，必须写入数据根或明确
评测目录而非 Git。工具不会启动训练，也不会自动接受 gated 数据许可。
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.data.motionmillion.common import (  # noqa: E402
    atomic_write_json,
    read_json,
    safe_torch_load,
    validate_motion_record,
)

QUOTAS = {
    "walk_run": 6,
    "dance": 6,
    "martial": 6,
    "ground": 6,
    "mirror": 2,
    "other": 6,
}
KEYWORDS = {
    "walk_run": ("walk", "run", "jog", "sprint"),
    "dance": ("dance", "dancing", "ballet", "hip hop"),
    "martial": ("martial", "punch", "kick", "karate", "kung fu", "boxing"),
    "ground": ("crawl", "lie", "lying", "floor", "roll over", "get up"),
}


def _is_mirror(motion_id: str) -> bool:
    name = motion_id.replace("\\", "/").rsplit("/", 1)[-1].lower()
    return name.startswith(("m_", "m-", "mirror_", "mirror-")) or (
        len(name) > 1 and name[0] == "m" and name[1].isdigit()
    ) or name.endswith(("_mirror", "-mirror", "_mirrored", "-mirrored"))


def _category(record: dict[str, Any]) -> str:
    if _is_mirror(str(record["motion_id"])):
        return "mirror"
    text = " ".join(str(value).lower() for value in record["captions"])
    for category, keywords in KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return category
    return "other"


def select_records(motion_root: Path, split: str = "train") -> list[dict[str, Any]]:
    manifest = read_json(motion_root / "manifests" / f"{split}.json")
    candidates: dict[str, list[dict[str, Any]]] = {key: [] for key in QUOTAS}
    for shard in manifest["shards"]:
        records = safe_torch_load(motion_root / str(shard["path"]))
        for record in records:
            validate_motion_record(record)
            candidates[_category(record)].append(record)
    selected = []
    for category, quota in QUOTAS.items():
        ordered = sorted(
            candidates[category],
            key=lambda record: hashlib.sha256(
                f"20260909:{record['motion_id']}".encode()
            ).hexdigest(),
        )
        if len(ordered) < quota:
            raise RuntimeError(
                f"pilot 类别 {category} 只有 {len(ordered)} 条，要求至少 {quota} 条"
            )
        selected.extend({"category": category, "record": row} for row in ordered[:quota])
    return selected


def render_pilot(args: argparse.Namespace) -> dict[str, Any]:
    motion_root = Path(args.motion_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    selected = select_records(motion_root, args.split)

    from gem.utils.smplx_utils import make_smplx
    from gem.utils.video_io_utils import save_video
    from scripts.demo.demo_utils import render_global_frames

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"请求 {device}，但 CUDA 不可用")
    model = make_smplx("supermotion", use_pca=False, flat_hand_mean=True).to(device).eval()
    faces = torch.from_numpy(np.asarray(model.faces, dtype=np.int64)).long()
    reports = []
    for index, item in enumerate(selected):
        record = item["record"]
        pose = record["pose"].float()
        trans = record["trans"].float()
        beta = record["beta"].float().repeat(len(pose), 1)
        vertices = []
        with torch.no_grad():
            for indices in torch.arange(len(pose)).split(args.chunk_frames):
                output = model(
                    global_orient=pose[indices, :3].to(device),
                    body_pose=pose[indices, 3:66].to(device),
                    transl=trans[indices].to(device),
                    betas=beta[indices].to(device),
                )
                vertices.append(output.vertices.detach().cpu().float())
        verts = torch.cat(vertices)
        vertical_offset = float(verts[..., 1].min())
        verts[..., 1] -= vertical_offset
        frames = render_global_frames(verts, faces, args.width, args.height)
        video = output_root / f"{index:03d}_{item['category']}.mp4"
        save_video(frames, str(video), fps=Fraction(30, 1))
        reports.append(
            {
                "index": index,
                "category": item["category"],
                "motion_id": record["motion_id"],
                "caption": record["captions"][0],
                "frames": len(pose),
                "video": str(video),
                "visualization_only_vertical_offset_m": vertical_offset,
            }
        )
        print(f"[pilot render] {index + 1}/{len(selected)} {record['motion_id']}", flush=True)
    report = {
        "schema_version": 1,
        "motion_root": str(motion_root),
        "split": args.split,
        "fps": 30,
        "quotas": QUOTAS,
        "records": reports,
    }
    atomic_write_json(output_root / "pilot_render_manifest.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--chunk-frames", type=int, default=64)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if min(args.width, args.height, args.chunk_frames) <= 0:
        raise SystemExit("width、height、chunk-frames 必须为正数")
    report = render_pilot(args)
    print(f"MotionMillion pilot render complete: {len(report['records'])} videos")


if __name__ == "__main__":
    main()
