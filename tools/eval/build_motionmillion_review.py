#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""生成 MotionMillion-Eval 126 条视频的离线人工评分页面。

工具严格按官方 prompt 文本顺序寻找 ``video-root/000.mp4`` 到 ``125.mp4``，并把
prompt、checkpoint SHA256、统一 seed/DDIM/CFG 与三个 1–4 分评分项写进单文件
HTML。页面不依赖网络，评分保存在浏览器 localStorage，可导出 JSON；它只服务无
GT 的人工 Text Alignment、Motion Smoothness、Physical Plausibility 评审，不参与
官方验证集 FID/R-Precision/Matching Score。
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.data.motionmillion.common import atomic_write_json, sha256_file  # noqa: E402


def _score_buttons(index: int, key: str) -> str:
    return " ".join(
        f'<label><input type="radio" name="{key}_{index}" value="{score}" '
        f'onchange="saveScore({index},\'{key}\',{score})">{score}</label>'
        for score in range(1, 5)
    )


def build_review(args: argparse.Namespace) -> dict:
    prompts = [
        line.strip()
        for line in Path(args.prompt_file).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if len(prompts) != 126 and not args.allow_nonofficial_count:
        raise ValueError(f"官方 MotionMillion-Eval 必须为 126 条，实际 {len(prompts)}")
    video_root = Path(args.video_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    rows = []
    cards = []
    for index, prompt in enumerate(prompts):
        video = video_root / f"{index:03d}.mp4"
        if not video.is_file():
            raise FileNotFoundError(f"缺少固定序号视频: {video}")
        relative_video = Path(os.path.relpath(video, output.parent)).as_posix()
        rows.append({"index": index, "prompt": prompt, "video": str(video)})
        cards.append(
            f"""
<section class="card">
  <h2>{index:03d}</h2><p>{html.escape(prompt)}</p>
  <video controls preload="metadata" src="{html.escape(relative_video)}"></video>
  <div>Text Alignment: {_score_buttons(index, 'alignment')}</div>
  <div>Motion Smoothness: {_score_buttons(index, 'smoothness')}</div>
  <div>Physical Plausibility: {_score_buttons(index, 'physical')}</div>
</section>"""
        )
    protocol = {
        "schema_version": 1,
        "prompt_count": len(prompts),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "seed": args.seed,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "ddim_steps": args.ddim_steps,
        "cfg_scale": args.cfg_scale,
        "postprocess": args.postprocess,
        "items": rows,
    }
    protocol_json = json.dumps(protocol, ensure_ascii=False).replace("</", "<\\/")
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>MotionMillion-Eval Review</title>
<style>
body{{font-family:sans-serif;max-width:1100px;margin:auto;background:#f4f4f4}}
.card{{background:white;margin:18px 0;padding:16px;border-radius:10px}}
video{{width:100%;max-height:620px;background:#111}} label{{margin-right:12px}}
</style></head><body>
<h1>MotionMillion-Eval Review</h1>
<p>checkpoint: {html.escape(str(checkpoint))}<br>seed={args.seed}, frames={args.num_frames},
fps={args.fps}, DDIM={args.ddim_steps}, CFG={args.cfg_scale}</p>
<button onclick="downloadScores()">Export scores.json</button>
{''.join(cards)}
<script>
const protocol={protocol_json};
const storageKey='motionmillion-review-'+protocol.checkpoint_sha256;
let scores=JSON.parse(localStorage.getItem(storageKey)||'{{}}');
function saveScore(i,k,v){{scores[i]=scores[i]||{{}};scores[i][k]=v;
localStorage.setItem(storageKey,JSON.stringify(scores));}}
for(const [i,values] of Object.entries(scores)) for(const [k,v] of Object.entries(values)){{
const node=document.querySelector(`input[name="${{k}}_${{i}}"][value="${{v}}"]`);if(node)node.checked=true;}}
function downloadScores(){{const payload={{...protocol,scores}};
const blob=new Blob([JSON.stringify(payload,null,2)],{{type:'application/json'}});
const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='scores.json';a.click();}}
</script></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(output)
    atomic_write_json(output.with_suffix(".protocol.json"), protocol)
    return {"output": str(output), **protocol}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--num-frames", type=int, default=120)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=2.5)
    parser.add_argument("--postprocess", default="shared-default")
    parser.add_argument("--allow-nonofficial-count", action="store_true")
    return parser


def main() -> None:
    report = build_review(build_parser().parse_args())
    print(f"MotionMillion review page complete: {report['output']}")


if __name__ == "__main__":
    main()
