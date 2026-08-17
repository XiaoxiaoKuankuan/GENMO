#!/usr/bin/env python3
"""Render original/generated pairs and build a synchronized comparison page."""

from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.demo.demo_music import mux_selected_audio, render_global_motion  # noqa: E402


DATASET_LABELS = {
    "aistpp": "AIST++",
    "aioz_gdance": "AIOZ-GDANCE",
    "finedance": "FineDance",
    "compas3d": "CoMPAS3D",
}


def probe_video(path: Path, expected_frames: int) -> None:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,r_frame_rate:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    video = [stream for stream in streams if stream.get("codec_type") == "video"]
    audio = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if len(video) != 1 or video[0].get("r_frame_rate") != "30/1" or not audio:
        raise RuntimeError(f"invalid comparison media streams: {path}")
    duration = float(payload["format"]["duration"])
    if abs(duration - expected_frames / 30.0) > 0.15:
        raise RuntimeError(
            f"duration mismatch for {path}: {duration} vs {expected_frames / 30.0}"
        )


def resolve_audio(staging: Path, item: dict[str, Any]) -> Path:
    if item.get("audio"):
        path = staging / item["audio"]
    else:
        path = Path(item["aist_audio_hint"])
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def render_original(motion: Path, audio: Path, output: Path, work: Path, width: int, height: int) -> None:
    with np.load(motion, allow_pickle=False) as source:
        pose = torch.from_numpy(np.ascontiguousarray(source["pose"], dtype=np.float32))
        transl = torch.from_numpy(np.ascontiguousarray(source["transl"], dtype=np.float32))
        betas = torch.from_numpy(np.ascontiguousarray(source["betas"], dtype=np.float32))
    frames = len(pose)
    render_dir = work / "original"
    render_dir.mkdir(parents=True, exist_ok=True)
    rendered = render_global_motion(
        render_dir,
        {
            "global_orient": pose[:, :3],
            "body_pose": pose[:, 3:66],
            "transl": transl,
            "betas": torch.zeros_like(betas),
        },
        width,
        height,
    )
    if rendered is None or not mux_selected_audio(
        rendered, audio, output, 0.0, frames / 30.0
    ):
        raise RuntimeError(f"original render/mux failed: {motion}")


def render_generated(
    item: dict[str, Any],
    audio: Path,
    checkpoint: Path,
    output: Path,
    work: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    generated_dir = work / "generated"
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/demo_music_only.py"),
        "--audio",
        str(audio),
        "--audio-duration-sec",
        f"{item['clip_duration_sec']:.9f}",
        "--num-frames",
        str(item["clip_frames"]),
        "--max-frames",
        str(item["clip_frames"] + 2),
        "--ckpt",
        str(checkpoint),
        "--output-dir",
        str(generated_dir),
        "--cfg-scale",
        str(args.cfg_scale),
        "--ddim-steps",
        str(args.ddim_steps),
        "--seed",
        str(args.seed + int(item["number"]) - 1),
        "--postproc",
        "--allow-physically-invalid",
        "--width",
        str(args.width),
        "--height",
        str(args.height),
    ]
    log_path = work / "generated.log"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = args.device
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    report_path = generated_dir / "demo_report.json"
    if not report_path.is_file():
        raise RuntimeError(
            f"generated demo failed with code {process.returncode}; see {log_path}"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    generated_video = generated_dir / "motion_with_audio.mp4"
    if not generated_video.is_file():
        raise RuntimeError(f"generated video missing; see {log_path}")
    shutil.copy2(generated_video, output)
    return report


def page(items: list[dict[str, Any]], output_root: Path, checkpoint: Path, args: argparse.Namespace) -> None:
    cards: list[str] = []
    for item in items:
        dataset = item["dataset"]
        sample = item["sample_id"]
        base = f"videos/{dataset}/{sample}"
        sanity = item.get("generated_sanity") or {}
        passed = bool(sanity.get("physical_sanity_pass", False))
        badge = "通过粗检" if passed else "需人工检查"
        badge_class = "pass" if passed else "warn"
        key = html.escape(f"{dataset}::{sample}", quote=True)
        cards.append(
            f'''<article class="card pair" data-key="{key}">
  <header><div><span class="dataset">{html.escape(DATASET_LABELS[dataset])}</span>
  <h2>{html.escape(sample)}</h2></div><span class="badge {badge_class}">{badge}</span></header>
  <div class="meta">{html.escape(item['split'])} · {item['clip_frames']} 帧 · {item['clip_duration_sec']:.2f} 秒 · seed {item['seed']}</div>
  <div class="videos">
    <section><h3>原始舞蹈（音频源）</h3><video class="original" preload="metadata" controls src="{base}_original.mp4"></video></section>
    <section><h3>模型生成（默认静音同步）</h3><video class="generated" preload="metadata" controls muted src="{base}_generated.mp4"></video></section>
  </div>
  <div class="actions"><button data-action="play">同步播放</button><button data-action="pause">暂停</button><button data-action="restart">回到开头</button><span class="sync">等待播放</span></div>
  <div class="ratings"><label>节奏相似度 <input data-rating="rhythm" type="range" min="1" max="5" step="1" value="3"><output>3</output></label>
  <label>动作/风格相似度 <input data-rating="style" type="range" min="1" max="5" step="1" value="3"><output>3</output></label></div>
</article>'''
        )
    html_text = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GENMO 四数据集原始/生成舞蹈对比</title><style>
:root{{--bg:#0b0d12;--panel:#151922;--line:#2b3240;--text:#edf2f7;--muted:#98a2b3;--blue:#66a3ff;--green:#42d392;--orange:#ffb454}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}}main{{max-width:1500px;margin:auto;padding:28px}}
.hero{{padding:24px;background:linear-gradient(135deg,#17213b,#151922);border:1px solid var(--line);border-radius:18px;margin-bottom:22px}}h1{{margin:0 0 8px;font-size:30px}}.hero p{{color:var(--muted);max-width:1000px}}.toolbar{{position:sticky;top:0;z-index:5;padding:12px 0;background:#0b0d12e8;backdrop-filter:blur(8px)}}button,select{{background:#252c3a;color:var(--text);border:1px solid #3a4558;border-radius:8px;padding:8px 13px;cursor:pointer}}button:hover{{border-color:var(--blue)}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:16px;margin:18px 0}}header{{display:flex;justify-content:space-between;gap:12px;align-items:center}}h2{{font-size:18px;margin:2px 0}}.dataset{{color:var(--blue);font-size:12px;text-transform:uppercase}}.badge{{font-size:12px;border-radius:99px;padding:4px 9px}}.pass{{background:#153d2f;color:var(--green)}}.warn{{background:#4a3219;color:var(--orange)}}.meta,.sync{{color:var(--muted)}}
.videos{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:12px}}.videos h3{{font-size:14px;margin:0 0 7px}}video{{display:block;width:100%;aspect-ratio:16/9;background:#050609;border-radius:10px}}.actions,.ratings{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:12px}}.ratings label{{display:flex;align-items:center;gap:8px;background:#10131a;padding:7px 10px;border-radius:8px}}input[type=range]{{accent-color:var(--blue)}}
@media(max-width:850px){{main{{padding:12px}}.videos{{grid-template-columns:1fr}}}}
</style></head><body><main>
<section class="hero"><h1>GENMO：原始舞蹈 vs 音乐条件生成</h1>
<p>共 {len(items)} 对 held-out 样本。左右使用同一段音乐、相同帧数和 30 FPS；默认左侧播放音频，右侧静音同步，避免双重回声。这里适合人工比较节奏、动作强度和风格；音乐条件生成并不以逐帧复制原舞蹈为目标，因此不展示容易误导的逐关节“相似度”。评分仅保存在当前浏览器。</p>
<div>Checkpoint: {html.escape(checkpoint.name)} · DDIM {args.ddim_steps} · CFG {args.cfg_scale}</div></section>
<div class="toolbar"><select id="filter"><option value="all">全部数据集</option>{''.join(f'<option value="{key}">{label}</option>' for key,label in DATASET_LABELS.items())}</select> <button id="pauseAll">全部暂停</button></div>
{''.join(cards)}
</main><script>
const pairs=[...document.querySelectorAll('.pair')];
function syncPair(card){{
 const a=card.querySelector('.original'),b=card.querySelector('.generated'),s=card.querySelector('.sync');
 let timer=null,syncing=false;
 b.muted=true;
 const stopSync=message=>{{syncing=false;if(timer!==null){{clearInterval(timer);timer=null}}if(message)s.textContent=message}};
 const pauseBoth=message=>{{stopSync(message);a.pause();b.pause()}};
 const restart=()=>{{pauseBoth('已回到开头');a.currentTime=0;b.currentTime=0}};
 const play=async()=>{{
  stopSync();syncing=true;
  const duration=Number.isFinite(a.duration)?a.duration:0;
  if(a.ended||(duration>0&&a.currentTime>=duration-.05))a.currentTime=0;
  b.currentTime=a.currentTime;
  const results=await Promise.allSettled([a.play(),b.play()]);
  if(!syncing)return;
  if(results.some(result=>result.status==='rejected')||a.paused||b.paused){{
   pauseBoth('同步播放失败，请重试');return;
  }}
  timer=setInterval(()=>{{
   if(!syncing||a.paused||b.paused||a.ended||b.ended){{pauseBoth('同步已停止');return}}
   const d=Math.abs(a.currentTime-b.currentTime);
   if(d>.08)b.currentTime=a.currentTime;
   s.textContent=`同步误差 ${{d.toFixed(3)}} 秒`;
  }},250);
 }};
 card.querySelector('[data-action=play]').onclick=play;
 card.querySelector('[data-action=pause]').onclick=()=>pauseBoth('已暂停');
 card.querySelector('[data-action=restart]').onclick=restart;
 a.addEventListener('seeking',()=>{{if(syncing)b.currentTime=a.currentTime}});
 a.addEventListener('pause',()=>{{if(syncing)pauseBoth('同步已停止')}});
 b.addEventListener('pause',()=>{{if(syncing)pauseBoth('同步已停止')}});
 a.addEventListener('ended',()=>{{if(syncing)pauseBoth('播放结束')}});
 b.addEventListener('ended',()=>{{if(syncing)pauseBoth('播放结束')}});
 return{{pause:()=>pauseBoth('已暂停')}};
}}
const syncControls=pairs.map(card=>{{
 const controls=syncPair(card);
 card.querySelectorAll('[data-rating]').forEach(input=>{{const id=`genmo-rating:${{card.dataset.key}}:${{input.dataset.rating}}`;const saved=localStorage.getItem(id);if(saved)input.value=saved;input.nextElementSibling.value=input.value;input.oninput=()=>{{input.nextElementSibling.value=input.value;localStorage.setItem(id,input.value)}}}});
 return controls;
}});
document.getElementById('pauseAll').onclick=()=>syncControls.forEach(controls=>controls.pause());
document.getElementById('filter').onchange=e=>pairs.forEach(c=>{{c.hidden=e.target.value!=='all'&&!c.dataset.key.startsWith(e.target.value+'::')}});
</script></body></html>'''
    (output_root / "index.html").write_text(html_text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--delete-staging", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    staging = args.staging.resolve()
    checkpoint = args.ckpt.resolve()
    output_root = args.output_root.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    manifest = json.loads((staging / "manifest.json").read_text())
    items = manifest["items"][: args.limit]
    output_root.mkdir(parents=True, exist_ok=True)
    work_root = output_root / "_work"
    results: list[dict[str, Any]] = []
    for position, item in enumerate(items, start=1):
        started = time.monotonic()
        dataset = item["dataset"]
        sample = item["sample_id"]
        final_dir = output_root / "videos" / dataset
        final_dir.mkdir(parents=True, exist_ok=True)
        original = final_dir / f"{sample}_original.mp4"
        generated = final_dir / f"{sample}_generated.mp4"
        audio = resolve_audio(staging, item)
        work = work_root / dataset / sample
        if not (original.is_file() and generated.is_file()):
            if work.exists():
                shutil.rmtree(work)
            work.mkdir(parents=True)
            print(f"[{position:02d}/{len(items):02d}] original {dataset}/{sample}", flush=True)
            render_original(staging / item["motion"], audio, original, work, args.width, args.height)
            print(f"[{position:02d}/{len(items):02d}] generated {dataset}/{sample}", flush=True)
            report = render_generated(item, audio, checkpoint, generated, work, args)
            item["generated_sanity"] = report.get("motion_sanity")
            shutil.rmtree(work)
        else:
            item["generated_sanity"] = {"physical_sanity_pass": True, "reused": True}
        probe_video(original, item["clip_frames"])
        probe_video(generated, item["clip_frames"])
        item["seed"] = args.seed + int(item["number"]) - 1
        results.append(item)
        page(results, output_root, checkpoint, args)
        print(
            f"[{position:02d}/{len(items):02d}] done {dataset}/{sample} "
            f"in {time.monotonic() - started:.1f}s",
            flush=True,
        )
    if work_root.exists():
        shutil.rmtree(work_root)
    page(results, output_root, checkpoint, args)
    if args.delete_staging and len(items) == len(manifest["items"]):
        shutil.rmtree(staging)
        print(f"removed staging data: {staging}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
