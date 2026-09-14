"""独立 GPU 常驻工作进程与逐任务视频渲染。

工作进程串行接收冻结的任务；同 checkpoint 复用 T5/GEM，仅在 DDIM 改变时更新
采样器。切换 checkpoint 或加载异常会释放旧引擎。渲染另起子进程，避免 Open3D
原生崩溃中断 HTTP 服务；视频经 H.264/yuv420p 转码和逐帧解码验证后才发布完成事件。
"""

from __future__ import annotations

import ctypes
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

from .storage import ROOT, T5_MODEL, atomic_json, fingerprint


def follow_parent() -> None:
    """Linux 父进程意外退出时结束子进程，防止遗留显存占用。"""
    if sys.platform == "linux":
        parent = os.getppid()
        ctypes.CDLL(None).prctl(1, signal.SIGTERM)
        if parent == 1 or os.getppid() != parent:
            raise SystemExit("父进程已退出")


def check_motion(path: Path, frames: int) -> None:
    import numpy as np

    with np.load(path, allow_pickle=False) as motion:
        for key, width in [("body_pose", 63), ("global_orient", 3), ("transl", 3), ("betas", 10)]:
            if motion[key].shape != (frames, width) or not np.isfinite(motion[key]).all():
                raise ValueError(f"动作 {key} 的形状或有限性校验失败")
        if float(motion["fps"]) != 30 or np.count_nonzero(motion["betas"]):
            raise ValueError("动作 FPS 或零体型契约校验失败")


def check_video(path: Path, frames: int) -> dict:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if (
            stream.codec_context.name != "h264"
            or stream.codec_context.format.name != "yuv420p"
            or float(stream.average_rate) != 30
            or (stream.width, stream.height) != (1280, 720)
        ):
            raise ValueError("视频必须为 1280×720、30 FPS 的 H.264/yuv420p")
        count = 0
        for frame in container.decode(stream):
            if frame.is_corrupt:
                raise ValueError("视频包含损坏帧")
            count += 1
        if count != frames:
            raise ValueError(f"视频帧数不符：得到 {count}，请求 {frames}")
    return {
        "codec": "h264",
        "pixel_format": "yuv420p",
        "fps": 30,
        "frames": count,
        "width": 1280,
        "height": 720,
        "fully_decoded": True,
    }


def render_job(output: Path, frames: int) -> None:
    follow_parent()
    os.chdir(ROOT)
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")
    import torch

    from scripts.demo.demo_smpl_text import render_global_video

    check_motion(output / "motion.npz", frames)
    payload = torch.load(output / "smpl_params.pt", map_location="cuda:0", weights_only=False)
    with torch.inference_mode():
        render_global_video(output, payload["body_params_global"], 1280, 720, 30)
    source = output / "global.mp4"
    if not source.is_file() or source.stat().st_size == 0:
        raise RuntimeError("Open3D 未输出有效视频，请检查 render.log 中的原始错误")
    converted = output / "video.pending.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(source),
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "17",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(converted),
        ],
        check=True,
        timeout=600,
    )
    media = check_video(converted, frames)
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(converted),
            "-frames:v",
            "1",
            "-vf",
            "scale=480:-2",
            str(output / "thumbnail.jpg"),
        ],
        check=True,
        timeout=60,
    )
    converted.replace(output / "video.mp4")
    atomic_json(output / "media_checks.json", media)
    source.unlink()


def run_renderer(output: Path, frames: int) -> dict:
    with (output / "render.log").open("w", encoding="utf-8") as log:
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "gem.runtime.text_motion_web.worker",
                    "--render",
                    str(output),
                    str(frames),
                ],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=1800,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("渲染超过 30 分钟，已结束本次渲染；可降低帧数重试") from exc
    if result.returncode:
        tail = (output / "render.log").read_text(errors="replace")[-4000:]
        raise RuntimeError(f"渲染进程退出（{result.returncode}）：\n{tail}")
    if not (output / "video.mp4").is_file() or not (output / "thumbnail.jpg").is_file():
        raise RuntimeError("渲染未产生视频或缩略图")
    return json.loads((output / "media_checks.json").read_text())


def worker_main(commands, events):
    follow_parent()
    os.chdir(ROOT)
    from gem.runtime.resident_text_motion import ResidentTextMotionEngine

    engine = None
    loaded = None
    try:
        while True:
            job = commands.get()
            if job is None:
                break
            job_id = job["id"]
            stage = "loading"
            started = time.monotonic()

            def report(state, job_id=job_id, started=started, **fields):
                events.put(
                    dict(
                        id=job_id,
                        status=state,
                        elapsed_seconds=time.monotonic() - started,
                        **fields,
                    )
                )

            try:
                report(stage)
                model = job["model"]
                identity = (model["path"], tuple(model["fingerprint"]))
                if fingerprint(Path(model["path"])) != model["fingerprint"]:
                    raise ValueError("checkpoint 已发生变化，请重新选择模型后生成")
                reused = engine is not None and loaded == identity
                if not reused:
                    if engine is not None:
                        engine.close()
                        engine = None
                    engine = ResidentTextMotionEngine(
                        ckpt_path=model["path"],
                        t5_model=T5_MODEL,
                        local_files_only=True,
                        ddim_steps=job["ddim_steps"],
                        guidance_scale=2.5,
                        warmup_enabled=False,
                        output_root=Path(job["task_dir"]) / "artifacts",
                        shape_mode="zero",
                    )
                    engine.initialize()
                    loaded = identity
                else:
                    engine.set_ddim_steps(job["ddim_steps"])
                stage = "generating"
                report(
                    stage,
                    engine_reused=reused,
                    max_text_len=engine.max_text_len,
                    load_seconds=time.monotonic() - started,
                )
                result = engine.generate(
                    dict(
                        prompt=job["prompt"],
                        num_frames=job["num_frames"],
                        fps=30,
                        seed=42,
                        request_id=job_id,
                        output_root=str(Path(job["task_dir"]) / "artifacts"),
                    )
                )
                if not result["ok"]:
                    raise RuntimeError(str(result.get("error", "动作生成失败")))
                output = Path(result["output_dir"])
                stage = "rendering"
                report(stage, output_dir=str(output), timing=result["timing"])
                render_start = time.monotonic()
                media = run_renderer(output, job["num_frames"])
                report(
                    "done",
                    output_dir=str(output),
                    media=media,
                    render_seconds=time.monotonic() - render_start,
                )
            except Exception as exc:
                traceback.print_exc()
                if stage == "loading" and engine is not None:
                    engine.close()
                    engine = None
                    loaded = None
                report("failed", failed_stage=stage, error=f"{type(exc).__name__}: {exc}")
    finally:
        if engine is not None:
            engine.close()


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--render":
        raise SystemExit(
            "用法：python -m gem.runtime.text_motion_web.worker --render OUTPUT FRAMES"
        )
    render_job(Path(sys.argv[2]).resolve(), int(sys.argv[3]))
