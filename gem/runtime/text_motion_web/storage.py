"""本地工作台的轻量持久化与固定参数。

JSON 通过同目录临时文件、fsync 和原子替换发布；不导入 CUDA 或模型库，
使 HTTP 进程启动和轮询不依赖 GPU 初始化。任务目录和历史保存在指定输出根目录。
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = (
    ROOT / "inputs/checkpoints/motionmillion_text_only_s190000_20260914/s190000.ckpt"
)
T5_MODEL = "/home/weili/.cache/huggingface/hub/models--t5-3b/snapshots/bed96aab9ee46012a5046386105ee5fd0ac572f0"
FIXED = dict(
    fps=30,
    seed=42,
    guidance_scale=2.5,
    shape_mode="zero",
    width=1280,
    height=720,
    t5_model=T5_MODEL,
    local_files_only=True,
    postproc=True,
)
TERMINAL = {"done", "failed"}
HISTORY_LIMIT = 60


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def fingerprint(path: Path) -> list[int]:
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
