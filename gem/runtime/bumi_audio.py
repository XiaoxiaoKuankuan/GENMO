"""BUMI 音频子进程管理：由 Bridge 或独立播放器复用。

使用 ffplay 播放选定的音乐区间；替换、停止和回收只针对本对象创建的子进程，
不阻塞 50 Hz 播放线程，不依赖 Redis、ROS、策略文件或训练框架。
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from pathlib import Path


class AudioController:
    """非阻塞管理 ffplay，绝不让音频进程阻塞 50 Hz 安全发布线程。"""

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.process: subprocess.Popen[bytes] | None = None
        self.lock = threading.RLock()

    def start(self, path: Path, start_sec: float, duration_sec: float) -> bool:
        with self.lock:
            self.stop("replace")
            if self.mode == "off":
                return False
            ffplay = shutil.which("ffplay")
            if ffplay is None:
                print("[Audio WARNING] ffplay 不可用", flush=True)
                return False
            self.process = subprocess.Popen(
                [
                    ffplay,
                    "-nodisp",
                    "-autoexit",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{start_sec:.9f}",
                    "-t",
                    f"{duration_sec:.9f}",
                    str(path),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True

    def stop(self, reason: str) -> None:
        del reason
        with self.lock:
            process, self.process = self.process, None
        if process is None or process.poll() is not None:
            return
        process.terminate()

        def reap() -> None:
            try:
                process.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

        threading.Thread(target=reap, daemon=True).start()
