"""Non-blocking client for the GMR BUMI3 MuJoCo visualization sidecar."""

from __future__ import annotations

import os
import select
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from gem.runtime.gmt_trajectory import BUMI_QPOS_DIM

VIEWER_READY_BYTE = b"V"
VIEWER_FRAME_BYTES = BUMI_QPOS_DIM * np.dtype("<f4").itemsize


def encode_qpos_viewer_frame(qpos: np.ndarray) -> bytes:
    """Encode one native BUMI qpos record for the C++ MuJoCo viewer."""

    values = np.asarray(qpos, dtype=np.float32)
    if values.shape != (BUMI_QPOS_DIM,):
        raise ValueError(f"viewer qpos must have shape ({BUMI_QPOS_DIM},)")
    if not np.isfinite(values).all():
        raise ValueError("viewer qpos must be finite")
    payload = np.asarray(values, dtype="<f4").tobytes(order="C")
    if len(payload) != VIEWER_FRAME_BYTES:
        raise RuntimeError("viewer qpos encoding has an invalid size")
    return payload


class GMRMujocoViewerClient:
    """Publish current 50 Hz GMR references without blocking the safety loop."""

    def __init__(
        self,
        command: Sequence[str | Path],
        *,
        cwd: str | Path | None = None,
        ready_timeout_seconds: float = 3.0,
        popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    ) -> None:
        if not command:
            raise ValueError("MuJoCo viewer command must not be empty")
        if ready_timeout_seconds <= 0.0:
            raise ValueError("viewer ready timeout must be > 0")
        self.command = [str(value) for value in command]
        self.process = popen(
            self.command,
            cwd=None if cwd is None else str(cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            bufsize=0,
        )
        if self.process.stdin is None or self.process.stdout is None:
            self._terminate()
            raise RuntimeError("failed to open MuJoCo viewer pipes")
        readable, _, _ = select.select(
            [self.process.stdout], [], [], float(ready_timeout_seconds)
        )
        ready = b"" if not readable else self.process.stdout.read(1)
        if ready != VIEWER_READY_BYTE:
            returncode = self.process.poll()
            self._terminate()
            detail = "readiness timeout" if returncode is None else f"exit={returncode}"
            raise RuntimeError(f"MuJoCo viewer failed to start ({detail})")
        self._fd = self.process.stdin.fileno()
        os.set_blocking(self._fd, False)
        self.dropped_frames = 0
        self.published_frames = 0
        self.last_error: str | None = None

    @property
    def alive(self) -> bool:
        return self.process.poll() is None

    def publish(self, qpos: np.ndarray) -> bool:
        payload = encode_qpos_viewer_frame(qpos)
        if not self.alive:
            self.last_error = f"viewer exited with code {self.process.returncode}"
            return False
        try:
            count = os.write(self._fd, payload)
        except BlockingIOError:
            self.dropped_frames += 1
            return False
        except (BrokenPipeError, OSError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False
        if count != len(payload):
            self.last_error = f"partial viewer write: {count}/{len(payload)} bytes"
            return False
        self.published_frames += 1
        return True

    def _terminate(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=1.0)

    def close(self) -> None:
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        if self.process.poll() is None:
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._terminate()

    def __enter__(self) -> GMRMujocoViewerClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
