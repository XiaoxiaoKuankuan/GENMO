# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""独立的定频播放调度器，供 BUMI 离线 GMT 发布入口使用。

从旧 SMPL 播放器原样抽出单调时钟 deadline 算法：发生调度延迟时跳过已经错过的
发送时刻，避免恢复后突发补发。模块不加载人体模型、GMR 或任何通信后端，
只接收调用者给出的时钟值，保持原有帧率校验、返回值和时间推进语义。
"""

from __future__ import annotations

import math


class MonotonicDeadline:
    """Fixed-rate deadline scheduler that skips missed sends instead of bursting."""

    def __init__(self, fps: float, start_time: float) -> None:
        if not math.isfinite(fps) or fps <= 0.0:
            raise ValueError("publish fps must be finite and > 0")
        self.period = 1.0 / float(fps)
        self.next_deadline = float(start_time)

    def seconds_until(self, now: float) -> float:
        return max(self.next_deadline - float(now), 0.0)

    def advance(self, now: float) -> int:
        """Advance one deadline and return how many stale deadlines were skipped."""
        now = float(now)
        candidate = self.next_deadline + self.period
        skipped = 0
        if candidate <= now:
            skipped = int(math.floor((now - candidate) / self.period)) + 1
            candidate += skipped * self.period
        self.next_deadline = candidate
        return skipped
