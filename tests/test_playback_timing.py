"""验证 BUMI 播放共用的定频 deadline 算法。

从旧 SMPL 播放器测试中保留错过发送时刻后的跳帧回归，确保模块迁移不改变调度
语义。测试只使用显式数值时钟，不睡眠、不启动网络服务、不发送机器人命令。
"""

import pytest

from gem.runtime.playback_timing import MonotonicDeadline


def test_deadline_skips_stale_sends_without_burst() -> None:
    deadline = MonotonicDeadline(10.0, 0.0)
    assert deadline.advance(0.0) == 0
    skipped = deadline.advance(0.55)
    assert skipped == 4
    assert deadline.next_deadline == pytest.approx(0.6)
    assert deadline.seconds_until(0.55) == pytest.approx(0.05)
