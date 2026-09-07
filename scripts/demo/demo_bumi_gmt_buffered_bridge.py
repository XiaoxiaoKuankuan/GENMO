#!/usr/bin/env python3
"""BUMI 整段缓存仿真播放桥的独立入口。

默认监听 7023，使用 gmt_buffered_frame_bumi Redis 键，避免覆盖原实时桥。完整轨迹
上传一次后，只发送心跳并读取 GMT 回报的帧号；动作、缓入和返回都由 GMT 策略步推进。
必须配合支持 gmt_mode:=buffered 的 GMT 接收端重新编译使用。默认不播放真实时间
音频，避免慢仿真与正常速度音乐错位；此模式不提高 Gazebo 实时率，也不用于实机。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.demo.demo_bumi_gmt_bridge import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                "--bind",
                "tcp://127.0.0.1:7023",
                "--redis-key",
                "gmt_buffered_frame_bumi",
                "--audio-playback",
                "off",
                "--ack-timeout-seconds",
                "300",
                *sys.argv[1:],
                "--playback-mode",
                "buffered",
            ]
        )
    )
