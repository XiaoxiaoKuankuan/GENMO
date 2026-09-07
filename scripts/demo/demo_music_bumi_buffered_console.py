#!/usr/bin/env python3
"""BUMI 先整首生成、再交给 GMT 仿真步播放的独立控制台。

模型、CUDA ONNX、DDIM、特征缓存和命令语法复用原实时控制台；区别是完整 qpos 生成
并收齐后才提交唯一一个最终块，不受在线生成高低水位或三秒续窗门槛限制。默认连接
7023 端口的 buffered 桥，原 7022 实时模式不受影响。可在 bumi> 输入音乐绝对路径。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.demo.demo_music_bumi_console import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(
        main(["--bridge", "tcp://127.0.0.1:7023", *sys.argv[1:], "--playback-mode", "buffered"])
    )
