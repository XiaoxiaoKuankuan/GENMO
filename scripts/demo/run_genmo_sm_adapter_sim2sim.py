#!/usr/bin/env python3
"""输入 WAV/MP3，在 MuJoCo 查看冻结 GENMO 参考与 SM＋Adapter 真实执行。

默认读取本机明确绑定的 Stage1/SM/Adapter 资产；首次音乐先生成固定 Stage2-A
名义参考，随后启动原生 MuJoCo 交互窗口，蓝色半透明机器人显示原始参考。
--zero-adapter 或窗口 B 键用于相同参考、相同初态的零 Adapter 对照。
--headless --mute --max-seconds N 提供有界真实动力学验证。所有输出写新目录，
不会启动训练、调用 WM 代替物理、覆盖 checkpoint 或中断已有 GPU 任务。
"""

import argparse
import os
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio", type=Path, required=True, help="完整音乐文件路径，WAV/MP3/FLAC等")
    p.add_argument("--config", type=Path, default=ROOT/"configs/closedloop/adapter_sim2sim_mujoco.yaml")
    p.add_argument("--adapter-checkpoint", type=Path)
    p.add_argument("--zero-adapter", action="store_true")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--mute", action="store_true")
    p.add_argument("--fast", action="store_true", help="仅无声无窗口验证：不等待实时播放")
    p.add_argument("--max-seconds", type=float, help="仿真时间上限；不截断用于生成的原音频")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, help="必须是尚不存在的独立目录")
    args = p.parse_args()
    if not args.audio.expanduser().is_file(): p.error("音乐文件不存在")
    args.audio = args.audio.expanduser().resolve()
    if args.max_seconds is not None and args.max_seconds < .02: p.error("max-seconds 至少0.02")
    if args.fast and not (args.headless and args.mute): p.error("--fast 只能与 --headless --mute 一起使用")
    library = Path(sys.prefix)/"lib"
    if not args.mute and (library/"libportaudio.so.2").exists():
        paths = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        if str(library) not in paths:
            env = dict(os.environ, LD_LIBRARY_PATH=os.pathsep.join([str(library), *filter(None, paths)]))
            os.execve(sys.executable, [sys.executable, "-B", str(Path(__file__).resolve()), *sys.argv[1:]], env)
    if args.headless: os.environ.setdefault("MUJOCO_GL", "egl")
    from omegaconf import OmegaConf
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    sys.path.insert(0, str(Path(config["supermimic"]["repo"]).expanduser()))
    from gem.closedloop.reference_adapter.mujoco_demo import run
    output = args.output or Path(config["output_root"])/(time.strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:8])
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    run(args, config, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
