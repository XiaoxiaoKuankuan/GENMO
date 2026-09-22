#!/usr/bin/env python3
"""BUMI 文本生成单条/常驻命令行入口。

支持带契约checkpoint的PyTorch推理，或deployment清单中的ONNX/TensorRT；本地T5只加载
一次。--console保留文本交互，--preview附加纯运动学窗口；不会连接控制器、ROS或训练服务器。
默认生成120帧、DDIM50、CFG2.5、seed42；产物按独立目录保存，窗口关闭不终止常驻引擎。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    from gem.runtime.bumi_text_runtime import T5_DEFAULT, ResidentBumiTextEngine

    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--deployment-manifest")
    parser.add_argument("--backend", choices=["torch", "onnx", "tensorrt"], default="torch")
    parser.add_argument("--prompt")
    parser.add_argument("--num-frames", type=int, default=120)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t5-model", default=T5_DEFAULT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default="outputs/bumi_text")
    parser.add_argument("--kinematics")
    parser.add_argument("--stats")
    parser.add_argument("--console", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--no-postproc", action="store_true")
    args = parser.parse_args()
    if not args.prompt and not args.console:
        parser.error("指定--prompt或--console")
    engine = ResidentBumiTextEngine(
        args.checkpoint,
        deployment_manifest=args.deployment_manifest,
        backend=args.backend,
        t5_model=args.t5_model,
        device=args.device,
        ddim_steps=args.ddim_steps,
        output_root=args.output_root,
        kinematics=args.kinematics,
        stats=args.stats,
    )
    player = None
    latest = None
    try:
        engine.initialize()
        if args.preview:
            from gem.runtime.bumi_text_viewer import TextPreviewPlayer

            player = TextPreviewPlayer(engine.robot_manifest, engine.asset_paths["kinematics"])

        def generate(prompt):
            nonlocal latest
            result = engine.generate(
                dict(
                    prompt=prompt,
                    num_frames=args.num_frames,
                    seed=args.seed,
                    postproc=not args.no_postproc,
                )
            )
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if not result["ok"]:
                return
            latest = Path(result["output_dir"])
            if player:
                import numpy as np

                with np.load(latest / "motion.npz", allow_pickle=False) as data:
                    player.play(data["qpos"])
            if args.render:
                from gem.runtime.text_motion_web.worker import run_renderer

                run_renderer(latest, args.num_frames)

        if args.prompt:
            generate(args.prompt)
        if args.console:
            print("输入文本或play 文本；frames N、steps N、pause、resume、stand、status、quit。")
            while True:
                try:
                    line = input("BUMI text> ").strip()
                except EOFError:
                    break
                if line in {"quit", "exit"}:
                    break
                try:
                    if line.startswith("frames "):
                        value = int(line.split()[1])
                        from gem.utils.sequence_contract import validate_generation_length

                        validate_generation_length(engine.contract["sequence"], value)
                        args.num_frames = value
                    elif line.startswith("steps "):
                        engine.set_ddim_steps(int(line.split()[1]))
                    elif line in {"pause", "resume", "stand"}:
                        if player:
                            getattr(player, line)()
                    elif line == "status":
                        print(
                            dict(
                                backend=args.backend,
                                frames=args.num_frames,
                                ddim_steps=engine.ddim_steps,
                                preview=None if player is None else player.state,
                            )
                        )
                    elif line:
                        generate(line[5:] if line.startswith("play ") else line)
                except Exception as exc:
                    print(f"错误：{exc}", flush=True)
        elif player:
            input("动画预览中，按回车退出…")
    finally:
        if player:
            player.close()
        engine.close()


if __name__ == "__main__":
    main()
