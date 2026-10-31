#!/usr/bin/env python3
"""从完整音乐或英文文本生成共享 Stage1 的 BUMI 轨迹。

入口将音频解码为原 EDGE35，或者将英文描述交给本地冻结 T5；每次只能选择一种
输入，文本必须声明持续秒数。使用共享模型、唯一动作统计量和相同滚动历史构造，
保存 qpos28/PT/NPZ、接触有效性及生成来源。附加尾部由模型真实生成，单独标出可
执行时长；不把这些运动学输出当作物理验收，不启动训练，也不覆盖已有输出目录。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio", type=Path)
    source.add_argument("--text")
    parser.add_argument("--duration-s", type=float, help="文本必须指定；音乐默认完整音频")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--kinematics", type=Path, default=ROOT / "configs/bumi/bumi_kinematics_robot_retargeter_fe934_v1.json")
    parser.add_argument("--text-encoder-path", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.text is not None and (not args.text.strip() or args.duration_s is None):
        parser.error("非空 --text 必须搭配 --duration-s")
    if args.duration_s is not None and (not math.isfinite(args.duration_s) or args.duration_s < .02):
        parser.error("--duration-s 必须至少0.02秒且有限")
    if args.audio and not args.audio.is_file():
        parser.error("--audio 文件不存在")
    import numpy as np
    import torch
    from gem.runtime.multimodal_stage1 import generate_stage1_reference
    torch.set_num_threads(args.threads)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    task = dict(id="generated", seed=args.seed, duration_s=args.duration_s)
    if args.audio:
        from gem.utils.music_features import extract_edge_baseline35
        music, metadata = extract_edge_baseline35(args.audio, target_fps=30)
        task.update(condition_kind="music", music_features=str(output / "music_features.pt"))
        task["duration_s"] = args.duration_s or metadata["selected_duration_sec"]
        torch.save(music, task["music_features"])
    else:
        task.update(condition_kind="text", text=args.text.strip())
    config = {name: str(getattr(args, name).expanduser().resolve())
              for name in ("checkpoint", "stats", "kinematics")}
    if args.text_encoder_path:
        config["text_encoder_path"] = str(args.text_encoder_path.expanduser().resolve())
    config.update(sample_steps=args.sample_steps, guidance_scale=args.guidance_scale)
    end = round(task["duration_s"] * 600)
    result = generate_stage1_reference(config, task, device=args.device,
                                       required_end_tick=(end // 12) * 12 + 132)
    torch.save(result, output / "reference.pt")
    np.savez_compressed(output / "motion.npz", qpos=result["world_qpos"][0].numpy(),
                        contact=result["contact"][0].numpy(), contact_valid=result["contact_valid"].numpy(),
                        joint_names=np.asarray(result["joint_names"]), fps=30,
                        execution_duration_s=task["duration_s"])
    report = dict(task=task, generation=result["generation_contract"],
                  generated_frames=result["world_qpos"].shape[1], evidence="kinematic_generation_only")
    (output / "generation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(output=str(output), **report), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
