#!/usr/bin/env python3
"""只读比较已经生成的音乐/文本 BUMI 轨迹并准备不含模型名的人工盲评包。

JSON manifest 的 tasks 各项需要 task_id、split(val/test)、condition_kind、整数seed、
duration_s、music_features或text，以及models字典（模型标签→reference.pt/motion.npz）。
文件相对路径按manifest所在目录解析；所有模型按同一任务时长截取，缺失文件显式
保留失败记录。输出必须为新目录，包含分模型/分模态运动学报告、盲评JSON/CSV、
匿名轨迹和独立揭盲key。已有生成合同会核对种子/时长/条件，没有元数据的旧导出
明确标为仅manifest声明。此工具不生成动作、不训练、不跑物理，也不产生文本语义分。
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--kinematics",
        type=Path,
        default=ROOT / "configs/bumi/bumi_kinematics_robot_retargeter_fe934_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blind-seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("threads 必须为正")
    import torch

    from gem.robots.bumi.multimodal_evaluation import evaluate_manifest

    torch.set_num_threads(args.threads)
    result = evaluate_manifest(
        args.manifest, args.kinematics, args.output, blind_seed=args.blind_seed
    )
    summary = dict(
        output=str(args.output.resolve()),
        tasks=len(result["pairing"]),
        complete_pairs=sum(x["complete"] for x in result["pairing"]),
        invalid_outputs=sum(x["status"] != "ok" for x in result["results"]),
        evidence=result["evidence"],
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    # 报告即使存在缺失也完整落盘，非零退出让自动化不能把不完整配对当验收通过。
    return 0 if summary["invalid_outputs"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
