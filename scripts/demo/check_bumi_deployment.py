#!/usr/bin/env python3
"""检查独立 BUMI 部署目录的资产、环境与可选单步推理。

默认只核验部署清单、完整文件哈希、BUMI 运动学/统计，并输出实际
Python 环境及版本；不连接 Bridge/Redis，不启动 ROS，也不发送机器人动作。传入
--inference 时才使用指定 ONNX/TensorRT 后端执行一次固定形状的零输入去噪，验证输出
形状和有限性。这只是安装检查，真实模型的完整 DDIM 数值验收在原 GENMO 仓库进行。
所有模型路径从清单位置解析，允许在不同电脑、不同工作目录直接复用整个部署目录。
--check-gmt 额外只读GMT当前ROS参数及policy元数据；未传时不要求GMT或ROS在线。
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gem.runtime.bumi_deployment_bundle import load_bumi_deployment_manifest  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-manifest", type=Path, required=True)
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--backend", choices=("tensorrt", "onnx"), default="tensorrt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--onnx-provider", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--check-gmt", action="store_true")
    parser.add_argument("--ros-master-uri")
    parser.add_argument("--gmt-container", default="noetic")
    parser.add_argument(
        "--robot-manifest", type=Path, help="可选：检查 MuJoCo 资源及原生关节契约，不打开窗口"
    )
    args = parser.parse_args(argv)
    import torch

    from gem.robots.bumi.endecoder import BumiEndecoder
    from gem.runtime.bumi_music_deploy import BumiOrtStepRunner, BumiTensorRTStepRunner
    from gem.runtime.gmt_trajectory import GmtPolicyContract

    bundle = load_bumi_deployment_manifest(args.deployment_manifest)
    endecoder = BumiEndecoder(
        bundle.paths["kinematics"], bundle.paths["stats"], enable_contact_targets=False
    )
    versions = {}
    for name in (
        "torch",
        "numpy",
        "scipy",
        "librosa",
        "soundfile",
        "onnxruntime-gpu",
        "onnxruntime",
        "redis",
        "pyzmq",
        "mujoco",
    ):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    result = {
        "pass": True,
        "manifest": str(bundle.manifest_path),
        "source_checkpoint_sha256": bundle.source_checkpoint_sha256,
        "checkpoint_file_required": False,
        "python": sys.executable,
        "versions": versions,
        "bumi_joint_count": len(endecoder.kinematics.joint_order),
        "gmt_checked": args.check_gmt,
        "inference_requested": args.inference,
    }
    if args.check_gmt:
        from gem.runtime.gmt_policy_source import resolve_bridge_policy

        path, source = resolve_bridge_policy(args)
        policy = GmtPolicyContract.from_onnx(path)
        policy.native_to_gmt_indices(endecoder.kinematics.joint_order)
        result.update(gmt_policy_path=str(path), gmt_policy_source=source)
    if args.robot_manifest is not None:
        from scripts.demo.bumi_mujoco_viewer import load_model

        model, _ = load_model(args.robot_manifest, bundle.paths["kinematics"])
        result["preview_model"] = {
            "nq": model.nq,
            "nv": model.nv,
            "manifest": str(args.robot_manifest),
        }
    if args.inference:
        device = torch.device(args.device)
        if args.backend == "tensorrt":
            runner = BumiTensorRTStepRunner(bundle.paths["engine"], device=device)
            result["tensorrt"] = runner.trt.__version__
            result["libnvinfer"] = runner.linked_tensorrt_version
        else:
            runner = BumiOrtStepRunner(
                bundle.paths["onnx"], device=device, provider=args.onnx_provider
            )
        with torch.inference_mode():
            outputs = runner(
                torch.zeros(1, 120, 30, device=device),
                torch.tensor([999], device=device),
                torch.zeros(1, 120, 35, device=device),
                torch.tensor([120], device=device),
                torch.tensor([2.5], device=device),
            )
        if not all(bool(torch.isfinite(value).all()) for value in outputs):
            raise RuntimeError("single-step outputs contain NaN/Inf")
        result["output_shapes"] = [list(value.shape) for value in outputs]
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
