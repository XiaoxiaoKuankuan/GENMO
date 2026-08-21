#!/usr/bin/env python3
"""验证同一 BUMI checkpoint 在 PyTorch、ONNX Runtime 与 TensorRT 间的 parity。

脚本固定同一段 120 帧 EDGE35、随机噪声、扩散 timestep、CFG 和 DDIM seed，先比较一个
去噪步，再分别执行完整确定性 DDIM。完整结果不仅比较归一化 93D，还经同一套 BUMI
Endecoder、qpos28 组合和 Torch FK 比较机器人根/关节及 22 个 body 位置。报告同时保存
误差统计和明确的 pass/fail；单步与完整 DDIM 分别使用独立容差，避免把合理的多步浮点
累积误差误判为单步导出错误。FP16 TensorRT 使用可配置的较宽容差，但所有后端都必须
有限且逐项达标。这里的 parity 是数值等价验证，不要求不同算子实现逐 bit 相同。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gem.runtime.bumi_music_deploy import (  # noqa: E402
    BUMI_MOTION_DIM,
    BumiOrtStepRunner,
    BumiTensorRTStepRunner,
)
from gem.runtime.bumi_music_onnx import (  # noqa: E402
    BumiMusicGuidedDenoiser,
    make_bumi_onnx_inputs,
)
from gem.runtime.music_only_trt import SlidingDDIMGenerator, sha256_file  # noqa: E402
from gem.utils.music_features import extract_edge_baseline35  # noqa: E402
from gem.utils.net_utils import load_pretrained_model  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio", type=Path)
    source.add_argument("--edge35", type=Path)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--exp", default="gem_bumi_music_only_4set_random_v1")
    parser.add_argument("--kinematics", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--onnx-provider", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--audio-start-sec", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timestep", type=int, default=999)
    parser.add_argument("--cfg-scale", type=float, default=2.5)
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument("--ort-atol", type=float, default=5.0e-3)
    parser.add_argument("--ort-rtol", type=float, default=5.0e-3)
    parser.add_argument("--trt-atol", type=float, default=3.0e-2)
    parser.add_argument("--trt-rtol", type=float, default=3.0e-2)
    parser.add_argument("--ort-full-atol", type=float, default=2.0e-2)
    parser.add_argument("--ort-full-rtol", type=float, default=2.0e-2)
    parser.add_argument("--trt-full-atol", type=float, default=3.0e-2)
    parser.add_argument("--trt-full-rtol", type=float, default=3.0e-2)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/tensorrt/bumi/parity.json")
    )
    return parser


def _load_edge35(args: argparse.Namespace) -> torch.Tensor:
    if args.audio is not None:
        features, _ = extract_edge_baseline35(
            args.audio.expanduser().resolve(strict=True),
            start_sec=args.audio_start_sec,
            duration_sec=4.0,
            target_fps=30,
        )
    else:
        try:
            value = torch.load(
                args.edge35.expanduser().resolve(strict=True),
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            value = torch.load(args.edge35.expanduser().resolve(strict=True), map_location="cpu")
        if not isinstance(value, torch.Tensor):
            raise ValueError("--edge35 must contain Tensor[T,35]")
        features = value.detach().cpu().float()
    if features.ndim != 2 or features.shape[1] != 35 or len(features) < 120:
        raise ValueError(f"parity EDGE35 must cover at least [120,35], got {features.shape}")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("parity EDGE35 contains NaN or Inf")
    return features[:120].contiguous()


def _comparison(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    left = reference.detach().float().cpu().numpy().astype(np.float64)
    right = candidate.detach().float().cpu().numpy().astype(np.float64)
    if left.shape != right.shape:
        return {"reference_shape": list(left.shape), "shape": list(right.shape), "pass": False}
    delta = np.abs(left - right)
    finite = bool(np.isfinite(right).all())
    close = bool(np.allclose(left, right, atol=atol, rtol=rtol))
    return {
        "shape": list(right.shape),
        "finite": finite,
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
        "atol": float(atol),
        "rtol": float(rtol),
        "pass": bool(finite and close),
    }


@torch.inference_mode()
def _decode(model: Any, motion: torch.Tensor) -> dict[str, torch.Tensor]:
    decoded = model.endecoder.decode(motion)
    default_z = model.endecoder.kinematics.default_qpos[2].to(motion)
    anchor = torch.stack(
        (default_z.new_zeros(()), default_z.new_zeros(()), default_z, default_z.new_zeros(()))
    )
    qpos = model.endecoder.compose_qpos(decoded, world_anchor=anchor)
    body = model.endecoder.kinematics.forward_kinematics(qpos)["body_pos_w"]
    return {"motion_93d": motion, "qpos28": qpos, "body_pos_w": body}


@torch.inference_mode()
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 2 <= args.ddim_steps <= 1000:
        raise ValueError("--ddim-steps must be in 2..1000")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("BUMI TensorRT parity requires a CUDA device")
    checkpoint = args.ckpt.expanduser().resolve(strict=True)
    onnx_path = args.onnx.expanduser().resolve(strict=True)
    engine_path = args.engine.expanduser().resolve(strict=True)
    kinematics = args.kinematics.expanduser().resolve(strict=True)
    stats = args.stats.expanduser().resolve(strict=True)
    os.environ["BUMI_KINEMATICS_PATH"] = str(kinematics)
    os.environ["BUMI_MUSIC_STATS_PATH"] = str(stats)
    music = _load_edge35(args)

    with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "configs")):
        cfg = compose(config_name="train", overrides=[f"exp={args.exp}"])
    model = instantiate(cfg.model, _recursive_=False)
    load_pretrained_model(model, checkpoint)
    model = model.eval().to(device)
    pytorch_step = BumiMusicGuidedDenoiser(model).eval().to(device)
    ort_step = BumiOrtStepRunner(
        onnx_path, device=device, provider=args.onnx_provider
    )
    trt_step = BumiTensorRTStepRunner(engine_path, device=device)
    if trt_step.manifest is None:
        raise RuntimeError("BUMI TensorRT parity requires engine.json")
    if trt_step.manifest.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise ValueError("TensorRT manifest checkpoint SHA does not match --ckpt")
    if trt_step.manifest.get("onnx_sha256") != sha256_file(onnx_path):
        raise ValueError("TensorRT manifest ONNX SHA does not match --onnx")
    inputs = make_bumi_onnx_inputs(
        music,
        seed=args.seed,
        timestep=args.timestep,
        guidance_scale=args.cfg_scale,
        device=device,
    )
    pt_single = pytorch_step(*inputs).float()
    ort_single = ort_step(*inputs).float()
    trt_single = trt_step(*inputs).clone().float()

    def full(step: Any) -> torch.Tensor:
        return SlidingDDIMGenerator(
            step,
            device=device,
            steps=args.ddim_steps,
            guidance_scale=args.cfg_scale,
            motion_dim=BUMI_MOTION_DIM,
        ).generate_window(music, valid_length=120, seed=args.seed)

    pt_motion = full(pytorch_step)
    ort_motion = full(ort_step)
    trt_motion = full(trt_step)
    pt_decoded = _decode(model, pt_motion)
    ort_decoded = _decode(model, ort_motion)
    trt_decoded = _decode(model, trt_motion)

    report: dict[str, Any] = {
        "contract_version": "genmo.bumi_music_tensorrt_parity.v1",
        "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
        "onnx": {"path": str(onnx_path), "sha256": sha256_file(onnx_path)},
        "engine": {"path": str(engine_path), "sha256": sha256_file(engine_path)},
        "seed": args.seed,
        "timestep": args.timestep,
        "cfg_scale": args.cfg_scale,
        "ddim_steps": args.ddim_steps,
        "single_step": {
            "pytorch_vs_onnx": _comparison(
                pt_single, ort_single, atol=args.ort_atol, rtol=args.ort_rtol
            ),
            "pytorch_vs_tensorrt": _comparison(
                pt_single, trt_single, atol=args.trt_atol, rtol=args.trt_rtol
            ),
        },
        "full_ddim": {},
    }
    for name in ("motion_93d", "qpos28", "body_pos_w"):
        report["full_ddim"][name] = {
            "pytorch_vs_onnx": _comparison(
                pt_decoded[name],
                ort_decoded[name],
                atol=args.ort_full_atol,
                rtol=args.ort_full_rtol,
            ),
            "pytorch_vs_tensorrt": _comparison(
                pt_decoded[name],
                trt_decoded[name],
                atol=args.trt_full_atol,
                rtol=args.trt_full_rtol,
            ),
        }
    comparisons = list(report["single_step"].values())
    for value in report["full_ddim"].values():
        comparisons.extend(value.values())
    report["final_pass"] = all(bool(value["pass"]) for value in comparisons)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    torch.save(
        {
            "contract_version": "genmo.bumi_motion_prediction.v1",
            "qpos": trt_decoded["qpos28"].detach().cpu(),
            "normalized_motion_93d": trt_motion.detach().cpu(),
            "fps": 30,
            "joint_names": list(model.endecoder.kinematics.joint_order),
            "quaternion_convention": "wxyz",
            "qpos_order": "mujoco_native",
        },
        output.with_suffix(".motion.pt"),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["final_pass"]:
        raise RuntimeError(f"BUMI TensorRT parity failed; inspect {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
