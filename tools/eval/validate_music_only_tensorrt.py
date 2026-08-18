#!/usr/bin/env python3
"""Validate fixed-shape PyTorch, ONNX and TensorRT streaming denoisers."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gem.datasets.aistpp.aistplusplus import (  # noqa: E402
    load_music_feature_tensor,
    validate_musicfeat_v2,
)
from gem.runtime.music_only_onnx import (  # noqa: E402
    MusicOnlyTensorRTDenoiser,
    make_onnx_inputs,
)
from gem.runtime.music_only_trt import (  # noqa: E402
    SlidingDDIMGenerator,
    TensorRTStepRunner,
    sha256_file,
)
from gem.utils.motion_utils import rollout_local_transl_vel  # noqa: E402
from gem.utils.music_features import extract_edge_baseline35  # noqa: E402
from gem.utils.net_utils import load_pretrained_model  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio", type=Path)
    source.add_argument("--music-embed", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--exp", default="gem_smpl_music_only_4set_physics_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timestep", type=int, default=999)
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument("--onnx-provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--atol", type=float, default=0.03)
    parser.add_argument("--rtol", type=float, default=0.03)
    parser.add_argument("--max-physics-delta", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=Path("outputs/tensorrt/validation.json"))
    return parser


def _music(args: argparse.Namespace) -> torch.Tensor:
    if args.audio is not None:
        value, _ = extract_edge_baseline35(
            args.audio.expanduser().resolve(strict=True),
            start_sec=0.0,
            duration_sec=4.0,
            target_fps=30,
        )
    else:
        value = load_music_feature_tensor(args.music_embed.expanduser().resolve(strict=True))
    validate_musicfeat_v2(value, source=args.audio or args.music_embed)
    if len(value) < 120:
        raise ValueError("TensorRT validation music must contain at least 120 feature frames")
    return value[:120].float().contiguous()


def _comparison(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    left = reference.detach().float().cpu().numpy().astype(np.float64)
    right = candidate.detach().float().cpu().numpy().astype(np.float64)
    delta = np.abs(left - right)
    return {
        "shape": list(right.shape),
        "finite": bool(np.isfinite(right).all()),
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
    }


class OrtStep:
    def __init__(self, path: Path, device: torch.device, provider: str) -> None:
        import onnxruntime as ort

        provider_name = (
            "CUDAExecutionProvider" if provider == "cuda" else "CPUExecutionProvider"
        )
        if provider_name not in ort.get_available_providers():
            raise RuntimeError(
                f"requested {provider_name}, available={ort.get_available_providers()}"
            )
        self.session = ort.InferenceSession(
            str(path.expanduser().resolve(strict=True)), providers=[provider_name]
        )
        self.device = device
        inputs = {value.name for value in self.session.get_inputs()}
        expected = {
            "noisy_motion",
            "diffusion_timestep",
            "music",
            "length",
            "guidance_scale",
        }
        if inputs != expected or [value.name for value in self.session.get_outputs()] != [
            "pred_motion"
        ]:
            raise RuntimeError("ONNX graph does not match the TensorRT deployment contract")

    def __call__(self, noisy, timestep, music, length, guidance):
        values = self.session.run(
            ["pred_motion"],
            {
                "noisy_motion": noisy.detach().cpu().numpy().astype(np.float32),
                "diffusion_timestep": timestep.detach().cpu().numpy().astype(np.int64),
                "music": music.detach().cpu().numpy().astype(np.float32),
                "length": length.detach().cpu().numpy().astype(np.int64),
                "guidance_scale": guidance.detach().cpu().numpy().astype(np.float32),
            },
        )[0]
        return torch.from_numpy(values).to(self.device)


def _derivative_rms(value: torch.Tensor, order: int, fps: float = 30.0) -> float:
    result = value.float()
    for _ in range(order):
        result = torch.diff(result, dim=0) * fps
    return float(torch.sqrt(torch.mean(torch.square(result))).cpu())


@torch.inference_mode()
def _physics_metrics(endecoder: torch.nn.Module, motion: torch.Tensor) -> dict[str, float]:
    decoded = endecoder.decode(motion.float().unsqueeze(0))
    body_pose = decoded["body_pose"]
    orient = decoded["global_orient_gv"]
    transl = rollout_local_transl_vel(decoded["local_transl_vel"], orient)
    frames = motion.shape[0]
    betas = torch.zeros(1, frames, 10, device=motion.device, dtype=torch.float32)
    joints, _, _ = endecoder.fk_v2(
        body_pose=body_pose,
        betas=betas,
        global_orient=orient,
        transl=transl,
        get_intermediate=True,
    )
    root = transl[0]
    fk = joints[0, :, :22]
    return {
        "root_velocity_rms": _derivative_rms(root, 1),
        "root_acceleration_rms": _derivative_rms(root, 2),
        "root_jerk_rms": _derivative_rms(root, 3),
        "fk_velocity_rms": _derivative_rms(fk, 1),
        "fk_acceleration_rms": _derivative_rms(fk, 2),
        "fk_jerk_rms": _derivative_rms(fk, 3),
    }


@torch.inference_mode()
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("TensorRT validation requires CUDA")
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    music = _music(args)
    with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "configs")):
        cfg = compose(config_name="train", overrides=[f"exp={args.exp}"])
    model = instantiate(cfg.model, _recursive_=False)
    load_pretrained_model(model, checkpoint)
    model = model.eval().to(device)
    pytorch_step = MusicOnlyTensorRTDenoiser(model).eval().to(device)
    onnx_step = OrtStep(args.onnx, device, args.onnx_provider)
    trt_step = TensorRTStepRunner(args.engine, device=device)
    if (
        trt_step.manifest is None
        or trt_step.manifest.get("checkpoint_sha256")
        != sha256_file(checkpoint)
    ):
        raise RuntimeError("engine manifest checkpoint SHA256 mismatch")

    inputs = make_onnx_inputs(
        music,
        seed=args.seed,
        timestep=args.timestep,
        guidance_scale=args.guidance_scale,
        device=device,
    )
    pt_step = pytorch_step(*inputs)
    ort_step = onnx_step(*inputs)
    trt_output = trt_step(*inputs).clone()
    single_ort = _comparison(pt_step, ort_step)
    single_trt = _comparison(pt_step, trt_output)
    single_ort["allclose"] = bool(
        torch.allclose(pt_step, ort_step, atol=args.atol, rtol=args.rtol)
    )
    single_trt["allclose"] = bool(
        torch.allclose(pt_step, trt_output, atol=args.atol, rtol=args.rtol)
    )

    full_outputs: dict[str, torch.Tensor] = {}
    full_times: dict[str, float] = {}
    for name, step in (("pytorch", pytorch_step), ("onnx", onnx_step), ("tensorrt", trt_step)):
        generator = SlidingDDIMGenerator(
            step,
            device=device,
            steps=args.ddim_steps,
            guidance_scale=args.guidance_scale,
        )
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        full_outputs[name] = generator.generate_window(
            music, valid_length=120, seed=args.seed
        ).clone()
        torch.cuda.synchronize(device)
        full_times[name] = time.perf_counter() - started

    physics = {
        name: _physics_metrics(model.endecoder, output)
        for name, output in full_outputs.items()
    }
    physics_delta = {
        key: abs(physics["tensorrt"][key] - physics["pytorch"][key])
        / max(abs(physics["pytorch"][key]), 1e-8)
        for key in physics["pytorch"]
    }
    finite = all(torch.isfinite(value).all() for value in full_outputs.values())
    physics_pass = max(physics_delta.values()) <= args.max_physics_delta
    final_pass = bool(
        single_ort["allclose"] and single_trt["allclose"] and finite and physics_pass
    )
    report = {
        "contract_version": "gem_music_only_trt_validation_v1",
        "checkpoint": str(checkpoint),
        "onnx": str(args.onnx.expanduser().resolve()),
        "engine": str(args.engine.expanduser().resolve()),
        "single_step": {"pytorch_vs_onnx": single_ort, "pytorch_vs_tensorrt": single_trt},
        "ddim_steps": args.ddim_steps,
        "full_motion": {
            "pytorch_vs_onnx": _comparison(full_outputs["pytorch"], full_outputs["onnx"]),
            "pytorch_vs_tensorrt": _comparison(
                full_outputs["pytorch"], full_outputs["tensorrt"]
            ),
            "finite": finite,
            "seconds": full_times,
        },
        "physics_metrics": physics,
        "tensorrt_relative_physics_delta": physics_delta,
        "max_allowed_physics_delta": args.max_physics_delta,
        "physics_pass": physics_pass,
        "final_pass": final_pass,
        "recommended_precision": (
            "current" if final_pass else "fp32 TensorRT (rebuild and validate again)"
        ),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not final_pass:
        raise RuntimeError(f"TensorRT validation failed; inspect {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
