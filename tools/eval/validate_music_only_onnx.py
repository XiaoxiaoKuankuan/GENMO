#!/usr/bin/env python3
"""Validate a music-only GEM ONNX denoiser against its PyTorch checkpoint.

The default check uses real EDGE35 music and identical noisy motion/timestep inputs
for one neural denoising step.  ``--full-ddim-steps`` additionally drives the same
repository DDIM scheduler with ONNX Runtime for an end-to-end 151-D diffusion sample.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import open_dict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gem.datasets.aistpp.aistplusplus import (  # noqa: E402
    load_music_feature_tensor,
    validate_musicfeat_v2,
)
from gem.runtime.music_only_onnx import (  # noqa: E402
    MusicOnlyGuidedDenoiser,
    make_onnx_inputs,
    output_statistics,
)
from gem.utils.music_features import extract_edge_baseline35  # noqa: E402
from gem.utils.net_utils import load_pretrained_model  # noqa: E402

DEFAULT_CHECKPOINT = Path("inputs/checkpoints/music_only_aistpp/version_1/last.ckpt")
DEFAULT_ONNX = Path("outputs/onnx/music_only_aistpp_s260000/music_only_denoiser.onnx")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio", type=Path)
    source.add_argument("--music-embed", type=Path)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--exp", default="gem_smpl_music_only")
    parser.add_argument("--seq-len", type=int, default=120)
    parser.add_argument("--audio-start-sec", type=float, default=0.0)
    parser.add_argument("--audio-duration-sec", type=float, default=4.0)
    parser.add_argument("--feature-start-frame", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timestep", type=int, default=999)
    parser.add_argument("--cfg-scale", type=float, default=2.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--provider", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--atol", type=float, default=5e-3)
    parser.add_argument("--rtol", type=float, default=5e-3)
    parser.add_argument(
        "--full-ddim-steps",
        type=int,
        default=0,
        help="0 checks one network step; a positive value also runs a full DDIM loop.",
    )
    parser.add_argument("--full-atol", type=float, default=2e-2)
    parser.add_argument("--full-rtol", type=float, default=2e-2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/onnx/music_only_aistpp_s260000/validation"),
    )
    return parser


def _load_music(args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, Any]]:
    if args.audio is not None:
        if not args.audio.is_file():
            raise FileNotFoundError(args.audio)
        music, feature_metadata = extract_edge_baseline35(
            args.audio,
            start_sec=args.audio_start_sec,
            duration_sec=args.audio_duration_sec,
            target_fps=30,
        )
        source = str(args.audio.resolve())
        metadata = {"kind": "audio", "path": source, **feature_metadata}
    else:
        if not args.music_embed.is_file():
            raise FileNotFoundError(args.music_embed)
        music = load_music_feature_tensor(args.music_embed)
        validate_musicfeat_v2(music, source=args.music_embed)
        source = str(args.music_embed.resolve())
        metadata = {"kind": "music_embed", "path": source}

    start = int(args.feature_start_frame)
    end = start + int(args.seq_len)
    if start < 0:
        raise ValueError("--feature-start-frame must be >= 0")
    if music.shape[0] < end:
        raise ValueError(
            f"music source has {music.shape[0]} frames, but [{start}:{end}] was requested. "
            "For WAV input, select an audio duration that yields at least --seq-len frames."
        )
    selected = music[start:end].contiguous().float()
    validate_musicfeat_v2(selected, source=source)
    metadata.update(
        {
            "original_feature_frames": int(music.shape[0]),
            "selected_start_frame": start,
            "selected_frames": int(selected.shape[0]),
            "selected_shape": list(selected.shape),
        }
    )
    return selected, metadata


def _providers(requested: str) -> list[str]:
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    if requested == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                f"CUDAExecutionProvider is unavailable; providers={sorted(available)}"
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if requested == "cpu":
        return ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _comparison(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    delta = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    return {
        "shape": list(candidate.shape),
        "finite": bool(np.isfinite(candidate).all()),
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "allclose": bool(np.allclose(reference, candidate, atol=atol, rtol=rtol)),
        "atol": float(atol),
        "rtol": float(rtol),
    }


class _TorchDenoiserCall:
    def __init__(
        self,
        wrapper: MusicOnlyGuidedDenoiser,
        music: torch.Tensor,
        length: torch.Tensor,
        scale: torch.Tensor,
    ):
        self.wrapper = wrapper
        self.music = music
        self.length = length
        self.scale = scale

    def __call__(self, x: torch.Tensor, timestep: torch.Tensor, **_kwargs):
        pred_motion, pred_camera, static_logits = self.wrapper(
            x, timestep, self.music, self.length, self.scale
        )
        return {
            "pred_x_start": pred_motion,
            "pred_cam": pred_camera,
            "static_conf_logits": static_logits,
        }


class _OrtDenoiserCall:
    def __init__(self, session: Any, music: np.ndarray, length: np.ndarray, scale: np.ndarray):
        self.session = session
        self.music = music
        self.length = length
        self.scale = scale

    def __call__(self, x: torch.Tensor, timestep: torch.Tensor, **_kwargs):
        values = self.session.run(
            None,
            {
                "noisy_motion": x.detach().cpu().numpy().astype(np.float32, copy=False),
                "diffusion_timestep": timestep.detach().cpu().numpy().astype(np.int64, copy=False),
                "music": self.music,
                "length": self.length,
                "guidance_scale": self.scale,
            },
        )
        return {
            "pred_x_start": torch.from_numpy(values[0]),
            "pred_cam": torch.from_numpy(values[1]),
            "static_conf_logits": torch.from_numpy(values[2]),
        }


def _run_full_ddim(
    *,
    diffusion: Any,
    callable_model: Any,
    noise: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return diffusion.ddim_sample_loop_with_aux(
        callable_model,
        tuple(noise.shape),
        noise=noise.to(device),
        clip_denoised=False,
        # GEM's DDIM implementation stores the previous x_start under y even
        # when this specialist callback does not consume it.
        model_kwargs={"y": {}},
        device=device,
        progress=True,
        eta=0.0,
    )


@torch.inference_mode()
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.ckpt.is_file():
        raise FileNotFoundError(args.ckpt)
    if not args.onnx.is_file():
        raise FileNotFoundError(args.onnx)
    if not 1 <= args.seq_len <= 120:
        raise ValueError("--seq-len must be in 1..120")
    if args.full_ddim_steps < 0:
        raise ValueError("--full-ddim-steps must be >= 0")
    if args.full_ddim_steps == 1 or args.full_ddim_steps > 1000:
        raise ValueError("--full-ddim-steps must be 0 or in 2..1000")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    music, source_metadata = _load_music(args)
    with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "configs")):
        cfg = compose(config_name="train", overrides=[f"exp={args.exp}"])
    print(f"[Validate] 加载 PyTorch specialist: {args.ckpt}")
    model = instantiate(cfg.model, _recursive_=False)
    load_pretrained_model(model, args.ckpt)
    model = model.eval().to(device)
    wrapper = MusicOnlyGuidedDenoiser(model).eval().to(device)
    pt_inputs = make_onnx_inputs(
        music,
        seed=args.seed,
        timestep=args.timestep,
        guidance_scale=args.cfg_scale,
        device=device,
    )
    pt_outputs = wrapper(*pt_inputs)

    import onnxruntime as ort

    selected_providers = _providers(args.provider)
    print(f"[Validate] 加载 ONNX Runtime: {args.onnx}; providers={selected_providers}")
    session = ort.InferenceSession(str(args.onnx), providers=selected_providers)
    expected_inputs = {
        "noisy_motion",
        "diffusion_timestep",
        "music",
        "length",
        "guidance_scale",
    }
    actual_inputs = {item.name for item in session.get_inputs()}
    if actual_inputs != expected_inputs:
        raise RuntimeError(f"unexpected ONNX inputs: {sorted(actual_inputs)}")
    ort_input = {
        "noisy_motion": pt_inputs[0].detach().cpu().numpy(),
        "diffusion_timestep": pt_inputs[1].detach().cpu().numpy(),
        "music": pt_inputs[2].detach().cpu().numpy(),
        "length": pt_inputs[3].detach().cpu().numpy(),
        "guidance_scale": pt_inputs[4].detach().cpu().numpy(),
    }
    ort_outputs = session.run(None, ort_input)

    output_names = ("pred_motion", "pred_camera", "static_conf_logits")
    comparisons = {
        name: _comparison(pt.detach().cpu().numpy(), ort_value, atol=args.atol, rtol=args.rtol)
        for name, pt, ort_value in zip(output_names, pt_outputs, ort_outputs)
    }
    single_step_pass = all(value["finite"] and value["allclose"] for value in comparisons.values())
    report: dict[str, Any] = {
        "checkpoint": str(args.ckpt.resolve()),
        "onnx": str(args.onnx.resolve()),
        "experiment": args.exp,
        "condition_list": list(cfg.pipeline.args.in_attr),
        "music_source": source_metadata,
        "seed": args.seed,
        "timestep": args.timestep,
        "cfg_scale": args.cfg_scale,
        "providers_requested": selected_providers,
        "providers_active": session.get_providers(),
        "pytorch_statistics": output_statistics(pt_outputs),
        "onnx_statistics": output_statistics(tuple(ort_outputs)),
        "single_step_comparison": comparisons,
        "single_step_pass": single_step_pass,
        "full_ddim": None,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(ort_outputs[0]), args.output_dir / "onnx_pred_motion_step.pt")

    full_pass = True
    if args.full_ddim_steps:
        # Rebuild the repository's existing sampler; the ONNX graph only replaces its
        # neural-network callback.  No new diffusion equations are introduced here.
        diffusion_cfg = model.pipeline.denoiser3d.model_cfg.diffusion
        diffusion_cfg.test_timestep_respacing = str(args.full_ddim_steps)
        with open_dict(diffusion_cfg):
            diffusion_cfg.gen_only_test_timestep_respacing = str(args.full_ddim_steps)
        model.pipeline.denoiser3d.init_diffusion()
        diffusion = model.pipeline.denoiser3d.test_gen_only_diffusion

        noise_cpu = pt_inputs[0].detach().cpu()
        torch_call = _TorchDenoiserCall(
            wrapper,
            pt_inputs[2],
            pt_inputs[3],
            pt_inputs[4],
        )
        print(f"[Validate] 运行 PyTorch {args.full_ddim_steps}-step DDIM")
        pt_final = _run_full_ddim(
            diffusion=diffusion,
            callable_model=torch_call,
            noise=noise_cpu,
            device=device,
        )

        ort_call = _OrtDenoiserCall(
            session,
            ort_input["music"],
            ort_input["length"],
            ort_input["guidance_scale"],
        )
        print(f"[Validate] 运行 ONNX Runtime {args.full_ddim_steps}-step DDIM")
        ort_final = _run_full_ddim(
            diffusion=diffusion,
            callable_model=ort_call,
            noise=noise_cpu,
            device=torch.device("cpu"),
        )
        pt_motion = pt_final["sample"].detach().cpu().numpy()
        ort_motion = ort_final["sample"].detach().cpu().numpy()
        full_comparison = _comparison(
            pt_motion,
            ort_motion,
            atol=args.full_atol,
            rtol=args.full_rtol,
        )
        full_pass = bool(full_comparison["finite"] and full_comparison["allclose"])
        report["full_ddim"] = {
            "steps": args.full_ddim_steps,
            "comparison": full_comparison,
            "pass": full_pass,
            "note": (
                "The existing Python/PyTorch DDIM scheduler was used for both runs; "
                "only the neural denoiser calls were executed by ONNX Runtime."
            ),
        }
        torch.save(torch.from_numpy(ort_motion), args.output_dir / "onnx_generated_motion_151d.pt")

    report["final_pass"] = bool(single_step_pass and full_pass)
    report_path = args.output_dir / "validation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[Validate] 报告: {report_path}")
    if not report["final_pass"]:
        raise RuntimeError("ONNX validation failed; inspect validation_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
