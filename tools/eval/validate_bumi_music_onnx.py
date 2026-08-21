#!/usr/bin/env python3
"""Validate BUMI music ONNX against its PyTorch checkpoint and decode qpos28.

The mandatory check compares one identical CFG denoising step.  With
``--full-ddim-steps`` the script also runs the repository DDIM scheduler twice,
using PyTorch and ONNX Runtime denoiser callbacks, then compares normalized 93D,
authoritative qpos28 and Torch FK positions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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

from gem.runtime.bumi_music_onnx import (  # noqa: E402
    BUMI_ONNX_CONTRACT_VERSION,
    BumiMusicGuidedDenoiser,
    make_bumi_onnx_inputs,
    tensor_statistics,
)
from gem.utils.music_features import extract_edge_baseline35  # noqa: E402
from gem.utils.net_utils import load_pretrained_model  # noqa: E402


def sha256_file(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def configure_assets(kinematics: Path | None, stats: Path | None) -> tuple[Path, Path]:
    if kinematics is not None:
        os.environ["BUMI_KINEMATICS_PATH"] = str(kinematics.expanduser().resolve())
    if stats is not None:
        os.environ["BUMI_MUSIC_STATS_PATH"] = str(stats.expanduser().resolve())
    paths = []
    for name in ("BUMI_KINEMATICS_PATH", "BUMI_MUSIC_STATS_PATH"):
        value = os.environ.get(name)
        if not value:
            raise RuntimeError(f"{name} is required; export it or pass the matching CLI option")
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")
        paths.append(path)
    return paths[0], paths[1]


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_music(args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, Any]]:
    if args.audio is not None:
        audio = args.audio.expanduser().resolve()
        if not audio.is_file():
            raise FileNotFoundError(audio)
        music, feature_metadata = extract_edge_baseline35(
            audio,
            start_sec=args.audio_start_sec,
            duration_sec=args.audio_duration_sec,
            target_fps=30,
        )
        metadata = {"kind": "audio", "path": str(audio), **feature_metadata}
    else:
        path = args.music_embed.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch_load(path)
        if not isinstance(payload, torch.Tensor):
            raise ValueError(f"--music-embed must contain Tensor[T,35]: {path}")
        music = payload.detach().cpu().float()
        metadata = {"kind": "music_embed", "path": str(path)}
    if music.ndim != 2 or music.shape[1] != 35 or not bool(torch.isfinite(music).all()):
        raise ValueError(f"music must be finite [T,35], got {music.shape}")
    start = int(args.feature_start_frame)
    end = start + int(args.seq_len)
    if start < 0 or end > music.shape[0]:
        raise ValueError(
            f"requested feature range [{start}:{end}] exceeds {music.shape[0]} frames"
        )
    selected = music[start:end].contiguous()
    metadata.update(
        {
            "original_feature_frames": int(music.shape[0]),
            "selected_start_frame": start,
            "selected_frames": int(selected.shape[0]),
        }
    )
    return selected, metadata


def providers(requested: str) -> list[str]:
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    if requested == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(f"CUDAExecutionProvider unavailable: {sorted(available)}")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if requested == "cpu":
        return ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def comparison(
    reference: np.ndarray, candidate: np.ndarray, *, atol: float, rtol: float
) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        return {
            "shape": list(candidate.shape),
            "reference_shape": list(reference.shape),
            "finite": bool(np.isfinite(candidate).all()),
            "allclose": False,
            "reason": "shape_mismatch",
            "atol": float(atol),
            "rtol": float(rtol),
        }
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


class TorchDenoiserCall:
    def __init__(
        self,
        wrapper: BumiMusicGuidedDenoiser,
        music: torch.Tensor,
        length: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        self.wrapper = wrapper
        self.music = music
        self.length = length
        self.scale = scale

    def __call__(self, x: torch.Tensor, timestep: torch.Tensor, **_kwargs: Any):
        motion = self.wrapper(x, timestep, self.music, self.length, self.scale)
        return {
            "pred_x_start": motion,
            "pred_x": motion,
            "pred_cam": None,
            "static_conf_logits": None,
        }


class OrtDenoiserCall:
    def __init__(
        self,
        session: Any,
        music: np.ndarray,
        length: np.ndarray,
        scale: np.ndarray,
    ) -> None:
        self.session = session
        self.music = music
        self.length = length
        self.scale = scale

    def __call__(self, x: torch.Tensor, timestep: torch.Tensor, **_kwargs: Any):
        motion = self.session.run(
            ["pred_motion"],
            {
                "noisy_motion": x.detach().cpu().numpy().astype(np.float32, copy=False),
                "diffusion_timestep": timestep.detach()
                .cpu()
                .numpy()
                .astype(np.int64, copy=False),
                "music": self.music,
                "length": self.length,
                "guidance_scale": self.scale,
            },
        )[0]
        tensor = torch.from_numpy(motion)
        return {
            "pred_x_start": tensor,
            "pred_x": tensor,
            "pred_cam": None,
            "static_conf_logits": None,
        }


def run_ddim(
    diffusion: Any,
    callable_model: Any,
    noise: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    result = diffusion.ddim_sample_loop_with_aux(
        callable_model,
        tuple(noise.shape),
        noise=noise.to(device),
        clip_denoised=False,
        model_kwargs={"y": {}},
        device=device,
        progress=True,
        eta=0.0,
    )
    return result["sample"].detach().cpu()


@torch.inference_mode()
def decode_motion(model: Any, motion: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    normalized = motion.to(device)
    decoded = model.endecoder.decode(normalized)
    qpos = model.endecoder.compose_qpos(decoded)
    fk = model.endecoder.kinematics.forward_kinematics(qpos)
    return {
        "normalized_motion_93d": normalized.detach().cpu(),
        "qpos_canonical": qpos.detach().cpu(),
        "body_position_fk": fk["body_pos_w"].detach().cpu(),
    }


def validate_metadata(
    path: Path,
    *,
    checkpoint: Path,
    kinematics: Path,
    stats: Path,
    seq_len: int,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"ONNX metadata is required: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("contract_version") != BUMI_ONNX_CONTRACT_VERSION:
        raise ValueError(f"unexpected ONNX contract: {metadata.get('contract_version')!r}")
    final_shapes = (metadata.get("checkpoint") or {}).get(
        "final_layer_weight_shapes", {}
    )
    if len(final_shapes) != 1 or next(iter(final_shapes.values()))[0] != 93:
        raise ValueError(f"ONNX metadata does not describe a 93D checkpoint: {final_shapes}")
    expected = {
        "checkpoint": (metadata.get("checkpoint") or {}).get("sha256"),
        "kinematics": (metadata.get("kinematics") or {}).get("sha256"),
        "stats": (metadata.get("stats") or {}).get("sha256"),
    }
    actual = {
        "checkpoint": sha256_file(checkpoint),
        "kinematics": sha256_file(kinematics),
        "stats": sha256_file(stats),
    }
    for name in expected:
        if expected[name] != actual[name]:
            raise ValueError(
                f"ONNX {name} SHA mismatch: metadata={expected[name]}, actual={actual[name]}"
            )
    if int(metadata.get("sequence_length", -1)) != int(seq_len):
        raise ValueError("ONNX metadata sequence length does not match --seq-len")
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio", type=Path)
    source.add_argument("--music-embed", type=Path)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--onnx-metadata", type=Path)
    parser.add_argument("--exp", default="gem_bumi_music_only_4set_random_v1")
    parser.add_argument("--kinematics", type=Path)
    parser.add_argument("--stats", type=Path)
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
    parser.add_argument("--full-ddim-steps", type=int, default=0)
    parser.add_argument("--full-atol", type=float, default=2e-2)
    parser.add_argument("--full-rtol", type=float, default=2e-2)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/onnx/bumi_music/validation")
    )
    return parser


@torch.inference_mode()
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.seq_len <= 120:
        raise ValueError("--seq-len must be in 1..120")
    if args.full_ddim_steps == 1 or not 0 <= args.full_ddim_steps <= 1000:
        raise ValueError("--full-ddim-steps must be 0 or in 2..1000")
    checkpoint = args.ckpt.expanduser().resolve()
    onnx_path = args.onnx.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)
    kinematics, stats = configure_assets(args.kinematics, args.stats)
    metadata_path = (
        args.onnx_metadata.expanduser().resolve()
        if args.onnx_metadata is not None
        else onnx_path.with_suffix(onnx_path.suffix + ".json")
    )
    validate_metadata(
        metadata_path,
        checkpoint=checkpoint,
        kinematics=kinematics,
        stats=stats,
        seq_len=args.seq_len,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    music, source_metadata = load_music(args)

    with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "configs")):
        cfg = compose(config_name="train", overrides=[f"exp={args.exp}"])
    model = instantiate(cfg.model, _recursive_=False)
    load_pretrained_model(model, checkpoint)
    model = model.eval().to(device)
    wrapper = BumiMusicGuidedDenoiser(model).eval().to(device)
    pt_inputs = make_bumi_onnx_inputs(
        music,
        seed=args.seed,
        timestep=args.timestep,
        guidance_scale=args.cfg_scale,
        device=device,
    )
    pt_step = wrapper(*pt_inputs).detach().cpu().numpy()

    import onnxruntime as ort

    selected_providers = providers(args.provider)
    session = ort.InferenceSession(str(onnx_path), providers=selected_providers)
    expected_inputs = {
        "noisy_motion",
        "diffusion_timestep",
        "music",
        "length",
        "guidance_scale",
    }
    if {item.name for item in session.get_inputs()} != expected_inputs:
        raise RuntimeError("ONNX input names do not match the BUMI contract")
    ort_input = {
        "noisy_motion": pt_inputs[0].detach().cpu().numpy(),
        "diffusion_timestep": pt_inputs[1].detach().cpu().numpy(),
        "music": pt_inputs[2].detach().cpu().numpy(),
        "length": pt_inputs[3].detach().cpu().numpy(),
        "guidance_scale": pt_inputs[4].detach().cpu().numpy(),
    }
    ort_step = session.run(["pred_motion"], ort_input)[0]
    step_comparison = comparison(pt_step, ort_step, atol=args.atol, rtol=args.rtol)
    step_pass = bool(step_comparison["finite"] and step_comparison["allclose"])
    report: dict[str, Any] = {
        "contract_version": "genmo.bumi_music_onnx_validation.v1",
        "checkpoint": str(checkpoint),
        "onnx": str(onnx_path),
        "onnx_metadata": str(metadata_path),
        "experiment": args.exp,
        "music_source": source_metadata,
        "seed": args.seed,
        "timestep": args.timestep,
        "cfg_scale": args.cfg_scale,
        "providers_requested": selected_providers,
        "providers_active": session.get_providers(),
        "pytorch_step_statistics": tensor_statistics(pt_step),
        "onnx_step_statistics": tensor_statistics(ort_step),
        "single_step_comparison": step_comparison,
        "single_step_pass": step_pass,
        "full_ddim": None,
    }
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(ort_step), output_dir / "onnx_pred_motion_step_93d.pt")

    full_pass = True
    if args.full_ddim_steps:
        diffusion_cfg = model.pipeline.denoiser3d.model_cfg.diffusion
        with open_dict(diffusion_cfg):
            diffusion_cfg.test_timestep_respacing = str(args.full_ddim_steps)
            diffusion_cfg.gen_only_test_timestep_respacing = str(args.full_ddim_steps)
        model.pipeline.denoiser3d.init_diffusion()
        diffusion = model.pipeline.denoiser3d.test_gen_only_diffusion
        noise = pt_inputs[0].detach().cpu()
        print(f"[BUMI ONNX] PyTorch DDIM steps={args.full_ddim_steps}")
        pt_motion = run_ddim(
            diffusion,
            TorchDenoiserCall(wrapper, pt_inputs[2], pt_inputs[3], pt_inputs[4]),
            noise,
            device,
        )
        print(f"[BUMI ONNX] ONNX Runtime DDIM steps={args.full_ddim_steps}")
        ort_motion = run_ddim(
            diffusion,
            OrtDenoiserCall(
                session,
                ort_input["music"],
                ort_input["length"],
                ort_input["guidance_scale"],
            ),
            noise,
            torch.device("cpu"),
        )
        pt_decoded = decode_motion(model, pt_motion, device)
        ort_decoded = decode_motion(model, ort_motion, device)
        comparisons = {
            "normalized_motion_93d": comparison(
                pt_motion.numpy(),
                ort_motion.numpy(),
                atol=args.full_atol,
                rtol=args.full_rtol,
            ),
            "qpos_canonical": comparison(
                pt_decoded["qpos_canonical"].numpy(),
                ort_decoded["qpos_canonical"].numpy(),
                atol=args.full_atol,
                rtol=args.full_rtol,
            ),
            "body_position_fk": comparison(
                pt_decoded["body_position_fk"].numpy(),
                ort_decoded["body_position_fk"].numpy(),
                atol=args.full_atol,
                rtol=args.full_rtol,
            ),
        }
        full_pass = all(item["finite"] and item["allclose"] for item in comparisons.values())
        report["full_ddim"] = {
            "steps": args.full_ddim_steps,
            "comparisons": comparisons,
            "pass": full_pass,
            "scheduler": "repository Python/PyTorch DDIM for both denoiser backends",
        }
        canonical_qpos = ort_decoded["qpos_canonical"].to(device)
        world_qpos = model.endecoder.codec.apply_world_anchor(
            canonical_qpos,
            torch.tensor([0.0, 0.0, 0.0], device=device),
        ).detach().cpu()
        artifact = {
            "contract_version": "genmo.bumi_motion_prediction.v1",
            "qpos": world_qpos[0],
            "qpos_canonical": ort_decoded["qpos_canonical"][0],
            "normalized_motion_93d": ort_motion[0],
            "fps": 30,
            "robot_name": "bumi",
            "joint_names": list(model.endecoder.kinematics.joint_order),
            "quaternion_convention": "wxyz",
            "qpos_order": "mujoco_native",
            "feature_dim": 93,
            "anchor_mode": model.endecoder.anchor_mode,
            "world_anchor_applied": True,
            "world_anchor": {"root_xy": [0.0, 0.0], "yaw": 0.0},
            "source": source_metadata,
            "checkpoint": str(checkpoint),
            "onnx": str(onnx_path),
            "seed": args.seed,
            "cfg_scale": args.cfg_scale,
            "ddim_steps": args.full_ddim_steps,
        }
        torch.save(artifact, output_dir / "onnx_bumi_motion.pt")

    report["final_pass"] = bool(step_pass and full_pass)
    report_path = output_dir / "validation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["final_pass"]:
        raise RuntimeError(f"BUMI ONNX validation failed; inspect {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
