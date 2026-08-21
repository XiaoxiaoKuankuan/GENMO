#!/usr/bin/env python3
"""Generate, audit, evaluate and optionally render BUMI qpos28 from music."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
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

from gem.robots.bumi.metrics import (  # noqa: E402
    compute_bumi_kinematic_metrics,
    metrics_to_json,
)
from gem.runtime.bumi_music_onnx import validate_bumi_checkpoint_state_dict  # noqa: E402
from gem.utils.music_features import EDGE_FEATURE_DIM, extract_edge_baseline35  # noqa: E402
from gem.utils.net_utils import load_pretrained_model  # noqa: E402


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def sha256_file(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_assets(kinematics: Path | None, stats: Path | None) -> tuple[Path, Path]:
    if kinematics is not None:
        os.environ["BUMI_KINEMATICS_PATH"] = str(kinematics.expanduser().resolve())
    if stats is not None:
        os.environ["BUMI_MUSIC_STATS_PATH"] = str(stats.expanduser().resolve())
    result = []
    for name in ("BUMI_KINEMATICS_PATH", "BUMI_MUSIC_STATS_PATH"):
        raw = os.environ.get(name)
        if not raw:
            raise RuntimeError(f"{name} is required; export it or pass the matching CLI option")
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")
        result.append(path)
    return result[0], result[1]


def load_edge35(path: Path) -> torch.Tensor:
    payload = torch_load(path)
    if not isinstance(payload, torch.Tensor):
        raise ValueError(f"precomputed EDGE35 file must contain Tensor[T,35]: {path}")
    features = payload.detach().cpu().float()
    if features.ndim != 2 or features.shape[1] != EDGE_FEATURE_DIM or features.shape[0] <= 0:
        raise ValueError(f"EDGE35 must have shape [T,35], got {features.shape}: {path}")
    if not bool(torch.isfinite(features).all()):
        raise ValueError(f"EDGE35 contains NaN or Inf: {path}")
    return features


def align_music_frames(
    features: torch.Tensor, num_frames: int | None
) -> tuple[torch.Tensor, torch.Tensor]:
    source_frames = int(features.shape[0])
    target_frames = source_frames if num_frames is None else int(num_frames)
    if target_frames <= 0:
        raise ValueError("--num-frames must be positive")
    if target_frames <= source_frames:
        return features[:target_frames].contiguous(), torch.ones(
            target_frames, dtype=torch.bool
        )
    padding = torch.zeros(target_frames - source_frames, 35, dtype=features.dtype)
    mask = torch.zeros(target_frames, dtype=torch.bool)
    mask[:source_frames] = True
    return torch.cat((features, padding), dim=0), mask


def cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(child) for key, child in value.items()}
    if isinstance(value, list):
        return [cpu_tree(child) for child in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(child) for child in value)
    return value


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    try:
        state = checkpoint.get("state_dict", checkpoint)
        if not isinstance(state, dict):
            raise ValueError(f"checkpoint has no valid state_dict: {path}")
        architecture = validate_bumi_checkpoint_state_dict(state)
        return {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "global_step": checkpoint.get("global_step") if isinstance(checkpoint, dict) else None,
            "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
            "state_dict_keys": len(state),
            **architecture,
        }
    finally:
        del checkpoint


def normalized_prediction(prediction: dict[str, Any], length: int) -> torch.Tensor:
    net_outputs = prediction.get("net_outputs")
    if not isinstance(net_outputs, dict):
        raise RuntimeError("BUMI prediction has no net_outputs")
    model_output = net_outputs.get("model_output")
    if not isinstance(model_output, dict):
        raise RuntimeError("BUMI prediction has no denoiser model_output")
    for name in ("pred_x", "pred_x_start", "pred_xstart"):
        value = model_output.get(name)
        if isinstance(value, torch.Tensor):
            if value.shape[-1] != 93:
                raise RuntimeError(f"BUMI denoiser output is not 93D: {value.shape}")
            return value[0, :length].detach().cpu()
    raise RuntimeError("BUMI prediction has no normalized pred_x")


def canonicalize_target(
    path: Path,
    *,
    start_frame: int,
    length: int,
    model: Any,
) -> torch.Tensor:
    payload = torch_load(path)
    if not isinstance(payload, dict):
        raise ValueError(f"target motion must be a dictionary: {path}")
    qpos = torch.as_tensor(payload.get("qpos")).float()
    end = int(start_frame) + int(length)
    if qpos.ndim != 2 or qpos.shape[1] != 28 or start_frame < 0 or end > qpos.shape[0]:
        raise ValueError(
            f"target qpos must cover [{start_frame}:{end}] with shape [T,28]; got {qpos.shape}"
        )
    if not bool(torch.isfinite(qpos).all()):
        raise ValueError("target qpos contains NaN or Inf")
    names = tuple(map(str, payload.get("joint_names", ())))
    if names and names != model.endecoder.kinematics.joint_order:
        raise ValueError("target joint_names do not match BUMI MuJoCo-native order")
    qpos = qpos[start_frame:end].to(model.endecoder.mean.device)
    return model.endecoder.codec.encode(qpos).canonical_qpos.detach().cpu()


def render_motion(
    *,
    artifact: Path,
    mjcf: Path,
    output: Path,
    camera: str | None,
    width: int,
    height: int,
) -> None:
    command = [
        sys.executable,
        str(REPO_ROOT / "tools/eval/render_bumi_motion.py"),
        "--motion",
        str(artifact),
        "--mjcf",
        str(mjcf),
        "--output",
        str(output),
        "--qpos-key",
        "qpos",
        "--width",
        str(width),
        "--height",
        str(height),
    ]
    if camera:
        command.extend(("--camera", camera))
    subprocess.run(command, check=True)


def mux_audio(
    *,
    video: Path,
    audio: Path,
    output: Path,
    start_sec: float,
    duration_sec: float,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("--mux-audio requires ffmpeg")
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video),
        "-ss",
        f"{start_sec:.9f}",
        "-t",
        f"{duration_sec:.9f}",
        "-i",
        str(audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output),
    ]
    subprocess.run(command, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--wav", type=Path)
    source.add_argument("--edge35", type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--exp", default="gem_bumi_music_only_4set_random_v1")
    parser.add_argument("--kinematics", type=Path)
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--duration-sec", type=float)
    parser.add_argument("--cfg-scale", type=float, default=2.5)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-motion", type=Path)
    parser.add_argument("--target-start-frame", type=int, default=0)
    parser.add_argument("--render-mjcf", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--camera")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--mux-audio", action="store_true")
    parser.add_argument("--world-root-x", type=float)
    parser.add_argument("--world-root-y", type=float)
    parser.add_argument("--world-root-z", type=float)
    parser.add_argument("--world-root-yaw", type=float)
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if not args.checkpoint.expanduser().is_file():
        raise FileNotFoundError(f"BUMI checkpoint does not exist: {args.checkpoint}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if not math.isfinite(args.start_sec) or args.start_sec < 0.0:
        raise ValueError("--start-sec must be finite and >= 0")
    if args.duration_sec is not None and (
        not math.isfinite(args.duration_sec) or args.duration_sec <= 0.0
    ):
        raise ValueError("--duration-sec must be finite and > 0")
    if args.ddim_steps < 2 or args.ddim_steps > 1000:
        raise ValueError("--ddim-steps must be in 2..1000")
    if not math.isfinite(args.cfg_scale) or args.cfg_scale < 0.0:
        raise ValueError("--cfg-scale must be finite and >= 0")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive")
    if args.video is not None and args.render_mjcf is None:
        raise ValueError("--video requires --render-mjcf")
    if args.mux_audio and (args.render_mjcf is None or args.wav is None):
        raise ValueError("--mux-audio requires both --render-mjcf and --wav")
    if args.target_start_frame < 0:
        raise ValueError("--target-start-frame must be >= 0")


@torch.inference_mode()
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_arguments(args)
    kinematics_path, stats_path = configure_assets(args.kinematics, args.stats)
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    report_path = (
        args.report.expanduser().resolve()
        if args.report is not None
        else output.with_suffix(output.suffix + ".json")
    )
    seed_everything(args.seed)

    feature_started = time.perf_counter()
    if args.wav is not None:
        source_path = args.wav.expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        features, feature_metadata = extract_edge_baseline35(
            source_path,
            start_sec=args.start_sec,
            duration_sec=args.duration_sec,
            target_fps=30,
        )
    else:
        source_path = args.edge35.expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        features = load_edge35(source_path)
        feature_metadata = {
            "feature_type": "edge_baseline35",
            "feature_frames": int(features.shape[0]),
            "source_path": str(source_path),
        }
    features, has_music = align_music_frames(features, args.num_frames)
    feature_seconds = time.perf_counter() - feature_started

    checkpoint_info = checkpoint_metadata(checkpoint)
    load_started = time.perf_counter()
    with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "configs")):
        cfg = compose(config_name="train", overrides=[f"exp={args.exp}"])
    model = instantiate(cfg.model, _recursive_=False)
    load_pretrained_model(model, checkpoint)
    if str(getattr(model, "motion_backend", "")) != "bumi":
        raise RuntimeError("selected experiment/checkpoint is not the BUMI backend")
    denoiser = model.pipeline.denoiser3d.denoiser
    if int(getattr(denoiser, "output_dim", -1)) != 93:
        raise RuntimeError("selected checkpoint does not use the BUMI 93D denoiser")
    with open_dict(model.pipeline.denoiser3d.model_cfg.diffusion):
        diffusion_cfg = model.pipeline.denoiser3d.model_cfg.diffusion
        diffusion_cfg.guidance_param = float(args.cfg_scale)
        diffusion_cfg.test_timestep_respacing = str(args.ddim_steps)
        diffusion_cfg.gen_only_test_timestep_respacing = str(args.ddim_steps)
    model.pipeline.denoiser3d.init_diffusion()
    device = torch.device(args.device)
    model = model.to(device).eval()
    load_seconds = time.perf_counter() - load_started

    anchor_values = (args.world_root_x, args.world_root_y, args.world_root_yaw)
    if args.world_root_z is not None and not all(value is not None for value in anchor_values):
        raise ValueError("--world-root-z requires complete world XY/yaw placement")
    if any(value is not None for value in anchor_values) and not all(
        value is not None for value in anchor_values
    ):
        raise ValueError("world placement requires root X, Y and yaw together")
    world_anchor = None
    if all(value is not None for value in anchor_values):
        world_anchor = {
            "root_xy": [args.world_root_x, args.world_root_y],
            "yaw": args.world_root_yaw,
        }
        if args.world_root_z is not None:
            world_anchor["anchor_z"] = args.world_root_z

    generation_started = time.perf_counter()
    prediction = model.predict(
        {
            "music_embed": features,
            "has_music_mask": has_music,
            "length": len(features),
            "music_path": str(source_path),
            "world_anchor": world_anchor,
        }
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    generation_seconds = time.perf_counter() - generation_started
    normalized_93d = normalized_prediction(prediction, len(features))

    artifact = {
        key: cpu_tree(value) for key, value in prediction.items() if key != "net_outputs"
    }
    artifact.update(
        {
            "contract_version": "genmo.bumi_motion_prediction.v1",
            "normalized_motion_93d": normalized_93d,
            "music_features": features,
            "has_music_mask": has_music,
            "feature_metadata": feature_metadata,
            "seed": args.seed,
            "cfg_scale": float(args.cfg_scale),
            "ddim_steps": args.ddim_steps,
            "experiment_config": str(args.exp),
            "checkpoint": checkpoint_info,
            "kinematics_path": str(kinematics_path),
            "kinematics_sha256": model.endecoder.kinematics.kinematics_sha256,
            "stats_path": str(stats_path),
            "stats_sha256": sha256_file(stats_path),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)

    pred_canonical = torch.as_tensor(artifact["qpos_canonical"]).float()
    target_canonical = None
    target_path = None
    if args.target_motion is not None:
        target_path = args.target_motion.expanduser().resolve()
        target_canonical = canonicalize_target(
            target_path,
            start_frame=args.target_start_frame,
            length=len(features),
            model=model,
        )
    ground_height = -float(model.endecoder.kinematics.default_qpos[2])
    metrics = compute_bumi_kinematic_metrics(
        pred_canonical,
        model.endecoder.kinematics,
        target_qpos=target_canonical,
        valid_mask=has_music,
        music_beats=features[:, 34],
        ground_height=ground_height,
    )

    rendered_video = None
    if args.render_mjcf is not None:
        mjcf = args.render_mjcf.expanduser().resolve()
        if not mjcf.is_file():
            raise FileNotFoundError(mjcf)
        final_video = (
            args.video.expanduser().resolve()
            if args.video is not None
            else output.with_suffix(".mp4")
        )
        if args.mux_audio:
            silent = final_video.with_name(final_video.stem + ".silent.mp4")
            render_motion(
                artifact=output,
                mjcf=mjcf,
                output=silent,
                camera=args.camera,
                width=args.width,
                height=args.height,
            )
            selected_duration = float(
                feature_metadata.get("selected_duration_sec", len(features) / 30.0)
            )
            mux_audio(
                video=silent,
                audio=source_path,
                output=final_video,
                start_sec=args.start_sec,
                duration_sec=selected_duration,
            )
        else:
            render_motion(
                artifact=output,
                mjcf=mjcf,
                output=final_video,
                camera=args.camera,
                width=args.width,
                height=args.height,
            )
        rendered_video = str(final_video)

    report = {
        "contract_version": "genmo.bumi_demo_report.v1",
        "output": str(output),
        "video": rendered_video,
        "source": str(source_path),
        "target": None if target_path is None else str(target_path),
        "qpos_shape": list(torch.as_tensor(artifact["qpos"]).shape),
        "normalized_motion_shape": list(normalized_93d.shape),
        "fps": artifact["fps"],
        "seed": args.seed,
        "cfg_scale": args.cfg_scale,
        "ddim_steps": args.ddim_steps,
        "checkpoint": checkpoint_info,
        "metrics": metrics_to_json(metrics),
        "timing_seconds": {
            "feature_extraction": feature_seconds,
            "model_load": load_seconds,
            "generation": generation_seconds,
        },
        "scope": "kinematic generation/evaluation; no GMT dynamics claim",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
