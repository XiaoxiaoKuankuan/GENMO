#!/usr/bin/env python3
"""Generate BUMI qpos28 directly from WAV or precomputed EDGE35 features."""

from __future__ import annotations

import argparse
import json
import random
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

from gem.utils.music_features import EDGE_FEATURE_DIM, extract_edge_baseline35  # noqa: E402
from gem.utils.net_utils import load_pretrained_model  # noqa: E402


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_edge35(path: Path) -> torch.Tensor:
    payload = _torch_load(path)
    if not isinstance(payload, torch.Tensor):
        raise ValueError(f"Precomputed EDGE35 file must contain Tensor[T,35]: {path}")
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
        return features[:target_frames].contiguous(), torch.ones(target_frames, dtype=torch.bool)
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
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--wav", type=Path)
    source.add_argument("--edge35", type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--duration-sec", type=float)
    parser.add_argument("--cfg-scale", type=float, default=2.5)
    parser.add_argument("--ddim-steps", default="50")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--world-root-x", type=float)
    parser.add_argument("--world-root-y", type=float)
    parser.add_argument("--world-root-z", type=float)
    parser.add_argument("--world-root-yaw", type=float)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"BUMI checkpoint does not exist: {args.checkpoint}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.wav is not None:
        features, feature_metadata = extract_edge_baseline35(
            args.wav,
            start_sec=args.start_sec,
            duration_sec=args.duration_sec,
        )
        music_path = args.wav.expanduser().resolve()
    else:
        features = load_edge35(args.edge35.expanduser().resolve())
        feature_metadata = {
            "feature_type": "edge_baseline35",
            "feature_frames": int(features.shape[0]),
            "source_path": str(args.edge35.expanduser().resolve()),
        }
        music_path = args.edge35.expanduser().resolve()
    features, has_music = align_music_frames(features, args.num_frames)

    with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "configs")):
        cfg = compose(config_name="train", overrides=["exp=gem_bumi_music_only_4set"])
    model = instantiate(cfg.model, _recursive_=False)
    load_pretrained_model(model, args.checkpoint)
    with open_dict(model.pipeline.denoiser3d.model_cfg.diffusion):
        model.pipeline.denoiser3d.model_cfg.diffusion.guidance_param = float(args.cfg_scale)
        model.pipeline.denoiser3d.model_cfg.diffusion.test_timestep_respacing = str(
            args.ddim_steps
        )
        model.pipeline.denoiser3d.model_cfg.diffusion.gen_only_test_timestep_respacing = str(
            args.ddim_steps
        )
    # Rebuild only the inference diffusion schedules after explicit CLI overrides.
    model.pipeline.denoiser3d.init_diffusion()
    model = model.to(torch.device(args.device)).eval()

    anchor_values = (args.world_root_x, args.world_root_y, args.world_root_yaw)
    if args.world_root_z is not None and not all(value is not None for value in anchor_values):
        raise ValueError("--world-root-z is valid only with complete world XY/yaw placement")
    if any(value is not None for value in anchor_values) and not all(
        value is not None for value in anchor_values
    ):
        raise ValueError(
            "World placement requires --world-root-x, --world-root-y, and --world-root-yaw together"
        )
    world_anchor = None
    if all(value is not None for value in anchor_values):
        world_anchor = {
            "root_xy": [args.world_root_x, args.world_root_y],
            "yaw": args.world_root_yaw,
        }
        if args.world_root_z is not None:
            world_anchor["anchor_z"] = args.world_root_z
    with torch.no_grad():
        prediction = model.predict(
            {
                "music_embed": features,
                "has_music_mask": has_music,
                "length": len(features),
                "music_path": str(music_path),
                "world_anchor": world_anchor,
            }
        )
    artifact = {
        key: cpu_tree(value)
        for key, value in prediction.items()
        if key != "net_outputs"
    }
    artifact["feature_metadata"] = feature_metadata
    artifact["seed"] = args.seed
    artifact["cfg_scale"] = float(args.cfg_scale)
    artifact["ddim_steps"] = str(args.ddim_steps)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    summary = {
        "output": str(output),
        "qpos_shape": list(artifact["qpos"].shape),
        "fps": artifact["fps"],
        "robot_name": artifact["robot_name"],
        "qpos_order": artifact["qpos_order"],
        "feature_dim": artifact["feature_dim"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
