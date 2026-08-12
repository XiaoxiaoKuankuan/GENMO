#!/usr/bin/env python3
"""Minimal inference entry point for a trained music-only GEM-SMPL specialist."""

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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gem.datasets.aistpp.aistplusplus import (  # noqa: E402
    load_music_feature_tensor,
    validate_musicfeat_v2,
)
from gem.utils.net_utils import load_pretrained_model  # noqa: E402
from scripts.demo.demo_music import build_music_only_data  # noqa: E402


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(child) for key, child in value.items()}
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--music-embed", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cfg-scale", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--postproc", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.ckpt.is_file():
        raise FileNotFoundError(args.ckpt)
    if not torch.cuda.is_available():
        raise RuntimeError("music-only inference requires a CUDA GPU")
    if not np.isfinite(args.cfg_scale) or args.cfg_scale < 0:
        raise ValueError("--cfg-scale must be finite and non-negative")

    music = load_music_feature_tensor(args.music_embed)
    validate_musicfeat_v2(music, source=args.music_embed)
    if music.shape[0] <= 0 or music.shape[0] > 120:
        raise ValueError(
            f"music-only smoke inference supports 1..120 frames; got {music.shape[0]}"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    with initialize_config_dir(
        version_base="1.3", config_dir=str(REPO_ROOT / "configs")
    ):
        cfg = compose(config_name="train", overrides=["exp=gem_smpl_music_only"])
    cfg.model_cfg.diffusion.guidance_param = float(args.cfg_scale)
    if list(cfg.pipeline.args.in_attr) != ["encoded_music"]:
        raise RuntimeError(f"unexpected specialist conditions: {list(cfg.pipeline.args.in_attr)}")

    model = instantiate(cfg.model, _recursive_=False)
    load_pretrained_model(model, args.ckpt)
    model = model.cuda().eval()
    denoiser = model.pipeline.denoiser3d.denoiser
    if model.text_condition_enabled or hasattr(denoiser, "embed_text"):
        raise RuntimeError("music-only specialist unexpectedly contains a text condition")

    data = build_music_only_data(music)
    with torch.no_grad():
        prediction = model.predict(data, static_cam=True, postproc=args.postproc)
    net_outputs = prediction["net_outputs"]
    body_global = net_outputs.get("pred_body_params_global")
    if not isinstance(body_global, dict):
        raise RuntimeError("prediction is missing pred_body_params_global")
    for field, value in body_global.items():
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise RuntimeError(f"pred_body_params_global.{field} is missing or non-finite")
    generated = net_outputs["model_output"]["pred_x"].detach().cpu()[0]
    if generated.shape != (music.shape[0], 151) or not torch.isfinite(generated).all():
        raise RuntimeError(f"invalid generated motion: {tuple(generated.shape)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(generated, args.output_dir / "generated_motion_151d.pt")
    torch.save(_cpu_tree(body_global), args.output_dir / "pred_body_params_global.pt")
    metadata = {
        "checkpoint": str(args.ckpt.resolve()),
        "music_feature_path": str(args.music_embed.resolve()),
        "music_shape": list(music.shape),
        "generated_shape": list(generated.shape),
        "generated_finite": True,
        "cfg_scale": float(args.cfg_scale),
        "seed": args.seed,
        "postproc": args.postproc,
        "condition_list": list(cfg.pipeline.args.in_attr),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
