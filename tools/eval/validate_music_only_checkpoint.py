#!/usr/bin/env python3
"""Generate and validate 151-D/SMPL motion from real music with a specialist ckpt."""

from __future__ import annotations

import argparse
import json
import os
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

from gem.datasets.aistpp.aistplusplus import (  # noqa: E402
    load_music_feature_tensor,
    validate_musicfeat_v2,
)
from gem.runtime.motion_sanity import evaluate_global_motion_sanity  # noqa: E402
from gem.utils.music_features import extract_edge_baseline35  # noqa: E402
from gem.utils.net_utils import load_pretrained_model  # noqa: E402
from scripts.demo.demo_music import (  # noqa: E402
    build_music_only_data,  # noqa: E402
    mux_selected_audio,
    render_global_motion,
)

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

DEFAULT_CHECKPOINT = Path("inputs/checkpoints/music_only_aistpp/version_1/last.ckpt")


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(child) for key, child in value.items()}
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio", type=Path)
    source.add_argument("--music-embed", type=Path)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audio-start-sec", type=float, default=0.0)
    parser.add_argument("--audio-duration-sec", type=float, default=4.0)
    parser.add_argument("--feature-start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=120)
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--cfg-scale", type=float, default=2.5)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--postproc", action="store_true")
    parser.add_argument("--render-video", action="store_true")
    parser.add_argument("--mux-audio", action="store_true")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser


def load_selected_music(args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, Any]]:
    if args.max_frames <= 0:
        raise ValueError("--max-frames must be > 0")
    if not 1 <= args.num_frames <= args.max_frames:
        raise ValueError(f"--num-frames must be in 1..{args.max_frames}")
    if args.feature_start_frame < 0:
        raise ValueError("--feature-start-frame must be >= 0")
    if args.audio is not None:
        if not args.audio.is_file():
            raise FileNotFoundError(args.audio)
        music, extraction = extract_edge_baseline35(
            args.audio,
            start_sec=args.audio_start_sec,
            duration_sec=args.audio_duration_sec,
            target_fps=30,
        )
        metadata: dict[str, Any] = {
            "kind": "audio",
            "path": str(args.audio.resolve()),
            **extraction,
        }
    else:
        if not args.music_embed.is_file():
            raise FileNotFoundError(args.music_embed)
        music = load_music_feature_tensor(args.music_embed)
        validate_musicfeat_v2(music, source=args.music_embed)
        metadata = {
            "kind": "music_embed",
            "path": str(args.music_embed.resolve()),
        }
    start = int(args.feature_start_frame)
    end = start + int(args.num_frames)
    if music.shape[0] < end:
        raise ValueError(f"music has {music.shape[0]} frames, but [{start}:{end}] was requested")
    selected = music[start:end].float().contiguous()
    validate_musicfeat_v2(selected, source=metadata["path"])
    metadata.update(
        {
            "original_feature_frames": int(music.shape[0]),
            "selected_start_frame": start,
            "selected_frames": int(selected.shape[0]),
        }
    )
    return selected, metadata


def _validate_body_group(group: Any, frames: int) -> dict[str, torch.Tensor]:
    if not isinstance(group, dict):
        raise RuntimeError("prediction is missing pred_body_params_global")
    expected = {
        "body_pose": (frames, 63),
        "global_orient": (frames, 3),
        "transl": (frames, 3),
        "betas": (frames, 10),
    }
    result: dict[str, torch.Tensor] = {}
    for name, shape in expected.items():
        value = group.get(name)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise RuntimeError(
                f"pred_body_params_global.{name} must have shape {shape}; "
                f"got {getattr(value, 'shape', None)}"
            )
        if not torch.isfinite(value).all():
            raise RuntimeError(f"pred_body_params_global.{name} contains NaN or Inf")
        result[name] = value.detach().cpu()
    return result


@torch.inference_mode()
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.ckpt.is_file():
        raise FileNotFoundError(args.ckpt)
    if not torch.cuda.is_available():
        raise RuntimeError("checkpoint generation validation requires a CUDA GPU")
    if args.ddim_steps < 2 or args.ddim_steps > 1000:
        raise ValueError("--ddim-steps must be in 2..1000")
    if not np.isfinite(args.cfg_scale) or args.cfg_scale < 0:
        raise ValueError("--cfg-scale must be finite and >= 0")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be > 0")
    if args.mux_audio and args.audio is None:
        raise ValueError("--mux-audio requires --audio so the original waveform is available")
    if args.mux_audio:
        args.render_video = True
    music, source_metadata = load_selected_music(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "configs")):
        cfg = compose(config_name="train", overrides=["exp=gem_smpl_music_only"])
    cfg.model_cfg.diffusion.guidance_param = float(args.cfg_scale)
    with open_dict(cfg.model_cfg.diffusion):
        cfg.model_cfg.diffusion.gen_only_test_timestep_respacing = str(args.ddim_steps)
    if list(cfg.pipeline.args.in_attr) != ["encoded_music"]:
        raise RuntimeError(f"unexpected condition list: {list(cfg.pipeline.args.in_attr)}")

    print(f"[Music] 加载 checkpoint: {args.ckpt}")
    model = instantiate(cfg.model, _recursive_=False)
    checkpoint = load_pretrained_model(model, args.ckpt)
    checkpoint_global_step = checkpoint.get("global_step")
    checkpoint_epoch = checkpoint.get("epoch")
    del checkpoint
    model = model.cuda().eval()
    denoiser = model.pipeline.denoiser3d.denoiser
    if model.text_condition_enabled or hasattr(denoiser, "embed_text"):
        raise RuntimeError("music-only specialist unexpectedly contains text conditioning")

    data = build_music_only_data(music)
    prediction = model.predict(data, static_cam=True, postproc=args.postproc)
    outputs = prediction["net_outputs"]
    raw_motion = outputs["model_output"]["pred_x"].detach().cpu()
    if tuple(raw_motion.shape) != (1, args.num_frames, 151):
        raise RuntimeError(f"unexpected raw motion shape: {tuple(raw_motion.shape)}")
    if not torch.isfinite(raw_motion).all():
        raise RuntimeError("generated 151-D motion contains NaN or Inf")
    body_global = _validate_body_group(prediction.get("body_params_global"), args.num_frames)
    motion_sanity = evaluate_global_motion_sanity(body_global)
    static_logits = outputs["model_output"].get("static_conf_logits")
    if not isinstance(static_logits, torch.Tensor) or not torch.isfinite(static_logits).all():
        raise RuntimeError("static_conf_logits is missing or non-finite")

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    report = {
        "checkpoint": str(args.ckpt.resolve()),
        "checkpoint_global_step": checkpoint_global_step,
        "checkpoint_epoch": checkpoint_epoch,
        "condition_list": list(cfg.pipeline.args.in_attr),
        "text_condition_enabled": False,
        "music_source": source_metadata,
        "music_shape": list(music.shape),
        "music_finite": bool(torch.isfinite(music).all()),
        "generated_motion_shape": list(raw_motion.shape),
        "generated_motion_finite": True,
        "body_parameter_shapes": {key: list(value.shape) for key, value in body_global.items()},
        "static_conf_logits_shape": list(static_logits.shape),
        "cfg_scale": args.cfg_scale,
        "ddim_steps": args.ddim_steps,
        "seed": args.seed,
        "postproc": args.postproc,
        "training_window_frames": 120,
        "uses_sliding_attention": bool(args.num_frames > 120),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "motion_sanity": motion_sanity,
        "final_pass": bool(motion_sanity["physical_sanity_pass"]),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(music, args.output_dir / "music_features.pt")
    torch.save(raw_motion[0], args.output_dir / "generated_motion_151d.pt")
    torch.save(_cpu_tree(body_global), args.output_dir / "pred_body_params_global.pt")

    rendered_video: Path | None = None
    muxed_video: Path | None = None
    if args.render_video:
        # The motion is evaluated independently of body shape.  Use a neutral body
        # for stable visual comparison while preserving pose/orientation/translation.
        render_params = dict(body_global)
        render_params["betas"] = torch.zeros_like(body_global["betas"])
        rendered_video = render_global_motion(
            args.output_dir, render_params, args.width, args.height
        )
        if rendered_video is None:
            raise RuntimeError("motion generation passed, but SMPL rendering failed")
    if args.mux_audio:
        muxed_video = args.output_dir / "motion_with_audio.mp4"
        audio_start = float(args.audio_start_sec) + float(args.feature_start_frame) / 30.0
        mux_ok = mux_selected_audio(
            rendered_video,
            args.audio,
            muxed_video,
            audio_start,
            float(args.num_frames) / 30.0,
        )
        if not mux_ok:
            raise RuntimeError("motion rendering passed, but ffmpeg audio mux failed")
    report["render"] = {
        "enabled": bool(args.render_video),
        "neutral_zero_betas": bool(args.render_video),
        "video": None if rendered_video is None else str(rendered_video.resolve()),
        "width": args.width,
        "height": args.height,
    }
    report["audio_mux"] = {
        "enabled": bool(args.mux_audio),
        "video": None if muxed_video is None else str(muxed_video.resolve()),
    }
    (args.output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["final_pass"]:
        print(f"[Music] 数值与物理粗检通过，结果位于: {args.output_dir.resolve()}")
        return 0
    print(
        "[Music] 张量 finite，但物理粗检失败；该 checkpoint 不能视为有效舞蹈模型。"
        f" 结果位于: {args.output_dir.resolve()}"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
