#!/usr/bin/env python3
"""Generate a music-only GEM-SMPL video from WAV or EDGE baseline35 features."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import open_dict

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

REPO_ROOT = Path(__file__).resolve().parents[1]
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
    build_music_only_data,
    mux_selected_audio,
    render_global_motion,
)

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
    source.add_argument("--audio", type=Path, help="Raw WAV/audio input")
    source.add_argument("--music-embed", type=Path, help="Precomputed [T,35] EDGE feature")
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audio-start-sec", type=float, default=0.0)
    parser.add_argument(
        "--audio-duration-sec",
        type=float,
        default=None,
        help="Selected audio duration; default uses the remainder of the audio",
    )
    parser.add_argument("--feature-start-frame", type=int, default=0)
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Frames to generate; default uses all selected features",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=600,
        help="Safety limit. The denoiser uses sliding 120-frame attention above 120 frames.",
    )
    parser.add_argument("--cfg-scale", type=float, default=2.5)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--postproc", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--no-mux-audio", action="store_true")
    parser.add_argument(
        "--allow-physically-invalid",
        action="store_true",
        help="Return success despite gross trajectory/orientation failure (diagnostics only)",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser


def select_music_window(
    music: torch.Tensor,
    *,
    start_frame: int,
    num_frames: int | None,
    max_frames: int,
    source: str,
) -> torch.Tensor:
    """Select a non-truncated inference range, including lengths above 120."""
    validate_musicfeat_v2(music, source=source)
    if start_frame < 0:
        raise ValueError("--feature-start-frame must be >= 0")
    if max_frames <= 0:
        raise ValueError("--max-frames must be > 0")
    if start_frame >= music.shape[0]:
        raise ValueError(
            f"feature start {start_frame} is outside music length {music.shape[0]}"
        )
    frames = int(music.shape[0] - start_frame) if num_frames is None else int(num_frames)
    if frames <= 0:
        raise ValueError("--num-frames must be > 0 when provided")
    if frames > max_frames:
        raise ValueError(
            f"requested {frames} frames exceeds --max-frames={max_frames}; "
            "select a shorter audio range or explicitly raise the safety limit"
        )
    end = start_frame + frames
    if end > music.shape[0]:
        raise ValueError(
            f"music has {music.shape[0]} frames, but [{start_frame}:{end}] was requested"
        )
    selected = music[start_frame:end].float().contiguous()
    validate_musicfeat_v2(selected, source=source)
    return selected


def load_selected_music(args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, Any]]:
    if args.audio is not None:
        if not args.audio.is_file():
            raise FileNotFoundError(args.audio)
        music, extraction = extract_edge_baseline35(
            args.audio,
            start_sec=args.audio_start_sec,
            duration_sec=args.audio_duration_sec,
            target_fps=30,
        )
        source = str(args.audio.resolve())
        metadata: dict[str, Any] = {"kind": "audio", "path": source, **extraction}
    else:
        if not args.music_embed.is_file():
            raise FileNotFoundError(args.music_embed)
        music = load_music_feature_tensor(args.music_embed)
        source = str(args.music_embed.resolve())
        metadata = {"kind": "music_embed", "path": source}
    selected = select_music_window(
        music,
        start_frame=args.feature_start_frame,
        num_frames=args.num_frames,
        max_frames=args.max_frames,
        source=source,
    )
    metadata.update(
        {
            "original_feature_frames": int(music.shape[0]),
            "selected_start_frame": int(args.feature_start_frame),
            "selected_frames": int(selected.shape[0]),
            "uses_sliding_attention": bool(selected.shape[0] > 120),
            "training_window_frames": 120,
        }
    )
    return selected, metadata


def _validate_body_group(group: Any, frames: int) -> dict[str, torch.Tensor]:
    if not isinstance(group, dict):
        raise RuntimeError("prediction is missing body_params_global")
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
                f"body_params_global.{name} must be {shape}, "
                f"got {getattr(value, 'shape', None)}"
            )
        if not torch.isfinite(value).all():
            raise RuntimeError(f"body_params_global.{name} contains NaN or Inf")
        result[name] = value.detach().cpu()
    return result


@torch.inference_mode()
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.ckpt.is_file():
        raise FileNotFoundError(args.ckpt)
    if not torch.cuda.is_available():
        raise RuntimeError("music-only diffusion inference requires a CUDA GPU")
    if not np.isfinite(args.cfg_scale) or args.cfg_scale < 0:
        raise ValueError("--cfg-scale must be finite and >= 0")
    if not 2 <= args.ddim_steps <= 1000:
        raise ValueError("--ddim-steps must be in 2..1000")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be > 0")
    music, source_metadata = load_selected_music(args)
    frames = int(music.shape[0])

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
        raise RuntimeError(f"unexpected conditions: {list(cfg.pipeline.args.in_attr)}")

    print(f"[Demo] 加载 checkpoint: {args.ckpt}")
    model = instantiate(cfg.model, _recursive_=False)
    checkpoint = load_pretrained_model(model, args.ckpt)
    checkpoint_step = checkpoint.get("global_step")
    del checkpoint
    model = model.cuda().eval()
    denoiser = model.pipeline.denoiser3d.denoiser
    if model.text_condition_enabled or hasattr(denoiser, "embed_text"):
        raise RuntimeError("music-only specialist unexpectedly contains text conditioning")
    if frames > denoiser.max_len:
        print(
            f"[Demo] {frames} frames > training window {denoiser.max_len}; "
            "using the repository's sliding local-attention path."
        )

    prediction = model.predict(
        build_music_only_data(music), static_cam=True, postproc=args.postproc
    )
    outputs = prediction["net_outputs"]
    generated = outputs["model_output"]["pred_x"].detach().cpu()
    if tuple(generated.shape) != (1, frames, 151) or not torch.isfinite(generated).all():
        raise RuntimeError(f"invalid generated 151D motion: {tuple(generated.shape)}")
    body_global = _validate_body_group(prediction.get("body_params_global"), frames)
    sanity = evaluate_global_motion_sanity(body_global)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(music, args.output_dir / "music_features.pt")
    torch.save(generated[0], args.output_dir / "generated_motion_151d.pt")
    torch.save(_cpu_tree(body_global), args.output_dir / "pred_body_params_global.pt")

    rendered: Path | None = None
    muxed: Path | None = None
    if not args.no_render:
        render_params = dict(body_global)
        render_params["betas"] = torch.zeros_like(body_global["betas"])
        rendered = render_global_motion(
            args.output_dir, render_params, args.width, args.height
        )
        if rendered is None:
            raise RuntimeError("SMPL motion rendering failed")
    if args.audio is not None and not args.no_mux_audio:
        if rendered is None:
            warnings.warn("audio mux skipped because --no-render was set", stacklevel=2)
        else:
            muxed = args.output_dir / "motion_with_audio.mp4"
            audio_start = args.audio_start_sec + args.feature_start_frame / 30.0
            if not mux_selected_audio(
                rendered, args.audio, muxed, audio_start, frames / 30.0
            ):
                raise RuntimeError("ffmpeg audio mux failed")

    report = {
        "checkpoint": str(args.ckpt.resolve()),
        "checkpoint_global_step": checkpoint_step,
        "condition_list": list(cfg.pipeline.args.in_attr),
        "music_source": source_metadata,
        "music_shape": list(music.shape),
        "generated_shape": list(generated.shape),
        "cfg_scale": args.cfg_scale,
        "ddim_steps": args.ddim_steps,
        "seed": args.seed,
        "postproc": args.postproc,
        "rendered_video": None if rendered is None else str(rendered.resolve()),
        "muxed_video": None if muxed is None else str(muxed.resolve()),
        "motion_sanity": sanity,
        "final_pass": bool(sanity["physical_sanity_pass"]),
    }
    (args.output_dir / "demo_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["final_pass"]:
        return 0
    message = (
        "生成文件已保存，但根轨迹/朝向物理粗检失败。不要把该结果视为有效模型输出。"
    )
    if args.allow_physically_invalid:
        warnings.warn(message, stacklevel=2)
        return 0
    print(f"[Demo] ERROR: {message}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
