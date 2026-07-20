#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Generate SMPL human motion from an arbitrary music file using full GEM DDIM."""

from __future__ import annotations

import argparse
import gc
import json
import random
import shutil
import subprocess
import warnings
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from gem.utils.music_features import EDGE_TARGET_FPS, extract_edge_baseline35

DEFAULT_CHECKPOINT = Path("inputs/pretrained/gem_smpl.ckpt")
DEFAULT_BODY_MODEL = Path("inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz")
DEFAULT_OUTPUT_ROOT = Path("outputs/music_demo")

_MUSIC_CHECKPOINT_ERROR = (
    "The supplied checkpoint does not contain the music-conditioned diffusion weights "
    "required for music-to-motion generation. Use the official gem_smpl.ckpt or a "
    "checkpoint trained with exp=gem_smpl; regression-only checkpoints are unsupported."
)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch before DDIM creates Gaussian noise."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_music_only_data(
    music_features: torch.Tensor,
    *,
    width: int = 1280,
    height: int = 720,
    focal: float | None = None,
) -> dict[str, Any]:
    """Build GEM's unified input contract with music as the only valid condition."""
    if not isinstance(music_features, torch.Tensor):
        raise TypeError("music_features must be a torch.Tensor")
    if music_features.ndim != 2 or music_features.shape[1] != 35:
        raise ValueError(
            f"music_features must have shape [L, 35]; got {tuple(music_features.shape)}"
        )
    if music_features.shape[0] <= 0:
        raise ValueError("music_features must contain at least one frame")
    if not torch.isfinite(music_features).all():
        raise ValueError("music_features contains NaN or Inf")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be greater than zero")
    resolved_focal = float(max(width, height) if focal is None else focal)
    if resolved_focal <= 0:
        raise ValueError("focal must be greater than zero")

    from gem.utils.geo_transform import compute_cam_angvel

    length = int(music_features.shape[0])
    K = torch.tensor(
        [
            [resolved_focal, 0.0, width / 2.0],
            [0.0, resolved_focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    K_fullimg = K.unsqueeze(0).repeat(length, 1, 1)
    R_w2c = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(length, 1, 1)
    R_for_velocity = R_w2c
    if length == 1:
        R_for_velocity = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)
    cam_angvel = compute_cam_angvel(R_for_velocity, padding_last=True)[:length].float()
    bbox = torch.tensor([width / 2.0, height / 2.0, 0.8 * min(width, height)], dtype=torch.float32)

    def false_mask() -> torch.Tensor:
        return torch.zeros(length, dtype=torch.bool)

    return {
        "music_embed": music_features.detach().cpu().float(),
        "kp2d": torch.zeros(length, 17, 3, dtype=torch.float32),
        "f_imgseq": torch.zeros(length, 1024, dtype=torch.float32),
        "R_w2c": R_w2c,
        "cam_angvel": cam_angvel,
        "cam_tvel": torch.zeros(length, 3, dtype=torch.float32),
        "K_fullimg": K_fullimg,
        "bbx_xys": bbox.unsqueeze(0).repeat(length, 1),
        "has_text": torch.tensor([False], dtype=torch.bool),
        "caption": "",
        "length": torch.tensor(length, dtype=torch.long),
        "meta": [{"mode": "default", "source": "music_only"}],
        "mask": {
            "has_music_mask": torch.ones(length, dtype=torch.bool),
            "has_img_mask": false_mask(),
            "has_2d_mask": false_mask(),
            "has_cam_mask": false_mask(),
            "has_audio_mask": false_mask(),
        },
    }


def validate_music_checkpoint(ckpt_path: str | Path) -> int:
    """Verify checkpoint music weights and return their exact input dimension."""
    path = Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(f"GEM checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    try:
        state_dict = checkpoint.get("state_dict", checkpoint)
        if not isinstance(state_dict, dict):
            raise RuntimeError(f"Checkpoint '{path}' does not contain a valid state_dict")
        music_keys = [str(key) for key in state_dict if "music_embedder" in str(key)]
        if not music_keys:
            raise RuntimeError(_MUSIC_CHECKPOINT_ERROR)
        fc1_entries = [
            (str(key), value)
            for key, value in state_dict.items()
            if str(key).endswith("music_embedder.fc1.weight")
        ]
        if not fc1_entries:
            raise RuntimeError(
                f"{_MUSIC_CHECKPOINT_ERROR} The first music Linear weight was not found."
            )
        _, weight = fc1_entries[0]
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise RuntimeError("Checkpoint music_embedder.fc1.weight is not a 2D tensor")
        return int(weight.shape[1])
    finally:
        del checkpoint
        gc.collect()


def inspect_model_music_input_dim(model: Any) -> int:
    """Validate model music conditioning and return the first Linear in_features."""
    if "encoded_music" not in model.pipeline.args.in_attr:
        raise RuntimeError("The loaded exp=gem_smpl model does not accept encoded_music")
    if not hasattr(model, "music_embedder"):
        raise RuntimeError("The loaded exp=gem_smpl model has no music_embedder")
    first_linear = next(
        (module for module in model.music_embedder.modules() if isinstance(module, nn.Linear)),
        None,
    )
    if first_linear is None:
        raise RuntimeError("model.music_embedder contains no nn.Linear layer")
    return int(first_linear.in_features)


def _resolve_checkpoint(path: Path) -> Path:
    """Resolve the full checkpoint, visibly downloading the official file if absent."""
    if path.is_file():
        return path
    if path != DEFAULT_CHECKPOINT:
        raise FileNotFoundError(
            f"The explicitly selected checkpoint does not exist: {path}. "
            f"Omit --ckpt_path to use/download the official {DEFAULT_CHECKPOINT}."
        )
    from gem.utils.hf_utils import download_checkpoint

    print(
        f"[Checkpoint] {path} is missing. Downloading the official full gem_smpl.ckpt "
        "from nvidia/GEM-X; this is a large, explicit download."
    )
    downloaded = Path(download_checkpoint())
    if not downloaded.is_file():
        raise FileNotFoundError(f"Downloaded checkpoint was not found at: {downloaded}")
    return downloaded


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(child) for child in value]
    return value


def _validate_body_group(group: Any, name: str, length: int) -> dict[str, torch.Tensor]:
    if not isinstance(group, dict):
        raise RuntimeError(f"GEM prediction is missing '{name}'")
    shapes = {
        "body_pose": (length, 63),
        "global_orient": (length, 3),
        "transl": (length, 3),
        "betas": (length, 10),
    }
    result: dict[str, torch.Tensor] = {}
    for field, shape in shapes.items():
        value = group.get(field)
        if value is None and field == "betas":
            warnings.warn(f"{name}.betas is absent; saving zero betas", stacklevel=2)
            value = torch.zeros(shape, dtype=torch.float32)
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"GEM prediction is missing tensor '{name}.{field}'")
        if tuple(value.shape) != shape:
            raise RuntimeError(
                f"Unexpected {name}.{field} shape {tuple(value.shape)}; expected {shape}"
            )
        if not torch.isfinite(value).all():
            raise RuntimeError(f"GEM output contains NaN or Inf in '{name}.{field}'")
        result[field] = value
    return result


def extract_motion_151d(pred: dict[str, Any], length: int) -> torch.Tensor:
    """Extract the pipeline's documented final ``model_output['pred_x']`` tensor."""
    net_outputs = pred.get("net_outputs")
    if not isinstance(net_outputs, dict):
        raise RuntimeError("GEM prediction is missing net_outputs")
    model_output = net_outputs.get("model_output")
    if not isinstance(model_output, dict) or "pred_x" not in model_output:
        keys = [] if not isinstance(model_output, dict) else sorted(model_output)
        raise RuntimeError(f"GEM model_output has no final pred_x tensor; keys={keys}")
    pred_x = model_output["pred_x"]
    if not isinstance(pred_x, torch.Tensor) or tuple(pred_x.shape) != (1, length, 151):
        raise RuntimeError(
            f"Final pred_x must have shape [1, {length}, 151]; got {getattr(pred_x, 'shape', None)}"
        )
    if not torch.isfinite(pred_x).all():
        raise RuntimeError("GEM final pred_x contains NaN or Inf")
    return pred_x[0].detach().cpu()


def _write_json(path: Path, value: Any, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _torch_save(path: Path, value: Any, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path)


def render_global_motion(
    sample_dir: Path,
    body_params_global: dict[str, torch.Tensor],
    width: int,
    height: int,
) -> Path | None:
    """Render the global-coordinate SMPL motion using existing demo helpers."""
    try:
        try:
            from demo_utils import normalize_global_verts, render_global_frames
        except ModuleNotFoundError:
            from scripts.demo.demo_utils import normalize_global_verts, render_global_frames
        from gem.utils.smplx_utils import make_smplx
        from gem.utils.video_io_utils import save_video

        body_model = make_smplx("supermotion").cuda().eval()
        faces = torch.from_numpy(body_model.faces.astype(np.int32)).long()
        vertices = normalize_global_verts(body_model, body_params_global)
        frames = render_global_frames(vertices, faces, width, height)
        output = sample_dir / "motion_global.mp4"
        save_video(frames, str(output), fps=Fraction(EDGE_TARGET_FPS, 1))
        return output
    except Exception as exc:
        warnings.warn(
            "SMPL parameters were saved, but global rendering failed. Check Open3D and "
            f"rendering assets. Original error: {exc}",
            stacklevel=2,
        )
        return None


def mux_selected_audio(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    start_sec: float,
    duration_sec: float,
    overwrite: bool,
) -> bool:
    """Mux the selected source-audio range into a rendered video with ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        warnings.warn(
            "ffmpeg is unavailable; motion generation succeeded but audio was not muxed",
            stacklevel=2,
        )
        return False
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {output_path}")
    command = [ffmpeg, "-y" if overwrite else "-n", "-i", str(video_path)]
    command.extend(["-ss", f"{start_sec:.9f}", "-t", f"{duration_sec:.9f}"])
    command.extend(
        [
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(output_path),
        ]
    )
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        warnings.warn(
            "ffmpeg failed; motion generation succeeded but audio was not muxed. "
            f"Error: {result.stderr.strip()}",
            stacklevel=2,
        )
        return False
    return True


def build_parser() -> argparse.ArgumentParser:
    """Build the music-only inference command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--ckpt_path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start_sec", type=float, default=0.0)
    parser.add_argument("--duration_sec", type=float)
    parser.add_argument("--max_frames", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--focal", type=float)
    parser.add_argument("--no_render", action="store_true")
    parser.add_argument("--no_postproc", action="store_true")
    parser.add_argument("--save_features", action="store_true")
    parser.add_argument("--mux_audio", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--feature_type", choices=("auto", "edge_baseline35"), default="auto")
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument(
        "--overwrite", action="store_true", help="Explicitly replace known output files"
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not args.audio.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {args.audio}")
    if args.audio.suffix.lower() not in {".wav", ".mp3", ".flac"}:
        raise ValueError("--audio must be a WAV, MP3, or FLAC file")
    if args.start_sec < 0:
        raise ValueError("--start_sec must be >= 0")
    if args.duration_sec is not None and args.duration_sec <= 0:
        raise ValueError("--duration_sec must be > 0")
    if args.max_frames <= 0 or args.num_samples <= 0:
        raise ValueError("--max_frames and --num_samples must be > 0")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be > 0")
    if args.focal is not None and args.focal <= 0:
        raise ValueError("--focal must be > 0")
    if args.guidance_scale < 0 or args.ddim_steps <= 0:
        raise ValueError("--guidance_scale must be >= 0 and --ddim_steps must be > 0")
    if not args.device.startswith("cuda"):
        raise RuntimeError(
            "Current GEM.predict() inference requires CUDA; --device must select a CUDA device"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; full GEM music diffusion inference requires CUDA")


def main(argv: list[str] | None = None) -> int:
    """Extract music features, run full GEM DDIM, save SMPL, and optionally render."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(args)
    if args.device != "cuda":
        torch.cuda.set_device(torch.device(args.device))

    ckpt_path = _resolve_checkpoint(args.ckpt_path)
    if not DEFAULT_BODY_MODEL.is_file():
        raise FileNotFoundError(f"Required SMPL-X body model does not exist: {DEFAULT_BODY_MODEL}")
    checkpoint_music_dim = validate_music_checkpoint(ckpt_path)
    print(f"[Audit] checkpoint music input dimension: {checkpoint_music_dim}")
    if checkpoint_music_dim != 35:
        raise RuntimeError(
            f"Checkpoint expects {checkpoint_music_dim} music features, not EDGE baseline35. "
            "No truncation, padding, PCA, or random projection will be applied."
        )
    feature_type = "edge_baseline35" if args.feature_type == "auto" else args.feature_type
    if feature_type != "edge_baseline35":
        raise RuntimeError(f"Unsupported music feature type: {feature_type}")

    features, feature_metadata = extract_edge_baseline35(
        args.audio,
        start_sec=args.start_sec,
        duration_sec=args.duration_sec,
    )
    length = int(features.shape[0])
    if length > args.max_frames:
        raise RuntimeError(
            f"Selected audio produced {length} frames, exceeding --max_frames={args.max_frames}. "
            "Use --start_sec/--duration_sec to select a shorter range. This demo does not "
            "silently truncate or stitch long diffusion generations."
        )
    data = build_music_only_data(
        features,
        width=args.width,
        height=args.height,
        focal=args.focal,
    )

    try:
        from demo_utils import load_model
    except ModuleNotFoundError:
        from scripts.demo.demo_utils import load_model

    print("[GEM] Loading full exp=gem_smpl PyTorch checkpoint with T5 disabled ...")
    model = load_model(str(ckpt_path), load_text_encoder=False)
    model_music_dim = inspect_model_music_input_dim(model)
    print(f"[Audit] model music input dimension: {model_music_dim}")
    if model_music_dim != checkpoint_music_dim or features.shape[1] != model_music_dim:
        raise RuntimeError(
            "Music dimension mismatch among checkpoint/model/features: "
            f"{checkpoint_music_dim}/{model_music_dim}/{features.shape[1]}; refusing to reshape"
        )
    denoiser3d = model.pipeline.denoiser3d
    if denoiser3d.regression_only:
        raise RuntimeError(_MUSIC_CHECKPOINT_ERROR)
    diff_cfg = denoiser3d.model_cfg.diffusion
    diff_cfg.guidance_param = args.guidance_scale
    diff_cfg.test_timestep_respacing = str(args.ddim_steps)
    diff_cfg.gen_only_test_timestep_respacing = str(args.ddim_steps)
    denoiser3d.init_diffusion()
    model.eval()
    print(
        f"[GEM] DDIM music generation enabled (not ONNX regression): frames={length}, "
        f"steps={args.ddim_steps}, CFG={args.guidance_scale}"
    )

    audio_root = args.output_root / args.audio.stem
    audio_root.mkdir(parents=True, exist_ok=True)
    if args.save_features:
        _torch_save(audio_root / "music_features.pt", features.float(), args.overwrite)
        _write_json(audio_root / "music_features_meta.json", feature_metadata, args.overwrite)
    run_config = {
        "audio_path": str(args.audio.resolve()),
        "checkpoint": str(ckpt_path.resolve()),
        "feature_type": feature_type,
        "music_input_dim": model_music_dim,
        "fps": EDGE_TARGET_FPS,
        "start_sec": args.start_sec,
        "requested_duration_sec": args.duration_sec,
        "selected_duration_sec": feature_metadata["selected_duration_sec"],
        "generated_frames": length,
        "generated_duration_sec": length / EDGE_TARGET_FPS,
        "seed": args.seed,
        "num_samples": args.num_samples,
        "width": args.width,
        "height": args.height,
        "focal": float(max(args.width, args.height) if args.focal is None else args.focal),
        "guidance_scale": args.guidance_scale,
        "ddim_steps": args.ddim_steps,
        "postproc": not args.no_postproc,
    }
    _write_json(audio_root / "run_config.json", run_config, args.overwrite)

    for sample_index in range(args.num_samples):
        sample_seed = args.seed + sample_index
        sample_dir = audio_root / f"sample_{sample_index:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        seed_everything(sample_seed)
        with torch.inference_mode():
            pred = model.predict(data, static_cam=True, postproc=not args.no_postproc)
        body_global = _validate_body_group(
            pred.get("body_params_global"), "body_params_global", length
        )
        body_incam = None
        if pred.get("body_params_incam") is not None:
            body_incam = _validate_body_group(
                pred["body_params_incam"], "body_params_incam", length
            )
        motion_151d = extract_motion_151d(pred, length)
        save_payload = {
            "body_params_global": _cpu_tree(body_global),
            "K_fullimg": data["K_fullimg"].cpu(),
            "fps": EDGE_TARGET_FPS,
            "audio_path": str(args.audio.resolve()),
            "start_sec": args.start_sec,
            "duration_sec": feature_metadata["selected_duration_sec"],
            "seed": sample_seed,
            "source": "music_only",
        }
        if body_incam is not None:
            save_payload["body_params_incam"] = _cpu_tree(body_incam)
        _torch_save(sample_dir / "smpl_params.pt", save_payload, args.overwrite)
        _torch_save(sample_dir / "motion_151d.pt", motion_151d, args.overwrite)
        print(f"[Output] sample_{sample_index:03d}: {sample_dir}")
        for field, value in body_global.items():
            print(f"  {field:<14}{list(value.shape)} finite={bool(torch.isfinite(value).all())}")

        rendered = None
        if not args.no_render:
            rendered = render_global_motion(
                sample_dir,
                _cpu_tree(body_global),
                args.width,
                args.height,
            )
        if args.mux_audio:
            if rendered is None:
                warnings.warn(
                    "--mux_audio requires a successful render; no muxed video was made",
                    stacklevel=2,
                )
            else:
                mux_selected_audio(
                    rendered,
                    args.audio,
                    sample_dir / "motion_with_audio.mp4",
                    args.start_sec,
                    length / EDGE_TARGET_FPS,
                    args.overwrite,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
