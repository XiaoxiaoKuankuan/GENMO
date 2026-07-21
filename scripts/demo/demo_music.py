#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Generate SMPL-X human motion from music with the full GEM DDIM model."""

from __future__ import annotations

import argparse
import gc
import json
import math
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

from gem.runtime.artifact_publish import (
    enforce_zero_shape,
    make_unique_output_paths,
    publish_ready_directory,
    safe_generation_prefix,
    utc_now_iso,
)
from gem.utils.music_features import (
    EDGE_BASELINE_FEATURE_NAMES,
    EDGE_FEATURE_DIM,
    EDGE_HOP_LENGTH,
    EDGE_SAMPLE_RATE,
    EDGE_TARGET_FPS,
    extract_edge_baseline35,
)

DEFAULT_CHECKPOINT = Path("inputs/pretrained/gem_smpl.ckpt")
DEFAULT_BODY_MODEL = Path("inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz")
DEFAULT_OUTPUT_ROOT = Path("outputs/music_motion")
SUPPORTED_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac"}

_MUSIC_CHECKPOINT_ERROR = (
    "The supplied checkpoint does not contain the music-conditioned diffusion "
    "weights required for music-to-motion generation. Use the official "
    "gem_smpl.ckpt or a checkpoint trained with exp=gem_smpl."
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
    """Build GEM's unified input with music as the only enabled condition."""
    if not isinstance(music_features, torch.Tensor):
        raise TypeError("music_features must be a torch.Tensor")
    if music_features.ndim != 2 or music_features.shape[1] != EDGE_FEATURE_DIM:
        raise ValueError(
            f"music_features must have shape [L, {EDGE_FEATURE_DIM}]; "
            f"got {tuple(music_features.shape)}"
        )
    if music_features.shape[0] <= 0:
        raise ValueError("music_features must contain at least one frame")
    if not torch.isfinite(music_features).all():
        raise ValueError("music_features contains NaN or Inf")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be greater than zero")
    resolved_focal = float(max(width, height) if focal is None else focal)
    if not math.isfinite(resolved_focal) or resolved_focal <= 0:
        raise ValueError("focal must be finite and greater than zero")

    from gem.utils.geo_transform import compute_cam_angvel

    length = int(music_features.shape[0])
    intrinsics = torch.tensor(
        (
            (resolved_focal, 0.0, width / 2.0),
            (0.0, resolved_focal, height / 2.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=torch.float32,
    )
    K_fullimg = intrinsics.unsqueeze(0).repeat(length, 1, 1)
    R_w2c = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(length, 1, 1)
    velocity_rotations = R_w2c
    if length == 1:
        velocity_rotations = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)
    cam_angvel = compute_cam_angvel(velocity_rotations, padding_last=True)[:length].float()
    bbox = torch.tensor(
        (width / 2.0, height / 2.0, 0.8 * min(width, height)), dtype=torch.float32
    )

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
        if not any("music_embedder" in str(key) for key in state_dict):
            raise RuntimeError(_MUSIC_CHECKPOINT_ERROR)
        candidates = [
            value
            for key, value in state_dict.items()
            if str(key).endswith("music_embedder.fc1.weight")
        ]
        if not candidates:
            raise RuntimeError(f"{_MUSIC_CHECKPOINT_ERROR} Missing first music Linear weight.")
        weight = candidates[0]
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise RuntimeError("Checkpoint music_embedder.fc1.weight is not a 2D tensor")
        input_dim = int(weight.shape[1])
        if input_dim != EDGE_FEATURE_DIM:
            raise RuntimeError(
                f"Checkpoint music input dimension is {input_dim}, but EDGE baseline35 is "
                f"{EDGE_FEATURE_DIM}. No truncation, padding, PCA, or projection is allowed."
            )
        return input_dim
    finally:
        del checkpoint
        gc.collect()


def inspect_model_music_input_dim(model: Any) -> int:
    """Validate full model music/DDIM support and return music input dimension."""
    if "encoded_music" not in model.pipeline.args.in_attr:
        raise RuntimeError(_MUSIC_CHECKPOINT_ERROR)
    if not hasattr(model, "music_embedder"):
        raise RuntimeError(_MUSIC_CHECKPOINT_ERROR)
    first_linear = next(
        (module for module in model.music_embedder.modules() if isinstance(module, nn.Linear)),
        None,
    )
    if first_linear is None:
        raise RuntimeError("model.music_embedder contains no nn.Linear layer")
    input_dim = int(first_linear.in_features)
    if input_dim != EDGE_FEATURE_DIM:
        raise RuntimeError(
            f"Loaded model expects {input_dim} music features, but EDGE baseline35 provides "
            f"{EDGE_FEATURE_DIM}; refusing to reshape the input."
        )
    if bool(model.pipeline.denoiser3d.regression_only):
        raise RuntimeError(_MUSIC_CHECKPOINT_ERROR)
    return input_dim


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
    """Recursively detach tensors to CPU for artifact serialization."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(child) for child in value)
    return value


def _validate_body_group(group: Any, name: str, length: int) -> dict[str, torch.Tensor]:
    """Validate one canonical SMPL-X parameter group without changing values."""
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
            warnings.warn(f"{name}.betas is absent; using zero betas", stacklevel=2)
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
    """Extract the final diffusion ``model_output['pred_x']`` diagnostic tensor."""
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
            f"Final pred_x must have shape [1, {length}, 151]; "
            f"got {getattr(pred_x, 'shape', None)}"
        )
    if not torch.isfinite(pred_x).all():
        raise RuntimeError("GEM final pred_x contains NaN or Inf")
    return pred_x[0].detach().cpu().float()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def music_generation_prefix(audio_path: Path, start_sec: float, seed: int) -> str:
    """Build the stable part of a unique direct-child generation name."""
    stem = safe_generation_prefix(audio_path.stem, fallback="music", limit=48)
    start = f"{start_sec:.3f}".replace("-", "m").replace(".", "p")
    return f"{stem}_start{start}_seed{seed}"


def build_music_metadata(
    *,
    args: argparse.Namespace,
    audio_path: Path,
    checkpoint: Path,
    feature_metadata: dict[str, Any],
    sample_seed: int,
    sample_index: int,
    num_frames: int,
    render_succeeded: bool,
    audio_mux_succeeded: bool,
    completed_at: str,
) -> dict[str, Any]:
    """Build the auditable JSON/top-level metadata for one generated sample."""
    selected_duration = float(feature_metadata["selected_duration_sec"])
    focal = float(max(args.width, args.height) if args.focal is None else args.focal)
    return {
        "source": "music_only",
        "shape_mode": "zero",
        "audio_path": str(audio_path.resolve()),
        "audio_name": audio_path.name,
        "audio_start_sec": float(args.start_sec),
        "audio_duration_sec": selected_duration,
        "original_audio_duration_sec": float(feature_metadata["original_duration_sec"]),
        "feature_type": "edge_baseline35",
        "feature_fps": EDGE_TARGET_FPS,
        "feature_dim": EDGE_FEATURE_DIM,
        "feature_names": list(EDGE_BASELINE_FEATURE_NAMES),
        "feature_frames": int(num_frames),
        "sample_rate": EDGE_SAMPLE_RATE,
        "hop_length": EDGE_HOP_LENGTH,
        "estimated_bpm": float(feature_metadata["estimated_or_prior_bpm"]),
        "bpm_source": str(feature_metadata["bpm_source"]),
        "seed": int(sample_seed),
        "sample_index": int(sample_index),
        "num_frames": int(num_frames),
        "generated_duration_sec": float(num_frames / EDGE_TARGET_FPS),
        "guidance_scale": float(args.guidance_scale),
        "ddim_steps": int(args.ddim_steps),
        "checkpoint": str(checkpoint.resolve()),
        "width": int(args.width),
        "height": int(args.height),
        "focal": focal,
        "postproc": not bool(args.no_postproc),
        "render_succeeded": bool(render_succeeded),
        "audio_mux_succeeded": bool(audio_mux_succeeded),
        "completed_at": completed_at,
    }


def write_music_artifacts(
    output_dir: Path,
    *,
    body_global: dict[str, torch.Tensor],
    body_incam: dict[str, torch.Tensor],
    raw_motion_151d: torch.Tensor,
    music_features: torch.Tensor,
    data: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    """Write every required non-READY file for one music generation."""
    output_dir.mkdir(parents=False, exist_ok=False)
    groups = enforce_zero_shape(
        {
            "body_params_global": _cpu_tree(body_global),
            "body_params_incam": _cpu_tree(body_incam),
        }
    )
    body_global_cpu = groups["body_params_global"]
    body_incam_cpu = groups["body_params_incam"]
    for name, group in groups.items():
        if torch.count_nonzero(group["betas"]).item() != 0:
            raise AssertionError(f"{name}.betas must be zero before saving")
    features_cpu = music_features.detach().cpu().float()
    if tuple(features_cpu.shape) != (metadata["num_frames"], EDGE_FEATURE_DIM):
        raise RuntimeError(
            f"music_features shape {tuple(features_cpu.shape)} does not match metadata"
        )
    if not torch.isfinite(features_cpu).all():
        raise RuntimeError("music_features contains NaN or Inf before saving")

    payload = {
        "body_params_global": body_global_cpu,
        "body_params_incam": body_incam_cpu,
        "K_fullimg": data["K_fullimg"].detach().cpu(),
        "bbx_xys": data["bbx_xys"].detach().cpu(),
        "fps": float(EDGE_TARGET_FPS),
        "num_frames": int(metadata["num_frames"]),
        "duration_sec": float(metadata["generated_duration_sec"]),
        "source": "music_only",
        "shape_mode": "zero",
        "audio_path": metadata["audio_path"],
        "audio_start_sec": metadata["audio_start_sec"],
        "audio_duration_sec": metadata["audio_duration_sec"],
        "feature_type": metadata["feature_type"],
        "feature_fps": metadata["feature_fps"],
        "feature_dim": metadata["feature_dim"],
        "estimated_bpm": metadata["estimated_bpm"],
        "bpm_source": metadata["bpm_source"],
        "seed": metadata["seed"],
        "sample_index": metadata["sample_index"],
        "guidance_scale": metadata["guidance_scale"],
        "ddim_steps": metadata["ddim_steps"],
        "checkpoint": metadata["checkpoint"],
        "metadata": dict(metadata),
    }
    torch.save(payload, output_dir / "smpl_params.pt")
    np.savez(
        output_dir / "motion.npz",
        body_pose=body_global_cpu["body_pose"].numpy(),
        global_orient=body_global_cpu["global_orient"].numpy(),
        transl=body_global_cpu["transl"].numpy(),
        betas=body_global_cpu["betas"].numpy(),
        fps=np.asarray(EDGE_TARGET_FPS, dtype=np.float32),
    )
    torch.save(raw_motion_151d.detach().cpu().float(), output_dir / "raw_motion_151d.pt")
    torch.save(features_cpu, output_dir / "music_features.pt")
    _write_json(output_dir / "metadata.json", metadata)
    (output_dir / "source_audio.txt").write_text(
        f"audio_path={metadata['audio_path']}\n"
        f"start_sec={metadata['audio_start_sec']:.9f}\n"
        f"duration_sec={metadata['audio_duration_sec']:.9f}\n",
        encoding="utf-8",
    )


def update_saved_metadata(output_dir: Path, metadata: dict[str, Any]) -> None:
    """Update completion/render fields before the hidden directory is published."""
    smpl_path = output_dir / "smpl_params.pt"
    payload = torch.load(smpl_path, map_location="cpu", weights_only=False)
    payload["metadata"] = dict(metadata)
    torch.save(payload, smpl_path)
    _write_json(output_dir / "metadata.json", metadata)


def render_global_motion(
    sample_dir: Path,
    body_params_global: dict[str, torch.Tensor],
    width: int,
    height: int,
) -> Path | None:
    """Render zero-shape global motion; return ``None`` on optional failures."""
    if torch.count_nonzero(body_params_global["betas"]).item() != 0:
        raise RuntimeError("rendering received non-zero betas under shape_mode=zero")
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
) -> bool:
    """Best-effort mux of the selected source range into the rendered video."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        warnings.warn(
            "ffmpeg is unavailable; motion generation succeeded but audio was not muxed",
            stacklevel=2,
        )
        return False
    command = [ffmpeg, "-y", "-i", str(video_path)]
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
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        warnings.warn(f"ffmpeg launch failed; motion remains valid: {exc}", stacklevel=2)
        return False
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
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument(
        "--shape_mode",
        choices=("zero",),
        default="zero",
        help="Use the neutral zero-beta SMPL-X shape for rendering and robot FK.",
    )
    parser.add_argument("--feature_type", choices=("auto", "edge_baseline35"), default="auto")
    parser.add_argument(
        "--save_features",
        action="store_true",
        help="Compatibility flag; READY generations always include music_features.pt.",
    )
    parser.add_argument("--no_render", action="store_true")
    parser.add_argument("--no_postproc", action="store_true")
    parser.add_argument("--mux_audio", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Compatibility flag; unique generation directories never replace older output.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    return parser


def validate_arguments(args: argparse.Namespace, *, require_runtime: bool | None = None) -> None:
    """Validate CLI values, requiring CUDA only for actual GEM inference."""
    if not args.audio.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {args.audio}")
    if args.audio.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
        raise ValueError("--audio must be a WAV, MP3, or FLAC file")
    if not math.isfinite(args.start_sec) or args.start_sec < 0:
        raise ValueError("--start_sec must be finite and >= 0")
    if args.duration_sec is not None and (
        not math.isfinite(args.duration_sec) or args.duration_sec <= 0
    ):
        raise ValueError("--duration_sec must be finite and > 0")
    if args.max_frames <= 0:
        raise ValueError("--max_frames must be > 0")
    if args.num_samples <= 0:
        raise ValueError("--num_samples must be > 0")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be > 0")
    if args.focal is not None and (not math.isfinite(args.focal) or args.focal <= 0):
        raise ValueError("--focal must be finite and > 0")
    if not math.isfinite(args.guidance_scale) or args.guidance_scale < 0:
        raise ValueError("--guidance_scale must be finite and >= 0")
    if args.ddim_steps <= 0:
        raise ValueError("--ddim_steps must be > 0")
    runtime_required = not args.dry_run if require_runtime is None else require_runtime
    if runtime_required:
        if not args.device.startswith("cuda"):
            raise RuntimeError(
                "Current GEM.predict() inference requires CUDA; --device must select CUDA"
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable; full GEM music diffusion inference requires CUDA. "
                "Use --dry_run to validate real music features on CPU."
            )


def _print_dry_run(
    audio_path: Path,
    features: torch.Tensor,
    feature_metadata: dict[str, Any],
    data: dict[str, Any],
) -> None:
    """Print the real feature and music-only input contract without loading GEM."""
    length = int(features.shape[0])
    masks = data["mask"]
    print("============================================================")
    print("GEM music-to-motion dry run (no checkpoint/GEM/CUDA/render)")
    print(f"Audio path:                 {audio_path.resolve()}")
    print(f"Original duration:          {feature_metadata['original_duration_sec']:.6f}s")
    print(f"Selected start:             {feature_metadata['selected_start_sec']:.6f}s")
    print(f"Selected duration:          {feature_metadata['selected_duration_sec']:.6f}s")
    print(
        f"Estimated BPM:              {feature_metadata['estimated_or_prior_bpm']:.6f} "
        f"({feature_metadata['bpm_source']})"
    )
    print(f"Feature FPS:                {EDGE_TARGET_FPS}")
    print(f"Feature shape:              {tuple(features.shape)}")
    print(f"Feature finite:             {bool(torch.isfinite(features).all())}")
    print(f"Onset peak count:           {int(torch.count_nonzero(features[:, 33]))}")
    print(f"Beat peak count:            {int(torch.count_nonzero(features[:, 34]))}")
    print(f"Expected generated duration:{length / EDGE_TARGET_FPS:.6f}s")
    print(f"music_embed:       {tuple(data['music_embed'].shape)}")
    print(f"kp2d:              {tuple(data['kp2d'].shape)}")
    print(f"f_imgseq:          {tuple(data['f_imgseq'].shape)}")
    print(f"K_fullimg:         {tuple(data['K_fullimg'].shape)}")
    print(f"cam_angvel:        {tuple(data['cam_angvel'].shape)}")
    print(f"has_music:         {int(masks['has_music_mask'].sum())} / {length}")
    print(f"has_img:           {int(masks['has_img_mask'].sum())} / {length}")
    print(f"has_2d:            {int(masks['has_2d_mask'].sum())} / {length}")
    print(f"has_cam:           {int(masks['has_cam_mask'].sum())} / {length}")
    print(f"has_audio:         {int(masks['has_audio_mask'].sum())} / {length}")
    print(f"GEM music mask count:       {int(masks['has_music_mask'].sum())}")
    print("Other condition mask counts: img=0 2d=0 cam=0 audio=0 text=0")


def main(argv: list[str] | None = None) -> int:
    """Extract EDGE features, run full GEM DDIM, and atomically publish samples."""
    args = build_parser().parse_args(argv)
    validate_arguments(args)

    feature_type = "edge_baseline35" if args.feature_type == "auto" else args.feature_type
    if feature_type != "edge_baseline35":
        raise RuntimeError(f"Unsupported music feature type: {feature_type}")
    features, feature_metadata = extract_edge_baseline35(
        args.audio,
        start_sec=args.start_sec,
        duration_sec=args.duration_sec,
        target_fps=EDGE_TARGET_FPS,
    )
    length = int(features.shape[0])
    if length > args.max_frames:
        raise RuntimeError(
            f"Selected audio produced {length} frames, exceeding --max_frames={args.max_frames}. "
            "Use --start_sec/--duration_sec to select a shorter range. This demo does not "
            "silently truncate or stitch long diffusion generations."
        )
    data = build_music_only_data(features, width=args.width, height=args.height, focal=args.focal)
    if args.dry_run:
        _print_dry_run(args.audio, features, feature_metadata, data)
        return 0

    if args.device != "cuda":
        torch.cuda.set_device(torch.device(args.device))
    ckpt_path = _resolve_checkpoint(args.ckpt_path)
    if not DEFAULT_BODY_MODEL.is_file():
        raise FileNotFoundError(f"Required SMPL-X body model does not exist: {DEFAULT_BODY_MODEL}")
    checkpoint_music_dim = validate_music_checkpoint(ckpt_path)
    print(f"[Audit] checkpoint music input dimension: {checkpoint_music_dim}")

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
    diff_cfg = denoiser3d.model_cfg.diffusion
    diff_cfg.guidance_param = args.guidance_scale
    diff_cfg.test_timestep_respacing = str(args.ddim_steps)
    diff_cfg.gen_only_test_timestep_respacing = str(args.ddim_steps)
    denoiser3d.init_diffusion()
    model.eval()
    print(
        f"[GEM] Full DDIM/CFG music generation (not ONNX regression): frames={length}, "
        f"steps={args.ddim_steps}, CFG={args.guidance_scale}"
    )

    for sample_index in range(args.num_samples):
        sample_seed = args.seed + sample_index
        seed_everything(sample_seed)
        with torch.inference_mode():
            pred = model.predict(data, static_cam=True, postproc=not args.no_postproc)
        body_global = _validate_body_group(
            pred.get("body_params_global"), "body_params_global", length
        )
        body_incam = _validate_body_group(
            pred.get("body_params_incam"), "body_params_incam", length
        )
        groups = enforce_zero_shape(
            {"body_params_global": body_global, "body_params_incam": body_incam}
        )
        body_global = groups["body_params_global"]
        body_incam = groups["body_params_incam"]
        raw_motion = extract_motion_151d(pred, length)
        temporary_dir, output_dir = make_unique_output_paths(
            args.output_root,
            music_generation_prefix(args.audio, args.start_sec, sample_seed),
        )
        provisional_completed_at = utc_now_iso()
        metadata = build_music_metadata(
            args=args,
            audio_path=args.audio,
            checkpoint=ckpt_path,
            feature_metadata=feature_metadata,
            sample_seed=sample_seed,
            sample_index=sample_index,
            num_frames=length,
            render_succeeded=False,
            audio_mux_succeeded=False,
            completed_at=provisional_completed_at,
        )
        try:
            write_music_artifacts(
                temporary_dir,
                body_global=body_global,
                body_incam=body_incam,
                raw_motion_151d=raw_motion,
                music_features=features,
                data=data,
                metadata=metadata,
            )
            rendered: Path | None = None
            if not args.no_render:
                rendered = render_global_motion(
                    temporary_dir, _cpu_tree(body_global), args.width, args.height
                )
            mux_succeeded = False
            if args.mux_audio:
                if rendered is None:
                    warnings.warn(
                        "--mux_audio requires a successful render; no muxed video was made",
                        stacklevel=2,
                    )
                else:
                    mux_succeeded = mux_selected_audio(
                        rendered,
                        args.audio,
                        temporary_dir / "motion_with_audio.mp4",
                        args.start_sec,
                        float(feature_metadata["selected_duration_sec"]),
                    )
            completed_at = utc_now_iso()
            metadata["render_succeeded"] = rendered is not None
            metadata["audio_mux_succeeded"] = mux_succeeded
            metadata["completed_at"] = completed_at
            update_saved_metadata(temporary_dir, metadata)
            publish_ready_directory(temporary_dir, output_dir, completed_at)
        except Exception:
            shutil.rmtree(temporary_dir, ignore_errors=True)
            raise

        print(f"[Shape] sample={sample_index} global/incam betas norm=0.000000")
        print(f"[Output] Published complete music motion with READY: {output_dir}")
        for field, value in body_global.items():
            print(f"  {field:<14}{list(value.shape)} finite={bool(torch.isfinite(value).all())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
