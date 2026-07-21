# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Generate a SMPL motion directly from a single text prompt with GEM.

This entry point deliberately does not import or run the video preprocessing
stack.  It encodes text first, releases T5-3B, then loads the full GEM-SMPL
diffusion checkpoint so that the two large models do not occupy GPU memory at
the same time.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import re
import shutil
import warnings
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gem.runtime.artifact_publish import (
    enforce_zero_shape,
    make_unique_output_paths,
    publish_ready_directory,
    utc_now_iso,
)

DEFAULT_CHECKPOINT = Path("inputs/pretrained/gem_smpl.ckpt")
DEFAULT_OUTPUT_ROOT = Path("outputs/text_motion")
DEFAULT_T5_MODEL = "t5-3b"
MAX_TEXT_LEN = 50
TEXT_EMBED_DIM = 1024

_TEXT_CHECKPOINT_ERROR = (
    "The supplied checkpoint does not contain the text-conditioned diffusion "
    "weights required for text-to-motion generation. Use the official "
    "gem_smpl.ckpt or a checkpoint trained with exp=gem_smpl. A checkpoint "
    "trained with exp=gem_smpl_regression cannot generate motion from text."
)


def build_text_only_data(
    prompt: str,
    text_embed: torch.Tensor,
    num_frames: int,
    width: int,
    height: int,
    bbox_scale: float,
) -> dict[str, Any]:
    """Build GEM input tensors for text-only generation and a static camera.

    The camera tensors only define the coordinate system used to decode the
    generated local motion.  All non-text condition masks are false, so none
    of these synthetic tensors condition the diffusion denoiser.
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be greater than 0")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must both be greater than 0")
    if not 0.0 < bbox_scale <= 1.5:
        raise ValueError("bbox_scale must satisfy 0 < bbox_scale <= 1.5")
    if not isinstance(text_embed, torch.Tensor):
        raise TypeError("text_embed must be a torch.Tensor")
    if tuple(text_embed.shape) != (MAX_TEXT_LEN, TEXT_EMBED_DIM):
        raise ValueError(
            f"text_embed must have shape ({MAX_TEXT_LEN}, {TEXT_EMBED_DIM}); "
            f"got {tuple(text_embed.shape)}"
        )

    from gem.utils.cam_utils import estimate_K
    from gem.utils.geo_transform import compute_cam_angvel

    length = num_frames
    K = estimate_K(width, height).float()
    K_fullimg = K.unsqueeze(0).repeat(length, 1, 1)
    R_w2c = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(length, 1, 1)

    # compute_cam_angvel computes consecutive rotations.  Give it two static
    # frames for the valid L=1 case, then retain the requested sequence length.
    R_for_velocity = R_w2c
    if length == 1:
        R_for_velocity = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)
    cam_angvel = compute_cam_angvel(R_for_velocity, padding_last=True)[:length].float()
    cam_tvel = torch.zeros(length, 3, dtype=torch.float32)

    bbox_size = min(width, height) * bbox_scale
    bbx_one = torch.tensor([width / 2.0, height / 2.0, bbox_size], dtype=torch.float32)
    bbx_xys = bbx_one.unsqueeze(0).repeat(length, 1)

    kp2d = torch.zeros(length, 17, 3, dtype=torch.float32)
    f_imgseq = torch.zeros(length, 1024, dtype=torch.float32)

    def false_mask() -> torch.Tensor:
        return torch.zeros(length, dtype=torch.bool)

    return {
        "kp2d": kp2d,
        "bbx_xys": bbx_xys,
        "K_fullimg": K_fullimg,
        "cam_angvel": cam_angvel,
        "cam_tvel": cam_tvel,
        "R_w2c": R_w2c,
        "f_imgseq": f_imgseq,
        "caption": prompt,
        "has_text": torch.tensor([True], dtype=torch.bool),
        "text_embed": text_embed.detach().float().cpu(),
        "length": torch.tensor(length, dtype=torch.long),
        "mask": {
            "has_img_mask": false_mask(),
            "has_2d_mask": false_mask(),
            "has_cam_mask": false_mask(),
            "has_audio_mask": false_mask(),
            "has_music_mask": false_mask(),
        },
        "meta": [
            {
                "mode": "default",
                "source": "text_only",
                "prompt": prompt,
            }
        ],
    }


def _resolve_text_encoder_settings(device: str, dtype: str) -> tuple[str, torch.dtype, str]:
    """Resolve CLI text-encoder settings to a device and torch dtype."""
    resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if resolved_device == "auto":
        resolved_device = "cpu"
    if resolved_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--text_encoder_device cuda was requested, but CUDA is not available. "
            "Use --text_encoder_device cpu or install a CUDA-enabled PyTorch build."
        )

    resolved_dtype = dtype
    if dtype == "auto":
        resolved_dtype = "float16" if resolved_device == "cuda" else "float32"
    torch_dtype = torch.float16 if resolved_dtype == "float16" else torch.float32
    return resolved_device, torch_dtype, resolved_dtype


def _load_t5_components_cached_first(
    model_name_or_path: str,
    torch_dtype: torch.dtype,
    local_files_only: bool,
    tokenizer_class: Any,
    encoder_class: Any,
) -> tuple[Any, Any, str]:
    """Load a complete local T5 cache before attempting network access.

    Transformers may perform a Hub metadata request even when all required
    files are cached. Trying strict local mode first makes repeat inference
    independent of transient or malformed proxy settings without accepting a
    partial cache or replacing text features with zeros.
    """
    attempts = (True,) if local_files_only else (True, False)
    failures: list[tuple[str, Exception]] = []
    for use_local_cache in attempts:
        tokenizer = None
        text_encoder = None
        source = "local cache/path" if use_local_cache else "Hugging Face Hub"
        try:
            tokenizer = tokenizer_class.from_pretrained(
                model_name_or_path,
                local_files_only=use_local_cache,
            )
            text_encoder = encoder_class.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                local_files_only=use_local_cache,
            )
            return tokenizer, text_encoder, source
        except Exception as exc:
            failures.append((source, exc))
            if text_encoder is not None:
                del text_encoder
            if tokenizer is not None:
                del tokenizer
            gc.collect()

    details = "; ".join(f"{source} failed: {type(exc).__name__}: {exc}" for source, exc in failures)
    proxy_hint = ""
    if "Unknown scheme for proxy URL" in details or "socks://" in details:
        proxy_hint = (
            " The configured socks:// proxy scheme is not supported by this httpx "
            "installation. For a local mixed HTTP proxy, use "
            "HTTP_PROXY=http://127.0.0.1:7897 and "
            "HTTPS_PROXY=http://127.0.0.1:7897, and unset ALL_PROXY/all_proxy. "
            "For a real SOCKS proxy, install SOCKS support and use socks5://."
        )
    attempted_sources = (
        "the local cache/path" if local_files_only else "the local cache/path or Hugging Face Hub"
    )
    raise RuntimeError(
        f"T5 files could not be loaded from {attempted_sources}. {details}.{proxy_hint}"
    )


def encode_prompt_t5(
    prompt: str,
    model_name_or_path: str,
    max_text_len: int,
    device: str,
    dtype: str,
    local_files_only: bool,
) -> torch.Tensor:
    """Encode one prompt exactly as ``GEM.encode_text`` and release T5."""
    resolved_device, torch_dtype, _ = _resolve_text_encoder_settings(device, dtype)
    tokenizer = None
    text_encoder = None
    try:
        from transformers import T5EncoderModel, T5Tokenizer

        tokenizer, text_encoder, load_source = _load_t5_components_cached_first(
            model_name_or_path,
            torch_dtype,
            local_files_only,
            T5Tokenizer,
            T5EncoderModel,
        )
        print(f"[Text] T5 loaded from {load_source}")
        text_encoder = text_encoder.to(resolved_device).eval()

        tokenized = tokenizer(
            [prompt],
            return_tensors="pt",
            padding="max_length",
            max_length=max_text_len,
            truncation=True,
        )
        input_ids = tokenized.input_ids.to(resolved_device)
        attention_mask = tokenized.attention_mask.to(resolved_device)
        with torch.inference_mode():
            output = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoded_text = output.last_hidden_state
        encoded_text = encoded_text[:, :max_text_len]
        encoded_text = encoded_text * attention_mask[:, :max_text_len].unsqueeze(-1)
        expected = (1, MAX_TEXT_LEN, TEXT_EMBED_DIM)
        if tuple(encoded_text.shape) != expected:
            raise RuntimeError(
                f"T5 text embedding has shape {tuple(encoded_text.shape)}, expected {expected}. "
                "Use the T5-3B encoder expected by GEM-SMPL."
            )
        text_embed = encoded_text[0].detach().float().cpu()
        assert tuple(text_embed.shape) == (MAX_TEXT_LEN, TEXT_EMBED_DIM)
        return text_embed
    except Exception as exc:
        if isinstance(exc, RuntimeError) and "T5 text embedding has shape" in str(exc):
            raise
        locality = " in the local cache/path" if local_files_only else " locally or online"
        raise RuntimeError(
            f"Unable to load or run T5 text encoder '{model_name_or_path}'{locality}. "
            "Provide --t5_model /path/to/t5-3b, check network access, or pre-download "
            "T5-3B. Text generation cannot use an all-zero embedding. "
            f"Original error: {exc}"
        ) from exc
    finally:
        if text_encoder is not None:
            del text_encoder
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch without forcing slow deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prompt_slug(prompt: str, max_length: int = 48) -> str:
    """Return a filesystem-safe, lowercase ASCII slug for a prompt."""
    slug = re.sub(r"[^a-z0-9_-]+", "_", prompt.lower())
    slug = slug.strip("_-")[:max_length].rstrip("_-")
    return slug or "text_motion"


def validate_text_generation_checkpoint(ckpt_path: str | Path) -> None:
    """Reject checkpoints that do not contain text-conditioned diffusion weights."""
    path = Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(f"GEM checkpoint does not exist: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"Unable to read GEM checkpoint '{path}': {exc}") from exc

    try:
        state_dict = checkpoint.get("state_dict", checkpoint)
        if not isinstance(state_dict, dict):
            raise RuntimeError(f"Checkpoint '{path}' does not contain a valid state_dict")
        keys = tuple(str(key) for key in state_dict)
        required_markers = ("embed_text", "text_encoder_layers", "gate_cross_attn")
        if not all(any(marker in key for key in keys) for marker in required_markers):
            raise RuntimeError(_TEXT_CHECKPOINT_ERROR)
    finally:
        del checkpoint
        gc.collect()


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate numerical CLI arguments with explicit error messages."""
    if args.num_frames <= 0:
        raise ValueError("--num_frames must be greater than 0")
    if args.fps <= 0:
        raise ValueError("--fps must be greater than 0")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must both be greater than 0")
    if not 0.0 < args.bbox_scale <= 1.5:
        raise ValueError("--bbox_scale must satisfy 0 < value <= 1.5")
    if args.ddim_steps <= 0:
        raise ValueError("--ddim_steps must be greater than 0")
    if args.guidance_scale < 0:
        raise ValueError("--guidance_scale must be greater than or equal to 0")


def resolve_prompt(prompt: str | None, prompt_file: Path | None) -> str:
    """Resolve and validate the mutually exclusive inline/file prompt input."""
    if (prompt is None) == (prompt_file is None):
        raise ValueError("Provide exactly one of --prompt or --prompt_file")
    if prompt_file is not None:
        try:
            prompt = prompt_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ValueError(f"Unable to read --prompt_file '{prompt_file}': {exc}") from exc
    else:
        prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("The text prompt must not be empty")
    return prompt


def _print_dry_run(data: dict[str, Any]) -> None:
    """Print the text-only input contract without loading T5 or GEM."""
    print("[Dry run] Text-only GEM input")
    for key in ("kp2d", "bbx_xys", "K_fullimg", "cam_angvel", "f_imgseq", "text_embed"):
        value = data[key]
        print(f"{key + ':':<16}{tuple(value.shape)} dtype={value.dtype}")
    label_map = {
        "has_img_mask": "has_img",
        "has_2d_mask": "has_2d",
        "has_cam_mask": "has_cam",
        "has_audio_mask": "has_audio",
        "has_music_mask": "has_music",
    }
    length = int(data["length"])
    for key, label in label_map.items():
        count = int(data["mask"][key].sum())
        print(f"{label + ':':<16}{count} / {length}")
    print("Video/YOLO/ByteTrack/ViTPose/HMR2: not loaded or used")


def _download_or_resolve_checkpoint(ckpt_path: Path) -> Path:
    """Return an existing checkpoint, downloading the official model if absent."""
    if ckpt_path.is_file():
        return ckpt_path
    from gem.utils.hf_utils import download_checkpoint

    print(f"[Checkpoint] '{ckpt_path}' was not found; downloading official gem_smpl.ckpt ...")
    downloaded = Path(download_checkpoint())
    if not downloaded.is_file():
        raise FileNotFoundError(f"Downloaded checkpoint was not found at: {downloaded}")
    return downloaded


def _assert_finite_tree(value: Any, field_name: str = "pred") -> None:
    """Raise with the precise field path when an inference tensor is non-finite."""
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise RuntimeError(f"GEM output contains NaN or Inf in field '{field_name}'")
    elif isinstance(value, dict):
        for key, child in value.items():
            _assert_finite_tree(child, f"{field_name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_finite_tree(child, f"{field_name}[{index}]")


def validate_smpl_prediction(
    pred: dict[str, Any], num_frames: int
) -> dict[str, dict[str, torch.Tensor]]:
    """Validate and return canonical global/in-camera SMPL parameter dictionaries."""
    _assert_finite_tree(pred)
    expected_shapes = {
        "body_pose": (num_frames, 63),
        "global_orient": (num_frames, 3),
        "transl": (num_frames, 3),
        "betas": (num_frames, 10),
    }
    validated: dict[str, dict[str, torch.Tensor]] = {}
    for group_name in ("body_params_global", "body_params_incam"):
        group = pred.get(group_name)
        if not isinstance(group, dict):
            raise RuntimeError(f"GEM prediction is missing '{group_name}'")
        canonical: dict[str, torch.Tensor] = {}
        for field, expected_shape in expected_shapes.items():
            value = group.get(field)
            if value is None and field == "betas":
                warnings.warn(
                    f"Checkpoint output '{group_name}.betas' is missing; using zero SMPL betas.",
                    stacklevel=2,
                )
                value = torch.zeros(expected_shape, dtype=torch.float32)
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(f"GEM prediction is missing tensor '{group_name}.{field}'")
            if tuple(value.shape) != expected_shape:
                raise RuntimeError(
                    f"Unexpected shape for '{group_name}.{field}': got {tuple(value.shape)}, "
                    f"expected {expected_shape}"
                )
            if not torch.isfinite(value).all():
                raise RuntimeError(
                    f"GEM output contains NaN or Inf in field '{group_name}.{field}'"
                )
            canonical[field] = value
        validated[group_name] = canonical
    return validated


def _to_cpu(value: Any) -> Any:
    """Recursively detach tensors to CPU for serialization."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_to_cpu(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(child) for child in value)
    return value


def unique_output_paths(output_root: Path, prompt: str, seed: int) -> tuple[Path, Path]:
    """Return same-filesystem temporary and unique final generation directories."""
    return make_unique_output_paths(output_root, f"{prompt_slug(prompt)}_seed{seed}")


def save_results(
    output_dir: Path,
    body_params: dict[str, dict[str, torch.Tensor]],
    data: dict[str, Any],
    prompt: str,
    args: argparse.Namespace,
    ckpt_path: Path,
    text_encoder_device: str,
    text_encoder_dtype: str,
    completed_at: str,
) -> None:
    """Save CPU SMPL parameters, NumPy motion arrays, prompt, and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    body_global = _to_cpu(body_params["body_params_global"])
    body_incam = _to_cpu(body_params["body_params_incam"])
    save_dict = {
        "body_params_global": body_global,
        "body_params_incam": body_incam,
        "K_fullimg": data["K_fullimg"].detach().cpu(),
        "bbx_xys": data["bbx_xys"].detach().cpu(),
        "prompt": prompt,
        "fps": args.fps,
        "seed": args.seed,
        "num_frames": args.num_frames,
        "guidance_scale": args.guidance_scale,
        "ddim_steps": args.ddim_steps,
        "checkpoint": str(ckpt_path),
        "source": "text_only",
        "shape_mode": args.shape_mode,
    }
    torch.save(save_dict, output_dir / "smpl_params.pt")
    np.savez(
        output_dir / "motion.npz",
        body_pose=body_global["body_pose"].numpy(),
        global_orient=body_global["global_orient"].numpy(),
        transl=body_global["transl"].numpy(),
        betas=body_global["betas"].numpy(),
        fps=np.asarray(args.fps, dtype=np.float32),
    )
    (output_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    metadata = {
        "prompt": prompt,
        "fps": args.fps,
        "seed": args.seed,
        "num_frames": args.num_frames,
        "duration_seconds": args.num_frames / args.fps,
        "width": args.width,
        "height": args.height,
        "bbox_scale": args.bbox_scale,
        "guidance_scale": args.guidance_scale,
        "ddim_steps": args.ddim_steps,
        "checkpoint": str(ckpt_path),
        "t5_model": args.t5_model,
        "text_encoder_device": text_encoder_device,
        "text_encoder_dtype": text_encoder_dtype,
        "source": "text_only",
        "shape_mode": args.shape_mode,
        "completed_at": completed_at,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def render_global_video(
    output_dir: Path,
    body_params_global: dict[str, torch.Tensor],
    width: int,
    height: int,
    fps: float,
) -> None:
    """Render only the generated global-coordinate motion to ``global.mp4``."""
    try:
        try:
            from demo_utils import normalize_global_verts, render_global_frames
        except ModuleNotFoundError:
            from scripts.demo.demo_utils import normalize_global_verts, render_global_frames
        from gem.utils.smplx_utils import make_smplx
        from gem.utils.video_io_utils import save_video

        body_model = make_smplx("supermotion")
        body_model.cuda().eval()
        smpl_faces = torch.from_numpy(body_model.faces.astype(np.int32)).long()
        verts_global = normalize_global_verts(body_model, body_params_global)
        global_frames = render_global_frames(verts_global, smpl_faces, width, height)
        # PyAV expects a rational frame rate; passing a Python float creates an
        # empty container before failing with a missing ``numerator`` attribute.
        save_video(global_frames, str(output_dir / "global.mp4"), fps=Fraction(str(fps)))
        print(f"[Render] Saved {output_dir / 'global.mp4'}")
    except Exception as exc:
        warnings.warn(
            "SMPL parameters were saved, but global rendering is unavailable or failed. "
            f"Install/check Open3D and the SMPL rendering assets. Original error: {exc}",
            stacklevel=2,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for text-only generation."""
    parser = argparse.ArgumentParser(
        description="Generate a SMPL motion from text with full GEM diffusion (no video)."
    )
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", type=str, help="Text motion description")
    prompt_group.add_argument("--prompt_file", type=Path, help="UTF-8 text prompt file")
    parser.add_argument("--ckpt_path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--num_frames", type=int, default=300)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shape_mode",
        choices=("zero",),
        default="zero",
        help="SMPL-X body-shape policy; robot-compatible generation requires zero.",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--bbox_scale", type=float, default=0.75)
    parser.add_argument("--t5_model", type=str, default=DEFAULT_T5_MODEL)
    parser.add_argument("--text_encoder_device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--text_encoder_dtype", choices=("auto", "float16", "float32"), default="auto"
    )
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--no_postproc", action="store_true")
    parser.add_argument("--no_render", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run text encoding, full GEM diffusion inference, saving, and optional rendering."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        validate_arguments(args)
        prompt = resolve_prompt(args.prompt, args.prompt_file)
    except ValueError as exc:
        parser.error(str(exc))

    if args.dry_run:
        placeholder = torch.zeros(MAX_TEXT_LEN, TEXT_EMBED_DIM, dtype=torch.float32)
        data = build_text_only_data(
            prompt,
            placeholder,
            args.num_frames,
            args.width,
            args.height,
            args.bbox_scale,
        )
        _print_dry_run(data)
        return 0

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Full GEM text-to-motion inference requires CUDA, but CUDA is not available. "
            "Use --dry_run to validate inputs on a CPU-only machine."
        )

    ckpt_path = _download_or_resolve_checkpoint(args.ckpt_path)
    print(f"[Checkpoint] Validating text diffusion weights in {ckpt_path} ...")
    validate_text_generation_checkpoint(ckpt_path)

    text_device, _, text_dtype = _resolve_text_encoder_settings(
        args.text_encoder_device, args.text_encoder_dtype
    )
    seed_everything(args.seed)
    print(f"[Text] Encoding prompt with {args.t5_model} on {text_device} ({text_dtype}) ...")
    text_embed = encode_prompt_t5(
        prompt=prompt,
        model_name_or_path=args.t5_model,
        max_text_len=MAX_TEXT_LEN,
        device=args.text_encoder_device,
        dtype=args.text_encoder_dtype,
        local_files_only=args.local_files_only,
    )
    data = build_text_only_data(
        prompt,
        text_embed,
        args.num_frames,
        args.width,
        args.height,
        args.bbox_scale,
    )
    del text_embed
    gc.collect()
    torch.cuda.empty_cache()

    try:
        from demo_utils import load_model
    except ModuleNotFoundError:
        from scripts.demo.demo_utils import load_model

    print("[GEM] Loading full gem_smpl diffusion model (T5 remains unloaded) ...")
    model = load_model(str(ckpt_path), load_text_encoder=False)
    denoiser3d = model.pipeline.denoiser3d
    if denoiser3d.regression_only:
        raise RuntimeError(_TEXT_CHECKPOINT_ERROR)

    diff_cfg = denoiser3d.model_cfg.diffusion
    diff_cfg.guidance_param = args.guidance_scale
    diff_cfg.test_timestep_respacing = str(args.ddim_steps)
    diff_cfg.gen_only_test_timestep_respacing = str(args.ddim_steps)
    denoiser3d.init_diffusion()
    model.eval()

    # Re-seed immediately before DDIM creates its initial Gaussian noise.
    seed_everything(args.seed)
    print(
        f"[GEM] Generating {args.num_frames} frames with DDIM={args.ddim_steps}, "
        f"CFG={args.guidance_scale} ..."
    )
    with torch.inference_mode():
        pred = model.predict(
            data,
            static_cam=True,
            postproc=not args.no_postproc,
        )
    body_params = validate_smpl_prediction(pred, args.num_frames)
    body_params = enforce_zero_shape(body_params)
    print("[Shape] mode=zero; global/incam betas norm=0.000000")

    temporary_dir, output_dir = unique_output_paths(args.output_root, prompt, args.seed)
    try:
        temporary_dir.mkdir(parents=False, exist_ok=False)
        if not args.no_render:
            render_global_video(
                temporary_dir,
                _to_cpu(body_params["body_params_global"]),
                args.width,
                args.height,
                args.fps,
            )
        completed_at = utc_now_iso()
        save_results(
            temporary_dir,
            body_params,
            data,
            prompt,
            args,
            ckpt_path,
            text_device,
            text_dtype,
            completed_at,
        )
        publish_ready_directory(temporary_dir, output_dir, completed_at)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    print(f"[Output] Published complete motion with READY: {output_dir}")
    for field, value in body_params["body_params_global"].items():
        print(f"  {field:<14}{list(value.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
