# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""One resident T5 + GEM runtime for text, music, and joint conditioning.

The runtime deliberately owns exactly one tokenizer, one T5 encoder, one full
GEM model, and one initialized DDIM sampler.  All caches contain CPU float32
data only, and all GEM generations are serialized by ``generation_lock``.
"""

from __future__ import annotations

import copy
import gc
import json
import math
import os
import random
import shutil
import threading
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np
import torch

from gem.runtime.artifact_publish import (
    atomic_write_json,
    enforce_zero_shape,
    make_unique_output_paths,
    publish_ready_directory,
    safe_generation_prefix,
    utc_now_iso,
)
from gem.runtime.resident_text_motion import (
    MAX_TEXT_LEN,
    TEXT_EMBED_DIM,
    encode_prompt_with_loaded_t5,
    get_cuda_memory_snapshot,
)
from gem.utils.music_features import (
    EDGE_BASELINE_FEATURE_NAMES,
    EDGE_FEATURE_DIM,
    EDGE_HOP_LENGTH,
    EDGE_SAMPLE_RATE,
    EDGE_TARGET_FPS,
    align_features_to_length,
    extract_edge_baseline35,
)

GenerationMode = Literal["text", "music", "text_music"]
SUPPORTED_AUDIO_SUFFIXES = frozenset({".wav", ".mp3", ".flac"})
FEATURE_CACHE_VERSION = "edge_baseline35_v1"

_CHECKPOINT_ERROR = (
    "The supplied checkpoint does not contain the text- and music-conditioned "
    "diffusion weights required by the multimodal service. Use the official "
    "gem_smpl.ckpt or a checkpoint trained with exp=gem_smpl."
)


class UnsupportedModeError(ValueError):
    """Raised when a request asks for an intentionally unsupported fusion mode."""


@dataclass(slots=True)
class MultimodalMotionRequest:
    """One fixed-duration request handled by the resident engine."""

    mode: GenerationMode
    prompt: str | None = None
    audio_path: str | None = None
    start_sec: float = 0.0
    seed: int = 42
    request_id: str | None = None
    metadata: dict[str, Any] | None = None


def seed_everything(seed: int) -> None:
    """Seed every random source immediately before DDIM noise is created."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_transformers_classes() -> tuple[Any, Any]:
    """Import T5 classes lazily so CPU-only protocol tests stay lightweight."""
    from transformers import T5EncoderModel, T5Tokenizer

    return T5Tokenizer, T5EncoderModel


def _load_t5_components(
    model_name_or_path: str,
    torch_dtype: torch.dtype,
    local_files_only: bool,
    tokenizer_class: Any,
    encoder_class: Any,
) -> tuple[Any, Any, str]:
    """Use the cache-first loader shared with the standalone text service."""
    from gem.runtime.resident_text_motion import _load_t5_components as load

    return load(
        model_name_or_path,
        torch_dtype,
        local_files_only,
        tokenizer_class,
        encoder_class,
    )


def _load_gem_model(checkpoint: Path) -> Any:
    """Load one complete GEM model while deferring DDIM initialization."""
    try:
        from demo_utils import load_model
    except ModuleNotFoundError:
        from scripts.demo.demo_utils import load_model

    return load_model(
        str(checkpoint),
        load_text_encoder=False,
        defer_diffusion_init=True,
    )


def _validate_multimodal_checkpoint(checkpoint_path: str | Path) -> int:
    """Validate text diffusion markers and the exact music input dimension once."""
    checkpoint_path = Path(checkpoint_path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"GEM checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    try:
        state_dict = checkpoint.get("state_dict", checkpoint)
        if not isinstance(state_dict, dict):
            raise RuntimeError(
                f"Checkpoint '{checkpoint_path}' does not contain a valid state_dict"
            )
        keys = tuple(str(key) for key in state_dict)
        text_markers = ("embed_text", "text_encoder_layers", "gate_cross_attn")
        if not all(any(marker in key for key in keys) for marker in text_markers):
            raise RuntimeError(_CHECKPOINT_ERROR)
        weights = [
            value
            for key, value in state_dict.items()
            if str(key).endswith("music_embedder.fc1.weight")
        ]
        if not weights:
            raise RuntimeError(f"{_CHECKPOINT_ERROR} Missing music_embedder.fc1.weight.")
        weight = weights[0]
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise RuntimeError("Checkpoint music_embedder.fc1.weight must be a 2D tensor")
        input_dim = int(weight.shape[1])
        if input_dim != EDGE_FEATURE_DIM:
            raise RuntimeError(
                f"Checkpoint music input dimension is {input_dim}, but EDGE baseline35 "
                f"provides {EDGE_FEATURE_DIM}. Reshaping or projecting is forbidden."
            )
        return input_dim
    finally:
        del checkpoint


def _inspect_loaded_model(model: Any, checkpoint_music_dim: int) -> int:
    """Confirm that the loaded full model exposes both conditions and DDIM."""
    from scripts.demo.demo_music import inspect_model_music_input_dim

    model_music_dim = inspect_model_music_input_dim(model)
    if model_music_dim != checkpoint_music_dim:
        raise RuntimeError(
            "Music input dimension differs between checkpoint and loaded model: "
            f"{checkpoint_music_dim}/{model_music_dim}"
        )
    if bool(model.pipeline.denoiser3d.regression_only):
        raise RuntimeError(_CHECKPOINT_ERROR)
    return model_music_dim


def build_text_music_data(
    prompt: str,
    text_embed: torch.Tensor,
    music_embed: torch.Tensor,
    *,
    width: int = 1280,
    height: int = 720,
    focal: float | None = None,
) -> dict[str, Any]:
    """Build one GEM batch containing simultaneous text and music conditions."""
    prompt = str(prompt).strip()
    if not prompt:
        raise ValueError("prompt must not be empty for text_music mode")
    if tuple(text_embed.shape) != (MAX_TEXT_LEN, TEXT_EMBED_DIM):
        raise ValueError(
            f"text_embed must have shape ({MAX_TEXT_LEN}, {TEXT_EMBED_DIM}), "
            f"got {tuple(text_embed.shape)}"
        )
    if (
        not isinstance(music_embed, torch.Tensor)
        or music_embed.ndim != 2
        or music_embed.shape[1] != EDGE_FEATURE_DIM
    ):
        raise ValueError(
            f"music_embed must have shape [L, {EDGE_FEATURE_DIM}], "
            f"got {getattr(music_embed, 'shape', None)}"
        )
    if not torch.isfinite(text_embed).all() or not torch.isfinite(music_embed).all():
        raise ValueError("text_embed and music_embed must contain only finite values")

    from scripts.demo.demo_music import build_music_only_data

    data = build_music_only_data(
        music_embed.detach().cpu().float(),
        width=width,
        height=height,
        focal=focal,
    )
    data["text_embed"] = text_embed.detach().cpu().float().contiguous()
    data["caption"] = prompt
    data["has_text"] = torch.tensor([True], dtype=torch.bool)
    data["meta"] = [
        {
            "mode": "default",
            "source": "text_music",
            "prompt": prompt,
            "fusion_mode": "joint_gem_condition",
        }
    ]
    if not data["mask"]["has_music_mask"].all():
        raise AssertionError("text_music input must enable the music mask")
    for key in (
        "has_img_mask",
        "has_2d_mask",
        "has_cam_mask",
        "has_audio_mask",
    ):
        if data["mask"][key].any():
            raise AssertionError(f"text_music input unexpectedly enables {key}")
    return data


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()
    if isinstance(value, dict):
        return {key: _cpu_tree(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(child) for child in value)
    return value


class ResidentMultimodalMotionEngine:
    """Resident fixed-duration generator sharing one T5, GEM, and DDIM."""

    def __init__(
        self,
        *,
        ckpt_path: str | Path,
        t5_model: str = "t5-3b",
        local_files_only: bool = False,
        device: str = "cuda:0",
        text_dtype: str = "float16",
        clip_frames: int = 120,
        clip_fps: int = 30,
        ddim_steps: int = 20,
        guidance_scale: float = 2.5,
        width: int = 1280,
        height: int = 720,
        focal: float | None = None,
        bbox_scale: float = 0.75,
        output_root: str | Path = "outputs/multimodal_motion",
        postproc: bool = True,
        text_cache_size: int = 128,
        music_cache_size: int = 32,
        allowed_audio_roots: list[str | Path] | tuple[str | Path, ...] | None = None,
        min_free_gib: float = 4.0,
        strict_memory: bool = False,
        warmup_enabled: bool = True,
        latest_file: str | Path | None = None,
        _allow_cpu_for_tests: bool = False,
    ) -> None:
        if clip_frames <= 0:
            raise ValueError("clip_frames must be > 0")
        if clip_fps != EDGE_TARGET_FPS:
            raise ValueError(f"clip_fps must be {EDGE_TARGET_FPS}")
        if ddim_steps <= 0:
            raise ValueError("ddim_steps must be > 0")
        if not math.isfinite(guidance_scale) or guidance_scale < 0:
            raise ValueError("guidance_scale must be finite and >= 0")
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be > 0")
        if focal is not None and (not math.isfinite(focal) or focal <= 0):
            raise ValueError("focal must be finite and > 0")
        if not 0.0 < bbox_scale <= 1.5:
            raise ValueError("bbox_scale must satisfy 0 < value <= 1.5")
        if text_dtype not in {"float16", "float32"}:
            raise ValueError("text_dtype must be float16 or float32")
        if text_cache_size < 0 or music_cache_size < 0:
            raise ValueError("cache sizes must be >= 0")
        if not math.isfinite(min_free_gib) or min_free_gib < 0:
            raise ValueError("min_free_gib must be finite and >= 0")

        self.ckpt_path = Path(ckpt_path).expanduser()
        self.t5_model = str(t5_model)
        self.local_files_only = bool(local_files_only)
        self.device = torch.device(device)
        self.text_dtype = text_dtype
        self.torch_text_dtype = torch.float16 if text_dtype == "float16" else torch.float32
        self.clip_frames = int(clip_frames)
        self.clip_fps = int(clip_fps)
        self.clip_duration_sec = self.clip_frames / self.clip_fps
        self.ddim_steps = int(ddim_steps)
        self.guidance_scale = float(guidance_scale)
        self.width = int(width)
        self.height = int(height)
        self.focal = None if focal is None else float(focal)
        self.bbox_scale = float(bbox_scale)
        self.output_root = Path(output_root).expanduser()
        self.postproc = bool(postproc)
        self.text_cache_capacity = int(text_cache_size)
        self.music_cache_capacity = int(music_cache_size)
        self.allowed_audio_roots = self._resolve_allowed_roots(allowed_audio_roots or ())
        self.min_free_gib = float(min_free_gib)
        self.strict_memory = bool(strict_memory)
        self.warmup_enabled = bool(warmup_enabled)
        self.latest_file = (
            Path(latest_file).expanduser()
            if latest_file is not None
            else self.output_root / "latest_ready.json"
        )
        self._allow_cpu_for_tests = bool(_allow_cpu_for_tests)

        self.tokenizer: Any | None = None
        self.text_encoder: Any | None = None
        self.gem_model: Any | None = None
        self.denoiser3d: Any | None = None
        self.generation_lock = threading.RLock()
        self._cache_lock = threading.Lock()
        self._text_cache: OrderedDict[tuple[str, str, int], torch.Tensor] = OrderedDict()
        self._music_cache: OrderedDict[tuple[Any, ...], tuple[torch.Tensor, dict[str, Any]]] = (
            OrderedDict()
        )
        # Public aliases ease migration from the two original resident engines.
        self.embedding_cache = self._text_cache
        self.feature_cache = self._music_cache
        self.text_cache_hits = 0
        self.text_cache_misses = 0
        self.music_cache_hits = 0
        self.music_cache_misses = 0
        self.request_count = 0
        self.successful_count = 0
        self.failed_count = 0
        self.ddim_init_count = 0
        self.sequence_number = self._read_previous_sequence()
        self.started_monotonic: float | None = None
        self.startup_timings: dict[str, float] = {}
        self.memory_stages: dict[str, dict[str, float]] = {}
        self.last_request: dict[str, Any] | None = None
        self.initialized = False

    @staticmethod
    def _resolve_allowed_roots(
        values: list[str | Path] | tuple[str | Path, ...],
    ) -> tuple[Path, ...]:
        roots: list[Path] = []
        for value in values:
            root = Path(value).expanduser().resolve(strict=True)
            if not root.is_dir():
                raise NotADirectoryError(f"Allowed audio root is not a directory: {root}")
            roots.append(root)
        return tuple(roots)

    def _read_previous_sequence(self) -> int:
        if not self.latest_file.is_file():
            return 0
        try:
            value = json.loads(self.latest_file.read_text(encoding="utf-8"))
            return max(0, int(value.get("sequence_number", 0)))
        except (OSError, TypeError, ValueError):
            return 0

    def _record_memory(self, stage: str) -> dict[str, float]:
        snapshot = get_cuda_memory_snapshot(self.device)
        self.memory_stages[stage] = snapshot
        print(
            f"[Multimodal] CUDA {stage}: "
            f"allocated={snapshot['allocated_gib']:.3f} GiB "
            f"reserved={snapshot['reserved_gib']:.3f} GiB "
            f"free={snapshot['free_gib']:.3f} GiB "
            f"peak={snapshot['max_allocated_gib']:.3f} GiB"
        )
        return snapshot

    def record_external_memory_stage(self, stage: str) -> dict[str, float]:
        """Record memory after an externally owned component, such as video, loads."""
        snapshot = self._record_memory(stage)
        self._check_free_memory(stage)
        return snapshot

    def _check_free_memory(self, stage: str) -> None:
        if self.device.type != "cuda":
            return
        free_gib = get_cuda_memory_snapshot(self.device)["free_gib"]
        if free_gib >= self.min_free_gib:
            return
        message = (
            f"free CUDA memory after {stage} is {free_gib:.3f} GiB, below "
            f"--min_free_gib={self.min_free_gib:.3f}"
        )
        if self.strict_memory:
            raise RuntimeError(message)
        warnings.warn(message, stacklevel=2)

    def initialize(self) -> None:
        """Load T5, GEM, and initialize one DDIM sampler exactly once."""
        with self.generation_lock:
            if self.initialized:
                return
            if self.device.type != "cuda" and not self._allow_cpu_for_tests:
                raise RuntimeError("The multimodal resident service requires CUDA")
            if self.device.type == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is not available for the multimodal service")
                torch.cuda.set_device(self.device)
                torch.cuda.reset_peak_memory_stats(self.device)

            self.started_monotonic = time.monotonic()
            started = time.perf_counter()
            self._record_memory("initialization")
            checkpoint_dim = _validate_multimodal_checkpoint(self.ckpt_path)
            self.startup_timings["checkpoint_seconds"] = time.perf_counter() - started

            started = time.perf_counter()
            tokenizer_class, encoder_class = _load_transformers_classes()
            self.tokenizer, self.text_encoder, source = _load_t5_components(
                self.t5_model,
                self.torch_text_dtype,
                self.local_files_only,
                tokenizer_class,
                encoder_class,
            )
            self.text_encoder = self.text_encoder.to(self.device).eval()
            self.startup_timings["t5_load_seconds"] = time.perf_counter() - started
            print(f"[Multimodal] T5 ready ({source})")
            self._record_memory("after T5 load")

            started = time.perf_counter()
            self.gem_model = _load_gem_model(self.ckpt_path)
            self.gem_model = self.gem_model.to(self.device).eval()
            _inspect_loaded_model(self.gem_model, checkpoint_dim)
            self.denoiser3d = self.gem_model.pipeline.denoiser3d
            self.startup_timings["gem_load_seconds"] = time.perf_counter() - started
            self._record_memory("after GEM load")

            started = time.perf_counter()
            diffusion = self.denoiser3d.model_cfg.diffusion
            diffusion.guidance_param = self.guidance_scale
            diffusion.test_timestep_respacing = str(self.ddim_steps)
            diffusion.gen_only_test_timestep_respacing = str(self.ddim_steps)
            self.denoiser3d.init_diffusion()
            self.ddim_init_count += 1
            self.startup_timings["ddim_init_seconds"] = time.perf_counter() - started
            print("[Multimodal] DDIM initialized once")

            self.initialized = True
            self._check_free_memory("model initialization")
            if self.warmup_enabled:
                self.warmup()
            self.startup_timings["total_seconds"] = time.monotonic() - self.started_monotonic
            self._record_memory("after warmup")
            self._check_free_memory("warmup")
            print("[Multimodal] SERVICE READY")

    @staticmethod
    def _normalize_prompt(value: str | None) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("prompt must be a non-empty string")
        return value.strip()

    def _text_key(self, prompt: str) -> tuple[str, str, int]:
        return prompt, self.t5_model, MAX_TEXT_LEN

    def _get_text_embedding(self, prompt: str) -> tuple[torch.Tensor, bool]:
        normalized = self._normalize_prompt(prompt)
        key = self._text_key(normalized)
        with self._cache_lock:
            cached = self._text_cache.get(key)
            if cached is not None:
                self._text_cache.move_to_end(key)
                self.text_cache_hits += 1
                return cached, True
            self.text_cache_misses += 1
        if self.tokenizer is None or self.text_encoder is None:
            raise RuntimeError("resident T5 is not initialized")
        embedding = encode_prompt_with_loaded_t5(
            normalized,
            self.tokenizer,
            self.text_encoder,
            self.device,
            MAX_TEXT_LEN,
        )
        with self._cache_lock:
            if self.text_cache_capacity > 0:
                self._text_cache[key] = embedding
                self._text_cache.move_to_end(key)
                while len(self._text_cache) > self.text_cache_capacity:
                    self._text_cache.popitem(last=False)
        return embedding, False

    def _validate_audio_path(self, value: str | None) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("audio_path must be a non-empty string")
        path = Path(value).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"Audio path is not a regular file: {path}")
        if path.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
            raise ValueError("audio_path must have a .wav, .mp3, or .flac suffix")
        if self.allowed_audio_roots and not any(
            path == root or path.is_relative_to(root) for root in self.allowed_audio_roots
        ):
            roots = ", ".join(str(root) for root in self.allowed_audio_roots)
            raise PermissionError(
                f"Resolved audio path is outside --allowed_audio_root: {path}; roots={roots}"
            )
        return path

    @staticmethod
    def _stable_time_key(value: float) -> int:
        return int(round(value * 1_000_000))

    def _music_key(self, path: Path, start_sec: float) -> tuple[Any, ...]:
        stat = path.stat()
        return (
            str(path),
            int(stat.st_dev),
            int(stat.st_ino),
            int(stat.st_size),
            int(stat.st_mtime_ns),
            self._stable_time_key(start_sec),
            self._stable_time_key(self.clip_duration_sec),
            FEATURE_CACHE_VERSION,
            EDGE_SAMPLE_RATE,
            EDGE_HOP_LENGTH,
            EDGE_TARGET_FPS,
        )

    def _get_music_features(
        self, path: Path, start_sec: float
    ) -> tuple[torch.Tensor, dict[str, Any], bool]:
        key = self._music_key(path, start_sec)
        with self._cache_lock:
            cached = self._music_cache.get(key)
            if cached is not None:
                self._music_cache.move_to_end(key)
                self.music_cache_hits += 1
                features, metadata = cached
                return features, copy.deepcopy(metadata), True
            self.music_cache_misses += 1

        features, metadata = extract_edge_baseline35(
            path,
            start_sec=start_sec,
            duration_sec=self.clip_duration_sec,
            target_fps=self.clip_fps,
        )
        features = features.detach().cpu().float().contiguous()
        if features.ndim != 2 or features.shape[1] != EDGE_FEATURE_DIM:
            raise RuntimeError(
                f"EDGE features must have shape [L, {EDGE_FEATURE_DIM}], "
                f"got {tuple(features.shape)}"
            )
        if not torch.isfinite(features).all():
            raise RuntimeError("EDGE music features contain NaN or Inf")
        difference = abs(int(features.shape[0]) - self.clip_frames)
        if difference > 2:
            raise RuntimeError(
                f"Music extraction produced {features.shape[0]} frames for a fixed "
                f"{self.clip_frames}-frame clip (difference={difference}). Only a "
                "boundary difference of at most 2 frames may be aligned."
            )
        aligned = (
            align_features_to_length(
                features,
                self.clip_frames,
                "trim_or_pad_last",
            )
            .detach()
            .cpu()
            .float()
            .contiguous()
        )
        if tuple(aligned.shape) != (self.clip_frames, EDGE_FEATURE_DIM):
            raise AssertionError("aligned music feature shape does not match fixed clip")
        metadata = copy.deepcopy(metadata)
        metadata["raw_feature_frames"] = int(features.shape[0])
        metadata["feature_frames"] = self.clip_frames
        metadata["alignment_policy"] = "trim_or_pad_last"
        with self._cache_lock:
            if self.music_cache_capacity > 0:
                self._music_cache[key] = (aligned, copy.deepcopy(metadata))
                self._music_cache.move_to_end(key)
                while len(self._music_cache) > self.music_cache_capacity:
                    self._music_cache.popitem(last=False)
        return aligned, metadata, False

    def clear_text_cache(self) -> int:
        """Clear CPU T5 embeddings without releasing either GPU model."""
        with self._cache_lock:
            removed = len(self._text_cache)
            self._text_cache.clear()
        return removed

    def clear_music_cache(self) -> int:
        """Clear CPU EDGE features without releasing either GPU model."""
        with self._cache_lock:
            removed = len(self._music_cache)
            self._music_cache.clear()
        return removed

    def clear_cache(self, target: str = "all") -> dict[str, int]:
        """Clear selected CPU caches."""
        if target not in {"all", "text", "music"}:
            raise ValueError("cache target must be all, text, or music")
        return {
            "text": self.clear_text_cache() if target in {"all", "text"} else 0,
            "music": self.clear_music_cache() if target in {"all", "music"} else 0,
        }

    def _validate_request(
        self, request: MultimodalMotionRequest | dict[str, Any]
    ) -> MultimodalMotionRequest:
        if isinstance(request, dict):
            allowed = {
                "mode",
                "prompt",
                "audio_path",
                "start_sec",
                "seed",
                "request_id",
                "metadata",
            }
            unknown = sorted(set(request) - allowed)
            if unknown:
                raise ValueError(f"unsupported request fields: {unknown}")
            request = MultimodalMotionRequest(**request)
        if not isinstance(request, MultimodalMotionRequest):
            raise TypeError("request must be MultimodalMotionRequest or a dictionary")
        if request.mode in {"video_text", "video_music", "video_text_music"}:
            raise UnsupportedModeError(
                "True video multimodal fusion is not supported by the real-time ONNX path."
            )
        if request.mode not in {"text", "music", "text_music"}:
            raise UnsupportedModeError(f"Unsupported generation mode: {request.mode}")
        if request.mode in {"text", "text_music"}:
            request.prompt = self._normalize_prompt(request.prompt)
        elif request.prompt is not None:
            raise ValueError("music mode does not accept prompt")
        if request.mode in {"music", "text_music"}:
            request.audio_path = str(self._validate_audio_path(request.audio_path))
            request.start_sec = float(request.start_sec)
            if not math.isfinite(request.start_sec) or request.start_sec < 0:
                raise ValueError("start_sec must be finite and >= 0")
        elif request.audio_path is not None:
            raise ValueError("text mode does not accept audio_path")
        if not isinstance(request.seed, int) or isinstance(request.seed, bool):
            raise TypeError("seed must be an integer")
        request.request_id = (
            str(request.request_id).strip() if request.request_id is not None else uuid4().hex
        )
        if not request.request_id:
            raise ValueError("request_id must not be empty")
        if request.metadata is not None and not isinstance(request.metadata, dict):
            raise TypeError("metadata must be a dictionary or null")
        return request

    def _build_input(
        self, request: MultimodalMotionRequest
    ) -> tuple[dict[str, Any], torch.Tensor | None, dict[str, Any] | None, dict[str, bool]]:
        from scripts.demo.demo_music import build_music_only_data
        from scripts.demo.demo_smpl_text import build_text_only_data

        text_embed: torch.Tensor | None = None
        music_features: torch.Tensor | None = None
        music_metadata: dict[str, Any] | None = None
        cache_hits = {"text": False, "music": False}
        if request.mode in {"text", "text_music"}:
            text_embed, cache_hits["text"] = self._get_text_embedding(request.prompt or "")
        if request.mode in {"music", "text_music"}:
            music_features, music_metadata, cache_hits["music"] = self._get_music_features(
                Path(request.audio_path or ""),
                request.start_sec,
            )

        if request.mode == "text":
            data = build_text_only_data(
                request.prompt or "",
                text_embed,
                self.clip_frames,
                self.width,
                self.height,
                self.bbox_scale,
            )
        elif request.mode == "music":
            data = build_music_only_data(
                music_features,
                width=self.width,
                height=self.height,
                focal=self.focal,
            )
        else:
            data = build_text_music_data(
                request.prompt or "",
                text_embed,
                music_features,
                width=self.width,
                height=self.height,
                focal=self.focal,
            )
        if int(data["length"]) != self.clip_frames:
            raise AssertionError("request attempted to override the fixed clip length")
        return data, music_features, music_metadata, cache_hits

    def _metadata(
        self,
        request: MultimodalMotionRequest,
        music_metadata: dict[str, Any] | None,
        completed_at: str,
    ) -> dict[str, Any]:
        source = {
            "text": "text_only",
            "music": "music_only",
            "text_music": "text_music",
        }[request.mode]
        metadata: dict[str, Any] = {
            "source": source,
            "shape_mode": "zero",
            "request_id": request.request_id,
            "request_metadata": copy.deepcopy(request.metadata or {}),
            "seed": request.seed,
            "num_frames": self.clip_frames,
            "fps": self.clip_fps,
            "generated_duration_sec": self.clip_duration_sec,
            "guidance_scale": self.guidance_scale,
            "ddim_steps": self.ddim_steps,
            "checkpoint": str(self.ckpt_path.resolve()),
            "width": self.width,
            "height": self.height,
            "focal": float(max(self.width, self.height) if self.focal is None else self.focal),
            "postproc": self.postproc,
            "service": "resident_multimodal_motion",
            "completed_at": completed_at,
        }
        if request.mode in {"text", "text_music"}:
            metadata["prompt"] = request.prompt
            metadata["t5_model"] = self.t5_model
        if request.mode in {"music", "text_music"}:
            assert music_metadata is not None
            metadata.update(
                {
                    "audio_path": request.audio_path,
                    "audio_start_sec": request.start_sec,
                    "audio_duration_sec": self.clip_duration_sec,
                    "original_audio_duration_sec": float(music_metadata["original_duration_sec"]),
                    "feature_type": "edge_baseline35",
                    "feature_fps": self.clip_fps,
                    "feature_dim": EDGE_FEATURE_DIM,
                    "feature_names": list(EDGE_BASELINE_FEATURE_NAMES),
                    "feature_frames": self.clip_frames,
                    "raw_feature_frames": int(
                        music_metadata.get("raw_feature_frames", self.clip_frames)
                    ),
                    "sample_rate": EDGE_SAMPLE_RATE,
                    "hop_length": EDGE_HOP_LENGTH,
                    "estimated_bpm": float(music_metadata["estimated_or_prior_bpm"]),
                    "bpm_source": str(music_metadata["bpm_source"]),
                }
            )
        if request.mode == "text_music":
            metadata.update(
                {
                    "fusion_mode": "joint_gem_condition",
                    "fusion_training_status": "zero_shot_cross_dataset",
                }
            )
        return metadata

    def _generation_prefix(self, request: MultimodalMotionRequest) -> str:
        if request.mode == "text":
            label = request.prompt or "text"
        elif request.mode == "music":
            label = Path(request.audio_path or "music").stem
        else:
            label = f"{Path(request.audio_path or 'music').stem}_{request.prompt or 'text'}"
        return f"{safe_generation_prefix(label, limit=56)}_{request.mode}_seed{request.seed}"

    def _write_artifacts(
        self,
        output_dir: Path,
        *,
        body_params: dict[str, dict[str, torch.Tensor]],
        data: dict[str, Any],
        request: MultimodalMotionRequest,
        music_features: torch.Tensor | None,
        metadata: dict[str, Any],
    ) -> None:
        output_dir.mkdir(parents=False, exist_ok=False)
        body_params = enforce_zero_shape(_cpu_tree(body_params))
        global_group = body_params["body_params_global"]
        incam_group = body_params["body_params_incam"]
        for name, group in body_params.items():
            if torch.count_nonzero(group["betas"]).item() != 0:
                raise AssertionError(f"{name}.betas must be zero")

        payload: dict[str, Any] = {
            "body_params_global": global_group,
            "body_params_incam": incam_group,
            "K_fullimg": data["K_fullimg"].detach().cpu(),
            "bbx_xys": data["bbx_xys"].detach().cpu(),
            "fps": float(self.clip_fps),
            "num_frames": self.clip_frames,
            "duration_sec": self.clip_duration_sec,
            "source": metadata["source"],
            "shape_mode": "zero",
            "seed": request.seed,
            "guidance_scale": self.guidance_scale,
            "ddim_steps": self.ddim_steps,
            "checkpoint": str(self.ckpt_path),
            "metadata": dict(metadata),
        }
        if request.prompt is not None:
            payload["prompt"] = request.prompt
        for key in (
            "audio_path",
            "audio_start_sec",
            "audio_duration_sec",
            "feature_type",
            "feature_fps",
            "feature_dim",
            "estimated_bpm",
            "bpm_source",
            "fusion_mode",
            "fusion_training_status",
        ):
            if key in metadata:
                payload[key] = metadata[key]
        torch.save(payload, output_dir / "smpl_params.pt")
        np.savez(
            output_dir / "motion.npz",
            body_pose=global_group["body_pose"].numpy(),
            global_orient=global_group["global_orient"].numpy(),
            transl=global_group["transl"].numpy(),
            betas=global_group["betas"].numpy(),
            fps=np.asarray(self.clip_fps, dtype=np.float32),
        )
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if request.prompt is not None:
            (output_dir / "prompt.txt").write_text(request.prompt + "\n", encoding="utf-8")
        if music_features is not None:
            torch.save(
                music_features.detach().cpu().float().contiguous(),
                output_dir / "music_features.pt",
            )
            (output_dir / "source_audio.txt").write_text(
                f"audio_path={metadata['audio_path']}\n"
                f"start_sec={metadata['audio_start_sec']:.9f}\n"
                f"duration_sec={metadata['audio_duration_sec']:.9f}\n",
                encoding="utf-8",
            )

    def _failure_response(
        self, request_id: str | None, exc: Exception, started: float
    ) -> dict[str, Any]:
        response = {
            "ok": False,
            "request_id": request_id,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "total_seconds": time.perf_counter() - started,
        }
        self.failed_count += 1
        self.last_request = response
        return response

    def generate(self, request: MultimodalMotionRequest | dict[str, Any]) -> dict[str, Any]:
        """Generate one fixed clip and atomically publish a READY directory."""
        total_started = time.perf_counter()
        request_id = (
            request.get("request_id")
            if isinstance(request, dict)
            else getattr(request, "request_id", None)
        )
        temporary_dir: Path | None = None
        with self.generation_lock:
            self.request_count += 1
            try:
                if not self.initialized or self.gem_model is None:
                    raise RuntimeError("multimodal resident engine is not initialized")
                request = self._validate_request(request)
                request_id = request.request_id
                if self.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(self.device)

                input_started = time.perf_counter()
                data, features, feature_metadata, cache_hits = self._build_input(request)
                input_seconds = time.perf_counter() - input_started

                seed_everything(request.seed)
                generation_started = time.perf_counter()
                with torch.inference_mode():
                    pred = self.gem_model.predict(
                        data,
                        static_cam=True,
                        postproc=self.postproc,
                    )
                from scripts.demo.demo_smpl_text import validate_smpl_prediction

                body_params = validate_smpl_prediction(pred, self.clip_frames)
                body_params = enforce_zero_shape(body_params)
                generation_seconds = time.perf_counter() - generation_started

                save_started = time.perf_counter()
                temporary_dir, output_dir = make_unique_output_paths(
                    self.output_root,
                    self._generation_prefix(request),
                )
                completed_at = utc_now_iso()
                metadata = self._metadata(request, feature_metadata, completed_at)
                self._write_artifacts(
                    temporary_dir,
                    body_params=body_params,
                    data=data,
                    request=request,
                    music_features=features,
                    metadata=metadata,
                )
                publish_ready_directory(temporary_dir, output_dir, completed_at)
                temporary_dir = None

                self.sequence_number += 1
                latest = {
                    "request_id": request.request_id,
                    "mode": request.mode,
                    "source": metadata["source"],
                    "output_dir": str(output_dir.resolve()),
                    "smpl_params": str((output_dir / "smpl_params.pt").resolve()),
                    "motion_npz": str((output_dir / "motion.npz").resolve()),
                    "ready": str((output_dir / "READY").resolve()),
                    "num_frames": self.clip_frames,
                    "fps": self.clip_fps,
                    "seed": request.seed,
                    "completed_at": completed_at,
                    "sequence_number": self.sequence_number,
                }
                atomic_write_json(self.latest_file, latest)
                save_seconds = time.perf_counter() - save_started
                total_seconds = time.perf_counter() - total_started
                response = {
                    "ok": True,
                    **latest,
                    "duration_seconds": self.clip_duration_sec,
                    "text_cache_hit": cache_hits["text"],
                    "music_cache_hit": cache_hits["music"],
                    "timing": {
                        "input_seconds": input_seconds,
                        "generation_seconds": generation_seconds,
                        "save_seconds": save_seconds,
                        "total_seconds": total_seconds,
                    },
                    "gpu": get_cuda_memory_snapshot(self.device),
                }
                self.successful_count += 1
                self.last_request = response
                print(
                    f"[Multimodal] request={request.request_id} mode={request.mode} "
                    f"frames={self.clip_frames} total={total_seconds:.3f}s "
                    f"output={output_dir}"
                )
                return response
            except torch.cuda.OutOfMemoryError as exc:
                if temporary_dir is not None:
                    shutil.rmtree(temporary_dir, ignore_errors=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return self._failure_response(request_id, exc, total_started)
            except Exception as exc:
                if temporary_dir is not None:
                    shutil.rmtree(temporary_dir, ignore_errors=True)
                return self._failure_response(request_id, exc, total_started)

    def generate_text(
        self, prompt: str, *, seed: int = 42, request_id: str | None = None
    ) -> dict[str, Any]:
        """Generate one text-conditioned fixed clip."""
        return self.generate(
            {
                "mode": "text",
                "prompt": prompt,
                "seed": seed,
                "request_id": request_id,
            }
        )

    def generate_music(
        self,
        audio_path: str,
        *,
        start_sec: float = 0.0,
        seed: int = 42,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Generate one music-conditioned fixed clip."""
        return self.generate(
            {
                "mode": "music",
                "audio_path": audio_path,
                "start_sec": start_sec,
                "seed": seed,
                "request_id": request_id,
            }
        )

    def generate_text_music(
        self,
        prompt: str,
        audio_path: str,
        *,
        start_sec: float = 0.0,
        seed: int = 42,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Generate one clip from simultaneous text and music conditions."""
        return self.generate(
            {
                "mode": "text_music",
                "prompt": prompt,
                "audio_path": audio_path,
                "start_sec": start_sec,
                "seed": seed,
                "request_id": request_id,
            }
        )

    def warmup(self) -> dict[str, Any]:
        """Warm both condition paths with one unpublished joint GEM prediction."""
        if not self.initialized or self.gem_model is None:
            raise RuntimeError("initialize the engine before warmup")
        started = time.perf_counter()
        prompt = "A person stands still."
        text_embed, _ = self._get_text_embedding(prompt)
        music = torch.zeros(self.clip_frames, EDGE_FEATURE_DIM, dtype=torch.float32)
        data = build_text_music_data(
            prompt,
            text_embed,
            music,
            width=self.width,
            height=self.height,
            focal=self.focal,
        )
        seed_everything(0)
        with torch.inference_mode():
            pred = self.gem_model.predict(
                data,
                static_cam=True,
                postproc=self.postproc,
            )
        from scripts.demo.demo_smpl_text import validate_smpl_prediction

        body_params = enforce_zero_shape(validate_smpl_prediction(pred, self.clip_frames))
        for group in body_params.values():
            if torch.count_nonzero(group["betas"]).item() != 0:
                raise AssertionError("warmup did not enforce zero betas")
        result = {
            "frames": self.clip_frames,
            "seconds": time.perf_counter() - started,
            "gpu": get_cuda_memory_snapshot(self.device),
        }
        self.startup_timings["warmup_seconds"] = result["seconds"]
        print("[Multimodal] WARMUP COMPLETE")
        return result

    def status(self) -> dict[str, Any]:
        """Return model identity, fixed contract, caches, counters, and memory."""
        with self._cache_lock:
            text_bytes = sum(
                tensor.numel() * tensor.element_size() for tensor in self._text_cache.values()
            )
            music_bytes = sum(
                tensor.numel() * tensor.element_size() for tensor, _ in self._music_cache.values()
            )
            text_size = len(self._text_cache)
            music_size = len(self._music_cache)
        uptime = (
            time.monotonic() - self.started_monotonic if self.started_monotonic is not None else 0.0
        )
        return {
            "initialized": self.initialized,
            "pid": os.getpid(),
            "device": str(self.device),
            "checkpoint": str(self.ckpt_path),
            "t5_model": self.t5_model,
            "gem_instances": int(self.gem_model is not None),
            "t5_instances": int(self.text_encoder is not None),
            "ddim_init_count": self.ddim_init_count,
            "ddim_steps": self.ddim_steps,
            "guidance_scale": self.guidance_scale,
            "clip_frames": self.clip_frames,
            "clip_fps": self.clip_fps,
            "clip_duration_sec": self.clip_duration_sec,
            "request_count": self.request_count,
            "successful_count": self.successful_count,
            "failed_count": self.failed_count,
            "text_cache_size": text_size,
            "text_cache_capacity": self.text_cache_capacity,
            "text_cache_hits": self.text_cache_hits,
            "text_cache_misses": self.text_cache_misses,
            "text_cache_bytes": text_bytes,
            "music_cache_size": music_size,
            "music_cache_capacity": self.music_cache_capacity,
            "music_cache_hits": self.music_cache_hits,
            "music_cache_misses": self.music_cache_misses,
            "music_cache_bytes": music_bytes,
            "allowed_audio_roots": [str(path) for path in self.allowed_audio_roots],
            "startup_timings": dict(self.startup_timings),
            "memory_stages": copy.deepcopy(self.memory_stages),
            "gpu_memory": get_cuda_memory_snapshot(self.device),
            "last_request": self.last_request,
            "uptime_seconds": uptime,
        }

    def close(self) -> None:
        """Release resident models only during explicit service shutdown."""
        with self.generation_lock:
            self.initialized = False
            self.clear_cache("all")
            self.denoiser3d = None
            self.gem_model = None
            self.text_encoder = None
            self.tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[Multimodal] SERVICE STOPPED")
