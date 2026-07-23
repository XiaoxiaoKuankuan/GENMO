# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Resident T5-3B + GEM-SMPL text-to-motion inference runtime.

The engine loads both models once, keeps them on one CUDA device, serializes
generation requests, and publishes the same READY artifact contract as the
single-shot text demo.
"""

from __future__ import annotations

import gc
import json
import os
import random
import shutil
import threading
import time
import warnings
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
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

MAX_TEXT_LEN = 50
TEXT_EMBED_DIM = 1024
GIB = float(1024**3)


@dataclass(slots=True)
class TextMotionRequest:
    """One immutable-generation request accepted by the resident engine."""

    prompt: str
    num_frames: int = 120
    fps: float = 30.0
    seed: int = 42
    request_id: str | None = None
    output_root: str | None = None
    metadata: dict[str, Any] | None = None


def normalize_prompt(prompt: str) -> str:
    """Strip surrounding whitespace without changing prompt case or content."""
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    normalized = prompt.strip()
    if not normalized:
        raise ValueError("prompt must not be empty")
    return normalized


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch immediately before DDIM noise creation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_t5_components(
    model_name_or_path: str,
    torch_dtype: torch.dtype,
    local_files_only: bool,
    tokenizer_class: Any,
    encoder_class: Any,
) -> tuple[Any, Any, str]:
    """Reuse the single-shot demo's cache-first T5 loading contract."""
    from scripts.demo.demo_smpl_text import _load_t5_components_cached_first

    return _load_t5_components_cached_first(
        model_name_or_path,
        torch_dtype,
        local_files_only,
        tokenizer_class,
        encoder_class,
    )


def encode_prompt_with_loaded_t5(
    prompt: str,
    tokenizer: Any,
    text_encoder: Any,
    device: str | torch.device,
    max_text_len: int = MAX_TEXT_LEN,
) -> torch.Tensor:
    """Encode text exactly like GEM while leaving tokenizer and T5 resident."""
    normalized = normalize_prompt(prompt)
    tokenized = tokenizer(
        [normalized],
        return_tensors="pt",
        padding="max_length",
        max_length=max_text_len,
        truncation=True,
    )
    input_ids = (
        tokenized["input_ids"] if isinstance(tokenized, dict) else tokenized.input_ids
    ).to(device)
    attention_mask = (
        tokenized["attention_mask"]
        if isinstance(tokenized, dict)
        else tokenized.attention_mask
    ).to(device)
    with torch.inference_mode():
        output = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
    encoded_text = output.last_hidden_state[:, :max_text_len]
    encoded_text = encoded_text * attention_mask[:, :max_text_len].unsqueeze(-1)
    expected = (1, max_text_len, TEXT_EMBED_DIM)
    if tuple(encoded_text.shape) != expected:
        raise RuntimeError(
            f"T5 text embedding has shape {tuple(encoded_text.shape)}, expected {expected}. "
            "Use the T5-3B encoder expected by GEM-SMPL."
        )
    return encoded_text[0].detach().float().cpu().contiguous()


def get_cuda_memory_snapshot(
    device: str | torch.device = "cuda:0",
) -> dict[str, float]:
    """Return allocator and physical CUDA memory counters in GiB."""
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        return {
            "total_gib": 0.0,
            "free_gib": 0.0,
            "allocated_gib": 0.0,
            "reserved_gib": 0.0,
            "max_allocated_gib": 0.0,
        }
    free_bytes, total_bytes = torch.cuda.mem_get_info(resolved)
    return {
        "total_gib": total_bytes / GIB,
        "free_gib": free_bytes / GIB,
        "allocated_gib": torch.cuda.memory_allocated(resolved) / GIB,
        "reserved_gib": torch.cuda.memory_reserved(resolved) / GIB,
        "max_allocated_gib": torch.cuda.max_memory_allocated(resolved) / GIB,
    }


def _load_transformers_classes() -> tuple[Any, Any]:
    from transformers import T5EncoderModel, T5Tokenizer

    return T5Tokenizer, T5EncoderModel


def _load_gem_model(checkpoint: Path) -> Any:
    try:
        from demo_utils import load_model
    except ModuleNotFoundError:
        from scripts.demo.demo_utils import load_model

    return load_model(
        str(checkpoint),
        load_text_encoder=False,
        defer_diffusion_init=True,
    )


def _validate_checkpoint(checkpoint: Path) -> None:
    from scripts.demo.demo_smpl_text import validate_text_generation_checkpoint

    validate_text_generation_checkpoint(checkpoint)


def _text_demo_helpers() -> Any:
    """Import existing single-shot contracts lazily to avoid code duplication."""
    from scripts.demo import demo_smpl_text

    return demo_smpl_text


class ResidentTextMotionEngine:
    """Keep T5-3B and GEM-SMPL resident on one device across requests."""

    def __init__(
        self,
        *,
        ckpt_path: str | Path,
        t5_model: str = "t5-3b",
        device: str = "cuda:0",
        text_dtype: str = "float16",
        local_files_only: bool = False,
        ddim_steps: int = 20,
        guidance_scale: float = 2.5,
        width: int = 1280,
        height: int = 720,
        bbox_scale: float = 0.75,
        output_root: str | Path = "outputs/text_motion",
        postproc: bool = True,
        shape_mode: str = "zero",
        embedding_cache_size: int = 128,
        min_free_gib: float = 2.0,
        warmup_frames: int = 30,
        warmup_prompt: str = "A person stands still.",
        warmup_enabled: bool = True,
        strict_memory: bool = False,
        latest_file: str | Path | None = None,
        max_frames: int = 900,
        _allow_cpu_for_tests: bool = False,
    ) -> None:
        if text_dtype not in {"float16", "float32"}:
            raise ValueError("text_dtype must be float16 or float32")
        if ddim_steps <= 0:
            raise ValueError("ddim_steps must be > 0")
        if guidance_scale < 0:
            raise ValueError("guidance_scale must be >= 0")
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be > 0")
        if not 0.0 < bbox_scale <= 1.5:
            raise ValueError("bbox_scale must satisfy 0 < value <= 1.5")
        if shape_mode != "zero":
            raise ValueError("resident robot-compatible service only supports shape_mode=zero")
        if embedding_cache_size < 0:
            raise ValueError("embedding_cache_size must be >= 0")
        if min_free_gib < 0:
            raise ValueError("min_free_gib must be >= 0")
        if warmup_frames <= 0 or max_frames <= 0:
            raise ValueError("warmup_frames and max_frames must be > 0")

        self.ckpt_path = Path(ckpt_path).expanduser()
        self.t5_model = t5_model
        self.device = torch.device(device)
        self.text_dtype = text_dtype
        self.torch_dtype = torch.float16 if text_dtype == "float16" else torch.float32
        self.local_files_only = local_files_only
        self.ddim_steps = int(ddim_steps)
        self.guidance_scale = float(guidance_scale)
        self.width = int(width)
        self.height = int(height)
        self.bbox_scale = float(bbox_scale)
        self.output_root = Path(output_root).expanduser()
        self.postproc = bool(postproc)
        self.shape_mode = shape_mode
        self.embedding_cache_size = int(embedding_cache_size)
        self.min_free_gib = float(min_free_gib)
        self.warmup_frames = int(warmup_frames)
        self.warmup_prompt = normalize_prompt(warmup_prompt)
        self.warmup_enabled = bool(warmup_enabled)
        self.strict_memory = bool(strict_memory)
        self.latest_file = (
            Path(latest_file).expanduser()
            if latest_file is not None
            else self.output_root / "latest_ready.json"
        )
        self.max_frames = int(max_frames)
        self._allow_cpu_for_tests = _allow_cpu_for_tests

        self.tokenizer: Any | None = None
        self.text_encoder: Any | None = None
        self.gem_model: Any | None = None
        self.denoiser3d: Any | None = None
        self.generation_lock = threading.RLock()
        self._cache_lock = threading.Lock()
        self.embedding_cache: OrderedDict[tuple[str, str, int], torch.Tensor] = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        self.request_count = 0
        self.successful_count = 0
        self.failed_count = 0
        self.startup_timings: dict[str, float] = {}
        self.last_request: dict[str, Any] | None = None
        self.initialized = False
        self.started_monotonic: float | None = None
        self.sequence_number = self._read_previous_sequence()

    def _read_previous_sequence(self) -> int:
        if not self.latest_file.is_file():
            return 0
        try:
            payload = json.loads(self.latest_file.read_text(encoding="utf-8"))
            return max(0, int(payload.get("sequence_number", 0)))
        except (OSError, ValueError, TypeError):
            return 0

    def _log_memory(self, stage: str) -> dict[str, float]:
        snapshot = get_cuda_memory_snapshot(self.device)
        print(
            f"[Resident] GPU {stage}: allocated={snapshot['allocated_gib']:.3f} GiB "
            f"reserved={snapshot['reserved_gib']:.3f} GiB "
            f"free={snapshot['free_gib']:.3f} GiB"
        )
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
        """Load T5, GEM and the configured DDIM sampler exactly once."""
        with self.generation_lock:
            if self.initialized:
                return
            if self.device.type != "cuda" and not self._allow_cpu_for_tests:
                raise RuntimeError("Resident T5 + GEM service requires a CUDA device")
            if self.device.type == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is not available for the resident service")
                torch.cuda.set_device(self.device)
                torch.cuda.reset_peak_memory_stats(self.device)

            self.started_monotonic = time.monotonic()
            stage = time.perf_counter()
            print("[Resident] CUDA initialized")
            self._log_memory("CUDA initialization")
            _validate_checkpoint(self.ckpt_path)
            self.startup_timings["cuda_and_checkpoint_seconds"] = time.perf_counter() - stage

            stage = time.perf_counter()
            print(
                f"[Resident] Loading T5-3B {self.text_dtype} on {self.device}"
            )
            tokenizer_class, encoder_class = _load_transformers_classes()
            self.tokenizer, self.text_encoder, source = _load_t5_components(
                str(self.t5_model),
                self.torch_dtype,
                self.local_files_only,
                tokenizer_class,
                encoder_class,
            )
            self.text_encoder = self.text_encoder.to(self.device).eval()
            self.startup_timings["t5_load_seconds"] = time.perf_counter() - stage
            print(f"[Resident] T5 ready ({source})")
            self._log_memory("T5 load")

            stage = time.perf_counter()
            print("[Resident] Loading GEM-SMPL")
            self.gem_model = _load_gem_model(self.ckpt_path)
            self.gem_model = self.gem_model.to(self.device).eval()
            self.denoiser3d = self.gem_model.pipeline.denoiser3d
            if self.denoiser3d.regression_only:
                raise RuntimeError(
                    "The supplied GEM checkpoint is regression-only and cannot generate "
                    "text-conditioned motion."
                )
            self.startup_timings["gem_load_seconds"] = time.perf_counter() - stage
            print("[Resident] GEM ready")
            self._log_memory("GEM load")

            stage = time.perf_counter()
            diff_cfg = self.denoiser3d.model_cfg.diffusion
            diff_cfg.guidance_param = self.guidance_scale
            diff_cfg.test_timestep_respacing = str(self.ddim_steps)
            diff_cfg.gen_only_test_timestep_respacing = str(self.ddim_steps)
            self.denoiser3d.init_diffusion()
            self.startup_timings["ddim_init_seconds"] = time.perf_counter() - stage
            print("[Resident] DDIM initialized once")

            self.initialized = True
            self._check_free_memory("model initialization")
            if self.warmup_enabled:
                self.warmup()
            self.startup_timings["total_seconds"] = (
                time.monotonic() - self.started_monotonic
            )
            self._check_free_memory("warmup")
            print("[Resident] SERVICE READY")

    def _cache_key(self, prompt: str) -> tuple[str, str, int]:
        return prompt, str(self.t5_model), MAX_TEXT_LEN

    def _encode_cached(self, prompt: str) -> tuple[torch.Tensor, bool]:
        normalized = normalize_prompt(prompt)
        key = self._cache_key(normalized)
        with self._cache_lock:
            cached = self.embedding_cache.get(key)
            if cached is not None:
                self.embedding_cache.move_to_end(key)
                self.cache_hits += 1
                return cached, True
            self.cache_misses += 1
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
            if self.embedding_cache_size > 0:
                self.embedding_cache[key] = embedding
                self.embedding_cache.move_to_end(key)
                while len(self.embedding_cache) > self.embedding_cache_size:
                    self.embedding_cache.popitem(last=False)
        return embedding, False

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """Return a cached or freshly encoded CPU float32 T5 embedding."""
        embedding, _ = self._encode_cached(prompt)
        return embedding

    def clear_cache(self) -> int:
        """Clear CPU embeddings without touching either resident GPU model."""
        with self._cache_lock:
            removed = len(self.embedding_cache)
            self.embedding_cache.clear()
        return removed

    def _validate_request(
        self, request: TextMotionRequest | dict[str, Any]
    ) -> TextMotionRequest:
        if isinstance(request, dict):
            allowed = {
                "request_id",
                "prompt",
                "num_frames",
                "fps",
                "seed",
                "output_root",
                "metadata",
            }
            unknown = sorted(set(request) - allowed)
            if unknown:
                raise ValueError(f"unsupported request fields: {unknown}")
            request = TextMotionRequest(**request)
        if not isinstance(request, TextMotionRequest):
            raise TypeError("request must be TextMotionRequest or a request dictionary")
        request.prompt = normalize_prompt(request.prompt)
        request.request_id = (
            str(request.request_id).strip() if request.request_id is not None else uuid4().hex
        )
        if not request.request_id:
            raise ValueError("request_id must not be empty")
        if not isinstance(request.num_frames, int) or isinstance(request.num_frames, bool):
            raise TypeError("num_frames must be an integer")
        if request.num_frames <= 0 or request.num_frames > self.max_frames:
            raise ValueError(f"num_frames must be in [1, {self.max_frames}]")
        request.fps = float(request.fps)
        if not np.isfinite(request.fps) or request.fps <= 0:
            raise ValueError("fps must be finite and > 0")
        if not isinstance(request.seed, int) or isinstance(request.seed, bool):
            raise TypeError("seed must be an integer")
        if request.metadata is not None and not isinstance(request.metadata, dict):
            raise TypeError("metadata must be a dictionary or null")
        return request

    def _failure_response(
        self,
        request_id: str | None,
        exc: Exception,
        total_started: float,
    ) -> dict[str, Any]:
        response = {
            "ok": False,
            "request_id": request_id,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "total_seconds": time.perf_counter() - total_started,
        }
        self.failed_count += 1
        self.last_request = response
        return response

    def generate(
        self, request: TextMotionRequest | dict[str, Any]
    ) -> dict[str, Any]:
        """Generate, validate and atomically publish one request without unloading models."""
        total_started = time.perf_counter()
        request_id = (
            request.get("request_id")
            if isinstance(request, dict)
            else getattr(request, "request_id", None)
        )
        temporary_dir: Path | None = None
        data = pred = body_params = None
        with self.generation_lock:
            self.request_count += 1
            try:
                if not self.initialized or self.gem_model is None:
                    raise RuntimeError("resident engine is not initialized")
                request = self._validate_request(request)
                request_id = request.request_id
                if self.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(self.device)

                started = time.perf_counter()
                text_embed, cache_hit = self._encode_cached(request.prompt)
                text_encode_seconds = time.perf_counter() - started

                started = time.perf_counter()
                demo = _text_demo_helpers()
                data = demo.build_text_only_data(
                    request.prompt,
                    text_embed,
                    request.num_frames,
                    self.width,
                    self.height,
                    self.bbox_scale,
                )
                input_build_seconds = time.perf_counter() - started

                seed_everything(request.seed)
                started = time.perf_counter()
                with torch.inference_mode():
                    pred = self.gem_model.predict(
                        data,
                        static_cam=True,
                        postproc=self.postproc,
                    )
                body_params = demo.validate_smpl_prediction(pred, request.num_frames)
                body_params = enforce_zero_shape(body_params)
                generation_seconds = time.perf_counter() - started

                started = time.perf_counter()
                output_root = (
                    Path(request.output_root).expanduser()
                    if request.output_root is not None
                    else self.output_root
                )
                prefix = (
                    f"{safe_generation_prefix(request.prompt, limit=48)}"
                    f"_seed{request.seed}"
                )
                temporary_dir, output_dir = make_unique_output_paths(output_root, prefix)
                temporary_dir.mkdir(parents=False, exist_ok=False)
                completed_at = utc_now_iso()
                args = SimpleNamespace(
                    fps=request.fps,
                    seed=request.seed,
                    num_frames=request.num_frames,
                    guidance_scale=self.guidance_scale,
                    ddim_steps=self.ddim_steps,
                    shape_mode=self.shape_mode,
                    width=self.width,
                    height=self.height,
                    bbox_scale=self.bbox_scale,
                    t5_model=self.t5_model,
                )
                demo.save_results(
                    temporary_dir,
                    body_params,
                    data,
                    request.prompt,
                    args,
                    self.ckpt_path,
                    str(self.device),
                    self.text_dtype,
                    completed_at,
                )
                metadata_path = temporary_dir / "metadata.json"
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata["request_id"] = request.request_id
                metadata["request_metadata"] = request.metadata or {}
                metadata_path.write_text(
                    json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                publish_ready_directory(temporary_dir, output_dir, completed_at)
                temporary_dir = None

                self.sequence_number += 1
                latest = {
                    "request_id": request.request_id,
                    "output_dir": str(output_dir.resolve()),
                    "smpl_params": str((output_dir / "smpl_params.pt").resolve()),
                    "motion_npz": str((output_dir / "motion.npz").resolve()),
                    "ready": str((output_dir / "READY").resolve()),
                    "prompt": request.prompt,
                    "num_frames": request.num_frames,
                    "fps": request.fps,
                    "completed_at": completed_at,
                    "sequence_number": self.sequence_number,
                }
                atomic_write_json(self.latest_file, latest)
                save_seconds = time.perf_counter() - started

                total_seconds = time.perf_counter() - total_started
                gpu = get_cuda_memory_snapshot(self.device)
                response = {
                    "ok": True,
                    "request_id": request.request_id,
                    "output_dir": str(output_dir.resolve()),
                    "smpl_params": latest["smpl_params"],
                    "motion_npz": latest["motion_npz"],
                    "num_frames": request.num_frames,
                    "duration_seconds": request.num_frames / request.fps,
                    "timing": {
                        "text_encode_seconds": text_encode_seconds,
                        "input_build_seconds": input_build_seconds,
                        "generation_seconds": generation_seconds,
                        "save_seconds": save_seconds,
                        "total_seconds": total_seconds,
                    },
                    "gpu": {
                        **gpu,
                        "peak_allocated_gib": gpu["max_allocated_gib"],
                    },
                    "text_cache_hit": cache_hit,
                    "sequence_number": self.sequence_number,
                }
                self.successful_count += 1
                self.last_request = response
                self._log_request(request, response)
                return response
            except torch.cuda.OutOfMemoryError as exc:
                if temporary_dir is not None:
                    shutil.rmtree(temporary_dir, ignore_errors=True)
                data = pred = body_params = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return self._failure_response(request_id, exc, total_started)
            except Exception as exc:
                if temporary_dir is not None:
                    shutil.rmtree(temporary_dir, ignore_errors=True)
                return self._failure_response(request_id, exc, total_started)
            finally:
                data = pred = body_params = None

    def _log_request(
        self, request: TextMotionRequest, response: dict[str, Any]
    ) -> None:
        timing = response["timing"]
        gpu = response["gpu"]
        print(f"[Request {self.request_count:04d}]")
        print(f"request_id={request.request_id}")
        print(f"prompt={request.prompt}")
        print(f"frames={request.num_frames}")
        print(f"fps={request.fps:g}")
        print(f"seed={request.seed}")
        print(f"text_cache_hit={response['text_cache_hit']}")
        print(f"text_encode_ms={timing['text_encode_seconds'] * 1000:.2f}")
        print(f"generation_ms={timing['generation_seconds'] * 1000:.2f}")
        print(f"save_ms={timing['save_seconds'] * 1000:.2f}")
        print(f"total_ms={timing['total_seconds'] * 1000:.2f}")
        print(f"output_dir={response['output_dir']}")
        print(f"gpu_allocated_gib={gpu['allocated_gib']:.3f}")
        print(f"gpu_free_gib={gpu['free_gib']:.3f}")

    def warmup(self) -> dict[str, Any]:
        """Run one unpublished request without changing request counters."""
        if not self.initialized or self.gem_model is None:
            raise RuntimeError("initialize models before warmup")
        started = time.perf_counter()
        demo = _text_demo_helpers()
        text_embed, cache_hit = self._encode_cached(self.warmup_prompt)
        data = demo.build_text_only_data(
            self.warmup_prompt,
            text_embed,
            self.warmup_frames,
            self.width,
            self.height,
            self.bbox_scale,
        )
        seed_everything(0)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        with torch.inference_mode():
            pred = self.gem_model.predict(data, static_cam=True, postproc=self.postproc)
        body_params = demo.validate_smpl_prediction(pred, self.warmup_frames)
        enforce_zero_shape(body_params)
        result = {
            "seconds": time.perf_counter() - started,
            "frames": self.warmup_frames,
            "cache_hit": cache_hit,
            "gpu": get_cuda_memory_snapshot(self.device),
        }
        self.startup_timings["warmup_seconds"] = result["seconds"]
        del data, pred, body_params, text_embed
        print("[Resident] Warmup complete")
        print("[Resident] WARMUP COMPLETE")
        self._log_memory("warmup")
        return result

    def status(self) -> dict[str, Any]:
        """Return model, cache, request, timing and memory service status."""
        uptime = (
            time.monotonic() - self.started_monotonic
            if self.started_monotonic is not None
            else 0.0
        )
        device_name = (
            torch.cuda.get_device_name(self.device)
            if self.device.type == "cuda" and torch.cuda.is_available()
            else str(self.device)
        )
        return {
            "initialized": self.initialized,
            "pid": os.getpid(),
            "device": str(self.device),
            "device_name": device_name,
            "checkpoint": str(self.ckpt_path),
            "t5_model": str(self.t5_model),
            "text_dtype": self.text_dtype,
            "ddim_steps": self.ddim_steps,
            "guidance_scale": self.guidance_scale,
            "shape_mode": self.shape_mode,
            "postproc": self.postproc,
            "request_count": self.request_count,
            "successful_count": self.successful_count,
            "failed_count": self.failed_count,
            "embedding_cache_size": len(self.embedding_cache),
            "embedding_cache_capacity": self.embedding_cache_size,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "last_request": self.last_request,
            "startup_timings": dict(self.startup_timings),
            "gpu_memory": get_cuda_memory_snapshot(self.device),
            "uptime_seconds": uptime,
        }

    def close(self) -> None:
        """Release both resident models only during explicit service shutdown."""
        with self.generation_lock:
            self.initialized = False
            self.clear_cache()
            self.denoiser3d = None
            self.gem_model = None
            self.text_encoder = None
            self.tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[Resident] SERVICE STOPPED")


def request_as_dict(request: TextMotionRequest) -> dict[str, Any]:
    """Return a JSON-compatible request dictionary."""
    return asdict(request)
