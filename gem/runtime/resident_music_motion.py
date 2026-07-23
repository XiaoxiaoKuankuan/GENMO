# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Resident GEM-SMPL music-to-motion inference runtime.

The runtime loads the full music-conditioned GEM checkpoint and initializes a
fixed DDIM/CFG sampler once. Each request decodes an audio range, reuses a
thread-safe CPU feature cache, and publishes the existing ``music_only`` READY
artifact contract without loading T5 or changing the robot streaming protocol.
"""

from __future__ import annotations

import copy
import gc
import json
import math
import os
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

import torch

from gem.runtime.artifact_publish import (
    atomic_write_json,
    enforce_zero_shape,
    make_unique_output_paths,
    publish_ready_directory,
    utc_now_iso,
)
from gem.runtime.resident_text_motion import (
    get_cuda_memory_snapshot,
    seed_everything,
)
from gem.utils.music_features import (
    EDGE_FEATURE_DIM,
    EDGE_HOP_LENGTH,
    EDGE_SAMPLE_RATE,
    EDGE_TARGET_FPS,
    extract_edge_baseline35,
)

SUPPORTED_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac"}
FEATURE_CACHE_VERSION = "edge_baseline35_v1"


@dataclass(slots=True)
class MusicMotionRequest:
    """One audio-range request accepted by the resident music engine."""

    audio_path: str
    start_sec: float = 0.0
    duration_sec: float | None = 10.0
    seed: int = 42
    request_id: str | None = None
    metadata: dict[str, Any] | None = None


def _load_gem_model(checkpoint: Path) -> Any:
    """Load GEM without T5 and defer DDIM initialization to the resident engine."""
    try:
        from demo_utils import load_model
    except ModuleNotFoundError:
        from scripts.demo.demo_utils import load_model

    return load_model(
        str(checkpoint),
        load_text_encoder=False,
        defer_diffusion_init=True,
    )


def _validate_checkpoint(checkpoint: Path) -> int:
    """Reuse the single-shot checkpoint music-weight audit."""
    from scripts.demo.demo_music import validate_music_checkpoint

    return validate_music_checkpoint(checkpoint)


def _music_demo_helpers() -> Any:
    """Import single-shot music contracts lazily to avoid a module cycle."""
    from scripts.demo import demo_music

    return demo_music


class ResidentMusicMotionEngine:
    """Keep one music-conditioned GEM-SMPL model resident across requests."""

    def __init__(
        self,
        *,
        ckpt_path: str | Path,
        device: str = "cuda:0",
        ddim_steps: int = 20,
        guidance_scale: float = 2.5,
        width: int = 1280,
        height: int = 720,
        focal: float | None = None,
        output_root: str | Path = "outputs/music_motion",
        postproc: bool = True,
        shape_mode: str = "zero",
        feature_cache_size: int = 32,
        min_free_gib: float = 2.0,
        strict_memory: bool = False,
        warmup_frames: int = 30,
        warmup_enabled: bool = True,
        latest_file: str | Path | None = None,
        max_frames: int = 600,
        allowed_audio_roots: list[str | Path] | tuple[str | Path, ...] | None = None,
        _allow_cpu_for_tests: bool = False,
    ) -> None:
        if ddim_steps <= 0:
            raise ValueError("ddim_steps must be > 0")
        if not math.isfinite(guidance_scale) or guidance_scale < 0:
            raise ValueError("guidance_scale must be finite and >= 0")
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be > 0")
        if focal is not None and (not math.isfinite(focal) or focal <= 0):
            raise ValueError("focal must be finite and > 0")
        if shape_mode != "zero":
            raise ValueError("resident music service only supports shape_mode=zero")
        if feature_cache_size < 0:
            raise ValueError("feature_cache_size must be >= 0")
        if not math.isfinite(min_free_gib) or min_free_gib < 0:
            raise ValueError("min_free_gib must be finite and >= 0")
        if warmup_frames <= 0 or max_frames <= 0:
            raise ValueError("warmup_frames and max_frames must be > 0")

        self.ckpt_path = Path(ckpt_path).expanduser()
        self.device = torch.device(device)
        self.ddim_steps = int(ddim_steps)
        self.guidance_scale = float(guidance_scale)
        self.width = int(width)
        self.height = int(height)
        self.focal = None if focal is None else float(focal)
        self.output_root = Path(output_root).expanduser()
        self.postproc = bool(postproc)
        self.shape_mode = shape_mode
        self.feature_cache_capacity = int(feature_cache_size)
        self.min_free_gib = float(min_free_gib)
        self.strict_memory = bool(strict_memory)
        self.warmup_frames = int(warmup_frames)
        self.warmup_enabled = bool(warmup_enabled)
        self.latest_file = (
            Path(latest_file).expanduser()
            if latest_file is not None
            else self.output_root / "latest_ready.json"
        )
        self.max_frames = int(max_frames)
        self.allowed_audio_roots = self._resolve_allowed_roots(allowed_audio_roots or ())
        self._allow_cpu_for_tests = bool(_allow_cpu_for_tests)

        self.gem_model: Any | None = None
        self.denoiser3d: Any | None = None
        self.generation_lock = threading.RLock()
        self._cache_lock = threading.Lock()
        self._feature_cache: OrderedDict[tuple[Any, ...], tuple[torch.Tensor, dict[str, Any]]] = (
            OrderedDict()
        )
        self.feature_cache_hits = 0
        self.feature_cache_misses = 0
        self.request_count = 0
        self.successful_count = 0
        self.failed_count = 0
        self.startup_timings: dict[str, float] = {}
        self.last_request: dict[str, Any] | None = None
        self.initialized = False
        self.started_monotonic: float | None = None
        self.sequence_number = self._read_previous_sequence()

    @staticmethod
    def _resolve_allowed_roots(values: tuple[str | Path, ...]) -> tuple[Path, ...]:
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
            payload = json.loads(self.latest_file.read_text(encoding="utf-8"))
            return max(0, int(payload.get("sequence_number", 0)))
        except (OSError, TypeError, ValueError):
            return 0

    def _log_memory(self, stage: str) -> dict[str, float]:
        snapshot = get_cuda_memory_snapshot(self.device)
        print(
            f"[ResidentMusic] GPU {stage}: "
            f"allocated={snapshot['allocated_gib']:.3f} GiB "
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
        """Validate, load and initialize the fixed GEM/DDIM stack exactly once."""
        with self.generation_lock:
            if self.initialized:
                return
            if self.device.type != "cuda" and not self._allow_cpu_for_tests:
                raise RuntimeError("Resident GEM music service requires a CUDA device")
            if self.device.type == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is not available for the resident music service")
                torch.cuda.set_device(self.device)
                torch.cuda.reset_peak_memory_stats(self.device)

            self.started_monotonic = time.monotonic()
            started = time.perf_counter()
            print(f"[ResidentMusic] Validating checkpoint: {self.ckpt_path}")
            checkpoint_dim = _validate_checkpoint(self.ckpt_path)
            if checkpoint_dim != EDGE_FEATURE_DIM:
                raise RuntimeError(
                    f"Checkpoint music input dimension is {checkpoint_dim}, expected "
                    f"{EDGE_FEATURE_DIM}"
                )
            self.startup_timings["checkpoint_seconds"] = time.perf_counter() - started

            started = time.perf_counter()
            print("[ResidentMusic] Loading GEM-SMPL once (T5 disabled)")
            self.gem_model = _load_gem_model(self.ckpt_path)
            self.gem_model = self.gem_model.to(self.device).eval()
            demo = _music_demo_helpers()
            model_dim = demo.inspect_model_music_input_dim(self.gem_model)
            if model_dim != checkpoint_dim:
                raise RuntimeError(
                    "Music dimension mismatch between checkpoint and loaded model: "
                    f"{checkpoint_dim}/{model_dim}"
                )
            self.denoiser3d = self.gem_model.pipeline.denoiser3d
            self.startup_timings["gem_load_seconds"] = time.perf_counter() - started
            self._log_memory("GEM load")

            started = time.perf_counter()
            diffusion = self.denoiser3d.model_cfg.diffusion
            diffusion.guidance_param = self.guidance_scale
            diffusion.test_timestep_respacing = str(self.ddim_steps)
            diffusion.gen_only_test_timestep_respacing = str(self.ddim_steps)
            self.denoiser3d.init_diffusion()
            self.startup_timings["ddim_init_seconds"] = time.perf_counter() - started
            print("[ResidentMusic] DDIM initialized once")

            self.initialized = True
            self._check_free_memory("model initialization")
            if self.warmup_enabled:
                self.warmup()
            self.startup_timings["total_seconds"] = time.monotonic() - self.started_monotonic
            self._check_free_memory("warmup")
            print("[ResidentMusic] SERVICE READY")

    def _validate_audio_path(self, value: str) -> Path:
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

    def _validate_request(self, request: MusicMotionRequest | dict[str, Any]) -> MusicMotionRequest:
        if isinstance(request, dict):
            allowed = {
                "audio_path",
                "start_sec",
                "duration_sec",
                "seed",
                "request_id",
                "metadata",
            }
            unknown = sorted(set(request) - allowed)
            if unknown:
                raise ValueError(f"unsupported request fields: {unknown}")
            request = MusicMotionRequest(**request)
        if not isinstance(request, MusicMotionRequest):
            raise TypeError("request must be MusicMotionRequest or a request dictionary")

        request.audio_path = str(self._validate_audio_path(request.audio_path))
        request.start_sec = float(request.start_sec)
        if not math.isfinite(request.start_sec) or request.start_sec < 0:
            raise ValueError("start_sec must be finite and >= 0")
        if request.duration_sec is not None:
            request.duration_sec = float(request.duration_sec)
            if not math.isfinite(request.duration_sec) or request.duration_sec <= 0:
                raise ValueError("duration_sec must be null or finite and > 0")
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

    @staticmethod
    def _stable_time_key(value: float | None) -> int | None:
        return None if value is None else int(round(value * 1_000_000))

    def _cache_key(
        self, path: Path, start_sec: float, duration_sec: float | None
    ) -> tuple[Any, ...]:
        stat = path.stat()
        return (
            str(path),
            int(stat.st_dev),
            int(stat.st_ino),
            int(stat.st_size),
            int(stat.st_mtime_ns),
            self._stable_time_key(start_sec),
            self._stable_time_key(duration_sec),
            FEATURE_CACHE_VERSION,
            EDGE_SAMPLE_RATE,
            EDGE_HOP_LENGTH,
            EDGE_TARGET_FPS,
        )

    def _get_features(
        self,
        path: Path,
        start_sec: float,
        duration_sec: float | None,
    ) -> tuple[torch.Tensor, dict[str, Any], bool]:
        key = self._cache_key(path, start_sec, duration_sec)
        with self._cache_lock:
            cached = self._feature_cache.get(key)
            if cached is not None:
                self._feature_cache.move_to_end(key)
                self.feature_cache_hits += 1
                features, metadata = cached
                return features, copy.deepcopy(metadata), True
            self.feature_cache_misses += 1

        features, metadata = extract_edge_baseline35(
            path,
            start_sec=start_sec,
            duration_sec=duration_sec,
            target_fps=EDGE_TARGET_FPS,
        )
        features = features.detach().cpu().to(dtype=torch.float32).contiguous()
        if features.ndim != 2 or features.shape[1] != EDGE_FEATURE_DIM:
            raise RuntimeError(
                f"EDGE features must have shape [L, {EDGE_FEATURE_DIM}], "
                f"got {tuple(features.shape)}"
            )
        if not torch.isfinite(features).all():
            raise RuntimeError("EDGE music features contain NaN or Inf")
        metadata = copy.deepcopy(metadata)
        with self._cache_lock:
            if self.feature_cache_capacity > 0:
                self._feature_cache[key] = (features, copy.deepcopy(metadata))
                self._feature_cache.move_to_end(key)
                while len(self._feature_cache) > self.feature_cache_capacity:
                    self._feature_cache.popitem(last=False)
        return features, metadata, False

    def clear_cache(self) -> int:
        """Clear only CPU music features, keeping the resident GEM model intact."""
        with self._cache_lock:
            removed = len(self._feature_cache)
            self._feature_cache.clear()
        return removed

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

    def generate(self, request: MusicMotionRequest | dict[str, Any]) -> dict[str, Any]:
        """Generate and atomically publish one audio-conditioned motion."""
        total_started = time.perf_counter()
        request_id = (
            request.get("request_id")
            if isinstance(request, dict)
            else getattr(request, "request_id", None)
        )
        temporary_dir: Path | None = None
        data = pred = groups = raw_motion = None
        with self.generation_lock:
            self.request_count += 1
            try:
                if not self.initialized or self.gem_model is None:
                    raise RuntimeError("resident music engine is not initialized")
                request = self._validate_request(request)
                request_id = request.request_id
                audio_path = Path(request.audio_path)
                if self.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(self.device)

                started = time.perf_counter()
                features, feature_metadata, cache_hit = self._get_features(
                    audio_path,
                    request.start_sec,
                    request.duration_sec,
                )
                feature_extract_seconds = time.perf_counter() - started
                length = int(features.shape[0])
                if length > self.max_frames:
                    raise RuntimeError(
                        f"Selected audio produced {length} frames, exceeding "
                        f"max_frames={self.max_frames}. Use start_sec/duration_sec to "
                        "select a shorter range; long motions are not silently stitched."
                    )

                started = time.perf_counter()
                demo = _music_demo_helpers()
                data = demo.build_music_only_data(
                    features,
                    width=self.width,
                    height=self.height,
                    focal=self.focal,
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
                body_global = demo._validate_body_group(
                    pred.get("body_params_global"), "body_params_global", length
                )
                body_incam = demo._validate_body_group(
                    pred.get("body_params_incam"), "body_params_incam", length
                )
                groups = enforce_zero_shape(
                    {
                        "body_params_global": body_global,
                        "body_params_incam": body_incam,
                    }
                )
                raw_motion = demo.extract_motion_151d(pred, length)
                generation_seconds = time.perf_counter() - started

                started = time.perf_counter()
                temporary_dir, output_dir = make_unique_output_paths(
                    self.output_root,
                    demo.music_generation_prefix(audio_path, request.start_sec, request.seed),
                )
                provisional_completed_at = utc_now_iso()
                args = SimpleNamespace(
                    start_sec=request.start_sec,
                    width=self.width,
                    height=self.height,
                    focal=self.focal,
                    guidance_scale=self.guidance_scale,
                    ddim_steps=self.ddim_steps,
                    no_postproc=not self.postproc,
                )
                metadata = demo.build_music_metadata(
                    args=args,
                    audio_path=audio_path,
                    checkpoint=self.ckpt_path,
                    feature_metadata=feature_metadata,
                    sample_seed=request.seed,
                    sample_index=0,
                    num_frames=length,
                    render_succeeded=False,
                    audio_mux_succeeded=False,
                    completed_at=provisional_completed_at,
                )
                metadata.update(
                    {
                        "request_id": request.request_id,
                        "request_metadata": copy.deepcopy(request.metadata or {}),
                        "service": "resident_music_motion",
                        "audio_decode_mode": feature_metadata.get("audio_decode_mode", "unknown"),
                    }
                )
                demo.write_music_artifacts(
                    temporary_dir,
                    body_global=groups["body_params_global"],
                    body_incam=groups["body_params_incam"],
                    raw_motion_151d=raw_motion,
                    music_features=features,
                    data=data,
                    metadata=metadata,
                )
                completed_at = utc_now_iso()
                metadata["completed_at"] = completed_at
                demo.update_saved_metadata(temporary_dir, metadata)
                publish_ready_directory(temporary_dir, output_dir, completed_at)
                temporary_dir = None

                self.sequence_number += 1
                latest = {
                    "request_id": request.request_id,
                    "output_dir": str(output_dir.resolve()),
                    "smpl_params": str((output_dir / "smpl_params.pt").resolve()),
                    "motion_npz": str((output_dir / "motion.npz").resolve()),
                    "music_features": str((output_dir / "music_features.pt").resolve()),
                    "ready": str((output_dir / "READY").resolve()),
                    "audio_path": str(audio_path),
                    "audio_start_sec": request.start_sec,
                    "audio_duration_sec": float(feature_metadata["selected_duration_sec"]),
                    "num_frames": length,
                    "fps": EDGE_TARGET_FPS,
                    "seed": request.seed,
                    "estimated_bpm": float(feature_metadata["estimated_or_prior_bpm"]),
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
                    "audio_path": str(audio_path),
                    "output_dir": latest["output_dir"],
                    "smpl_params": latest["smpl_params"],
                    "motion_npz": latest["motion_npz"],
                    "music_features": latest["music_features"],
                    "num_frames": length,
                    "fps": EDGE_TARGET_FPS,
                    "duration_seconds": length / EDGE_TARGET_FPS,
                    "estimated_bpm": latest["estimated_bpm"],
                    "feature_cache_hit": cache_hit,
                    "sequence_number": self.sequence_number,
                    "timing": {
                        "feature_extract_seconds": feature_extract_seconds,
                        "input_build_seconds": input_build_seconds,
                        "generation_seconds": generation_seconds,
                        "save_seconds": save_seconds,
                        "total_seconds": total_seconds,
                    },
                    "gpu": {
                        **gpu,
                        "peak_allocated_gib": gpu["max_allocated_gib"],
                    },
                }
                self.successful_count += 1
                self.last_request = response
                self._log_request(request, response)
                return response
            except torch.cuda.OutOfMemoryError as exc:
                if temporary_dir is not None:
                    shutil.rmtree(temporary_dir, ignore_errors=True)
                data = pred = groups = raw_motion = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return self._failure_response(request_id, exc, total_started)
            except Exception as exc:
                if temporary_dir is not None:
                    shutil.rmtree(temporary_dir, ignore_errors=True)
                return self._failure_response(request_id, exc, total_started)
            finally:
                data = pred = groups = raw_motion = None

    def _log_request(self, request: MusicMotionRequest, response: dict[str, Any]) -> None:
        timing = response["timing"]
        print(f"[ResidentMusic Request {self.request_count:04d}]")
        print(f"request_id={request.request_id}")
        print(f"audio={request.audio_path}")
        print(f"range={request.start_sec:g}s + {request.duration_sec}")
        print(f"frames={response['num_frames']} fps={EDGE_TARGET_FPS}")
        print(f"seed={request.seed}")
        print(f"feature_cache_hit={response['feature_cache_hit']}")
        print(f"feature_extract_ms={timing['feature_extract_seconds'] * 1000:.2f}")
        print(f"generation_ms={timing['generation_seconds'] * 1000:.2f}")
        print(f"save_ms={timing['save_seconds'] * 1000:.2f}")
        print(f"total_ms={timing['total_seconds'] * 1000:.2f}")
        print(f"output_dir={response['output_dir']}")

    def warmup(self) -> dict[str, Any]:
        """Run one deterministic unpublished music request."""
        if not self.initialized or self.gem_model is None:
            raise RuntimeError("initialize the GEM model before warmup")
        started = time.perf_counter()
        demo = _music_demo_helpers()
        features = torch.zeros(self.warmup_frames, EDGE_FEATURE_DIM, dtype=torch.float32)
        features[::10, 33] = 1.0
        features[::15, 34] = 1.0
        data = demo.build_music_only_data(
            features,
            width=self.width,
            height=self.height,
            focal=self.focal,
        )
        seed_everything(0)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        with torch.inference_mode():
            pred = self.gem_model.predict(
                data,
                static_cam=True,
                postproc=self.postproc,
            )
        groups = enforce_zero_shape(
            {
                "body_params_global": demo._validate_body_group(
                    pred.get("body_params_global"),
                    "body_params_global",
                    self.warmup_frames,
                ),
                "body_params_incam": demo._validate_body_group(
                    pred.get("body_params_incam"),
                    "body_params_incam",
                    self.warmup_frames,
                ),
            }
        )
        for group in groups.values():
            if torch.count_nonzero(group["betas"]).item() != 0:
                raise AssertionError("warmup did not enforce zero betas")
        result = {
            "seconds": time.perf_counter() - started,
            "frames": self.warmup_frames,
            "gpu": get_cuda_memory_snapshot(self.device),
        }
        self.startup_timings["warmup_seconds"] = result["seconds"]
        del data, pred, groups, features
        print("[ResidentMusic] WARMUP COMPLETE")
        self._log_memory("warmup")
        return result

    def status(self) -> dict[str, Any]:
        """Return model, cache, request, timing and memory service status."""
        with self._cache_lock:
            cache_size = len(self._feature_cache)
            cache_bytes = sum(
                features.numel() * features.element_size()
                for features, _metadata in self._feature_cache.values()
            )
        uptime = (
            time.monotonic() - self.started_monotonic if self.started_monotonic is not None else 0.0
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
            "ddim_steps": self.ddim_steps,
            "guidance_scale": self.guidance_scale,
            "shape_mode": self.shape_mode,
            "postproc": self.postproc,
            "request_count": self.request_count,
            "successful_count": self.successful_count,
            "failed_count": self.failed_count,
            "feature_cache_size": cache_size,
            "feature_cache_capacity": self.feature_cache_capacity,
            "feature_cache_hits": self.feature_cache_hits,
            "feature_cache_misses": self.feature_cache_misses,
            "feature_cache_bytes": cache_bytes,
            "allowed_audio_roots": [str(root) for root in self.allowed_audio_roots],
            "last_request": self.last_request,
            "startup_timings": dict(self.startup_timings),
            "gpu_memory": get_cuda_memory_snapshot(self.device),
            "uptime_seconds": uptime,
        }

    def close(self) -> None:
        """Release the resident model only during explicit service shutdown."""
        with self.generation_lock:
            self.initialized = False
            self.clear_cache()
            self.denoiser3d = None
            self.gem_model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[ResidentMusic] SERVICE STOPPED")


def request_as_dict(request: MusicMotionRequest) -> dict[str, Any]:
    """Return a JSON-compatible request dictionary."""
    return asdict(request)
