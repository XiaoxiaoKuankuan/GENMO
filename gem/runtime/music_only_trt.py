# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""TensorRT/DDIM sliding-window runtime for physical music-only deployment.

TensorRT owns exactly one CFG-guided x-start prediction.  The deterministic
20-step DDIM scheduler and hard overlap inpainting stay in PyTorch CUDA so a
window never performs a device-to-host round trip between denoising steps.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import json
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol

import numpy as np
import torch

from gem.diffusion_utils.model_util import create_gaussian_diffusion
from gem.utils.rotation_conversions import axis_angle_to_matrix

WINDOW_FRAMES = 120
OVERLAP_FRAMES = 30
STRIDE_FRAMES = WINDOW_FRAMES - OVERLAP_FRAMES
MUSIC_DIM = 35
MOTION_DIM = 151
SOURCE_FPS = 30
DEFAULT_DDIM_STEPS = 20
DEFAULT_GUIDANCE_SCALE = 2.5


def _parse_version(value: str) -> tuple[int, int, int]:
    parts = str(value).split(".")
    if len(parts) < 2 or not all(part.isdigit() for part in parts[:2]):
        raise RuntimeError(f"cannot parse TensorRT version {value!r}")
    patch = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
    return int(parts[0]), int(parts[1]), patch


def linked_tensorrt_version() -> str:
    """Read the actual local ``libnvinfer`` version, not package metadata."""
    library_path = ctypes.util.find_library("nvinfer")
    if not library_path:
        raise RuntimeError("libnvinfer is not visible to the dynamic linker")
    library = ctypes.CDLL(library_path)
    try:
        get_version = library.getInferLibVersion
    except AttributeError as exc:
        raise RuntimeError(
            f"{library_path} does not export getInferLibVersion"
        ) from exc
    get_version.restype = ctypes.c_int32
    encoded = int(get_version())
    if encoded <= 0:
        raise RuntimeError(f"libnvinfer returned an invalid version integer: {encoded}")
    major = encoded // 10_000
    minor = encoded % 10_000 // 100
    patch = encoded % 100
    return f"{major}.{minor}.{patch}"


def validate_tensorrt_installation(trt_module: object) -> str:
    """Require the Python binding and linked runtime to share major/minor ABI."""
    binding = str(getattr(trt_module, "__version__", ""))
    runtime = linked_tensorrt_version()
    if _parse_version(binding)[:2] != _parse_version(runtime)[:2]:
        raise RuntimeError(
            "TensorRT Python binding/runtime mismatch: "
            f"binding={binding}, libnvinfer={runtime}. Install a binding matching "
            "the local libnvinfer major/minor version."
        )
    return runtime


class DenoiserStep(Protocol):
    """Callable contract shared by TensorRT and CPU/GPU test doubles."""

    def __call__(
        self,
        noisy_motion: torch.Tensor,
        diffusion_timestep: torch.Tensor,
        music: torch.Tensor,
        length: torch.Tensor,
        guidance_scale: torch.Tensor,
    ) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class SlidingWindow:
    index: int
    start: int
    valid_length: int
    known_length: int

    @property
    def end(self) -> int:
        return self.start + self.valid_length

    @property
    def new_start(self) -> int:
        return self.known_length

    @property
    def new_length(self) -> int:
        return self.valid_length - self.known_length


def plan_sliding_windows(num_frames: int) -> list[SlidingWindow]:
    """Plan fixed 120-frame windows with an exact 30-frame overlap."""
    if num_frames <= 0:
        raise ValueError("num_frames must be > 0")
    result: list[SlidingWindow] = []
    start = 0
    index = 0
    while start < num_frames:
        valid = min(WINDOW_FRAMES, num_frames - start)
        known = 0 if index == 0 else min(OVERLAP_FRAMES, valid)
        if index > 0 and valid <= known:
            break
        result.append(SlidingWindow(index, start, valid, known))
        if start + valid >= num_frames:
            break
        start += STRIDE_FRAMES
        index += 1
    return result


def exact_motion_frame_count(
    feature_frames: int,
    duration_sec: float | None,
    *,
    fps: int = SOURCE_FPS,
) -> int:
    """Return a half-open duration in frames (20 seconds is exactly 600)."""
    if feature_frames <= 0:
        raise ValueError("feature_frames must be > 0")
    if fps <= 0:
        raise ValueError("fps must be > 0")
    if duration_sec is None:
        return int(feature_frames)
    if not math.isfinite(float(duration_sec)) or duration_sec <= 0.0:
        raise ValueError("duration_sec must be finite and > 0")
    requested = max(1, int(math.floor(float(duration_sec) * fps + 1e-7)))
    if requested > int(feature_frames):
        raise ValueError(
            f"selected audio only produced {feature_frames} feature frames, "
            f"fewer than the requested {requested} frames"
        )
    return requested


def padded_music_window(features: torch.Tensor, window: SlidingWindow) -> torch.Tensor:
    """Return one finite ``[120,35]`` window, padding only outside valid length."""
    if features.ndim != 2 or features.shape[1] != MUSIC_DIM:
        raise ValueError(f"features must have shape [T,{MUSIC_DIM}]")
    selected = features[window.start : window.end].to(dtype=torch.float32)
    if len(selected) != window.valid_length:
        raise ValueError("window is outside the music feature timeline")
    if not torch.isfinite(selected).all():
        raise ValueError("music features contain NaN or Inf")
    if len(selected) < WINDOW_FRAMES:
        selected = torch.cat(
            (selected, selected[-1:].expand(WINDOW_FRAMES - len(selected), -1)), dim=0
        )
    return selected.contiguous()


def derive_window_seed(request_seed: int, window_index: int) -> int:
    payload = f"genmo-trt-v1:{int(request_seed)}:{int(window_index)}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & 0x7FFF_FFFF


def _diffusion_config(steps: int) -> SimpleNamespace:
    if not 2 <= int(steps) <= 1000:
        raise ValueError("DDIM steps must be in 2..1000")
    return SimpleNamespace(
        train_timestep_respacing="",
        test_timestep_respacing=str(int(steps)),
        noise_schedule="cosine",
        sigma_small=True,
    )


class SlidingDDIMGenerator:
    """Deterministic DDIM with hard x0 overlap inpainting at every step."""

    def __init__(
        self,
        denoiser: DenoiserStep,
        *,
        device: torch.device | str,
        steps: int = DEFAULT_DDIM_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        motion_dim: int = MOTION_DIM,
    ) -> None:
        if not math.isfinite(float(guidance_scale)) or guidance_scale < 0.0:
            raise ValueError("guidance_scale must be finite and >= 0")
        if int(motion_dim) <= 0:
            raise ValueError("motion_dim must be > 0")
        self.denoiser = denoiser
        self.device = torch.device(device)
        self.steps = int(steps)
        self.guidance_scale = float(guidance_scale)
        self.motion_dim = int(motion_dim)
        self.diffusion = create_gaussian_diffusion(_diffusion_config(self.steps), training=False)
        if self.diffusion.num_timesteps != self.steps:
            raise RuntimeError("diffusion respacing did not produce the requested step count")
        self._alphas_cumprod = torch.as_tensor(
            self.diffusion.alphas_cumprod, device=self.device, dtype=torch.float32
        )
        self._alphas_cumprod_prev = torch.as_tensor(
            self.diffusion.alphas_cumprod_prev, device=self.device, dtype=torch.float32
        )
        self._timestep_map = torch.as_tensor(
            self.diffusion.timestep_map, device=self.device, dtype=torch.long
        )

    def _q_sample_at(
        self,
        x0: torch.Tensor,
        step: int,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        alpha = self._alphas_cumprod[step].to(dtype=x0.dtype)
        return alpha.sqrt() * x0 + (1.0 - alpha).sqrt() * noise

    @torch.inference_mode()
    def generate_window(
        self,
        music: torch.Tensor,
        *,
        valid_length: int,
        seed: int,
        known_x0: torch.Tensor | None = None,
        trace_hook: Callable[[int, torch.Tensor, torch.Tensor | None], None] | None = None,
    ) -> torch.Tensor:
        """Generate one normalized ``[120,motion_dim]`` window.

        ``trace_hook`` receives ``(step, x_t_after_overwrite, pred_x0)`` and is
        intended for contract tests and diagnostics only.
        """
        if music.shape != (WINDOW_FRAMES, MUSIC_DIM):
            raise ValueError(f"music must have shape [{WINDOW_FRAMES},{MUSIC_DIM}]")
        if not 1 <= int(valid_length) <= WINDOW_FRAMES:
            raise ValueError("valid_length must be in 1..120")
        if known_x0 is not None:
            if known_x0.shape != (OVERLAP_FRAMES, self.motion_dim):
                raise ValueError(
                    f"known_x0 must have shape [{OVERLAP_FRAMES},{self.motion_dim}]"
                )
            if valid_length <= OVERLAP_FRAMES:
                raise ValueError("an inpainted window must contain at least one new frame")

        music_b = music.to(self.device, dtype=torch.float32).unsqueeze(0).contiguous()
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))
        x_t = torch.randn(
            (1, WINDOW_FRAMES, self.motion_dim),
            device=self.device,
            dtype=torch.float32,
            generator=generator,
        )
        known = None if known_x0 is None else known_x0.to(self.device).float().unsqueeze(0)
        known_noise = None if known is None else x_t[:, :OVERLAP_FRAMES].clone()
        length = torch.tensor([int(valid_length)], device=self.device, dtype=torch.long)
        guidance = torch.tensor(
            [self.guidance_scale], device=self.device, dtype=torch.float32
        )

        for step in range(self.steps - 1, -1, -1):
            if known is not None:
                known_xt = self._q_sample_at(known, step, known_noise)
                x_t[:, :OVERLAP_FRAMES] = known_xt
            if trace_hook is not None:
                trace_hook(step, x_t.detach().clone(), None)

            timestep = self._timestep_map[step : step + 1]
            pred_x0 = self.denoiser(x_t, timestep, music_b, length, guidance).float()
            if pred_x0.shape != x_t.shape or not torch.isfinite(pred_x0).all():
                raise RuntimeError(
                    f"denoiser returned invalid pred_motion {tuple(pred_x0.shape)}"
                )
            if known is not None:
                pred_x0[:, :OVERLAP_FRAMES] = known

            alpha = self._alphas_cumprod[step].to(dtype=x_t.dtype)
            alpha_prev = self._alphas_cumprod_prev[step].to(dtype=x_t.dtype)
            eps = (x_t - alpha.sqrt() * pred_x0) / (1.0 - alpha).sqrt().clamp_min(1e-12)
            x_t = alpha_prev.sqrt() * pred_x0 + (1.0 - alpha_prev).sqrt() * eps

            if known is not None:
                if step == 0:
                    x_t[:, :OVERLAP_FRAMES] = known
                else:
                    x_t[:, :OVERLAP_FRAMES] = self._q_sample_at(
                        known, step - 1, known_noise
                    )
            if trace_hook is not None:
                trace_hook(step, x_t.detach().clone(), pred_x0.detach().clone())

        if known is not None:
            x_t[:, :OVERLAP_FRAMES] = known
        return x_t[0].contiguous()


class StreamingSmplDecoder:
    """Decode only committed frames while preserving exact root rollout state."""

    def __init__(self, endecoder: torch.nn.Module, device: torch.device | str) -> None:
        self.endecoder = endecoder.to(device).eval()
        self.device = torch.device(device)
        self.reset()

    def reset(self) -> None:
        self._previous_orient: torch.Tensor | None = None
        self._previous_local_velocity: torch.Tensor | None = None
        self._translation: torch.Tensor | None = None
        self.frames_decoded = 0

    @torch.inference_mode()
    def decode_new(
        self,
        normalized_motion: torch.Tensor,
        *,
        start: int,
        end: int,
    ) -> dict[str, torch.Tensor]:
        if normalized_motion.ndim != 2 or normalized_motion.shape[1] != MOTION_DIM:
            raise ValueError(f"normalized_motion must have shape [T,{MOTION_DIM}]")
        if not 0 <= start < end <= len(normalized_motion):
            raise ValueError("invalid committed motion slice")
        decoded = self.endecoder.decode(
            normalized_motion[start:end].to(self.device).float().unsqueeze(0)
        )
        body_pose = decoded["body_pose"][0]
        orient = decoded["global_orient_gv"][0]
        local_velocity = decoded["local_transl_vel"][0]
        translations: list[torch.Tensor] = []
        for current_orient, current_velocity in zip(orient, local_velocity):
            if self._translation is None:
                self._translation = torch.zeros(3, device=self.device, dtype=torch.float32)
            else:
                assert self._previous_orient is not None
                assert self._previous_local_velocity is not None
                rotation = axis_angle_to_matrix(self._previous_orient)
                self._translation = self._translation + rotation @ self._previous_local_velocity
            translations.append(self._translation.clone())
            self._previous_orient = current_orient.detach().clone()
            self._previous_local_velocity = current_velocity.detach().clone()
            self.frames_decoded += 1
        transl = torch.stack(translations)
        return {
            "body_pose": body_pose,
            "global_orient": orient,
            "transl": transl,
            "betas": torch.zeros(
                len(body_pose), 10, device=self.device, dtype=body_pose.dtype
            ),
        }


def sha256_file(path: str | Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def gpu_fingerprint(device: torch.device | str = "cuda:0") -> dict[str, object]:
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("TensorRT physical deployment requires CUDA")
    index = resolved.index if resolved.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_memory": properties.total_memory,
        "torch_cuda": torch.version.cuda,
    }


def engine_cache_key(
    *,
    onnx_sha256: str,
    checkpoint_sha256: str,
    tensorrt_version: str,
    precision: str,
    gpu: dict[str, object],
) -> str:
    value = {
        "contract": "gem_music_only_trt_v2",
        "onnx_sha256": onnx_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "tensorrt_version": str(tensorrt_version),
        "precision": str(precision),
        "gpu": gpu,
        "inputs": [1, WINDOW_FRAMES, MOTION_DIM],
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class TensorRTStepRunner:
    """One resident TensorRT execution context with persistent CUDA tensors."""

    REQUIRED_INPUTS = {
        "noisy_motion": (1, WINDOW_FRAMES, MOTION_DIM),
        "diffusion_timestep": (1,),
        "music": (1, WINDOW_FRAMES, MUSIC_DIM),
        "length": (1,),
        "guidance_scale": (1,),
    }
    REQUIRED_OUTPUT = (1, WINDOW_FRAMES, MOTION_DIM)

    def __init__(
        self,
        engine_path: str | Path,
        *,
        device: torch.device | str = "cuda:0",
        use_cuda_graph: bool = True,
        require_manifest: bool = True,
    ) -> None:
        self.engine_path = Path(engine_path).expanduser().resolve(strict=True)
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("TensorRTStepRunner requires a CUDA device")
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT Python bindings are missing. Install bindings matching the "
                "deployment machine's libnvinfer before starting physical mode."
            ) from exc
        self.linked_tensorrt_version = validate_tensorrt_installation(trt)
        self.trt = trt
        self.manifest = self._validate_manifest(require_manifest)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine: {self.engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("failed to create TensorRT execution context")
        self._lock = threading.Lock()
        self._buffers: dict[str, torch.Tensor] = {}
        self._input_names: set[str] = set()
        self._output_name: str | None = None
        self._allocate_and_bind()
        self.cuda_graph: torch.cuda.CUDAGraph | None = None
        if use_cuda_graph:
            self._try_capture_cuda_graph()

    def _validate_manifest(self, required: bool) -> dict[str, object] | None:
        manifest_path = self.engine_path.parent / "engine.json"
        if not manifest_path.is_file():
            if required:
                raise RuntimeError(
                    f"TensorRT engine manifest is missing: {manifest_path}. "
                    "Physical mode only accepts fingerprinted engines."
                )
            return None
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("contract_version") != "gem_music_only_trt_engine_v1":
            raise RuntimeError("unsupported TensorRT engine manifest contract")
        if payload.get("engine_sha256") != sha256_file(self.engine_path):
            raise RuntimeError("TensorRT engine SHA256 does not match its manifest")
        expected_version = str(payload.get("tensorrt_version", ""))
        actual_version = str(self.trt.__version__)
        if expected_version.split(".")[:2] != actual_version.split(".")[:2]:
            raise RuntimeError(
                f"TensorRT engine was built with {expected_version}, runtime is {actual_version}"
            )
        expected_library = str(payload.get("libnvinfer_version", expected_version))
        if expected_library.split(".")[:2] != self.linked_tensorrt_version.split(".")[:2]:
            raise RuntimeError(
                "TensorRT engine libnvinfer mismatch: "
                f"built with {expected_library}, runtime is {self.linked_tensorrt_version}"
            )
        expected_gpu = payload.get("gpu", {})
        actual_gpu = gpu_fingerprint(self.device)
        if (
            expected_gpu.get("name") != actual_gpu["name"]
            or expected_gpu.get("compute_capability") != actual_gpu["compute_capability"]
        ):
            raise RuntimeError(
                f"TensorRT engine GPU fingerprint mismatch: {expected_gpu} != {actual_gpu}"
            )
        expected_cache_key = engine_cache_key(
            onnx_sha256=str(payload.get("onnx_sha256", "")),
            checkpoint_sha256=str(payload.get("checkpoint_sha256", "")),
            tensorrt_version=expected_version,
            precision=str(payload.get("precision", "")),
            gpu=actual_gpu,
        )
        if payload.get("cache_key") != expected_cache_key:
            raise RuntimeError(
                "TensorRT engine cache fingerprint does not match this runtime/GPU"
            )
        return payload

    @staticmethod
    def _torch_dtype(np_dtype: np.dtype) -> torch.dtype:
        mapping = {
            np.dtype(np.float32): torch.float32,
            np.dtype(np.float16): torch.float16,
            np.dtype(np.int64): torch.int64,
            np.dtype(np.int32): torch.int32,
            np.dtype(np.bool_): torch.bool,
        }
        try:
            return mapping[np_dtype]
        except KeyError as exc:
            raise RuntimeError(f"unsupported TensorRT tensor dtype: {np_dtype}") from exc

    def _allocate_and_bind(self) -> None:
        trt = self.trt
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            shape = tuple(int(value) for value in self.engine.get_tensor_shape(name))
            if any(value <= 0 for value in shape):
                raise RuntimeError(f"TensorRT engine has a dynamic/unresolved shape for {name}")
            dtype = self._torch_dtype(np.dtype(trt.nptype(self.engine.get_tensor_dtype(name))))
            tensor = torch.empty(shape, dtype=dtype, device=self.device).contiguous()
            self._buffers[name] = tensor
            self.context.set_tensor_address(name, tensor.data_ptr())
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._input_names.add(name)
            else:
                if self._output_name is not None:
                    raise RuntimeError("deployment engine must expose exactly one output")
                self._output_name = name
        if self._input_names != set(self.REQUIRED_INPUTS):
            raise RuntimeError(
                f"TensorRT input contract mismatch: {sorted(self._input_names)}"
            )
        for name, expected in self.REQUIRED_INPUTS.items():
            if tuple(self._buffers[name].shape) != expected:
                raise RuntimeError(
                    f"TensorRT input {name} must be {expected}, got {tuple(self._buffers[name].shape)}"
                )
        if self._output_name != "pred_motion" or tuple(
            self._buffers[self._output_name].shape
        ) != self.REQUIRED_OUTPUT:
            raise RuntimeError("TensorRT pred_motion output contract mismatch")

    def _execute(self) -> None:
        stream = torch.cuda.current_stream(self.device)
        if not self.context.execute_async_v3(stream_handle=stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 returned false")

    def _try_capture_cuda_graph(self) -> None:
        try:
            self._execute()
            torch.cuda.synchronize(self.device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._execute()
            self.cuda_graph = graph
        except Exception:
            self.cuda_graph = None

    def __call__(
        self,
        noisy_motion: torch.Tensor,
        diffusion_timestep: torch.Tensor,
        music: torch.Tensor,
        length: torch.Tensor,
        guidance_scale: torch.Tensor,
    ) -> torch.Tensor:
        values = {
            "noisy_motion": noisy_motion,
            "diffusion_timestep": diffusion_timestep,
            "music": music,
            "length": length,
            "guidance_scale": guidance_scale,
        }
        with self._lock:
            for name, value in values.items():
                destination = self._buffers[name]
                if tuple(value.shape) != tuple(destination.shape):
                    raise ValueError(
                        f"{name} must be {tuple(destination.shape)}, got {tuple(value.shape)}"
                    )
                destination.copy_(value.to(device=self.device, dtype=destination.dtype))
            if self.cuda_graph is None:
                self._execute()
            else:
                self.cuda_graph.replay()
            assert self._output_name is not None
            return self._buffers[self._output_name]
