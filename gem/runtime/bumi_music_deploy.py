# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""BUMI 音乐模型的 ONNX/TensorRT 与长音乐滑窗部署运行时。

本模块把固定形状的 ``[1,120,93]`` 单步去噪图封装成统一调用接口，并在图外执行
与训练仓库相同的确定性 DDIM。长音乐严格使用 120 帧窗口、30 帧重叠和 90 帧步长；
重叠段在每一个 DDIM 步都做硬覆盖。由于 BUMI 93D 特征采用“每段首帧 XY/航向角”
局部坐标，本模块不会直接拼接不同窗口的局部特征，而是先把已提交的世界系 qpos28
重新编码为下一窗的局部重叠条件，再把新窗口放回同一世界锚点。这样既保持模型训练
分布，又保证输出给机器人/GMT 的根轨迹、四元数和关节序列连续。

TensorRT 引擎使用独立的 BUMI 指纹合约，缓存键包含 ONNX、checkpoint、TensorRT ABI、
精度和 GPU 指纹，避免误加载 SMPL 151D 引擎或在不同显卡上复用不兼容 plan。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gem.robots.bumi.endecoder import BumiEndecoder
from gem.runtime.music_only_trt import (
    MUSIC_DIM,
    OVERLAP_FRAMES,
    WINDOW_FRAMES,
    SlidingDDIMGenerator,
    TensorRTStepRunner,
    derive_window_seed,
    gpu_fingerprint,
    padded_music_window,
    plan_sliding_windows,
    sha256_file,
)

BUMI_MOTION_DIM = 93
BUMI_ENGINE_CONTRACT = "gem_bumi_music_trt_engine_v1"


def bumi_engine_cache_key(
    *,
    onnx_sha256: str,
    checkpoint_sha256: str,
    tensorrt_version: str,
    precision: str,
    gpu: dict[str, object],
) -> str:
    """返回只适用于 BUMI 93D 固定形状引擎的可复现缓存键。"""

    value = {
        "contract": BUMI_ENGINE_CONTRACT,
        "onnx_sha256": str(onnx_sha256),
        "checkpoint_sha256": str(checkpoint_sha256),
        "tensorrt_version": str(tensorrt_version),
        "precision": str(precision),
        "gpu": gpu,
        "inputs": [1, WINDOW_FRAMES, BUMI_MOTION_DIM],
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class BumiOrtStepRunner:
    """严格校验固定形状 BUMI ONNX 图并执行一个 CFG 去噪步。"""

    REQUIRED_INPUTS = {
        "noisy_motion": (1, WINDOW_FRAMES, BUMI_MOTION_DIM),
        "diffusion_timestep": (1,),
        "music": (1, WINDOW_FRAMES, MUSIC_DIM),
        "length": (1,),
        "guidance_scale": (1,),
    }

    def __init__(
        self,
        onnx_path: str | Path,
        *,
        device: torch.device | str = "cpu",
        provider: str = "cpu",
    ) -> None:
        import onnxruntime as ort

        self.path = Path(onnx_path).expanduser().resolve(strict=True)
        self.device = torch.device(device)
        provider_name = {
            "cpu": "CPUExecutionProvider",
            "cuda": "CUDAExecutionProvider",
        }.get(str(provider).lower())
        if provider_name is None:
            raise ValueError("provider must be 'cpu' or 'cuda'")
        if provider_name not in ort.get_available_providers():
            raise RuntimeError(
                f"requested {provider_name}, available={ort.get_available_providers()}"
            )
        self.session = ort.InferenceSession(str(self.path), providers=[provider_name])
        input_shapes = {
            value.name: tuple(int(item) for item in value.shape)
            for value in self.session.get_inputs()
        }
        if input_shapes != self.REQUIRED_INPUTS:
            raise RuntimeError(f"BUMI ONNX input contract mismatch: {input_shapes}")
        outputs = self.session.get_outputs()
        if len(outputs) != 1 or outputs[0].name != "pred_motion" or tuple(
            int(item) for item in outputs[0].shape
        ) != (1, WINDOW_FRAMES, BUMI_MOTION_DIM):
            raise RuntimeError("BUMI ONNX pred_motion output contract mismatch")

    def __call__(
        self,
        noisy_motion: torch.Tensor,
        diffusion_timestep: torch.Tensor,
        music: torch.Tensor,
        length: torch.Tensor,
        guidance_scale: torch.Tensor,
    ) -> torch.Tensor:
        values = self.session.run(
            ["pred_motion"],
            {
                "noisy_motion": noisy_motion.detach().cpu().numpy().astype(np.float32),
                "diffusion_timestep": diffusion_timestep.detach().cpu().numpy().astype(np.int64),
                "music": music.detach().cpu().numpy().astype(np.float32),
                "length": length.detach().cpu().numpy().astype(np.int64),
                "guidance_scale": guidance_scale.detach().cpu().numpy().astype(np.float32),
            },
        )[0]
        return torch.from_numpy(values).to(self.device)


class BumiTensorRTStepRunner(TensorRTStepRunner):
    """带 BUMI 专用 plan 指纹检查的常驻 TensorRT 单步运行器。"""

    REQUIRED_INPUTS = {
        "noisy_motion": (1, WINDOW_FRAMES, BUMI_MOTION_DIM),
        "diffusion_timestep": (1,),
        "music": (1, WINDOW_FRAMES, MUSIC_DIM),
        "length": (1,),
        "guidance_scale": (1,),
    }
    REQUIRED_OUTPUT = (1, WINDOW_FRAMES, BUMI_MOTION_DIM)

    def _validate_manifest(self, required: bool) -> dict[str, object] | None:
        manifest_path = self.engine_path.parent / "engine.json"
        if not manifest_path.is_file():
            if required:
                raise RuntimeError(f"BUMI TensorRT engine manifest is missing: {manifest_path}")
            return None
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("contract_version") != BUMI_ENGINE_CONTRACT:
            raise RuntimeError("unsupported BUMI TensorRT engine manifest contract")
        if payload.get("engine_sha256") != sha256_file(self.engine_path):
            raise RuntimeError("BUMI TensorRT engine SHA256 does not match its manifest")
        if payload.get("input_shape") != [1, WINDOW_FRAMES, BUMI_MOTION_DIM]:
            raise RuntimeError("BUMI TensorRT manifest input_shape is not [1,120,93]")
        if payload.get("output_shape") != [1, WINDOW_FRAMES, BUMI_MOTION_DIM]:
            raise RuntimeError("BUMI TensorRT manifest output_shape is not [1,120,93]")
        expected_version = str(payload.get("tensorrt_version", ""))
        actual_version = str(self.trt.__version__)
        if expected_version.split(".")[:2] != actual_version.split(".")[:2]:
            raise RuntimeError(
                f"BUMI engine was built with TensorRT {expected_version}, runtime is {actual_version}"
            )
        expected_library = str(payload.get("libnvinfer_version", expected_version))
        if expected_library.split(".")[:2] != self.linked_tensorrt_version.split(".")[:2]:
            raise RuntimeError(
                "BUMI engine libnvinfer mismatch: "
                f"built={expected_library}, runtime={self.linked_tensorrt_version}"
            )
        actual_gpu = gpu_fingerprint(self.device)
        expected_gpu = payload.get("gpu", {})
        if (
            not isinstance(expected_gpu, dict)
            or expected_gpu.get("name") != actual_gpu["name"]
            or expected_gpu.get("compute_capability") != actual_gpu["compute_capability"]
        ):
            raise RuntimeError(f"BUMI TensorRT GPU fingerprint mismatch: {expected_gpu}")
        expected_key = bumi_engine_cache_key(
            onnx_sha256=str(payload.get("onnx_sha256", "")),
            checkpoint_sha256=str(payload.get("checkpoint_sha256", "")),
            tensorrt_version=expected_version,
            precision=str(payload.get("precision", "")),
            gpu=actual_gpu,
        )
        if payload.get("cache_key") != expected_key:
            raise RuntimeError("BUMI TensorRT engine cache fingerprint mismatch")
        return payload


@dataclass(frozen=True, slots=True)
class BumiGeneratedChunk:
    """一个已经提交、可以安全发送给桥接端的 30 Hz 世界系 qpos 后缀。"""

    window_index: int
    absolute_start_frame: int
    total_frames: int
    qpos: torch.Tensor
    is_last: bool


@dataclass(frozen=True, slots=True)
class BumiSlidingResult:
    """长音乐生成结果；qpos 与 chunks 使用完全相同的提交边界。"""

    qpos: torch.Tensor
    chunks: tuple[BumiGeneratedChunk, ...]


def _yaw_from_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    value = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = value.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class BumiSlidingQposGenerator:
    """把 120/30 的局部 93D 滑窗提交为连续世界系 qpos28。"""

    def __init__(
        self,
        denoiser: Any,
        endecoder: BumiEndecoder,
        *,
        device: torch.device | str,
        steps: int = 20,
        guidance_scale: float = 2.5,
        overlap_atol: float = 2.0e-4,
    ) -> None:
        if not isinstance(endecoder, BumiEndecoder):
            raise TypeError("BumiSlidingQposGenerator requires BumiEndecoder")
        if not math.isfinite(float(overlap_atol)) or overlap_atol <= 0.0:
            raise ValueError("overlap_atol must be finite and > 0")
        self.device = torch.device(device)
        self.endecoder = endecoder.to(self.device).eval()
        self.generator = SlidingDDIMGenerator(
            denoiser,
            device=self.device,
            steps=steps,
            guidance_scale=guidance_scale,
            motion_dim=BUMI_MOTION_DIM,
        )
        self.overlap_atol = float(overlap_atol)

    def _anchor_for(self, first_world_qpos: torch.Tensor | None) -> torch.Tensor:
        default_z = self.endecoder.kinematics.default_qpos[2].to(self.device)
        if first_world_qpos is None:
            return torch.stack((default_z.new_zeros(()), default_z.new_zeros(()), default_z, default_z.new_zeros(())))
        yaw = _yaw_from_wxyz(first_world_qpos[3:7])
        return torch.stack((first_world_qpos[0], first_world_qpos[1], default_z, yaw))

    def _known_features(self, overlap_qpos: torch.Tensor) -> torch.Tensor:
        encoded = self.endecoder.codec.encode(overlap_qpos.to(self.device))
        return self.endecoder.normalize(encoded.physical_features).contiguous()

    def _verify_overlap(
        self, expected: torch.Tensor, candidate: torch.Tensor, window_index: int
    ) -> None:
        linear_delta = torch.cat(
            ((expected[:, :3] - candidate[:, :3]).abs(), (expected[:, 7:] - candidate[:, 7:]).abs()),
            dim=-1,
        ).amax()
        expected_quat = expected[:, 3:7]
        candidate_quat = candidate[:, 3:7]
        quat_dot = (expected_quat * candidate_quat).sum(dim=-1).abs().clamp(max=1.0)
        quat_delta = (1.0 - quat_dot).amax()
        if float(linear_delta) > self.overlap_atol or float(quat_delta) > self.overlap_atol:
            raise RuntimeError(
                "BUMI sliding overlap reconstruction failed at window "
                f"{window_index}: linear_max={float(linear_delta):.6g}, "
                f"quat_1_minus_abs_dot={float(quat_delta):.6g}"
            )

    @torch.inference_mode()
    def generate(self, music: torch.Tensor, *, seed: int = 42) -> BumiSlidingResult:
        """生成任意长度 EDGE35；不足 120 帧的尾窗只提交有效帧。"""

        features = torch.as_tensor(music).detach().float().cpu()
        if features.ndim != 2 or features.shape[1] != MUSIC_DIM or len(features) <= 0:
            raise ValueError(f"music must have shape [T,{MUSIC_DIM}] with T > 0")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("music contains NaN or Inf")
        total_frames = len(features)
        committed = torch.empty((0, 28), dtype=torch.float32, device=self.device)
        chunks: list[BumiGeneratedChunk] = []
        windows = plan_sliding_windows(total_frames)
        for window in windows:
            known_world = None
            known_x0 = None
            if window.known_length:
                known_world = committed[
                    window.start : window.start + window.known_length
                ]
                if len(known_world) != OVERLAP_FRAMES:
                    raise RuntimeError("BUMI committed overlap does not cover the next window")
                known_x0 = self._known_features(known_world)
            normalized = self.generator.generate_window(
                padded_music_window(features, window),
                valid_length=window.valid_length,
                seed=derive_window_seed(seed, window.index),
                known_x0=known_x0,
            )[: window.valid_length]
            decoded = self.endecoder.decode(normalized)
            anchor = self._anchor_for(None if known_world is None else known_world[0])
            world_window = self.endecoder.compose_qpos(decoded, world_anchor=anchor)
            if known_world is not None:
                self._verify_overlap(
                    known_world, world_window[:OVERLAP_FRAMES], window.index
                )
                world_window = torch.cat(
                    (known_world, world_window[OVERLAP_FRAMES:]), dim=0
                )
            new_qpos = world_window[window.known_length :].contiguous()
            if len(new_qpos) != window.new_length:
                raise RuntimeError("BUMI sliding window committed an unexpected suffix")
            absolute_start = window.start + window.known_length
            if absolute_start != len(committed):
                raise RuntimeError("BUMI sliding commit is not contiguous")
            committed = torch.cat((committed, new_qpos), dim=0)
            chunks.append(
                BumiGeneratedChunk(
                    window_index=window.index,
                    absolute_start_frame=absolute_start,
                    total_frames=total_frames,
                    qpos=new_qpos.detach().cpu(),
                    is_last=window.end == total_frames,
                )
            )
        if len(committed) != total_frames or not chunks or not chunks[-1].is_last:
            raise RuntimeError("BUMI sliding generation did not cover the requested timeline")
        return BumiSlidingResult(
            qpos=committed.detach().cpu(), chunks=tuple(chunks)
        )


__all__ = [
    "BUMI_ENGINE_CONTRACT",
    "BUMI_MOTION_DIM",
    "BumiGeneratedChunk",
    "BumiOrtStepRunner",
    "BumiSlidingQposGenerator",
    "BumiSlidingResult",
    "BumiTensorRTStepRunner",
    "bumi_engine_cache_key",
]
