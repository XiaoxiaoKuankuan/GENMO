# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""BUMI 音乐模型的 ONNX/TensorRT 与长音乐滑窗部署运行时。

本模块把固定形状的 ``[1,120,93]`` 单步去噪图封装成统一调用接口，并在图外执行
与训练仓库相同的确定性 DDIM。长音乐严格使用 120 帧窗口、30 帧重叠和 90 帧步长。
每个窗口按训练时的独立完整 crop 分布生成，随后把下一窗口的根旋转对齐到统一轨迹航向，
在双侧真实预测的重叠区融合世界水平位移、绝对根高和关节，并对根四元数执行最短弧
SLERP。全部窗口融合后只积分一次水平根位移，再重建可发送的连续 qpos chunks；根位置
连续性因此来自单一积分链，而不是每个窗口各自积分后再修补位置。

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
from gem.robots.bumi.feature_codec import (
    BUMI_REPRESENTATION_CONTRACT_VERSION,
    make_quaternion_continuous,
    normalize_quaternion_wxyz,
)
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
from gem.utils.rotation_conversions import quaternion_multiply

BUMI_MOTION_DIM = 93
BUMI_ENGINE_CONTRACT = "gem_bumi_music_trt_engine_v2"
BUMI_SLIDING_QPOS_CONTRACT_VERSION = "genmo.bumi_sliding_motion_overlap_add.v3"


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
        if (
            len(outputs) != 1
            or outputs[0].name != "pred_motion"
            or tuple(int(item) for item in outputs[0].shape) != (1, WINDOW_FRAMES, BUMI_MOTION_DIM)
        ):
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
        if payload.get("representation_contract_version") != BUMI_REPRESENTATION_CONTRACT_VERSION:
            raise RuntimeError("BUMI TensorRT engine uses the wrong motion representation")
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
    """完整重叠融合后、可以安全发送给桥接端的 30 Hz 世界系 qpos 后缀。"""

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


def _slerp_wxyz(first: torch.Tensor, second: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """沿最短弧插值一组 wxyz 四元数，并保持时间符号连续。"""

    first = normalize_quaternion_wxyz(first)
    second = normalize_quaternion_wxyz(second)
    if first.shape != second.shape or first.shape[-1] != 4:
        raise ValueError("SLERP quaternion inputs must have the same [...,4] shape")
    if alpha.shape != (*first.shape[:-1], 1):
        raise ValueError("SLERP alpha must have shape [...,1] matching quaternions")
    dot = (first * second).sum(dim=-1, keepdim=True)
    second = torch.where(dot < 0.0, -second, second)
    dot = (first * second).sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    denominator = sin_theta.clamp_min(1.0e-8)
    spherical = (
        torch.sin((1.0 - alpha) * theta) / denominator * first
        + torch.sin(alpha * theta) / denominator * second
    )
    linear = normalize_quaternion_wxyz((1.0 - alpha) * first + alpha * second)
    result = torch.where(sin_theta > 1.0e-6, spherical, linear)
    return make_quaternion_continuous(normalize_quaternion_wxyz(result))


def _align_motion_state_rotation_to_reference(
    candidate: torch.Tensor, reference_first: torch.Tensor
) -> torch.Tensor:
    """只对齐候选状态的根 yaw；heading-local 位移无需旋转或平移。"""

    if candidate.ndim != 2 or candidate.shape[1] != 28 or len(candidate) <= 0:
        raise ValueError("candidate motion state must have shape [T,28] with T > 0")
    if reference_first.shape != (28,):
        raise ValueError("reference_first must have shape [28]")
    candidate = candidate.clone()
    candidate_quat = normalize_quaternion_wxyz(candidate[:, 3:7])
    reference_quat = normalize_quaternion_wxyz(reference_first[3:7])
    delta_yaw = _yaw_from_wxyz(reference_quat) - _yaw_from_wxyz(candidate_quat[0])
    half_yaw = delta_yaw * 0.5
    delta_quaternion = torch.stack(
        (
            torch.cos(half_yaw),
            torch.zeros_like(half_yaw),
            torch.zeros_like(half_yaw),
            torch.sin(half_yaw),
        )
    )
    aligned_quaternion = quaternion_multiply(
        delta_quaternion.expand_as(candidate_quat), candidate_quat
    )
    return torch.cat(
        (
            candidate[:, :3],
            make_quaternion_continuous(aligned_quaternion),
            candidate[:, 7:],
        ),
        dim=-1,
    )


def _heading_delta_to_world(
    delta_xy_heading: torch.Tensor, quaternion: torch.Tensor
) -> torch.Tensor:
    yaw = _yaw_from_wxyz(quaternion)
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    return torch.stack(
        (
            cosine * delta_xy_heading[:, 0] - sine * delta_xy_heading[:, 1],
            sine * delta_xy_heading[:, 0] + cosine * delta_xy_heading[:, 1],
        ),
        dim=-1,
    )


def _world_delta_to_heading(delta_xy_world: torch.Tensor, quaternion: torch.Tensor) -> torch.Tensor:
    yaw = _yaw_from_wxyz(quaternion)
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    return torch.stack(
        (
            cosine * delta_xy_world[:, 0] + sine * delta_xy_world[:, 1],
            -sine * delta_xy_world[:, 0] + cosine * delta_xy_world[:, 1],
        ),
        dim=-1,
    )


def _blend_motion_state_overlap(previous: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    """融合速度、高度、旋转和关节，不在窗口内独立积分根位置。"""

    if previous.shape != candidate.shape or previous.ndim != 2 or previous.shape[1] != 28:
        raise ValueError("motion state overlap inputs must have the same [T,28] shape")
    if len(previous) <= 0:
        raise ValueError("qpos overlap must not be empty")
    alpha = (
        torch.arange(1, len(previous) + 1, device=previous.device, dtype=previous.dtype)
        / float(len(previous) + 1)
    ).unsqueeze(-1)
    blended_quaternion = _slerp_wxyz(previous[:, 3:7], candidate[:, 3:7], alpha)
    previous_delta_world = _heading_delta_to_world(previous[:, :2], previous[:, 3:7])
    candidate_delta_world = _heading_delta_to_world(candidate[:, :2], candidate[:, 3:7])
    blended_delta_world = torch.lerp(previous_delta_world, candidate_delta_world, alpha)
    blended_delta_heading = _world_delta_to_heading(blended_delta_world, blended_quaternion)
    return torch.cat(
        (
            blended_delta_heading,
            torch.lerp(previous[:, 2:3], candidate[:, 2:3], alpha),
            blended_quaternion,
            torch.lerp(previous[:, 7:], candidate[:, 7:], alpha),
        ),
        dim=-1,
    )


class BumiSlidingQposGenerator:
    """融合独立 120 帧预测，并对最终水平根位移统一积分一次。"""

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
            return torch.stack(
                (
                    default_z.new_zeros(()),
                    default_z.new_zeros(()),
                    default_z,
                    default_z.new_zeros(()),
                )
            )
        yaw = _yaw_from_wxyz(first_world_qpos[3:7])
        return torch.stack((first_world_qpos[0], first_world_qpos[1], default_z, yaw))

    def _verify_alignment(
        self, reference_first: torch.Tensor, candidate_first: torch.Tensor, window_index: int
    ) -> None:
        yaw_delta = _yaw_from_wxyz(reference_first[3:7]) - _yaw_from_wxyz(candidate_first[3:7])
        yaw_delta = torch.atan2(torch.sin(yaw_delta), torch.cos(yaw_delta)).abs()
        if float(yaw_delta) > self.overlap_atol:
            raise RuntimeError(
                "BUMI sliding rotation alignment failed at window "
                f"{window_index}: yaw_abs={float(yaw_delta):.6g}"
            )

    @staticmethod
    def _motion_state(decoded: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            (
                decoded["root_delta_xy_heading"],
                decoded["root_height_offset"],
                decoded["root_rot_local_quat"],
                decoded["joint_dof"],
            ),
            dim=-1,
        )

    @torch.inference_mode()
    def generate(self, music: torch.Tensor, *, seed: int = 42) -> BumiSlidingResult:
        """生成任意长度 EDGE35；独立窗口对齐融合后只返回最终连续轨迹。"""

        features = torch.as_tensor(music).detach().float().cpu()
        if features.ndim != 2 or features.shape[1] != MUSIC_DIM or len(features) <= 0:
            raise ValueError(f"music must have shape [T,{MUSIC_DIM}] with T > 0")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("music contains NaN or Inf")
        total_frames = len(features)
        stitched_state: torch.Tensor | None = None
        windows = plan_sliding_windows(total_frames)
        for window in windows:
            normalized = self.generator.generate_window(
                padded_music_window(features, window),
                valid_length=window.valid_length,
                seed=derive_window_seed(seed, window.index),
                known_x0=None,
            )[: window.valid_length]
            decoded = self.endecoder.decode(normalized)
            candidate_state = self._motion_state(decoded)
            if stitched_state is None:
                stitched_state = candidate_state
                continue
            if window.known_length != OVERLAP_FRAMES:
                raise RuntimeError("BUMI sliding window must have a full 30-frame overlap")
            overlap_end = window.start + window.known_length
            if overlap_end != len(stitched_state):
                raise RuntimeError("BUMI stitched timeline does not end at the overlap boundary")
            aligned_state = _align_motion_state_rotation_to_reference(
                candidate_state, stitched_state[window.start]
            )
            self._verify_alignment(stitched_state[window.start], aligned_state[0], window.index)
            blended_overlap = _blend_motion_state_overlap(
                stitched_state[window.start : overlap_end],
                aligned_state[: window.known_length],
            )
            stitched_state = torch.cat(
                (
                    stitched_state[: window.start],
                    blended_overlap,
                    aligned_state[window.known_length :],
                ),
                dim=0,
            )
        if stitched_state is None or len(stitched_state) != total_frames:
            raise RuntimeError("BUMI sliding generation did not cover the requested timeline")
        stitched_state = torch.cat(
            (
                stitched_state[:, :3],
                make_quaternion_continuous(stitched_state[:, 3:7]),
                stitched_state[:, 7:],
            ),
            dim=-1,
        ).contiguous()
        stitched = self.endecoder.compose_qpos(
            {
                "root_delta_xy_heading": stitched_state[:, :2],
                "root_height_offset": stitched_state[:, 2:3],
                "root_rot_local_quat": stitched_state[:, 3:7],
                "joint_dof": stitched_state[:, 7:],
            },
            world_anchor=self._anchor_for(None),
        ).contiguous()
        chunks: list[BumiGeneratedChunk] = []
        for window in windows:
            absolute_start = window.start + window.known_length
            chunk_qpos = stitched[absolute_start : window.end].contiguous()
            if len(chunk_qpos) != window.new_length:
                raise RuntimeError("BUMI final chunk has an unexpected suffix length")
            chunks.append(
                BumiGeneratedChunk(
                    window_index=window.index,
                    absolute_start_frame=absolute_start,
                    total_frames=total_frames,
                    qpos=chunk_qpos.detach().cpu(),
                    is_last=window.end == total_frames,
                )
            )
        if not chunks or not chunks[-1].is_last:
            raise RuntimeError("BUMI sliding generation did not produce a terminal chunk")
        return BumiSlidingResult(qpos=stitched.detach().cpu(), chunks=tuple(chunks))


__all__ = [
    "BUMI_ENGINE_CONTRACT",
    "BUMI_MOTION_DIM",
    "BUMI_SLIDING_QPOS_CONTRACT_VERSION",
    "BumiGeneratedChunk",
    "BumiOrtStepRunner",
    "BumiSlidingQposGenerator",
    "BumiSlidingResult",
    "BumiTensorRTStepRunner",
    "bumi_engine_cache_key",
]
