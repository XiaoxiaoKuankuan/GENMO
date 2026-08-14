"""ONNX export boundary for the music-only GEM diffusion specialist.

The complete GEM generator contains an iterative DDIM scheduler and SMPL decoding.
Exporting that Python control flow as one giant graph would duplicate the 16-layer
Transformer many times.  This module therefore exports the reusable neural-network
part of one DDIM step: EDGE baseline35 embedding, conditional/unconditional music
branches, classifier-free guidance, and the 151-D denoiser prediction.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

MUSIC_FEATURE_DIM = 35
MOTION_FEATURE_DIM = 151


def validate_music_only_export_model(model: nn.Module) -> None:
    """Fail early unless *model* is the strict music-only diffusion specialist."""
    in_attr = list(model.pipeline.args.in_attr)
    if in_attr != ["encoded_music"]:
        raise RuntimeError(
            "ONNX music-only export requires pipeline.args.in_attr == "
            f"['encoded_music']; got {in_attr}"
        )
    if not hasattr(model, "music_embedder"):
        raise RuntimeError("music-only model has no music_embedder")
    if bool(getattr(model.pipeline.denoiser3d, "regression_only", False)):
        raise RuntimeError("music-only ONNX export requires the diffusion model")
    denoiser = model.pipeline.denoiser3d.denoiser
    if bool(getattr(denoiser, "encode_text", False)):
        raise RuntimeError("music-only denoiser unexpectedly enables text cross-attention")
    if int(getattr(denoiser, "output_dim", -1)) != MOTION_FEATURE_DIM:
        raise RuntimeError(
            f"expected a {MOTION_FEATURE_DIM}-D motion denoiser, got "
            f"{getattr(denoiser, 'output_dim', None)}"
        )


class MusicOnlyGuidedDenoiser(nn.Module):
    """One music-conditioned, classifier-free-guided diffusion network step.

    Inputs:
        noisy_motion: ``float32[B, T, 151]`` normalized noisy motion ``x_t``.
        diffusion_timestep: ``int64[B]`` timestep on the original 0..999 schedule.
        music: ``float32[B, T, 35]`` EDGE baseline35 features at 30 Hz.
        length: ``int64[B]`` valid lengths, each in ``[1, T]``.
        guidance_scale: ``float32[1]`` CFG scale (2.5 by default in GEM).

    Outputs are the guided x-start motion, weak-perspective camera prediction, and
    static-joint logits.  Only the music branch exists in this graph.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        validate_music_only_export_model(model)
        if model.endecoder.obs_indices_dict is None:
            model.endecoder.build_obs_indices_dict()

        self.music_embedder = model.music_embedder
        self.denoiser = model.pipeline.denoiser3d.denoiser
        self.sample_indices_dict = dict(model.endecoder.obs_indices_dict)
        self.latent_dim = int(model.latent_dim)
        self.max_len = int(self.denoiser.max_len)

        self.use_condition_exists = bool(model.model_cfg.use_cond_exists_as_input)
        if self.use_condition_exists:
            embedders = getattr(model, "cond_exists_embedder", None)
            if embedders is None or "encoded_music" not in embedders:
                raise RuntimeError(
                    "use_cond_exists_as_input=True but encoded_music exists embedder is missing"
                )
            self.music_exists_embedder = embedders["encoded_music"]

    def _condition_pair(self, music: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        conditional = self.music_embedder(music)
        unconditional = torch.zeros_like(conditional)
        if self.use_condition_exists:
            ones = torch.ones_like(conditional[..., :1])
            zeros = torch.zeros_like(conditional[..., :1])
            conditional = self.music_exists_embedder(torch.cat([conditional, ones], dim=-1))
            unconditional = self.music_exists_embedder(torch.cat([unconditional, zeros], dim=-1))
        return conditional, unconditional

    def _denoise(
        self,
        noisy_motion: torch.Tensor,
        diffusion_timestep: torch.Tensor,
        condition: torch.Tensor,
        length: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.denoiser(
            noisy_motion,
            diffusion_timestep,
            y={"f_cond": condition, "length": length},
            inputs={},
            sample_indices_dict=self.sample_indices_dict,
        )

    @staticmethod
    def _guided(
        conditional: torch.Tensor,
        unconditional: torch.Tensor,
        guidance_scale: torch.Tensor,
    ) -> torch.Tensor:
        scale = guidance_scale.reshape(1, 1, 1).to(dtype=conditional.dtype)
        return unconditional + scale * (conditional - unconditional)

    def forward(
        self,
        noisy_motion: torch.Tensor,
        diffusion_timestep: torch.Tensor,
        music: torch.Tensor,
        length: torch.Tensor,
        guidance_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        conditional, unconditional = self._condition_pair(music)
        out_cond = self._denoise(noisy_motion, diffusion_timestep, conditional, length)
        out_uncond = self._denoise(noisy_motion, diffusion_timestep, unconditional, length)

        pred_motion = self._guided(
            out_cond["pred_x_start"], out_uncond["pred_x_start"], guidance_scale
        )
        pred_camera = self._guided(out_cond["pred_cam"], out_uncond["pred_cam"], guidance_scale)
        static_conf_logits = self._guided(
            out_cond["static_conf_logits"],
            out_uncond["static_conf_logits"],
            guidance_scale,
        )
        return pred_motion, pred_camera, static_conf_logits


def make_onnx_inputs(
    music: torch.Tensor,
    *,
    seed: int = 42,
    timestep: int = 999,
    guidance_scale: float = 2.5,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create deterministic, contract-checked inputs for export or validation."""
    if not isinstance(music, torch.Tensor):
        raise TypeError("music must be a torch.Tensor")
    if music.ndim == 2:
        music = music.unsqueeze(0)
    if music.ndim != 3 or music.shape[-1] != MUSIC_FEATURE_DIM:
        raise ValueError(
            f"music must have shape [T, {MUSIC_FEATURE_DIM}] or "
            f"[B, T, {MUSIC_FEATURE_DIM}]; got {tuple(music.shape)}"
        )
    if music.shape[1] <= 0:
        raise ValueError("music must contain at least one frame")
    if not torch.isfinite(music).all():
        raise ValueError("music contains NaN or Inf")
    if not 0 <= int(timestep) <= 999:
        raise ValueError("timestep must be in the original diffusion range 0..999")
    if not torch.isfinite(torch.tensor(float(guidance_scale))) or float(guidance_scale) < 0:
        raise ValueError("guidance_scale must be finite and >= 0")

    batch, frames = music.shape[:2]
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noisy_motion = torch.randn(
        batch, frames, MOTION_FEATURE_DIM, generator=generator, dtype=torch.float32
    )
    timestep_tensor = torch.full((batch,), int(timestep), dtype=torch.long)
    length = torch.full((batch,), int(frames), dtype=torch.long)
    scale = torch.tensor([float(guidance_scale)], dtype=torch.float32)
    return tuple(
        value.to(device) for value in (noisy_motion, timestep_tensor, music.float(), length, scale)
    )


def output_statistics(outputs: tuple[Any, ...]) -> dict[str, dict[str, Any]]:
    """Return compact finite/range statistics for the three ONNX outputs."""
    names = ("pred_motion", "pred_camera", "static_conf_logits")
    result: dict[str, dict[str, Any]] = {}
    for name, raw in zip(names, outputs):
        value = torch.as_tensor(raw).detach().cpu()
        result[name] = {
            "shape": list(value.shape),
            "dtype": str(value.numpy().dtype),
            "finite": bool(torch.isfinite(value).all()),
            "min": float(value.min()),
            "max": float(value.max()),
            "mean": float(value.float().mean()),
            "std": float(value.float().std()),
        }
    return result
