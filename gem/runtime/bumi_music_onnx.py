"""ONNX boundary for the BUMI-native music-only diffusion model.

The exported graph contains EDGE35 embedding, conditional/unconditional CFG and
one 93-D x-start denoiser step.  DDIM scheduling and authoritative
93D -> qpos28 -> Torch FK decoding deliberately stay outside the graph so the
same repository diffusion equations and robot contract are used by PyTorch and
ONNX Runtime.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from gem.robots.bumi.feature_codec import BUMI_REPRESENTATION_CONTRACT_VERSION

BUMI_ONNX_CONTRACT_VERSION = "genmo.bumi_music_guided_denoiser_step.v2"
BUMI_MOTION_FEATURE_DIM = 93
MUSIC_FEATURE_DIM = 35


def validate_bumi_checkpoint_state_dict(
    state_dict: dict[str, Any],
) -> dict[str, Any]:
    """Validate the checkpoint-side 93D/music/no-head architecture before loading."""

    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("BUMI checkpoint state_dict must be a non-empty dictionary")
    music = {
        str(key): list(value.shape)
        for key, value in state_dict.items()
        if str(key).endswith("music_embedder.fc1.weight") and isinstance(value, torch.Tensor)
    }
    final = {
        str(key): list(value.shape)
        for key, value in state_dict.items()
        if str(key).endswith("denoiser.final_layer.fc2.weight") and isinstance(value, torch.Tensor)
    }
    if len(music) != 1 or next(iter(music.values()))[-1] != MUSIC_FEATURE_DIM:
        raise ValueError(
            f"checkpoint must contain exactly one EDGE35 music fc1 weight; got {music}"
        )
    if len(final) != 1 or next(iter(final.values()))[0] != BUMI_MOTION_FEATURE_DIM:
        raise ValueError(f"checkpoint must contain exactly one 93D final fc2 weight; got {final}")
    forbidden = sorted(
        str(key)
        for key in state_dict
        if ".pred_cam_head." in str(key) or ".static_conf_head." in str(key)
    )
    if forbidden:
        raise ValueError(
            "formal BUMI no-contact checkpoint must not contain camera/static heads; "
            f"found {forbidden[:4]}"
        )
    return {"music_weight_shapes": music, "final_layer_weight_shapes": final}


def validate_bumi_music_export_model(model: nn.Module) -> None:
    """Reject anything other than the formal BUMI music-only 93-D model."""

    if str(getattr(model, "motion_backend", "")) != "bumi":
        raise RuntimeError("BUMI ONNX export requires motion_backend='bumi'")
    in_attr = list(model.pipeline.args.in_attr)
    if in_attr != ["encoded_music"]:
        raise RuntimeError(
            f"BUMI ONNX export requires pipeline.args.in_attr == ['encoded_music']; got {in_attr}"
        )
    if not hasattr(model, "music_embedder"):
        raise RuntimeError("BUMI music-only model has no music_embedder")
    if bool(getattr(model.pipeline.denoiser3d, "regression_only", False)):
        raise RuntimeError("BUMI ONNX export requires a diffusion checkpoint")
    denoiser = model.pipeline.denoiser3d.denoiser
    if bool(getattr(denoiser, "encode_text", False)):
        raise RuntimeError("BUMI music denoiser unexpectedly enables text encoding")
    if int(getattr(denoiser, "output_dim", -1)) != BUMI_MOTION_FEATURE_DIM:
        raise RuntimeError(
            f"BUMI ONNX export requires a {BUMI_MOTION_FEATURE_DIM}-D denoiser, got "
            f"{getattr(denoiser, 'output_dim', None)}"
        )
    if bool(getattr(denoiser, "pred_cam_head", False)):
        raise RuntimeError("formal BUMI ONNX export requires pred_cam_dim=0")
    if bool(getattr(denoiser, "static_conf_head", False)):
        raise RuntimeError("formal BUMI ONNX export requires static_conf_dim=0")
    endecoder = getattr(model, "endecoder", None)
    if int(getattr(endecoder, "feat_dim", -1)) != BUMI_MOTION_FEATURE_DIM:
        raise RuntimeError("BUMI ONNX export requires the authoritative 93-D Endecoder")
    if (
        str(getattr(endecoder, "representation_contract_version", ""))
        != BUMI_REPRESENTATION_CONTRACT_VERSION
    ):
        raise RuntimeError(
            "BUMI ONNX export requires representation contract "
            f"{BUMI_REPRESENTATION_CONTRACT_VERSION!r}"
        )

    first_music_linear = next(
        (module for module in model.music_embedder.modules() if isinstance(module, nn.Linear)),
        None,
    )
    if first_music_linear is None or int(first_music_linear.in_features) != MUSIC_FEATURE_DIM:
        raise RuntimeError("BUMI checkpoint music input must be raw EDGE35")


class BumiMusicGuidedDenoiser(nn.Module):
    """One fixed-batch BUMI classifier-free-guided denoising step.

    Conditional and unconditional samples are concatenated internally, so one
    Transformer invocation produces the two CFG branches.  The public graph is
    intentionally fixed to batch=1 for TensorRT deployment and outputs only the
    normalized 93-D x-start prediction used by DDIM.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        validate_bumi_music_export_model(model)
        if model.endecoder.obs_indices_dict is None:
            model.endecoder.build_obs_indices_dict()
        self.music_embedder = model.music_embedder
        self.denoiser = model.pipeline.denoiser3d.denoiser
        self.sample_indices_dict = dict(model.endecoder.obs_indices_dict)
        self.use_condition_exists = bool(model.model_cfg.use_cond_exists_as_input)
        if self.use_condition_exists:
            embedders = getattr(model, "cond_exists_embedder", None)
            if embedders is None or "encoded_music" not in embedders:
                raise RuntimeError(
                    "use_cond_exists_as_input=True but encoded_music embedder is missing"
                )
            self.music_exists_embedder = embedders["encoded_music"]

    def _condition_pair(self, music: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        conditional = self.music_embedder(music)
        unconditional = torch.zeros_like(conditional)
        if self.use_condition_exists:
            ones = torch.ones_like(conditional[..., :1])
            zeros = torch.zeros_like(conditional[..., :1])
            conditional = self.music_exists_embedder(torch.cat((conditional, ones), dim=-1))
            unconditional = self.music_exists_embedder(torch.cat((unconditional, zeros), dim=-1))
        return conditional, unconditional

    def forward(
        self,
        noisy_motion: torch.Tensor,
        diffusion_timestep: torch.Tensor,
        music: torch.Tensor,
        length: torch.Tensor,
        guidance_scale: torch.Tensor,
    ) -> torch.Tensor:
        conditional, unconditional = self._condition_pair(music)
        paired_output = self.denoiser(
            torch.cat((noisy_motion, noisy_motion), dim=0),
            torch.cat((diffusion_timestep, diffusion_timestep), dim=0),
            y={
                "f_cond": torch.cat((conditional, unconditional), dim=0),
                "length": torch.cat((length, length), dim=0),
            },
            inputs={},
            sample_indices_dict=self.sample_indices_dict,
        )["pred_x_start"]
        # Export is fixed to public batch=1, therefore the paired tensor always
        # contains conditional at index 0 and unconditional at index 1.
        pred_cond = paired_output[0:1]
        pred_uncond = paired_output[1:2]
        scale = guidance_scale.reshape(1, 1, 1).to(dtype=pred_cond.dtype)
        return pred_uncond + scale * (pred_cond - pred_uncond)


def make_bumi_onnx_inputs(
    music: torch.Tensor,
    *,
    seed: int = 42,
    timestep: int = 999,
    guidance_scale: float = 2.5,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build deterministic fixed-batch inputs for export and parity checks."""

    if not isinstance(music, torch.Tensor):
        raise TypeError("music must be a torch.Tensor")
    if music.ndim == 2:
        music = music.unsqueeze(0)
    if music.ndim != 3 or tuple(music.shape[:1]) != (1,) or music.shape[-1] != 35:
        raise ValueError("music must have shape [T,35] or [1,T,35]")
    if music.shape[1] <= 0:
        raise ValueError("music must contain at least one frame")
    if not bool(torch.isfinite(music).all()):
        raise ValueError("music contains NaN or Inf")
    if not 0 <= int(timestep) <= 999:
        raise ValueError("timestep must be in the original diffusion range 0..999")
    scale_value = torch.tensor(float(guidance_scale), dtype=torch.float32)
    if not bool(torch.isfinite(scale_value)) or float(guidance_scale) < 0.0:
        raise ValueError("guidance_scale must be finite and >= 0")

    frames = int(music.shape[1])
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noisy_motion = torch.randn(
        1,
        frames,
        BUMI_MOTION_FEATURE_DIM,
        generator=generator,
        dtype=torch.float32,
    )
    timestep_tensor = torch.tensor([int(timestep)], dtype=torch.long)
    length = torch.tensor([frames], dtype=torch.long)
    scale = scale_value.reshape(1)
    return tuple(
        value.to(device) for value in (noisy_motion, timestep_tensor, music.float(), length, scale)
    )


def tensor_statistics(value: Any) -> dict[str, Any]:
    tensor = torch.as_tensor(value).detach().cpu().float()
    return {
        "shape": list(tensor.shape),
        "finite": bool(torch.isfinite(tensor).all()),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "mean": float(tensor.mean()),
        "std": float(tensor.std()),
    }


__all__ = [
    "BUMI_MOTION_FEATURE_DIM",
    "BUMI_ONNX_CONTRACT_VERSION",
    "MUSIC_FEATURE_DIM",
    "BumiMusicGuidedDenoiser",
    "make_bumi_onnx_inputs",
    "tensor_statistics",
    "validate_bumi_checkpoint_state_dict",
    "validate_bumi_music_export_model",
]
