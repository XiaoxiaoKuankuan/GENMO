"""Explicit, report-producing SMPL-music to BUMI-music checkpoint adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from gem.utils.ckpt_compat import remap_legacy_state_dict


def _is_exact_shared_key(key: str) -> bool:
    prefixes = (
        "music_embedder.",
        "cond_exists_embedder.encoded_music.",
        "pipeline.denoiser3d.denoiser.blocks.",
        "pipeline.denoiser3d.denoiser.embed_timestep.",
        "pipeline.denoiser3d.denoiser.sequence_pos_encoder.",
        # Small test modules may expose the same components at their root.
        "blocks.",
        "embed_timestep.",
        "sequence_pos_encoder.",
    )
    return key.startswith(prefixes)


def _is_add_cond_weight(key: str) -> bool:
    return key.endswith("add_cond_linear.weight")


def _is_add_cond_bias(key: str) -> bool:
    return key.endswith("add_cond_linear.bias")


def _is_expected_skip(key: str) -> bool:
    markers = (
        "endecoder",
        "body_model",
        "smpl",
        "final_layer",
        "pred_cam",
        "static_conf_head",
        "betas",
        "body_pose",
        "stats",
    )
    return any(marker in key for marker in markers)


def _checkpoint_state(
    checkpoint_or_path: str | Path | dict[str, Any],
) -> tuple[dict[str, Any], dict[str, torch.Tensor], str, dict[str, Any]]:
    if isinstance(checkpoint_or_path, (str, Path)):
        path = Path(checkpoint_or_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"SMPL music checkpoint does not exist: {path}")
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")
        source = str(path)
    elif isinstance(checkpoint_or_path, dict):
        checkpoint = checkpoint_or_path
        source = "<in-memory>"
    else:
        raise TypeError("checkpoint must be a path or checkpoint dictionary")
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    raw_state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(raw_state, dict) or not raw_state:
        raise ValueError("checkpoint does not contain a non-empty state_dict")
    for key, value in raw_state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError(f"checkpoint state_dict entry {key!r} is not a tensor")
    state, compat = remap_legacy_state_dict(raw_state)
    return checkpoint, dict(state), source, compat


def adapt_smpl_music_checkpoint_to_bumi(
    model: nn.Module,
    checkpoint_or_path: str | Path | dict[str, Any],
    *,
    report_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load only allowlisted representation-independent weights.

    The add-condition projection is special: GEM concatenates ``[f_cond, xt]``.
    Consequently only its first ``latent_dim`` condition columns and its bias
    are copied.  No SMPL motion-input column is ever reinterpreted as BUMI.
    """

    checkpoint, source_state, source, compat = _checkpoint_state(checkpoint_or_path)
    target_state = model.state_dict()
    adapted_state = {key: value.detach().clone() for key, value in target_state.items()}
    report: dict[str, Any] = {
        "contract_version": "genmo.bumi_checkpoint_adaptation.v1",
        "adapter": "smpl_music_to_bumi",
        "source_checkpoint": source,
        "compatibility_remap": compat,
        "loaded_exact": [],
        "loaded_partial": [],
        "reinitialized": [],
        "skipped_expected": [],
        "missing_expected": [],
        "unexpected": [],
        "unclassified_shape_mismatch": [],
    }
    loaded_target_keys: set[str] = set()

    for key, source_value in source_state.items():
        target_value = target_state.get(key)
        if target_value is None:
            category = "skipped_expected" if _is_expected_skip(key) else "unexpected"
            report[category].append(key)
            continue
        if _is_add_cond_weight(key):
            if source_value.ndim != 2 or target_value.ndim != 2:
                report["unclassified_shape_mismatch"].append(
                    {
                        "key": key,
                        "source_shape": list(source_value.shape),
                        "target_shape": list(target_value.shape),
                        "reason": "add_cond_linear.weight must be 2D",
                    }
                )
                continue
            latent_dim = int(target_value.shape[0])
            if (
                source_value.shape[0] != latent_dim
                or source_value.shape[1] <= latent_dim
                or target_value.shape[1] <= latent_dim
            ):
                report["unclassified_shape_mismatch"].append(
                    {
                        "key": key,
                        "source_shape": list(source_value.shape),
                        "target_shape": list(target_value.shape),
                        "reason": "cannot identify [condition, motion] column boundary",
                    }
                )
                continue
            adapted = target_value.detach().clone()
            adapted[:, :latent_dim] = source_value[:, :latent_dim].to(adapted)
            adapted_state[key] = adapted
            loaded_target_keys.add(key)
            report["loaded_partial"].append(
                {
                    "key": key,
                    "copied": f"condition columns [0:{latent_dim})",
                    "kept_initialized": f"BUMI motion columns [{latent_dim}:{target_value.shape[1]})",
                    "source_shape": list(source_value.shape),
                    "target_shape": list(target_value.shape),
                }
            )
            continue
        if _is_add_cond_bias(key):
            if source_value.shape != target_value.shape:
                report["unclassified_shape_mismatch"].append(
                    {
                        "key": key,
                        "source_shape": list(source_value.shape),
                        "target_shape": list(target_value.shape),
                        "reason": "add_cond_linear.bias shape differs",
                    }
                )
                continue
            adapted_state[key] = source_value.to(target_value).detach().clone()
            loaded_target_keys.add(key)
            report["loaded_partial"].append(
                {
                    "key": key,
                    "copied": "full bias",
                    "source_shape": list(source_value.shape),
                    "target_shape": list(target_value.shape),
                }
            )
            continue
        if _is_exact_shared_key(key):
            if source_value.shape != target_value.shape:
                report["unclassified_shape_mismatch"].append(
                    {
                        "key": key,
                        "source_shape": list(source_value.shape),
                        "target_shape": list(target_value.shape),
                        "reason": "allowlisted shared tensor shape differs",
                    }
                )
                continue
            adapted_state[key] = source_value.to(target_value).detach().clone()
            loaded_target_keys.add(key)
            report["loaded_exact"].append(key)
            continue
        if _is_expected_skip(key):
            report["skipped_expected"].append(key)
        elif source_value.shape != target_value.shape:
            report["unclassified_shape_mismatch"].append(
                {
                    "key": key,
                    "source_shape": list(source_value.shape),
                    "target_shape": list(target_value.shape),
                    "reason": "same-name tensor is outside the explicit allowlist",
                }
            )
        else:
            report["unexpected"].append(key)

    for key in target_state:
        if key in loaded_target_keys:
            continue
        if _is_exact_shared_key(key) or _is_add_cond_weight(key) or _is_add_cond_bias(key):
            if key not in source_state:
                report["missing_expected"].append(key)
        report["reinitialized"].append(key)

    for key in (
        "loaded_exact",
        "reinitialized",
        "skipped_expected",
        "missing_expected",
        "unexpected",
    ):
        report[key] = sorted(set(report[key]))
    if report["unclassified_shape_mismatch"]:
        details = json.dumps(report["unclassified_shape_mismatch"], indent=2, sort_keys=True)
        raise RuntimeError(
            "BUMI checkpoint adaptation found unclassified shape mismatches; "
            f"refusing to continue:\n{details}"
        )

    # adapted_state contains every current key, so this call cannot silently
    # omit newly introduced model parameters.
    nn.Module.load_state_dict(model, adapted_state, strict=True)
    if report_path is not None:
        path = Path(report_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return checkpoint, report


__all__ = ["adapt_smpl_music_checkpoint_to_bumi"]
