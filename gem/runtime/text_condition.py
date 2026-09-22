# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""训练与推理共用的文本条件形状和有效位校验。

保留预计算特征、单条文本和空条件的既有约定；空条件至少保留一个零 token，防止
注意力全 padding 导致非有限值。本模块只依赖 PyTorch，不导入人体模型或训练服务。
"""

from __future__ import annotations

import torch


def prepare_precomputed_text_embed(
    text_embed: torch.Tensor,
    *,
    expected_dim: int,
    expected_length: int = 50,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Validate and batch a precomputed GEM text embedding.

    ``GEM.predict`` accepts either a single ``[T, C]`` embedding or an
    already-batched ``[1, T, C]`` embedding.  Keeping this validation in a
    small helper also makes the input contract testable without constructing
    the full diffusion model.
    """
    if not isinstance(text_embed, torch.Tensor):
        raise TypeError(
            "data['text_embed'] must be a torch.Tensor with shape "
            f"[{expected_length}, {expected_dim}] or [1, {expected_length}, {expected_dim}]"
        )
    if text_embed.ndim == 2:
        text_embed = text_embed.unsqueeze(0)
    if text_embed.ndim != 3:
        raise ValueError(
            "data['text_embed'] must have 2 or 3 dimensions; expected "
            f"[{expected_length}, {expected_dim}] or [1, {expected_length}, {expected_dim}], "
            f"got {tuple(text_embed.shape)}"
        )
    if text_embed.shape[0] != 1:
        raise ValueError(
            "GEM.predict() currently supports exactly one precomputed text prompt; "
            f"got batch dimension {text_embed.shape[0]}"
        )
    if text_embed.shape[1] != expected_length:
        raise ValueError(
            f"Precomputed text embedding length must be {expected_length}; "
            f"got {text_embed.shape[1]}"
        )
    if text_embed.shape[2] != expected_dim:
        raise ValueError(
            f"Precomputed text embedding feature dimension must match the denoiser "
            f"encoded_text_dim ({expected_dim}); got {text_embed.shape[2]}"
        )
    if device is not None:
        text_embed = text_embed.to(device=device)
    return text_embed


def prepare_predict_text_condition(
    data: dict,
    *,
    max_text_len: int,
    encoded_text_dim: int,
    device: torch.device | str,
    encode_text_fn=None,
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """Prepare the single-sample text condition used by :meth:`GEM.predict`.

    A false ``has_text`` value takes a dedicated zero path and never invokes
    the T5 encoder. This is important for music-only inference, where loading
    T5-3B is unnecessary and an empty caption must not become a valid prompt.
    """
    raw_has_text = data.get("has_text")
    if raw_has_text is None:
        raw_has_text = bool(data.get("caption", ""))
    has_text = torch.as_tensor(raw_has_text, dtype=torch.bool, device=device).reshape(-1)
    if has_text.numel() != 1:
        raise ValueError(
            f"GEM.predict() expects one has_text value; got shape {tuple(has_text.shape)}"
        )
    caption = [str(data.get("caption", ""))] if bool(has_text.item()) else [""]
    if not bool(has_text.item()):
        encoded_text = torch.zeros(
            (1, int(max_text_len), int(encoded_text_dim)),
            dtype=torch.float32,
            device=device,
        )
    else:
        if encode_text_fn is None:
            raise RuntimeError("has_text=True requires a text encoder or precomputed text_embed")
        encoded_text = encode_text_fn(caption, has_text)
        expected = (1, int(max_text_len), int(encoded_text_dim))
        if tuple(encoded_text.shape) != expected:
            raise ValueError(
                f"Text encoder returned {tuple(encoded_text.shape)}; expected {expected}"
            )
        encoded_text = encoded_text.to(device=device)
    return caption, has_text, encoded_text


def prepare_text_attention_mask(
    text_embed: torch.Tensor,
    text_attention_mask: torch.Tensor | None,
    *,
    has_text: torch.Tensor | None = None,
) -> torch.Tensor:
    """校验并补全 T5 token 有效位掩码。

    新的 MotionMillion 数据会显式提供 mask；旧数据和旧 demo 没有该字段时，
    从非零 embedding 推断。对于明确标记为无文本的兼容样本，只保留一个零
    token 为有效位，避免 ``MultiheadAttention`` 收到整行全 padding 后产生 NaN。
    有文本却没有任何有效 token 属于数据错误，会在进入 denoiser 前直接拒绝。
    """
    if text_embed.ndim != 3:
        raise ValueError(f"text_embed must be [B,T,C], got {tuple(text_embed.shape)}")
    batch_size, token_count = text_embed.shape[:2]
    if text_attention_mask is None:
        mask = text_embed.detach().abs().any(dim=-1)
    else:
        mask = torch.as_tensor(text_attention_mask, device=text_embed.device, dtype=torch.bool)
        if mask.ndim == 1 and batch_size == 1:
            mask = mask.unsqueeze(0)
        if tuple(mask.shape) != (batch_size, token_count):
            raise ValueError(
                "text_attention_mask must match [B,T] of text_embed; "
                f"got {tuple(mask.shape)} versus {(batch_size, token_count)}"
            )

    empty_rows = ~mask.any(dim=1)
    if empty_rows.any():
        if has_text is None:
            raise ValueError("text_attention_mask contains an all-padding sample")
        has_text = torch.as_tensor(has_text, device=text_embed.device, dtype=torch.bool).reshape(-1)
        if has_text.numel() != batch_size:
            raise ValueError(f"has_text must have {batch_size} values, got {has_text.numel()}")
        invalid = empty_rows & has_text
        if invalid.any():
            bad = invalid.nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(
                f"text samples {bad} are marked has_text=True but contain no valid token"
            )
        mask = mask.clone()
        mask[empty_rows, 0] = True
    return mask
