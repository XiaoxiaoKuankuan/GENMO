"""常驻 T5 文本编码公共接口。

对单条有效文本执行固定 token 长度编码，并将 padding 特征置零。返回连续的 CPU
float32 特征；tokenizer 与 T5 由调用者管理，模块不加载 SMPL 模型或启动旧文本服务。
"""

from __future__ import annotations

from typing import Any

import torch

MAX_TEXT_LEN = 150
TEXT_EMBED_DIM = 1024


def normalize_prompt(prompt: str) -> str:
    """Strip surrounding whitespace without changing prompt case or content."""
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    normalized = prompt.strip()
    if not normalized:
        raise ValueError("prompt must not be empty")
    return normalized


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
    input_ids = (tokenized["input_ids"] if isinstance(tokenized, dict) else tokenized.input_ids).to(
        device
    )
    attention_mask = (
        tokenized["attention_mask"] if isinstance(tokenized, dict) else tokenized.attention_mask
    ).to(device)
    with torch.inference_mode():
        output = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
    encoded_text = output.last_hidden_state[:, :max_text_len]
    encoded_text = encoded_text * attention_mask[:, :max_text_len].unsqueeze(-1)
    expected = (1, max_text_len, TEXT_EMBED_DIM)
    if tuple(encoded_text.shape) != expected:
        raise RuntimeError(
            f"T5 text embedding has shape {tuple(encoded_text.shape)}, expected {expected}. "
            "Use the T5-3B encoder expected by BUMI text."
        )
    return encoded_text[0].detach().float().cpu().contiguous()
