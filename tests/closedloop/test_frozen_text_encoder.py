"""冻结文本编码器的缓存、去重、补零和反向传播边界测试。

使用明确的微型替身编码器验证实现，不下载模型、不依赖 GPU，也不把本测试视为真实
T5 数值验收。覆盖 LRU 容量小于 batch、重复原文、空文本拒绝、150-token 输出及缓存
精度一致性，确保冻结条件仍允许后续可训练层正常反向传播。
"""

from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

from gem.runtime.frozen_text_encoder import FrozenTextEncoder


class TinyTokenizer:
    def __call__(self, texts, **kwargs):
        ids = torch.zeros(len(texts), 150, dtype=torch.long)
        mask = ids.clone()
        for i, text in enumerate(texts):
            tokens = [ord(c) % 32 + 1 for c in text[:149]] + [1]
            ids[i, : len(tokens)] = torch.tensor(tokens)
            mask[i, : len(tokens)] = 1
        return SimpleNamespace(input_ids=ids, attention_mask=mask)


class TinyEncoder(torch.nn.Module):
    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(
            last_hidden_state=input_ids[..., None].expand(-1, -1, 1024).float() / 7
        )


def test_frozen_text_lru_dedup_mask_and_downstream_gradients():
    encoder = FrozenTextEncoder.__new__(FrozenTextEncoder)
    encoder.micro_batch_size, encoder.cache_entries = 2, 1
    encoder.cache = OrderedDict()
    encoder.hits = encoder.misses = 0
    encoder.device = torch.device("cpu")
    encoder.model, encoder.tokenizer = TinyEncoder(), TinyTokenizer()
    features, mask = encoder.encode([" first ", "second", "first", "third"], "cpu")
    assert features.shape == (4, 150, 1024) and mask.dtype == torch.bool
    assert len(encoder.cache) == 1 and encoder.misses == 3
    assert not features[~mask].any() and not features.requires_grad
    torch.testing.assert_close(features[0], features[2], rtol=0, atol=0)
    cached, cached_mask = encoder.encode(["third"], "cpu")
    torch.testing.assert_close(cached[0], features[3], rtol=0, atol=0)
    torch.testing.assert_close(cached_mask[0], mask[3])
    layer = torch.nn.Linear(1024, 8)
    layer(features).sum().backward()
    assert torch.isfinite(layer.weight.grad).all()
    with pytest.raises(ValueError, match="空文本"):
        encoder.encode([" "], "cpu")
