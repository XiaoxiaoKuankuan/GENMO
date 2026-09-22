"""按实际采样文本运行冻结的本地 T5，避免预计算数百万条未被抽到的描述。

编码器只接受显式本地资产，记录权重、配置和分词器 SHA；每个训练 rank 独立绑定设备，
以小批次编码未命中的原文，并把有效 token 的 FP16 特征放入有界 CPU LRU。
返回值保持现有 150×1024、padding 清零及 attention mask 契约。该普通 Python 对象
不把冻结 T5 注册到训练模型，避免进入优化器或每个 checkpoint；恢复时另行核对资产。
缓存不改变原文、caption 抽样或事件区间，空文本、非有限编码和设备迁移均显式报错。
"""

from collections import OrderedDict
from pathlib import Path

import torch

from gem.runtime.bumi_text_contract import sha256_file


class FrozenTextEncoder:
    def __init__(self, model_path, *, micro_batch_size=32, cache_entries=100000):
        self.root = Path(model_path).expanduser().resolve(strict=True)
        self.micro_batch_size, self.cache_entries = int(micro_batch_size), int(cache_entries)
        if self.micro_batch_size < 1 or self.cache_entries < 1:
            raise ValueError("T5 micro batch/cache 必须为正")
        self.assets = {
            "t5_" + name.replace(".", "_"): {
                "path": str(self.root / name),
                "sha256": sha256_file(self.root / name),
            }
            for name in ("model.safetensors", "config.json", "spiece.model")
        }
        self.model, self.tokenizer, self.device = None, None, None
        self.cache = OrderedDict()
        self.hits, self.misses = 0, 0

    def _load(self, device):
        if self.model is not None:
            if self.device != device:
                raise ValueError("冻结 T5 不能跨 rank/device 复用")
            return
        from transformers import T5EncoderModel, T5Tokenizer

        self.tokenizer = T5Tokenizer.from_pretrained(str(self.root), local_files_only=True)
        self.model = (
            T5EncoderModel.from_pretrained(
                str(self.root),
                local_files_only=True,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            )
            .eval()
            .requires_grad_(False)
            .to(device)
        )
        if self.model.config.d_model != 1024:
            raise ValueError("BUMI 文本必须使用 1024D T5")
        self.device = device

    def encode(self, captions, device):
        device = torch.device(device)
        texts = [caption.strip() for caption in captions]
        if not texts or any(not text for text in texts):
            raise ValueError("在线 T5 不能用空文本替代真实条件")
        self._load(device)
        # batch 内去重；暂存引用防止小容量 LRU 在本批次中途淘汰仍需使用的特征。
        values = {text: self.cache[text] for text in dict.fromkeys(texts) if text in self.cache}
        missing = [text for text in dict.fromkeys(texts) if text not in values]
        self.hits += len(texts) - len(missing)
        self.misses += len(missing)
        for start in range(0, len(missing), self.micro_batch_size):
            part = missing[start : start + self.micro_batch_size]
            token = self.tokenizer(
                part, padding="max_length", truncation=True, max_length=150, return_tensors="pt"
            )
            mask = token.attention_mask.to(device)
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                feature = self.model(
                    input_ids=token.input_ids.to(device), attention_mask=mask
                ).last_hidden_state
            if feature.shape != (len(part), 150, 1024) or not torch.isfinite(feature).all():
                raise ValueError("冻结 T5 输出 shape 错误或包含非有限值")
            feature = feature.detach().to(device="cpu", dtype=torch.float16)
            for i, text in enumerate(part):
                length = int(token.attention_mask[i].sum())
                values[text] = feature[i, :length].clone()
        output = torch.zeros(len(texts), 150, 1024, device=device, dtype=torch.float32)
        attention_mask = torch.zeros(len(texts), 150, device=device, dtype=torch.bool)
        for i, text in enumerate(texts):
            value = values[text]
            output[i, : len(value)] = value.to(device=device, dtype=torch.float32)
            attention_mask[i, : len(value)] = True
            self.cache[text] = value
            self.cache.move_to_end(text)
            while len(self.cache) > self.cache_entries:
                self.cache.popitem(last=False)
        return output, attention_mask
