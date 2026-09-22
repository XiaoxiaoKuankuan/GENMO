"""文本分支公共模块与部署依赖边界的回归测试。

通过独立解释器阻断已退役后端，确认当前训练、数据与推理模块可以独立导入；
同时检查 T5 padding、差分和区间统计的关键语义。测试不加载真实模型、不使用 GPU，
不创建正式输出目录；这些检查不代表生成动作质量或实际 TensorRT 引擎验收。
"""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gem.robots.bumi.motion_quality import (
    _angular_speed_wxyz,
    _central_difference,
    mask_to_intervals,
    safe_intervals_from_bad_mask,
)
from gem.runtime.text_encoding import encode_prompt_with_loaded_t5

ROOT = Path(__file__).resolve().parents[2]


def test_current_modules_do_not_import_retired_backends():
    program = """
import importlib.abc
import sys
retired = ("gem.gem", "gem.runtime.music_only_trt", "gem.runtime.resident_text_motion",
           "gem.robots.bumi.legacy_motion", "gem.robots.bumi.quality_filter",
           "tools.data.music_dance", "tools.data.motionmillion", "tools.data.bumi.filter_sonic")
class BlockRetired(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == p or fullname.startswith(p + ".") for p in retired):
            raise AssertionError("重新引入旧依赖: " + fullname)
sys.meta_path.insert(0, BlockRetired())
import gem.bumi_text_gem
import gem.runtime.bumi_text_tensorrt
import gem.runtime.text_motion_web.models
import tools.data.bumi.prepare_bumi_text
import tools.data.bumi.umr_text_preprocess
import tools.data.kitml.prepare_kitml_amass
import tools.export.bumi_text
assert not any(n.startswith("gem.network.hmr2") or n == "smplx" for n in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-B", "-c", program],
        cwd=ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        check=True,
        timeout=60,
    )


def test_public_t5_encoder_masks_padding_and_rejects_empty_prompt():
    def tokenizer(prompts, **kwargs):
        assert prompts == ["walk"] and kwargs["max_length"] == 150
        return {
            "input_ids": torch.zeros(1, 150, dtype=torch.long),
            "attention_mask": (torch.arange(150) < 3)[None],
        }

    def encoder(**kwargs):
        return SimpleNamespace(last_hidden_state=torch.ones(1, 150, 1024))

    result = encode_prompt_with_loaded_t5(" walk ", tokenizer, encoder, "cpu")
    assert result.shape == (150, 1024) and result.dtype == torch.float32
    assert result[:3].eq(1).all() and result[3:].eq(0).all()
    with pytest.raises(ValueError, match="empty"):
        encode_prompt_with_loaded_t5(" ", tokenizer, encoder, "cpu")


def test_public_quality_statistics_keep_time_and_interval_semantics():
    time = np.arange(8)[:, None] / 30
    np.testing.assert_allclose(_central_difference(time, 30), 1)
    np.testing.assert_allclose(_angular_speed_wxyz(np.tile([1.0, 0, 0, 0], (8, 1)), 30), 0)
    mask = np.array([False, True, True, False, False, False, False, False])
    assert mask_to_intervals(mask) == ((1, 3),)
    assert safe_intervals_from_bad_mask(mask, halo_frames=1, minimum_frames=2) == ((4, 8),)
