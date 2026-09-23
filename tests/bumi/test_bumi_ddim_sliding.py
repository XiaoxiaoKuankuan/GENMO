"""验证 BUMI 复用的 DDIM 滑窗工具，保持 30D 测试动作与当前表示一致。

从 SMPL/TensorRT 混合测试中拆出窗口覆盖、padding、确定性噪声和 known-prefix
采样约束；不再测试 SMPL 解码。prefix 用例仅验证共享采样器能力，不宣称当前
BUMI 长音乐部署采用硬前缀，也不表示实现了 closed-loop 训练。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gem.robots.bumi.feature_codec import BUMI_FEATURE_DIM
from gem.runtime.music_only_trt import (
    OVERLAP_FRAMES,
    SlidingDDIMGenerator,
    derive_window_seed,
    exact_motion_frame_count,
    padded_music_window,
    plan_sliding_windows,
)


class FakeStep:
    def __call__(self, noisy, timestep, music, length, guidance):
        del length, guidance
        signal = music.mean(dim=-1, keepdim=True)
        time = timestep.float().reshape(1, 1, 1) / 1000.0
        return noisy * 0.05 + signal.expand_as(noisy) * 0.1 + time


def test_window_plan_commits_every_frame_once_and_duration_is_exact() -> None:
    windows = plan_sliding_windows(601)
    committed: list[int] = []
    for window in windows:
        committed.extend(range(window.start + window.new_start, window.end))
    assert committed == list(range(601))
    assert windows[0].known_length == 0
    assert all(value.known_length == 30 for value in windows[1:])
    assert exact_motion_frame_count(601, 20.0) == 600
    assert exact_motion_frame_count(601, None) == 601
    with pytest.raises(ValueError, match="fewer than"):
        exact_motion_frame_count(599, 20.0)
    music = torch.arange(601 * 35, dtype=torch.float32).reshape(601, 35)
    last = padded_music_window(music, windows[-1])
    assert last.shape == (120, 35)
    torch.testing.assert_close(last[60], last[-1])


def test_hard_inpainting_is_applied_before_and_after_every_ddim_step() -> None:
    generator = SlidingDDIMGenerator(FakeStep(), motion_dim=BUMI_FEATURE_DIM, device="cpu", steps=5)
    music = torch.randn(120, 35)
    known = torch.randn(OVERLAP_FRAMES, BUMI_FEATURE_DIM)
    traces: list[tuple[int, torch.Tensor, torch.Tensor | None]] = []
    result = generator.generate_window(
        music,
        valid_length=120,
        seed=17,
        known_x0=known,
        trace_hook=lambda step, xt, pred: traces.append((step, xt, pred)),
    )
    assert torch.equal(result[:OVERLAP_FRAMES], known)
    before = [(step, xt) for step, xt, pred in traces if pred is None]
    after = [(step, xt, pred) for step, xt, pred in traces if pred is not None]
    assert len(before) == len(after) == 5

    top_step, top_xt = before[0]
    top_alpha = float(generator.diffusion.alphas_cumprod[top_step])
    known_noise = (top_xt[0, :OVERLAP_FRAMES] - np.sqrt(top_alpha) * known) / np.sqrt(
        1.0 - top_alpha
    )
    for step, xt in before:
        expected = generator._q_sample_at(known.unsqueeze(0), step, known_noise.unsqueeze(0))
        torch.testing.assert_close(xt[:, :OVERLAP_FRAMES], expected)
    for step, xt, pred in after:
        assert pred is not None
        assert torch.equal(pred[0, :OVERLAP_FRAMES], known)
        expected = (
            known
            if step == 0
            else generator._q_sample_at(known.unsqueeze(0), step - 1, known_noise.unsqueeze(0))[0]
        )
        torch.testing.assert_close(xt[0, :OVERLAP_FRAMES], expected)


def test_window_noise_is_deterministic() -> None:
    generator = SlidingDDIMGenerator(FakeStep(), motion_dim=BUMI_FEATURE_DIM, device="cpu", steps=4)
    music = torch.randn(120, 35)
    first = generator.generate_window(music, valid_length=120, seed=9)
    second = generator.generate_window(music, valid_length=120, seed=9)
    third = generator.generate_window(music, valid_length=120, seed=10)
    assert torch.equal(first, second)
    assert not torch.equal(first, third)
    assert derive_window_seed(42, 3) == derive_window_seed(42, 3)
    assert derive_window_seed(42, 3) != derive_window_seed(42, 4)


def test_window_noise_can_use_an_explicit_canonical_rng_device() -> None:
    generator = SlidingDDIMGenerator(
        FakeStep(), motion_dim=BUMI_FEATURE_DIM, device="cpu", noise_device="cpu", steps=2
    )
    music = torch.zeros(120, 35)
    first_step_inputs: list[torch.Tensor] = []
    generator.generate_window(
        music,
        valid_length=120,
        seed=42,
        trace_hook=lambda _step, x_t, pred: (
            first_step_inputs.append(x_t) if pred is None and not first_step_inputs else None
        ),
    )
    expected_rng = torch.Generator(device="cpu")
    expected_rng.manual_seed(42)
    expected = torch.randn(
        (1, 120, BUMI_FEATURE_DIM),
        device="cpu",
        dtype=torch.float32,
        generator=expected_rng,
    )
    assert generator.noise_device == torch.device("cpu")
    assert torch.equal(first_step_inputs[0], expected)


def test_inpainted_window_requires_one_new_frame() -> None:
    generator = SlidingDDIMGenerator(FakeStep(), motion_dim=BUMI_FEATURE_DIM, device="cpu", steps=2)
    with pytest.raises(ValueError, match="new frame"):
        generator.generate_window(
            torch.zeros(120, 35),
            valid_length=30,
            seed=1,
            known_x0=torch.zeros(30, BUMI_FEATURE_DIM),
        )
