"""CPU contracts for TensorRT-style DDIM sliding deployment."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from gem.pipeline.gem_pipeline import get_body_params_w_Rt_v2
from gem.runtime.music_only_trt import (
    MOTION_DIM,
    OVERLAP_FRAMES,
    SlidingDDIMGenerator,
    StreamingSmplDecoder,
    derive_window_seed,
    exact_motion_frame_count,
    padded_music_window,
    plan_sliding_windows,
)
from gem.utils.motion_utils import (
    init_rollout_w_Rt_state,
    rollout_local_transl_vel,
    rollout_step_w_Rt,
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
    generator = SlidingDDIMGenerator(FakeStep(), device="cpu", steps=5)
    music = torch.randn(120, 35)
    known = torch.randn(OVERLAP_FRAMES, MOTION_DIM)
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
    known_noise = (
        top_xt[0, :OVERLAP_FRAMES] - np.sqrt(top_alpha) * known
    ) / np.sqrt(1.0 - top_alpha)
    for step, xt in before:
        expected = generator._q_sample_at(known.unsqueeze(0), step, known_noise.unsqueeze(0))
        torch.testing.assert_close(xt[:, :OVERLAP_FRAMES], expected)
    for step, xt, pred in after:
        assert pred is not None
        assert torch.equal(pred[0, :OVERLAP_FRAMES], known)
        expected = (
            known
            if step == 0
            else generator._q_sample_at(
                known.unsqueeze(0), step - 1, known_noise.unsqueeze(0)
            )[0]
        )
        torch.testing.assert_close(xt[0, :OVERLAP_FRAMES], expected)


def test_window_noise_is_deterministic() -> None:
    generator = SlidingDDIMGenerator(FakeStep(), device="cpu", steps=4)
    music = torch.randn(120, 35)
    first = generator.generate_window(music, valid_length=120, seed=9)
    second = generator.generate_window(music, valid_length=120, seed=9)
    third = generator.generate_window(music, valid_length=120, seed=10)
    assert torch.equal(first, second)
    assert not torch.equal(first, third)
    assert derive_window_seed(42, 3) == derive_window_seed(42, 3)
    assert derive_window_seed(42, 3) != derive_window_seed(42, 4)


class FakeEndecoder(nn.Module):
    def decode(self, value):
        batch, length = value.shape[:2]
        return {
            "body_pose": torch.zeros(batch, length, 63, device=value.device),
            "global_orient_gv": value[..., :3],
            "local_transl_vel": value[..., 3:6],
        }


def test_streaming_root_rollout_matches_one_shot_rollout() -> None:
    torch.manual_seed(4)
    motion = torch.zeros(200, 151)
    motion[:, :3] = torch.randn(200, 3) * 0.03
    motion[:, 3:6] = torch.randn(200, 3) * 0.01
    decoder = StreamingSmplDecoder(FakeEndecoder(), "cpu")
    first = decoder.decode_new(motion[:120], start=0, end=120)
    second_window = motion[90:]
    second = decoder.decode_new(second_window, start=30, end=len(second_window))
    streamed = torch.cat((first["transl"], second["transl"]), dim=0)
    expected = rollout_local_transl_vel(
        motion[None, :, 3:6], motion[None, :, :3]
    )[0]
    torch.testing.assert_close(streamed, expected, atol=1e-6, rtol=1e-6)
    identity_camera_delta = torch.tensor(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ).expand(1, len(motion), -1)
    batch_reference = get_body_params_w_Rt_v2(
        global_orient_gv=motion[None, :, :3],
        local_transl_vel=motion[None, :, 3:6],
        global_orient_c=motion[None, :, :3],
        cam_angvel=identity_camera_delta,
    )
    torch.testing.assert_close(
        streamed, batch_reference["transl"][0], atol=1e-6, rtol=1e-6
    )
    assert torch.equal(streamed[0], torch.zeros(3))


def test_shared_rollout_step_has_no_first_frame_phase_shift() -> None:
    orientations = torch.zeros(5, 3)
    velocities = torch.tensor(
        [[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0], [0.4, 0.0, 0.0], [9.0, 0.0, 0.0]]
    )
    state = init_rollout_w_Rt_state(orientations[0], orientations[0])
    actual = []
    for index in range(len(orientations)):
        params, state = rollout_step_w_Rt(
            state,
            orientations[index],
            orientations[index],
            local_transl_vel_prev=None if index == 0 else velocities[index - 1],
            local_transl_vel_curr=velocities[index],
        )
        actual.append(params["transl"][0])
    expected = rollout_local_transl_vel(velocities[None], orientations[None])[0]
    torch.testing.assert_close(torch.stack(actual), expected)


def test_inpainted_window_requires_one_new_frame() -> None:
    generator = SlidingDDIMGenerator(FakeStep(), device="cpu", steps=2)
    with pytest.raises(ValueError, match="new frame"):
        generator.generate_window(
            torch.zeros(120, 35),
            valid_length=30,
            seed=1,
            known_x0=torch.zeros(30, 151),
        )
