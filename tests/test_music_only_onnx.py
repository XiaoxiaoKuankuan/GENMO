"""CPU contract tests for the music-only ONNX export boundary."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from gem.runtime.music_only_onnx import (
    MusicOnlyGuidedDenoiser,
    MusicOnlyTensorRTDenoiser,
    make_onnx_inputs,
    validate_music_only_export_model,
)


class _FakeDenoiser(nn.Module):
    output_dim = 151
    encode_text = False
    max_len = 120

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, xt, timesteps, y, inputs, sample_indices_dict):
        self.calls += 1
        del inputs, sample_indices_dict
        cond = y["f_cond"].mean(dim=-1, keepdim=True)
        time = timesteps.float().view(-1, 1, 1) / 1000.0
        pred = xt * 0.25 + cond.expand_as(xt) + time
        return {
            "pred_x_start": pred,
            "pred_x": pred,
            "pred_cam": cond.expand(*cond.shape[:-1], 3),
            "static_conf_logits": cond.expand(*cond.shape[:-1], 6),
        }


class _FakeEndecoder:
    def __init__(self):
        self.obs_indices_dict = None

    def build_obs_indices_dict(self):
        self.obs_indices_dict = {"betas": (126, 136)}


class _FakeModel(nn.Module):
    def __init__(self, in_attr=None):
        super().__init__()
        in_attr = ["encoded_music"] if in_attr is None else in_attr
        denoiser = _FakeDenoiser()
        self.pipeline = SimpleNamespace(
            args=SimpleNamespace(in_attr=in_attr),
            denoiser3d=SimpleNamespace(denoiser=denoiser, regression_only=False),
        )
        self.music_embedder = nn.Linear(35, 8)
        self.cond_exists_embedder = nn.ModuleDict({"encoded_music": nn.Linear(9, 8)})
        self.model_cfg = SimpleNamespace(use_cond_exists_as_input=True)
        self.endecoder = _FakeEndecoder()
        self.latent_dim = 8


def test_export_boundary_contains_only_music_and_cfg_matches_formula() -> None:
    torch.manual_seed(3)
    wrapper = MusicOnlyGuidedDenoiser(_FakeModel()).eval()
    music = torch.randn(2, 7, 35)
    noisy, timestep, music, length, _ = make_onnx_inputs(music, seed=9)
    zero = torch.tensor([0.0])
    one = torch.tensor([1.0])
    scale = torch.tensor([2.5])
    uncond = wrapper(noisy, timestep, music, length, zero)
    cond = wrapper(noisy, timestep, music, length, one)
    guided = wrapper(noisy, timestep, music, length, scale)
    for uncond_value, cond_value, guided_value in zip(uncond, cond, guided):
        assert torch.allclose(guided_value, uncond_value + 2.5 * (cond_value - uncond_value))
        assert torch.isfinite(guided_value).all()
    assert guided[0].shape == (2, 7, 151)
    assert guided[1].shape == (2, 7, 3)
    assert guided[2].shape == (2, 7, 6)


def test_tensorrt_export_batches_cfg_and_returns_only_motion() -> None:
    torch.manual_seed(5)
    legacy_model = _FakeModel()
    deploy_model = _FakeModel()
    deploy_model.load_state_dict(legacy_model.state_dict())
    legacy = MusicOnlyGuidedDenoiser(legacy_model).eval()
    deployment = MusicOnlyTensorRTDenoiser(deploy_model).eval()
    music = torch.randn(1, 120, 35)
    inputs = make_onnx_inputs(music, seed=2, guidance_scale=2.5)
    expected = legacy(*inputs)[0]
    actual = deployment(*inputs)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (1, 120, 151)
    assert deploy_model.pipeline.denoiser3d.denoiser.calls == 1


def test_export_rejects_non_specialist_conditions() -> None:
    model = _FakeModel(in_attr=["encoded_music", "f_imgseq"])
    with pytest.raises(RuntimeError, match="in_attr"):
        validate_music_only_export_model(model)


@pytest.mark.parametrize("shape", [(120, 34), (1, 120, 36), (35,)])
def test_make_onnx_inputs_rejects_wrong_music_shape(shape) -> None:
    with pytest.raises(ValueError, match="music must have shape"):
        make_onnx_inputs(torch.zeros(shape))


def test_make_onnx_inputs_is_deterministic_and_rejects_nonfinite() -> None:
    music = torch.zeros(120, 35)
    first = make_onnx_inputs(music, seed=17)
    second = make_onnx_inputs(music, seed=17)
    assert all(torch.equal(left, right) for left, right in zip(first, second))
    assert first[0].shape == (1, 120, 151)
    assert first[1].dtype == torch.long
    bad = music.clone()
    bad[0, 0] = torch.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        make_onnx_inputs(bad)
    with pytest.raises(ValueError, match="guidance_scale"):
        make_onnx_inputs(music, guidance_scale=-1)
