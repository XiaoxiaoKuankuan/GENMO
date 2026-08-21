"""CPU contract tests for the BUMI-native ONNX denoiser boundary."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from gem.runtime.bumi_music_onnx import (
    BumiMusicGuidedDenoiser,
    make_bumi_onnx_inputs,
    validate_bumi_checkpoint_state_dict,
    validate_bumi_music_export_model,
)


class FakeDenoiser(nn.Module):
    output_dim = 93
    encode_text = False
    max_len = 120
    pred_cam_head = False
    static_conf_head = False

    def forward(self, xt, timesteps, y, inputs, sample_indices_dict):
        del inputs, sample_indices_dict
        condition = y["f_cond"].mean(dim=-1, keepdim=True)
        time = timesteps.float().view(-1, 1, 1) / 1000.0
        length_term = y["length"].float().view(-1, 1, 1) * 1.0e-5
        prediction = xt * 0.25 + condition.expand_as(xt) + time + length_term
        return {
            "pred_x_start": prediction,
            "pred_x": prediction,
            "pred_cam": None,
            "static_conf_logits": None,
        }


class FakeEndecoder:
    feat_dim = 93

    def __init__(self):
        self.obs_indices_dict = None

    def build_obs_indices_dict(self):
        self.obs_indices_dict = {
            "root_pos_local": (0, 3),
            "root_rot_local": (3, 9),
            "joint_dof": (9, 30),
            "body_link_pos_local": (30, 93),
        }


class FakeBumiModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.motion_backend = "bumi"
        self.music_embedder = nn.Linear(35, 8)
        self.cond_exists_embedder = nn.ModuleDict(
            {"encoded_music": nn.Linear(9, 8)}
        )
        self.model_cfg = SimpleNamespace(use_cond_exists_as_input=True)
        self.endecoder = FakeEndecoder()
        self.pipeline = SimpleNamespace(
            args=SimpleNamespace(in_attr=["encoded_music"]),
            denoiser3d=SimpleNamespace(
                denoiser=FakeDenoiser(), regression_only=False
            ),
        )


def test_bumi_cfg_is_internal_and_outputs_only_motion93() -> None:
    torch.manual_seed(4)
    wrapper = BumiMusicGuidedDenoiser(FakeBumiModel()).eval()
    music = torch.randn(120, 35)
    noisy, timestep, music, length, _ = make_bumi_onnx_inputs(music, seed=7)
    unconditional = wrapper(
        noisy, timestep, music, length, torch.tensor([0.0])
    )
    conditional = wrapper(noisy, timestep, music, length, torch.tensor([1.0]))
    guided = wrapper(noisy, timestep, music, length, torch.tensor([2.5]))
    torch.testing.assert_close(
        guided, unconditional + 2.5 * (conditional - unconditional)
    )
    assert guided.shape == (1, 120, 93)
    assert torch.isfinite(guided).all()


def test_bumi_onnx_inputs_are_deterministic_and_strict() -> None:
    music = torch.zeros(120, 35)
    first = make_bumi_onnx_inputs(music, seed=19)
    second = make_bumi_onnx_inputs(music, seed=19)
    assert all(torch.equal(left, right) for left, right in zip(first, second))
    assert first[0].shape == (1, 120, 93)
    assert first[1].dtype == torch.long
    with pytest.raises(ValueError, match="shape"):
        make_bumi_onnx_inputs(torch.zeros(2, 120, 35))
    with pytest.raises(ValueError, match="guidance_scale"):
        make_bumi_onnx_inputs(music, guidance_scale=-1.0)
    bad = music.clone()
    bad[0, 0] = torch.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        make_bumi_onnx_inputs(bad)


def test_bumi_export_rejects_wrong_backend_and_optional_heads() -> None:
    model = FakeBumiModel()
    model.motion_backend = "smpl"
    with pytest.raises(RuntimeError, match="motion_backend"):
        validate_bumi_music_export_model(model)
    model.motion_backend = "bumi"
    model.pipeline.denoiser3d.denoiser.pred_cam_head = True
    with pytest.raises(RuntimeError, match="pred_cam_dim=0"):
        validate_bumi_music_export_model(model)


def test_checkpoint_contract_rejects_smpl151_and_optional_heads() -> None:
    valid = {
        "music_embedder.fc1.weight": torch.zeros(8, 35),
        "pipeline.denoiser3d.denoiser.final_layer.fc2.weight": torch.zeros(93, 8),
    }
    report = validate_bumi_checkpoint_state_dict(valid)
    assert next(iter(report["final_layer_weight_shapes"].values())) == [93, 8]
    wrong = dict(valid)
    wrong["pipeline.denoiser3d.denoiser.final_layer.fc2.weight"] = torch.zeros(151, 8)
    with pytest.raises(ValueError, match="93D"):
        validate_bumi_checkpoint_state_dict(wrong)
    wrong = dict(valid)
    wrong["pipeline.denoiser3d.denoiser.pred_cam_head.fc1.weight"] = torch.zeros(8, 8)
    with pytest.raises(ValueError, match="camera/static"):
        validate_bumi_checkpoint_state_dict(wrong)


def test_bumi_wrapper_exports_and_matches_onnxruntime(tmp_path) -> None:
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    torch.manual_seed(5)
    wrapper = BumiMusicGuidedDenoiser(FakeBumiModel()).eval()
    inputs = make_bumi_onnx_inputs(torch.randn(12, 35), seed=23)
    reference = wrapper(*inputs).detach().numpy()
    output = tmp_path / "fake_bumi.onnx"
    torch.onnx.export(
        wrapper,
        inputs,
        str(output),
        opset_version=18,
        input_names=[
            "noisy_motion",
            "diffusion_timestep",
            "music",
            "length",
            "guidance_scale",
        ],
        output_names=["pred_motion"],
        dynamo=False,
    )
    onnx.checker.check_model(str(output))
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    names = (
        "noisy_motion",
        "diffusion_timestep",
        "music",
        "length",
        "guidance_scale",
    )
    candidate = session.run(
        ["pred_motion"],
        {name: value.detach().numpy() for name, value in zip(names, inputs)},
    )[0]
    np = pytest.importorskip("numpy")
    np.testing.assert_allclose(reference, candidate, atol=1e-5, rtol=1e-5)
