"""Contracts for the independent music-only GEM-SMPL specialist."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from gem.datasets.aistpp.aistplusplus import (
    resolve_music_motion_alignment,
    select_aist_temporal_window,
    validate_aist_metric_translation,
    validate_musicfeat_v2,
)
from gem.gem import GEM
from gem.network.gem_cfg_sampler import ClassifierFreeSampleModel
from gem.network.gem_denoiser import NetworkEncoderRoPE
from gem.network.gem_diffusion import (
    GEMDiffusion,
    apply_regression_targets_to_2d_only,
)
from gem.pipeline.gem_pipeline import compute_extra_global_loss
from gem.utils.ckpt_compat import remap_legacy_state_dict

REPO_ROOT = Path(__file__).resolve().parents[1]


def _compose(exp: str):
    with initialize_config_dir(
        version_base="1.3", config_dir=str(REPO_ROOT / "configs")
    ):
        return compose(config_name="train", overrides=[f"exp={exp}"])


def test_music_only_and_generalist_configs_compose() -> None:
    specialist = _compose("gem_smpl_music_only")
    assert list(specialist.train_datasets) == ["aistpp_train"]
    assert list(specialist.pipeline.args.train_modes) == ["diffusion"]
    assert list(specialist.pipeline.args.in_attr) == ["encoded_music"]
    assert specialist.pipeline.args.encoded_music_dim == 35
    assert specialist.pipeline.args.disable_random_null_condition is True
    assert specialist.model.model_cfg.text_encoder is None
    assert specialist.network.model_cfg.denoiser.encode_text is False
    assert specialist.train_datasets.aistpp_train.strict_music_alignment is True
    assert specialist.train_datasets.aistpp_train.load_raw_music_audio is False
    assert specialist.train_datasets.aistpp_train.enable_contact_supervision is True
    assert specialist.train_datasets.aistpp_train.aist_world_up_axis == "y"
    assert specialist.train_datasets.aistpp_train.validate_metric_translation is True
    assert specialist.test_datasets.aistpp_music_eval.enable_contact_supervision is True
    assert specialist.test_datasets.aistpp_music_eval.aist_world_up_axis == "y"
    assert specialist.scheduler.interval == "step"
    assert list(specialist.scheduler.scheduler.milestones) == [70000, 100000]

    generalist = _compose("gem_smpl")
    assert list(generalist.pipeline.args.train_modes) == ["regression", "diffusion"]
    assert "encoded_music" in generalist.pipeline.args.in_attr
    assert "f_imgseq" in generalist.pipeline.args.in_attr
    assert generalist.network.model_cfg.denoiser.encode_text is True
    assert generalist.train_datasets.aistpp_train.enable_contact_supervision is False
    assert "aist_world_up_axis" not in generalist.train_datasets.aistpp_train
    assert generalist.scheduler.interval == "epoch"


def _static_contact_loss(enabled: bool) -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.zeros(1, 4, 6, requires_grad=True)
    inputs = {
        "mask": {
            "valid": torch.ones(1, 4, dtype=torch.bool),
            "spv_incam_only": torch.zeros(1, dtype=torch.bool),
            "2d_only": torch.zeros(1, dtype=torch.bool),
        },
        "static_gt_mask": torch.tensor([enabled]),
        "static_gt": torch.ones_like(logits),
    }
    outputs = {
        "decode_dict": {},
        "model_output": {"static_conf_logits": logits},
    }
    pipeline = SimpleNamespace(
        endecoder=None,
        weights=OmegaConf.create({"transl_w": 0.0, "static_conf_bce": 1.0}),
        args=OmegaConf.create({"static_conf": {"vel_thr": 0.15}}),
    )
    loss, loss_dict = compute_extra_global_loss(inputs, outputs, pipeline, "diffusion")
    return loss_dict["static_conf_loss"], logits


def test_music_only_contact_mask_enables_bce_and_gradient() -> None:
    contact_loss, logits = _static_contact_loss(enabled=True)
    assert torch.allclose(contact_loss, torch.tensor(0.6931472), atol=1e-6)
    contact_loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) > 0


def test_legacy_invalid_contact_mask_still_disables_bce() -> None:
    contact_loss, logits = _static_contact_loss(enabled=False)
    assert contact_loss.item() == 0.0
    contact_loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) == 0


def _target_case(mask: torch.Tensor, with_regression: bool) -> tuple[torch.Tensor, dict]:
    target = torch.zeros(2, 4, 151)
    inputs = {"mask": {"2d_only": mask}}
    if with_regression:
        prediction = torch.full_like(target, 7.0, requires_grad=True)
        inputs["regression_outputs"] = {
            "model_output": {"pred_x_start": prediction}
        }
    return target, inputs


def test_diffusion_only_fully_supervised_needs_no_regression_outputs() -> None:
    target, inputs = _target_case(torch.tensor([False, False]), False)
    actual = apply_regression_targets_to_2d_only(target, inputs)
    assert torch.equal(actual, torch.zeros_like(actual))


def test_diffusion_only_2d_without_regression_is_explicit_error() -> None:
    target, inputs = _target_case(torch.tensor([True, False]), False)
    with pytest.raises(RuntimeError, match="diffusion-only cannot train 2d_only"):
        apply_regression_targets_to_2d_only(target, inputs)


def test_diffusion_2d_with_regression_keeps_legacy_replacement() -> None:
    mask = torch.tensor([True, False])
    target, inputs = _target_case(mask, True)
    target[1] = 3.0
    actual = apply_regression_targets_to_2d_only(target, inputs)
    assert torch.all(actual[0] == 7.0)
    assert torch.all(actual[1] == 3.0)
    assert not actual.requires_grad


@pytest.mark.parametrize("channels", [34, 36])
def test_music_feature_wrong_width_fails(channels: int) -> None:
    with pytest.raises(ValueError, match=r"\[L, 35\]"):
        validate_musicfeat_v2(torch.zeros(120, channels))


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_music_feature_nonfinite_fails(bad_value: float) -> None:
    feature = torch.zeros(120, 35)
    feature[3, 4] = bad_value
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_musicfeat_v2(feature)


def test_music_feature_120x35_passes() -> None:
    validate_musicfeat_v2(torch.zeros(120, 35))


def test_aist_metric_translation_rejects_unscaled_centimetres() -> None:
    metric = torch.tensor([[0.0, 1.8, 0.0], [0.02, 1.8, 0.01], [0.04, 1.8, 0.02]])
    stats = validate_aist_metric_translation(metric, sequence_id="metric")
    assert stats["median_root_step_m"] < 0.1

    centimetres = metric * 90.0
    with pytest.raises(ValueError, match="not in GEM metric scale.*smpl_scaling"):
        validate_aist_metric_translation(centimetres, sequence_id="stale")


@pytest.mark.parametrize(
    ("sequence_length", "expected_length"),
    [(119, 119), (120, 120), (121, 120), (240, 120)],
)
def test_safe_training_crop_boundaries(
    sequence_length: int, expected_length: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("numpy.random.randint", lambda low, high: high - 1)
    start, length = select_aist_temporal_window(
        sequence_length=sequence_length,
        target_length=120,
        random_crop=True,
    )
    assert length == expected_length
    assert 0 <= start <= sequence_length - length
    if sequence_length > 120:
        assert start == sequence_length - 120


def test_deterministic_eval_clip_is_centered() -> None:
    assert select_aist_temporal_window(
        sequence_length=240,
        target_length=120,
        random_crop=False,
        eval_clip_mode="center",
    ) == (60, 120)


@pytest.mark.parametrize("difference", [0, 1, 2])
def test_strict_alignment_accepts_at_most_two_frames(difference: int) -> None:
    length, info = resolve_music_motion_alignment(
        sequence_id="seq",
        motion_frames=120,
        music_frames=120 + difference,
        music_feature_path="music.pt",
        strict=True,
        max_mismatch=2,
    )
    assert length == 120
    assert info["difference"] == difference


def test_strict_alignment_rejects_three_frames_with_context() -> None:
    with pytest.raises(
        ValueError,
        match=r"sequence_id=seq.*F_motion=120.*F_music=123.*difference=3.*music.pt",
    ):
        resolve_music_motion_alignment(
            sequence_id="seq",
            motion_frames=120,
            music_frames=123,
            music_feature_path="music.pt",
            strict=True,
            max_mismatch=2,
        )


def _condition_test_model() -> SimpleNamespace:
    music_embedder = nn.Linear(35, 32, bias=False)
    nn.init.constant_(music_embedder.weight, 1.0)
    return SimpleNamespace(
        pipeline=SimpleNamespace(
            args=OmegaConf.create(
                {
                    "in_attr": ["encoded_music"],
                    "disable_random_null_condition": True,
                }
            )
        ),
        condition_source={
            "image": ["f_imgseq"],
            "2d": ["obs", "f_cliffcam"],
            "camera": ["f_cam_angvel", "f_cam_tvel"],
            "audio": ["encoded_audio"],
            "music": ["encoded_music"],
        },
        music_mask_prob=0.1,
        audio_mask_prob=0.0,
        music_embedder=music_embedder,
        latent_dim=32,
        not_add_features=["encoded_music"],
        no_exist_keys=["obs", "observed_motion_3d", "multi_text_embed"],
        model_cfg=OmegaConf.create({"use_cond_exists_as_input": False}),
    )


def test_music_cfg_dropout_is_whole_sample_and_only_music_enters_f_cond() -> None:
    batch_size, length = 128, 6
    true = torch.ones(batch_size, length, dtype=torch.bool)
    false = torch.zeros_like(true)
    batch = {
        "B": batch_size,
        "L": length,
        "device": torch.device("cpu"),
        "has_text": torch.zeros(batch_size, dtype=torch.bool),
        "condition_mask": {
            "has_img_mask": false,
            "has_2d_mask": false,
            "has_cam_mask": false,
            "has_audio_mask": false,
            "has_music_mask": true,
            "j2d_visible_mask": torch.zeros(batch_size, length, 17, dtype=torch.bool),
        },
        "length": torch.full((batch_size,), length, dtype=torch.long),
        "music_embed": torch.ones(batch_size, length, 35),
        "target_x": torch.zeros(batch_size, length, 151),
    }
    torch.manual_seed(7)
    output = GEM.create_condition_mask(
        _condition_test_model(), batch, OmegaConf.create({}), "diffusion", train=True
    )
    active = output["f_cond"].abs().sum(dim=-1) > 0
    assert torch.equal(active, active[:, :1].expand_as(active))
    assert (~active[:, 0]).any() and active[:, 0].any()
    assert torch.count_nonzero(output["f_uncond"]) == 0
    assert output["f_cond"].shape == (batch_size, length, 32)


def test_encode_text_false_builds_no_text_modules() -> None:
    denoiser = NetworkEncoderRoPE(
        output_dim=151,
        xt_dim=151,
        latent_dim=32,
        num_layers=1,
        num_heads=4,
        encode_text=False,
    )
    assert denoiser.encode_text is False
    assert not hasattr(denoiser, "embed_text")
    assert not hasattr(denoiser, "text_encoder_layers")


class _CfgRecorder:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, x, _timesteps, y, **_kwargs):
        self.calls.append(y)
        return {"pred": x + y["f_cond"]}


def test_cfg_sampler_supports_music_without_encoded_text() -> None:
    recorder = _CfgRecorder()
    sampler = ClassifierFreeSampleModel(recorder)
    x = torch.ones(1, 2, 3)
    result = sampler(
        x,
        torch.zeros(1, dtype=torch.long),
        y={
            "f_cond": torch.full_like(x, 2),
            "f_uncond": torch.zeros_like(x),
            "scale": 2.5,
        },
    )
    assert torch.equal(recorder.calls[1]["f_cond"], torch.zeros_like(x))
    assert torch.equal(result["pred"], torch.full_like(x, 6))


def test_cfg_sampler_keeps_generalist_text_nulling() -> None:
    recorder = _CfgRecorder()
    sampler = ClassifierFreeSampleModel(recorder)
    x = torch.zeros(1, 2, 3)
    encoded_text = torch.ones(1, 4, 5)
    sampler(
        x,
        torch.zeros(1, dtype=torch.long),
        y={
            "f_cond": torch.ones_like(x),
            "f_uncond": torch.zeros_like(x),
            "encoded_text": encoded_text,
            "scale": 1.0,
        },
    )
    assert torch.equal(recorder.calls[0]["encoded_text"], encoded_text)
    assert torch.count_nonzero(recorder.calls[1]["encoded_text"]) == 0


def test_current_condition_exists_weights_are_not_remapped_to_denoiser() -> None:
    key = "cond_exists_embedder.encoded_music.0.weight"
    remapped, report = remap_legacy_state_dict({key: torch.ones(2, 2)})
    assert list(remapped) == [key]
    assert report["renamed"] == 0


def test_tiny_music_only_diffusion_forward_loss_and_backward() -> None:
    model_cfg = OmegaConf.create(
        {
            "diffusion": {
                "sampler": "ddim",
                "train_timestep_respacing": "",
                "test_timestep_respacing": "2",
                "gen_only_test_timestep_respacing": "2",
                "schedule_sampler_type": "uniform",
                "noise_schedule": "cosine",
                "sigma_small": True,
                "guidance_param": 2.5,
                "ddim_eta": 0.0,
            },
            "denoiser": {
                "_target_": "gem.network.gem_denoiser.NetworkEncoderRoPE",
                "output_dim": 151,
                "xt_dim": 151,
                "latent_dim": 32,
                "num_layers": 1,
                "num_heads": 4,
                "mlp_ratio": 2,
                "encode_text": False,
                "args": {"pred_fullcam": False},
            },
        }
    )
    args = OmegaConf.create({"out_attr": {"pred_cam": 3}})
    model = GEMDiffusion(model_cfg=model_cfg, args=args, latent_dim=32)
    model.train()
    batch_size, length = 2, 120
    target = torch.randn(batch_size, length, 151)
    output = model.forward_train(
        {
            "length": torch.full((batch_size,), length, dtype=torch.long),
            "motion": target,
            "f_cond": torch.randn(batch_size, length, 32),
            "f_empty": torch.zeros(batch_size, length, 32),
            "mask": {
                "valid": torch.ones(batch_size, length, dtype=torch.bool),
                "2d_only": torch.zeros(batch_size, dtype=torch.bool),
            },
            "sample_indices_dict": {
                "body_pose": (0, 126),
                "betas": (126, 136),
                "global_orient": (136, 142),
                "global_orient_gv": (142, 148),
                "local_transl_vel": (148, 151),
            },
        },
        mode="diffusion",
    )
    assert output["pred_x"].shape == (batch_size, length, 151)
    loss = torch.nn.functional.mse_loss(output["pred_x"], target)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert any(grad is not None and torch.isfinite(grad).all() for grad in grads)
