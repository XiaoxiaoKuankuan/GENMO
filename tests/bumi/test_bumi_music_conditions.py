"""验证 BUMI 音乐条件和共享扩散主干，不依赖已退役的 SMPL 实验配置。

从旧 specialist 混合测试中保留 EDGE35、数据时间对齐、CFG 和音乐权重键回归；
新增部分直接调用 BumiMusicGEM 的条件处理，并以 30D 运动加 2D 接触 head 的
微型网络验证前向和反向传播。所有输入为 CPU 合成张量，不加载正式权重或训练数据。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from gem.bumi_gem import BumiMusicGEM
from gem.datasets.aistpp.aistplusplus import (
    resolve_music_motion_alignment,
    select_aist_temporal_window,
    validate_aist_metric_translation,
    validate_musicfeat_v2,
)
from gem.network.gem_cfg_sampler import ClassifierFreeSampleModel
from gem.network.gem_diffusion import GEMDiffusion
from gem.robots.bumi.feature_codec import BUMI_FEATURE_SLICES
from gem.utils.ckpt_compat import remap_legacy_state_dict


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


def test_current_condition_exists_weights_are_not_remapped_to_denoiser() -> None:
    key = "cond_exists_embedder.encoded_music.0.weight"
    remapped, report = remap_legacy_state_dict({key: torch.ones(2, 2)})
    assert list(remapped) == [key]
    assert report["renamed"] == 0


def test_bumi_music_dropout_is_causal_to_valid_music_and_whole_sample() -> None:
    batch_size, length = 128, 8
    embedder = nn.Linear(35, 32, bias=False)
    nn.init.constant_(embedder.weight, 1.0)
    model = SimpleNamespace(
        music_embedder=embedder,
        music_mask_prob=0.1,
        model_cfg=OmegaConf.create({"use_cond_exists_as_input": False}),
    )
    valid = torch.ones(batch_size, length, dtype=torch.bool)
    valid[:, -2:] = False
    batch = {
        "B": batch_size,
        "L": length,
        "device": torch.device("cpu"),
        "music_embed": torch.ones(batch_size, length, 35),
        "condition_mask": {"has_music_mask": torch.ones_like(valid)},
        "mask": {"valid": valid},
        "length": torch.full((batch_size,), length - 2, dtype=torch.long),
        "target_x": torch.ones(batch_size, length, 30),
    }
    torch.manual_seed(7)
    result = BumiMusicGEM.create_condition_mask(model, batch, None, None, train=True)
    active = result["f_cond"].abs().sum(-1) > 0
    assert active[:, 0].any() and (~active[:, 0]).any()
    assert torch.equal(active[:, :-2], active[:, :1].expand(-1, length - 2))
    assert not active[:, -2:].any()
    assert torch.count_nonzero(result["f_uncond"]) == 0
    assert torch.count_nonzero(result["motion"][:, -2:]) == 0
    assert torch.equal(result["music_dropout_mask"], ~active[:, 0])


def test_tiny_bumi_music_diffusion_forward_and_both_heads_backward() -> None:
    args = OmegaConf.create(
        {
            "motion_backend": "bumi",
            "out_attr": {"static_conf_logits": 2},
        }
    )
    config = OmegaConf.create(
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
                "output_dim": 30,
                "xt_dim": 30,
                "njoints": 30,
                "pred_cam_dim": 0,
                "static_conf_dim": 2,
                "avgbeta": False,
                "latent_dim": 32,
                "num_layers": 1,
                "num_heads": 4,
                "mlp_ratio": 2,
                "encode_text": False,
                "input_remove_global": False,
                "args": {"pred_fullcam": False},
            },
        }
    )
    model = GEMDiffusion(
        model_cfg=config,
        args=args,
        latent_dim=32,
        observed_motion_3d_dim=30,
        encoded_music_dim=35,
    ).train()
    assert not hasattr(model.denoiser, "embed_text")
    batch_size, length = 2, 120
    target = torch.randn(batch_size, length, 30)
    output = model.forward_train(
        {
            "length": torch.full((batch_size,), length, dtype=torch.long),
            "motion": target,
            "f_cond": torch.randn(batch_size, length, 32),
            "f_empty": torch.zeros(batch_size, length, 32),
            "mask": {"valid": torch.ones(batch_size, length, dtype=torch.bool)},
            "sample_indices_dict": dict(BUMI_FEATURE_SLICES),
        },
        mode="diffusion",
    )
    assert output["pred_x"].shape == (batch_size, length, 30)
    assert output["static_conf_logits"].shape == (batch_size, length, 2)
    loss = torch.nn.functional.mse_loss(output["pred_x"], target)
    loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
        output["static_conf_logits"], torch.ones_like(output["static_conf_logits"])
    )
    assert torch.isfinite(loss)
    loss.backward()
    for head in (model.denoiser.final_layer, model.denoiser.static_conf_head):
        grads = [value.grad for value in head.parameters() if value.requires_grad]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(torch.count_nonzero(g) for g in grads)
