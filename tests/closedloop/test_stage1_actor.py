"""验证 Stage 1 Actor 对现有数据契约的适配及条件扩散的隔离性质。

本文件直接复用第 3 步 Dataset 测试夹具，构造真实的 120 帧、30 Hz、physical qpos30
样本；不会再定义另一套 batch 字段。模型使用真实 BumiEndecoder、原 NetworkEncoderRoPE
和 timm 音乐 MLP，仅缩小隐层与层数。测试所用非单位均值/方差明确标注为 placeholder，
只写入 pytest 临时目录；它们用于发现漏归一化和错误占位问题，不是正式训练统计量。

覆盖条件与监督隔离、可变 H/P、逐坐标前缀约束、每一步 DDIM 的已知值保持、仅音乐 CFG、
非零无效槽的隔离、新残差的初始兼容性和可学习性，以及历史顺序/相对时间的可辨识性。
测试保留原 qpos30 解码和独立 contact2 输出，不启动正式训练，不构成动力学或实机证明。
"""

from __future__ import annotations

import copy
import json

import pytest
import torch
from timm.models.vision_transformer import Mlp

from gem.closedloop.actor import Stage1Actor
from gem.closedloop.contracts import (
    PROPRIO_SLICES,
    STAGE1_CONDITION_KEYS,
    validate_stage1_condition_batch,
    validate_stage1_training_batch,
)
from gem.closedloop.stage1_dataset import collate_stage1_training_samples
from gem.network.gem_denoiser import NetworkEncoderRoPE
from gem.robots.bumi.endecoder import STATS_CONTRACT_VERSION, BumiEndecoder
from gem.robots.bumi.feature_codec import (
    BUMI_ANCHOR_MODE,
    BUMI_FEATURE_SLICES,
    BUMI_REPRESENTATION_CONTRACT_VERSION,
)
from gem.robots.bumi.kinematics import BumiKinematics
from tests.closedloop.test_stage1_dataset import KINEMATICS_PATH, _dataset, _write_dataset


@pytest.fixture
def actor_factory(tmp_path):
    """创建小型真实 Actor；所有临时文件均由 pytest basetemp 统一回收。"""

    torch.manual_seed(812)
    kinematics = BumiKinematics(KINEMATICS_PATH)
    stats_path = tmp_path / "placeholder_qpos30_stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "contract_version": STATS_CONTRACT_VERSION,
                "representation_contract_version": BUMI_REPRESENTATION_CONTRACT_VERSION,
                "robot_name": "bumi",
                "feature_dim": 30,
                "anchor_mode": BUMI_ANCHOR_MODE,
                "quaternion_convention": "wxyz",
                "feature_slices": dict(BUMI_FEATURE_SLICES),
                "joint_names": list(kinematics.joint_order),
                "kinematics_sha256": kinematics.kinematics_sha256,
                "mean": torch.linspace(-0.3, 0.2, 30).tolist(),
                "std": torch.linspace(0.4, 1.3, 30).tolist(),
                "training_clip_std_min": 0.01,
                "root_height_reference_m": float(kinematics.default_qpos[2]),
                "is_placeholder": True,
            }
        ),
        encoding="utf-8",
    )
    root = _write_dataset(tmp_path / "dataset", length=180)

    def make(*, history_steps=50, prefix=6, starts=(45, 47), proprio_scales=(1, 1, 1, 1)):
        dataset = _dataset(
            root,
            history_steps=history_steps,
            prefix_min_frames=prefix,
            prefix_max_frames=prefix,
        )
        batch = collate_stage1_training_samples(
            [dataset.get_window(0, start_frame=start) for start in starts]
        )
        endecoder = BumiEndecoder(
            KINEMATICS_PATH,
            stats_path,
            allow_placeholder_stats=True,
            enable_contact_targets=False,
        )
        denoiser = NetworkEncoderRoPE(
            output_dim=30,
            xt_dim=30,
            max_len=120,
            latent_dim=32,
            num_layers=2,
            num_heads=4,
            mlp_ratio=2.0,
            pred_cam_dim=0,
            static_conf_dim=2,
            dropout=0.0,
            avgbeta=False,
            njoints=30,
            encode_text=False,
            use_text_pos_enc=False,
            allow_autoregressive=False,
        )
        # 让真实注意力参与计算，避免原 block 的零 gate 掩盖 padding 泄漏。
        with torch.no_grad():
            for block in denoiser.blocks:
                block.gate_msa.fill_(0.35)
                block.gate_mlp.fill_(0.2)
        actor = Stage1Actor(
            endecoder=endecoder,
            denoiser=denoiser,
            music_embedder=Mlp(35, hidden_features=64, out_features=32, drop=0.0),
            cond_exists_embedder=Mlp(33, hidden_features=32, out_features=32, drop=0.0),
            history_hidden_dim=24,
            history_steps=history_steps,
            proprio_scales=proprio_scales,
            music_mask_prob=0.0,
            diffusion_steps=1000,
        )
        actor.eval()
        return actor, batch

    return make


def _conditions(batch):
    return {key: batch[key].clone() for key in STAGE1_CONDITION_KEYS}


def _activate_branches(actor):
    """模拟经过学习的新增分支，让隔离测试不被零初始化平凡满足。"""

    with torch.no_grad():
        for branch in (actor.history_encoder, actor.prefix_encoder):
            torch.nn.init.normal_(branch.out_proj.weight, std=0.1)
            branch.out_proj.bias.fill_(0.05)


def _assert_same_samples(first, second, valid=None):
    for key in ("qpos30", "contact", "contact_logits", "normalized", "qpos"):
        left, right = first[key], second[key]
        if valid is not None:
            left, right = left[valid], right[valid]
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_dataset_adapter_normalizes_physical_prefix_and_masks_unknown(actor_factory):
    actor, batch = actor_factory(proprio_scales=(1, 2, 3, 4))
    batch["proprio_history_valid"][:, :7] = False
    batch["proprio_history"][:, :7] = 900.0
    batch["known_qpos30"][~batch["known_qpos30_mask"]] = -700.0
    adapted = actor.adapt_conditions(batch)
    known = batch["known_qpos30_mask"]
    expected = actor.endecoder.normalize(batch["known_qpos30"])
    torch.testing.assert_close(adapted["known_x"][known], expected[known])
    assert torch.count_nonzero(adapted["known_x"][~known]) == 0
    assert torch.count_nonzero(adapted["history"][:, :7]) == 0
    torch.testing.assert_close(adapted["music_embed"], batch["music_features"])
    for scale, (start, stop) in zip((1, 2, 3, 4), PROPRIO_SLICES.values()):
        torch.testing.assert_close(
            adapted["history"][:, 7:, start:stop],
            batch["proprio_history"][:, 7:, start:stop] / scale,
        )
    torch.testing.assert_close(
        adapted["history_relative_times"][:, 7:],
        (batch["proprio_history_times"] - batch["decision_time"][:, None])[:, 7:],
        check_dtype=False,
    )


def test_zero_residual_keeps_original_music_condition_and_denoiser(actor_factory):
    actor, batch = actor_factory()
    adapted = actor.adapt_conditions(batch)
    expected = actor.music_embedder(batch["music_features"])
    expected = actor.cond_exists_embedder(
        torch.cat((expected, batch["music_valid"][..., None].to(expected)), dim=-1)
    )
    actual = actor.encode_conditions(adapted)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    time = torch.tensor([5, 13])
    output = actor.training_forward(batch, timesteps=time, noise=torch.randn(2, 120, 30))
    previous = actor.denoiser(
        output["xt"],
        time,
        y={"f_cond": expected, "length": batch["future_valid"].sum(dim=1)},
        inputs={},
    )
    unknown = ~batch["known_qpos30_mask"]
    torch.testing.assert_close(
        output["pred_x_start"][unknown], previous["pred_x_start"][unknown], rtol=0, atol=0
    )
    torch.testing.assert_close(
        output["static_conf_logits"], previous["static_conf_logits"], rtol=0, atol=0
    )


def test_sampling_only_conditions_and_target_mutation_has_no_effect(actor_factory):
    actor, batch = actor_factory()
    _activate_branches(actor)
    conditions = _conditions(batch)
    changed = copy.deepcopy(batch)
    changed["target_qpos30"].fill_(234.0)
    changed["target_contact"].fill_(1.0)
    changed["target_qpos30_valid"].fill_(False)
    changed["target_contact_valid"].fill_(False)
    noise = torch.randn_like(batch["known_qpos30"])
    expected = actor.sample(conditions, steps=3, noise=noise)
    for value in (batch, changed):
        torch.testing.assert_close(
            actor.encode_conditions(actor.adapt_conditions(conditions)),
            actor.encode_conditions(actor.adapt_conditions(value)),
            rtol=0,
            atol=0,
        )
        _assert_same_samples(expected, actor.sample(value, steps=3, noise=noise))
    assert expected["qpos30"].shape == (2, 120, 30)
    assert expected["contact"].shape == (2, 120, 2)
    assert expected["qpos"].shape == (2, 120, 28)


def test_nonzero_invalid_history_unknown_prefix_and_missing_music_are_isolated(actor_factory):
    actor, batch = actor_factory(starts=(168, 170))
    _activate_branches(actor)
    conditions = _conditions(batch)
    conditions["proprio_history_valid"][:, :11] = False
    conditions["music_valid"][:, 3:5] = False
    altered = copy.deepcopy(conditions)
    altered["proprio_history"][~altered["proprio_history_valid"]] = 19_000.0
    altered["known_qpos30"][~altered["known_qpos30_mask"]] = -31_000.0
    altered["music_features"][~altered["music_valid"]] = 47_000.0
    noise = torch.randn_like(batch["known_qpos30"])
    second_noise = noise.clone()
    second_noise[~conditions["future_valid"]] = -15_000.0
    first = actor.sample(conditions, steps=3, noise=noise)
    second = actor.sample(altered, steps=3, noise=second_noise)
    _assert_same_samples(first, second, valid=conditions["future_valid"])


@pytest.mark.parametrize("history_steps", [1, 7, 50, 61])
def test_empty_history_and_zero_prefix_are_finite_with_configurable_h(actor_factory, history_steps):
    actor, batch = actor_factory(history_steps=history_steps, prefix=0, starts=(0,))
    _activate_branches(actor)
    assert not batch["proprio_history_valid"].any()
    assert not batch["known_qpos30_mask"].any()
    output = actor.sample(_conditions(batch), steps=2)
    for key in ("qpos30", "qpos", "contact", "normalized"):
        assert torch.isfinite(output[key]).all(), key


@pytest.mark.parametrize("prefix", [0, 1, 6, 119])
def test_each_sampling_step_preserves_partial_coordinate_prefix(actor_factory, prefix):
    actor, batch = actor_factory(prefix=prefix)
    _activate_branches(actor)
    mask = batch["known_qpos30_mask"]
    expected = actor.endecoder.normalize(batch["known_qpos30"])
    if prefix:
        assert mask[:, prefix - 1, 2:].all()
        assert not mask[:, prefix - 1, :2].any()
    output = actor.sample(_conditions(batch), steps=4, return_trace=True)
    assert len(output["trace"]) >= 4
    for state in (*output["trace"], output["normalized"]):
        torch.testing.assert_close(state[mask], expected[mask], atol=0, rtol=0)
    # 最终 physical 值必须逐位保留；不能依赖 normalize/denormalize 的近似往返。
    assert torch.equal(output["qpos30"][mask], batch["known_qpos30"][mask])


def test_training_noise_changes_only_unknown_and_target_normalization_is_explicit(actor_factory):
    actor, batch = actor_factory(starts=(168, 170))
    time = torch.tensor([9, 17])
    noise = torch.randn_like(batch["target_qpos30"])
    first = actor.training_forward(batch, timesteps=time, noise=noise)
    second = actor.training_forward(batch, timesteps=time, noise=noise + 3.0)
    mask = batch["known_qpos30_mask"]
    known_x = actor.endecoder.normalize(batch["known_qpos30"])
    for result in (first, second):
        torch.testing.assert_close(result["xt"][mask], known_x[mask], atol=0, rtol=0)
        torch.testing.assert_close(result["pred_x_start"][mask], known_x[mask], atol=0, rtol=0)
        valid = batch["target_qpos30_valid"]
        expected = actor.endecoder.normalize(batch["target_qpos30"])
        torch.testing.assert_close(result["target_x_start"][valid], expected[valid])
    valid_unknown = batch["target_qpos30_valid"] & ~mask
    assert torch.all(first["xt"][valid_unknown] != second["xt"][valid_unknown])
    torch.testing.assert_close(
        first["xt"][~batch["future_valid"]],
        second["xt"][~batch["future_valid"]],
        atol=0,
        rtol=0,
    )


def test_invalid_target_padding_and_terminal_delta_never_enter_denoising(actor_factory):
    actor, batch = actor_factory(starts=(168, 170))
    _activate_branches(actor)
    altered = copy.deepcopy(batch)
    altered["target_qpos30"][~altered["target_qpos30_valid"]] = 62_000.0
    noise = torch.randn_like(batch["target_qpos30"])
    time = torch.tensor([400, 700])
    first = actor.training_forward(batch, timesteps=time, noise=noise)
    second = actor.training_forward(altered, timesteps=time, noise=noise)
    for key in ("xt", "target_x_start", "pred_x_start", "static_conf_logits"):
        torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)
    for index, valid in enumerate(batch["future_valid"]):
        terminal = int(valid.sum()) - 1
        assert not batch["target_qpos30_valid"][index, terminal, :2].any()
        assert batch["target_qpos30_valid"][index, terminal, 2:].all()


def test_prefix_mask_is_a_condition_even_when_normalized_known_value_is_zero(actor_factory):
    actor, batch = actor_factory()
    _activate_branches(actor)
    conditions = _conditions(batch)
    conditions["known_qpos30"][..., 12] = actor.endecoder.mean[12]
    altered = copy.deepcopy(conditions)
    altered["known_qpos30_mask"][..., 12] = False
    first = actor.adapt_conditions(conditions)
    second = actor.adapt_conditions(altered)
    torch.testing.assert_close(first["known_x"], second["known_x"], atol=0, rtol=0)
    assert not torch.equal(actor.encode_conditions(first), actor.encode_conditions(second))


def test_all_padding_inference_does_not_produce_all_mask_attention_nan(actor_factory):
    actor, batch = actor_factory(prefix=0)
    _activate_branches(actor)
    conditions = _conditions(batch)
    conditions["future_valid"].fill_(False)
    conditions["music_valid"].fill_(False)
    conditions["proprio_history_valid"].fill_(False)
    output = actor.sample(conditions, steps=2)
    for key in ("qpos30", "qpos", "contact", "contact_logits", "normalized"):
        assert torch.isfinite(output[key]).all(), key
        assert torch.count_nonzero(output[key]) == 0, key


def test_cfg_drops_only_music_and_uses_same_history_prefix_inputs(actor_factory):
    actor, batch = actor_factory()
    _activate_branches(actor)
    adapted = actor.adapt_conditions(batch)
    captures = {"history": [], "prefix": []}

    def capture(name):
        def hook(_module, args):
            captures[name].append(tuple(value.detach().clone() for value in args))

        return hook

    handles = [
        actor.history_encoder.register_forward_pre_hook(capture("history")),
        actor.prefix_encoder.register_forward_pre_hook(capture("prefix")),
    ]
    try:
        conditional = actor.encode_conditions(adapted, drop_music=torch.zeros(2, dtype=torch.bool))
        unconditional = actor.encode_conditions(adapted, drop_music=torch.ones(2, dtype=torch.bool))
    finally:
        for handle in handles:
            handle.remove()
    for name, values in captures.items():
        assert len(values) == 2, name
        for first, second in zip(*values):
            torch.testing.assert_close(first, second, atol=0, rtol=0)
    original_music = actor.music_embedder(batch["music_features"])
    cond_music = actor.cond_exists_embedder(
        torch.cat((original_music, torch.ones(2, 120, 1)), dim=-1)
    )
    uncond_music = actor.cond_exists_embedder(torch.zeros(2, 120, 33))
    torch.testing.assert_close(
        conditional - unconditional, cond_music - uncond_music, atol=2e-7, rtol=2e-5
    )
    sample_conditions = []

    def record_denoiser(_module, _args, kwargs):
        sample_conditions.append(kwargs["y"]["f_cond"].detach().clone())

    handle = actor.denoiser.register_forward_pre_hook(record_denoiser, with_kwargs=True)
    try:
        actor.sample(_conditions(batch), steps=2, guidance_scale=2.5)
    finally:
        handle.remove()
    # 采样真实经过相同两种条件；batch 拼接与分次调用均允许。
    individual = [value for captured in sample_conditions for value in captured.split(2)]
    assert any(torch.equal(value, conditional) for value in individual)
    assert any(torch.equal(value, unconditional) for value in individual)
    assert all(
        torch.equal(value, conditional) or torch.equal(value, unconditional) for value in individual
    )


def test_new_residual_branches_receive_gradients_from_real_actor(actor_factory):
    actor, batch = actor_factory()
    actor.train()
    output = actor.training_forward(
        batch, timesteps=torch.tensor([7, 21]), noise=torch.randn(2, 120, 30)
    )
    unknown = batch["target_qpos30_valid"] & ~batch["known_qpos30_mask"]
    loss = (output["pred_x_start"][unknown] - output["target_x_start"][unknown]).square().mean()
    loss.backward()
    for branch in (actor.history_encoder, actor.prefix_encoder):
        gradient = branch.out_proj.weight.grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
    optimizer = torch.optim.SGD(
        list(actor.history_encoder.parameters()) + list(actor.prefix_encoder.parameters()), lr=0.1
    )
    before = actor.encode_conditions(actor.adapt_conditions(batch)).detach()
    optimizer.step()
    after = actor.encode_conditions(actor.adapt_conditions(batch)).detach()
    assert not torch.equal(before, after)


def test_history_order_and_relative_time_change_learned_condition(actor_factory):
    actor, batch = actor_factory(history_steps=7)
    _activate_branches(actor)
    conditions = _conditions(batch)
    assert conditions["proprio_history_valid"].all()
    conditions["proprio_history"] = torch.randn_like(conditions["proprio_history"])
    baseline = actor.encode_conditions(actor.adapt_conditions(conditions))
    reversed_order = copy.deepcopy(conditions)
    reversed_order["proprio_history"] = conditions["proprio_history"].flip(1)
    delayed = copy.deepcopy(conditions)
    delayed["proprio_history_times"] -= 0.35
    # 整体时钟平移不应改变相对时间；历史单独变旧则是新的有效条件。
    translated = copy.deepcopy(conditions)
    for key in ("proprio_history_times", "future_times", "decision_time"):
        translated[key] += 16.0
    for changed in (reversed_order, delayed, translated):
        validate_stage1_condition_batch(changed, history_steps=7)
    assert not torch.allclose(
        baseline, actor.encode_conditions(actor.adapt_conditions(reversed_order)), atol=1e-6, rtol=0
    )
    assert not torch.allclose(
        baseline, actor.encode_conditions(actor.adapt_conditions(delayed)), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        baseline, actor.encode_conditions(actor.adapt_conditions(translated)), atol=2e-6, rtol=1e-6
    )


def test_training_uses_existing_validator_and_rejects_wrong_history_shape(actor_factory):
    actor, batch = actor_factory(history_steps=7)
    validate_stage1_training_batch(batch, history_steps=7)
    wrong = copy.deepcopy(batch)
    wrong["proprio_history"] = wrong["proprio_history"][:, :-1]
    with pytest.raises(ValueError, match="proprio_history"):
        actor.training_forward(wrong)
    wrong = copy.deepcopy(batch)
    wrong["known_qpos30_mask"][:, 10, 12] = True
    with pytest.raises(ValueError, match="prefix"):
        actor.sample(wrong, steps=2)
