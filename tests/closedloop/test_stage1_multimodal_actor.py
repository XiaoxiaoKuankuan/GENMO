"""音乐/英文文本联合Stage1的条件隔离、梯度和DDP固定图回归。

测试复用真实BUMI编解码器与既有临时音乐数据，仅把Transformer缩成两层32维。
文本使用明确的随机夹具，不加载T5权重、不运行正式训练。覆盖互斥输入、token padding、
缺失文本sentinel、整个文本块残差门控、单处CFG丢弃、前缀保持、逐样本loss和双rank
连续反传。临时checkpoint/进程通信仅写pytest basetemp，不能视为生成质量或物理证明。
"""

import copy

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from timm.models.vision_transformer import Mlp

from gem.closedloop.actor import Stage1Actor
from gem.closedloop.checkpoint import load_stage1_checkpoint, save_stage1_checkpoint
from gem.closedloop.contracts import validate_stage1_condition_batch, validate_stage1_training_batch
from gem.closedloop.training import stage1_conditions, prepare_text_batch
from gem.network.gem_denoiser import NetworkEncoderRoPE
from gem.robots.bumi.losses import _masked_mean, _PER_SAMPLE_REDUCTION
from gem.robots.bumi.endecoder import BumiEndecoder
from tests.closedloop.test_stage1_actor import actor_factory


def _new_joint(endecoder):
    denoiser = NetworkEncoderRoPE(output_dim=30, xt_dim=30, max_len=120,
        latent_dim=32, num_layers=2, num_heads=4, mlp_ratio=2, dropout=0,
        pred_cam_dim=0, static_conf_dim=2, avgbeta=False, njoints=30,
        encode_text=True, encoded_text_dim=1024, text_mask_prob=0,
        text_encoder_cfg={"mode": "all", "cross_attn_type": "mha"},
        use_text_pos_enc=True, allow_autoregressive=False)
    actor = Stage1Actor(endecoder, denoiser, Mlp(35, hidden_features=64, out_features=32, drop=0),
        Mlp(33, hidden_features=32, out_features=32, drop=0), history_hidden_dim=24, music_mask_prob=0,
        semantic_mask_prob=0)
    return actor


@pytest.fixture
def joint_actor(actor_factory):
    old, batch = actor_factory()
    actor = _new_joint(old.endecoder)
    with torch.no_grad():
        for block in actor.denoiser.text_encoder_layers.values():
            block.gate_cross_attn.fill_(0.3)
            block.gate_mlp.fill_(0.2)
    batch.update(text_features=torch.randn(2, 150, 1024),
        text_attention_mask=torch.zeros(2, 150, dtype=torch.bool),
        has_text=torch.tensor([False, True]))
    batch["text_attention_mask"][1, :7] = True
    batch["music_valid"][1] = False
    actor.eval()
    return actor, batch


def _prediction(actor, conditions):
    adapted = actor.adapt_conditions(conditions)
    return actor._denoise(torch.zeros_like(adapted["known_x"]),
        torch.tensor([4, 7]), adapted, actor.encode_conditions(adapted))["pred_x_start"]


def test_joint_masks_exclusivity_and_whole_text_block_gate(joint_actor):
    actor, batch = joint_actor
    validate_stage1_training_batch(batch)
    conditions = stage1_conditions(batch)
    expected = _prediction(actor, conditions)
    noisy = copy.deepcopy(conditions)
    noisy["text_features"][~noisy["text_attention_mask"]] = 918.0
    torch.testing.assert_close(_prediction(actor, noisy), expected, rtol=0, atol=0)
    # 缺失文本样本的整个文本块必须无影响，包括其FFN，而非仅屏蔽cross-attention。
    with torch.no_grad():
        for block in actor.denoiser.text_encoder_layers.values():
            block.mlp.fc2.bias.add_(11)
    actual = _prediction(actor, conditions)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    assert not torch.allclose(actual[1], expected[1])
    bad = copy.deepcopy(conditions)
    bad["music_valid"][1] = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_stage1_condition_batch(bad)


def test_semantic_dropout_keeps_fixed_graph_and_sampling_prefix(joint_actor):
    actor, batch = joint_actor
    actor.train()
    actor.semantic_mask_prob = 1
    out = actor(batch)
    assert out["semantic_dropout_mask"].all()
    (out["pred_x_start"].square().mean() + out["static_conf_logits"].square().mean()).backward()
    assert not [(name, p.grad is None) for name, p in actor.named_parameters() if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
    actor.eval()
    generated = actor.sample(stage1_conditions(batch), steps=2, guidance_scale=2.5)
    known = batch["known_qpos30_mask"]
    assert torch.isfinite(generated["qpos30"]).all()
    assert torch.equal(generated["qpos30"][known], batch["known_qpos30"][known])


def test_joint_checkpoint_contract_round_trip(joint_actor, tmp_path):
    actor, _ = joint_actor
    path = tmp_path / "joint.pt"
    payload = save_stage1_checkpoint(actor, path, config={}, global_step=0)
    assert payload["stage1_contract_version"] == "genmo.bumi_closedloop.music_text.v1"
    restored = copy.deepcopy(actor)
    load_stage1_checkpoint(restored, path)
    for key, value in actor.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])


def test_music_only_onnx_rejects_joint_conditions(joint_actor):
    from gem.runtime.closedloop_stage1_onnx import Stage1GuidedDenoiser
    with pytest.raises(ValueError, match="multimodal PyTorch"):
        Stage1GuidedDenoiser(joint_actor[0])


def test_per_sample_reduction_and_zero_support():
    value = torch.tensor([[1., 1., 1.], [9., 9., 9.], [99., 99., 99.]], requires_grad=True)
    mask = torch.tensor([[True, True, True], [True, False, False], [False, False, False]])
    token = _PER_SAMPLE_REDUCTION.set(True)
    try:
        loss = _masked_mean(value, mask)
        assert loss.item() == 5
        loss.backward()
        assert value.grad[2].eq(0).all()
    finally:
        _PER_SAMPLE_REDUCTION.reset(token)
    assert _masked_mean(value, mask).item() == 3


def test_online_encoder_only_real_pending_rows(joint_actor):
    actor, batch = joint_actor
    calls = []
    class Encoder:
        def encode(self, captions, device):
            calls.extend(captions)
            torch.randn(19)  # 模拟首次载入冻结T5时初始化模块消耗随机数。
            return torch.ones(1, 150, 1024), torch.ones(1, 150, dtype=torch.bool)
    actor.__dict__["_frozen_text_encoder"] = Encoder()
    batch["captions"] = ["", "walk forward"]
    batch["text_needs_encoding"] = torch.tensor([False, True])
    batch["text_attention_mask"][1] = False
    rng = torch.get_rng_state().clone()
    prepared = prepare_text_batch(actor, batch)
    assert torch.equal(rng, torch.get_rng_state())
    assert calls == ["walk forward"]
    validate_stage1_training_batch(prepared)


def _ddp_worker(rank, rendezvous, assets, state, batch):
    torch.set_num_threads(1)
    actor = _new_joint(BumiEndecoder(*assets, allow_placeholder_stats=True, enable_contact_targets=False))
    actor.load_state_dict(state)
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2)
    try:
        model = DistributedDataParallel(actor.train(), find_unused_parameters=False, broadcast_buffers=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        for step in range(3):
            local = copy.deepcopy(batch)
            text = (rank + step) % 2 == 1
            local["has_text"].fill_(text)
            local["text_attention_mask"].fill_(False)
            local["text_attention_mask"][:, :3] = text
            local["music_valid"] = local["future_valid"] & (not text)
            actor.semantic_mask_prob = 1.0 if step == 2 else 0.0
            optimizer.zero_grad(set_to_none=True)
            out = model(local)
            (out["pred_x_start"].square().mean() + out["static_conf_logits"].square().mean()).backward()
            assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
            optimizer.step()
        for parameter in model.parameters():
            reference = parameter.detach().clone()
            dist.broadcast(reference, src=0)
            assert torch.equal(reference, parameter)
    finally:
        dist.destroy_process_group()


def test_two_rank_alternating_modalities_fixed_graph(joint_actor, tmp_path):
    actor, batch = joint_actor
    torch.multiprocessing.start_processes(_ddp_worker,
        args=(str(tmp_path / "gloo_init"),
            (str(actor.endecoder.kinematics.kinematics_path), actor.endecoder.stats_path),
            actor.state_dict(), batch), nprocs=2, start_method="spawn")
