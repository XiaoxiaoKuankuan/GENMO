"""从零Actor的合同回归，复用原Stage1小型真实Transformer和显式测试数据。

这些CPU测试只验证数学、mask、无泄漏与导数接线，不能作为真实Newton或学生执行收益。
固定噪声比较旧60帧监督；未来分支单独检查梯度；采样前后逐模块模式一致。静态构造测试
禁止调用旧权重加载函数并核对随机种子，不生成任何正式训练资产。
"""

import copy

import torch

from gem.closedloop.joint_physics.actor import JointPhysicsActor
from gem.closedloop.mb_control.contracts import ForecastContext


def test_supervision_parity_masks_and_future_gradient(actor_factory):
    base, batch = actor_factory(motion_frames=60, starts=(45, 48))
    actor = JointPhysicsActor(base).eval()
    t = torch.tensor([0, 999])
    noise = torch.randn(2, 60, 30)
    old = base.training_forward(batch, t, noise)
    new = actor.denoise_supervised(batch, None, t, noise)
    for key in ("pred_x_start", "static_conf_logits", "target_x_start", "xt"):
        torch.testing.assert_close(new[key], old[key], rtol=0, atol=0)
    known = batch["known_qpos30_mask"]
    torch.testing.assert_close(new["xt"][known], new["target_x_start"][known], rtol=0, atol=0)
    assert torch.equal(new["unknown_mask"], batch["target_qpos30_valid"] & ~known)
    contexts = [ForecastContext(torch.randn(1, 58), torch.ones(1, dtype=torch.bool), tick, tick+120)
                for tick in (900, 960)]
    out = actor.denoise_supervised(batch, contexts, t, noise)
    out["pred_x_start"].square().mean().backward()
    assert actor.future_encoder.output.weight.grad.norm() > 0
    # 原架构零初始化的末层在第一次更新前允许阻断更早Future层，这是预期行为。
    assert torch.isfinite(actor.future_encoder.output.weight.grad).all()


def test_labels_not_conditions_and_invalid_future_zero(actor_factory):
    base, batch = actor_factory(motion_frames=60, starts=(45,))
    actor = JointPhysicsActor(base).eval()
    changed = copy.deepcopy(batch)
    changed["target_qpos30"][~batch["known_qpos30_mask"]] += 17
    changed["target_contact"] = 1 - changed["target_contact"]
    changed["meta"][0]["ground_supervision"] = {"sentinel": "not_actor_input"}
    for k, v in base.adapt_conditions(batch).items():
        torch.testing.assert_close(v, base.adapt_conditions(changed)[k], rtol=0, atol=0)
    with torch.no_grad():
        actor.future_encoder.output.weight.fill_(1)
        actor.future_encoder.output.bias.fill_(2)
    invalid = ForecastContext(torch.full((1, 58), float("nan")), torch.zeros(1, dtype=torch.bool), 900, 1020)
    assert torch.equal(actor.future_residual(batch, [invalid]), torch.zeros(1, 60, 32))


def test_sample_restores_all_module_modes(actor_factory):
    base, batch = actor_factory(motion_frames=60, starts=(45,))
    actor = JointPhysicsActor(base).train()
    actor.stage1.music_embedder.eval()
    before = [m.training for m in actor.modules()]
    context = ForecastContext.invalid(batch["music_features"], 900)
    with torch.no_grad():
        output = actor.sample_differentiable(batch, context, torch.randn(1, 60, 30))
    assert [m.training for m in actor.modules()] == before
    assert output["qpos30"].shape == (1, 60, 30)
    mask = batch["known_qpos30_mask"]
    torch.testing.assert_close(output["qpos30"][mask], batch["known_qpos30"][mask], rtol=0, atol=0)


def test_random_static_construction_never_loads_genmo(monkeypatch):
    from gem.closedloop.joint_physics.config import ROOT, build_actor, read_config, seed_all
    import gem.closedloop.training as legacy
    def forbidden(*args, **kwargs):
        raise AssertionError("禁止旧GENMO权重初始化")
    for name in ("load_stage1_checkpoint", "warm_start_weights", "initialize_stage1_weights"):
        monkeypatch.setattr(legacy, name, forbidden)
    cfg = read_config(ROOT / "configs/closedloop/joint_physics_from_scratch.yaml")
    cfg["stage1"]["model"].update(latent_dim=32, num_layers=2, num_heads=4, history_hidden_dim=24)
    values = []
    for seed in (72, 72, 73):
        seed_all(seed)
        actor, _ = build_actor(cfg, "cpu")
        values.append(actor.stage1.denoiser.add_cond_linear.weight.detach().clone())
        assert actor.initialization == "static_config_random_no_genmo_checkpoint"
    assert torch.equal(values[0], values[1])
    assert not torch.equal(values[0], values[2])


def test_future_batch_preserves_independent_times_masks_and_gradients(actor_factory):
    base, batch = actor_factory(motion_frames=60, starts=(45, 48, 51))
    actor = JointPhysicsActor(base).eval()
    with torch.no_grad():
        actor.future_encoder.output.weight.normal_(0, .01)
        actor.future_encoder.output.bias.fill_(.1)
    individual = copy.deepcopy(actor.future_encoder)
    contexts = [ForecastContext(torch.randn(1, 58, requires_grad=True), torch.tensor([valid]),
                                tick, tick + advance)
                for valid, tick, advance in ((True, 900, 120), (False, 960, 60), (True, 1020, 240))]
    legacy_contexts = [ForecastContext(c.features.detach().clone().requires_grad_(True),
                                      c.valid, c.request_tick, c.forecast_tick) for c in contexts]
    calls = []
    hook = actor.future_encoder.input.register_forward_hook(lambda *args: calls.append(1))
    batched = actor.future_residual(batch, contexts)
    hook.remove()
    expected = torch.cat([individual(c, batch["future_times"][i:i+1])
                          for i, c in enumerate(legacy_contexts)])
    assert len(calls) == 1
    torch.testing.assert_close(batched, expected, rtol=2e-5, atol=1e-6)
    assert batched[1].count_nonzero() == 0
    weights = torch.randn_like(batched)
    (batched * weights).sum().backward()
    (expected * weights).sum().backward()
    for actual, old in zip(actor.future_encoder.parameters(), individual.parameters()):
        torch.testing.assert_close(actual.grad, old.grad, rtol=2e-5, atol=2e-5)
    for actual, old in zip(contexts, legacy_contexts):
        torch.testing.assert_close(actual.features.grad, old.features.grad, rtol=2e-5, atol=1e-6)


def test_bf16_neural_autocast_keeps_physical_outputs_and_parameters_fp32(actor_factory):
    base, batch = actor_factory(motion_frames=60, starts=(45,))
    actor = JointPhysicsActor(base).train()
    actor.precision = "bf16_mixed"
    context = ForecastContext(torch.randn(1, 58), torch.ones(1, dtype=torch.bool), 900, 1020)
    observed = []
    hook = actor.future_encoder.output.register_forward_hook(
        lambda module, inputs, output: observed.append(output.dtype))
    out = actor.denoise_supervised(batch, [context], torch.tensor([100]),
                                  torch.randn(1, 60, 30), torch.tensor([False]))
    hook.remove()
    assert observed == [torch.bfloat16]
    for key in ("pred_x_start", "static_conf_logits", "target_x_start", "xt", "noise"):
        assert out[key].dtype == torch.float32
        assert torch.isfinite(out[key]).all()
    known = batch["known_qpos30_mask"]
    torch.testing.assert_close(out["pred_x_start"][known], out["target_x_start"][known], rtol=0, atol=0)
    (out["pred_x_start"].square().mean() + out["static_conf_logits"].square().mean()).backward()
    assert actor.future_encoder.output.weight.grad.norm() > 0
    assert all(p.dtype == torch.float32 for p in actor.parameters())
    assert all(torch.isfinite(p.grad).all() for p in actor.parameters() if p.grad is not None)
