"""真实权重梯度、模型隔离、lambda-return和完整恢复的数值验证。

此文件不运行物理；回放数据夹具只验证存储/优化器接口，不能视为真实监督验证。
执行项和music-only项独立检查Adapter→Actor→独立WM导数，故意detach负测必须断图。
checkpoint在pytest临时目录保存，不覆盖任何正式模型；所有冻结资产逐张量比较。
"""

from dataclasses import replace

import pytest
import torch

from gem.closedloop.reference_adapter.critic import lambda_returns
from gem.closedloop.reference_adapter.dynamics import frozen_parameters
from gem.closedloop.reference_adapter.contracts import map_tensors


def test_lambda_returns_terminal_vs_truncation():
    reward = torch.tensor([[1.], [2.]])
    values = torch.tensor([[3.], [4.], [5.]], requires_grad=True)
    terminal = lambda_returns(reward, torch.tensor([[1.], [0.]]), values, .99, .95)
    truncated = lambda_returns(reward, torch.ones_like(reward), values, .99, .95)
    assert terminal[-1].item() == 2
    assert truncated[-1].item() == pytest.approx(6.95)
    assert not terminal.requires_grad


@pytest.mark.parametrize("horizon", [8, 16])
def test_real_weights_bptt_isolated_input_gradients(system, horizon):
    engine, z = system
    engine.bptt_steps = horizon
    bundle = engine.step.bundle
    before = bundle.frozen_snapshot()
    with engine.dynamics.adapter_phase(), frozen_parameters(engine.critic):
        trajectory = engine.rollout(z)
        print("actual_weight_rollout", horizon, len(trajectory.rows), trajectory.boundary)
        execution = trajectory.rows[-1].end.executed_state[:, 34:].square().mean()
        music = sum(r.raw["music"].sum() for r in trajectory.rows)
        bias = engine.step.adapter.net[-1].bias
        execution_grad, = torch.autograd.grad(execution, bias, retain_graph=True)
        music_grad, = torch.autograd.grad(music, bias, retain_graph=True)
        assert torch.isfinite(execution_grad).all() and execution_grad.norm() > 0
        assert torch.isfinite(music_grad).all() and music_grad.norm() > 0
        trajectory.objective.sum().backward()
        assert all(p.grad is None for p in bundle.parameters())
        assert all(p.grad is None for p in engine.dynamics.parameters())
        assert all(p.grad is None for p in engine.critic.parameters())
        assert len(trajectory.rows) == horizon
    bundle.assert_unchanged(before)


@pytest.mark.parametrize("component", ["music", "execution"])
def test_adapter_music_only_finite_difference_and_detach_negative(system, component):
    engine, z = system
    engine.bptt_steps = 6
    bias = engine.step.adapter.net[-1].bias
    with engine.dynamics.adapter_phase(), frozen_parameters(engine.critic):
        def objective():
            trajectory = engine.rollout(z)
            return (sum(r.raw["music"].sum() for r in trajectory.rows) if component == "music"
                    else trajectory.rows[-1].end.executed_state[:, 34:].square().mean())
        value = objective()
        grad, = torch.autograd.grad(value, bias)
        direction = grad / grad.norm()
        original = bias.detach().clone()
        eps = 1e-3
        with torch.no_grad():
            bias.copy_(original+eps*direction)
            plus = objective()
            bias.copy_(original-eps*direction)
            minus = objective()
            bias.copy_(original)
        finite = (plus-minus)/(2*eps)
        print("adapter_component_fd", component, float(grad.norm()), float(finite))
        torch.testing.assert_close(finite, grad.norm(), atol=.01, rtol=.15)
        step = engine.dynamics.step
        engine.dynamics.step = lambda *args: step(*args).detach()
        try:
            assert not objective().requires_grad
        finally:
            engine.dynamics.step = step


def tensor_replay(engine, z, count=10):
    """纯接口夹具，不是物理标签；测试报告必须保留此证据边界。"""
    from gem.closedloop.reference_adapter.replay import RealReplay
    replay = RealReplay()
    for split in ("train", "validation"):
        state = replace(z, episode_id=split, song_id=split, split=split)
        with torch.no_grad():
            for i in range(count):
                proposal = engine.step.propose(state, zero=True)
                row = engine.step.advance(state, proposal, engine.dynamics.step(state.executed_state, proposal.action, state.xi), origin="real")
                if i == count-1:
                    row = replace(row, terminated=torch.ones_like(row.terminated), reason="fixture_end", continuation=torch.zeros_like(row.continuation))
                row = map_tensors(row, lambda x: x.detach())
                replay.add(row)
                state = row.end
    return replay


def test_replay_reset_guards_and_independent_wm_update(system):
    engine, z = system
    replay = tensor_replay(engine, z)
    w = replay.sample_windows(2, 8)
    assert w["state"].shape == (2, 8, 55)
    with pytest.raises(ValueError):
        engine.dynamics.supervised_loss(w | {"origin": "imagined"})
    with pytest.raises(ValueError):
        replay.add(next(iter(replay.episodes.values()))[0])
    before = engine.step.bundle.frozen_snapshot()
    norm_before = {k: v.clone() for k, v in engine.dynamics.model.state_dict().items() if "norm." in k}
    optimizer = torch.optim.Adam(engine.dynamics.model.net.parameters(), lr=3e-5)
    # 改真实标签夹具以检测独立WM优化器确实生效，仍只属于数值接口测试。
    w["next_state"] = w["next_state"].clone()
    w["next_state"][..., 34:] += .01
    old = engine.dynamics.step(z.executed_state, torch.zeros(1, 21), z.xi).detach()
    engine.dynamics.fit(w, optimizer)
    new = engine.dynamics.step(z.executed_state, torch.zeros(1, 21), z.xi).detach()
    assert not torch.equal(old, new)
    for k, v in norm_before.items():
        assert torch.equal(v, engine.dynamics.model.state_dict()[k])
    engine.step.bundle.assert_unchanged(before)
    with engine.dynamics.adapter_phase():
        with pytest.raises(RuntimeError):
            engine.dynamics.fit(w, optimizer)


def test_full_checkpoint_optimizers_rng_music_and_resume(system, tmp_path):
    from gem.closedloop.reference_adapter.trainer import AdapterTrainer
    from gem.closedloop.reference_adapter.checkpoint import save_checkpoint, load_checkpoint
    engine, z = system
    replay = tensor_replay(engine, z)
    engine.bptt_steps = 6
    trainer = AdapterTrainer(engine, replay, dict(batch_size=1, wm_fit_window=8))
    trainer.warmup_real(epochs=1)
    report = trainer.update()
    assert report["update"] == 1 and report["component_parameter_gradient_norm"]["sm"] > 0
    ck = tmp_path/"full.pt"
    save_checkpoint(ck, trainer, behavior={"reference_steps": 10}, cursor={"episode": 2})
    expected_random = torch.rand(4)
    expected_adapter = {k: v.clone() for k, v in trainer.adapter.state_dict().items()}
    with pytest.raises(FileExistsError):
        save_checkpoint(ck, trainer)
    with torch.no_grad():
        trainer.adapter.net[-1].bias.add_(1)
    restored = load_checkpoint(ck, trainer, behavior={"reference_steps": 10})
    assert torch.equal(torch.rand(4), expected_random)
    assert restored["physics_resume"] == "new_episode_required_solver_hidden_state_not_saved"
    for k, v in expected_adapter.items():
        assert torch.equal(v, trainer.adapter.state_dict()[k])
    assert trainer.adapter_optimizer.state and trainer.critic_optimizer.state and trainer.wm_optimizer.state
    assert trainer.dynamics.model_version == 1 and trainer.update_index == 1
    memory = list(trainer.replay.episodes.values())[0][-1].end.music_state
    assert memory.last_paid_tick is not None and memory.valid.sum() > 0
    # 相同优化器、RNG和回放恢复后，下一次完整更新逐参数一致。
    trainer.update()
    next_adapter = {k: v.clone() for k, v in trainer.adapter.state_dict().items()}
    next_critic = {k: v.clone() for k, v in trainer.critic.state_dict().items()}
    next_wm = {k: v.clone() for k, v in trainer.dynamics.state_dict().items()}
    load_checkpoint(ck, trainer, behavior={"reference_steps": 10})
    trainer.update()
    for module, expected in ((trainer.adapter, next_adapter), (trainer.critic, next_critic), (trainer.dynamics, next_wm)):
        for k, v in expected.items():
            torch.testing.assert_close(module.state_dict()[k], v, atol=0, rtol=0)


def test_music_only_action_finite_difference(system):
    engine, z = system
    engine.bptt_steps = 6
    original_step = engine.dynamics.step
    offset = torch.zeros(1, 21, requires_grad=True)
    calls = 0
    def perturbed_step(state, action, xi):
        nonlocal calls
        calls += 1
        return original_step(state, action + (offset if calls == 1 else 0), xi)
    engine.dynamics.step = perturbed_step
    def objective():
        nonlocal calls
        calls = 0
        return sum(r.raw["music"].sum() for r in engine.rollout(z).rows)
    try:
        with engine.dynamics.adapter_phase(), frozen_parameters(engine.critic):
            grad, = torch.autograd.grad(objective(), offset)
            direction = grad/grad.norm()
            with torch.no_grad():
                offset.copy_(direction*1e-3)
                plus = objective()
                offset.copy_(-direction*1e-3)
                minus = objective()
            derivative = (plus-minus)/2e-3
            print("music_action_fd", float(grad.norm()), float(derivative))
            assert grad.norm() > 0
            torch.testing.assert_close(derivative, grad.norm(), atol=.01, rtol=.15)
    finally:
        engine.dynamics.step = original_step
