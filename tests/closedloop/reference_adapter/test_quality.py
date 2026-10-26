"""正式训练门槛与有损回放的针对性回归。

使用明确标注的数值转移夹具检查协议，不声称物理可靠。覆盖Critic固定epoch不能
解锁、坏预测被拒绝、真实reset前缀丢失不再当MC、尾支持不足不外推、音乐尾部和
失败优先级，并比较microbatch与逐轨迹反传。临时模型只写pytest临时目录。
"""

from dataclasses import replace

import pytest
import torch

from gem.closedloop.reference_adapter.quality import critic_gate, prediction_gate
from gem.closedloop.reference_adapter.critic import calibration
from gem.closedloop.reference_adapter.replay import RealReplay
from gem.closedloop.reference_adapter.trainer import AdapterTrainer
from gem.closedloop.reference_adapter.nominal import check_support, required_support_tick
from gem.closedloop.reference_adapter.dynamics import frozen_parameters
from test_learning import tensor_replay


CRITIC = dict(min_train_samples=128, min_validation_samples=64, min_explained_variance=.1,
              min_mse_improvement=.2, min_near_failure_samples=16, max_near_failure_mse_ratio=.8)


def test_critic_gate_rejects_constant_bad_and_missing_failure():
    target = torch.arange(128).float()/128
    near = torch.arange(128) < 16
    good = calibration(target, target, near, constant_value=.5)
    assert critic_gate(good, CRITIC, train_samples=128)["passed"]
    bad = calibration(torch.ones_like(target)*.5, target, near, constant_value=.5)
    assert not critic_gate(bad, CRITIC, train_samples=128)["passed"]
    assert not critic_gate(good | {"near_failure_samples": 0}, CRITIC)["passed"]
    with pytest.raises(ValueError):
        critic_gate(good, None)


def test_warmup_epochs_do_not_unlock_without_quality(system):
    engine, z = system
    trainer = AdapterTrainer(engine, tensor_replay(engine, z, 16), {"critic_quality": CRITIC})
    report = trainer.warmup_real(epochs=1)
    assert not trainer.critic_ready
    assert not report["quality_gate"]["passed"]
    with pytest.raises(RuntimeError, match="Critic"):
        trainer.update()


def test_wm_gate_requires_all_three_horizons():
    limits = dict(root_position_m=.05, root_rotation_rad=.1, joint_position_rad=.1, joint_velocity_rad_s=2.)
    values = {str(h): {k: 0. for k in limits} for h in (1, 8, 16)}
    assert prediction_gate(values, limits)["passed"]
    values["16"]["root_position_m"] = .06
    assert not prediction_gate(values, limits)["passed"]
    values["16"]["root_position_m"] = float("nan")
    assert not prediction_gate(values, limits)["passed"]


def test_replay_truncated_head_and_reference_end_cannot_be_mc(system):
    engine, z = system
    source = tensor_replay(engine, z, 16)
    rows = source.complete_episodes("train")[0]
    replay = RealReplay(capacity=8)
    for row in rows:
        replay.add(row)
    assert not replay.complete_episodes("train")
    assert replay.episode_metadata["train"]["head_truncated"]
    assert replay.sample_windows(1, 8)["state"].shape == (1, 8, 55)
    restored = RealReplay()
    restored.load_state_dict(replay.state_dict())
    assert not restored.complete_episodes("train")
    reference = RealReplay()
    for row in rows[:-1]+[replace(rows[-1], reason="reference_end")]:
        reference.add(row)
    assert not reference.complete_episodes("train")
    with pytest.raises(ValueError, match="真实MC"):
        AdapterTrainer(engine, reference, {}).real_targets(list(reference.episodes.values()))


def test_tail_support_and_noncommensurate_music_end(system, monkeypatch):
    engine, z = system
    assert required_support_tick(80) == 204
    check_support(z.nominal_reference, 80)
    with pytest.raises(ValueError, match="尾部不足"):
        check_support(z.nominal_reference, z.nominal_reference.upper)
    monkeypatch.setattr(engine.step.runtime, "termination", lambda state, ref, soft=False: state.new_zeros(len(state)) if soft else torch.zeros(len(state), dtype=torch.bool))
    z = replace(z, audio=replace(z.audio, features=z.audio.features[:, :5], valid=z.audio.valid[:, :5]))
    engine.bptt_steps = 16
    with engine.dynamics.adapter_phase(), frozen_parameters(engine.critic):
        trajectory = engine.rollout(z)
    assert trajectory.rows[-1].end.tick == 72 and trajectory.rows[-1].reason == "music_end"
    assert trajectory.component_objectives["terminal_value"] == 0
    terminal = trajectory.rows[-1].start
    monkeypatch.setattr(engine.step.runtime, "termination", lambda state, ref, soft=False: state.new_ones(len(state)) if soft else torch.ones(len(state), dtype=torch.bool))
    proposal = engine.step.propose(terminal)
    assert engine.step.advance(terminal, proposal, terminal.executed_state, origin="real").reason == "physical_failure"


def test_microbatch_objective_and_all_parameter_gradient_parity(system):
    engine, z = system
    engine.bptt_steps = 8
    params = tuple(engine.step.adapter.parameters())
    def run(micro):
        engine.step.adapter.zero_grad(set_to_none=True)
        values, pending = [], []
        with engine.dynamics.adapter_phase(), frozen_parameters(engine.critic):
            for i in range(2):
                value = engine.rollout(z).objective.mean()/2
                values.append(value.detach())
                pending.append(value)
                if len(pending) == micro:
                    torch.stack(pending).sum().backward()
                    pending.clear()
        return torch.stack(values), [p.grad.clone() for p in params]
    a, ga = run(1)
    b, gb = run(2)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    for x, y in zip(ga, gb):
        torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-5)


def test_zero_replay_reprice_preserves_physics_and_matches_music(system):
    from gem.closedloop.reference_adapter.baseline import reprice_zero_replay, combined_s0
    from gem.closedloop.reference_adapter.music_reward import MusicReward
    engine, z = system
    replay = tensor_replay(engine, z, 16)
    old = list(replay.episodes["train"])
    reprice_zero_replay(replay, engine.step.rewards.music, .2)
    for a, b in zip(old, replay.episodes["train"]):
        torch.testing.assert_close(a.reward, b.reward, atol=1e-6, rtol=1e-6)
        assert torch.equal(a.action, b.action) and torch.equal(a.end.executed_state, b.end.executed_state)
        assert a.end.music_state.last_paid_tick == b.end.music_state.last_paid_tick
    reprice_zero_replay(replay, MusicReward(torch.ones(21)*2), .2)
    assert not torch.equal(old[-1].end.music_state.energy, replay.episodes["train"][-1].end.music_state.energy)
    assert torch.equal(old[-1].weighted["sm"], replay.episodes["train"][-1].weighted["sm"])
    stats = [dict(source="zero_adapter_real_training_episodes", joint_names=range(21), samples=2,
                  sum_square_qd=torch.ones(21)*8)]
    torch.testing.assert_close(combined_s0(stats), torch.ones(21)*2)
