"""滚动边界、真实/想象音乐一致、终止语义与物理采集分段测试。

环境替身仅核验控制接口与reset次数，不模拟物理、不形成真实收益报告。给定同一
执行状态，真实与想象必须逐项一致；滚动边界不读取未来计划，过去名义缓存不可改。
"""

from dataclasses import replace

import pytest
import torch

from gem.closedloop.reference_adapter.collector import PhysicalCollector
from gem.closedloop.reference_adapter.dynamics import frozen_parameters
from gem.closedloop.reference_adapter.evaluation import beat_report, validate_comparison
from gem.closedloop.reference_adapter.features import critic_features


def test_real_imagined_single_step_reward_and_music_memory_identical(system):
    engine, z = system
    for _ in range(7):
        proposal = engine.step.propose(z)
        next_state = engine.dynamics.step(z.executed_state, proposal.action, z.xi)
        real = engine.step.advance(z, proposal, next_state, origin="real")
        imagined = engine.step.advance(z, proposal, next_state, origin="imagined")
        for name in real.weighted:
            torch.testing.assert_close(real.weighted[name], imagined.weighted[name], atol=0, rtol=0)
        assert real.end.music_state.last_paid_tick == imagined.end.music_state.last_paid_tick
        assert torch.equal(real.end.music_state.energy, imagined.end.music_state.energy)
        z = real.end


def test_rolling_boundary_bootstrap_and_unknown_plan_excluded(system):
    engine, z = system
    z = replace(z, mode="rolling", plan_boundary_tick=36)
    with engine.dynamics.adapter_phase(), frozen_parameters(engine.critic):
        trajectory = engine.rollout(z)
    assert len(trajectory.rows) == 3
    assert not trajectory.rows[-1].terminated.any()
    assert trajectory.rows[-1].truncated
    assert trajectory.boundary == "known_nominal_plan_boundary_bootstrap"
    assert trajectory.rows[-1].continuation.item() > 0
    for row in trajectory.rows:
        assert row.end.nominal_reference.version == z.nominal_reference.version
    with pytest.raises(ValueError):
        engine.step.propose(replace(z, known_at_tick=z.tick+12))


def test_music_end_no_bootstrap_or_future_completion(system):
    engine, z = system
    short_audio = replace(z.audio, features=z.audio.features[:, :4], valid=z.audio.valid[:, :4])
    z = replace(z, audio=short_audio)
    with engine.dynamics.adapter_phase(), frozen_parameters(engine.critic):
        trajectory = engine.rollout(z)
    assert trajectory.rows[-1].reason == "music_end"
    assert trajectory.rows[-1].continuation.item() == 0
    assert trajectory.component_objectives["terminal_value"].item() == 0
    assert trajectory.rows[-1].end.tick == 60
    assert trajectory.rows[-1].end.music_state.last_paid_tick == 36


def test_segments_never_reset_and_committed_nominal_prefix_guard(system):
    engine, z = system
    class InterfaceOnlyEnvironment:
        def __init__(self):
            self.resets = 0
            self.state = z.executed_state
        def reset(self, ids, initial):
            self.resets += 1
            self.state = initial
        def observe(self):
            return dict(state=self.state, xi=z.xi, previous_action=z.previous_action)
        def step_physics(self, action):
            # 仅接口替身：显式原样返回，不是物理执行。
            return self.state
    env = InterfaceOnlyEnvironment()
    collector = PhysicalCollector(env, engine.step)
    collector.reset(z.executed_state, z.nominal_reference, z.audio, episode_id="interface", song_id="interface", mode="rolling")
    first, second = collector.collect(2), collector.collect(2)
    assert env.resets == 1 and second[0].start.tick == first[-1].end.tick
    corrupted = z.nominal_reference.world_qpos.clone()
    corrupted[:, 0, 0] += .1
    with pytest.raises(ValueError):
        collector.install_nominal(replace(z.nominal_reference, version=1, world_qpos=corrupted), known_at_tick=48)
    collector.install_nominal(replace(z.nominal_reference, version=1), known_at_tick=48)
    assert torch.equal(collector.state.current_pose.q, second[-1].end.current_pose.q)


def test_bidirectional_metrics_and_budget_mismatch(system):
    _, z = system
    qd = torch.zeros(100, 21)
    report = beat_report(qd, torch.arange(100)*12, z.audio)
    assert report["motion_beat_count"] == 0 and report["legacy_alignment"] is None
    reports = {k: dict(task_contract=["same"], budget=dict(updates=1, real_steps=100), episodes=[]) for k in ("zero", "no_music", "music")}
    assert set(validate_comparison(reports)) == set(reports)
    reports["music"]["budget"]["real_steps"] += 1
    with pytest.raises(ValueError):
        validate_comparison(reports)


def test_critic_retains_absolute_nominal_root_for_original_reward(system):
    engine, z = system
    before = critic_features(z, engine.step.adapter, engine.step.bundle, engine.step.runtime.kin)
    world = z.nominal_reference.world_qpos.clone()
    world[..., 0] += .3
    shift = z.current_pose.p.new_tensor([.3, 0., 0.])
    shifted = replace(z, nominal_reference=replace(z.nominal_reference, world_qpos=world),
                      left_pose=replace(z.left_pose, p=z.left_pose.p+shift),
                      current_pose=replace(z.current_pose, p=z.current_pose.p+shift))
    after = critic_features(shifted, engine.step.adapter, engine.step.bundle, engine.step.runtime.kin)
    assert (after[0, -57]-before[0, -57]).item() == pytest.approx(.3)
    assert before.shape[-1] == 2835
