"""交替训练修复的真实SM权重数值回归，不启动物理仿真或训练任务。

使用真实冻结Actor/WM检查：被拒绝的投影不能进入模拟物理或Replay；单环境重置
必须产生新episode且其余环境不受影响；想象microbatch拒绝不计Adam，剩余样本仍
有真实BPTT梯度。探索与任务采样使用独立随机流，噪声后的动作同时进入执行和历史。
诊断覆盖raw/投影/首帧/Goal/Actor变化及梯度穿透，所有检查只证明代码语义，不能
替代同初态真实物理配对效果。全部夹具来自会话CPU资产，不写正式checkpoint。
"""

import torch
import pytest
from supermimic.networks import Critic
from test_alternating import setup
from gem.closedloop.reference_adapter.alternating_learning import policy_update, PolicyProjectionRejected
from gem.closedloop.reference_adapter.vector_collector import VectorCollector
from gem.closedloop.reference_adapter.vector_replay import VectorReplay


def numerical_failure(mask):
    error = RuntimeError('fixture_projection_numerical_failure')
    error.bad_mask = mask
    error.certificate = dict(feasible=~mask)
    return error


def test_collector_recovers_only_bad_environment_before_sending_action(system, monkeypatch):
    engine, control, z = setup(system)
    class Env:
        N = 2
        calls = 0
        def step_physics(self, action):
            self.calls += 1
            self.applied = action.clone()
            return z['state'].clone()
    env = Env()
    collector = VectorCollector(env, control)
    collector.state = {k: v.clone() for k, v in z.items()}
    collector.projection_recovery_attempts = 1
    original = control.propose
    calls, resets = [], []
    def propose(state, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise numerical_failure(torch.tensor([True, False]))
        return original(state, **kwargs)
    def reset(ids, **kwargs):
        resets.append(ids.clone())
        collector.state['episode'][ids] += 100
    monkeypatch.setattr(control, 'propose', propose)
    monkeypatch.setattr(collector, 'reset', reset)
    replay = VectorReplay(100, 2, 'cpu', compute_mc=False)
    row = collector.step(replay, auto_reset=True)
    assert env.calls == 1 and len(resets) == 1
    torch.testing.assert_close(resets[0], torch.tensor([0]))
    assert row['start']['episode'][0] == z['episode'][0]+100
    assert row['start']['episode'][1] == z['episode'][1]
    torch.testing.assert_close(collector.last_constraint_state['episode'], z['episode'])
    torch.testing.assert_close(replay.data['z_episode'][0], row['start']['episode'])
    assert collector.projection_recoveries == 1
    assert collector.projection_recovery_events[0]['action_sent'] is False


def test_collector_never_resets_evaluation_to_hide_constraint_failure(system, monkeypatch):
    _, control, z = setup(system)
    class Env:
        N = 2
        def step_physics(self, action):
            pytest.fail('不可行参考不得发送物理动作')
    collector = VectorCollector(Env(), control)
    collector.state = z
    collector.projection_recovery_attempts = 2
    monkeypatch.setattr(control, 'propose', lambda *a, **kw: (_ for _ in ()).throw(
        numerical_failure(torch.tensor([False, True]))))
    with pytest.raises(RuntimeError, match='fixture_projection'):
        collector.step(split='validation')
    assert collector.real_transitions == 0 and collector.projection_recoveries == 0


def test_mixture_exploration_keeps_task_rng_and_executed_action_identity(system):
    _, control, z = setup(system)
    class Env:
        N = 2
        def step_physics(self, action):
            self.applied = action.clone()
            return z['state'].clone()
    env = Env()
    collector = VectorCollector(env, control)
    collector.state = z
    before = collector.generator.get_state().clone()
    row = collector.step(action_noise=.02, exploration_fraction=1., exploration_std=.1, auto_reset=False)
    torch.testing.assert_close(collector.generator.get_state(), before)
    torch.testing.assert_close(row['proposal']['exploration_std'], torch.full((2,), .1))
    torch.testing.assert_close(env.applied, row['end']['action'])
    assert not torch.equal(env.applied, row['proposal']['policy_action'])


def test_rejected_imagination_microbatch_does_not_discard_valid_update(system, monkeypatch):
    engine, control, z = setup(system)
    critic = Critic(2835)
    optimizer = torch.optim.Adam(control.adapter.parameters(), lr=3e-5)
    original = control.propose
    calls = []
    def propose(state, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise numerical_failure(torch.tensor([True]))
        return original(state, **kwargs)
    monkeypatch.setattr(control, 'propose', propose)
    report, _ = policy_update(control, engine.dynamics, critic, optimizer, z, 2,
                             terminal_value=False, microbatch=1)
    assert report['accepted_samples'] == 1 and report['rejected_samples'] == 1
    assert report['optimizer_step_completed'] and report['parameter_delta'] > 0
    assert all(int(v['step']) == 1 for v in optimizer.state.values())


def test_all_projection_rejections_preserve_parameters_and_optimizer(system, monkeypatch):
    engine, control, z = setup(system)
    critic = Critic(2835)
    optimizer = torch.optim.Adam(control.adapter.parameters(), lr=3e-5)
    previous = [p.detach().clone() for p in control.adapter.parameters()]
    def propose(state, **kwargs):
        raise numerical_failure(torch.ones(len(state['state']), dtype=torch.bool))
    monkeypatch.setattr(control, 'propose', propose)
    with pytest.raises(PolicyProjectionRejected) as info:
        policy_update(control, engine.dynamics, critic, optimizer, z, 2,
                      terminal_value=False, microbatch=1)
    assert info.value.report['accepted_samples'] == 0
    assert info.value.report['rejected_samples'] == 2
    assert not optimizer.state
    for before, after in zip(previous, control.adapter.parameters()):
        torch.testing.assert_close(before, after, atol=0, rtol=0)
        assert after.grad is None


def test_update_diagnostics_report_projection_and_first_frame_gradients(system):
    engine, control, z = setup(system)
    optimizer = torch.optim.Adam(control.adapter.parameters(), lr=3e-5)
    report, _ = policy_update(control, engine.dynamics, Critic(2835), optimizer, z, 2,
                             terminal_value=False, microbatch=1, diagnostics=True)
    gradient = report['projection_gradient']
    assert gradient['raw_residual_norm'] > 0 and gradient['residual_norm'] > 0
    assert gradient['raw_over_projected_norm'] == pytest.approx(1.)
    assert gradient['physical_gain_proven'] is False
    changes = report['update_effect']['change_distributions']
    for key in ('raw_residual', 'projected_residual', 'next_committed_first_residual',
                'next_committed_joint_q', 'next_committed_joint_qd', 'goal', 'actor_action'):
        assert changes[key]['maximum'] > 0


def test_root_orientation_ablation_preserves_270_shape_and_other_channels(system):
    _, control, z = setup(system)
    control.disable_root_orientation = True
    with torch.no_grad():
        control.adapter.net[-1].bias.fill_(.01)
    proposal = control.propose(z)
    assert proposal['raw_residual'].shape == (2, 10, 27)
    assert proposal['raw_residual'][..., 3:6].count_nonzero() == 0
    assert proposal['raw_residual'][..., 6:].count_nonzero() > 0


def test_old_checkpoint_extra_fields_keep_counters_and_add_only_metadata():
    from gem.closedloop.reference_adapter.alternating_entry import normalize_extra_state
    old = dict(step_acc=.5, best=dict(adapter_updates=20), evaluations=[20],
               completed_checkpoints=['old.pt'], precision_scales={'joint_position': .2})
    restored = normalize_extra_state(old)
    for key, value in old.items():
        assert restored[key] == value
    assert restored['zero_evaluations'] == {}
    assert restored['consecutive_projection_rejections'] == 0
    assert restored['best_short'] is None
    assert 'zero_evaluations' not in old


def test_complete_checkpoint_preserves_new_rng_replay_and_disabled_critic(system, tmp_path, monkeypatch):
    from gem.closedloop.reference_adapter.vector_checkpoint import save_vector, load_vector
    from gem.closedloop.reference_adapter.vector_learning import TangentReplay
    from gem.closedloop.reference_adapter.alternating_entry import normalize_extra_state
    from gem.closedloop.reference_adapter.alternating_learning import enable_wm_normalization
    engine, control, z = setup(system)
    enable_wm_normalization(engine.dynamics)
    critic = Critic(2835)
    critic.requires_grad_(False)
    modules = dict(adapter=control.adapter, wm=engine.dynamics.model.net, critic=critic)
    optimizers = {key: torch.optim.Adam(module.parameters(), lr=3e-5) for key, module in modules.items()}
    for key in ('adapter', 'wm'):
        sum(p.square().mean() for p in modules[key].parameters()).backward()
        optimizers[key].step(); optimizers[key].zero_grad(set_to_none=True)
    class Env:
        N = 2
    collector = VectorCollector(Env(), control)
    collector.state = {k: v.clone() for k, v in z.items()}
    monkeypatch.setattr(collector, 'reset', lambda **kwargs: None)
    replay = VectorReplay(80, 2, 'cpu', compute_mc=False)
    replay.add(z, z['action'], z['state'], torch.ones(2), torch.ones(2),
               torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool), 1)
    torch.randn((7,), generator=collector.action_generator)
    extra = normalize_extra_state(dict(action_noise_rng=collector.action_generator.get_state(),
                                      evaluations=[0], consecutive_projection_rejections=2))
    expected_noise = torch.randn((5,), generator=collector.action_generator)
    config = dict(music={}, rewards=control.weights, reference_steps=10, control_hz=50)
    expected = {key: {name: value.clone() for name, value in module.state_dict().items()}
                for key, module in modules.items()}
    path = tmp_path/'complete_runtime_recovery_fixture.pt'
    save_vector(path, engine, critic, optimizers, replay, TangentReplay(1), collector,
                dict(wm_updates=1, accepted_updates=1), config, extra_state=extra)
    with torch.no_grad():
        for module in modules.values():
            for parameter in module.parameters():
                parameter.zero_()
    restored = load_vector(path, engine, critic, optimizers, replay, TangentReplay(1), collector, config)
    saved = torch.load(path, map_location='cpu', weights_only=False)
    restored_extra = normalize_extra_state(saved['extra_state'])
    collector.action_generator.set_state(restored_extra['action_noise_rng'])
    torch.testing.assert_close(torch.randn((5,), generator=collector.action_generator), expected_noise)
    assert restored == dict(wm_updates=1, accepted_updates=1)
    assert restored_extra['evaluations'] == [0] and restored_extra['consecutive_projection_rejections'] == 2
    assert not replay.compute_mc and len(replay) == 2
    torch.testing.assert_close(saved['collector_state']['music'], z['music'])
    torch.testing.assert_close(saved['collector_state']['plan'], z['plan'])
    for key, module in modules.items():
        for name, value in module.state_dict().items():
            torch.testing.assert_close(value, expected[key][name], atol=0, rtol=0)
        assert bool(optimizers[key].state) == (key != 'critic')
