"""真实SM权重下向量优化路径的语义回归，不将合成参考测试冒充物理验收。

对照旧控制路径核验零/非零十帧残差的Goal570、Actor、六项SM奖励与输入梯度；
独立音乐时钟逐步核验现有事件公式。紧凑Replay测试reset边界、随机起点MC标识、
环形淘汰和恢复。真实64至2048环境压力及动作方向另由GPU审计入口执行。
"""

import torch
import pytest

from gem.closedloop.reference_adapter.vector_bank import VectorBank
from gem.closedloop.reference_adapter.vector_control import VectorControl, select_state
from gem.closedloop.reference_adapter.vector_replay import VectorReplay
from gem.closedloop.reference_adapter.vector_learning import GroupNormalizedCritic, tangent_loss
from gem.closedloop.reference_adapter.reference import reconstruct
from gem.closedloop.reference_adapter.music_reward import MusicState


def vector(system, count=1):
    engine, old = system
    task = dict(id='fixture', song_id='fixture_song', split='train', seed=42)
    bank = VectorBank.from_entries([(task, old.executed_state, old.nominal_reference, old.audio)],
                                  engine.step.runtime.kin, engine.step.rewards.music)
    control = VectorControl(engine, bank)
    observed = dict(state=old.executed_state.expand(count, -1).clone(), xi=old.xi.expand(count, -1).clone())
    z = control.initial(observed, torch.zeros(count, dtype=torch.long), torch.zeros(count, dtype=torch.long), torch.arange(count))
    return engine, old, control, z


@pytest.mark.parametrize('nonzero', [False, True])
def test_vector_goal_actor_reward_gradient(system, nonzero):
    # fixture音频需留下真实halo，不能以裁剪索引伪造缺失名义点。
    engine, old = system
    from gem.closedloop.reference_adapter.music_reward import AudioTrack
    from dataclasses import replace
    old = replace(old, audio=AudioTrack.from_features(old.audio.features[:, :-10]))
    engine, old, control, z = vector((engine, old))
    if nonzero:
        with torch.no_grad():
            engine.step.adapter.net[-1].bias.normal_(std=.01)
    for t in range(25):
        old_proposal = engine.step.propose(old)
        new = control.propose(z)
        torch.testing.assert_close(new['goal'], old_proposal.corrected_reference.goal, atol=3e-6, rtol=2e-5)
        torch.testing.assert_close(new['action'], old_proposal.action, atol=5e-6, rtol=2e-5)
        nxt = engine.dynamics.step(old.executed_state, old_proposal.action, old.xi).detach()
        legacy = engine.step.advance(old, old_proposal, nxt, origin='real')
        end, reward, raw, terms, cont, done, failed = control.advance(z, new, nxt)
        torch.testing.assert_close(reward, legacy.reward, atol=1e-6, rtol=2e-5)
        for key in ('body_pos', 'body_ori', 'body_linvel', 'body_angvel', 'root_pos', 'root_ori',
                    'music', 'activity', 'magnitude', 'smooth', 'overlap', 'reference_limit'):
            torch.testing.assert_close(raw[key], legacy.raw[key], atol=1e-5, rtol=2e-5)
        old = legacy.end.detached()
        z = {k: v.detach() for k, v in end.items()}
    residual = torch.randn(1, 10, 27)*.001
    a, b = residual.clone().requires_grad_(), residual.clone().requires_grad_()
    original_goal = reconstruct(old, a, engine.step.runtime.kin).goal
    fast_goal = control.reference(z, b)[2]
    target = torch.randn_like(original_goal)
    ga, = torch.autograd.grad((original_goal*target).sum(), a)
    gb, = torch.autograd.grad((fast_goal*target).sum(), b)
    torch.testing.assert_close(ga, gb, atol=2e-5, rtol=1e-4)


def test_vector_music_independent_clocks(system):
    from dataclasses import replace
    from gem.closedloop.reference_adapter.music_reward import AudioTrack
    engine, old = system
    old = replace(old, audio=AudioTrack.from_features(old.audio.features[:, :-10]))
    engine, old, control, z = vector((engine, old), count=3)
    bank, music = control.bank, control.music
    memories = [MusicState.empty(old.executed_state) for _ in range(3)]
    memory = torch.zeros(3, 20, 4)
    starts = torch.tensor([0, 5, 11])
    torch.manual_seed(5)
    for i in range(45):
        qd, nominal = torch.randn(3, 21), torch.randn(3, 21)
        memory = bank.memory_push(memory, qd, nominal, music)
        reward, activity, _ = bank.music_reward(memory, torch.zeros(3, dtype=torch.long), starts+i, music)
        for j in range(3):
            memories[j] = music.push(memories[j], qd[j:j+1], nominal[j:j+1], int(starts[j]+i)*12)
            memories[j], expected, expected_activity, _ = music.settle(memories[j], old.audio)
            torch.testing.assert_close(reward[j:j+1], expected, atol=1e-6, rtol=2e-5)
            torch.testing.assert_close(activity[j:j+1], expected_activity)


def test_compact_replay_reset_mc_and_restore():
    replay = VectorReplay(80, 2, 'cpu')
    for t in range(25):
        frame = torch.tensor([5+t, 50+t])
        z = dict(state=torch.ones(2, 55)*t, xi=torch.ones(2, 25), frame=frame,
                 start_frame=torch.tensor([5, 50]), episode=torch.tensor([10, 11]), song=torch.tensor([0, 1]))
        done = torch.tensor([t == 24, t == 24])
        replay.add(z, torch.zeros(2, 21), z['state']+1, torch.ones(2), (~done).float(), done,
                   torch.tensor([False, t == 24]), 0)
    assert replay.completed_segments == 2 and replay.completed_full_songs == 0
    windows = replay.windows(64, 8)
    torch.testing.assert_close(windows['next_state'][:, :-1], windows['state'][:, 1:])
    samples = replay.mc_samples(100)
    assert len(samples[1]) == 50
    expected = sum(.99**j for j in range(25))
    torch.testing.assert_close(replay.data['mc'][0], torch.full((2,), expected))
    restored = VectorReplay(80, 2, 'cpu')
    restored.load_state_dict(replay.state_dict())
    torch.testing.assert_close(restored.data['mc'][:25], replay.data['mc'][:25])
    # 环形头被淘汰后不能把剩余尾巴变成完整MC。
    truncated = VectorReplay(40, 2, 'cpu')
    for t in range(25):
        z = dict(state=torch.zeros(2, 55), xi=torch.zeros(2, 25), frame=torch.full((2,), t),
                 start_frame=torch.zeros(2, dtype=torch.long), episode=torch.zeros(2, dtype=torch.long), song=torch.zeros(2, dtype=torch.long))
        done = torch.full((2,), t == 24)
        truncated.add(z, torch.zeros(2, 21), z['state'], torch.ones(2), (~done).float(), done, done, 0)
    assert truncated.mc_samples(100) is None
    assert truncated.head_truncated_terminals == 2


def test_critic_training_only_group_normalization(system):
    engine, _ = system
    critic = GroupNormalizedCritic(2835, engine.step.adapter.input_spec)
    x, y = torch.randn(128, 2835), torch.randn(128)*2+6
    with pytest.raises(ValueError):
        critic.fit_normalization(x, y, split='validation')
    critic.fit_normalization(x, y, split='train')
    before = critic.mean.clone()
    critic.fit_normalization(x+100, y, split='train')
    torch.testing.assert_close(before, critic.mean)
    x.requires_grad_()
    grad, = torch.autograd.grad(critic(x).sum(), x)
    assert torch.isfinite(grad).all() and grad.norm() > 0


def test_batched_imagination_and_dense_update(system):
    from dataclasses import replace
    from gem.closedloop.reference_adapter.music_reward import AudioTrack
    from gem.closedloop.reference_adapter.vector_learning import rollout, adapter_update
    from gem.closedloop.reference_adapter.dynamics import frozen_parameters
    engine, old = system
    old = replace(old, audio=AudioTrack.from_features(old.audio.features[:, :-10]))
    engine, old, control, z = vector((engine, old), count=4)
    critic = GroupNormalizedCritic(2835, engine.step.adapter.input_spec)
    snapshot = engine.step.bundle.frozen_snapshot()
    with engine.dynamics.adapter_phase(), frozen_parameters(critic):
        batch = rollout(control, engine.dynamics, critic, z, 8, terminal_value=False)
        single = rollout(control, engine.dynamics, critic, select_state(z, slice(0, 1)), 8, terminal_value=False)
        torch.testing.assert_close(batch['objective'], single['objective'].expand(4), atol=1e-5, rtol=1e-4)
        grad, = torch.autograd.grad(batch['objective'].mean(), engine.step.adapter.net[-1].bias)
        assert grad.norm() > 0 and torch.isfinite(grad).all()
    optimizer = torch.optim.Adam(engine.step.adapter.parameters(), lr=1e-4)
    report, labels = adapter_update(control, engine.dynamics, critic, optimizer, z, 1, terminal_value=False, microbatch=2)
    assert labels is None and report['phase'] == 'dense_tracking_no_terminal_value'
    assert report['parameter_delta'] > 0 and report['action_rms_delta'] > 0 and report['goal_rms_delta'] > 0
    engine.step.bundle.assert_unchanged(snapshot)


def test_vector_complete_checkpoint_restores_optimizers(system, tmp_path):
    from dataclasses import replace
    from types import SimpleNamespace
    from gem.closedloop.reference_adapter.music_reward import AudioTrack
    from gem.closedloop.reference_adapter.vector_learning import TangentReplay
    from gem.closedloop.reference_adapter.vector_checkpoint import save_vector, load_vector
    engine, old = system
    old = replace(old, audio=AudioTrack.from_features(old.audio.features[:, :-10]))
    engine, old, control, z = vector((engine, old), count=2)
    critic = GroupNormalizedCritic(2835, engine.step.adapter.input_spec)
    modules = dict(adapter=engine.step.adapter, critic=critic, wm=engine.dynamics.model.net)
    optimizers = {k:torch.optim.Adam(m.parameters(), lr=1e-4) for k,m in modules.items()}
    # 这里只验证三个优化器状态序列化，不称这些代数更新为真实训练/物理验收。
    for k,module in modules.items():
        optimizer=optimizers[k]
        sum(p.square().mean() for p in module.parameters()).backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
    replay = VectorReplay(80, 2, 'cpu')
    replay.add(z, z['action'], z['state'], torch.ones(2), torch.zeros(2), torch.ones(2,dtype=torch.bool), torch.zeros(2,dtype=torch.bool), 0)
    validation = VectorReplay(80, 2, 'cpu')
    validation.add(z, z['action'], z['state'], torch.ones(2), torch.zeros(2), torch.ones(2,dtype=torch.bool), torch.zeros(2,dtype=torch.bool), 0)
    generator=torch.Generator().manual_seed(42)
    collector=SimpleNamespace(state=z, episode_serial=2, real_transitions=2, reset_count=2, vector_steps=1,
        generator=generator, device='cpu', reset=lambda **kwargs:None)
    config=dict(music={},rewards=control.weights,reference_steps=10,control_hz=50)
    expected={k:{n:v.clone() for n,v in m.state_dict().items()} for k,m in modules.items()}
    path=tmp_path/'vector_full.pt'
    save_vector(path,engine,critic,optimizers,replay,TangentReplay(),collector,dict(wm_updates=1,accepted_updates=0),config,validation)
    with torch.no_grad():
        for module in modules.values():
            for p in module.parameters():p.zero_()
    empty_validation=VectorReplay(80,2,'cpu')
    restored=load_vector(path,engine,critic,optimizers,replay,TangentReplay(),collector,config,empty_validation)
    assert restored['wm_updates']==1 and len(empty_validation)==2
    for k,module in modules.items():
        for name,value in module.state_dict().items():torch.testing.assert_close(value,expected[k][name])
        assert optimizers[k].state_dict()['state']


def test_contact_refinement_never_hides_negative_or_noise():
    import copy
    from gem.closedloop.reference_adapter.vector_audit import resolve_contact_case
    criteria = dict(repeatability_max_abs=1e-4, root_position_m=.05)
    probe = dict(kind='residual_offset', direction_status='uncertain_contact_change', passed=False,
                 physical_return_difference=.0055, noise_floor=1e-6, contact_matches_baseline=False)
    coarse = dict(passed=False, real_terminal=False, repeat_contacts_equal=True, initial_max_abs=0.,
                  repeatability_max_abs=1e-6, model_errors=dict(root_position_m=.0005), directions=[probe])
    trial = copy.deepcopy(coarse)
    trial.update(passed=True)
    trial['directions'][0].update(passed=True, direction_status='positive', contact_matches_baseline=True,
                                  physical_return_difference=.0002)
    assert resolve_contact_case(coarse, [trial, copy.deepcopy(trial)], criteria)['passed']
    assert not resolve_contact_case(coarse, [trial], criteria)['passed']
    negative = copy.deepcopy(trial)
    negative['directions'][0].update(direction_status='negative', passed=False, physical_return_difference=-.0002)
    assert not resolve_contact_case(coarse, [trial, negative], criteria)['passed']
    # 已通过的大尺度也不能掩盖新测得的小尺度高信噪比反向。
    assert not resolve_contact_case(trial, [trial, negative], criteria)['passed']
    invalid_pair = copy.deepcopy(negative)
    invalid_pair['initial_max_abs'] = .1
    assert resolve_contact_case(trial, [trial, invalid_pair], criteria)['passed']
    assert not resolve_contact_case(coarse, [trial, invalid_pair], criteria)['passed']
    noisy = copy.deepcopy(trial)
    noisy['directions'][0].update(direction_status='uncertain_below_noise', passed=False, physical_return_difference=1e-8)
    assert not resolve_contact_case(coarse, [trial, noisy], criteria)['passed']
    crossed = copy.deepcopy(trial)
    crossed['directions'][0]['contact_matches_baseline'] = False
    assert not resolve_contact_case(coarse, [trial, crossed], criteria)['passed']
    original_negative = copy.deepcopy(coarse)
    original_negative['directions'][0].update(direction_status='negative', physical_return_difference=-.0055)
    assert not resolve_contact_case(original_negative, [trial, trial], criteria)['passed']
