"""联合 Stage2 的缺失音频、来源隔离、均衡回放与评价选择回归。

临时fixture只验证任务/缓存/采样语义，不生成正式训练数据，不把合成状态当真实
物理结果。可用真实SM资产时额外核验同一2654维网络的文本条件执行与梯度路径；
无资产时仅该数值集成案例明确skip，其余合同测试始终执行。测试不启动仿真训练。
"""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gem.closedloop.reference_adapter.application import read_tasks, raw_task
from gem.closedloop.reference_adapter.music_reward import AudioTrack, MusicReward, MusicState
from gem.closedloop.reference_adapter.tasks import condition_contract, task_end_tick, modality_metrics
from gem.closedloop.reference_adapter.vector_bank import VectorBank
from gem.closedloop.reference_adapter.vector_replay import VectorReplay
from gem.closedloop.reference_adapter.alternating_entry import validation_ranking, select_validation_best
from gem.closedloop.reference_adapter.evaluation import beat_report


def test_text_task_no_audio_file_duration_and_source_split(tmp_path):
    state = np.zeros(55, dtype=np.float32); state[3] = 1.
    np.save(tmp_path/'state.npy', state)
    task = dict(id='walk', source_group_id='mother', condition_kind='text', text='walk forward',
                duration_s=2.01, split='train', seed=42, initial_state=str(tmp_path/'state.npy'))
    path = tmp_path/'tasks.json'
    path.write_text(json.dumps([task]))
    checked = read_tasks(path, preparing=True)[0]
    music, initial = raw_task(checked, 'cpu')
    assert music is None and initial.shape == (1, 55)
    assert checked['song_id'] == 'mother' and task_end_tick(task) == 1206
    assert 'text_sha256' in condition_contract(task)
    path.write_text(json.dumps([task, dict(task, id='walk2', split='validation')]))
    with pytest.raises(ValueError, match='泄漏'):
        read_tasks(path, preparing=True)
    with pytest.raises(ValueError, match='时长|声明'):
        task_end_tick({k: v for k, v in task.items() if k != 'duration_s'})


def test_no_audio_features_zero_and_activity_retained():
    like = torch.zeros(2, 55)
    audio = AudioTrack.without_audio(1206, like=like)
    assert audio.control_end_tick == 1206 and audio.end_tick == 1220
    features, valid, times = audio.preview(37*12)
    assert not features.any() and not valid.any() and not times.any()
    reward = MusicReward(torch.ones(21))
    memory = MusicState.empty(like)
    for step in range(25):
        memory = reward.push(memory, torch.ones(2, 21)*.05, torch.ones(2, 21), step*12)
        memory, music, activity, info = reward.settle(memory, audio)
        assert torch.equal(music, torch.zeros(2)) and not info['music_valid'].any()
    assert activity.min() > 0
    report = beat_report(torch.zeros(20, 21), torch.arange(20)*12, audio)
    assert report['applicable'] is False and report['f1'] is None


def fake_bank():
    bank = VectorBank.__new__(VectorBank)
    bank.device = 'cpu'
    bank.task_modalities = torch.tensor([0]*9+[1])
    bank.by_split = {'train': torch.arange(10)}
    return bank


def test_balanced_environment_tasks_including_single_environment_resets():
    bank = fake_bank(); generator = torch.Generator().manual_seed(123)
    songs = bank.sample_tasks(100, 'train', generator=generator)
    assert int(bank.task_modalities[songs].sum()) == 50
    modes = [int(bank.task_modalities[bank.sample_tasks(1, 'train', generator=generator)][0]) for _ in range(200)]
    assert 60 < sum(modes) < 140  # 单环境复位也必须随机两模态，不能因奇数批量永远选music。


def replay_fixture():
    replay = VectorReplay(1000, 10, 'cpu', compute_mc=False, task_modalities=torch.tensor([0]*9+[1]))
    for step in range(24):
        # 文本仅占实际Replay的10%，更新batch仍需精确50:50。
        marker = torch.arange(10).float()
        z = dict(state=marker[:, None].expand(-1, 55), xi=torch.zeros(10, 25),
                 frame=torch.full((10,), step), start_frame=torch.zeros(10, dtype=torch.long),
                 episode=torch.arange(10), song=torch.arange(10))
        done = torch.zeros(10, dtype=torch.bool)
        replay.add(z, torch.zeros(10, 21), z['state'], torch.zeros(10), torch.ones(10), done, done, 0)
    return replay


def test_wm_and_adapter_batches_balanced_with_continuous_windows():
    torch.manual_seed(42)
    replay = replay_fixture()
    starts = replay.starts(64, horizon=64, ends=torch.full((10,), 200))
    assert int((starts['song'] == 9).sum()) == 32
    windows = replay.windows(64, 8)
    assert int((windows['state'][:, 0, 0] == 9).sum()) == 32
    assert torch.equal(windows['state'][:, 1:], windows['next_state'][:, :-1])
    clone = VectorReplay(1000, 10, 'cpu', compute_mc=False, task_modalities=torch.tensor([0]*9+[1]))
    clone.load_state_dict(replay.state_dict())
    with pytest.raises(ValueError, match='采样合同'):
        VectorReplay(1000, 10, 'cpu').load_state_dict(replay.state_dict())
    # 所有文本起点尾部无H64支持时，明确失败，不偷偷退回音乐数据。
    with pytest.raises(ValueError, match='H64'):
        replay.starts(64, horizon=64, ends=torch.tensor([200]*9+[10]))


def test_modality_evaluation_and_best_are_equal_weighted():
    def row(kind, done, root, tracking):
        return dict(condition_kind=kind, full_task_complete=done, root_position_rmse_m=root,
            full_horizon_tracking=tracking, jerk_rms=10., contact_foot_slip_mps=.01,
            music={'f1': .8 if kind == 'music' else None})
    rows = [row('music', True, .1, .09)]*9+[row('text', False, .3, .03)]
    groups = modality_metrics(rows)
    assert groups['text']['mean_beat_f1'] is None
    report = dict(by_modality=groups)
    assert validation_ranking(report) == pytest.approx((.5, -.2, .06))
    better = copy.deepcopy(report)
    better['by_modality']['text']['completion_rate'] = 1.
    # 完成率优先于根误差、tracking，且两个模态均显示在best记录。
    better['by_modality']['text']['mean_root_position_rmse_m'] = .5
    best = select_validation_best(None, report, better, None, adapter_updates=10, scope='full_task')
    assert best['adapter_updates'] == 10 and set(best['by_modality']) == {'music', 'text'}


def test_nominal_v2_checks_text_condition_and_duration(tmp_path, monkeypatch):
    from gem.closedloop.reference_adapter import nominal
    state = tmp_path/'state.npy'; state.write_bytes(b'initial')
    task = dict(condition_kind='text', text='wave', duration_s=2.)
    contract = condition_contract(task) | dict(initial_sha256=nominal.file_digest(state), seed=42, forecast_enabled=False)
    payload = dict(schema=nominal.MULTIMODAL_NOMINAL_SCHEMA, source='frozen_genmo_accepted_reference',
                   contract=contract, world_qpos=torch.zeros(1, 100, 28), joint_names=[], start_tick=0)
    path = tmp_path/'nominal.pt'; torch.save(payload, path)
    monkeypatch.setattr(nominal, 'make_nominal', lambda *a, **kw: 'nominal')
    result, _ = nominal.load_nominal(path, SimpleNamespace(spec=None), device='cpu', initial_path=state, seed=42, task=task)
    assert result == 'nominal'
    for changed in (dict(task, text='sit'), dict(task, duration_s=3.)):
        with pytest.raises(ValueError, match='不匹配'):
            nominal.load_nominal(path, SimpleNamespace(spec=None), device='cpu', initial_path=state, seed=42, task=changed)


def test_asset_preparation_uses_allowed_caption_interval_and_named_initial(tmp_path, monkeypatch):
    from tools.prepare_stage2_multimodal_assets import prepare_tasks
    from gem.closedloop import stage1_multimodal_dataset as datasets
    from tools.data.bumi import prepare_multimodal_stage1 as prepare
    names = [f'joint_{i}' for i in range(21)]
    qpos = torch.zeros(100, 28); qpos[:, 3] = 1; qpos[:, 2] = .65
    qpos[:, 7:] = torch.arange(21).float()[None]
    sources = {}
    allow = {'entries': {}}
    for split in ('train', 'val', 'test'):
        for kind in ('music', 'text'):
            key = f'{split}_{kind}'
            record = dict(frames=100, captions=['wrong interval', 'wave hand'], caption_intervals=[[0, 5], [10, 90]])
            row = dict(sample_id=key)
            source = SimpleNamespace(modality=kind, record_keys=[key, 'excluded'], rows=[row, row],
                release=SimpleNamespace(read_record=lambda index, record=record: record),
                kinematics=SimpleNamespace(joint_order=names), data_identity={'manifest': f'{key}.json'},
                reader=SimpleNamespace(load_aligned_sequence=lambda row, **kwargs: dict(qpos=qpos, music_path='actual_music.pt')))
            sources[(split, kind)] = source
            allow['entries'][f'{split}/{kind}'] = [key]
    monkeypatch.setattr(datasets, '_load_allowlist', lambda data: allow)
    monkeypatch.setattr(datasets, 'make_multimodal_source', lambda entry, source_name, split, **kwargs: sources[(split, source_name)])
    monkeypatch.setattr(prepare, '_record_identities', lambda source, index, aliases: [source.record_keys[index]])
    data = {'datasets': {split: {'music': {}, 'text': {}} for split in ('train', 'val', 'test')}}
    bundle = SimpleNamespace(spec=SimpleNamespace(joint_names=list(reversed(names))))
    tasks = prepare_tasks(data, bundle, tmp_path, train_per_source=1, heldout_per_source=1)
    assert len(tasks) == 6 and {t['split'] for t in tasks} == {'train', 'validation', 'evaluation'}
    text = next(task for task in tasks if task['condition_kind'] == 'text')
    assert text['text'] == 'wave hand' and text['source_interval_frames'] == [10, 90]
    assert text['duration_ticks'] == 79*20 and 'music_features' not in text
    initial = np.load(text['initial_state'])
    np.testing.assert_array_equal(initial[13:34], np.arange(20, -1, -1))
    assert not initial[7:13].any() and not initial[34:].any()


def test_multimodal_cache_generation_records_support_without_physics(tmp_path, monkeypatch):
    from gem.closedloop.reference_adapter import nominal
    from gem.runtime import multimodal_stage1
    names = tuple(f'j{i}' for i in range(21))
    initial = tmp_path/'initial.npy'; initial.write_bytes(b'initial')
    checkpoint = tmp_path/'stage1.pt'; checkpoint.write_bytes(b'multimodal checkpoint')
    task = dict(condition_kind='text', text='wave', duration_s=2., initial_state=str(initial), seed=7)
    seen = {}
    def generate(cfg, passed_task, *, device, required_end_tick):
        seen.update(required_end_tick=required_end_tick, names=cfg['robot_spec_joint_names'])
        world = torch.zeros(1, (required_end_tick+19)//20+1, 28); world[..., 3] = 1
        return dict(world_qpos=world, joint_names=names, start_tick=0, contact=torch.zeros(1, world.shape[1], 2))
    monkeypatch.setattr(multimodal_stage1, 'generate_stage1_reference', generate)
    bundle = SimpleNamespace(spec=SimpleNamespace(joint_names=names))
    path = tmp_path/'cache.pt'
    contract = nominal.prepare_cache(None, bundle, {'checkpoint': str(checkpoint)}, task, None,
                                     torch.zeros(1, 55), path, bootstrap_latency_ticks=0)
    assert seen['required_end_tick'] == 1332 and seen['names'] == list(names)
    assert contract['actual_support_tick'] >= 1332 and contract['task_end_tick'] == 1200
    loaded, saved = nominal.load_nominal(path, bundle, device='cpu', initial_path=initial, seed=7, task=task)
    assert loaded.upper >= 1332 and saved['genmo_checkpoint_sha256'] == nominal.file_digest(checkpoint)


def test_text_vector_adapter_same_2654_inputs_and_gradients(system):
    from gem.closedloop.reference_adapter.vector_control import VectorControl
    engine, z = system
    audio = AudioTrack.without_audio(1000, like=z.executed_state)
    task = dict(id='text', song_id='wave', source_group_id='wave', condition_kind='text', split='train')
    bank = VectorBank.from_entries([(task, z.executed_state, z.nominal_reference, audio)],
                                  engine.step.runtime.kin, engine.step.rewards.music)
    control = VectorControl(engine, bank)
    state = control.initial(dict(state=z.executed_state, xi=z.xi), torch.zeros(1, dtype=torch.long),
                            torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long))
    fields = control.fields(state)
    assert control.packed(fields).shape == (1, 2654)
    assert not fields['music'].any() and not fields['music_valid'].any() and not fields['music_times'].any()
    proposal = control.propose(state)
    assert proposal['residual'].shape == (1, 10, 27)
    proposal['action'].square().mean().backward()
    assert engine.step.adapter.net[-1].weight.grad.norm() > 0
    assert all(p.grad is None for p in engine.step.bundle.actor.parameters())


def test_two_modalities_real_mujoco_H8_and_H64_update(system):
    """真实单环境依次各16步，只验物理标签/更新链；合成参考不证明生成质量或硬件。"""
    from gem.closedloop.reference_adapter.mujoco_env import MujocoPhysicalEnv
    from gem.closedloop.reference_adapter.vector_control import VectorControl, detach_state
    from gem.closedloop.reference_adapter.alternating_learning import wm_update, policy_update
    from supermimic.networks import Critic
    engine, original = system
    audio = AudioTrack.from_features(original.audio.features[:, :70])
    text = AudioTrack.without_audio(audio.end_tick, like=original.executed_state)
    entries = [(dict(id=kind, song_id=kind, split='train', condition_kind=kind), original.executed_state,
                original.nominal_reference, track) for kind, track in (('music', audio), ('text', text))]
    bank = VectorBank.from_entries(entries, engine.step.runtime.kin, engine.step.rewards.music)
    control = VectorControl(engine, bank)
    control.adapter.disable_root_xy = True
    env = MujocoPhysicalEnv(engine.step.bundle.spec)
    replay = VectorReplay(128, 1, 'cpu', compute_mc=False, task_modalities=bank.task_modalities)
    frozen = engine.step.bundle.frozen_snapshot()
    completed = []
    for kind in (0, 1):
        with torch.no_grad():
            z = control.initial(env.reset(original.executed_state), torch.tensor([kind]),
                                torch.zeros(1, dtype=torch.long), torch.tensor([kind]))
            steps = 0
            for _ in range(16):
                proposal = control.propose(z, zero=True)
                actual = env.step_physics(proposal['action'])
                end, reward, raw, _, continuation, done, failed = control.advance(z, proposal, actual)
                replay.add(z, proposal['action'], actual, reward, continuation, done, failed, 0)
                if kind:
                    assert not raw['music'].any() and not raw['music_valid'].any()
                steps += 1; z = detach_state(end)
                if bool(done.any()):
                    break
            completed.append(steps)
            assert steps >= 8
    wm_before = next(engine.dynamics.model.net.parameters()).detach().clone()
    wm = wm_update(engine.dynamics, torch.optim.Adam(engine.dynamics.model.net.parameters(), lr=1e-6),
                   replay.windows(2, 8))
    assert np.isfinite(wm['wm_loss']) and not torch.equal(wm_before, next(engine.dynamics.model.net.parameters()))
    critic = Critic(2835).requires_grad_(False)
    starts = replay.starts(2, horizon=64, ends=bank.ends)
    before = engine.step.adapter.net[-1].weight.detach().clone()
    report, trajectory = policy_update(control, engine.dynamics, critic,
        torch.optim.Adam(engine.step.adapter.parameters(), lr=1e-6), starts, 64,
        terminal_value=False, microbatch=2)
    assert trajectory is None and not torch.equal(before, engine.step.adapter.net[-1].weight)
    assert all(torch.isfinite(p).all() for p in engine.step.adapter.parameters())
    engine.step.bundle.assert_unchanged(frozen)
    print(json.dumps(dict(actual_mujoco_steps_by_modality=completed, wm_real_H8_updates=1,
        adapter_H64_updates=1, frozen_actor_unchanged=True,
        evidence='bounded_CPU_synthetic_reference_real_physics_not_full_song_or_hardware')))
