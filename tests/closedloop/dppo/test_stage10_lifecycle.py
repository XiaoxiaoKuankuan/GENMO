"""第十步多轮训练与中断恢复入口的合成CPU生命周期检查。

本测试用小型torch网络、合成UpperTransition及假的worker/完整数据目录替代GPU、
GENMO采样和物理执行；因此只证明入口的多轮调度、真实优化器状态保存/恢复、逐条
rollout落盘、预算延续和失败发布边界，不证明真实机器人动力学或完整数据集可用。
实际复用RunManager、RolloutWriter、StepJournal、固定目标与Critic更新、完整
save_checkpoint/load_checkpoint。所有文件均在pytest临时目录，测试不连接服务器。
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import signal
from types import SimpleNamespace
import uuid

import numpy as np
import pytest
import torch
import yaml

from gem.closedloop.dppo.buffer import UpperTransition
from gem.closedloop.dppo.run_management import RunManager
from tools import train_closedloop_stage10 as entry


class TinyActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.endecoder = SimpleNamespace(mean=torch.zeros(30), std=torch.ones(30))
        self.interface_config = dict(synthetic_lifecycle_fixture=True)


class TinyCritic(torch.nn.Module):
    def __init__(self, **_):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, context, remaining):
        return self.weight.expand(len(remaining)) + context['decision_time'].float() * .01


class TinyBC:
    def __init__(self, *_):
        self.bc_update_steps = 0
        self.generator = torch.Generator().manual_seed(88)

    def state_dict(self):
        return dict(bc_update_steps=self.bc_update_steps, rng=self.generator.get_state())

    def load_state_dict(self, state):
        self.bc_update_steps = state['bc_update_steps']
        self.generator.set_state(state['rng'])


class SyntheticCatalog:
    def __init__(self, *_):
        self.identity = dict(schema='synthetic-full-catalog', complete_splits=['train', 'val', 'test'])
        self.samples = {split: {name: [dict(dataset=name, manifest_sha256='a' * 64,
            row=dict(sample_id=f'{name}-{split}', split=split))]
            for name in ('AIST++', 'AIOZ-GDANCE', 'FineDance', 'Mine')} for split in ('train', 'val', 'test')}

    def audit_files(self, callback, **_):
        for split, sources in self.samples.items():
            for dataset in sources:
                callback(dict(dataset=dataset, split=split))
        return dict(status='passed', data_content_sha256='b' * 64, synthetic=True)

    def load_music(self, sample):
        return np.zeros((90, 35), dtype=np.float32)


class SyntheticSampler:
    def __init__(self, catalog, *, seed, **_):
        self.catalog = catalog
        self.rng = np.random.default_rng(seed)
        self.draw_count = self.controls = 0

    def next_task(self):
        source = tuple(self.catalog.samples['train'])[self.draw_count % 4]
        self.draw_count += 1
        sample = copy.deepcopy(self.catalog.samples['train'][source][0])
        return dict(sample=sample, music=self.catalog.load_music(sample),
                    music_start_frame=int(self.rng.integers(0, 20)))

    def record_execution(self, task, controls):
        self.controls += controls

    def coverage(self):
        return dict(draw_count=self.draw_count, control_steps=self.controls)

    def state_dict(self):
        return dict(draw_count=self.draw_count, controls=self.controls, rng=self.rng.bit_generator.state)

    def load_state_dict(self, value):
        self.draw_count, self.controls = value['draw_count'], value['controls']
        self.rng.bit_generator.state = value['rng']


def observation(tick):
    time = tick / 600
    return dict(music_features=torch.zeros(1, 120, 35), music_valid=torch.ones(1, 120, dtype=torch.bool),
        proprio_history=torch.zeros(1, 50, 48), proprio_history_valid=torch.ones(1, 50, dtype=torch.bool),
        proprio_history_times=((torch.arange(50, dtype=torch.float64) - 49) / 50 + time)[None],
        known_qpos30=torch.zeros(1, 120, 30), known_qpos30_mask=torch.zeros(1, 120, 30, dtype=torch.bool),
        future_valid=torch.ones(1, 120, dtype=torch.bool), future_times=(torch.arange(120, dtype=torch.float64) / 30 + time)[None],
        decision_time=torch.tensor([time], dtype=torch.float64))


class SyntheticEnvironment:
    def __init__(self, config, backend, builder, policy, budget, output):
        self.config, self.backend, self.policy, self.budget = config, backend, policy, budget
        self.output = Path(output)
        self.output.mkdir(parents=True)
        self.policy_version = self.iteration = self.episode_count = self.attempt = self.decision = 0
        self.latency_budget_s = .1

    def reset_task(self, sample, music, *, seed, phase, music_start_frame=0):
        self.episode_count += 1
        self.phase = phase
        self.episode_id = str(uuid.uuid4())
        self.tick = 600

    def step(self):
        self.budget.reserve(self.phase, generations=1, control_steps=1, physics_steps=4)
        begin = self.tick
        self.tick += 12
        self.attempt += 1
        self.backend.sequence += 1
        self.backend.journal.append_result(dict(backend_session_id=self.backend.session_id,
            mutation_seq=self.backend.sequence, result=dict(synthetic_tick=self.tick)))
        value = UpperTransition(identity=dict(run_id=self.config['stage9']['run_id'],
            backend_session_id=self.backend.session_id, episode_id=self.episode_id,
            decision_id=self.decision, policy_version=self.policy_version),
            context=observation(begin), next_context=observation(self.tick),
            chain=torch.zeros(self.policy.steps + 1, 120, 30), old_log_prob=torch.zeros(self.policy.steps, dtype=torch.float64),
            free_mask=torch.ones(120, 30, dtype=torch.bool), rewards=torch.tensor([.02], dtype=torch.float64),
            old_value=0., next_value=0., control_tick_begin=begin, control_tick_end=self.tick,
            executed_control_steps=1, executed_physics_steps=4,
            metadata=dict(remaining_music_seconds=3. - (begin - 600) / 600,
                          next_remaining_music_seconds=3. - (self.tick - 600) / 600))
        self.decision += 1
        return value


@pytest.fixture
def lifecycle(tmp_path, monkeypatch):
    config = yaml.safe_load((entry.ROOT / 'configs/closedloop/stage10_prepare_server1.yaml').read_text())
    config['runtime'].update(genmo_device='cpu', torch_threads=1)
    config['stage10']['training'].update(rollout_upper_steps=2, critic_steps=2, critic_batch=2,
        actor_lr=1e-6, actor_lr_candidates=[1e-6], denoising_steps=2)
    config['model']['ddim_steps'] = 2
    config['stage10']['limits'] = dict(accepted_iterations=4, optimizer_attempts=12, generations=100,
                                     control_steps=200, physics_steps=800)
    config['stage10']['storage'].update(min_free_bytes=1, max_run_bytes=128 * 1024**2,
        checkpoint_reserve_bytes=1024**2, rollout_chunk_size=1)
    for name in ('checkpoint', 'stats', 'kinematics', 'gmt_policy', 'compat_profile', 'isaac_contract'):
        path = tmp_path / f'{name}.fixture'
        path.write_text(f'synthetic asset {name}')
        config['paths'][name] = str(path)
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config))
    harness = SimpleNamespace(config=config, config_path=config_path, worker_starts=0,
        actor_calls=0, failure_on_actor_call=None, failure_on_frozen_call=None, frozen_calls=0,
        environments=[], backend_sessions=[])

    class FakeWorkers:
        def __init__(self, *_):
            self.temp = SimpleNamespace(name=str(tmp_path / ('worker-' + str(uuid.uuid4()))))
            self.entries, self.shutdown = [], {}

        def start(self, *_):
            harness.worker_starts += 1
            client = object()
            self.entries.append(dict(client=client))
            return client

        def close(self):
            self.shutdown = dict(gmt=dict(closed=True, process_exit_code=0, policy_unchanged=True,
                                         runtime_parameters_unchanged=True))

    class FakeBackend:
        def __init__(self, client, journal, **_):
            self.client, self.journal = client, journal
            self.session_id, self.sequence = str(uuid.uuid4()), 0
            harness.backend_sessions.append(self.session_id)

        def call(self, method, **_):
            assert method == 'verify_frozen'
            harness.frozen_calls += 1
            return dict(policy_unchanged=harness.frozen_calls != harness.failure_on_frozen_call,
                runtime_parameters_unchanged=True, execution_journal=dict(executed_seq=self.sequence, acked_seq=self.sequence))

    def environment(*args):
        value = SyntheticEnvironment(*args)
        harness.environments.append(value)
        return value

    def actor_update(policy, optimizer, transitions, targets, *, reserve_attempt, bc, **_):
        harness.actor_calls += 1
        if harness.actor_calls == harness.failure_on_actor_call:
            raise RuntimeError('synthetic non-candidate optimizer failure')
        reserve_attempt()
        optimizer.zero_grad(set_to_none=True)
        policy.actor.weight.sum().backward()
        optimizer.step()
        bc.bc_update_steps += 1
        torch.rand((), generator=bc.generator)
        return dict(parameters_changed=True, ppo_only_gradient_norm=1., total_gradient_norm=1.,
                    optimizer_steps=1, bc=dict(bc_update_steps=bc.bc_update_steps))

    def preflight(cfg, **_):
        assets = {key: entry.sha256_file(cfg['paths'][key]) for key in
            ('checkpoint', 'stats', 'kinematics', 'gmt_policy', 'compat_profile', 'isaac_contract')}
        return dict(ready=True, asset_sha256=assets, repositories={})

    monkeypatch.setattr(entry, 'runtime_preflight', preflight)
    monkeypatch.setattr(entry, 'FullMusicCatalog', SyntheticCatalog)
    monkeypatch.setattr(entry, 'FullMusicSampler', SyntheticSampler)
    monkeypatch.setattr(entry, '_sources', lambda *a: dict(source_manifest_sha256='c' * 64))
    monkeypatch.setattr(entry, 'verify_source_provenance', lambda *a: dict(unchanged=True))
    monkeypatch.setattr(entry, 'load_actor', lambda cfg: (TinyActor(),
        SimpleNamespace(model=SimpleNamespace(proprio_scales=[1.] * 48)), dict(synthetic=True)))
    monkeypatch.setattr(entry, 'UpperCritic', TinyCritic)
    monkeypatch.setattr(entry, 'DPPODiffusionPolicy', lambda actor, *, steps, **kwargs: SimpleNamespace(actor=actor, steps=steps))
    monkeypatch.setattr(entry, 'SupervisedAnchor', TinyBC)
    monkeypatch.setattr(entry, 'Workers', FakeWorkers)
    monkeypatch.setattr(entry, 'AcknowledgedBackend', FakeBackend)
    monkeypatch.setattr(entry, 'UpperEnvironment', environment)
    monkeypatch.setattr(entry, 'BumiKinematics', lambda *a: None)
    monkeypatch.setattr(entry, 'BumiMotionFeatureCodec', lambda *a: None)
    monkeypatch.setattr(entry, 'OnlineConditionBuilder', lambda *a: None)
    monkeypatch.setattr(entry, 'calibrate', lambda *a: dict(synthetic=True, latency_budget_s=.1))
    monkeypatch.setattr(entry, 'probability_check', lambda *a, **k: dict(passed=True))
    monkeypatch.setattr(entry, 'actor_update', actor_update)
    monkeypatch.setattr(entry, 'analytic_kl', lambda *a: dict(mean_joint_kl=.001))
    return harness


def run(lifecycle, output, *, stop, resume=None):
    args = ['--config', str(lifecycle.config_path), '--mode', 'train', '--output-dir', str(output),
            '--stop-after-iteration', str(stop)]
    if resume is not None:
        args += ['--resume', str(resume)]
    return entry.main(args)


def latest(output):
    descriptor = json.loads((output / 'latest.json').read_text())
    return torch.load(output / descriptor['path'], weights_only=False, map_location='cpu'), descriptor


def test_two_iterations_then_resume_one_more_preserves_optimizer_sampler_and_values(lifecycle, tmp_path):
    output = tmp_path / 'run'
    assert run(lifecycle, output, stop=2) == 0
    second, second_descriptor = latest(output)
    assert second['state']['iteration'] == second['state']['actor_updates'] == 2
    assert second['state']['critic_updates'] == 4
    assert second['samplers']['music']['draw_count'] == second['samplers']['bc']['bc_update_steps'] == 2
    assert int(next(iter(second['actor_optimizer']['state'].values()))['step']) == 2
    # 磁盘不可变转移中的旧价值必须已经在写入时采样，不能只在内存计算GAE后更新。
    reports = sorted(output.glob('sessions/*/iterations/*/summary.json'))
    for report_path in reports:
        report = json.loads(report_path.read_text())
        targets = torch.load(output / report['targets_path'], weights_only=False)
        manifest_path = output / report['rollout_manifest']
        manifest = json.loads(manifest_path.read_text())
        values = []
        for chunk in manifest['chunks']:
            chunk_path = manifest_path.parent / chunk['path']
            for record in json.loads(chunk_path.read_text())['records']:
                row = torch.load(chunk_path.parent / record['path'], weights_only=False)
                values.append(row.old_value)
                assert row.identity['policy_version'] == report['policy_version_before']
        assert torch.equal(torch.tensor(values, dtype=torch.float64), targets['old_values'])
        assert any(value != 0. for value in values)
    assert run(lifecycle, output, stop=3, resume='latest') == 0
    third, third_descriptor = latest(output)
    assert third['state']['iteration'] == third['state']['policy_version'] == third['state']['actor_updates'] == 3
    assert third['state']['critic_updates'] == 6
    assert third['samplers']['music']['draw_count'] == third['samplers']['bc']['bc_update_steps'] == 3
    assert int(next(iter(third['actor_optimizer']['state'].values()))['step']) == 3
    assert not torch.equal(second['actor']['weight'], third['actor']['weight'])
    assert second_descriptor['session_id'] != third_descriptor['session_id']
    assert len(set(lifecycle.backend_sessions)) == 2
    ledger = json.loads((output / 'budget.json').read_text())
    assert ledger['used'] == dict(accepted_iterations=3, optimizer_attempts=3, generations=6,
                                  control_steps=6, physics_steps=24)


def test_non_candidate_failure_does_not_publish_a_partial_iteration(lifecycle, tmp_path):
    output = tmp_path / 'run'
    lifecycle.failure_on_actor_call = 2
    assert run(lifecycle, output, stop=2) == 1
    checkpoint, descriptor = latest(output)
    assert checkpoint['state']['iteration'] == descriptor['iteration'] == 1
    failed = [json.loads(path.read_text()) for path in output.glob('sessions/*/iterations/*/summary.json')]
    assert sorted(item['status'] for item in failed) == ['accepted', 'failed']
    assert json.loads((output / 'budget.json').read_text())['used']['generations'] == 4
    # 上次失败后Critic可能已经更新；显式恢复须从已发布完整状态而不是这些半更新对象继续。
    lifecycle.failure_on_actor_call = None
    assert run(lifecycle, output, stop=2, resume='latest') == 0
    restored, _ = latest(output)
    assert restored['state']['iteration'] == 2
    assert restored['state']['critic_updates'] == 4
    assert restored['samplers']['bc']['bc_update_steps'] == 2
    assert json.loads((output / 'budget.json').read_text())['used']['generations'] == 6


def test_frozen_failure_after_actor_step_keeps_previous_latest(lifecycle, tmp_path):
    output = tmp_path / 'run'
    lifecycle.failure_on_frozen_call = 2
    assert run(lifecycle, output, stop=2) == 1
    checkpoint, descriptor = latest(output)
    assert checkpoint['state']['iteration'] == descriptor['iteration'] == 1
    used = json.loads((output / 'budget.json').read_text())['used']
    assert used['optimizer_attempts'] == 2 and used['accepted_iterations'] == 1


def test_resume_identity_mismatch_is_rejected_before_starting_worker(lifecycle, tmp_path):
    output = tmp_path / 'run'
    assert run(lifecycle, output, stop=1) == 0
    before = (output / 'latest.json').read_bytes()
    Path(lifecycle.config['paths']['stats']).write_text('changed statistics identity')
    assert run(lifecycle, output, stop=2, resume='latest') == 1
    assert lifecycle.worker_starts == 1
    assert (output / 'latest.json').read_bytes() == before
    # 失败也必须释放run锁，允许用户修复配置后显式再恢复。
    with RunManager(output, resume=True):
        pass


def test_first_iteration_failure_can_resume_the_explicit_initial_state(lifecycle, tmp_path):
    output = tmp_path / 'run'
    lifecycle.failure_on_actor_call = 1
    assert run(lifecycle, output, stop=1) == 1
    assert not (output / 'latest.json').exists()
    initial = output / 'checkpoints' / 'initial.pt'
    assert torch.load(initial, weights_only=False)['state']['iteration'] == 0
    lifecycle.failure_on_actor_call = None
    assert run(lifecycle, output, stop=1, resume=initial) == 0
    checkpoint, _ = latest(output)
    assert checkpoint['state']['iteration'] == checkpoint['state']['actor_updates'] == 1
    assert checkpoint['state']['critic_updates'] == 2
    assert checkpoint['samplers']['bc']['bc_update_steps'] == 1
    assert json.loads((output / 'budget.json').read_text())['used']['generations'] == 4


def test_resume_rejects_an_older_published_iteration_before_worker_start(lifecycle, tmp_path):
    output = tmp_path / 'run'
    assert run(lifecycle, output, stop=2) == 0
    earlier = next(output.glob('checkpoints/stage10_000001_*.pt'))
    before = (output / 'latest.json').read_bytes()
    assert run(lifecycle, output, stop=3, resume=earlier) == 1
    assert lifecycle.worker_starts == 1
    assert (output / 'latest.json').read_bytes() == before


def test_budget_setup_failure_releases_the_writer_lock(lifecycle, tmp_path):
    output = tmp_path / 'run'
    assert run(lifecycle, output, stop=1) == 0
    lifecycle.config['stage10']['limits']['generations'] += 1
    lifecycle.config_path.write_text(yaml.safe_dump(lifecycle.config))
    with pytest.raises(ValueError, match='budget'):
        run(lifecycle, output, stop=2, resume='latest')
    assert lifecycle.worker_starts == 1
    with RunManager(output, resume=True):
        pass


def test_final_metrics_failure_restores_signal_handlers_and_releases_lock(lifecycle, tmp_path, monkeypatch):
    output = tmp_path / 'run'
    previous = {value: signal.getsignal(value) for value in (signal.SIGINT, signal.SIGTERM)}
    original = RunManager.append_metrics
    def append_metrics(manager, value):
        if value.get('event') == 'session_end':
            raise OSError('synthetic final metrics disk failure')
        return original(manager, value)
    monkeypatch.setattr(RunManager, 'append_metrics', append_metrics)
    with pytest.raises(OSError, match='final metrics disk failure'):
        run(lifecycle, output, stop=1)
    assert {value: signal.getsignal(value) for value in previous} == previous
    with RunManager(output, resume=True):
        pass
    assert latest(output)[0]['state']['iteration'] == 1
    assert not list(output.glob('sessions/*/completion.json'))


def test_metrics_failure_after_publication_preserves_accepted_evidence_and_resume(lifecycle, tmp_path, monkeypatch):
    output = tmp_path / 'run'
    original = RunManager.append_metrics

    def append_metrics(manager, value):
        if value.get('event') == 'iteration_accepted':
            raise OSError('synthetic post-publication metrics failure')
        return original(manager, value)

    monkeypatch.setattr(RunManager, 'append_metrics', append_metrics)
    assert run(lifecycle, output, stop=2) == 1
    checkpoint, descriptor = latest(output)
    assert checkpoint['state']['iteration'] == descriptor['iteration'] == 1
    summary_path = output / descriptor['metadata']['iteration_summary']
    accepted_evidence = summary_path.read_bytes()
    assert json.loads(accepted_evidence)['status'] == 'accepted'
    failed_session = next(output.glob('sessions/*'))
    assert json.loads((failed_session / 'summary.json').read_text())['status'] == 'failed'
    completion_path = failed_session / 'completion.json'
    if completion_path.exists():
        completion = json.loads(completion_path.read_text())
        assert completion['status'] == 'failed' and completion['exit_code'] != 0
    # 指标失败不应撤销已发布的完整模型，也不能改写被latest引用的接受证据。
    monkeypatch.setattr(RunManager, 'append_metrics', original)
    assert run(lifecycle, output, stop=2, resume='latest') == 0
    restored, _ = latest(output)
    assert restored['state']['iteration'] == restored['state']['actor_updates'] == 2
    assert restored['state']['critic_updates'] == 4
    assert restored['samplers']['bc']['bc_update_steps'] == 2
    assert summary_path.read_bytes() == accepted_evidence


def test_signal_during_update_stops_only_after_publishing_the_iteration(lifecycle, tmp_path, monkeypatch):
    output = tmp_path / 'run'
    original = entry.actor_update
    def actor_update(*args, **kwargs):
        result = original(*args, **kwargs)
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        return result
    monkeypatch.setattr(entry, 'actor_update', actor_update)
    assert run(lifecycle, output, stop=3) == 0
    checkpoint, _ = latest(output)
    assert checkpoint['state']['iteration'] == 1
    summary = json.loads(next(output.glob('sessions/*/summary.json')).read_text())
    assert summary['stopped_on_signal'] and summary['final_iteration'] == 1
    assert lifecycle.actor_calls == 1
