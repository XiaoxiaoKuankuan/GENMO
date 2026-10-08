"""八路周期评估连接、固定初始价值诊断与本rank RNG合同的CPU验收。

使用真实evaluation计划、逐控制奖励证据、merge和periodic_monitor，替换物理环境、
分布式通信及预算租约，检查_evaluate首次建立固定参考、100轮仅内存候选、300轮
完整保存后候选绑定、恢复时读取原初始参考，以及每轮TensorBoard/JSONL曲线。
另以非均分world验证sample分片的credit数量，以四个sample八rank验证空rank合同。
Critic用有限真实奖励的GAE targets并保留初始bootstrap，只验固定回归诊断正确性，
不代表仿真质量或训练效果。CUDA RNG API用计数替身证明local模式不触碰其他GPU。
全部运行产物位于pytest tmp_path，不启动GPU/Isaac或任何正式训练。
"""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from gem.closedloop.dppo import evaluation
from gem.closedloop.dppo import parallel_training as training
from gem.closedloop.dppo.budget import atomic_json
from gem.closedloop.dppo.periodic_monitor import (
    build_balanced_plan,
    build_fixed_critic_reference,
    fixed_critic_diagnostic,
    load_best_pointers,
    mark_saved_evaluation,
    merge_periodic_reports,
    shard_plan,
)
from tests.closedloop.dppo.test_periodic_monitor import Catalog
from tests.test_evaluation import Env


class Policy:
    def __init__(self):
        self.actor = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.zeros_(self.actor.weight)
        self.steps, self.kernel_config = 1, {'kernel': 'cpu_test_gaussian'}
    def transition_parameters(self, context, x, step):
        return dict(mean=x+self.actor.weight.mean(), std=torch.ones_like(x))


class Critic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.))
    def forward(self, context, remaining):
        return context['x'].reshape(-1)+self.weight


class IndependentEnv(Env):
    def __init__(self, config, backend, builder, loaded, budget, output):
        super().__init__(loaded)
        self.config, self.output = config, output
        self.output.joinpath('raw_samples').mkdir()
        self.episode = 0
    def reset_task(self, *args, **kwargs):
        super().reset_task(*args, **kwargs)
        self.episode += 1
        x = torch.zeros(1, 120, 30)
        params = self.policy.transition_parameters({'x': torch.ones(1, 1)}, x, 0)
        trace = dict(conditions={'x': torch.ones(1, 1)}, chain=torch.stack([x, x], 1),
            old_means=params['mean'].detach()[:, None], old_stds=params['std'].detach()[:, None],
            free_mask=torch.ones_like(x, dtype=torch.bool))
        self.raw = self.output/'raw_samples'/f'{self.episode:06d}.pt'
        torch.save({'trace': trace}, self.raw)
    def step(self, deterministic=False):
        begin = self.steps
        row = super().step(deterministic)
        row['metadata'].update(raw_sample_path=str(self.raw), remaining_music_seconds=.2,
                               next_remaining_music_seconds=0.)
        return SimpleNamespace(**row, context={'x': torch.ones(1, 1)}, next_context=None,
            identity=dict(backend_session_id='cpu_backend', episode_id=self.episode, policy_version=self.policy_version),
            control_tick_begin=begin, control_tick_end=self.steps, transition_valid=True)


def context(tmp_path, monkeypatch, *, samples=1, world=1):
    output, session = tmp_path/'run', tmp_path/'run'/'sessions'/'test'
    session.mkdir(parents=True)
    loaded, critic = Policy(), Critic()
    class Collective:
        rank, world_size, device = 0, world, torch.device('cpu')
        def all_gather_object(self, value):
            if isinstance(value, float):
                return [value]*world
            return [value]
    c = SimpleNamespace(distributed=Collective(), generators={}, output=output, session=session,
        config={'stage9': {'episode_seconds': 30.}, 'runtime': {'timing_contract': 'deployment_critical.v2'}},
        state=dict(iteration=0, policy_version=0), identity={'source': 'immutable'}, actor=loaded.actor,
        critic=critic, policy=loaded, backend=SimpleNamespace(journal=object()), builder=None, writer=None,
        guard=SimpleNamespace(account_file=lambda path: None), budget=None,
        settings={'gamma_upper': .99, 'lambda_upper': .95, 'episode_seconds': 30.},
        env=SimpleNamespace(latency_budget_s=.5, policy_version=0, iteration=0), catalog=Catalog(tmp_path),
        stage={'evaluation': {'samples_per_source': samples, 'seeds': [42, 1729],
                             'selection_seed': 42, 'episode_seconds': 10.}})
    credits = []
    def phase(context, name, per_rank):
        credits.append(copy.deepcopy(per_rank))
        path = session/'phases'/name/'rank00'
        path.mkdir(parents=True)
        return path, name, object()
    monkeypatch.setattr(training, '_new_phase', phase)
    monkeypatch.setattr(training, 'root_call', lambda collective, fn: fn())
    monkeypatch.setattr(training, 'local_call', lambda collective, fn: fn())
    monkeypatch.setattr(training, 'finish_lease', lambda *args: {'used': 0})
    monkeypatch.setattr(training, 'UpperEnvironment', IndependentEnv)
    monkeypatch.setattr(training, 'GuardedStepJournal', lambda *args: SimpleNamespace(close=lambda: None))
    return c, credits


def test_evaluate_connects_identity_fixed_references_curves_and_real_saved_boundary(tmp_path, monkeypatch):
    c, _ = context(tmp_path, monkeypatch)
    original = c.backend.journal
    first = training._evaluate(c, 'initial')
    assert c.backend.journal is original
    assert first['evaluation_identity']['rng_scope'] == 'current_cuda_device'
    assert first['task_count'] == 8 and first['fixed_actor_drift']['mean_joint_kl'] == pytest.approx(0.)
    assert first['fixed_critic_diagnostic']['available'] and first['fixed_critic_diagnostic']['transition_count'] == 8
    assert first['selection']['best_saved'] is None
    atomic_json(c.output/'evaluation_baseline.json', first)
    reference = json.loads((c.output/'fixed_diagnostic_reference.json').read_text())
    c.state['iteration'] = 100
    with torch.no_grad():
        c.actor.weight.add_(.1)
    second = training._evaluate(c, '000100')
    assert second['fixed_actor_drift']['mean_joint_kl'] > 0
    assert second['fixed_actor_drift']['reference_sha256'] == first['fixed_actor_drift']['reference_sha256']
    assert second['selection']['best_saved'] is None
    # 恢复后重新从初始manifest读取，而不是采用100轮的新raw trace。
    del c._periodic_fixed_reference
    c.state['iteration'] = 300
    checkpoint = c.output/'checkpoints'/'saved300.pt'
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b'complete_checkpoint_placeholder')
    atomic_json(c.output/'latest.json', dict(iteration=300, path='checkpoints/saved300.pt', sha256='a'*64,
        metadata={'actor_model_fingerprint': evaluation._model_fingerprint(c.actor)}))
    third = training._evaluate(c, '000300')
    assert third['selection']['best_saved']['iteration'] == 300
    assert third['selection']['best_saved']['checkpoint']['path'] == 'checkpoints/saved300.pt'
    assert json.loads((c.output/'fixed_diagnostic_reference.json').read_text()) == reference
    rows = [json.loads(line) for line in (c.output/'curves.jsonl').read_text().splitlines()]
    assert [row['step'] for row in rows] == [0, 100, 300]
    assert 'validation/fixed_critic_diagnostic/mse' in rows[0]['metrics']


def test_credit_counts_actual_sample_shards_for_nondivisible_world(tmp_path, monkeypatch):
    c, credits = context(tmp_path, monkeypatch, samples=4, world=3)
    class Finished(Exception):
        pass
    def inspect_phase(context, name, per_rank):
        credits.append(per_rank)
        raise Finished()
    monkeypatch.setattr(training, '_new_phase', inspect_phase)
    with pytest.raises(Finished):
        training._evaluate(c, 'initial')
    assert [credit['generations'] for credit in credits[0]] == [12*22, 10*22, 10*22]


def test_credit_limits_use_each_rank_latency(tmp_path, monkeypatch):
    c, credits = context(tmp_path, monkeypatch, samples=4, world=3)
    c.distributed.all_gather_object = lambda value: [.5, 3., 1.1] if isinstance(value, float) else [value]
    class Finished(Exception):
        pass
    def inspect_phase(context, name, per_rank):
        credits.append(per_rank)
        raise Finished()
    monkeypatch.setattr(training, '_new_phase', inspect_phase)
    with pytest.raises(Finished):
        training._evaluate(c, 'initial')
    expected = [training.collection_credit(count*22, 10., latency)
                for count, latency in zip([12, 10, 10], [.5, 3., 1.1])]
    assert credits[0] == expected


def test_small_evaluation_allows_empty_ranks_and_merge_checks_active_ranks(tmp_path):
    catalog, loaded = Catalog(tmp_path), Policy()
    plan = build_balanced_plan(catalog, samples_per_source=1)
    assert [len(shard_plan(plan, rank, 8)['tasks']) for rank in range(8)] == [2, 2, 2, 2, 0, 0, 0, 0]
    reports = []
    for rank in range(4):
        directory = tmp_path/f'rank{rank}'
        evaluation.evaluate_policy(Env(loaded), loaded, shard_plan(plan, rank, 8), directory, catalog=catalog)
        reports.append(directory/'report.json')
    result = merge_periodic_reports(plan, reports)
    assert result['task_count'] == 8
    with pytest.raises(ValueError, match='rank'):
        merge_periodic_reports(plan, reports[:3])


def test_empty_rank_does_not_execute_policy(tmp_path, monkeypatch):
    c, _ = context(tmp_path, monkeypatch, samples=1, world=8)
    c.distributed.rank = 7
    def forbidden(*args, **kwargs):
        raise AssertionError('Empty evaluation shard executed Actor/physics')
    monkeypatch.setattr(training, 'evaluate_policy', forbidden)
    # 本替身无其他rank，merge应拒绝缺少完整报告；到达该错误前本rank不能执行策略。
    with pytest.raises(ValueError, match='no rank reports'):
        training._evaluate(c, 'initial')


def test_resume_best_pointer_prefers_durable_eval_and_ignores_superseded_tail(tmp_path):
    root = tmp_path/'run'
    root.mkdir()
    current = dict(iteration=300, plan_sha256='plan', evaluation_identity={'contract': 2})
    initial = dict(iteration=0, plan_sha256='plan', evaluation_identity={'contract': 2}, session_id='initial', score=0.)
    saved = root/'saved300.pt'
    saved.write_bytes(b'complete')
    durable = dict(initial, iteration=300, score=1., saved=True,
                   checkpoint={'iteration': 300, 'path': 'saved300.pt', 'sha256': 'a'*64})
    atomic_json(root/'best_saved.json', durable)
    assert load_best_pointers(root, current, state={'best_saved': None})['best_saved'] == durable
    stale = dict(initial, iteration=350, session_id='old', score=10.)
    atomic_json(root/'best_observed.json', stale)
    atomic_json(root/'superseded_tails'/'restore.json', dict(durable_iteration=300,
        previous_accepted={'iteration': 390, 'session_id': 'old'}))
    # 即使新session已再次走到400轮，旧session的350轮候选也不能复活。
    later = dict(current, iteration=400)
    result = load_best_pointers(root, later, state={'best_observed': initial})
    assert result['best_observed'] == initial and result['best_saved'] == durable


def test_mark_saved_rejects_wrong_actor_or_iteration(tmp_path, monkeypatch):
    c, _ = context(tmp_path, monkeypatch)
    first = training._evaluate(c, 'initial')
    checkpoint = c.output/'saved.pt'
    checkpoint.write_bytes(b'durable')
    descriptor = dict(iteration=100, path='saved.pt', sha256='a'*64)
    with pytest.raises(ValueError, match='iteration'):
        mark_saved_evaluation(first, first, descriptor, actor_fingerprint=evaluation._model_fingerprint(c.actor), output_dir=c.output)
    descriptor['iteration'] = 0
    with pytest.raises(ValueError, match='Actor'):
        mark_saved_evaluation(first, first, descriptor, actor_fingerprint='wrong', output_dir=c.output)


def test_fixed_critic_targets_preserve_bootstrap_and_models_are_readonly():
    critic = Critic()
    critic.train()
    row = SimpleNamespace(context={'x': torch.tensor([[2.]])}, next_context={'x': torch.tensor([[3.]])},
        identity={'backend_session_id': 'b', 'episode_id': 1, 'policy_version': 0}, rewards=[1.]*25,
        executed_control_steps=25, control_tick_begin=0, control_tick_end=25, terminated=False, truncated=True,
        transition_valid=True, metadata={'remaining_music_seconds': 2., 'next_remaining_music_seconds': 1.5})
    reference = build_fixed_critic_reference([row], critic, torch.device('cpu'), gamma_upper=.99, lambda_upper=.95)
    expected = sum(.99**(index/25) for index in range(25))+.99*3.
    assert reference['records'][0]['target'] == pytest.approx(expected)
    before = fixed_critic_diagnostic(critic, reference)
    assert before['explained_variance'] is None and critic.training
    assert before['mse'] == pytest.approx((2.-expected)**2)
    with torch.no_grad():
        critic.weight.add_(1.)
    after = fixed_critic_diagnostic(critic, reference)
    assert after['targets'] == before['targets'] and after['predictions'] == [3.]
    assert not hasattr(row, 'old_value') and critic.weight.grad is None


def test_rank_local_cuda_rng_never_uses_all_devices(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 3)
    monkeypatch.setattr(torch.cuda, 'get_rng_state', lambda device: calls.append(('get', device)) or torch.tensor([9], dtype=torch.uint8))
    monkeypatch.setattr(torch.cuda, 'set_rng_state', lambda state, device: calls.append(('set', device)))
    def forbidden(*args, **kwargs):
        raise AssertionError('rank evaluation touched all GPUs')
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', forbidden)
    monkeypatch.setattr(torch.cuda, 'set_rng_state_all', forbidden)
    state = evaluation._capture_rng(local_cuda_only=True)
    evaluation._restore_rng(state)
    assert calls == [('get', 3), ('set', 3)]
