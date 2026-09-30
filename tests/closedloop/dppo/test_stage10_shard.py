"""独立评估分片入口的CPU生命周期、身份隔离和持久执行证据测试。

真实复用原Stage10生命周期夹具产生小型完整checkpoint，再由新增分片入口严格
恢复模型、优化器与RNG。GPU网络、完整数据加载和物理worker由合成夹具替代，
因此这里只证明入口没有优化、没有改写原身份，以及故障/信号不发布成功分片。
分片划分与严格合并在evaluation_shards专用测试中单独验证；本文件用明确的发布
桩隔离这部分，保留RunManager、预算、真实SQLite日志与checkpoint读写。
"""
from __future__ import annotations

import json
from pathlib import Path
import signal

import pytest
import torch

import test_stage10_lifecycle as lifecycle_fixtures
from test_stage10_lifecycle import latest, run
from gem.closedloop.dppo.buffer import StepJournal
from gem.closedloop.dppo.run_management import RunManager

lifecycle = lifecycle_fixtures.lifecycle
from tools import train_closedloop_stage10 as training
from tools.eval import run_closedloop_stage10_shard as shard


def reply(*, sequence=1, count=1, session='synthetic-worker'):
    return dict(backend_session_id=session, mutation_seq=sequence, operation='advance', ok=True,
        result=dict(executed_control_steps=count, executed_physics_steps=count * 4,
            physics_count_exact=True, transition_valid=True, control_tick_begin=600,
            control_tick_end=600 + 12 * count, episode_id='episode',
            trace=[dict(episode_id='episode', tick=600 + 12 * (index + 1)) for index in range(count)]))


def frozen(sequence=1, session='synthetic-worker'):
    return dict(policy_unchanged=True, runtime_parameters_unchanged=True,
        execution_journal=dict(backend_session_id=session, executed_seq=sequence,
                               acked_seq=sequence, outstanding_seq=None))


@pytest.fixture
def evaluation(lifecycle, tmp_path, monkeypatch):
    train_output = tmp_path/'training'
    assert run(lifecycle, train_output, stop=2) == 0
    checkpoint, descriptor = latest(train_output)
    lifecycle.checkpoint = train_output/descriptor['path']
    lifecycle.expected_identity = checkpoint['identity']
    lifecycle.expected_weight = checkpoint['actor']['weight'].clone()
    lifecycle.evaluation_calls = 0
    lifecycle.interrupt_evaluation = False
    lifecycle.published = []
    parent = dict(plan_sha256='parent', split='val', tasks=[dict(task_id='one'), dict(task_id='two')])
    child = dict(plan_sha256='child', split='val', tasks=[dict(task_id='one')])
    monkeypatch.setattr(training, 'build_evaluation_tasks', lambda *a, **k: parent)
    monkeypatch.setattr(shard, 'partition_evaluation_plan', lambda plan, index, count: child)

    def evaluate(env, policy, plan, output, *, actor_identity, frozen_modules, progress, **_):
        lifecycle.evaluation_calls += 1
        assert env.disk_guard is not None
        assert torch.equal(policy.actor.weight.detach(), lifecycle.expected_weight)
        assert actor_identity['sha256'] == training.sha256_file(lifecycle.checkpoint)
        assert actor_identity['iteration'] == 2
        progress(dict(event='evaluation_episode_start'))
        env.budget.reserve('evaluation', generations=1, control_steps=1, physics_steps=4)
        env.backend.sequence += 1
        env.backend.journal.append_result(reply(sequence=env.backend.sequence, session=env.backend.session_id))
        if lifecycle.interrupt_evaluation:
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            progress(dict(event='evaluation_episode_end'))
            progress(dict(event='evaluation_episode_start'))
        Path(output).mkdir(parents=True)
        result = dict(status='passed', actor_identity=actor_identity, plan_sha256=plan['plan_sha256'])
        training.atomic_json(Path(output)/'report.json', result)
        return result

    def publish(output, **kwargs):
        summary = json.loads(Path(kwargs['session_summary']).read_text())
        completion = json.loads(Path(kwargs['completion']).read_text())
        assert summary['status'] == completion['status'] == 'passed'
        assert completion['summary_sha256'] == training.sha256_file(kwargs['session_summary'])
        assert kwargs['training_identity'] == lifecycle.expected_identity
        assert summary['execution_journal_integrity']['complete']
        assert json.loads((Path(output)/'run.json').read_text())['identity'] == lifecycle.expected_identity
        lifecycle.published.append(summary)
        training.atomic_json(Path(output)/'shard_manifest.json', dict(synthetic=True, status='passed'))
        return Path(output)/'shard_manifest.json'

    monkeypatch.setattr(training, 'evaluate_policy', evaluate)
    monkeypatch.setattr(shard, 'publish_shard_manifest', publish)
    # 生命周期worker的冻结返回只用于训练；本分片须额外绑定真实落盘序号的worker UUID。
    original_backend = training.AcknowledgedBackend
    class Backend(original_backend):
        def call(self, method, **kwargs):
            value = super().call(method, **kwargs)
            value['execution_journal'].update(backend_session_id=self.session_id, outstanding_seq=None)
            return value
    monkeypatch.setattr(training, 'AcknowledgedBackend', Backend)
    return lifecycle


def evaluate_main(evaluation, output):
    return shard.main(['--config', str(evaluation.config_path), '--checkpoint', str(evaluation.checkpoint),
        '--output-dir', str(output), '--shard-index', '1', '--shard-count', '7'])


def test_loaded_checkpoint_is_evaluated_without_new_optimization_or_identity_change(evaluation, tmp_path):
    calls = evaluation.actor_calls
    output = tmp_path/'shard'
    assert evaluate_main(evaluation, output) == 0
    assert evaluation.actor_calls == calls and evaluation.evaluation_calls == 1
    summary = evaluation.published[0]
    assert summary['resume']['restored_full_state'] and not summary['resume']['training_resume']
    assert summary['initial_iteration'] == summary['final_iteration'] == 2
    assert summary['budget']['used'] == dict(accepted_iterations=0, optimizer_attempts=0,
        generations=1, control_steps=1, physics_steps=4)
    assert summary['extension_source_provenance']['included_in_original_checkpoint_identity'] is False
    assert summary['extension_source_unchanged']['unchanged']
    assert summary['resume']['checkpoint_unchanged_during_load']
    assert summary['evaluated_checkpoint_unchanged']
    assert (output/'shard_manifest.json').is_file()


def test_changed_original_identity_refuses_before_evaluation_worker(evaluation, tmp_path):
    before = evaluation.worker_starts
    Path(evaluation.config['paths']['stats']).write_text('changed asset identity')
    output = tmp_path/'shard'
    assert evaluate_main(evaluation, output) == 1
    assert evaluation.worker_starts == before and evaluation.evaluation_calls == 0
    assert not (output/'shard_manifest.json').exists()
    completion = json.loads(next(output.glob('sessions/*/completion.json')).read_text())
    assert completion['status'] == 'failed' and completion['exit_code'] == 1
    with RunManager(output, resume=True):
        pass


def test_checkpoint_selected_learning_rate_can_differ_from_initial_configuration(evaluation, tmp_path):
    saved = torch.load(evaluation.checkpoint, map_location='cpu', weights_only=False)
    saved['state']['selected_actor_lr'] = 3e-7
    for group in saved['actor_optimizer']['param_groups']:
        group['lr'] = 3e-7
    evaluation.checkpoint = tmp_path/'calibrated_actor_checkpoint.pt'
    torch.save(saved, evaluation.checkpoint)
    assert evaluate_main(evaluation, tmp_path/'shard') == 0
    assert evaluation.published[0]['resume']['actor_optimizer_lrs'] == [3e-7]
    assert evaluation.config['stage10']['training']['actor_lr'] == 1e-6


def test_checkpoint_optimizer_and_selected_learning_rate_mismatch_refuses_worker(evaluation, tmp_path):
    saved = torch.load(evaluation.checkpoint, map_location='cpu', weights_only=False)
    saved['state']['selected_actor_lr'] = 3e-7
    evaluation.checkpoint = tmp_path/'inconsistent_optimizer_checkpoint.pt'
    torch.save(saved, evaluation.checkpoint)
    before = evaluation.worker_starts
    output = tmp_path/'shard'
    assert evaluate_main(evaluation, output) == 1
    assert evaluation.worker_starts == before and evaluation.evaluation_calls == 0
    assert not (output/'shard_manifest.json').exists()


def test_checkpoint_changed_during_load_is_rejected_before_worker(evaluation, tmp_path, monkeypatch):
    original = training.load_checkpoint
    def load(*args, **kwargs):
        state = original(*args, **kwargs)
        Path(args[0]).write_bytes(b'synthetic replacement during load')
        return state
    monkeypatch.setattr(training, 'load_checkpoint', load)
    before = evaluation.worker_starts
    output = tmp_path/'shard'
    assert evaluate_main(evaluation, output) == 1
    assert evaluation.worker_starts == before and evaluation.evaluation_calls == 0
    assert not (output/'shard_manifest.json').exists()


def test_checkpoint_changed_during_evaluation_cannot_publish_shard(evaluation, tmp_path, monkeypatch):
    original = training.evaluate_policy
    def evaluate(*args, **kwargs):
        result = original(*args, **kwargs)
        evaluation.checkpoint.write_bytes(b'synthetic replacement during evaluation')
        return result
    monkeypatch.setattr(training, 'evaluate_policy', evaluate)
    output = tmp_path/'shard'
    assert evaluate_main(evaluation, output) == 1
    summary = json.loads(next(output.glob('sessions/*/summary.json')).read_text())
    assert not summary['evaluated_checkpoint_unchanged'] and summary['status'] == 'failed'
    assert not (output/'shard_manifest.json').exists()


def test_signal_stops_at_next_episode_boundary_without_success_publication(evaluation, tmp_path):
    before = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
    evaluation.interrupt_evaluation = True
    output = tmp_path/'shard'
    assert evaluate_main(evaluation, output) == 130
    assert not (output/'shard_manifest.json').exists()
    completion = json.loads(next(output.glob('sessions/*/completion.json')).read_text())
    assert completion['status'] == 'incomplete' and completion['exit_code'] == 130
    assert {number: signal.getsignal(number) for number in before} == before
    with RunManager(output, resume=True):
        pass


def test_session_end_failure_cannot_publish_completion_or_shard(evaluation, tmp_path, monkeypatch):
    original = RunManager.append_metrics
    def append(manager, value):
        if value.get('event') == 'session_end':
            raise OSError('synthetic session metrics failure')
        return original(manager, value)
    monkeypatch.setattr(RunManager, 'append_metrics', append)
    output = tmp_path/'shard'
    with pytest.raises(OSError, match='session metrics failure'):
        evaluate_main(evaluation, output)
    assert not list(output.glob('sessions/*/completion.json'))
    assert not (output/'shard_manifest.json').exists()
    with RunManager(output, resume=True):
        pass


def test_execution_journal_proves_exact_physics_and_ack_watermark(tmp_path):
    path = tmp_path/'execution.sqlite'
    with StepJournal(path) as journal:
        journal.append_result(reply(count=3))
    result = shard.verify_execution_journal(path, frozen())
    assert result['complete'] and result['executed_control_steps'] == 3
    assert result['executed_physics_steps'] == 12 and result['mutation_count'] == 1


@pytest.mark.parametrize('failure', ['trace', 'sequence', 'sha', 'ack'])
def test_execution_journal_rejects_missing_trace_sequence_sha_or_ack(tmp_path, failure):
    path = tmp_path/'execution.sqlite'
    value = reply(sequence=2 if failure == 'sequence' else 1)
    if failure == 'trace':
        value['result']['trace'] = []
    with StepJournal(path) as journal:
        journal.append_result(value)
        if failure == 'sha':
            journal.connection.execute('UPDATE replies SET sha256=?', ('0' * 64,))
            journal.connection.commit()
    state = frozen(sequence=2 if failure == 'sequence' else 1)
    if failure == 'ack':
        state['execution_journal']['acked_seq'] = 0
    with pytest.raises((ValueError, RuntimeError)):
        shard.verify_execution_journal(path, state)
