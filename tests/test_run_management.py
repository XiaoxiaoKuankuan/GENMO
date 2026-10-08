"""第十步运行管理的CPU故障测试。

测试在pytest临时目录内创建真实UpperTransition、SQLite执行日志与小型完整checkpoint，
验证预算断点续用、未知执行不退款、逐条写入且不能覆盖、SHA清单与latest发布原子性、
运行锁、磁盘配额、session隔离和安全停止信号。测试不构建真实GENMO/GMT，不启动
GPU或物理仿真；所有产物由临时目录负责清理，不改动正式训练数据。
"""
from __future__ import annotations

import json
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest
import torch

from gem.closedloop.dppo.buffer import UpperTransition
from gem.closedloop.dppo.checkpoint import save_checkpoint
from gem.closedloop.dppo import run_management as management
from gem.closedloop.dppo.run_management import (
    BudgetExceeded, DiskCapacityError, DiskGuard, GuardedStepJournal, RolloutWriter,
    RunManager, StopSignal, TrainingBudget, file_sha256,
)


def limits():
    return dict(accepted_iterations=4, optimizer_attempts=12, generations=100,
                control_steps=200, physics_steps=800)


def context(tick):
    time = tick / 600
    return dict(music_features=torch.zeros(1, 120, 35), music_valid=torch.ones(1, 120, dtype=torch.bool),
        proprio_history=torch.zeros(1, 50, 48), proprio_history_valid=torch.ones(1, 50, dtype=torch.bool),
        proprio_history_times=((torch.arange(50, dtype=torch.float64) - 49) / 50 + time)[None],
        known_qpos30=torch.zeros(1, 120, 30), known_qpos30_mask=torch.zeros(1, 120, 30, dtype=torch.bool),
        future_valid=torch.ones(1, 120, dtype=torch.bool),
        future_times=(torch.arange(120, dtype=torch.float64) / 30 + time)[None],
        decision_time=torch.tensor([time], dtype=torch.float64))


def transition(index, policy_version=0):
    return UpperTransition(identity=dict(run_id='run', backend_session_id='s', episode_id='e',
        decision_id=index, policy_version=policy_version), context=context(index * 12),
        next_context=context((index + 1) * 12), chain=torch.zeros(2, 120, 30),
        old_log_prob=torch.zeros(1, dtype=torch.float64), free_mask=torch.ones(120, 30, dtype=torch.bool),
        rewards=torch.tensor([.02], dtype=torch.float64), old_value=0., next_value=0.,
        control_tick_begin=index * 12, control_tick_end=(index + 1) * 12,
        executed_control_steps=1, executed_physics_steps=4)


def complete_checkpoint(path, iteration):
    actor, critic = torch.nn.Linear(2, 2), torch.nn.Linear(2, 1)
    save_checkpoint(path, actor=actor, critic=critic,
        actor_optimizer=torch.optim.AdamW(actor.parameters()), critic_optimizer=torch.optim.AdamW(critic.parameters()),
        state=dict(iteration=iteration, buffer_size=0, pending_plan=False), identity={}, config={})
    return path


def test_budget_persists_attempts_separately_and_refunds_only_known_execution(tmp_path):
    path = tmp_path / 'budget.json'
    budget = TrainingBudget(path, limits())
    budget.reserve('main', generations=1, control_steps=25, physics_steps=100)
    budget.settle_control('main', 25, dict(executed_control_steps=20, executed_physics_steps=82, physics_count_exact=True))
    budget.reserve('update', optimizer_attempts=3)
    budget.accept_iteration()
    persisted = TrainingBudget(path, limits())
    assert persisted.state_dict()['used'] == dict(accepted_iterations=1, optimizer_attempts=3,
        generations=1, control_steps=20, physics_steps=82)
    before = path.read_bytes()
    persisted.settle_control('main', 5, dict(physics_count_exact=False))
    assert path.read_bytes() == before
    with pytest.raises(BudgetExceeded):
        persisted.reserve('update', optimizer_attempts=10)
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match='cannot silently change'):
        TrainingBudget(path, dict(limits(), generations=101))


@pytest.mark.parametrize('amount', [-1, True, 1.5])
def test_invalid_budget_reservation_does_not_change_disk(tmp_path, amount):
    path = tmp_path / 'budget.json'
    budget = TrainingBudget(path, limits())
    before = path.read_bytes()
    with pytest.raises(ValueError):
        budget.reserve('main', generations=amount)
    assert path.read_bytes() == before


def test_budget_write_failure_leaves_the_previous_state(tmp_path, monkeypatch):
    budget = TrainingBudget(tmp_path / 'budget.json', limits())
    before = budget.state_dict()
    monkeypatch.setattr(management, '_atomic_json', lambda *a, **k: (_ for _ in ()).throw(OSError('disk failure')))
    with pytest.raises(OSError, match='disk failure'):
        budget.reserve('main', generations=1)
    assert budget.state_dict() == before
    assert json.loads(budget.path.read_text()) == before


def test_full_control_settlement_validates_without_copy_or_write(tmp_path, monkeypatch):
    budget = TrainingBudget(tmp_path/'budget.json', limits())
    budget.reserve('main', control_steps=25, physics_steps=100)
    before = budget.path.read_bytes()
    monkeypatch.setattr(budget, '_save', lambda *_: pytest.fail('Zero refund must not write'))
    monkeypatch.setattr(management.copy, 'deepcopy', lambda *_: pytest.fail('Zero refund must not copy history'))
    budget.settle_control('main', 25, dict(executed_control_steps=25, executed_physics_steps=100))
    assert budget.path.read_bytes() == before
    with pytest.raises(ValueError, match='reserved'):
        budget.settle_control('unknown', 25, dict(executed_control_steps=25, executed_physics_steps=100))
    with pytest.raises(ValueError, match='inconsistent'):
        budget.settle_control('main', 25, dict(executed_control_steps=25, executed_physics_steps=99))


def test_guarded_journal_encodes_once_and_keeps_original_payload_sha(tmp_path, monkeypatch):
    import hashlib
    import numpy as np
    from gem.closedloop.dppo import buffer as buffer_module
    reply = dict(backend_session_id='会话', mutation_seq=1, result=dict(
        array=np.arange(24, dtype=np.float32).reshape(3, 8), nonfinite=float('nan')))
    original = buffer_module._journal_value
    expected = json.dumps(original(reply), ensure_ascii=False, sort_keys=True,
                          separators=(',', ':'), allow_nan=False)
    calls = []
    def counted(value):
        if value is reply:
            calls.append(1)
        return original(value)
    monkeypatch.setattr(buffer_module, '_journal_value', counted)
    guard = DiskGuard(tmp_path, max_run_bytes=10**8, min_free_bytes=0)
    journal = GuardedStepJournal(tmp_path/'execution.sqlite', guard)
    try:
        assert journal.append_result(reply)
        assert calls == [1]
        payload, digest = journal._journal.connection.execute('SELECT payload, sha256 FROM replies').fetchone()
        assert payload == expected and digest == hashlib.sha256(expected.encode()).hexdigest()
        assert not journal.append_result(reply)
    finally:
        journal.close()


def test_corrupt_budget_totals_and_unknown_counters_are_rejected(tmp_path):
    path = tmp_path / 'budget.json'
    budget = TrainingBudget(path, limits())
    with pytest.raises(ValueError, match='unknown'):
        budget.reserve('update', iterations=1)
    payload = budget.state_dict()
    payload['used']['generations'] = 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='totals'):
        TrainingBudget(path, limits())


def test_rollout_append_is_immutable_chunked_and_sha_verified(tmp_path):
    writer = RolloutWriter(tmp_path / 'rollout', policy_version=0, chunk_size=2)
    first = writer.append(transition(0))
    before = (first.read_bytes(), first.stat().st_mtime_ns)
    writer.append(transition(1))
    writer.append(transition(2))
    assert (first.read_bytes(), first.stat().st_mtime_ns) == before
    with pytest.raises(ValueError, match='duplicate'):
        writer.append(transition(1))
    with pytest.raises(ValueError, match='policy version'):
        writer.append(transition(3, policy_version=1))
    manifest_path = writer.finish()
    manifest = json.loads(manifest_path.read_text())
    assert manifest['complete'] and manifest['transition_count'] == 3
    assert manifest['executed_control_steps'] == 3 and manifest['executed_physics_steps'] == 12
    assert [chunk['record_count'] for chunk in manifest['chunks']] == [2, 1]
    for chunk in manifest['chunks']:
        path = writer.directory / chunk['path']
        assert file_sha256(path) == chunk['sha256']
        for record in json.loads(path.read_text())['records']:
            record_path = path.parent / record['path']
            assert file_sha256(record_path) == record['sha256']
            value = torch.load(record_path, weights_only=False, map_location='cpu')
            assert isinstance(value, UpperTransition)
            value.validate()
    with pytest.raises(RuntimeError, match='completed rollout'):
        writer.append(transition(3))
    with pytest.raises(RuntimeError, match='already been published'):
        writer.finish()


def test_interrupted_rollout_is_not_published_or_reopened_for_overwrite(tmp_path):
    path = tmp_path / 'rollout'
    writer = RolloutWriter(path, policy_version=0)
    record = writer.append(transition(0))
    assert record.is_file() and not (path / 'manifest.json').exists()
    with pytest.raises(FileExistsError):
        RolloutWriter(path, policy_version=0)
    invalid = transition(1)
    invalid.transition_valid = False
    with pytest.raises(ValueError, match='valid transitions'):
        writer.append(invalid)


def test_rollout_final_manifest_failure_preserves_records(tmp_path, monkeypatch):
    writer = RolloutWriter(tmp_path / 'rollout', policy_version=0)
    record = writer.append(transition(0))
    real = management._atomic_json
    def fail_final(path, value, **kwargs):
        if Path(path) == writer.directory / 'manifest.json':
            raise OSError('final manifest failure')
        return real(path, value, **kwargs)
    monkeypatch.setattr(management, '_atomic_json', fail_final)
    with pytest.raises(OSError, match='final manifest failure'):
        writer.finish()
    assert record.is_file() and not (writer.directory / 'manifest.json').exists()
    monkeypatch.setattr(management, '_atomic_json', real)
    assert writer.finish().is_file()


def test_disk_guard_rejects_reserve_and_run_quota_without_deleting_evidence(tmp_path, monkeypatch):
    path = tmp_path / 'evidence.bin'
    path.write_bytes(b'x' * 100)
    guard = DiskGuard(tmp_path, min_free_bytes=50, max_run_bytes=200)
    monkeypatch.setattr(management.shutil, 'disk_usage', lambda path: SimpleNamespace(free=100))
    with pytest.raises(DiskCapacityError, match='free space'):
        guard.check(51)
    monkeypatch.setattr(management.shutil, 'disk_usage', lambda path: SimpleNamespace(free=1000))
    with pytest.raises(DiskCapacityError, match='byte quota'):
        guard.check(101)
    path.write_bytes(b'y' * 150)
    guard.account_file(path)
    assert guard.used_bytes == 150
    assert guard.check(50)['used_bytes'] == 150
    assert path.read_bytes() == b'y' * 150


def test_rollout_disk_preflight_fails_before_serialization(tmp_path, monkeypatch):
    guard = DiskGuard(tmp_path, max_run_bytes=1000)
    writer = RolloutWriter(tmp_path / 'rollout', policy_version=0, disk_guard=guard)
    monkeypatch.setattr(torch, 'save', lambda *a, **k: (_ for _ in ()).throw(AssertionError('must not serialize')))
    with pytest.raises(DiskCapacityError):
        writer.append(transition(0))
    assert not list(writer.directory.rglob('*.pt'))


def test_run_lock_resume_attempts_and_session_metrics_are_preserved(tmp_path):
    run = tmp_path / 'run'
    with RunManager(run) as manager:
        manager.budget(limits()).reserve('main', generations=1)
        first = manager.iteration_dir(1)
        metrics = manager.append_metrics(dict(event='iteration_start', iteration=1))
        old_session = manager.session_id
        assert manager.iteration_dir(1) == first
        with pytest.raises(RuntimeError, match='another training writer'):
            RunManager(run, resume=True)
        with pytest.raises(ValueError, match='plain identifier'):
            manager.iteration_dir(2, '../escape')
    with pytest.raises(RuntimeError, match='closed'):
        manager.append_metrics(dict(event='illegal write'))
    with pytest.raises(FileExistsError, match='empty'):
        RunManager(run)
    # 已故障session的半行原样保留；恢复session单独追加。
    with metrics.open('ab') as stream:
        stream.write(b'{"interrupted":')
    old_bytes = metrics.read_bytes()
    with RunManager(run, resume=True) as resumed:
        assert resumed.session_id != old_session
        assert resumed.iteration_dir(1) != first
        assert resumed.budget(limits()).state_dict()['used']['generations'] == 1
        assert resumed.append_metrics(dict(event='recovered')) != metrics
    assert metrics.read_bytes() == old_bytes


def test_guarded_journal_keeps_duplicate_reply_semantics_and_one_file_per_attempt(tmp_path):
    with RunManager(tmp_path / 'run') as manager:
        journal = manager.journal(1)
        reply = dict(backend_session_id='session', mutation_seq=1, result=dict(episode_id='episode'))
        assert journal.append_result(reply)
        assert not journal.append_result(reply)
        assert len(journal) == 1
        with pytest.raises(ValueError, match='different payload'):
            journal.append_result(dict(reply, result=dict(episode_id='other')))
        with pytest.raises(FileExistsError):
            manager.journal(1)
        second = manager.journal(2)
        assert second.path != journal.path
        assert second.append_result(dict(backend_session_id='session', mutation_seq=2, result={}))
        assert second.path.is_file()
        journal.close()


def test_checkpoint_latest_only_advances_after_complete_atomic_publication(tmp_path, monkeypatch):
    with RunManager(tmp_path / 'run') as manager:
        budget = manager.budget(limits())
        budget.accept_iteration()
        first = complete_checkpoint(manager.run_dir / 'checkpoints' / 'first.pt', 1)
        manager.publish_checkpoint(1, first, dict(validation='passed'))
        before = (manager.run_dir / 'latest.json').read_bytes()
        assert manager.latest_checkpoint() == first
        budget.accept_iteration()
        second = complete_checkpoint(manager.run_dir / 'checkpoints' / 'second.pt', 2)
        real = management._atomic_json
        def fail_latest(path, value, **kwargs):
            if Path(path).name == 'latest.json':
                raise OSError('latest publication failed')
            return real(path, value, **kwargs)
        monkeypatch.setattr(management, '_atomic_json', fail_latest)
        with pytest.raises(OSError, match='latest publication failed'):
            manager.publish_checkpoint(2, second)
        assert (manager.run_dir / 'latest.json').read_bytes() == before
        assert manager.latest_checkpoint() == first
        assert second.is_file() and budget.state_dict()['used']['accepted_iterations'] == 2
        # 不覆盖刚才已落盘的发布描述；新attempt/session恢复可重试同一checkpoint。
        monkeypatch.setattr(management, '_atomic_json', real)
    with RunManager(tmp_path / 'run', resume=True) as resumed:
        resumed.budget(limits())
        resumed.publish_checkpoint(2, second)
        assert resumed.latest_checkpoint() == second
        with pytest.raises(ValueError, match='advance'):
            resumed.publish_checkpoint(2, second)
        with second.open('ab') as stream:
            stream.write(b'corrupt')
        with pytest.raises(ValueError, match='SHA'):
            resumed.latest_checkpoint()


def test_latest_rejects_weights_only_pending_or_external_checkpoints(tmp_path):
    with RunManager(tmp_path / 'run') as manager:
        path = manager.run_dir / 'weights.pt'
        torch.save(dict(actor={}), path)
        with pytest.raises(ValueError, match='complete Actor/Critic'):
            manager.publish_checkpoint(1, path)
        full = complete_checkpoint(manager.run_dir / 'full.pt', 1)
        payload = torch.load(full, weights_only=False)
        payload['state']['pending_plan'] = True
        torch.save(payload, full)
        with pytest.raises(ValueError, match='no-pending'):
            manager.publish_checkpoint(1, full)
        external = complete_checkpoint(tmp_path / 'external.pt', 1)
        with pytest.raises(ValueError, match='inside this run'):
            manager.publish_checkpoint(1, external)
        assert not (manager.run_dir / 'latest.json').exists()


def test_signal_requests_stop_without_throwing_and_restores_handlers():
    previous = {value: signal.getsignal(value) for value in (signal.SIGINT, signal.SIGTERM)}
    with StopSignal() as stop:
        assert not stop.stop_requested
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        assert stop.stop_requested and stop.signal_number == signal.SIGTERM
    assert {value: signal.getsignal(value) for value in previous} == previous


def test_disk_daily_checks_do_not_rescan_history_or_allow_external_evidence(tmp_path, monkeypatch):
    run = tmp_path / 'run'
    run.mkdir()
    guard = DiskGuard(run, max_run_bytes=1000000)
    monkeypatch.setattr(management.os, 'walk', lambda *a, **k: (_ for _ in ()).throw(AssertionError('history rescan')))
    path = run / 'new.bin'
    path.write_bytes(b'abc')
    guard.account_file(path)
    assert guard.check()['used_bytes'] == 3
    with pytest.raises(ValueError, match='inside this run'):
        RolloutWriter(tmp_path / 'outside', policy_version=0, disk_guard=guard)
    with pytest.raises(ValueError, match='inside this run'):
        GuardedStepJournal(tmp_path / 'outside.sqlite', guard)
    assert not (tmp_path / 'outside').exists() and not (tmp_path / 'outside.sqlite').exists()


def test_metrics_reject_nonfinite_values_before_appending(tmp_path):
    with RunManager(tmp_path / 'run') as manager:
        path = manager.append_metrics(dict(event='good', value=1.))
        before = path.read_bytes()
        with pytest.raises(ValueError):
            manager.append_metrics(dict(event='bad', value=float('nan')))
        assert path.read_bytes() == before


def test_latest_pointer_cannot_silently_diverge_from_publication(tmp_path):
    with RunManager(tmp_path / 'run') as manager:
        full = complete_checkpoint(manager.run_dir / 'full.pt', 1)
        manager.publish_checkpoint(1, full)
        path = manager.run_dir / 'latest.json'
        value = json.loads(path.read_text())
        value['iteration'] = 9
        path.write_text(json.dumps(value))
        with pytest.raises(ValueError, match='immutable checkpoint publication'):
            manager.latest_checkpoint()
