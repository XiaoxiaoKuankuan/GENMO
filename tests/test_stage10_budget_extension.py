"""第十步同一运行显式扩预算的CPU故障与恢复测试。

测试使用临时目录和真实预算账本，验证提高上限时原消耗及阶段计数不变，恢复只能沿
不可覆盖事件的SHA链继续；没有扩展证据、缩小上限、回退消耗、篡改事件均必须拒绝。
模拟“扩展事件落盘、预算替换失败”确保孤立事件不会授权扩限，重试保留旧证据。
测试不创建真实Actor/GMT、不占GPU，不删除任何正式checkpoint或训练数据。
"""
from __future__ import annotations

import copy
import json

import pytest

from gem.closedloop.dppo.run_management import RunManager, TrainingBudget, validate_budget_progress


def limits():
    return dict(accepted_iterations=6, optimizer_attempts=18, generations=1000,
                control_steps=10000, physics_steps=40000)


def expanded():
    return dict(accepted_iterations=40, optimizer_attempts=120, generations=8000,
                control_steps=200000, physics_steps=800000)


def extend(budget, values=None):
    return budget.extend_limits(expanded() if values is None else values,
        reason='完成准备验收，显式提高后续运行上限', checkpoint_sha256='a'*64, config_sha256='b'*64)


def test_expansion_preserves_spent_budget_and_resumes_with_full_evidence(tmp_path):
    budget = TrainingBudget(tmp_path/'budget.json', limits())
    budget.reserve('main', generations=4, control_steps=100, physics_steps=400)
    budget.reserve('update', optimizer_attempts=3, accepted_iterations=1)
    initial = budget.state_dict()
    descriptor = extend(budget)
    after = budget.state_dict()
    assert after['used'] == initial['used'] and after['phases'] == initial['phases']
    assert after['limit_extensions'] == [descriptor]
    assert json.loads((tmp_path/descriptor['path']).read_text())['checkpoint_sha256'] == 'a'*64
    restored = TrainingBudget(tmp_path/'budget.json', expanded())
    validate_budget_progress(initial, restored.state_dict())
    restored.reserve('update', optimizer_attempts=3, accepted_iterations=1)
    second = dict(expanded(), accepted_iterations=41, optimizer_attempts=123)
    extend(restored, second)
    final = TrainingBudget(tmp_path/'budget.json', second).state_dict()
    validate_budget_progress(initial, final)
    validate_budget_progress(after, final)
    assert final['used']['accepted_iterations'] == 2
    assert len(final['limit_extensions']) == 2


@pytest.mark.parametrize('change', [dict(accepted_iterations=5), dict(physics_steps=39999), {}])
def test_extension_cannot_shrink_or_leave_limits_unchanged(tmp_path, change):
    budget = TrainingBudget(tmp_path/'budget.json', limits())
    before = budget.path.read_bytes()
    with pytest.raises(ValueError):
        extend(budget, dict(limits(), **change))
    assert budget.path.read_bytes() == before
    assert not (tmp_path/'budget_extensions').exists()


@pytest.mark.parametrize('arguments', [dict(reason=' '), dict(checkpoint_sha256='short'), dict(config_sha256=None)])
def test_extension_requires_explicit_reason_and_identity(tmp_path, arguments):
    budget = TrainingBudget(tmp_path/'budget.json', limits())
    kwargs = dict(reason='next segment', checkpoint_sha256='a'*64, config_sha256='b'*64)
    kwargs.update(arguments)
    with pytest.raises(ValueError):
        budget.extend_limits(expanded(), **kwargs)
    assert not (tmp_path/'budget_extensions').exists()


def test_orphan_extension_after_atomic_budget_failure_does_not_authorize_limits(tmp_path, monkeypatch):
    budget = TrainingBudget(tmp_path/'budget.json', limits())
    budget.reserve('main', generations=1)
    before = budget.state_dict()
    with monkeypatch.context() as patch:
        patch.setattr(budget, '_save', lambda _: (_ for _ in ()).throw(OSError('budget replace failed')))
        with pytest.raises(OSError, match='replace failed'):
            extend(budget)
    assert budget.state_dict() == before
    assert TrainingBudget(budget.path, limits()).state_dict() == before
    with pytest.raises(ValueError, match='cannot silently change'):
        TrainingBudget(budget.path, expanded())
    orphan = list((tmp_path/'budget_extensions').glob('*.json'))
    assert len(orphan) == 1
    orphan_bytes = orphan[0].read_bytes()
    extend(budget)
    assert len(list((tmp_path/'budget_extensions').glob('*.json'))) == 2
    assert len(TrainingBudget(budget.path, expanded()).state_dict()['limit_extensions']) == 1
    assert orphan[0].read_bytes() == orphan_bytes


def test_changed_event_fails_sha_before_resume(tmp_path):
    budget = TrainingBudget(tmp_path/'budget.json', limits())
    entry = extend(budget)
    path = tmp_path/entry['path']
    payload = json.loads(path.read_text())
    payload['reason'] = 'tampered'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='SHA mismatch'):
        TrainingBudget(budget.path, expanded())


def test_resume_rejects_unrecorded_limits_replaced_history_and_spent_rollback(tmp_path):
    budget = TrainingBudget(tmp_path/'budget.json', limits())
    budget.reserve('main', generations=5)
    initial = budget.state_dict()
    fake = copy.deepcopy(initial)
    fake['limits'] = expanded()
    with pytest.raises(ValueError, match='without budget extension evidence'):
        validate_budget_progress(initial, fake)
    extend(budget)
    expanded_state = budget.state_dict()
    fake = copy.deepcopy(expanded_state)
    fake['used']['generations'] = 4
    with pytest.raises(ValueError, match='roll back spent'):
        validate_budget_progress(initial, fake)
    fake = copy.deepcopy(expanded_state)
    fake['limit_extensions'][0]['sha256'] = 'c'*64
    with pytest.raises(ValueError, match='extension history'):
        validate_budget_progress(expanded_state, fake)


def test_manager_extension_mode_is_read_only_until_verified_expansion(tmp_path):
    run = tmp_path/'run'
    with RunManager(run) as manager:
        original = manager.budget(limits()).state_dict()
    with RunManager(run, resume=True) as manager:
        budget = manager.budget(expanded(), for_extension=True)
        assert budget.state_dict() == original
        assert json.loads((run/'budget.json').read_text()) == original
        extend(budget)
    with RunManager(run, resume=True) as manager:
        assert manager.budget(expanded()).state_dict()['used'] == original['used']


def test_iteration_capacity_stops_before_collection_without_charging(tmp_path):
    budget = TrainingBudget(tmp_path/'budget.json', limits())
    assert budget.iteration_capacity(3)['can_start']
    budget.reserve('update', optimizer_attempts=16)
    before = budget.path.read_bytes()
    assert budget.iteration_capacity(3) == dict(can_start=False,
        exhausted={'optimizer_attempts': dict(required=3, remaining=2)})
    assert budget.path.read_bytes() == before
    budget.reserve('update', accepted_iterations=6)
    assert set(budget.iteration_capacity(3)['exhausted']) == {'optimizer_attempts', 'accepted_iterations'}


def test_extension_descriptor_and_limit_chain_tampering_is_rejected(tmp_path):
    budget = TrainingBudget(tmp_path/'budget.json', limits())
    extend(budget)
    state = budget.state_dict()
    state['limits']['accepted_iterations'] += 1
    budget.path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match='extension chain'):
        TrainingBudget(budget.path, state['limits'])
