"""源码记录不再阻塞恢复、固定评估及配对统计的回归。

使用实际加载器恢复含非空Adam和BC状态的小模型；调用真实固定评估入口，复现
旧参考携带旧源码表的情形，并在物理阶段开始处以替身结束。另用既有真实小型
评估报告检查配对统计、TensorBoard和最佳指针读取。源码修改/缺失必须通过，
模型形状、实际奖励、采样核和执行条件不兼容仍须拒绝，避免把去掉源码门禁误写
成绕过概率或张量校验。文件仅写pytest临时目录，不执行正式PhysX或改写旧模型。
显式新实验允许重新绑定初始化来源，并逐张量检查两模型、非空Adam和BC原样恢复；
普通resume仍拒绝来源变化，新实验仍拒绝kernel、reward或学习率的未授权变化。
"""
import copy
import json
from types import SimpleNamespace as NS

import pytest
import torch

from gem.closedloop.dppo.execution_compatibility import same_execution, contains_execution
from gem.closedloop.dppo.gradient_merge import BCGradientController
from gem.closedloop.dppo.training_checkpoint import SCHEMA, load_training_checkpoint


def test_nested_source_records_are_optional_and_input_is_unchanged():
    before = {'training_identity': {'kernel': {'eta': .1}, 'source_files': {'genmo': {'a': 'old'}}}}
    after = copy.deepcopy(before)
    after['training_identity']['source_files'] = {'gmt': {'new': 'new'}}
    original = copy.deepcopy(before)
    assert same_execution(before, after)
    del after['training_identity']['source_files']
    assert same_execution(before, after)
    assert before == original
    after['training_identity']['kernel']['eta'] = .2
    assert not same_execution(before, after)
    assert contains_execution({'reward': 1, 'other': 2}, {'reward': 1, 'source_files': {'a': 'sha'}})
    assert not contains_execution({'reward': 2}, {'reward': 1})


@pytest.mark.parametrize('change', ['source', 'missing_source', 'kernel', 'shape'])
def test_resume_preserves_real_model_adam_and_bc_without_source_gate(tmp_path, change):
    actor, critic = torch.nn.Linear(2, 2), torch.nn.Linear(2, 1)
    aopt, copt = torch.optim.AdamW(actor.parameters()), torch.optim.AdamW(critic.parameters())
    for model, optimizer in ((actor, aopt), (critic, copt)):
        model(torch.ones(2, 2)).sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    control = BCGradientController()
    control.update(2., 1.)
    old = dict(kernel={'eta': .1}, source_files={'genmo': {'updater.py': 'old'}})
    payload = copy.deepcopy(dict(schema=SCHEMA, identity=old, actor=actor.state_dict(), critic=critic.state_dict(),
        actor_optimizer=aopt.state_dict(), critic_optimizer=copt.state_dict(), bc_controller=control.state_dict()))
    target = copy.deepcopy(old)
    target['source_files'] = {'genmo': {'updater.py': 'new'}}
    if change == 'missing_source':
        del payload['identity']['source_files']
    if change == 'kernel':
        target['kernel']['eta'] = .2
    if change == 'shape':
        payload['actor']['weight'] = torch.ones(3, 2)
    path = tmp_path/'saved.pt'
    torch.save(payload, path)
    with torch.no_grad():
        actor.weight.add_(10.)
    prior = actor.weight.clone()
    aopt.state.clear()
    control.value = control.minimum
    kwargs = dict(identity=target, actor=actor, critic=critic, actor_optimizer=aopt,
                  critic_optimizer=copt, controller=control)
    if change in ('kernel', 'shape'):
        with pytest.raises(ValueError):
            load_training_checkpoint(path, **kwargs)
        assert torch.equal(actor.weight, prior)
        assert not aopt.state
    else:
        loaded = load_training_checkpoint(path, **kwargs)
        assert torch.equal(actor.weight, payload['actor']['weight'])
        assert aopt.state and control.value == payload['bc_controller']['value']
        assert loaded['identity'] == payload['identity']  # 原记录未被盖上当前源码。


@pytest.mark.parametrize('change', ['provenance', 'resume', 'kernel', 'reward', 'actor_lr'])
def test_new_experiment_rebinds_only_provenance_and_preserves_learning_state(tmp_path, change):
    from tests.closedloop.dppo.test_actor_lr_initialization import _equal, _networks, _snapshot

    source = _networks(1e-7)
    for name in ('actor', 'critic'):
        optimizer = source[name + '_optimizer']
        source[name](torch.ones(2, 3)).square().sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    source['controller'].update(2., 1.)
    old = dict(actor_initialization={'kind': 'explicit_checkpoint', 'checkpoint_sha256': 'previous'},
               kernel={'eta': .1}, reward={'track_weight': 2.5}, training={'actor_lr': 1e-7})
    payload = dict(schema=SCHEMA, identity=copy.deepcopy(old), **_snapshot(source))
    payload['bc_controller'] = payload.pop('controller')
    path = tmp_path / 'nonempty_adam.pt'
    torch.save(payload, path)
    target = _networks(1e-7)
    before = _snapshot(target)
    identity = copy.deepcopy(old)
    identity['actor_initialization'] = {'kind': 'source_stage1', 'source_global_step': 595000}
    if change == 'kernel':
        identity['kernel']['eta'] = .2
    elif change == 'reward':
        identity['reward']['track_weight'] = 1.
    elif change == 'actor_lr':
        identity['training']['actor_lr'] = 1e-6
    original_identity = copy.deepcopy(identity)
    if change == 'provenance':
        loaded = load_training_checkpoint(path, identity=identity, **target, new_experiment=True)
        for name in ('actor', 'critic', 'actor_optimizer', 'critic_optimizer'):
            _equal(target[name].state_dict(), payload[name])
        _equal(target['controller'].state_dict(), payload['bc_controller'])
        for name in ('actor_optimizer', 'critic_optimizer'):
            assert target[name].state
            assert all(state['step'] == 1 for state in target[name].state.values())
        assert loaded['identity'] == old
    else:
        with pytest.raises(ValueError, match='actual training configuration differs'):
            load_training_checkpoint(path, identity=identity, **target, new_experiment=change != 'resume')
        _equal(_snapshot(target), before)
    assert identity == original_identity
    assert torch.load(path, weights_only=False)['identity'] == old


@pytest.mark.parametrize('changed_reward', [False, True])
def test_actual_periodic_entry_passes_old_source_reference(tmp_path, monkeypatch, changed_reward):
    from gem.closedloop.dppo import evaluation_runtime as runtime
    from gem.closedloop.dppo.periodic_monitor import build_balanced_plan
    from tests.closedloop.dppo.test_periodic_monitor import Catalog
    catalog = Catalog(tmp_path)
    options = dict(samples_per_source=1, seeds=[42, 1729], selection_seed=42, episode_seconds=10.,
                   start_mode='beginning', unique_music=False, split='val')
    plan = build_balanced_plan(catalog, **{k: v for k, v in options.items() if k != 'episode_seconds'}, seconds=10.)
    old = dict(reward={'track_weight': 2.5}, source_files={'genmo': {'vector_generation.py': 'old'}})
    new = copy.deepcopy(old)
    new['source_files'] = {'genmo': {'vector_generation.py': 'new'}}
    if changed_reward:
        new['reward']['track_weight'] = 1.
    recorded = dict(schema='genmo.closedloop.stage10.periodic_execution.v3', training_identity=old,
        timing_contract='test', kernel={'eta': .1}, execution_profile=None,
        critical_latency_budgets=[.32]*8, episode_seconds=10., rng_scope='current_cuda_device')
    (tmp_path/'fixed_diagnostic_reference.json').write_text(json.dumps(
        dict(plan_sha256=plan['plan_sha256'], evaluation_identity=recorded)))
    c = NS(generators={}, backend=NS(journal=None), catalog=catalog, stage={'evaluation': options},
        distributed=NS(world_size=8, rank=0, all_gather_object=lambda value: [value]*8),
        env=NS(latency_budget_s=.32), config={'runtime': {'timing_contract': 'test'}},
        identity=new, policy=NS(kernel_config={'eta': .1}), state={}, output=tmp_path)
    monkeypatch.setattr(runtime, 'capture_local_rng', lambda generators: {})
    monkeypatch.setattr(runtime, 'root_call', lambda distributed, fn: fn())

    class PhysicalPhaseReached(Exception):
        pass

    def begin(*args):
        raise PhysicalPhaseReached

    monkeypatch.setattr(runtime, '_new_phase', begin)
    if changed_reward:
        with pytest.raises(ValueError, match='evaluation contract'):
            runtime._evaluate_impl(c, 'periodic_50')
    else:
        with pytest.raises(PhysicalPhaseReached):
            runtime._evaluate_impl(c, 'periodic_50')


def test_paired_statistics_dashboard_and_best_pointer_accept_source_changes(tmp_path):
    from gem.closedloop.dppo.periodic_monitor import merge_periodic_reports, load_best_pointers
    from gem.closedloop.dppo.tensorboard_metrics import evaluation_scalars
    from tests.closedloop.dppo.test_periodic_monitor import evaluate_ranks
    from tools.eval.compare_closedloop_stage10 import compare_evaluations
    plan, paths = evaluate_ranks(tmp_path)
    before = merge_periodic_reports(plan, paths, evaluation_identity={'reward': 1, 'source_files': {'a': 'old'}})
    after = merge_periodic_reports(plan, paths, baseline=before,
        evaluation_identity={'reward': 1, 'source_files': {'a': 'new'}})
    assert after['paired_baseline']['new_failures'] == 0
    assert evaluation_scalars(after, before, bootstrap_repeats=20)['Reward/paired_gain_per_second'] == 0.
    parent = copy.deepcopy(before)
    parent.update(iteration=0, saved=False)
    (tmp_path/'best_observed.json').write_text(json.dumps(parent))
    after['iteration'] = 1
    assert load_best_pointers(tmp_path, after)['best_observed'] is not None
    left, right = tmp_path/'before.json', tmp_path/'after.json'
    left.write_text(json.dumps(before))
    right.write_text(json.dumps(after))
    assert compare_evaluations(left, right)['paired_task_count'] == plan['task_count']
