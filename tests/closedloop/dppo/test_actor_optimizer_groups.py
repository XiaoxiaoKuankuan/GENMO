"""分模块Actor优化器的实际Adam增量、事务与恢复测试。

构造含Transformer层、输出MLP及其他参数的小网络，先产生非空Adam状态，再与
独立直接Adam计算比较回溯结果。验证共同缩放、学习率增长、失败/异常原子恢复、
参数名字绑定及存盘恢复，不以仅执行optimizer.step作为有效更新证据。
所有文件位于pytest临时目录；此处的解析KL只检查事务，真实联合概率与物理效果
分别由既有完整链测试及服务器八卡实验验证，不将合成小网络冒充GENMO实验。
"""
import copy
import pytest
import torch

from gem.closedloop.dppo.actor_optimizer import (
    make_actor_optimizer, validate_actor_optimizer, actor_base_lr,
    actor_learning_rates, set_actor_base_lr)
from gem.closedloop.dppo.actor_backtracking import backtracked_step
from gem.closedloop.dppo.actor_step_diagnostics import StepDiagnostics
from gem.closedloop.dppo.checkpoint import _validate_optimizer_state
from gem.closedloop.dppo.gradient_merge import BCGradientController
from tests.closedloop.dppo.test_actor_backtracking import Collective, equal_state


def network():
    actor = torch.nn.Module()
    actor.denoiser = torch.nn.Module()
    actor.denoiser.blocks = torch.nn.ModuleList([torch.nn.Linear(3, 3) for _ in range(2)])
    actor.denoiser.final_layer = torch.nn.Linear(3, 2)
    actor.condition = torch.nn.Linear(2, 3)
    actor.frozen = torch.nn.Parameter(torch.ones(2), requires_grad=False)
    return actor


def settings():
    return dict(actor_lr=1e-6, actor_trainable_scope='all', actor_lr_multipliers=dict(
        transformer=.1, qpos_output=1., other=.1, transformer_overrides={'1': .025}))


def fill_gradients(actor, value):
    for parameter in actor.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.full_like(parameter, value)


def prepared():
    actor = network()
    optimizer = make_actor_optimizer(actor, settings())
    fill_gradients(actor, .7)
    optimizer.step()
    fill_gradients(actor, .2)
    return actor, optimizer, BCGradientController()


def test_group_backtrack_matches_direct_nonempty_adam_and_preserves_ratios():
    actor, optimizer, controller = prepared()
    reference = copy.deepcopy(actor)
    expected = make_actor_optimizer(reference, settings())
    expected.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    fill_gradients(reference, .2)
    set_actor_base_lr(expected, 2.5e-7)
    expected.step()
    rng = torch.get_rng_state().clone()
    def evaluate():
        torch.rand(7)
        return dict(chain_joint_kl=dict(mean=actor_base_lr(optimizer)),
                    active_clip_fraction=0.)
    result = backtracked_step(actor, optimizer, controller, evaluate=evaluate,
        limits=dict(chain_joint_mean=3e-7), distributed=Collective(), max_attempts=4)
    assert result['attempt_count'] == 3
    assert equal_state(actor.state_dict(), reference.state_dict())
    assert equal_state(optimizer.state_dict(), expected.state_dict())
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(float(state['step']) == 2 for state in optimizer.state.values())
    assert actor_learning_rates(optimizer) == pytest.approx(dict(
        transformer_00=2.5e-8, transformer_01=6.25e-9, qpos_output=2.5e-7, other=2.5e-8))
    set_actor_base_lr(optimizer, min(1e-6, actor_base_lr(optimizer) * 1.25))
    assert actor_base_lr(optimizer) == pytest.approx(3.125e-7)
    validate_actor_optimizer(actor, optimizer)


def test_explicit_fifty_times_lr_executes_adam_without_hidden_ceiling():
    actor, optimizer, _ = prepared()
    reference = copy.deepcopy(actor)
    expected = make_actor_optimizer(reference, settings())
    expected.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    fill_gradients(reference, .2)
    for group in expected.param_groups:
        group['lr'] = 1.5e-5 * group['lr_multiplier']
    expected.step()
    set_actor_base_lr(optimizer, 1.5e-5)
    validate_actor_optimizer(actor, optimizer)
    optimizer.step()
    assert actor_base_lr(optimizer) == pytest.approx(1.5e-5)
    assert equal_state(actor.state_dict(), reference.state_dict())
    assert equal_state(optimizer.state_dict(), expected.state_dict())
    assert all(float(state['step']) == 2 for state in optimizer.state.values())


@pytest.mark.parametrize('learning_rate', [0., -1e-5, float('nan'), float('inf')])
def test_explicit_lr_still_rejects_nonpositive_and_nonfinite(learning_rate):
    actor, optimizer, _ = prepared()
    with pytest.raises(ValueError, match='finite and positive'):
        set_actor_base_lr(optimizer, learning_rate)
    for group in optimizer.param_groups:
        group['lr'] = learning_rate
    with pytest.raises(ValueError, match='finite and positive'):
        validate_actor_optimizer(actor, optimizer)


@pytest.mark.parametrize('exception', [False, True])
def test_rejection_restores_all_groups_adam_controller_and_rng(exception):
    actor, optimizer, controller = prepared()
    state = copy.deepcopy((actor.state_dict(), optimizer.state_dict(), controller.state_dict()))
    rng = torch.get_rng_state().clone()
    def evaluate():
        torch.rand(5)
        controller.value = .2
        if exception:
            raise RuntimeError('injected multi-group failure')
        return dict(chain_joint_kl=dict(mean=1.))
    if exception:
        with pytest.raises(RuntimeError, match='multi-group'):
            backtracked_step(actor, optimizer, controller, evaluate=evaluate,
                limits=dict(chain_joint_mean=.1), distributed=Collective())
    else:
        result = backtracked_step(actor, optimizer, controller, evaluate=evaluate,
            limits=dict(chain_joint_mean=.1), distributed=Collective(), max_attempts=3)
        assert not result['accepted'] and result['attempt_count'] == 3
    assert equal_state(state, (actor.state_dict(), optimizer.state_dict(), controller.state_dict()))
    assert torch.equal(torch.get_rng_state(), rng)


def test_checkpoint_restores_group_binding_and_rejects_same_shape_parameter_swap(tmp_path):
    actor, optimizer, _ = prepared()
    set_actor_base_lr(optimizer, 5e-7)
    path = tmp_path / 'optimizer.pt'
    torch.save(optimizer.state_dict(), path)
    saved = torch.load(path, weights_only=False)
    restored = make_actor_optimizer(actor, settings())
    _validate_optimizer_state(restored, saved)
    restored.load_state_dict(saved)
    validate_actor_optimizer(actor, restored)
    assert equal_state(optimizer.state_dict(), restored.state_dict())
    broken = copy.deepcopy(saved)
    broken['param_groups'][0]['param_names'][0] = 'denoiser.blocks.1.weight'
    with pytest.raises(ValueError, match='identity'):
        _validate_optimizer_state(restored, broken)
    broken = copy.deepcopy(saved)
    broken['param_groups'][0]['lr'] *= 2
    with pytest.raises(ValueError, match='ratios'):
        _validate_optimizer_state(restored, broken)


def test_group_diagnostics_reconstruct_actual_adam_lr_and_detect_duplicate_or_frozen():
    actor, optimizer, _ = prepared()
    diagnostics = StepDiagnostics(actor)
    optimizer.step()
    result = diagnostics.finish(optimizer, .1, 1.)
    assert result['modules']['transformer_00']['changed_elements'] > 0
    assert result['modules']['transformer_01']['adam_subtraction_fp64_delta_rms'] < (
        result['modules']['transformer_00']['adam_subtraction_fp64_delta_rms'])
    assert result['learning_rates'] == actor_learning_rates(optimizer)
    optimizer.param_groups[0]['params'].append(actor.frozen)
    with pytest.raises(ValueError, match='frozen'):
        validate_actor_optimizer(actor, optimizer)
