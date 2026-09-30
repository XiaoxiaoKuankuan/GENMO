"""第九步完整 checkpoint 的 CPU 恢复和拒绝错配测试。

使用两个独立小网络与真实 AdamW 状态验证模型、优化器、采样器、计数及 Python/NumPy/
Torch/具名 generator 的连续恢复。特别覆盖同形状参数交换顺序时不能静默错配 Adam
状态，以及未知RNG名称在模型发生修改之前被拒绝。所有文件只写 pytest 临时目录；
测试禁用 CUDA 探测，不初始化 GPU，不模拟或宣称恢复 PhysX 内部环境状态。
"""
from __future__ import annotations

import copy
import random

import numpy as np
import pytest
import torch

from gem.closedloop.dppo.checkpoint import capture_rng, load_checkpoint, restore_rng, save_checkpoint
from gem.closedloop.dppo.rewards import resolve_reward_config


class Sampler:
    def __init__(self):
        self.position = 3
    def state_dict(self):
        return {'position': self.position}
    def load_state_dict(self, state):
        self.position = state['position']


@pytest.fixture(autouse=True)
def cpu_rng_scope(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    original = capture_rng()
    yield
    restore_rng(original)


def _objects():
    actor = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    critic = torch.nn.Linear(2, 1)
    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=.002)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=.003)
    for module, optimizer in ((actor, actor_optimizer), (critic, critic_optimizer)):
        optimizer.zero_grad(set_to_none=True)
        module(torch.tensor([[.2, -.4]])).square().sum().backward()
        optimizer.step()
    return dict(actor=actor, critic=critic, actor_optimizer=actor_optimizer,
                critic_optimizer=critic_optimizer, identity={'asset_sha': 'test', 'policy': 'v1'},
                samplers={'music': Sampler()}, generators={'collector': torch.Generator().manual_seed(819)})


def _save(path, objects):
    save_checkpoint(path, **objects, state={'buffer_size': 0, 'pending_plan': False, 'policy_version': 4},
                    config={'test': 'cpu'})


def _draw(objects):
    return random.random(), float(np.random.random()), torch.rand(4), torch.rand(4, generator=objects['generators']['collector'])


def _equal(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            _equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            _equal(a, b)
    else:
        assert first == second


def test_full_state_rng_and_next_optimizer_step_replay(tmp_path):
    objects = _objects()
    path = tmp_path/'complete.pt'
    _save(path, objects)
    expected_draw = _draw(objects)
    expected_actor = copy.deepcopy(objects['actor'].state_dict())
    expected_adam = copy.deepcopy(objects['actor_optimizer'].state_dict())
    optimizer = objects['actor_optimizer']
    def advance():
        optimizer.zero_grad(set_to_none=True)
        objects['actor'](torch.tensor([[.3, .1]])).sum().backward()
        optimizer.step()
    advance()
    expected_next = copy.deepcopy(objects['actor'].state_dict())
    objects['samplers']['music'].position = 78
    state = load_checkpoint(path, **objects)
    assert state['policy_version'] == 4 and objects['samplers']['music'].position == 3
    _equal(expected_actor, objects['actor'].state_dict())
    _equal(expected_adam, objects['actor_optimizer'].state_dict())
    _equal(expected_draw, _draw(objects))
    advance()
    _equal(expected_next, objects['actor'].state_dict())


def test_optimizer_parameter_name_order_rejected_before_model_mutation(tmp_path):
    objects = _objects()
    path = tmp_path/'ordered.pt'
    _save(path, objects)
    with torch.no_grad():
        next(objects['actor'].parameters()).add_(3)
    original = copy.deepcopy(objects['actor'].state_dict())
    objects['actor_optimizer'] = torch.optim.AdamW(list(objects['actor'].parameters())[::-1], lr=.002)
    with pytest.raises(ValueError, match='parameter names'):
        load_checkpoint(path, **objects)
    _equal(original, objects['actor'].state_dict())


@pytest.mark.parametrize('mismatch', ['rng', 'sampler', 'identity'])
def test_named_states_and_identity_mismatch_rejected_before_loading(tmp_path, mismatch):
    objects = _objects()
    path = tmp_path/'identity.pt'
    _save(path, objects)
    with torch.no_grad():
        next(objects['actor'].parameters()).add_(1)
    original = copy.deepcopy(objects['actor'].state_dict())
    if mismatch == 'rng':
        objects['generators'] = {'wrong': torch.Generator()}
    elif mismatch == 'sampler':
        objects['samplers'] = {}
    else:
        objects['identity'] = {'asset_sha': 'other'}
    with pytest.raises(ValueError):
        load_checkpoint(path, **objects)
    _equal(original, objects['actor'].state_dict())


def test_checkpoint_requires_collection_boundary_and_cleans_failed_atomic_write(tmp_path, monkeypatch):
    objects = _objects()
    path = tmp_path/'atomic.pt'
    with pytest.raises(ValueError, match='empty buffer'):
        save_checkpoint(path, **objects, state={'buffer_size': 1}, config={})
    _save(path, objects)
    previous = path.read_bytes()
    def fail(*args, **kwargs):
        raise OSError('injected disk write failure')
    monkeypatch.setattr(torch, 'save', fail)
    with pytest.raises(OSError, match='injected'):
        _save(path, objects)
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize('mismatch', ['legacy_tracking', 'tracking_std', 'tracking_weights'])
def test_tracking_reward_identity_change_rejects_full_resume_before_any_state_mutation(tmp_path, mismatch):
    """Tracking 定义或参数变化必须形成新奖励身份，不能继续旧 Critic/Adam/GAE 语义。"""
    objects = _objects()
    current = resolve_reward_config()
    saved_reward = copy.deepcopy(current)
    if mismatch == 'legacy_tracking':
        saved_reward.pop('tracking')
        saved_reward['track_mix'] = dict(joint_pos=.45, joint_vel=.25, ee_height=.15,
                                         yaw=.10, root_position=.05)
        saved_reward['scales'].update(joint_pos_rad=.22, joint_vel_rad_s=1.4,
                                     ee_height_m=.07, yaw_rad=.6, root_position_m=.4)
    elif mismatch == 'tracking_std':
        saved_reward['tracking']['std']['joint_pos'] *= 2
    else:
        saved_reward['tracking']['weights']['joint_pos'] -= .1
        saved_reward['tracking']['weights']['joint_vel'] += .1
    objects['identity']['reward'] = saved_reward
    path = tmp_path / 'previous_tracking.pt'
    _save(path, objects)
    objects['identity']['reward'] = current
    with torch.no_grad():
        next(objects['actor'].parameters()).add_(5)
        next(objects['critic'].parameters()).sub_(3)
    objects['samplers']['music'].position = 77
    before = {name: copy.deepcopy(objects[name].state_dict())
              for name in ('actor', 'critic', 'actor_optimizer', 'critic_optimizer')}
    # 用抽样结果而非重新实现 RNG 编码，验证拒绝路径没有消耗任何独立随机流。
    rng_before = capture_rng(objects['generators'])
    expected_draw = _draw(objects)
    restore_rng(rng_before, objects['generators'])
    with pytest.raises(ValueError, match='identity mismatch'):
        load_checkpoint(path, **objects)
    for name, expected in before.items():
        _equal(expected, objects[name].state_dict())
    assert objects['samplers']['music'].position == 77
    _equal(expected_draw, _draw(objects))
