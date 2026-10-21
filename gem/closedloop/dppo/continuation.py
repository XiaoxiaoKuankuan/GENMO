"""DPPO 更换奖励协议时的显式学习状态继承与关闭 BC 的空接口。

输入必须是原 DPPO 完整 checkpoint，核验文件 SHA、Stage1/GMT 资产、15+4+1
采样核和实际 Actor/Adam 结构后，只继承 Actor、Actor Adam、学习率调度和累计
Actor 计数。旧 Critic、价值标签、BC 状态、物理中间态和旧预算不进入新实验；
新目录以新奖励重新采完整 MC 并拟合 Critic，父实验只读且保留可追溯身份。
普通 resume 仍由完整 checkpoint 加载器处理，不把协议迁移冒充精确恢复。

DisabledBC 不读取 Stage1 数据、不执行网络前反向或推进随机游标；它只满足
现有事务/快照的状态接口，任何意外 BC 调用都会报错，避免仅系数为零仍训练。
"""
from __future__ import annotations

import copy
from pathlib import Path
import torch

from .checkpoint import _validate_model_state, _validate_optimizer_state
from .evaluation import _model_fingerprint
from .training_checkpoint import SCHEMA, file_sha256


class DisabledBC:
    def state_dict(self):
        return {'schema': 'genmo.stage2.bc_disabled.v1'}

    def load_state_dict(self, state):
        if state != self.state_dict():
            raise ValueError('Disabled BC cannot restore an active BC sampler')

    def backward(self, *args, **kwargs):
        raise RuntimeError('BC is disabled: no supervised forward/backward is allowed')


def initialize_actor_continuation(spec, *, actor, optimizer, lr_controller, assets, kernel,
                                  restore_learning=True):
    if set(spec) != {'path', 'sha256', 'mode'} or spec['mode'] != 'actor_adam_new_reward':
        raise ValueError('Explicit Actor/Adam continuation into a new reward experiment required')
    path = Path(spec['path']).resolve(strict=True)
    if file_sha256(path) != spec['sha256']:
        raise ValueError('Parent DPPO checkpoint SHA differs')
    payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    identity = payload.get('identity', {})
    if payload.get('schema') != SCHEMA or identity.get('assets') != assets or identity.get('kernel') != kernel:
        raise ValueError('Parent DPPO schema, assets or 15+4+1 behavior kernel differs')
    _validate_model_state(actor, payload['actor'], '继承 DPPO Actor')
    _validate_optimizer_state(optimizer, payload['actor_optimizer'])
    copy.deepcopy(lr_controller).load_state_dict(payload['scheduler'])
    counters = {key: payload['state'][key] for key in
                ('iteration', 'rollouts_completed', 'actor_updates', 'policy_version')}
    if any(type(value) is not int or value < 0 for value in counters.values()):
        raise ValueError('Invalid parent Actor progress')
    if restore_learning:
        actor.load_state_dict(payload['actor'], strict=True)
        if _model_fingerprint(actor) != payload['actor_fingerprint']:
            raise ValueError('Parent Actor tensor fingerprint differs')
        optimizer.load_state_dict(payload['actor_optimizer'])
        lr_controller.load_state_dict(payload['scheduler'])
    return dict(kind='dppo_actor_adam_new_reward', path=str(path), sha256=spec['sha256'],
                actor_fingerprint=payload['actor_fingerprint'], parent_progress=counters,
                inherited_actor_adam_states=len(payload['actor_optimizer']['state']),
                parent_reward=identity['reward'], parent_termination=identity['physical_configuration']['termination'],
                critic_restored=False, bc_restored=False, physical_world_restored=False,
                old_reward_labels_reused=False, old_budget_imported=False)
