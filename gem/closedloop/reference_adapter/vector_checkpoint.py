"""独立向量实验的完整checkpoint和可恢复状态，不复制只读歌曲库。

保存三个网络/优化器、训练归一化、动态Replay、切向真实数据、音乐/残差历史、
独立采样RNG和accepted计数。每个文件使用唯一名称并原子创建，来源SHA只用于追溯。
恢复不能还原求解器隐藏状态，因此保留学习回放但必须从新的真实episode开始。
"""

import random
import time
from pathlib import Path
import numpy as np
import torch

from supermimic.bridge.bundle import atomic_save
from .checkpoint import shared_cpu_state

SCHEMA = 'genmo.adapter.vector.independent.v1'


def save_vector(path, engine, critic, optimizers, replay, tangent, collector, state, config, validation_replay=None, extra_state=None):
    payload = dict(schema=SCHEMA, adapter=engine.step.adapter.state_dict(), dynamics=engine.dynamics.state_dict(),
        critic=critic.state_dict(), optimizers={k: v.state_dict() for k, v in optimizers.items()},
        frozen_bundle=engine.step.bundle.state_dict(), input_spec=engine.step.adapter.input_spec,
        music_s0=engine.step.rewards.music.s0, config=config, state=state,
        replay=replay.state_dict(), tangent=tangent.data, tangent_total=tangent.total_pairs,
        validation_replay=validation_replay.state_dict() if validation_replay is not None else None,
        collector_state=collector.state, collector_episode_serial=collector.episode_serial,
        collector_real_transitions=collector.real_transitions, collector_resets=collector.reset_count,
        collector_real_by_role=getattr(collector, 'real_by_role', {}),
        collector_vector_steps=collector.vector_steps, collector_rng=collector.generator.get_state(),
        rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                 cuda=torch.cuda.get_rng_state_all()),
        extra_state=extra_state,
        task_contracts=collector.bank.task_contracts if config.get('task_sampling') else None,
        physics_resume='new_episode_required_solver_hidden_state_not_saved')
    atomic_save(shared_cpu_state(payload), Path(path))


def load_vector(path, engine, critic, optimizers, replay, tangent, collector, config, validation_replay=None):
    ck = torch.load(path, weights_only=False, map_location='cpu')
    if ck.get('schema') != SCHEMA or ck['input_spec'] != engine.step.adapter.input_spec:
        raise ValueError('不是相同十帧Adapter完整向量checkpoint')
    for key in ('music', 'rewards', 'reference_steps', 'control_hz'):
        if ck['config'][key] != config[key]:
            raise ValueError('恢复的奖励或时钟合同不同')
    if ck['config'].get('sm_mdp', {}) != config.get('sm_mdp', {}):
        raise ValueError('恢复的SM误差尺度或真实终止条件不同，不能沿用旧奖励Replay')
    if config.get('task_sampling') and ck.get('task_contracts') != collector.bank.task_contracts:
        raise ValueError('恢复的联合条件/Stage1名义缓存不同，旧Replay索引不再代表相同任务')
    for k, v in engine.step.bundle.state_dict().items():
        if not torch.equal(v.cpu(), ck['frozen_bundle'][k]):
            raise ValueError('冻结资产发生变化')
    if not torch.equal(ck['music_s0'], engine.step.rewards.music.s0.cpu()):
        raise ValueError('音乐固定标尺不一致')
    engine.step.adapter.load_state_dict(ck['adapter']); engine.dynamics.load_state_dict(ck['dynamics'])
    critic.load_state_dict(ck['critic'])
    for k, opt in optimizers.items():
        opt.load_state_dict(ck['optimizers'][k])
    replay.load_state_dict(ck['replay'])
    tangent.data = {k: v.to(collector.device) for k, v in ck['tangent'].items()}
    tangent.total_pairs = ck['tangent_total']
    if ck['validation_replay'] is not None and validation_replay is not None:
        validation_replay.load_state_dict(ck['validation_replay'])
    collector.episode_serial = ck['collector_episode_serial']
    collector.real_transitions = ck['collector_real_transitions']; collector.reset_count = ck['collector_resets']
    collector.vector_steps = ck['collector_vector_steps']; collector.generator.set_state(ck['collector_rng'])
    collector.real_by_role = ck.get('collector_real_by_role', {})
    random.setstate(ck['rng']['python']); np.random.set_state(ck['rng']['numpy']); torch.set_rng_state(ck['rng']['torch'])
    torch.cuda.set_rng_state_all(ck['rng']['cuda'])
    engine.dynamics.model_version = ck['state']['wm_updates']
    collector.reset(split='train', random_start=True, perturb=True)
    return ck['state']
