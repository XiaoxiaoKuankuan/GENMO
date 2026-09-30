"""第九步训练的完整状态保存及严格恢复。

Stage1 checkpoint 只用于 Actor 权重初始化；本模块保存的 Stage9 文件才包含独立
Critic、两个优化器、策略版本、监督步数、任务采样器、预算快照和全部随机数状态。
恢复发生在对象构造之后，且只能从合法采集边界开始新 worker session；不伪装成恢复
PhysX 内部状态，不复用 pending plan 或旧 on-policy Buffer。写入使用同目录原子替换。
"""
from __future__ import annotations

import copy
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

VERSION = 'genmo.closedloop.stage9.full_state.v1'


def capture_rng(generators=None):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                generators={k: g.get_state() for k, g in (generators or {}).items()},
                generator_devices={k:str(g.device) for k,g in (generators or {}).items()})


def _validate_rng(state, generators=None):
    """先检查拓扑与状态格式，避免恢复一半才发现命名RNG不匹配。"""
    if set(state['generators']) != set(generators or {}):
        raise ValueError('Independent RNG names differ from checkpoint')
    if state['generator_devices'] != {k:str(g.device) for k,g in (generators or {}).items()}:
        raise ValueError('Independent RNG devices differ from checkpoint')
    cuda_count=torch.cuda.device_count() if torch.cuda.is_available() else 0
    if len(state['cuda'])!=cuda_count:
        raise RuntimeError('CUDA RNG topology differs from checkpoint')
    random.Random().setstate(state['python'])
    np.random.RandomState().set_state(state['numpy'])
    torch.Generator().set_state(state['torch'])
    for name, generator in (generators or {}).items():
        torch.Generator(device=generator.device).set_state(state['generators'][name])


def restore_rng(state, generators=None):
    _validate_rng(state,generators)
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])
    for name, generator in (generators or {}).items():
        generator.set_state(state['generators'][name])


def _optimizer_layout(model, optimizer):
    """用参数名字绑定组及组内顺序，防止Adam状态被错配到同形状的其他权重。"""
    names={id(parameter):name for name,parameter in model.named_parameters()}
    seen=set(); groups=[]
    for group in optimizer.param_groups:
        entries=[]
        for parameter in group['params']:
            key=id(parameter)
            if key not in names or key in seen:
                raise ValueError('Optimizer contains foreign or duplicate parameters')
            seen.add(key);entries.append(names[key])
        groups.append(entries)
    return dict(optimizer_class=f'{type(optimizer).__module__}.{type(optimizer).__qualname__}',
                parameter_groups=groups)


def _validate_model_state(model, state, label):
    current=model.state_dict()
    if set(current)!=set(state):
        raise ValueError(f'{label} checkpoint parameter names differ')
    for name,value in current.items():
        saved=state[name]
        if not isinstance(saved,torch.Tensor) or saved.shape!=value.shape or saved.dtype!=value.dtype:
            raise ValueError(f'{label} checkpoint shape/dtype differs: {name}')
        if (saved.is_floating_point() or saved.is_complex()) and not torch.isfinite(saved).all():
            raise ValueError(f'{label} checkpoint contains nonfinite parameter: {name}')


def _validate_boundary(state):
    if state.get('pending_plan') or state.get('buffer_size', 0):
        raise ValueError('Checkpoint requires an empty buffer and no pending plan')


def _validate_optimizer_state(optimizer, saved):
    """加载前检查Adam等优化器的参数映射与有限状态，禁止部分恢复后才发现损坏。"""
    if len(saved['param_groups'])!=len(optimizer.param_groups):
        raise ValueError('Optimizer state group count differs')
    allowed=set()
    for current,previous in zip(optimizer.param_groups,saved['param_groups']):
        if len(current['params'])!=len(previous['params']):
            raise ValueError('Optimizer state parameter count differs')
        for parameter,key in zip(current['params'],previous['params']):
            if key in allowed:
                raise ValueError('Optimizer state contains duplicate parameter keys')
            allowed.add(key)
            for field,value in saved['state'].get(key,{}).items():
                if isinstance(value,torch.Tensor):
                    if not torch.isfinite(value).all():
                        raise ValueError(f'Optimizer state contains nonfinite {field}')
                    if field in ('exp_avg','exp_avg_sq','max_exp_avg_sq','momentum_buffer') and value.shape!=parameter.shape:
                        raise ValueError(f'Optimizer state tensor shape differs: {field}')
    if not set(saved['state']).issubset(allowed):
        raise ValueError('Optimizer state refers to unknown parameter keys')


def save_checkpoint(path, *, actor, critic, actor_optimizer, critic_optimizer,
                    state, identity, config, samplers=None, generators=None):
    _validate_boundary(state)
    if {id(p) for p in actor.parameters()} & {id(p) for p in critic.parameters()}:
        raise ValueError('Actor and Critic must not share parameters')
    layout=dict(actor=_optimizer_layout(actor,actor_optimizer),critic=_optimizer_layout(critic,critic_optimizer))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(version=VERSION, actor=actor.state_dict(), critic=critic.state_dict(),
                   actor_optimizer=actor_optimizer.state_dict(), critic_optimizer=critic_optimizer.state_dict(),
                   state=state, identity=identity, config=config, rng=capture_rng(generators),
                   samplers={k: sampler.state_dict() for k, sampler in (samplers or {}).items()},
                   optimizer_layout=layout,
                   restore_environment='new_worker_session_and_reset')
    descriptor,temporary_name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    temporary=Path(temporary_name)
    try:
        with os.fdopen(descriptor,'wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def load_checkpoint(path, *, actor, critic, actor_optimizer, critic_optimizer,
                    identity, samplers=None, generators=None):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('version') != VERSION or payload['identity'] != identity:
        raise ValueError('Stage9 checkpoint version or asset identity mismatch')
    if set(payload['samplers']) != set(samplers or {}):
        raise ValueError('Sampler names differ from checkpoint')
    _validate_boundary(payload['state'])
    layout=dict(actor=_optimizer_layout(actor,actor_optimizer),critic=_optimizer_layout(critic,critic_optimizer))
    if payload.get('optimizer_layout')!=layout:
        raise ValueError('Optimizer parameter names, group order or class differ from checkpoint')
    _validate_model_state(actor,payload['actor'],'Actor')
    _validate_model_state(critic,payload['critic'],'Critic')
    _validate_optimizer_state(actor_optimizer,payload['actor_optimizer'])
    _validate_optimizer_state(critic_optimizer,payload['critic_optimizer'])
    _validate_rng(payload['rng'],generators)
    actor.load_state_dict(payload['actor'], strict=True)
    critic.load_state_dict(payload['critic'], strict=True)
    actor_optimizer.load_state_dict(payload['actor_optimizer'])
    critic_optimizer.load_state_dict(payload['critic_optimizer'])
    for name, sampler in (samplers or {}).items():
        sampler.load_state_dict(payload['samplers'][name])
    restore_rng(payload['rng'], generators)
    return copy.deepcopy(payload['state'])
