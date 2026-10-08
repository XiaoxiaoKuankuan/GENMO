"""第九步训练的完整状态保存及严格恢复。

Stage1 checkpoint 只用于 Actor 权重初始化；本模块保存的 Stage9 文件才包含独立
Critic、两个优化器、策略版本、监督步数、任务采样器、预算快照和全部随机数状态。
v2 为多 rank 第二阶段保存一套共享模型/优化器和各 rank 独立执行、sampler、RNG；
每个进程只捕获自己的 CUDA 设备。完整恢复严格匹配 world_size/资产/参数映射，
评估使用单独的 weights-only 入口，不把加载权重声称为恢复训练状态。
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
VERSION_V2 = 'genmo.closedloop.stage10.full_state.v2'
FULL_STATE_VERSIONS = (VERSION, VERSION_V2)


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


def capture_rank_state(rank, *, state, samplers=None, generators=None):
    """只捕获当前 rank 的 CUDA 设备，避免每个进程初始化其他七张卡的 RNG。"""
    if type(rank) is not int or rank < 0:
        raise ValueError('Rank must be a nonnegative integer')
    _validate_boundary(state, explicit=True)
    cuda_device = torch.cuda.current_device() if torch.cuda.is_available() else None
    rng = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
               cuda=[] if cuda_device is None else [torch.cuda.get_rng_state(cuda_device)],
               cuda_device=cuda_device, cuda_scope='local_device',
               generators={name: generator.get_state() for name, generator in (generators or {}).items()},
               generator_devices={name: str(generator.device) for name, generator in (generators or {}).items()})
    return dict(rank=rank, state=copy.deepcopy(state), rng=rng,
                samplers={name: sampler.state_dict() for name, sampler in (samplers or {}).items()})


def _validate_rank_states(rank_states):
    if not isinstance(rank_states, (list, tuple)) or not rank_states:
        raise ValueError('V2 checkpoint requires every rank state')
    for rank, item in enumerate(rank_states):
        if not isinstance(item, dict) or type(item.get('rank')) is not int or item['rank'] != rank:
            raise ValueError('Rank states must contain each rank exactly once in rank order')
        if not {'state', 'rng', 'samplers'}.issubset(item) or not isinstance(item['samplers'], dict):
            raise ValueError('Incomplete rank state')
        _validate_boundary(item['state'], explicit=True)
        rng = item['rng']
        if rng.get('cuda_scope') != 'local_device':
            raise ValueError('V2 rank RNG must have local-device scope')
        device = rng.get('cuda_device')
        if (device is not None and (type(device) is not int or device < 0)) or len(rng['cuda']) != (0 if device is None else 1):
            raise ValueError('Invalid local CUDA RNG descriptor')
        random.Random().setstate(rng['python'])
        np.random.RandomState().set_state(rng['numpy'])
        torch.Generator().set_state(rng['torch'])
        if set(rng['generators']) != set(rng['generator_devices']):
            raise ValueError('Rank generator identities differ')
        for value in [*rng['cuda'], *rng['generators'].values()]:
            if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
                raise ValueError('Rank RNG states must be one-dimensional byte tensors')
    return len(rank_states)


def _validate_shared_rank_boundary(state, rank_states):
    for item in rank_states:
        for key in ('iteration', 'policy_version'):
            if key in state and item['state'].get(key) != state[key]:
                raise ValueError(f'Rank {key} differs from shared checkpoint boundary')


def _validate_rank_rng(rng, generators=None):
    if set(rng['generators']) != set(generators or {}):
        raise ValueError('Independent RNG names differ from checkpoint')
    if rng['generator_devices'] != {name: str(generator.device) for name, generator in (generators or {}).items()}:
        raise ValueError('Independent RNG devices differ from checkpoint')
    current = torch.cuda.current_device() if torch.cuda.is_available() else None
    if rng['cuda_device'] != current:
        raise RuntimeError('Local CUDA RNG topology differs from checkpoint')
    for name, generator in (generators or {}).items():
        torch.Generator(device=generator.device).set_state(rng['generators'][name])
    if current is not None:
        torch.Generator(device=f'cuda:{current}').set_state(rng['cuda'][0])


def _restore_rank_rng(rng, generators=None):
    random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch'])
    if rng['cuda_device'] is not None:
        torch.cuda.set_rng_state(rng['cuda'][0], rng['cuda_device'])
    for name, generator in (generators or {}).items():
        generator.set_state(rng['generators'][name])


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


def _validate_boundary(state, *, explicit=False):
    if explicit and (state.get('buffer_size') != 0 or state.get('pending_plan') is not False):
        raise ValueError('V2 checkpoint requires an explicit empty-buffer/no-pending boundary')
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
                    state, identity, config, samplers=None, generators=None,
                    version=VERSION, rank_states=None):
    if version not in FULL_STATE_VERSIONS:
        raise ValueError('Unsupported full-state checkpoint version')
    _validate_boundary(state, explicit=version == VERSION_V2)
    if version == VERSION_V2:
        world_size = _validate_rank_states(rank_states)
        _validate_shared_rank_boundary(state, rank_states)
        if samplers or generators:
            raise ValueError('V2 sampler/RNG state belongs in rank_states, not the shared state')
    elif rank_states is not None:
        raise ValueError('Rank states require checkpoint version V2')
    if {id(p) for p in actor.parameters()} & {id(p) for p in critic.parameters()}:
        raise ValueError('Actor and Critic must not share parameters')
    layout=dict(actor=_optimizer_layout(actor,actor_optimizer),critic=_optimizer_layout(critic,critic_optimizer))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(version=version, actor=actor.state_dict(), critic=critic.state_dict(),
                   actor_optimizer=actor_optimizer.state_dict(), critic_optimizer=critic_optimizer.state_dict(),
                   state=state, identity=identity, config=config,
                   optimizer_layout=layout,
                   restore_environment='new_worker_session_and_reset')
    if version == VERSION_V2:
        payload.update(world_size=world_size, rank_states=rank_states)
    else:
        payload.update(rng=capture_rng(generators),
                       samplers={name: sampler.state_dict() for name, sampler in (samplers or {}).items()})
    descriptor,temporary_name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    temporary=Path(temporary_name)
    try:
        with os.fdopen(descriptor,'wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary,path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def load_checkpoint(path, *, actor, critic, actor_optimizer, critic_optimizer,
                    identity, samplers=None, generators=None, rank=None, world_size=None):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    version = payload.get('version')
    if version not in FULL_STATE_VERSIONS or payload['identity'] != identity:
        raise ValueError('Stage9 checkpoint version or asset identity mismatch')
    if version == VERSION_V2:
        saved_world_size = _validate_rank_states(payload['rank_states'])
        _validate_shared_rank_boundary(payload['state'], payload['rank_states'])
        if (type(world_size) is not int or world_size != saved_world_size
                or payload.get('world_size') != saved_world_size
                or type(rank) is not int or not 0 <= rank < saved_world_size):
            raise ValueError('V2 full resume requires matching world size and a valid local rank')
        local = payload['rank_states'][rank]
    else:
        if rank is not None or world_size is not None:
            raise ValueError('V1 checkpoint cannot restore distributed rank state')
        local = payload
    if set(local['samplers']) != set(samplers or {}):
        raise ValueError('Sampler names differ from checkpoint')
    _validate_boundary(payload['state'], explicit=version == VERSION_V2)
    layout=dict(actor=_optimizer_layout(actor,actor_optimizer),critic=_optimizer_layout(critic,critic_optimizer))
    if payload.get('optimizer_layout')!=layout:
        raise ValueError('Optimizer parameter names, group order or class differ from checkpoint')
    _validate_model_state(actor,payload['actor'],'Actor')
    _validate_model_state(critic,payload['critic'],'Critic')
    _validate_optimizer_state(actor_optimizer,payload['actor_optimizer'])
    _validate_optimizer_state(critic_optimizer,payload['critic_optimizer'])
    if version == VERSION_V2:
        _validate_rank_rng(local['rng'], generators)
    else:
        _validate_rng(local['rng'], generators)
    # 先在独立副本检查 sampler 身份，拒绝错误恢复时不动模型、优化器或真实采样位置。
    for name, sampler in (samplers or {}).items():
        validator = getattr(sampler, 'validate_state_dict', None)
        if validator is not None:
            validator(copy.deepcopy(local['samplers'][name]))
        else:
            copy.deepcopy(sampler).load_state_dict(copy.deepcopy(local['samplers'][name]))
    actor.load_state_dict(payload['actor'], strict=True)
    critic.load_state_dict(payload['critic'], strict=True)
    actor_optimizer.load_state_dict(payload['actor_optimizer'])
    critic_optimizer.load_state_dict(payload['critic_optimizer'])
    for name, sampler in (samplers or {}).items():
        sampler.load_state_dict(local['samplers'][name])
    if version == VERSION_V2:
        _restore_rank_rng(local['rng'], generators)
    else:
        restore_rng(local['rng'], generators)
    result = copy.deepcopy(payload['state'])
    if version == VERSION_V2:
        result['local_rank_state'] = copy.deepcopy(local['state'])
    return result


def load_weights_checkpoint(path, *, actor, critic=None, identity=None):
    """显式评估入口：只校验并恢复权重，不恢复训练计数、优化器、sampler 或 RNG。"""
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('version') not in FULL_STATE_VERSIONS:
        raise ValueError('Unsupported evaluation checkpoint version')
    if identity is not None and payload.get('identity') != identity:
        raise ValueError('Evaluation checkpoint asset identity mismatch')
    _validate_model_state(actor, payload['actor'], 'Actor')
    if critic is not None:
        _validate_model_state(critic, payload['critic'], 'Critic')
    actor.load_state_dict(payload['actor'], strict=True)
    if critic is not None:
        critic.load_state_dict(payload['critic'], strict=True)
    return dict(version=payload['version'], state=copy.deepcopy(payload['state']),
                identity=copy.deepcopy(payload['identity']), restore_mode='weights_only',
                rank_execution_states=[copy.deepcopy(item['state']) for item in payload.get('rank_states', [])],
                optimizer_restored=False, rng_restored=False, samplers_restored=False)
