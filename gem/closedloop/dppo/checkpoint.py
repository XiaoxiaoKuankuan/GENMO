"""当前完整checkpoint和Critic初始化共同使用的张量与优化器校验。

只验证参数名字、形状、dtype、有限值及Adam状态的参数映射，不读取或转换旧训练
格式。当前完整保存/恢复统一由training_checkpoint负责；Stage1权重由独立的
gem.closedloop.checkpoint读取，二者不会互相冒充训练恢复。
"""
from __future__ import annotations

import torch


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


def _validate_optimizer_state(optimizer, saved):
    """加载前检查Adam等优化器的参数映射与有限状态，禁止部分恢复后才发现损坏。"""
    if len(saved['param_groups'])!=len(optimizer.param_groups):
        raise ValueError('Optimizer state group count differs')
    allowed=set()
    for current,previous in zip(optimizer.param_groups,saved['param_groups']):
        for identity in ('name','param_names','lr_multiplier'):
            if current.get(identity)!=previous.get(identity):
                raise ValueError('Optimizer group identity differs: '+identity)
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
    if any('lr_multiplier' in group for group in optimizer.param_groups):
        from .actor_optimizer import actor_base_lr
        from types import SimpleNamespace
        actor_base_lr(SimpleNamespace(param_groups=saved['param_groups']))
