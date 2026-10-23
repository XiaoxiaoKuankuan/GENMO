"""Stage2 Actor分模块AdamW及同比例学习率控制。

输入为Stage1原架构中已经标记可训练的参数和显式倍率配置。Transformer按真实
denoiser.blocks层号分组，原qpos输出MLP单独分组，其余允许训练的参数归入other。
不新增网络、不改梯度、不改Gaussian核；每组实际学习率等于共同基础LR乘固定倍率。
完整链KL回溯及恢复增长只改变共同基础LR，避免单组回退破坏相对更新方向。

参数名称、组名与倍率进入Adam状态和配置身份，加载前核对，不把单组旧Adam迁移
为分组Adam。每个可训练参数必须且只能出现一次，冻结参数不得进入优化器。
未声明倍率的既有实验仍为一个Adam组；这不改变其概率或参数更新数学定义。
本模块只构造及校验优化器，不选择候选、不启动训练、不判断物理收益。
"""
from __future__ import annotations

import math
import torch


def actor_parameter_group(name):
    if name.startswith('denoiser.blocks.'):
        return 'transformer_' + name.split('.')[2].zfill(2)
    if name.startswith('denoiser.final_layer.'):
        return 'qpos_output'
    return 'other'


def validate_lr_multipliers(settings):
    values = settings.get('actor_lr_multipliers')
    if values is None:
        return None
    if (settings.get('actor_trainable_scope', 'all') != 'all'
            or set(values) != {'transformer', 'qpos_output', 'other', 'transformer_overrides'}):
        raise ValueError('Module LR requires all Actor parameters and explicit fixed multipliers')
    for name in ('transformer', 'qpos_output', 'other'):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError('Actor LR multiplier must be finite in (0,1]')
    overrides = values['transformer_overrides']
    if not isinstance(overrides, dict):
        raise ValueError('Transformer LR overrides must map canonical layer indices to multipliers')
    for layer, value in overrides.items():
        if (not isinstance(layer, str) or not layer.isdigit() or str(int(layer)) != layer
                or not 0 <= int(layer) < 16 or isinstance(value, bool)
                or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1):
            raise ValueError('Invalid Transformer layer LR override')
    return values


def make_actor_optimizer(actor, settings):
    values = validate_lr_multipliers(settings)
    named = [(n, p) for n, p in actor.named_parameters() if p.requires_grad]
    if values is None:
        optimizer = torch.optim.AdamW([p for _, p in named], lr=settings['actor_lr'], weight_decay=0.)
    else:
        groups = {}
        for name, parameter in named:
            key = actor_parameter_group(name)
            if key not in groups:
                multiplier = (values['transformer_overrides'].get(str(int(key.rsplit('_', 1)[1])),
                    values['transformer']) if key.startswith('transformer_') else values[key])
                groups[key] = dict(name=key, lr_multiplier=multiplier,
                    lr=settings['actor_lr'] * multiplier, params=[], param_names=[])
            groups[key]['params'].append(parameter)
            groups[key]['param_names'].append(name)
        for layer in values['transformer_overrides']:
            if 'transformer_' + layer.zfill(2) not in groups:
                raise ValueError('Transformer LR override refers to an absent trainable layer')
        optimizer = torch.optim.AdamW(list(groups.values()), lr=settings['actor_lr'], weight_decay=0.)
    validate_actor_optimizer(actor, optimizer)
    return optimizer


def actor_base_lr(optimizer):
    rates = [float(group['lr']) / float(group.get('lr_multiplier', 1.)) for group in optimizer.param_groups]
    if not rates or not all(math.isfinite(lr) and lr > 0 for lr in rates):
        raise ValueError('Explicit experimental Actor base LR must be finite and positive')
    if any(not math.isclose(lr, rates[0], rel_tol=1e-12, abs_tol=0.) for lr in rates):
        raise ValueError('Actor parameter groups lost their fixed LR ratios')
    return rates[0]


def set_actor_base_lr(optimizer, learning_rate):
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('Explicit experimental Actor base LR must be finite and positive')
    for group in optimizer.param_groups:
        group['lr'] = learning_rate * group.get('lr_multiplier', 1.)


def actor_learning_rates(optimizer):
    return {group.get('name', 'uniform'): float(group['lr']) for group in optimizer.param_groups}


def validate_actor_optimizer(actor, optimizer):
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError('Actor must use AdamW')
    expected = {id(p): n for n, p in actor.named_parameters() if p.requires_grad}
    actual, names = [], set()
    for group in optimizer.param_groups:
        if len(optimizer.param_groups) > 1:
            if not {'name', 'lr_multiplier', 'param_names'} <= group.keys():
                raise ValueError('Multiple Actor groups require explicit names and fixed LR multipliers')
            if group['name'] in names:
                raise ValueError('Actor optimizer group names must be unique')
            names.add(group['name'])
        multiplier = group.get('lr_multiplier', 1.)
        if not math.isfinite(multiplier) or not 0 < multiplier <= 1:
            raise ValueError('Invalid optimizer LR multiplier')
        group_names = [expected.get(id(p)) for p in group['params']]
        if 'param_names' in group and group['param_names'] != group_names:
            raise ValueError('Actor group parameter names differ or include frozen parameters')
        actual.extend(id(p) for p in group['params'])
        if any(p.dtype != torch.float32 for p in group['params']):
            raise ValueError('FP32 master parameters required')
        if getattr(actor, '_stage2_row_gradient_hooks', []) and group['weight_decay'] != 0:
            raise ValueError('Row-frozen Actor parameters require AdamW weight_decay=0')
    if set(actual) != set(expected) or len(actual) != len(expected):
        raise ValueError('AdamW must contain every trainable Actor parameter exactly once and no frozen parameters')
    actor_base_lr(optimizer)
