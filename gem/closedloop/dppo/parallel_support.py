"""第二阶段八卡独立采集的配置、同步、预算租约和张量辅助接口。

本模块只组织既有 UpperTransition，不另建训练数据格式。各卡保留完整随机链，
仅交换小型样本索引、统计量和必要恢复状态；梯度已经使用全局实际样本数作为分母，
通信必须使用 SUM。资源在物理执行之前由唯一根进程预占，各卡另写持久化使用账本；
所有执行都确认完成后才能退回明确未用的额度，进程失败或执行未知时不退款。

原始条件、旧概率和旧价值在同一 rollout 内不可变；可学习条件编码不能跨参数更新
复用。这里还提供一次整轮权威快照和无磁盘的张量状态广播，供 KL 拒绝时恢复模型和
Adam 状态。当前训练的规模、延迟与检查频率由 training_config 统一解析。
"""
from __future__ import annotations

import copy
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .run_management import TrainingBudget




def root_call(collective, function):
    """根进程异常也广播，避免其他 rank 永久等待一个不会再到达的结果。"""
    result = None
    if collective.rank == 0:
        try:
            result = dict(ok=True, value=function())
        except Exception as error:
            result = dict(ok=False, error=f'{type(error).__name__}: {error}')
    result = collective.broadcast_object(result)
    if not result['ok']:
        raise RuntimeError(result['error'])
    return result['value']


def local_call(collective, function):
    """无内部 collective 的本地阶段结束后统一检查；失败数据不进入训练。"""
    value, failure = None, None
    try:
        value = function()
    except Exception as error:
        failure = f'rank {collective.rank}: {type(error).__name__}: {error}'
    failures = collective.all_gather_object(failure)
    if any(failures):
        raise RuntimeError('; '.join(item for item in failures if item))
    return value


def build_global_manifest(rows, collective):
    from .executed_action_mask import action_mask
    local = [dict(owner_rank=collective.rank, local_index=i,
                  valid=bool(row.transition_valid), has_free=bool(action_mask(row).any()))
             for i, row in enumerate(rows)]
    return [entry for shard in collective.all_gather_object(local) for entry in shard]


def capture_local_rng(generators=None):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                generators={k: g.get_state() for k, g in (generators or {}).items()})


def restore_local_rng(value, generators=None):
    random.setstate(value['python']); np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch'])
    if value['cuda'] is not None:
        torch.cuda.set_rng_state(value['cuda'])
    for key, generator in (generators or {}).items():
        generator.set_state(value['generators'][key])


def cpu_snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(item) for item in value)
    return copy.deepcopy(value)


def broadcast_state(value, collective):
    """小结构经 Gloo，大张量经张量组逐个广播，不 pickle 整个 Adam 到控制组。"""
    tensors = []
    def pack(item):
        if isinstance(item, torch.Tensor):
            index = len(tensors); tensors.append(item)
            return ('tensor', index, tuple(item.shape), item.dtype)
        if isinstance(item, dict):
            return ('dict', [(key, pack(child)) for key, child in item.items()])
        if isinstance(item, (tuple, list)):
            return (type(item).__name__, [pack(child) for child in item])
        return ('value', item)
    shape = collective.broadcast_object(pack(value) if collective.rank == 0 else None)
    def unpack(item):
        kind = item[0]
        if kind == 'tensor':
            _, index, size, dtype = item
            tensor = (tensors[index].to(collective.device).contiguous() if collective.rank == 0
                      else torch.empty(size, dtype=dtype, device=collective.device))
            if tensor.numel():
                dist.broadcast(tensor, src=0, group=collective.tensor_group)
            return tensor.cpu().clone()
        if kind == 'dict':
            return {key: unpack(child) for key, child in item[1]}
        if kind in ('tuple', 'list'):
            children = [unpack(child) for child in item[1]]
            return tuple(children) if kind == 'tuple' else children
        return item[1]
    return unpack(shape)


def begin_lease(collective, manager, root_budget, directory, phase, per_rank, guard=None):
    """按 rank 单独预占，失败/中断时保守保留预占消耗；成功才能对账退款。"""
    def allocate():
        current = root_budget.summary()
        total = {key: sum(item[key] for item in per_rank) for key in per_rank[0]}
        if any(current['used'][key] + count > current['limits'][key] for key, count in total.items()):
            raise RuntimeError('Insufficient global budget for the complete parallel collection lease')
        entries = [(f'{phase}/rank{rank}', amounts) for rank, amounts in enumerate(per_rank)]
        if hasattr(root_budget, 'reserve_many'):
            root_budget.reserve_many(entries)
        else:
            for name, amounts in entries:
                root_budget.reserve(name, **amounts)
        return True
    root_call(collective, allocate)
    credit = per_rank[collective.rank]
    limits = dict(accepted_iterations=1, optimizer_attempts=1, **credit)
    path = Path(directory) / 'resource_lease.json'
    budget = TrainingBudget(path, limits, disk_guard=guard)
    return budget


def finish_lease(collective, root_budget, local_budget, phase, per_rank):
    reports = collective.all_gather_object(local_budget.state_dict())
    def settle():
        entries = [(f'{phase}/rank{rank}', per_rank[rank],
                    {key: report['used'][key] for key in per_rank[rank]}, f'{phase}/rank{rank}')
                   for rank, report in enumerate(reports)]
        if hasattr(root_budget, 'settle_many'):
            root_budget.settle_many(entries)
        else:
            for name, reserved, used, identity in entries:
                root_budget.settle_lease(name, reserved, used, lease_id=identity)
        return root_budget.summary()
    return root_call(collective, settle)


def collection_credit(count, episode_seconds, latency_budget_s):
    # 单次生成最多推进一个有限 episode 加边界余量，每次 reset 的预热也先计入。
    controls = count * (math.ceil((episode_seconds + 2 * max(.5, latency_budget_s) + 2) * 50) + 50)
    return dict(generations=max(1, int(count)), control_steps=max(1, controls), physics_steps=4*max(1, controls))


def check_kl_limits(report, settings):
    failures = []
    tests = [('mean_joint_kl', settings['kl_stop_joint']),
             ('max_joint_kl', settings.get('kl_max_internal')),
             ('max_chain_joint_kl', settings.get('kl_max_chain'))]
    for key, limit in tests:
        if limit is not None and (key not in report or not math.isfinite(report[key]) or report[key] > limit):
            failures.append(key)
    maximum = max((row['mean_joint_kl'] for row in report['per_denoising_step']), default=float('nan'))
    limit = settings.get('kl_max_step_mean')
    if limit is not None and (not math.isfinite(maximum) or maximum > limit):
        failures.append('max_step_mean_joint_kl')
    if failures:
        raise RuntimeError(f'Final whole-rollout KL rejected: {failures}; fixed learning rate is unchanged')


def lightweight_fingerprint(module):
    """明确标为抽样统计而非完整哈希；仅检查各参数固定位置样本的有限性。"""
    values = []
    for parameter in module.parameters():
        flat = parameter.detach().reshape(-1)
        if flat.numel():
            positions = torch.linspace(0, flat.numel()-1, min(16, flat.numel()), device=flat.device).long()
            values.append(flat[positions].double())
    sampled = torch.cat(values)
    if not torch.isfinite(sampled).all():
        raise FloatingPointError('Nonfinite model fingerprint sample')
    return sampled.cpu().tolist()
