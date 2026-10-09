"""第二阶段八卡独立采集的配置、同步、预算租约和张量辅助接口。

本模块只组织既有 UpperTransition，不另建训练数据格式。各卡保留完整随机链，
仅交换小型样本索引、统计量和必要恢复状态；梯度已经使用全局实际样本数作为分母，
通信必须使用 SUM。资源在物理执行之前由唯一根进程预占，各卡另写持久化使用账本；
所有执行都确认完成后才能退回明确未用的额度，进程失败或执行未知时不退款。

原始条件、旧概率和旧价值在同一 rollout 内不可变；可学习条件编码不能跨参数更新
复用。这里还提供一次整轮权威快照和无磁盘的张量状态广播，供 KL 拒绝时恢复模型和
Adam 状态。新版本配置显式绑定多步更新、延迟和检查频率，旧 v1 入口保持兼容。
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


def validate_v2_configuration(config):
    stage, settings = config['stage10'], config['stage10']['training']
    performance = stage.get('performance', {})
    if config['runtime'].get('backend') == 'gpu_vectorized.v1':
        runtime = config['runtime']
        if (type(runtime.get('num_envs')) is not int or not 1 <= runtime['num_envs'] <= 20
                or runtime.get('physics_device') != 'cuda:0'
                or performance.get('numerical_layout') != 'sample_matrix_bmm_fp32.v1'
                or settings.get('cfg_batch') is not True
                or settings.get('rollout_upper_steps_per_rank') != 20
                or settings.get('rollout_upper_steps') != 160
                or settings.get('denoising_steps') != 20):
            raise ValueError('GPU vector production requires genuine environments, 20 local chains and the accepted batch policy')
        if (runtime.get('prefix_deadline_contract') != 'available_reference_deadline_cap.v1'
                or runtime.get('vector_audit_contract') != 'nested_world_journal_excluded.v1'
                or runtime.get('vector_device_contract') != 'isaac_global_gpu_index.v1'
                or runtime.get('vector_fragment_contract') != 'available_reference_fragment_boundary.v3'):
            raise ValueError('GPU vector timing/prefix contracts must be explicit')
        if runtime.get('vector_reward_contract') not in (None, 'stage10.gpu_vector_continuous_reward.v1'):
            raise ValueError('Unknown vector reward computation contract')
        wait = runtime.get('vector_batch_wait_s')
        if isinstance(wait, bool) or not isinstance(wait,(int,float)) or not math.isfinite(wait) or not 0 <= wait <= 1:
            raise ValueError('GPU ready queue requires bounded explicit wait seconds')
    for key in ('tensor_cache_max_bytes', 'module_gradient_every', 'value_snapshot_batch_size',
                'disk_full_scan_every', 'asset_full_check_every'):
        if key in performance and (type(performance[key]) is not int or performance[key] < 1):
            raise ValueError(f'performance.{key} must be a positive integer')
    if performance.get('journal_format', 'json.v1') not in ('json.v1', 'genmo.execution_journal.ndarray.v2'):
        raise ValueError('Unknown journal format')
    layout = performance.get('numerical_layout', 'legacy_step_lane')
    if layout not in ('legacy_step_lane', 'sample_matrix_bmm_fp32.v1'):
        raise ValueError('Unknown numerical_layout')
    if layout != 'legacy_step_lane' and performance.get('execution_batch_size') is not None:
        raise ValueError('New sample-matrix layout must not specify a legacy fixed shape')
    if type(performance.get('profiler_every', 1)) is not int or performance.get('profiler_every', 1) < 0:
        raise ValueError('profiler_every must be zero or a positive integer')
    if type(performance.get('defer_tensor_checks', False)) is not bool:
        raise ValueError('defer_tensor_checks must be boolean')
    shape = performance.get('execution_batch_size')
    if shape is not None and (type(shape) is not int or shape not in (1, 2, 4)
                              or settings['denoising_steps'] % shape):
        raise ValueError('execution_batch_size must divide all denoising steps and be 1/2/4')
    if stage.get('distributed') != dict(world_size=8, backend='nccl', collection='all_ranks'):
        raise ValueError('Stage10 v2 requires eight synchronous independent collectors')
    if 'actor_lr_candidates' in settings:
        raise ValueError('Fixed-lr v2 must not contain learning-rate candidates')
    for name in ('ppo_epochs', 'actor_minibatch_internal_transitions', 'denoising_microbatch',
                 'max_actor_optimizer_steps', 'rollout_upper_steps_per_rank'):
        if type(settings.get(name)) is not int or settings[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    if settings['actor_minibatch_internal_transitions'] % settings['denoising_steps']:
        raise ValueError('Actor optimizer minibatch must contain complete denoising chains')
    if settings['rollout_upper_steps'] != 8 * settings['rollout_upper_steps_per_rank']:
        raise ValueError('Global rollout count must equal eight local collector counts')
    if not 0 < settings['kl_soft_stop_joint'] <= settings['kl_stop_joint']:
        raise ValueError('Soft KL threshold must be positive and no greater than final hard threshold')
    if settings.get('kl_check_mode', 'post_step_full') not in ('post_step_full', 'pre_step_plus_final'):
        raise ValueError('Unknown KL check mode')
    if settings.get('critic_update_mode', 'distributed') not in ('distributed', 'rank0_broadcast'):
        raise ValueError('Critic mode must be distributed or explicit rank0_broadcast comparison')
    if type(settings.get('x0_diagnostic_every', 0)) is not int or settings.get('x0_diagnostic_every', 0) < 0:
        raise ValueError('x0 diagnostic interval must be zero (disabled) or a positive integer')
    objective = settings.get('objective_logprob_reduction', 'joint_sum')
    if objective not in ('joint_sum', 'free_coordinate_mean'):
        raise ValueError('Unknown PPO probability objective')
    if objective != 'joint_sum' and not settings.get('alternative_objective_acknowledged', False):
        raise ValueError('Alternative objective requires explicit learning-rate/clip/BC acknowledgement')
    for key in ('kl_max_internal', 'kl_max_step_mean', 'kl_max_chain'):
        value = settings.get(key)
        if value is not None and (isinstance(value, bool) or not math.isfinite(value) or value <= 0):
            raise ValueError(f'{key} must be positive or null (diagnostic only)')
    if config['runtime'].get('timing_contract') != 'deployment_critical.v2':
        raise ValueError('Stage10 v2 requires the explicit deployment critical-path timing contract')
    for key in ('checkpoint_every_iterations', 'archive_queue_size'):
        if type(stage['storage'].get(key)) is not int or stage['storage'][key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for value in stage['checks'].values():
        if type(value) is not int or value < 1:
            raise ValueError('Periodic checks require positive integer intervals')
    for key in ('every_iterations', 'samples_per_source'):
        if type(stage['evaluation'].get(key)) is not int or stage['evaluation'][key] < 1:
            raise ValueError(f'Evaluation {key} must be positive')


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
    local = [dict(owner_rank=collective.rank, local_index=i,
                  valid=bool(row.transition_valid), has_free=bool(row.free_mask.any()))
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
