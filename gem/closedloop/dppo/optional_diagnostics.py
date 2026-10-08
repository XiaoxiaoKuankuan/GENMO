"""第二阶段显式启用的小 Critic 通信对照与逐步 x0 位移诊断。

本文件不被正式训练入口自动调用，不改变默认的八卡 SUM 梯度 Critic、80 次更新、
Actor 学习率或概率目标。critic_update_root_broadcast 供调用方显式比较通信策略：
只把 Critic 必需的条件、音乐剩余时间和固定 returns 汇集到 rank 0，沿用原更新器的
全局 minibatch、随机索引及损失归一化，随后广播参数和完整优化器状态。绝不传输
去噪链、奖励或 Actor 数据。计时包含汇集、更新、广播三个阶段；性能结论仍需相同
状态、相同采样索引和硬件上的另行有限对照。整轮失败回滚仍由调用方负责。

capture_x0_reference 在 Actor 更新前保存本 rank 的真实 pred_x_start CPU 快照；
x0_change_local 在相同旧链状态、条件及自由坐标上比较更新后的 x0，报告逐步 RMS、
最大绝对位移及平均绝对位移。它不把 DDIM 核均值倒推成 x0，也不把 x0 位移冒充 KL
或最终动作质量。由于基线日志未保存旧 x0，这个可选模式明确多做更新前后各一次
完整诊断前向，单独报告成本；链、掩码或条件被修改时拒绝混用参考。

本模块不采集环境、不写 checkpoint、不启动训练、不保存文件；本地失败在下一次
collective 前统一传播。所有上层轨迹、旧概率、GAE 和优势保持只读。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist

from .parallel_support import broadcast_state, cpu_snapshot
from .updater_v2 import _local_phase, _manifest, _owned, _parameters, critic_update_local


def _synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def _root_critic_inputs(transitions, targets, manifest, distributed):
    """按全局清单顺序汇集必要条件张量；Gloo 控制组只交换形状等小元数据。"""
    selected = [i for i, item in enumerate(manifest) if item['valid']]
    owned = _owned(selected, manifest, distributed)
    with _local_phase(distributed, 'optional_critic_input_validation'):
        if not selected:
            raise ValueError('Root Critic comparison requires valid transitions')
        values = torch.as_tensor(targets['returns']).detach()
        if values.shape != (len(transitions),):
            raise ValueError('Fixed returns must align with the complete local shard')
        rows = [transitions[index] for _, index in owned]
        specification = None
        for row in rows:
            fields = []
            for name, value in sorted(row.context.items()):
                if not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[0] != 1:
                    raise ValueError('Critic context fields require a leading singleton batch')
                fields.append((name, tuple(value.shape[1:]), value.dtype))
            if specification is not None and fields != specification:
                raise ValueError('Critic context shape or dtype differs inside the local shard')
            specification = fields
        remaining = [float(row.metadata['remaining_music_seconds']) for row in rows]
        if (any(not math.isfinite(value) or value < 0 for value in remaining)
                or any(not torch.isfinite(values[index]) for _, index in owned)):
            raise ValueError('Critic requires finite fixed returns and nonnegative remaining time')
    descriptions = distributed.all_gather_object(dict(specification=specification,
        positions=[position for position, _ in owned]))
    available = [entry['specification'] for entry in descriptions if entry['positions']]
    if not available or any(item != available[0] for item in available):
        raise ValueError('Critic condition interfaces differ between ranks')
    specification = available[0]
    maximum = max(len(entry['positions']) for entry in descriptions)
    ordered_positions = [position for entry in descriptions for position in entry['positions']]
    if sorted(ordered_positions) != list(range(len(selected))):
        raise ValueError('Critic global selected indices are incomplete or duplicated')
    root_fields, payload_bytes = {}, 0
    fields = specification + [('__fixed_returns', (), torch.float32), ('__remaining_seconds', (), torch.float32)]
    if any(name.startswith('__') for name, _, _ in specification):
        raise ValueError('Critic condition key conflicts with reserved diagnostic fields')
    for name, shape, dtype in fields:
        with _local_phase(distributed, 'optional_critic_gather_tensor_allocation'):
            local = torch.zeros((maximum, *shape), dtype=dtype, device=distributed.device)
            if rows:
                if name == '__fixed_returns':
                    source = values[[index for _, index in owned]]
                elif name == '__remaining_seconds':
                    source = torch.tensor(remaining)
                else:
                    source = torch.cat([row.context[name] for row in rows], 0)
                local[:len(rows)] = source.to(device=distributed.device, dtype=dtype)
            gathered = [torch.empty_like(local) for _ in descriptions] if distributed.rank == 0 else None
        dist.gather(local, gather_list=gathered, dst=0, group=distributed.tensor_group)
        with _local_phase(distributed, 'optional_critic_gather_tensor_reorder'):
            if distributed.rank == 0:
                field = torch.empty((len(selected), *shape), dtype=dtype, device=distributed.device)
                for description, value in zip(descriptions, gathered):
                    field[description['positions']] = value[:len(description['positions'])]
                root_fields[name] = field
            payload_bytes += len(selected) * local[0].numel() * local.element_size()
    with _local_phase(distributed, 'optional_critic_root_rows'):
        root_rows, root_targets = None, None
        if distributed.rank == 0:
            root_rows = [SimpleNamespace(context={name: root_fields[name][i:i + 1] for name, _, _ in specification},
                metadata={'remaining_music_seconds': float(root_fields['__remaining_seconds'][i])},
                transition_valid=True, free_mask=torch.tensor([manifest[index]['has_free']]))
                for i, index in enumerate(selected)]
            root_targets = {'returns': root_fields['__fixed_returns']}
    return root_rows, root_targets, payload_bytes


def critic_update_root_broadcast(critic, optimizer, transitions, targets, *, global_manifest=None,
                                distributed=None, steps=80, batch_size=32, generator=None,
                                grad_clip_norm=1.):
    """显式单卡 Critic 更新后广播；调用方持有整轮快照，本函数不隐式切换默认模式。"""
    device = next(critic.parameters()).device
    _synchronize(device)
    started = time.perf_counter()
    with _local_phase(distributed, 'optional_critic_manifest'):
        if type(steps) is not int or not 1 <= steps <= 80:
            raise ValueError('Explicit root Critic comparison supports 1 to 80 steps')
        manifest = _manifest(transitions, targets, distributed, global_manifest)
    if distributed is None:
        report = critic_update_local(critic, optimizer, transitions, targets, global_manifest=manifest,
            steps=steps, batch_size=batch_size, generator=generator, grad_clip_norm=grad_clip_norm)
        _synchronize(device)
        return dict(report, execution_mode='root_update_then_broadcast',
            timings=dict(gather_seconds=0., update_seconds=time.perf_counter() - started, broadcast_seconds=0.),
            critic_payload_bytes=0, optimizer_state_broadcast=False)
    root_rows, root_targets, payload_bytes = _root_critic_inputs(transitions, targets, manifest, distributed)
    _synchronize(device)
    gathered_at = time.perf_counter()
    report, optimizer_state = None, None
    with _local_phase(distributed, 'optional_critic_root_update'):
        if distributed.rank == 0:
            report = critic_update_local(critic, optimizer, root_rows, root_targets, steps=steps,
                batch_size=batch_size, generator=generator, grad_clip_norm=grad_clip_norm)
            optimizer_state = cpu_snapshot(optimizer.state_dict())
        _synchronize(device)
    updated_at = time.perf_counter()
    distributed.broadcast_module(critic)
    optimizer_state = broadcast_state(optimizer_state, distributed)
    with _local_phase(distributed, 'optional_critic_optimizer_restore'):
        optimizer.load_state_dict(optimizer_state)
        optimizer.zero_grad(set_to_none=True)
        critic.eval()
        _synchronize(device)
    timings = dict(gather_seconds=gathered_at - started, update_seconds=updated_at - gathered_at,
                   broadcast_seconds=time.perf_counter() - updated_at)
    rank_timings = distributed.all_gather_object(timings)
    report = distributed.broadcast_object(report)
    return dict(report, execution_mode='root_update_then_broadcast',
        timings={key: max(row[key] for row in rank_timings) for key in timings},
        rank_timings=rank_timings, critic_payload_bytes=payload_bytes, optimizer_state_broadcast=True)


def _row_signature(row):
    digest = hashlib.sha256()
    for key, tensor in [('chain', row.chain), ('free_mask', row.free_mask), *sorted(row.context.items())]:
        value = tensor.detach().cpu().contiguous()
        digest.update(str((key, value.dtype, tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class X0Reference:
    """仅供本轮临时诊断的内存对象；不是训练轨迹或 checkpoint 的新持久化格式。"""
    kernel: str
    manifest: tuple
    row_signatures: tuple
    predictions: tuple
    policy_version_label: str
    extra_internal_forwards: int


def _x0_rows(policy, transitions, global_manifest, distributed):
    from .trainer import _validate_actor_transitions
    manifest = _manifest(transitions, {}, distributed, global_manifest)
    selected = [i for i, item in enumerate(manifest) if item['valid'] and item['has_free']]
    if not selected:
        raise ValueError('x0 diagnostic requires valid transitions with free coordinates')
    owned = _owned(selected, manifest, distributed)
    rows = [transitions[index] for _, index in owned]
    if rows:
        _validate_actor_transitions(policy, rows)
    return manifest, selected, owned, rows


@torch.no_grad()
def capture_x0_reference(policy, transitions, *, global_manifest=None, distributed=None,
                         denoising_microbatch=4, policy_version_label='before_update'):
    """更新前显式多做一次诊断前向；返回独立 CPU x0，不跨 optimizer step 复用编码。"""
    with _local_phase(distributed, 'optional_x0_reference_forward'):
        if type(denoising_microbatch) is not int or denoising_microbatch < 1:
            raise ValueError('x0 diagnostic microbatch must be positive')
        manifest, selected, _, rows = _x0_rows(policy, transitions, global_manifest, distributed)
        device = next(policy.actor.parameters()).device
        values = [torch.empty_like(row.chain[:-1], device='cpu') for row in rows]
        flat = [(index, step) for index in range(len(rows)) for step in range(policy.steps)]
        for start in range(0, len(flat), denoising_microbatch):
            chunk = flat[start:start + denoising_microbatch]
            parameters, mask = _parameters(policy, [rows[index] for index, _ in chunk],
                                          [step for _, step in chunk], device)
            prediction = parameters.get('pred_x_start')
            if prediction is None or prediction.shape != mask.shape or not torch.isfinite(prediction[mask]).all():
                raise ValueError('Policy must expose finite pred_x_start on every free coordinate')
            for offset, (index, step) in enumerate(chunk):
                values[index][step].copy_(prediction[offset].cpu())
        result = X0Reference(json.dumps(policy.kernel_config, sort_keys=True),
            tuple((item['owner_rank'], item['local_index'], item['valid'], item['has_free']) for item in manifest),
            tuple(_row_signature(row) for row in rows), tuple(values), str(policy_version_label), len(selected) * policy.steps)
    return result


@torch.no_grad()
def x0_change_local(policy, transitions, reference, *, global_manifest=None, distributed=None,
                    denoising_microbatch=4):
    """在固定旧链和条件上记录逐步真实 x0 差异；自由坐标以外不参与分母或最大值。"""
    with _local_phase(distributed, 'optional_x0_change_forward'):
        if type(denoising_microbatch) is not int or denoising_microbatch < 1:
            raise ValueError('x0 diagnostic microbatch must be positive')
        manifest, selected, owned, rows = _x0_rows(policy, transitions, global_manifest, distributed)
        expected_manifest = tuple((item['owner_rank'], item['local_index'], item['valid'], item['has_free']) for item in manifest)
        if (not isinstance(reference, X0Reference) or reference.manifest != expected_manifest
                or reference.kernel != json.dumps(policy.kernel_config, sort_keys=True)
                or reference.row_signatures != tuple(_row_signature(row) for row in rows)):
            raise ValueError('x0 reference differs from the fixed chain, condition, mask or kernel')
        device = next(policy.actor.parameters()).device
        output = torch.zeros((len(rows), policy.steps, 4), dtype=torch.float64, device=device)
        flat = [(index, step) for index in range(len(rows)) for step in range(policy.steps)]
        for start in range(0, len(flat), denoising_microbatch):
            chunk = flat[start:start + denoising_microbatch]
            parameters, mask = _parameters(policy, [rows[index] for index, _ in chunk],
                                          [step for _, step in chunk], device)
            prediction = parameters.get('pred_x_start')
            if prediction is None or prediction.shape != mask.shape:
                raise ValueError('Policy must expose pred_x_start matching its free mask')
            old = torch.stack([reference.predictions[index][step] for index, step in chunk]).to(device)
            difference = (prediction.double() - old.double()).masked_fill(~mask, 0.)
            if not torch.isfinite(difference).all():
                raise FloatingPointError('Nonfinite x0 change on a free coordinate')
            values = torch.stack((difference.square().sum((-2, -1)), difference.abs().sum((-2, -1)),
                                  difference.abs().flatten(1).max(1).values, mask.sum((-2, -1))), 1)
            for offset, (index, step) in enumerate(chunk):
                output[index, step] = values[offset]
    if distributed is not None:
        output = distributed.gather_rows(output.reshape(len(rows), policy.steps * 4),
                                        [position for position, _ in owned], len(selected)).reshape(len(selected), policy.steps, 4)
    output = output.cpu()
    report = []
    for step in range(policy.steps):
        squared, absolute, maximum, count = output[:, step].unbind(1)
        report.append(dict(step_index=step, free_coordinate_x0_rms=float((squared.sum() / count.sum()).sqrt()),
            mean_upper_x0_rms=float((squared / count).sqrt().mean()),
            max_upper_x0_rms=float((squared / count).sqrt().max()),
            mean_absolute_x0_change=float(absolute.sum() / count.sum()), max_absolute_x0_change=float(maximum.max()),
            free_coordinate_count=int(count.sum())))
    return dict(scope='normalized_x0_on_fixed_old_chain_free_coordinates_not_final_action_quality',
        policy_version_label=reference.policy_version_label, included_upper_transitions=len(selected),
        reference_extra_internal_forwards=reference.extra_internal_forwards,
        comparison_extra_internal_forwards=len(selected) * policy.steps, per_denoising_step=report)
