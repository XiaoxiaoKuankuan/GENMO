"""第十步共享采集数据上的多 GPU 同步学习与唯一主进程调度。

首版保留 rank 0 的实际 GMT 执行、完整数据采样、固定 GAE、预算和 checkpoint
发布。所有 GPU 持有同一个 Actor/Critic 的副本，按全局 batch 分摊梯度计算，
以 SUM collective 合并已按全局样本数归一化的梯度，再执行相同优化器步骤。
这不是多个独立实验，也不把评估分片充当训练。控制消息使用 Gloo，CUDA 张量
使用 NCCL；CPU 小模型验证允许同一个 Gloo group 完成全部通信。

每次更新前由主进程在共享运行目录写临时 batch，各 rank 读取同一 SHA 文件；
集体完成后删除交换文件。只有主进程持有正式 run 锁、写预算和 checkpoint。
每次更新核对所有 rank 的 Actor/Critic 参数及 buffer 指纹，任何不一致立即停止。
BC 仅主进程采样并反传一次，trainer 负责将其梯度准确合并到同步 PPO 梯度。

新运行和 resume 均由既有单采集主进程恢复权威状态：Actor/Critic、两个优化器、
音乐及 BC 采样器、随机状态、预算和策略版本。初始化时把已恢复模型和完整优化器
状态同步给所有 rank，并核验 optimizer 的动量、步数及参数组指纹。其他 rank 不
独立采样轨迹或 BC；Critic minibatch 由主进程广播，因此无需引入另一套续训随机源。
仅恢复已发布 checkpoint，丢弃中断时未接受的 rollout；不支持跨拓扑或跨任务身份。
异常交由 torchrun 终止同一作业的其他进程，不继续不完整的分布式更新。

梯度SUM归约另记录当前设备stream的CUDA事件，presence和每个梯度bucket均保留
真实通信区间；记录时不逐桶强制同步，外层阶段完成后统一query或等待末event。
CPU替身使用阻塞all_reduce的wall计时。张量本地bytes、host调用wall与stream耗时
分别报告，未完成记录继续待取，不用零值冒充缺测，也不改变归约/优化算法。
"""
from __future__ import annotations

import copy
import hashlib
import time
import weakref
from pathlib import Path
from uuid import uuid4

import torch
import torch.distributed as dist

from gem.closedloop.dppo import trainer
from gem.closedloop.dppo.critic import UpperCritic
from gem.closedloop.dppo.policy import DPPODiffusionPolicy
from gem.closedloop.frozen_actor import _fingerprint
from gem.robots.bumi.kinematics import sha256_file
from .performance import measure


def _optimizer_fingerprint(optimizer):
    """核验完整优化器状态；设备位置允许随rank变化，数值、精度和参数组必须一致。"""
    digest = hashlib.sha256()

    def visit(value):
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(f'tensor:{tensor.dtype}:{tuple(tensor.shape)}:'.encode())
            digest.update(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()))
        elif isinstance(value, dict):
            digest.update(b'dict:')
            for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
                visit(key)
                visit(value[key])
        elif isinstance(value, (tuple, list)):
            digest.update(f'{type(value).__name__}:{len(value)}:'.encode())
            for item in value:
                visit(item)
        else:
            digest.update(f'{type(value).__name__}:{value!r}:'.encode())

    visit(optimizer.state_dict())
    return digest.hexdigest()


class DistributedCollectives:
    def __init__(self, rank, world_size, *, control_group=None, tensor_group=None, device='cpu',
                 measure_gradient_communication=False):
        self.rank, self.world_size = int(rank), int(world_size)
        self.control_group, self.tensor_group = control_group, tensor_group
        self.device = torch.device(device)
        self._gradient_timing_pending = []
        self._gradient_timing_sequence = 0
        self._gradient_timing_enabled = bool(measure_gradient_communication)
        self._gradient_layouts = weakref.WeakKeyDictionary()
        self._gradient_buffers = {}

    def enable_gradient_timing(self, enabled=True):
        """新监视入口显式开启；旧入口没有consumer时不积压CUDA事件或历史记录。"""
        self._gradient_timing_enabled = bool(enabled)

    def _timed_gradient_reduce(self, tensor, record, kind):
        """仅包围真实SUM collective；CUDA事件延后收集，不增加逐桶同步。"""
        if record is None:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.tensor_group)
            return
        events = None
        if tensor.device.type == 'cuda':
            stream = torch.cuda.current_stream(tensor.device)
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record(stream)
            events = (begin, end, stream)
        started = time.perf_counter()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.tensor_group)
        wall = time.perf_counter()-started
        if events is not None:
            events[1].record(events[2])
        record['collectives'].append(dict(kind=kind, device=str(tensor.device),
            tensor_bytes=tensor.numel()*tensor.element_size(), host_api_seconds=wall,
            cpu_wall_seconds=wall if events is None else None, events=events))

    def collect_gradient_timings(self, *, synchronize=False):
        """外层阶段结束后取走完成的归约计时；默认只query，不等待GPU。

        CUDA值是SUM collective所在stream的真实event区间，包含通信等待；CPU值是
        阻塞all_reduce的wall耗时。pack/copy/优化器计算不在区间内，tensor_bytes是
        本地归约张量大小，不冒充网络传输量。synchronize=True仅等待每个stream的
        最后一个结束event，随后统一读取，绝不在sum_gradients内逐桶同步。
        尚未完成的调用继续保留；没有完成样本时seconds为None而非伪造为零。
        """
        pending = self._gradient_timing_pending
        if synchronize:
            ends = {}
            for record in pending:
                for collective in record['collectives']:
                    if collective['events'] is not None:
                        _, end, stream = collective['events']
                        ends[(collective['device'], stream.cuda_stream)] = end
            for end in ends.values():
                end.synchronize()
        complete, remaining = [], []
        for record in pending:
            if not all(c['events'] is None or c['events'][1].query() for c in record['collectives']):
                remaining.append(record)
                continue
            collectives = []
            for collective in record['collectives']:
                event = collective['events']
                seconds = collective['cpu_wall_seconds'] if event is None else event[0].elapsed_time(event[1])/1000.
                if not 0. <= seconds < float('inf'):
                    raise RuntimeError('Invalid completed gradient communication timing')
                collectives.append(dict(kind=collective['kind'], device=collective['device'],
                    tensor_bytes=collective['tensor_bytes'], host_api_seconds=collective['host_api_seconds'],
                    seconds=float(seconds), clock='host_wall_cpu' if event is None else 'cuda_stream_event'))
            complete.append(dict(sequence=record['sequence'], module=record['module'],
                call_wall_seconds=record['call_wall_seconds'], collectives=collectives))
        self._gradient_timing_pending = remaining
        by_module = {}
        for record in complete:
            aggregate = by_module.setdefault(record['module'], dict(call_count=0, collective_count=0,
                seconds=0., host_api_seconds=0., call_wall_seconds=0., tensor_bytes=0))
            aggregate['call_count'] += 1
            aggregate['call_wall_seconds'] += record['call_wall_seconds']
            for collective in record['collectives']:
                aggregate['collective_count'] += 1
                for key in ('seconds', 'host_api_seconds', 'tensor_bytes'):
                    aggregate[key] += collective[key]
        return dict(schema='genmo.closedloop.stage10.gradient_communication_timing.v1', rank=self.rank,
            completed_call_count=len(complete), pending_call_count=len(remaining),
            seconds=sum(v['seconds'] for v in by_module.values()) if complete else None,
            by_module=by_module, calls=complete, scope='sum_collective_stream_elapsed_excludes_pack_and_copy')

    def broadcast_object(self, value, src=0):
        objects = [value if self.rank == src else None]
        dist.broadcast_object_list(objects, src=src, group=self.control_group)
        return objects[0]

    def sum_tensor(self, tensor):
        result = tensor.detach().to(self.device).clone()
        dist.all_reduce(result, op=dist.ReduceOp.SUM, group=self.tensor_group)
        return result.to(tensor.device)

    def all_gather_object(self, value):
        """仅交换小型索引、计数和恢复状态；随机链保留在所属 rank。"""
        values = [None] * self.world_size
        dist.all_gather_object(values, value, group=self.control_group)
        return values

    def max_tensor(self, tensor):
        result = tensor.detach().to(self.device).clone()
        dist.all_reduce(result, op=dist.ReduceOp.MAX, group=self.tensor_group)
        return result.to(tensor.device)

    def barrier(self):
        dist.barrier(group=self.control_group)

    def sum_gradients(self, module):
        started, timing = None, None
        if self._gradient_timing_enabled:
            started = time.perf_counter()
            self._gradient_timing_sequence += 1
            timing = dict(sequence=self._gradient_timing_sequence, module=type(module).__name__, collectives=[])
        parameters = list(module.parameters())
        present = torch.tensor([p.grad is not None for p in parameters], dtype=torch.int32, device=self.device)
        self._timed_gradient_reduce(present, timing, 'gradient_presence')
        active = tuple(value > 0 for value in present.tolist())
        signature = (tuple((id(p), tuple(p.shape), p.dtype, p.device) for p in parameters), active)
        layouts = self._gradient_layouts.setdefault(module, {})
        if signature not in layouts:
            groups = {}
            for parameter, used in zip(parameters, active):
                if used:
                    groups.setdefault((parameter.device, parameter.dtype), []).append(parameter)
            layout = []
            limit = 32*1024*1024
            for key, entries in groups.items():
                pending, size = [], 0
                for parameter in entries:
                    required = parameter.numel()*parameter.element_size()
                    if pending and size+required > limit:
                        layout.append((key, pending)); pending, size = [], 0
                    pending.append(parameter); size += required
                if pending:
                    layout.append((key, pending))
            if len(layouts) >= 4:
                layouts.pop(next(iter(layouts)))
            layouts[signature] = layout
        else:
            layout = layouts[signature]
        for parameter, used in zip(parameters, active):
            if not used:
                parameter.grad = None
        # 每种device/dtype只复用最大单桶缓冲，避免缓存一份完整模型梯度。
        # 同stream的pack -> SUM -> copy按序执行，不额外除以world_size。
        for key, entries in layout:
            count = sum(parameter.numel() for parameter in entries)
            buffer = self._gradient_buffers.get(key)
            if buffer is None or buffer.numel() < count:
                buffer = torch.empty(count, device=key[0], dtype=key[1])
                self._gradient_buffers[key] = buffer
            flat, offset = buffer[:count], 0
            with measure('communication.gradient_pack', gpu=True):
                for parameter in entries:
                    target = flat[offset:offset+parameter.numel()].view_as(parameter)
                    target.zero_() if parameter.grad is None else target.copy_(parameter.grad)
                    offset += parameter.numel()
            self._timed_gradient_reduce(flat, timing, 'gradient_bucket')
            offset = 0
            with measure('communication.gradient_unpack', gpu=True):
                for parameter in entries:
                    if parameter.grad is None:
                        parameter.grad = torch.empty_like(parameter)
                    parameter.grad.copy_(flat[offset:offset+parameter.numel()].view_as(parameter))
                    offset += parameter.numel()
        if timing is not None:
            timing['call_wall_seconds'] = time.perf_counter()-started
            self._gradient_timing_pending.append(timing)

    def gather_rows(self, local_rows, local_indices, total_count):
        entries = [None] * self.world_size
        dist.all_gather_object(entries, (list(local_indices), torch.as_tensor(local_rows).cpu()), group=self.control_group)
        ordered = [None] * total_count
        for indices, values in entries:
            if len(indices) != len(values):
                raise ValueError('Distributed row/index count mismatch')
            for index, row in zip(indices, values):
                if not 0 <= index < total_count or ordered[index] is not None:
                    raise ValueError('Distributed duplicate or invalid global row')
                ordered[index] = row
        if any(row is None for row in ordered):
            raise ValueError('Distributed global rows are incomplete')
        return torch.stack(ordered).double()

    @torch.no_grad()
    def broadcast_module(self, module):
        for value in list(module.parameters()) + list(module.buffers()):
            if value.numel():
                contiguous = value.detach().contiguous()
                dist.broadcast(contiguous, src=0, group=self.tensor_group)
                value.copy_(contiguous)

    def fingerprints(self, actor, critic, phase):
        local = dict(rank=self.rank, device=str(self.device), actor=_fingerprint(actor), critic=_fingerprint(critic))
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, local, group=self.control_group)
        if len({(r['actor'], r['critic']) for r in gathered}) != 1:
            raise RuntimeError(f'Distributed model replicas diverged after {phase}')
        return dict(phase=phase, world_size=self.world_size, replicas_identical=True, ranks=gathered)


class DistributedLearner:
    """rank 0 调度已有训练入口，其他 rank 仅执行同步学习命令。"""
    def __init__(self, collectives, output_dir, preflight=None):
        self.distributed = collectives
        self.output_dir = Path(output_dir).resolve()
        self.preflight_result = preflight
        self.evidence = []
        self.attached = False
        self.sequence = 0
        self.actor = self.critic = self.policy = None
        self.actor_optimizer = self.critic_optimizer = None

    def preflight(self, config, *, check_gpu):
        # GPU空闲检查在任何rank创建CUDA上下文前完成，避免把本作业当外部占用。
        if self.preflight_result is None:
            raise RuntimeError('Distributed launch did not complete resource preflight')
        return copy.deepcopy(self.preflight_result)

    def attach(self, config, actor, critic, actor_optimizer, critic_optimizer, policy):
        self.actor, self.critic, self.policy = actor, critic, policy
        self.actor_optimizer, self.critic_optimizer = actor_optimizer, critic_optimizer
        self.distributed.broadcast_object(dict(operation='initialize', config=config))
        self._synchronize_initial()

    def _synchronize_initial(self):
        self.distributed.broadcast_module(self.actor)
        self.distributed.broadcast_module(self.critic)
        optimizer_evidence = self._synchronize_optimizers()
        evidence = self.distributed.fingerprints(self.actor, self.critic, 'initialization')
        evidence['optimizers'] = optimizer_evidence
        self.evidence.append(evidence)
        self.attached = True
        if self.distributed.rank == 0:
            print(f'[DISTRIBUTED] {self.distributed.world_size} GPU replicas initialized and identical', flush=True)

    def _synchronize_optimizers(self):
        """初始化时共享完整Adam状态，防止resume后worker错误地从空动量重新开始。"""
        path = None
        message = None
        if self.distributed.rank == 0:
            directory = self.output_dir / 'distributed_exchange'
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f'optimizers_{uuid4().hex}.pt'
            torch.save(dict(actor=self.actor_optimizer.state_dict(), critic=self.critic_optimizer.state_dict()), path)
            message = dict(path=str(path), sha256=sha256_file(path))
        message = self.distributed.broadcast_object(message)
        source = Path(message['path'])
        if source.parent != self.output_dir / 'distributed_exchange' or sha256_file(source) != message['sha256']:
            raise ValueError('Distributed optimizer path or SHA mismatch')
        if self.distributed.rank != 0:
            payload = torch.load(source, map_location='cpu', weights_only=False)
            self.actor_optimizer.load_state_dict(payload['actor'])
            self.critic_optimizer.load_state_dict(payload['critic'])
            del payload
        local = dict(rank=self.distributed.rank, actor=_optimizer_fingerprint(self.actor_optimizer),
                     critic=_optimizer_fingerprint(self.critic_optimizer))
        gathered = [None] * self.distributed.world_size
        dist.all_gather_object(gathered, local, group=self.distributed.control_group)
        if len({(row['actor'], row['critic']) for row in gathered}) != 1:
            raise RuntimeError('Distributed optimizer replicas differ after initialization or resume')
        dist.barrier(group=self.distributed.control_group)
        if path is not None:
            path.unlink()
        return dict(replicas_identical=True, state_source='rank0_authoritative_checkpoint_or_new_state', ranks=gathered)

    def _publish(self, operation, transitions, targets, kwargs):
        self.sequence += 1
        directory = self.output_dir / 'distributed_exchange'
        directory.mkdir(exist_ok=True)
        path = directory / f'batch_{self.sequence:06d}.pt'
        torch.save(dict(transitions=transitions, targets=targets, kwargs=kwargs), path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.distributed.broadcast_object(dict(operation=operation, path=str(path), sha256=digest))
        return path

    def _finish(self, operation, result, path=None):
        evidence = self.distributed.fingerprints(self.actor, self.critic, operation)
        self.evidence.append(evidence)
        dist.barrier(group=self.distributed.control_group)
        if path is not None:
            path.unlink()
        if isinstance(result, dict):
            result['distributed'] = evidence
        return result

    def critic_update(self, critic, optimizer, transitions, targets, **kwargs):
        wire = {key: value for key, value in kwargs.items() if key != 'generator'}
        path = self._publish('critic', transitions, targets, wire)
        result = trainer.critic_update(critic, optimizer, transitions, targets,
                                      distributed=self.distributed, **kwargs)
        return self._finish('critic', result, path)

    def actor_update(self, policy, optimizer, transitions, targets, **kwargs):
        wire = {key: value for key, value in kwargs.items()
                if key not in ('bc', 'reserve_attempt', 'calibration_progress')}
        path = self._publish('actor', transitions, targets, wire)
        result = trainer.actor_update(policy, optimizer, transitions, targets,
                                     distributed=self.distributed, **kwargs)
        return self._finish('actor', result, path)

    def analytic_kl(self, policy, transitions):
        path = self._publish('kl', transitions, None, {})
        result = trainer.analytic_kl(policy, transitions, distributed=self.distributed)
        return self._finish('kl', result, path)

    def worker_loop(self):
        while True:
            message = self.distributed.broadcast_object(None)
            operation = message['operation']
            if operation == 'stop':
                return
            if operation == 'initialize':
                config = copy.deepcopy(message['config'])
                config['runtime']['genmo_device'] = str(self.distributed.device)
                self.actor, train_config, _ = trainer.load_actor(config)
                self.critic = UpperCritic(qpos_mean=self.actor.endecoder.mean, qpos_std=self.actor.endecoder.std,
                    proprio_scales=tuple(train_config.model.proprio_scales)).to(self.distributed.device)
                settings = config['stage9']
                self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=settings['actor_lr'], weight_decay=0.)
                self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=settings['critic_lr'], weight_decay=0.)
                self.policy = DPPODiffusionPolicy(self.actor, steps=settings['denoising_steps'], eta=settings['eta'],
                    std_floor=settings['std_floor'], guidance_scale=settings['guidance_scale'])
                self._synchronize_initial()
                continue
            if not self.attached or operation not in ('critic', 'actor', 'kl'):
                raise ValueError('Unknown distributed learner command')
            path = Path(message['path'])
            if path.parent != self.output_dir / 'distributed_exchange' or hashlib.sha256(path.read_bytes()).hexdigest() != message['sha256']:
                raise ValueError('Distributed batch path or SHA mismatch')
            batch = torch.load(path, map_location='cpu', weights_only=False)
            transitions, targets, kwargs = batch['transitions'], batch['targets'], batch['kwargs']
            if operation == 'critic':
                result = trainer.critic_update(self.critic, self.critic_optimizer, transitions, targets,
                    distributed=self.distributed, **kwargs)
            elif operation == 'actor':
                result = trainer.actor_update(self.policy, self.actor_optimizer, transitions, targets,
                    distributed=self.distributed, **kwargs)
            else:
                result = trainer.analytic_kl(self.policy, transitions, distributed=self.distributed)
            self._finish(operation, result)
            del batch, transitions, targets, result

    def close(self):
        self.distributed.broadcast_object(dict(operation='stop'))
