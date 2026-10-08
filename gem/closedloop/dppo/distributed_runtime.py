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
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from uuid import uuid4

import torch
import torch.distributed as dist

from gem.closedloop.dppo import trainer
from gem.closedloop.dppo.critic import UpperCritic
from gem.closedloop.dppo.policy import DPPODiffusionPolicy
from gem.closedloop.frozen_actor import _fingerprint
from gem.robots.bumi.kinematics import sha256_file


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
    def __init__(self, rank, world_size, *, control_group=None, tensor_group=None, device='cpu'):
        self.rank, self.world_size = int(rank), int(world_size)
        self.control_group, self.tensor_group = control_group, tensor_group
        self.device = torch.device(device)

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
        parameters = list(module.parameters())
        present = torch.tensor([p.grad is not None for p in parameters], dtype=torch.int32, device=self.device)
        dist.all_reduce(present, op=dist.ReduceOp.SUM, group=self.tensor_group)
        buckets = {}
        for parameter, count in zip(parameters, present.tolist()):
            if count == 0:
                parameter.grad = None
                continue
            gradient = torch.zeros_like(parameter) if parameter.grad is None else parameter.grad.contiguous()
            buckets.setdefault((gradient.device, gradient.dtype), []).append((parameter, gradient))
        # 已按全局 minibatch 分母归一化，只 SUM，不再除以 world_size。
        # 分桶最多 32 MiB；超大单个参数独占桶，避免为全模型再分配一份梯度。
        limit = 32 * 1024 * 1024
        for entries in buckets.values():
            pending, size = [], 0
            def flush():
                if not pending:
                    return
                flat = torch.cat([gradient.reshape(-1) for _, gradient in pending])
                dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=self.tensor_group)
                offset = 0
                for parameter, gradient in pending:
                    gradient.copy_(flat[offset:offset + gradient.numel()].view_as(gradient))
                    parameter.grad = gradient
                    offset += gradient.numel()
            for parameter, gradient in entries:
                required = gradient.numel() * gradient.element_size()
                if pending and size + required > limit:
                    flush()
                    pending, size = [], 0
                pending.append((parameter, gradient))
                size += required
            flush()

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
