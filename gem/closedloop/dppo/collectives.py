"""当前八卡学习的通信原语与真实梯度归约计时。

只保留正式同进程训练使用的广播、对象集合、全局 SUM、梯度桶及模型一致性核验。
PPO/BC 梯度在本地按全局有效分母累计后统一同步，不重复除以 world_size。
CUDA 事件只度量实际通信，未完成的计时返回缺测，不用零冒充结果。
本模块不创建环境、不采集 rollout、不调度旧 rank0 独占采样，也不管理恢复文件。
"""
from __future__ import annotations
import time
import weakref
import torch
import torch.distributed as dist
from gem.closedloop.frozen_actor import _fingerprint
from .performance import measure


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
