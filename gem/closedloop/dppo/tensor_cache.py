"""单轮、单rank的只读学习张量缓存及可反传条件缓存。

RolloutTensorCache在采集完成后一次组织条件、完整链、旧高斯核、掩码和固定目标。
默认整轮驻留设备；超过显式字节上限时使用固定完整链块和有界LRU，按块搬运并记录
传输次数，绝不在每个去噪步隐式重新拼接CPU原始对象。旧数据不进入计算图，缓存
绑定本轮对象身份和随机核，结束必须close，不能沿用到另一个rollout。

ConditionGraphCache让每条链只编码一次。训练时临时叶子累计各去噪微批的条件梯度，
随后将累计梯度一次传回原编码图，因此历史、前缀、音乐编码器仍获得完整PPO梯度。
它不跨optimizer.step复用，不保留denoiser图，不使用retain_graph。只读KL路径则直接
复用条件表示。输入张量与可训练条件表示有不同生命周期，不能混用。
"""
from __future__ import annotations

from collections import OrderedDict
import json
import math

import torch

from .performance import measure


class RolloutTensorCache:
    def __init__(self, transitions, targets, device, *, max_device_bytes=256*1024**2):
        self.device = torch.device(device)
        self.rows = tuple(transitions)
        self.lookup = {id(row): i for i, row in enumerate(self.rows)}
        if len(self.lookup) != len(self.rows):
            raise ValueError('Rollout cache cannot contain duplicate transition objects')
        if type(max_device_bytes) is not int or max_device_bytes <= 0:
            raise ValueError('Tensor cache budget must be a positive byte count')
        self.closed = False
        self.blocks, self.index_cache = OrderedDict(), {}
        self.transfer_count, self.transferred_bytes = 0, 0
        self.identities = [dict(getattr(row, 'identity', {})) for row in self.rows]
        self.kernel = None
        self.host = {}
        with measure('learning.tensor_cache_prepare'):
            if not self.rows:
                self.block_size, self.max_blocks = 1, 1
                self.mode = 'device'
                self.bytes = 0
                return
            for key in self.rows[0].context:
                self.host['context/'+key] = torch.cat([row.context[key].detach().cpu() for row in self.rows], 0)
            prototype = next((row for row in self.rows if row.chain is not None), None)
            if prototype is not None:
                trace = prototype.metadata['sampler_trace']
                self.kernel = json.dumps(trace['kernel_config'], sort_keys=True)
                extractors = dict(chain=lambda r: r.chain, old_log_prob=lambda r: r.old_log_prob,
                    free_mask=lambda r: r.free_mask,
                    old_means=lambda r: r.metadata['sampler_trace']['old_means'][0],
                    old_stds=lambda r: r.metadata['sampler_trace']['old_stds'][0])
                for name, extract in extractors.items():
                    template = extract(prototype).detach().cpu()
                    values = [extract(row).detach().cpu() if row.chain is not None else torch.zeros_like(template)
                              for row in self.rows]
                    self.host[name] = torch.stack(values)
            for key in ('advantages', 'returns', 'valid', 'old_values', 'next_values'):
                if key in targets:
                    value = torch.as_tensor(targets[key]).detach().cpu().clone()
                    if value.shape != (len(self.rows),):
                        raise ValueError(f'Cache target {key} must align with local rollout')
                    self.host[key] = value
            self.host['remaining_music_seconds'] = torch.tensor([
                row.metadata.get('remaining_music_seconds', 0.) for row in self.rows], dtype=torch.float32)
            self.host['local_indices'] = torch.arange(len(self.rows), dtype=torch.long)
            self.host['valid_lengths'] = self.host['context/future_valid'].sum(1)
        self.bytes = sum(value.numel()*value.element_size() for value in self.host.values())
        per_row = max(1, math.ceil(self.bytes/len(self.rows)))
        if per_row > max_device_bytes:
            raise MemoryError(f'One complete rollout chain requires {per_row} cache bytes, budget={max_device_bytes}')
        self.mode = 'device' if self.bytes <= max_device_bytes else 'pinned_blocks'
        self.block_size = len(self.rows) if self.mode == 'device' else max(1, min(8, max_device_bytes//per_row))
        self.max_blocks = max(1, max_device_bytes//(per_row*self.block_size))
        if self.mode == 'pinned_blocks' and self.device.type == 'cuda':
            self.host = {key: value.pin_memory() for key, value in self.host.items()}
        if self.mode == 'device':
            self._block(0)

    def _block(self, index):
        if self.closed:
            raise RuntimeError('Rollout tensor cache already released')
        if index in self.blocks:
            self.blocks.move_to_end(index)
            return self.blocks[index]
        while len(self.blocks) >= self.max_blocks:
            self.blocks.popitem(last=False)
        start = index*self.block_size
        with measure('learning.tensor_cache_transfer', gpu=True):
            block = {key: value[start:start+self.block_size].to(self.device, non_blocking=True)
                     for key, value in self.host.items()}
        self.transfer_count += 1
        self.transferred_bytes += sum(value.numel()*value.element_size() for value in block.values())
        self.blocks[index] = block
        return block

    def _positions(self, rows):
        if self.closed:
            raise RuntimeError('Rollout tensor cache already released')
        try:
            return tuple(self.lookup[id(row)] for row in rows)
        except KeyError as error:
            raise ValueError('Transition belongs to a different rollout cache') from error

    def get(self, name, rows, steps=None):
        positions = self._positions(rows)
        # 按原顺序保留跨块尾批；常规整轮驻留走一次index_select/高级索引。
        block_ids = {index//self.block_size for index in positions}
        if len(block_ids) != 1:
            return torch.cat([self.get(name, [row], None if steps is None else [step])
                for row, step in zip(rows, steps if steps is not None else [None]*len(rows))])
        block_id = next(iter(block_ids))
        values = self._block(block_id)[name]
        key = (positions, None if steps is None else tuple(steps))
        if key not in self.index_cache:
            self.index_cache[key] = (torch.tensor([i % self.block_size for i in positions], device=self.device),
                None if steps is None else torch.tensor(steps, device=self.device))
        indices, time = self.index_cache[key]
        return values[indices] if time is None else values[indices, time]

    def context(self, rows):
        self._positions(rows)
        return {key.removeprefix('context/'): self.get(key, rows) for key in self.host if key.startswith('context/')}

    def report(self):
        return dict(mode=self.mode, bytes=self.bytes, block_upper_chains=self.block_size,
                    block_transfers=self.transfer_count, transferred_bytes=self.transferred_bytes,
                    upper_transitions=len(self.rows))

    def close(self):
        self.closed = True
        self.blocks.clear()
        self.host.clear()
        self.index_cache.clear()


class ConditionGraphCache:
    def __init__(self, policy, tensor_cache=None):
        self.policy, self.tensor_cache = policy, tensor_cache
        self.training = torch.is_grad_enabled()
        self.signature = policy._parameter_signature()
        self.entries = {}
        self.finished = False

    def prepare(self, rows, context):
        if self.finished or self.policy._parameter_signature() != self.signature:
            raise ValueError('Condition graph cache cannot cross an optimizer step or completed backward')
        entries = []
        for row in rows:
            key = id(row)
            if key not in self.entries:
                single = (self.tensor_cache.context([row]) if self.tensor_cache is not None else
                          {key: value.to(context[key].device) for key, value in row.context.items()})
                original = self.policy.prepare_conditions(single)
                entry = dict(original=original, leaf={})
                for name in ('conditional', 'unconditional'):
                    feature = original[name]
                    entry['leaf'][name] = (feature.detach().requires_grad_(True)
                        if self.training and feature is not None and feature.requires_grad else feature)
                self.entries[key] = entry
            entries.append(self.entries[key])
        prepared = dict(entries[0]['original'])
        prepared['adapted'] = {key: torch.cat([entry['original']['adapted'][key] for entry in entries])
                               for key in prepared['adapted']}
        for name in ('conditional', 'unconditional'):
            prepared[name] = (None if entries[0]['leaf'][name] is None else
                              torch.cat([entry['leaf'][name] for entry in entries]))
        prepared['inputs'] = self.policy._input_signature(context)
        return prepared

    def backward(self):
        if self.finished:
            raise RuntimeError('Condition encoder gradients already propagated')
        if self.policy._parameter_signature() != self.signature:
            raise ValueError('Actor changed before condition encoder backward')
        if self.training:
            with measure('actor.condition_encoder_backward', gpu=True):
                for entry in self.entries.values():
                    outputs, gradients = [], []
                    for name in ('conditional', 'unconditional'):
                        original, leaf = entry['original'][name], entry['leaf'][name]
                        if original is not None and original.requires_grad and leaf.grad is not None:
                            outputs.append(original)
                            gradients.append(leaf.grad)
                    if outputs:
                        torch.autograd.backward(outputs, gradients)
        self.finished = True
        self.entries.clear()


class KLResultCache:
    """同一参数版本的逐链完整KL/噪声诊断缓存；任何身份变化都拒绝复用。"""
    def __init__(self, policy, transitions, records, manifest=None):
        self.identity = self._identity(policy, transitions, manifest)
        self.records = {index: tuple(value.detach().clone() for value in values)
                        for index, values in records.items()}

    @staticmethod
    def _identity(policy, transitions, manifest):
        def row_identity(row):
            trace = row.metadata.get('sampler_trace', {})
            tensors = [row.chain, row.old_log_prob, row.free_mask, trace.get('old_means'), trace.get('old_stds'), *row.context.values()]
            return (id(row), tuple((id(value), value._version) for value in tensors if isinstance(value, torch.Tensor)))
        return (tuple(row_identity(row) for row in transitions),
                tuple((id(p), p._version) for p in policy.actor.parameters()),
                json.dumps(policy.kernel_config, sort_keys=True),
                json.dumps(manifest, sort_keys=True),
                'old_path_joint_free_sum_per_internal.v1')

    def validate(self, policy, transitions, manifest=None):
        if self.identity != self._identity(policy, transitions, manifest):
            raise ValueError('KL cache rollout, Actor parameter version, kernel or aggregation changed')
