"""整轮更新失败后的逐字节恢复核验，只在异常路径执行。

递归摘要绑定张量/数组的类型、精度、形状与全部字节，绑定容器的键、顺序及标量。
设备位置允许不同，以便比较 rank0 CPU 权威快照和各 GPU 恢复后的模型/Adam；数值
不使用容差，不漏查 NumPy RNG 数组的中间元素。Actor、Critic、两个优化器、BC 与
本地独立 RNG 分项报告。此模块不保存第二份整轮状态，不改变预算，也不在正常
minibatch 热路径扫描大模型；恢复不一致时由调用方发布失败证据并终止会话。
"""
from __future__ import annotations

import hashlib
import struct
import numpy as np
import torch


def exact_state_digest(value):
    digest = hashlib.sha256()
    def frame(data):
        digest.update(len(data).to_bytes(8, 'little'))
        digest.update(data)
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            frame(f'tensor:{tensor.dtype}:{tuple(tensor.shape)}'.encode())
            frame(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()))
        elif isinstance(item, np.ndarray):
            if item.dtype.hasobject:
                raise TypeError('Object arrays are not valid training recovery state')
            frame(f'ndarray:{item.dtype.str}:{item.shape}'.encode())
            frame(memoryview(np.ascontiguousarray(item)).cast('B'))
        elif isinstance(item, dict):
            frame(f'dict:{len(item)}'.encode())
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                visit(key); visit(item[key])
        elif isinstance(item, (tuple, list)):
            frame(f'{type(item).__name__}:{len(item)}'.encode())
            for child in item: visit(child)
        elif isinstance(item, float):
            frame(b'float64'); frame(struct.pack('!d', item))
        elif isinstance(item, (str, int, bool, bytes, type(None), np.generic)):
            frame(type(item).__name__.encode()); frame(repr(item).encode())
        else:
            raise TypeError(f'Unsupported recovery state: {type(item).__name__}')
    visit(value)
    return digest.hexdigest()


def compare_recovered_state(expected, actual):
    if set(expected) != set(actual):
        raise ValueError('Recovered state fields differ')
    fields = {}
    for key in expected:
        before, after = exact_state_digest(expected[key]), exact_state_digest(actual[key])
        fields[key] = dict(expected_sha256=before, recovered_sha256=after, identical=before == after)
    return dict(scope='complete_state_exact_bytes', fields=fields,
                identical=all(field['identical'] for field in fields.values()))
