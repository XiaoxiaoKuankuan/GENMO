"""执行日志的规范二进制编码、旧 JSON 读取与独立内容校验。

v2 使用固定魔数、大小端无关长度头、排序 JSON 元数据和 C 连续数组原始字节。
数组描述符保存 dtype（含字节序）、shape、offset 和 size；不用 pickle，不同步
压缩，不将浮点数组展开为 Python 列表。SHA256 覆盖完整已编码字节，明确不同于
v1 JSON 的 SHA。对象共享关系不影响字节：相同逻辑值的深拷贝得到同一编码。
解码严格验证长度、数组覆盖范围和 dtype，拒绝截断/尾随数据；哈希由读取接口核验。
允许保存 NaN/Inf 故障证据，独立审计随后决定其是否是有效训练转移，不丢掉故障现场。
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping
from pathlib import Path
import numpy as np

MAGIC = b'GENMOJ2\0'
FORMAT = 'genmo.execution_journal.ndarray.v2'
MAX_BYTES = 256*1024**2


def encode_binary(value):
    parts, offset = [], 0

    def visit(item):
        nonlocal offset
        if hasattr(item, 'detach'):
            item = item.detach().cpu().numpy()
        if isinstance(item, np.ndarray):
            if item.dtype.hasobject or item.dtype.kind not in 'biufUS':
                raise TypeError('Unsupported journal array dtype')
            raw = item.tobytes(order='C')
            spec = dict(dtype=item.dtype.str, shape=list(item.shape), offset=offset, size=len(raw))
            parts.append(raw)
            offset += len(raw)
            if offset > MAX_BYTES:
                raise ValueError('Journal reply exceeds byte limit')
            return {'__journal_array_v2__':spec}
        if isinstance(item, np.generic):
            return visit(item.item())
        if isinstance(item, Mapping):
            if any(not isinstance(key, str) for key in item):
                raise TypeError('Journal mapping keys must be strings')
            if set(item) in ({'__journal_array_v2__'}, {'__nonfinite__'}):
                raise ValueError('Reserved journal metadata key')
            return {key:visit(item[key]) for key in sorted(item)}
        if isinstance(item, (list, tuple)):
            return [visit(child) for child in item]
        if isinstance(item, float) and not math.isfinite(item):
            return {'__nonfinite__':repr(item)}
        if isinstance(item, Path):
            return str(item)
        if item is None or isinstance(item, (str, int, float, bool)):
            return item
        raise TypeError(f'Unsupported journal value {type(item).__name__}')

    metadata = json.dumps(dict(format=FORMAT, value=visit(value)), ensure_ascii=False,
                          sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if len(metadata)+offset > MAX_BYTES:
        raise ValueError('Journal reply exceeds byte limit')
    return MAGIC+struct.pack('!QQ', len(metadata), offset)+metadata+b''.join(parts)


def decode_payload(payload, *, sha256=None):
    raw = payload.encode('utf-8') if isinstance(payload, str) else bytes(payload)
    if sha256 is not None and hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError('Journal SHA256 mismatch')
    if not raw.startswith(MAGIC):
        def legacy(item):
            if isinstance(item, dict):
                if set(item) == {'__array__','shape','values'}:
                    dtype = np.dtype(item['__array__'])
                    if dtype.hasobject or dtype.kind not in 'biufUS':
                        raise ValueError('Invalid legacy journal dtype')
                    return np.asarray(legacy(item['values']), dtype=dtype).reshape(item['shape'])
                if set(item) == {'__nonfinite__'}:
                    return float(item['__nonfinite__'])
                return {key:legacy(value) for key,value in item.items()}
            if isinstance(item, list):
                return [legacy(value) for value in item]
            return item
        return legacy(json.loads(raw))
    if len(raw) < 24:
        raise ValueError('Truncated journal header')
    meta_size, data_size = struct.unpack('!QQ', raw[8:24])
    if not meta_size or meta_size+data_size > MAX_BYTES or len(raw) != 24+meta_size+data_size:
        raise ValueError('Invalid journal frame lengths')
    header = json.loads(raw[24:24+meta_size])
    if set(header) != {'format','value'} or header['format'] != FORMAT:
        raise ValueError('Unknown journal format')
    data = memoryview(raw)[24+meta_size:]
    cursor = 0

    def visit(item):
        nonlocal cursor
        if isinstance(item, dict):
            if set(item) == {'__journal_array_v2__'}:
                spec = item['__journal_array_v2__']
                if set(spec) != {'dtype','shape','offset','size'}:
                    raise ValueError('Invalid journal array descriptor')
                shape, offset, size = spec['shape'], spec['offset'], spec['size']
                if (type(offset) is not int or type(size) is not int or size < 0 or offset != cursor
                        or offset+size > data_size or not isinstance(shape,list) or len(shape)>32
                        or any(type(n) is not int or n<0 for n in shape)):
                    raise ValueError('Invalid journal array bounds')
                dtype = np.dtype(spec['dtype'])
                if dtype.hasobject or dtype.kind not in 'biufUS' or math.prod(shape)*dtype.itemsize != size:
                    raise ValueError('Invalid journal dtype or shape byte count')
                cursor += size
                return np.frombuffer(data, dtype=dtype, count=math.prod(shape), offset=offset).reshape(shape).copy()
            if set(item) == {'__nonfinite__'}:
                if item['__nonfinite__'] not in ('nan','inf','-inf'):
                    raise ValueError('Invalid nonfinite marker')
                return float(item['__nonfinite__'])
            return {key:visit(item[key]) for key in sorted(item)}
        if isinstance(item, list):
            return [visit(value) for value in item]
        return item
    result = visit(header['value'])
    if cursor != data_size:
        raise ValueError('Unreferenced journal bytes')
    return result
