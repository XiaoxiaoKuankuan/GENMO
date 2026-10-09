"""完整控制轨迹的列式无损传输表示，供 GMT 和训练侧共享。

连续控制行按相同字段递归组织：数值数组沿控制步维堆叠，Python 数值转列，固定
字符串/静态列表只保存一次并附内容 SHA。每条控制步和四个物理子步仍完整保留，
可还原原逐行协议；异构故障行保留原值，不删除不方便合批的证据。此模块仅依赖
标准库和 NumPy，不依赖 Actor、Torch 或 Isaac。显式 v1 块身份不冒充旧回复字节。
"""
import copy
import hashlib
import json
import numpy as np

VERSION = 'genmo.control_trace.columns.v1'


def pack_trace(rows):
    if not isinstance(rows, list) or len(rows) > 25:
        raise ValueError('Trace requires at most 25 actual control rows')

    def column(values):
        first = values[0]
        if all(isinstance(v,np.ndarray) and v.dtype==first.dtype and v.shape==first.shape for v in values) if isinstance(first,np.ndarray) else False:
            return dict(kind='array', values=np.stack(values))
        if isinstance(first,dict) and all(isinstance(v,dict) and v.keys()==first.keys() for v in values):
            return dict(kind='mapping', fields={k:column([v[k] for v in values]) for k in sorted(first)})
        if isinstance(first,(list,tuple)) and all(isinstance(v,type(first)) and len(v)==len(first) for v in values):
            return dict(kind='tuple' if isinstance(first,tuple) else 'list',
                        fields=[column([v[i] for v in values]) for i in range(len(first))])
        if first is None or isinstance(first,(str,int,float,bool)):
            if all(type(v) is type(first) and v==first for v in values):
                raw = json.dumps(first,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()
                return dict(kind='constant',value=first,sha256=hashlib.sha256(raw).hexdigest())
            if type(first) in (int,float,bool) and all(type(v) is type(first) for v in values):
                array = np.asarray(values)
                if not array.dtype.hasobject:
                    return dict(kind='scalar',values=array)
        return dict(kind='rows', values=copy.deepcopy(values))

    return dict(schema=VERSION, count=len(rows), columns=column(rows) if rows else None)


def unpack_trace(block):
    if set(block) != {'schema','count','columns'} or block['schema'] != VERSION:
        raise ValueError('Unsupported trajectory block')
    count = block['count']
    if type(count) is not int or not 0 <= count <= 25:
        raise ValueError('Invalid trajectory row count')
    if count == 0:
        if block['columns'] is not None:
            raise ValueError('Empty trace has columns')
        return []

    def decode(node):
        kind = node['kind']
        if kind == 'mapping':
            columns = {k:decode(v) for k,v in node['fields'].items()}
            return [{k:v[i] for k,v in columns.items()} for i in range(count)]
        if kind in ('list','tuple'):
            columns = [decode(v) for v in node['fields']]
            return [(tuple(v[i] for v in columns) if kind=='tuple' else [v[i] for v in columns]) for i in range(count)]
        if kind == 'constant':
            raw = json.dumps(node['value'],ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()
            if hashlib.sha256(raw).hexdigest() != node['sha256']:
                raise ValueError('Static trace content SHA mismatch')
            return [copy.deepcopy(node['value']) for _ in range(count)]
        if kind in ('array','scalar','rows'):
            values = node['values']
            if len(values) != count:
                raise ValueError('Trace column count mismatch')
            if kind in ('array','scalar') and (not isinstance(values,np.ndarray) or values.dtype.hasobject):
                raise ValueError('Invalid trace array column')
            return [(values[i].copy() if kind=='array' else values[i].item() if kind=='scalar' else copy.deepcopy(values[i])) for i in range(count)]
        raise ValueError('Invalid trace column kind')
    return decode(block['columns'])


def expand_feedback(reply):
    """在 journal 已保存并 ACK 后，为奖励/旧审计恢复完整逐行视图。"""
    if isinstance(reply,dict) and 'trace_block' in reply:
        if 'trace' in reply:
            raise ValueError('Reply cannot contain two authoritative traces')
        return {k:v for k,v in reply.items() if k!='trace_block'} | {'trace':unpack_trace(reply['trace_block'])}
    return reply
