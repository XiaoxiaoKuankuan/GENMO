"""完整控制轨迹的列式无损传输表示，供 GMT 和训练侧共享。

连续控制行按相同字段递归组织：数值数组沿控制步维堆叠，Python 数值转列，固定
字符串/静态列表只保存一次并附内容 SHA。每条控制步和四个物理子步仍完整保留，
可还原原逐行协议；异构故障行保留原值，不删除不方便合批的证据。此模块仅依赖
标准库和 NumPy，不依赖 Actor、Torch 或 Isaac。显式 v1 块身份不冒充旧回复字节。
"""
import copy
import hashlib
import json
from functools import lru_cache
import numpy as np

VERSION = 'genmo.control_trace.columns.v1'


@lru_cache(maxsize=8192)
def _scalar_sha(type_name, representation, value):
    # 类型与repr显式入key，避免True/1和+0.0/-0.0在Python哈希中相等而误用SHA。
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()
    return hashlib.sha256(raw).hexdigest()


def _constant_sha(value):
    if type(value) in (type(None), str, int, float, bool):
        return _scalar_sha(type(value).__name__, repr(value), value)
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()
    return hashlib.sha256(raw).hexdigest()


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
                return dict(kind='constant',value=first,sha256=_constant_sha(first))
            if type(first) in (int,float,bool) and all(type(v) is type(first) for v in values):
                array = np.asarray(values)
                if not array.dtype.hasobject:
                    return dict(kind='scalar',values=array)
        return dict(kind='rows', values=copy.deepcopy(values))

    return dict(schema=VERSION, count=len(rows), columns=column(rows) if rows else None)


class _TraceRow(dict):
    """权威列块的只读逐行适配器：只展开消费者实际访问的字段。

    块结构和每个常量SHA先完整验证；访问延迟不等于省略验证或物理证据。仅用于
    journal已持久化后的奖励/状态消费。显式复制或序列化时完整展开，防止dict子类
    的内部字段缓存被误当成全部证据；数组视图只读，普通协议缺省仍返回独立字典。
    """
    def __init__(self, fields, index):
        self._fields,self._index=fields,index
    def __getitem__(self,key):
        if not dict.__contains__(self,key):
            dict.__setitem__(self,key,_lazy_value(self._fields[key],self._index))
        return dict.__getitem__(self,key)
    def __iter__(self):return iter(self._fields)
    def __len__(self):return len(self._fields)
    def __contains__(self,key):return key in self._fields
    def keys(self):return self._fields.keys()
    def items(self):return ((key,self[key]) for key in self._fields)
    def values(self):return (self[key] for key in self._fields)
    def get(self,key,default=None):return self[key] if key in self._fields else default
    def copy(self):return {key:self[key] for key in self._fields}
    def __deepcopy__(self,memo):return {key:copy.deepcopy(self[key],memo) for key in self._fields}
    def __reduce_ex__(self,protocol):return dict,(self.copy(),)
    def __setitem__(self,*args):raise TypeError('Trace row is read-only')
    def __delitem__(self,*args):raise TypeError('Trace row is read-only')
    def update(self,*args,**kwargs):raise TypeError('Trace row is read-only')
    def clear(self):raise TypeError('Trace row is read-only')
    def pop(self,*args):raise TypeError('Trace row is read-only')
    def popitem(self):raise TypeError('Trace row is read-only')
    def setdefault(self,*args):raise TypeError('Trace row is read-only')


def _validate_lazy_columns(node,count):
    kind=node['kind']
    if kind=='mapping':
        for value in node['fields'].values():_validate_lazy_columns(value,count)
    elif kind in ('list','tuple'):
        for value in node['fields']:_validate_lazy_columns(value,count)
    elif kind=='constant':
        if _constant_sha(node['value'])!=node['sha256']:raise ValueError('Static trace content SHA mismatch')
    elif kind in ('array','scalar','rows'):
        value=node['values']
        if len(value)!=count:raise ValueError('Trace column count mismatch')
        if kind in ('array','scalar') and (not isinstance(value,np.ndarray) or value.dtype.hasobject):
            raise ValueError('Invalid trace array column')
    else:raise ValueError('Invalid trace column kind')


def _lazy_value(node,index):
    kind=node['kind']
    if kind=='mapping':return _TraceRow(node['fields'],index)
    if kind in ('list','tuple'):
        values=(_lazy_value(value,index) for value in node['fields'])
        return tuple(values) if kind=='tuple' else list(values)
    if kind=='constant':
        value=node['value']
        return value if type(value) in (type(None),str,int,float,bool) else copy.deepcopy(value)
    if kind=='array':
        value=node['values'][index].view();value.setflags(write=False);return value
    if kind=='scalar':return node['values'][index].item()
    if kind=='rows':return copy.deepcopy(node['values'][index])
    raise ValueError('Invalid trace column kind')


def unpack_trace(block, *, readonly_views=False, lazy=False):
    if set(block) != {'schema','count','columns'} or block['schema'] != VERSION:
        raise ValueError('Unsupported trajectory block')
    count = block['count']
    if type(count) is not int or not 0 <= count <= 25:
        raise ValueError('Invalid trajectory row count')
    if count == 0:
        if block['columns'] is not None:
            raise ValueError('Empty trace has columns')
        return []
    if lazy:
        if not readonly_views:raise ValueError('Lazy trace access requires read-only views')
        _validate_lazy_columns(block['columns'],count)
        return [_lazy_value(block['columns'],index) for index in range(count)]

    def decode(node):
        kind = node['kind']
        if kind == 'mapping':
            columns = {k:decode(v) for k,v in node['fields'].items()}
            return [{k:v[i] for k,v in columns.items()} for i in range(count)]
        if kind in ('list','tuple'):
            columns = [decode(v) for v in node['fields']]
            return [(tuple(v[i] for v in columns) if kind=='tuple' else [v[i] for v in columns]) for i in range(count)]
        if kind == 'constant':
            if _constant_sha(node['value']) != node['sha256']:
                raise ValueError('Static trace content SHA mismatch')
            value = node['value']
            # pack_trace 的常量仅含不可变内建标量。共享标量不会使行之间产生
            # 可写别名；旧证据若包含可变常量，仍逐行复制，保留读取兼容性。
            if type(value) in (type(None), str, int, float, bool):
                return [value] * count
            return [copy.deepcopy(value) for _ in range(count)]
        if kind in ('array','scalar','rows'):
            values = node['values']
            if len(values) != count:
                raise ValueError('Trace column count mismatch')
            if kind in ('array','scalar') and (not isinstance(values,np.ndarray) or values.dtype.hasobject):
                raise ValueError('Invalid trace array column')
            if kind == 'array' and readonly_views:
                result = [values[i].view() for i in range(count)]
                for item in result: item.setflags(write=False)
                return result
            return [(values[i].copy() if kind=='array' else values[i].item() if kind=='scalar' else copy.deepcopy(values[i])) for i in range(count)]
        raise ValueError('Invalid trace column kind')
    return decode(block['columns'])


def expand_feedback(reply, *, readonly_views=False, lazy=False):
    """在 journal 已保存并 ACK 后，为奖励/旧审计恢复完整逐行视图。"""
    if isinstance(reply,dict) and 'trace_block' in reply:
        if 'trace' in reply:
            raise ValueError('Reply cannot contain two authoritative traces')
        return {k:v for k,v in reply.items() if k!='trace_block'} | {'trace':unpack_trace(reply['trace_block'], readonly_views=readonly_views,lazy=lazy)}
    return reply


def concatenate_trace_blocks(blocks):
    """拼接同一环境的真实连续区间，不先展开全部控制步对象。

    数值列一次连接；常量与结构保持原协议，少量异构叶子独立恢复后重组。这里只
    合并已经持久化/确认区间的表示，不改变环境身份、执行顺序或最多25步的边界。
    """
    counts = [b['count'] for b in blocks]
    if not blocks or any(b['schema'] != VERSION for b in blocks) or sum(counts) > 25:
        raise ValueError('Cannot concatenate incompatible or oversized control blocks')
    useful = [(b['columns'], b['count']) for b in blocks if b['count']]
    if not useful: return pack_trace([])
    def combine(parts):
        first = parts[0][0]
        kinds = {node['kind'] for node,_ in parts}
        if kinds == {'mapping'} and all(node['fields'].keys() == first['fields'].keys() for node,_ in parts):
            return dict(kind='mapping', fields={k:combine([(n['fields'][k],c) for n,c in parts]) for k in first['fields']})
        if len(kinds)==1 and first['kind'] in ('list','tuple') and all(len(n['fields'])==len(first['fields']) for n,_ in parts):
            return dict(kind=first['kind'],fields=[combine([(n['fields'][i],c) for n,c in parts]) for i in range(len(first['fields']))])
        if kinds == {'constant'} and all(n['sha256']==first['sha256'] for n,_ in parts): return first
        if len(kinds)==1 and first['kind'] in ('array','scalar'):
            return dict(kind=first['kind'],values=np.concatenate([n['values'] for n,_ in parts]))
        values=[]
        for node,count in parts:
            values.extend(unpack_trace(dict(schema=VERSION,count=count,columns=node)))
        return pack_trace(values)['columns']
    return dict(schema=VERSION,count=sum(counts),columns=combine(useful))
