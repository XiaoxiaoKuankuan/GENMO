"""第二阶段训练的分层性能计时与调用计数。

CPU 区间使用 perf_counter；GPU 区间同时记录当前 stream 的 CUDA Event，事件在
轮次边界统一读取，不在每个算子后执行设备全局同步。嵌套区间明确标记为包含关系，
不能直接相加冒充关键路径。未启用时接口为空操作，不改变随机数、梯度或执行时序。
每个 rank 独立持有记录器，采集计算、collective 等待和整轮墙钟必须分别命名。
序列化报告只包含标量统计和调用次数，不复制模型、执行轨迹或训练张量。
"""
from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

import torch

_ACTIVE = ContextVar('stage10_performance', default=None)


class PhaseProfiler:
    def __init__(self, device='cpu', rank=0):
        self.device, self.rank = torch.device(device), rank
        self.rows = defaultdict(lambda: dict(calls=0, host_seconds=0., cuda_seconds=0.))
        self.events = []
        self.started = time.perf_counter()

    @contextmanager
    def span(self, name, *, gpu=False):
        events = None
        if gpu and self.device.type == 'cuda':
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            stream = torch.cuda.current_stream(self.device)
            begin.record(stream)
            events = (begin, end, stream)
        start = time.perf_counter()
        try:
            yield
        finally:
            row = self.rows[name]
            row['calls'] += 1
            row['host_seconds'] += time.perf_counter() - start
            if events is not None:
                events[1].record(events[2])
                self.events.append((name, *events))

    def report(self):
        # 只等待各 stream 最后一个事件，不使用 torch.cuda.synchronize。
        ends = {stream.cuda_stream: end for _, _, end, stream in self.events}
        for end in ends.values():
            end.synchronize()
        for name, begin, end, _ in self.events:
            self.rows[name]['cuda_seconds'] += begin.elapsed_time(end) / 1000.
        self.events.clear()
        return dict(schema='genmo.stage10.performance.v1', rank=self.rank,
                    elapsed_wall_seconds=time.perf_counter()-self.started,
                    intervals='inclusive_nested_do_not_sum',
                    stages={name: dict(row) for name, row in sorted(self.rows.items())})


def activate(profiler):
    """返回 ContextVar token；调用者结束轮次时必须 deactivate。"""
    return _ACTIVE.set(profiler)


def deactivate(token):
    _ACTIVE.reset(token)


@contextmanager
def measure(name, *, gpu=False):
    profiler = _ACTIVE.get()
    if profiler is None:
        yield
    else:
        with profiler.span(name, gpu=gpu):
            yield


def profiled(name, *, gpu=False):
    def decorate(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            with measure(name, gpu=gpu):
                return function(*args, **kwargs)
        return wrapper
    return decorate
