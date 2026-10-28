"""交替训练的真实墙钟与CUDA流分段计时，不在每个小算子后同步GPU。

外层Timeline只在进入/退出测量区间同步一次，内部用CUDA事件保留Adapter、投影、
Actor、WM、奖励、反向和优化器的时间。反向重算单独标记，嵌套区域为inclusive，
不能把所有条目直接相加。事件时间包含同一流等待CPU提交的间隙；外层同步墙钟
用于吞吐结论，Kineto独立短trace用于算子分析。未启用时只保留原profiler标签。
上下文变量使测试和正式入口共用实现，不改变张量、损失、autograd或训练随机流。
"""

from contextlib import contextmanager
from contextvars import ContextVar
import time
import torch

_active = ContextVar('adapter_timeline', default=None)
_phase = ContextVar('adapter_timing_phase', default=None)
# PyTorch自定义autograd可能在引擎线程执行，Python ContextVar不会自动传播。
# 每进程只有一个串行策略更新，使用进程级只读回退覆盖这些反向区域。
_global_active = None
_global_phase = 'forward'


@contextmanager
def phase(name):
    global _global_phase
    old = _global_phase
    _global_phase = name
    token = _phase.set(name)
    try:
        yield
    finally:
        _phase.reset(token)
        _global_phase = old


@contextmanager
def scope(name):
    timer = _active.get() or _global_active
    with torch.profiler.record_function(name):
        if timer is None:
            yield
        else:
            start = torch.cuda.Event(enable_timing=True) if timer.cuda else None
            end = torch.cuda.Event(enable_timing=True) if timer.cuda else None
            begin = time.perf_counter()
            if start is not None:
                start.record()
            try:
                yield
            finally:
                if end is not None:
                    end.record()
                timer.events.append((_phase.get() or _global_phase, name, start, end, time.perf_counter()-begin))


class Timeline:
    def __init__(self, device):
        self.device = torch.device(device)
        self.cuda = self.device.type == 'cuda'
        self.events = []

    def __enter__(self):
        global _global_active
        if self.cuda:
            torch.cuda.synchronize(self.device)
        self.begin = time.perf_counter()
        self.token = _active.set(self)
        self.previous_global = _global_active
        _global_active = self
        return self

    def __exit__(self, *args):
        global _global_active
        if self.cuda:
            torch.cuda.synchronize(self.device)
        self.seconds = time.perf_counter()-self.begin
        _active.reset(self.token)
        _global_active = self.previous_global

    def report(self):
        result = {}
        for phase_name, name, start, end, cpu in self.events:
            key = phase_name+'/'+name
            row = result.setdefault(key, dict(calls=0, stream_ms=0., cpu_dispatch_ms=0.))
            row['calls'] += 1
            row['stream_ms'] += start.elapsed_time(end) if self.cuda else cpu*1000
            row['cpu_dispatch_ms'] += cpu*1000
        return dict(synchronized_wall_seconds=self.seconds, regions=result,
                    semantics='inclusive_stream_events_not_additive; backward_recompute_separate')
