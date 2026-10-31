"""联合训练分阶段计时：CPU用墙钟，CUDA用流内事件，按日志间隔统一读取。

不在每步计时后同步GPU。事件仅保存当前日志区间，记录日志时等待末事件并归约八个rank
的紧凑结果，同时报告最慢rank。计时器不参与RNG、模型或optimizer checkpoint；无CUDA时
仍可测CPU路径。真实磁盘读量来自进程/proc/self/io，不将mmap映射字节冒充磁盘读取。
"""
from collections import defaultdict
from pathlib import Path
import time

import torch
from torch import distributed as dist

STAGES = ('data_wait','sidecar_read','condition_prepare','host_to_device','network_forward',
          'stage1_loss','future_forward_loss','physical_forward_backward','backward_ddp',
          'optimizer_step','logging','sidecar_refresh','checkpoint_save','gradient_balance','total_step')


class Performance:
    def __init__(self, device):
        self.device = torch.device(device)
        self.cpu = defaultdict(float)
        self.events = []
        self.steps = 0

    def add(self, name, seconds):
        self.cpu[name] += seconds

    def call(self, name, function, *args, **kwargs):
        started = time.perf_counter()
        first = last = None
        if self.device.type == 'cuda':
            first, last = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            first.record()
        result = function(*args, **kwargs)
        if last is not None:
            last.record(); self.events.append((name, first, last))
        self.add(name, time.perf_counter()-started)
        return result

    def collect(self, rank, world_size, cache):
        cuda = defaultdict(float)
        if self.events:
            self.events[-1][2].synchronize()
            for name, first, last in self.events:
                cuda[name] += first.elapsed_time(last)/1000
        count = max(1,self.steps)
        io = {}
        if Path('/proc/self/io').exists():
            io = {key:int(value) for key,value in
                  (line.split(':') for line in Path('/proc/self/io').read_text().splitlines())}
        value = dict(rank=rank, steps=self.steps,
            cpu_seconds_per_step={key:self.cpu[key]/count for key in STAGES},
            cuda_seconds_per_step={key:cuda[key]/count for key in STAGES}, cache=cache,
            disk_bytes_read=io.get('read_bytes',0), disk_read_scope='whole_rank_process',
            memory_allocated=torch.cuda.memory_allocated(self.device) if self.device.type=='cuda' else 0)
        ranks = [value]
        if world_size > 1:
            ranks = [None]*world_size
            dist.all_gather_object(ranks,value)
        result = dict(ranks=ranks, slowest_rank=max(ranks,key=lambda r:r['cpu_seconds_per_step']['total_step'])['rank'],
                      max_step_seconds=max(r['cpu_seconds_per_step']['total_step'] for r in ranks))
        self.cpu.clear(); self.events.clear(); self.steps=0
        return result
