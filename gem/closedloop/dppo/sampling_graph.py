"""只用于无梯度扩散采样的固定形状CUDA Graph，不替换PPO反向。

把原denoiser的同一组FP32算子捕获为GPU图，减少20步循环中重复的Python调度。
不使用编译器重排、TF32、混合精度或图反向；有梯度的PPO、BC调用原forward，
因此不会复用静态梯度缓冲。每个有效batch独立捕获，没有复制环境充当有效样本。
参数对象、状态键和冻结规则不改变；图直接读取原参数存储，原位Adam更新可见。
若参数/缓冲存储或精度改变，旧图失效。输出克隆后返回，防止下一次图重放改写
已交付给其他环境线程的去噪证据。缓存有界，启动捕获耗时须单列，不能算作稳态
提速。启用前必须对真实模型逐条链通过原零更新概率和完整输出数值检查。
"""
from __future__ import annotations
from collections import OrderedDict
import time
import torch


class SamplingDenoiserGraph:
    def __init__(self, module, *, max_shapes=8):
        self.module = module
        self.original = module.forward
        self.max_shapes = max_shapes
        self.graphs = OrderedDict()
        self.capture_seconds = 0.
        self.replays = self.eager_gradient_calls = 0
        self.storage = None

    def _storage(self):
        return tuple((id(t), t.data_ptr(), t.dtype, t.device, tuple(t.shape))
                     for t in (*self.module.parameters(), *self.module.buffers()))

    def __call__(self, x, timestep, y=None, inputs=None, **kwargs):
        if torch.is_grad_enabled() or x.device.type != 'cuda':
            self.eager_gradient_calls += int(torch.is_grad_enabled())
            return self.original(x, timestep, y=y, inputs=inputs, **kwargs)
        if self.module.training or inputs or kwargs or set(y) != {'f_cond', 'length'}:
            raise ValueError('Sampling graph requires the validated eval DPPO tensor interface')
        if torch.backends.cuda.matmul.allow_tf32 or torch.is_autocast_enabled():
            raise ValueError('Sampling graph requires strict FP32 execution')
        arguments = (x, timestep, y['f_cond'], y['length'])
        key = tuple((tuple(t.shape), t.dtype, t.device) for t in arguments)
        storage = self._storage()
        if storage != self.storage:
            self.graphs.clear(); self.storage = storage
        if key not in self.graphs:
            start = time.perf_counter()
            static = tuple(t.detach().clone() for t in arguments)
            stream = torch.cuda.Stream(device=x.device)
            stream.wait_stream(torch.cuda.current_stream(x.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    output = self.original(static[0], static[1],
                        y=dict(f_cond=static[2], length=static[3]), inputs={})
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = self.original(static[0], static[1],
                    y=dict(f_cond=static[2], length=static[3]), inputs={})
            self.graphs[key] = (graph, static, output)
            self.capture_seconds += time.perf_counter()-start
            while len(self.graphs) > self.max_shapes: self.graphs.popitem(last=False)
        self.graphs.move_to_end(key)
        graph, static, output = self.graphs[key]
        for target, source in zip(static, arguments): target.copy_(source)
        graph.replay(); self.replays += 1
        # Actor只消费这两个head；不可将静态输出视图交给异步采集线程。
        return {key: output[key].clone() for key in ('pred_x_start', 'static_conf_logits')}

    def report(self):
        return dict(contract='eager_fp32_sampling_cuda_graph.v1', shapes=[str(k) for k in self.graphs],
                    capture_seconds=self.capture_seconds, replays=self.replays,
                    eager_gradient_calls=self.eager_gradient_calls, graph_backward=False)


def install_sampling_graph(actor, *, max_shapes=8):
    if isinstance(actor.denoiser.forward, SamplingDenoiserGraph):
        raise ValueError('Sampling graph already installed')
    wrapper = SamplingDenoiserGraph(actor.denoiser, max_shapes=max_shapes)
    actor.denoiser.forward = wrapper
    return wrapper
