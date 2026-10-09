"""保持原FP32算子与单链形状的采样条件编码CUDA图，默认不启用。

历史GRU有50次顺序状态更新，条件在一条20步链中只编码一次，但Python逐算子提交
仍可能占用时间。此模块只捕获已验证条件的纯张量编码段，不改变GRU顺序、音乐或
前缀公式，不合并不同链的数值归约，也不改变随机数。真实环境条件分别写入静态
输入并重放；输出必须克隆，不能让下一条链覆盖上一条的编码。

只有no_grad、eval、CUDA、FP32模式允许图执行；PPO/BC有梯度调用直接走原函数，
因此ConditionGraphCache仍保留正确编码梯度。图读取原参数存储，原位优化器更新
自动可见；替换参数/缓冲、改变guidance或输入形状时重新捕获。捕获费用单列，正式
启用前必须用真实八卡模型验证完整随机链逐元素一致、原概率门槛和全部梯度。
"""
from __future__ import annotations
from collections import OrderedDict
import time
import torch


class SamplingConditionGraph:
    def __init__(self, policy, *, max_shapes=4):
        self.policy = policy
        self.original = policy._encode_prepared
        self.graphs = OrderedDict()
        self.max_shapes = max_shapes
        self.storage = None
        self.capture_seconds = 0.
        self.replays = self.eager_gradient_calls = 0

    def __call__(self, adapted):
        device = adapted['known_x'].device
        if torch.is_grad_enabled() or device.type != 'cuda':
            self.eager_gradient_calls += int(torch.is_grad_enabled())
            return self.original(adapted)
        if self.policy.actor.training or torch.backends.cuda.matmul.allow_tf32 or torch.is_autocast_enabled():
            raise ValueError('Condition graph requires eval, strict FP32, and no autocast')
        if adapted['known_x'].shape[0] != 1:
            raise ValueError('Condition graph preserves the single-chain numeric contract')
        actor = self.policy.actor
        storage = (self.policy.guidance_scale, tuple((id(t), t.data_ptr(), t.dtype, t.device, tuple(t.shape))
            for t in (*actor.parameters(), *actor.buffers())))
        if storage != self.storage:
            self.graphs.clear()
            self.storage = storage
        key = tuple((name, tuple(t.shape), t.dtype, t.device) for name, t in sorted(adapted.items()))
        if key not in self.graphs:
            started = time.perf_counter()
            static = {name: t.detach().clone() for name, t in adapted.items()}
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    output = self.original(static)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = self.original(static)
            self.graphs[key] = (graph, static, output)
            self.capture_seconds += time.perf_counter()-started
            while len(self.graphs) > self.max_shapes:
                self.graphs.popitem(last=False)
        self.graphs.move_to_end(key)
        graph, static, output = self.graphs[key]
        for name, target in static.items():
            target.copy_(adapted[name])
        graph.replay()
        self.replays += 1
        return tuple(None if value is None else value.clone() for value in output)

    def report(self):
        return dict(contract='eager_fp32_condition_sampling_cuda_graph.v1',
            shapes=len(self.graphs), capture_seconds=self.capture_seconds, replays=self.replays,
            eager_gradient_calls=self.eager_gradient_calls, graph_backward=False)


def install_condition_sampling_graph(policy):
    if isinstance(policy._encode_prepared, SamplingConditionGraph):
        raise ValueError('Condition sampling graph already installed')
    wrapper = SamplingConditionGraph(policy)
    policy._encode_prepared = wrapper
    return wrapper
