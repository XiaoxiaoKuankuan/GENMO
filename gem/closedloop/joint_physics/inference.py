"""只用于部署评价的FP32 CUDA Graph去噪器与请求计时。

捕获原Transformer单次前向，输入噪声/时间/音乐历史未来融合条件和长度每次重新拷入；
不替换网络参数、CFG公式、20步DDIM或codec。训练和需要导数的采样始终走原路径。
图仅绑定本次已冻结评价权重和固定批量，不能跨optimizer更新复用；不同形状单独捕获。
前瞻优化只消除固定旧参考下重复观测/trim，不改SM/WM算法。启用前须比较整链真实输出。
"""

import time
import torch


class GraphDenoiser:
    def __init__(self, base):
        self.base, self.graphs = base, {}

    def __call__(self, x, t, adapted, condition):
        if torch.is_grad_enabled() or self.base.training or not x.is_cuda:
            raise RuntimeError('CUDA Graph仅用于GPU eval/no_grad；训练保留原导数路径')
        x = self.base._constrain(x,adapted)
        length = adapted['future_valid'].sum(1).clamp_min(1)
        key = (tuple(x.shape),str(x.device),str(x.dtype))
        if key not in self.graphs:
            inputs = [value.clone() for value in (x,t,condition,length)]
            stream = torch.cuda.Stream(device=x.device)
            stream.wait_stream(torch.cuda.current_stream(x.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.base.denoiser(inputs[0],inputs[1],y={'f_cond':inputs[2],'length':inputs[3]},inputs={})
            torch.cuda.current_stream(x.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):
                result = self.base.denoiser(inputs[0],inputs[1],y={'f_cond':inputs[2],'length':inputs[3]},inputs={})
            self.graphs[key] = graph,inputs,result
        graph,inputs,result = self.graphs[key]
        for destination,value in zip(inputs,(x,t,condition,length)):
            destination.copy_(value)
        graph.replay()
        return {'pred_x_start':self.base._constrain(result['pred_x_start'],adapted),
                'static_conf_logits':torch.where(adapted['future_valid'][...,None],result['static_conf_logits'],0.)}


class GraphTrackingMonitor:
    """捕获原SM评价奖励中的FK，避免Python逐关节发射拖慢50Hz控制。

    仅评价且无梯度，原reward公式/参数/终止检查保持不变；所有状态、动作和实际参考每步
    拷入图。首次捕获和数值对照必须在正式计时之前完成，不把启动开销藏入控制时延。
    返回clone防止图重放改写上一控制步的统计。不会用于执行Loss或训练。
    """
    def __init__(self, adapter):
        self.adapter, self.config, self.graph = adapter, {}, None

    @torch.no_grad()
    def warmup(self, state, action, previous, reference):
        from torch.utils._pytree import tree_flatten
        if not state.is_cuda:
            raise ValueError('控制监控CUDA Graph需要目标GPU')
        began = time.perf_counter()
        eager = self.adapter.runtime.reward(state,action,previous,reference)
        leaves,self.input_spec = tree_flatten((state,action,previous,reference))
        if not all(isinstance(value,torch.Tensor) for value in leaves):
            raise ValueError('SM参考必须保持权威纯张量结构')
        self.inputs = [value.clone() for value in leaves]
        from torch.utils._pytree import tree_unflatten
        arguments = tree_unflatten(self.inputs,self.input_spec)
        stream = torch.cuda.Stream(device=state.device)
        stream.wait_stream(torch.cuda.current_stream(state.device))
        with torch.cuda.stream(stream):
            for _ in range(3):self.adapter.runtime.reward(*arguments)
        torch.cuda.current_stream(state.device).wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=stream):
            self.result = self.adapter.runtime.reward(*arguments)
        self.graph.replay()
        actual,_ = tree_flatten(self.result)
        expected,_ = tree_flatten(eager)
        error = max(float((a-b).abs().max()) for a,b in zip(actual,expected))
        if error>1e-6:raise ValueError('控制监控图与原SM奖励不一致')
        return {'seconds':time.perf_counter()-began,'max_abs_error':error,'precision':'fp32',
                'scope':'original_SM_reward_and_FK_only','before_physics_episode':True}

    def __call__(self, state, action, previous, reference, **unused):
        from torch.utils._pytree import tree_flatten
        if torch.is_grad_enabled() or self.graph is None:
            raise RuntimeError('控制监控图只能用于完成预热的无梯度评价')
        leaves,spec = tree_flatten((state,action,previous,reference))
        if spec!=self.input_spec or any(a.shape!=b.shape for a,b in zip(leaves,self.inputs)):
            raise ValueError('控制监控状态/参考形状变化，不能复用图')
        for destination,value in zip(self.inputs,leaves):destination.copy_(value)
        self.graph.replay()
        reward,components = self.result
        reward = reward.clone()
        return reward, {'sm_reward':reward,**{k:v.clone() for k,v in components.items()}}


def configure(actor, backend):
    if backend not in ('eager','fast_forecast','cuda_graph'):
        raise ValueError('未知推理后端')
    actor.fast_forecast = backend!='eager'
    actor.inference_denoise = GraphDenoiser(actor.stage1) if backend=='cuda_graph' else None


def timed_request(actor,adapter,state,ticket,noise):
    from gem.closedloop.mb_control.forecast import forecast_rollout
    def sync():
        if noise.is_cuda:torch.cuda.current_stream(noise.device).synchronize()
    sync()
    start = time.perf_counter()
    with torch.no_grad():
        context = forecast_rollout(state,adapter,ticket.first_influence_tick,
                                   reuse_current_observation=actor.fast_forecast).context
        sync(); middle = time.perf_counter()
        plan = actor.generate(state,ticket,context,noise)
        sync(); end = time.perf_counter()
    return plan,context,{'forecast_seconds':middle-start,'generation_decode_seconds':end-middle,'total_seconds':end-start}
