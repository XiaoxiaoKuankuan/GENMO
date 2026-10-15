"""纯推理denoiser的有界CUDA Graph候选，不改变扩散、随机采样或反向数学。

只捕获eval且no_grad的张量前向；PPO有梯度前向和Stage1 BC继续执行原函数。输入
使用真实批量形状，不填充或复制样本；同形状出现三次才捕获，最多保留两个图，
变长尾批可执行原前向。随机噪声仍由外层独立generator产生并写入静态输入。
图读取当前参数存储，optimizer原位更新后仍使用新权重；替换参数/缓冲区存储则
重建，禁止复用旧地址。返回独立输出，避免下一次replay覆盖此前动作或contact。

这是显式候选，图构建失败直接报错；不修改稳定参考核、不自动切换精度，不把
仅推理通过视为学习或物理通过。固定静态缓冲占用更多显存以减少CPU发射开销，
捕获数量、命中数和回退到预先声明的有梯度/冷形状路径均有可读统计。
"""
from collections import OrderedDict
import torch


def install_inference_graph(policy,*,maximum_graphs=2):
    module=policy.actor.denoiser
    if maximum_graphs<1 or hasattr(module,'_stage10_eager_forward'):
        raise ValueError('Invalid CUDA graph capacity or an existing execution wrapper')
    original=module.forward;graphs=OrderedDict();seen={}
    counters=dict(captures=0,replays=0,cold_calls=0,gradient_calls=0,evictions=0)
    def storage_signature():
        return tuple((id(v),v.data_ptr(),tuple(v.shape),v.dtype,v.device) for v in
                     (*module.parameters(),*module.buffers()))
    signature=storage_signature()
    def forward(xt,timesteps,y=None,inputs=None,**kwargs):
        nonlocal signature
        if torch.is_grad_enabled() or module.training:
            counters['gradient_calls']+=1
            return original(xt,timesteps,y=y,inputs=inputs,**kwargs)
        if not xt.is_cuda or set(y or {})!={'f_cond','length'} or inputs or kwargs:
            raise ValueError('CUDA graph inference requires pure CUDA Stage1 denoiser tensors')
        values=(xt,timesteps,y['f_cond'],y['length'])
        key=tuple((tuple(v.shape),v.dtype,v.device) for v in values)
        current=storage_signature()
        if current!=signature:graphs.clear();seen.clear();signature=current
        seen[key]=seen.get(key,0)+1
        if key not in graphs and seen[key]<3:
            counters['cold_calls']+=1
            return original(xt,timesteps,y=y,inputs=inputs,**kwargs)
        if key not in graphs:
            if len(graphs)>=maximum_graphs:graphs.popitem(last=False);counters['evictions']+=1
            static=tuple(v.detach().clone() for v in values)
            def call():return original(static[0],static[1],y={'f_cond':static[2],'length':static[3]},inputs={})
            stream=torch.cuda.Stream(device=xt.device)
            stream.wait_stream(torch.cuda.current_stream(xt.device))
            with torch.cuda.stream(stream):
                for _ in range(3):call()
            torch.cuda.current_stream(xt.device).wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):outputs=call()
            torch.cuda.current_stream(xt.device).wait_stream(stream)
            graphs[key]=(static,graph,outputs);counters['captures']+=1
        static,graph,outputs=graphs[key];graphs.move_to_end(key)
        for target,value in zip(static,values):target.copy_(value)
        graph.replay();counters['replays']+=1
        return {name:value.clone() for name,value in outputs.items()}
    module._stage10_eager_forward=original;module.forward=forward
    policy.cuda_graph_counters=counters
    policy.numerical_execution.update(cuda_graph=dict(contract='eager_math_eval_nograd_static_buffers.v1',
        maximum_graphs=maximum_graphs,capture_after_calls=3,grad_path='eager',output_storage='independent_clone'))
