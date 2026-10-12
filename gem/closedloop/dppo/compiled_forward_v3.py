"""共享纯推理编译前向及显式重算反向的Stage10 v3数值候选。

采样、KL与PPO的实际前向全部调用同一个no_grad编译函数，避免旧AOT推理图与
训练图使用不同融合归约，同时不在采样/KL中创建整网络梯度图。PPO通过独立
autograd边界保存输入与参数版本，反向重算原张量网络并求VJP；条件梯度继续
回到ConditionGraphCache。BC真实train/dropout路径不进入该包装。

这是一条有额外反向重算成本的显式数值/显存候选，不宣称与旧FP32逐位相同。
必须比较全部梯度、Adam、输出、KL和完整工作量速度；若不合格或更慢就不采用。
编译仅覆盖纯denoiser，不覆盖RPC/文件/预算/物理循环，不自动回退。编译输入
先detach只发生在自定义forward内部，backward显式返回原条件与参数梯度，不能
将此当作可训练编码器永久detach。参数在前后向之间改变会被SavedTensor检测。
"""
from __future__ import annotations
import torch


def install_shared_forward(policy):
    module=policy.actor.denoiser
    if hasattr(module,'_stage10_eager_forward'):raise ValueError('Denoiser execution wrapper already installed')
    original=module.forward
    parameters=tuple(p for p in module.parameters() if p.requires_grad)
    def tensor_forward(xt,timesteps,condition,length):
        result=original(xt,timesteps,y={'f_cond':condition,'length':length},inputs={})
        return result['pred_x_start'],result['static_conf_logits']
    compiled=torch.compile(tensor_forward,fullgraph=True,dynamic=False,
        options={'epilogue_fusion':False,'max_autotune':False})

    class SharedForward(torch.autograd.Function):
        @staticmethod
        def forward(ctx,xt,timesteps,condition,length,*weights):
            ctx.set_materialize_grads(False)
            ctx.save_for_backward(xt,timesteps,condition,length,*weights)
            with torch.no_grad():return compiled(xt.detach(),timesteps,condition.detach(),length)
        @staticmethod
        def backward(ctx,action_gradient,contact_gradient):
            xt,timesteps,condition,length,*weights=ctx.saved_tensors
            with torch.enable_grad():
                x=xt.detach().requires_grad_(ctx.needs_input_grad[0])
                c=condition.detach().requires_grad_(ctx.needs_input_grad[2])
                action,contact=tensor_forward(x,timesteps,c,length)
                variables=[];positions=[]
                for pos,tensor in [(0,x),(2,c),*[(4+i,p) for i,p in enumerate(weights)]]:
                    if tensor.requires_grad:variables.append(tensor);positions.append(pos)
                outputs=[];incoming=[]
                for value,gradient in [(action,action_gradient),(contact,contact_gradient)]:
                    if gradient is not None:outputs.append(value);incoming.append(gradient)
                grads=torch.autograd.grad(outputs,variables,incoming,allow_unused=True)
            result=[None]*len(ctx.needs_input_grad)
            for pos,grad in zip(positions,grads):result[pos]=grad
            return tuple(result)

    def forward(xt,timesteps,y=None,inputs=None,**kwargs):
        if module.training:return original(xt,timesteps,y=y,inputs=inputs,**kwargs)
        if set(y or {})!={'f_cond','length'} or inputs or kwargs:raise ValueError('Shared compile accepts only pure Stage1 tensor inputs')
        action,contact=SharedForward.apply(xt,timesteps,y['f_cond'],y['length'],*parameters)
        return dict(pred_x_start=action,static_conf_logits=contact)
    module._stage10_eager_forward=original;module.forward=forward
    policy.numerical_execution.update(compiler='inductor_shared_true_nograd_forward_recomputed_vjp.v3',
        inference_requires_grad=False,backward_recomputes_forward=True,compile_implicit_fallback=False)
