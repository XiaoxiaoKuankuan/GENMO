"""服务器2容量优化的存储所有权和真实CUDA Graph回归。

小网络独立验证梯度bucket视图不互相污染、非空Adam持续更新、图输出不会被随后
replay覆盖、参数原位更新被图读到、有梯度计算仍走可训练路径。CUDA用例需真实GPU，
没有GPU时明确跳过；这些单元测试不替代真实s595000的概率/梯度及物理规模验收。
"""
from types import SimpleNamespace
import copy
import pytest
import torch
from gem.closedloop.dppo.gradient_merge import BCGradientController,merge_global_gradients


def test_paired_bucket_views_preserve_nonempty_adam_and_unused_parameters():
    torch.manual_seed(25)
    model=torch.nn.Sequential(torch.nn.Linear(4,7),torch.nn.Linear(7,2))
    expected=copy.deepcopy(model)
    optim=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=0.)
    reference=torch.optim.AdamW(expected.parameters(),lr=.002,weight_decay=0.)
    controller=BCGradientController(initial=.1,mode='fixed')
    for step in range(3):
        optim.zero_grad(set_to_none=True);reference.zero_grad(set_to_none=True)
        ppo={n:torch.randn_like(p) for n,p in model.named_parameters()}
        bc={n:torch.randn_like(p) for n,p in model.named_parameters()}
        wanted={n:ppo[n]+.1*bc[n] for n in ppo}
        merge_global_gradients(model,ppo,bc,controller,bucket_bytes=128)
        for (name,p),q in zip(model.named_parameters(),expected.parameters()):
            torch.testing.assert_close(p.grad,wanted[name],atol=3e-7,rtol=1e-6)
            q.grad=wanted[name]
        optim.step();reference.step()
        for p,q in zip(model.parameters(),expected.parameters()):torch.testing.assert_close(p,q,atol=1e-7,rtol=0.)
    assert all(int(s['step'])==3 for s in optim.state.values())


@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires Server2 CUDA')
def test_cuda_graph_reads_current_weights_and_returns_owned_outputs():
    from gem.closedloop.dppo.cuda_graph_forward import install_inference_graph
    class Denoiser(torch.nn.Module):
        def __init__(self):super().__init__();self.linear=torch.nn.Linear(4,4,device='cuda')
        def forward(self,x,t,y=None,inputs=None):
            value=self.linear(x)+y['f_cond']+t[:,None,None]
            return dict(pred_x_start=value,static_conf_logits=value[...,:2])
    module=Denoiser().eval();original=module.forward
    policy=SimpleNamespace(actor=SimpleNamespace(denoiser=module),numerical_execution={})
    install_inference_graph(policy)
    x=torch.randn(5,3,4,device='cuda');t=torch.arange(5,device='cuda');y=dict(f_cond=torch.randn_like(x),length=torch.full((5,),3,device='cuda'))
    with torch.no_grad():
        for _ in range(4):first=module(x,t,y,inputs={})
        saved=first['pred_x_start'].clone()
        module.linear.weight.add_(.01)
        second=module(x+1,t,y,inputs={})
        torch.testing.assert_close(second['pred_x_start'],original(x+1,t,y,inputs={})['pred_x_start'],atol=0,rtol=0)
        torch.testing.assert_close(first['pred_x_start'],saved,atol=0,rtol=0)
    module(x,t,y,inputs={})['pred_x_start'].square().mean().backward()
    assert module.linear.weight.grad is not None
    assert policy.cuda_graph_counters['replays']>=3 and policy.cuda_graph_counters['gradient_calls']==1
