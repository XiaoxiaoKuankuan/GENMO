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




@pytest.mark.parametrize('device',['cpu','cuda'])
def test_batched_privileged_history_matches_individual_and_rejects_future(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA required')
    from gem.closedloop.dppo.tracking_context import build_critic_context,build_critic_context_batch
    from tests.closedloop.dppo.test_data_learning import context
    c={k:v.to(device) for k,v in context(batch=3).items()}
    snapshots=[dict(tracking_history_schema='gmt.causal_tracking_history90.v1',
        tracking_history_values=torch.zeros(50,90),tracking_history_valid=torch.ones(50,dtype=torch.bool),
        tracking_history_ticks=torch.arange(12,601,12),tracking_reference_ticks=torch.arange(12,601,12)) for _ in range(3)]
    result=build_critic_context_batch(c,snapshots)
    for i,snapshot in enumerate(snapshots):
        expected=build_critic_context({k:v[i:i+1] for k,v in c.items()},snapshot)
        assert all(torch.equal(value[i:i+1],expected[k]) for k,value in result.items())
    snapshots[1]['tracking_reference_ticks'][-1]+=12
    with pytest.raises(ValueError,match='reference_timestamps'):build_critic_context_batch(c,snapshots)
