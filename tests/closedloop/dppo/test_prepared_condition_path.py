"""服务器1八卡上的已编码条件直达路径回归。

使用每rank独立CUDA设备，对比公共严格输入接口和缓存直达接口的动作、contact、
完整参数梯度及重复链索引。缓存仅在当前固定参数阶段有效，原环境输入在阶段末
核验版本；不能跨optimizer更新，也不能让更改后的旧输入冒充已编码输入。
此处小模型隔离缓存组织逻辑，真实模型梯度/Adam/概率验收使用独立封存rollout工具。
"""
import os
from types import SimpleNamespace
import pytest
import torch
from gem.closedloop.dppo.policy import DPPODiffusionPolicy
from gem.closedloop.dppo.tensor_cache import ConditionGraphCache
from gem.closedloop.dppo.execution_checks import policy_phase
from tests.closedloop.test_stage1_actor import actor_factory, _conditions, _activate_branches


def test_prepared_only_outputs_and_encoder_gradients(actor_factory):
    if not torch.cuda.is_available(): pytest.skip('Server1 eight CUDA ranks required')
    device=f"cuda:{int(os.environ['LOCAL_RANK'])}";torch.cuda.set_device(device)
    actor,batch=actor_factory(starts=(45,47));actor.to(device);_activate_branches(actor)
    policy=DPPODiffusionPolicy(actor,cfg_batch=True,numerical_layout='sample_matrix_bmm_fp32.v1',
                              precision_mode='fp32_fast',defer_checks=True)
    context={k:v.to(device) for k,v in _conditions(batch).items()}
    rows=[SimpleNamespace(context={k:v[i:i+1] for k,v in context.items()}) for i in range(2)]
    steps=torch.tensor([18,19],device=device)
    state=torch.randn(2,120,30,device=device)
    with policy_phase(policy):
        expected=policy.transition_parameters(context,state,steps)
        (expected['mean'].square().mean()+expected['contact_logits'].square().mean()).backward()
    gradients={n:p.grad.detach().clone() for n,p in actor.named_parameters() if p.grad is not None}
    actor.zero_grad(set_to_none=True)
    with policy_phase(policy):
        graph=ConditionGraphCache(policy);graph.prime(rows)
        actual=graph.transition(rows,state,steps)
        for name in expected:torch.testing.assert_close(actual[name],expected[name],atol=0,rtol=0)
        (actual['mean'].square().mean()+actual['contact_logits'].square().mean()).backward()
        graph.backward()
    for name,p in actor.named_parameters():
        if name in gradients:torch.testing.assert_close(p.grad,gradients[name],atol=3e-5,rtol=2e-4)
    with torch.no_grad(),policy_phase(policy):
        graph=ConditionGraphCache(policy);graph.prime(rows)
        repeated=graph.transition([rows[1],rows[0],rows[1]],state[[1,0,1]],steps[[1,0,1]])
        torch.testing.assert_close(repeated['mean'],expected['mean'][[1,0,1]],atol=1e-6,rtol=1e-5)
        graph.validate_inputs()
        rows[0].context['known_qpos30'].add_(1)
        with pytest.raises(ValueError,match='Condition source changed'):graph.validate_inputs()
