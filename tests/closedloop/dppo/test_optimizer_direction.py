"""整批方向诊断的独立解析验证：内积符号、梯度恢复与异常边界。

使用CPU小线性层和显式已知梯度，不运行物理、不加载正式模型。验证额外整批反向
不会替换已有小批梯度对象，Adam状态与参数保持不变；真实接受增量的一阶目标预测
与独立向量公式一致，并验证反向异常时原梯度仍恢复。这些测试不代表物理收益。
"""
from types import SimpleNamespace
import pytest
import torch
from gem.closedloop.dppo.optimizer_direction import FullRolloutDirection


def test_exact_gradient_geometry_and_original_gradient_storage_restored():
    actor=torch.nn.Linear(2,1,bias=False,dtype=torch.float32)
    actor.weight.grad=torch.tensor([[3.,4.]])
    gradient=actor.weight.grad
    before={name:p.detach().clone() for name,p in actor.named_parameters()}
    def backward():actor.weight.grad=torch.tensor([[0.,2.]])
    probe=FullRolloutDirection.capture(actor,SimpleNamespace(rank=0,sum_gradients=lambda _:None),backward)
    assert actor.weight.grad is gradient
    assert probe.geometry['cosine_minibatch_full_gradient']==pytest.approx(.8)
    with torch.no_grad():actor.weight.add_(torch.tensor([[.25,-.5]]))
    report=probe.finish(before)
    assert report['modules']['all']['linearized_full_objective_gain']==pytest.approx(1.)
    assert report['modules']['all']['cosine_update_full_objective_gradient']==pytest.approx(2/(5**.5))
    assert not probe.gradients


def test_backward_exception_restores_original_gradient():
    actor=torch.nn.Linear(1,1);actor.weight.grad=torch.ones_like(actor.weight)
    gradient=actor.weight.grad
    def fail():
        actor.weight.grad=torch.full_like(actor.weight,9.)
        raise RuntimeError('expected')
    with pytest.raises(RuntimeError,match='expected'):
        FullRolloutDirection.capture(actor,SimpleNamespace(rank=0,sum_gradients=lambda _:None),fail)
    assert actor.weight.grad is gradient and actor.bias.grad is None
