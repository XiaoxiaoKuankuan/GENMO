"""执行平台诊断的统计回归，不加载模型或创建仿真。

验证参数梯度的分组、夹角和物理权重换算，以及临时捕获器在异常退出后恢复函数。
这些测试只保证诊断工具不会伪造统计或污染同进程函数；真实模型结论来自独立实跑产物。
"""

from types import SimpleNamespace

import pytest
import torch

from tools.audit_joint_physics_plateau import capture_terms, gradient_stats


def test_parameter_gradient_weight_and_conflict():
    named=[('future_encoder.weight',None),('stage1.denoiser.weight',None)]
    data=[torch.tensor([3.,4.]),torch.tensor([0.,2.])]
    physics=[-2*data[0],torch.tensor([2.,0.])]
    stats=gradient_stats(named,data,physics,.02)
    assert stats['future_encoder']['cosine']==pytest.approx(-1.)
    assert stats['future_encoder']['weighted_execution_to_data']==pytest.approx(.04)
    assert stats['stage1.denoiser']['cosine']==0.
    assert stats['all']['data_norm']==pytest.approx(29**.5)
    assert stats['all']['execution_norm']==pytest.approx(104**.5)


def test_capture_restores_on_failure():
    original=lambda: (torch.tensor([2.]),torch.tensor([0.]),{'joint':torch.tensor([3.])})
    candidate=lambda: 'old'
    module=SimpleNamespace(execution_terms=original,candidate_reference=candidate)
    with pytest.raises(RuntimeError):
        with capture_terms(module,'new') as rows:
            module.execution_terms()
            assert rows==[{'loss':2.,'joint':3.}]
            assert module.candidate_reference()=='new'
            raise RuntimeError('test')
    assert module.execution_terms is original and module.candidate_reference is candidate


def test_global_gradient_accumulation_with_unused_parameters():
    from tools.audit_joint_physics_global_gradients import accumulate
    total=[None,None]
    first=torch.tensor([2.,4.])
    accumulate(total,[first,None],.5)
    accumulate(total,[torch.tensor([4.,2.]),torch.tensor([6.])],.5)
    assert torch.equal(total[0],torch.tensor([3.,3.]))
    assert torch.equal(total[1],torch.tensor([3.]))
    assert torch.equal(first,torch.tensor([2.,4.]))
