"""服务器1八卡运行的补偿权重梯度候选底层回归。

比较行连续和转置view、尾维和抵消输入，检查FP32累加、当前非默认stream及自定义
算子的fake/AOT组织；这些是小算子诊断，不能替代真实rollout完整梯度/Adam/KL验收。
测试只使用LOCAL_RANK对应GPU，不在本地或CPU上假装完成Tensor Core验证。
"""
import os
import pytest
import torch
from gem.closedloop.dppo.compensated_gemm import compensated_gemm


@pytest.mark.parametrize('products',[3,6])
@pytest.mark.parametrize('transpose',[False,True])
def test_compensated_fp32_output_on_current_stream(products,transpose):
    if not torch.cuda.is_available():pytest.skip('Server1 CUDA required')
    device=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(device)
    stream=torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        a=torch.randn((113,257) if transpose else (257,113),device=device)
        if transpose:a=a.t()
        b=torch.randn(113,67,device=device)
        expected=(a.double()@b.double()).float()
        actual=compensated_gemm(a,b,products)
        assert actual.dtype==torch.float32
        torch.testing.assert_close(actual,expected,atol=3e-4 if products==3 else 4e-5,rtol=2e-5)
    stream.synchronize()


def test_compensated_weight_gradient_does_not_allocate_sample_weight_tensor():
    if not torch.cuda.is_available():pytest.skip('Server1 CUDA required')
    from gem.closedloop.dppo.batch_execution import _SampleLinear
    device=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(device)
    x=torch.randn(8,17,31,device=device,requires_grad=True)
    w=torch.randn(29,31,device=device,requires_grad=True)
    dy=torch.randn(8,17,29,device=device)
    value=_SampleLinear.apply(x,w,None,'cublas_bf16x6','sample_bmm')
    value.backward(dy)
    torch.testing.assert_close(w.grad,(dy.reshape(-1,29).double().t()@x.detach().reshape(-1,31).double()).float(),atol=3e-5,rtol=2e-4)
    assert w.grad.shape==w.shape and x.grad.shape==x.shape
