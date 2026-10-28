"""2026-10-28根SO(3)投影回归：输出精度、安全余量、非线性隐式导数和历史梯度。

这些固定CPU数值夹具验证真实求解映射的有限差分，不以梯度非零或测试通过代替
真实物理跟踪收益。历史锚点必须进入隐式KKT参数导数；已可行零输出严格保持零。
"""
import torch
from types import SimpleNamespace
from supermimic.rotations import so3_exp
from gem.closedloop.reference_adapter.reference_constraints import ReferenceProjection


def fixture(dtype=torch.float64):
    bounds=torch.tensor([.03]*3+[.05]*3+[.1]*21,dtype=dtype)
    p=ReferenceProjection(SimpleNamespace(joint_names=['joint']*21),bounds,iterations=1200)
    time=torch.arange(13,dtype=dtype)*.02
    rotation=so3_exp(torch.stack((.2*torch.sin(time*5),.1*torch.cos(time*7),time*2),-1))[None]
    raw=torch.zeros(1,10,27,dtype=dtype)
    raw[...,3:6]=torch.tensor([.008,-.007,.006],dtype=dtype)
    left=torch.zeros(1,27,dtype=dtype);anchor=torch.zeros_like(left)
    return p,raw,left,anchor,rotation


def test_SO3_implicit_kkt_matches_proposal_and_history_finite_difference():
    p,raw,left,anchor,rotation=fixture()
    raw.requires_grad_();left.requires_grad_();anchor.requires_grad_()
    output=p(raw,left,anchor,rotation)
    probe=torch.linspace(.3,1.,output.numel(),dtype=output.dtype).reshape_as(output)
    gradients=torch.autograd.grad((output*probe).sum(),(raw,left,anchor))
    torch.manual_seed(340)
    for field,gradient in enumerate(gradients):
        direction=torch.zeros_like((raw,left,anchor)[field]);direction[...,3:6]=torch.randn_like(direction[...,3:6])*.001
        plus=[v.detach().clone() for v in (raw,left,anchor)];minus=[v.detach().clone() for v in (raw,left,anchor)]
        epsilon=1e-4;plus[field]+=epsilon*direction;minus[field]-=epsilon*direction
        finite=((p(*plus,rotation)-p(*minus,rotation))*probe).sum()/(2*epsilon)
        torch.testing.assert_close((gradient*direction).sum(),finite,atol=2e-5,rtol=.025)
    assert float(p.last_statistics['sqp_stationarity_residual64'].max())<.002


def test_SO3_cast_float32_stays_inside_actual_acceleration_budget():
    p,raw,left,anchor,rotation=fixture(torch.float32)
    result=p(raw,left,anchor,rotation)
    report=p.certificate(result,left,anchor,rotation)
    assert report['feasible'].all()
    assert float(report['angular_acceleration_excess'].max())<=0
    assert int(p.last_statistics['sqp_active_count'])==1
    assert float(p.last_statistics['so3_float32_excess'].max())<=0


def test_SO3_axis_aligned_constraint_has_finite_curvature_backward():
    p,raw,left,anchor,rotation=fixture()
    rotation=torch.eye(3,dtype=raw.dtype).expand(1,13,3,3)
    raw.zero_();raw[...,5]=.01;raw.requires_grad_()
    value=p(raw,left,anchor,rotation)
    gradient=torch.autograd.grad(value.sum(),raw)[0]
    assert gradient.isfinite().all()
