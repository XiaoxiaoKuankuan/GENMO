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


def test_sparse_SO3_derivatives_equal_full_30_dimension_autodiff():
    """局部6/9维精确组装必须与原30维Jacobian及非线性Hessian一致。"""
    from gem.closedloop.reference_adapter.so3_derivatives import build
    p, raw, left, anchor, rotation = fixture()
    x = (raw[..., 3:6]/(p.velocity[3:6]*.02)).transpose(1, 2).flatten(1)
    _, jac, hess = build(p, x)
    expected_jac = torch.func.vmap(torch.func.jacrev(p.angular_constraints, argnums=0))(x, left[:, 3:6], anchor[:, 3:6], rotation)
    torch.testing.assert_close(jac(x, left[:, 3:6], anchor[:, 3:6], rotation), expected_jac, atol=1e-10, rtol=1e-9)
    lam = torch.linspace(0, 1, expected_jac.shape[1], dtype=x.dtype)[None]
    def curvature(value, old, current, nominal, multiplier):
        return (p.angular_constraints(value, old, current, nominal)*multiplier).sum()
    expected_hess = torch.func.vmap(torch.func.hessian(curvature, argnums=0))(x, left[:, 3:6], anchor[:, 3:6], rotation, lam)
    torch.testing.assert_close(hess(x, left[:, 3:6], anchor[:, 3:6], rotation, lam), expected_hess, atol=1e-9, rtol=1e-8)


def test_SO3_axis_aligned_constraint_has_finite_curvature_backward():
    p,raw,left,anchor,rotation=fixture()
    rotation=torch.eye(3,dtype=raw.dtype).expand(1,13,3,3)
    raw.zero_();raw[...,5]=.01;raw.requires_grad_()
    value=p(raw,left,anchor,rotation)
    gradient=torch.autograd.grad(value.sum(),raw)[0]
    assert gradient.isfinite().all()


def test_all_eight_saved_failures_strict_budget_stationarity_and_gradients():
    import json
    from pathlib import Path
    fixture=json.loads((Path(__file__).parent/'fixtures/so3_failed_alternating_20261028.json').read_text())
    torch.manual_seed(912)
    for case in fixture['cases']:
        p=ReferenceProjection(SimpleNamespace(joint_names=fixture['joint_names']),torch.tensor(fixture['bounds']))
        raw,left,anchor,rotation=[torch.tensor(case[k]) for k in ('raw','left','anchor','rotation')]
        value=p(raw,left,anchor,rotation)
        certificate=p.certificate(value,left,anchor,rotation)
        assert certificate['feasible'].all(), (case['gpu'],certificate)
        assert float(certificate['angular_acceleration_excess'].max())<=0
        assert float(p.last_statistics['sqp_stationarity_residual64'].max())<1e-4
        assert int(p.last_statistics['sqp_iterations'].max())<20
        # 用双精度对求解映射做中央差分，避免FP32量化噪声掩盖真实导数。
        p=p.double()
        raw,left,anchor,rotation=[v.double() for v in (raw,left,anchor,rotation)]
        for v in (raw,left,anchor):v.requires_grad_()
        value=p(raw,left,anchor,rotation);probe=torch.randn_like(value)
        gradients=torch.autograd.grad((value*probe).sum(),(raw,left,anchor))
        for i,gradient in enumerate(gradients):
            direction=torch.zeros_like((raw,left,anchor)[i])
            direction[...,3:6]=torch.randn_like(direction[...,3:6])*.001
            plus=[v.detach().clone() for v in (raw,left,anchor)]
            minus=[v.detach().clone() for v in (raw,left,anchor)]
            epsilon=1e-4;plus[i]+=epsilon*direction;minus[i]-=epsilon*direction
            finite=((p(*plus,rotation)-p(*minus,rotation))*probe).sum()/(2*epsilon)
            torch.testing.assert_close((gradient*direction).sum(),finite,atol=2e-6,rtol=.005,
                                       msg=f'GPU{case["gpu"]}, field={i}')


def test_actual_32_sample_imagination_batch_reaches_kkt_stationarity():
    """真实批量反例：不能只按很小的SQP步长宣布求解完成。"""
    import json
    from pathlib import Path
    case=json.loads((Path(__file__).parent/'fixtures/so3_actual_batch_20261028.json').read_text())
    p=ReferenceProjection(SimpleNamespace(joint_names=case['joint_names']),torch.tensor(case['bounds']))
    raw,left,anchor,rotation=[torch.tensor(case[k]) for k in ('raw','left','anchor','rotation')]
    raw.requires_grad_()
    out=p(raw,left,anchor,rotation)
    assert p.certificate(out,left,anchor,rotation)['feasible'].all()
    assert float(p.last_statistics['sqp_stationarity_residual64'].max())<2e-6
    assert int(p.last_statistics['sqp_iterations'].max())<12
    assert torch.autograd.grad(out.sum(),raw)[0].isfinite().all()
    # CPU/GPU批处理会有小幅舍入差异；固定整批上下文并检验历史和提案的导数。
    p=p.double();raw=raw.detach().double().requires_grad_()
    left=left.double().requires_grad_();anchor=anchor.double().requires_grad_();rotation=rotation.double()
    out=p(raw,left,anchor,rotation);torch.manual_seed(729);probe=torch.randn_like(out)
    gradients=torch.autograd.grad((out*probe).sum(),(raw,left,anchor))
    for i,gradient in enumerate(gradients):
        direction=torch.zeros_like(gradient);direction[...,3:6]=torch.randn_like(direction[...,3:6])*.001
        values=[raw.detach(),left.detach(),anchor.detach()];plus=[x.clone() for x in values];minus=[x.clone() for x in values]
        epsilon=1e-4;plus[i]+=epsilon*direction;minus[i]-=epsilon*direction
        finite=((p(*plus,rotation)-p(*minus,rotation))*probe).sum()/(2*epsilon)
        torch.testing.assert_close((gradient*direction).sum(),finite,atol=2e-5,rtol=.01)
