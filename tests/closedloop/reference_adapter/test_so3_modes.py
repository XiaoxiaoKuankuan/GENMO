"""四种根姿态模式及活跃面QP的数学回归，不启动仿真或训练。

使用固定真实投影合同验证零输出、历史接缝、三次时间构造的精确自动微分，以及
活跃面KKT与双精度IPM的同目标一致性。模式改变只影响根旋转，非根通道和网络
270维接口不变；数值测试不能替代Server-1真实配对整歌评价。
"""

from types import SimpleNamespace
import pytest
import torch
from gem.closedloop.reference_adapter.reference_constraints import ReferenceProjection
from gem.closedloop.reference_adapter.so3_projection import _active_set_qp, _ipm_qp


def projector(mode):
    return ReferenceProjection(SimpleNamespace(joint_names=['joint']*21),
        torch.tensor([.03]*3+[.05]*3+[.1]*21, dtype=torch.float64),
        root_orientation_mode=mode, rotation_acceleration=6. if mode == 'relaxed_so3' else 3.)


@pytest.mark.parametrize('mode', ['strict_so3', 'relaxed_so3', 'smooth_so3', 'no_root_orientation'])
def test_zero_and_nonroot_contract(mode):
    p=projector(mode)
    raw=torch.zeros(2,10,27,dtype=torch.float64,requires_grad=True)
    history=torch.zeros(2,27,dtype=torch.float64)
    nominal=torch.eye(3,dtype=torch.float64).expand(2,13,3,3)
    out=p(raw,history,history,nominal)
    assert torch.equal(out,raw)
    assert p.certificate(out,history,history,nominal)['feasible'].all()
    gradient=torch.autograd.grad(out.sum(),raw)[0]
    torch.testing.assert_close(gradient[...,6:],torch.ones_like(gradient[...,6:]))
    if mode=='no_root_orientation':
        assert torch.count_nonzero(gradient[...,3:6])==0
        assert int(p.last_statistics['sqp_active_count'])==0


def test_cubic_fit_history_and_directional_derivative():
    torch.manual_seed(99);p=projector('smooth_so3')
    raw=(torch.randn(2,10,27,dtype=torch.float64)*.001).requires_grad_()
    left=(torch.randn(2,27,dtype=torch.float64)*.0001).requires_grad_()
    anchor=(torch.randn(2,27,dtype=torch.float64)*.0001).requires_grad_()
    out=p.smooth_rotation(raw,left,anchor)
    full=torch.cat((left[:,None,3:6],anchor[:,None,3:6],out[...,3:6]),1)
    torch.testing.assert_close(full.diff(n=4,dim=1),torch.zeros_like(full[:,4:]),atol=1e-14,rtol=0)
    probe=torch.randn_like(out)
    gs=torch.autograd.grad((probe*out).sum(),(raw,left,anchor))
    ds=[torch.randn_like(x)*.001 for x in (raw,left,anchor)]
    eps=1e-5
    plus=p.smooth_rotation(*[x+eps*d for x,d in zip((raw,left,anchor),ds)])
    minus=p.smooth_rotation(*[x-eps*d for x,d in zip((raw,left,anchor),ds)])
    torch.testing.assert_close(sum((g*d).sum() for g,d in zip(gs,ds)),((plus-minus)*probe).sum()/(2*eps))


def test_active_set_qp_certificate_and_ipm_equivalence():
    torch.manual_seed(33)
    matrix=torch.eye(30,dtype=torch.float64).expand(16,-1,-1).clone()
    lo=torch.full((16,30),-.2,dtype=torch.float64);hi=-lo
    target=torch.randn(16,30,dtype=torch.float64)
    # 一组使用准确热启动活跃面，其余仅部分面，可失败但绝不能误证书通过。
    warm=target.clamp(-.2,.2);warm[8:]=0
    x,good,_=_active_set_qp(target,lo,hi,matrix,warm)
    ipm,_,err=_ipm_qp(target,lo,hi,matrix,60)
    assert good[:8].all() and err.max()<1e-8
    # 盒约束有解析真解；IPM在边界附近以互补残差停止，原始变量误差可达sqrt(mu)。
    torch.testing.assert_close(x[good],target[good].clamp(-.2,.2),atol=1e-12,rtol=1e-12)
    assert ((x[good]-target[good]).square().sum(-1) <=
            (ipm[good]-target[good]).square().sum(-1)+1e-8).all()


def test_mode_budgets_cannot_silently_change():
    with pytest.raises(ValueError,match='预算'):
        ReferenceProjection(SimpleNamespace(joint_names=['joint']*21),torch.ones(27),
                            root_orientation_mode='strict_so3',rotation_acceleration=6.)


def test_hard_rotation_budget_and_switch_do_not_change_explicit_regularizer():
    from gem.closedloop.reference_adapter.precision import PrecisionObjective
    precision=PrecisionObjective(rotation_velocity_scale=.3,rotation_acceleration_scale=3.,
                                 rotation_acceleration_penalty=True)
    torch.manual_seed(84)
    v=torch.randn(2,12,27,dtype=torch.double,requires_grad=True)
    a=v.diff(dim=1)/.02
    results=[]
    for mode in ('strict_so3','relaxed_so3','smooth_so3','no_root_orientation'):
        p=projector(mode)
        vs,acs,mask=precision.reference_cost_scales(p)
        loss=(v/vs).square().mean()+((a/acs).square()*mask).mean()
        results.append((loss,torch.autograd.grad(loss,v,retain_graph=True)[0]))
    for loss,grad in results[1:]:
        torch.testing.assert_close(loss,results[0][0],rtol=0,atol=0)
        torch.testing.assert_close(grad,results[0][1],rtol=0,atol=0)
    # 即使关硬角加速度约束，显式奖励开关仍独立生效。
    p=ReferenceProjection(SimpleNamespace(joint_names=['joint']*21),torch.ones(27,dtype=torch.double),
                          constrain_root_angular_acceleration=False)
    _,acs,mask=precision.reference_cost_scales(p)
    assert p.acceleration_mask[3:6].sum()==0 and mask[3:6].sum()==3
    torch.testing.assert_close(acs[3:6],torch.full((3,),3.,dtype=torch.double))


def test_quality_stop_requires_consecutive_full_validation():
    from gem.closedloop.reference_adapter.alternating_entry import validation_degradation
    rules=dict(root_rmse_relative_increase=.25,jerk_relative_increase=.20,
        tracking_relative_drop=.02,completion_absolute_drop=.25,consecutive_full_evaluations=2)
    zero=dict(max_steps=None,mean_root_position_rmse_m=.1,mean_jerk_rms=100.,
              mean_full_horizon_tracking=.09,completion_rate=1.)
    bad=dict(zero,mean_root_position_rmse_m=.14)
    first=validation_degradation(None,zero,bad,rules)
    assert first['consecutive']==1 and not first['stop']
    second=validation_degradation(first,zero,bad,rules)
    assert second['stop']
    assert not validation_degradation(first,zero,zero,rules)['stop']
    assert validation_degradation(first,zero,dict(bad,max_steps=256),rules)==first


def test_relaxed_true_stationarity_uses_correct_nonnegative_dual_faces():
    import json
    from pathlib import Path
    case=json.loads((Path(__file__).parent/'fixtures/relaxed_dual_pruning.json').read_text())
    p=ReferenceProjection(SimpleNamespace(joint_names=case['joint_names']),torch.tensor(case['bounds']),**case['options'])
    raw,left,anchor,rotation=[torch.tensor(case[k]) for k in ('raw','left','anchor','nominal_rotation')]
    raw.requires_grad_()
    value=p(raw,left,anchor,rotation)
    assert p.certificate(value,left,anchor,rotation)['feasible'].all()
    assert p.last_statistics['sqp_stationarity_residual64'].max()<2e-6
    assert torch.autograd.grad(value.sum(),raw)[0].isfinite().all()
    p=p.double();raw=raw.detach().double().requires_grad_()
    left,anchor,rotation=left.double(),anchor.double(),rotation.double()
    torch.manual_seed(710);probe=torch.randn_like(raw);direction=torch.zeros_like(raw)
    direction[...,3:6]=torch.randn_like(direction[...,3:6])*.001
    value=p(raw,left,anchor,rotation);g=torch.autograd.grad((probe*value).sum(),raw)[0]
    eps=1e-4
    finite=((p(raw+eps*direction,left,anchor,rotation)-p(raw-eps*direction,left,anchor,rotation))*probe).sum()/(2*eps)
    torch.testing.assert_close((g*direction).sum(),finite,atol=2e-6,rtol=.01)
