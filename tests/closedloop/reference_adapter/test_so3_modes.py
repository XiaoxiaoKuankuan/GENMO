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
