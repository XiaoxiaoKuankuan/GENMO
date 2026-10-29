"""绝对根XY与实际Goal软连续性的数值、可观测性及奖励梯度回归。

使用已有真实冻结SM权重夹具验证：heading坐标、t+1参考对齐、远距离Huber梯度、
左右锚点/halo/绝对时刻重叠、Actor Goal同一速度定义和参数有限差分。合成名义
动作只作数值回归，不冒充真实物理改善；不创建仿真或修改正式checkpoint。
"""

from types import SimpleNamespace
import torch

from supermimic.rotations import so3_exp
from gem.closedloop.reference_adapter.application import direct_reference_config
from gem.closedloop.reference_adapter.precision import (
    PrecisionObjective, absolute_root_xy, reference_continuity_values, REFERENCE_NAMES)
from gem.closedloop.reference_adapter.objective_diagnostics import reward_gradients
from gem.closedloop.reference_adapter.vector_control import detach_state
from test_direct_reference import direct_system


def test_root_heading_huber_far_gradient_and_z_independence():
    state = torch.zeros(1, 55, dtype=torch.float64)
    state[0, :3] = torch.tensor([1., 2., 99.])
    state.requires_grad_()
    target = SimpleNamespace(root_p=torch.zeros(1,3,dtype=torch.float64),
        root_R=so3_exp(torch.tensor([[0.,0.,torch.pi/2]],dtype=torch.float64)))
    cost, raw = absolute_root_xy(state, target)
    torch.testing.assert_close(raw['root_xy_heading_x_m'], torch.tensor([2.],dtype=torch.float64))
    torch.testing.assert_close(raw['root_xy_heading_y_m'], torch.tensor([-1.],dtype=torch.float64))
    gradient, = torch.autograd.grad(cost.sum(), state)
    torch.testing.assert_close(gradient[0,:3], torch.tensor([2.5,2.5,0.],dtype=torch.float64))
    assert torch.autograd.gradcheck(lambda x: absolute_root_xy(x, target)[0], (state,))


def test_no_false_world_position_feedback(system):
    _, _, control, z = direct_system(system)
    shifted = dict(z, state=z['state'].clone())
    shifted['state'][:, :2] += torch.tensor([2.,-3.])
    original, moved = control.propose(z), control.propose(shifted)
    torch.testing.assert_close(control.packed(control.fields(z)), control.packed(control.fields(shifted)), atol=0,rtol=0)
    torch.testing.assert_close(original['action'], moved['action'], atol=0, rtol=0)
    ref = control.bank.ref(z['song'], z['frame'])
    assert absolute_root_xy(shifted['state'], ref)[0] > absolute_root_xy(z['state'], ref)[0]


def test_continuity_zero_and_actual_goal_seams_halo(system):
    _, _, c, z = direct_system(system)
    zero = c.propose(z, zero=True)
    assert all(torch.equal(v, torch.zeros_like(v)) for v in reference_continuity_values(c,z,zero).values())
    z = dict(z, frame=torch.full_like(z['frame'], 5), plan_valid=torch.ones_like(z['plan_valid']))
    residual = (torch.randn(1,10,27)*.001).requires_grad_()
    left = (torch.randn(1,27)*.001).requires_grad_()
    anchor = (torch.randn(1,27)*.001).requires_grad_()
    z = dict(z, left=left, anchor=anchor, plan=torch.cat((residual[:, :1], residual[:, :9]),1).detach())
    def features(r):
        cur, fut, _, halo = c.reference(z,r)
        return reference_continuity_values(c,z,dict(current=cur,future=fut,residual=r,halo=halo))
    value = features(residual)
    assert value['joint_overlap'].abs().max() == 0
    assert value['root_overlap'].abs().max() < 1e-8
    baseline = c.reference(dict(z,left=torch.zeros_like(left),anchor=torch.zeros_like(anchor)), torch.zeros_like(residual))
    current,future,_,_ = c.reference(z,residual)
    expected = torch.cat((current.qd[:,None],future.qd),1)-torch.cat((baseline[0].qd[:,None],baseline[1].qd),1)
    torch.testing.assert_close(value['joint_velocity'], expected)
    torch.testing.assert_close(value['joint_acceleration'], expected.diff(dim=1)/.02)
    # 第一接缝的两侧历史锚点和第10帧halo均参与实际Goal导数成本。
    loss = value['joint_acceleration'].square().mean()+value['root_acceleration'].square().mean()
    grads = torch.autograd.grad(loss,(residual,left,anchor),retain_graph=True)
    assert all(g.isfinite().all() and g.norm()>0 for g in grads)
    assert grads[0][:,-1].norm()>0
    direction = torch.randn_like(residual); direction /= direction.norm()
    eps=1e-5
    def objective(r):
        f=features(r)
        return f['joint_acceleration'].square().mean()+f['root_acceleration'].square().mean()
    fd=(objective(residual.detach()+eps*direction)-objective(residual.detach()-eps*direction))/(2*eps)
    torch.testing.assert_close((grads[0]*direction).sum(),fd,rtol=.02,atol=.05)


def test_new_reward_single_dt_and_frozen_actual_network_gradients(system):
    engine,_,c,z=direct_system(system)
    c.adapter.disable_root_xy=True
    settings=dict(weight=.4,root_xy_weight=.03,root_xy_scale=.2,
                  soft_reference_weights=dict.fromkeys(REFERENCE_NAMES,.001))
    cfg=direct_reference_config(dict(alternating=dict(precision=settings)))
    assert cfg['alternating']['precision']==settings
    assert cfg['alternating']['objective_version'].endswith('v9')
    c.precision=PrecisionObjective(**settings)
    with torch.no_grad(): c.adapter.net[-1].bias.normal_(std=.01)
    frozen=c.bundle.frozen_snapshot()
    with engine.dynamics.adapter_phase():
        p=c.propose(z)
        state=engine.dynamics.step(z['state'],p['action'],z['xi'])
        _,reward,raw,terms,*_=c.advance(z,p,state)
        expected,_=absolute_root_xy(state,c.bank.ref(z['song'],z['frame']+1))
        torch.testing.assert_close(terms['root_non_saturating'],-.02*.03*expected)
        torch.testing.assert_close(raw['root_non_saturating'],expected)
        grad,=torch.autograd.grad(reward.sum(),c.adapter.net[-1].bias)
        assert grad.isfinite().all() and grad.norm()>0
        bias=c.adapter.net[-1].bias
        saved=bias.detach().clone()
        direction=grad/grad.norm()
        def value(offset):
            with torch.no_grad():
                bias.copy_(saved+offset*direction)
                pp=c.propose(z)
                ss=engine.dynamics.step(z['state'],pp['action'],z['xi'])
                return c.advance(z,pp,ss)[1].sum()
        fd=(value(.001)-value(-.001))/.002
        with torch.no_grad(): bias.copy_(saved)
        torch.testing.assert_close(grad.norm(),fd,rtol=.04,atol=1e-4)
        # 成本开关不改变参考、Actor动作或原SM奖励。
        c.precision=None
        p2=c.propose(z)
        _,_,_,old,*_=c.advance(z,p2,state)
        torch.testing.assert_close(p['goal'],p2['goal'],atol=0,rtol=0)
        torch.testing.assert_close(p['action'],p2['action'],atol=0,rtol=0)
        torch.testing.assert_close(terms['sm'],old['sm'],atol=0,rtol=0)
    c.precision=PrecisionObjective(**settings)
    before=[v.detach().clone() for v in c.adapter.parameters()]
    report,gradients=reward_gradients(c,engine.dynamics,engine.critic,detach_state(z),4,microbatch=1)
    assert report['parameter_gradient_norms']['root_non_saturating']>0
    assert report['root_joint_gradient_cosine'] is not None
    assert all(g.isfinite().all() for g in gradients.values())
    assert all(torch.equal(p,q) for p,q in zip(before,c.adapter.parameters()))
    c.bundle.assert_unchanged(frozen)
