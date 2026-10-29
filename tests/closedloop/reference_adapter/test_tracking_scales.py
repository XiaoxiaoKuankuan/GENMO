"""用户指定SM奖励尺度与真实失败阈值的数值回归。

使用真实冻结SM权重及RobotSpec生成的参考，检查指数尺度、原六项权重和dt只乘
一次；独立配置实例不得污染原MDP或Actor。高度/重力投影z阈值直接复用原SM
hard/soft termination，覆盖严格大于边界、原始参考相对性及输入梯度。此处
只有张量数值证据，不把合成状态当作真实物理改善。
"""

import copy
from dataclasses import asdict

import pytest
import torch

from supermimic.mdp import flatten_state, unflatten_state
from supermimic.rotations import so3_exp
from gem.closedloop.reference_adapter.rewards import configure_sm_mdp, sm_mdp_contract
from gem.closedloop.reference_adapter.reference import nominal_window


OVERRIDES = dict(root_pos_std=.2, root_ori_std=.3, body_pos_std=.1,
                 body_ori_std=.15, term_height=.2, term_tilt=.6)


def test_tracking_scales_keep_weights_dt_and_input_gradients(system):
    engine, z = system
    original = engine.step.runtime
    before = asdict(original.mdp.cfg)
    runtime = copy.copy(original); runtime.mdp = copy.copy(original.mdp)
    configure_sm_mdp(runtime, OVERRIDES)
    assert asdict(original.mdp.cfg) == before
    assert runtime.mdp.cfg is not original.mdp.cfg
    assert runtime.mdp.cfg.body_linvel_std == before['body_linvel_std']
    assert runtime.mdp.cfg.body_angvel_std == before['body_angvel_std']
    nominal, _, _ = nominal_window(z.nominal_reference, 12, runtime.kin)
    p, R, v, w, q, qd = unflatten_state(z.executed_state)
    state = flatten_state((p+.015, R@so3_exp(torch.tensor([[.03, -.02, .01]])), v, w, q+.02, qd+.1))
    state = state.detach().requires_grad_()
    action = z.previous_action+.01
    reward, terms = runtime.reward(state, action, z.previous_action, nominal, diagnostics=True)
    expected_track = sum(weight*torch.exp(-terms['error_'+name]/getattr(runtime.mdp.cfg, name+'_std')**2)
                         for name, weight in sm_mdp_contract(runtime)['weights'].items())*.02
    torch.testing.assert_close(terms['track'], expected_track)
    expected = expected_track+.02*(runtime.mdp.cfg.w_action_rate*terms['action_rate']+
                                    runtime.mdp.cfg.w_joint_limit*terms['joint_limit'])
    torch.testing.assert_close(reward, expected)
    gradient, = torch.autograd.grad(reward.sum(), state, retain_graph=True)
    formula_gradient, = torch.autograd.grad(expected.sum(), state)
    torch.testing.assert_close(gradient, formula_gradient)
    assert torch.isfinite(gradient).all() and gradient.norm() > 0
    assert all(not p.requires_grad for p in engine.step.bundle.actor.parameters())


def test_original_reference_height_tilt_failure_and_soft_threshold(system):
    engine, z = system
    runtime = copy.copy(engine.step.runtime); runtime.mdp = copy.copy(runtime.mdp)
    configure_sm_mdp(runtime, OVERRIDES)
    nominal, _, _ = nominal_window(z.nominal_reference, 12, runtime.kin)
    n = 7
    p = torch.zeros(n, 3, dtype=torch.double)
    p[:, 2] = torch.tensor([.1999, .2, .2001, 0., 0., 0., 0.], dtype=torch.double)
    R = torch.eye(3, dtype=torch.double).expand(n, -1, -1).clone()
    angles = torch.acos(torch.tensor([.4001, .4, .3999], dtype=torch.double))
    rv = torch.zeros(3, 3, dtype=torch.double); rv[:, 0] = angles
    R[3:6] = so3_exp(rv)
    zero = torch.zeros(n, 3, dtype=torch.double)
    joints = torch.zeros(n, 21, dtype=torch.double)
    nominal = nominal._replace(root_p=zero, root_R=torch.eye(3, dtype=torch.double).expand(n, -1, -1))
    S = (p, R, zero, zero, joints, joints)
    hard = runtime.mdp.termination(S, None, ref=nominal)
    assert hard[[0, 1, 2, 3, 5, 6]].tolist() == [False, False, True, False, True, False]
    soft = runtime.mdp.soft_termination(S, None, ref=nominal)
    torch.testing.assert_close(soft[[1, 4]], torch.full((2,), .5, dtype=torch.double), atol=3e-5, rtol=0)
    # 同样倾斜的原名义参考不会被错误地拿世界竖直方向作为目标。
    assert not runtime.mdp.termination(S, None, ref=nominal._replace(root_p=p, root_R=R)).any()


@pytest.mark.parametrize('bad', [{'root_pos_weight': .2}, {'term_height': 0.}, {'body_ori_std': float('nan')}])
def test_invalid_scale_configuration_is_rejected_at_entry(system, bad):
    with pytest.raises(ValueError, match='SM误差尺度'):
        configure_sm_mdp(system[0].step.runtime, bad)
