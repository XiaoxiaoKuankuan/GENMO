"""直接十帧参考的行为与真实冻结网络梯度回归。

输入来自仓库固定SuperMimic权重、RobotSpec和合成名义缓存，验证旧配置中的导数
预算不会重新启用投影或惩罚；明显超过旧加速度预算的有界提案须原样构造Goal570。
使用真实Actor/WM检查输入梯度、原物理终止及已删除奖励的缺席，不用合成状态宣称
整歌物理改善。Server-1真实复查使用独立结果目录，既有checkpoint和旧结果只读。
"""

import copy
from dataclasses import replace

import torch

from gem.closedloop.reference_adapter.application import direct_reference_config
from gem.closedloop.reference_adapter.precision import PrecisionObjective
from gem.closedloop.reference_adapter.paired_evaluation import _feasible_evaluation_proposal
from gem.closedloop.reference_adapter.music_reward import AudioTrack
from test_vector import vector


def direct_system(system):
    engine, old = system
    old = replace(old, audio=AudioTrack.from_features(old.audio.features[:, :-10]))
    return vector((engine, old), count=1)


def test_old_checkpoint_settings_cannot_restore_deleted_constraints():
    old = dict(rewards=dict(music=.001, activity=.03, smooth=100., magnitude=100.,
                           overlap=100., reference_limit=100.),
               sm_mdp=dict(term_height=.2, term_tilt=.6),
               alternating=dict(projection=dict(rotation_acceleration=3., joint_acceleration=10.),
                   precision=dict(weight=.4, acceleration_weight=100., velocity_weight=100.,
                                  action_weight=100., rotation_acceleration_penalty=True),
                   proposal_feasibility_weight=100., projection_recovery_attempts=2,
                   max_consecutive_projection_rejections=5, compile_so3_derivatives=True,
                   root_orientation=True, root_xy=False),
               training=dict(adapter_lr=1e-5))
    saved = copy.deepcopy(old)
    cfg = direct_reference_config(old)
    assert old == saved
    assert cfg['rewards'] == dict(music=.001, activity=.03)
    assert cfg['alternating']['precision'] == dict(weight=.4)
    assert 'projection' not in cfg['alternating']
    assert 'proposal_feasibility_weight' not in cfg['alternating']
    assert cfg['sm_mdp'] == old['sm_mdp']
    assert cfg['training'] == old['training']


def test_large_reference_derivatives_do_not_reject_or_modify_goal(system):
    _, _, control, z = direct_system(system)
    control.adapter.disable_root_xy = True
    control.precision = PrecisionObjective()
    with torch.no_grad():
        bias = control.adapter.net[-1].bias.view(10, 27)
        bias[::2] = .8
        bias[1::2] = -.8
    proposal = control.propose(z)
    assert control.projection is None and control.constraint_beta == 1.
    torch.testing.assert_close(proposal['residual'], proposal['raw_residual'], atol=0, rtol=0)
    assert proposal['goal'].shape == (1, 570)
    assert 'zero_action' not in proposal
    d = torch.cat((z['left'][:, None], z['anchor'][:, None], proposal['residual']), 1)
    acceleration = d.diff(n=2, dim=1)/.02**2
    assert acceleration[..., 6:].abs().max() > 100.
    assert acceleration[..., 3:6].abs().max() > 3.
    ids, _, accepted, failures = _feasible_evaluation_proposal(control, z, torch.arange(1), zero=False)
    assert ids.tolist() == [0] and not failures
    torch.testing.assert_close(accepted['goal'], proposal['goal'], atol=0, rtol=0)


def test_execution_objective_retains_gradients_without_reference_penalties(system):
    engine, _, control, z = direct_system(system)
    control.adapter.disable_root_xy = True
    control.precision = PrecisionObjective()
    with torch.no_grad():
        control.adapter.net[-1].bias.normal_(std=.03)
    frozen = control.bundle.frozen_snapshot()
    with engine.dynamics.adapter_phase():
        proposal = control.propose(z)
        nxt = engine.dynamics.step(z['state'], proposal['action'], z['xi'])
        _, reward, raw, terms, _, _, _ = control.advance(z, proposal, nxt)
        removed = {'action_deviation', 'added_reference_velocity', 'added_reference_acceleration',
                   'reference_velocity', 'proposal_feasibility', 'magnitude', 'smooth',
                   'overlap', 'reference_limit'}
        assert not removed.intersection(terms)
        assert not removed.intersection(raw)
        assert {'sm', 'music', 'activity', 'precision_joint_position', 'precision_joint_velocity'} <= terms.keys()
        gradient, = torch.autograd.grad(reward.sum(), control.adapter.net[-1].bias)
    assert gradient.isfinite().all() and gradient.norm() > 0
    assert all(p.grad is None for p in control.bundle.actor.parameters())
    control.bundle.assert_unchanged(frozen)
