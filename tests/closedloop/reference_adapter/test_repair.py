"""失败方法修复的数值合同测试，不将合成状态或CPU测试称为真实物理收益。

用极端交替残差和跨周期提交验证参考速度/加速度上界及halo；真实SM权重验证零策略
Goal/Actor不变、SO(3)可微和有限差分。概率测试比较独立高斯权威密度，排除被关闭
根XY维度；GAE测试真实终止与行政截断。模型身份测试防止复用旧WM方向许可。
"""

from dataclasses import replace
import copy
import torch

from gem.closedloop.reference_adapter.reference_constraints import constrain_plan
from gem.closedloop.reference_adapter.real_returns import gae_returns
from gem.closedloop.reference_adapter.physical_ppo import gaussian_logprob, active_mask, ppo_update
from gem.closedloop.reference_adapter.parameter_audit import audit_matches
from gem.closedloop.reference_adapter.update_diagnostics import parameter_identity, fixed_probes, update_effect
from gem.closedloop.reference_adapter.music_reward import AudioTrack
from test_vector import vector


def constrained_system(system, count=4):
    engine, old = system
    old = replace(old, audio=AudioTrack.from_features(old.audio.features[:, :-10]))
    engine, old, control, z = vector((engine, old), count=count)
    control.adapter.disable_root_xy = True
    control.constraint_beta = .05
    return engine, control, z


def test_adversarial_constraints_across_committed_seam_and_halo():
    torch.manual_seed(2)
    bounds = torch.tensor([.03]*3+[.05]*3+[.1]*21, dtype=torch.float64)
    left = torch.zeros(32, 27, dtype=torch.float64)
    anchor = left.clone()
    beta, dt = .05, .02
    for step in range(300):
        proposal = torch.randint(2, (32, 10, 27)).double().mul(2).sub(1)*bounds
        proposal *= (-1)**step
        saved = anchor.clone()
        plan = constrain_plan(proposal, left, anchor, beta)
        halo = 2*plan[:, -1:]-plan[:, -2:-1]
        points = torch.cat((left[:, None], anchor[:, None], plan, halo), 1)
        v = points.diff(dim=1)/dt
        a = v.diff(dim=1)/dt
        assert (v.abs() <= 2*bounds*beta/dt+1e-10).all()
        assert (a.abs() <= 4*bounds*beta**2/dt**2+1e-10).all()
        torch.testing.assert_close(v[:, -1], v[:, -2])
        torch.testing.assert_close(anchor, saved)
        assert (plan.abs() <= bounds+1e-12).all()
        left, anchor = anchor, plan[:, 0]


def test_zero_constraints_preserve_real_SM_and_new_root_cost(system):
    engine, control, z = constrained_system(system)
    bounded = control.propose(z)
    control.constraint_beta = 1
    original = control.propose(z)
    torch.testing.assert_close(bounded['goal'], original['goal'], rtol=0, atol=0)
    torch.testing.assert_close(bounded['action'], original['action'], rtol=0, atol=0)
    control.constraint_beta = .05
    control.root_cost_weight, control.velocity_cost_weight = 1., .05
    next_state = z['state'].clone()
    next_state[:, 0] += 5
    next_state.requires_grad_()
    _, _, raw, weighted, *_ = control.advance(z, bounded, next_state)
    assert (raw['root_pos'] < 1e-20).all()
    grad, = torch.autograd.grad(weighted['root_non_saturating'].sum(), next_state)
    assert (grad[:, 0].abs() > .1).all()
    fields = control.packed(control.fields(z))
    shifted = dict(z, state=z['state'].clone())
    shifted['state'][:, :2] += 5
    torch.testing.assert_close(control.packed(control.fields(shifted)), fields, atol=0, rtol=0)


def test_independent_270_constraints_SO3_finite_difference(system):
    engine, control, z = constrained_system(system, 1)
    latent = (torch.randn(1, 270)*.2).requires_grad_()
    proposal = control.propose(z, latent=latent)
    assert proposal['residual'].shape == (1, 10, 27)
    assert proposal['goal'].shape == (1, 570)
    assert not proposal['residual'][..., :2].count_nonzero()
    target = torch.randn_like(proposal['goal'])
    value = (proposal['goal']*target).sum()
    grad, = torch.autograd.grad(value, latent)
    assert (grad.reshape(1, 10, 27).abs().sum(-1) > 0).all()
    # 约束后的变化量很小，float32在原Goal的常数项上发生消减；有限差分用float64。
    saved_reference = control.bank.reference
    control.bank.reference = type(saved_reference)(*(v.double() for v in saved_reference))
    z64 = {k: v.double() if v.is_floating_point() else v for k, v in z.items()}
    raw = latent.detach().double().requires_grad_()
    def goal64(x):
        plan = constrain_plan(control.adapter.decode(x), z64['left'], z64['anchor'], .05)
        return control.reference(z64, plan)[2]
    target64 = target.double()
    grad, = torch.autograd.grad((goal64(raw)*target64).sum(), raw)
    direction = torch.randn_like(raw)
    direction /= direction.norm()
    eps = 1e-4
    a = (goal64(raw.detach()+eps*direction)*target64).sum()
    b = (goal64(raw.detach()-eps*direction)*target64).sum()
    torch.testing.assert_close((a-b)/(2*eps), (grad*direction).sum(), atol=1e-7, rtol=1e-4)
    control.bank.reference = saved_reference
    diagnostics = control.reference_diagnostics(z, proposal)
    assert all(torch.isfinite(x).all() for x in diagnostics.values())
    nominal_before = control.bank.reference.q.clone()
    control.reference_diagnostics(z, control.propose(z, latent=latent.detach()*100))
    torch.testing.assert_close(control.bank.reference.q, nominal_before, atol=0, rtol=0)


def test_exact_latent_probability_with_noninvertible_XY_projection(system):
    engine, control, z = constrained_system(system)
    mask = active_mask(control.adapter, 'cpu')
    assert int(mask.sum()) == 250
    mean = torch.randn(4, 270, requires_grad=True)
    latent = torch.randn_like(mean)
    std = .2
    log = gaussian_logprob(latent, mean, std, mask)
    expected = torch.distributions.Independent(torch.distributions.Normal(mean[:, mask], std), 1).log_prob(latent[:, mask])
    torch.testing.assert_close(log, expected)
    changed = latent.clone(); changed[:, ~mask] += 100
    torch.testing.assert_close(gaussian_logprob(changed, mean, std, mask), log, rtol=0, atol=0)
    grad, = torch.autograd.grad(log.sum(), mean)
    assert grad[:, ~mask].count_nonzero() == 0


def test_gae_does_not_cross_real_reset_but_bootstraps_administrative_end():
    reward = torch.ones(3, 2)
    value = torch.zeros_like(reward)
    nxt = torch.ones_like(reward)*10
    done = torch.tensor([[False, False], [True, False], [False, False]])
    adv, targets = gae_returns(reward, value, nxt, done, gamma=.9, lam=1)
    assert adv[1, 0] == 1
    assert adv[2, 1] == 10
    torch.testing.assert_close(adv[0, 0], torch.tensor(10.9))
    assert not targets.requires_grad


def test_audit_requires_exact_WM_parameters_not_only_version(system):
    engine, control, z = constrained_system(system)
    report = dict(identity=dict(wm_version=engine.dynamics.model_version,
        wm_sha256=parameter_identity(engine.dynamics), adapter_candidate_sha256=parameter_identity(control.adapter)))
    assert audit_matches(report, engine.dynamics, control.adapter)
    with torch.no_grad():
        next(engine.dynamics.model.net.parameters()).flatten()[0] += .001
    assert not audit_matches(report, engine.dynamics, control.adapter)


def test_no_change_is_diagnosed_and_PPO_uses_no_WM_gradient(system):
    engine, control, z = constrained_system(system, 32)
    probes = fixed_probes(z)
    with torch.no_grad():
        before = control.propose(probes)
    report = update_effect(control, probes, before, torch.tensor(0.))
    assert not report['effective'] and report['diagnosis'] == 'parameter_unchanged'
    x = control.packed(control.fields(z)).detach()
    mean = control.adapter.net(x).detach()
    latent = mean+torch.randn_like(mean)*.2
    mask = active_mask(control.adapter, 'cpu')
    dataset = dict(probes=probes, x=x, latent=latent, old_logprob=gaussian_logprob(latent, mean, .2, mask),
                   advantage=torch.randn(len(x)), std=.2)
    frozen = engine.step.bundle.frozen_snapshot()
    optimizer = torch.optim.Adam(control.adapter.parameters(), lr=3e-5)
    result = ppo_update(control, optimizer, dataset, batch=16)
    assert result['optimizer_steps'] == 1 and result['likelihood_dimension'] == 250
    assert result['effective']
    assert all(p.grad is None for p in engine.dynamics.parameters())
    engine.step.bundle.assert_unchanged(frozen)
