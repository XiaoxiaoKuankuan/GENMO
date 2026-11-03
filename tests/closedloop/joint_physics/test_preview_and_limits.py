"""连续启动前缀与执行机械限位的数值回归，不运行正式训练或Newton。

预览测试复用小型原Stage1 Transformer、原codec及完整DDIM20，核对12帧连续输入与
逐坐标mask，确认首帧之后仍保留原动作变化且输入不会被原地修改。限位测试使用明确
角度与解析导数，覆盖上下界、安全余量、最坏关节、有限差分和32步统计；它们证明
约束公式和监控口径，不作为真实机器人或整曲执行成功证据。临时文件由pytest回收。
"""

import copy
import math
import json
from types import SimpleNamespace

import pytest
import torch

from gem.closedloop.joint_physics.checkpoint import check_restore_contract
from gem.closedloop.joint_physics.config import ROOT, read_config
from gem.closedloop.joint_physics.losses import execution_limit_diagnostics, joint_limit_constraint
from tools.eval.generate_stage1_music_dances import generate


def constraint_options():
    return dict(joint_limit_constraint=dict(enabled=True, margin_rad=math.pi/180,
                                           scale_rad=math.pi/180, weight=5.))


def test_continuous_preview_prefix_and_original_ddim20(actor_factory, tmp_path):
    actor, batch = actor_factory(motion_frames=60, starts=(0,), prefix=12)
    startup = actor.endecoder.codec.decode_to_canonical_qpos(batch['target_qpos30'])[0, :12].clone()
    # 明确构造连续移动的已知前缀，防止夹具恰好静止使“重复首帧”错误漏检。
    startup[:, 0] += torch.arange(12)*.002
    original = startup.clone()
    calls = []

    class Runtime:
        def sample(self, conditions, steps, guidance_scale, noise):
            assert steps == 20 and guidance_scale == 2.5
            calls.append({k: v.clone() for k, v in conditions.items()})
            return actor.sample(conditions, steps=steps, guidance_scale=guidance_scale, noise=noise)

    qpos, _, valid, windows = generate(actor, Runtime(), torch.zeros(66, 35), 66, 42,
        tmp_path, window_frames=60, prefix_frames=12, startup_qpos=startup)
    torch.testing.assert_close(qpos[:12], original, rtol=0, atol=0)
    torch.testing.assert_close(startup, original, rtol=0, atol=0)
    assert not torch.equal(qpos[0], qpos[11])
    assert not valid[:12].any() and valid[12:].all()
    assert len(calls) == 2 and all(w['prefix_max_abs_error'] == 0 for w in windows)
    assert not calls[0]['known_qpos30_mask'][:, 12:].any()
    assert not calls[0]['proprio_history_valid'][:, :-1].any()
    with pytest.raises(ValueError, match='连续启动前缀'):
        generate(actor, Runtime(), torch.zeros(66, 35), 66, 42, tmp_path,
                 window_frames=60, prefix_frames=12, startup_qpos=startup[:1])


def test_mechanical_limit_gradient_worst_joint_and_finite_difference():
    kin = SimpleNamespace(q_lo=torch.full((21,), -.17), q_hi=torch.full((21,), .17))
    cfg = constraint_options()
    q = torch.zeros(1, 21, requires_grad=True)
    with torch.no_grad():
        q[0, 0], q[0, 1] = .19, -.185
    penalty, excess = joint_limit_constraint(q, kin, cfg)
    grad, = torch.autograd.grad(penalty.sum(), q)
    assert torch.isfinite(grad).all() and grad[0, 0] > 0 and grad[0, 1] < 0
    assert torch.count_nonzero(grad) == 2
    assert penalty > ((.02+math.pi/180)/(math.pi/180))**2  # 最坏关节不除21。
    torch.testing.assert_close(excess[0, :2], torch.tensor([.02, .015]), atol=1e-7, rtol=0)
    direction = torch.zeros_like(q); direction[0, 0] = 1.
    eps = 1e-4
    plus = joint_limit_constraint(q.detach()+eps*direction, kin, cfg)[0]
    minus = joint_limit_constraint(q.detach()-eps*direction, kin, cfg)[0]
    torch.testing.assert_close((grad*direction).sum(), ((plus-minus)/(2*eps)).sum(), rtol=2e-4, atol=.01)
    assert joint_limit_constraint(torch.zeros_like(q), kin, cfg)[0].item() == 0
    near = torch.zeros_like(q); near[0, 0] = .165
    p, v = joint_limit_constraint(near, kin, cfg)
    assert p > 0 and not v.any()  # 机械限位以内的安全余量提前给梯度。


def test_h32_limit_diagnostics_and_explicit_continuation_contract():
    kin = SimpleNamespace(q_lo=torch.full((21,), -.17), q_hi=torch.full((21,), .17))
    states = torch.zeros(32, 1, 55); states[7, 0, 13] = .20
    sums, maxima = execution_limit_diagnostics(states, kin, constraint_options())
    torch.testing.assert_close(sums[:4], torch.tensor([1., 32., 1., 1.]))
    assert sums[5] == 1 and sums[6:].sum() == 0
    assert maxima[0] == pytest.approx(.03)
    old = read_config(ROOT/'configs/closedloop/joint_physics_formal_200k_gradshare30_8gpu_server2.yaml')
    new = read_config(ROOT/'configs/closedloop/joint_physics_formal_200k_joint_limits_8gpu_server2.yaml')
    old_contract = {k: old[k] for k in ('physics', 'execution_loss', 'training')}
    new_contract = {k: new[k] for k in old_contract}
    assert check_restore_contract(old_contract, new_contract, continue_objective=True) == 'add_execution_joint_limit_constraint'
    with pytest.raises(ValueError):
        check_restore_contract(old_contract, new_contract)
    bad = copy.deepcopy(new_contract); bad['physics']['horizon'] = 16
    with pytest.raises(ValueError):
        check_restore_contract(old_contract, bad, continue_objective=True)


def test_old_newton_report_only_reused_as_diagnostic(tmp_path):
    from gem.closedloop.joint_physics.execution import ValidatedSupport
    old = {'prediction_mode': 'denoised_x0', 'loss': {'hard_safety': True}}
    report = dict(schema='genmo.joint_physics.validation.v2', assets={}, supported_horizons=[32],
                  objective=old, forecast_passed=True,
                  state_envelope=dict(schema='height_gravity_body_velocity_q_qd_v1',
                                      minimum=[-1.]*52, maximum=[1.]*52, padding=0.),
                  action_min=[-1.]*21, action_max=[1.]*21, source_ids=['fixture'])
    path = tmp_path/'report.json'; path.write_text(json.dumps(report))
    new = copy.deepcopy(old); new['loss'].update(constraint_options())
    with pytest.raises(ValueError, match='绑定当前执行目标'):
        ValidatedSupport(path, {}, 32, 'cpu', new)
    diagnostic = ValidatedSupport(path, {}, 32, 'cpu', new, diagnostic_only=True)
    assert not diagnostic.objective_verified
    unrelated = copy.deepcopy(new); unrelated['target'] = 'changed_target'
    with pytest.raises(ValueError, match='绑定当前执行目标'):
        ValidatedSupport(path, {}, 32, 'cpu', unrelated, diagnostic_only=True)
