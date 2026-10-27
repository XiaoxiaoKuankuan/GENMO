"""冻结真实SM资产的局部导数测试，使用明确标记的张量起点而非伪称Newton轨迹。

这里只检查reference→SM→WM→执行Loss链、H归一化和冻结参数/buffer；真实状态、同起点
Newton方向及学生有效更新由独立CLI验收。所有几何/速度均调用当前bridge。
"""

from dataclasses import replace

import torch
import pytest

from gem.closedloop.joint_physics.actor import JointPhysicsActor
from gem.closedloop.joint_physics.config import build_adapter, read_config, ROOT
from gem.closedloop.joint_physics.contracts import Window
from gem.closedloop.joint_physics.contracts import PhysicsRejected
from gem.closedloop.joint_physics.execution import execute_window
from gem.closedloop.joint_physics.losses import execution_terms, normalized_window
from gem.closedloop.mb_control.contracts import ClosedLoopState, RequestTicket, ForecastContext
from gem.closedloop.mb_control.reference_buffer import ReferenceBuffer
from supermimic.mdp import flatten_state


def test_real_frozen_weights_execution_gradient_and_fd(actor_factory):
    cfg = read_config(ROOT / "configs/closedloop/joint_physics_from_scratch.yaml")
    adapter = build_adapter(cfg, "cpu")
    frozen = adapter.bundle.frozen_snapshot()
    base, batch = actor_factory(motion_frames=60, starts=(0,), prefix=12)
    actor = JointPhysicsActor(base)
    world = base.endecoder.codec.decode_to_canonical_qpos(batch["target_qpos30"])
    ticks = torch.arange(60)*20
    ref = ReferenceBuffer(ticks, world, torch.zeros(1, 60, 2), torch.zeros_like(ticks), adapter.runtime.kin, adapter.to_sm)
    current, _, _ = ref.read_current_future(0)
    state = flatten_state((current.root_p, current.root_R, current.root_v, current.root_w, current.q, current.qd))
    xi = torch.zeros(1, 25)
    xi[:, 0] = 1
    obs = adapter.runtime.observation(state, torch.zeros(1, 21), xi, current)
    z = ClosedLoopState(0, state, xi, torch.zeros(1, 21), obs[:, None].expand(-1, 10, -1),
        torch.zeros(1, 50, 48), torch.zeros(1, 50, dtype=torch.bool), ref, torch.zeros(1, 60, 35), 0, 1200)
    first = ref.first_influence(0, 240)
    ticket = RequestTicket(0, 0, 0, 240, first, 12, 0)
    window = Window(z, replace(z, tick=first), ticket, torch.tensor([[0., 0., float(base.endecoder.codec.default_root_height), 0.]]),
        ref, 8, "tensor_fixture_not_newton", {"old_plan_committed_tick": 0, "last_real_tick": 1188, "split": "train", "start_origin": "tensor_fixture_only"})
    prediction = base.endecoder.normalize(batch["target_qpos30"]).detach().requires_grad_()
    logits = torch.zeros(1, 60, 2)
    with pytest.raises(PhysicsRejected,match='degenerate_or_nonfinite_rotation') as invalid:
        execute_window(window, prediction*float('nan'), logits, actor, adapter, cfg['execution_loss'],
                       prediction_origin='gt_interface_only')
    assert invalid.value.details['action_origin']=='not_executed'
    assert invalid.value.details['prediction_origin']=='gt_interface_only'
    assert invalid.value.details['first_reject_tick']==window.request.tick
    # 原夹具是零膝角张量起点，WM下一步负膝角违反硬限位，必须拒绝，不能算学生成功。
    with pytest.raises(PhysicsRejected, match="executed_hard_joint_limit"):
        execute_window(window, prediction, logits, actor, adapter, cfg["execution_loss"])
    # 以下只保留旧接口求导对照。正式read_config禁止关闭硬检查；此结果无训练验收含义。
    cfg["execution_loss"]["hard_safety"] = False
    result = execute_window(window, prediction, logits, actor, adapter, cfg["execution_loss"])
    assert result["safety"].item() == 0
    violating = state.clone()
    violating[:, 13:34] = adapter.runtime.kin.q_hi + .2
    _, safety, diagnostics = execution_terms(violating, current, adapter.runtime.kin, cfg["execution_loss"])
    assert safety.item() == 0 and diagnostics["joint_limit_violation_rad"].item() > .1
    gradient, = torch.autograd.grad(result["loss"], prediction)
    assert torch.isfinite(gradient).all() and gradient.norm() > 0
    direction = torch.zeros_like(prediction)
    direction[:, 12:20, 10] = torch.sin(torch.linspace(0, torch.pi, 8))
    eps = .002
    with torch.no_grad():
        plus = execute_window(window, prediction+eps*direction, logits, actor, adapter, cfg["execution_loss"])["loss"]
        minus = execute_window(window, prediction-eps*direction, logits, actor, adapter, cfg["execution_loss"])["loss"]
    fd = (plus-minus)/(2*eps)
    automatic = (gradient*direction).sum()
    torch.testing.assert_close(automatic, fd, atol=1e-4, rtol=.03)
    adapter.bundle.train(True)
    actor.train()
    adapter.bundle.assert_unchanged(frozen)
    assert not adapter.bundle.training
    assert len(result["coverage"]["supported_source_ticks"]) < 60
    assert result["actions"].shape == (8, 1, 21)
    assert not torch.equal(result["actions"][0], result["actions"][-1])
    # 不加任何BC：学生真实去噪输出的执行Loss必须单独到达GENMO和Future投影。
    context = ForecastContext(torch.ones(1, 58), torch.ones(1, dtype=torch.bool), 0, first)
    student = actor.denoise_supervised(batch, [context], torch.tensor([50]), torch.randn(1,60,30),
                                      torch.zeros(1, dtype=torch.bool))
    execution = execute_window(window, student["pred_x_start"], student["static_conf_logits"], actor, adapter, cfg["execution_loss"])
    backbone, future = torch.autograd.grad(execution["loss"],
        (base.denoiser.add_cond_linear.weight, actor.future_encoder.output.weight))
    assert torch.isfinite(backbone).all() and backbone.norm() > 0
    assert torch.isfinite(future).all() and future.norm() > 0
    adapter.bundle.assert_unchanged(frozen)


def test_horizon_normalization():
    a = normalized_window([torch.tensor([2.])] * 8, .99)
    b = normalized_window([torch.tensor([2.])] * 16, .99)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(a, normalized_window([torch.tensor([2.])] * 64, .99))


def test_real_start_replay_gate_rejects_mismatch_and_nonfinite():
    """只测重放门禁，不把此假环境当作Newton证据；漂移时不得执行候选控制。"""
    from types import SimpleNamespace
    from gem.closedloop.joint_physics.evaluation import replay_to_start
    class ReplayFixture:
        state = torch.zeros(1,55)
        def reset(self,*args):
            pass
        def step_physics(self,*args):
            raise AssertionError('零时刻起点不应先执行控制')
        def observe(self):
            return {'state':self.state}
    env=ReplayFixture()
    window=SimpleNamespace(start=SimpleNamespace(state=torch.zeros(1,55),tick=0))
    episode={'initial_state':torch.zeros(55),'action':torch.empty(0,21)}
    assert replay_to_start(env,episode,window,atol=1e-5)==0.
    for value in (.01,float('nan')):
        env.state=torch.full((1,55),value)
        with pytest.raises(PhysicsRejected,match='newton_replay_start_mismatch') as error:
            replay_to_start(env,episode,window,atol=1e-5)
        assert error.value.details['action_origin']=='not_executed'
        assert error.value.details['first_reject_tick']==0
