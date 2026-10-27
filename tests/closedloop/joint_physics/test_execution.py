"""冻结真实SM资产的局部导数测试，使用明确标记的张量起点而非伪称Newton轨迹。

这里只检查reference→SM→WM→执行Loss链、H归一化和冻结参数/buffer；真实状态、同起点
Newton方向及学生有效更新由独立CLI验收。所有几何/速度均调用当前bridge。
"""

from dataclasses import replace
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4].parent.parent / "supermimic/SuperMimic"))
from gem.closedloop.joint_physics.actor import JointPhysicsActor
from gem.closedloop.joint_physics.config import build_adapter, read_config, ROOT
from gem.closedloop.joint_physics.contracts import Window
from gem.closedloop.joint_physics.execution import execute_window
from gem.closedloop.joint_physics.losses import normalized_window
from gem.closedloop.mb_control.contracts import ClosedLoopState, RequestTicket
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
    result = execute_window(window, prediction, logits, actor, adapter, cfg["execution_loss"])
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


def test_horizon_normalization():
    a = normalized_window([torch.tensor([2.])] * 8, .99)
    b = normalized_window([torch.tensor([2.])] * 16, .99)
    torch.testing.assert_close(a, b)
