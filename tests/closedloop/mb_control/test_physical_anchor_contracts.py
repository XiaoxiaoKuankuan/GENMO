"""真实接管新合同的纯张量回归，覆盖在途价值状态与参考覆盖的因果边界。

测试检查Critic在任意50Hz时刻读取完整在途请求、冻结价值参数仍保留输入梯度，
参考版本替换不修改后来覆盖点，以及旧长链报告不能授权短时训练。这里的张量时轴
只用于工程合同测试，不是机器人轨迹、Newton验收、物理方向或实际训练收益证据。
"""

from dataclasses import replace
import pytest
import torch

from gem.closedloop.mb_control.contracts import PendingGeneration, GeneratedPlan
from gem.closedloop.mb_control.critic import PlanAwareCritic
from gem.closedloop.mb_control.imagination import replace_plan_version
from gem.closedloop.mb_control.scheduler import Scheduler
from gem.closedloop.mb_control.trainer import frozen_parameters
from gem.closedloop.mb_control.validation import select_short_horizon


def test_pending_value_has_full_request_and_input_derivatives(tensor_state):
    scheduler = Scheduler(120)
    ticket = scheduler.reserve(tensor_state)
    noise = torch.randn(1, 60, 30, requires_grad=True)
    request_state = tensor_state.state.clone().requires_grad_(True)
    pending = PendingGeneration(replace(tensor_state, state=request_state), ticket, noise, 7)
    state = replace(tensor_state, tick=12, pending=True, pending_request=pending)
    state.validate()
    critic = PlanAwareCritic(scheduler, width=32, layers=1, heads=4)
    with frozen_parameters(critic):
        critic(state).sum().backward()
    assert noise.grad is not None and noise.grad.norm() > 0
    assert request_state.grad is not None and request_state.grad.norm() > 0
    assert all(p.grad is None for p in critic.parameters())
    with pytest.raises(ValueError, match="完整请求"):
        replace(state, pending_request=None).validate()


def test_overlay_keeps_later_plan_sources_constant(tensor_state):
    ticket = Scheduler(120).reserve(tensor_state)
    values = tensor_state.reference.world_qpos.clone().requires_grad_(True)
    plan = GeneratedPlan(ticket, values, torch.zeros(1, 60, 2))
    cache = tensor_state.reference.merge_plan(plan, 120, 240).detached()
    versions = cache.source_versions.clone()
    versions[-10:] = 2
    cache = replace(cache, source_versions=versions, version=2)
    result = replace_plan_version(cache, plan, 1)
    torch.testing.assert_close(result.world_qpos[:, -10:], cache.world_qpos[:, -10:], rtol=0, atol=0)
    result.world_qpos[..., 7:].sum().backward()
    assert values.grad[:, -10:].count_nonzero() == 0
    assert values.grad[:, ticket.prefix_frames:-10].abs().sum() > 0


def test_old_long_horizon_report_cannot_unlock_anchor_method():
    cfg = {"imagination": {"mode": "physical_anchor_short_bptt", "horizon_steps": "auto", "critic_bootstrap": True}}
    with pytest.raises(ValueError, match="短时"):
        select_short_horizon(cfg, {"supported_control_steps": 102, "passed": True})


def test_replay_microbatch_requires_distinct_current_version_anchors():
    from types import SimpleNamespace
    from gem.closedloop.mb_control.replay import RecentReplay
    replay = RecentReplay(4, 42)
    for version in (0, 1, 1):
        replay.items.append(SimpleNamespace(generation=SimpleNamespace(request=SimpleNamespace(actor_version=version))))
    batch = replay.sample(2, actor_version=1)
    assert len({id(s) for s in batch}) == 2
    with pytest.raises(ValueError, match="独立真实样本不足"):
        replay.sample(2, actor_version=0)
