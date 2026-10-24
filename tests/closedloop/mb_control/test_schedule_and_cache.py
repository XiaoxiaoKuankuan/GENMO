"""缓存支持、晚到接管和覆盖梯度的合同测试源，未在本轮执行。

使用确定时间格和张量源点验证插值/导数halo造成的首次影响，不用mock仿真或回报。
对两次合并检查旧支持点保持与被覆盖尾部无直接梯度，区分影响与中心消费时间。
"""

from dataclasses import replace

import pytest
import torch

from gem.closedloop.mb_control.contracts import GeneratedPlan, RequestTicket, ReferenceExhausted
from gem.closedloop.mb_control.scheduler import Scheduler


def test_eager_requests_use_first_grid_after_resolution(tensor_state):
    ticket = Scheduler(120).reserve(tensor_state)
    assert Scheduler(120).next_request_tick(ticket, 120) == 120
    assert Scheduler(120).next_request_tick(ticket, 132) == 180
    assert Scheduler(120).next_request_tick(ticket, 0) == 60
    fixed = Scheduler(120, request_mode="fixed_interval")
    assert fixed.next_request_tick(ticket, 120) == 300
    assert fixed.next_request_tick(ticket, 324) == 360


def test_halo_changes_first_influence_and_center_consumption(tensor_state):
    ref = tensor_state.reference
    assert 140 in ref.dependency_support(0).tolist()
    assert ref.first_influence(0, 160) == 12
    assert ref.first_center_consumed(0, 160) == 144
    with pytest.raises(ReferenceExhausted):
        ref.read_current_future(1104)


def test_latency_not_visible_to_reservation_and_late_plan_rejected(tensor_state):
    scheduler = Scheduler(120)
    ticket = scheduler.reserve(tensor_state)
    assert ticket.predicted_ready_tick == 120
    assert ticket.first_mutable_source_tick == 260
    plan = GeneratedPlan(ticket, tensor_state.reference.world_qpos + .1, torch.zeros(1, 60, 2))
    before = tensor_state.reference.world_qpos.clone()
    committed, event = scheduler.commit(replace(tensor_state, tick=120, pending=True), plan, actual_ready_tick=119)
    assert event["accepted"] and event["first_influence_tick"] == 120
    torch.testing.assert_close(committed.reference.world_qpos[:, :13], before[:, :13], rtol=0, atol=0)
    late, event = scheduler.commit(replace(tensor_state, tick=132, pending=True), plan, actual_ready_tick=121)
    assert not event["accepted"] and late.reference is tensor_state.reference
    assert ticket == scheduler.reserve(tensor_state)
    with pytest.raises(ValueError):
        scheduler.commit(replace(committed, pending=True), plan, actual_ready_tick=119)


def test_two_commits_preserve_only_uncovered_plan_graph(tensor_state):
    first_values = tensor_state.reference.world_qpos.clone().requires_grad_(True)
    first_ticket = RequestTicket(0, 0, 120, 260, 120, 13, 0)
    first = tensor_state.reference.merge_plan(GeneratedPlan(first_ticket, first_values, torch.zeros(1, 60, 2)), 120, 240)
    second_values = first_values.detach().clone().requires_grad_(True)
    second_ticket = RequestTicket(1, 300, 420, 560, 420, 13, 1)
    second = first.merge_plan(GeneratedPlan(second_ticket, second_values, torch.zeros(1, 60, 2)), 420, 540)
    second.world_qpos[..., 7:].sum().backward()
    assert first_values.grad[:, 13:28, 7:].abs().sum() > 0
    assert first_values.grad[:, 28:, 7:].count_nonzero() == 0
    assert second_values.grad[:, 13:, 7:].abs().sum() > 0
    torch.testing.assert_close(tensor_state.reference.world_qpos[:, :13], second.world_qpos[:, :13], rtol=0, atol=0)
