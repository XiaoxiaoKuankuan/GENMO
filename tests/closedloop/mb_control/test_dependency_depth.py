"""最长WM调用链计数的精确回归，使用缓存真支持集而非物理结果替身。

构造提前到达且安全裕量仍保护未来源点的计划，验证提交不会提前增加执行状态深度；
首次真正消费的速度halo会引入计划依赖。另验证GENMO前缀与终点Critic缓存可以提前
读取未来计划，必须独立计入，不得为缩短时域漏记这些依赖。测试不评价WM物理质量。
"""

from dataclasses import replace

import torch

from gem.closedloop.mb_control.contracts import GeneratedPlan
from gem.closedloop.mb_control.dependency import ReferenceDependencies
from gem.closedloop.mb_control.scheduler import Scheduler


def test_early_commit_depth_starts_at_actual_consumption(tensor_state):
    scheduler = Scheduler(360, latency_margin_ticks=36)
    ticket = scheduler.reserve(tensor_state)
    deps = ReferenceDependencies(tensor_state.reference)
    plan = GeneratedPlan(ticket, tensor_state.reference.world_qpos.clone(), torch.zeros(1, 60, 2))
    committed, event = scheduler.commit(replace(tensor_state, tick=348, pending=True), plan, actual_ready_tick=342)
    assert event["accepted"]
    deps.register(committed.reference, 33)
    influence = event["first_influence_tick"]
    assert influence == 396
    assert deps.control_depth(committed.reference, 348) == 0
    assert deps.control_depth(committed.reference, influence - 12) == 0
    assert deps.control_depth(committed.reference, influence) == 33
    depth = 29
    for tick in range(348, 408, 12):
        depth = max(depth, deps.control_depth(committed.reference, tick)) + 1
    assert depth == 34  # 不能把提交时刻29误当成已经依赖33步未来的状态。
    assert deps.cache_depth(committed.reference) == 33
    next_ticket = scheduler.reserve(replace(committed, tick=360, decision_id=1))
    assert deps.prefix_depth(committed.reference, next_ticket) == 33


def test_overwritten_future_version_does_not_pollute_current_support(tensor_state):
    scheduler = Scheduler(120)
    deps = ReferenceDependencies(tensor_state.reference)
    ticket = scheduler.reserve(tensor_state)
    plan = GeneratedPlan(ticket, tensor_state.reference.world_qpos.clone(), torch.zeros(1, 60, 2))
    committed, _ = scheduler.commit(replace(tensor_state, tick=120, pending=True), plan, actual_ready_tick=120)
    deps.register(committed.reference, 100)
    assert deps.control_depth(committed.reference, 0) == 0
    assert deps.control_depth(committed.reference, 120) == 100
    assert deps.control_depth(committed.reference, 120, future_steps=0) == 0
def test_forecast_physical_ticks_are_independent_of_graph_depth():
    """93个物理控制步可具有101步图依赖；缺失中间状态仍必须被拒绝。"""
    from gem.closedloop.mb_control.dependency import paired_forecast_timeline
    meta = {"episode_start_tick": 0, "request_tick": 720, "forecast_tick": 1116,
            "branch_control_steps": 33, "recorded_control_steps": 93,
            "state_ticks": list(range(12, 1117, 12)), "dependency_steps": 101}
    assert paired_forecast_timeline(meta, dict(meta), expected_forecast_tick=1116)
    missing = dict(meta, state_ticks=meta["state_ticks"][:70]+meta["state_ticks"][71:])
    assert not paired_forecast_timeline(missing, missing, expected_forecast_tick=1116)
    assert not paired_forecast_timeline(meta, dict(meta, forecast_tick=1104), expected_forecast_tick=1116)

