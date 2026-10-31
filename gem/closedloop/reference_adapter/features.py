"""部署可用Adapter输入与特权Critic输入的分离构造。

Adapter只读取已经使用的SM历史、原Goal、编码器关节/速度与IMU相对姿态，以及固定
名义计划和已知音频。xi和完整state55仅加入Critic；旧历史不重新用名义参考计算。
参考及音乐时钟是绝对tick派生的确定量，没有可学习的音乐偏移或真实未来状态。
"""

import torch

from supermimic.mdp import unflatten_state, normalize_xi
from supermimic.rotations import mat_to_6d
from .reference import nominal_pose, pose_residual, nominal_window


def adapter_fields(z, adapter, bundle, nominal_current, nominal_goal):
    _, R, _, _, q, qd = unflatten_state(z.executed_state)
    music, valid, times = z.audio.preview(z.tick)
    plan = z.previous_plan
    b = len(q)
    if plan is not None:
        if plan.ticks.min() < z.nominal_reference.lower or plan.ticks.max() > z.nominal_reference.upper:
            raise ValueError("当前名义计划必须覆盖上一计划的已知重叠点")
        old_pose_relative = pose_residual(plan.corrected_pose, nominal_pose(z.nominal_reference, plan.ticks), z.heading[:, None]) / adapter.bounds
    else:
        old_pose_relative = q.new_zeros(b, 10, 27)
    anchors = []
    for tick, pose in ((max(z.tick-12, z.nominal_reference.lower), z.left_pose), (z.tick, z.current_pose)):
        base = nominal_pose(z.nominal_reference, z.nominal_reference.ticks.new_tensor([tick])).select((slice(None), 0))
        anchors.append(pose_residual(pose, base, z.heading) / adapter.bounds)
    return {
        "control_history": bundle.actor.obs_norm(z.control_history),
        "nominal_goal": bundle.actor.goal_norm(nominal_goal),
        "observable_error": torch.cat((nominal_current.q-q, (nominal_current.qd-qd)*.1,
                                        mat_to_6d(R.transpose(-1, -2) @ nominal_current.root_R)), -1),
        "previous_plan": plan.residual / adapter.bounds if plan else q.new_zeros(b, 10, 27),
        "previous_corrected_plan": old_pose_relative,
        "plan_times": ((plan.ticks-z.tick).to(q)/600)[None].expand(b, -1) if plan else q.new_zeros(b, 10),
        "plan_valid": plan.valid.to(q) if plan else q.new_zeros(b, 10),
        "committed_anchors": torch.stack(anchors, 1),
        "music": music, "music_valid": valid.to(q), "music_times": times,
        "music_memory": z.music_state.features(),
    }


def critic_features(z, adapter, bundle, kin):
    from gem.closedloop.mb_control.contracts import ReferenceExhausted
    try:
        current, _, goal = nominal_window(z.nominal_reference, z.tick, kin)
        goal_valid = 1.
    except ReferenceExhausted:
        # 只允许终点的显式无效goal；对应真结束的bootstrap必须为0。
        from supermimic.bridge.reference import build_reference
        from supermimic.motion import select_reference
        current = select_reference(build_reference(z.nominal_reference.pose_at,
            z.nominal_reference.ticks.new_tensor([z.tick]), z.nominal_reference.lower, z.nominal_reference.upper, kin),
            (slice(None), 0))
        goal, goal_valid = z.executed_state.new_zeros(len(z.executed_state), 570), 0.
    fields = adapter_fields(z, adapter, bundle, current, goal)
    clock = z.executed_state.new_tensor([
        (z.tick-z.audio.start_tick)/600, max(0, z.audio.control_end_tick-z.tick)/600,
        max(0, z.nominal_reference.upper-z.tick)/600,
        (z.plan_boundary_tick-z.tick)/600 if z.plan_boundary_tick is not None else -1.,
        (z.tick-z.known_at_tick)/600, float(z.mode == "rolling"), goal_valid,
        (z.tick-z.music_state.last_paid_tick)/600 if z.music_state.last_paid_tick is not None else -1.])
    scheduling = z.scheduling or {}
    pending = scheduling.get("pending_request")
    history = scheduling.get("latency_history", ())
    schedule_features = z.executed_state.new_tensor([
        float(pending is not None),
        (pending.ticket.request_tick-z.tick)/600 if pending else 0.,
        (pending.ticket.predicted_ready_tick-z.tick)/600 if pending else 0.,
        (pending.ticket.first_mutable_source_tick-z.tick)/600 if pending else 0.,
        (scheduling.get("next_request_tick", z.tick)-z.tick)/600,
        sum(history)/len(history)/600 if history else 0.])
    # Goal570以名义当前根为相对原点，单独补当前世界参考才能辨识原SM根位置跟踪误差。
    # 这57维只进入Critic；Adapter仍不读取世界根位置/速度。
    current_reference = torch.cat((current.root_p, mat_to_6d(current.root_R), current.root_v, current.root_w,
                                   current.q, current.qd*.1), -1)
    return torch.cat((adapter.input_spec.flatten(fields), z.executed_state, normalize_xi(z.xi),
                      z.previous_action, z.heading.flatten(1), clock[None].expand(len(goal), -1),
                      schedule_features[None].expand(len(goal), -1), current_reference), -1)


def critic_dimension(adapter):
    return adapter.input_spec.dimension + 55 + 25 + 21 + 9 + 8 + 6 + 57
