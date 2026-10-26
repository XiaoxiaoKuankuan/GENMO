"""从本轮真实训练转移生成Critic困难区间诊断标签，标签不进入任何模型输入。

按已经执行的区间误差区分腿部、根高度、非偏航姿态困难样本；按同一episode的
真实控制时间寻找本轮可见的失败前5秒及PPO切分边界附近状态。这里允许事后分组，
只用于留出预测统计，不能用作Actor条件或改变价值目标。失败尚未出现在本轮时
不会猜测其未来标签；音乐位置超过100秒与连续物理运行超过100秒明确分开。
返回每条转移的分组名，跨rank仅交换这些轻量标签，避免再采完整MC或保存扩散链。
"""
from __future__ import annotations


GROUPS = ('large_leg_error', 'large_root_height_error', 'large_non_yaw_error',
          'reference_switch', 'rollout_boundary_1s', 'observed_failure_before_5s',
          'music_position_after_100s', 'continuous_execution_after_100s')


def subgroup_labels(rows):
    def episode(row):
        ident = row.identity
        return (ident.get('backend_session_id'), ident.get('env_id'), ident.get('episode_id'))
    failures, boundaries = {}, {}
    for row in rows:
        if row.metadata.get('terminal_snapshot', {}).get('terminated', False):
            failures.setdefault(episode(row), []).append(row.control_tick_end)
        if row.truncated and not row.terminated and not row.metadata.get('physical_time_limit_reached', False):
            boundaries.setdefault(episode(row), []).append(row.control_tick_end)
    result = []
    for row in rows:
        flags = set()
        details = row.metadata.get('reward_details', [])
        for detail in details:
            raw = detail.get('components', {}).get('track', {}).get('raw', {})
            physical = detail.get('diagnostics', {}).get('physical_state', {})
            if raw.get('leg_joint_pos_squared_error', 0.) > .04:
                flags.add('large_leg_error')
            if abs(physical.get('root_height_error_m', 0.)) > .05:
                flags.add('large_root_height_error')
            if abs(physical.get('non_yaw_orientation_error_rad', 0.)) > .2:
                flags.add('large_non_yaw_error')
        if len(row.metadata.get('consumed_plan_ids', [])) > 1:
            flags.add('reference_switch')
        if any(0 <= tick-row.control_tick_begin <= 600 for tick in boundaries.get(episode(row), [])):
            flags.add('rollout_boundary_1s')
        if any(0 <= tick-row.control_tick_begin <= 3000 for tick in failures.get(episode(row), [])):
            flags.add('observed_failure_before_5s')
        elapsed = max(0., (row.control_tick_begin-600)/600.)
        if elapsed+row.metadata.get('music_start_frame', 0)/30. > 100:
            flags.add('music_position_after_100s')
        if elapsed > 100:
            flags.add('continuous_execution_after_100s')
        result.append(sorted(flags))
    return result
