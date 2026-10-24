"""真实执行参考的姿态可执行性诊断，明确区分heading退化和实际大幅倾斜。

输入为控制侧车中冻结GMT的同一时刻参考/实际anchor四元数，使用世界系wxyz。
通过旋转矩阵的前向轴水平投影识别heading病态，用重力方向夹角衡量不依赖yaw
分解的姿态差，并分别记录参考与实际相对竖直的倾斜、俯仰和原生产non-yaw误差。
一个信号不能证明动作不可行：near-vertical只标记参考前向轴接近竖直；只有同时
存在实际大倾斜才标记联合风险。自动隔离还需多次物理失败和跨轮次证据。
输出只用于采样诊断/隔离审计，不修改姿态、奖励、GMT、终止条件或固定评估任务。
缺证据返回unknown，不能用零值代表正常。本模块仅使用NumPy，不产生仿真操作。
"""
from __future__ import annotations

import math
from collections.abc import Mapping
import numpy as np


def pose_diagnostics(reference, actual, *, non_yaw=None, horizontal_threshold=.15,
                     tilt_threshold_rad=math.pi/3):
    try:
        q = np.asarray([reference, actual], dtype=np.float64)
    except (ValueError,TypeError):
        return dict(available=False,reason='invalid_quaternion_shape_or_dtype')
    if q.shape != (2, 4) or not np.isfinite(q).all():
        return dict(available=False, reason='missing_or_invalid_quaternion')
    norm = np.linalg.norm(q, axis=1)
    if not np.allclose(norm, 1., atol=1e-4, rtol=0.):
        return dict(available=False, reason='nonunit_quaternion')
    q = q/norm[:, None]  # 仅诊断消除FP32单位长度舍入，不改原控制数据。
    w, x, y, z = q.T
    forward = np.stack((1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y)), 1)
    up = np.stack((2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)), 1)
    horizontal = np.linalg.norm(forward[:, :2], axis=1)
    pitch = np.arcsin(np.clip(-forward[:, 2], -1., 1.))
    tilt = np.arccos(np.clip(up[:, 2], -1., 1.))
    # world重力在各自body系中的方向；避免先提取不稳定heading再相减。
    gravity = np.stack((2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), 1)
    gravity_error = float(np.arccos(np.clip(np.dot(gravity[0], gravity[1]), -1., 1.)))
    heading_risk = bool(horizontal[0] < horizontal_threshold)
    actual_tilt = bool(tilt[1] > tilt_threshold_rad)
    result = dict(available=True, reference_forward_horizontal_norm=float(horizontal[0]),
        actual_forward_horizontal_norm=float(horizontal[1]), reference_pitch_rad=float(pitch[0]),
        actual_pitch_rad=float(pitch[1]), reference_tilt_rad=float(tilt[0]), actual_tilt_rad=float(tilt[1]),
        gravity_direction_error_rad=gravity_error, reference_heading_degenerate=heading_risk,
        actual_large_tilt=actual_tilt, combined_feasibility_risk=heading_risk and actual_tilt,
        production_non_yaw_error_rad=None if non_yaw is None else float(non_yaw),
        interpretation='diagnostic_not_physical_infeasibility_proof')
    return result


def execution_pose_diagnostics(row):
    tracking = row.get('motion_tracking')
    if (not isinstance(tracking, Mapping) or tracking.get('coordinate_frame') != 'world'
            or tracking.get('quaternion_order') != 'wxyz'):
        return dict(available=False, reason='tracking_pose_not_recorded')
    try:
        result = pose_diagnostics(tracking['reference']['anchor_quat_w'],
            tracking['actual']['anchor_quat_w'],
            non_yaw=row.get('errors', {}).get('non_yaw_orientation_error_rad'))
        result.update(anchor_body_name=tracking.get('anchor_body_name'), control_tick=int(row['tick']))
        return result
    except (ValueError, KeyError, TypeError, OverflowError):
        return dict(available=False, reason='invalid_tracking_pose')


def summarize_execution(rows):
    # 每控制步诊断，但仅保存极值/计数/末态，避免把正式checkpoint变成原始轨迹仓库。
    poses = [execution_pose_diagnostics(row) for row in rows]
    valid = [p for p in poses if p['available']]
    return dict(schema='genmo.stage2.feasibility.v1', controls=len(poses), valid_controls=len(valid),
        unknown_controls=len(poses)-len(valid),
        heading_degenerate_controls=sum(p['reference_heading_degenerate'] for p in valid),
        actual_large_tilt_controls=sum(p['actual_large_tilt'] for p in valid),
        combined_risk_controls=sum(p['combined_feasibility_risk'] for p in valid),
        minimum_reference_forward_horizontal_norm=min(
            (p['reference_forward_horizontal_norm'] for p in valid), default=None),
        maximum_actual_tilt_rad=max((p['actual_tilt_rad'] for p in valid), default=None),
        terminal=poses[-1] if poses else dict(available=False, reason='no_executed_control'))
