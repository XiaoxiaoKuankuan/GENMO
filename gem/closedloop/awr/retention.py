"""历史自蒸馏修复实验的保留门槛重现工具，当前AWR训练不再调用。

自蒸馏无需提高回报，但必须在同任务、同种子、同物理和奖励身份下保持完成能力、
执行时长及动作质量。本模块只读取完整物理报告，不按Loss挑选模型、不删除失败、
不把动作更慢自动当成改善。该旧规则仅用于解释已完成的历史报告，不再阻止AWR。
"""
from __future__ import annotations

import math
import numpy as np
from .evaluation import paired_comparison


def retention_comparison(baseline, current, limits):
    # 历史修复实验的重现接口；当前AWR运行不调用此门槛。
    paired = paired_comparison(baseline, current, selection_policy="legacy_physical_gates")
    before, after = (r["aggregate"]["overall"] for r in (baseline, current))
    ratios = {}
    def ratio(name, left, right):
        if left is None or right is None or not math.isfinite(left+right) or left < 0 or right < 0:
            raise ValueError(f"Missing/nonfinite retention metric: {name}")
        value = right/max(left, 1e-12)
        ratios[name] = value
        return value
    gates = {key: paired["gates"][key] for key in (
        "completion_not_decreased", "failure_count_not_increased", "no_new_paired_failure")}
    gates["duration_retained"] = ratio("duration", before["executed_seconds"], after["executed_seconds"]) >= limits["min_duration_ratio"]
    gates["return_retained"] = ratio("return", before["reward_sum"], after["reward_sum"]) >= limits["min_return_ratio"]
    activity = paired["activity_ratio"]
    gates["activity_retained_without_growth"] = (activity is not None and
        limits["min_activity_ratio"] <= activity <= limits["max_motion_ratio"])
    # 同任务等权比较，不让长曲提前失败截短后的统计掩盖动作强度上升。
    for key in ("generated_joint_speed_rms_rad_s", "generated_joint_acceleration_rms_rad_s2",
                "generated_joint_jerk_rms_rad_s3"):
        values = [float(np.mean([t["generated"][key] for t in r["task_summaries"]]))
                  for r in (baseline, current)]
        gates[key] = ratio(key, *values) <= limits["max_motion_ratio"]
    for key in ("root_position_error_m", "joint_position_rmse_rad"):
        values = [r["physical_metrics"]["overall"]["original_errors_mean"][key]
                  for r in (baseline, current)]
        gates[key] = ratio(key, *values) <= limits["max_error_ratio"]
    return dict(accepted=all(gates.values()), gates=gates, ratios=ratios, limits=limits,
                paired=paired, scope="full_music_physical_self_distillation_retention_not_improvement")
