"""保存真实控制证据中的AWR评价诊断，不重算或修改奖励。

逐控制步读取GMT已发布的八项Tracking、原接触/滑动代价、原始误差和实际Root位置；
数值列随Episode进入二进制分片。摘要按真实控制样本统计，Root范围按完整Episode
计算，生成参考范围单独标明包含未执行后缀。足端滑动保持原支持球速度代理语义，
不能解释为新的真实足底传感器量。完整消费报告直接使用公共参考时间审计。
"""
from __future__ import annotations

import numpy as np
from gem.closedloop.common.execution_provenance import consumption_report
from gem.closedloop.common.rewards import TRACKING_FUNCTIONS


class PhysicalMetrics:
    def __init__(self):
        self.rows, self.provenance, self.generated_ranges = [], [], []

    def observe(self, task, result):
        metadata = result["metadata"] if isinstance(result, dict) else result.metadata
        self.provenance.append(metadata.get("compact_execution_provenance"))
        q = np.asarray(metadata["generated"]["qpos_world"], dtype=np.float64)
        self.generated_ranges.append(np.ptp(q[:, :3], axis=0))
        for row in metadata["execution_trace"]:
            components = row["reward_primitives"]["components"]
            contact = components["contact"]
            samples = contact["raw"]["samples"]
            foot_contacts, support_speeds = [], []
            for sample in samples:
                forces = np.asarray(sample["foot_net_contact_forces_w_n"])
                foot_contacts.append(np.linalg.norm(forces, axis=-1) > sample["foot_contact_force_threshold_n"])
                for name, speed in sample["slide_proxy"].items():
                    speed = np.atleast_1d(np.asarray(speed, dtype=np.float64))
                    selected = np.atleast_1d(np.asarray(sample["support_selected"][name], dtype=bool))
                    support_speeds.extend(speed[selected].tolist())
            self.rows.append(dict(tick=int(row["tick"]), root=np.asarray(row["actual_qpos"][:3], dtype=np.float64),
                tracking=[float(components["track"]["normalized"]["scores"][k]) for k in TRACKING_FUNCTIONS],
                errors={k: float(v) for k, v in row["errors"].items()},
                slide_cost=float(contact["normalized"]["slide"]),
                bad_contact_cost=float(contact["normalized"]["bad_contact"]),
                foot_contact_fraction=float(np.mean(foot_contacts)),
                support_speed_sum_m_s=float(sum(support_speeds)), support_speed_count=len(support_speeds)))

    def result(self):
        columns = {}
        if self.rows:
            for key in self.rows[0]:
                if key != "errors":
                    columns[key] = np.asarray([r[key] for r in self.rows])
            for key in self.rows[0]["errors"]:
                columns[key] = np.asarray([r["errors"][key] for r in self.rows])
            if not all(np.isfinite(a).all() for a in columns.values()):
                raise ValueError("Physical evaluation diagnostics contain nonfinite evidence")
        return dict(schema="genmo.awr.physical_metrics.v1", columns=columns,
                    tracking_order=list(TRACKING_FUNCTIONS),
                    generated_root_ranges_m=np.asarray(self.generated_ranges),
                    reference_consumption=consumption_report(self.provenance))


def summarize_metrics(episodes):
    diagnostics = [e["awr_physical_metrics"] for e in episodes]
    columns = [d["columns"] for d in diagnostics if d["columns"]]
    if not columns:
        return dict(available=False, reason="no_actual_control_samples")
    merged = {k: np.concatenate([c[k] for c in columns]) for k in columns[0]}
    speed_count = int(merged["support_speed_count"].sum())
    return dict(available=True, actual_control_samples=len(merged["tick"]),
        tracking_eight_mean=dict(zip(diagnostics[0]["tracking_order"], merged["tracking"].mean(axis=0).tolist())),
        original_errors_mean={k: float(v.mean()) for k, v in merged.items()
                              if k.endswith(("_error_m", "_error_rad", "_rmse_m", "_rmse_rad", "_rmse_rad_s"))},
        slide_cost_mean=float(merged["slide_cost"].mean()),
        bad_contact_cost_mean=float(merged["bad_contact_cost"].mean()),
        foot_contact_fraction=float(merged["foot_contact_fraction"].mean()),
        supported_sphere_speed_mean_m_s=(float(merged["support_speed_sum_m_s"].sum()/speed_count)
                                        if speed_count else None),
        slide_semantics="original_supported_sphere_tangent_velocity_proxy",
        actual_root_peak_to_peak_m_per_episode=[np.ptp(c["root"], axis=0).tolist() for c in columns],
        actual_root_path_length_m_per_episode=[float(np.linalg.norm(np.diff(c["root"], axis=0), axis=1).sum())
                                              for c in columns],
        reward_modified=False)
