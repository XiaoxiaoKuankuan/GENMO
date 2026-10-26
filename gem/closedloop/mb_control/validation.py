"""正式训练的独立验证覆盖与物理状态支持合同。

P3覆盖按已有数据来源、未用于训练的split、名义物理和实际预测时域核对，不要求
每首训练音乐出现在审计清单。状态支持由真实审计轨迹建立：使用根高度、重力方向、
机体系速度、关节位置和速度，去除世界XY/yaw平移旋转自由度；超域拒绝样本，不
裁剪状态或扩大WM允许时域。源码提交只记录，固定模型资产仍须一致。
"""

import torch


class UnsupportedModelState(ValueError):
    pass


def support_features(state):
    from supermimic.mdp import unflatten_state
    p, R, v, w, q, qd = unflatten_state(state)
    Rt = R.transpose(-1, -2)
    return torch.cat((p[..., 2:3], R[..., 2, :], (Rt @ v[..., None]).squeeze(-1),
                      (Rt @ w[..., None]).squeeze(-1), q, qd), -1)


def state_envelope(real_states):
    features = support_features(real_states)
    return {"schema": "height_gravity_body_velocity_q_qd_v1", "minimum": features.amin(0).tolist(),
            "maximum": features.amax(0).tolist(), "real_state_count": len(features)}


def combine_envelopes(envelopes):
    if not envelopes:
        raise ValueError("缺少真实状态覆盖")
    return {"schema": envelopes[0]["schema"],
            "minimum": torch.tensor([e["minimum"] for e in envelopes]).amin(0).tolist(),
            "maximum": torch.tensor([e["maximum"] for e in envelopes]).amax(0).tolist(),
            "real_state_count": sum(e["real_state_count"] for e in envelopes)}


class ModelSupport:
    def __init__(self, envelope, *, device):
        if envelope.get("schema") != "height_gravity_body_velocity_q_qd_v1":
            raise ValueError("需要真实P3状态支持报告")
        self.lower = torch.tensor(envelope["minimum"], device=device)
        self.upper = torch.tensor(envelope["maximum"], device=device)
        # 固定数值容差用于浮点与小扰动；不从验收结果反推扩大范围。
        self.margin = self.lower.new_tensor([.01] + [.02]*3 + [.05]*3 + [.1]*3 + [.02]*21 + [.1]*21)

    def __call__(self, state):
        f = support_features(state.detach())
        if not torch.isfinite(f).all():
            raise FloatingPointError("WM状态非有限")
        if ((f < self.lower - self.margin) | (f > self.upper + self.margin)).any():
            raise UnsupportedModelState("当前状态超出独立P3真实状态支持，拒绝执行反馈更新")


def validate_coverage(tasks, report):
    coverage = report.get("coverage", {})
    if coverage.get("physics") != "nominal_flat_newton_mujoco_warp":
        raise ValueError("P3缺少匹配物理合同")
    audited = coverage.get("tasks", [])
    train_ids = {t["id"] for t in tasks}
    heldout = [t for t in audited if t.get("split") in ("val", "test") and t["id"] not in train_ids]
    sources = {(t.get("distribution_id"), t.get("source")) for t in heldout}
    requested = {(t.get("distribution_id"), t.get("source")) for t in tasks}
    if any(None in pair for pair in requested) or not requested <= sources:
        raise ValueError("训练数据来源缺少独立held-out P3覆盖")
    if not report.get("validated_state_envelope"):
        raise ValueError("P3必须提供真实状态分布支持，不能只检查歌曲标签")


def select_short_horizon(cfg, report):
    """只认可新方法逐时域实测报告，旧supported_control_steps永不用于短时授权。"""
    if report.get("schema") != "physical_anchor_short_bptt.v1" or report.get("passed") is not True:
        raise ValueError("真实anchor短时物理精度/方向尚未通过")
    if not report.get("frozen_unchanged") or not report.get("forecast_prediction_passed") or not report.get("phase_coverage_passed"):
        raise ValueError("冻结合同、预测条件或真实运动阶段覆盖不足")
    candidates = [h for h in report.get("supported_horizons", []) if h in (16, 24, 32) and
                  report["horizons"][str(h)]["passed"] is True]
    if not candidates or report.get("legal_real_anchors", 0) < 1:
        raise ValueError("没有合法真实执行样本或合格16/24/32时域")
    requested = cfg["imagination"].get("horizon_steps")
    selected = min(candidates) if requested in (None, "auto") else requested
    if selected not in candidates:
        raise ValueError("配置短时域未获得对应物理验收")
    if cfg["imagination"].get("critic_bootstrap") is not True:
        raise ValueError("短时训练必须启用独立上层Critic终点价值")
    cfg["imagination"]["horizon_steps"] = selected
    return selected
