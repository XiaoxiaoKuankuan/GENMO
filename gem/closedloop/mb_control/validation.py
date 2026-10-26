"""正式训练的独立验证覆盖与物理状态支持合同。

P3覆盖按已有数据来源、未用于训练的split、名义物理和实际预测时域核对，不要求
每首训练音乐出现在审计清单。状态支持由真实审计轨迹建立：使用根高度、重力方向、
机体系速度、关节位置和速度，去除世界XY/yaw平移旋转自由度；超域拒绝样本，不
裁剪状态或扩大WM允许时域。源码提交只记录，固定模型资产仍须一致。
"""

import torch


class UnsupportedModelState(ValueError):
    def __init__(self,message,details=None):
        super().__init__(message);self.details=details or {}


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


def training_support_envelope(cfg, report):
    """新时域审计不能自动扩大旧支持域；配置旧边界时使用两者交集。"""
    import json
    from pathlib import Path
    current = report["validated_state_envelope"]
    path = cfg.get("imagination", {}).get("state_support_reference_report")
    if not path:
        return current
    previous = json.loads(Path(path).read_text())
    if previous.get("frozen_sha256") != report.get("frozen_sha256"):
        raise ValueError("固定状态支持域必须来自同一冻结SM/WM资产")
    old = previous["validated_state_envelope"]
    if old["schema"] != current["schema"] or len(old["minimum"]) != len(current["minimum"]):
        raise ValueError("状态支持坐标定义不一致")
    lower = [max(a,b) for a,b in zip(old["minimum"],current["minimum"])]
    upper = [min(a,b) for a,b in zip(old["maximum"],current["maximum"])]
    if any(a>b for a,b in zip(lower,upper)):
        raise ValueError("新旧已验证状态支持没有共同范围")
    return {"schema":current["schema"],"minimum":lower,"maximum":upper,
        "real_state_count":current["real_state_count"],
        "definition":"intersection_with_preserved_prior_support",
        "prior_report":str(Path(path).resolve())}


class ModelSupport:
    def __init__(self, envelope, *, device):
        if envelope.get("schema") != "height_gravity_body_velocity_q_qd_v1":
            raise ValueError("需要真实P3状态支持报告")
        self.lower = torch.tensor(envelope["minimum"], device=device)
        self.upper = torch.tensor(envelope["maximum"], device=device)
        # 固定数值容差用于浮点与小扰动；不从验收结果反推扩大范围。
        self.margin = self.lower.new_tensor([.01] + [.02]*3 + [.05]*3 + [.1]*3 + [.02]*21 + [.1]*21)

    def __call__(self, state, tick=None):
        f = support_features(state.detach())
        if not torch.isfinite(f).all():
            raise FloatingPointError("WM状态非有限")
        outside=(f < self.lower-self.margin)|(f > self.upper+self.margin)
        if outside.any():
            labels=["root_height"]+["gravity_"+x for x in "xyz"]+["body_velocity_"+x for x in "xyz"]+["body_angular_"+x for x in "xyz"]+[f"joint_position_sm_{i}" for i in range(21)]+[f"joint_velocity_sm_{i}" for i in range(21)]
            indices=outside.nonzero().tolist()
            raise UnsupportedModelState("当前状态超出独立P3真实状态支持，拒绝执行反馈更新",{"tick":tick,
                "components":[{"name":labels[j],"value":float(f[i,j]),"lower":float(self.lower[j]-self.margin[j]),
                               "upper":float(self.upper[j]+self.margin[j])} for i,j in indices]})


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
    if report.get("physics_determinism", "not_guaranteed") != cfg.get("execution", {}).get("physics_determinism", "not_guaranteed"):
        raise ValueError("短时配对报告与本次Newton确定性设置不一致")
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
    teacher_only_short = (cfg.get("teacher", {}).get("enabled") and
                          cfg["training"].get("terminal_value_weight") == 0.)
    if cfg["imagination"].get("critic_bootstrap") is not True and not teacher_only_short:
        raise ValueError("短时训练必须启用独立上层Critic终点价值")
    if cfg.get("teacher", {}).get("enabled") and report.get("reward_contract") != cfg["rewards"]:
        raise ValueError("教师音乐目标需要对应新奖励的实际H物理方向验证")
    if cfg.get("teacher",{}).get("target_mode")=="online_consumed_reference_v1":
        from .short_audit import teacher_target_contract
        if report.get("teacher_target_contract")!=teacher_target_contract(cfg):
            raise ValueError("新在线音乐目标、对齐或Actor损失改变，需要重做H16/H32物理验证")
        if not report.get("criteria",{}).get("require_dance_main_direction") or not report["horizons"][str(selected)].get("dance_main_direction_passed"):
            raise ValueError("新音乐目标需独立验证dance主任务及左右腿物理方向，不能仅由组合目标通过代替")
    if cfg.get("teacher", {}).get("enabled") and report.get("teacher_seed")!=cfg["teacher"]["seed"]:
        raise ValueError("物理方向必须对应同一个固定教师音乐时间线")
    import json
    from pathlib import Path
    path = cfg["imagination"].get("anchor_gradient_report")
    if not path:
        raise ValueError("短时训练需要对应真实anchor的新执行梯度报告")
    gradient = json.loads(Path(path).read_text())
    if (gradient.get("method") != "physical_anchor_short_bptt" or gradient.get("passed") is not True or
            gradient.get("horizon") != selected or gradient.get("frozen_sha256") != report.get("frozen_sha256")):
        raise ValueError("真实anchor执行梯度报告不匹配或未通过")
    if cfg.get("teacher", {}).get("enabled") and gradient.get("reward_contract") != cfg["rewards"]:
        raise ValueError("真实anchor梯度必须验证当前冻结教师的新奖励")
    if cfg.get("teacher",{}).get("target_mode")=="online_consumed_reference_v1":
        if (gradient.get("teacher_target_contract")!=teacher_target_contract(cfg) or
                gradient.get("known_prefix_global_gradient_max_abs")!=0. or
                not gradient.get("mutable_global_gradient_norm",0.)>0.):
            raise ValueError("当前固定在线音乐保持目标尚未通过真实anchor可修改参考梯度验证")
    cfg["imagination"]["horizon_steps"] = selected
    return selected
