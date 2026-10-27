"""真实首次影响起点的冻结SM/WM闭环可微执行。

候选x0经原codec/参考缓存进入SM，每步从当前预测状态更新75维观测、历史和上一动作，
再次求新动作。这里只detach固定真实起点和数据目标，不detach候选/动作/预测链。支持域
门禁只读detach副本；失败整窗丢弃执行梯度而保留BC。短窗覆盖和预览步数独立报告。
"""

from dataclasses import replace
import json
from pathlib import Path

import torch

from gem.closedloop.mb_control.validation import ModelSupport, UnsupportedModelState
from gem.closedloop.mb_control.contracts import ReferenceExhausted
from .contracts import PhysicsRejected
from .losses import execution_terms, normalized_window
from .reference import candidate_reference


class ValidatedSupport:
    def __init__(self, report_path, assets, horizon, device, objective):
        report = json.loads(Path(report_path).read_text())
        if report["schema"] not in ("genmo.joint_physics.validation.v1", "genmo.joint_physics.validation.v2") or report["assets"] != assets:
            raise ValueError("必须使用本方法与同一资产的新Newton报告")
        if horizon not in report["supported_horizons"]:
            raise ValueError("本方法当前H尚未通过实际物理方向/精度验证")
        if report.get("objective") != objective or not report.get("forecast_passed"):
            raise ValueError("物理方向报告需绑定当前执行目标并通过真实请求前瞻验证")
        self.state_guard = ModelSupport(report["state_envelope"], device=device)
        self.lower = torch.tensor(report["action_min"], device=device)
        self.upper = torch.tensor(report["action_max"], device=device)
        self.source_ids = set(report["source_ids"])
        self.report = report
        self.horizon = horizon
        self.certificates = report.get("certificates", {}).get(str(horizon), {})

    def certified(self, tags):
        """来源ID只追溯；须与同H的真实动作/接触类型认证匹配，旧报告不可扩到新来源。"""
        return bool(tags) and all(self.certificates.get(tag, {}).get("passed") for tag in tags)

    def timesteps_for(self, tags):
        if not self.certified(tags):
            return []
        return sorted(set.intersection(*(set(self.certificates[tag]["audited_timesteps"]) for tag in tags)))

    def for_tags(self, tags):
        if not self.certified(tags):
            raise PhysicsRejected("uncertified_motion_type", {"required_motion_types": tags})
        certificates = [self.certificates[tag] for tag in tags]
        guard = object.__new__(ValidatedSupport)
        envelope = dict(certificates[0]["state_envelope"])
        envelope["minimum"] = torch.tensor([c["state_envelope"]["minimum"] for c in certificates]).amax(0).tolist()
        envelope["maximum"] = torch.tensor([c["state_envelope"]["maximum"] for c in certificates]).amin(0).tolist()
        guard.state_guard = ModelSupport(envelope, device=self.lower.device)
        guard.lower = self.lower.new_tensor([c["action_min"] for c in certificates]).amax(0)
        guard.upper = self.upper.new_tensor([c["action_max"] for c in certificates]).amin(0)
        return guard

    def check_state(self, state):
        try:
            self.state_guard(state)
        except (UnsupportedModelState, FloatingPointError) as error:
            raise PhysicsRejected("wm_state_out_of_domain", getattr(error, "details", {})) from error

    def check_action(self, action):
        if not torch.isfinite(action).all():
            raise PhysicsRejected("nonfinite_sm_action")
        outside = ((action.detach() < self.lower-.02) | (action.detach() > self.upper+.02)).nonzero().tolist()
        if outside:
            raise PhysicsRejected("action_out_of_validated_domain", {"components": [
                {"joint_index": j, "value": float(action[i,j].detach()), "lower": float(self.lower[j]-.02),
                 "upper": float(self.upper[j]+.02)} for i,j in outside]})

    def __call__(self, state, action):
        try:
            self.state_guard(state)
        except (UnsupportedModelState, FloatingPointError) as error:
            raise PhysicsRejected("wm_state_out_of_domain", getattr(error, "details", {})) from error
        if not torch.isfinite(action).all() or bool(((action.detach() < self.lower-.02) | (action.detach() > self.upper+.02)).any()):
            outside = ((action.detach() < self.lower-.02) | (action.detach() > self.upper+.02)).nonzero().tolist()
            raise PhysicsRejected("action_out_of_validated_domain", {"components": [
                {"joint_index": j, "value": float(action[i,j].detach()), "lower": float(self.lower[j]-.02),
                 "upper": float(self.upper[j]+.02)} for i,j in outside]})


def execute_window(window, predicted_x0, logits, actor, adapter, cfg, *, support=None, transition=None):
    window.validate()
    reference = candidate_reference(window, predicted_x0, logits, actor.stage1.endecoder.codec,
                                    actor.stage1.endecoder)
    enforce = cfg.get("hard_safety", True)
    def hard_joint(q, reason):
        kin = adapter.runtime.kin
        outside = ((q.detach() < kin.q_lo-1e-4) | (q.detach() > kin.q_hi+1e-4)).nonzero().tolist()
        if outside:
            raise PhysicsRejected(reason, {"first_reject_tick": window.start.tick, "components": outside,
                                            "maximum_violation_rad": float(torch.maximum(kin.q_lo-q.detach(), q.detach()-kin.q_hi).max())})
    if enforce:
        hard_joint(reference.world_qpos[...,7:][...,adapter.to_sm], "reference_hard_joint_limit")
    first = reference.first_influence(window.request.tick, window.ticket.first_mutable_source_tick)
    if first != window.start.tick:
        raise PhysicsRejected("first_influence_mismatch")
    z = replace(window.start.detached(), reference=reference)
    if support is not None:
        try:
            support.check_state(z.state)
        except PhysicsRejected as error:
            error.details.update(first_reject_tick=z.tick, failure_phase="real_initial_state", action_origin="not_executed")
            raise PhysicsRejected("real_start_out_of_domain",error.details) from error
    losses, safety, terms, states, actions, support_ticks, center = [], [], [], [], [], set(), 0
    failure_steps = []
    for step in range(window.horizon):
        rejection_tick, phase = z.tick, "reference_and_action"
        try:
            dependency = reference.dependency_support(z.tick)
            mutable = dependency[dependency >= window.ticket.first_mutable_source_tick]
            support_ticks.update(int(x) for x in mutable)
            source = reference.ticks.new_tensor([z.tick])
            lo, hi, fraction = reference._indices(source)
            center += int(bool((reference.ticks[lo] >= window.ticket.first_mutable_source_tick).any() or
                          ((reference.ticks[hi] >= window.ticket.first_mutable_source_tick) & (fraction != 0)).any()))
            z = adapter.refresh_current_observation(z)
            _, _, goal = z.reference.read_current_future(z.tick)
            action = adapter.bundle.act(z.sm_history, goal)
            if not torch.isfinite(action).all():
                raise PhysicsRejected("nonfinite_sm_action")
            if support is not None:
                support.check_action(action)
            rejection_tick, phase = z.tick+12, "predicted_state"
            next_state = adapter.bundle.step_model(z.state, action, z.xi) if transition is None else transition(action)
            if not torch.isfinite(next_state).all():
                raise PhysicsRejected("nonfinite_predicted_state")
            if enforce:
                from supermimic.mdp import unflatten_state
                hard_joint(unflatten_state(next_state)[4], "executed_hard_joint_limit")
            if support is not None:
                support.check_state(next_state)
            z = adapter.advance(z, next_state, action, origin="imagined" if transition is None else "real")
            target, _, _ = window.target.read_current_future(z.tick, 0)
            target = target._make(t.detach() for t in target)
            loss, limit, components = execution_terms(z.state, target, adapter.runtime.kin, cfg)
            # 只记录原SM失败判据，不以失败后reset拼接、不改变训练损失或覆盖门禁。
            with torch.no_grad():
                if bool(adapter.runtime.termination(z.state.detach(), target).any()):
                    failure_steps.append(step + 1)
                    if enforce:
                        raise PhysicsRejected("sm_physical_termination")
        except PhysicsRejected as error:
            error.details.update(first_reject_tick=rejection_tick, control_step=step, failure_phase=phase,
                                 action_origin="frozen_sm_from_student_denoised_x0")
            raise
        except ReferenceExhausted as error:
            raise PhysicsRejected("incomplete_reference_window") from error
        if not torch.isfinite(loss).all() or not torch.isfinite(limit).all():
            raise PhysicsRejected("nonfinite_execution")
        losses.append(loss)
        safety.append(limit)
        terms.append(components)
        states.append(z.state)
        actions.append(action)
    if not support_ticks:
        raise PhysicsRejected("no_legal_new_reference_influence")
    return {"loss": normalized_window(losses, cfg["discount"]).mean(),
        "safety": normalized_window(safety, cfg["discount"]).mean(),
        "terms": {k: normalized_window([t[k] for t in terms], cfg["discount"]).mean() for k in terms[0]},
        "states": torch.stack(states), "actions": torch.stack(actions),
        "failure_steps": failure_steps,
        "coverage": {"horizon": window.horizon, "preview_only_steps": window.horizon-center,
            "center_consumption_steps": center, "supported_source_ticks": sorted(support_ticks),
            "supported_source_points": len(support_ticks), "mutable_source_points": 60-window.ticket.prefix_frames,
            "coverage_ratio": len(support_ticks)/(60-window.ticket.prefix_frames),
            "request_tick": window.request.tick, "first_influence_tick": first,
            "end_tick": z.tick, "start_origin": window.provenance["start_origin"],
            "complete_60_frame_optimization": False}}
