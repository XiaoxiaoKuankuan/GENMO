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
        if report["schema"] != "genmo.joint_physics.validation.v1" or report["assets"] != assets:
            raise ValueError("必须使用本方法与同一资产的新Newton报告")
        if horizon not in report["supported_horizons"]:
            raise ValueError("本方法当前H尚未通过实际物理方向/精度验证")
        if report.get("objective") != objective or not report.get("forecast_passed"):
            raise ValueError("物理方向报告需绑定当前执行目标并通过真实请求前瞻验证")
        self.state_guard = ModelSupport(report["state_envelope"], device=device)
        self.lower = torch.tensor(report["action_min"], device=device)
        self.upper = torch.tensor(report["action_max"], device=device)
        self.source_ids = set(report["source_ids"])

    def __call__(self, state, action):
        try:
            self.state_guard(state)
        except (UnsupportedModelState, FloatingPointError) as error:
            raise PhysicsRejected("wm_state_out_of_domain") from error
        if not torch.isfinite(action).all() or bool(((action.detach() < self.lower-.02) | (action.detach() > self.upper+.02)).any()):
            raise PhysicsRejected("action_out_of_validated_domain")


def execute_window(window, predicted_x0, logits, actor, adapter, cfg, *, support=None, transition=None):
    window.validate()
    reference = candidate_reference(window, predicted_x0, logits, actor.stage1.endecoder.codec,
                                    actor.stage1.endecoder)
    first = reference.first_influence(window.request.tick, window.ticket.first_mutable_source_tick)
    if first != window.start.tick:
        raise PhysicsRejected("first_influence_mismatch")
    z = replace(window.start.detached(), reference=reference)
    losses, safety, terms, states, actions, support_ticks, center = [], [], [], [], [], set(), 0
    for _ in range(window.horizon):
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
            next_state = adapter.bundle.step_model(z.state, action, z.xi) if transition is None else transition(action)
            if support is not None:
                support(next_state, action)
            z = adapter.advance(z, next_state, action, origin="imagined" if transition is None else "real")
            target, _, _ = window.target.read_current_future(z.tick, 0)
            target = target._make(t.detach() for t in target)
            loss, limit, components = execution_terms(z.state, target, adapter.runtime.kin, cfg)
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
        "coverage": {"horizon": window.horizon, "preview_only_steps": window.horizon-center,
            "center_consumption_steps": center, "supported_source_ticks": sorted(support_ticks),
            "supported_source_points": len(support_ticks), "mutable_source_points": 60-window.ticket.prefix_frames,
            "coverage_ratio": len(support_ticks)/(60-window.ticket.prefix_frames),
            "request_tick": window.request.tick, "first_influence_tick": first,
            "end_tick": z.tick, "start_origin": window.provenance["start_origin"],
            "complete_60_frame_optimization": False}}
