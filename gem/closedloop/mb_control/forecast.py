"""旧参考驱动的因果闭环前瞻，不读取未来真值或当次实际完成延迟。

从请求状态循环读取旧目标、执行冻结SM Actor、推进同一个WM、重建75维观测历史。
输出对应首次可能影响控制之前的状态及相对旧参考58维误差。函数不永久加no_grad，
在线调用者可禁用图，多决策想象可保留对早先状态和缓存的导数。valid只表示路径完整。
"""

from dataclasses import dataclass

import torch

from supermimic.mdp import unflatten_state
from supermimic.rotations import mat_to_6d, yaw_mat
from .contracts import ForecastContext


@dataclass(frozen=True)
class ForecastResult:
    context: ForecastContext
    predicted: object
    old_reference: object
    control_steps: int
    dependency_steps: int


def forecast_rollout(state, adapter, forecast_tick, *, support_guard=None,
                     reference_dependencies=None, initial_dependency_steps=0,
                     transition_fn=None, on_step=None, reuse_current_observation=False,
                     allow_nonfinite_context=False):
    if state.pending or forecast_tick < state.tick or (forecast_tick - state.tick) % 12:
        raise ValueError("前瞻必须从无pending请求预测到未来控制网格")
    predicted = state
    depth = initial_dependency_steps
    while predicted.tick < forecast_tick:
        reference_depth = (reference_dependencies.control_depth(predicted.reference, predicted.tick)
                           if reference_dependencies is not None else 0)
        depth = max(depth, reference_depth) + 1
        # 固定旧参考前瞻中advance已重建下一时刻观测；无需再重做同一FK/观测。
        # 默认保留旧路径供数值对照；短前瞻不trim可避免每步重复验证不可变源张量。
        if not reuse_current_observation or predicted.tick == state.tick:
            predicted = adapter.refresh_current_observation(predicted)
        _, _, goal = predicted.reference.read_current_future(predicted.tick)
        with torch.profiler.record_function("mb/forecast_sm_actor"):
            action = adapter.bundle.act(predicted.sm_history, goal)
        with torch.profiler.record_function("mb/forecast_world_model"):
            next_state = (adapter.bundle.step_model(predicted.state, action, predicted.xi)
                          if transition_fn is None else transition_fn(action))
        if support_guard is not None:
            support_guard(next_state)
        predicted = adapter.advance(predicted, next_state, action, origin="imagined" if transition_fn is None else "real",
                                    trim_reference=not reuse_current_observation)
        if on_step is not None:
            on_step(predicted)
    ref, _, _ = predicted.reference.read_current_future(forecast_tick)
    p, R, v, w, q, qd = unflatten_state(predicted.state)
    Ht = yaw_mat(ref.root_R).transpose(-1, -2)
    local = lambda value: (Ht @ value.unsqueeze(-1)).squeeze(-1)
    identity = torch.eye(3, dtype=R.dtype, device=R.device).expand_as(R)
    error = torch.cat((local(p - ref.root_p),
                       mat_to_6d(ref.root_R.transpose(-1, -2) @ R) - mat_to_6d(identity),
                       local(v - ref.root_v), local(w - ref.root_w), q - ref.q, qd - ref.qd,
                       p.new_full((len(p), 1), (forecast_tick - state.tick) / 600)), -1)
    # 离线批量预处理在返回后按样本检查整段状态；单个坏请求不应使整批合法请求失效。
    # 在线及训练调用保持默认严格行为，禁止把该选项用于放宽实际执行安全条件。
    if not allow_nonfinite_context and not bool(torch.isfinite(error).all()):
        raise FloatingPointError("冻结WM前瞻产生非有限状态")
    context_valid = (torch.isfinite(error).all(dim=-1) if allow_nonfinite_context
                     else torch.ones(len(p), dtype=torch.bool, device=p.device))
    ctx = ForecastContext(error, context_valid, state.tick, forecast_tick)
    if reference_dependencies is not None:
        depth = max(depth, reference_dependencies.control_depth(predicted.reference, forecast_tick, future_steps=0))
    return ForecastResult(ctx, predicted, ref, (forecast_tick - state.tick) // 12, depth)
