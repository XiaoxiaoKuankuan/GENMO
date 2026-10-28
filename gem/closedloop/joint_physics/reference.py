"""sidecar请求快照、首次参考影响起点及数据固定坐标系的装配。

旧参考只从采集时已提交的独立缓存读取，监督未来标签不用于扩充缓存。真实历史由每个
控制时刻的state/obs还原；FutureContext只由请求时快照与旧计划经冻结WM预测。执行起点
取候选首次影响之前的真实状态并detach；目标使用同源原数据的固定episode坐标，整窗不
重新对齐。预测参考在codec/插值/SM路径上保留导数。
"""

from dataclasses import replace
from functools import lru_cache

import torch

from gem.closedloop.mb_control.contracts import ClosedLoopState, ForecastContext, GeneratedPlan, RequestTicket, ReferenceExhausted
from gem.closedloop.mb_control.reference_buffer import ReferenceBuffer
from gem.closedloop.mb_control.validation import UnsupportedModelState
from .contracts import PhysicsRejected, Window, frame_tick
from .data import slice_batch


def attach_supervised_sidecars(batch, store, adapter, codec, horizon, device):
    """在CPU装配无导数监督条件，整批搬运，避免逐样本参考校验触发CUDA同步。

    复用原attach_sidecars的全部时间/前缀/连续窗口和失败回退语义；不运行SM/WM，不改变
    独立物理子批的可微执行。只移动最终条件及58维缓存，从不创建物理Window。
    CPU/GPU FP32条件编码允许舍入差异，使用数值等价回归而非声称逐位相同。
    """
    from gem.closedloop.training import batch_to_device
    if batch["target_qpos30"].device.type != "cpu":
        raise ValueError("监督sidecar必须在数据搬上GPU之前装配")
    updated, contexts, reasons = prepare_supervised_conditions(batch, store, codec, horizon)
    features = torch.cat([context.features for context in contexts]).to(device)
    valid = torch.cat([context.valid for context in contexts]).to(device)
    moved = [ForecastContext(features[i:i+1], valid[i:i+1], context.request_tick, context.forecast_tick)
             for i, context in enumerate(contexts)]
    return batch_to_device(updated, device), moved, reasons


def pack_supervised_conditions(batch, store, codec, horizon):
    """CPU预取线程装配并打包条件；所有张量可随batch合批pin/H2D，不推进任何RNG。"""
    batch, contexts, reasons = prepare_supervised_conditions(batch, store, codec, horizon)
    batch['_forecast_features'] = torch.cat([c.features for c in contexts])
    batch['_forecast_valid'] = torch.cat([c.valid for c in contexts])
    batch['_forecast_clocks'] = [(c.request_tick, c.forecast_tick) for c in contexts]
    batch['_sidecar_reasons'] = reasons
    return batch


def unpack_supervised_conditions(batch):
    """在目标设备恢复ForecastContext视图；时刻仍是各样本独立的CPU整数。"""
    features, valid = batch.pop('_forecast_features'), batch.pop('_forecast_valid')
    contexts = [ForecastContext(features[i:i+1], valid[i:i+1], request, forecast)
                for i, (request, forecast) in enumerate(batch.pop('_forecast_clocks'))]
    return batch, contexts, batch.pop('_sidecar_reasons')


@lru_cache(maxsize=8192)
def _condition_clock(prefix, at_origin, upper_relative):
    """只用CPU整数时间和权威ReferenceBuffer公式，不构建物理Window或读取未来状态。"""
    request = 0 if at_origin else 60
    ticks = torch.arange((upper_relative + request)//20 + 1, dtype=torch.long)*20
    pose = torch.zeros(1, len(ticks), 28); pose[..., 3] = 1
    ref = ReferenceBuffer(ticks, pose, torch.zeros(1, len(ticks), 2),
                          torch.zeros_like(ticks), None, tuple(range(21)))
    mutable = request + prefix*20
    protection = int(ref.dependency_support(request-12).max()) if request else -20
    if mutable <= protection:
        raise PhysicsRejected('protected_reference_support')
    first = ref.first_influence(request, mutable)
    if request+(prefix-1)*20 > ref.upper:
        raise ReferenceExhausted('已知前缀超出旧计划')
    return first-request


def prepare_supervised_conditions(batch, store, codec, horizon):
    """轻量A路径：仅装配真实历史、合法前缀和离线Future；不创建Window或SM状态。

    原路径在离线Future拒绝前已写入的合法真实历史仍保留。时钟、窗口连续性和旧计划
    provenance逐项等价检查；未知target、接触与地面监督标签不参与条件构建。
    """
    updated = dict(batch)
    for key in ('known_qpos30', 'proprio_history', 'proprio_history_valid', 'proprio_history_times'):
        updated[key] = batch[key].clone()
    updated['meta'] = [dict(m) for m in batch['meta']]
    contexts, reasons = [], []
    for i, meta in enumerate(batch['meta']):
        tick = frame_tick(meta['decision_frame'])
        context = ForecastContext.invalid(batch['target_qpos30'][i:i+1], tick)
        reason = 'missing_real_sidecar'
        if store is not None:
            try:
                episode = store.episode(meta)
                index, p = tick//12, int(batch['prefix_frames'][i])
                if tick % 60 or index >= len(episode['state']):
                    raise PhysicsRejected('missing_real_time')
                if p < 1:
                    raise PhysicsRejected('insufficient_known_prefix')
                upper = int(episode['visible_upper'][index])
                first = tick + _condition_clock(p, tick == 0, upper-tick)
                if first//12 >= len(episode['state']):
                    raise PhysicsRejected('missing_real_time')
                if int(episode['committed_tick'][index]) > tick:
                    raise PhysicsRejected('future_plan_leak')
                if first+horizon*12 > (len(episode['state'])-1)*12:
                    raise PhysicsRejected('incomplete_real_window')
                # 在真实30Hz源点取旧计划；原codec进行相同的四元数规范化/局部anchor编码。
                frame = meta['decision_frame']
                prefix = episode['reference_qpos'][frame:frame+p][None]
                shift = prefix.new_tensor([0., 0., episode['ground_height_world_m']] + [0.]*25)
                features, _ = codec.encode_condition_features(prefix+shift)
                known = torch.cat((features, features.new_zeros(1, 60-p, 30)), 1)
                h = torch.arange(index-49, index+1)
                valid = h >= 0
                updated['proprio_history'][i] = torch.where(valid[:, None], episode['proprio'][h.clamp_min(0)], 0.)
                updated['proprio_history_valid'][i] = valid
                updated['proprio_history_times'][i] = (tick-torch.arange(49, -1, -1)*12).double()/600
                updated['known_qpos30'][i:i+1] = torch.where(batch['known_qpos30_mask'][i:i+1], known, 0.)
                updated['meta'][i]['proprio_source'] = 'real_continuous_newton_sidecar'
                context = store.request(episode, frame, p, 'cpu')
                if context.forecast_tick != first:
                    raise ValueError('离线前瞻时刻与首次影响合同不匹配')
                reason = 'available'
            except (PhysicsRejected, ReferenceExhausted) as error:
                reason = getattr(error, 'reason', 'reference_support_missing')
            except FloatingPointError:
                reason = 'nonfinite_forecast'
            except UnsupportedModelState:
                reason = 'forecast_out_of_domain'
        contexts.append(context); reasons.append(reason)
    return updated, contexts, reasons


def episode_reference(episode, adapter, device, *, visible_upper=None, lower_tick=0):
    end = len(episode['reference_qpos']) if visible_upper is None else visible_upper // 20 + 1
    begin = max(0, int(lower_tick)//20)
    qpos = episode['reference_qpos'][begin:end].to(device)[None]
    ticks = torch.arange(begin, end, dtype=torch.long, device=device) * 20
    return ReferenceBuffer(ticks, qpos, qpos.new_zeros((1, end-begin, 2)),
        episode["source_versions"][begin:end].to(device), adapter.runtime.kin, adapter.to_sm,
        version=int(episode["source_versions"][:end].max()))


def real_state(episode, tick, reference, adapter, device):
    if tick % 12 or tick < 0 or tick // 12 >= len(episode["state"]):
        raise PhysicsRejected("missing_real_time")
    i = tick // 12
    state = episode["state"][i:i+1].to(device)
    xi = episode["xi"].to(device)[None]
    obs_index = torch.arange(i-9, i+1).clamp_min(0)
    sm = episode["observation"][obs_index].to(device)[None]
    history_index = torch.arange(i-49, i+1)
    valid = (history_index >= 0).to(device)[None]
    history = episode["proprio"][history_index.clamp_min(0)].to(device)[None]
    history = torch.where(valid[..., None], history, 0.)
    return ClosedLoopState(tick, state, xi, episode["previous_action"][i:i+1].to(device), sm,
        history, valid, reference, state.new_zeros((1, 1, 35)), 0,
        int(episode["source_frames"])*20, history_origin="real", task_id=episode["source_key"])


def make_window(episode, sample, adapter, codec, horizon):
    device = sample["target_qpos30"].device
    meta = sample["meta"][0]
    request_tick = frame_tick(meta["decision_frame"])
    if request_tick % 60 or request_tick // 12 >= len(episode["state"]):
        raise PhysicsRejected("missing_real_time")
    p = int(sample["prefix_frames"][0])
    if p < 1:
        raise PhysicsRejected("insufficient_known_prefix")
    i = request_tick // 12
    upper = int(episode["visible_upper"][i])
    # 留足request-12的左差分及其插值端点；右边保留候选和H窗口全部支持。
    lower = max(0, ((request_tick-24)//20)*20)
    old = episode_reference(episode, adapter, device, visible_upper=upper, lower_tick=lower)
    old = replace(old, version=int(episode["plan_version"][i]))
    first_mutable = request_tick + p * 20
    protection = int(old.dependency_support(request_tick - 12).max()) if request_tick else -20
    if first_mutable <= protection:
        raise PhysicsRejected("protected_reference_support")
    first_influence = old.first_influence(request_tick, first_mutable)
    request = real_state(episode, request_tick, old, adapter, device)
    start = real_state(episode, first_influence, old, adapter, device)
    ticket = RequestTicket(0, request_tick, request_tick, first_mutable, first_influence, p, old.version)
    # 原codec在原数据坐标中编码；物理地面偏移只作为固定世界变换，不进入Actor特征。
    ticks = request_tick + torch.arange(p, device=device) * 20
    prefix = old.source_at(ticks)
    ground_shift = prefix.new_tensor([0., 0., episode["ground_height_world_m"]] + [0.] * 25)
    features, encoded_anchor = codec.encode_condition_features(prefix + ground_shift)
    known = torch.cat((features, features.new_zeros((1, 60-p, 30))), 1)
    known = torch.where(sample["known_qpos30_mask"], known, 0.)
    anchor = torch.cat((encoded_anchor.position_w[:, 0], encoded_anchor.yaw.reshape(1, 1)), -1)
    anchor = anchor - anchor.new_tensor([0., 0., episode["ground_height_world_m"], 0.])
    target_upper = min((len(episode['reference_qpos'])-1)*20,
                       ((first_influence+horizon*12+144+19)//20)*20)
    target = episode_reference(episode, adapter, device, visible_upper=target_upper, lower_tick=lower)
    window = Window(request, start, ticket, anchor.detach(), target, horizon, episode["source_key"],
        {"old_plan_committed_tick": int(episode["committed_tick"][i]), "old_plan_visible_upper": upper,
         "old_plan_version": old.version, "old_plan_origin": "teacher_forced_committed_before_request",
         "last_real_tick": (len(episode["state"])-1)*12, "split": episode["split"],
         "start_origin": "real_before_first_candidate_influence", "local_reanchor": False})
    window.validate()
    return window, known


def attach_sidecars(batch, store, adapter, codec, horizon, support_guard=None):
    contexts, windows, reasons = [], [], []
    updated = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    updated["meta"] = [dict(m) for m in batch["meta"]]
    for i, meta in enumerate(batch["meta"]):
        tick = frame_tick(meta["decision_frame"])
        context = ForecastContext.invalid(batch["target_qpos30"][i:i+1], tick)
        window, reason = None, "missing_real_sidecar"
        if store is not None:
            try:
                episode = store.episode(meta)
                window, known = make_window(episode, slice_batch(batch, i), adapter, codec, horizon)
                updated["proprio_history"][i:i+1] = window.request.genmo_history
                updated["proprio_history_valid"][i:i+1] = window.request.genmo_valid
                updated["proprio_history_times"][i] = (tick - torch.arange(49, -1, -1, device=known.device)*12).double()/600
                updated["known_qpos30"][i:i+1] = known
                updated["meta"][i]["proprio_source"] = "real_continuous_newton_sidecar"
                context = store.request(episode, meta["decision_frame"], window.ticket.prefix_frames,
                                        known.device, support_guard)
                if context.forecast_tick != window.start.tick:
                    raise ValueError("离线前瞻时刻与当前首次影响合同不匹配")
                reason = "available"
            except (PhysicsRejected, ReferenceExhausted) as error:
                window, reason = None, getattr(error, "reason", "reference_support_missing")
            except FloatingPointError:
                window, reason = None, "nonfinite_forecast"
            except UnsupportedModelState:
                window, reason = None, "forecast_out_of_domain"
        contexts.append(context)
        windows.append(window)
        reasons.append(reason)
    return updated, contexts, windows, reasons


def candidate_reference(window, predicted_x0, contact_logits, codec, endecoder):
    physical = endecoder.denormalize(predicted_x0.float())
    rotation = physical[..., 3:9]
    # 6D表示两行独立性检查在decode前进行；拒绝而非替换成GT。
    a, b = rotation[..., :3], rotation[..., 3:]
    if not torch.isfinite(physical).all() or bool((a.norm(dim=-1) < 1e-6).any()) or bool((torch.cross(a, b, dim=-1).norm(dim=-1) < 1e-6).any()):
        raise PhysicsRejected("degenerate_or_nonfinite_rotation")
    canonical = codec.decode_to_canonical_qpos(physical)
    world = codec.apply_world_anchor(canonical, window.anchor)
    plan = GeneratedPlan(window.ticket, world, contact_logits.sigmoid())
    protection = int(window.request.reference.dependency_support(window.request.tick-12).max()) if window.request.tick else -20
    return window.request.reference.merge_plan(plan, window.request.tick, protection)
