"""仅在参考投影失败路径保存可独立复现的CPU证据，不改变训练数值逻辑。

输入来自实际失败的投影调用，包含原始十帧残差、已提交历史、名义SO(3)轨迹及
投影器参数；递归detach并复制到CPU，避免保留BPTT图或占用训练显存。想象异常
同时记录真实起点身份、失败想象时刻、模型版本和当前更新的有效样本身份。
常规完整checkpoint保存最近一次失败，恢复后保留这份历史；正常投影不复制张量。
此证据只说明数值拒样及其复现边界，不把想象样本当作真实物理失败。
"""

import torch


def cpu_snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device='cpu', copy=True)
    if isinstance(value, dict):
        return {k: cpu_snapshot(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_snapshot(v) for v in value)
    return value


def projection_snapshot(projector, raw, left, anchor, nominal_rotation):
    """只由异常处理器调用；保留整个失败批次以复现批处理求解的数值上下文。"""
    inputs = dict(raw=raw, left=left, anchor=anchor, nominal_rotation=nominal_rotation)
    return dict(schema='reference_projection_failure.v1', inputs=cpu_snapshot(inputs),
        requires_grad={k: bool(v.requires_grad) if isinstance(v, torch.Tensor) else False
                       for k, v in inputs.items()}, gradient_enabled=torch.is_grad_enabled(),
        joint_names=tuple(projector.joint_names), buffers=cpu_snapshot(projector.state_dict()),
        options={k: getattr(projector, k) for k in ('iterations', 'coupled_iterations',
            'tolerance', 'rotation_velocity', 'rotation_acceleration',
            'constrain_root_angular_acceleration')}, original_device=str(raw.device))


def retain_partial_failure(control, error, start, starts, *, offset, horizon, context):
    identity = {k: starts[k] for k in ('song', 'frame', 'start_frame', 'episode') if k in starts}
    control.last_imagination_failure = dict(
        schema='alternating_partial_projection_failure.v1', origin='imagination_not_physics',
        context=cpu_snapshot(context), offset=offset, horizon=horizon,
        imagination_step=getattr(error, 'imagination_step', None),
        bad_mask=cpu_snapshot(error.bad_mask), state=cpu_snapshot(start),
        failure_state=cpu_snapshot(getattr(error, 'imagination_state', None)),
        batch_identity=cpu_snapshot(identity),
        projection_inputs=getattr(error, 'projection_inputs', None),
        certificate=cpu_snapshot(getattr(error, 'certificate', {})),
        diagnostic=cpu_snapshot(getattr(error, 'diagnostic', {})), error=str(error))


def checkpoint_projection_evidence(extra, control):
    """调用方已经保存CPU现场；常规checkpoint只挂接，不再复制大张量。"""
    evidence = getattr(control, 'last_imagination_failure', None)
    if evidence is not None:
        extra['imagination_failure'] = evidence


def restore_projection(evidence, device='cpu'):
    """离线重建真实失败调用，不需要Actor/WM或歌曲缓存；不自动运行求解。"""
    from types import SimpleNamespace
    from .reference_constraints import ReferenceProjection
    buffers = {k: v.to(device) for k, v in evidence['buffers'].items()}
    projector = ReferenceProjection(SimpleNamespace(joint_names=evidence['joint_names']),
        buffers['bounds'], **evidence['options']).to(device)
    projector.load_state_dict(buffers)
    inputs = {k: (v.to(device=device, copy=True).requires_grad_(evidence['requires_grad'][k])
                  if isinstance(v, torch.Tensor) else v) for k, v in evidence['inputs'].items()}
    return projector, inputs
