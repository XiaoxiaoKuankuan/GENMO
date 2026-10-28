"""固定多歌曲探针对策略更新的实际影响诊断，不把一次浮点零变化当成数值崩溃。

在真实因果起点中按歌曲及均匀时间位置选取探针，保存完整输入、Goal和动作变化分布。
区分参数没有变化、参数变化低于输出精度、Tanh平坦区和可测有效更新；固定探针逐项
记录网络原始残差、投影残差、下一已提交帧、Goal和动作的更新传播，不把输出非零或
Tanh未饱和解释为物理收益。真实物理误差由配对评价提供，不能用本诊断替代。
参数SHA用于绑定物理审计的确切网络身份，与源码Git SHA无关。
"""

import hashlib
import torch
from supermimic.rotations import so3_log
from .vector_control import select_state


def parameter_identity(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def fixed_probes(starts, per_song=16):
    indices = []
    for song in starts['song'].unique(sorted=True):
        ids = (starts['song'] == song).nonzero().flatten()
        ids = ids[starts['frame'][ids].argsort(stable=True)]
        indices.append(ids[torch.linspace(0, len(ids)-1, min(per_song, len(ids)), device=ids.device).long()])
    return {k: v.detach().clone() for k, v in select_state(starts, torch.cat(indices)).items()}


def _distribution(value):
    value = value.detach().float().flatten()
    return dict(mean=float(value.mean()), p50=float(value.median()),
                p95=float(value.quantile(.95)), p99=float(value.quantile(.99)),
                maximum=float(value.max()), zero_fraction=float((value == 0).float().mean()))


def _rms_by_sample(value):
    return value.flatten(1).square().mean(-1).sqrt()


@torch.no_grad()
def update_effect(control, probes, before, parameter_delta):
    after = control.propose(probes)
    action = (after['action']-before['action']).square().mean(-1).sqrt()
    goal = (after['goal']-before['goal']).square().mean(-1).sqrt()
    logits = control.adapter.net(control.packed(control.fields(probes)))
    derivative = 1-logits.tanh().square()
    result = dict(parameter_delta=float(parameter_delta), action_rms_delta=float(action.square().mean().sqrt()),
                  goal_rms_delta=float(goal.square().mean().sqrt()), probe_count=len(action),
                  tanh_exact_zero_derivative_fraction=float((derivative == 0).float().mean()),
                  tanh_saturation_fraction=float((logits.tanh().abs() >= .98).float().mean()))
    result['per_song'] = {}
    for song in probes['song'].unique(sorted=True):
        mask = probes['song'] == song
        result['per_song'][str(int(song))] = {
            name: dict(p50=float(v[mask].median()), p95=float(v[mask].quantile(.95)), maximum=float(v[mask].max()),
                       zero_fraction=float((v[mask] == 0).float().mean())) for name, v in [('action', action), ('goal', goal)]}
    meaningful = (action > 1e-7) & (goal > 1e-7)
    result['meaningful_fraction'] = float(meaningful.float().mean())
    result['effective'] = bool(parameter_delta > 0) and bool(meaningful.any())
    result['diagnosis'] = ('effective' if result['effective'] else 'parameter_unchanged' if parameter_delta == 0 else
        'tanh_saturation' if result['tanh_exact_zero_derivative_fraction'] > .5 else 'output_resolution_or_projection')
    raw_before = before.get('raw_residual', before['residual'])
    raw_after = after.get('raw_residual', after['residual'])
    raw_delta = _rms_by_sample(raw_after-raw_before)
    projected_delta = _rms_by_sample(after['residual']-before['residual'])
    first_delta = _rms_by_sample(after['residual'][:, 0]-before['residual'][:, 0])
    changes = dict(raw_residual=raw_delta, projected_residual=projected_delta,
        next_committed_first_residual=first_delta, goal=goal, actor_action=action,
        next_committed_joint_q=_rms_by_sample(after['future'].q[:, 0]-before['future'].q[:, 0]),
        next_committed_joint_qd=_rms_by_sample(after['future'].qd[:, 0]-before['future'].qd[:, 0]),
        next_committed_root_position=_rms_by_sample(after['future'].root_p[:, 0]-before['future'].root_p[:, 0]),
        next_committed_root_rotation=_rms_by_sample(so3_log(
            after['future'].root_R[:, 0]@before['future'].root_R[:, 0].transpose(-1, -2))))
    result['change_distributions'] = {k: _distribution(v) for k, v in changes.items()}
    result['residual_before_after'] = {
        name: dict(before=_distribution(_rms_by_sample(old)), after=_distribution(_rms_by_sample(new)))
        for name, old, new in [('raw', raw_before, raw_after),
                              ('projected', before['residual'], after['residual']),
                              ('next_committed_first', before['residual'][:, 0], after['residual'][:, 0])]}
    identifiable = raw_delta > 1e-9
    result['projection_update_transfer'] = dict(
        identifiable_count=int(identifiable.sum()),
        projected_over_raw=_distribution(projected_delta[identifiable]/raw_delta[identifiable])
            if bool(identifiable.any()) else None,
        first_frame_over_raw=_distribution(first_delta[identifiable]/raw_delta[identifiable])
            if bool(identifiable.any()) else None,
        scope='same_causal_probes_parameter_update_not_physical_tracking_improvement')
    return result
