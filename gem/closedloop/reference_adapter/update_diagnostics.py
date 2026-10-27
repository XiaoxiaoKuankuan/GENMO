"""固定多歌曲探针对策略更新的实际影响诊断，不把一次浮点零变化当成数值崩溃。

在真实因果起点中按歌曲及均匀时间位置选取探针，保存完整输入、Goal和动作变化分布。
区分参数没有变化、参数变化低于输出精度、Tanh平坦区和可测有效更新；只有非有限值
才抛出数值异常。调用者对失活返回值暂停优化并完整保存，不增加有效更新计数。
参数SHA用于绑定物理审计的确切网络身份，与源码Git SHA无关。
"""

import hashlib
import torch
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
    return result
