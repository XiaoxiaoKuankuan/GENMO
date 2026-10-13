"""固定真实数据上的梯度、参数与末端去噪输出比较。

本模块只读取给定的命名张量和行为链，报告梯度参与范围、模块方向、相对误差及
最后两步的输出变化。当前计算验收可以直接调用这些函数，不再通过旧训练脚本
间接加载更新器；所有概率门槛和对照容差由调用者明确传入。
"""
from __future__ import annotations
import torch

def module_directions(current, reference):
    groups = {}
    for name, value in current.items():
        other = reference[name]
        if value is None or other is None:
            continue
        key = '.'.join(name.split('.')[:3])
        a, b = value.double().reshape(-1), other.double().reshape(-1)
        row = groups.setdefault(key, [0., 0., 0., 0.])
        for i, scalar in enumerate((a.dot(a), b.dot(b), a.dot(b), (a-b).dot(a-b))):
            row[i] += float(scalar)
    return {key: dict(cosine=None if a*b == 0 else dot/(a*b)**.5,
                     relative_l2=None if b == 0 else (delta/b)**.5)
            for key, (a, b, dot, delta) in groups.items()}



@torch.no_grad()
def terminal_outputs(policy, rows, device):
    context = {key: torch.cat([row.context[key] for row in rows[:4]]).to(device)
               for key in rows[0].context}
    prepared = policy.prepare_conditions(context)
    result = []
    for step in (policy.steps-2, policy.steps-1):
        state = torch.stack([row.chain[step] for row in rows[:4]]).to(device)
        value = policy.transition_parameters(context, state, step, prepared=prepared)
        result.append(value['mean'].cpu())
    return torch.stack(result)



def compare_named(current, reference, *, atol, rtol):
    if current.keys() != reference.keys():
        raise AssertionError('Named tensor coverage changed')
    rows = {}
    for name, a in current.items():
        b = reference[name]
        if a is None or b is None:
            if a is not None or b is not None:
                raise AssertionError(f'Gradient participation changed: {name}')
            rows[name] = dict(gradient_present=False)
            continue
        diff, base = (a.double()-b.double()), b.double()
        rows[name] = dict(max_abs=float(diff.abs().max()),
            relative_l2=float(diff.norm()/base.norm().clamp_min(1e-30)),
            passed=bool(torch.allclose(a,b,atol=atol,rtol=rtol)))
    return dict(passed=all(r.get('passed', True) for r in rows.values()),
                atol=atol, rtol=rtol, parameter_tensors=len(rows), tensors=rows)
