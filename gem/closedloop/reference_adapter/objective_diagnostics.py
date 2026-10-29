"""训练歌曲上的BPTT分项梯度标定与定期诊断，不更新任何模型或优化器。

输入来自真实Replay起点，只在起点detach；随后复用正式闭环imagine，保留Actor、
WM、历史和参考计划的输入梯度。逐项计算Adapter参数梯度、范数及与原SM六项
Tracking/关节精度的夹角，用于事前固定小权重及日志解释，不作为收益门禁。
所有方向均指最大化该项奖励；成本已带负号和唯一一次dt。函数不读取评价歌曲，
不把WM梯度诊断当成真实物理方向认证。只在小批量定期运行，避免每轮重复多次BPTT。
"""

import time
import torch

from .alternating_learning import imagine
from .dynamics import frozen_parameters
from .vector_control import detach_state, select_state


def reward_gradients(control, dynamics, critic, starts, horizon, *, microbatch=32, gamma=.99):
    if not bool(torch.isin(starts['song'], control.bank.by_split['train']).all()):
        raise ValueError('目标梯度标定/诊断只能读取train真实起点')
    begin = time.perf_counter()
    parameters = tuple(control.adapter.parameters())
    size = sum(p.numel() for p in parameters)
    total = len(starts['state'])
    gradients, means = {}, {}
    with dynamics.adapter_phase(), frozen_parameters(critic):
        for offset in range(0, total, microbatch):
            z = detach_state(select_state(starts, slice(offset, offset+microbatch)))
            trajectory = imagine(control, dynamics, critic, z, horizon, terminal_value=False, gamma=gamma)
            terms = dict(trajectory['terms'], sm_tracking=trajectory['tracking'])
            terms['joint_precision'] = terms['precision_joint_position']+terms['precision_joint_velocity']
            for i, (name, values) in enumerate(terms.items()):
                term = values.sum()/total
                gradients.setdefault(name, z['state'].new_zeros(size))
                means[name] = means.get(name, 0)+term.detach()
                if term.requires_grad:
                    pieces = torch.autograd.grad(term, parameters, retain_graph=i < len(terms)-1, allow_unused=True)
                    gradients[name].add_(torch.cat([torch.zeros_like(p).flatten() if g is None else g.detach().flatten()
                                                   for p, g in zip(parameters, pieces)]))
            del trajectory, terms
    norms = {k: g.norm() for k, g in gradients.items()}
    def cosine(a, b):
        denom = norms[a]*norms[b]
        return float((gradients[a]@gradients[b])/denom) if float(denom) > 1e-20 else None
    report = dict(horizon=horizon, samples=total, train_songs=int(starts['song'].unique().numel()),
        weighted_discounted_means={k: float(v) for k, v in means.items()},
        parameter_gradient_norms={k: float(v) for k, v in norms.items()},
        cosine_with_sm_tracking={k: cosine(k, 'sm_tracking') for k in gradients},
        root_joint_gradient_cosine=cosine('root_non_saturating', 'joint_precision'),
        direction_convention='maximize_weighted_reward_full_closed_loop_BPTT',
        seconds=time.perf_counter()-begin, evidence='train_only_WM_gradients_not_physical_improvement')
    return report, gradients
