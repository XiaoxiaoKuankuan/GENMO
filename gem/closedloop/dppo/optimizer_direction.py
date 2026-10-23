"""用完整真实rollout的PPO梯度解释小批次Adam更新方向，不改变训练目标。

数据流：调用方提供当前Actor版本上的完整链反向函数；暂存已经SUM的小批次梯度，
使用配置指定可训练步的原流式联合PPO重算整批梯度并做全局SUM，随后逐张量恢复原梯度。仅rank0
保留整批梯度，接受更新后结合更新前参数测量真实Adam增量的一阶目标预测及夹角。
整批梯度与小批次梯度都采用loss最小化符号，报告的一阶目标增益因此取负内积。

本模块不执行optimizer.step，不修改Adam、学习率、BC、优势、概率、采样或物理状态。
诊断额外执行一次整批反向且占用rank0一份梯度，默认生产训练不启用；它解释方向，
不能把一阶预测或代理目标变化称为物理收益，也不能把逐模块KL相加替代共同更新KL。
"""
from __future__ import annotations

import math
import torch

from .gradient_merge import take_gradients, _group


class FullRolloutDirection:
    def __init__(self, actor, gradients, geometry):
        self.actor, self.gradients, self.geometry = actor, gradients, geometry

    @classmethod
    def capture(cls, actor, distributed, backward):
        """所有rank参加反向与SUM；无论成功或异常均恢复原小批次梯度对象。"""
        minibatch = {name:p.grad for name,p in actor.named_parameters()
                     if p.requires_grad and p.grad is not None}
        for parameter in actor.parameters():
            if parameter.requires_grad: parameter.grad = None
        try:
            backward()
            distributed.sum_gradients(actor)
            full = take_gradients(actor)
            if distributed.rank != 0:
                return None
            device = next(actor.parameters()).device
            values = torch.zeros(3, dtype=torch.float64, device=device)
            for name in set(minibatch) | set(full):
                a, b = minibatch.get(name), full.get(name)
                if a is not None: values[0] += a.double().square().sum()
                if b is not None: values[1] += b.double().square().sum()
                if a is not None and b is not None: values[2] += (a.double()*b.double()).sum()
            a2, b2, dot = values.cpu().tolist()
            return cls(actor, full, dict(minibatch_gradient_norm=math.sqrt(a2),
                full_rollout_gradient_norm=math.sqrt(b2),
                cosine_minibatch_full_gradient=dot/math.sqrt(a2*b2) if a2*b2 > 0 else None))
        finally:
            for name, parameter in actor.named_parameters():
                if parameter.requires_grad: parameter.grad = minibatch.get(name)

    @torch.no_grad()
    def finish(self, before):
        totals = {}
        for name, parameter in self.actor.named_parameters():
            if not parameter.requires_grad: continue
            delta = parameter.double()-before[name].double()
            gradient = self.gradients.get(name)
            values = torch.stack((delta.square().sum(), delta.new_zeros(()) if gradient is None else
                gradient.double().square().sum(), delta.new_zeros(()) if gradient is None else
                -(delta*gradient.double()).sum()))
            for key in ('all', _group(name)):
                totals[key] = values.clone() if key not in totals else totals[key]+values
        modules = {}
        for key, values in zip(totals, torch.stack(list(totals.values())).cpu().tolist()):
            d2, g2, improvement = values
            modules[key] = dict(linearized_full_objective_gain=improvement,
                cosine_update_full_objective_gradient=improvement/math.sqrt(d2*g2) if d2*g2 > 0 else None)
        self.gradients.clear()
        return dict(**self.geometry, modules=modules,
            scope='current_parameter_version_full_rollout_true_clipped_PPO_gradient_not_physical_return')
