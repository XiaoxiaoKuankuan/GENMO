"""真实固定链的逐去噪步骤梯度诊断，不改变训练状态或随机采样身份。

以完整20步PPO目标的全局分母计算每个步骤的独立梯度，保留原clip、优势、折扣、
精确联合概率和编译FP32/局部FP64执行合同。所有rank按同一步骤SUM梯度，统计
各模块范数；范数不可直接相加，也不冒称Adam实际更新贡献。条件编码保留计算图，
按当前步骤的全部微批回传，诊断结束恢复原grad引用。无optimizer.step、无新物理
采集、不重写old概率。它是一次显式诊断的额外计算，不计入普通训练耗时。
"""
from __future__ import annotations
import math
import torch
from gem.closedloop.dppo.batch_execution import MicrobatchGradientAccumulator
from gem.closedloop.dppo.execution_checks import policy_phase,require_tensor
from gem.closedloop.dppo.policy import masked_joint_log_prob
from gem.closedloop.dppo.tensor_cache import ConditionGraphCache
from gem.closedloop.dppo.updater_v2 import _parameters,_old_kernel,_joint_kl


def step_contributions(policy,rows,targets,cache,collective,*,microbatch=128):
    selected=[r for i,r in enumerate(rows) if bool(targets['valid'][i]) and bool(r.free_mask.any())]
    global_chains=sum(collective.all_gather_object(len(selected)))
    if not global_chains:raise ValueError('Gradient diagnostic requires actual valid chains')
    actor=policy.actor;original={n:p.grad for n,p in actor.named_parameters()};reports=[]
    try:
        for step in range(policy.steps):
            actor.zero_grad(set_to_none=True)
            stats=torch.zeros(5,dtype=torch.float64,device=collective.device)
            with policy_phase(policy):
                accumulator=MicrobatchGradientAccumulator(actor)
                conditions=ConditionGraphCache(policy,cache);conditions.prime(selected)
                for begin in range(0,len(selected),microbatch):
                    current=selected[begin:begin+microbatch];steps=[step]*len(current)
                    parameters,mask=_parameters(policy,current,steps,collective.device,cache,conditions)
                    observed=cache.get('chain',current,[step+1]*len(current))
                    logprob=masked_joint_log_prob(observed,parameters['mean'],parameters['std'],mask)
                    ratio=(logprob-cache.get('old_log_prob',current,steps)).exp()
                    advantage=cache.get('advantages',current)*cache.denoising_discounts(steps,20,.99)
                    loss=-torch.minimum(ratio*advantage,ratio.clamp(.99,1.01)*advantage).sum()/(global_chains*20)
                    require_tensor(torch.isfinite(loss),'Invalid diagnostic loss',FloatingPointError)
                    loss.backward();accumulator.add()
                    old_mean,old_std=_old_kernel(current,steps,collective.device,cache)
                    kl=_joint_kl(parameters,mask,old_mean,old_std).detach()
                    stats+=torch.stack((kl.sum(),ratio.detach().sum(),((ratio<.99)|(ratio>1.01)).sum(),
                        loss.detach(),kl.new_tensor(len(current))))
                accumulator.finish();conditions.backward()
            collective.sum_gradients(actor);stats=collective.sum_tensor(stats).cpu().tolist()
            if collective.rank==0:
                squares={}
                for name,p in actor.named_parameters():
                    if p.grad is not None:
                        key=name.split('.')[0];squares[key]=squares.get(key,0.)+float(p.grad.double().square().sum())
                norm=math.sqrt(sum(squares.values()))
                if not math.isfinite(norm):raise FloatingPointError('Nonfinite per-step gradient')
                reports.append(dict(step=step,gradient_norm=norm,
                    module_gradient_norms={k:math.sqrt(v) for k,v in squares.items()},
                    mean_joint_kl=stats[0]/stats[4],ratio_mean=stats[1]/stats[4],clip_fraction=stats[2]/stats[4],
                    objective_contribution=stats[3],global_chains=int(stats[4])))
    finally:
        for name,p in actor.named_parameters():p.grad=original[name]
    return dict(per_step=reports,normalization='full_global_valid_chains_times_20',
        gradient_reduction='global_SUM',individual_norms_are_not_additive=True,
        optimizer_steps=0,old_probabilities_unchanged=True)
