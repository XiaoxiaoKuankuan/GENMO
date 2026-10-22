"""同一完整行为rollout的真实联合Clipped PPO目标分组审计。

输入为生产高斯路径得到的每链四步FP64统计、固定GAE优势及原始来源标签。每条链
只计算一个完整联合ratio；分组只用于分析，不删除尾部、不修改概率、优势或loss。
输出全体、四数据来源、优势正负和当前KL尾部的目标、联合KL及剪裁比例。尾部组
会随参数变化，故只作为当次分布描述，不能跨更新冒充相同样本的因果对照。
本模块只处理小型统计张量，不执行Transformer、Adam或物理仿真。
"""
from __future__ import annotations

import torch

from .gaussian_probability import chain_diagnostics


def grouped_objective(statistics, advantages, sources, *, clip,before_statistics=None):
    from .gaussian_probability import ppo_surrogate
    objective=ppo_surrogate(statistics,advantages,clip=clip)['objective']
    kl=statistics['joint_kl'];adv=advantages.double()
    groups={'all':torch.ones_like(adv,dtype=torch.bool),'advantage_positive':adv>0,'advantage_negative':adv<0}
    for source in sorted(set(sources)):
        groups['source/'+source]=torch.tensor([s==source for s in sources],device=adv.device)
    cutoff=torch.quantile(kl,.99) if len(kl) else kl.new_tensor(0.)
    groups['KL_top1pct']=kl>=cutoff;groups['KL_ordinary']=kl<cutoff
    previous=(None if before_statistics is None else
        ppo_surrogate(before_statistics,advantages,clip=clip)['objective'])
    return {name:dict(count=int(mask.sum()),objective_mean=float(objective[mask].mean()),
        gain_from_behavior=float((objective-adv)[mask].mean()),KL_mean=float(kl[mask].mean()),
        gain_from_previous_same_membership=None if previous is None else float((objective-previous)[mask].mean()))
        for name,mask in groups.items() if mask.any()}
