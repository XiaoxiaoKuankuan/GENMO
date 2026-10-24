"""逐去噪步骤 DPPO 的两遍流式计算。

第一遍返回 [N,K,8] 的单步真实高斯统计；每步独立计算 PPO 裁剪分支，共享链上的
上层 GAE。第二遍按相同参数版本重算每步 log-ratio，以其独立导数反传。
损失分母为全局物理链数 N 乘可训练步数 K，不将 K 步当成 K 倍物理样本。
每个步骤内自由坐标的概率仍求和，不能用坐标平均冒充真实概率。
条件编码按最多1024条链分块释放，梯度跨块SUM；微批/分块不增加Adam次数。
冻结前段、确定性末步、前缀和padding不参与概率，重算误差必须低于1e-4。
本模块不改变物理、奖励、噪声或行为数据，不直接执行优化器更新。
"""
from __future__ import annotations

import torch

from .execution_checks import require_tensor
from .gaussian_probability import gaussian_statistics
from .step_probability import step_statistics, step_surrogate
from .optimization_support import _parameters, _old_kernel
from .tensor_cache import ConditionGraphCache


@torch.no_grad()
def evaluate_chain_records(policy, rows, cache, *, microbatch, physical_changes=False):
    """返回[N,K,8]：K为合同指定的随机步数，统计列定义不随训练步数改变。"""
    device=next(policy.actor.parameters()).device
    records=torch.zeros(len(rows),len(policy.trainable_steps),8,device=device,dtype=torch.float64)
    conditions=ConditionGraphCache(policy,cache,include_frozen=False)
    conditions.prime(rows)
    pairs=[(i,s) for i in range(len(rows)) for s in policy.trainable_steps]
    for offset in range(0,len(pairs),microbatch):
        chunk=pairs[offset:offset+microbatch]
        samples=[rows[i] for i,_ in chunk];steps=[s for _,s in chunk]
        params,mask=_parameters(policy,samples,steps,device,cache,conditions)
        old_mean,old_std=_old_kernel(samples,steps,device,cache)
        stats=gaussian_statistics(cache.get('chain',samples,[s+1 for s in steps]),
            old_mean,old_std,params['mean'],params['std'],mask)
        changes=[torch.zeros_like(stats['joint_kl']) for _ in range(4)]
        if physical_changes:
            delta=params['mean'].double()-old_mean
            changes[0]=(delta.square().masked_fill(~mask,0.).sum((-2,-1))/stats['free_count'].clamp_min(1)).sqrt()
            physical=delta*policy.actor.endecoder.std.double()
            for j,section in enumerate((slice(0,3),slice(3,9),slice(9,30)),1):
                submask=mask[:,:,section]
                changes[j]=(physical[:,:,section].square().masked_fill(~submask,0.).sum((-2,-1))/
                    submask.sum((-2,-1)).clamp_min(1)).sqrt()
        index=torch.tensor([i for i,_ in chunk],device=device)
        records[index,cache.step_indices(steps)-policy.trainable_steps[0]]=torch.stack((stats['joint_kl'],
            stats['mean_coordinate_kl'],stats['joint_log_ratio'],stats['free_count'],*changes),1)
    conditions.validate_inputs()
    return records


def from_records(records):
    return step_statistics(records[:,:,2],records[:,:,0],records[:,:,3])


def backward_chain_surrogate(policy, rows, cache, records, *, clip, global_chain_count, microbatch,
                             condition_chunk_chains=1024):
    """按链块释放条件图、逐微批重算梯度；None只用于无界参考，不执行参数更新。"""
    if global_chain_count<1:
        raise ValueError('Global denominator must count complete valid chains')
    if condition_chunk_chains is not None and (type(condition_chunk_chains)is not int or condition_chunk_chains<1):
        raise ValueError('Condition chunk chains must be a positive integer or None')
    device=next(policy.actor.parameters()).device
    # 完整rollout汇总的小型FP64统计可位于CPU；反向索引和优势在Actor设备。
    # 仅搬运[N,K,8]统计，不改变行为链或概率精度，原GPU路径为零拷贝。
    records=records.to(device=device)
    if not rows:
        return dict(loss=records.new_zeros(()),recompute_max_error=records.new_zeros(()))
    surrogate=step_surrogate(from_records(records),cache.get('advantages',rows),clip=clip)
    denominator=global_chain_count*len(policy.trainable_steps)
    coefficients=-surrogate['objective_log_gradient']/denominator
    chain_chunk=len(rows) if condition_chunk_chains is None else condition_chunk_chains
    errors=[]
    for begin in range(0,len(rows),chain_chunk):
        end=min(begin+chain_chunk,len(rows))
        conditions=ConditionGraphCache(policy,cache,include_frozen=False)
        conditions.prime(rows[begin:end])
        # i始终是本rank原链索引；块边界不重新归一化，每个去噪步使用自己的PPO系数。
        pairs=[(i,s) for i in range(begin,end) for s in policy.trainable_steps]
        for offset in range(0,len(pairs),microbatch):
            chunk=pairs[offset:offset+microbatch]
            samples=[rows[i] for i,_ in chunk];steps=[s for _,s in chunk]
            indices=torch.tensor([i for i,_ in chunk],device=device)
            params,mask=_parameters(policy,samples,steps,device,cache,conditions)
            old_mean,old_std=_old_kernel(samples,steps,device,cache)
            stats=gaussian_statistics(cache.get('chain',samples,[s+1 for s in steps]),
                old_mean,old_std,params['mean'],params['std'],mask)
            errors.append((stats['joint_log_ratio'].detach()-records[indices,cache.step_indices(steps)-policy.trainable_steps[0],2]).abs().max())
            (stats['joint_log_ratio']*coefficients[indices,cache.step_indices(steps)-policy.trainable_steps[0]]).sum().backward()
        conditions.backward()
        # 未反传的KL诊断图仍可经最后一次params/stats持有ConditionGather梯度池；
        # 新块prime之前一起释放，避免相邻两块FP64条件池同时驻留。
        del params,stats,conditions
    maximum=torch.stack(errors).max()
    require_tensor(maximum<=1e-4,'Streaming chain recomputation changed probabilities within one Actor version',FloatingPointError)
    return dict(loss=-surrogate['objective'].sum().detach()/denominator,recompute_max_error=maximum)
