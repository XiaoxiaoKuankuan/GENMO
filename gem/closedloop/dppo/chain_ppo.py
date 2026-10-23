"""完整扩散链PPO的两遍流式计算，覆盖合同指定的全部随机步而不保存整条反向图。

第一遍读取真实rollout中配置指定的全部可训练随机步与旧核，得到每链FP64统计；
上层只据这些步骤之和决定一个PPO裁剪分支和一个GAE系数。第二遍在相同Actor
版本上逐微批重算联合log-ratio，以已固定的完整链导数反传。
第二遍按最多1024条链分块：每条链只编码一次条件，在块内完成其全部随机步的
去噪微批反向后，立即反传并释放该块条件图，参数梯度跨块累计。可显式传入None
保留无界条件图作为等价性验证参考；这不是新的优化器minibatch或独立Adam步骤。
全局链数始终是唯一loss分母，训练步之间不平均、不折扣，不对展开的内部转移
单独执行PPO clipping。这里只限制反向条件图，第一遍只读条件与rollout缓存另有边界。

缓存只保存小型统计量；行为链和旧核来自不可变RolloutTensorCache。冻结步骤和
确定性末步不进入高斯概率。首尾参数版本、条件和重算值受校验；异常交给外层分布式
事务统一拒绝与回滚。本模块不执行Adam、不改变噪声、奖励、物理状态或BC定义。
"""
from __future__ import annotations

import torch

from .execution_checks import require_tensor
from .gaussian_probability import gaussian_statistics, chain_statistics, ppo_surrogate
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
    return chain_statistics(records[:,:,2],records[:,:,0],records[:,:,3])


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
    surrogate=ppo_surrogate(from_records(records),cache.get('advantages',rows),clip=clip,reduction='joint_sum')
    coefficients=-surrogate['objective_log_gradient']/global_chain_count
    chain_chunk=len(rows) if condition_chunk_chains is None else condition_chunk_chains
    errors=[]
    for begin in range(0,len(rows),chain_chunk):
        end=min(begin+chain_chunk,len(rows))
        conditions=ConditionGraphCache(policy,cache,include_frozen=False)
        conditions.prime(rows[begin:end])
        # i始终是本rank原链索引；块边界不重新归一化，也不拆开一条链的PPO系数。
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
            (stats['joint_log_ratio']*coefficients[indices]).sum().backward()
        conditions.backward()
        # 未反传的KL诊断图仍可经最后一次params/stats持有ConditionGather梯度池；
        # 新块prime之前一起释放，避免相邻两块FP64条件池同时驻留。
        del params,stats,conditions
    maximum=torch.stack(errors).max()
    require_tensor(maximum<=1e-4,'Streaming chain recomputation changed probabilities within one Actor version',FloatingPointError)
    return dict(loss=-surrogate['objective'].sum().detach()/global_chain_count,recompute_max_error=maximum)
