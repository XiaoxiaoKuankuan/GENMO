"""独立的DPPO去噪时间步抽样实验合同。

每条真实行为链仍完整采集20步，old概率保持采集值。均匀实验对PPO loss无放回
抽取K个内部转移，入选概率K/T，逆概率权重T/K；按全量链×T归一化等价于抽样
loss之和除以链×K。分层K8固定保留18/19，从0至17均匀选6，对前18步乘3、末两步
乘1，再除以全量链×20。两方案均按原始步号计算去噪折扣，不改变BC频率。

全步模式不消费新随机数。抽样模式每个optimizer minibatch由rank0的可恢复Actor
生成器产生一个种子并广播，再按全局链身份派生独立CPU生成器，分片方式不影响
选择。保存种子、选择SHA和步号直方图可重建每个样本。抽样KL只是软停止估计；
最终硬KL必须仍在所有真实链、全部T步上计算，不由本模块改变或缓存拼接。
"""
from __future__ import annotations
import hashlib
import json
import torch

CONTRACT = 'dppo.uniform_denoising_without_replacement.v1'
STRATIFIED_CONTRACT = 'dppo.stratified_terminal_two_plus_uniform_six.v1'


def validate_sampling(total_steps, sampled_steps, strategy='uniform'):
    count = validate_step_count(total_steps, sampled_steps)
    if strategy not in ('uniform', 'stratified8'):
        raise ValueError('Unknown denoising sampling strategy')
    if strategy == 'stratified8' and (total_steps != 20 or count != 8):
        raise ValueError('stratified8 requires exactly 20 full steps and 8 selected steps')
    return count


def validate_step_count(total_steps, sampled_steps):
    count = total_steps if sampled_steps is None else sampled_steps
    if type(count) is not int or not 1 <= count <= total_steps:
        raise ValueError('denoising_steps_per_chain must be an integer in [1, full diffusion steps]')
    return count


def chain_steps(seed, global_index, total_steps, sampled_steps, strategy='uniform'):
    count = validate_sampling(total_steps, sampled_steps, strategy)
    if count == total_steps:
        return list(range(total_steps))
    if type(seed) is not int or seed < 0 or type(global_index) is not int or global_index < 0:
        raise ValueError('Sampled denoising steps require a nonnegative seed and global chain identity')
    contract = STRATIFIED_CONTRACT if strategy == 'stratified8' else CONTRACT
    digest = hashlib.blake2b(f'{contract}:{seed}:{global_index}'.encode(), digest_size=8).digest()
    generator = torch.Generator().manual_seed(int.from_bytes(digest, 'little') % (2**63-1))
    # 排序仅方便连续状态读取，不改变无放回抽样分布。
    if strategy == 'stratified8':
        return sorted(torch.randperm(18, generator=generator)[:6].tolist()) + [18,19]
    return sorted(torch.randperm(total_steps, generator=generator)[:count].tolist())


def plan_record(seed, global_indices, total_steps, sampled_steps, strategy='uniform'):
    count = validate_sampling(total_steps, sampled_steps, strategy)
    selection = [(index, chain_steps(seed,index,total_steps,count) if strategy=='uniform' else
                  chain_steps(seed,index,total_steps,count,strategy)) for index in global_indices]
    histogram = [0]*total_steps
    for _, steps in selection:
        for step in steps: histogram[step] += 1
    result = dict(contract=CONTRACT, full_steps=total_steps, selected_steps_per_chain=count,
        seed=seed, inclusion_probability=count/total_steps, inverse_probability_weight=total_steps/count,
        selection_sha256=hashlib.sha256(json.dumps(selection,separators=(',',':')).encode()).hexdigest(),
        step_histogram=histogram, full_20_step_hard_kl_required=True,
        pre_step_kl_scope='all_minibatch_steps' if count==total_steps else 'uniform_sample_estimate_not_full_KL',
        loss_normalization='sum_selected_objectives_divided_by_global_chains_times_selected_steps')
    if strategy == 'stratified8':
        result.update(contract=STRATIFIED_CONTRACT,strategy=strategy,
            inclusion_probability=[1/3]*18+[1.,1.],inverse_probability_weight=[3.]*18+[1.,1.],
            selected_chain_steps=[[index,steps] for index,steps in selection],
            pre_step_kl_scope='inverse_probability_weighted_estimate_not_full_KL',
            loss_normalization='sum_inverse_probability_weighted_objectives_divided_by_global_chains_times_20')
    return result
