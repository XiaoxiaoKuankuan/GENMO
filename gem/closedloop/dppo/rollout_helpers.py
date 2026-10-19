"""当前统一世界采集器使用的独立片段GAE和真实批量扩散链拆分。

片段按真实环境、episode与行为版本分别计算，最后才做全局优势标准化。拆分
只保留原链的真实行和条件，不制造数据或复制历史。旧每环境线程采集器已删除。
"""
from __future__ import annotations

import copy
import torch


def fixed_fragment_targets(fragments, critic, device, *, critic_version, gamma_upper=.99, lambda_upper=.95):
    """分别按环境/episode/连续区间计算GAE；返回按环境顺序的行，留待全局归一化一次。"""
    from .trainer import fixed_targets
    targets = [fixed_targets(rows,critic,device, gamma_upper=gamma_upper,lambda_upper=lambda_upper,
                             reuse_values=True,critic_version=critic_version,normalize=False)
               for rows in fragments if rows]
    if not targets:
        return [],fixed_targets([],critic,device,gamma_upper=gamma_upper,lambda_upper=lambda_upper,
            reuse_values=True,critic_version=critic_version,normalize=False)
    result = {}
    for key in targets[0]:
        values = [item[key] for item in targets]
        if torch.is_tensor(values[0]):
            result[key] = torch.cat(values)
        elif not all(value==values[0] for value in values):
            raise ValueError('Independent fragment return contracts differ')
        else:
            result[key] = values[0]
    return [row for rows in fragments for row in rows],result


def split_trace(trace, index, batch_size):
    result = {}
    for key,value in trace.items():
        if key == 'conditions':
            result[key] = {name:tensor[index:index+1] for name,tensor in value.items()}
        elif torch.is_tensor(value) and key not in ('timestep_map','probability_steps'):
            if value.shape[0] != batch_size:
                raise ValueError(f'Unexpected batched trace field {key}')
            result[key] = value[index:index+1]
        else:
            result[key] = copy.deepcopy(value)
    return result
