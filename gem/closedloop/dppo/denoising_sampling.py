"""固定中间四步DPPO目标：完整20步行为，仅索引15至18具有可训练随机转移。

这四步平均是明确的代理目标，不是全20步联合概率比。前15步冻结、末步确定性，
不参与PPO分母或Gaussian KL。所有链都选择相同四步，各epoch仅打乱链顺序。
不再保留旧K8/末四步的训练兼容入口；运行身份变更后必须重新采样。
"""
from __future__ import annotations
import hashlib
import json

TRAINABLE_STEPS=(15,16,17,18)
CONTRACT='dppo.frozen15_train4_deterministic1.v1'


def validate_step_count(total_steps,sampled_steps):
    if total_steps!=20 or sampled_steps!=4:raise ValueError('The new actor has exactly four trainable stochastic steps')
    return 4


def validate_sampling(total_steps,sampled_steps,strategy='intermediate4'):
    if strategy!='intermediate4':raise ValueError('Old denoising objectives cannot run under the new policy identity')
    return validate_step_count(total_steps,sampled_steps)


def chain_steps(seed,global_index,total_steps,sampled_steps,strategy='intermediate4'):
    validate_sampling(total_steps,sampled_steps,strategy)
    return list(TRAINABLE_STEPS)


def plan_record(seed,global_indices,total_steps,sampled_steps,strategy='intermediate4'):
    validate_sampling(total_steps,sampled_steps,strategy)
    selections=[(i,list(TRAINABLE_STEPS)) for i in global_indices]
    return dict(contract=CONTRACT,strategy=strategy,full_steps=20,selected_steps_per_chain=4,seed=None,
        selection_sha256=hashlib.sha256(json.dumps(selections,separators=(',',':')).encode()).hexdigest(),
        step_histogram=[len(global_indices) if i in TRAINABLE_STEPS else 0 for i in range(20)],
        full_20_step_hard_kl_required=False,gaussian_kl_steps=list(TRAINABLE_STEPS),
        pre_step_kl_scope='all_trainable_four_steps',objective_scope='four_step_free_coordinate_mean_proxy',
        loss_normalization='sum_four_objectives_divided_by_global_chains_times_four')
