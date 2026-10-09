"""大规模 Stage10 的固定任务效果检查记录，不自动扩展训练预算。

在配置的第100/200/500轮使用同一验证计划、种子和部署合同，与初始化模型配对。
同一动作的两个种子先归为一个组，按来源分别重采样动作组，四来源等权汇总；避免
把同一动作的两个种子当作完全独立样本。区间只是有限验证集上的诊断证据，不代表
总体收敛保证。失败数和执行时长同时守门，不能仅凭 loss、参数变化或 KL 通过验收。

独立 NumPy Generator 不改变训练/采样 RNG。输出包含有效更新、KL、Critic 及下一步
检查建议，持久化到会话证据。该函数不会启动训练、改学习率、修改已有预算或自动
续跑1000/10000轮；用户授权的500轮依旧只是资源上限。
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np


def summarize_effect(evaluation, update, *, iteration, repeats=4096):
    paired = evaluation.get('paired_baseline')
    if paired is None:
        raise ValueError('Effect checkpoint requires a matched initial evaluation')
    tasks = {task['task_id']: task for task in evaluation['plan']['tasks']}
    if set(tasks) != {row['task_id'] for row in paired['pairs']}:
        raise ValueError('Effect checkpoint cannot omit fixed validation tasks')
    grouped = defaultdict(lambda: defaultdict(list))
    for row in paired['pairs']:
        task = tasks[row['task_id']]
        grouped[task['dataset']][task['sample_id']].append(row['reward_delta'])
    if len(grouped) != 4 or any(len(samples) < 2 for samples in grouped.values()):
        raise ValueError('Effect checkpoint requires four sources with independent action groups')
    rng = np.random.default_rng(181738)
    draws, means, counts = [], [], {}
    for source, samples in sorted(grouped.items()):
        values = np.array([np.mean(deltas) for _, deltas in sorted(samples.items())], dtype=np.float64)
        draws.append(values[rng.integers(0, len(values), size=(repeats, len(values)))].mean(1))
        means.append(float(values.mean())); counts[source] = len(values)
    lower, upper = np.quantile(np.stack(draws).mean(0), [.025,.975]).tolist()
    safeguards = (paired['new_failures'] <= paired['failure_recovered'] and paired['duration_delta']['mean'] >= 0)
    reliable = bool(safeguards and lower > 0)
    actor = update['actor']
    return dict(schema='stage10.effect_checkpoint.v1', iteration=iteration,
        paired_source_balanced_reward_delta=float(np.mean(means)),
        cluster_bootstrap_95_interval=[lower,upper], action_groups_by_source=counts,
        resampling_scope='within_source_action_groups_with_paired_seeds_fixed_validation_subset',
        safeguard_passed=safeguards, reliable_improvement_on_fixed_subset=reliable,
        validation_is_not_convergence_guarantee=True,
        actor_optimizer_steps=actor['optimizer_steps'], actor_early_stop=actor['early_stop_reason'],
        parameter_update_observations=[step.get('parameter_update_observation') for step in actor['steps']],
        gradient_contributions=[step.get('gradient_contributions') for step in actor['steps']],
        actual_policy_mean_change=update['kl'].get('effective_mean_change'),
        final_mean_joint_kl=update['kl']['mean_joint_kl'], critic=update['critic'],
        recommendation='review_fixed_subset_gain_before_user_authorized_extension' if reliable else
            'inspect_effective_actor_change_BC_PPO_reward_KL_stop_and_value_before_extension',
        automatically_extend_training=False)
