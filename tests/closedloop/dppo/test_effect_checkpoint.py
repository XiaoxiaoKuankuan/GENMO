"""效果检查的配对分组与安全指标回归，仅在服务器1测试进程执行。

构造四来源、每来源四动作、每动作两个种子，验证同动作种子先聚合、正收益区间、
正负抵消与新物理失败守门。该测试只证明统计与记录逻辑，不代表任何真实模型收益。
"""
import pytest
from gem.closedloop.dppo.effect_checkpoint import summarize_effect


def example():
    tasks, pairs = [], []
    for source in range(4):
        for sample in range(4):
            for seed in (42, 1729):
                task = dict(task_id=f'{source}:{sample}:{seed}', dataset=str(source), sample_id=str(sample))
                tasks.append(task); pairs.append(dict(task_id=task['task_id'], reward_delta=1.))
    evaluation = dict(plan=dict(tasks=tasks), paired_baseline=dict(pairs=pairs,
        new_failures=0, failure_recovered=0, duration_delta=dict(mean=0.)))
    update = dict(actor=dict(optimizer_steps=8, early_stop_reason=None, steps=[]),
                  kl=dict(mean_joint_kl=.01), critic={})
    return evaluation, update


def test_action_grouped_interval_never_counts_two_seeds_as_two_motions():
    evaluation, update = example()
    result = summarize_effect(evaluation, update, iteration=200)
    assert result['action_groups_by_source'] == {str(i):4 for i in range(4)}
    assert result['cluster_bootstrap_95_interval'] == [1.,1.]
    assert result['reliable_improvement_on_fixed_subset']
    assert not result['automatically_extend_training']
    for index, row in enumerate(evaluation['paired_baseline']['pairs']):row['reward_delta'] = 1. if index%2 else -1.
    result = summarize_effect(evaluation, update, iteration=200)
    assert result['cluster_bootstrap_95_interval'] == [0.,0.]
    assert not result['reliable_improvement_on_fixed_subset']


def test_reward_increase_cannot_hide_more_failures_or_missing_tasks():
    evaluation, update = example(); evaluation['paired_baseline']['new_failures'] = 1
    assert not summarize_effect(evaluation, update, iteration=100)['reliable_improvement_on_fixed_subset']
    evaluation['paired_baseline']['pairs'].pop()
    with pytest.raises(ValueError, match='omit'):summarize_effect(evaluation, update, iteration=100)
