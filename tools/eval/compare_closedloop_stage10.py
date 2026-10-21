#!/usr/bin/env python3
"""当前第二阶段固定歌曲、种子与物理合同的逐任务配对比较。

读取periodic_monitor产生的当前32任务合并报告，重验原episode与无损控制侧车
的SHA、完整任务计划、奖励积分聚合及模型不变证据。保留新增失败、恢复失败、
双方失败和双方成功四类，分别报告总回报、真实时长、每秒回报与活动度。
仅处理当前报告，不加载退役分片验收、旧checkpoint或历史配置；固定子集结果
始终标记为非完整留出集验收。只读输入，输出须为新路径，不运行模型或物理。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gem.closedloop.dppo.evaluation import _publish_json, _stats, aggregate_evaluation
from gem.closedloop.dppo.compact_evidence import read_episode
from gem.closedloop.dppo.training_checkpoint import file_sha256 as sha256
from gem.closedloop.dppo.execution_compatibility import without_source_records
import json

VERSION = 'genmo.closedloop.stage10.paired_comparison.v1'
FAILURE_GROUPS = ('both_without_failure', 'new_failure', 'failure_recovered', 'both_failed')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f'Nonfinite JSON {value}')))


def _episode_metrics(episode):
    values = aggregate_evaluation([episode])['overall']
    result = {'episode_reward': values['reward_sum'], 'executed_seconds': values['executed_seconds'],
              'executed_control_steps': values['executed_control_steps'], 'physical_failure': int(episode['physical_failure']),
              'rejection_count': values['rejection_count'], 'finite_invalid_reference_count': values['finite_invalid_reference_count'],
              'reward_per_executed_second': values['reward_per_executed_second'],
              'event_reward_sum': values['reward_sum']-sum(row['integrated_reward_sum'] for row in values['components'].values())}
    for name, row in values['activity'].items():
        result['activity.'+name] = row['mean']
    for name, row in values['raw_errors'].items():
        result['raw_error.'+name] = row['mean']
    for name in ('prefix_frames', 'latency_seconds'):
        for statistic in ('mean', 'p50', 'p95', 'max'):
            result[name+'.'+statistic] = values[name][statistic]
    for name, row in values['components'].items():
        result['component_integral.'+name] = row['integrated_reward_sum']
    return result


def _load_periodic(report_path):
    """核验当前固定任务及不可变来源文件，不要求训练先落checkpoint。"""
    from gem.closedloop.dppo.periodic_monitor import VERSION as PERIODIC_VERSION
    from gem.closedloop.dppo.periodic_monitor import _episode_row, _plan_valid, _verify_episode
    report_path = Path(report_path).resolve(strict=True)
    report = read_json(report_path)
    require(report.get('schema') == PERIODIC_VERSION+'.merge' and report.get('status') == 'passed'
            and report.get('mode') == 'periodic_subset', 'Input is not a complete periodic subset merge')
    plan = report['plan']
    _plan_valid(plan)
    require(report['plan_sha256'] == plan['plan_sha256'] and 'periodic_partition' not in plan,
            'Periodic comparison requires the full fixed subset parent plan')
    assertions = report['assertions']
    require(assertions.get('all_fixed_tasks_included') is True and assertions.get('rank_models_unchanged') is True
            and assertions.get('original_task_identity_and_noise_preserved') is True
            and assertions.get('full_heldout_acceptance') is False, 'Periodic evidence assertions incomplete')
    require(report.get('evaluation_identity') is not None and bool(report['actor_identity'].get('model_fingerprint')),
            'Periodic comparison requires explicit execution/reward identity and memory model fingerprint')
    references = report['source_episode_manifests']
    episodes = [read_episode(reference['path']) for reference in references]
    require(len(episodes) == len(references) == plan['task_count'] == report['task_count'], 'Periodic report omits tasks')
    for task, episode, reference in zip(plan['tasks'], episodes, references):
        path = Path(reference['path']).resolve(strict=True)
        require(path.is_file() and sha256(path) == reference['sha256'],
                'Periodic original episode bytes differ')
        require(episode.get('transition_valid') is True and reference['task_id'] == task['task_id'],
                'Periodic episode invalid or out of order')
        _verify_episode(episode, task, plan['episode_seconds'])
    require(aggregate_evaluation(episodes) == report['aggregate']
            and [_episode_row(e) for e in episodes] == report['episode_metrics'], 'Periodic aggregate cannot be reproduced')
    adapted = dict(report, episode_seconds_limit=plan['episode_seconds'],
        training_identity=report['evaluation_identity'],
        extension_provenance=dict(source_manifest_sha256=sha256(Path(__file__))))
    return adapted, plan, episodes, dict(path=str(report_path), sha256=sha256(report_path))


def _comparison_group(pairs, initial_episodes, final_episodes):
    metrics = sorted({name for row in pairs for name in row['initial']})
    task_statistics = {}
    for name in metrics:
        sides = {side: _stats([row[side][name] for row in pairs if row[side].get(name) is not None]) for side in ('initial', 'final', 'delta')}
        task_statistics[name] = sides
    initial = aggregate_evaluation(initial_episodes)['overall']
    final = aggregate_evaluation(final_episodes)['overall']
    return {'paired_task_count': len(pairs), 'task_weighted_metrics': task_statistics,
            'actual_execution_aggregate': {'initial': initial, 'final': final},
            'failure_pair_counts': {name: sum(row['failure_pair']==name for row in pairs) for name in FAILURE_GROUPS}}


def paired_execution_identity(identity):
    """配对Actor效果时要求物理/奖励/核和执行时钟相同，源码仅记录。

    新run的Critic预热文件、优化器设置和Actor初始化来源不决定冻结Actor评估的
    状态转移。它们保留在原报告中，但不能要求Stage1和更新Actor的初始化身份相同。
    其余实际执行字段保留，不自动兼容不同奖励或随机核。
    """
    identity=without_source_records(identity)
    if not isinstance(identity,dict) or 'training_identity' not in identity:return identity
    required=('schema','assets','kernel','reward','runtime','timing','physical_episode','ppo_rollout',
        'generation_horizon','physical_configuration','data_catalog','source_stage1_step',
        'torch_version','recovery')
    training=identity['training_identity']
    require(all(k in training for k in required),'Paired evaluation lacks explicit execution identity')
    return dict(identity,training_identity={k:training[k] for k in required})


def compare_evaluations(initial_report, final_report):
    """完整读取和核验后生成比较；调用方负责选择新的输出文件，函数不写输入。"""
    initial, parent, initial_episodes, initial_source = _load_periodic(initial_report)
    final, other_parent, final_episodes, final_source = _load_periodic(final_report)
    require(parent==other_parent, 'Initial/final full parent plan, task noise or episode duration differs')
    require(paired_execution_identity(initial['training_identity'])==paired_execution_identity(final['training_identity']),
            'Initial/final reward, physics, stochastic kernel or execution configuration differs')
    require(initial['episode_seconds_limit']==final['episode_seconds_limit'] and initial['deterministic']==final['deterministic'], 'Initial/final evaluation semantics differ')
    pairs = []
    for task, before, after in zip(parent['tasks'], initial_episodes, final_episodes):
        a, b = _episode_metrics(before), _episode_metrics(after)
        # 零控制步的合法终止只有事件惩罚，没有连续误差/奖励；显式缺测，不能删除任务。
        keys = sorted(set(a)|set(b))
        a, b = ({key: values.get(key, 0. if key.startswith('component_integral.') else None)
                 for key in keys} for values in (a, b))
        failure_pair = (('both_without_failure', 'new_failure'), ('failure_recovered', 'both_failed'))[bool(before['physical_failure'])][bool(after['physical_failure'])]
        pairs.append({key: task[key] for key in ('task_id', 'dataset', 'sample_id', 'seed', 'music_start_frame', 'noise_index_start')}
                     | {'failure_pair': failure_pair, 'initial': a, 'final': b,
                        'delta': {key: None if a[key] is None or b[key] is None else b[key]-a[key] for key in a}})
    def group(indices):
        return _comparison_group([pairs[index] for index in indices], [initial_episodes[index] for index in indices], [final_episodes[index] for index in indices])
    indices = list(range(len(pairs)))
    result = {'schema': VERSION, 'status': 'passed', 'sources': {'initial': initial_source, 'final': final_source},
              'checkpoints': {'initial': initial['actor_identity'], 'final': final['actor_identity']},
              'plan_sha256': parent['plan_sha256'], 'split': parent['split'], 'paired_task_count': len(pairs),
              'paired_sample_count': parent['selected_sample_count'], 'seeds': parent['seeds'],
              'episode_seconds_limit': parent['episode_seconds'], 'implementation_sha256': initial['extension_provenance']['source_manifest_sha256'],
              'comparison_implementation_sha256': sha256(Path(__file__)),
              'all_planned_tasks_included': True, 'optimized_during_comparison': False,
              'metric_semantics': {'task_weighted': 'Each fixed task has equal weight; failures and shortened episodes remain included.',
                 'actual_execution': 'Existing aggregate_evaluation over all real controls; physical failures shorten exposure and can cause survivor bias.',
                 'delta': 'final minus initial; signs alone do not establish better policy quality.',
                 'reward': 'Original recorded reward and component integrals; no reward recomputation or replacement.',
                 'interpretation': 'A bounded before/after evaluation does not prove convergence or long-term improvement.'},
              'overall': group(indices),
              'by_source': {source: group([index for index in indices if pairs[index]['dataset']==source]) for source in sorted({row['dataset'] for row in pairs})},
              'by_seed': {str(seed): group([index for index in indices if pairs[index]['seed']==seed]) for seed in parent['seeds']},
              'by_failure_pair': {name: group([index for index in indices if pairs[index]['failure_pair']==name]) for name in FAILURE_GROUPS},
              'pairs': pairs}
    result.update(mode='periodic_subset', full_heldout_acceptance=False)
    execution=initial['training_identity'].get('training_identity')
    if execution is not None:
        from gem.closedloop.dppo.actor_optimization import digest
        result.update(execution_kernel_sha256=digest(execution['kernel']),
                      execution_reward_sha256=digest(execution['reward']),
                      evaluated_actor_initialization=final['training_identity']['training_identity']['actor_initialization'])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial-report', required=True, type=Path)
    parser.add_argument('--final-report', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    result = compare_evaluations(args.initial_report, args.final_report)
    _publish_json(args.output, result)
    print(f"Stage10 paired comparison passed: {result['paired_task_count']} tasks; output={args.output}")
    return 0


if __name__=='__main__':
    raise SystemExit(main())
