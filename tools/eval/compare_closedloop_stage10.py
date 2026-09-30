#!/usr/bin/env python3
"""第十步完整 held-out 评估的只读、逐任务配对比较。

输入必须是 evaluation_shards 合并发布的两个完整报告，具有同一份父计划、完整
task_id 集合、采样噪声起点、音乐切点、最长 episode 时长、训练/奖励/执行身份和
分片工具实现 SHA；两个 checkpoint SHA 必须不同。完成标记、独立验收、报告和
逐 episode 文件的 SHA 都重新核验。不会加载模型、调用 GPU、执行优化或重定义
奖励；所有全体/来源/seed 奖励统计直接复用 aggregate_evaluation。

输出保留每个 task 的初始值、最终值和差值，同时区分固定任务平均与真实执行区间
加权指标。失败提前结束会缩短实际执行时长，控制区间平均可能存在幸存偏差，因此
总回报、执行时长、失败及拒绝必须共同报告；不能仅从奖励率上涨推断策略提升。
四类失败配对始终完整列出：双方未失败、新增失败、失败恢复、双方失败，不删除
不利样本。少量更新前后结果仅是固定任务的小步比较，不证明收敛或长期改进。

命令行只在指定的新 output 文件发布比较 JSON，不修改输入、源代码、checkpoint
或原始评估证据；正式原始记录必须来自可信运行。临时文件原子发布且拒绝覆盖。
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gem.closedloop.dppo.evaluation import aggregate_evaluation, _digest, _publish_json, _stats
from tools.eval.audit_closedloop_stage10 import read_json, require, sha256


VERSION = 'genmo.closedloop.stage10.paired_comparison.v1'
MERGE_VERSION = 'genmo.closedloop.stage10.evaluation_merge.v1'
FAILURE_GROUPS = ('both_without_failure', 'new_failure', 'failure_recovered', 'both_failed')


def _path(root, reference):
    value = (root/reference).resolve()
    require(value.is_relative_to(root) and value.is_file(), 'Comparison artifact missing or outside merged directory')
    return value


def _finite(value, name):
    require(type(value) in (int, float) and math.isfinite(value), f'Nonfinite comparison evidence: {name}')
    return float(value)


def _close(left, right, name):
    require(math.isclose(_finite(left, name), _finite(right, name), abs_tol=1e-8, rel_tol=1e-9), f'Comparison evidence differs: {name}')


def _load_merged(report_path):
    report_path = Path(report_path).resolve()
    root = report_path.parent
    report = read_json(report_path)
    completion = read_json(root/'completion.json')
    audit_path = root/'independent_audit.json'
    audit = read_json(audit_path)
    report_sha = sha256(report_path)
    require(completion.get('schema')==MERGE_VERSION+'.completion' and completion.get('status')=='passed', 'Merged evaluation has no successful completion marker')
    require(completion.get('report_sha256')==report_sha and completion.get('audit_sha256')==sha256(audit_path), 'Merge completion SHA differs')
    require(audit.get('schema')==MERGE_VERSION and audit.get('status')=='passed' and audit.get('report_sha256')==report_sha, 'Independent merge audit did not pass or names different report')
    require(report.get('merge_schema')==MERGE_VERSION and report.get('status')=='passed', 'Input must be a complete successful merged evaluation')
    assertions = report['merge_assertions']
    require(assertions.get('exact_full_parent_task_set') is True and assertions.get('duplicate_tasks')==0
            and assertions.get('original_task_identity_and_noise_preserved') is True
            and assertions.get('all_shards_independently_audited') is True, 'Merge did not establish complete unaltered tasks')
    require(bool(audit.get('shards')) and all(row.get('status')=='passed' for row in audit['shards']), 'A shard was not independently accepted')
    require(report.get('updates_performed') is False and report.get('training_buffer_used') is False
            and report.get('actor_gradients_unchanged') is True and report.get('rng_restored') is True,
            'Evaluation changed optimizer/gradient/RNG boundary')
    require(bool(report.get('networks_unchanged')) and all(report['networks_unchanged'].values())
            and report['network_fingerprints_before']==report['network_fingerprints_after'], 'Evaluation changed a network')
    checkpoint = Path(report['actor_identity']['checkpoint']).resolve()
    require(checkpoint.is_file() and sha256(checkpoint)==report['actor_identity']['sha256'], 'Explicit evaluated checkpoint file SHA differs')
    plan = read_json(_path(root, report['selection_path']))
    require(_digest({key: value for key, value in plan.items() if key!='plan_sha256'})==plan['plan_sha256']==report['plan_sha256'], 'Parent plan digest differs')
    require(plan.get('requested_eval_count')=='all' and plan['selected_sample_count']==plan['complete_pool_count']
            and plan['split'] in ('val', 'test') and len(plan['seeds'])>=2, 'Comparison requires the full held-out paired-sample pool and multiple seeds')
    tasks = plan['tasks']
    ids = [task['task_id'] for task in tasks]
    require(len(ids)==len(set(ids))==plan['task_count']==plan['selected_sample_count']*len(plan['seeds']), 'Parent tasks duplicate or omit a sample/seed')
    require(len(tasks)==report['requested_task_count']==report['completed_episode_count']==len(report['episode_manifests']), 'Merged episode count differs from full parent plan')
    extension = report['extension_provenance']
    require(extension.get('included_in_original_checkpoint_identity') is False and bool(extension.get('files')), 'Missing independent evaluation implementation identity')
    require(_digest([(row['relative_path'], row['sha256']) for row in extension['files']])==extension['source_manifest_sha256'], 'Evaluation implementation manifest SHA differs')
    episodes = []
    for task, reference in zip(tasks, report['episode_manifests']):
        path = _path(root, reference['path'])
        require(sha256(path)==reference['sha256'], 'Episode bytes differ from merged publication')
        episode = read_json(path)
        require(episode.get('transition_valid') is True and reference.get('transition_valid') is True
                and episode['task_id']==reference['task_id']==task['task_id'], 'Episode invalid or not aligned with full task order')
        for key in ('dataset', 'sample_id', 'split', 'seed', 'music_start_frame', 'noise_index_start'):
            require(episode[key]==task[key], f'Episode task or noise changed: {key}')
        require(episode['episode_seconds_limit']==report['episode_seconds_limit']==plan['episode_seconds'], 'Episode configured duration changed')
        _close(episode['full_music_seconds'], task['full_music_seconds'], 'full music duration')
        _close(episode['remaining_music_seconds'], task['remaining_music_seconds'], 'remaining music duration')
        controls = [row for decision in episode['decisions'] for row in decision['controls']]
        seconds = sum(_finite(row['dt_s'], 'control dt') for row in controls)
        maximum = task['remaining_music_seconds'] if plan['episode_seconds'] is None else min(task['remaining_music_seconds'], plan['episode_seconds'])
        require(seconds<=maximum+.02000001, 'Executed time exceeds the agreed task window')
        for decision in episode['decisions']:
            require(len(decision['controls'])==decision['executed_control_steps'] and decision['executed_physics_steps']==4*decision['executed_control_steps'], 'Episode actual control/physics counts differ')
            _close(decision['reward_sum'], sum(row['reward'] for row in decision['controls'])+decision['zero_step_event_reward'], 'decision actual reward')
        _close(episode['reward_sum'], sum(decision['reward_sum'] for decision in episode['decisions']), 'episode actual reward')
        episodes.append(episode)
    require(aggregate_evaluation(episodes)==report['aggregate'], 'Merged reward aggregate cannot be reproduced')
    return report, plan, episodes, {'path': str(report_path), 'sha256': report_sha}


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


def compare_evaluations(initial_report, final_report):
    """完整读取和核验后生成比较；调用方负责选择新的输出文件，函数不写输入。"""
    initial, parent, initial_episodes, initial_source = _load_merged(initial_report)
    final, other_parent, final_episodes, final_source = _load_merged(final_report)
    require(parent==other_parent, 'Initial/final full parent plan, task noise or episode duration differs')
    require(initial['actor_identity']['sha256']!=final['actor_identity']['sha256'], 'Comparison requires two different checkpoint files')
    require(initial['training_identity']==final['training_identity'], 'Initial/final training, reward or execution identity differs')
    require(initial['extension_provenance']['source_manifest_sha256']==final['extension_provenance']['source_manifest_sha256'], 'Initial/final evaluation implementation SHA differs')
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
