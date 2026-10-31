"""固定随机音乐面板的评价身份与分组汇总，不参与训练或修改物理结果。

读取预先抽样的128首独立音乐清单及参考覆盖记录，只允许实际有完整参考的任务进入
仿真；原初态失败必须保留在请求分母和失败列表中，不换歌、不补造模型结果。
逐歌曲报告保留原始数据划分、全局音乐身份和Adapter已见标签。混合总体用于主要
质量报告，Adapter未见音乐子集用于本轮开发集选模；它不是未使用过的独立测试集。
两次零Adapter回放先分别保存原始报告，再等权平均汇总标量，不合成虚假的轨迹或
episode。该文件只依赖标准库，方便用合成数据回归，不能替代真实动力学验证。
"""

import hashlib
import json
from pathlib import Path
import statistics


def panel_metadata(settings, tasks):
    selection_path = Path(settings['selection_manifest'])
    selection = json.loads(selection_path.read_text())
    coverage = json.loads(Path(settings['reference_coverage']).read_text())
    selected = {t['id']: t for t in selection}
    available = {t['id']: t for t in tasks}
    missing = {t['task'] for t in coverage['unavailable']}
    requested = settings['requested_song_count']
    if (len(selection) != requested or len(selected) != requested
            or len({t['global_music_group_id'] for t in selection}) != requested
            or len(available) != len(tasks) or set(available) & missing
            or set(available) | missing != set(selected)
            or coverage['requested'] != requested or coverage['evaluable'] != len(tasks)):
        raise ValueError('固定音乐面板的独立音乐数、参考覆盖或失败分母不一致')
    for task in tasks:
        for key in ('song_id', 'split', 'global_music_group_id', 'adapter_training_seen'):
            if task[key] != selected[task['id']][key]:
                raise ValueError(f'固定面板任务身份改变: {task["id"]} {key}')
    return dict(requested_song_count=requested, paired_evaluable_songs=len(tasks),
        reference_generation_failures=coverage['unavailable'],
        selection_manifest_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        evaluable_tasks_sha256=hashlib.sha256(Path(settings['tasks']).read_bytes()).hexdigest(),
        selection_population='adapter_unseen_music_development_panel_not_pristine_test',
        reference_failure_policy='preserve_original_initial_state_no_replacement')


def annotate_panel_report(report, tasks, metadata):
    by_id = {t['id']: t for t in tasks}
    for episode in report['episodes']:
        task = by_id[episode['task']]
        for key in ('split', 'source', 'dataset_split', 'global_music_group_id',
                    'adapter_training_seen', 'adapter_training_exact_task_seen'):
            episode[key] = task.get(key)
    report.update(metadata)
    return report


def panel_subset(report, seen=None):
    episodes = [e for e in report['episodes'] if seen is None or e['adapter_training_seen'] is seen]
    if not episodes:
        raise ValueError('固定面板选模/报告子集不能为空')
    result = dict(report, episodes=episodes, measured_evaluation_envs=len(episodes),
        report_population='all_available' if seen is None else ('adapter_seen' if seen else 'adapter_unseen'))
    fields = ('full_horizon_tracking', 'root_position_rmse_m', 'root_xy_rmse_m', 'jerk_rms',
              'joint_q_rmse_rad', 'joint_qd_rmse_rad_s', 'body_orientation_rmse_rad', 'contact_foot_slip_mps')
    for key in fields:
        values = [e[key] for e in episodes if e.get(key) is not None]
        result['mean_'+key] = statistics.mean(values) if values else None
    result.update(completion_rate=statistics.mean(float(e['full_music_complete']) for e in episodes),
        mean_tracking=statistics.mean(e['sm_terms'].get('track', 0.) for e in episodes),
        mean_beat_f1=statistics.mean(e['music'].get('f1') or 0. for e in episodes),
        constraint_failure_count=sum(e.get('constraint_failure', False) for e in episodes),
        measured_transitions=sum(e['steps'] for e in episodes))
    return result


def mean_baseline_reports(reports):
    """只平均重复回放的汇总标量；逐歌配对必须使用两份原始报告。"""
    result = {k: v for k, v in reports[0].items() if k != 'episodes'}
    for key in result:
        if key.startswith('mean_') or key == 'completion_rate':
            values = [r[key] for r in reports if r[key] is not None]
            result[key] = statistics.mean(values) if values else None
    result.update(baseline_repetitions=len(reports),
                  summary_aggregation='equal_weight_independent_repeat_summaries_no_synthetic_episodes')
    return result
