"""汇总64首训练样本的两次基线/两次最好模型整歌物理评价并固定展示选择。

输入为互斥清单分片的完整评价JSON与同次真实轨迹，不读取训练损失、不重新选择模型。
逐首检查初态、物理参数、音乐预算及模型身份，先对同首两次运行求均值，再按歌曲等权
汇总整体与来源；重复差异单独报告。展示排序预先固定为完成率差异、整歌Tracking绝对
差值、根RMSE绝对差值，去除同音乐身份/音频重复后取三首，改善和退化一并保留。
结果仅代表本轮Stage2实际使用的训练缓存，不代表完整原始数据集或未见音乐泛化。
输出新JSON目录，保留逐歌指标、重复波动、选择依据及真实轨迹路径供渲染和复核。
"""

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = {
    'full_horizon_tracking': 1, 'root_position_rmse_m': -1, 'root_orientation_rmse_rad': -1,
    'joint_q_rmse_rad': -1, 'joint_qd_rmse_rad_s': -1, 'body_orientation_rmse_rad': -1,
    'jerk_rms': -1, 'contact_foot_slip_mps': -1, 'total_return': 1, 'discounted_return': 1,
    'mean_abs_joint_speed': 0, 'joint_range_mean': 0,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows, seen, models = [], set(), set()
    manifests = sorted(args.root.glob('shard*/manifest.json'))
    if not manifests:
        raise ValueError('没有完整评价分片')
    task_manifest = json.loads(manifests[0].read_text())['config']['tasks']
    tasks = {t['id']: t for t in json.loads(Path(task_manifest).read_text()) if t['split'] == 'train'}
    for path in manifests:
        directory = path.parent
        manifest = json.loads(path.read_text())
        complete = json.loads((directory/'completed.json').read_text())
        if complete['optimizer_steps'] or not complete['actor_unchanged']:
            raise ValueError('评价改变了模型')
        models.add((manifest['checkpoint'], manifest['adapter_sha256']))
        reports = [json.loads((directory/f'train_{mode}.json').read_text())
                   for mode in ('zero', 'repeat', 'adapter', 'adapter_repeat')]
        if any(not report['full_song_evaluation'] for report in reports):
            raise ValueError('不接受行政截断作为整歌结果')
        by_id = [{episode['task']: episode for episode in report['episodes']} for report in reports]
        if any(set(index) != set(manifest['task_ids']) for index in by_id):
            raise ValueError('分片评价歌曲不完整')
        for column, task_id in enumerate(manifest['task_ids']):
            if task_id in seen:
                raise ValueError('分片重复评价歌曲')
            seen.add(task_id)
            episodes = [index[task_id] for index in by_id]
            for episode in episodes[1:]:
                for key in ('initial_state_sha256', 'physical_xi_sha256', 'evaluation_budget_steps', 'planned_steps'):
                    if episode[key] != episodes[0][key]:
                        raise ValueError(f'{task_id} 配对身份不符：{key}')
            task = tasks[task_id]
            metrics = {}
            for key, direction in METRICS.items():
                values = [episode[key] for episode in episodes]
                if any(v is None for v in values):
                    continue
                baseline, adapter = float(np.mean(values[:2])), float(np.mean(values[2:]))
                metrics[key] = dict(baseline=baseline, adapter=adapter, delta=adapter-baseline,
                    change_percent=100*(adapter/baseline-1) if baseline else None,
                    baseline_repeat_abs=abs(values[1]-values[0]), adapter_repeat_abs=abs(values[3]-values[2]),
                    favorable_direction=direction)
            completion = [float(episode['full_music_complete']) for episode in episodes]
            stem = f'{column:03d}_{task_id}.json'
            rows.append(dict(task=task_id, song_id=task['song_id'], source=task['source'],
                audio_sha256=task['audio_sha256'], planned_seconds=episodes[0]['planned_steps']*.02,
                baseline_completion=float(np.mean(completion[:2])), adapter_completion=float(np.mean(completion[2:])),
                reasons=[episode['reason'] for episode in episodes], metrics=metrics,
                baseline_trajectory=str(directory/'train_zero_trajectories'/stem),
                adapter_trajectory=str(directory/'train_adapter_trajectories'/stem),
                first_run_baseline=episodes[0], first_run_adapter=episodes[2]))
    if seen != set(tasks) or len(models) != 1:
        raise ValueError('训练清单覆盖不完整或分片模型不同')

    def aggregate(subset):
        summary = dict(songs=len(subset), baseline_completion=np.mean([r['baseline_completion'] for r in subset]),
                       adapter_completion=np.mean([r['adapter_completion'] for r in subset]), metrics={})
        for key, direction in METRICS.items():
            values = [r['metrics'][key] for r in subset if key in r['metrics']]
            if not values:
                continue
            baseline = float(np.mean([v['baseline'] for v in values]))
            adapter = float(np.mean([v['adapter'] for v in values]))
            summary['metrics'][key] = dict(baseline=baseline, adapter=adapter, delta=adapter-baseline,
                change_percent=100*(adapter/baseline-1) if baseline else None,
                improved_songs=sum(v['delta']*direction > 0 for v in values) if direction else None,
                regressed_songs=sum(v['delta']*direction < 0 for v in values) if direction else None,
                mean_baseline_repeat_abs=float(np.mean([v['baseline_repeat_abs'] for v in values])),
                mean_adapter_repeat_abs=float(np.mean([v['adapter_repeat_abs'] for v in values])))
        return summary

    ranked = sorted(rows, key=lambda r: (abs(r['adapter_completion']-r['baseline_completion']),
        abs(r['metrics']['full_horizon_tracking']['delta']), abs(r['metrics']['root_position_rmse_m']['delta'])), reverse=True)
    selected, audio_seen, songs_seen = [], set(), set()
    for row in ranked:
        if row['audio_sha256'] in audio_seen or row['song_id'] in songs_seen:
            continue
        selected.append(row)
        audio_seen.add(row['audio_sha256']); songs_seen.add(row['song_id'])
        if len(selected) == 3:
            break
    summary = dict(scope='all_64_training_cache_songs_used_by_this_Stage2_run_not_all_original_training_data',
        evaluation_repetitions_per_model=2, measured_episodes=4*len(rows), model=list(models)[0],
        overall=aggregate(rows), by_source={s: aggregate([r for r in rows if r['source'] == s]) for s in sorted({r['source'] for r in rows})},
        selection_rule='absolute_completion_difference_then_absolute_full_horizon_tracking_difference_then_absolute_root_RMSE_difference_unique_audio',
        selection_boundary='three_largest_differences_are_selected_examples_not_an_unbiased_average',
        selected=selected, episodes=rows)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    (args.output/'selected.json').write_text(json.dumps(selected, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps(dict(overall=summary['overall'], selected=[dict(task=r['task'], tracking=r['metrics']['full_horizon_tracking'], root=r['metrics']['root_position_rmse_m']) for r in selected]), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
