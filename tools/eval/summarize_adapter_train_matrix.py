"""汇总同一批完整歌曲的候选模型物理结果，并保留逐歌及来源分组证据。

输入是evaluate_adapter_train_matrix的只读候选清单、两次零基线和候选完整整歌报告。
默认保留原64首train合同；清单可显式声明128首及混合split，仍检查每首配对身份，
并分别汇总原划分、来源、本轮Adapter训练已见/未见。失败歌曲不删除，混合结果
不称为纯未见泛化；每个误差项报告有效数量和P90，避免把缺测静默当零。
按歌曲先平均重复，再对64首等权汇总；保留完成率、失败原因、六项Tracking、根及
关节误差、jerk、求解器接触脚滑和音乐。完成率优先，其次根RMSE、完整预算Tracking、
jerk，选出的模型只称为这16个候选中的训练集最好，不代表测试集最优或泛化证明。
确认阶段可读取同一候选的额外两次完整物理回放；视频只选双方完成且Tracking与根
误差均改善、音频不同的五首。按完整预算Tracking增益排序，明确它们是展示正例；
不能以这五首替代完整64首均值，也不隐藏失败或退化歌曲。所有排序来自实际报告，
不使用训练loss或想象奖励。输出JSON和CSV保留候选范围、重复噪声与逐歌曲审计。
"""

import argparse
import csv
import json
from pathlib import Path
import statistics


METRICS = ('full_horizon_tracking', 'root_position_rmse_m', 'root_orientation_rmse_rad',
           'joint_q_rmse_rad', 'joint_qd_rmse_rad_s', 'body_orientation_rmse_rad',
           'jerk_rms', 'contact_foot_slip_mps')
TERMS = ('body_pos', 'body_ori', 'body_linvel', 'body_angvel', 'root_pos', 'root_ori')


def aggregate_rows(rows):
    overall = dict(songs=len(rows), baseline_completion=statistics.mean(r['baseline_completion'] for r in rows),
                   adapter_completion=statistics.mean(r['adapter_completion'] for r in rows), metrics={})
    for key in sorted({key for row in rows for key in row['metrics']}):
        values = [r['metrics'][key] for r in rows if key in r['metrics']]
        a, b = (statistics.mean(v[k] for v in values) for k in ('baseline', 'adapter'))
        after = sorted(v['adapter'] for v in values)
        overall['metrics'][key] = dict(baseline=a, adapter=b, delta=b-a, change_percent=100*(b/a-1) if a else None,
            valid_songs=len(values), adapter_median=statistics.median(after),
            adapter_p90=after[min(len(after)-1, int(.9*len(after)))],
            baseline_mean_repeat_range=statistics.mean(v['baseline_repeat_range'] for v in values),
            candidate_mean_repeat_range=statistics.mean(v['candidate_repeat_range'] for v in values),
            improved_songs=sum(v['delta'] > 0 if key.startswith(('tracking_', 'full_horizon')) or key=='beat_f1'
                               else v['delta'] < 0 for v in values))
    return overall


def compare_reports(baselines, candidates, expected_songs=64, allowed_splits=('train',)):
    reports = [*baselines, *candidates]
    if any(not r['full_song_evaluation'] for r in reports):
        raise ValueError('仅完整音乐预算可进入训练集比较')
    if any(r['sm_mdp_contract'] != reports[0]['sm_mdp_contract'] for r in reports):
        raise ValueError('奖励和物理终止标准不一致')
    indexes = [{e['task']: e for e in r['episodes']} for r in reports]
    ids = set(indexes[0])
    if len(ids) != expected_songs or any(set(i) != ids for i in indexes):
        raise ValueError(f'每份报告必须覆盖相同{expected_songs}首歌曲')
    if any(len(r['episodes']) != expected_songs or len({e['song'] for e in r['episodes']}) != expected_songs
           for r in reports):
        raise ValueError('不能以重复歌曲充当独立样本')
    rows = []
    for task in sorted(ids):
        episodes = [i[task] for i in indexes]
        for e in episodes:
            if e['split'] not in allowed_splits:
                raise ValueError('实际歌曲划分不在声明的评价范围内')
            for k in ('song', 'split', 'initial_state_sha256', 'physical_xi_sha256', 'evaluation_budget_steps', 'planned_steps'):
                if e[k] != episodes[0][k]:
                    raise ValueError(f'{task}的配对身份不同：{k}')
        metrics = {}
        for key in (*METRICS, 'beat_f1', *('tracking_'+t for t in TERMS)):
            def value(e):
                if key == 'beat_f1':
                    return e['music'].get('f1', 0.)
                if key.startswith('tracking_'):
                    return e['sm_terms'].get(key[9:])
                return e.get(key)
            before, after = [value(e) for e in episodes[:len(baselines)]], [value(e) for e in episodes[len(baselines):]]
            if any(v is None for v in before+after):
                continue
            a, b = statistics.mean(before), statistics.mean(after)
            metrics[key] = dict(baseline=a, adapter=b, delta=b-a,
                change_percent=100*(b/a-1) if a else None,
                baseline_repeat_range=max(before)-min(before), candidate_repeat_range=max(after)-min(after))
        rows.append(dict(task=task, song=episodes[0]['song'], split=episodes[0]['split'],
            source=episodes[0].get('source', 'unknown'),
            adapter_training_seen=episodes[0].get('adapter_training_seen', episodes[0]['split']=='train'),
            planned_seconds=episodes[0]['planned_steps']*.02,
            baseline_completion=statistics.mean(float(e['full_music_complete']) for e in episodes[:len(baselines)]),
            adapter_completion=statistics.mean(float(e['full_music_complete']) for e in episodes[len(baselines):]),
            baseline_reasons=[e['reason'] for e in episodes[:len(baselines)]],
            adapter_reasons=[e['reason'] for e in episodes[len(baselines):]],
            executed_seconds=[[e['steps']*.02 for e in episodes[:len(baselines)]],
                              [e['steps']*.02 for e in episodes[len(baselines):]]], metrics=metrics))
    groups = {key: {str(value): aggregate_rows([row for row in rows if row[key]==value])
                   for value in sorted({row[key] for row in rows}, key=str)}
              for key in ('split', 'source', 'adapter_training_seen')}
    return dict(overall=aggregate_rows(rows), groups=groups, songs=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--confirmation', type=Path)
    args = p.parse_args()
    manifest = json.loads((args.root/'candidates.json').read_text())
    options = dict(expected_songs=manifest.get('expected_song_count', 64),
                   allowed_splits=manifest.get('evaluation_splits', ['train']))
    comparisons = []
    for candidate in manifest['candidates']:
        paths = list(args.root.glob('worker*/'+candidate['label']+'.json'))
        if len(paths) != 1:
            raise ValueError(f'候选未完成或重复：{candidate["label"]}')
        directory = paths[0].parent
        baseline = [json.loads((directory/(n+'.json')).read_text()) for n in ('zero', 'zero_repeat')]
        reports = [json.loads(paths[0].read_text())]
        result = compare_reports(baseline, reports, **options)
        result.update(candidate=candidate, report=str(paths[0]))
        comparisons.append(result)
    def ranking(result):
        x = result['overall']; m = x['metrics']
        return (x['adapter_completion'], -m['root_position_rmse_m']['adapter'],
                m['full_horizon_tracking']['adapter'], -m['jerk_rms']['adapter'])
    comparisons.sort(key=ranking, reverse=True)
    best = comparisons[0]
    confirmed = None
    selected = []
    if args.confirmation:
        directory = args.confirmation
        label = best['candidate']['label']
        baseline = [json.loads((directory/(n+'.json')).read_text()) for n in ('zero', 'zero_repeat')]
        reports = [json.loads((directory/(n+'.json')).read_text()) for n in (label, label+'_repeat1')]
        confirmed = compare_reports(baseline, reports, **options)
        confirmed['candidate'] = best['candidate']
        eligible = [r for r in confirmed['songs'] if r['baseline_completion']==1 and r['adapter_completion']==1
                    and r['metrics']['full_horizon_tracking']['delta']>0
                    and r['metrics']['root_position_rmse_m']['delta']<0]
        eligible.sort(key=lambda r: (r['metrics']['full_horizon_tracking']['delta'],
                                    -r['metrics']['root_position_rmse_m']['delta']), reverse=True)
        seen = set()
        for row in eligible:
            paths = list((directory/'zero_trajectories').glob('*_'+row['task']+'.json'))
            if len(paths) != 1:
                raise ValueError('缺少同次真实导出轨迹')
            metadata = json.loads(paths[0].read_text())
            audio = metadata['task']['audio_sha256']
            if audio in seen:
                continue
            seen.add(audio)
            selected.append(dict(**row, baseline_trajectory=str(paths[0]),
                adapter_trajectory=str(directory/(label+'_trajectories')/paths[0].name),
                source=metadata['task']['source'], audio_sha256=audio))
            if len(selected)==5:
                break
    args.output.mkdir(parents=True, exist_ok=False)
    summary = dict(scope=manifest.get('scope', 'training_set_in_sample_model_selection_not_unseen_test'),
        expected_song_count=options['expected_songs'], evaluation_splits=options['allowed_splits'],
        selection_rule=manifest['selection_rule'], candidate_count=len(comparisons),
        best_candidate=best['candidate'], comparisons=comparisons, confirmation=confirmed,
        selected_examples=selected, selection_boundary='five_positive_examples_not_64_song_average')
    (args.output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    (args.output/'selected.json').write_text(json.dumps(selected, ensure_ascii=False, indent=2))
    with (args.output/'candidate_comparison.csv').open('w', newline='') as f:
        writer = csv.writer(f); writer.writerow(['candidate','completion',*METRICS])
        for result in comparisons:
            writer.writerow([result['candidate']['label'],result['overall']['adapter_completion'],
                             *(result['overall']['metrics'][k]['adapter'] for k in METRICS)])
    print(json.dumps(dict(best=best['candidate'], overall=(confirmed or best)['overall'],
                         selected=[r['task'] for r in selected]), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
