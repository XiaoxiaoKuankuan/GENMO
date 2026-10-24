"""复算八个方向在参考生效前观测完全一致的共同任务子集。

输入为 summarize_decisive_physics.py 产出的八份带 SHA 核验来源的 JSON；只依据
新参考生效前十二个控制步的奖励和八项误差筛选，不按候选收益筛选。输出共同任务、
奖励/活动度/跟踪误差/实际参考变化及正负方向差的音乐聚类区间。该筛选只证明记录的
早期观测一致，不证明全部 PhysX 隐状态一致；小音乐队列、部分来源只有一首音乐，
bootstrap 区间不能代替独立重复实验。只读分析，不载入 Actor、不启动仿真或训练。
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.summarize_decisive_physics import cluster_summary, ERRORS


def summarize(folder):
    data = [json.loads((folder / f'A_rank{i:02d}.json').read_text()) for i in range(8)]
    valid = [{r['task_id'] for r in d['tasks']
              if r['pre_influence_reward_max_difference'] == 0
              and r['pre_influence_tracking_max_difference'] == 0} for d in data]
    ids = set.intersection(*valid)
    if not ids:
        raise ValueError('没有八方向共同通过参考生效前观测检查的任务')
    out = dict(scope='all_eight_directions_exact_pre_influence_intersection_no_post_reward_filter',
               limitation='early_observations_not_all_PhysX_hidden_state; exploratory_small_cluster_intervals',
               task_ids=sorted(ids), passing_counts=[len(v) for v in valid], candidates=[], directions=[], contrasts={})
    for i, d in enumerate(data):
        rows = [r for r in d['tasks'] if r['task_id'] in ids]
        for row in rows:
            for key in ERRORS:
                row[key + '_delta'] = row['final']['tracking'][key] - row['initial']['tracking'][key]
        refs = [r for r in d['reference_tasks'] if r['task_id'] in ids]
        out['candidates'].append(dict(rank=i, label=('A+', 'A-', 'B+', 'B-', 'C+', 'C-', 'D+', 'D-')[i],
            rate=cluster_summary(rows, 'reward_per_second_delta'),
            metrics={k: cluster_summary(rows, k + '_delta') for k in
                     ('full_task_reward', 'failure', 'activity', 'music_alignment', 'completion')},
            tracking={k: cluster_summary(rows, k + '_delta') for k in ERRORS},
            executed_reference={k: cluster_summary(refs, k) for k in
                                ('root_position_rms_m', 'root_angle_rms_rad', 'leg_rms_rad')},
            tasks=rows))
    derivatives = []
    for i, label in enumerate('ABCD'):
        positive = {r['task_id']: r for r in data[2*i]['tasks']}
        negative = {r['task_id']: r for r in data[2*i+1]['tasks']}
        rows = [{**positive[t], 'gain': (positive[t]['reward_per_second_delta'] -
                                       negative[t]['reward_per_second_delta']) / 2} for t in sorted(ids)]
        derivatives.append(rows)
        out['directions'].append(dict(direction=label, **cluster_summary(rows, 'gain')))
    for left, right in ((0, 1), (0, 2), (3, 0), (3, 1), (3, 2)):
        rows = [{**a, 'gain': a['gain'] - b['gain']} for a, b in zip(derivatives[left], derivatives[right])]
        out['contrasts']['ABCD'[left] + '_minus_' + 'ABCD'[right]] = cluster_summary(rows, 'gain')
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    result = summarize(args.folder)
    path = args.folder / 'strict_initial_state_subset.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(tasks=len(result['task_ids']), passing_counts=result['passing_counts'],
                          directions=result['directions'], contrasts=result['contrasts']), ensure_ascii=False))
