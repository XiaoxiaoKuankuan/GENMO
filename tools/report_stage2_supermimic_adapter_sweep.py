"""只读汇总八个独立GPU实验的进程、accepted计数和相同预算真实评价。

仅读取各组自己的status/evaluation JSON，不连接GPU、不导入Torch、不修改训练状态。
每60秒生成当前汇总与共同已完成评价预算下的排名；完整歌曲完成率优先，其次原始
GENMO参考Tracking，音乐F1为次要指标。WM监督/预热绝不计为有效Adapter更新。
"""

import argparse
import json
import os
from pathlib import Path
import time


def collect(root):
    groups = []
    for folder in sorted(root.glob('gpu[0-7]')):
        status_path = folder/'status.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else dict(phase='not_started', accepted_updates=0)
        manifest_path = folder/'manifest.json'
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        command = manifest.get('command', [])
        initial_accepted = 0
        if '--resume' in command:
            # 本项目完整文件名含接受计数，恢复链父目录的latest可核实同一路径。
            checkpoint = Path(command[command.index('--resume')+1])
            import re
            match = re.search(r'_(\d{6})_\d+\.pt$', checkpoint.name)
            if match:
                initial_accepted = int(match.group(1))
        process_start = manifest.get('status', {}).get('started_unix', status.get('started_unix', time.time()))
        new_updates = status['accepted_updates']-initial_accepted
        status['process_initial_accepted'] = initial_accepted
        status['process_accepted_updates'] = new_updates
        status['accepted_updates_per_hour'] = new_updates*3600/max(1, time.time()-process_start)
        status['lifetime_updates_per_hour'] = status['accepted_updates']*3600/max(1, time.time()-status.get('started_unix', time.time()))
        pid = status.get('pid')
        alive = False
        if pid:
            try:
                os.kill(pid, 0); alive = True
            except ProcessLookupError:
                pass
        # 源码维护后的完整恢复延续同一个实验；沿各组自己的恢复链读取已有真实评价。
        # 不混入另一组结果，也不要求恢复后重复已经完成的相同预算评价。
        ancestry, visited, cursor = [], set(), folder
        while cursor not in visited and cursor.is_dir():
            visited.add(cursor); ancestry.append(cursor)
            manifest_path = cursor/'manifest.json'
            command = json.loads(manifest_path.read_text()).get('command', []) if manifest_path.exists() else []
            if '--resume' not in command:
                break
            cursor = Path(command[command.index('--resume')+1]).parent
        evaluations = {}
        for ancestor in reversed(ancestry):
            for path in ancestor.glob('evaluation_*.json'):
                if path.stem == 'evaluation_zero':
                    continue
                value = json.loads(path.read_text())
                value['evaluation_source'] = str(path)
                evaluations[int(value['accepted_updates'])] = value
        groups.append(dict(group=folder.name, directory=str(folder), pid_alive=alive, status=status,
                           evaluations=evaluations))
    common = set.intersection(*(set(g['evaluations']) for g in groups)) if len(groups) == 8 else set()
    budget = max(common) if common else None
    ranking = []
    if budget is not None:
        for group in groups:
            value = group['evaluations'][budget]
            ranking.append(dict(group=group['group'], accepted_updates=budget, completion_rate=value['completion_rate'],
                mean_tracking=value['mean_tracking'], mean_beat_f1=value['mean_beat_f1'],
                tracking_change_from_zero=value['tracking_change_from_zero']))
        ranking.sort(key=lambda v:(v['completion_rate'], v['mean_tracking'], v['mean_beat_f1']), reverse=True)
    for g in groups:
        latest = max(g['evaluations']) if g['evaluations'] else None
        g['latest_evaluation'] = g['evaluations'].get(latest)
        g.pop('evaluations')
    return dict(unix=time.time(), comparison_budget=budget, ranking=ranking, groups=groups,
        note='未达到共同真实评价预算的组不参与同预算排名；监督预热不计accepted')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True); p.add_argument('--watch', action='store_true')
    args=p.parse_args(); root=Path(args.root).resolve()
    while True:
        result=collect(root)
        temp=root/'comparison_latest.tmp'
        temp.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        os.replace(temp, root/'comparison_latest.json')
        with (root/'comparison_history.jsonl').open('a') as f:
            f.write(json.dumps(result, ensure_ascii=False, allow_nan=False)+'\n')
        print(json.dumps(dict(unix=result['unix'], comparison_budget=result['comparison_budget'],
            groups=[dict(group=g['group'], pid=g['status'].get('pid'), alive=g['pid_alive'],
                         phase=g['status']['phase'], accepted=g['status']['accepted_updates']) for g in result['groups']]),
            ensure_ascii=False), flush=True)
        if not args.watch:
            return
        time.sleep(60)


if __name__ == '__main__':
    main()
