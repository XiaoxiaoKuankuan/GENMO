"""真实整曲参考版本的生命周期覆盖审计。

只读取完整真实控制轨迹中当时缓存的源时刻和版本，不使用最终缓存倒推过去。
逐50Hz控制输入统计实际姿态中心插值端点，区分速度halo和未来goal；跟踪参考点
被下一份计划覆盖的时刻，分别统计曾作为当前姿态消费、仅预览、消费前已被覆盖、
歌曲结束尚未来得及执行的源点。输出是实际调度覆盖，不代表模型梯度或物理收益。
"""
import argparse
from collections import Counter,defaultdict
import gc
import json
from pathlib import Path
import torch
from .drift_diagnostics import stats,write_json


def scan(source,version):
    data=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
    controls=data['controls']['data'];generations=data['generations']['data']
    seen=defaultdict(set);center=defaultdict(set);goal=defaultdict(set);lost=defaultdict(set)
    previous={};end_tick=controls[-1]['data']['end']['data']['tick']
    for control in controls:
        state=control['data']['start']['data'];cache=state['reference'];t=state['tick']
        tick_list=cache['ticks'].tolist();owners=cache['source_versions'].tolist();current=dict(zip(tick_list,owners))
        for point,owner in current.items():
            if owner:seen[owner].add(point)
            old=previous.get(point)
            if old and old!=owner:lost[old].add(point)
        def endpoints(t):
            lo=(t-tick_list[0])//20;out=[lo]
            if (t-tick_list[0])%20:out.append(lo+1)
            return [i for i in out if 0<=i<len(tick_list)]
        for i in endpoints(t):
            if owners[i]:center[owners[i]].add(tick_list[i])
        for future in range(t,t+121,12):
            for query in (future,max(tick_list[0],future-12),future+12):
                for i in endpoints(query):
                    if owners[i]:goal[owners[i]].add(tick_list[i])
        previous=current
    task=controls[0]['data']['start']['data']['task_id'];rows=[]
    for generation in generations:
        g=generation['data'];v=g['reference_version']
        if not g['accepted'] or v not in seen:continue
        ticket=g['plan']['data']['ticket']['data'];owned=seen[v]
        eligible={t for t in owned if t<=end_tick}
        rows.append({'version':version,'task':task,'reference_version':v,'request_tick':ticket['request_tick'],
            'actually_owned_source_points':len(owned),'within_music_source_points':len(eligible),
            'ever_center_consumed_source_points':len(center[v]),'goal_but_never_center_source_points':len(goal[v]-center[v]),
            'overwritten_before_center_source_points':len((lost[v]-center[v])&eligible),
            'after_song_end_source_points':len(owned-eligible),
            'center_consumed_fraction':len(center[v])/len(owned),
            'within_music_overwritten_before_center_fraction':len((lost[v]-center[v])&eligible)/max(1,len(eligible))})
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run=Path(args.run);output=Path(args.output);output.mkdir(exist_ok=False)
    rows=[];torch.set_num_threads(2)
    for version in [0,100,300,600,700]:
        directory=run/'evaluations/fixed_delay_replay'/('initial_genmo_baseline' if version==0 else f'actor_{version:06}')
        for source in directory.glob('shard_*/*_real_controls.pt'):
            rows.extend(scan(source,version));gc.collect()
        write_json(output/'summary.json',{str(v):{k:stats([r[k] for r in rows if r['version']==v]) for k in
            ['actually_owned_source_points','ever_center_consumed_source_points','goal_but_never_center_source_points',
             'overwritten_before_center_source_points','center_consumed_fraction','within_music_overwritten_before_center_fraction']}
            for v in sorted({r['version'] for r in rows})})
    with (output/'reference_lifetimes.jsonl').open('w') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')


if __name__=='__main__':main()
