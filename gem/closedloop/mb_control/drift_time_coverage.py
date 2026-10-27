"""真实训练请求的音乐时间覆盖补充审计。

只读原正式任务列表、音乐特征长度和逐次训练日志，按真实请求身份去重，统计真正
完成更新的请求在歌曲四分位中的分布及重复音乐时刻。不同episode、collector的同
音乐时刻仍是不同物理样本，另外记录去掉这些身份后的时间覆盖，不能把两者混称。
不会加载GPU模型、改变采样器、放宽世界模型支持域或写回正式运行目录。
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from .application import read_music_features
from .drift_diagnostics import write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run=Path(args.run)
    cfg=json.loads((run/'manifest.json').read_text())['config']
    tasks=json.loads(Path(cfg['tasks']).read_text())
    length={t['id']:len(read_music_features(t['music_features']))*20 for t in tasks}
    unique={}
    for line in (run/'training.jsonl').open():
        record=json.loads(line)
        if record.get('accepted_actor_updates',0)>700:break
        for sample in record.get('sample_distribution',[]):
            if sample.get('accepted'):
                identity=tuple(sample.get(k) for k in ['collector','episode','task','request_tick'])
                unique[identity]=sample
    rows=list(unique.values())
    def summarize(part):
        return {'physical_requests':len(part),
            'distinct_music_time_pairs':len({(r['task'],r['request_tick']) for r in part}),
            'music_quartiles':dict(Counter(min(3,int(4*r['request_tick']/length[r['task']])) for r in part)),
            'first_2_seconds':sum(r['request_tick']<1200 for r in part)}
    write_json(Path(args.output),{'all':summarize(rows),
        'by_task':{t['id']:{**summarize([r for r in rows if r['task']==t['id']]),'music_seconds':length[t['id']]/600} for t in tasks},
        'limitation':'same music/time in another physical episode is not the same physical state; one seed only'})


if __name__=='__main__':main()
