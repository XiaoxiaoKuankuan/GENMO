"""只读重审完整P0原始证据，纠正把训练墙钟误当部署时钟的审核错误。

只接受modeled_deployment.v3：仿真到达由固定profile决定，GPU计算耗时是性能记录。
重新核验全部任务的初始物理/GMT/缓存、逐50Hz奖励八项误差和部署到达/提交/消费
完整时序，同时逐元素核验每个当前与bootstrap状态张量、每条世界参考和接触/种子。
不重写原失败报告，不删除任何失败样本，不重新采样；保存带原报告SHA的新独立审核。
任何真实部署时钟、状态、生成参考或物理误差差异均使重审失败，禁止放行实验C。
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gem.closedloop.dppo.decisive_diagnostic import compare_zero_reports
from gem.closedloop.dppo.budget import atomic_json


def audit(row):
    torch.set_num_threads(2)
    result=compare_zero_reports(row['before'],row['after'])
    a=torch.load(Path(row['before']).parent.parent/'causal_references.pt',map_location='cpu',weights_only=False)
    b=torch.load(Path(row['after']).parent.parent/'causal_references.pt',map_location='cpu',weights_only=False)
    if len(a['conditions'])!=len(b['conditions']) or len(a['references'])!=len(b['references']):raise ValueError('Different complete condition/reference counts')
    contexts=references=0
    for x,y in zip(a['conditions'],b['conditions']):
        if (x.task_id,x.control_tick_begin,x.control_tick_end)!=(y.task_id,y.control_tick_begin,y.control_tick_end):
            raise ValueError('Condition task or physical interval differs')
        for key in ('critic_context','next_critic_context'):
            left,right=x.metadata.get(key),y.metadata.get(key)
            if (left is None)!=(right is None):raise ValueError('Bootstrap validity differs')
            if left is None:continue
            if set(left)!=set(right) or any(not torch.equal(left[k],right[k]) for k in left):
                raise ValueError('Causal state differs in '+key)
            contexts+=1
    for x,y in zip(a['references'],b['references']):
        if x['task_id']!=y['task_id']:raise ValueError('Reference task differs')
        for k in ('qpos_world','qpos30','contact','source_ticks','seed','decision_tick','prefix_frames'):
            if not np.array_equal(x['generated'][k],y['generated'][k]):raise ValueError('Generated reference/noise differs: '+k)
        references+=1
    result.update(rank=row['rank'],before=row['before'],after=row['after'],exact_causal_contexts=contexts,
        exact_generated_references=references,exact_generated_noise=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    config=yaml.safe_load(next(a.run.glob('sessions/*/resolved_config.yaml')).read_text())
    if config['runtime']['timing_contract']!='modeled_deployment.v3':raise ValueError('Wall time may drive this deployment')
    original=a.run/'p0/summary.json';target=a.run/'p0/revalidated_summary.json'
    if target.exists():raise FileExistsError(target)
    prior=json.loads(original.read_text())
    with ProcessPoolExecutor(max_workers=a.workers) as pool:rows=list(pool.map(audit,prior['results']))
    result=dict(status='passed' if all(r['passed'] for r in rows) else 'failed',results=rows,
        world_reuse=False,tasks_per_slot=1,training_updates=0,
        original_gate_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),original_gate_status=prior['status'],
        corrected_check='actual arrival/commit/consumption ticks; GPU generation_wall_seconds excluded from deployment clock',
        physical_samples_removed=0,physics_rerun=False)
    atomic_json(target,result)
    print(json.dumps(dict(status=result['status'],ranks=len(rows),tasks=sum(len(r['tasks']) for r in rows),
        exact_causal_contexts=sum(r['exact_causal_contexts'] for r in rows),exact_generated_references=sum(r['exact_generated_references'] for r in rows))))
    if result['status']!='passed':raise ValueError('Complete P0 remains failed')


if __name__=='__main__':main()
