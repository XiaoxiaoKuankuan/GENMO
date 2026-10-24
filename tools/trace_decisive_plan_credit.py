"""将完整物理评估中的后续跟踪奖励追到最初生成的自由参考帧。

输入是SHA绑定的完整评估报告和诊断B导出的逐计划时间线。逐个50Hz控制步按
奖励实际读取的中心参考及线性插值质量分配统计，复制的已知前缀递归追到父计划；
输出每条计划的生成/提交/前瞻/中心时刻、后续被使用的秒数、奖励和八项RMS。
这些统计说明奖励目标来源及延迟，不声称反事实因果奖励归属；GMT前瞻已能提前
改变控制，不能据此移动GAE奖励。没有模型更新、物理重放或实际状态监督目标。
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from functools import lru_cache
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gem.closedloop.dppo.compact_evidence import report_episodes
from gem.closedloop.dppo.execution_provenance import source_frame_support
from gem.closedloop.dppo.budget import atomic_json


def origin_resolver(plans):
    """离散30Hz前缀只按确切时间映射；禁止把不齐帧映射向下取整。"""
    @lru_cache(maxsize=None)
    def origin(name,frame):
        if name.startswith('bootstrap:'):return 'bootstrap'
        record=plans[name]
        if frame>=record['prefix_frames']:return name
        parent=record['parent_plan_id']
        if parent.startswith('bootstrap:'):return 'bootstrap'
        index,remainder=divmod(record['decision_tick']+20*frame-plans[parent]['decision_tick'],20)
        if remainder:raise ValueError('Unaligned inherited reference frame')
        return origin(parent,index)
    return origin


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--credit',type=Path,required=True)
    args=parser.parse_args();bytask=defaultdict(dict)
    for line in (args.credit/'plans.jsonl').read_text().splitlines():
        row=json.loads(line);bytask[row['task_id']][row['plan_id']]=row
    report=json.loads((args.run/'evaluation_baseline.json').read_text());rows=[];total=assigned=bootstrap=0.;same=later=0.
    for episode in report_episodes(report):
        plans=bytask[episode['task_id']];origin=origin_resolver(plans);accum={}
        for decision in episode['decisions']:
            steps={step['control_tick_end']:step for step in decision['compact_execution_provenance']['steps']}
            for control in decision['controls']:
                step=steps[control['tick']];name=step['reference_plan_id'];total+=control['reward']
                if name.startswith('bootstrap:'):bootstrap+=control['reward'];continue
                for frame,weight in source_frame_support(step['reference_tick'],plans[name]['decision_tick']):
                    source=origin(name,frame)
                    if source=='bootstrap':bootstrap+=weight*control['reward'];continue
                    item=accum.setdefault(source,dict(seconds=0.,reward=0.,same_transition_seconds=0.,later_transition_seconds=0.,squared_error_integrals=defaultdict(float),first_reward_tick=control['tick'],last_reward_tick=control['tick']))
                    dt=weight*control['dt_s'];item['seconds']+=dt;item['reward']+=weight*control['reward'];assigned+=weight*control['reward'];item['last_reward_tick']=control['tick']
                    field='same_transition_seconds' if source==decision['identity']['plan_id'] else 'later_transition_seconds';item[field]+=dt
                    if field=='same_transition_seconds':same+=dt
                    else:later+=dt
                    for key,value in control['raw_errors'].items():
                        if key.endswith('_squared_error'):item['squared_error_integrals'][key]+=dt*value
        for name,plan in plans.items():
            item=accum.get(name);row=dict(plan)
            row['later_tracking']=None if item is None else dict(item,rms={key:math.sqrt(value/item['seconds']) for key,value in item['squared_error_integrals'].items()},reward_per_second=item['reward']/item['seconds'],first_reward_delay_seconds=(item['first_reward_tick']-plan['decision_tick'])/600)
            rows.append(row)
    if not math.isclose(total,assigned+bootstrap,rel_tol=1e-10,abs_tol=1e-8):raise ValueError('Reference reward mass not conserved')
    with (args.credit/'plan_future_tracking.jsonl').open('x') as handle:
        for row in rows:handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    summary=dict(status='passed',plans=len(rows),plans_with_center_reward=sum(r['later_tracking'] is not None for r in rows),
        total_control_reward=total,generated_reference_reward=assigned,bootstrap_reference_reward=bootstrap,
        same_transition_center_seconds=same,later_transition_center_seconds=later,
        scope='reference_target_attribution_only_not_causal_reward_decomposition; zero_step_events_not_assigned_to_reference')
    atomic_json(args.credit/'plan_future_tracking_summary.json',summary);print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
