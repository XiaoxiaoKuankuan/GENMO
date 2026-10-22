#!/usr/bin/env python3
"""只读核验固定2Hz与连续滚动Stage1的有限PhysX配对结果。

输入为两次原评估器产生的完整periodic合并报告。先核对原始episode侧车、SHA、
奖励聚合和固定音乐/种子，要求相同Actor、Gaussian核、奖励和物理参数；仅允许
调度/参考更新不同。再从逐控制来源递归追踪继承前缀，区分窗口读取、中心消费、
新计划替换和未完成尾段。每份计划报告真实源帧、权重及延迟，不预设执行15帧。

本工具不训练、不重算奖励、不删除失败任务。参考频率提高与跟踪改善分开报告；
未实际进入中心的计划不虚填延迟，尾部未消费不能当作永久无效。输出独立JSON，
不能写回或覆盖输入。部署时间来自同一实测profile，训练墙钟不是机器人周期。
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.eval.compare_closedloop_stage10 import _load_periodic, _episode_metrics
from gem.closedloop.dppo.execution_provenance import consumption_report
from gem.closedloop.dppo.evaluation import _stats


def schedule_metrics(episodes):
    records=[];intervals=[];ready_read=[];state_center=[];prefixes=[];frames=[];durations=[]
    shared_position=[];shared_velocity=[];splices={};unresolved=0.
    for episode in episodes:
        decisions=episode['decisions']
        evidence=[d.get('compact_execution_provenance') for d in decisions]
        result=consumption_report(evidence)
        if result['status']=='unverified':raise ValueError('Physical comparison lacks retained control provenance')
        unresolved+=sum(result['unresolved_source_support_weight'].values())
        starts=[]
        for d in decisions:
            commit=d.get('reference_commit')
            if commit:
                shared_position.append(commit['max_position_error'])
                shared_velocity.append(commit['max_velocity_error'])
                for key,value in (commit.get('splice_continuity') or {}).get('metrics',{}).items():
                    splices.setdefault(key,[]).append(value)
            plan_id=d['identity']['plan_id']
            p=result['plans'][plan_id]
            starts.append(p['decision_tick'])
        intervals.extend((b-a)/600. for a,b in zip(starts,starts[1:]))
        replacement={p['parent_plan_id']:dict(tick=p['commit_tick'],by=key)
            for key,p in result['plans'].items() if p['commit_tick'] is not None}
        for key,p in result['plans'].items():
            first=p['first_generated_current_tick'];window=p['first_generated_window_tick']
            ready=p.get('generation_ready_tick')
            direct=p['observed_current_generated_frames']
            seconds=p['generated_current_control_weight']*.02
            if ready is not None and window is not None:ready_read.append((window-ready)/600.)
            if first is not None:state_center.append((first-p['decision_tick'])/600.)
            prefixes.append(p['prefix_frames']);frames.append(len(direct));durations.append(seconds)
            records.append(dict(task_id=episode['task_id'],dataset=episode['dataset'],seed=episode['seed'],
                plan_id=key,decision_tick=p['decision_tick'],commit_tick=p['commit_tick'],
                generation_ready_tick=ready,prefix_frames=p['prefix_frames'],generation_frames=p['generation_frames'],
                first_window_tick=p['first_window_tick'],first_center_tick=p['first_current_tick'],
                first_generated_window_tick=window,first_generated_center_tick=first,
                last_generated_center_tick=p['last_generated_current_tick'],
                replacement=replacement.get(key),actual_generated_source_frames=direct,
                current_source_frame_weights=p['current_generated_frame_weight'],
                actual_source_weighted_seconds=seconds,
                source_support_in_window=p['window_generated_frame_weight'],
                right_censored_at_episode_end=True))
    mean=sum(intervals)/len(intervals) if intervals else None
    return dict(decision_intervals_seconds=_stats(intervals),effective_generation_hz=None if not mean else 1/mean,
        ready_to_first_generated_future_read_seconds=_stats(ready_read),
        state_to_first_generated_center_seconds=_stats(state_center),prefix_frames=_stats(prefixes),
        executed_generated_source_frames_per_plan=_stats(frames),
        source_weighted_execution_seconds_per_plan=_stats(durations),
        shared_reference_max_position_error=_stats(shared_position),shared_reference_max_velocity_error=_stats(shared_velocity),
        splice_metrics={k:_stats(v) for k,v in splices.items()},unresolved_source_support_weight=unresolved,
        plans=records,physical_time_semantics='measured single-request discrete-event delay; all 50Hz controls and 200Hz substeps executed',
        consumption_semantics='recursive source-frame lineage, linear position/joints and SLERP supports; not a causal influence score')


def compare(fixed,rolling):
    a,plan,ae,ar=_load_periodic(fixed)
    b,other,be,br=_load_periodic(rolling)
    if plan!=other:raise ValueError('Task/music/seed/initial-state protocol changed')
    if a['actor_identity']['model_fingerprint']!=b['actor_identity']['model_fingerprint']:
        raise ValueError('Stage1 models differ')
    ai=a['evaluation_identity']['training_identity'];bi=b['evaluation_identity']['training_identity']
    for key in ('assets','kernel','reward','physical_configuration','generation_horizon'):
        if ai[key]!=bi[key]:raise ValueError('Non-scheduling change: '+key)
    if ai['timing']['deployment_profile']!=bi['timing']['deployment_profile']:
        raise ValueError('Different measured latency distributions')
    pairs=[]
    for x,y in zip(ae,be):
        xm,ym=_episode_metrics(x),_episode_metrics(y)
        pairs.append(dict(task_id=x['task_id'],fixed=xm,rolling=ym,delta={
            key:ym[key]-xm[key] for key in xm if xm[key] is not None and ym.get(key) is not None}))
    return dict(schema='genmo.continuous_rolling.physical_comparison.v1',status='completed',
        optimized=False,source_reports=dict(fixed=ar,rolling=br),task_count=len(pairs),
        plan=plan,actor_fingerprint=a['actor_identity']['model_fingerprint'],
        fixed=dict(physical=a['aggregate'],scheduling=schedule_metrics(ae)),
        rolling=dict(physical=b['aggregate'],scheduling=schedule_metrics(be)),pairs=pairs,
        interpretation='有限冻结Stage1调度对照；频率增加不等于跟踪改善，不证明DPPO长期收益')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixed',type=Path,required=True)
    parser.add_argument('--rolling',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=compare(args.fixed,args.rolling)
    with args.output.open('x') as f:json.dump(result,f,ensure_ascii=False,allow_nan=False,indent=2)
    print(json.dumps({k:result[k] for k in ('status','task_count','optimized')},ensure_ascii=False))


if __name__=='__main__':main()
