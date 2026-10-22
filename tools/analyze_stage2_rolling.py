"""审核连续滚动真实采集的时钟、参考发布和变长奖励账本。

输入是当前分支collect-only生成的不可变fixed_data_validation目录（按窗口顺序），
逐rank校验文件SHA、完整随机链、Actor版本和参考合同，跨窗口追踪同一物理episode
的连续20ms奖励。只使用实际保存的commit、21点GMT窗口和重置前Critic条件；不
移动奖励、不拟造未消费自由帧、不把前瞻观察当作当前执行。输出紧凑JSON审计报告，
不更新模型、不启动PhysX；必须配合真实采集日志判断运行成功及性能。
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gem.closedloop.dppo.diagnostic_rollout import load_rollout
from gem.closedloop.dppo.rolling import SCHEMA, REFERENCE_CONTRACT


def audit(cases):
    begun=time.perf_counter()
    durations=Counter();prefixes=Counter();failures=Counter();rejections=Counter();versions=set()
    last={};last_content={};kernel=None;windows=[];timeouts=[];commits=cancelled=controls=overwrites=0
    future_first=[];center_first=[];all_plans={};observed_future={};observed_center={}
    boundary_target_errors=[];gammas=set()
    for case in cases:
        info=json.loads((case/'case.json').read_text());window=Counter()
        for rank in range(8):
            payload=load_rollout(case/f'rank{rank:02d}')
            schedule=payload['identity']['runtime'].get('generation_schedule',{})
            assert schedule.get('schema')==SCHEMA and schedule.get('reference_contract')==REFERENCE_CONTRACT
            if kernel is None:kernel=payload['kernel']
            assert payload['kernel']==kernel
            gamma=float(payload['identity']['training']['gamma_upper']);gammas.add(gamma)
            gamma_control=gamma**(1/25)
            rows=payload['transitions'];window['chains']+=len(rows)
            for row_index,row in enumerate(rows):
                key=tuple(row.identity[k] for k in ('backend_session_id','env_id','episode_id'))
                start,end=int(row.control_tick_begin),int(row.control_tick_end)
                n=int(row.executed_control_steps)
                assert start%60==0 and end-start==12*n and len(row.rewards)==n and row.executed_physics_steps==4*n
                assert n>0
                if key in last:assert last[key]==start,('overlap_or_gap',key,last[key],start)
                last[key]=end;versions.add(row.identity['policy_version']);durations[n]+=1
                window['controls']+=n;controls+=n
                meta=row.metadata;packet=meta['compact_execution_provenance']
                assert len(packet['steps'])==n
                own=str(row.identity['plan_id']);own_origin=packet['plans'][own]
                prefixes[own_origin['prefix_frames']]+=1
                for plan,origin in packet['plans'].items():
                    k=(*key,plan)
                    if k in all_plans:assert all_plans[k]==origin
                    all_plans[k]=origin
                commit=meta.get('published')
                if commit:
                    commits+=1;window['commits']+=1
                    tick=int(commit['effective_tick'])
                    assert tick%12==0 and start<=tick<=end
                    assert tick==meta['simulated_arrival_tick']
                    assert commit['protected_modification_count']==0
                    assert commit['reference_contract']==REFERENCE_CONTRACT
                    for replaced in commit['overwritten_reference_ranges']:
                        assert replaced['begin_tick']>tick
                        assert replaced['end_tick']>replaced['begin_tick']
                        overwrites+=1
                if meta.get('pending_cancelled_at_boundary'):cancelled+=1;window['cancelled']+=1
                if meta.get('rejection'):rejections[meta['rejection']['code']]+=1
                if row.terminated:failures[row.reason]+=1;window['terminal']+=1
                if meta.get('physical_time_limit_reached'):
                    assert not row.terminated and row.truncated and meta['time_limit_bootstrap']
                    assert end==12600 and row.next_context is not None
                    assert abs(float(row.next_context['decision_time'].reshape(-1)[0])-21.)<1e-8
                    timeouts.append(dict(rank=rank,episode=key[-1],tick=end,next_value=float(row.next_value)))
                    window['timeouts']+=1
                if row.transition_valid and (row.terminated or meta.get('physical_time_limit_reached')):
                    # 独立标量重算真实边界目标，防止“保存了末态”却实际漏用自举。
                    target=math.fsum(gamma_control**j*float(r) for j,r in enumerate(row.rewards))
                    target+=gamma_control**max(n-1,0)*float(meta.get('event_reward',0.))
                    if not row.terminated:target+=gamma_control**n*float(row.next_value)
                    error=abs(target-float(payload['targets']['returns'][row_index]))
                    assert error<1e-8,('boundary_bootstrap_target_differs',key,error)
                    boundary_target_errors.append(error)
                for index,step in enumerate(packet['steps']):
                    begin=int(step['control_tick_begin'])
                    assert begin==start+12*index and step['control_tick_end']==begin+12
                    assert step['consumed_reference_ticks']==list(range(begin-120,begin+121,12))
                    v=step['gmt_reference_version']
                    assert v==step['reward_reference_version']
                    assert v>=last_content.get(key,-1);last_content[key]=v
                    plans=[packet['plan_dictionary'][i] for i in step['consumed_plan_indices']]
                    for p in set(plans[11:]):observed_future.setdefault((*key,p),begin)
                    observed_center.setdefault((*key,plans[10]),begin)
            del rows,payload
        assert window['chains']==info['real_global_chains']
        windows.append(dict(case=str(case),**window))
    assert len(versions)==1,'mixed behavior policy versions'
    assert len(gammas)==1,'mixed physical-time discounts'
    for k,origin in all_plans.items():
        commit=origin['commit_tick']
        if commit is None:continue
        start=origin['decision_tick']
        if k in observed_future:
            assert observed_future[k]>=commit
            future_first.append((observed_future[k]-start)/600)
        if k in observed_center:
            assert observed_center[k]>=commit
            center_first.append((observed_center[k]-start)/600)
    def stats(values):
        import numpy as np
        if not values:return dict(count=0)
        return dict(count=len(values),minimum=min(values),p50=float(np.median(values)),p95=float(np.quantile(values,.95)),maximum=max(values))
    return dict(status='passed',schema='genmo.continuous_rolling.physical_audit.v1',windows=windows,
        behavior_policy_versions=sorted(versions),total_control_steps=controls,total_physics_steps=4*controls,
        duration_control_histogram=dict(sorted(durations.items())),prefix_frame_histogram=dict(sorted(prefixes.items())),
        successful_commits=commits,cancelled_at_boundary=cancelled,rejections=dict(rejections),terminal_reasons=dict(failures),
        timeout_count=len(timeouts),timeout_samples=timeouts[:8],overwritten_reference_ranges=overwrites,
        gamma_upper=next(iter(gammas)),boundary_targets_checked=len(boundary_target_errors),
        maximum_boundary_target_error=max(boundary_target_errors,default=0.),
        first_plan_future_observation_seconds=stats(future_first),first_plan_current_center_seconds=stats(center_first),
        first_plan_scope='reference_plan_identity_includes_inherited_prefix_not_all_new_free_coordinates',
        actor_updated=False,reward_units_unchanged=True,all_retained_controls_contiguous_once=True,
        input_reward_reference_versions_equal=True,seconds=time.perf_counter()-begun)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',nargs='+',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    result=audit(args.cases)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
