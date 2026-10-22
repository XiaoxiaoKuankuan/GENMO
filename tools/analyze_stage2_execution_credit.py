"""只读审计真实参考消费、递归前缀来源、跨rollout执行及延迟回报。

输入一个或多个带SHA的真实fixed_data_validation。先按物理episode合并控制记录，
递归解析继承前缀的原始生成计划，再分别统计中心、前瞻、回看与未观测支持。
输出每计划/环境/数据来源/前缀/8秒时间段统计和可选逐控制压缩证据，位置关节使用
线性插值权重，姿态只声明相邻源帧的SLERP依赖，速度另保留50Hz差分邻点与符号系数。

同一episode连续片段用于只读回报比较，不合并为PPO batch；Actor版本边界切断
价值分析。真正终止的尾部才可称无自举MC；正常截断保留next_value自举，不能声称
完整episode MC。缺失历史、越界支持和右截断明确标记，不补造执行、不搬移奖励，
不更改Actor/Critic观测、奖励公式、策略或任何物理时间。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
import gzip
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from gem.closedloop.dppo.budget import atomic_json
from gem.closedloop.dppo.diagnostic_rollout import load_rollout
from gem.closedloop.dppo.execution_provenance import consumption_report,source_frame_support
from gem.closedloop.dppo.returns import compute_gae
from gem.closedloop.dppo.training_config import configuration
from tools.analyze_stage2_reference_consumption import token,source_origin,temporal_window_support


def distribution(values):
    x=torch.tensor([v for v in values if v is not None],dtype=torch.float64)
    if not x.numel():return dict(count=0)
    return dict(count=x.numel(),mean=float(x.mean()),p50=float(torch.quantile(x,.5)),
        p95=float(torch.quantile(x,.95)),min=float(x.min()),max=float(x.max()))


def episode_analysis(key,entries,*,control_stream=None,gamma=.99,lam=.98):
    records=[row.metadata['compact_execution_provenance'] for _,row,_ in entries
        if 'compact_execution_provenance' in row.metadata]
    report=consumption_report(records)
    if not records:return [],[],dict(episode=key,status='unverified_missing_provenance')
    plans=report['plans'];temporal=temporal_window_support(records,plans);controls={}
    by_plan={str(row.identity.get('plan_id')):(case,row,target) for case,row,target in entries}
    for record in records:
        for item in record['steps']:
            ids=item.get('consumed_plan_ids')
            if ids is None:ids=[record['plan_dictionary'][i] for i in item['consumed_plan_indices']]
            controls[item['control_tick_begin']]=(item['consumed_reference_ticks'],ids)
    @lru_cache(maxsize=None)
    def support(plan,tick):
        if plan.startswith('bootstrap:'):return (('bootstrap',None,1.),)
        item=plans[plan];result=[]
        for frame,weight in source_frame_support(tick,item['decision_tick'],item.get('generation_frames',60)):
            original=source_origin(plans,plan,frame)
            result.append((None if original is None else original[0],None if original is None else original[1],weight))
        return tuple(result)
    first_center={};first_window={};first_future={};case_consumed=defaultdict(set)
    row_intervals=[(row.control_tick_begin,row.control_tick_end,case) for case,row,_ in entries]
    row_intervals.sort()
    interval=0
    for begin,(ticks,ids) in sorted(controls.items()):
        while interval+1<len(row_intervals) and begin>=row_intervals[interval][1]:interval+=1
        case=row_intervals[interval][2]
        for index,(tick,plan) in enumerate(zip(ticks,ids)):
            for original,frame,weight in support(plan,tick):
                if original not in plans:continue
                first_window.setdefault(original,begin)
                if index>10:first_future.setdefault(original,begin)
                if index==10:
                    first_center.setdefault(original,begin)
                    case_consumed[(original,frame)].add(case)
        if control_stream is not None:
            # 中心点具有实际21点窗口，所以确定是权威数组内部点；不把片段边界当差分端点。
            pose=support(ids[10],ticks[10]);velocity=[]
            for index,factor in ((9,-25.),(11,25.)):
                velocity.extend(dict(plan_id=p,frame=f,coefficient_per_second=w*factor)
                    for p,f,w in support(ids[index],ticks[index]))
            control_stream.write(json.dumps(dict(episode=key,control_tick=begin,
                center_label_plan=ids[10],pose_source=[dict(plan_id=p,frame=f,weight=w) for p,f,w in pose],
                position_joint_velocity_source=velocity,linear_coefficients_scope='before_production_FP32_rounding',
                quaternion='SLERP_on_pose_support_nonlinear',
                angular_velocity='relative_rotation_of_SLERP_at_control_minus12_plus12_nonlinear',
                body_velocity='FK_current_pose_plus_root_twist_and_joint_velocity_support',
                velocity_pose_dependencies=list({(p,f) for index in (9,10,11)
                    for p,f,w in support(ids[index],ticks[index])})),ensure_ascii=False)+'\n')
    begin=min(controls);end=max(controls)+12
    continuous=len(controls)*12==end-begin
    details=[]
    for plan,item in plans.items():
        pair=by_plan.get(plan)
        case,row,_=pair if pair else (None,None,None)
        prefix=item.get('prefix_frames');n=item.get('generation_frames',60)
        center=set(item['observed_current_generated_frames'])
        future={i for i,v in enumerate(temporal[plan]['future']) if v>0}
        free=set(range(prefix,n)) if prefix is not None else set()
        later=[v['commit_tick'] for p,v in plans.items() if p!=plan and v['commit_tick'] is not None
            and item['commit_tick'] is not None and v['commit_tick']>item['commit_tick']]
        overwritten=[f for f in free-center if continuous and begin<=item['decision_tick']+20*f<end
            and any(t<=item['decision_tick']+20*f for t in later)]
        cross=[f for f in center if case is not None and any(c!=case for c in case_consumed[(plan,f)])]
        def delay(t):return None if t is None else (t-item['decision_tick'])/600
        details.append(dict(episode=key,plan_id=plan,source=(row.metadata.get('training_task',{}).get('dataset','unknown') if row else 'outside_case'),
            case=case,prefix_frames=prefix,total_generated_frames=n,new_free_frames=len(free),
            decision_tick=item['decision_tick'],time_segment_8s=item['decision_tick']//4800,
            center_source_frames=sorted(center),center_count=len(center),
            future_only_source_frames=sorted(future-center),future_only_count=len(future-center),
            overwritten_in_observed_range=overwritten,right_censored_frames=sorted(f for f in free-center if item['decision_tick']+20*f>=end),
            cross_rollout_source_frames=sorted(cross),cross_rollout_count=len(cross),
            decision_to_commit_seconds=delay(item['commit_tick']),
            decision_to_first_window_seconds=delay(first_window.get(plan)),
            decision_to_first_future_seconds=delay(first_future.get(plan)),
            decision_to_first_center_seconds=delay(first_center.get(plan)),
            generation_wall_seconds=item.get('generation_wall_seconds'),
            clock_model_ready_tick=item.get('generation_ready_tick'),
            ready_to_commit_seconds=(None if item.get('generation_ready_tick') is None or item['commit_tick'] is None
                else (item['commit_tick']-item['generation_ready_tick'])/600),
            full_observed_horizon=continuous and begin<=item['decision_tick'] and item['decision_tick']+(n-1)*20<end))
    values=[]
    ordered=sorted(entries,key=lambda x:x[1].control_tick_begin)
    fragments=[]
    for entry in ordered:
        row=entry[1]
        if not fragments or fragments[-1][-1][1].control_tick_end!=row.control_tick_begin or fragments[-1][-1][1].terminated or (
                fragments[-1][-1][1].identity['policy_version']!=row.identity['policy_version']):fragments.append([])
        fragments[-1].append(entry)
    for fragment in fragments:
        rows=[v[1] for v in fragment];n=len(rows)
        args=dict(rewards=[r.rewards for r in rows],values=[r.old_value for r in rows],next_values=[r.next_value for r in rows],
            executed_steps=[r.executed_control_steps for r in rows],bootstrap_mask=[not r.terminated for r in rows],
            continuation_mask=[i<n-1 for i in range(n)],valid=[r.transition_valid for r in rows],
            event_rewards=[r.metadata.get('event_reward',0.) for r in rows],gamma_upper=gamma,normalize=False)
        mc=compute_gae(**args,lambda_upper=1.)['returns']
        linked=compute_gae(**args,lambda_upper=lam)['returns']
        for i,(case,row,target) in enumerate(fragment):
            plan=str(row.identity.get('plan_id'));first=first_center.get(plan)
            rewards=row.rewards.double();ticks=torch.arange(len(rewards))*12+row.control_tick_begin
            # 没在保留区间看见首次消费时只能说尚未观察，不能假定此计划永远不消费。
            before_mask=torch.ones_like(ticks,dtype=torch.bool) if first is None else ticks<first
            future_tick=first_future.get(plan)
            before_future=torch.ones_like(ticks,dtype=torch.bool) if future_tick is None else ticks<future_tick
            values.append(dict(episode=key,plan_id=plan,case=case,policy_version=row.identity['policy_version'],
                source=row.metadata.get('training_task',{}).get('dataset','unknown'),
                controls=len(rewards),controls_before_generated_center=int(before_mask.sum()),
                controls_before_generated_future_window=int(before_future.sum()),
                reward_before_generated_center=float(rewards[before_mask].sum()),total_reward=float(rewards.sum()),
                reward_before_generated_future_window=float(rewards[before_future].sum()),
                generated_center_not_observed=first is None,terminated=row.terminated,truncated=row.truncated,
                end_reason=getattr(row,'reason',None),
                near_real_termination=bool(rows[-1].terminated and i>=n-3),
                near_physical_failure=bool(rows[-1].terminated and getattr(rows[-1],'reason',None) not in
                    (None,'music_end','task_complete') and i>=n-3),
                rollout_end=not row.terminated and (i==n-1 or (i+1<n and fragment[i+1][0]!=case)),
                bootstrap_value=row.next_value,old_value=row.old_value,frozen_gae_target=float(target),
                linked_gae_target=float(linked[i]),lambda1_return=float(mc[i]),
                lambda1_type='complete_observed_tail_no_bootstrap' if rows[-1].terminated else 'right_censored_bootstrapped_not_full_MC',
                critic_error_lambda1=float(row.old_value-mc[i]),
                gae_target_error_lambda1=float(target-mc[i])))
    return details,values,dict(episode=key,status=report['status'],control_count=len(controls),
        source_lineage_complete=report['source_lineage_complete'],continuous_control_range=continuous,
        begin_tick=begin,end_tick=end,closed_episode=bool(ordered[-1][1].terminated))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',nargs='+',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--control-evidence',action='store_true')
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    config=configuration(args.config,genmo_repo=ROOT);episodes=defaultdict(list)
    missing=0
    for case in args.cases:
        for rank in sorted(case.glob('rank[0-9][0-9]')):
            payload=load_rollout(rank)
            for i,row in enumerate(payload['transitions']):
                if 'compact_execution_provenance' not in row.metadata:missing+=1
                episodes[token(row.identity)[:3]].append((str(case),row,payload['targets']['returns'][i]))
    stream=gzip.open(args.output_dir/'actual_center_support.jsonl.gz','wt',encoding='utf8') if args.control_evidence else None
    plans=[];values=[];ranges=[]
    try:
        for key,entries in episodes.items():
            p,v,r=episode_analysis(key,entries,control_stream=stream,
                gamma=config['stage10']['training']['gamma_upper'],lam=config['stage10']['training']['lambda_upper'])
            plans.extend(p);values.extend(v);ranges.append(r)
    finally:
        if stream is not None:stream.close()
    for name,items in [('plans',plans),('delay_and_values',values)]:
        with gzip.open(args.output_dir/(name+'.jsonl.gz'),'wt',encoding='utf8') as f:
            for row in items:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    summaries={}
    for field in ('source','prefix_frames','time_segment_8s','environment'):
        groups=defaultdict(list)
        for row in plans:groups[str(row['episode'][1] if field=='environment' else row[field])].append(row)
        summaries[field]={k:dict(plans=len(rows),**{name:distribution([r[name] for r in rows]) for name in
            ('center_count','future_only_count','cross_rollout_count','decision_to_commit_seconds',
             'decision_to_first_center_seconds')}) for k,rows in groups.items()}
    controls=sum(v['controls'] for v in values);early=sum(v['controls_before_generated_center'] for v in values)
    early_future=sum(v['controls_before_generated_future_window'] for v in values)
    report=dict(cases=[str(p) for p in args.cases],episodes=ranges,missing_compact_transitions=missing,
        plans=len(plans),by=summaries,controls=controls,controls_before_generated_center=early,
        before_center_control_fraction=early/controls if controls else None,
        controls_before_generated_future_window=early_future,
        before_future_window_control_fraction=early_future/controls if controls else None,
        complete_horizon_plan_count=sum(p['full_observed_horizon'] for p in plans),
        complete_horizon_center_count=distribution([p['center_count'] for p in plans if p['full_observed_horizon']]),
        cross_rollout_plan_count=sum(p['cross_rollout_count']>0 for p in plans),
        critic_groups={name:dict(samples=len(rows),critic_error_lambda1=distribution([r['critic_error_lambda1'] for r in rows]),
            gae_target_error_lambda1=distribution([r['gae_target_error_lambda1'] for r in rows]))
            for name,rows in [('all',values),('near_termination',[r for r in values if r['near_real_termination']]),
                ('near_physical_failure',[r for r in values if r['near_physical_failure']]),
                ('rollout_end',[r for r in values if r['rollout_end']]),
                ('full_observed_tail',[r for r in values if r['lambda1_type']=='complete_observed_tail_no_bootstrap'])]},
        velocity_evidence='actual_center_21point_window_confirms_interior_difference_support; '
            'linear_position_joint_coefficients_separate_from_nonlinear_rotation_FK',
        causal_scope='direct_center_future_window_and_Transformer_temporal_coupling_are_distinct; '
            'unconsumed_probability_factors_are_not_deleted',
        MC_scope='only_true_termination_gives_no_bootstrap_observed_tail; clipped_rollout_is_not_complete_episode_MC')
    atomic_json(args.output_dir/'summary.json',report)
    print(json.dumps(dict(plans=len(plans),controls=controls,missing=missing),ensure_ascii=False))


if __name__=='__main__':main()
