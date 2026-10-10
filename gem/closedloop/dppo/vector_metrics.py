"""GPU多环境曲线的按真实执行统计，区分批量计算时间与各环境等待时间。

每个执行奖励分项按已经执行的控制区间积分相加，含轮末物理尾段；拒绝、超时按
上层转移计数，不漏记终止。一次真实GENMO批量的条件/去噪/解码耗时只加一次，
不能把交付给8个环境的同一批计时再加8遍。逐环境critical延迟和排队仍保留原值，
它们允许重叠且不等于墙钟。此模块只读固定rollout与采集报告，不影响奖励和GAE。
"""
from __future__ import annotations


def attach_vector_metrics(rows, report):
    report['rejections']=sum(bool(row.metadata.get('rejection')) for row in rows)
    report['timeouts']=sum('timeout' in str(row.reason or '').lower() for row in rows)
    report['reward_component_sums']={}
    report['generation_timing_totals']={}
    report['actor_phase_totals']={}
    report['modeled_delay_ticks']=[]
    report['prefix_frames']=[]
    report['boundary_wait_by_environment']=[]
    report['new_reference_first_gmt_input']=[]
    report['columnar_reward_fallbacks']=0
    first={}
    for row in rows:
        for plan,tick in row.metadata.get('consumed_plan_first_gmt_input_tick',{}).items():
            key=(row.identity.get('env_id'),row.identity['episode_id'],plan)
            first[key]=min(tick,first.get(key,tick))
    for row in rows:
        key=(row.identity.get('env_id'),row.identity['episode_id'],row.identity['plan_id'])
        report['new_reference_first_gmt_input'].append(dict(env_id=key[0],episode_id=key[1],plan_id=key[2],
            tick=first.get(key),observed=key in first))
        tail=row.metadata.get('fragment_tail')
        if tail is not None:
            total=float(row.rewards.sum())+row.metadata.get('event_reward',0.)
            report['boundary_wait_by_environment'].append(dict(env_id=key[0],episode_id=key[1],
                control_steps=tail['executed_control_steps'],reward_sum=tail['reward_sum'],
                fraction_of_last_transition_reward=None if total==0 else tail['reward_sum']/total))
        report['columnar_reward_fallbacks']=max(report['columnar_reward_fallbacks'],
            row.metadata.get('reward_columnar_fallbacks',0))
        clock=row.metadata.get('timing',{}).get('deployment_clock')
        if clock is not None: report['modeled_delay_ticks'].append(clock['delay_ticks'])
        prefix=row.metadata.get('generated',{}).get('prefix_frames')
        if prefix is not None: report['prefix_frames'].append(prefix)
        for detail in row.metadata.get('reward_details',[]):
            for name,value in detail.get('components',{}).items():
                value=value.get('integrated_reward') if isinstance(value,dict) else value
                if isinstance(value,(int,float)):
                    report['reward_component_sums'][name]=report['reward_component_sums'].get(name,0.)+value
        for name,value in row.metadata.get('timing',{}).items():
            if isinstance(value,(int,float)):
                report['generation_timing_totals'][name]=report['generation_timing_totals'].get(name,0.)+value
    for batch in report['batches']:
        for name,value in batch.get('components',{}).items():
            if isinstance(value,(int,float)) and not isinstance(value,bool):
                report['actor_phase_totals'][name]=report['actor_phase_totals'].get(name,0.)+value
    report['actor_timing_interval_scopes']=sorted({b['components']['interval_scope'] for b in report['batches']
        if 'interval_scope' in b.get('components',{})})
    report['generation_batch_wall_seconds']=sum(b['generation_seconds'] for b in report['batches'])
    for source,target,scale in (('modeled_delay_ticks','modeled_delay_seconds',1/600.),('prefix_frames','prefix_frames',1.)):
        values=report[source]
        if values:
            report[target+'_mean']=sum(values)/len(values)*scale
            report[target+'_min']=min(values)*scale
            report[target+'_max']=max(values)*scale
    report['generation_pipeline_totals']={}
    report['effective_generation_batch_histogram']={}
    for batch in report['batches']:
        size=str(batch['effective_rows'])
        report['effective_generation_batch_histogram'][size]=report['effective_generation_batch_histogram'].get(size,0)+1
        for name,value in (batch.get('pipeline_timing') or {}).items():
            if name.endswith('_seconds') and isinstance(value,(int,float)):
                report['generation_pipeline_totals'][name]=report['generation_pipeline_totals'].get(name,0.)+value
    report['generation_timing_scope']='per_environment_latency_sum_can_overlap'
    report['actor_phase_scope']='each_real_batch_once_no_per_environment_duplication'
    return report
