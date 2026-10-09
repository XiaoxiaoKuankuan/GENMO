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
    for row in rows:
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
            report['actor_phase_totals'][name]=report['actor_phase_totals'].get(name,0.)+value
    report['generation_batch_wall_seconds']=sum(b['generation_seconds'] for b in report['batches'])
    report['generation_timing_scope']='per_environment_latency_sum_can_overlap'
    report['actor_phase_scope']='each_real_batch_once_no_per_environment_duplication'
    return report
