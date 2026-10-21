"""汇总有限连续Stage2训练及相同完整音乐任务的物理结果。

输入为独立训练目录及证据输出目录。通过既有完整配对校验器读取Stage1基线与各轮
评估，保留所有失败、奖励组成、时长与活动度；另对双方共同执行的时间前缀统计八项
真实平方跟踪误差，避免提前失败、缩短轨迹使误差看起来下降。原完整任务回报与失败
结果保持完整，共同前缀只是附加诊断，不删除提前失败代价，也不作反事实补全。

四来源每秒回报等权；不把不同来源的音乐数或播放时长当成任务权重。置信区间按
音乐聚合两个种子，在来源内重采样，不把两个种子误当两首独立音乐。训练统计区分
计划、执行、接受更新与真实整批目标，以及每轮墙钟和显存。此工具只读源报告，
不训练、不挑选checkpoint、不改变奖励、不自动开训或因统计阈值阻止训练。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gem.closedloop.dppo.compact_evidence import report_episodes
from tools.eval.compare_closedloop_stage10 import compare_evaluations

ERRORS=('anchor_pos','anchor_ori','body_pos','body_ori','body_lin_vel','body_ang_vel','foot_height','leg_joint_pos')


def balanced_rate_and_interval(pairs, *, draws=10000):
    """按来源等权、音乐为cluster，失败任务与两个种子均保留。"""
    groups=defaultdict(lambda:defaultdict(list))
    for pair in pairs:
        groups[pair['dataset']][pair['sample_id']].append(pair)
    if not groups:raise ValueError('No paired tasks')
    first=[];last=[];bootstrap=[];rng=np.random.default_rng(20261021)
    for music in groups.values():
        clusters=list(music.values())
        a=np.array([np.mean([p['initial']['reward_per_executed_second'] for p in c]) for c in clusters])
        b=np.array([np.mean([p['final']['reward_per_executed_second'] for p in c]) for c in clusters])
        if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Missing actual paired reward rates')
        first.append(float(a.mean()));last.append(float(b.mean()))
        indices=rng.integers(len(clusters),size=(draws,len(clusters)))
        bootstrap.append((b-a)[indices].mean(axis=1))
    initial=float(np.mean(first));final=float(np.mean(last))
    interval=np.quantile(np.mean(bootstrap,axis=0),[.025,.975]).tolist()
    return dict(initial=initial,final=final,delta=final-initial,
        relative_change=(final/initial-1) if initial else None,
        music_cluster_bootstrap_95=interval,music_clusters=sum(len(v) for v in groups.values()),
        interpretation='fixed_train_music_only; interval_crossing_zero_does_not_establish_overall_gain')


def matched_control_prefix(before, after):
    """只比较双方真实执行且时间对齐的控制，不外推失败后的轨迹。"""
    if before['task_id']!=after['task_id']:raise ValueError('Different physical tasks')
    a=[r for d in before['decisions'] for r in d['controls']]
    b=[r for d in after['decisions'] for r in d['controls']]
    n=min(len(a),len(b));duration=0.;errors={side:{k:0. for k in ERRORS} for side in ('initial','final')}
    for x,y in zip(a[:n],b[:n]):
        dt=float(x['dt_s'])
        if dt<=0 or not math.isclose(dt,float(y['dt_s']),rel_tol=0.,abs_tol=1e-12):
            raise ValueError('Unaligned control intervals')
        # 控制顺序与dt定义相对同一固定任务起点；不同世界session的绝对tick不可直接比较。
        for side,row in (('initial',x),('final',y)):
            for k in ERRORS:
                value=float(row['raw_errors'][k+'_squared_error'])
                if not math.isfinite(value) or value<0:raise ValueError('Invalid squared physical error')
                errors[side][k]+=dt*value
        duration+=dt
    return dict(task_id=before['task_id'],dataset=before['dataset'],controls=n,seconds=duration,
        squared_error_integrals=errors,initial_controls=len(a),final_controls=len(b),
        full_task_failures=[bool(before['physical_failure']),bool(after['physical_failure'])])


def physical_summary(comparison, initial_report, final_report):
    physical=comparison['overall']['actual_execution_aggregate']
    def rms(side,k):return math.sqrt(physical[side]['raw_errors'][k+'_squared_error']['mean'])
    errors={k:dict(initial=rms('initial',k),final=rms('final',k),
        relative_change=rms('final',k)/rms('initial',k)-1 if rms('initial',k) else None) for k in ERRORS}
    prefixes=[matched_control_prefix(a,b) for a,b in zip(report_episodes(initial_report),report_episodes(final_report))]
    duration=sum(p['seconds'] for p in prefixes)
    common={k:{side:math.sqrt(sum(p['squared_error_integrals'][side][k] for p in prefixes)/duration)
               if duration else None for side in ('initial','final')} for k in ERRORS}
    return dict(tasks=comparison['paired_task_count'],failure_pairs=comparison['overall']['failure_pair_counts'],
        source_balanced_reward_rate=balanced_rate_and_interval(comparison['pairs']),all_actual_tracking_rms=errors,
        actual_seconds={side:physical[side]['executed_seconds'] for side in ('initial','final')},
        completed={side:physical[side]['music_completion_count'] for side in ('initial','final')},
        reward_integral={side:physical[side]['reward_sum'] for side in ('initial','final')},
        actual_activity={side:physical[side]['activity']['actual_activity_rad_s']['mean'] for side in ('initial','final')},
        reward_component_integrals={side:{k:v['integrated_reward_sum'] for k,v in physical[side]['components'].items()}
            for side in ('initial','final')},
        common_executed_prefix=dict(seconds=duration,tracking_rms=common,tasks=prefixes,
            scope='auxiliary_common_time_only; failures_and_full_returns_still_reported'),
        by_source={source:dict(reward_rate_delta=v['task_weighted_metrics']['reward_per_executed_second']['delta']['mean'],
            failure_pairs=v['failure_pair_counts']) for source,v in comparison['by_source'].items()})


def training_summary(run):
    records=[]
    for path in sorted((run/'metrics').glob('*.jsonl')):
        for line in path.read_text().splitlines():
            row=json.loads(line)
            if 'state' not in row or 'actor' not in row or row.get('mode')!='dppo_bc':continue
            actor=row['actor'];history=actor.get('full_rollout_objective_history',[])
            values=[r['objective_mean'] for r in history]
            records.append(dict(iteration=row['state']['iteration'],label=row['label'],
                eligible_chains=actor['eligible_chains'],planned_steps=actor['planned_optimizer_steps'],
                executed_steps=actor['executed_optimizer_steps'],accepted_steps=actor['accepted_optimizer_steps'],
                stop_reason=actor.get('early_stop_reason'),
                full_rollout_objective_gain=values[-1]-values[0] if values else None,
                each_accepted_objective_nondecreasing=all(b>=a-1e-10 for a,b in zip(values,values[1:])),
                nominal_lr=actor['nominal_learning_rate'],accepted_lrs=[s['learning_rate'] for s in actor['steps']],
                chain_joint_kl=row['trainable4_kl']['chain_joint_kl'],
                actual_bc_samples=actor['actual_global_bc_samples'],efficiency=row['efficiency']))
    return sorted(records,key=lambda r:r['iteration'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    reports=[]
    for path in args.run_dir.glob('sessions/*/evaluations/*.json'):
        report=json.loads(path.read_text())
        if report.get('status')=='passed':reports.append((int(report['iteration']),path,report))
    reports.sort(key=lambda r:r[0])
    initial=next((r for r in reports if r[0]==0),None)
    if initial is None:raise ValueError('No completed Stage1 baseline in this run')
    summaries=[]
    for iteration,path,report in reports:
        if iteration==0:continue
        comparison=compare_evaluations(initial[1],path)
        target=args.output_dir/f'paired_iteration{iteration:06d}.json'
        target.write_text(json.dumps(comparison,ensure_ascii=False,indent=2,allow_nan=False))
        summary=physical_summary(comparison,initial[2],report)
        summaries.append(dict(iteration=iteration,report=str(path),**summary))
    result=dict(schema='genmo.stage2.continuous_finite_validation.v1',run_dir=str(args.run_dir),
        baseline=str(initial[1]),training=training_summary(args.run_dir),physical=summaries,
        no_automatic_training_launch=True)
    (args.output_dir/'continuous_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps(dict(iterations=[r['iteration'] for r in result['training']],
        evaluated=[r['iteration'] for r in summaries]),ensure_ascii=False))


if __name__=='__main__':main()
