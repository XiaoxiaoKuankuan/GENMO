"""审计并分析三模型独立音乐组完整物理对照，输出可复核的小型JSON。

只接受全部rank正常关闭、三模型任务覆盖一致且轨迹/报告SHA正确的完整结果。
逐50Hz控制步核对奖励积分与事件罚分；真实失败保留，行政截断和基础设施错误
直接拒绝。完成率、每任务每秒回报与池化每秒回报分别统计，并同时给出各来源
和四来源等权汇总，避免长歌曲、较多舞者或重复种子改变来源权重。

比较使用同任务同seed的共同已执行前缀；跟踪项用对称乘积分解精确区分原始
tracking score与activity gate贡献，额外执行尾段及终止事件另列，不删早失败。
同时分析实际/参考关节运动、软硬边界、音乐同步和独立完整MC价值误差。
置信区间按音乐group成组抽样，两个seed始终一起，val/test各自计算；不把同组
种子当成更多独立音乐，也不把联合PPO+BC训练的效果单独归因于PPO。
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


MODELS = ('stage1', 'iteration147', 'current')
COMPONENTS = ('track', 'music', 'stable', 'alive', 'cmd', 'torque', 'contact', 'joint_limit')
METRICS = ('completed', 'reward_sum', 'reward_per_second', 'seconds', 'activity',
           'target_activity', 'activity_gate', 'actual_speed_rms', 'reference_speed_rms',
           'soft_joint_fraction', 'soft_any_frame_fraction', 'hard_joint_fraction',
           'reference_soft_joint_fraction', 'reference_hard_joint_fraction', 'score_track')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def close(actual, expected, message, *, atol=2e-7):
    if not np.allclose(actual, expected, atol=atol, rtol=2e-7):
        raise ValueError(message)


def boundary(q, limits, tolerance=0.):
    return (q < limits[:, 0]-tolerance) | (q > limits[:, 1]+tolerance)


def boundary_cost(q, safe, hard):
    """实际奖励的边界接近程度：软硬区间宽为零时，恰好触界仍计1。"""
    lw,uw=safe[:,0]-hard[:,0],hard[:,1]-safe[:,1]
    lower=np.divide(safe[:,0]-q,lw,out=np.zeros_like(q),where=lw>0)
    upper=np.divide(q-safe[:,1],uw,out=np.zeros_like(q),where=uw>0)
    lower=np.where((lw==0)&(q<=hard[:,0]),1.,lower)
    upper=np.where((uw==0)&(q>=hard[:,1]),1.,upper)
    return np.clip(np.maximum(lower,upper),0.,1.)


def slice_metrics(a, end=None):
    """同一真实控制前缀上的指标；边界分母为控制步×21关节。"""
    q = a['actual_joint_gmt'][:end]
    ref = a['reference_joint_gmt'][:end]
    count = len(q)
    if count < 1:
        raise ValueError('At least one actual control required')
    soft = boundary_cost(q, a['safe_limits'][0],a['hard_limits'][0])>0
    hard = boundary(q, a['hard_limits'][0], 1e-6)
    result = dict(seconds=count/50, reward_sum=float(a['reward'][:end].sum()),
        activity=float(a['activity'][:end].mean()), target_activity=float(a['target_activity'][:end].mean()),
        activity_gate=float(a['activity_gate'][:end].mean()),
        actual_speed_rms=float(np.sqrt(np.mean(a['joint_velocity'][:end]**2))),
        reference_speed_rms=float(np.sqrt(np.mean((np.diff(ref, axis=0)*50)**2))) if count>1 else 0.,
        soft_joint_fraction=float(soft.mean()), soft_any_frame_fraction=float(soft.any(axis=1).mean()),
        hard_joint_fraction=float(hard.mean()),
        reference_soft_joint_fraction=float((boundary_cost(ref, a['safe_limits'][0],a['hard_limits'][0])>0).mean()),
        reference_hard_joint_fraction=float(boundary(ref, a['hard_limits'][0], 1e-6).mean()),
        score_track=float(a['score_track'][:end].mean()))
    result['reward_per_second'] = result['reward_sum']/result['seconds']
    return result


def paired_prefix(a, b):
    """精确分解共同时间的跟踪收益，保留不等长真实执行尾段。"""
    n = min(len(a['tick']), len(b['tick']))
    close(a['tick'][:n]-a['tick'][0], b['tick'][:n]-b['tick'][0], 'Unaligned paired controls', atol=0.)
    close(a['target_activity'][:n], b['target_activity'][:n], 'Paired target activity differs')
    ga, gb = a['activity_gate'][:n], b['activity_gate'][:n]
    sa, sb = a['score_track'][:n], b['score_track'][:n]
    score = 2.5*.02*(sb-sa)*(ga+gb)/2
    gate = 2.5*.02*(gb-ga)*(sa+sb)/2
    track = b['reward_track'][:n]-a['reward_track'][:n]
    close(score+gate, track, 'Tracking product attribution does not reconstruct reward')
    delta = {key: float((b['reward_'+key][:n]-a['reward_'+key][:n]).sum()) for key in COMPONENTS}
    delta['event'] = float((b['reward'][:n]-a['reward'][:n]).sum()-sum(delta.values()))
    tail = float(b['reward'][n:].sum()-a['reward'][n:].sum())
    event = float(b['upper_event_reward'].sum()-a['upper_event_reward'].sum())
    total = float(b['reward'].sum()+b['upper_event_reward'].sum()-a['reward'].sum()-a['upper_event_reward'].sum())
    close(sum(delta.values())+tail+event, total, 'Paired total reward accounting differs')
    am, bm = slice_metrics(a, n), slice_metrics(b, n)
    return dict(common_seconds=n/50, total_reward_delta=total, common_components_delta=delta,
        common_components_delta_per_second={k:v/(n/50) for k,v in delta.items()},
        tracking_score_contribution=float(score.sum()), tracking_gate_contribution=float(gate.sum()),
        tracking_score_contribution_per_second=float(score.sum()/(n/50)),
        tracking_gate_contribution_per_second=float(gate.sum()/(n/50)),
        unequal_tail_reward_delta=tail, zero_control_event_delta=event,
        common_metrics_a=am, common_metrics_b=bm,
        common_metric_delta={key:bm[key]-am[key] for key in am})


def bootstrap_groups(rows, key, *, seed=713, draws=4000):
    """source内重抽独立音乐组，再四来源等权；同组seed先平均。"""
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[row['dataset']][row['group_id']].append(row[key])
    generator = np.random.default_rng(seed)
    means, samples, counts = [], [], {}
    for source, mapping in sorted(groups.items()):
        values = np.array([np.mean(items) for _, items in sorted(mapping.items())], dtype=np.float64)
        means.append(float(values.mean()))
        samples.append(values[generator.integers(0, len(values), size=(draws,len(values)))].mean(axis=1))
        counts[source] = len(values)
    estimates = np.mean(samples, axis=0)
    return dict(mean=float(np.mean(means)), ci95=np.quantile(estimates,[.025,.975]).tolist(),
                groups_by_source=counts, resampling_unit='music_group_with_all_seeds', draws=draws)


def critic_statistics(prediction, target):
    p, t = np.asarray(prediction), np.asarray(target)
    error = p-t
    return dict(states=len(t), mse=float(np.mean(error**2)), bias=float(np.mean(error)),
        mae=float(np.mean(np.abs(error))), explained_variance=float(1-np.var(error)/np.var(t)) if np.var(t)>1e-10 else None,
        mean_prediction=float(p.mean()), mean_mc=float(t.mean()))


def aggregate(rows):
    seconds = sum(row['seconds'] for row in rows)
    values = {key: float(np.mean([row[key] for row in rows])) for key in METRICS}
    return dict(episodes=len(rows), music_groups=len({r['group_id'] for r in rows}),
        completed_count=sum(row['completed'] for row in rows), failures=sum(row['failed'] for row in rows),
        end_reasons=dict(Counter(row['end_reason'] for row in rows)), means=values,
        pooled_reward_per_second=sum(r['reward_sum'] for r in rows)/seconds,
        total_executed_seconds=seconds, total_reward=sum(r['reward_sum'] for r in rows),
        components_per_second={key:float(np.mean([r['component_sums'][key]/r['seconds'] for r in rows])) for key in COMPONENTS},
        zero_control_events=sum(r['zero_control_event_reward_sum'] for r in rows))


def analyze(root):
    spec = read(root/'comparison_spec.json')
    completion = read(root/'completion.json')
    if completion['status']!='passed' or completion['updates_performed']:
        raise ValueError('Comparison must finish successfully without updates')
    if spec['smoke']:
        raise ValueError('Administrative smoke cannot be analyzed as a complete-song result')
    plans = {}
    for split, item in spec['plans'].items():
        path = root/item['file']
        if sha(path)!=item['sha256']:
            raise ValueError('Prepared group plan changed')
        plans[split] = {t['task_id']:t for t in read(path)['tasks']}
    for name in completion['ranks']:
        item = read(root/name)
        if item['status']!='passed' or any(not m['actor_unchanged'] for m in item['models']):
            raise ValueError('Incomplete rank or Actor changed during evaluation')
    arrays, episodes, curve, joint_rows, mc = {}, [], [], [], defaultdict(list)
    for model in MODELS:
        seen = set()
        for directory in sorted(root.glob('rank*/'+model+'/*')):
            if not directory.is_dir() or (directory/'empty.json').exists():
                continue
            split = directory.name
            manifest = read(directory/'trajectory_manifest.json')
            rp = directory/'report/report.json'
            if sha(rp)!=manifest['report_sha256']:
                raise ValueError('Evaluation report SHA changed')
            report = read(rp)
            if report['status']!='passed' or not report['networks_unchanged'] or report['updates_performed'] or report['training_buffer_used']:
                raise ValueError('Evaluation did not preserve frozen models')
            evidence = {m['task_id']:m for m in report['episode_manifests']}
            for meta in manifest['records']:
                task_id = meta['task_id'];task = meta['task']
                if task_id in seen or task != plans[split][task_id] or not meta['complete_episode'] or meta['truncated']:
                    raise ValueError('Incomplete, duplicate or mismatched comparison task')
                seen.add(task_id)
                path = directory/'trajectories'/meta['trajectory_file']
                if sha(path)!=meta['sha256'] or read(path.with_suffix('.json'))!=meta:
                    raise ValueError('Actual trajectory SHA or metadata changed')
                ep = evidence[task_id];epath = directory/'report'/ep['path']
                if sha(epath)!=ep['sha256']:
                    raise ValueError('Episode reward evidence changed')
                episode = read(epath)
                if not episode['transition_valid'] or episode['truncated']:
                    raise ValueError('Invalid or administrative episode')
                with np.load(path) as z:
                    a = {k:z[k] for k in z.files}
                arrays[(model,task_id)] = a
                n = len(a['tick'])
                if n!=meta['actual_control_steps'] or n*4!=meta['actual_physics_steps'] or not np.all(np.diff(a['tick'])==12):
                    raise ValueError('Real physical clock mismatch')
                controls = [c for d in episode['decisions'] for c in d['controls']]
                close(a['reward'],[c['reward'] for c in controls],'Per-control reward differs')
                for key in COMPONENTS:
                    close(a['reward_'+key],[c['components'][key]['integrated_reward'] for c in controls],key+' integral differs')
                continuous = sum(a['reward_'+key] for key in COMPONENTS)
                close(continuous+np.array([c['event_reward'] for c in controls]),a['reward'],'Components and failure reward do not sum')
                close(a['reward'].sum()+a['upper_event_reward'].sum(),episode['reward_sum'],'Episode reward sum differs')
                close(a['reward_music'],0.,'Music reward must be disabled')
                actual_cost=boundary_cost(a['actual_joint_gmt'],a['safe_limits'][0],a['hard_limits'][0])
                reference_cost=boundary_cost(a['reference_joint_gmt'],a['safe_limits'][0],a['hard_limits'][0])
                close(.5*actual_cost.max(axis=1)+.5*reference_cost.max(axis=1),a['score_joint_limit'],
                    'Joint boundary cost does not reconstruct active reward')
                if not np.isfinite(a['mc_return']).all():
                    raise ValueError('Complete episode requires finite MC')
                row = dict(model=model,task_id=task_id,split=split,dataset=task['dataset'],sample_id=task['sample_id'],
                    group_id=task['sample']['group_id'],seed=task['seed'],end_reason=meta['end_reason'],
                    completed=int(meta['end_reason'] in ('music_end','task_complete')),failed=int(episode['physical_failure']),
                    full_music_seconds=task['full_music_seconds'],trajectory=str(path.relative_to(root)),trajectory_sha256=meta['sha256'],
                    zero_control_event_reward_sum=meta['zero_control_event_reward_sum'],
                    **slice_metrics(a))
                row['reward_sum'] = episode['reward_sum']
                row['reward_per_second'] = row['reward_sum']/row['seconds']
                row['component_sums'] = {key:float(a['reward_'+key].sum()) for key in COMPONENTS}
                row['motion_diagnostic'] = episode['motion_diagnostic']
                row['generated_reference_diagnostic'] = episode['generated_reference_diagnostic']
                row['critic'] = critic_statistics(a['value_prediction'],a['mc_return'])
                row['initial_value_prediction']=float(a['value_prediction'][0])
                row['initial_mc_return']=float(a['mc_return'][0])
                row['raw_error_means'] = {key:float(np.mean([c['raw_errors'][key] for c in controls]))
                    for key in controls[0]['raw_errors'] if key!='control_tick'}
                row['last_5s_raw_error_means'] = {key:float(np.mean([c['raw_errors'][key] for c in controls[-250:]]))
                    for key in controls[0]['raw_errors'] if key!='control_tick'}
                row['last_control_raw_errors']=controls[-1]['raw_errors']
                row['prefix_range'] = [int(a['prefix_frames'].min()),int(a['prefix_frames'].max())]
                episodes.append(row)
                for j,name in enumerate(meta['gmt_joint_names']):
                    q,ref=a['actual_joint_gmt'][:,j],a['reference_joint_gmt'][:,j]
                    lo,hi=a['safe_limits'][0,j];hl,hh=a['hard_limits'][0,j]
                    joint_rows.append(dict(model=model,task_id=task_id,split=split,dataset=row['dataset'],group_id=row['group_id'],joint=name,
                        actual_soft_fraction=float(np.mean(actual_cost[:,j]>0)),reference_soft_fraction=float(np.mean(reference_cost[:,j]>0)),
                        actual_mean_cost=float(actual_cost[:,j].mean()),reference_mean_cost=float(reference_cost[:,j].mean()),
                        actual_max_cost_frame_fraction=float(np.mean((actual_cost[:,j]>0)&(actual_cost[:,j]==actual_cost.max(axis=1)))),
                        reference_max_cost_frame_fraction=float(np.mean((reference_cost[:,j]>0)&(reference_cost[:,j]==reference_cost.max(axis=1)))),
                        actual_hard_fraction=float(np.mean((q<hl-1e-6)|(q>hh+1e-6))),reference_hard_fraction=float(np.mean((ref<hl-1e-6)|(ref>hh+1e-6))),
                        actual_max_violation_rad=float(np.maximum(np.maximum(hl-q,q-hh),0).max()),
                        actual_range_rad=float(np.ptp(q)),reference_range_rad=float(np.ptp(ref))))
                for i in range(0,n,500):
                    b={k:v[i:i+500] if k not in ('safe_limits','hard_limits') and len(v)==n else v for k,v in a.items()}
                    curve.append(dict(model=model,task_id=task_id,split=split,dataset=row['dataset'],group_id=row['group_id'],
                        time_start=i/50,**slice_metrics(b)))
                # value误差按上层真实状态统计，长歌自然有更多状态，另保留逐episode统计。
                for idx,(p,t,remaining) in enumerate(zip(a['value_prediction'],a['mc_return'],a['remaining_music_seconds'])):
                    seconds_to_end=(n*12+int(a['tick'][0])-12-int(a['upper_tick'][idx]))/600
                    mc[(split,model)].append((float(p),float(t),float(remaining),float(seconds_to_end),row['dataset'],bool(row['failed'])))
        expected = {k for tasks in plans.values() for k in tasks}
        if seen!=expected:
            raise ValueError(f'{model}: task coverage missing {expected-seen}, extra {seen-expected}')
    summaries, paired, critic = {}, {}, {}
    for split in plans:
        summaries[split],paired[split],critic[split] = {},{},{}
        for model in MODELS:
            rows=[r for r in episodes if r['model']==model and r['split']==split]
            sources=sorted({r['dataset'] for r in rows})
            by_source={s:aggregate([r for r in rows if r['dataset']==s]) for s in sources}
            summaries[split][model]=dict(overall=aggregate(rows),by_source=by_source,
                source_balanced_means={key:float(np.mean([v['means'][key] for v in by_source.values()])) for key in METRICS},
                source_balanced_component_rates={key:float(np.mean([v['components_per_second'][key] for v in by_source.values()])) for key in COMPONENTS})
            values=mc[(split,model)]
            def stat(items):return critic_statistics([v[0] for v in items],[v[1] for v in items]) if items else None
            critic[split][model]=dict(overall=stat(values),by_source={s:stat([v for v in values if v[4]==s]) for s in sources},
                by_remaining_seconds={f'{lo}-{hi}':stat([v for v in values if lo<=v[2]<hi]) for lo,hi in ((0,10),(10,30),(30,60),(60,300))},
                last_10_seconds_before_failure=stat([v for v in values if v[5] and v[3]<=10]))
        for before,after in (('stage1','iteration147'),('stage1','current'),('iteration147','current')):
            rows=[]
            erows={r['task_id']:r for r in episodes if r['split']==split and r['model']==before}
            brows={r['task_id']:r for r in episodes if r['split']==split and r['model']==after}
            for tid in plans[split]:
                ra,rb=erows[tid],brows[tid]
                item=dict(task_id=tid,split=split,dataset=ra['dataset'],group_id=ra['group_id'],seed=ra['seed'],
                    **{k:rb[k]-ra[k] for k in METRICS},**paired_prefix(arrays[(before,tid)],arrays[(after,tid)]))
                rows.append(item)
            paired[split][after+'_minus_'+before]=dict(tasks=rows,
                bootstrap={key:bootstrap_groups(rows,key) for key in METRICS},
                common_bootstrap={key:bootstrap_groups([dict(dataset=r['dataset'],group_id=r['group_id'],value=r['common_metric_delta'][key]) for r in rows],'value')
                    for key in ('reward_per_second','activity','actual_speed_rms','reference_speed_rms','soft_joint_fraction','score_track','activity_gate')})
    return dict(schema='genmo.stage10.group_comparison_analysis.v1',status='passed',models=spec['models'],
        comparison_id=spec['comparison_id'],tasks_per_model=sum(len(v) for v in plans.values()),episodes=episodes,
        summaries=summaries,paired=paired,critic=critic,joint_rows=joint_rows,time_curves=curve,
        verification=dict(identical_task_and_noise=True,report_and_trajectory_sha=True,all_control_rewards_reconciled=True,
            music_reward_zero=True,all_complete_no_administrative_truncation=True,frozen_models_unchanged=True),
        scope=dict(main_weighting='equal_sources_equal_music_groups_equal_seeds',
            uncertainty='source_stratified_paired_music_group_bootstrap_not_independent_seeds',
            reference_speed='finite_difference_of_actually_consumed_50Hz_reference_includes_replanning',
            hard_boundary='beyond_asset_hard_limits_by_more_than_1e-6_radians_not_simulator_crash',
            soft_boundary='positive_active_reward_cost_includes_touch_when_soft_equals_hard',
            music='full_available_paired_sample_not_uncropped_original_audio',
            critic='on_policy_complete_MC_per_checkpoint_critic_state_weighted',
            attribution='exact_tracking_reward_product_decomposition_not_causal_PPO_vs_BC_ablation'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    result=analyze(args.root.resolve())
    with args.output.open('x') as stream:json.dump(result,stream,ensure_ascii=False,allow_nan=False,indent=2)
    print(json.dumps(dict(status=result['status'],tasks_per_model=result['tasks_per_model'],output=str(args.output)),ensure_ascii=False))


if __name__=='__main__':main()
