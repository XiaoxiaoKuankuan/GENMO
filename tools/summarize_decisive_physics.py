"""汇总决定性A/C实验的同任务物理差分、音乐聚类区间与执行参考变化。

输入为独立诊断目录，各rank代表一个明确候选而非同一Actor的分片。核验任务计划、
episode JSON和控制侧车SHA，按真实时间重算奖励、失败和八项跟踪误差；失败后的
未执行部分不外推，额外报告共同执行前缀。两个种子随同一首音乐一起bootstrap。
参考变化由保存的世界qpos计划和实际GMT中心来源插值重建，不把整段生成变化当作
实际消费变化。A同时报告正负差分及相对打乱/随机方向的差分；C保留合法性拒绝计数。
本入口只读模型和物理产物，不创建仿真或优化器；置信区间只描述固定小队列。
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gem.closedloop.dppo.budget import atomic_json
from gem.closedloop.dppo.compact_evidence import read_episode
from gem.closedloop.dppo.evaluation import _digest
from gem.closedloop.dppo.execution_provenance import source_frame_support
from tools.summarize_stage2_continuous_validation import ERRORS,matched_control_prefix,tracking_distribution


def cluster_summary(rows,field):
    groups=defaultdict(lambda:defaultdict(list))
    for row in rows:groups[row['dataset']][row['sample_id']].append(float(row[field]))
    rng=np.random.default_rng(20261024);boots=[];means=[]
    for source,musics in sorted(groups.items()):
        values=np.asarray([np.mean(v) for _,v in sorted(musics.items())]);means.append(values.mean())
        boots.append(rng.choice(values,(10000,len(values)),replace=True).mean(1))
    return dict(mean=float(np.mean(means)),music_cluster_95=np.quantile(np.mean(boots,axis=0),[.025,.975]).tolist(),
        music_clusters=sum(len(v) for v in groups.values()),task_count=len(rows),weighting='equal_sources_then_music_then_seed')


def load_report(path):
    path=Path(path);report=json.loads(path.read_text());plan=json.loads((path.parent/'selection.json').read_text())
    declared=plan.pop('plan_sha256');actual=_digest(plan);plan['plan_sha256']=declared
    if declared!=actual or report['plan_sha256']!=declared or report['status']!='passed':raise ValueError('Unverified evaluation plan/report')
    episodes=[]
    for item in report['episode_manifests']:
        file=path.parent/item['path']
        if hashlib.sha256(file.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Episode SHA mismatch')
        episode=read_episode(file)
        if episode['task_id']!=item['task_id'] or not episode['transition_valid']:raise ValueError('Episode mismatch')
        episodes.append(episode)
    if len(episodes)!=len(plan['tasks']):raise ValueError('Incomplete cohort')
    return report,plan,episodes


def metrics(episode):
    controls=[c for d in episode['decisions'] for c in d['controls']];seconds=sum(c['dt_s'] for c in controls)
    reward=sum(c['reward'] for c in controls)+sum(d['zero_step_event_reward'] for d in episode['decisions'])
    if not math.isclose(reward,episode['reward_sum'],abs_tol=1e-6):raise ValueError('Complete reward differs')
    return dict(reward_per_second=reward/seconds,full_task_reward=reward,seconds=seconds,
        failure=float(episode['physical_failure']),completion=float(episode['end_reason'] in ('music_end','task_complete')),
        activity=float(np.mean([c['activity']['actual_activity_rad_s'] for c in controls if c['activity'].get('valid')])),
        music_alignment=float(episode['motion_diagnostic'].get('music_alignment',0.)),
        tracking={k:math.sqrt(sum(c['raw_errors'][k+'_squared_error']*c['dt_s'] for c in controls)/seconds) for k in ERRORS},
        generated_diagnostic=episode['generated_reference_diagnostic'])


def interpolate(qpos,support):
    out=np.zeros(28,dtype=np.float64)
    for index,weight in support:out+=weight*qpos[index]
    q0=qpos[support[0][0],3:7];q0=q0/np.linalg.norm(q0)
    if len(support)==1:out[3:7]=q0;return out
    q1=qpos[support[1][0],3:7];q1=q1/np.linalg.norm(q1);dot=np.dot(q0,q1)
    if dot<0:q1=-q1;dot=-dot
    t=support[1][1]
    if dot>.9995:q=(1-t)*q0+t*q1;q=q/np.linalg.norm(q)
    else:
        angle=np.arccos(np.clip(dot,-1,1));q=(np.sin((1-t)*angle)*q0+np.sin(t*angle)*q1)/np.sin(angle)
    out[3:7]=q;return out


def executed_references(report_path):
    path=Path(report_path).parent.parent/'causal_references.pt'
    payload=torch.load(path,map_location='cpu',weights_only=False)
    records=payload['references'];plans={r['identity']['plan_id']:r['generated'] for r in records};result={}
    for record in records:
        p=record['provenance']
        if p is None:continue
        for step in p['steps']:
            names=step.get('consumed_plan_ids') or [p['plan_dictionary'][i] for i in step['consumed_plan_indices']]
            plan=plans.get(names[10])
            if plan is None:continue
            origin=p['plans'][names[10]]['decision_tick'];tick=step['control_tick_begin']
            support=source_frame_support(tick,origin)
            result[(record['task_id'],tick)]=interpolate(np.asarray(plan['qpos_world'],dtype=np.float64),support)
    return result


def reference_continuity(before,after,kin):
    """沿真实50Hz消费序列检查拼接后增量，覆盖参考银行重新接入前缀的边界。"""
    identities=sorted(set(before)&set(after));checks=[];limits=kin.joint_limits.numpy()
    for task in sorted({key[0] for key in identities}):
        keys=[key for key in identities if key[0]==task];ticks=np.asarray([key[1] for key in keys])
        a=np.stack([before[key] for key in keys]);b=np.stack([after[key] for key in keys])
        contiguous=np.diff(ticks)==12;second=contiguous[:-1]&contiguous[1:]
        bodies_a=kin.forward_body_positions(torch.tensor(a,dtype=torch.float32)).numpy()
        bodies_b=kin.forward_body_positions(torch.tensor(b,dtype=torch.float32)).numpy()
        if not np.isfinite(bodies_b).all():raise ValueError('Nonfinite executed FK')
        excess_a=np.maximum(limits[:,0]-a[:,7:],a[:,7:]-limits[:,1])
        excess_b=np.maximum(limits[:,0]-b[:,7:],b[:,7:]-limits[:,1])
        row=dict(task_id=task,controls=len(keys),new_limit_coordinate_count=int(((excess_b>1e-5)&(excess_a<=1e-5)).sum()),
            baseline_limit_coordinate_count=int((excess_a>1e-5).sum()),candidate_limit_coordinate_count=int((excess_b>1e-5).sum()),
            maximum_additional_limit_excess_rad=float(np.maximum(excess_b-np.maximum(excess_a,0),0).max()))
        for name,first,last in [('joint',a[:,7:],b[:,7:]),('root_position',a[:,:3],b[:,:3]),('body_position',bodies_a,bodies_b)]:
            va=np.diff(first,axis=0)/.02;vb=np.diff(last,axis=0)/.02
            row[name+'_velocity_baseline_max']=float(np.abs(va[contiguous]).max())
            row[name+'_velocity_candidate_max']=float(np.abs(vb[contiguous]).max())
            row[name+'_velocity_delta_max']=float(np.abs((vb-va)[contiguous]).max())
            row[name+'_acceleration_delta_max']=float(np.abs(np.diff(vb-va,axis=0)[second]/.02).max())
        checks.append(row)
    return dict(scope='matched_actual_50Hz_center_reference_including_plan_splices; derivatives_do_not_cross_gaps; all_body_FK_includes_feet',
        tasks=checks,new_limit_coordinate_count=sum(r['new_limit_coordinate_count'] for r in checks),
        maxima={key:max(r[key] for r in checks) for key in checks[0] if key.endswith('_max') or key=='maximum_additional_limit_excess_rad'})


def compare_one(work):
    rank,kind,base_path,new_path,output,leg_indices,kinematics_path=work;torch.set_num_threads(2)
    br,bp,before=load_report(base_path);nr,np_,after=load_report(new_path)
    if br['plan_sha256']!=nr['plan_sha256'] or [x['task_id'] for x in before]!=[x['task_id'] for x in after]:raise ValueError('Unpaired physics')
    rows=[];prefixes=[];pair_keys=[]
    for a,b in zip(before,after):
        if any(a[k]!=b[k] for k in ('dataset','sample_id','seed','music_start_frame','noise_index_start')):raise ValueError('Task initial state or noise differs')
        first,last=metrics(a),metrics(b)
        row=dict(task_id=a['task_id'],dataset=a['dataset'],sample_id=a['sample_id'],seed=a['seed'],initial=first,final=last)
        for key in ('reward_per_second','full_task_reward','seconds','failure','activity','music_alignment','completion'):
            row[key+'_delta']=last[key]-first[key];row[key+'_initial']=first[key];row[key+'_final']=last[key]
        rows.append(row);prefixes.append(matched_control_prefix(a,b));pair_keys.append({k:row[k] for k in ('task_id','dataset','sample_id','seed')})
    duration=sum(r['seconds'] for r in prefixes)
    tracking={k:{side:math.sqrt(sum(p['squared_error_integrals'][side][k] for p in prefixes)/duration) for side in ('initial','final')} for k in ERRORS}
    actual_a,actual_b=executed_references(base_path),executed_references(new_path)
    differences=defaultdict(list)
    for identity in sorted(set(actual_a)&set(actual_b)):
        a,b=actual_a[identity],actual_b[identity];delta=b-a
        angle=2*np.arccos(np.clip(abs(np.dot(a[3:7],b[3:7])),0,1))
        differences[identity[0]].append((float(np.sum(delta[:3]**2)),float(angle**2),float(np.mean(delta[leg_indices]**2))))
    refs=[]
    for row in rows:
        values=np.asarray(differences[row['task_id']]);r={k:row[k] for k in ('task_id','dataset','sample_id','seed')}
        if not len(values):raise ValueError('Missing actual consumed references')
        r.update(controls=len(values),root_position_rms_m=float(np.sqrt(values[:,0].mean())),root_angle_rms_rad=float(np.sqrt(values[:,1].mean())),leg_rms_rad=float(np.sqrt(values[:,2].mean())))
        refs.append(r)
    from gem.robots.bumi.kinematics import BumiKinematics
    continuity=reference_continuity(actual_a,actual_b,BumiKinematics(kinematics_path))
    result=dict(rank=rank,kind=kind,baseline=str(base_path),candidate=str(new_path),plan_sha256=br['plan_sha256'],
        cohort='four_sources_eight_music_two_seeds_fixed_train_music_exploratory_no_multiplicity_correction',
        metrics={field:cluster_summary(rows,field+'_delta') for field in ('reward_per_second','full_task_reward','seconds','failure','activity','music_alignment','completion')},
        baseline_metrics={field:cluster_summary(rows,field+'_initial') for field in ('reward_per_second','full_task_reward','failure','activity')},
        candidate_metrics={field:cluster_summary(rows,field+'_final') for field in ('reward_per_second','full_task_reward','failure','activity')},
        common_prefix_tracking_rms=tracking,tracking_confidence=tracking_distribution(pair_keys,prefixes),
        executed_reference_changes={k:cluster_summary(refs,k) for k in ('root_position_rms_m','root_angle_rms_rad','leg_rms_rad')},
        executed_reference_checks=continuity,
        reference_scope='actual_GMT_center_interpolation_from_world_plans; bootstrap_reference_excluded; no_counterfactual_after_failure',
        tasks=rows,reference_tasks=refs)
    atomic_json(Path(output)/f'{kind}_rank{rank:02d}.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--kind',choices=['A','C'],required=True);parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    import yaml
    config_path=next(args.run.glob('sessions/*/resolved_config.yaml'));config=yaml.safe_load(config_path.read_text())
    from gem.robots.bumi.kinematics import BumiKinematics
    kin=BumiKinematics(config['paths']['kinematics'])
    leg=[7+i for i,n in enumerate(kin.joint_order) if any(s in n.lower() for s in ('hip','knee','ankle'))]
    phase=config_path.parent/'phases';base='A_baseline' if args.kind=='A' else 'C_Stage1';work=[]
    labels=(['A_plus','A_minus','B_plus','B_minus','C_plus','C_minus','D_plus','D_minus'] if args.kind=='A' else
            ['C_height_plus','C_height_minus','C_yaw_plus','C_yaw_minus','C_leg_plus','C_leg_minus','C_smooth10','C_smooth20'])
    for rank,label in enumerate(labels):
        before=phase/f'decisive_{base}'/f'rank{rank:02d}'/'report/report.json'
        after=phase/f'decisive_{label}'/f'rank{rank:02d}'/'report/report.json'
        if not before.exists() or not after.exists():raise FileNotFoundError(f'Incomplete paired phase: {before}, {after}')
        work.append((rank,args.kind,before,after,args.output_dir,leg,config['paths']['kinematics']))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:results=list(pool.map(compare_one,work))
    summary=dict(kind=args.kind,candidates=[dict(label=label,**{k:r[k] for k in ('rank','metrics','baseline_metrics','candidate_metrics','common_prefix_tracking_rms','executed_reference_changes','executed_reference_checks')}) for label,r in zip(labels,results)])
    task_ids=[r['task_id'] for r in results[0]['tasks']]
    spreads=[]
    for i,task in enumerate(task_ids):
        values=[r['tasks'][i]['reward_per_second_initial'] for r in results]
        spreads.append(max(values)-min(values))
    summary['repeated_baseline_max_reward_rate_spread']=max(spreads)
    if args.kind=='A':
        derivatives=[]
        for i,label in enumerate(('A','B','C','D')):
            positive,negative=results[2*i:2*i+2];rows=[]
            for a,b in zip(positive['tasks'],negative['tasks']):
                rows.append(dict(task_id=a['task_id'],dataset=a['dataset'],sample_id=a['sample_id'],seed=a['seed'],
                    symmetric_rate_gain=(a['reward_per_second_delta']-b['reward_per_second_delta'])/2))
            derivatives.append(dict(direction=label,**cluster_summary(rows,'symmetric_rate_gain'),tasks=rows))
        summary['symmetric_directional_gain']=derivatives
        contrasts={}
        for left,right in ((0,1),(0,2),(3,0),(3,1),(3,2)):
            rows=[{**a,'contrast':a['symmetric_rate_gain']-b['symmetric_rate_gain']} for a,b in zip(derivatives[left]['tasks'],derivatives[right]['tasks'])]
            contrasts[derivatives[left]['direction']+'_minus_'+derivatives[right]['direction']]=cluster_summary(rows,'contrast')
        summary['direction_contrasts']=contrasts
    else:
        summary['legality']=[json.loads((args.run/'decisive'/f'results_rank{rank:02d}.json').read_text())['reference_audit'] for rank in range(8)]
    atomic_json(args.output_dir/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='candidates'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
