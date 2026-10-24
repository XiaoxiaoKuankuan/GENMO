"""只读诊断B：完整物理MC、局部Critic误差及连续参考的实际延迟信用。

读取已完成评估的SHA绑定控制档案和同次保存的因果Critic预测，独立从50Hz积分
奖励反向计算无自举MC；行政截断单独列出，不以零尾值冒充完整回报。按大跟踪误差、
参考切换、失败前及100秒后分组，同时重建8秒片段的GAE和lambda=1自举误差。
来源账本沿已知前缀追到原始生成计划，统计本转移奖励实际使用新/旧参考的比例，
并导出生成、到达、提交、前瞻、中心消费的逐计划时间。工具不移动奖励、不训练
网络、不把执行后状态作为Actor目标。所有结果进入明确的新诊断目录。
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gem.closedloop.dppo.budget import atomic_json
from gem.closedloop.dppo.compact_evidence import report_episodes
from gem.closedloop.dppo.returns import compute_gae
from gem.closedloop.dppo.execution_provenance import source_frame_support
from gem.closedloop.robofft.execution_weights import merged_consumption


def key(identity):return json.dumps(identity,sort_keys=True,separators=(',',':'))


def discounted_complete(decisions,gamma_upper):
    """无模型、无bootstrap，独立逐控制步递推；积分奖励不重复乘dt。"""
    gamma=gamma_upper**(1/25);result=[0.]*len(decisions);value=0.;previous=None
    for d in decisions:
        if len(d['controls'])!=d['executed_control_steps']:raise ValueError('Control count mismatch')
        for control in d['controls']:
            if previous is not None and control['tick']!=previous+12:raise ValueError('Noncontiguous physical controls')
            if not math.isclose(control['dt_s'],.02,abs_tol=1e-12):raise ValueError('Incorrect control dt')
            previous=control['tick']
    for i in range(len(decisions)-1,-1,-1):
        d=decisions[i];controls=d['controls']
        immediate=sum(gamma**j*float(r['reward']) for j,r in enumerate(controls))
        immediate+=gamma**max(len(controls)-1,0)*d['zero_step_event_reward']
        value=immediate+gamma**len(controls)*value;result[i]=value
    return np.asarray(result)


def group_summary(rows):
    if not rows:return dict(states=0,tasks=0)
    result=dict(states=len(rows),tasks=len({r['task_id'] for r in rows}))
    for metric in ('value_error','gae_error','bootstrap_error'):
        x=np.asarray([r[metric] for r in rows]);result[metric]=dict(bias=float(x.mean()),rmse=float(np.sqrt(np.mean(x*x))),
            p05=float(np.quantile(x,.05)),p95=float(np.quantile(x,.95)),maximum=float(x.max()))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--behavior-root',type=Path)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    report=json.loads((args.run/'evaluation_baseline.json').read_text())
    reference=json.loads((args.run/'fixed_diagnostic_reference.json').read_text())
    if report['plan_sha256']!=reference['plan_sha256']:raise ValueError('Plan differs')
    predictions={};source_files=[]
    for rank in reference['ranks']:
        descriptor=rank.get('critic')
        if descriptor is None:continue
        path=Path(descriptor['path']);sha=hashlib.sha256(path.read_bytes()).hexdigest()
        if sha!=descriptor['sha256']:raise ValueError('Causal prediction reference SHA differs')
        data=torch.load(path,map_location='cpu',weights_only=False)
        if data['reference_sha256']!=descriptor['reference_sha256']:raise ValueError('Causal condition binding differs')
        source_files.append(dict(path=str(path),sha256=sha))
        for row in data['records']:
            identity=key(row['identity'])
            if identity in predictions:raise ValueError('Repeated value identity')
            predictions[identity]=dict(prediction=float(row['initial_prediction']),begin=row['control_tick_begin'],end=row['control_tick_end'],
                known_frames=int(row['critic_context']['known_qpos30_mask'].any(-1).sum()),remaining=row['remaining_music_seconds'])
        del data
    gamma=float(report['evaluation_identity']['training_identity']['training']['gamma_upper'])
    lam=float(report['evaluation_identity']['training_identity']['training']['lambda_upper'])
    states=[];plans=[];episodes=[];excluded=[];new_credit=old_credit=unresolved=0.
    for episode in report_episodes(report):
        decisions=episode['decisions'];task=episode['task_id']
        print('MC/provenance',episode['dataset'],episode['sample_id'],episode['seed'],flush=True)
        wrappers=[SimpleNamespace(identity=d['identity'],metadata=dict(compact_execution_provenance=d.get('compact_execution_provenance'))) for d in decisions]
        consumption=merged_consumption(wrappers)
        plan_map={name:item for record,_ in consumption.values() for name,item in record.get('plans',{}).items()}
        def origin(plan,frame,depth=0):
            if plan.startswith('bootstrap:'):return 'bootstrap'
            if depth>1000:return None
            item=plan_map.get(plan)
            if item is None or item.get('prefix_frames') is None:return None
            if frame>=item['prefix_frames']:return plan
            parent=item.get('parent_plan_id')
            if not parent:return None
            if parent.startswith('bootstrap:'):return 'bootstrap'
            previous=plan_map.get(parent)
            if previous is None:return None
            absolute=item['decision_tick']+frame*20;index,remainder=divmod(absolute-previous['decision_tick'],20)
            if remainder:return None
            return origin(parent,index,depth+1)
        for name,item in plan_map.items():
            record=dict(task_id=task,dataset=episode['dataset'],sample_id=episode['sample_id'],seed=episode['seed'],plan_id=name)
            for field in ('decision_tick','generation_ready_tick','commit_tick','first_window_tick','first_current_tick',
                          'first_generated_window_tick','first_generated_current_tick','last_generated_current_tick','prefix_frames','parent_plan_id'):
                record[field]=item.get(field)
            for field in ('generation_ready_tick','commit_tick','first_generated_window_tick','first_generated_current_tick'):
                record[field+'_delay_seconds']=None if item.get(field) is None else (item[field]-item['decision_tick'])/600
            record['center_generated_frames']=sum(x>0 for x in item['current_generated_frame_weight'])
            record['window_generated_frames']=sum(x>0 for x in item['window_generated_frame_weight'])
            plans.append(record)
        per_credit=[];switch=[]
        for decision in decisions:
            own=decision['identity']['plan_id'];p=decision.get('compact_execution_provenance') or {};own_mass=total=0.;changed=False
            for step in p.get('steps',[]):
                plan=step['reference_plan_id'];total+=1
                if plan.startswith('bootstrap:'):old_credit+=1;continue
                item=plan_map.get(plan)
                if item is None:unresolved+=1;continue
                for frame,weight in source_frame_support(step['reference_tick'],item['decision_tick']):
                    source=origin(plan,frame)
                    if source==own:own_mass+=weight;new_credit+=weight
                    elif source is None:unresolved+=weight
                    else:old_credit+=weight
                dictionary=p.get('plan_dictionary',[])
                consumed=step.get('consumed_plan_ids') or [dictionary[i] for i in step['consumed_plan_indices']]
                changed |= len(set(consumed))>1
            per_credit.append(own_mass/max(total,1));switch.append(changed)
        complete=bool(decisions and decisions[-1]['terminated'] and not decisions[-1]['truncated'] and not episode['truncated'])
        if not complete:
            excluded.append(dict(task_id=task,reason=episode['end_reason'],scope='excluded_from_complete_MC_only'));continue
        target=discounted_complete(decisions,gamma)
        saved=[predictions[key(d['identity'])] for d in decisions]
        value=np.array([x['prediction'] for x in saved]);n=len(decisions)
        gae=np.empty(n);boot=np.empty(n);cuts=[];begin=0
        for end in range(1,n+1):
            if end==n or saved[end-1]['end']-saved[begin]['begin']>=4800:
                part=decisions[begin:end]
                kwargs=dict(rewards=[[c['reward'] for c in d['controls']] for d in part],values=value[begin:end],
                    next_values=[value[j+1] if j+1<n else 0. for j in range(begin,end)],
                    executed_steps=[d['executed_control_steps'] for d in part],bootstrap_mask=[not d['terminated'] for d in part],
                    continuation_mask=[j+1<len(part) for j in range(len(part))],
                    event_rewards=[d['zero_step_event_reward'] for d in part],gamma_upper=gamma,normalize=False)
                gae[begin:end]=compute_gae(**kwargs,lambda_upper=lam)['returns'].numpy()
                boot[begin:end]=compute_gae(**kwargs,lambda_upper=1.)['returns'].numpy()
                if end<n:cuts.append(saved[end-1]['end'])
                begin=end
        final=saved[-1]['end']
        for i,d in enumerate(decisions):
            errors=[c['raw_errors'] for c in d['controls']]
            root=max((math.sqrt(r['anchor_pos_squared_error']) for r in errors),default=0.)
            orientation=max((math.sqrt(r['anchor_ori_squared_error']) for r in errors),default=0.)
            time_s=(saved[i]['begin']-600)/600+episode['music_start_frame']/30
            states.append(dict(task_id=task,dataset=episode['dataset'],sample_id=episode['sample_id'],seed=episode['seed'],
                decision=i,music_time_seconds=time_s,physical_time_seconds=(saved[i]['begin']-600)/600,
                prediction=float(value[i]),complete_mc=float(target[i]),gae=float(gae[i]),lambda1=float(boot[i]),
                value_error=float(value[i]-target[i]),gae_error=float(gae[i]-target[i]),bootstrap_error=float(boot[i]-target[i]),
                root_error=root,orientation_error=orientation,reference_switch=switch[i],new_reference_reward_fraction=per_credit[i],
                before_failure=episode['physical_failure'] and 0<=final-saved[i]['begin']<=5*600,
                before_cut=any(0<t-saved[i]['begin']<=600 for t in cuts),after_100_seconds=time_s>=100,
                known_reference_frames=saved[i]['known_frames'],physical_failure=episode['physical_failure']))
        episodes.append(dict(task_id=task,dataset=episode['dataset'],sample_id=episode['sample_id'],seed=episode['seed'],
            failure=episode['physical_failure'],end_reason=episode['end_reason'],states=n,mc_initial=float(target[0]),
            prediction_initial=float(value[0]),bias=float((value-target).mean()),rmse=float(np.sqrt(np.mean((value-target)**2)))))
    threshold=float(np.quantile([r['root_error'] for r in states],.9))
    groups={'all':states,'large_root_error_top10pct':[r for r in states if r['root_error']>=threshold]}
    for field in ('reference_switch','before_failure','before_cut','after_100_seconds'):
        groups[field]=[r for r in states if r[field]]
    summary=dict(status='passed',source_run=str(args.run),actor_identity=report['actor_identity'],source_files=source_files,
        critic_prediction_scope='SHA_bound_same_evaluation_frozen_Critic_initial_prediction_no_refit',
        target_scope='independent_complete_control_reward_MC_no_bootstrap',
        gamma_upper=gamma,lambda_upper=lam,discount={str(s):gamma**(s/.5) for s in (.3,.5,1,5,10,20,100)},
        half_life_seconds=.5*math.log(.5)/math.log(gamma),groups={k:group_summary(v) for k,v in groups.items()},
        episodes=episodes,excluded_administrative=excluded,large_error_threshold=threshold,
        reward_reference_credit=dict(current_generated_mass=new_credit,previous_generated_mass=old_credit,unresolved=unresolved,
            current_generated_fraction=new_credit/max(new_credit+old_credit+unresolved,1)),
        critic_observability=dict(known_prefix=True,tracking_history_seconds=1.,music_remaining=True,
            full_committed_future_buffer=False,pending_plan_schedule=False,explicit_reference_source_ledger=False),
        inference='partial_reference_state_can_alias_future_returns; MC_errors_alone_do_not_prove_causal_action_ranking')
    for field in ('generation_ready_tick_delay_seconds','commit_tick_delay_seconds','first_generated_window_tick_delay_seconds','first_generated_current_tick_delay_seconds'):
        x=[r[field] for r in plans if r[field] is not None]
        summary.setdefault('delays',{})[field]=dict(count=len(x),mean=float(np.mean(x)),p50=float(np.median(x)),p95=float(np.quantile(x,.95)))
    if args.behavior_root:
        coverage=[]
        for path in sorted(args.behavior_root.glob('behavior_rank*.pt')):
            data=torch.load(path,map_location='cpu',weights_only=False)
            for row in data['rows']:
                metadata=row.metadata
                coverage.append(dict(rank=path.stem,start_seconds=metadata['music_start_frame']/30,
                    absolute_music_seconds=metadata['music_start_frame']/30+metadata['physical_episode_elapsed_seconds'],
                    episode_elapsed_seconds=metadata['physical_episode_elapsed_seconds'],
                    full_music_seconds=metadata['full_music_num_frames']/30))
        summary['sampled_training_coverage']=dict(states=len(coverage),after100=sum(r['absolute_music_seconds']>=100 for r in coverage),
            largest_history_seconds=max((r['episode_elapsed_seconds'] for r in coverage),default=0.),
            distinction='random_late_music_phase_reset_does_not_reproduce_100s_accumulated_root_drift')
        atomic_json(args.output_dir/'training_coverage.json',coverage)
    for name,records in [('states',states),('plans',plans)]:
        with (args.output_dir/(name+'.jsonl')).open('w') as f:
            for row in records:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    atomic_json(args.output_dir/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ('groups','reward_reference_credit','delays','discount')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
