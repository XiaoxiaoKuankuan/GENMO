"""从已保存的八卡计算与真实物理证据重建原DPPO三项疑问报告。

读取固定rollout目标、裁剪/学习率对照、正式历史和独立整曲评价；按已发布SHA
校验真实消费数组，核对实际评价Actor指纹及任务/物理身份，保留失败任务。
完整链替代目标明确标为反事实诊断，不冒充当前生产目标。物理对照同时报告
任务回报、每秒回报、失败/完成数和执行时长；消费参考仅在真实共同控制时刻
对齐，状态分叉后不宣称同状态因果比较。输出写在独立诊断目录，不选择模型、
启动训练或改写正式记录，缺少证据时立即报错而非补零或排除失败样本。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from tools.stage2_dppo_diagnostic_math import summarize_records
from gem.closedloop.dppo.optimization_support import balanced_epoch_order


def read(path):return json.loads(Path(path).read_text())


def available_mean(values):
    values=[v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def joined_objectives(payload):
    results=[]
    advantage=payload['advantages'].double()
    for index,record in enumerate([payload['initial'],*payload['after_steps']]):
        log_ratio=record[...,0].sum(1).double();ratio=log_ratio.exp()
        if not torch.isfinite(ratio).all():raise FloatingPointError('完整链诊断溢出')
        results.append(dict(step=index,log_ratio_min=float(log_ratio.min()),log_ratio_max=float(log_ratio.max()),
            unclipped=float((ratio*advantage).mean()),
            clipped={str(c):float(torch.minimum(ratio*advantage,ratio.clamp(1-c,1+c)*advantage).mean())
                     for c in [.01,.2,.5]}))
    return dict(scope='counterfactual_joint_chain_objective_not_production_loss',steps=results)


def actual_minibatches(case,summary,probabilities):
    counts=[read(case/f'rank{rank:02d}/manifest.json')['real_upper_transitions'] for rank in range(8)]
    total=sum(counts)
    if total!=len(probabilities['advantages']):
        raise ValueError('轻量小批重建仅接受本次所有真实行均有效的case，不能假设缺失有效Mask')
    manifest=[dict(owner_rank=rank,local_index=i,valid=True,has_free=True)
              for rank,n in enumerate(counts) for i in range(n)]
    state=torch.load(case/'rank00/rank_state.pt',map_location='cpu',weights_only=False)
    generator=torch.Generator();generator.set_state(state['rng']['generators']['actor'])
    order=balanced_epoch_order(list(range(total)),manifest,generator)
    records=[probabilities['initial'],*probabilities['after_steps']]
    results=[];start=0
    for step,entry in enumerate(summary['optimizer']['steps']):
        indices=order[start:start+entry['global_chain_count']];start+=len(indices)
        actual=summarize_records(records[step][indices],probabilities['advantages'][indices],clip=summary['clip'])
        difference=-actual['production_objective']-entry['ppo_loss']
        if abs(difference)>1e-8 or abs(actual['active_clip_fraction']-entry['objective_clip_fraction'])>1e-12:
            raise ValueError('重建的小批PPO目标或裁剪不能复现原更新器')
        proposed={}
        for clip in [.01,.005,.003,.001,.0005,.0003]:
            s=summarize_records(records[step][indices],probabilities['advantages'][indices],clip=clip)
            proposed[str(clip)]=dict(active_clip_fraction=s['active_clip_fraction'],
                                    ordinary_clip_fraction=s['ordinary_clip_fraction'])
        results.append(dict(step=step+1,global_chains=len(indices),production_loss_difference=difference,
                            hypothetical_clip_on_same_actual_pre_step_minibatch=proposed))
    if start!=total:raise ValueError('本次单Epoch重建没有覆盖全部真实训练样本')
    return dict(scope='exact_preserved_RNG_and_original_balanced_order_reproduces_logged_PPO_loss_not_an_extra_update',steps=results)


def current_consumption_delta(x,y,ia,ib):
    """以前一真实控制结束的同计划/同tick中心参考核验本次GMT当前输入。"""
    mask=(ia>0)&(ib>0)
    ia,ib=ia[mask],ib[mask]
    if not len(ia):return dict(verified_controls=0,rms_rad=None)
    px=x['gmt_window_plan_ids'][ia,10];py=y['gmt_window_plan_ids'][ib,10]
    mask=(x['reward_reference_tick'][ia-1]==x['control_begin_tick'][ia])
    mask &= (y['reward_reference_tick'][ib-1]==y['control_begin_tick'][ib])
    mask &= (x['reward_center_plan_id'][ia-1]==px)&(y['reward_center_plan_id'][ib-1]==py)
    mask &= np.asarray([not str(v).startswith('bootstrap:') for v in px])
    mask &= np.asarray([not str(v).startswith('bootstrap:') for v in py])
    delta=y['reward_center_joint_pos'][ib[mask]-1]-x['reward_center_joint_pos'][ia[mask]-1]
    return dict(verified_controls=int(mask.sum()),rms_rad=float(np.sqrt(np.mean(delta**2))) if len(delta) else None,
        proof='same_immutable_plan_and_exact_reward_end_tick_equals_next_GMT_current_input_tick')


def physics(root,expected_fingerprint):
    report=read(root/'evaluation_baseline.json')
    summaries=list((root/'sessions').glob('*/summary.json'))
    if len(summaries)!=1:raise ValueError('物理评价必须是唯一独立会话')
    summary=read(summaries[0])
    if summary['status']!='passed' or summary['exit_code']!=0 or not summary['source_unchanged']:
        raise ValueError('物理评价未正常结束或来源变化')
    if report['actor_identity']['model_fingerprint']!=expected_fingerprint:
        raise ValueError('实际评价模型被覆盖或指纹不匹配')
    consume={}
    for path in (root/'consumed').glob('rank*/*.json'):
        header=read(path)
        if header['task_id'] in consume:raise ValueError('消费证据任务重复')
        target=path.parent/header['arrays']
        if hashlib.sha256(target.read_bytes()).hexdigest()!=header['arrays_sha256']:
            raise ValueError('消费数组SHA不匹配')
        consume[header['task_id']]=dict(header=header,path=target)
    tasks={row['task_id']:row for row in report['episode_metrics']}
    if set(consume)!=set(tasks) or len(tasks)!=report['task_count']:raise ValueError('消费证据未覆盖全部任务')
    aggregate=report['aggregate']['overall']
    balanced_rate=float(np.mean([group['reward_per_executed_second']
                               for group in report['aggregate']['by_source'].values()]))
    result=dict(tasks=report['task_count'],completed=aggregate['music_completion_count'],
        failures=aggregate['physical_failure_count'],duration=aggregate['executed_seconds'],
        administrative_truncations=aggregate['administrative_truncation_count'],
        infrastructure_failures=aggregate['infrastructure_failure_count'],
        source_balanced_task_return=report['source_balanced_reward'],source_balanced_return_per_second=balanced_rate,
        total_return=aggregate['reward_sum'],raw_errors=aggregate['raw_errors'],
        tracking=aggregate['components'],actual_actor_fingerprint=expected_fingerprint,
        consumed_controls=sum(c['header']['controls'] for c in consume.values()))
    return report,tasks,consume,result


def paired(before,after):
    base,base_tasks,base_consume,_=before
    candidate,tasks,consume,_=after
    if base['plan']!=candidate['plan']:raise ValueError('物理对照任务不一致')
    a=base['evaluation_identity'];b=candidate['evaluation_identity']
    for key in ['timing_contract','kernel','execution_profile','critical_latency_budgets','episode_seconds','physics']:
        if a.get(key)!=b.get(key):raise ValueError('物理/数值执行身份改变: '+key)
    for key in ['assets','reward','timing','physical_configuration','source_files','generation_horizon','data_catalog']:
        if a['training_identity'][key]!=b['training_identity'][key]:raise ValueError('物理训练身份改变: '+key)
    samples=[]
    for task_id,row in tasks.items():
        original=base_tasks[task_id]
        with np.load(base_consume[task_id]['path'],allow_pickle=False) as x,np.load(consume[task_id]['path'],allow_pickle=False) as y:
            ticks,ia,ib=np.intersect1d(x['control_begin_tick'],y['control_begin_tick'],return_indices=True)
            if not len(ticks):raise ValueError('物理轨迹没有共同执行控制时刻')
            current_x=x['gmt_window_plan_ids'][ia,10];current_y=y['gmt_window_plan_ids'][ib,10]
            mask=np.asarray([not str(v).startswith('bootstrap:') for v in current_x])
            mask &= np.asarray([not str(v).startswith('bootstrap:') for v in current_y])
            delta=y['reward_center_joint_pos'][ib]-x['reward_center_joint_pos'][ia]
            current=current_consumption_delta(x,y,ia,ib)
            task=next(t for t in base['plan']['tasks'] if t['task_id']==task_id)
            samples.append(dict(task_id=task_id,source=row['dataset'],sample_id=task['sample_id'],group_id=task['sample']['group_id'],
                seed=row['seed'],return_delta=row['reward']-original['reward'],
                candidate_failure=row['physical_failure'],baseline_failure=original['physical_failure'],
                duration_delta=row['executed_seconds']-original['executed_seconds'],common_controls=len(ticks),
                common_generated_current_controls=int(mask.sum()),
                actual_gmt_current_reference_delta=current,
                actual_reward_center_joint_delta_rms_rad=float(np.sqrt(np.mean(delta[mask]**2))) if mask.any() else None,
                first_control_actual_joint_max_difference=float(np.max(np.abs(x['actual_joint_pos'][0]-y['actual_joint_pos'][0])))))
    sources=sorted({s['source'] for s in samples});groups={}
    for source in sources:
        song_names=sorted({s['group_id'] for s in samples if s['source']==source})
        groups[source]=[np.mean([s['return_delta'] for s in samples if s['source']==source and s['group_id']==song]) for song in song_names]
    rng=np.random.default_rng(20261022)
    bootstrap=np.mean([rng.choice(values,size=(10000,len(values)),replace=True).mean(1) for values in groups.values()],axis=0)
    return dict(source_balanced_paired_task_return_delta=float(np.mean([np.mean(v) for v in groups.values()])),
        paired_95_interval=np.quantile(bootstrap,[.025,.975]).tolist(),
        cluster_scope='source_balanced_music_group_clusters_fixed_training_music_not_generalization_claim',
        newly_failed_tasks=[s['task_id'] for s in samples if s['candidate_failure'] and not s['baseline_failure']],
        reference_change_scope='real_GMT_current_verified_by_immutable_plan_and_tick_plus_reward_center_after_state_divergence_not_same_state_causal_comparison',
        initial_actual_joint_max_difference=max(s['first_control_actual_joint_max_difference'] for s in samples),
        actual_reward_center_joint_delta_rms_rad_mean=available_mean([s['actual_reward_center_joint_delta_rms_rad'] for s in samples]),
        actual_gmt_current_reference_delta_rms_rad_mean=available_mean([s['actual_gmt_current_reference_delta']['rms_rad'] for s in samples]),
        actual_gmt_current_reference_verified_controls=sum(s['actual_gmt_current_reference_delta']['verified_controls'] for s in samples),
        samples=samples)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve(strict=True)
    replay=read(root/'replay/summary.json');origin=read(root/'origin.json')
    if replay['source_checkpoint_sha256']!=origin['checkpoint_sha']:raise ValueError('计算验证来源不是已停止正式模型')
    if hashlib.sha256((Path(replay['case'])/'case.json').read_bytes()).hexdigest()!=replay['case_sha256']:
        raise ValueError('固定真实case身份被改写')
    report=dict(origin=origin,history=read(root/'history.json'),replay=replay,clips={},
                production_preservation=read(root/'production_preservation.json'),input_probability_sha256={})
    if (root/'baseline_repeatability.json').exists():
        report['baseline_repeatability']=read(root/'baseline_repeatability.json')
    for tag in ['clip_0_01','clip_0_005','clip_0_003']:
        folder=root/'replay'/tag;summary=read(folder/'summary.json')
        probabilities=torch.load(folder/'probability_records.pt',map_location='cpu',weights_only=False)
        report['input_probability_sha256'][tag]=hashlib.sha256((folder/'probability_records.pt').read_bytes()).hexdigest()
        report['clips'][tag]=dict(summary=summary,joint_counterfactual=joined_objectives(probabilities),
            common_clip_001_objective=summarize_records(probabilities['final'],probabilities['advantages'],clip=.01)['production_objective'],
            actual_minibatch_audit=actual_minibatches(Path(replay['case']),summary,probabilities),
            all_actual_full_objective_steps_positive=all(s['actual_full_objective_gain']>0 for s in summary['step_history']),
            free_coordinate_counts=torch.unique(probabilities['final'][...,2]).tolist())
    report['shifted_clip_gradients']=read(root/'replay/clip_0_01/same_shifted_model_clip_gradients.json')
    report['lr_sweep']=read(root/'replay/lr_sweep.json')
    queue=read(root/'physics_queue.json')
    if queue['status']!='passed':raise ValueError('有限物理对照尚未全部完成')
    report['physics_queue']=queue;report['physics']={};report['paired_physics']={}
    higher=root/'replay_lr2/clip_0_01/summary.json'
    if higher.exists():report['higher_lr_full_round']=read(higher)
    baseline=None
    for run in queue['runs']:
        tag=run['tag']
        if tag=='baseline':expected=read(root/'capture/latest.json')['metadata']['actor_model_fingerprint']
        elif tag=='lr2':expected=report['higher_lr_full_round']['actor_fingerprint']
        else:expected=report['clips'][tag]['summary']['actor_fingerprint']
        data=physics(Path(run['output']),expected)
        report['physics'][tag]=data[3]
        if tag=='baseline':baseline=data
        else:report['paired_physics'][tag]=paired(baseline,data)
    output=root/'verification_report.json';temporary=output.with_suffix('.tmp')
    temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2));temporary.replace(output)
    print(json.dumps(dict(report=str(output),
        physics={k:{name:value for name,value in v.items() if name not in ['raw_errors','tracking']} for k,v in report['physics'].items()},
        paired_physics={k:{name:value for name,value in v.items() if name!='samples'} for k,v in report['paired_physics'].items()}),
        ensure_ascii=False,indent=2))


if __name__=='__main__':main()
