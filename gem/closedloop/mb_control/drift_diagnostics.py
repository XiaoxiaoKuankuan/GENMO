"""音乐舞蹈持续漂移的真实证据重算。

输入正式运行的不可变完整评价、逐控制步缓存和训练日志。A目标严格复现每个真实
首次影响anchor的一次SE(2)对齐H16积分；B只在episode初态对齐一次后比较整曲；
C保留世界误差，D复用按真实关节名称计算的内容/速度相位。并重建新参考进入
当前目标、未来goal与后续覆盖的时序，不能把计划长度当实际训练覆盖。
仅CPU读取与计算；输出JSON/JSONL，不写源运行、不运行优化器、不改变教师缓存。
"""
from collections import Counter,defaultdict
from dataclasses import replace
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch


def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))


def stats(values):
    x=np.asarray([v for v in values if v is not None],dtype=float)
    return {'n':len(x),'mean':float(x.mean()) if len(x) else None,
        'min':float(x.min()) if len(x) else None,'max':float(x.max()) if len(x) else None,
        'p50':float(np.quantile(x,.5)) if len(x) else None,'p95':float(np.quantile(x,.95)) if len(x) else None}


def sample_counts(rows):
    return {'count':len(rows),'unique_requests':len({(r.get('collector'),r.get('episode'),r.get('task'),r.get('request_tick')) for r in rows}),
        **{k:dict(Counter(r.get(k,'unknown') for r in rows)) for k in ['task','source','speed_bin','motion_class','contact','amplitude_bin']},
        **{k:stats([r.get(k) for r in rows]) for k in ['speed_rms_rad_s','root_speed_m_s','leg_speed_rad_s','joint_range_rad']},
        'contact_transition_samples':sum(r.get('contact_changes',0)>0 for r in rows)}


def audit_logs(run,output):
    candidates=[];gradients=defaultdict(list);effects=defaultdict(list);funnel=defaultdict(list)
    with (run/'training.jsonl').open() as stream:
        for line in stream:
            d=json.loads(line)
            if 'output_trust_region' not in d:continue
            if d['accepted_actor_updates']>700:break
            for r in d.get('sample_distribution',[]):funnel['E_effective' if r.get('accepted') else 'D_drawn_rejected'].append(r)
            for a in d['output_trust_region']['attempts']:
                m=a.get('metrics',{});before=m.get('losses_before',{});after=m.get('losses_after',{})
                row={'attempted_update':d['update'],'accepted_actor_updates':d['accepted_actor_updates'],
                    'candidate_attempt':a['attempt'],'accepted':a['accepted'],'reasons':a['reasons'],
                    'loss_before':before,'loss_after':after,'outputs':m.get('outputs',[]),
                    'actual_parameter_delta_l2':a.get('parameter_delta_l2'),
                    'raw_gradient_dot_actual_parameter_delta':a.get('raw_gradient_dot_actual_parameter_delta',{}),
                    'weighted_gradient_dot_actual_parameter_delta':a.get('weighted_gradient_dot_actual_parameter_delta',{}),
                    'samples':d.get('sample_distribution',[])}
                if 'dance' in before and 'dance' in after:
                    row.update(dance_delta=after['dance']-before['dance'],
                        allowed_increase=1e-5+.005*abs(before['dance']))
                candidates.append(row)
                if a['accepted']:
                    for k,v in row['weighted_gradient_dot_actual_parameter_delta'].items():effects[k].append(v)
            if d.get('accepted_execution_feedback_update'):
                for module,terms in d['module_gradient_metrics'].items():
                    for term,v in terms.items():gradients[module+'/'+term].append(v)
    with (output/'candidate_history.jsonl').open('w') as f:
        for r in candidates:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    accepted=[r for r in candidates if r['accepted']]
    worse=[r for r in accepted if r.get('dance_delta',0)>0]
    write_json(output/'accepted_dance_worsening.json',worse)
    for line in (run/'sampling_funnel.jsonl').open():
        d=json.loads(line)
        if d['actor_version']>=700:break
        funnel['A_all_real_requests'].extend(d.get('A_all_real_requests',[]))
        funnel['B_anchor_outcomes'].extend(d.get('B_anchor_outcomes',[]))
        funnel['B_formed_H_anchor'].extend(r for r in d.get('B_anchor_outcomes',[]) if r.get('stage')=='B_formed_H_anchor')
        funnel['B_legal_H_anchor'].extend(r for r in d.get('BC_support_classification',[]) if r.get('legal') is True)
        funnel['C_outside_support'].extend(r for r in d.get('BC_support_classification',[]) if r.get('legal') is False)
        # Support records retain actual offending component/tick; do not relabel these as legal.
        funnel['BC_support_classification'].extend(d.get('BC_support_classification',[]))
    funnel['D_drawn']=funnel['E_effective']+funnel['D_drawn_rejected']
    support_failures=funnel['C_outside_support']
    write_json(output/'support_failures.json',{'components':dict(Counter(c['name'] for r in support_failures for c in (r.get('support_details') or {}).get('components',[]))),
        'prediction_steps':dict(Counter((r.get('support_details') or {}).get('prediction_step') for r in support_failures)),
        'by_speed':{k:sample_counts([r for r in support_failures if r.get('speed_bin')==k]) for k in ['low','medium','high']}})
    report={'candidate_count':len(candidates),'accepted':len(accepted),'accepted_dance_worse':len(worse),
        'accepted_worsening_delta':stats([r['dance_delta'] for r in worse]),
        'module_gradients':{k:{field:stats([v[field] for v in vs]) for field in ['raw_norm','weighted_norm','cosine_with_dance']} for k,vs in gradients.items()},
        'actual_weighted_gradient_dot_adamw_delta':{k:{**stats(v),'descent':sum(x<0 for x in v),'ascent':sum(x>0 for x in v)} for k,v in effects.items()},
        'funnel':{k:sample_counts(v) for k,v in funnel.items()},
        'next_online_rollout_counterfactual':'not_in_optimizer_logs; requires independent replay, not inferred from next unpaired minibatch'}
    report['accepted_output_statistics']={group:{key:stats([o.get(group,{}).get(key) for r in accepted for o in r['outputs']])
        for key in (['joint_rms_rad','root_rms_m','orientation_rms_rad'] if group in ['step','local','global','global_absolute_unaligned'] else ['range_ratio','speed_ratio','velocity_phase_cosine'])}
        for group in ['step','local','global','global_absolute_unaligned','global_activity','local_activity']}
    write_json(output/'log_audit.json',report)
    return report


def anchor_coverage(sample):
    version=sample.generation.reference_version;future_only=0;center=0;support_sources=set();goal_steps=0
    exact_center_steps=0;exact_center_points=set();first_exact_center=None;current_halo_only=0
    for step in sample.transitions:
        ref=step.start.reference;t=step.start.tick
        def selected(support):return support[ref.source_versions[(support-ref.lower)//20]==version]
        goal=selected(ref.dependency_support(t));current=selected(ref.dependency_support(t,future_steps=0))
        goal_steps+=bool(len(goal));future_only+=bool(len(goal)) and not bool(len(current))
        center+=bool(len(current));support_sources.update(current.tolist())
        lo,hi,fraction=ref._indices(ref.ticks.new_tensor([t]))
        indices=torch.unique(torch.cat((lo,hi[fraction!=0])))
        exact=ref.ticks[indices[ref.source_versions[indices]==version]]
        exact_center_steps+=bool(len(exact));exact_center_points.update(exact.tolist())
        current_halo_only+=bool(len(current)) and not bool(len(exact))
        if len(exact) and first_exact_center is None:first_exact_center=t
    ticket=sample.generation.request.ticket
    initial=sample.transitions[0].start.reference;final=sample.transitions[-1].end.reference
    owned=initial.ticks[initial.source_versions==version]
    still=set(final.ticks[final.source_versions==version].tolist())
    common=[int(t) for t in owned if final.lower<=int(t)<=final.upper]
    overwritten=sum(t not in still for t in common)
    return {'request_tick':ticket.request_tick,'anchor_tick':sample.state.tick,
        'first_influence_tick':ticket.first_influence_tick,'first_mutable_source_tick':ticket.first_mutable_source_tick,
        'h16_goal_influence_steps':goal_steps,'h16_current_reference_support_steps':center,
        'h16_future_goal_only_steps':future_only,'h16_unique_current_source_frames':len(support_sources),
        'h16_current_source_ticks':sorted(support_sources),'h16_exact_center_consumption_steps':exact_center_steps,
        'h16_exact_center_source_frames':len(exact_center_points),'h16_current_velocity_halo_only_steps':current_halo_only,
        'first_actual_center_consumed_tick_in_H':first_exact_center,'overwritten_source_fraction':overwritten/max(1,len(common)),
        'overwritten_denominator':len(common),'old_source_frames_outside_final_cache_excluded':len(owned)-len(common)}


@torch.no_grad()
def offline_audit(cfg,run,output,versions):
    from .application import build_components
    from .checkpoint import unpack_value
    from .music_repair_audit import anchors_from_episode
    from .dance_losses import execution_loss,AnchorAlignment
    from .sample_distribution import describe_sample
    from supermimic.motion import ReferenceBatch
    from supermimic.rotations import quat_to_mat,yaw_mat,so3_log
    torch.set_num_threads(4);output.mkdir(parents=True,exist_ok=False)
    write_json(output/'input_manifest.json',{'source_run':str(run),'versions':versions,'source_config':cfg,'started_unix':time.time()})
    audit_logs(run,output)
    local_cfg=json.loads(json.dumps(cfg));local_cfg['model']['compile_networks']=False;local_cfg['teacher']['enabled']=False
    actor,adapter,*_=build_components(local_cfg,'cpu');del actor
    rows=[];all_coverage=[];all_samples=[]
    def targets(fixed,ticks):return ReferenceBatch(*(v.flatten(0,1) for v in fixed.reference_at_ticks(ticks)))
    def loss_chunks(state,target,alignment=None):
        values=[];parts=defaultdict(list)
        for begin in range(0,len(state),256):
            ref=ReferenceBatch(*(v[begin:begin+256] for v in target))
            if alignment is not None:
                align=alignment if len(alignment.rotation)==1 else AnchorAlignment(alignment.rotation[begin:begin+256],alignment.translation[begin:begin+256])
                ref=align.reference(ref)
            v,groups=execution_loss(state[begin:begin+256],ref,adapter.runtime.kin,cfg['rewards']['dance_scales'])
            values.append(v)
            for k,x in groups.items():parts[k].append(x)
        return torch.cat(values),{k:torch.cat(x) for k,x in parts.items()}
    for version in versions:
        folder=run/'evaluations/fixed_delay_replay'/('initial_genmo_baseline' if version==0 else f'actor_{version:06}')
        for source in sorted(folder.glob('shard_*/*_real_controls.pt')):
            payload=torch.load(source,map_location='cpu',weights_only=False,mmap=True);memo={}
            controls=unpack_value(payload['controls'],adapter,'cpu',memo);generations=unpack_value(payload['generations'],adapter,'cpu',memo)
            samples=anchors_from_episode(controls,generations,16)
            state=torch.cat([s.end.state for s in controls]);ticks=torch.tensor([s.end.tick for s in controls]);fixed=controls[0].start.teacher_reference
            target=targets(fixed,ticks);initial_alignment=AnchorAlignment.at_anchor(controls[0].start.state,fixed.read_current_future(controls[0].start.tick,0)[0])
            B,B_parts=loss_chunks(state,target,initial_alignment);C,C_parts=loss_chunks(state,target)
            A=[];A_integral=[];anchor_rows=[];discount=.02*.99**torch.arange(16)
            for sample in samples:
                tick=torch.tensor([s.end.tick for s in sample.transitions]);actual=torch.cat([s.end.state for s in sample.transitions])
                align=AnchorAlignment.at_anchor(sample.state.state,fixed.read_current_future(sample.state.tick,0)[0])
                values,parts=execution_loss(actual,align.reference(targets(fixed,tick)),adapter.runtime.kin,cfg['rewards']['dance_scales'])
                A.append(float(values.mean()));A_integral.append(float((values*discount).sum()))
                coverage=anchor_coverage(sample);coverage.update(version=version,task=controls[0].start.task_id)
                e=next((e for e in payload['events'] if e.get('reference_version')==sample.generation.reference_version),{})
                coverage['first_center_consumed_tick']=e.get('first_center_consumed_tick');all_coverage.append(coverage)
                desc=describe_sample(sample,adapter);desc.update(version=version);all_samples.append(desc)
                anchor_rows.append({'tick':sample.state.tick,'A_mean':A[-1],'A_integral':A_integral[-1],
                    'episode_window_loss':float(B[((tick-ticks[0])//12)].mean()),'coverage':coverage})
            yaw=so3_log(yaw_mat(target.root_R).transpose(-1,-2)@yaw_mat(quat_to_mat(state[:,3:7])))[:,2]
            metrics=json.loads((source.parent/'physical_summary.json').read_text())['physical_results']
            metric=next(r for r in metrics if Path(r['trajectory_artifact']).stem==source.name.replace('_real_controls.pt','_physical'))
            row={'version':version,'task':metric['task'],'source':metric['source'],'controls':len(controls),'anchors':len(samples),
                'A_anchor_H16_mean':stats(A),'A_anchor_H16_integral':stats(A_integral),
                'B_episode_once_loss':float(B.mean()),'C_world_loss':float(C.mean()),
                'C_original_report_parity':abs(float(C.mean())-metric['physical_metrics_mean']['dance_loss']),
                'C_world_root_rms_m':float((state[:,:3]-target.root_p).square().mean().sqrt()),
                'C_world_yaw_rms_rad':float(yaw.square().mean().sqrt()),'C_final_yaw_rad':float(yaw[-1]),
                'D_music_content':metric['music_content'],'B_components':{k:float(v.mean()) for k,v in B_parts.items()},
                'quarter_B_loss':[float(x.mean()) for x in torch.tensor_split(B,4)],'quarter_yaw_rms':[float(x.square().mean().sqrt()) for x in torch.tensor_split(yaw,4)],
                'anchor_rows':anchor_rows,'source_controls':str(source)}
            rows.append(row);write_json(output/'trajectory_objectives.json',rows)
            print(json.dumps({k:row[k] for k in ['version','task','anchors','B_episode_once_loss','C_world_loss','C_original_report_parity']},ensure_ascii=False),flush=True)
            del payload,memo,controls,generations,samples,state
            gc.collect()
    with (output/'request_coverage.jsonl').open('w') as f:
        for row in all_coverage:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    write_json(output/'evaluation_sample_distribution.json',{str(v):sample_counts([r for r in all_samples if r['version']==v]) for v in versions})
    summary={str(v):{'songs':len([r for r in rows if r['version']==v]),
        **{k:float(np.mean([r[k] for r in rows if r['version']==v])) for k in ['B_episode_once_loss','C_world_loss','C_world_root_rms_m','C_world_yaw_rms_rad']},
        'A_anchor_H16_mean':float(np.mean([r['A_anchor_H16_mean']['mean'] for r in rows if r['version']==v])),
        'coverage':{k:stats([r[k] for r in all_coverage if r['version']==v]) for k in ['h16_future_goal_only_steps','h16_current_reference_support_steps','h16_unique_current_source_frames','h16_exact_center_consumption_steps','h16_exact_center_source_frames','h16_current_velocity_halo_only_steps','overwritten_source_fraction']}}
        for v in versions}
    write_json(output/'summary.json',summary)
