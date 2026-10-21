"""把真实紧凑GMT消费记录与同一旧行为链的逐帧联合KL关联。

输入必须是带SHA的fixed_data_validation和同一case产生的chain_attribution.pt。
按backend/env/episode合并真实21点窗口和中心消费，再用plan登记的decision_tick、
commit_tick映射30Hz相邻源帧。分类包含未提交、中心消费、仅窗口观察、当前保留区间
未观察及来源缺失。只报告当前证据覆盖范围，不把缺失或右截断帧称为从未执行。

旧数据无compact_execution_provenance时输出unverified。历史reference_credit账本
只可用于独立说明首次窗口/中心时间，不能补造当前case的逐步消费。脚本只用CPU，
不加载Actor、不启动物理、不改奖励或原始数据；新候选的KL仍是旧状态条件分布诊断。
若存在同条件输出探针，按原控制来源先线性组合关节角变化向量，再统计50Hz中心
参考的RMS；源计划缺少探针时明确不覆盖。此为固定旧条件的局部反事实参考变化，
不是连贯重规划后的轨迹，也不是机器人实际运动或世界系根姿态重建。
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from gem.closedloop.dppo.budget import atomic_json
from gem.closedloop.dppo.diagnostic_rollout import load_rollout
from gem.closedloop.dppo.execution_provenance import consumption_report
from gem.closedloop.dppo.execution_provenance import source_frame_support
from gem.closedloop.dppo.training_checkpoint import file_sha256


def token(identity):
    return tuple(str(identity.get(k,'')) for k in ('backend_session_id','env_id','episode_id','plan_id'))


def source_origin(plans,plan,frame,seen=()):
    if plan.startswith('bootstrap:'):return 'bootstrap',frame
    if plan in seen:raise ValueError('Cyclic source lineage in temporal attribution')
    item=plans.get(plan)
    if item is None or item.get('prefix_frames') is None:return None
    if frame>=item['prefix_frames']:return plan,frame
    parent=item.get('parent_plan_id')
    if parent and parent.startswith('bootstrap:'):return 'bootstrap',frame
    previous=plans.get(parent)
    if previous is None:return None
    index,remainder=divmod(item['decision_tick']+frame*20-previous['decision_tick'],20)
    if remainder or not 0<=index<previous.get('generation_frames',60):raise ValueError('Unaligned actual source prefix')
    return source_origin(plans,parent,index,(*seen,plan))


def temporal_window_support(records,plans):
    """按实际control_begin区分回看、中心、前瞻，追继承源；不改变原执行记录。

    完整21点窗口同时含过去和未来，window-only不能直接叫前瞻。这里复用已核验
    登记及源帧插值公式，仅增加只读时间分类，并核对三类支持之和等于原完整窗口。
    """
    import math
    result={key:{side:[0.]*len(item['window_generated_frame_weight']) for side in ('past','current','future')}
        for key,item in plans.items()}
    controls={}
    for record in records:
        for row in record['steps']:
            ids=row.get('consumed_plan_ids')
            if ids is None:ids=[record['plan_dictionary'][i] for i in row['consumed_plan_indices']]
            controls[row['control_tick_begin']]=(row['consumed_reference_ticks'],ids)
    for begin,(ticks,ids) in controls.items():
        for tick,plan in zip(ticks,ids):
            if plan.startswith('bootstrap:'):continue
            item=plans[plan];side='past' if tick<begin else 'future' if tick>begin else 'current'
            for frame,weight in source_frame_support(tick,item['decision_tick'],item.get('generation_frames',60)):
                source=source_origin(plans,plan,frame)
                if source is not None and source[0]!='bootstrap':result[source[0]][side][source[1]]+=weight
    for key,item in plans.items():
        for frame,weight in enumerate(item['window_generated_frame_weight']):
            if not math.isclose(sum(result[key][side][frame] for side in ('past','current','future')),weight,
                    rel_tol=1e-10,abs_tol=1e-8):raise AssertionError('Past/current/future support fails full-window conservation')
    return result


def joint_reference_changes(episodes,consumed,probes,rows_by_id):
    """仅在所有相邻生成源都有探针时计算关节插值；未知源绝不假设为零。"""
    deltas={}
    for payload in probes:
        before,after=payload['before'],payload['after']
        if before['identities']!=after['identities']:raise ValueError('Probe before/after identities differ')
        for i,identity in enumerate(before['identities']):
            key=token(identity)
            if key not in rows_by_id:raise ValueError('Probe identity absent from supplied real behavior case')
            delta=(after['qpos'][i,:,7:]-before['qpos'][i,:,7:]).double()
            if delta.shape!=(60,21) or not torch.isfinite(delta).all():raise ValueError('Probe must contain finite canonical joint deltas')
            if key in deltas and not torch.equal(deltas[key],delta):raise ValueError('Conflicting same-chain output probes')
            deltas[key]=delta
    changes=[];uncovered=0;bootstrap_only=0;controls_total=0
    for episode,records in episodes.items():
        plans=consumed[episode]['plans'];controls={}
        for record in records:
            for row in record['steps']:
                if 'consumed_plan_ids' in row:plan=row['consumed_plan_ids'][10]
                else:plan=record['plan_dictionary'][row['consumed_plan_indices'][10]]
                controls[row['control_tick_begin']]=(row['consumed_reference_ticks'][10],plan)
        for begin,(tick,plan) in controls.items():
            controls_total+=1
            if plan.startswith('bootstrap:'):bootstrap_only+=1;continue
            change=torch.zeros(21,dtype=torch.float64);covered=True;generated=False
            for frame,weight in source_frame_support(tick,plans[plan]['decision_tick'],plans[plan].get('generation_frames',60)):
                origin=source_origin(plans,plan,frame)
                if origin is None:covered=False;break
                if origin[0]=='bootstrap':continue
                generated=True;key=(*episode,origin[0])
                if key not in deltas:covered=False;break
                change+=weight*deltas[key][origin[1]]
            if not covered:uncovered+=1
            elif not generated:bootstrap_only+=1
            else:changes.append(dict(episode=list(episode),control_begin=begin,plan_id=plan,
                joint_rms_rad=float(change.square().mean().sqrt()),joint_max_abs_rad=float(change.abs().max())))
    values=torch.tensor([r['joint_rms_rad'] for r in changes],dtype=torch.float64)
    return dict(status='verified_conditional_joint_interpolation' if changes else 'unverified_no_covered_generated_centers',
        probed_chains=len(deltas),retained_controls=controls_total,covered_generated_centers=len(changes),
        missing_probe_or_lineage_centers=uncovered,bootstrap_only_centers=bootstrap_only,
        coverage_fraction=len(changes)/controls_total if controls_total else None,
        joint_rms_rad=dict(mean=float(values.mean()),p50=float(values.median()),p95=float(torch.quantile(values,.95)),
            max=float(values.max())) if values.numel() else None,
        largest_changes=sorted(changes,key=lambda r:r['joint_rms_rad'],reverse=True)[:12],
        scope='old_actual_50Hz_source_support_applied_to_independent_same_condition_probe_deltas_not_coherent_new_rollout_or_physics',
        interpolation='linear_delta_vectors_before_RMS_not_weighted_average_of_frame_RMS',
        world_root_change='unverified_missing_world_alignment_not_inferred_from_model_root_features')


def associate(case,chain_payload,probes=(),*,execution_cases=()):
    episodes=defaultdict(list);rows_by_id={};missing=0
    # 附加窗口只补充真实执行来源，不增加任何PPO链或混合行为策略概率。
    paths=[path for root in (case,*execution_cases) for path in sorted(Path(root).glob('rank[0-9][0-9]'))]
    for path in paths:
        payload=load_rollout(path)
        for row in payload['transitions']:
            key=token(row.identity);rows_by_id[key]=row
            record=row.metadata.get('compact_execution_provenance')
            if record is not None:episodes[key[:3]].append(record)
            else:missing+=1
    consumed={key:consumption_report(records) for key,records in episodes.items()}
    temporal={key:temporal_window_support(records,consumed[key]['plans']) for key,records in episodes.items()}
    names=('generated_not_committed','current_consumed_source_support','window_observed_not_current',
        'not_observed_in_retained_controls','unverified_missing_provenance')
    totals={name:dict(joint_kl=0.,free_coordinates=0,frame_log_ratio_signed=0.,frame_log_ratio_abs_sum=0.) for name in names}
    temporal_names=('generated_not_committed','current_consumed_source_support','future_window_only_source_support',
        'past_window_only_source_support','not_observed_in_retained_controls','unverified_missing_provenance')
    temporal_totals={name:dict(joint_kl=0.,free_coordinates=0) for name in temporal_names}
    details=[]
    for meta in chain_payload['metadata']:
        key=token(meta['identity']);row=rows_by_id.get(key)
        if row is None:raise ValueError('Attribution chain identity is absent from the supplied real rollout')
        episode=consumed.get(key[:3]);plan=(episode or {}).get('plans',{}).get(key[3])
        lineage=bool((episode or {}).get('source_lineage_complete'))
        current=(plan or {}).get('current_generated_frame_weight',[0.]*60);window=(plan or {}).get('window_generated_frame_weight',[0.]*60)
        time_support=temporal.get(key[:3],{}).get(key[3],{'past':[0.]*60,'current':[0.]*60,'future':[0.]*60})
        counts={name:0 for name in names}
        for frame,(kl,log) in enumerate(zip(meta['kl_by_frame'],meta['log_ratio_by_frame'])):
            if plan is not None and plan['commit_tick'] is None:category='generated_not_committed'
            elif plan is None or not lineage:category='unverified_missing_provenance'
            elif current[frame]>0:category='current_consumed_source_support'
            elif window[frame]>0:category='window_observed_not_current'
            else:category='not_observed_in_retained_controls'
            item=totals[category];item['joint_kl']+=kl
            item['free_coordinates']+=int(row.free_mask[frame].sum())*4
            item['frame_log_ratio_signed']+=log;item['frame_log_ratio_abs_sum']+=abs(log);counts[category]+=1
            temporal_category=category
            if category=='window_observed_not_current':
                temporal_category='future_window_only_source_support' if time_support['future'][frame]>0 else 'past_window_only_source_support'
            temporal_totals[temporal_category]['joint_kl']+=kl
            temporal_totals[temporal_category]['free_coordinates']+=int(row.free_mask[frame].sum())*4
        details.append(dict(identity=meta['identity'],source=meta['source'],music_frame=meta['music_frame'],
            prefix_frames=meta['prefix_frames'],frame_categories=counts,
            current_frames=(plan or {}).get('observed_current_frames'),
            window_only_frames=(plan or {}).get('observed_lookahead_only_frames'),
            current_generated_frames=[i for i,v in enumerate(current) if v>0],
            future_window_generated_frames=[i for i,v in enumerate(time_support['future']) if v>0],
            past_window_generated_frames=[i for i,v in enumerate(time_support['past']) if v>0],
            first_window_tick=(plan or {}).get('first_window_tick'),first_current_tick=(plan or {}).get('first_current_tick')))
    total=sum(v['joint_kl'] for v in totals.values())
    for item in totals.values():
        item['kl_fraction']=item['joint_kl']/total if total else None
        item['mean_coordinate_kl']=item['joint_kl']/item['free_coordinates'] if item['free_coordinates'] else None
    for item in temporal_totals.values():item['kl_fraction']=item['joint_kl']/total if total else None
    return dict(status='verified_in_retained_ranges' if episodes else 'unverified',
        missing_compact_transitions=missing,by_consumption=totals,by_temporal_consumption=temporal_totals,chains=details,
        consumed_joint_probe=joint_reference_changes(episodes,consumed,probes,rows_by_id) if probes else None,
        temporal_scope='actual_window_tick_vs_control_begin; current_priority_then_future_then_past; source_lineage_resolved',
        reason=None if episodes else 'Historical learning-only rollout omitted real per-control GMT windows',
        association_scope='adjacent_source_pose_support_not_full_velocity_or_physical_causality',
        root_feature_association_is_temporal_index_not_world_pose_causality=True,
        unobserved_frames_are_not_proof_of_never_executed=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--chain-attribution',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--probe-vectors-dir',type=Path,help='可选固定条件探针目录；默认在同一归因实验目录自动发现')
    parser.add_argument('--execution-case',type=Path,action='append',default=[],help='只补充后续真实消费记录，不加入概率batch')
    args=parser.parse_args();torch.set_num_threads(4)
    if args.output.exists():raise FileExistsError('Independent consumption report must not overwrite evidence')
    experiment=json.loads((args.chain_attribution.parent/'summary.json').read_text())
    case_sha=file_sha256(args.case/'case.json')
    if experiment['case_sha256']!=case_sha:raise ValueError('Coordinate attribution and supplied real case differ')
    data=torch.load(args.chain_attribution,map_location='cpu',weights_only=False)
    directory=args.probe_vectors_dir or args.chain_attribution.parent
    paths=sorted(directory.glob('probe_vectors_rank*.pt')) or sorted(directory.glob('ordinary_probe_vectors_rank*.pt'))
    probes=[torch.load(path,map_location='cpu',weights_only=False) for path in paths]
    result=associate(args.case,data,probes,execution_cases=args.execution_case)
    result['probe_vector_sha256']={str(p):file_sha256(p) for p in paths}
    result.update(case_sha256=case_sha,chain_attribution_sha256=file_sha256(args.chain_attribution))
    atomic_json(args.output,result)


if __name__=='__main__':main()
