"""把真实紧凑GMT消费记录与同一旧行为链的逐帧联合KL关联。

输入必须是带SHA的fixed_data_validation和同一case产生的chain_attribution.pt。
按backend/env/episode合并真实21点窗口和中心消费，再用plan登记的decision_tick、
commit_tick映射30Hz相邻源帧。分类包含未提交、中心消费、仅窗口观察、当前保留区间
未观察及来源缺失。只报告当前证据覆盖范围，不把缺失或右截断帧称为从未执行。

旧数据无compact_execution_provenance时输出unverified。历史reference_credit账本
只可用于独立说明首次窗口/中心时间，不能补造当前case的逐步消费。脚本只用CPU，
不加载Actor、不启动物理、不改奖励或原始数据；新候选的KL仍是旧状态条件分布诊断。
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


def temporal_window_support(records,plans):
    """按实际control_begin区分回看、中心、前瞻，追继承源；不改变原执行记录。

    完整21点窗口同时含过去和未来，window-only不能直接叫前瞻。这里复用已核验
    登记及源帧插值公式，仅增加只读时间分类，并核对三类支持之和等于原完整窗口。
    """
    import math
    result={key:{side:[0.]*len(item['window_generated_frame_weight']) for side in ('past','current','future')}
        for key,item in plans.items()}
    def origin(plan,frame,seen=()):
        if plan.startswith('bootstrap:'):return None
        if plan in seen:raise ValueError('Cyclic source lineage in temporal attribution')
        item=plans.get(plan)
        if item is None or item.get('prefix_frames') is None:return None
        if frame>=item['prefix_frames']:return plan,frame
        parent=item.get('parent_plan_id');previous=plans.get(parent)
        if previous is None:return None
        index,remainder=divmod(item['decision_tick']+frame*20-previous['decision_tick'],20)
        if remainder or not 0<=index<previous.get('generation_frames',60):raise ValueError('Unaligned actual source prefix')
        return origin(parent,index,(*seen,plan))
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
                source=origin(plan,frame)
                if source is not None:result[source[0]][side][source[1]]+=weight
    for key,item in plans.items():
        for frame,weight in enumerate(item['window_generated_frame_weight']):
            if not math.isclose(sum(result[key][side][frame] for side in ('past','current','future')),weight,
                    rel_tol=1e-10,abs_tol=1e-8):raise AssertionError('Past/current/future support fails full-window conservation')
    return result


def associate(case,chain_payload):
    episodes=defaultdict(list);rows_by_id={};missing=0
    for path in sorted(Path(case).glob('rank[0-9][0-9]')):
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
    args=parser.parse_args();torch.set_num_threads(4)
    if args.output.exists():raise FileExistsError('Independent consumption report must not overwrite evidence')
    experiment=json.loads((args.chain_attribution.parent/'summary.json').read_text())
    case_sha=file_sha256(args.case/'case.json')
    if experiment['case_sha256']!=case_sha:raise ValueError('Coordinate attribution and supplied real case differ')
    data=torch.load(args.chain_attribution,map_location='cpu',weights_only=False)
    result=associate(args.case,data)
    result.update(case_sha256=case_sha,chain_attribution_sha256=file_sha256(args.chain_attribution))
    atomic_json(args.output,result)


if __name__=='__main__':main()
