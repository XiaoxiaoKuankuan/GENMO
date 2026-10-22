"""只读拆解真实DPPO优势的奖励/价值来源，并核对奖励与GMT消费计划的时间关系。

输入是已校验SHA的固定rollout目录，可包含多个连续8秒采集窗口。逐rank读取后，
按backend session、环境、episode、策略和原始采集窗口分别重建生产GAE，不跨重置
或不同价值快照拼接。将优势线性拆为真实奖励项、价值项及窗口尾部自举项，核对
拆解之和与保存标签一致；后两项是GAE代数贡献，不能直接称为Critic估计误差。

对保留的完整物理证据，以实际21点GMT输入和成功提交身份递归追溯保护前缀来源，
分开记录当前中心、未来前瞻和控制末奖励参考。没有完整控制账本时明确标记缺测。
奖励只核对50Hz原始账本，不重分配、不重复积分、不修改PPO标签，也不把暴露比例
解释成严格物理因果。中途截断的轨迹不是完整MC；真实Critic误差另由完整MC报告判断。

输出为小型JSON摘要及逐状态压缩证据，禁止写入正式训练目录、模型或原始数据。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
import gzip
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch

from gem.closedloop.dppo.budget import atomic_json
from gem.closedloop.dppo.diagnostic_rollout import load_rollout
from gem.closedloop.dppo.returns import compute_gae
from gem.closedloop.dppo.execution_provenance import source_frame_support


def decomposition(rows, *, gamma, lam):
    """在同一真实GAE片段上做线性分解，保持事件和变长折扣的生产定义。"""
    n=len(rows)
    values=[r.old_value for r in rows]
    nxt=[r.next_value for r in rows]
    arguments=dict(executed_steps=[r.executed_control_steps for r in rows],
        bootstrap_mask=[not r.terminated and r.next_context is not None for r in rows],
        continuation_mask=[i+1<n and not r.terminated and not r.truncated for i,r in enumerate(rows)],
        gamma_upper=gamma,lambda_upper=lam,normalize=False)
    rewards=[r.rewards for r in rows]
    events=[r.metadata.get('event_reward',0.) for r in rows]
    zeros=[torch.zeros_like(torch.as_tensor(r)) for r in rewards]
    total=compute_gae(rewards,values,nxt,event_rewards=events,**arguments)['advantages_raw']
    reward=compute_gae(rewards,[0.]*n,[0.]*n,event_rewards=events,**arguments)['advantages_raw']
    value=compute_gae(zeros,values,nxt,**arguments)['advantages_raw']
    tail=compute_gae(zeros,[0.]*n,[0.]*(n-1)+[nxt[-1]],**arguments)['advantages_raw']
    torch.testing.assert_close(total,reward+value,atol=1e-9,rtol=1e-10)
    return total,reward,value,tail


def analyze_case(case):
    result=[];controls=set();reward_checks=0
    for rank in sorted(case.glob('rank[0-9][0-9]')):
        payload=load_rollout(rank)
        rows=payload['transitions'];settings=payload['identity']['training']
        origins={}
        for row in rows:
            generated=row.metadata.get('generated',{})
            if generated:
                origins[str(row.identity['plan_id'])]=dict(decision_tick=int(generated['decision_tick']),
                    prefix_frames=int(generated['prefix_frames']),parent=generated.get('parent_plan_id'))

        @lru_cache(maxsize=None)
        def origin(plan,frame):
            if plan.startswith('bootstrap:'):return 'bootstrap'
            item=origins.get(plan)
            if item is None:return None
            if frame>=item['prefix_frames']:return plan
            parent=item['parent']
            if parent is None:return None
            if str(parent).startswith('bootstrap:'):return 'bootstrap'
            previous=origins.get(str(parent))
            if previous is None:return None
            index,remainder=divmod(item['decision_tick']+frame*20-previous['decision_tick'],20)
            if remainder or not 0<=index<60:raise ValueError('Invalid inherited-prefix source time')
            return origin(str(parent),index)

        def share(plan,tick,wanted):
            if str(plan).startswith('bootstrap:'):return 0.,0.
            item=origins.get(str(plan))
            if item is None:return 0.,1.
            own=unknown=0.
            for frame,weight in source_frame_support(tick,item['decision_tick']):
                resolved=origin(str(plan),frame)
                own+=weight*(resolved==wanted)
                unknown+=weight*(resolved is None)
            return own,unknown

        groups=defaultdict(list)
        for index,row in enumerate(rows):
            key=tuple(row.identity.get(k,0) for k in ('backend_session_id','env_id','episode_id','policy_version'))
            groups[key].append((index,row))
        components={}
        for pairs in groups.values():
            pairs.sort(key=lambda p:p[1].control_tick_begin)
            fragments=[];part=[]
            for pair in pairs:
                if part and (part[-1][1].truncated or part[-1][1].terminated or
                             part[-1][1].control_tick_end!=pair[1].control_tick_begin):
                    fragments.append(part);part=[]
                part.append(pair)
            if part:fragments.append(part)
            for part in fragments:
                values=decomposition([r for _,r in part],gamma=settings['gamma_upper'],lam=settings['lambda_upper'])
                for offset,(index,row) in enumerate(part):components[index]=[float(v[offset]) for v in values]
        for index,row in enumerate(rows):
            total,reward,value,tail=components[index]
            stored=float(payload['targets']['advantages_raw'][index])
            if abs(total-stored)>1e-7:raise ValueError('Reconstructed GAE differs from immutable target')
            task=row.metadata.get('training_task',{});plan=str(row.identity['plan_id'])
            record=dict(case=str(case),rank=rank.name,local_index=index,identity=row.identity,
                source=task.get('dataset'),sample_id=task.get('sample_id'),music_start_frame=task.get('music_start_frame'),
                control_tick_begin=row.control_tick_begin,control_tick_end=row.control_tick_end,
                terminated=row.terminated,truncated=row.truncated,reason=row.reason,
                gamma=settings['gamma_upper'],value=row.old_value,next_value=row.next_value,
                advantage=float(payload['targets']['advantages'][index]),raw_advantage=stored,
                reward_component=reward,value_component=value,window_tail_bootstrap_component=tail,
                control_reward_sum=float(torch.as_tensor(row.rewards).sum()),
                controls=row.executed_control_steps,physical_time_limit=row.metadata.get('physical_time_limit_reached',False),
                time_limit_bootstrap=row.metadata.get('time_limit_bootstrap',False))
            trace=row.metadata.get('execution_trace')
            if trace is None:
                record['reference_audit']='missing_full_execution_use_compact_analysis'
            else:
                current=[];future=[];reward_ref=[];unknown=0.
                for j,control in enumerate(trace):
                    begin=int(control['control_tick_begin']);end=int(control['tick'])
                    key=tuple(row.identity.get(k,0) for k in ('backend_session_id','env_id','episode_id'))+(begin,)
                    if key in controls:raise ValueError('Duplicated physical reward interval')
                    controls.add(key)
                    ticks=control['consumed_reference_ticks'];ids=control['consumed_plan_ids']
                    if len(ticks)!=21 or ticks[10]!=begin or end-begin!=12:
                        raise ValueError('Actual GMT window or 50Hz control time differs')
                    own,missing=share(ids[10],ticks[10],plan);current.append(own);unknown+=missing
                    exposure=[share(ids[k],ticks[k],plan)[0] for k in range(11,21)]
                    future.append(sum(exposure)/10)
                    reward_ref.append(share(control['reference_plan_id'],control['reference_tick'],plan)[0])
                    details=row.metadata.get('reward_details')
                    if details is not None:
                        # 回放奖励张量可能为FP32，原JSON明细为FP64；只容许对应一次量化误差。
                        dtype=torch.as_tensor(row.rewards).dtype
                        tolerance=2*torch.finfo(dtype).eps if dtype.is_floating_point else 0.
                        # 生产约定将真实事件只加到最后一个已执行控制奖励，零步事件另存。
                        event=float(row.metadata.get('event_penalty_total',0.)) if j+1==len(trace) else 0.
                        expected=details[j]['reward']+event
                        if details[j]['tick']!=end or not math.isclose(expected,float(row.rewards[j]),
                                rel_tol=tolerance,abs_tol=1e-12):
                            raise ValueError('Stored control reward differs: '+json.dumps(dict(
                                end=end,detail_tick=details[j]['tick'],detail_reward=details[j]['reward'],
                                event_reward=event,stored=float(row.rewards[j]),dtype=str(dtype))))
                        reward_checks+=1
                record.update(reference_audit='actual_GMT_window_recursive_prefix_source',
                    own_current_share=sum(current)/len(current) if current else None,
                    own_future_mean_exposure=sum(future)/len(future) if future else None,
                    own_reward_reference_share=sum(reward_ref)/len(reward_ref) if reward_ref else None,
                    controls_without_own_current=sum(v==0. for v in current),
                    controls_without_own_future=sum(v==0. for v in future),unresolved_current_support=unknown)
            result.append(record)
        del payload,rows
    return result,dict(unique_controls=len(controls),reward_details_checked=reward_checks)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',type=Path,nargs='+',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();args.output_dir.mkdir(exist_ok=False,parents=True)
    records=[];audits=[]
    for case in args.cases:
        part,audit=analyze_case(case);records.extend(part);audits.append(audit)
    def summarize(part):
        a=torch.tensor([r['advantage'] for r in part],dtype=torch.float64)
        ordered=sorted(part,key=lambda r:r['advantage'],reverse=True)
        top=ordered[:max(1,len(ordered)//100)]
        measured=[r for r in part if 'own_current_share' in r]
        controls=sum(r['controls'] for r in measured)
        return dict(states=len(part),advantage_min=float(a.min()),advantage_max=float(a.max()),
            positive_count=int((a>0).sum()),top_one_percent_mean_advantage=sum(r['advantage'] for r in top)/len(top),
            top_one_percent_reward_component_mean=sum(r['reward_component'] for r in top)/len(top),
            top_one_percent_value_component_mean=sum(r['value_component'] for r in top)/len(top),
            controls_with_full_reference_evidence=controls,
            controls_without_own_current=sum(r['controls_without_own_current'] for r in measured),
            controls_without_own_future=sum(r['controls_without_own_future'] for r in measured),
            unresolved_current_support=sum(r['unresolved_current_support'] for r in measured))
    report=dict(overall=summarize(records),by_source={source:summarize([r for r in records if r['source']==source])
        for source in sorted({r['source'] for r in records})},case_audits=audits,
        highest_advantages=sorted(records,key=lambda r:r['advantage'],reverse=True)[:24],
        limits='GAE_value_contribution_is_not_true_MC_error; input_exposure_is_not_strict_causality; reward_ledger_unchanged')
    with gzip.open(args.output_dir/'states.jsonl.gz','wt',encoding='utf8') as stream:
        for record in records:stream.write(json.dumps(record,ensure_ascii=False)+'\n')
    atomic_json(args.output_dir/'summary.json',report)
    print(json.dumps(report['overall'],ensure_ascii=False))


if __name__=='__main__':main()
