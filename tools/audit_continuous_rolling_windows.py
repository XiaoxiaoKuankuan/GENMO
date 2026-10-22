#!/usr/bin/env python3
"""只读审计连续滚动PhysX采集的跨8秒窗口、行为链与价值时间合同。

读取collect-only产生的实际rollout及其SHA清单，不加载/更新Actor权重。按环境和
episode核对所有50Hz控制区间连续，历史不越过决策时刻，30Hz音乐/参考时间对齐，
前缀在Stage1已训练范围内，四步PPO仍来自同一行为版本。合并相邻窗口的紧凑来源
账本，报告旧窗口计划在下一窗口继续被中心消费的真实控制数；不从commit推断消费。

只验证真实保留范围，不把reset前预热记成Actor动作，不伪恢复中途PhysX状态。
窗口末端GAE自举沿用实际next_context；输出独立JSON，缺失数据明确失败而不补造。
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from collections import defaultdict

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from gem.closedloop.dppo.diagnostic_rollout import load_rollout
from gem.closedloop.dppo.reference_credit import validate_reward_intervals
from gem.closedloop.dppo.execution_provenance import consumption_report, source_frame_support
from tools.analyze_stage2_reference_consumption import source_origin


def audit_world_windows(cases):
    """缺失世界步数或概率验收也必须失败，不能把没有证据当作通过。"""
    previous={};worlds=[];probabilities=[]
    for window,case in enumerate(cases):
        probability=case.get('full_zero_update_probability',{})
        if not probability.get('passed') or probability.get('checked_steps')!=list(range(19)):
            raise ValueError('Missing complete zero-update stochastic-chain audit')
        probabilities.append(probability)
        ranks=case.get('collection_by_rank',[])
        if not ranks or previous and len(ranks)!=len(previous):
            raise ValueError('Missing or inconsistent distributed world evidence')
        for rank,item in enumerate(ranks):
            record=item.get('world_window',item.get('collector',{}).get('world_window'))
            if record is None:
                raise ValueError('Missing actual shared-world control evidence')
            begin,end=record['begin_control'],record['current_control']
            if (record['actual_world_controls']!=400 or not record['reached'] or end-begin!=400
                    or end!=record['target_control'] or rank in previous and begin!=previous[rank]):
                raise ValueError('Shared physical windows are not contiguous eight-second intervals')
            previous[rank]=end
            worlds.append(dict(window=window,rank=rank,**record))
    return worlds,probabilities


def audit(run):
    sessions=list(Path(run).glob('sessions/*/summary.json'))
    if len(sessions)!=1:raise ValueError('Exactly one finite collection session required')
    summary=json.loads(sessions[0].read_text())
    if summary['status']!='passed':raise ValueError('Collection did not finish successfully')
    state=summary['final_state']
    if state['actor_updates'] or state['critic_updates']:raise ValueError('Expected frozen Stage1 collection')
    cases=sorted(sessions[0].parent.glob('fixed_data_validation/iteration*'))
    if len(cases)<2:raise ValueError('Cross-window evidence requires two real world windows')
    rows_by_episode=defaultdict(list);records=defaultdict(list);generated_at={};versions=set();counts=[]
    for window,case in enumerate(cases):
        total=0;steps=0;prefixes=[]
        for path in sorted(case.glob('rank*')):
            if not path.is_dir():continue
            payload=load_rollout(path)
            for row in payload['transitions']:
                key=(path.name,row.identity['backend_session_id'],row.identity['env_id'],row.identity['episode_id'])
                versions.add(row.identity['policy_version'])
                begin,end=row.control_tick_begin,row.control_tick_end
                if begin%60 or end-begin!=12*row.executed_control_steps or row.executed_physics_steps!=4*row.executed_control_steps:
                    raise ValueError('Noncontiguous physical/control/source clocks')
                c=row.context
                if not torch.allclose(c['decision_time'].double(),torch.tensor(begin/600.,dtype=torch.float64)):
                    raise ValueError('Behavior condition not sampled at its actual decision time')
                history=c['proprio_history_times'][c['proprio_history_valid']]
                if history.numel() and bool((history>begin/600.+1e-10).any()):raise ValueError('Actor read future history')
                mask=c['known_qpos30_mask'];prefix=int(mask.any(-1).sum())
                if not 6<=prefix<=30 or c['future_times'].shape[-1]!=60:raise ValueError('Invalid dynamic prefix/horizon')
                expected=begin/600.+torch.arange(60,dtype=torch.float64)/30.
                if not torch.allclose(c['future_times'].double().reshape(-1),expected,atol=1e-10,rtol=0):
                    raise ValueError('Incorrect source/music alignment')
                provenance=row.metadata.get('compact_execution_provenance')
                if provenance is None:raise ValueError('Missing actual consumption evidence')
                rows_by_episode[key].append(row);records[key].append((window,provenance))
                generated_at[(key,row.identity['plan_id'])]=window
                total+=1;steps+=row.executed_control_steps;prefixes.append(prefix)
        counts.append(dict(window=window,chains=total,actual_actor_controls=steps,
            prefix_min=min(prefixes),prefix_max=max(prefixes)))
    if len(versions)!=1:raise ValueError('Mixed behavior versions in frozen collection')
    crossing=0;episodes_continuing=0;unresolved=0.;delayed_rewards=0;all_controls=0
    for key,rows in rows_by_episode.items():
        validate_reward_intervals(rows)
        items=records[key]
        if len({w for w,_ in items})>1:episodes_continuing+=1
        result=consumption_report([r for _,r in items]);plans=result['plans']
        unresolved+=sum(result['unresolved_source_support_weight'].values())
        for window,record in items:
            for row in record['steps']:
                begin=row['control_tick_begin'];plan=row.get('consumed_plan_ids')
                plan=plan[10] if plan is not None else record['plan_dictionary'][row['consumed_plan_indices'][10]]
                if plan.startswith('bootstrap:'):continue
                origins=[source_origin(plans,plan,frame) for frame,_ in source_frame_support(begin,plans[plan]['decision_tick'],60)]
                if any(origin and origin[0]!='bootstrap' and generated_at.get((key,origin[0]),window)<window for origin in origins):crossing+=1
        for row in rows:
            p=plans[row.identity['plan_id']];first=p['first_generated_current_tick']
            delayed_rewards+=sum(1 for tick in range(row.control_tick_begin,row.control_tick_end,12) if first is None or tick<first)
            all_controls+=row.executed_control_steps
    worlds,probabilities=audit_world_windows(summary['fixed_data_cases'])
    return dict(schema='genmo.rolling.windows_audit.v1',status='passed',windows=counts,
        continuing_physical_episodes=episodes_continuing,previous_window_generated_source_center_controls=crossing,
        unresolved_source_support_weight=unresolved,behavior_versions=sorted(versions),
        actor_updates=state['actor_updates'],critic_updates=state['critic_updates'],
        rewards_before_this_action_first_direct_source_control=delayed_rewards,actual_actor_controls=all_controls,
        delayed_reward_semantics='只读统计，不按plan重新分配奖励；窗口影响、直接中心和动力学影响不同',
        summary_path=str(sessions[0]),shared_world_intervals=worlds,probability_audit=probabilities)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=audit(args.run)
    with args.output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
