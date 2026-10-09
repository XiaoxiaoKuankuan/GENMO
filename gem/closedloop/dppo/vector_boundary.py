"""把多环境轮末继续执行的真实控制区间并入最后一条上层转移。

各环境采满2或3条后仍属于同一个GPU场景，不能停止记录却继续移动机器人。
提前完成的环境沿已提交参考继续执行，直到本rank所有环境达到共同决策边界；
额外控制步保留原奖励公式、完整journal和实际时长，并入最后一条扩散动作的
半马尔可夫转移，不额外生成动作或扩大160条rollout。物理终止、音乐结束和行政
上限单独处理。已采满额度的环境最多等到仍支持前缀及GMT前瞻的决策边界，随后
按行政截断保留bootstrap并在下一轮重置，不把缺少未来参考误判成物理终止。
其他未到边界的环境下一轮继续真实状态/历史，GAE仍在片段边界截断。
副作用之前先预占有限执行额度，只有带ACK身份的精确回复才结算未用额度。
"""
from __future__ import annotations
import copy
import torch
from .env_adapter import ExecutionIntegrityError

FRAGMENT_CONTRACT='available_reference_fragment_boundary.v2'


def fragment_reference_boundary(snapshot, *, minimum_prefix=12):
    """返回同时支持最小前缀、GMT十控制步前瞻及差分的最后0.5秒决策点。"""
    source=int(snapshot['source_end_tick'])
    valid=int(snapshot['reference_valid_end_tick'])
    latest_deadline=((min(valid,source-12)-120)//12)*12
    latest_prefix=source-(minimum_prefix-1)*20
    boundary=(min(latest_deadline,latest_prefix)//300)*300
    if boundary<int(snapshot['tick']):
        raise ExecutionIntegrityError('Completed GPU fragment has no causal bootstrap reference')
    return boundary


def finish_vector_fragment(env,row,*,continue_episode):
    previous=env.snapshot
    remaining=min(env.music_end_tick,env.soft_end_tick)-previous['tick']
    limit=max(0,remaining//12) if continue_episode and not previous['done'] else 0
    requested_limit=limit
    supported_tick=None
    if continue_episode and not previous['done']:
        supported_tick=fragment_reference_boundary(previous)
        limit=min(limit,(supported_tick-previous['tick'])//12)
    if limit>2000:raise ExecutionIntegrityError('Vector fragment drain exceeds finite resource bound')
    end_reason='music_end' if env.music_end_tick<=env.soft_end_tick else 'collection_limit'
    reference_limited=limit<requested_limit
    if reference_limited:end_reason='reference_horizon_truncated'
    if limit:env.budget.reserve(env.phase,control_steps=limit,physics_steps=4*limit)
    result=env.backend.call('drain_fragment',max_control_steps=limit,end_reason=end_reason,
        advance_id=f"{previous['episode_id']}:fragment:{env.backend.sequence+1}")
    if limit:env.budget.settle_control(env.phase,limit,result)
    actual=result['executed_control_steps'];trace=result['trace'];snapshot=result['snapshot']
    if (not result.get('transition_valid') or not result.get('physics_count_exact') or
            len(trace)!=actual or not 0<=actual<=limit or result['executed_physics_steps']!=4*actual or
            snapshot['episode_id']!=previous['episode_id'] or snapshot['tick']!=previous['tick']+12*actual):
        raise ExecutionIntegrityError('Invalid vector fragment drain feedback')
    values=[];details=[]
    for n,step in enumerate(trace):
        if step['episode_id']!=row.identity['episode_id'] or step['env_id']!=row.identity['env_id'] or step['tick']!=previous['tick']+12*(n+1):
            raise ExecutionIntegrityError('Cross-environment or discontinuous fragment tail')
        reward=env.reward.evaluate_step(step)
        if not reward.get('transition_valid'):raise ExecutionIntegrityError(f'Invalid tail reward: {reward.get("errors")}')
        values.append(float(reward['reward']));details.append(reward)
    newly_failed=bool(snapshot.get('terminated') and not previous.get('terminated'))
    penalty=env.reward.event_reward('physical_failure') if newly_failed else 0.
    if values:values[-1]+=penalty
    elif penalty:
        if row.rewards.numel():row.rewards[-1]+=penalty
        else:row.metadata['event_reward']=row.metadata.get('event_reward',0.)+penalty
    if values:row.rewards=torch.cat((row.rewards,torch.tensor(values,dtype=row.rewards.dtype)))
    row.control_tick_end=snapshot['tick'];row.executed_control_steps+=actual;row.executed_physics_steps+=4*actual
    env.snapshot=snapshot
    reference_exhausted=reference_limited and actual==limit
    row.terminated=bool(row.terminated or snapshot.get('terminated') or snapshot['tick']>=env.music_end_tick)
    row.truncated=not row.terminated
    row.reason=snapshot.get('reason') or ('music_end' if row.terminated else
        'reference_horizon_truncated' if reference_exhausted else 'collector_fragment_boundary')
    row.next_context=None if row.terminated else env.preview_context()[0]
    row.metadata['next_remaining_music_seconds']=env.remaining_music()
    row.metadata['terminal_snapshot']=copy.deepcopy(snapshot)
    row.metadata['reward_details'].extend(details)
    row.metadata['event_penalty_total']+=penalty
    consumed={str(p) for step in trace for p in step.get('consumed_plan_ids',[]) if p is not None}
    row.metadata['consumed_plan_ids']=sorted(set(row.metadata['consumed_plan_ids'])|consumed)
    row.metadata['fragment_tail']=dict(executed_control_steps=actual,executed_physics_steps=4*actual,
        begin_tick=previous['tick'],end_tick=snapshot['tick'],contract=FRAGMENT_CONTRACT,
        execution_sequence=result['mutation_seq'],budget_reserved_controls=limit,
        requested_control_steps=requested_limit,maximum_supported_decision_tick=supported_tick,
        reference_horizon_truncated=reference_exhausted)
    row.validate()
    return actual, bool(row.terminated or snapshot['tick']>=env.soft_end_tick or
                       not continue_episode or reference_exhausted)
