"""真实GMT参考消费的紧凑来源记录与30Hz源帧映射，不推断缺失历史。

输入只取后端已经确认的21点GMT窗口、中心点、奖励参考和成功commit。600Hz整数
时钟下30Hz源帧间隔20tick，50Hz参考间隔12tick；每个目标参考映射至相邻两个
源帧及其精确线性权重。该权重描述关节/位置插值支持，姿态使用相同相邻帧的SLERP，
速度与角速度还含差分支持，不能把两帧权重当作完整动力学因果权重。

紧凑记录可在固定数据full_execution=False时保留，不保存机器人大状态。窗口首次
出现、中心实际消费和奖励参考分别统计，计划起点来自权威登记，禁止假设前15帧
执行。历史只有首次窗口时间而无逐步轨迹时返回unverified，不伪造中心消费。
本模块不改变参考调度、奖励、GAE或真实物理执行；跨rollout完整性由记录范围注明。
"""
from __future__ import annotations

SCHEMA='genmo.stage2.compact_execution_provenance.v1'


def source_frame_support(reference_tick,decision_tick,frames=60):
    relative=int(reference_tick)-int(decision_tick)
    left,remainder=divmod(relative,20)
    support=[(left,1-remainder/20)]
    if remainder:support.append((left+1,remainder/20))
    if any(frame<0 or frame>=frames for frame,_ in support):
        raise ValueError('Reference tick lacks real source-frame interpolation support')
    return support


def compact_execution(rows,*,ledger,plan_id,decision_tick,commit):
    steps=[];dictionary=[];lookup={}
    for row in rows:
        ticks=[int(t) for t in row['consumed_reference_ticks']]
        plans=[str(p) for p in row['consumed_plan_ids']]
        begin=int(row['control_tick_begin'])
        if len(ticks)!=21 or ticks[10]!=begin or len(plans)!=21:
            raise ValueError('Compact provenance requires the actual complete GMT window')
        for plan in plans:
            if plan not in lookup:lookup[plan]=len(dictionary);dictionary.append(plan)
        steps.append(dict(control_tick_begin=begin,control_tick_end=int(row['tick']),
            consumed_reference_ticks=ticks,consumed_plan_indices=[lookup[plan] for plan in plans],
            reference_tick=int(row['reference_tick']),reference_plan_id=str(row['reference_plan_id'])))
    origins={str(key):dict(decision_tick=int(item['decision_tick']),commit_tick=item['commit_tick'])
        for key,item in ledger.records.items() if not str(key).startswith('bootstrap:')}
    origins[str(plan_id)]=dict(decision_tick=int(decision_tick),commit_tick=(None if commit is None else int(commit['effective_tick'])))
    return dict(schema=SCHEMA,plans=origins,steps=steps,plan_dictionary=dictionary,
        scope='actual_control_window_current_center_and_reward_reference_in_this_transition')


def consumption_report(records,*,frames=60):
    available=[record for record in records if record is not None]
    if not available:return dict(status='unverified',reason='No real per-control execution provenance retained')
    plans={};controls={}
    for record in available:
        if record.get('schema')!=SCHEMA:raise ValueError('Unknown compact provenance schema')
        for key,item in record['plans'].items():
            if key in plans and plans[key]!=item:raise ValueError('Conflicting plan commit/origin identity')
            plans[key]=item
        for raw in record['steps']:
            row=dict(raw)
            if 'consumed_plan_indices' in row:
                row['consumed_plan_ids']=[record['plan_dictionary'][i] for i in row.pop('consumed_plan_indices')]
            # 调用者按物理episode分组；同一episode控制只允许完整一致的重复视图。
            key=row['control_tick_begin']
            if key in controls and controls[key]!=row:raise ValueError('Conflicting real control provenance')
            controls[key]=row
    result={key:dict(**item,window_frame_weight=[0.]*frames,current_frame_weight=[0.]*frames,
        reward_frame_weight=[0.]*frames,first_window_tick=None,first_current_tick=None) for key,item in plans.items()}
    for begin,row in sorted(controls.items()):
        occurrences=[('window',tick,plan) for tick,plan in zip(row['consumed_reference_ticks'],row['consumed_plan_ids'])]
        occurrences.extend((('current',row['consumed_reference_ticks'][10],row['consumed_plan_ids'][10]),
            ('reward',row['reference_tick'],row['reference_plan_id'])))
        for kind,tick,plan in occurrences:
            if plan.startswith('bootstrap:'):continue
            if plan not in result:raise ValueError('Consumed reference origin missing from actual plan registry')
            item=result[plan]
            if item['commit_tick'] is None or begin<item['commit_tick']:
                raise ValueError('Observed reference before successful commit')
            for frame,weight in source_frame_support(tick,item['decision_tick'],frames):
                item[kind+'_frame_weight'][frame]+=weight
            if kind in ('window','current') and item['first_'+kind+'_tick'] is None:item['first_'+kind+'_tick']=begin
    for item in result.values():
        current=item['current_frame_weight'];window=item['window_frame_weight']
        item['observed_current_frames']=[i for i,v in enumerate(current) if v>0]
        item['observed_lookahead_only_frames']=[i for i,v in enumerate(window) if v>0 and current[i]==0]
        item['unobserved_frames']=[i for i,v in enumerate(window) if v==0]
        item['unobserved_is_not_proof_of_never_consumed']=True
    return dict(status='verified_in_retained_control_range',plans=result,control_count=len(controls),
        pose_interpolation='linear_joint_position_and_position_SLERP_orientation_adjacent_source_support',
        scope='finite_retained_controls_not_complete_episode_unless_explicitly_collected')
