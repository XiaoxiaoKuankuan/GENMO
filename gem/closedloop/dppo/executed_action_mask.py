"""按GMT真实消费来源限定逐去噪PPO的动作坐标，完整远景仍照常生成。

输入为同一rank已结束的物理rollout及每次决策保留的紧凑参考来源记录。以
backend/env/episode分组，合并成功提交的计划登记，读取每个50Hz控制开始时
GMT窗口中心与相邻一个控制点的真实计划编号和600Hz时间，再映射到30Hz
源帧。相邻点只覆盖速度/差分所需支持，不将整个21点前瞻窗口当成已执行。
复制前缀沿parent_plan_id追溯到实际生成该源帧的旧决策，不给复制它的新决策
记一次动作。只对本批同策略版本中实际拥有这些坐标的决策建立ppo_action_mask。

掩码严格是采样free_mask的子集，单独保存；原条件、完整链、旧高斯参数、原始
采样概率及free_mask均保持不变。生成/物理/奖励和时间GAE不变；学习只在该掩码
上重算每步高斯边缘概率、PPO裁剪及KL。远景的参数共享和注意力仍可间接变化，
这里不承诺远景输出冻结，也不宣称这是全动作联合策略梯度的无偏替代。

跨PPO边界的旧版本来源只统计，不把它的帧塞给本批新决策；尚未在本批消费的
动作没有PPO坐标，但其真实物理转移仍保留给Critic与统计。不猜测未观察到的未来，
缺少来源、时间不合法、同一控制重复或来源循环均报错，不静默放宽掩码。
"""
from __future__ import annotations

from collections import defaultdict

import torch

from .execution_provenance import SCHEMA, source_frame_support

CONTRACT='genmo.ppo.executed_source_frames.v1'


def action_mask(row):
    """旧合同仍用原采样自由度；新合同必须已有采集后固定的独立掩码。"""
    scope=row.metadata.get('sampler_trace',{}).get('kernel_config',{}).get('actor_probability_scope')
    if scope==CONTRACT and (row.metadata.get('ppo_action_scope')!=CONTRACT or 'ppo_action_mask' not in row.metadata):
        raise ValueError('Executed-frame behavior requires its saved PPO mask, never full-horizon fallback')
    return row.metadata.get('ppo_action_mask',row.free_mask)


def bind_executed_action_masks(rows):
    """一次性解析完整本批消费来源；原始奖励保留，返回可审计的选择计数。"""
    groups=defaultdict(list)
    for row in rows:
        key=tuple(row.identity.get(k) for k in ('backend_session_id','env_id','episode_id'))
        groups[key].append(row)
    report=dict(contract=CONTRACT,physical_transitions=len(rows),actor_transitions=0,
        sampled_free_coordinates=0,ppo_coordinates=0,controls=0,
        previous_rollout_source_support=0,bootstrap_source_support=0,
        reward_ownership='original_physical_time_GAE_not_reassigned',
        selected_reference_points='actual_GMT_center_and_one_adjacent_control_each_side')
    for group in groups.values():
        plans={};controls={};owners={};selected=defaultdict(set)
        for row in group:
            plan=str(row.identity['plan_id'])
            if plan in owners:raise ValueError('Repeated decision plan in execution-mask rollout')
            owners[plan]=row
            record=row.metadata.get('compact_execution_provenance')
            if record is None or record.get('schema')!=SCHEMA:
                raise ValueError('Executed-frame PPO requires complete physical provenance')
            begins=[int(c['control_tick_begin']) for c in record['steps']]
            if (begins!=list(range(row.control_tick_begin,row.control_tick_end,12))
                    or len(begins)!=row.executed_control_steps):
                raise ValueError('Executed-frame provenance does not cover every actual control')
            for name,item in record['plans'].items():
                old=plans.setdefault(str(name),{})
                for field,value in item.items():
                    if field in old and old[field] is not None and value is not None and old[field]!=value:
                        raise ValueError('Conflicting reference source identity')
                    if value is not None or field not in old:old[field]=value
            for control in record['steps']:
                begin=int(control['control_tick_begin'])
                if begin in controls:raise ValueError('Duplicate physical control in execution-mask rollout')
                ticks=control['consumed_reference_ticks']
                ids=[record['plan_dictionary'][i] for i in control['consumed_plan_indices']]
                if ticks!=list(range(begin-120,begin+121,12)) or len(ids)!=21:
                    raise ValueError('Executed-frame mask requires actual complete GMT reference window')
                if int(control['control_tick_end'])!=begin+12 or int(control['reference_tick'])!=begin+12:
                    raise ValueError('Executed-frame control/reward clock differs')
                if control.get('gmt_reference_version')!=control.get('reward_reference_version'):
                    raise ValueError('Reward and GMT reference versions differ')
                controls[begin]=(control,ticks,ids)

        def resolve(plan,frame,seen=()):
            if plan.startswith('bootstrap:'):return None
            if plan in seen:raise ValueError('Cyclic reference prefix lineage')
            item=plans.get(plan)
            if item is None or item.get('prefix_frames') is None:
                raise ValueError('Missing reference source lineage for executed-frame PPO')
            if not 0<=frame<item['generation_frames']:raise ValueError('Source frame outside generated plan')
            if frame>=item['prefix_frames']:return plan,frame
            parent=item.get('parent_plan_id')
            if parent is None:raise ValueError('Copied prefix lacks its true generating parent')
            parent=str(parent)
            if parent.startswith('bootstrap:'):return None
            previous=plans.get(parent)
            if previous is None:raise ValueError('Copied prefix parent absent from retained lineage')
            absolute=item['decision_tick']+frame*20
            index,remainder=divmod(absolute-previous['decision_tick'],20)
            if remainder:raise ValueError('Copied source prefix is not aligned to its parent')
            return resolve(parent,index,(*seen,plan))

        for begin,(control,ticks,ids) in sorted(controls.items()):
            report['controls']+=1
            for index in (9,10,11):
                plan=str(ids[index])
                if plan.startswith('bootstrap:'):
                    report['bootstrap_source_support']+=1;continue
                item=plans.get(plan)
                if item is None or item.get('commit_tick') is None or item['commit_tick']>begin:
                    raise ValueError('Consumed reference lacks a confirmed earlier commit')
                for frame,_ in source_frame_support(ticks[index],item['decision_tick'],item['generation_frames']):
                    source=resolve(plan,frame)
                    if source is None:report['bootstrap_source_support']+=1
                    elif source[0] not in owners:report['previous_rollout_source_support']+=1
                    else:
                        owner=owners[source[0]]
                        if owner.identity['policy_version']!=group[0].identity['policy_version']:
                            raise ValueError('Mixed policy versions in execution-mask rollout')
                        selected[source[0]].add(source[1])
        for plan,row in owners.items():
            if 'ppo_action_mask' in row.metadata:raise ValueError('Executed-frame mask must be bound exactly once')
            mask=torch.zeros_like(row.free_mask)
            frames=sorted(selected[plan])
            if frames:
                mask[frames]=row.free_mask[frames]
                # 第f帧根位置累计之前各帧按各自heading旋转的xy增量；两者都是真实支持。
                mask[:max(frames),:2]=row.free_mask[:max(frames),:2]
                mask[:max(frames),3:9]=row.free_mask[:max(frames),3:9]
            if bool((mask & ~row.free_mask).any()):raise ValueError('PPO mask escaped sampled free coordinates')
            row.metadata['ppo_action_mask']=mask
            row.metadata['ppo_action_scope']=CONTRACT
            row.metadata['ppo_execution_frames']=frames
            report['sampled_free_coordinates']+=int(row.free_mask.sum())
            report['ppo_coordinates']+=int(mask.sum())
            report['actor_transitions']+=int(bool(mask.any()))
    report['coordinate_fraction']=(report['ppo_coordinates']/report['sampled_free_coordinates']
        if report['sampled_free_coordinates'] else 0.)
    return report
