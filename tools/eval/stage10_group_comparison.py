"""第二阶段三模型完整音乐对照的取样、真实轨迹归档与价值诊断。

按数据清单的全局音乐 group_id 去重，每组事先选择最长的有效配对样本，长度
相同时按 sample_id 排序；不根据模型表现挑选样本。val/test 分别建计划，同组
多个舞者或裁剪不增加独立样本数，多个随机种子始终绑定同一音乐组。

逐50Hz控制步保存实际Isaac qpos、参考关节、速度、关节限制和奖励分项；这些
数组供MuJoCo运动学回放及配对原因分析，不把生成动作冒充真实执行。完整MC
回报沿真实控制步反向折扣，奖励已经积分，不再乘dt；行政截断不产生MC真值。
评估不创建优化器，不保留无用途的完整扩散链，不修改训练和恢复身份。
"""
from __future__ import annotations

import copy
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from gem.closedloop.dppo.evaluation import _digest, _publish_json, build_evaluation_tasks


def representatives(samples, maximum=None):
    groups = defaultdict(list)
    for sample in samples:
        groups[sample['group_id']].append(sample)
    chosen = [sorted(rows, key=lambda s: (-int(s['row']['num_frames']), s['row']['sample_id']))[0]
              for _, rows in sorted(groups.items())]
    if maximum is not None:
        if type(maximum) is not int or maximum < 1:
            raise ValueError('Positive explicit group cap required')
        chosen = chosen[:maximum]
    return chosen


def music_group_plan(catalog, split, seeds, *, maximum=None, seconds=None):
    parent = build_evaluation_tasks(catalog, split=split, eval_count='all', seeds=seeds,
                                    episode_seconds=seconds)
    selected = [sample for rows in catalog.samples[split].values()
                for sample in representatives(rows, maximum)]
    keys = {(s['dataset'], s['row']['sample_id']) for s in selected}
    plan = copy.deepcopy(parent)
    plan.pop('plan_sha256')
    plan.update(parent_all_sample_plan_sha256=parent['plan_sha256'],
                selection_unit='unique_global_music_group',
                selection_rule='all_groups_longest_available_paired_sample_tie_lexical_before_model_results',
                requested_eval_count='all_music_groups' if maximum is None else 'explicit_smoke_group_cap',
                selected_sample_count=len(selected), selected_unique_group_count=len(selected),
                selected_unique_audio_count=len({s['row']['source_audio_sha256'] for s in selected}),
                selected_group_counts_by_source={source: len(representatives(rows, maximum))
                    for source, rows in catalog.samples[split].items()},
                complete_music_scope='full_available_paired_sample_not_uncropped_original_recording',
                tasks=[task for task in parent['tasks'] if (task['dataset'], task['sample_id']) in keys])
    plan['task_count'] = len(plan['tasks'])
    plan['plan_sha256'] = _digest(plan)
    return plan


def partition_by_music(plan, world_size):
    """最长音乐组优先分配，两个种子一起分片；三种模型复用完全相同分片。"""
    groups = defaultdict(list)
    for task in plan['tasks']:
        groups[task['sample']['group_id']].append(task)
    assignments, load = [[] for _ in range(world_size)], [0.] * world_size
    for group, tasks in sorted(groups.items(), key=lambda item: (-sum(t['remaining_music_seconds'] for t in item[1]), item[0])):
        rank = min(range(world_size), key=lambda i: (load[i], i))
        assignments[rank].extend(t['task_id'] for t in tasks)
        load[rank] += sum(t['remaining_music_seconds'] for t in tasks)
    result = []
    for rank, ids in enumerate(assignments):
        child = copy.deepcopy(plan)
        child.pop('plan_sha256')
        child['tasks'] = [t for t in plan['tasks'] if t['task_id'] in set(ids)]
        child.update(task_count=len(ids), parent_plan_sha256=plan['plan_sha256'],
                     group_partition=dict(rank=rank, world_size=world_size, expected_music_seconds=load[rank]))
        child['selected_sample_count'] = len(ids) // len(plan['seeds'])
        child['selected_unique_group_count'] = child['selected_sample_count']
        child['plan_sha256'] = _digest(child)
        result.append(child)
    assert sorted(t['task_id'] for p in result for t in p['tasks']) == sorted(t['task_id'] for t in plan['tasks'])
    return result


def control_mc_returns(reward_segments, gamma_upper=.99, event_rewards=None):
    """每0.5秒gamma_upper；输入是已积分的每控制步reward，返回各上层状态MC。"""
    if not 0 < float(gamma_upper) <= 1:
        raise ValueError('Invalid discount')
    events = [0.] * len(reward_segments) if event_rewards is None else list(event_rewards)
    if len(events) != len(reward_segments) or not np.isfinite(events).all():
        raise ValueError('One finite boundary event per upper transition required')
    gamma = gamma_upper ** (1 / 25)
    value, result = 0., []
    for rewards, event in reversed(list(zip(reward_segments,events))):
        rewards = np.asarray(rewards, dtype=np.float64)
        if rewards.ndim != 1 or not np.isfinite(rewards).all():
            raise ValueError('Finite one-dimensional integrated rewards required')
        value = float(np.dot(gamma ** np.arange(len(rewards)), rewards)
                      + gamma ** max(len(rewards)-1,0) * event + gamma ** len(rewards) * value)
        result.append(value)
    return np.asarray(result[::-1], dtype=np.float64)


class EpisodeTrajectorySink:
    """每episode发布紧凑实测数组；所有reward/control/tick数量必须一致。"""
    def __init__(self, output, critic, *, joint_names, gmt_joint_names, gamma_upper=.99):
        self.output, self.critic = Path(output), critic
        self.output.mkdir(parents=True, exist_ok=False)
        self.joint_names, self.gmt_joint_names = list(joint_names), list(gmt_joint_names)
        self.gamma_upper = gamma_upper
        self.active = None
        self.arrays, self.upper, self.rewards, self.records = defaultdict(list), [], [], []

    def __call__(self, task, transition):
        if self.active not in (None, task['task_id']):
            raise RuntimeError('An unfinished episode cannot be silently discarded')
        self.active = task['task_id']
        rows = transition.metadata['execution_trace']
        details = transition.metadata['reward_details']
        rewards = transition.rewards.detach().cpu().numpy().copy()
        if len(rows) != len(details) or len(rows) != len(rewards):
            raise ValueError('Physical trace and reward counts differ')
        device = next(self.critic.parameters()).device
        with torch.no_grad():
            value = float(self.critic({k: v.to(device) for k, v in transition.context.items()},
                torch.tensor([transition.metadata['remaining_music_seconds']], device=device)).item())
        self.upper.append(dict(tick=transition.control_tick_begin, prediction=value,
            remaining_music_seconds=transition.metadata['remaining_music_seconds'], controls=len(rows),
            event_reward=float(transition.metadata.get('event_reward',0.)),
            prefix_frames=int(transition.context['known_qpos30_mask'][0].any(-1).sum())))
        self.rewards.append(rewards)
        for row, detail, reward in zip(rows, details, rewards):
            if row['completed_physics_steps'] != 4 or not row['state_valid']:
                raise ValueError('Every physical control requires four real trusted substeps')
            values = dict(tick=row['tick'], qpos=row['actual_qpos'], joint_velocity=row['actual_joint_vel'],
                actual_joint_gmt=row['actual_joint_pos_gmt'], reference_joint_gmt=row['reference']['joint_pos'],
                joint_position_target=row['joint_position_target'], applied_action=row['applied_action'], reward=reward,
                activity=detail['activity']['actual_activity_rad_s'], target_activity=detail['activity']['target_activity_rad_s'],
                activity_gate=detail['activity']['gate'])
            for name, item in detail['components'].items():
                values['reward_'+name] = item['integrated_reward']
                values['score_'+name] = item['score']
            for name, value in values.items():
                self.arrays[name].append(np.array(value, copy=True))
            if 'hard_limits' not in self.arrays:
                diagnostic = row['physical_diagnostics']
                self.arrays['hard_limits'] = [np.array(diagnostic['joint_pos_limits_rad'], copy=True)]
                limits = detail['components']['joint_limit']['raw']['safe_limits_rad']
                self.arrays['safe_limits'] = [np.asarray(limits)]
        if not (transition.terminated or transition.truncated):
            return
        complete = transition.terminated and not transition.truncated
        arrays = {k: np.asarray(v) for k, v in self.arrays.items()}
        if len(arrays['tick']) > 1 and not np.all(np.diff(arrays['tick']) == 12):
            raise ValueError('Real 50Hz trajectory has missing or duplicate controls')
        arrays.update(upper_tick=np.array([u['tick'] for u in self.upper]),
            value_prediction=np.array([u['prediction'] for u in self.upper]),
            remaining_music_seconds=np.array([u['remaining_music_seconds'] for u in self.upper]),
            prefix_frames=np.array([u['prefix_frames'] for u in self.upper]),
            upper_event_reward=np.array([u['event_reward'] for u in self.upper]),
            mc_return=control_mc_returns(self.rewards,self.gamma_upper,[u['event_reward'] for u in self.upper])
                if complete else np.full(len(self.upper),np.nan))
        path = self.output / (task['task_id']+'.npz')
        with path.open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        from gem.closedloop.dppo.training_checkpoint import file_sha256
        record = dict(schema='genmo.stage10.actual_trajectory_comparison.v1',task_id=task['task_id'],
            task=task, trajectory_file=path.name, sha256=file_sha256(path), fps=50, quaternion_convention='wxyz',
            qpos_order='mujoco_native', joint_names=self.joint_names, gmt_joint_names=self.gmt_joint_names,
            complete_episode=complete, end_reason=transition.reason, terminated=transition.terminated,
            truncated=transition.truncated, actual_control_steps=len(arrays['tick']),
            actual_physics_steps=4*len(arrays['tick']), control_reward_sum=float(arrays['reward'].sum()),
            zero_control_event_reward_sum=float(arrays['upper_event_reward'].sum()),
            value_scope='current_checkpoint_critic_true_MC_only_for_complete_episode',
            motion_scope='actual_Isaac_execution_kinematic_MuJoCo_replay_not_MuJoCo_dynamics')
        _publish_json(path.with_suffix('.json'),record)
        self.records.append(record)
        self.active=None; self.arrays=defaultdict(list); self.upper=[]; self.rewards=[]
