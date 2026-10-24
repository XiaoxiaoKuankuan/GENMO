"""正式训练的学习状态回退、整段留出划分和物理退化判据。

输入为当前模型及 Adam、真实 rollout 的 episode 身份和独立配对评估报告。
快照只保存可训练参数、buffer 和完整优化器状态，恢复时不回退物理交互、预算或
随机数；被拒绝的数据本轮结束后丢弃。Critic 按完整 episode 留出，避免相邻状态
分别进入训练和验证。固定旧自举标签仅限制预测漂移，不冒充完整 MC 真值。
物理判据使用同任务评估的新失败和音乐成簇奖励区间；没有触发退化不等于有效。
本模块不改变 Actor 输入、扩散采样、奖励、GMT 或半马尔可夫回报。
"""
from __future__ import annotations

import hashlib
import math
import json
import torch


def capture_learning(module, optimizer):
    from .training_checkpoint import immutable_cpu_snapshot
    return immutable_cpu_snapshot(dict(
        parameters={k: v for k, v in module.named_parameters() if v.requires_grad},
        buffers=dict(module.named_buffers()), optimizer=optimizer.state_dict()))


@torch.no_grad()
def restore_learning(module, optimizer, saved):
    current = {k: v for k, v in module.named_parameters() if v.requires_grad}
    if set(current) != set(saved['parameters']):
        raise ValueError('Learning rollback trainable parameter mapping changed')
    for key, value in current.items():
        value.copy_(saved['parameters'][key])
    for key, value in module.named_buffers():
        value.copy_(saved['buffers'][key])
    optimizer.load_state_dict(saved['optimizer'])
    optimizer.zero_grad(set_to_none=True)


def split_episodes(selected, manifest, identities, fraction):
    """相同 episode 的所有有效状态进入同一集合，排序不消耗训练 RNG。"""
    if not 0 < fraction < .5:
        raise ValueError('Critic holdout fraction must be between zero and one half')
    keys = {}
    for index in selected:
        item = manifest[index]
        identity = identities[(item['owner_rank'], item['local_index'])]
        keys[index] = (item['owner_rank'], identity['backend_session_id'],
                       identity.get('env_id', 0), identity['episode_id'])
    groups = sorted(set(keys.values()), key=lambda k: hashlib.sha256(repr(k).encode()).digest())
    if len(groups) < 2:
        raise ValueError('Critic validation requires at least two independent episodes')
    held = set(groups[:max(1, math.ceil(len(groups)*fraction))])
    return ([i for i in selected if keys[i] not in held],
            [i for i in selected if keys[i] in held])


def actor_guard_failures(metrics, references, settings):
    failures = {}
    key = 'ppo_zero_gradient_fraction'
    if metrics[key] > settings['max_ppo_zero_gradient_fraction']:
        failures[key] = metrics[key]
    for key in ('joint_change_rms_rad', 'near_joint_change_rms_rad', 'normalized_change_rms'):
        values = [(r['actions'], r.get(key)) for r in references if r.get(key) is not None]
        if not values:
            continue
        value = math.sqrt(sum(n*x*x for n, x in values)/sum(n for n, _ in values))
        if value > settings['max_'+key]:
            failures[key] = value
    return failures


def physical_guard(result, baseline, settings):
    paired = result.get('paired_baseline') or {}
    alert = paired.get('regression_alert') or {}
    relative = ((result['source_balanced_reward']-baseline['source_balanced_reward']) /
                max(abs(baseline['source_balanced_reward']), 1e-8))
    reasons = []
    if paired.get('new_failures', 0) > settings['max_new_failures']:
        reasons.append('new_physical_failure')
    if alert.get('triggered') and relative < -settings['max_relative_reward_drop']:
        reasons.append('paired_music_cluster_reward_regression')
    changes = {}
    for key, limit in settings.get('max_metric_increase', {}).items():
        old = baseline['physical_quality'][key]
        new = result['physical_quality'][key]
        if old is None or new is None or not math.isfinite(old) or not math.isfinite(new):
            raise ValueError('Required physical quality metric is unavailable: '+key)
        changes[key] = (new-old)/max(abs(old), 1e-8)
        if changes[key] > limit:
            reasons.append('physical_metric_regression:'+key)
    if 'max_activity_drop' in settings:
        old, new = (v['physical_quality']['actual_activity_rad_s'] for v in (baseline,result))
        if old is None or new is None:
            raise ValueError('Physical activity guard requires actual robot activity')
        changes['actual_activity_rad_s'] = (new-old)/max(abs(old), 1e-8)
        if changes['actual_activity_rad_s'] < -settings['max_activity_drop']:
            reasons.append('actual_motion_activity_reduced')
    return dict(passed=not reasons, reasons=reasons, relative_reward_change=relative,
                new_failures=paired.get('new_failures', 0), upper90=alert.get('upper90'),
                physical_metric_relative_changes=changes,
                scope='paired_short_physics_regression_guard_not_improvement_acceptance')


def physical_quality_metrics(report):
    """从无损50Hz证据计算非偏航误差、活动度和平滑度，不修改奖励或训练标签。"""
    from .compact_evidence import report_episodes
    keys = ('non_yaw_orientation_error_rad', 'root_height_error_m',
            'leg_joint_pos_squared_error', 'foot_height_squared_error')
    sums = {k: 0. for k in (*keys, 'actual_activity_rad_s')}
    seconds, jerks = 0., []
    for episode in report_episodes(report):
        for decision in episode['decisions']:
            for row in decision['controls']:
                dt = row['dt_s']; seconds += dt
                for key in keys:
                    sums[key] += row['raw_errors'][key]*dt
                sums['actual_activity_rad_s'] += row['activity']['actual_activity_rad_s']*dt
        motion = episode.get('motion_diagnostic') or {}
        if motion.get('joint_jerk_rms_rad_s3') is not None:
            jerks.append(motion['joint_jerk_rms_rad_s3'])
    if seconds <= 0:
        raise ValueError('Physical guard requires nonempty executed control evidence')
    return dict({k: v/seconds for k,v in sums.items()},
                joint_jerk_rms_rad_s3=sum(jerks)/len(jerks) if jerks else None)


def validate_formal_configuration(config):
    stage = config['stage10']
    critic = stage['critic'].get('validation_guard') or {}
    actor = stage['training']['robofft'].get('update_guard') or {}
    evaluation = stage['evaluation']
    physics = evaluation.get('physical_guard') or {}
    if (stage['critic']['complete_mc']['enabled'] or evaluation.get('persistent_world') is not False or
            evaluation['episode_seconds'] is None or not 0 < evaluation['episode_seconds'] <= 20 or
            not 1 <= evaluation.get('every_actor_iterations', 0) <= 5 or
            stage['run_control']['max_training_iterations'] != stage['limits']['accepted_iterations'] or
            not 1 <= stage['run_control']['max_training_iterations'] <= 200):
        raise ValueError('Formal guard requires <=200 iterations, no complete MC and fresh <=20s physics every <=5 iterations')
    required = dict(holdout_fraction=critic.get('holdout_fraction'),
                    max_fixed_mse_increase=critic.get('max_fixed_mse_increase'),
                    max_ppo_zero_gradient_fraction=actor.get('max_ppo_zero_gradient_fraction'),
                    max_relative_reward_drop=physics.get('max_relative_reward_drop'))
    required.update({key: actor.get(key) for key in ('max_joint_change_rms_rad',
                    'max_near_joint_change_rms_rad', 'max_normalized_change_rms')})
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 < v < 1 for v in required.values()):
        raise ValueError('Formal guard requires explicit finite positive subunit limits')
    if (not 0 < critic['holdout_fraction'] < .5 or type(critic.get('patience_epochs')) is not int or
            critic['patience_epochs'] < 1 or physics.get('max_new_failures') != 0):
        raise ValueError('Formal Critic episode holdout, patience and zero-new-failure guards required')


def fixed_reference_for_update(c):
    if hasattr(c, '_periodic_fixed_reference'):
        return c._periodic_fixed_reference[1]
    if not hasattr(c, '_critic_guard_reference'):
        from .training_checkpoint import file_sha256
        descriptor = json.loads((c.output/'fixed_diagnostic_reference.json').read_text())
        record, = [v['critic'] for v in descriptor['ranks'] if v['rank'] == c.distributed.rank]
        if file_sha256(record['path']) != record['sha256']:
            raise ValueError('Fixed Critic reference payload integrity mismatch')
        c._critic_guard_reference = torch.load(record['path'], map_location='cpu', weights_only=False)
    return c._critic_guard_reference


def rollback_quality_stop(c, report):
    """先保存失败候选，再恢复上一合格评估模型及两个 Adam；已用预算与计数不退款。"""
    from .parallel_support import root_call, local_call
    from .training import _poll_checkpoint, _snapshot
    from .training_checkpoint import file_sha256
    from .checkpoint import _validate_model_state, _validate_optimizer_state
    from .budget import atomic_json
    root_call(c.distributed, lambda: _poll_checkpoint(c, wait=True))
    safe = root_call(c.distributed, lambda: json.loads((c.output/'quality_last_safe.json').read_text()))
    def restore():
        path = c.output/safe['path']
        if file_sha256(path) != safe['sha256']:
            raise ValueError('Safe learning checkpoint integrity mismatch')
        payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        for name in ('actor', 'critic'):
            _validate_model_state(getattr(c, name), payload[name], 'physical rollback '+name)
            _validate_optimizer_state(getattr(c, name+'_optimizer'), payload[name+'_optimizer'])
        for name in ('actor', 'critic'):
            getattr(c, name).load_state_dict(payload[name])
            getattr(c, name+'_optimizer').load_state_dict(payload[name+'_optimizer'])
            getattr(c, name+'_optimizer').zero_grad(set_to_none=True)
    local_call(c.distributed, restore)
    c.state['policy_version'] += 1
    c.state['quality_stop'].update(restored_checkpoint=safe, physical_budget_and_attempts_retained=True,
                                   counters_are_cumulative_attempt_history=True)
    # 先发布停止标记：即使最终checkpoint写盘失败，监督器也不能绕过质量停止重启。
    root_call(c.distributed, lambda: atomic_json(c.output/'quality_stop.json', c.state['quality_stop']))
    _snapshot(c, 'quality_rollback_controlled_stop', wait=True)
    report['quality_stop'] = c.state['quality_stop']
