"""显式 Stage2 权重导入和 RoboFFT 优化变体的完整模型 warm-start。

输入必须是完整、可识别的第二阶段 checkpoint，绑定相同 Stage1/统计量/机器人资产。
仅验证并载入可训练 Actor；冻结前段/末段仍从共同 Stage1 源重建，Critic、两个 Adam、
rollout、概率缓存、MC RNG 和进度全部由新实验重新初始化。源码 SHA 仅在来源记录中
保留，不作为权重导入门禁。普通 --resume 使用 RoboFFT 自身完整 schema 和恢复协议。
独立的优化变体入口只允许改变 Actor 优化超参数、时间权重和并行采集数量；核对
行为核、价值目标、资产及物理契约后保留 Actor/Critic 与两个 Adam 的动量。新目录
重新计轮、重新采集、重新固定 MC/时间权重并重新评估，绝不复用旧 rollout 或 RNG。
"""
from __future__ import annotations

from pathlib import Path
import copy
import torch
from . import CHECKPOINT_SCHEMA
from ..dppo.training_checkpoint import SCHEMA, schema_for_identity, file_sha256
from ..dppo.checkpoint import _validate_model_state, _validate_optimizer_state


def initialize_actor_weights_only(path, actor, *, assets):
    payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    previous = payload.get('identity', {})
    if payload.get('schema') not in (SCHEMA, CHECKPOINT_SCHEMA, 'genmo.stage2.future_committed_critic.checkpoint.v1'):
        raise ValueError('Explicit weights-only initialization requires a recognized Stage2 checkpoint')
    for key in ('checkpoint', 'stats', 'kinematics'):
        if previous.get('assets', {}).get(key) != assets[key]:
            raise ValueError('Weights-only Stage1/normalization/robot asset mismatch: '+key)
    _validate_model_state(actor, payload['actor'], 'RoboFFT weights-only Actor')
    actor.load_state_dict(payload['actor'], strict=True)
    return dict(kind='explicit_actor_weights_only_new_robofft_experiment',
        path=str(Path(path).resolve()), sha256=file_sha256(path), source_schema=payload['schema'],
        source_iteration=payload.get('state', {}).get('iteration'),
        source_files=previous.get('source_files'), source_version_is_informational=True,
        optimizer_restored=False, critic_restored=False, rollout_reused=False, rng_restored=False)


def initialize_optimization_variant(path, *, identity, actor, critic, actor_optimizer, critic_optimizer):
    """仅同一 RoboFFT 行为/价值目标 warm-start；所有门禁先于任何模型写入。"""
    payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if payload.get('schema') != schema_for_identity(identity):
        raise ValueError('Optimization variant requires a complete RoboFFT checkpoint')
    previous = payload['identity']
    for key in ('assets', 'kernel', 'critic', 'reward', 'timing', 'dataset', 'physical_episode',
                'ppo_rollout', 'generation_horizon', 'physical_configuration', 'data_catalog'):
        if previous.get(key) != identity.get(key):
            raise ValueError('RoboFFT variant changes behavior or value contract: ' + key)
    for section, allowed in (
        ('runtime', ('num_envs', 'generation_batch')),
        ('training', ('actor_lr', 'actor_lr_multipliers', 'ppo_epochs', 'ppo_clip',
                      'action_microbatch', 'robofft', 'rollout_upper_steps', 'rollout_upper_steps_per_rank',
                      'max_actor_optimizer_steps', 'critic_steps', 'actor_minibatch_mc_evaluations'))):
        old, new = copy.deepcopy(previous[section]), copy.deepcopy(identity[section])
        for key in allowed:
            old.pop(key, None); new.pop(key, None)
        if old != new:
            raise ValueError('RoboFFT variant changes protected contract: ' + section)
    old, new = (copy.deepcopy(v['training']['robofft']) for v in (previous, identity))
    for key in ('beta_neg', 'mc_microbatch', 'execution_weighting'):
        old.pop(key, None); new.pop(key, None)
    if old != new or not payload['state'].get('mc_preheat_complete'):
        raise ValueError('RoboFFT variant requires unchanged MC estimator and a calibrated source Critic')
    _validate_model_state(actor, payload['actor'], 'RoboFFT variant Actor')
    _validate_model_state(critic, payload['critic'], 'RoboFFT variant Critic')
    states = []
    for optimizer, name in ((actor_optimizer, 'actor_optimizer'), (critic_optimizer, 'critic_optimizer')):
        saved = dict(payload[name], param_groups=[dict(g) for g in payload[name]['param_groups']])
        if len(saved['param_groups']) != len(optimizer.param_groups):
            raise ValueError('RoboFFT variant optimizer groups differ')
        for old_group, current in zip(saved['param_groups'], optimizer.param_groups):
            # 参数映射及 Adam 动量保留，显式新配置决定学习率。
            old_group['lr'] = current['lr']
            if 'lr_multiplier' in current:
                old_group['lr_multiplier'] = current['lr_multiplier']
        _validate_optimizer_state(optimizer, saved)
        states.append(saved)
    actor.load_state_dict(payload['actor'], strict=True)
    critic.load_state_dict(payload['critic'], strict=True)
    actor_optimizer.load_state_dict(states[0]); critic_optimizer.load_state_dict(states[1])
    return dict(kind='explicit_robofft_optimization_variant_new_experiment',
        path=str(Path(path).resolve()), sha256=file_sha256(path), source_iteration=payload['state']['iteration'],
        source_actor_updates=payload['state']['actor_updates'], source_files=previous.get('source_files'),
        source_version_is_informational=True, actor_and_adam_preserved=True, critic_and_adam_preserved=True,
        inherited_mc_preheat_for_unchanged_value_contract=True, independent_new_mc_validation=False,
        rollout_reused=False, rng_restored=False, counters_restarted=True, fresh_paired_evaluation_required=True)
