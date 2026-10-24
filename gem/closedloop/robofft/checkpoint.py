"""显式 Stage2 权重导入和 RoboFFT 优化变体的完整模型 warm-start。

输入必须是完整、可识别的第二阶段 checkpoint，绑定相同 Stage1/统计量/机器人资产。
仅验证并载入可训练 Actor；冻结前段/末段仍从共同 Stage1 源重建，Critic、两个 Adam、
rollout、概率缓存、MC RNG 和进度全部由新实验重新初始化。源码 SHA 仅在来源记录中
保留，不作为权重导入门禁。普通 --resume 使用 RoboFFT 自身完整 schema 和恢复协议。
独立的优化变体入口只允许改变 Actor 优化超参数、时间权重和并行采集数量；核对
行为核、价值目标、资产及物理契约后保留 Actor/Critic 与两个 Adam 的动量。新目录
重新计轮、重新采集、重新固定 MC/时间权重并重新评估，绝不复用旧 rollout 或 RNG。
显式未来Critic短实验只继承旧Critic共享权重，新增分支必须零投影、Critic Adam重建；
输入输出初值等同旧Critic，但不继承新结构的MC验收结论，不触发完整轨迹采集。
用户授权的正式入口只从相同未来Critic结构导入两模型和两个Adam，允许增加验证
与更新约束；源MC验收标记按事实保留，重新计200轮目标，不冒充原实验完整恢复。
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


def initialize_optimization_variant(path, *, identity, actor, critic, actor_optimizer, critic_optimizer,
                                    future_critic_trial=False, future_critic_formal=False):
    """仅同一 RoboFFT 行为/价值目标 warm-start；所有门禁先于任何模型写入。"""
    payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    expected_schema=CHECKPOINT_SCHEMA if future_critic_trial else schema_for_identity(identity)
    if payload.get('schema') != expected_schema:
        raise ValueError('Optimization variant requires a complete RoboFFT checkpoint')
    previous = payload['identity']
    if future_critic_formal and (future_critic_trial or
            previous.get('critic',{}).get('type')!='asymmetric_tracking_committed_temporal' or
            identity.get('critic',{}).get('type')!='asymmetric_tracking_committed_temporal' or
            identity['critic']['complete_mc']['enabled']):
        raise ValueError('Guarded formal initialization requires the same future Critic and no new complete MC')
    if future_critic_trial:
        if (previous['critic']['type']!='asymmetric_tracking_temporal'
                or identity['critic']['type']!='asymmetric_tracking_committed_temporal'
                or identity['critic']['complete_mc']['enabled'] or critic_optimizer.state):
            raise ValueError('Future Critic trial requires legacy shared weights, disabled new MC and fresh Critic Adam')
        old_critic,new_critic=(copy.deepcopy(v['critic']) for v in (previous,identity))
        for item in (old_critic,new_critic):
            item.pop('type');item['complete_mc'].pop('enabled')
        if old_critic!=new_critic:raise ValueError('Future Critic trial changes shared architecture or value fitting')
    else:
        old_critic,new_critic=(copy.deepcopy(v.get('critic')) for v in (previous,identity))
        if future_critic_formal:
            old_critic.pop('validation_guard',None);new_critic.pop('validation_guard',None)
        if old_critic!=new_critic:
            raise ValueError('RoboFFT variant changes behavior or value contract: critic')
    for key in ('assets', 'kernel', 'reward', 'timing', 'dataset', 'physical_episode',
                'ppo_rollout', 'generation_horizon', 'physical_configuration', 'data_catalog'):
        if previous.get(key) != identity.get(key):
            raise ValueError('RoboFFT variant changes behavior or value contract: ' + key)
    for section, allowed in (
        ('runtime', ('num_envs', 'generation_batch')+(
            ('future_reference_context','paired_state_audit') if future_critic_trial else ())),
        ('training', ('actor_lr', 'actor_lr_multipliers', 'ppo_epochs', 'ppo_clip',
                      'action_microbatch', 'robofft', 'rollout_upper_steps', 'rollout_upper_steps_per_rank',
                      'max_actor_optimizer_steps', 'critic_steps', 'actor_minibatch_mc_evaluations'))):
        old, new = copy.deepcopy(previous[section]), copy.deepcopy(identity[section])
        for key in allowed:
            old.pop(key, None); new.pop(key, None)
        if old != new:
            raise ValueError('RoboFFT variant changes protected contract: ' + section)
    old, new = (copy.deepcopy(v['training']['robofft']) for v in (previous, identity))
    for key in ('beta_neg', 'mc_microbatch', 'execution_weighting')+ (('update_guard',) if future_critic_formal else ()):
        old.pop(key, None); new.pop(key, None)
    if old != new or (not future_critic_formal and not payload['state'].get('mc_preheat_complete')):
        raise ValueError('RoboFFT variant requires unchanged MC estimator and a calibrated source Critic')
    _validate_model_state(actor, payload['actor'], 'RoboFFT variant Actor')
    critic_state=payload['critic']
    if future_critic_trial:
        current=critic.state_dict();new_keys={k for k in current if k.startswith('future_reference_encoder.')}
        if not getattr(critic,'future_reference',False) or set(critic_state)!=set(current)-new_keys:
            raise ValueError('Future Critic shared parameter mapping differs')
        for key in ('future_reference_encoder.projection.weight','future_reference_encoder.metadata.2.weight'):
            if torch.count_nonzero(current[key]):raise ValueError('Future Critic projection must initialize to exact zero')
        critic_state={**current,**critic_state}
    _validate_model_state(critic, critic_state, 'RoboFFT variant Critic')
    states = []
    optimizers=((actor_optimizer,'actor_optimizer'),) if future_critic_trial else (
        (actor_optimizer,'actor_optimizer'),(critic_optimizer,'critic_optimizer'))
    for optimizer, name in optimizers:
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
    critic.load_state_dict(critic_state, strict=True)
    actor_optimizer.load_state_dict(states[0])
    if not future_critic_trial:critic_optimizer.load_state_dict(states[1])
    return dict(kind=('explicit_robofft_future_critic_guarded_formal' if future_critic_formal else
                     'explicit_robofft_future_critic_bounded_trial' if future_critic_trial else
                      'explicit_robofft_optimization_variant_new_experiment'),
        path=str(Path(path).resolve()), sha256=file_sha256(path), source_iteration=payload['state']['iteration'],
        source_actor_updates=payload['state']['actor_updates'], source_files=previous.get('source_files'),
        source_version_is_informational=True, actor_and_adam_preserved=True,
        critic_and_adam_preserved=not future_critic_trial,critic_shared_weights_preserved=True,
        future_branch_zero_initialized=bool(future_critic_trial),critic_optimizer_fresh=bool(future_critic_trial),
        inherited_mc_preheat_for_unchanged_value_contract=bool(payload['state'].get('mc_preheat_complete')) and not future_critic_trial,
        independent_new_mc_validation=False,
        rollout_reused=False, rng_restored=False, counters_restarted=True, fresh_paired_evaluation_required=True)
