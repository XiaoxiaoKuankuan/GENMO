"""当前连续世界Actor优化配置及实际参数身份。

当前训练只接受四步独立坐标平均代理Clipped PPO，共享一个上层GAE优势。显式绑定
共同基础学习率和固定模块倍率、全局链批量、Epoch、独立坐标KL与可选联合尾部保护。
历史联合配置仅供只读识别；旧joint-score函数仅留作数学参考。新门限仍待真实rollout
校准，未标定联合尾部必须显式仅诊断，不能沿用旧目标的门限或声称物理安全。
本模块只校验配置，不启动训练；物理收益仍由真实完整歌曲执行结果判断。
"""
from __future__ import annotations

import hashlib
import json
import math

ACTOR_OBJECTIVE='stepwise_coordinate_mean_clipped_ppo'
from ..robofft import OBJECTIVE as ROBOFFT_OBJECTIVE
ACTOR_OBJECTIVES=(ACTOR_OBJECTIVE,ROBOFFT_OBJECTIVE)
OPTIMIZATION_SCHEMA='genmo.stage2.stepwise_coordinate_mean_actor.v1'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def optimization_binding(stage):
    t=stage['training']
    from ..robofft import OBJECTIVE
    if t['actor_objective']==OBJECTIVE:
        from ..robofft.configuration import optimization_binding as binding
        return binding(stage)
    from .actor_optimizer import validate_lr_multipliers
    multipliers=validate_lr_multipliers(t)
    return dict(schema=('genmo.stage2.full_chain_optimization_binding.v1' if t['actor_objective']=='full_chain_clipped_ppo'
        else 'genmo.stage2.stepwise_coordinate_mean_optimization_binding.v1'),
        actor_lr=t['actor_lr'],actor_lr_control=t['actor_lr_control'],actor_trainable_scope=t.get('actor_trainable_scope','all'),
        actor_optimizer=dict(type='AdamW',parameter_groups=1 if multipliers is None else 'named_modules',
            **({'lr_multipliers':multipliers} if multipliers is not None else {}),
            betas=[.9,.999],eps=1e-8,weight_decay=0.),
        ppo_epochs=t['ppo_epochs'],global_minibatch_chains=stage['scale']['actor_minibatch_chains'],
        objective=t['actor_objective'],joint_reduction=t['objective_logprob_reduction'],ppo_clip=t['ppo_clip'],
        gamma_denoising=t['gamma_denoising'],gradient_clip=t['grad_clip_norm'],
        bc_samples=t['bc_samples_per_rollout'],bc_min_global_batch=t['bc_min_global_batch'],bc_controller=stage['bc_gradient'],
        chain_clip_saturation=t['max_active_clip_fraction'],
        hard_kl={key:value for key,value in stage['kl'].items() if key!='calibration'},
        **({'soft_chain_joint_kl':t['kl_soft_stop_chain_joint']} if t['actor_objective']=='full_chain_clipped_ppo'
            else {'soft_mean_coordinate_kl':t['kl_soft_stop_mean_coordinate']}),rollout=stage['ppo_rollout'],
        trust_region=t['trust_region'],
        gamma_upper=t['gamma_upper'],lambda_upper=t['lambda_upper'],dataset=stage['dataset'],
        mc_start_distribution=stage['critic']['complete_mc']['start_distribution'],
        run_contract=stage['actor_optimization'],
        **({'trainable_denoising_steps':list(range(9,19))} if t.get('ppo_steps_per_chain')==10 else {}))


def _validate_legacy_configuration(config,maximum_steps):
    s=config['stage10'];t=s['training']
    from .actor_scope import SCOPES
    if t.get('actor_trainable_scope','all') not in SCOPES:raise ValueError('Unknown Actor trainable scope')
    if (s.get('actor_optimization')!={'schema':'genmo.stage2.full_chain_actor.v1'}
            or 'actor_experiment' in s or t['actor_objective']!='full_chain_clipped_ppo'
            or t['ppo_epochs']<1 or maximum_steps<1
            or s['scale']['actor_minibatch_chains']<1
            or t['objective_logprob_reduction']!='joint_sum' or t['gamma_denoising']!=1.
            or t['actor_lr_control']!='joint_kl_backtracking' or not math.isfinite(t['actor_lr']) or t['actor_lr']<=0):
        raise ValueError('Explicit full-chain joint PPO with one undiscounted GAE and fixed Actor LR ratios required')
    trust=t['trust_region']
    trust.setdefault('minimum_trial_lr_ratio',.0625)
    if (set(trust)-{'objective_aware','objective_tolerance','objective_search_attempts'}!={'maximum_lr','growth','margin','max_attempts','minimum_trial_lr_ratio'}
            or trust['maximum_lr']!=t['actor_lr'] or not 1<=trust['growth']<=2
            or not 0<trust['margin']<1 or type(trust['max_attempts'])is not int
            or not 1<=trust['max_attempts']<=10
            or not 0<trust['minimum_trial_lr_ratio']<=1):
        raise ValueError('Explicit bounded joint-KL backtracking required')
    if s['limits']['optimizer_attempts']<s['limits']['accepted_iterations']*maximum_steps*trust['max_attempts']:
        raise ValueError('Budget must cover every planned optimizer update')
    if s['kl']['reduction']!='chain_joint':raise ValueError('Trust-region must use complete-chain joint KL')
    for key in ('mean_coordinate','maximum_step_mean_coordinate','p95_coordinate','mean_joint'):
        if s['kl'].get(key) is not None:raise ValueError('Coordinate/per-step KL is diagnostic only for full-chain PPO')
    for key in ('chain_joint_mean','chain_joint_p95','chain_joint_p99','chain_joint_max'):
        value=s['kl'][key]
        if not math.isfinite(value) or value<=0:raise ValueError('Explicit positive finite chain KL limits required')
    bounds=[s['kl'][key] for key in ('chain_joint_mean','chain_joint_p95','chain_joint_p99','chain_joint_max')]
    if bounds!=sorted(bounds) or not 0<t['kl_soft_stop_chain_joint']<=bounds[0]:
        raise ValueError('Soft chain KL and hard mean/tail thresholds must be ordered')
    if not 0<t['max_active_clip_fraction']<=1:
        raise ValueError('Explicit active-clip finite-experiment limit required')
    if s['kl'].get('maximum_active_clip')!=t['max_active_clip_fraction']:
        raise ValueError('Full-rollout active-clip acceptance must match mini-batch guard')
    expected=dict(status='candidate_requires_joint_calibration',scope='four_trainable_steps_only',
        optimization_sha256=digest(optimization_binding(s)))
    if s['kl']['calibration'].get('optimization_sha256')=='auto':
        s['kl']['calibration']['optimization_sha256']=expected['optimization_sha256']
    if s['kl']['calibration']!=expected:
        raise ValueError('Optimization binding differs; regenerate the identity for the actual configuration')
    from .reference_credit import CREDIT_CONTRACT
    if (s['run_control']['max_training_iterations']>2000 or s['limits']['accepted_iterations']>2000
            or s['ppo_rollout']['world_seconds']!=8 or t['lambda_upper']!=.98
            or not math.isfinite(t['gamma_upper']) or not 0<t['gamma_upper']<=1
            or s['storage']['checkpoint_interval_iterations']!=100
            or s['ppo_rollout']['credit_contract']!=CREDIT_CONTRACT
            or s['critic']['complete_mc']['start_distribution']!='match_training_reset.v1'):
        raise ValueError('Current 8s/.98/2000-iteration continuous-world training contract required')
    # 有限验证可以更频繁地完整评估；评估间隔不改变概率核、奖励或价值目标。
    evaluation_interval=s['evaluation']['every_actor_iterations']
    if (type(evaluation_interval)is not int or evaluation_interval<1
            or s['critic']['complete_mc']['every_actor_iterations']!=100):
        raise ValueError('Positive complete-song evaluation interval and MC every 100 required')
    return optimization_binding(s)


def validate_optimization_configuration(config,maximum_steps):
    """历史配置仅可只读解析；当前训练只允许显式逐步坐标平均代理合同。"""
    s=config['stage10'];t=s['training']
    from ..robofft import OBJECTIVE
    if t.get('actor_objective')==OBJECTIVE:
        from ..robofft.configuration import validate_configuration
        return validate_configuration(config,maximum_steps)
    if t.get('actor_objective')=='full_chain_clipped_ppo':
        return _validate_legacy_configuration(config,maximum_steps)
    if (s.get('actor_optimization')!={'schema':OPTIMIZATION_SCHEMA} or 'actor_experiment' in s
            or t.get('actor_objective')!=ACTOR_OBJECTIVE or t.get('actor_trainable_scope','all')!='all'
            or t['objective_logprob_reduction']!='free_coordinate_mean' or t['gamma_denoising']!=1.
            or t['actor_lr_control']!='coordinate_kl_backtracking' or t.get('bc_enabled') is not False
            or t['ppo_epochs']<1 or maximum_steps<1 or s['scale']['actor_minibatch_chains']<1
            or not math.isfinite(t['actor_lr']) or t['actor_lr']<=0):
        raise ValueError('Explicit stepwise coordinate-mean PPO, all Transformer layers and disabled BC required')
    trust=t['trust_region'];trust.setdefault('minimum_trial_lr_ratio',.0625)
    if (set(trust)-{'objective_aware','objective_tolerance','objective_search_attempts'}!=
            {'maximum_lr','growth','margin','max_attempts','minimum_trial_lr_ratio'}
            or trust['maximum_lr']!=t['actor_lr'] or not 1<=trust['growth']<=2
            or not 0<trust['margin']<1 or type(trust['max_attempts'])is not int
            or not 1<=trust['max_attempts']<=10 or not 0<trust['minimum_trial_lr_ratio']<=1):
        raise ValueError('Explicit bounded multi-group coordinate-KL backtracking required')
    if s['kl']['reduction']!='free_coordinate_mean' or s['kl'].get('mean_joint') is not None:
        raise ValueError('Use separate coordinate-mean controls and optional chain-joint tail diagnostics')
    for key in ('mean_coordinate','maximum_step_mean_coordinate','p95_coordinate'):
        value=s['kl'].get(key)
        if value is None or not math.isfinite(value) or value<=0:raise ValueError('Explicit positive coordinate KL limits required')
    if (not 0<t['kl_soft_stop_mean_coordinate']<=s['kl']['mean_coordinate'] or
            any(s['kl'][key]<s['kl']['mean_coordinate'] for key in ('maximum_step_mean_coordinate','p95_coordinate'))):
        raise ValueError('Coordinate KL mean, soft stop and step/tail bounds must be ordered')
    bounds=[s['kl'].get('chain_joint_'+name) for name in ('mean','p95','p99','max')]
    tail_status=s['kl']['calibration'].get('joint_tail_status')
    if any(v is not None for v in bounds):
        if (any(v is None or not math.isfinite(v) or v<=0 for v in bounds) or bounds!=sorted(bounds)
                or tail_status not in ('candidate_explicit_tail_limits_not_physically_validated','calibrated_empirical')):
            raise ValueError('Joint KL protection needs explicit independently calibrated/candidate tail limits')
    elif tail_status!='uncalibrated_diagnostic_only':
        raise ValueError('Absent joint KL limits must explicitly declare uncalibrated diagnostic-only status')
    if not 0<t['max_active_clip_fraction']<=1 or s['kl'].get('maximum_active_clip')!=t['max_active_clip_fraction']:
        raise ValueError('Stepwise active clip guards must agree')
    expected=dict(status='candidate_requires_stepwise_calibration',scope=(
        'four_trainable_steps_only' if t['ppo_steps_per_chain']==4 else 'ten_trainable_steps_only'),
        joint_tail_status=tail_status,optimization_sha256=digest(optimization_binding(s)))
    if s['kl']['calibration'].get('optimization_sha256')=='auto':
        s['kl']['calibration']['optimization_sha256']=expected['optimization_sha256']
    if s['kl']['calibration']!=expected:
        raise ValueError('Stepwise optimization binding differs from the actual configuration')
    from .reference_credit import CREDIT_CONTRACT
    if (s['limits']['optimizer_attempts']<s['limits']['accepted_iterations']*maximum_steps*trust['max_attempts']
            or s['run_control']['max_training_iterations']>2000 or s['limits']['accepted_iterations']>2000
            or s['ppo_rollout']['world_seconds']!=8 or t['lambda_upper']!=.98
            or not math.isfinite(t['gamma_upper']) or not 0<t['gamma_upper']<=1
            or s['storage']['checkpoint_interval_iterations']!=100
            or s['ppo_rollout']['credit_contract']!=CREDIT_CONTRACT
            or s['critic']['complete_mc']['start_distribution']!='match_training_reset.v1'
            or s['critic']['complete_mc']['every_actor_iterations']!=100
            or type(s['evaluation']['every_actor_iterations'])is not int or s['evaluation']['every_actor_iterations']<1):
        raise ValueError('Keep the existing continuous-world, Value, evaluation and resource contracts')
    return optimization_binding(s)


def require_current_actor_objective(settings):
    """旧配置只读可识别，但不能从当前训练入口误启动旧联合PPO。"""
    if settings.get('actor_objective') not in ACTOR_OBJECTIVES:
        raise ValueError('Historical full-chain config is read-only; use stage2_stepwise_mean.yaml for Actor training')
