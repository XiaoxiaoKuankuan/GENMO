"""当前连续世界Actor优化配置及实际参数身份。

接受配置指定四步或十步的逐去噪步骤Clipped PPO，每步独立裁剪并共享上层GAE，loss对步数取平均。显式绑定
共同基础学习率和固定模块倍率、全局链批量、Epoch、BC批量下限、单步条件KL阈值。旧坐标
代理训练和单次joint-score训练入口退役；joint-score函数仅留作数学梯度参考。
单步KL初始阈值是待真实rollout标定的候选，不能沿用坐标平均阈值或声称物理安全。
本模块只校验配置，不启动训练；物理收益仍由真实完整歌曲执行结果判断。
"""
from __future__ import annotations

import hashlib
import json
import math

ACTOR_OBJECTIVE='denoising_step_clipped_ppo'
ACTOR_OBJECTIVES=('denoising_step_clipped_ppo',)
OPTIMIZATION_SCHEMA='genmo.stage2.denoising_step_actor.v1'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def optimization_binding(stage):
    t=stage['training']
    from .actor_optimizer import validate_lr_multipliers
    multipliers=validate_lr_multipliers(t)
    from .denoising_contract import trainable_steps
    return dict(schema='genmo.stage2.denoising_step_optimization_binding.v1',
        candidate_acceptance='finite_KL_only_first_qualified.v1',
        **({'trainable_gaussian_steps':list(trainable_steps(t['frozen_prefix_steps'],t['denoising_steps']))} if t['frozen_prefix_steps']!=15 else {}),
        **({'ppo_action_scope':t['ppo_action_scope']} if t.get('ppo_action_scope') is not None else {}),
        actor_lr=t['actor_lr'],actor_lr_control=t['actor_lr_control'],actor_trainable_scope=t.get('actor_trainable_scope','all'),
        actor_optimizer=dict(type='AdamW',parameter_groups=1 if multipliers is None else 'named_modules',
            **({'lr_multipliers':multipliers} if multipliers is not None else {}),
            betas=[.9,.999],eps=1e-8,weight_decay=0.),
        ppo_epochs=t['ppo_epochs'],global_minibatch_chains=stage['scale']['actor_minibatch_chains'],
        objective=t['actor_objective'],joint_reduction=t['objective_logprob_reduction'],step_reduction=t['objective_step_reduction'],ppo_clip=t['ppo_clip'],
        gamma_denoising=t['gamma_denoising'],gradient_clip=t['grad_clip_norm'],
        bc_samples=t['bc_samples_per_rollout'],bc_min_global_batch=t['bc_min_global_batch'],bc_controller=stage['bc_gradient'],
        step_clip_saturation=t['max_active_clip_fraction'],
        hard_kl={key:value for key,value in stage['kl'].items() if key!='calibration'},
        soft_denoising_step_kl=t['kl_soft_stop_denoising_step'],rollout=stage['ppo_rollout'],
        trust_region=t['trust_region'],
        gamma_upper=t['gamma_upper'],lambda_upper=t['lambda_upper'],dataset=stage['dataset'],
        mc_start_distribution=stage['critic']['complete_mc']['start_distribution'],
        run_contract=stage['actor_optimization'])


def validate_optimization_configuration(config,maximum_steps):
    s=config['stage10'];t=s['training']
    from .actor_scope import SCOPES
    if t.get('actor_trainable_scope','all') not in SCOPES:raise ValueError('Unknown Actor trainable scope')
    if (s.get('actor_optimization')!={'schema':OPTIMIZATION_SCHEMA}
            or 'actor_experiment' in s or t['actor_objective'] not in ACTOR_OBJECTIVES
            or t['ppo_epochs']<1 or maximum_steps<1
            or s['scale']['actor_minibatch_chains']<1
            or t.get('objective_step_reduction')!='mean' or t['objective_logprob_reduction']!='joint_sum' or t['gamma_denoising']!=1.
            or t['actor_lr_control']!='step_kl_backtracking' or not math.isfinite(t['actor_lr']) or t['actor_lr']<=0):
        raise ValueError('Explicit step-mean denoising PPO with one undiscounted GAE and fixed Actor LR ratios required')
    trust=t['trust_region']
    trust.setdefault('minimum_trial_lr_ratio',.0625)
    if (set(trust)!={'maximum_lr','growth','margin','max_attempts','minimum_trial_lr_ratio'}
            or trust['maximum_lr']!=t['actor_lr'] or not 1<=trust['growth']<=2
            or not 0<trust['margin']<1 or type(trust['max_attempts'])is not int
            or not 1<=trust['max_attempts']<=10
            or not 0<trust['minimum_trial_lr_ratio']<=1):
        raise ValueError('Explicit bounded joint-KL backtracking required')
    if s['limits']['optimizer_attempts']<s['run_control']['max_training_iterations']*maximum_steps*trust['max_attempts']:
        raise ValueError('Budget must cover every planned optimizer update')
    if s['kl']['reduction']!='denoising_step':raise ValueError('Trust-region must use individual denoising-step joint KL')
    for key in ('mean_coordinate','maximum_step_mean_coordinate','p95_coordinate','mean_joint'):
        if s['kl'].get(key) is not None:raise ValueError('Coordinate-average KL is diagnostic only for step DPPO')
    for key in ('denoising_step_mean','denoising_step_p95','denoising_step_p99','denoising_step_max'):
        value=s['kl'][key]
        if not math.isfinite(value) or value<=0:raise ValueError('Explicit positive finite chain KL limits required')
    bounds=[s['kl'][key] for key in ('denoising_step_mean','denoising_step_p95','denoising_step_p99','denoising_step_max')]
    if bounds!=sorted(bounds) or not 0<t['kl_soft_stop_denoising_step']<=bounds[0]:
        raise ValueError('Soft step KL and hard mean/tail thresholds must be ordered')
    if not 0<t['max_active_clip_fraction']<=1:
        raise ValueError('Explicit active-clip finite-experiment limit required')
    if s['kl'].get('maximum_active_clip')!=t['max_active_clip_fraction']:
        raise ValueError('Full-rollout active-clip acceptance must match mini-batch guard')
    from .denoising_contract import calibration_scope
    expected=dict(status='candidate_requires_step_calibration',scope=calibration_scope(t['ppo_steps_per_chain']),
        optimization_sha256=digest(optimization_binding(s)))
    if s['kl']['calibration'].get('optimization_sha256')=='auto':
        s['kl']['calibration']['optimization_sha256']=expected['optimization_sha256']
    if s['kl']['calibration']!=expected:
        raise ValueError('Optimization binding differs; regenerate the identity for the actual configuration')
    from .reference_credit import CREDIT_CONTRACT
    if (s['run_control']['max_training_iterations']>2000 or s['limits']['accepted_iterations']>2000
            or s['ppo_rollout']['world_seconds']!=8 or t['lambda_upper']!=.98
            or not math.isfinite(t['gamma_upper']) or not 0<t['gamma_upper']<=1
            or s['ppo_rollout']['credit_contract']!=CREDIT_CONTRACT
            or s['critic']['complete_mc']['start_distribution']!='match_training_reset.v1'):
        raise ValueError('Current 8s/.98/2000-iteration continuous-world training contract required')
    # 有限验证可以更频繁地完整评估；评估间隔不改变概率核、奖励或价值目标。
    evaluation_interval=s['evaluation']['every_actor_iterations']
    if (type(evaluation_interval)is not int or evaluation_interval<1
            or s['critic']['complete_mc']['every_actor_iterations']!=100):
        raise ValueError('Positive complete-song evaluation interval and MC every 100 required')
    return optimization_binding(s)
