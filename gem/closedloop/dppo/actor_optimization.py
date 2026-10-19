"""当前连续世界Actor优化配置及实际参数身份。

默认Clipped DPPO支持完整四步、多全局minibatch和一次统一学习率；joint_score_pg
保留为显式单次全批量可选目标，只有它启用逐Adam回溯。两者共用Stage1 BC、真实
行为统计、GAE和整轮四步KL保护。绑定实际目标、LR、批量、BC、折扣和KL定义，
不把未启用的joint-score回溯参数计入DPPO实际优化合同，也不兼容旧Critic输入。
本模块只校验配置，不启动训练；物理收益仍由真实完整歌曲执行结果判断。
"""
from __future__ import annotations

import hashlib
import json
import math

ACTOR_OBJECTIVE='clipped_dppo'
ACTOR_OBJECTIVES=('clipped_dppo','joint_score_pg')
OPTIMIZATION_SCHEMA='genmo.stage2.continuous_actor.v2'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def optimization_binding(stage):
    t=stage['training']
    return dict(schema='genmo.stage2.actor_optimization_binding.v3',
        actor_lr=t['actor_lr'],actor_lr_control=t['actor_lr_control'],
        actor_optimizer=dict(type='AdamW',parameter_groups=1,betas=[.9,.999],eps=1e-8,weight_decay=0.),
        ppo_epochs=t['ppo_epochs'],global_minibatch_chains=stage['scale']['actor_minibatch_chains'],
        objective=t['actor_objective'],proxy_reduction=t['objective_logprob_reduction'],ppo_clip=t['ppo_clip'],
        gamma_denoising=t['gamma_denoising'],gradient_clip=t['grad_clip_norm'],
        bc_samples=t['bc_samples_per_rollout'],bc_controller=stage['bc_gradient'],
        backtracking=t['optimizer_backtracking'] if t['actor_objective']=='joint_score_pg' else None,
        hard_kl={key:value for key,value in stage['kl'].items() if key!='calibration'},
        soft_kl=t['kl_soft_stop_coordinate'],rollout=stage['ppo_rollout'],
        gamma_upper=t['gamma_upper'],lambda_upper=t['lambda_upper'],dataset=stage['dataset'],
        mc_start_distribution=stage['critic']['complete_mc']['start_distribution'],
        run_contract=stage['actor_optimization'])


def validate_optimization_configuration(config,maximum_steps):
    s=config['stage10'];t=s['training']
    if (s.get('actor_optimization')!={'schema':OPTIMIZATION_SCHEMA}
            or 'actor_experiment' in s or t['actor_objective'] not in ACTOR_OBJECTIVES
            or t['ppo_epochs']<1 or maximum_steps<1
            or s['scale']['actor_minibatch_chains']<1
            or t['objective_logprob_reduction']!='free_coordinate_mean'
            or t['actor_lr_control']!='fixed' or not math.isfinite(t['actor_lr']) or t['actor_lr']<=0):
        raise ValueError('Explicit clipped_dppo or single-step joint-score with one uniform Actor LR required')
    if t['actor_objective']=='joint_score_pg' and (t['ppo_epochs']!=1 or maximum_steps!=1):
        raise ValueError('joint-score requires exactly one full-rollout optimizer step')
    bt=t['optimizer_backtracking']
    if (set(bt)!={'max_attempts','factor'} or type(bt['max_attempts']) is not int
            or not 1<=bt['max_attempts']<=10 or bt['factor']!=.5):
        raise ValueError('Bounded optimizer-step backtracking required')
    attempts=bt['max_attempts'] if t['actor_objective']=='joint_score_pg' else maximum_steps
    if s['limits']['optimizer_attempts']<s['limits']['accepted_iterations']*attempts:
        raise ValueError('Budget must include every planned Actor step and any joint-score retries')
    for key in ('mean_coordinate','maximum_step_mean_coordinate','p95_coordinate'):
        if not math.isfinite(s['kl'][key]) or s['kl'][key]<=0:
            raise ValueError('Four-step KL limits must be positive and finite')
    for key in ('chain_joint_mean','chain_joint_p95','chain_joint_max'):
        value=s['kl'][key]
        if value is not None and (not math.isfinite(value) or value<=0):
            raise ValueError('Empirical joint-chain KL gates must be null or positive finite')
    expected=dict(status='not_physically_validated',scope='four_trainable_steps_only',
        optimization_sha256=digest(optimization_binding(s)))
    if s['kl']['calibration']!=expected:
        raise ValueError('Optimization binding differs; regenerate the identity for the actual configuration')
    from .reference_credit import CREDIT_CONTRACT
    # 保存频率由training_config校验正整数，不属于Actor优化合同，不能写死为100轮。
    if (s['run_control']['max_training_iterations']>2000 or s['limits']['accepted_iterations']>2000
            or s['ppo_rollout']['world_seconds']!=8 or t['lambda_upper']!=.98 or t['gamma_upper']!=.99
            or s['ppo_rollout']['credit_contract']!=CREDIT_CONTRACT
            or s['critic']['complete_mc']['start_distribution']!='match_training_reset.v1'):
        raise ValueError('Current 8s/.98/2000-iteration continuous-world training contract required')
    if any(container['every_actor_iterations']!=100 for container in
            (s['evaluation'],s['critic']['complete_mc'])):
        raise ValueError('Complete-song evaluation and MC run every 100 accepted iterations')
    return optimization_binding(s)
