"""当前第二阶段唯一Actor优化合同与可追踪参数身份。

采用一次全局on-policy joint-score策略梯度，加原Stage1 BC和有界Adam回溯。所有
可训练Actor参数共享学习率；四个随机步骤的联合score按实际有效链数归一化。
本模块绑定真实LR、批量、BC、GAE、回溯、四步KL及连续世界时钟，不再接受旧
坐标PPO、学习率扫描、负对照或二十轮试验配置，也不转换旧配置和checkpoint。

KL是实际更新保护，配置身份和通过数值检查都不代表物理学习收益已获证明。
完整歌曲评估、MC和安全停止仍由唯一训练入口负责；此处不启动GPU或仿真。
"""
from __future__ import annotations

import hashlib
import json
import math

ACTOR_OBJECTIVE='joint_score_pg'
OPTIMIZATION_SCHEMA='genmo.stage2.continuous_joint_score.v1'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def optimization_binding(stage):
    t=stage['training']
    return dict(schema='genmo.stage2.actor_optimization_binding.v2',
        actor_lr=t['actor_lr'],actor_lr_control=t['actor_lr_control'],
        actor_optimizer=dict(type='AdamW',parameter_groups=1,betas=[.9,.999],eps=1e-8,weight_decay=0.),
        ppo_epochs=t['ppo_epochs'],global_minibatch_chains=stage['scale']['actor_minibatch_chains'],
        objective=t['actor_objective'],proxy_reduction=t['objective_logprob_reduction'],ppo_clip=t['ppo_clip'],
        gamma_denoising=t['gamma_denoising'],gradient_clip=t['grad_clip_norm'],
        bc_samples=t['bc_samples_per_rollout'],bc_controller=stage['bc_gradient'],
        backtracking=t['optimizer_backtracking'],
        hard_kl={key:value for key,value in stage['kl'].items() if key!='calibration'},
        soft_kl=t['kl_soft_stop_coordinate'],rollout=stage['ppo_rollout'],
        gamma_upper=t['gamma_upper'],lambda_upper=t['lambda_upper'],dataset=stage['dataset'],
        mc_start_distribution=stage['critic']['complete_mc']['start_distribution'],
        run_contract=stage['actor_optimization'])


def validate_optimization_configuration(config,maximum_steps):
    s=config['stage10'];t=s['training']
    if (s.get('actor_optimization')!={'schema':OPTIMIZATION_SCHEMA}
            or 'actor_experiment' in s or t['actor_objective']!=ACTOR_OBJECTIVE
            or t['ppo_epochs']!=1 or maximum_steps!=1
            or s['scale']['actor_minibatch_chains']!=16384
            or t['objective_logprob_reduction']!='free_coordinate_mean'
            or t['actor_lr_control']!='fixed' or not math.isfinite(t['actor_lr']) or t['actor_lr']<=0):
        raise ValueError('Only the current uniform-LR single-step joint-score configuration is supported')
    bt=t['optimizer_backtracking']
    if (set(bt)!={'max_attempts','factor'} or type(bt['max_attempts']) is not int
            or not 1<=bt['max_attempts']<=10 or bt['factor']!=.5):
        raise ValueError('Bounded optimizer-step backtracking required')
    if s['limits']['optimizer_attempts']<s['limits']['accepted_iterations']*bt['max_attempts']:
        raise ValueError('Budget must include every rejected Adam attempt')
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
    if (s['run_control']['max_training_iterations']>2000 or s['limits']['accepted_iterations']>2000
            or s['ppo_rollout']['world_seconds']!=8 or t['lambda_upper']!=.98 or t['gamma_upper']!=.99
            or s['storage']['checkpoint_interval_iterations']!=100
            or s['ppo_rollout']['credit_contract']!=CREDIT_CONTRACT
            or s['critic']['complete_mc']['start_distribution']!='match_training_reset.v1'):
        raise ValueError('Current 8s/.98/2000-iteration continuous-world training contract required')
    if any(container['every_actor_iterations']!=100 for container in
            (s['evaluation'],s['critic']['complete_mc'])):
        raise ValueError('Complete-song evaluation and MC run every 100 accepted iterations')
    return optimization_binding(s)
