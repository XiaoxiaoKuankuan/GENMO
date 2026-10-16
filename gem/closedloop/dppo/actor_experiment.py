"""高学习率有限实验的配置身份和门禁，不借用旧低学习率的标定结论。

实验绑定实际统一LR、全局链批量、epoch、目标、BC、Adam超参数、回溯及软硬KL。
允许原门槛的1/3/10倍独立探索，但明确不代表物理质量通过，累计最多20轮；每轮
只有一次on-policy提交。该入口只授权服务器2有限实验，恢复仍严格比较完整运行
身份。正式发布需要包含新优化合同的v2物理证据，旧v1证据不可为本实验背书。
"""
from __future__ import annotations
import math

from .kl_calibration import digest


def optimization_binding(stage):
    t = stage['training']
    return dict(schema='genmo.stage2.actor_optimization_binding.v1',
        actor_lr=t['actor_lr'], actor_lr_control=t.get('actor_lr_control','fixed'),
        actor_optimizer=dict(type='AdamW', parameter_groups=1, betas=[.9,.999], eps=1e-8, weight_decay=0.),
        ppo_epochs=t['ppo_epochs'], global_minibatch_chains=stage['scale']['actor_minibatch_chains'],
        objective=t.get('actor_objective','coordinate_dppo'),
        proxy_reduction=t['objective_logprob_reduction'], ppo_clip=t['ppo_clip'],
        gamma_denoising=t['gamma_denoising'], gradient_clip=t['grad_clip_norm'],
        bc_samples=t['bc_samples_per_rollout'], bc_controller=stage['bc_gradient'],
        backtracking=t.get('optimizer_backtracking'),
        hard_kl={k:v for k,v in stage['kl'].items() if k!='calibration'},
        soft_kl=t['kl_soft_stop_coordinate'])


def validate_experiment_configuration(config, maximum_steps):
    s = config['stage10']; t = s['training']; trial = s.get('actor_experiment', {})
    if (trial.get('schema') != 'genmo.stage2.uniform_lr_experiment.v2'
            or trial.get('scope') != 'server2_finite_not_formal'
            or t['actor_objective'] not in ('coordinate_dppo','joint_score_pg')
            or t['ppo_epochs'] != 1 or maximum_steps != 1
            or s['scale']['actor_minibatch_chains'] != 16384
            or t['objective_logprob_reduction'] != 'free_coordinate_mean'
            or t['actor_lr_control'] != 'fixed'
            or not 0 < t['actor_lr'] <= 1e-5):
        raise ValueError('Explicit uniform-LR single on-policy full-batch finite experiment required')
    if s['run_control']['max_training_iterations'] > 20 or s['limits']['accepted_iterations'] > 20:
        raise ValueError('Exploratory high-LR experiments cannot automatically start long training')
    bt = t.get('optimizer_backtracking', {})
    if (set(bt) != {'max_attempts','factor'} or type(bt['max_attempts']) is not int
            or not 1 <= bt['max_attempts'] <= 10 or bt['factor'] != .5):
        raise ValueError('Bounded optimizer-step backtracking required')
    if s['limits']['optimizer_attempts'] < s['limits']['accepted_iterations']*bt['max_attempts']:
        raise ValueError('Budget must include every rejected Adam attempt')
    multiplier = trial.get('kl_multiplier')
    if multiplier not in (1,3,10):
        raise ValueError('Only explicitly separated 1x/3x/10x KL experiments are defined')
    for k,v in dict(mean_coordinate=.0003, maximum_step_mean_coordinate=.0006, p95_coordinate=.0007).items():
        if not math.isclose(s['kl'][k],v*multiplier,rel_tol=1e-12,abs_tol=0.):
            raise ValueError('Experimental KL thresholds differ from the named multiplier')
    for key in ('chain_joint_mean','chain_joint_p95','chain_joint_max'):
        value=s['kl'].get(key)
        if value is not None and (not math.isfinite(value) or value<=0):
            raise ValueError('Empirical chain-joint KL gate must be null or positive finite')
    if s['kl']['calibration'] != dict(status='exploratory_not_physically_validated',
            scope='four_trainable_steps_only', optimization_sha256=digest(optimization_binding(s))):
        raise ValueError('Experimental optimization binding differs; old calibration cannot authorize new LR/objective')
    for container in (s['evaluation'], s['critic']['complete_mc']):
        every=container.get('every_actor_iterations')
        if every is not None and every not in (2,4):
            raise ValueError('Finite experiments must evaluate at an explicit 2 or 4 accepted-iteration interval')
    return optimization_binding(s)
