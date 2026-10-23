"""Stage10 独立配置解析、训练规模推导与新旧身份隔离。

本模块只接受显式当前配置，自动从GPU数、环境数、连续决策数及完整链minibatch
推导rollout、optimizer次数、四个随机训练步访问数和BC每步样本量，不继承160条或四次更新
限制。联合链PPO和联合链KL为训练目标，coordinate KL仅作辅助诊断，不能复用一个
无量纲数字。初始参数是待有限标定的候选，任何算法或数值合同变化新建run。

Stage1与GMT均绑定真实SHA和统计量，全部环境时序与奖励保留旧明确配置。检查
CPU/GPU资源、模型加载和完整数据审计由入口执行；这里不启动训练、不扩预算，
不接受旧checkpointschema或隐式配置恢复。日常/严格证据仅改变明确数据生命周期。
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import yaml

from .rewards import resolve_reward_config

VERSION='genmo.closedloop.stage2.full_chain_joint.v1'


def _source_counts(value):
    """显式四来源配额，不把bool或缺来源的字典当有效任务数。"""
    from .full_dataset import SOURCES
    return (type(value)is int and value>0 or isinstance(value,dict) and set(value)==set(SOURCES)
            and all(type(n)is int and n>0 for n in value.values()))


def configuration(path,*,genmo_repo=None,gmt_repo=None):
    c=yaml.safe_load(Path(path).read_text())
    if c.get('stage10',{}).get('version')!=VERSION or 'base_config' in c:
        raise ValueError('New 当前 full configuration required')
    if genmo_repo:c['paths']['genmo_repo']=str(Path(genmo_repo).resolve())
    if gmt_repo:c['paths']['gmt_repo']=str(Path(gmt_repo).resolve())
    from .rolling import validate_schedule, generation_capacity, rolling_enabled
    validate_schedule(c)
    s=c['stage10'];t=s['training'];scale=s['scale'];world=s['distributed']['world_size']
    precision=s['precision']
    if (precision['mode']!='native_fp32' or precision['attention']!='sdpa_math'
            or precision['linear_execution']!='stable_tf32x3' or precision['compile']
            or not precision['precise_output'] or precision.get('cuda_graph',False)
            or precision.get('weight_gradient','direct_joint_gemm')!='direct_joint_gemm'):
        raise ValueError('Only the validated stable_tf32x3 FP32 execution is supported')
    from .time_sampling import TIME_SAMPLING_CONTRACT, MOTION_RESET_CONTRACT
    dataset = s['dataset']
    if (dataset.get('random_start') is not True or dataset.get('sampling_contract') != TIME_SAMPLING_CONTRACT
            or dataset.get('initialization') != MOTION_RESET_CONTRACT or dataset.get('split') != 'train'):
        raise ValueError('Actor training requires explicit random time sampling and paired motion initialization')
    bin_seconds, standing = dataset.get('time_bin_seconds'), dataset.get('standing_start_probability')
    if (type(bin_seconds) not in (int, float) or not math.isfinite(bin_seconds) or bin_seconds <= 0
            or not math.isclose(bin_seconds * 30, round(bin_seconds * 30), abs_tol=1e-9)
            or type(standing) not in (int, float) or not math.isfinite(standing) or not 0 <= standing < 1):
        raise ValueError('Explicit positive 30Hz time bins and standing probability in [0,1) required')
    control=s['run_control']
    if set(control)!={'max_training_iterations'}:
        raise ValueError('Only max_training_iterations is supported; total walltime limits are removed')
    maximum=control['max_training_iterations']
    if type(maximum)is not int or maximum<1 or maximum>s['limits']['accepted_iterations']:
        raise ValueError('Cumulative PPO iteration limit must be positive and within the resource budget')
    checkpoint_interval=s['storage'].get('checkpoint_interval_iterations')
    if checkpoint_interval is not None and (type(checkpoint_interval)is not int or checkpoint_interval<1):
        raise ValueError('Checkpoint iteration interval must be a positive integer')
    if c['runtime'].get('evidence_level','full') not in ('full','training'):
        raise ValueError('runtime.evidence_level must be full or training')
    interval=c['runtime'].get('reward_audit_interval_controls',100)
    if type(interval) is not int or interval<1:raise ValueError('reward_audit_interval_controls must be a positive integer')
    if any(key.startswith('vector_boundary_wait') for key in c['runtime']):
        raise ValueError('Retired vector_boundary_wait settings are not used by world-time collection')
    if 'effect_check_iterations' in s['evaluation']:
        raise ValueError('effect_check_iterations is unsupported; use actual interaction evaluation intervals')
    eval_envs=s['evaluation'].get('envs_per_rank',1)
    mc_envs=s['critic']['complete_mc'].get('envs_per_rank',eval_envs)
    if type(mc_envs)is not int or not 1<=mc_envs<=64:
        raise ValueError('Complete MC requires 1..64 independent slots per GPU')
    if type(eval_envs)is not int or not 1<=eval_envs<=64 or type(s['evaluation'].get('persistent_world',False))is not bool:
        raise ValueError('Evaluation requires 1..64 slots in one world and explicit persistent_world boolean')
    if s['evaluation'].get('start_mode','beginning') not in ('beginning','center','mixed') or type(s['evaluation'].get('unique_music',False))is not bool:
        raise ValueError('Invalid evaluation music selection configuration')
    selection=s['evaluation'].get('samples_per_source',4)
    train_monitor=s['evaluation'].get('purpose')=='training_progress'
    if train_monitor and s['evaluation'].get('original_split')!='train':
        raise ValueError('Training progress evaluation requires explicit original train music')
    if (s['evaluation'].get('split','val') not in (('train',) if train_monitor else ('val','test')) or not
            (_source_counts(selection) or selection=='all_unique' and s['evaluation'].get('unique_music',False))):
        raise ValueError('Held-out evaluation requires val/test; training progress requires explicit train purpose, and positive source counts or all_unique music selection')
    if s['evaluation'].get('episode_seconds') is None and s['evaluation'].get('start_mode','beginning')!='beginning':
        raise ValueError('Complete song evaluation cannot start in the middle')
    for key in ('eta', 'guidance_scale'):
        if not math.isfinite(t[key]) or t[key] < 0:
            raise ValueError(f'Invalid explicit policy setting {key}')
    if c['model']['guidance_scale'] != t['guidance_scale']:
        raise ValueError('model and training guidance_scale differ')
    if 'episode_seconds' in t:raise ValueError('physical_episode and ppo_rollout must be separate')
    seconds=s['physical_episode']['safety_limit_seconds']
    if seconds is not None and (not math.isfinite(seconds) or seconds<=0 or seconds%0.5):raise ValueError('Physical safety limit must lie on the decision grid')
    if s['physical_episode'].get('time_limit_bootstrap',True) is not True:
        raise ValueError('Physical time limits truncate sampling and bootstrap the pre-reset Value')
    if s['ppo_rollout']['world_seconds']<=0 or s['ppo_rollout']['world_seconds']%0.5:raise ValueError('Rollout world window must lie on the decision grid')
    if s['ppo_rollout']['collection_contract']!='shared_world_sim_time.v1':raise ValueError('Current collection uses shared world simulation time')
    if 'decisions_per_environment' in scale:raise ValueError('Per-environment decision quotas are no longer supported')
    horizon=s['generation_horizon']
    if type(horizon['frames'])is not int or horizon['frames'] not in (60,120) or horizon['fps']!=30:raise ValueError('Actual GENMO horizon must be 60/120 frames at 30Hz')
    t['generation_frames']=horizon['frames']
    mc=s['critic'].get('complete_mc',{})
    if s['critic']['complete_mc'].get('start_distribution','beginning') not in ('beginning','match_training_reset.v1'):
        raise ValueError('Unknown complete-MC reset distribution')
    artifact=s['critic'].get('preheat_artifact')
    if artifact and (not mc.get('enabled') or set(artifact)!={'path','sha256'}
            or not isinstance(artifact['path'],str) or not Path(artifact['path']).is_absolute()
            or not isinstance(artifact['sha256'],str) or len(artifact['sha256'])!=64):
        raise ValueError('Explicit matching complete-MC Critic initialization artifact required')
    if mc.get('enabled'):
        count=mc.get('training_samples_per_source',mc['samples_per_source'])
        if type(count)is not int or count<1:raise ValueError('MC training_samples_per_source must be positive')
        if not _source_counts(mc['samples_per_source']) or type(mc['epochs'])is not int or mc['epochs']<1:
            raise ValueError('Complete MC supervision requires positive real episode count and epochs')
        if not mc['mix_candidates'] or any(not math.isfinite(v) or not 0<=v<=1 for v in mc['mix_candidates']):raise ValueError('Invalid MC/TD mixing candidates')
        if set(mc['training_seeds']) & set(mc['validation_seeds']):raise ValueError('MC validation seeds must be independent')
        for value in mc.get('quality_gate',{}).values():
            if not math.isfinite(value) or not 0<=value<1:raise ValueError('Invalid independent MC quality threshold')
        refit=mc.get('refit_if_ev_below')
        if refit is not None and (not isinstance(refit,(int,float)) or not 0<refit<=1):
            raise ValueError('refit_if_ev_below must be null or in (0, 1]')
    n=c['runtime']['num_envs'];decisions=generation_capacity(c,s['ppo_rollout']['world_seconds']);chains=n*decisions*world
    for k,v in [('world',world),('environments',n),('decisions',decisions),('optimizer chains',scale['actor_minibatch_chains']),
                ('Actor microbatch',t['denoising_microbatch']),('KL microbatch',t['kl_microbatch'])]:
        if type(v)is not int or v<1:raise ValueError(f'{k} must be positive integer')
    if world!=8 or s['distributed']['backend']!='nccl' or s['distributed']['collection']!='all_ranks':
        raise ValueError('当前 production requires eight synchronous GPU ranks')
    from .denoising_contract import trainable_steps
    active_steps=trainable_steps(t['frozen_prefix_steps']);step_count=len(active_steps)
    if (t['denoising_steps']!=20 or t['ppo_step_strategy']!=f'intermediate{step_count}' or
            t['ppo_steps_per_chain']!=step_count or t['ppo_epochs'] not in (1,2,3,5)):
        raise ValueError('Explicit 15+4+1 or 9+10+1 configuration must match all trainable steps')
    if len(t['std_schedule'])!=20 or t['std_schedule'][-1]!=0. or any(
            not math.isfinite(x) or x<=0 for x in t['std_schedule'][:-1]):
        raise ValueError('Nineteen positive stochastic floors and an explicit zero deterministic final are required')
    if s['critic']['type']!='asymmetric_tracking_temporal' or 'freeze_encoders' in s['critic']:
        raise ValueError('Independent trainable tracking history Critic required')
    if not 0<s['critic']['history_learning_rate']<=s['critic']['learning_rate']:
        raise ValueError('Invalid separate history GRU learning rate')
    if s['kl']['calibration']['scope']!=('four_trainable_steps_only' if step_count==4 else 'ten_trainable_steps_only'):
        raise ValueError('KL scope must identify the configured trainable Gaussian interval')
    if not 0<c['timing']['calibration_reservation_seconds']<=.5:
        raise ValueError('Calibration needs an explicit bounded reservation, not a measured delay')
    if c['runtime']['timing_contract']!='modeled_deployment.v3' or c['timing']['control_ticks']!=12 or c['timing']['physics_ticks']!=3:
        raise ValueError('Keep calibrated deployment clock and 50Hz/200Hz four physical substeps')
    if t['objective_logprob_reduction']!='joint_sum' or t['gamma_denoising']!=1.:
        raise ValueError('Full-chain joint PPO requires sum and one undiscounted upper advantage')
    if not 0<t['ppo_clip']<1 or not math.isfinite(t['actor_lr']) or t['actor_lr']<=0:
        raise ValueError('Invalid independently calibrated Actor learning rate/clip')
    if t.get('advantage_control','none')!='none':
        raise ValueError('Retired advantage-control experiments are not supported')
    if type(t.get('gradient_snr_diagnostic',False)) is not bool:
        raise ValueError('gradient_snr_diagnostic must be boolean')
    diagnostic_interval=s['checks'].get('actor_diagnostics_every',100)
    if type(diagnostic_interval) is not int or diagnostic_interval<1:
        raise ValueError('actor_diagnostics_every must be a positive iteration interval')
    if t['actor_lr_control']!='joint_kl_backtracking' or t['kl_rejection']!='rollback_abort':
        raise ValueError('AdamW with fixed LR ratios, joint-KL backtracking and final whole-rollout rejection required')
    if any(key in t for key in ('actor_lr_bounds','kl_target_coordinate','kl_control_metric')):
        raise ValueError('Retired adaptive-LR settings are not supported')
    steps=t['ppo_epochs']*math.ceil(chains/scale['actor_minibatch_chains'])
    for key,default in (('bc_enabled',True),('audit_full_rollout_objective',False)):
        if type(t.get(key,default)) is not bool:raise ValueError(key+' must be boolean')
    bc_total=t['bc_samples_per_rollout']
    minimum_bc=t['bc_min_global_batch']
    if t.get('bc_enabled',True) and (type(minimum_bc)is not int or minimum_bc<256):
        raise ValueError('Production BC requires at least 256 global samples per Adam step')
    bc_total=max(bc_total,steps*minimum_bc) if t.get('bc_enabled',True) else 0
    t['bc_samples_per_rollout']=bc_total
    if type(bc_total)is not int or bc_total<0 or t.get('bc_enabled',True) and bc_total<1:
        raise ValueError('BC total must be positive; actual ragged optimizer steps determine its allocation')
    t.update(rollout_upper_steps=chains,rollout_upper_steps_per_rank=n*decisions,
        actor_minibatch_internal_transitions=scale['actor_minibatch_chains']*step_count,
        max_actor_optimizer_steps=steps,bc_batch=bc_total//steps,
        critic_steps=s['critic']['epochs']*math.ceil(chains/s['critic']['global_batch']),
        critic_batch=s['critic']['global_batch'],critic_lr=s['critic']['learning_rate'])
    c['stage9']=dict(t,reward=resolve_reward_config(s['reward']),seed=s['seed'],run_id='assigned_training_run',
        bc_data_root=c['paths']['data_root'],bc_distribution='all_ranks')
    if s['limits']['physics_steps']!=4*s['limits']['control_steps']:
        raise ValueError('Physical resource budget must preserve four substeps')
    # 共享世界所有reset/决策都对齐25控制步网格；每格每环境至多一条Actor转移。
    # collector再次按真实片段校验此上限，失败/预热只减少有效链，不能偷偷多做PG更新。
    maximum_steps=t['ppo_epochs']*math.ceil(chains/scale['actor_minibatch_chains'])
    if t.get('bc_enabled',True) and bc_total<maximum_steps:raise ValueError('BC must cover every possible actual optimizer step')
    if s['limits']['optimizer_attempts']<(maximum if rolling_enabled(c) else s['limits']['accepted_iterations'])*maximum_steps:
        raise ValueError('Optimizer attempt budget must cover the complete declared workload')
    from .actor_optimization import validate_optimization_configuration
    validate_optimization_configuration(c, maximum_steps)
    if c['runtime']['generation_batch']>n:raise ValueError('No duplicated environments to pad effective GENMO batches')
    s['derived']=dict(world_size=world,real_chains=chains,local_chains=n*decisions,planned_actor_steps=steps,
        local_ppo_internal_visits=n*decisions*t['ppo_steps_per_chain']*t['ppo_epochs'],
        local_trainable_kl_visits=n*decisions*step_count,bc_per_step=t['bc_batch'],bc_per_rollout=bc_total)
    s['derived'].update(count_semantics='nominal_capacity_only_actual_manifest_controls_optimization',
        maximum_actor_steps=maximum_steps)
    return c
