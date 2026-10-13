"""Stage10 独立配置解析、训练规模推导与新旧身份隔离。

本模块只接受显式当前配置，自动从GPU数、环境数、连续决策数及完整链minibatch
推导rollout、optimizer次数、K8访问数和BC每步样本量，不继承160条或四次更新
限制。代理目标与joint诊断、coordinate/joint KL门槛分别定义，不能复用一个
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

VERSION='genmo.closedloop.stage10.continuous_horizon.v1'


def configuration(path,*,genmo_repo=None,gmt_repo=None):
    c=yaml.safe_load(Path(path).read_text())
    if c.get('stage10',{}).get('version')!=VERSION or 'base_config' in c:
        raise ValueError('New 当前 full configuration required')
    if genmo_repo:c['paths']['genmo_repo']=str(Path(genmo_repo).resolve())
    if gmt_repo:c['paths']['gmt_repo']=str(Path(gmt_repo).resolve())
    s=c['stage10'];t=s['training'];scale=s['scale'];world=s['distributed']['world_size']
    if 'effect_check_iterations' in s['evaluation']:
        raise ValueError('effect_check_iterations is unsupported; use actual interaction evaluation intervals')
    for key in ('eta', 'guidance_scale'):
        if not math.isfinite(t[key]) or t[key] < 0:
            raise ValueError(f'Invalid explicit policy setting {key}')
    if c['model']['guidance_scale'] != t['guidance_scale']:
        raise ValueError('model and training guidance_scale differ')
    if 'episode_seconds' in t:raise ValueError('physical_episode and ppo_rollout must be separate')
    seconds=s['physical_episode']['safety_limit_seconds']
    if seconds is not None and (not math.isfinite(seconds) or seconds<=0 or seconds%0.5):raise ValueError('Physical safety limit must lie on the decision grid')
    if s['ppo_rollout']['world_seconds']<=0 or s['ppo_rollout']['world_seconds']%0.5:raise ValueError('Rollout world window must lie on the decision grid')
    if s['ppo_rollout']['collection_contract']!='shared_world_sim_time.v1':raise ValueError('Current collection uses shared world simulation time')
    if 'decisions_per_environment' in scale:raise ValueError('Per-environment decision quotas are no longer supported')
    horizon=s['generation_horizon']
    if type(horizon['frames'])is not int or horizon['frames'] not in (60,120) or horizon['fps']!=30:raise ValueError('Actual GENMO horizon must be 60/120 frames at 30Hz')
    t['generation_frames']=horizon['frames']
    mc=s['critic'].get('complete_mc',{})
    artifact=s['critic'].get('preheat_artifact')
    if artifact and (not mc.get('enabled') or set(artifact)!={'path','sha256'}
            or not isinstance(artifact['path'],str) or not Path(artifact['path']).is_absolute()
            or not isinstance(artifact['sha256'],str) or len(artifact['sha256'])!=64):
        raise ValueError('Explicit matching complete-MC Critic initialization artifact required')
    if mc.get('enabled'):
        if type(mc['samples_per_source'])is not int or mc['samples_per_source']<1 or type(mc['epochs'])is not int or mc['epochs']<1:
            raise ValueError('Complete MC supervision requires positive real episode count and epochs')
        if not mc['mix_candidates'] or any(not math.isfinite(v) or not 0<=v<=1 for v in mc['mix_candidates']):raise ValueError('Invalid MC/TD mixing candidates')
        if set(mc['training_seeds']) & set(mc['validation_seeds']):raise ValueError('MC validation seeds must be independent')
        for value in mc.get('quality_gate',{}).values():
            if not math.isfinite(value) or not 0<=value<1:raise ValueError('Invalid independent MC quality threshold')
    n=c['runtime']['num_envs'];decisions=math.ceil(s['ppo_rollout']['world_seconds']/.5);chains=n*decisions*world
    for k,v in [('world',world),('environments',n),('decisions',decisions),('optimizer chains',scale['actor_minibatch_chains']),
                ('Actor microbatch',t['denoising_microbatch']),('KL microbatch',t['kl_microbatch'])]:
        if type(v)is not int or v<1:raise ValueError(f'{k} must be positive integer')
    if world!=8 or s['distributed']['backend']!='nccl' or s['distributed']['collection']!='all_ranks':
        raise ValueError('当前 production requires eight synchronous GPU ranks')
    if t['denoising_steps']!=20 or t['ppo_steps_per_chain'] not in (8,20) or t['ppo_epochs']!=2:
        raise ValueError('Full20 behavior and explicit Uniform8/Full20 two-epoch PPO required')
    if len(t['std_schedule'])!=20 or any(not math.isfinite(x) or x<=0 for x in t['std_schedule']):
        raise ValueError('Actual noise schedule must define twenty positive finite floors')
    if c['runtime']['timing_contract']!='modeled_deployment.v3' or c['timing']['control_ticks']!=12 or c['timing']['physics_ticks']!=3:
        raise ValueError('Keep calibrated deployment clock and 50Hz/200Hz four physical substeps')
    if t['objective_logprob_reduction'] not in ('joint_sum','free_coordinate_mean'):
        raise ValueError('Unknown explicit PPO proxy objective')
    if not 0<t['ppo_clip']<1 or not math.isfinite(t['actor_lr']) or t['actor_lr']<=0:
        raise ValueError('Invalid independently calibrated Actor learning rate/clip')
    steps=t['ppo_epochs']*math.ceil(chains/scale['actor_minibatch_chains'])
    bc_total=t['bc_samples_per_rollout']
    if type(bc_total)is not int or bc_total<1:
        raise ValueError('BC total must be positive; actual ragged optimizer steps determine its allocation')
    t.update(rollout_upper_steps=chains,rollout_upper_steps_per_rank=n*decisions,
        actor_minibatch_internal_transitions=scale['actor_minibatch_chains']*20,
        max_actor_optimizer_steps=steps,bc_batch=bc_total//steps,
        critic_steps=s['critic']['epochs']*math.ceil(chains/s['critic']['global_batch']),
        critic_batch=s['critic']['global_batch'],critic_lr=s['critic']['learning_rate'])
    c['stage9']=dict(t,reward=resolve_reward_config(s['reward']),seed=s['seed'],run_id='assigned_training_run',
        bc_data_root=c['paths']['data_root'],bc_distribution='all_ranks')
    if s['limits']['physics_steps']!=4*s['limits']['control_steps']:
        raise ValueError('Physical resource budget must preserve four substeps')
    maximum_steps=t['ppo_epochs']*math.ceil(n*(decisions+1)*world/scale['actor_minibatch_chains'])
    if bc_total<maximum_steps:raise ValueError('BC must cover every possible actual optimizer step')
    if s['limits']['optimizer_attempts']<s['limits']['accepted_iterations']*maximum_steps:
        raise ValueError('Optimizer attempt budget must cover the complete declared workload')
    if c['runtime']['generation_batch']>n:raise ValueError('No duplicated environments to pad effective GENMO batches')
    s['derived']=dict(world_size=world,real_chains=chains,local_chains=n*decisions,planned_actor_steps=steps,
        local_ppo_internal_visits=n*decisions*t['ppo_steps_per_chain']*t['ppo_epochs'],
        local_full_kl_visits=n*decisions*20,bc_per_step=t['bc_batch'],bc_per_rollout=bc_total)
    s['derived'].update(count_semantics='nominal_capacity_only_actual_manifest_controls_optimization',
        maximum_actor_steps=maximum_steps)
    return c
