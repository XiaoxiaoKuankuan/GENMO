"""大规模Stage10的唯一采样/学习规模推导和授权预算缺口报告。

输入活动环境数、每环境真实决策数、完整链minibatch与PPO/Critic epochs，统一推导
local/global rollout、内部样本数和计划optimizer步数。缓存、manifest和writer读取
同一结果，禁止保留旧160条/4步限制或用内部微批改变实际训练工作量。
推导不会增加资源授权；500轮所需生成、控制/四子步、评估和存储按独立字段报告，
预算不足明确给出缺口。控制预算为部署profile支持的决策时长上界加最坏逐决策reset
warmup，实际账本仍按真实物理结算；不会因为配置规模增大自动降低采样总数。
"""
import math


def derive_training_scale(config):
    stage = config['stage10']
    scale = stage.get('scale')
    if scale is None: return
    runtime, train = config['runtime'], stage['training']
    world, envs = stage['distributed']['world_size'], runtime['num_envs']
    decisions, chains = scale['decisions_per_environment'], scale['actor_minibatch_chains']
    critic_epochs = scale['critic_epochs']
    for name, value in dict(world=world, envs=envs, decisions=decisions, chains=chains, critic_epochs=critic_epochs).items():
        if type(value) is not int or value < 1: raise ValueError(f'Invalid scale {name}')
    local, total = envs*decisions, world*envs*decisions
    derived = dict(rollout_upper_steps_per_rank=local, rollout_upper_steps=total,
        actor_minibatch_internal_transitions=chains*train['denoising_steps'],
        max_actor_optimizer_steps=train['ppo_epochs']*math.ceil(total/chains),
        critic_steps=critic_epochs*math.ceil(total/train['critic_batch']), critic_epochs=critic_epochs)
    for key, value in derived.items():
        if key in train and train[key] != value:
            raise ValueError(f'Conflicting manually specified {key}: {train[key]} != derived {value}')
        train[key] = value
    stage['derived_scale'] = dict(version='stage10.scale.v1', world_size=world, environments_per_rank=envs,
        decisions_per_environment=decisions, local_upper_transitions=local, global_upper_transitions=total,
        global_internal_transitions=total*train['denoising_steps'],
        ppo_local_internal_samples=local*train['denoising_steps']*train['ppo_epochs'],
        actor_minibatch_chains=chains, planned_actor_steps=derived['max_actor_optimizer_steps'],
        planned_critic_steps=derived['critic_steps'], global_bc_samples_per_iteration=derived['max_actor_optimizer_steps']*train['bc_batch'])


def budget_requirements(config, *, measured_evidence_bytes_per_transition=None):
    stage, train = config['stage10'], config['stage10']['training']
    rounds, count = stage['limits']['accepted_iterations'], train['rollout_upper_steps']
    evaluation = stage['evaluation']
    evaluation_rounds = sorted({0,rounds,*range(evaluation['every_iterations'],rounds+1,evaluation['every_iterations']),
        *(value for value in evaluation.get('effect_check_iterations',[]) if 0<value<=rounds)})
    evals = len(evaluation_rounds)
    tasks = 4*evaluation['samples_per_source']*len(evaluation['seeds'])*evals
    delay = max(config['timing']['deployment_profile']['samples_seconds'])
    controls_per_decision = math.ceil(max(.5, math.ceil(delay/.5)*.5)*50)
    envs = config['runtime']['num_envs']*stage['distributed']['world_size']
    calibration_generations=envs*(config['timing']['calibration_warmup']+config['timing']['calibration_samples'])
    # 明确采用正常无提前终止场景，不能把取profile最大时长的估计称为数学下界。
    requirements = dict(generations=rounds*count+tasks*math.ceil(evaluation['episode_seconds']/.5)+calibration_generations,
        optimizer_attempts=rounds*train['max_actor_optimizer_steps'],
        control_steps=rounds*count*controls_per_decision+tasks*math.ceil(evaluation['episode_seconds']*50)
            +(2*envs+tasks)*50)
    requirements['physics_steps'] = requirements['control_steps']*4
    worst = requirements['control_steps']+(rounds*count-envs)*50
    deficits = {key:max(0, value-stage['limits'][key]) for key,value in requirements.items()}
    return dict(planned_iterations=rounds, normal_no_failure_scenario_requirements=requirements,
        control_upper_with_reset_every_decision=worst, physics_upper_with_reset_every_decision=4*worst,
        authorization_limits=dict(stage['limits']), scenario_deficits=deficits,
        evaluation_rounds=evaluation_rounds,evaluation_tasks=tasks,calibration_generations=calibration_generations,
        implicit_budget_increase=False,
        evidence_bytes=None if measured_evidence_bytes_per_transition is None else
            rounds*count*measured_evidence_bytes_per_transition,
        scope='fresh_run_no_failure_scenario_max_profile_duration_with_warmup_not_a_guaranteed_lower_bound')
