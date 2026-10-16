"""当前闭环训练的独立物理评估、价值诊断和实际权重登记。

评估使用独立任务计划、物理世界、预算及随机数作用域，训练世界保持在原边界。
每次报告包含真实奖励、执行失败、固定初始策略漂移与独立轨迹价值误差；只有
存在同一轮的实际权重和一致身份时才登记 best。评估结束后统一恢复训练随机状态。
本模块不包含训练更新器，也不提供旧训练方案的兼容入口。
"""
from __future__ import annotations
import copy
import json
import math
import os
import time
from pathlib import Path
import torch

from .budget import atomic_json
from .env_adapter import UpperEnvironment
from .evaluation import evaluate_policy
from .parallel_support import (begin_lease, collection_credit, finish_lease, capture_local_rng,
    restore_local_rng, local_call, root_call)
from .run_management import GuardedStepJournal
from gem.robots.bumi.kinematics import sha256_file
from .tensorboard_metrics import evaluation_scalars

def _new_phase(c, name, credits):
    path = c.session/'phases'/name/f'rank{c.distributed.rank:02d}'
    path.mkdir(parents=True, exist_ok=False)
    key = f'{c.session.name}/{name}'
    budget = begin_lease(c.distributed, c.manager, c.budget, path, key, credits)
    return path, key, budget



def _restore_evaluation_state(c, journal, old_backend_journal, rng):
    """所有 rank 在同一评估 finally 调用；恢复块无 collective，异常统一传播。

    评估主体各个可能失败的计算阶段已通过 local_call/root_call 同步失败；这里继续
    使用相同协议，保证单卡关闭 journal 或恢复 RNG 失败时其他卡不会进入计时汇总。
    即使关闭 journal 失败，也必须尝试恢复训练 journal 和独立随机数。
    """
    def restore():
        try:
            journal.close()
        finally:
            c.backend.journal = old_backend_journal
            restore_local_rng(rng, c.generators)
    local_call(c.distributed, restore)



def _record_overhead(c, name, started):
    overhead = getattr(c, 'overhead', None)
    if overhead is not None:
        overhead[name] = overhead.get(name, 0.) + time.perf_counter() - started


def _evaluate(c, label, *, write_report=True):
    if getattr(c, 'vector', None) is not None:
        from .vector_evaluation import isolated_vector_evaluation
        with isolated_vector_evaluation(c,label):
            return _evaluate_impl(c,label,write_report=write_report)
    return _evaluate_impl(c,label,write_report=write_report)



def _evaluate_impl(c, label, *, write_report=True):
    evaluation_started = time.perf_counter()
    from .evaluation import _digest, _model_fingerprint, _stats
    from .periodic_monitor import (
        build_balanced_plan,
        build_fixed_critic_reference,
        choose_best,
        fixed_chain_drift,
        fixed_critic_diagnostic,
        load_best_pointers,
        load_reference_traces,
        log_metrics,
        mark_saved_evaluation,
        merge_periodic_reports,
        shard_plan,
        snapshot_critic_transition,
        summarize_critic_predictions,
    )
    # 一个独立 UpperEnvironment 隔离所有 episode/decision/noise 计数；物理后端在下一
    # train reset 重新建立状态，因此不假称恢复 PhysX 中途状态。
    rng = capture_local_rng(c.generators)
    old_backend_journal = c.backend.journal
    plan = root_call(c.distributed, lambda: build_balanced_plan(c.catalog,
        samples_per_source=c.stage['evaluation']['samples_per_source'],
        seeds=c.stage['evaluation']['seeds'], selection_seed=c.stage['evaluation']['selection_seed'],
        seconds=c.stage['evaluation']['episode_seconds'],
        start_mode=c.stage['evaluation'].get('start_mode','beginning'),
        unique_music=c.stage['evaluation'].get('unique_music',False),split=c.stage['evaluation'].get('split','val')))
    # 分片单位是样本，同一样本全部seed留在同rank；预算必须按实际child任务数计算。
    shards = [shard_plan(plan, rank, c.distributed.world_size) for rank in range(c.distributed.world_size)]
    latencies = c.distributed.all_gather_object(c.env.latency_budget_s)
    per_rank = []
    for rank, child in enumerate(shards):
        seconds=c.stage['evaluation']['episode_seconds']
        if seconds is None:
            controls=sum(math.ceil(t['remaining_music_seconds']*50)+75 for t in child['tasks'])
            generations=sum(math.ceil(t['remaining_music_seconds']/.5)+5 for t in child['tasks'])
            per_rank.append(dict(control_steps=max(1,controls),physics_steps=4*max(1,controls),generations=max(1,generations)))
        else:
            decisions=len(child['tasks'])*(math.ceil(seconds/.5)+2)
            per_rank.append(collection_credit(decisions,seconds,latencies[rank]))
        slots=c.stage['evaluation'].get('envs_per_rank',1)
        # 预付覆盖最坏的多slot不等长尾部；结算仅使用实际执行/分配的物理步。
        for key in ('control_steps','physics_steps'):per_rank[-1][key]*=slots
    evaluation_identity = dict(schema='genmo.closedloop.stage10.periodic_execution.v3',
        training_identity=c.identity, timing_contract=c.config['runtime']['timing_contract'],
        kernel=c.policy.kernel_config, execution_profile=c.state.get('execution_profile'),
        critical_latency_budgets=latencies,
        episode_seconds=c.stage['evaluation']['episode_seconds'], rng_scope='current_cuda_device')
    if getattr(c, 'vector', None) is not None:
        evaluation_identity['physics'] = dict(backend='gpu_vectorized.v1',evaluation_envs_per_rank=c.stage['evaluation'].get('envs_per_rank',1),
            training_envs_per_rank=c.vector.num_envs,train_world_preserved=True,
            vector_audit_contract=c.config['runtime']['vector_audit_contract'],
            vector_reward_contract=c.config['runtime'].get('vector_reward_contract'),
            vector_device_contract=c.config['runtime'].get('vector_device_contract'),
            vector_fragment_contract=c.config['runtime'].get('vector_fragment_contract'),
            prefix_deadline_contract=c.config['runtime']['prefix_deadline_contract'])
    baseline_path, reference_path = c.output/'evaluation_baseline.json', c.output/'fixed_diagnostic_reference.json'
    baseline = root_call(c.distributed, lambda: json.loads(baseline_path.read_text()) if baseline_path.exists() else None)
    references = root_call(c.distributed, lambda: json.loads(reference_path.read_text()) if reference_path.exists() else None)
    if references is not None and (references['plan_sha256'] != plan['plan_sha256']
            or references['evaluation_identity'] != evaluation_identity):
        raise ValueError('Fixed initial diagnostic reference differs from current evaluation contract')
    if references is None and (label != 'initial' or baseline is not None):
        raise ValueError('Periodic evaluation requires the durable initial diagnostic reference')
    path, lease, budget = _new_phase(c, f'eval_{label}', per_rank)
    journal = GuardedStepJournal(path/'execution_journal.sqlite', c.guard,
        format=c.stage.get('performance', {}).get('journal_format', 'json.v1'))
    c.backend.journal = journal
    config = copy.deepcopy(c.config)
    config.setdefault('runtime',{})['evidence_level']='full'
    env_type = UpperEnvironment
    if getattr(c, 'vector', None) is not None:
        from .vector_environment import VectorUpperEnvironment
        env_type = VectorUpperEnvironment
    driver=None
    if getattr(getattr(c,'active_evaluation_world',None),'num_envs',1)>1:
        from .evaluation_world import EvaluationWorldDriver
        driver=EvaluationWorldDriver(c,config,budget,path);env=driver.env
    else:env = env_type(config, c.backend, c.builder, c.policy, budget, path)
    env.disk_guard = c.guard
    env.latency_budget_s, env.policy_version, env.iteration = c.env.latency_budget_s, c.env.policy_version, c.env.iteration
    initial_rows = []
    from .training_config import VERSION as CURRENT_TRAINING_VERSION
    fresh_critic_enabled = c.stage.get('version')==CURRENT_TRAINING_VERSION
    def capture_initial(task,row):
        if (references is None or fresh_critic_enabled) and hasattr(row,'context') and hasattr(row,'identity'):
            initial_rows.append(snapshot_critic_transition(row))
    try:
        def perform():
            if not shards[c.distributed.rank]['tasks']:
                return None
            return evaluate_policy(env, c.policy, shards[c.distributed.rank],
                path/'report', catalog=c.catalog, episode_seconds=c.stage['evaluation']['episode_seconds'],
                actor_identity=dict(model_fingerprint=_model_fingerprint(c.actor), iteration=c.state['iteration'],
                    policy_version=c.state['policy_version'], timing_contract=c.config['runtime']['timing_contract'],
                    kernel=c.policy.kernel_config, scope='periodic_in_memory'), frozen_modules={'critic':c.critic},
                local_cuda_only=True,transition_sink=capture_initial,world_driver=driver,
                progress=lambda event: print(f'[EVAL rank={c.distributed.rank}] {event}', flush=True))
        local_call(c.distributed, perform)
        reports = c.distributed.all_gather_object(str(path/'report/report.json') if shards[c.distributed.rank]['tasks'] else None)
        reports = [report for report in reports if report is not None]
        result = root_call(c.distributed, lambda: merge_periodic_reports(plan, reports, baseline=baseline,
                                                                       evaluation_identity=evaluation_identity))
        def diagnose():
            if not shards[c.distributed.rank]['tasks']:
                return None
            if not hasattr(c, '_periodic_fixed_reference'):
                if references is None:
                    actor_reference = load_reference_traces(path/'raw_samples', max_chains=4)
                    critic_reference = None
                    if initial_rows:
                        critic_reference = build_fixed_critic_reference(initial_rows, c.critic, c.distributed.device,
                            gamma_upper=c.settings['gamma_upper'], lambda_upper=c.settings['lambda_upper'])
                    description = dict(rank=c.distributed.rank,
                        actor_records=[dict(path=r['path'], sha256=r['sha256']) for r in actor_reference['records']],
                        actor_reference_sha256=actor_reference['reference_sha256'], critic=None)
                    if critic_reference is not None:
                        target = path/'fixed_critic_reference.pt'
                        temporary = target.with_suffix('.pt.tmp')
                        torch.save(critic_reference, temporary)
                        with temporary.open('rb') as stream:
                            os.fsync(stream.fileno())
                        os.replace(temporary, target)
                        c.guard.account_file(target)
                        description['critic'] = dict(path=str(target.resolve()), sha256=sha256_file(target),
                                                    reference_sha256=critic_reference['reference_sha256'])
                else:
                    matches = [r for r in references['ranks'] if r['rank'] == c.distributed.rank]
                    if len(matches) != 1:
                        raise ValueError('Fixed reference rank missing or duplicated')
                    description = matches[0]
                    for record in description['actor_records']:
                        if sha256_file(record['path']) != record['sha256']:
                            raise ValueError('Initial Actor reference raw trace SHA differs')
                    actor_reference = load_reference_traces([r['path'] for r in description['actor_records']], max_chains=4)
                    if actor_reference['reference_sha256'] != description['actor_reference_sha256']:
                        raise ValueError('Fixed Actor reference identity differs')
                    critic_reference = None
                    if description['critic'] is not None:
                        record = description['critic']
                        if sha256_file(record['path']) != record['sha256']:
                            raise ValueError('Fixed Critic reference file SHA differs')
                        critic_reference = torch.load(record['path'], map_location='cpu', weights_only=False)
                        if critic_reference['reference_sha256'] != record['reference_sha256']:
                            raise ValueError('Fixed Critic reference identity differs')
                c._periodic_fixed_reference = (actor_reference, critic_reference, description)
            actor_reference, critic_reference, description = c._periodic_fixed_reference
            fresh_critic = None
            if fresh_critic_enabled:
                # 初始固定targets用于可比的历史诊断；新独立轨迹另用当前V自举，
                # 两者必须分开。这里只缓存因果价值输入及真实奖励，不保留随机链。
                fresh_reference = (critic_reference if references is None else
                    build_fixed_critic_reference(initial_rows, c.critic, c.distributed.device,
                        gamma_upper=c.settings['gamma_upper'], lambda_upper=c.settings['lambda_upper']))
                fresh_critic = dict(predictions=[r['initial_prediction'] for r in fresh_reference['records']],
                    targets=[r['target'] for r in fresh_reference['records']],
                    reference_sha256=fresh_reference['reference_sha256'])
            return dict(reference=description, actor=fixed_chain_drift(c.policy, actor_reference, include_values=True),
                        critic=None if critic_reference is None else fixed_critic_diagnostic(c.critic, critic_reference),
                        fresh_critic=fresh_critic)
        diagnostics = c.distributed.all_gather_object(local_call(c.distributed, diagnose))
        diagnostics = [diagnostic for diagnostic in diagnostics if diagnostic is not None]
        c.state['budget'] = finish_lease(c.distributed, c.budget, budget, lease, per_rank)
        def publish():
            if references is None:
                atomic_json(reference_path, dict(schema='genmo.closedloop.stage10.fixed_diagnostics.v2',
                    plan_sha256=plan['plan_sha256'], evaluation_identity=evaluation_identity,
                    initial_iteration=c.state['iteration'], ranks=[d['reference'] for d in diagnostics]))
            result['iteration'] = c.state['iteration']
            result['session_id'] = c.session.name
            values = [value for diagnostic in diagnostics for value in diagnostic['actor']['joint_kl_values']]
            stats = _stats(values)
            result['fixed_actor_drift'] = dict(reference_sha256=_digest([d['reference'] for d in diagnostics]),
                internal_transition_count=len(values), mean_joint_kl=stats['mean'], p95_joint_kl=stats['p95'],
                max_joint_kl=stats['max'], diagnostic_scope='fixed_initial_chain_gaussian_drift_not_physics_quality')
            critics = [d['critic'] for d in diagnostics]
            if any(d is not None for d in critics) and not all(d is not None for d in critics):
                raise ValueError('Fixed Critic reference must be available on every rank or explicitly unavailable')
            result['fixed_critic_diagnostic'] = (dict(available=False, reason='initial_evaluation_did_not_supply_value_contexts')
                if not all(d is not None for d in critics) else dict(available=True,
                    **summarize_critic_predictions([p for d in critics for p in d['predictions']],
                                                  [t for d in critics for t in d['targets']]),
                    reference_sha256=_digest([d['reference_sha256'] for d in critics]),
                    diagnostic_scope='fixed_initial_eval_gae_targets_with_initial_critic_bootstrap'))
            if fresh_critic_enabled:
                fresh = [d['fresh_critic'] for d in diagnostics]
                result['fresh_critic_diagnostic'] = dict(available=True,
                    **summarize_critic_predictions([p for d in fresh for p in d['predictions']],
                                                  [t for d in fresh for t in d['targets']]),
                    reference_sha256=_digest([d['reference_sha256'] for d in fresh]),
                    critic_updates=c.state.get('critic_updates'),
                    diagnostic_scope='fresh_independent_physics_current_critic_bootstrapped_GAE_not_unbiased_V_truth')
            initial = result if baseline is None else baseline
            previous_best = load_best_pointers(c.output, result, state=c.state)
            choice = choose_best(result, initial, **previous_best, iteration=c.state['iteration'])
            checkpoint = None
            latest_path = c.output/'latest.json'
            if latest_path.exists():
                descriptor = json.loads(latest_path.read_text())
                if descriptor['iteration'] == c.state['iteration']:
                    checkpoint = descriptor
            elif c.state['iteration'] == 0 and (c.output/'checkpoints/initial.pt').is_file():
                checkpoint = dict(iteration=0, path='checkpoints/initial.pt',
                                  sha256=sha256_file(c.output/'checkpoints/initial.pt'))
            if checkpoint is not None:
                choice = mark_saved_evaluation(result, initial, checkpoint,
                    actor_fingerprint=_model_fingerprint(c.actor), output_dir=c.output,
                    best_observed=choice['best_observed'], best_saved=choice['best_saved'])
            result['selection'] = choice
            for key in ('best_observed', 'best_saved'):
                if choice[key] is not None:
                    atomic_json(c.output/f'{key}.json', choice[key])
            return result
        result = root_call(c.distributed, publish)
        c.state.update(best_observed=result['selection']['best_observed'], best_saved=result['selection']['best_saved'])
        c.last_evaluation = result
    finally:
        try:
            if driver is not None:local_call(c.distributed,driver.close)
        finally:_restore_evaluation_state(c, journal, old_backend_journal, rng)
    result['wall_seconds'] = max(c.distributed.all_gather_object(time.perf_counter()-evaluation_started))
    result['wall_time_scope'] = 'evaluation_including_state_restore_excluding_final_timing_record_publish'
    def publish_timing():
        written = time.perf_counter()
        if write_report:atomic_json(c.session/'evaluations'/f'{label}.json', result)
        _record_overhead(c, 'evaluation_json_write_seconds', written)
        log_metrics(c.writer, c.output/'curves.jsonl', c.state['iteration'], dict(
            reward=result['source_balanced_reward'], duration=result['mean_executed_seconds'],
            physical_failure_count=result['physical_failure_count'], by_source=result['by_source'],
            paired_baseline=result.get('paired_baseline'), fixed_actor_drift=result['fixed_actor_drift'],
            fixed_critic_diagnostic=result['fixed_critic_diagnostic'],
            fresh_critic_diagnostic=result.get('fresh_critic_diagnostic'), wall_seconds=result['wall_seconds']),
            prefix='validation',dashboard_metrics=evaluation_scalars(result,baseline) if c.writer is not None else None)
    root_call(c.distributed, publish_timing)
    return result



def _evaluate_and_record(c, report, label, *, legacy_label=None):
    """评估与最终计时落盘成功后登记 SHA，避免会话漏报终态验证或接受被改写报告。"""
    result = _evaluate(c, label)
    path = c.session/'evaluations'/f'{label}.json'
    artifact = root_call(c.distributed, lambda: dict(label=label, iteration=c.state['iteration'],
        path=str(path.relative_to(c.output)), sha256=sha256_file(path)))
    report.setdefault('evaluation_artifacts', []).append(artifact)
    report['evaluations'].append(label if legacy_label is None else legacy_label)
    return result
