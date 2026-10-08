"""第二阶段 v2 的八路同步闭环训练主循环。

八个 rank 各自运行 GENMO 采样与独立冻结 GMT/CPU PhysX 后端，每轮各采二十条真实
上层转移；完整链留在本 rank，统一的样本索引、全局优势和 SUM 梯度连接成同一个
Actor/Critic 训练任务。Actor 固定学习率由配置指定，默认 5e-9，每轮两个 epoch、
最多四次 minibatch 更新；旧概率和价值目标不随 minibatch 改写。每次 Actor step
记录 PPO／BC 分解梯度，rank 0 因而会临时保留一份 PPO 梯度快照。

根进程独占预算、日志与完整断点发布，先向每个采集器预占有界额度，再按确认的本地
持久账本结算。所有副作用仍由 journal 先记录再确认；故障丢弃本轮并交给 torchrun
终止同一作业。KL 验收只在完整 rollout 上决定是否发布策略；拒绝时恢复整轮模型、
优化器与 BC 状态，物理消耗不退还。完整 checkpoint 以外层轮次按 300 保存，普通轮
仅封存执行证据并交给有界归档队列。周期评估使用独立环境对象和隔离随机状态，不把
评估任务放入训练 Buffer，也不在每 100 轮额外写模型权重。

本入口复用现有模型、UpperTransition、奖励、任务目录和严格来源身份；新版本与
旧单采集 v1 完整恢复严格区分。默认停止轮次仍由命令行给出，不自动开启长期训练。
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import random
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from .budget import atomic_json
from .checkpoint import VERSION_V2, capture_rank_state, load_checkpoint, save_checkpoint
from .critic import UpperCritic
from .env_adapter import UpperEnvironment
from .evaluation import evaluate_policy
from .execution_profile import probe_profiles, select_profile
from .full_dataset import FullMusicCatalog, FullMusicSampler
from .long_run import LongRunMaintenance
from .parallel_support import (begin_lease, broadcast_state, build_global_manifest, capture_local_rng,
    check_kl_limits, collection_credit, cpu_snapshot, finish_lease, lightweight_fingerprint,
    local_call, restore_local_rng, root_call)
from .policy import DPPODiffusionPolicy
from .returns import normalize_advantages_global
from .rpc import AcknowledgedBackend
from .run_management import DiskGuard, GuardedStepJournal, RolloutWriter, RunManager, StopSignal
from .trainer import (SupervisedAnchor, actor_update_v2, analytic_kl_local, critic_update_local,
    fixed_targets, load_actor, populate_values, probability_check_local, trainable_actor_parameters)
from gem.closedloop.baseline_provenance import verify_source_provenance
from gem.closedloop.online_conditions import OnlineConditionBuilder
from gem.robots.bumi.feature_codec import BumiMotionFeatureCodec
from gem.robots.bumi.kinematics import BumiKinematics, sha256_file


def _synchronize_models(context, *, full=False, phase='iteration'):
    c = context
    if full:
        from .distributed_runtime import _optimizer_fingerprint
        evidence = c.distributed.fingerprints(c.actor, c.critic, phase)
        optimizers = c.distributed.all_gather_object(dict(actor=_optimizer_fingerprint(c.actor_optimizer),
                                                       critic=_optimizer_fingerprint(c.critic_optimizer)))
        if any(value != optimizers[0] for value in optimizers):
            raise RuntimeError('Optimizer replicas differ at a complete checkpoint boundary')
        return dict(scope='full_model_and_optimizer_sha256', models=evidence, optimizers=optimizers)
    local = dict(actor=lightweight_fingerprint(c.actor), critic=lightweight_fingerprint(c.critic))
    reports = c.distributed.all_gather_object(local)
    if any(value != reports[0] for value in reports):
        raise RuntimeError('Replica sampled statistics differ; complete diagnostic required')
    return dict(scope='sampled_parameter_values_not_full_hash', world_size=c.distributed.world_size, passed=True)


def checkpoint_due(outer_iteration, interval=300, *, normal_end=False):
    """保存仅取决于外层轮次；结束时允许额外同步恢复点，不读取 Actor 步数。"""
    if type(outer_iteration) is not int or outer_iteration < 0 or type(interval) is not int or interval < 1:
        raise ValueError('Checkpoint interval and outer iteration must be valid integers')
    return outer_iteration == 0 or normal_end or outer_iteration % interval == 0


def _write_checkpoint(c, *, reason):
    from tools.train_closedloop_stage10 import capture_execution_state
    from .evaluation import _model_fingerprint
    from .periodic_monitor import log_metrics
    previous = root_call(c.distributed, lambda: json.loads((c.output/'latest.json').read_text())
                         if (c.output/'latest.json').exists() else None)
    if previous is not None and previous['iteration'] == c.state['iteration']:
        c.latest_checkpoint = previous
        return previous
    started = time.perf_counter()
    _synchronize_models(c, full=True, phase=f'checkpoint_{c.state["iteration"]}')
    c.state['budget'] = root_call(c.distributed, lambda: c.budget.state_dict())
    c.state['last_durable_checkpoint_iteration'] = c.state['iteration']
    local_state = dict(c.state, **capture_execution_state(c.env), buffer_size=0, pending_plan=False)
    rank_state = capture_rank_state(c.distributed.rank, state=local_state,
        samplers=c.samplers, generators=c.generators)
    states = c.distributed.all_gather_object(rank_state)
    def publish():
        index = c.state['iteration']
        if (c.output/'latest.json').exists():
            latest = json.loads((c.output/'latest.json').read_text())
            if latest['iteration'] == index:
                return latest
        c.state['budget'] = c.budget.state_dict()
        c.state['last_durable_checkpoint_iteration'] = index
        name = 'initial.pt' if index == 0 else f'stage10_{index:06d}_{c.manager.session_id}.pt'
        path = c.output/'checkpoints'/name
        c.manager.check_disk(c.stage['storage']['checkpoint_reserve_bytes'], refresh=True)
        save_checkpoint(path, actor=c.actor, critic=c.critic, actor_optimizer=c.actor_optimizer,
            critic_optimizer=c.critic_optimizer, state=c.state, identity=c.identity, config=c.base_config,
            version=VERSION_V2, rank_states=states)
        c.manager.disk_guard.account_file(path)
        if index:
            c.manager.publish_checkpoint(index, path, metadata=dict(reason=reason, world_size=c.distributed.world_size,
                actor_updates=c.state['actor_updates'], critic_updates=c.state['critic_updates'],
                timing_contract=c.config['runtime']['timing_contract'],
                actor_model_fingerprint=_model_fingerprint(c.actor)))
            c.maintenance.prune_checkpoints()
        return dict(iteration=index, path=str(path.relative_to(c.output)), sha256=sha256_file(path))
    saved = root_call(c.distributed, publish)
    c.state['last_durable_checkpoint_iteration'] = saved['iteration']
    c.latest_checkpoint = saved
    elapsed = max(c.distributed.all_gather_object(time.perf_counter()-started))
    root_call(c.distributed, lambda: log_metrics(c.writer, c.output/'curves.jsonl', c.state['iteration'],
        dict(seconds=elapsed, iteration=c.state['iteration']), prefix='checkpoint'))
    return saved


def _flush_archive_metrics(c):
    """仅根训练线程写曲线；后台只返回已完成计时，不持有 TensorBoard writer。"""
    if c.maintenance is None or not hasattr(c.maintenance, 'drain_archive_timings'):
        return
    from .periodic_monitor import log_metrics
    telemetry = c.maintenance.drain_archive_timings()
    for record in telemetry['records']:
        c.manager.append_metrics(dict(event='execution_archive_completed', **record))
        log_metrics(c.writer, c.output/'curves.jsonl', record['iteration'], record, prefix='archive')
    if telemetry['enqueues']:
        log_metrics(c.writer, c.output/'curves.jsonl', c.state['iteration'],
            {f'entry{i}': row for i, row in enumerate(telemetry['enqueues'])}, prefix='archive_queue')


def _new_phase(c, name, credits):
    path = c.session/'phases'/name/f'rank{c.distributed.rank:02d}'
    path.mkdir(parents=True, exist_ok=False)
    key = f'{c.session.name}/{name}'
    budget = begin_lease(c.distributed, c.manager, c.budget, path, key, credits)
    return path, key, budget


def _calibrate_and_profile(c, restored=False):
    from tools.train_closedloop_stage10 import calibrate, capture_execution_state, restore_execution_state
    previous = capture_execution_state(c.env) if restored else None
    rng = capture_local_rng(c.generators)
    samples = c.base_config['timing']['calibration_warmup'] + c.base_config['timing']['calibration_samples']
    credits = c.distributed.all_gather_object(
        collection_credit(samples + 2, c.settings['episode_seconds'], c.env.latency_budget_s))
    path, lease, budget = _new_phase(c, 'calibration', credits)
    c.env.budget, c.env.output = budget, path
    path.joinpath('raw_samples').mkdir()
    def perform():
        source = next(iter(c.catalog.samples['train']))
        sample = c.catalog.samples['train'][source][0]
        c.env.reset_task(sample, c.catalog.load_music(sample), seed=c.env.config['stage9']['seed'], phase='calibration')
        conditions, _ = c.env.preview_context()
        conditions = {key: value.to(c.distributed.device) for key, value in conditions.items()}
        return probe_profiles(c.policy, conditions,
            maximum_microbatch=c.settings['denoising_microbatch'], seed=c.stage['seed']+c.distributed.rank,
            reserve_generation=lambda: budget.reserve('profile', generations=1))
    reports = c.distributed.all_gather_object(local_call(c.distributed, perform))
    profile = select_profile(reports, required=c.state.get('execution_profile') if restored else None)
    c.profile = profile
    c.policy.cfg_batch = profile['cfg_batch']
    # 校准必须使用最终通过概率门槛的执行方式；CFG回退后不可沿用更快路径的时延。
    calibration = local_call(c.distributed, lambda: calibrate(c.env, c.catalog, path,
        c.base_config['timing']['calibration_warmup'], c.base_config['timing']['calibration_samples']))
    atomic_json(path/'profile.json', dict(calibration=calibration, profiles=reports[c.distributed.rank], selected=profile))
    if restored:
        # 新环境只校验运行路径，恢复已有延迟合同；校准的请求编号仍消耗，不复用。
        if max(calibration['durations']) > previous['latency_budget_s']:
            raise RuntimeError('Restored latency contract is insufficient for the selected execution profile')
        attempts = c.env.attempt
        restore_execution_state(c.env, dict(previous, policy_version=c.state['policy_version'],
                                           iteration=c.state['iteration']), spent_generations=attempts)
    restore_local_rng(rng, c.generators)
    c.profile = profile
    c.policy.cfg_batch = profile['cfg_batch']
    c.state['execution_profile'] = profile
    c.state['budget'] = finish_lease(c.distributed, c.budget, budget, lease, credits)
    return reports


def _evaluate(c, label):
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
        seconds=c.stage['evaluation']['episode_seconds']))
    # 分片单位是样本，同一样本全部seed留在同rank；预算必须按实际child任务数计算。
    shards = [shard_plan(plan, rank, c.distributed.world_size) for rank in range(c.distributed.world_size)]
    latencies = c.distributed.all_gather_object(c.env.latency_budget_s)
    per_rank = []
    for rank, child in enumerate(shards):
        count = len(child['tasks'])
        decisions = count * (math.ceil(c.stage['evaluation']['episode_seconds']/.5)+2)
        per_rank.append(collection_credit(decisions, c.stage['evaluation']['episode_seconds'], latencies[rank]))
    evaluation_identity = dict(schema='genmo.closedloop.stage10.periodic_execution.v2',
        training_identity=c.identity, timing_contract=c.config['runtime']['timing_contract'],
        kernel=c.policy.kernel_config, execution_profile=c.state.get('execution_profile'),
        critical_latency_budgets=latencies,
        episode_seconds=c.stage['evaluation']['episode_seconds'], rng_scope='current_cuda_device')
    baseline_path, reference_path = c.output/'evaluation_baseline.json', c.output/'fixed_diagnostic_reference.json'
    baseline = root_call(c.distributed, lambda: json.loads(baseline_path.read_text()) if baseline_path.exists() else None)
    references = root_call(c.distributed, lambda: json.loads(reference_path.read_text()) if reference_path.exists() else None)
    if references is not None and (references['plan_sha256'] != plan['plan_sha256']
            or references['evaluation_identity'] != evaluation_identity):
        raise ValueError('Fixed initial diagnostic reference differs from current evaluation contract')
    if references is None and (label != 'initial' or baseline is not None):
        raise ValueError('Periodic evaluation requires the durable initial diagnostic reference')
    path, lease, budget = _new_phase(c, f'eval_{label}', per_rank)
    journal = GuardedStepJournal(path/'execution_journal.sqlite', c.guard)
    c.backend.journal = journal
    config = copy.deepcopy(c.config)
    env = UpperEnvironment(config, c.backend, c.builder, c.policy, budget, path)
    env.disk_guard = c.guard
    env.latency_budget_s, env.policy_version, env.iteration = c.env.latency_budget_s, c.env.policy_version, c.env.iteration
    initial_rows = []
    if references is None:
        actual_step = env.step
        def capture_initial(*args, **kwargs):
            row = actual_step(*args, **kwargs)
            if hasattr(row, 'context') and hasattr(row, 'identity'):
                initial_rows.append(snapshot_critic_transition(row))
            return row
        env.step = capture_initial
    try:
        def perform():
            if not shards[c.distributed.rank]['tasks']:
                return None
            return evaluate_policy(env, c.policy, shards[c.distributed.rank],
                path/'report', catalog=c.catalog, episode_seconds=c.stage['evaluation']['episode_seconds'],
                actor_identity=dict(model_fingerprint=_model_fingerprint(c.actor), iteration=c.state['iteration'],
                    policy_version=c.state['policy_version'], timing_contract=c.config['runtime']['timing_contract'],
                    kernel=c.policy.kernel_config, scope='periodic_in_memory'), frozen_modules={'critic':c.critic},
                local_cuda_only=True,
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
            return dict(reference=description, actor=fixed_chain_drift(c.policy, actor_reference, include_values=True),
                        critic=None if critic_reference is None else fixed_critic_diagnostic(c.critic, critic_reference))
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
            atomic_json(c.session/'evaluations'/f'{label}.json', result)
            log_metrics(c.writer, c.output/'curves.jsonl', c.state['iteration'], dict(
                reward=result['source_balanced_reward'], duration=result['mean_executed_seconds'],
                physical_failure_count=result['physical_failure_count'], by_source=result['by_source'],
                paired_baseline=result.get('paired_baseline'), fixed_actor_drift=result['fixed_actor_drift'],
                fixed_critic_diagnostic=result['fixed_critic_diagnostic']), prefix='validation')
            return result
        result = root_call(c.distributed, publish)
        c.state.update(best_observed=result['selection']['best_observed'], best_saved=result['selection']['best_saved'])
        c.last_evaluation = result
        return result
    finally:
        journal.close()
        c.backend.journal = old_backend_journal
        restore_local_rng(rng, c.generators)


def _collect(c, index):
    from tools.train_closedloop_stage10 import collect_rollout
    directory = c.session/'iterations'/f'{index:06d}'
    path = directory/f'rank{c.distributed.rank:02d}'
    path.mkdir(parents=True, exist_ok=False)
    count = c.settings['rollout_upper_steps_per_rank']
    credits = c.distributed.all_gather_object(collection_credit(count, c.settings['episode_seconds'], c.env.latency_budget_s))
    lease = f'{c.session.name}/iteration{index}'
    budget = begin_lease(c.distributed, c.manager, c.budget, path, lease, credits)
    c.journal.close()
    c.journal = GuardedStepJournal(path/'execution_journal.sqlite', c.guard)
    c.backend.journal = c.journal
    c.env.output, c.env.budget = path, budget
    path.joinpath('raw_samples').mkdir()
    writer = RolloutWriter(path/'rollout', policy_version=c.state['policy_version'],
                          chunk_size=c.stage['storage']['rollout_chunk_size'], disk_guard=c.guard)
    started = time.perf_counter()
    def perform():
        return collect_rollout(c.env, c.sampler, count, writer, check_disk=c.guard.check,
            value_snapshot=lambda row: populate_values([row], c.critic, c.distributed.device,
                                                       critic_version=c.state['critic_updates']))
    buffer, report = local_call(c.distributed, perform)
    report['seconds'] = time.perf_counter()-started
    report['rank'] = c.distributed.rank
    report['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(c.distributed.device)
    report['rejections'] = sum(bool(row.metadata.get('rejection')) for row in buffer.transitions)
    report['timeouts'] = sum('timeout' in str(row.reason or '').lower() for row in buffer.transitions)
    report['generation_timing_totals'] = {}
    report['actor_phase_totals'] = {}
    report['reward_component_sums'] = {}
    for row in buffer.transitions:
        for key, value in row.metadata.get('timing', {}).items():
            if isinstance(value, (int, float)):
                report['generation_timing_totals'][key] = report['generation_timing_totals'].get(key, 0.) + value
        for key, value in row.metadata.get('timing', {}).get('actor_phases', {}).items():
            report['actor_phase_totals'][key] = report['actor_phase_totals'].get(key, 0.) + value
        for detail in row.metadata.get('reward_details', []):
            for key, value in detail.get('components', {}).items():
                if isinstance(value, dict):
                    value = value.get('integrated_reward')
                if isinstance(value, (int, float)):
                    report['reward_component_sums'][key] = report['reward_component_sums'].get(key, 0.) + value

    c.state['budget'] = finish_lease(c.distributed, c.budget, budget, lease, credits)
    local_targets = local_call(c.distributed, lambda: fixed_targets(buffer.transitions, c.critic, c.distributed.device,
        gamma_upper=c.settings['gamma_upper'], lambda_upper=c.settings['lambda_upper'],
        reuse_values=True, critic_version=c.state['critic_updates'], normalize=False))
    targets = normalize_advantages_global(local_targets, distributed=c.distributed)
    torch.save(targets, path/'fixed_targets.pt')
    c.guard.account_file(path/'fixed_targets.pt')
    manifest = build_global_manifest(buffer.transitions, c.distributed)
    if len(manifest) != c.settings['rollout_upper_steps']:
        raise RuntimeError('Collectors did not produce the configured global batch')
    atomic_json(path/'collection.json', report)
    return buffer, targets, manifest, directory, path, report


def _update(c, buffer, targets, manifest, index):
    rows = buffer.transitions
    full = index == c.initial_iteration+1 or index % c.stage['checks']['full_probability_every'] == 0
    indices = None if full else [i for i, row in enumerate(manifest)
        if row['local_index'] < c.stage['checks']['sentinel_chains_per_rank']]
    check = probability_check_local(c.policy, rows, global_manifest=manifest, global_indices=indices, distributed=c.distributed,
                                    denoising_microbatch=c.profile['microbatch'])
    check.update(scope='full_rollout_before_first_step' if full else 'rank_sentinel_all_denoising_steps')
    # 一份共享模型/Adam 快照只在 rank0 创建；各 rank 只保存自己的轻量更新随机状态。
    backup = (cpu_snapshot(dict(actor=c.actor.state_dict(), critic=c.critic.state_dict(),
        actor_optimizer=c.actor_optimizer.state_dict(), critic_optimizer=c.critic_optimizer.state_dict(),
        bc=None if c.bc is None else c.bc.state_dict())) if c.distributed.rank == 0 else None)
    rng = capture_local_rng(c.generators)
    actor, critic, kl = {}, {}, {}
    timings = dict(critic_seconds=0., actor_seconds=0., kl_seconds=0.)
    active_phase = None
    x0_reference = None
    if c.settings.get('x0_diagnostic_every', 0) and index % c.settings['x0_diagnostic_every'] == 0:
        from .optional_diagnostics import capture_x0_reference
        x0_reference = capture_x0_reference(c.policy, rows, global_manifest=manifest, distributed=c.distributed,
            denoising_microbatch=c.profile['microbatch'])
    try:
        active_phase = 'critic_seconds'
        started = time.perf_counter()
        critic_updater = critic_update_local
        if c.settings.get('critic_update_mode', 'distributed') == 'rank0_broadcast':
            from .optional_diagnostics import critic_update_root_broadcast
            critic_updater = critic_update_root_broadcast
        critic = critic_updater(c.critic, c.critic_optimizer, rows, targets,
            global_manifest=manifest, distributed=c.distributed, steps=c.settings['critic_steps'],
            batch_size=c.settings['critic_batch'], generator=c.generators['critic'],
            grad_clip_norm=c.settings['critic_grad_clip_norm'])
        timings[active_phase] = time.perf_counter()-started
        active_phase = 'actor_seconds'
        started = time.perf_counter()
        actor = actor_update_v2(c.policy, c.actor_optimizer, rows, targets, global_manifest=manifest,
            distributed=c.distributed, bc=c.bc, bc_weight=c.settings['bc_weight'], clip=c.settings['ppo_clip'],
            gamma_denoising=c.settings['gamma_denoising'], grad_clip_norm=c.settings['grad_clip_norm'],
            ppo_epochs=c.settings['ppo_epochs'], actor_minibatch_internal_transitions=c.settings['actor_minibatch_internal_transitions'],
            denoising_microbatch=c.profile['microbatch'], max_optimizer_steps=c.settings['max_actor_optimizer_steps'],
            soft_kl_limit=c.settings['kl_soft_stop_joint'], objective_logprob_reduction=c.settings['objective_logprob_reduction'],
            generator=c.generators['actor'], gradient_diagnostics=True,
            reserve_attempt=lambda: root_call(c.distributed,
                lambda: c.budget.reserve('update', optimizer_attempts=1)))
        timings[active_phase] = time.perf_counter()-started
        active_phase = 'kl_seconds'
        started = time.perf_counter()
        kl = analytic_kl_local(c.policy, rows, global_manifest=manifest, distributed=c.distributed,
                               denoising_microbatch=c.profile['microbatch'])
        timings[active_phase] = time.perf_counter()-started
        active_phase = None
        check_kl_limits(kl, c.settings)
        if x0_reference is not None:
            from .optional_diagnostics import x0_change_local
            kl['x0_diagnostic'] = x0_change_local(c.policy, rows, x0_reference,
                global_manifest=manifest, distributed=c.distributed, denoising_microbatch=c.profile['microbatch'])
    except Exception as failure:
        if active_phase is not None:
            timings[active_phase] = time.perf_counter()-started
        error_message = str(failure)
        restored = broadcast_state(backup, c.distributed)
        c.actor.load_state_dict(restored['actor']); c.critic.load_state_dict(restored['critic'])
        c.actor_optimizer.load_state_dict(restored['actor_optimizer'])
        c.critic_optimizer.load_state_dict(restored['critic_optimizer'])
        if c.bc is not None:
            c.bc.load_state_dict(restored['bc'])
        restore_local_rng(rng, c.generators)
        c.actor_optimizer.zero_grad(set_to_none=True); c.critic_optimizer.zero_grad(set_to_none=True)
        root_call(c.distributed, lambda: atomic_json(c.session/f'rejected_{index:06d}.json',
            dict(iteration=index, actor=actor, critic=critic, kl=kl, rolled_back=True, error=error_message,
                 actor_lr=c.settings['actor_lr'], optimizer_attempts_charged=True,
                 probability_check=check, timings=timings)))
        raise
    del backup
    communication = (c.distributed.collect_gradient_timings(synchronize=True)
                     if hasattr(c.distributed, 'collect_gradient_timings') else {'scope':'unavailable'})
    return dict(communication=communication, actor=actor, critic=critic, kl=kl, probability_check=check,
                timings=timings)


def run_parallel(args, config, collective, preflight):
    """所有 rank 共同调用；参数 args 来自正式八卡入口。"""
    from tools import train_closedloop_stage10 as legacy
    from tools.eval.run_closedloop_baseline import Workers
    c = SimpleNamespace(distributed=collective, manager=None, maintenance=None, budget=None,
        workers=None, backend=None, journal=None, writer=None, output=args.output_dir.resolve(),
        base_config=copy.deepcopy(config), config=copy.deepcopy(config), stage=config['stage10'],
        settings=config['stage9'], state=None)
    if hasattr(collective, 'enable_gradient_timing'):
        collective.enable_gradient_timing()
    stop = StopSignal(); stop.install()
    report = dict(schema='genmo.closedloop.stage10.session.v2', status='running', iterations=[], evaluations=[])
    error = None
    try:
        def initialize():
            c.manager = RunManager(c.output, resume=bool(args.resume),
                min_free_bytes=c.stage['storage']['min_free_bytes'], max_run_bytes=c.stage['storage']['max_run_bytes'])
            c.budget = c.manager.budget(c.stage['limits'])
            c.maintenance = LongRunMaintenance(c.manager, c.stage)
            return dict(session_id=c.manager.session_id, run_id=c.manager.run_id)
        run = root_call(collective, initialize)
        c.session = c.output/'sessions'/run['session_id']
        c.rank_dir = c.session/f'rank{collective.rank:02d}'
        c.rank_dir.mkdir(parents=True)
        c.config['runtime'].update(rank=collective.rank, genmo_device=str(collective.device))
        c.config['stage9']['run_id'] = run['run_id']
        c.config['stage9']['seed'] = (c.stage['seed'] + 1000003*collective.rank) % 2**32
        c.config['output_root'] = str(c.output)
        c.settings = c.config['stage9']
        # 每个 guard 约束本次 session 的全部本地文件；根进程每轮核对整 run 上限。
        c.guard = DiskGuard(c.session, min_free_bytes=c.stage['storage']['min_free_bytes'],
                            max_run_bytes=c.stage['storage']['max_run_bytes'])
        random.seed(c.stage['seed']); np.random.seed(c.stage['seed']); torch.manual_seed(c.stage['seed'])
        c.catalog = FullMusicCatalog(config['paths']['data_root'])
        audit_count = 0
        def audit_progress(record):
            nonlocal audit_count
            audit_count += 1
            if audit_count % 100 == 0:
                print(f'[DATA_AUDIT] {audit_count} {record["dataset"]}/{record["split"]}', flush=True)
        audit = root_call(collective, lambda: c.catalog.audit_files(audit_progress,
            require_audio=c.stage['dataset']['require_audio']))
        local_call(collective, lambda: c.catalog.apply_audit(audit))
        provenance = root_call(collective, lambda: legacy._sources(config, preflight))
        c.actor, train_config, loading = local_call(collective, lambda: load_actor(c.config))
        loading.update(source_checkpoint_sha256=preflight['asset_sha256']['checkpoint'])
        root_call(collective, lambda: atomic_json(c.session/'actor_loading.json', loading))
        c.critic = UpperCritic(qpos_mean=c.actor.endecoder.mean, qpos_std=c.actor.endecoder.std,
            proprio_scales=tuple(train_config.model.proprio_scales)).to(collective.device)
        collective.broadcast_module(c.actor); collective.broadcast_module(c.critic)
        c.actor_optimizer = torch.optim.AdamW(trainable_actor_parameters(c.actor), lr=c.settings['actor_lr'], weight_decay=0.)
        c.critic_optimizer = torch.optim.AdamW(c.critic.parameters(), lr=c.settings['critic_lr'], weight_decay=0.)
        c.policy = DPPODiffusionPolicy(c.actor, steps=c.settings['denoising_steps'], eta=c.settings['eta'],
            std_floor=c.settings['std_floor'], guidance_scale=c.settings['guidance_scale'],
            cfg_batch=c.settings['cfg_batch'], std_schedule=c.settings.get('std_schedule'))
        c.sampler = FullMusicSampler(c.catalog, split='train', seed=c.stage['seed']+1000003*collective.rank,
            window_seconds=c.settings['episode_seconds'], random_start=c.stage['dataset']['random_start'],
            source_probabilities=c.stage['dataset']['source_probabilities'])
        c.bc = local_call(collective, lambda: SupervisedAnchor(c.config, c.actor, train_config) if collective.rank == 0 else None)
        c.samplers = dict(music=c.sampler, **({'bc':c.bc} if c.bc is not None else {}))
        c.generators = dict(critic=torch.Generator().manual_seed(c.stage['seed']+2002),
                            actor=torch.Generator().manual_seed(c.stage['seed']+3003))
        c.identity = legacy.identity(config, preflight, provenance, c.catalog, audit, c.actor)
        c.state = dict(iteration=0, policy_version=0, actor_updates=0, critic_updates=0,
            optimizer_attempts=0, buffer_size=0, pending_plan=False, selected_actor_lr=c.settings['actor_lr'],
            last_durable_checkpoint_iteration=0)
        if args.resume:
            resume = root_call(collective, lambda: str(c.manager.latest_checkpoint() if (c.output/'latest.json').exists()
                                                       else c.output/'checkpoints/initial.pt'))
            if args.resume != 'latest' and Path(args.resume).resolve() != Path(resume).resolve():
                raise ValueError('Resume requires latest complete checkpoint of this run')
            c.state = local_call(collective, lambda: load_checkpoint(resume, actor=c.actor, critic=c.critic,
                actor_optimizer=c.actor_optimizer, critic_optimizer=c.critic_optimizer, identity=c.identity,
                samplers=c.samplers, generators=c.generators, rank=collective.rank, world_size=collective.world_size))
            root_call(collective, lambda: legacy.validate_resume_budget(c.state['budget'], c.budget.state_dict()))
            root_call(collective, lambda: c.manager.reconcile_accepted(c.state['iteration']))
            root_call(collective, lambda: c.maintenance.recover_archives())
        else:
            root_call(collective, lambda: atomic_json(c.output/'run.json', dict(schema='genmo.closedloop.stage10.run.v2',
                mode='train',
                identity=c.identity, initialization=dict(mode='stage1_actor_weights_only', critic='new'),
                input_config_sha256=sha256_file(args.config))))
        c.initial_iteration = c.state['iteration']
        report.update(session_id=c.session.name, initial_iteration=c.initial_iteration, world_size=collective.world_size,
            identity=c.identity, resume=(dict(requested=args.resume, training_resume=True,
                checkpoint=resume, sha256=sha256_file(resume), initial_iteration=c.initial_iteration,
                restored_full_state=True, old_buffer_discarded=True) if args.resume else None),
            data_audit='data_audit.json')
        def session_start():
            atomic_json(c.session/'data_audit.json', audit)
            atomic_json(c.session/'source_identity.json', provenance)
            (c.session/'resolved_config.yaml').write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
            atomic_json(c.session/'session_start.json', dict(report, schema='genmo.closedloop.stage10.session_start.v2'))
        root_call(collective, session_start)
        if c.initial_iteration >= args.stop_after_iteration:
            raise ValueError('Stop iteration must exceed restored iteration')
        _synchronize_models(c, full=True, phase='initialization')
        child_config = copy.deepcopy(c.config)
        child_config['runtime']['genmo_device'] = 'cuda:0'
        child_config['runtime']['asset_conversion_dir'] = str(c.rank_dir/'usd_assets')
        config_path = c.rank_dir/'resolved_config.yaml'
        config_path.write_text(yaml.safe_dump(child_config, allow_unicode=True, sort_keys=False))
        c.workers = Workers(c.config, c.rank_dir)
        socket = Path(c.workers.temp.name)/'gmt.sock'
        visible = os.environ['CUDA_VISIBLE_DEVICES'].split(',')[collective.rank].strip()
        client = local_call(collective, lambda: c.workers.start('gmt', [config['paths']['isaac_python'], '-B',
            str(Path(config['paths']['gmt_repo'])/'scripts/rsl_rl/serve_frozen_gmt.py'), '--config', str(config_path),
            '--socket', str(socket), '--headless'], config['paths']['gmt_repo'], socket,
            environment={'CUDA_VISIBLE_DEVICES':visible}, strip_distributed=True))
        c.journal = GuardedStepJournal(c.rank_dir/'bootstrap_journal.sqlite', c.guard)
        c.backend = AcknowledgedBackend(client, c.journal, socket_path=socket, timeout_s=config['runtime']['rpc_timeout_s'])
        c.builder = OnlineConditionBuilder(BumiMotionFeatureCodec(BumiKinematics(config['paths']['kinematics'])))
        c.env = UpperEnvironment(c.config, c.backend, c.builder, c.policy, None, c.rank_dir/'bootstrap')
        c.env.disk_guard = c.guard
        if args.resume:
            c.env.policy_version, c.env.iteration = c.state['policy_version'], c.state['iteration']
            spent_by_rank = root_call(collective, lambda: [sum(value.get('generations', 0)
                for key, value in c.budget.state_dict()['phases'].items()
                if key.endswith(f'/rank{rank}') and '/eval_' not in key) for rank in range(collective.world_size)])
            saved_execution = c.state.pop('local_rank_state')
            legacy.restore_execution_state(c.env, saved_execution, spent_generations=spent_by_rank[collective.rank])
            atomic_json(c.rank_dir/'execution_restore.json', dict(saved_attempt=saved_execution['attempt'],
                restored_attempt=c.env.attempt, reason='retain_per_rank_spent_generation_gap'))
        c.state['profile_checks'] = _calibrate_and_profile(c, restored=bool(args.resume))
        if not args.resume:
            _write_checkpoint(c, reason='initial')
        if collective.rank == 0:
            from torch.utils.tensorboard import SummaryWriter
            c.writer = SummaryWriter(str(c.output/'tensorboard'))
        baseline_path = c.output/'evaluation_baseline.json'
        if not baseline_path.exists():
            baseline = _evaluate(c, 'initial')
            root_call(collective, lambda: atomic_json(baseline_path, baseline))
            report['evaluations'].append('initial')
        from .periodic_monitor import log_metrics
        while c.state['iteration'] < args.stop_after_iteration:
            latencies = collective.all_gather_object(c.env.latency_budget_s)
            def capacity():
                result = c.budget.iteration_capacity(c.settings['max_actor_optimizer_steps'])
                required = dict(generations=0, control_steps=0, physics_steps=0)
                evaluation = c.stage['evaluation']
                for rank, latency in enumerate(latencies):
                    samples = (4*evaluation['samples_per_source'] + collective.world_size-1-rank)//collective.world_size
                    decisions = samples*len(evaluation['seeds'])*(math.ceil(evaluation['episode_seconds']/.5)+2)
                    for credit in (collection_credit(c.settings['rollout_upper_steps_per_rank'], c.settings['episode_seconds'], latency),
                                   collection_credit(decisions, evaluation['episode_seconds'], latency)):
                        for key, value in credit.items():
                            required[key] += value
                state = c.budget.state_dict()
                for key, value in required.items():
                    left = state['limits'][key]-state['used'][key]
                    if left < value:
                        result['exhausted'][key] = dict(required=value, remaining=left,
                            includes_final_evaluation_reserve=True)
                result['can_start'] = not result['exhausted']
                return dict(stop=stop.stop_requested or c.maintenance.expired(), capacity=result)
            control = root_call(collective, capacity)
            requested = collective.all_gather_object(stop.stop_requested)
            if control['stop'] or any(requested) or not control['capacity']['can_start']:
                report['stop_reason'] = 'signal_walltime_or_budget'
                report['stop_details'] = control
                break
            index = c.state['iteration'] + 1
            root_call(collective, lambda: c.manager.check_disk(c.stage['storage']['checkpoint_reserve_bytes'], refresh=True))
            torch.cuda.reset_peak_memory_stats(collective.device)
            start = time.perf_counter()
            buffer, targets, manifest, directory, path, collection = _collect(c, index)
            update = _update(c, buffer, targets, manifest, index)
            update['gpu_memory_by_rank'] = {f'rank{i}': row for i, row in enumerate(collective.all_gather_object(dict(
                allocated_bytes=torch.cuda.memory_allocated(collective.device),
                reserved_bytes=torch.cuda.memory_reserved(collective.device),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(collective.device))))}
            gmt = local_call(collective, lambda: c.backend.call('verify_frozen'))
            local_call(collective, lambda: legacy._assert_frozen(gmt))
            sources_ok = root_call(collective, lambda: verify_source_provenance(provenance)['unchanged'])
            if not sources_ok:
                raise RuntimeError('Training source changed during the run')
            replica = _synchronize_models(c, full=index % c.stage['checks']['full_fingerprint_every'] == 0)
            frozen_by_rank = collective.all_gather_object(gmt)
            summaries = collective.all_gather_object(collection)
            rewards = collective.all_gather_object(dict(reward=sum(float(t.rewards.sum())+float(t.metadata.get('event_reward',0.)) for t in buffer.transitions),
                control_steps=sum(t.executed_control_steps for t in buffer.transitions),
                physical_failures=sum(bool(t.metadata.get('terminal_snapshot', {}).get('terminated', False)) for t in buffer.transitions)))
            buffer.clear()
            c.journal.close()
            closed = collective.all_gather_object(str(c.journal.path))
            budget = root_call(collective, lambda: (c.budget.accept_iteration(), c.budget.state_dict())[1])
            c.state.update(iteration=index, policy_version=c.state['policy_version']+1,
                actor_updates=c.state['actor_updates']+update['actor']['optimizer_steps'],
                critic_updates=c.state['critic_updates']+update['critic']['optimizer_steps'],
                optimizer_attempts=budget['used']['optimizer_attempts'], budget=budget)
            c.env.iteration, c.env.policy_version = index, c.state['policy_version']
            seconds = max(collective.all_gather_object(time.perf_counter()-start))
            reward = sum(row['reward'] for row in rewards); duration = sum(row['control_steps'] for row in rewards)*.02
            summary = dict(schema='genmo.closedloop.stage10.iteration.v2', iteration=index, status='accepted',
                policy_version_before=c.state['policy_version']-1, policy_version_after=c.state['policy_version'],
                global_manifest=manifest, actor_updates_total=c.state['actor_updates'], critic_updates_total=c.state['critic_updates'],
                gmt_frozen_by_rank=frozen_by_rank, source_unchanged={'unchanged':sources_ok},
                **update, collectors=summaries, replicas=replica,
                reward=reward, executed_seconds=duration, reward_per_second=reward/duration if duration else None,
                physical_failures=sum(row['physical_failures'] for row in rewards), seconds=seconds,
                transitions_per_second=len(manifest)/seconds, simulated_seconds_per_second=duration/seconds,
                actor_lr=c.settings['actor_lr'], kl_limit=c.settings['kl_stop_joint'], budget=budget)
            def seal():
                atomic_json(directory/'summary.json', summary)
                c.manager.seal_iteration(directory, index, closed_journals=closed)
                c.manager.append_metrics(dict(event='iteration_accepted', **summary))
                curves = {key: value for key, value in summary.items() if key not in ('global_manifest', 'budget', 'replicas')}
                curves['actor_minibatches'] = {f'step{i+1}': row for i, row in enumerate(update['actor']['steps'])}
                curves['denoising_steps'] = {f'step{i}': row for i, row in enumerate(update['kl']['per_denoising_step'])}
                curves['collectors'] = {f'rank{i}': dict(row, wait_after_collection_seconds=
                    max(r['seconds'] for r in summaries)-row['seconds']) for i, row in enumerate(summaries)}
                log_metrics(c.writer, c.output/'curves.jsonl', index, curves)
                return True
            root_call(collective, seal)
            if checkpoint_due(index, c.stage['storage']['checkpoint_every_iterations'], normal_end=index == args.stop_after_iteration):
                _write_checkpoint(c, reason='periodic' if index % c.stage['storage']['checkpoint_every_iterations'] == 0 else 'normal_end')
            root_call(collective, lambda: c.maintenance.enqueue_archive(directory))
            root_call(collective, lambda: _flush_archive_metrics(c))
            if index % c.stage['evaluation']['every_iterations'] == 0:
                _evaluate(c, f'{index:06d}'); report['evaluations'].append(index)
            report['iterations'].append(index)
            if collective.rank == 0:
                print(f'[ACCEPTED v2] iteration={index} actor_steps={update["actor"]["optimizer_steps"]} '
                      f'lr={c.settings["actor_lr"]} KL={update["kl"]["mean_joint_kl"]:.8g} seconds={seconds:.2f}', flush=True)
        if c.state['iteration'] > c.initial_iteration:
            _write_checkpoint(c, reason='controlled_end')
            if c.state['iteration'] not in report['evaluations']:
                _evaluate(c, f'final_{c.state["iteration"]:06d}')
        report.update(status='passed', final_state=c.state)
    except BaseException as caught:
        error = caught
        report.update(status='failed', error=dict(type=type(caught).__name__, message=str(caught), traceback=traceback.format_exc()))
        traceback.print_exc()
    finally:
        if c.workers is not None:
            if c.backend is not None and c.workers.entries:
                c.workers.entries[-1]['client'] = c.backend.client
            try:
                c.workers.close()
                report['worker_shutdown'] = c.workers.shutdown
                if not error and any(item.get('close_error') or item.get('forced_shutdown')
                    or item.get('process_exit_code') != 0 for item in c.workers.shutdown.values()):
                    raise RuntimeError('Frozen GMT did not shut down cleanly')
            except Exception as caught:
                error = error or caught
        try:
            if c.journal is not None:
                c.journal.close()
            if hasattr(c, 'identity'):
                report['source_unchanged'] = root_call(collective, lambda: verify_source_provenance(provenance))
                report['original_assets_unchanged'] = root_call(collective,
                    lambda: all(sha256_file(config['paths'][key]) == digest
                                for key, digest in preflight['asset_sha256'].items()))
                if not report['source_unchanged']['unchanged'] or not report['original_assets_unchanged']:
                    error = error or RuntimeError('Source or original assets changed during this session')
            # root drains all immutable archives before releasing the run lock.
            def close_manager():
                if c.manager is not None:
                    try:
                        # 先drain和记录计时，再close释放写锁。
                        if c.maintenance is not None:
                            c.maintenance.drain()
                    finally:
                        _flush_archive_metrics(c)
                        c.manager.close()
            root_call(collective, close_manager)
            if c.writer is not None:
                c.writer.close()
            failures = collective.all_gather_object(None if error is None else f'{type(error).__name__}: {error}')
            if any(failures):
                error = error or RuntimeError('; '.join(value for value in failures if value))
            report.update(status='failed' if error else 'passed', exit_code=1 if error else 0,
                          shutdown_failures=failures, archives_drained=not any(failures))
            if hasattr(c, 'rank_dir'):
                atomic_json(c.rank_dir/'session_summary.json', report)
                shutdown = collective.all_gather_object(report.get('worker_shutdown'))
                rank_reports = collective.all_gather_object(dict(rank=collective.rank, status=report['status'],
                    worker_shutdown=report.get('worker_shutdown'), exit_code=report['exit_code']))
                def complete():
                    report['worker_shutdown_by_rank'] = shutdown
                    report['rank_reports'] = rank_reports
                    atomic_json(c.session/'summary.json', report)
                    atomic_json(c.session/'completion.json', dict(schema='genmo.closedloop.stage10.session_completion.v2',
                        session_id=c.session.name, status=report['status'], exit_code=report['exit_code'],
                        summary_sha256=sha256_file(c.session/'summary.json'),
                        archive_shutdown_complete=not any(failures)))
                root_call(collective, complete)
        except Exception as caught:
            error = error or caught
            traceback.print_exc()
        stop.restore()
    return 1 if error else 0
