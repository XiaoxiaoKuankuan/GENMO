"""第二阶段八 rank、多 minibatch 与稀疏 checkpoint 的只读产物审计。

本模块由原 Stage10 审计器仅在 run.v2 时调用；v1 的单次 Actor 更新、学习率候选
和逐轮 checkpoint 严格规则保持原样。v2 以不可变 iteration seal 选择接受轮次，
逐 rank 核验原始 transition、自由坐标、旧概率、执行步数和独立 GAE，再在全局
上层样本上重新验证优势标准化。Actor 更新次数按真实 minibatch 累计，不能把
外层轮次当作 Adam 步数；普通轮明确标记没有模型文件，不虚构逐轮完整恢复点。

恢复链从接受指针、session 的明确完整恢复证据及原 checkpoint 发布记录回溯。
未持久尾部必须有 immutable superseded 事件，保留其执行字节和已花资源，仅从
当前策略链中排除。v2 归档绑定独立 seal，可以在没有同轮 checkpoint 时审计。
所有模型只用 CPU mmap 读取元数据；不执行模型 forward、GPU、PhysX 或训练。
运行时梯度和 replica 报告只能证明其声明的检查范围，不能证明动作质量或收敛。

正常结束还必须有绑定 SHA 的初始化和终态周期评估：重新读取各 rank 的计划、
逐任务证据及其 SHA，重算奖励、执行时长和物理失败统计，再与合并报告和固定
初始基线比较。物理跌倒是有效的策略评估结果，不冒充基础设施故障；缺任务、
证据不一致或评估改变模型/RNG 才使这项审计失败。旧 v2 缺少新增证据时仍能读取
其他历史，但评估项明确缺失，不补写历史，也不宣称已经通过完整正式训练验收。

v2 的 old/next value 权威来源是已经核验 SHA 的不可变 rollout；生产 fixed_targets
只保存 GAE 结果，不强制重复序列化这两列。审计用独立 NumPy 递推重算目标，若
历史目标文件包含可选的重复旧值则额外交叉核验，不补写文件或改变原始归档 SHA。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from gem.closedloop.dppo.budget_ledger import expand_budget_reference, read_budget_state
from gem.closedloop.dppo.checkpoint import VERSION_V2, _validate_rank_states
from gem.closedloop.dppo.run_management import TrainingBudget
from tools.eval.audit_stage10_vector_helpers import is_vector,lane_sessions,audit_vector_rows,audit_vector_checkpoint
from tools.eval.audit_closedloop_stage10 import (
    _physical,
    _retired_checkpoint,
    _session_evidence,
    archived_execution,
    audit_data,
    audit_rollout,
    budget_check,
    close,
    integer,
    number,
    read_json,
    require,
    resolve,
    sha256,
)


def _session(root, directory):
    start = read_json(directory/'session_start.json')
    require(start.get('schema')=='genmo.closedloop.stage10.session_start.v2'
            and start.get('session_id')==directory.name and start.get('world_size')==8,
            'V2 session requires its eight-rank durable start identity')
    integer(start['initial_iteration'], 'V2 initial iteration')
    summary_path = directory/'summary.json'
    summary = read_json(summary_path) if summary_path.exists() else None
    if summary is not None:
        require(summary.get('schema')=='genmo.closedloop.stage10.session.v2'
                and summary.get('session_id')==directory.name, 'V2 session summary identity differs')
        for key in ('initial_iteration', 'resume'):
            require(summary.get(key)==start.get(key), 'V2 session differs from durable start/resume evidence')
    evidence = _session_evidence(root, directory)
    return dict(start=start, summary=summary, directory=directory, started_at=evidence['first_time'])


def _completion(session, *, recovered=False):
    directory, summary = session['directory'], session['summary']
    marker = directory/'completion.json'
    if summary is None:
        require(not marker.exists(), 'Completion exists without a V2 summary')
        if not recovered:
            raise FileNotFoundError(directory/'summary.json')
        return dict(disposition='recovered_interruption')
    if marker.exists():
        completion = read_json(marker)
        require(completion.get('schema') in ('genmo.closedloop.stage10.session_completion.v1',
                'genmo.closedloop.stage10.session_completion.v2')
                and completion.get('summary_sha256')==sha256(directory/'summary.json')
                and completion.get('status')==summary.get('status')
                and completion.get('exit_code')==summary.get('exit_code'), 'V2 completion marker differs')
    elif not recovered:
        raise FileNotFoundError(marker)
    if not recovered:
        require(summary.get('status')=='passed' and summary.get('exit_code')==0,
                'V2 terminal session did not complete successfully')
        require(summary.get('source_unchanged', {}).get('unchanged') is True
                and summary.get('original_assets_unchanged') is True,
                'V2 session source/assets verification is missing or failed')
        ranks = summary.get('rank_reports')
        require(isinstance(ranks, list) and len(ranks)==8, 'V2 completion requires all rank shutdown reports')
        for rank, report in enumerate(ranks):
            require(report.get('rank', rank)==rank and report.get('status')=='passed', 'A V2 rank did not complete')
            gmt = report.get('worker_shutdown', {}).get('gmt', {})
            require(gmt.get('policy_unchanged') is True and gmt.get('runtime_parameters_unchanged') is True
                    and gmt.get('process_exit_code')==0 and not gmt.get('forced_shutdown')
                    and not gmt.get('close_error'), 'A V2 frozen worker did not close cleanly')
    else:
        if 'source_unchanged' in summary:
            require(summary['source_unchanged'].get('unchanged') is True, 'Recovered V2 sources changed')
        if 'original_assets_unchanged' in summary:
            require(summary['original_assets_unchanged'] is True, 'Recovered V2 original assets changed')
    return dict(status=summary.get('status'), recovered=recovered)


def _periodic_evaluations(root, session, identity, *, require_final=False):
    """只读重放评估汇总；终态必须绑定真实报告，不把物理失败当成执行证据失败。"""
    from gem.closedloop.dppo.periodic_monitor import merge_periodic_reports

    summary, directory = session['summary'], session['directory']
    require(summary is not None, 'V2 evaluation requires a completed session summary')
    artifacts = summary.get('evaluation_artifacts')
    if artifacts is None:
        raise FileNotFoundError(f'{directory}/evaluation_artifacts: legacy V2 has no SHA-bound evaluation evidence')
    require(isinstance(artifacts, list), 'V2 evaluation artifacts must be a list')
    declared = [f'{label:06d}' if type(label) is int else label for label in summary.get('evaluations', [])]
    require(all(isinstance(label, str) and label for label in declared)
            and len(declared)==len(set(declared)), 'V2 evaluation labels are invalid or repeated')
    labels = [item.get('label') for item in artifacts]
    require(len(labels)==len(set(labels)) and set(labels)==set(declared),
            'V2 evaluation SHA records differ from completed evaluation labels')
    config = yaml.safe_load((directory/'resolved_config.yaml').read_text())
    expected = config['stage10']['evaluation']
    baseline_path = root/'evaluation_baseline.json'
    baseline = read_json(baseline_path)
    require(baseline.get('iteration')==0 and baseline.get('status')=='passed',
            'V2 periodic baseline is not the completed initial policy evaluation')
    def rebuild(value, owner, label, reference):
        paths = sorted((owner/'phases'/f'eval_{label}').glob('rank*/report/report.json'))
        require(len(paths)==min(8, value['plan']['selected_sample_count']), 'V2 periodic rank reports are missing')
        for rank_path in paths:
            rank_report = read_json(rank_path)
            selection = read_json(resolve(root, rank_report['selection_path'], rank_path.parent))
            require(selection['periodic_partition']['world']==8, 'V2 periodic evaluation is not eight-rank partitioned')
            actor = rank_report.get('actor_identity', {})
            require(actor.get('iteration')==value['iteration']
                    and actor.get('policy_version')==value['iteration'],
                    'V2 periodic rank report evaluated another policy version')
        recomputed = merge_periodic_reports(value['plan'], paths, baseline=reference,
                                            evaluation_identity=value['evaluation_identity'])
        require(all(value.get(key)==result for key, result in recomputed.items()),
                'V2 periodic aggregate or episode metrics cannot be reproduced from original evidence')
    # 初始 session 即使后来中断，也必须从其完整原始任务重算基线，不能只相信根目录副本。
    initial_directory = resolve(root, str(root/'sessions'/baseline['session_id']))
    require(read_json(initial_directory/'evaluations/initial.json')==baseline,
            'V2 durable initial baseline differs from its original report')
    rebuild(baseline, initial_directory, 'initial', None)
    records = []
    for item in artifacts:
        label = item['label']
        require(label=='initial' or label.isdecimal() or
                (label.startswith('final_') and label[6:].isdecimal()), 'V2 evaluation label is not canonical')
        iteration = integer(item['iteration'], 'V2 evaluated iteration')
        require(iteration==(0 if label=='initial' else int(label.removeprefix('final_'))),
                'V2 evaluation label differs from policy iteration')
        path = resolve(root, item['path'])
        require(path==directory/'evaluations'/f'{label}.json' and sha256(path)==item['sha256'],
                'V2 evaluation report path/SHA differs from its session publication')
        report = read_json(path)
        require(report.get('iteration')==iteration and report.get('session_id')==directory.name,
                'V2 evaluation report belongs to another session or policy iteration')
        plan = report['plan']
        require(plan['catalog_identity']==identity['dataset'] and plan['seeds']==expected['seeds']
                and plan['periodic_subset']['samples_per_source']==expected['samples_per_source']
                and plan['episode_seconds']==expected['episode_seconds']
                and plan['task_count']==4*expected['samples_per_source']*len(expected['seeds']),
                'V2 periodic plan differs from the configured dataset, samples, seeds or duration')
        require(report['plan_sha256']==baseline['plan_sha256'] and plan==baseline['plan'],
                'V2 periodic evaluation changed the fixed initial plan')
        evaluation_identity = report['evaluation_identity']
        require(evaluation_identity.get('training_identity')==identity
                and evaluation_identity==baseline['evaluation_identity'],
                'V2 periodic evaluation changed its training or execution identity')
        rebuild(report, directory, label, None if iteration==0 else baseline)
        for publication_path in (root/'checkpoints/publications').glob('*.json'):
            publication = read_json(publication_path)
            if publication.get('session_id')==directory.name and publication.get('iteration')==iteration:
                fingerprint = publication['metadata'].get('actor_model_fingerprint')
                if fingerprint is None:
                    raise FileNotFoundError(f'{publication_path}: legacy V2 lacks evaluation/model fingerprint binding')
                require(report['actor_identity']['model_fingerprint']==fingerprint,
                        'V2 periodic evaluation differs from the published checkpoint model')
        if iteration==0:
            require(report==baseline, 'V2 durable initial baseline differs from its published report')
        records.append(dict(label=label, iteration=iteration, path=str(path.relative_to(root)),
            sha256=item['sha256'], task_count=report['task_count'],
            physical_failure_count=report['physical_failure_count'],
            mean_executed_seconds=report['mean_executed_seconds'],
            source_balanced_reward=report['source_balanced_reward'],
            infrastructure_complete=True, physical_failures_are_valid_policy_outcomes=True))
    if require_final:
        final = summary['final_state']['iteration']
        require(any(row['iteration']==final and row['label']!='initial' for row in records),
                'V2 successful terminal session has no final-policy evaluation')
    return dict(reports=records, full_heldout_acceptance=False,
                scope='SHA-bound fixed periodic subset and independently recomputed execution metrics')


def _targets(root, summary, rows, contract, *, normalization):
    """兼容生产 v2 目标格式；旧值来自不可变链，独立重算按真实时间的 GAE。"""
    target = torch.load(_physical(resolve(root, summary['targets_path'])), map_location='cpu',
                        weights_only=False, mmap=True)
    size = len(rows)
    def vector(name):
        value = torch.as_tensor(target[name]).double().cpu().numpy()
        require(value.shape==(size,) and np.isfinite(value).all(), f'Invalid V2 target vector: {name}')
        return value
    old = np.asarray([row['old_value'] for row in rows], dtype=np.float64)
    nxt = np.asarray([row['next_value'] for row in rows], dtype=np.float64)
    require(np.isfinite(old).all() and np.isfinite(nxt).all(), 'V2 immutable rollout values are nonfinite')
    duplicates = []
    for name, expected in (('old_values', old), ('next_values', nxt)):
        if name in target:
            require(np.allclose(vector(name), expected, atol=1e-7, rtol=0),
                    'V2 duplicated old/next values differ from immutable rollout values')
            duplicates.append(name)
    gamma, lam = float(contract['gamma_upper'])**(1/25), float(contract['lambda_upper'])**(1/25)
    close(target['gamma_low'], gamma, 'V2 gamma_low')
    close(target['lambda_low'], lam, 'V2 lambda_low')
    valid = torch.as_tensor(target['valid'])
    require(valid.dtype==torch.bool and valid.shape==(size,) and bool(valid.all()),
            'Accepted V2 rollout contains invalid fixed targets')
    discounted = np.asarray([sum((gamma**i)*reward for i, reward in enumerate(row['rewards']))+
        (gamma**max(row['count']-1, 0))*row['event_reward'] for row in rows], dtype=np.float64)
    advantages = np.zeros(size, dtype=np.float64)
    for index in range(size-1, -1, -1):
        row = rows[index]
        advantages[index] = discounted[index]-old[index]
        if not row['terminated'] and row['has_next']:
            advantages[index] += gamma**row['count']*nxt[index]
        if index+1<size:
            following = rows[index+1]
            continuous = (not row['terminated'] and not row['truncated'] and row['end']==following['begin']
                and row['identity'].get('env_id')==following['identity'].get('env_id')
                and all(row['identity'][key]==following['identity'][key]
                        for key in ('backend_session_id', 'episode_id', 'policy_version')))
            if continuous:
                advantages[index] += (gamma*lam)**row['count']*advantages[index+1]
    mean, std = normalization
    require(target.get('advantages_normalized') is True and
            target.get('advantage_normalization_scope')=='global_valid_upper_transitions',
            'V2 advantages must be globally normalized exactly once')
    close(target['advantage_global_mean'], mean, 'V2 global advantage mean')
    close(target['advantage_global_std'], std, 'V2 global advantage std')
    differences = {}
    for name, expected in (('discounted_rewards', discounted), ('advantages_raw', advantages),
            ('returns', advantages+old), ('advantages', (advantages-mean)/max(float(std), 1e-8))):
        difference = float(np.max(np.abs(vector(name)-expected)))
        require(difference<=1e-7, f'Independent V2 fixed-target/GAE mismatch: {name} {difference}')
        differences[name] = difference
    return dict(max_abs_differences=differences,
        value_source='SHA-verified immutable rollout pre-update old_value/next_value',
        optional_duplicate_columns_verified=duplicates)


def _seal(root, path):
    seal = read_json(path)
    require(seal.get('schema')=='genmo.closedloop.stage10.iteration_seal.v2'
            and seal.get('journals_closed') is True, 'V2 iteration is not sealed with closed journals')
    index = integer(seal['iteration'], 'Sealed outer iteration', 1)
    directory = path.parent
    require(directory.name==f'{index:06d}' and directory.parent.name=='iterations'
            and directory.parent.parent.name==seal.get('session_id'), 'V2 seal directory/session identity differs')
    summary_path = directory/'summary.json'
    summary = read_json(summary_path)
    require(sha256(summary_path)==seal['summary_sha256'] and summary.get('iteration')==index
            and summary.get('schema')=='genmo.closedloop.stage10.iteration.v2'
            and summary.get('status')=='accepted', 'V2 seal binds a changed or unaccepted summary')
    names = set()
    for member in seal['members']:
        name = member['path']
        relative = Path(name)
        require(isinstance(name, str) and name and not relative.is_absolute() and '..' not in relative.parts
                and str(relative)==name and name not in names, 'Invalid or duplicated V2 seal member')
        require(name not in ('summary.json', 'seal_manifest.json', 'archive_manifest.json',
                'execution_evidence.tar.gz') and not name.endswith(('-wal', '-shm', '.tmp')),
                'V2 seal contains retained metadata or an open temporary journal')
        names.add(name)
        integer(member['size_bytes'], 'V2 seal member size')
        require(isinstance(member['sha256'], str) and len(member['sha256'])==64, 'V2 seal member SHA missing')
    require(names, 'V2 seal contains no execution evidence')
    return dict(seal=seal, summary=summary, directory=directory, path=path)


def _verify_members(root, record):
    directory, seal = record['directory'], record['seal']
    for member in seal['members']:
        path = resolve(root, str(directory/member['path']))
        physical = _physical(path)
        require(physical.is_file() and not physical.is_symlink()
                and physical.stat().st_size==member['size_bytes'] and sha256(path)==member['sha256'],
                'V2 sealed execution member size/SHA differs')


def _step_sampling(step, indices, contract):
    """核对抽样分母和可重建步号，旧全步记录继续可读；硬KL覆盖另外核验。"""
    from gem.closedloop.dppo.denoising_sampling import plan_record, validate_step_count
    full = contract['denoising_steps']
    selected = validate_step_count(full, contract.get('denoising_steps_per_chain'))
    require(step['internal_transitions']==len(indices)*selected,
            'V2 optimizer minibatch sampled denominator differs')
    sampling = step.get('denoising_sampling')
    if selected != full or sampling is not None:
        require(isinstance(sampling, dict) and step.get('full_internal_transitions')==len(indices)*full,
                'V2 sampled update lacks full behavior chain accounting')
        expected = plan_record(sampling.get('seed'), indices, full, selected)
        require(sampling==expected, 'V2 denoising sampling identity, weights or coverage differ')
    return selected


def _update(summary, contract):
    probability = summary['probability_check']
    require(probability.get('passed') is True and probability.get('scope') in (
        'full_rollout_before_first_step', 'rank_sentinel_all_denoising_steps'), 'V2 old probability check missing')
    for key, limit in (('max_abs_log_probability_difference', 1e-4), ('max_abs_ratio_minus_one', 1e-3),
                       ('max_abs_independent_gaussian_difference', 1e-8)):
        require(0<=number(probability[key], key)<=limit, 'V2 old probability tolerance exceeded')
    actor, critic = summary['actor'], summary['critic']
    steps = integer(actor['optimizer_steps'], 'V2 Actor optimizer steps', 1)
    require(steps<=contract['max_actor_optimizer_steps'] and steps==actor['optimizer_attempts']==len(actor['steps'])
            and actor.get('old_statistics_fixed') is True and actor.get('hard_kl_pending') is True
            and actor.get('rollback_scope')=='caller_owned_whole_rollout', 'V2 Actor step/fixed-statistics contract differs')
    require(actor.get('objective_logprob_reduction')==contract.get('objective_logprob_reduction', 'joint_sum'),
            'V2 objective differs from the bound training contract')
    manifest = summary['global_manifest']
    eligible = [i for i, item in enumerate(manifest) if item['valid'] and item['has_free']]
    require(actor['included_upper_transitions']==len(eligible)
            and actor['excluded_upper_transitions']==len(manifest)-len(eligible), 'V2 Actor eligibility differs')
    orders = actor['epoch_orders']
    require(len(orders)==actor['epochs_started'] and 1<=len(orders)<=contract['ppo_epochs']
            and all(sorted(order)==eligible for order in orders), 'V2 PPO epoch does not cover eligible chains once')
    upper_batch = contract['actor_minibatch_internal_transitions']//contract['denoising_steps']
    expected = [(epoch, order[start:start+upper_batch]) for epoch, order in enumerate(orders)
                for start in range(0, len(order), upper_batch)]
    require(steps<=len(expected), 'V2 Actor steps exceed the declared epoch/minibatch schedule')
    bc_samples = 0
    for position, (step, (epoch, indices)) in enumerate(zip(actor['steps'], expected), 1):
        require(step['optimizer_step']==position and step['epoch']==epoch and step['global_upper_indices']==indices,
                'V2 optimizer minibatch indices/denominator differ')
        selected_steps = _step_sampling(step, indices, contract)
        require(number(step['ppo_only_gradient_norm'], 'V2 PPO-only gradient')>0
                and number(step['total_gradient_norm'], 'V2 total gradient')>0, 'V2 Actor gradient is not positive')
        require(0<=number(step['clip_fraction'], 'V2 clip fraction')<=1
                and step.get('ratio_scope')=='before_this_minibatch_step_against_fixed_rollout_old_policy',
                'V2 ratio is not measured against fixed rollout probabilities')
        require(step['learning_rates'] and all(rate==summary['actor_lr'] for rate in step['learning_rates']),
                'V2 fixed Actor learning rate changed')
        if contract['bc_weight']>0:
            bc = step['bc']
            require(isinstance(bc, dict) and bc.get('batch_size')==contract.get('bc_batch', 2),
                    'V2 BC must run once per Actor step with its global batch')
            if 'weight' in bc:
                close(bc['weight'], contract['bc_weight'], 'V2 BC weight', 0.)
            bc_samples += bc['batch_size']
    require(actor['bc_global_samples']==bc_samples, 'V2 BC global sample count differs')
    if 'denoising_steps_per_chain' in actor or selected_steps != contract['denoising_steps']:
        require(actor.get('denoising_steps_per_chain')==selected_steps
                and actor.get('full_denoising_steps')==contract['denoising_steps']
                and actor.get('applied_internal_sample_visits')==sum(s['internal_transitions'] for s in actor['steps'])
                and actor.get('planned_internal_sample_visits')==len(eligible)*selected_steps*contract['ppo_epochs']
                and actor.get('full_objective_internal_transitions')==len(eligible)*contract['denoising_steps']*contract['ppo_epochs'],
                'V2 sampled/full objective visit counts differ')
        used_chains = {index for step in actor['steps'] for index in step['global_upper_indices']}
        require(actor.get('applied_unique_upper_chains')==len(used_chains),
                'V2 applied unique chain coverage differs')
        close(actor['applied_unique_chain_fraction'], len(used_chains)/len(eligible), 'V2 applied chain fraction')
    require(critic['optimizer_steps']==contract['critic_steps']
            and len(critic['losses'])==critic['optimizer_steps'], 'V2 Critic optimizer count differs')
    require(summary['actor_lr']==contract['actor_lr'] and summary['kl_limit']==contract['kl_stop_joint'],
            'V2 learning-rate/KL bound differs')
    kl = summary['kl']
    require(0<=number(kl['mean_joint_kl'], 'V2 joint KL')<=contract['kl_stop_joint'], 'V2 accepted KL exceeds hard limit')
    require(len(kl['per_denoising_step'])==contract['denoising_steps'], 'V2 KL step coverage differs')
    if selected_steps != contract['denoising_steps']:
        full_count = len(eligible)*contract['denoising_steps']
        require(kl.get('included_upper_transitions')==len(eligible)
                and [row.get('step_index') for row in kl['per_denoising_step']]==list(range(contract['denoising_steps']))
                and kl.get('effective_mean_change',{}).get('checked_internal_transitions')==full_count
                and kl.get('fresh_internal_forwards',0)+kl.get('reused_upper_transitions',0)*contract['denoising_steps']==full_count,
                'V2 sampled PPO requires final KL on every eligible chain and every behavior step')
    close(sum(number(row['mean_joint_kl'], 'V2 step KL') for row in kl['per_denoising_step'])/
          contract['denoising_steps'], kl['mean_joint_kl'], 'V2 mean KL across denoising steps')
    for field, key in (('max_joint_kl', 'kl_max_internal'), ('max_chain_joint_kl', 'kl_max_chain'),
                       ('max_step_mean_joint_kl', 'kl_max_step_mean')):
        if contract.get(key) is not None:
            require(0<=number(kl[field], field)<=contract[key], 'V2 configured KL maximum exceeded')
    frozen = summary['gmt_frozen_by_rank']
    require(len(frozen)==8 and len({row['execution_journal']['backend_session_id'] for row in frozen})==8,
            'V2 collectors do not have eight independent backend sessions')
    for row in frozen:
        require(row.get('policy_unchanged') is True and row.get('runtime_parameters_unchanged') is True
                and row['execution_journal']['executed_seq']==row['execution_journal']['acked_seq'],
                'V2 frozen backend changed or has an unacknowledged mutation')
        if 'lane_journals' in row:lane_sessions(row,len(row['lane_journals']))
    all_sessions=[r['execution_journal']['backend_session_id'] for r in frozen]
    all_sessions += [lane['backend_session_id'] for r in frozen for lane in r.get('lane_journals',[])]
    require(len(set(all_sessions))==len(all_sessions), 'V2 world/lane identities overlap across ranks')
    require(summary['source_unchanged'].get('unchanged') is True, 'V2 accepted sources changed')
    replica = summary['replicas']
    if replica.get('scope')=='sampled_parameter_values_not_full_hash':
        require(replica.get('passed') is True and replica.get('world_size')==8, 'V2 sampled replica check failed')
    else:
        models = replica['models']
        require(replica.get('scope')=='full_model_and_optimizer_sha256' and models.get('world_size')==8
                and models.get('replicas_identical') is True and len(models['ranks'])==8
                and len(replica['optimizers'])==8
                and all(value==replica['optimizers'][0] for value in replica['optimizers']),
                'V2 full model/optimizer replica check failed')
    return dict(actor_optimizer_steps=steps, optimizer_attempts=steps, critic_optimizer_steps=critic['optimizer_steps'],
                bc_global_samples=bc_samples, replica_check_scope=replica['scope'],
                parameter_change_scope='Runtime optimizer/gradient evidence; no per-iteration weight comparison')


def _audit_bc_rank_states(rank_states, contract, actor_updates, *, complete=True):
    """按不可变训练合同核对BC归属、局部样本量和更新计数，保留旧rank0格式。"""
    distribution = contract.get('bc_distribution', 'rank0')
    require(distribution in ('rank0', 'all_ranks'), 'V2 BC distribution is unknown')
    world = len(rank_states)
    global_batch = integer(contract.get('bc_batch', 2), 'V2 BC global batch', 1)
    sharded = distribution == 'all_ranks'
    require(not sharded or global_batch % world == 0, 'V2 BC global batch cannot be evenly sharded')
    local_batch = global_batch // world if sharded else global_batch
    active = contract['bc_weight'] > 0
    streams = []
    for rank, item in enumerate(rank_states):
        bc = item['samplers'].get('bc')
        owner = sharded or rank == 0
        require(owner or bc is None, 'V2 BC state is present on an unassigned rank')
        require(not (owner and active) or isinstance(bc, dict), 'V2 active BC rank state is missing')
        if bc is None:
            continue
        require(bc.get('batch_size') == local_batch, 'V2 BC local batch differs from the global normalization')
        require(bc.get('bc_update_steps') == (actor_updates if active else 0),
                'V2 BC accumulated updates differ from accepted Actor steps')
        if sharded and active and complete:
            generator = bc.get('generator')
            require(isinstance(generator, torch.Tensor) and generator.dtype == torch.uint8
                    and generator.ndim == 1, 'V2 sharded BC generator state is missing')
            # 各rank独立种子流，断点不能被rank0的BC快照覆盖。
            streams.append(generator.cpu().numpy().tobytes())
    require(not (sharded and active and complete) or len(set(streams)) == world,
            'V2 sharded BC RNG streams were duplicated across ranks')
    return dict(distribution=distribution, global_batch=global_batch, local_batch=local_batch,
                updates=actor_updates if active else 0,
                rng_scope=('distinct_rank_streams_checked' if sharded and active and complete else
                           'unavailable_in_retired_metadata' if not complete else 'not_sharded_or_inactive'))


def _checkpoint(root, publication, identity, summary, actor_updates, critic_updates, rank_rows):
    path = resolve(root, publication['path'])
    metadata = publication['metadata']
    require(metadata.get('world_size')==8 and metadata.get('actor_updates')==actor_updates
            and metadata.get('critic_updates')==critic_updates, 'V2 publication optimizer/topology metadata differs')
    if path.is_file():
        require(path.stat().st_size==publication['size_bytes'] and sha256(path)==publication['sha256'],
                'V2 published checkpoint size/SHA differs')
        saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        require({'actor', 'critic', 'actor_optimizer', 'critic_optimizer', 'rank_states',
                 'optimizer_layout', 'config'}.issubset(saved), 'V2 checkpoint lacks shared training state')
        _validate_rank_states(saved['rank_states'])
        interval = integer(saved['config']['stage10']['storage']['checkpoint_every_iterations'],
                           'V2 checkpoint outer interval', 1)
        require(metadata.get('reason') in ('periodic', 'normal_end', 'controlled_end')
                and (metadata['reason']!='periodic' or publication['iteration']%interval==0),
                'V2 periodic checkpoint was triggered by another counter')
        storage = dict(storage='complete_checkpoint', original_bytes_revalidated=True)
    else:
        saved, storage = _retired_checkpoint(root, path, publication)
    require(saved.get('version')==VERSION_V2 and saved.get('identity')==identity and saved.get('world_size')==8
            and saved.get('restore_environment')=='new_worker_session_and_reset', 'V2 checkpoint identity/topology differs')
    state = saved['state']
    require(state['iteration']==summary['iteration'] and state['policy_version']==summary['policy_version_after']
            and state.get('buffer_size')==0 and state.get('pending_plan') is False
            and state['actor_updates']==actor_updates and state['critic_updates']==critic_updates,
            'V2 checkpoint outer iteration or accumulated optimizer counts differ')
    require(len(saved['rank_states'])==8, 'V2 checkpoint is missing a rank')
    bc_audit = _audit_bc_rank_states(saved['rank_states'], identity['training_contract'], actor_updates,
                                   complete=storage['original_bytes_revalidated'])
    counters = []
    for rank, item in enumerate(saved['rank_states']):
        local = item['state']
        require(item['rank']==rank and local['iteration']==state['iteration']
                and local['policy_version']==state['policy_version']
                and local.get('buffer_size')==0 and local.get('pending_plan') is False,
                'V2 rank is not at the shared empty-buffer boundary')
        music = item['samplers']['music']
        require(music.get('split')=='train' and music.get('catalog_identity')==identity['dataset'],
                'V2 rank sampler differs from the full training catalog')
        if is_vector(identity):
            require(item['rng'].get('cuda_scope')=='local_device', 'V2 RNG must have rank-local scope')
            counters.append(audit_vector_checkpoint(local,rank_rows[rank],identity,rank))
            continue
        decision, attempt = integer(local['decision'], 'V2 decision'), integer(local['attempt'], 'V2 attempt')
        require(decision<=attempt and decision==rank_rows[rank][-1]['identity']['decision_id']+1,
                'V2 saved rank decision does not follow its real rollout')
        integer(local['episode_count'], 'V2 episode count')
        require(number(local['latency_budget_s'], 'V2 latency budget')>0, 'V2 rank latency budget invalid')
        require(item['rng'].get('cuda_scope')=='local_device', 'V2 RNG must have rank-local scope')
        counters.append(dict(rank=rank, decision=decision, attempt=attempt, episode_count=local['episode_count']))
    require(state['budget']==publication['budget'], 'V2 checkpoint/publication budget differs')
    budget_check(expand_budget_reference(root, state['budget']), expand_budget_reference(root, summary['budget']))
    for group in saved['actor_optimizer']['param_groups']:
        close(group['lr'], identity['training_contract']['actor_lr'], 'V2 checkpoint fixed LR', 0.)
    return dict(path=str(path.relative_to(root)), iteration=state['iteration'], actor_updates=actor_updates,
                critic_updates=critic_updates, rank_execution_counters=counters, bc_state_audit=bc_audit, **storage)


def audit_training_v2(root, run, audit, result, minimum_iterations, require_resume):
    require(run.get('mode')=='train' and run['identity'].get('distributed_training', {}).get('world_size')==8,
            'V2 audit requires explicit eight-rank parallel training')
    identity, contract = run['identity'], run['identity']['training_contract']
    sessions = {}
    for directory in sorted((root/'sessions').iterdir()):
        if not directory.is_dir():
            continue
        session = audit.check(f'v2_session_start:{directory.name}', lambda d=directory: _session(root, d))
        if session is not None:
            sessions[directory.name] = session
            audit.checks[-1]['details'] = dict(session_id=directory.name,
                initial_iteration=session['start']['initial_iteration'], summary_present=session['summary'] is not None)
    records = {}
    for path in sorted((root/'sessions').glob('*/iterations/*/seal_manifest.json')):
        record = audit.check(f'v2_seal:{path.relative_to(root)}', lambda p=path: _seal(root, p))
        if record is not None:
            key = (record['seal']['session_id'], record['seal']['iteration'])
            require(key not in records, 'Duplicate V2 seal for one session/outer iteration')
            records[key] = record
            audit.checks[-1]['details'] = dict(session_id=key[0], iteration=key[1],
                member_count=len(record['seal']['members']), summary_sha256=record['seal']['summary_sha256'])
    accepted = read_json(root/'accepted.json')
    require(accepted.get('schema')=='genmo.closedloop.stage10.accepted_iteration.v2' and 'seal' in accepted,
            'V2 accepted pointer does not name a sealed iteration')
    last = resolve(root, accepted['seal'])
    require(sha256(last)==accepted['seal_sha256'], 'V2 accepted pointer seal SHA differs')
    owner, final = accepted['session_id'], integer(accepted['iteration'], 'V2 accepted iteration', 1)
    publications = {}
    for path in sorted((root/'checkpoints/publications').glob('*.json')):
        value = read_json(path)
        require(value.get('schema')=='genmo.closedloop.stage10.checkpoint_publication.v1', 'V2 publication schema differs')
        publications[(value['session_id'], value['iteration'])] = (value, path)
    selected, owners, resumes = [], set(), []
    while True:
        require(owner in sessions and owner not in owners, 'V2 recovery chain lacks a start or contains a cycle')
        owners.add(owner)
        start = sessions[owner]['start']
        require(start.get('identity')==identity, 'V2 session start identity changed')
        initial = start['initial_iteration']
        require(final>=initial, 'V2 session accepted watermark precedes restored state')
        for index in range(initial+1, final+1):
            require((owner, index) in records, 'V2 accepted outer sequence has a missing seal')
            selected.append(records[(owner, index)])
        resume = start.get('resume')
        if initial==0 and resume is None:
            break
        require(resume and resume.get('training_resume') is True and resume.get('restored_full_state') is True
                and resume.get('old_buffer_discarded') is True and resume.get('initial_iteration')==initial,
                'V2 recovery lacks strict full-state/no-old-buffer evidence')
        resume_path = resolve(root, resume['checkpoint'])
        if initial==0:
            require(resume_path.name=='initial.pt' and sha256(resume_path)==resume['sha256'], 'V2 initial resume SHA differs')
            saved = torch.load(resume_path, map_location='cpu', weights_only=False, mmap=True)
            require(saved.get('version')==VERSION_V2 and saved.get('identity')==identity
                    and saved.get('world_size')==8 and saved['state']['iteration']==0
                    and saved['state'].get('buffer_size')==0 and saved['state'].get('pending_plan') is False
                    and {'actor_optimizer', 'critic_optimizer', 'optimizer_layout', 'rank_states'}.issubset(saved),
                    'V2 initial resume is not a full-state initial checkpoint')
            _validate_rank_states(saved['rank_states'])
            resumes.append(dict(session_id=owner, initial_iteration=0, final_iteration=final,
                                checkpoint=str(resume_path.relative_to(root))))
            break
        matches = [(key, value) for key, value in publications.items()
                   if key[1]==initial and resolve(root, value[0]['path'])==resume_path
                   and value[0]['sha256']==resume['sha256']]
        require(len(matches)==1 and matches[0][0][0]!=owner, 'V2 resume does not identify one prior durable publication')
        previous_record = records.get((matches[0][0][0], initial))
        following = records.get((owner, initial+1))
        require(previous_record is not None and following is not None, 'V2 resume has no pre/post rollout evidence')
        old_backends = [row['execution_journal']['backend_session_id'] for row in previous_record['summary']['gmt_frozen_by_rank']]
        new_backends = [row['execution_journal']['backend_session_id'] for row in following['summary']['gmt_frozen_by_rank']]
        require(set(old_backends).isdisjoint(new_backends), 'V2 resume reused an old physical backend session')
        resumes.append(dict(session_id=owner, initial_iteration=initial, final_iteration=final,
                            checkpoint=str(resume_path.relative_to(root))))
        owner, final = matches[0][0]
    selected.sort(key=lambda record: record['seal']['iteration'])
    selected_keys = {(record['seal']['session_id'], record['seal']['iteration']) for record in selected}
    superseded = set()
    events = []
    for path in sorted((root/'superseded_tails').glob('*.json')):
        event = read_json(path)
        previous = event['previous_accepted']
        require(event.get('schema')=='genmo.closedloop.stage10.superseded_tail.v2'
                and event.get('spent_budget_preserved') is True and event.get('session_id')==path.stem
                and path.stem in sessions and sessions[path.stem]['start']['initial_iteration']==event['durable_iteration'],
                'V2 superseded event does not bind its actual restored session')
        require(sha256(resolve(root, previous['seal']))==previous['seal_sha256'], 'V2 superseded pointer SHA differs')
        previous_seal = read_json(resolve(root, previous['seal']))
        require(previous_seal['session_id']==previous['session_id'] and previous_seal['iteration']==previous['iteration'],
                'V2 superseded pointer identifies another sealed state')
        for index in range(event['durable_iteration']+1, previous['iteration']+1):
            key = (previous['session_id'], index)
            require(key in records and key not in selected_keys, 'V2 superseded event removed a canonical or missing seal')
            superseded.add(key)
        events.append(dict(path=str(path.relative_to(root)), durable_iteration=event['durable_iteration'],
                           previous_accepted_iteration=previous['iteration']))
    require(set(records)-selected_keys<=superseded,
            'Noncanonical accepted V2 seal lacks explicit superseded-tail evidence')
    terminal = accepted['session_id']
    for name, session in sessions.items():
        recovered = name!=terminal and any(
            sessions[successor]['started_at'] is not None and session['started_at'] is not None
            and sessions[successor]['started_at']>session['started_at'] for successor in owners if successor!=name)
        audit.check(f'v2_session_completion:{name}', lambda s=session, r=recovered: _completion(s, recovered=r))
    evaluation_records = {}
    for name in sorted(owners):
        session = sessions[name]
        summary = session['summary']
        if summary is None or summary.get('status')!='passed':
            # 恢复链允许历史中断，不为未正常完成的旧进程虚构终态验证。
            continue
        evaluation_records[name] = audit.check(f'v2_periodic_evaluations:{name}',
            lambda s=session: _periodic_evaluations(root, s, identity, require_final=True))
    data_lookup = None
    for name in owners:
        def data_check(name=name):
            start = sessions[name]['start']
            data = read_json(resolve(root, start.get('data_audit', 'data_audit.json'), sessions[name]['directory']))
            lookup = audit_data(data)
            require(data['identity']==identity['dataset'] and data['data_content_sha256']==identity['data_content_sha256'],
                    'V2 data identity changed')
            return lookup
        lookup = audit.check(f'v2_data:{name}', data_check)
        if lookup is not None:
            data_lookup = lookup
            audit.checks[-1]['details'] = dict(sample_count=len(lookup), complete_manifest_scan=True)
    seen, counts, checkpoints = set(), {}, {}
    actor_total, critic_total, controls, previous_budget = 0, 0, 0, None
    for record in selected:
        summary, directory = record['summary'], record['directory']
        index = summary['iteration']
        def iteration_check(record=record, summary=summary, directory=directory, index=index):
            nonlocal actor_total, critic_total, controls, previous_budget
            require(data_lookup is not None, 'V2 complete dataset evidence is missing')
            require(summary['policy_version_before']==index-1 and summary['policy_version_after']==index,
                    'V2 policy advancement differs from outer iteration')
            require(len(summary['collectors'])==8, 'V2 accepted batch is missing a collector')
            reference = dict(targets_path=str(directory/'fixed_targets.pt'))
            rank_rows, rollouts, rank_summaries, raw = [], [], [], []
            with archived_execution(root, reference) as archive:
                _verify_members(root, record)
                for rank, collection in enumerate(summary['collectors']):
                    manifest = resolve(root, collection['rollout_manifest'])
                    require(manifest==directory/f'rank{rank:02d}'/'rollout/manifest.json', 'V2 collector manifest belongs to another rank')
                    local = dict(rollout_manifest=str(manifest), targets_path=str(manifest.parent.parent/'fixed_targets.pt'),
                        policy_version_before=index-1, collected_upper_transitions=collection['transition_count'], collection=collection)
                    rows, rollout = audit_rollout(root, local, data_lookup, contract, seen, identity=identity)
                    require(len(rows)==contract['rollout_upper_steps_per_rank'], 'V2 local rollout count differs')
                    if is_vector(identity):
                        audit_vector_rows(rows,collection,summary['gmt_frozen_by_rank'][rank],identity)
                    else:
                        decisions = [row['identity']['decision_id'] for row in rows]
                        require(all(right==left+1 for left, right in zip(decisions, decisions[1:])), 'V2 rank rollout decision IDs are not contiguous')
                        require(all(row['identity']['backend_session_id']==summary['gmt_frozen_by_rank'][rank]['execution_journal']['backend_session_id']
                                    for row in rows), 'V2 rollout belongs to another frozen backend')
                    rank_rows.append(rows)
                    rollouts.append(rollout)
                    rank_summaries.append(local)
                    target = torch.load(_physical(resolve(root, local['targets_path'])), weights_only=False, map_location='cpu', mmap=True)
                    raw.extend(torch.as_tensor(target['advantages_raw']).double().tolist())
                mean, std = float(np.mean(raw)), float(np.std(raw))
                targets = [_targets(root, local, rows, contract, normalization=(mean, std))
                           for local, rows in zip(rank_summaries, rank_rows)]
                for local in rank_summaries:
                    target = torch.load(_physical(resolve(root, local['targets_path'])), weights_only=False, map_location='cpu', mmap=True)
                    require(target['advantage_global_count']==len(raw)==contract['rollout_upper_steps'], 'V2 global normalization count differs')
            expected = [dict(owner_rank=rank, local_index=i, valid=True, has_free=row['free_coordinate_count']>0)
                        for rank, rows in enumerate(rank_rows) for i, row in enumerate(rows)]
            require(summary['global_manifest']==expected, 'V2 global sample ownership/free mask differs from actual rollout')
            update = _update(summary, contract)
            actor_total += update['actor_optimizer_steps']
            critic_total += update['critic_optimizer_steps']
            require(summary['actor_updates_total']==actor_total and summary['critic_updates_total']==critic_total,
                    'V2 cumulative optimizer counts differ from accepted minibatches')
            budget_check(expand_budget_reference(root, summary['budget']), expand_budget_reference(root, previous_budget))
            previous_budget = summary['budget']
            controls += sum(row['control_steps'] for row in rollouts)
            key = (record['seal']['session_id'], index)
            checkpoint = dict(storage='not_saved_this_iteration', iteration=index)
            if key in publications:
                checkpoint = _checkpoint(root, publications[key][0], identity, summary, actor_total, critic_total, rank_rows)
                checkpoints[index] = checkpoint
            counts[index] = update
            return dict(outer_iteration=index, actor_updates_total=actor_total, rollout_by_rank=rollouts,
                        fixed_targets_by_rank=targets, global_advantage_count=len(raw), update=update,
                        checkpoint=checkpoint, execution_archive=archive)
        audit.check(f'v2_iteration:{index}', iteration_check)
    # 作废尾部仍核对原件/归档 SHA，避免把恢复当作掩盖改写证据的理由。
    for key in sorted(set(records)-selected_keys):
        record = records[key]
        def historical_check(record=record, key=key):
            with archived_execution(root, dict(targets_path=str(record['directory']/'fixed_targets.pt'))) as archive:
                _verify_members(root, record)
            _update(record['summary'], contract)
            return dict(disposition='superseded_unsaved' if key in superseded else 'unselected', archive=archive)
        audit.check(f'v2_historical_seal:{key[0]}:{key[1]}', historical_check)
    def continuity():
        indices = [record['seal']['iteration'] for record in selected]
        require(len(indices)>=minimum_iterations and indices==list(range(1, accepted['iteration']+1)),
                'V2 accepted outer iteration sequence has gaps or too few updates')
        require(len(counts)==len(indices), 'V2 accepted iteration verification failed')
        return dict(accepted_outer_iterations=len(indices), actor_optimizer_steps=actor_total)
    audit.check('multiple_fresh_policy_iterations', continuity)
    audit.check('resume_then_optimize', lambda: require(not require_resume or resumes,
        'No V2 full-state resume followed by optimization') or resumes)
    def durable_check():
        latest = read_json(root/'latest.json')
        path = resolve(root, latest['publication'])
        require(read_json(path)=={key: value for key, value in latest.items() if key!='publication'}, 'V2 latest publication differs')
        require(latest['iteration'] in checkpoints and checkpoints[latest['iteration']]['storage']=='complete_checkpoint'
                and latest['iteration']<=accepted['iteration'], 'V2 latest is not a validated durable checkpoint')
        if sessions[terminal]['summary'] is not None and sessions[terminal]['summary'].get('status')=='passed':
            require(latest['iteration']==accepted['iteration'], 'V2 controlled completion did not save its final accepted state')
            final_state = sessions[terminal]['summary']['final_state']
            require(final_state['iteration']==accepted['iteration'] and final_state['actor_updates']==actor_total
                    and final_state['critic_updates']==critic_total, 'V2 final session optimizer/outer counters differ')
            budget_check(expand_budget_reference(root, final_state['budget']), expand_budget_reference(root, latest['budget']))
        return dict(accepted_iteration=accepted['iteration'], durable_iteration=latest['iteration'])
    audit.check('published_checkpoint_and_budget', durable_check)
    def persisted():
        budget = budget_check(read_budget_state(root/'budget.json'))
        if read_json(root/'budget.json').get('schema') == 'genmo.closedloop.stage10.budget.v1':
            TrainingBudget(root/'budget.json', budget['limits'])
        for record in records.values():
            budget_check(budget, expand_budget_reference(root, record['summary']['budget']))
        for session in sessions.values():
            if session['summary'] is not None and 'final_state' in session['summary']:
                budget_check(budget, expand_budget_reference(root, session['summary']['final_state']['budget']))
        require(budget['used']['accepted_iterations']>=len(selected)
                and budget['used']['optimizer_attempts']>=actor_total
                and budget['used']['control_steps']>=controls, 'V2 budget omitted accepted execution/optimizer consumption')
        return dict(used=budget['used'], spent_superseded_budget_preserved=True)
    audit.check('persisted_budget_monotonic', persisted)
    result.update(version='genmo.closedloop.stage10.audit.v2',
        periodic_evaluations=evaluation_records,
        execution_counter_contract='rank_local_verified' if len(counts)==len(selected) else 'required_not_verified',
        recovery_history=dict(sessions=[dict(session_id=name, selected=name in owners,
            disposition='terminal' if name==terminal else 'recovered_history') for name in sessions],
            superseded_tails=events, superseded_seal_count=len(superseded)),
        checkpoint_storage=dict(complete_checkpoints=sum(row['storage']=='complete_checkpoint' for row in checkpoints.values()),
            intentionally_retired_checkpoints=sum(row['storage']=='retired_metadata_only' for row in checkpoints.values()),
            outer_iterations_without_checkpoint=len(selected)-len(checkpoints),
            all_historical_weight_bytes_revalidated=False,
            sparse_checkpoint_scope='Only saved publications contain restorable model/optimizer bytes'))
