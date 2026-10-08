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
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from gem.closedloop.dppo.checkpoint import VERSION_V2, _validate_rank_states
from gem.closedloop.dppo.run_management import TrainingBudget
from tools.eval.audit_closedloop_stage10 import (
    _physical,
    _retired_checkpoint,
    _session_evidence,
    archived_execution,
    audit_data,
    audit_rollout,
    audit_targets,
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
        require(step['optimizer_step']==position and step['epoch']==epoch and step['global_upper_indices']==indices
                and step['internal_transitions']==len(indices)*contract['denoising_steps'],
                'V2 optimizer minibatch indices/denominator differ')
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
    require(critic['optimizer_steps']==contract['critic_steps']
            and len(critic['losses'])==critic['optimizer_steps'], 'V2 Critic optimizer count differs')
    require(summary['actor_lr']==contract['actor_lr'] and summary['kl_limit']==contract['kl_stop_joint'],
            'V2 learning-rate/KL bound differs')
    kl = summary['kl']
    require(0<=number(kl['mean_joint_kl'], 'V2 joint KL')<=contract['kl_stop_joint'], 'V2 accepted KL exceeds hard limit')
    require(len(kl['per_denoising_step'])==contract['denoising_steps'], 'V2 KL step coverage differs')
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
        require(not rank or 'bc' not in item['samplers'], 'V2 BC state must belong to rank zero only')
        if rank==0 and identity['training_contract']['bc_weight']>0:
            require('bc' in item['samplers'], 'V2 active BC state is missing from rank zero')
        decision, attempt = integer(local['decision'], 'V2 decision'), integer(local['attempt'], 'V2 attempt')
        require(decision<=attempt and decision==rank_rows[rank][-1]['identity']['decision_id']+1,
                'V2 saved rank decision does not follow its real rollout')
        integer(local['episode_count'], 'V2 episode count')
        require(number(local['latency_budget_s'], 'V2 latency budget')>0, 'V2 rank latency budget invalid')
        require(item['rng'].get('cuda_scope')=='local_device', 'V2 RNG must have rank-local scope')
        counters.append(dict(rank=rank, decision=decision, attempt=attempt, episode_count=local['episode_count']))
    require(state['budget']==publication['budget'], 'V2 checkpoint/publication budget differs')
    budget_check(state['budget'], summary['budget'])
    for group in saved['actor_optimizer']['param_groups']:
        close(group['lr'], identity['training_contract']['actor_lr'], 'V2 checkpoint fixed LR', 0.)
    return dict(path=str(path.relative_to(root)), iteration=state['iteration'], actor_updates=actor_updates,
                critic_updates=critic_updates, rank_execution_counters=counters, **storage)


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
    terminal = accepted['session_id']
    for name, session in sessions.items():
        recovered = name!=terminal and any(
            sessions[successor]['started_at'] is not None and session['started_at'] is not None
            and sessions[successor]['started_at']>session['started_at'] for successor in owners if successor!=name)
        audit.check(f'v2_session_completion:{name}', lambda s=session, r=recovered: _completion(s, recovered=r))
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
                    rows, rollout = audit_rollout(root, local, data_lookup, contract, seen)
                    require(len(rows)==contract['rollout_upper_steps_per_rank'], 'V2 local rollout count differs')
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
                targets = [audit_targets(root, local, rows, contract, normalization=(mean, std))
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
            budget_check(summary['budget'], previous_budget)
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
            budget_check(final_state['budget'], latest['budget'])
        return dict(accepted_iteration=accepted['iteration'], durable_iteration=latest['iteration'])
    audit.check('published_checkpoint_and_budget', durable_check)
    def persisted():
        budget = budget_check(read_json(root/'budget.json'))
        TrainingBudget(root/'budget.json', budget['limits'])
        for record in records.values():
            budget_check(budget, record['summary']['budget'])
        for session in sessions.values():
            if session['summary'] is not None and 'final_state' in session['summary']:
                budget_check(budget, session['summary']['final_state']['budget'])
        require(budget['used']['accepted_iterations']>=len(selected)
                and budget['used']['optimizer_attempts']>=actor_total
                and budget['used']['control_steps']>=controls, 'V2 budget omitted accepted execution/optimizer consumption')
        return dict(used=budget['used'], spent_superseded_budget_preserved=True)
    audit.check('persisted_budget_monotonic', persisted)
    result.update(version='genmo.closedloop.stage10.audit.v2',
        execution_counter_contract='rank_local_verified' if len(counts)==len(selected) else 'required_not_verified',
        recovery_history=dict(sessions=[dict(session_id=name, selected=name in owners,
            disposition='terminal' if name==terminal else 'recovered_history') for name in sessions],
            superseded_tails=events, superseded_seal_count=len(superseded)),
        checkpoint_storage=dict(complete_checkpoints=sum(row['storage']=='complete_checkpoint' for row in checkpoints.values()),
            intentionally_retired_checkpoints=sum(row['storage']=='retired_metadata_only' for row in checkpoints.values()),
            outer_iterations_without_checkpoint=len(selected)-len(checkpoints),
            all_historical_weight_bytes_revalidated=False,
            sparse_checkpoint_scope='Only saved publications contain restorable model/optimizer bytes'))
