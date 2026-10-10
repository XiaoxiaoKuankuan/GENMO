#!/usr/bin/env python3
"""第十步连续训练、恢复后再优化和独立评估产物的只读 CPU 审计。

审计逐轮 rollout/chunk/transition 清单、SHA、策略版本、实际执行步数、旧概率精度、
固定目标、网络隔离报告、联合 KL、完整 checkpoint 元数据与预算。固定 GAE 使用
fixed_targets.pt 中冻结的 old/next values 与逐条归档值先交叉核对，再独立重算。默认训练验收要求至少两轮接受更新，以及新 session
恢复完整状态后再次接受更新，不能把恢复后只采集当作续训成功。
成功 session 还必须有 session_end 指标成功写入后发布的 completion.json；完成标记
绑定 summary.json 的文件 SHA、成功状态和零退出码，避免把中途写出的摘要当作完成。
训练发布链从 latest 和实际恢复的 checkpoint 路径/SHA 回溯；失败尝试和未选发布
保留在报告中，不再按全目录同一轮编号合并。硬中断缺少 session 摘要时，可用已
落盘的 session_start 与不可变发布继续检查已接受轮次，但未恢复的终态仍不通过。
新 identity 的 base_seed/execution_contract 同时触发执行计数强校验：decision
对应真实归档最后一条加一，attempt 对应已耗 generation，恢复后的首 decision
必须延续所加载状态。旧 identity 明示 legacy_not_recorded，不冒称噪声计数完整。

评估分支核对显式 checkpoint 身份、完整 val/test 池计划、多 seed、网络不变和
逐 episode 文件，独立重算其聚合统计；评估不进入训练 Buffer。所有输入只读，
checkpoint 使用 CPU mmap，仅检查元信息与状态布局，不遍历大模型权重，不重新
跑 Actor、Critic、GMT、PhysX 或 GPU。清单 SHA 是当前文件内容核验；网络更新及
冻结语义来自运行时证据，不能据此宣称收敛、动作质量提高或真实硬件安全。
长期运行可将执行文件无损归档；审计逐轮在临时目录解包并核对每个原文件 SHA，
结束即清理。已按明确保留策略回收的旧 checkpoint 必须具备不可变 retirement
记录和发布时元数据，报告明确标记仅复核留存元数据，不能重新核验已删除权重。

--allow-incomplete 仅把缺失的后续证据列为 not_run；已存在的不一致、失败、预算
超限或旧数据跨策略版本仍然失败。只在显式 --output 指定路径创建一个新的报告，
绝不覆盖原始运行产物。PyTorch 文件必须来自可信运行，需 weights_only=False
读取原始 UpperTransition；评估不执行文件内定义的模型 forward。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
import shutil
import tarfile
import tempfile

import numpy as np
import torch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gem.closedloop.dppo.evaluation import aggregate_evaluation
from gem.closedloop.dppo.rollout_storage import load_rollout_record
from gem.closedloop.dppo.run_management import TrainingBudget, validate_budget_progress

VERSION = 'genmo.closedloop.stage10.audit.v1'
BUDGET_KEYS = {'accepted_iterations', 'optimizer_attempts', 'generations', 'control_steps', 'physics_steps'}
ROLLOUT_IDENTITY_KEYS = ('run_id', 'backend_session_id', 'episode_id', 'decision_id', 'policy_version')
_ARCHIVED_FILES = ContextVar('stage10_audit_archived_files', default={})


def _physical(path):
    """保留逻辑路径用于身份校验，只把文件读取映射到本轮临时解包位置。"""
    path = Path(path).resolve()
    return _ARCHIVED_FILES.get().get(path, path)


@contextmanager
def archived_execution(root, summary):
    """只读恢复一轮原始字节；拒绝重复成员、链接及越界路径，临时占用有界。"""
    directory = resolve(root, summary['targets_path']).parent
    manifest_path = directory/'archive_manifest.json'
    if not manifest_path.exists():
        yield None
        return
    manifest = read_json(manifest_path)
    require(manifest.get('schema') in ('genmo.closedloop.stage10.execution_archive.v1',
            'genmo.closedloop.stage10.execution_archive.v2'), 'Wrong execution archive schema')
    if manifest['schema'].endswith('.v2'):
        seal_path = directory/'seal_manifest.json'
        seal = read_json(seal_path)
        require(seal.get('schema')=='genmo.closedloop.stage10.iteration_seal.v2'
                and seal.get('journals_closed') is True
                and sha256(seal_path)==manifest.get('seal_sha256')
                and seal.get('members')==manifest.get('members')
                and seal.get('iteration')==manifest.get('iteration'), 'Execution archive differs from immutable seal')
    require(manifest.get('archive')=='execution_evidence.tar.gz', 'Unexpected execution archive filename')
    from gem.closedloop.dppo.archive_store import archive_path
    archive = archive_path(root, directory)
    require(archive.is_file() and not archive.is_symlink(), 'Execution archive must be a regular file')
    require(archive.stat().st_size==manifest['archive_size_bytes'] and sha256(archive)==manifest['archive_sha256'],
            'Execution archive size/SHA mismatch')
    records = {}
    for record in manifest['members']:
        name = record['path']
        relative = Path(name)
        require(isinstance(name, str) and name and not relative.is_absolute()
                and '..' not in relative.parts and str(relative)==name, 'Execution archive member path escapes iteration')
        require(name not in records, 'Duplicate execution archive manifest member')
        require(name not in ('summary.json', 'lr_progress.json', 'seal_manifest.json', 'superseded.json',
                'archive_manifest.json', 'execution_evidence.tar.gz'),
                'Execution archive cannot replace retained publication metadata')
        integer(record['size_bytes'], 'archive member size')
        records[name] = record
    require(records and sum(row['size_bytes'] for row in records.values())==manifest['original_size_bytes'],
            'Execution archive original byte count differs')
    with tempfile.TemporaryDirectory(prefix='stage10-audit-execution-') as temporary:
        mapping, seen = {}, set()
        with tarfile.open(archive, 'r:gz') as stream:
            for member in stream:
                relative = Path(member.name)
                require(not relative.is_absolute() and '..' not in relative.parts,
                        'Execution archive tar member escapes iteration')
                if member.isdir():
                    continue
                require(member.isfile() and member.name in records and member.name not in seen,
                        'Execution archive unexpected, duplicate or nonregular member')
                seen.add(member.name)
                record = records[member.name]
                require(member.size==record['size_bytes'], 'Execution archive member size differs')
                destination = Path(temporary)/relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with stream.extractfile(member) as source, destination.open('xb') as output:
                    shutil.copyfileobj(source, output, 1024*1024)
                require(sha256(destination)==record['sha256'], 'Execution archive member SHA mismatch')
                logical = (directory/relative).resolve()
                require(logical.is_relative_to(directory), 'Execution archive logical member escapes iteration')
                if logical.exists():
                    require(logical.is_file() and not logical.is_symlink()
                            and logical.stat().st_size==record['size_bytes'] and sha256(logical)==record['sha256'],
                            'Remaining original file differs from execution archive')
                mapping[logical] = destination
        require(seen==set(records), 'Execution archive is missing manifest members')
        token = _ARCHIVED_FILES.set(mapping)
        try:
            yield dict(manifest=str(manifest_path.relative_to(root)), archive_sha256=manifest['archive_sha256'],
                       original_bytes=manifest['original_size_bytes'], archive_bytes=manifest['archive_size_bytes'],
                       verified_original_members=len(records), lossless_original_bytes_verified=True)
        finally:
            _ARCHIVED_FILES.reset(token)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(_physical(path).read_text(encoding='utf-8'),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f'Nonfinite JSON {value}')))


def sha256(path):
    value = hashlib.sha256()
    with _physical(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def number(value, name):
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value), f'{name}: finite number required')
    return float(value)


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, f'{name}: integer >= {minimum} required')
    return value


def close(left, right, name, tolerance=1e-8):
    require(math.isclose(number(left, name), number(right, name), abs_tol=tolerance, rel_tol=1e-8), f'{name}: {left} != {right}')


def resolve(root, reference, base=None):
    require(isinstance(reference, str) and bool(reference), 'Missing run artifact reference')
    value = Path(reference)
    candidates = [value] if value.is_absolute() else [root/value, (base or root)/value]
    path = next((item.resolve() for item in candidates if _physical(item).exists()), candidates[0].resolve())
    require(path.is_relative_to(root), f'Artifact reference escapes run: {reference}')
    return path


def budget_check(budget, previous=None):
    require(budget.get('schema')=='genmo.closedloop.stage10.budget.v1', 'Wrong Stage10 budget schema')
    require(set(budget['used'])==BUDGET_KEYS and set(budget['limits'])==BUDGET_KEYS, 'Incomplete Stage10 budget counters')
    for key in BUDGET_KEYS:
        used = integer(budget['used'][key], key)
        limit = integer(budget['limits'][key], f'{key} limit', 1)
        require(used <= limit, f'{key} exceeds persisted budget')
        require(sum(integer(phase.get(key, 0), key) for phase in budget['phases'].values())==used, f'{key} phase sum differs')
    if previous is not None:
        validate_budget_progress(previous, budget)
    require(budget['limits']['physics_steps']==4*budget['limits']['control_steps'], 'Physics limit must contain four substeps/control')
    return budget


def audit_data(data):
    require(data.get('status')=='passed' and data.get('complete_manifest_scan') is True, 'Data scan is not complete/passed')
    identity = data['identity']
    manifests, counts = identity['manifest_sha256'], identity['sample_counts']
    expected_sources = {'AIST++', 'AIOZ-GDANCE', 'FineDance', 'Mine'}
    require(set(counts)=={'train', 'val', 'test'}, 'All train/val/test splits are required')
    require(len(manifests)==12 and all(set(counts[split])==expected_sources for split in counts), 'All twelve source/split manifests are required')
    records = data['records']
    require(len(records)==data['sample_count']==sum(sum(values.values()) for values in counts.values()), 'Full data audit record count differs')
    observed, lookup, owners = Counter(), {}, {key: {} for key in ('group_id', 'audio_sha256', 'source_motion_sha256')}
    for record in records:
        source, split, sample = record['dataset'], record['split'], str(record['sample_id'])
        key = (source, split, sample)
        require(key not in lookup, 'Duplicated audited source/split/sample')
        require(record['manifest_sha256']==manifests[f'{source}/{split}'], 'Data record manifest SHA mismatch')
        require(bool(record.get('motion_payload_sha256')) and bool(record.get('music_feature_sha256')), 'Motion/music payload identity missing')
        if data.get('require_audio'):
            require(record.get('audio_verified') is True and bool(record.get('audio_sha256')), 'Required audio was not verified')
        for field, mapping in owners.items():
            value = record.get(field)
            if value:
                require(mapping.get(value, split)==split, f'Data leakage across splits: {field}')
                mapping[value] = split
        observed[(source, split)] += 1
        lookup[key] = record
    for split, values in counts.items():
        for source, count in values.items():
            require(observed[(source, split)]==count, 'Per-source full dataset count differs')
    digest = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    require(digest==data['data_content_sha256'], 'Full data content record digest mismatch')
    return lookup


def audit_rollout(root, summary, data_lookup, contract, seen_paths, *, identity=None):
    path = resolve(root, summary['rollout_manifest'])
    manifest = read_json(path)
    before = summary['policy_version_before']
    require(manifest.get('schema') in ('genmo.closedloop.stage10.rollout.v1', 'genmo.closedloop.stage10.rollout.v2') and manifest.get('complete') is True, 'Rollout has no complete publication')
    require(manifest['policy_version']==before, 'Rollout policy version differs from iteration')
    rows, controls, physics, sources = [], 0, 0, Counter()
    block_cache = {}
    for chunk_reference in manifest['chunks']:
        chunk_path = resolve(root, chunk_reference['path'], path.parent)
        require(sha256(chunk_path)==chunk_reference['sha256'], 'Chunk manifest SHA mismatch')
        chunk = read_json(chunk_path)
        require(chunk.get('schema') in ('genmo.closedloop.stage10.rollout_chunk.v1', 'genmo.closedloop.stage10.rollout_chunk.v2') and chunk['policy_version']==before, 'Chunk schema or policy version differs')
        require(len(chunk['records'])==chunk['record_count']==chunk_reference['record_count'], 'Chunk record counts differ')
        for record in chunk['records']:
            record_path = resolve(root, record['path'], chunk_path.parent)
            record_identity = (record_path, record.get('index'))
            require(record_identity not in seen_paths, 'A rollout transition file/index was reused across iterations')
            seen_paths.add(record_identity)
            item = load_rollout_record(record_path, record, rank_directory=path.parent.parent,
                                       physical=_physical, cache=block_cache)
            # Writer 的清单只发布稳定五键；原始转移另有 env/request/plan/parent 诊断身份。
            require(set(record['identity'])==set(ROLLOUT_IDENTITY_KEYS), 'Rollout manifest identity must contain exactly the five published keys')
            require({key: item.identity[key] for key in ROLLOUT_IDENTITY_KEYS}==record['identity']
                    and item.identity['policy_version']==before, 'Stored transition policy identity differs')
            require(item.transition_valid is True, 'Invalid transition entered accepted rollout')
            count = integer(item.executed_control_steps, 'executed control steps')
            require(count==record['executed_control_steps'] and item.executed_physics_steps==record['executed_physics_steps']==4*count, 'Actual control/physics counts differ')
            require(item.control_tick_end-item.control_tick_begin==12*count, 'Actual time differs from control count')
            require(item.rewards.shape==(count,) and bool(torch.isfinite(item.rewards).all()), 'Actual rewards incomplete or nonfinite')
            steps = int(contract['denoising_steps'])
            require(item.old_log_prob.dtype==torch.float64 and item.old_log_prob.shape==(steps,) and bool(torch.isfinite(item.old_log_prob).all()), 'Old denoising probabilities are invalid')
            require(item.chain.dtype==torch.float32 and item.chain.shape==(steps+1, 120, 30), 'Stored denoising chain dtype/shape differs')
            expected_mask = (item.context['future_valid'][..., None] & ~item.context['known_qpos30_mask'])[0]
            require(torch.equal(item.free_mask, expected_mask), 'Committed prefix/padding differs from free action mask')
            kernel = item.metadata['sampler_trace']['kernel_config']
            for key, config_key in (('steps', 'denoising_steps'), ('eta', 'eta'), ('std_floor', 'std_floor'), ('guidance_scale', 'guidance_scale')):
                require(kernel[key]==contract[config_key], f'Stored denoising kernel differs: {key}')
            require(kernel['log_prob_reduction']=='joint_sum_fp64' and kernel['cfg_policy']=='music_only_shared_history_and_prefix', 'Wrong probability reduction or CFG policy')
            task = item.metadata['training_task']
            if item.metadata.get('timing_contract')=='modeled_deployment.v3':
                require(identity is not None, 'Modeled clock audit requires full execution identity')
                from tools.eval.audit_stage10_vector_helpers import audit_modeled_clock
                audit_modeled_clock(item.metadata,task,item.control_tick_begin,identity)
            require(task['split']=='train', 'val/test sample leaked into training Buffer')
            audited = data_lookup[(task['dataset'], 'train', str(task['sample_id']))]
            require(task['manifest_sha256']==audited['manifest_sha256'], 'Training task does not match complete train manifest')
            integer(task['music_start_frame'], 'training music start')
            require(task['music_start_frame']<audited['num_frames'], 'Training music start outside real song')
            from tools.eval.audit_stage10_vector_helpers import audit_boundary_wait_metadata
            tail = audit_boundary_wait_metadata(item, identity or {})
            rows.append(dict(identity=dict(item.identity), rewards=item.rewards.double().tolist(),
                count=count, begin=item.control_tick_begin, end=item.control_tick_end,
                terminated=bool(item.terminated), truncated=bool(item.truncated),
                has_next=item.next_context is not None, event_reward=float(item.metadata.get('event_reward', 0.)),
                old_value=float(item.old_value), next_value=float(item.next_value),
                free_coordinate_count=int(item.free_mask.sum()), audited_fragment_tail=tail))
            controls += count
            physics += item.executed_physics_steps
            sources[task['dataset']] += 1
            del item
    require(len(rows)==manifest['transition_count']==summary['collected_upper_transitions'], 'Published rollout transition count differs')
    require(controls==manifest['executed_control_steps']==summary['collection']['control_steps'] and physics==manifest['executed_physics_steps'], 'Rollout totals differ')
    require(summary['collection'].get('full_train_pool') is True, 'Training collection used a preselected subset')
    return rows, dict(transition_count=len(rows), control_steps=controls, physics_steps=physics, source_transition_counts=dict(sources))


def audit_targets(root, summary, rows, contract, *, normalization=None):
    target = torch.load(_physical(resolve(root, summary['targets_path'])), map_location='cpu', weights_only=False, mmap=True)
    size = len(rows)
    def vector(name):
        result = torch.as_tensor(target[name]).double().cpu().numpy()
        require(result.shape==(size,) and np.isfinite(result).all(), f'Invalid fixed target vector: {name}')
        return result
    old, nxt = vector('old_values'), vector('next_values')
    require(np.allclose(old, [row['old_value'] for row in rows], atol=1e-7, rtol=0)
            and np.allclose(nxt, [row['next_value'] for row in rows], atol=1e-7, rtol=0),
            'Fixed old/next values differ from immutable rollout pre-update values')
    gamma, lam = float(contract['gamma_upper'])**(1/25), float(contract['lambda_upper'])**(1/25)
    close(target['gamma_low'], gamma, 'gamma_low')
    close(target['lambda_low'], lam, 'lambda_low')
    require(torch.as_tensor(target['valid']).bool().shape==(size,) and bool(torch.as_tensor(target['valid']).all()), 'Accepted rollout contains invalid fixed targets')
    discounted = np.asarray([sum((gamma**i)*reward for i, reward in enumerate(row['rewards']))+
                             (gamma**max(row['count']-1, 0))*row['event_reward'] for row in rows], dtype=np.float64)
    advantages = np.zeros(size)
    for index in range(size-1, -1, -1):
        row = rows[index]
        advantages[index] = discounted[index]-old[index]
        if not row['terminated'] and row['has_next']:
            advantages[index] += gamma**row['count']*nxt[index]
        if index+1<size:
            following = rows[index+1]
            continuous = (not row['terminated'] and not row['truncated'] and row['end']==following['begin']
                and all(row['identity'][key]==following['identity'][key] for key in ('backend_session_id', 'episode_id', 'policy_version')))
            if continuous:
                advantages[index] += (gamma*lam)**row['count']*advantages[index+1]
    mean, std = (advantages.mean(), float(advantages.std())) if normalization is None else normalization
    normalized = (advantages-mean)/max(float(std), 1e-8)
    if normalization is not None:
        require(target.get('advantage_normalization_scope')=='global_valid_upper_transitions',
                'V2 advantages must use global upper-transition normalization')
        close(target['advantage_global_mean'], mean, 'Global advantage mean')
        close(target['advantage_global_std'], std, 'Global advantage std')
    differences = {}
    for name, expected in (('discounted_rewards', discounted), ('advantages_raw', advantages), ('returns', advantages+old), ('advantages', normalized)):
        difference = float(np.max(np.abs(vector(name)-expected)))
        require(difference<=1e-7, f'Independent fixed-target/GAE mismatch: {name} {difference}')
        differences[name] = difference
    return dict(max_abs_differences=differences, value_source='fixed_targets.pt cross-checked against immutable pre-update rollout values')


def audit_update(summary, contract):
    probability = summary['probability_check']
    require(probability.get('passed') is True, 'Old-policy probability check failed')
    for key, threshold in (('max_abs_log_probability_difference', 1e-4), ('max_abs_ratio_minus_one', 1e-3), ('max_abs_independent_gaussian_difference', 1e-8)):
        require(0<=number(probability[key], key)<=threshold, f'Pre-update probability check exceeded tolerance: {key}')
    actor, critic = summary['actor'], summary['critic']
    require(actor.get('parameters_changed') is True and actor.get('critic_unchanged') is True and actor.get('optimizer_steps')==1, 'Actor update or Critic isolation evidence failed')
    require(critic.get('parameters_changed') is True and critic.get('actor_unchanged') is True, 'Critic update or Actor isolation evidence failed')
    require(number(actor['ppo_only_gradient_norm'], 'PPO-only gradient')>0, 'DPPO itself has no positive gradient')
    limit = number(summary.get('kl_limit', contract['kl_stop_joint']), 'joint KL limit')
    require(0<=number(summary['kl']['mean_joint_kl'], 'accepted joint KL')<=limit, 'Accepted update exceeds joint KL limit')
    calibration = actor.get('lr_calibration')
    require(isinstance(calibration, dict) and calibration.get('accepted_updates')==1
            and calibration.get('base_state_restored_per_candidate') is True and calibration.get('gradients_reused') is True,
            'Accepted Actor step has no valid bounded LR calibration evidence')
    candidates = calibration['candidates']
    require(1<=len(candidates)<=3 and calibration['attempt_count']==len(candidates), 'Optimizer candidate attempt counts differ')
    valid = []
    rates = []
    for candidate in candidates:
        rate = number(candidate['lr'], 'candidate lr')
        require(0<rate<=1e-6, 'Unbounded Actor LR candidate')
        rates.append(rate)
        eligible = number(candidate['kl']['mean_joint_kl'], 'candidate KL')<=limit and candidate['parameter_change']['changed_count']>0
        require(candidate['accepted']==eligible, 'Candidate acceptance does not match KL and real parameter change')
        if eligible:
            valid.append(candidate)
    require(all(left<right for left, right in zip(rates, rates[1:])) and valid, 'Candidate LR order or acceptance invalid')
    chosen = max(valid, key=lambda item: item['lr'])
    close(calibration['selected_lr'], chosen['lr'], 'maximum eligible learning rate', 0.)
    close(summary['kl']['mean_joint_kl'], chosen['kl']['mean_joint_kl'], 'selected checkpoint KL')
    gmt = summary['gmt_frozen']
    require(gmt.get('policy_unchanged') is True and gmt.get('runtime_parameters_unchanged') is True, 'GMT changed during accepted update')
    watermarks = gmt.get('execution_journal', {})
    require(watermarks.get('executed_seq')==watermarks.get('acked_seq'), 'Accepted update has outstanding physics mutation')
    require(summary['source_unchanged'].get('unchanged') is True, 'Sources changed during accepted update')
    return dict(optimizer_attempts=len(candidates), selected_lr=chosen['lr'], mean_joint_kl=summary['kl']['mean_joint_kl'])


def _counter_contract(identity):
    present = {key for key in ('base_seed', 'execution_contract') if key in identity}
    if not present:
        return 'legacy_not_recorded'
    require(len(present)==2, 'Execution counter identity requires both base_seed and execution_contract')
    require(integer(identity['base_seed'], 'base_seed')<2**32, 'base_seed must fit uint32')
    require(isinstance(identity['execution_contract'], dict) and bool(identity['execution_contract']), 'Missing execution contract')
    return 'required'


def _execution_counters(state, identity, rows=None):
    if _counter_contract(identity)=='legacy_not_recorded':
        return dict(contract='legacy_not_recorded')
    result = {key:integer(state.get(key), f'checkpoint {key}') for key in ('decision', 'attempt', 'episode_count')}
    result['latency_budget_s'] = number(state.get('latency_budget_s'), 'checkpoint latency_budget_s')
    require(result['latency_budget_s']>0, 'Checkpoint latency budget must be positive')
    require(result['decision']<=result['attempt'], 'Checkpoint decision exceeds generation attempts')
    require(result['attempt']==state['budget']['used']['generations'], 'Checkpoint attempt differs from spent generation budget')
    if rows is not None:
        decisions = [integer(row['identity']['decision_id'], 'archived decision_id') for row in rows]
        require(decisions, 'Execution counter verification requires archived rollout rows')
        require(all(right==left+1 for left, right in zip(decisions, decisions[1:])), 'Archived decision IDs are not contiguous')
        require(result['decision']==decisions[-1]+1, 'Checkpoint decision differs from last archived decision plus one')
        result.update(first_archived_decision_id=decisions[0], last_archived_decision_id=decisions[-1])
    result['contract'] = 'verified'
    return result


def _retired_checkpoint(root, path, publication):
    """核验旧权重的有意回收证据；只返回删除前留存的元数据，不伪造权重复核。"""
    retirement_path = root/'checkpoints'/'retired'/f'{path.stem}.json'
    require(retirement_path.is_file() and not retirement_path.is_symlink(),
            'Missing checkpoint has no immutable retirement record')
    retirement = read_json(retirement_path)
    require(retirement.get('schema')=='genmo.closedloop.stage10.checkpoint_retirement.v1'
            and retirement.get('purpose')=='bounded_checkpoint_retention', 'Wrong checkpoint retirement schema/purpose')
    require(path.parent==root/'checkpoints' and path.name!='initial.pt', 'Retirement may only reference old run checkpoints')
    for key in ('iteration', 'path', 'sha256', 'size_bytes'):
        require(retirement.get(key)==publication[key], 'Checkpoint retirement differs from original publication')
    original_publication = resolve(root, retirement['publication'])
    require(original_publication.parent==root/'checkpoints'/'publications'
            and read_json(original_publication)==publication, 'Retirement names another immutable publication')
    policy = retirement['policy']
    keep_last = integer(policy['keep_last'], 'checkpoint keep_last', 2)
    keep_every = integer(policy['keep_every'], 'checkpoint keep_every', 1)
    require(publication['iteration'] % keep_every != 0, 'Retirement cannot remove a milestone checkpoint')
    replacement = retirement['replacement']
    replacement_publication_path = resolve(root, replacement['publication'])
    require(replacement_publication_path.parent==root/'checkpoints'/'publications',
            'Retirement replacement must name a run publication')
    replacement_publication = read_json(replacement_publication_path)
    require(replacement_publication.get('schema')=='genmo.closedloop.stage10.checkpoint_publication.v1',
            'Retirement replacement has invalid publication schema')
    for key in ('iteration', 'path', 'sha256'):
        require(replacement.get(key)==replacement_publication[key], 'Retirement replacement publication differs')
    saved = retirement.get('audit_state')
    if isinstance(saved, dict) and saved.get('version')=='genmo.closedloop.stage10.full_state.v2':
        successors = {read_json(path)['iteration'] for path in (root/'checkpoints/publications').glob('*.json')
                      if publication['iteration']<read_json(path)['iteration']<=replacement['iteration']}
        require(len(successors)>=keep_last, 'Retirement removed one of the latest protected checkpoints')
    else:
        require(replacement['iteration']>=publication['iteration']+keep_last,
                'Retirement removed one of the latest protected checkpoints')
    latest = read_json(root/'latest.json')
    require(latest['iteration']>=replacement['iteration'] and latest['path']!=publication['path'],
            'Retirement preceded replacement publication or removed latest')
    require(isinstance(saved, dict), 'Retirement lacks original checkpoint audit metadata')
    return saved, dict(storage='retired_metadata_only', original_bytes_revalidated=False,
                       retirement=str(retirement_path.relative_to(root)), sha256=publication['sha256'])


def audit_checkpoint(root, summary, identity, update, publication=None, rows=None):
    path = resolve(root, summary['checkpoint'])
    if publication is None:
        publications = []
        for publication_path in (root/'checkpoints/publications').glob('*.json'):
            candidate = read_json(publication_path)
            if candidate.get('iteration')==summary['iteration'] and resolve(root, candidate['path'])==path:
                publications.append(candidate)
        require(len(publications)==1, 'Accepted checkpoint requires exactly one matching immutable publication')
        publication = publications[0]
    require(publication.get('schema')=='genmo.closedloop.stage10.checkpoint_publication.v1'
            and resolve(root, publication['path'])==path, 'Checkpoint publication schema/path differs')
    require(publication['budget']==summary['budget'], 'Checkpoint publication budget differs')
    published_summary = read_json(resolve(root, publication['metadata']['iteration_summary']))
    require(published_summary==summary, 'Checkpoint publication binds another iteration summary')
    if path.is_file():
        require(path.stat().st_size==publication['size_bytes'] and sha256(path)==publication['sha256'],
                'Accepted checkpoint bytes differ from immutable publication')
        saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        required = {'actor', 'critic', 'actor_optimizer', 'critic_optimizer', 'state', 'rng', 'samplers', 'identity', 'config', 'optimizer_layout'}
        storage = dict(storage='complete_checkpoint', original_bytes_revalidated=True, sha256=publication['sha256'])
    else:
        saved, storage = _retired_checkpoint(root, path, publication)
        required = {'actor_optimizer', 'state', 'samplers', 'identity', 'restore_environment'}
    require(saved.get('version')=='genmo.closedloop.stage9.full_state.v1' and required.issubset(saved), 'Checkpoint lacks full training state')
    require(saved['identity']==identity and saved.get('restore_environment')=='new_worker_session_and_reset', 'Checkpoint identity/environment restore semantics differ')
    state = saved['state']
    require(publication['session_id']==state.get('session_id'), 'Checkpoint publication belongs to another session')
    require(state['iteration']==summary['iteration'] and state['policy_version']==summary['policy_version_after'], 'Checkpoint iteration/policy watermark differs')
    require(state['actor_updates']==state['iteration'] and state['buffer_size']==0 and state['pending_plan'] is False, 'Checkpoint is not a complete accepted-update boundary')
    require(saved['samplers']['music']['split']=='train' and saved['samplers']['music']['catalog_identity']==identity['dataset'], 'Checkpoint sampler does not bind the complete train catalog')
    for group in saved['actor_optimizer']['param_groups']:
        close(group['lr'], update['selected_lr'], 'Checkpoint optimizer selected LR', 0.)
    close(state['selected_actor_lr'], update['selected_lr'], 'Checkpoint selected LR', 0.)
    budget_check(state['budget'])
    require(state['budget']==summary['budget'], 'Checkpoint and iteration budget snapshots differ')
    result = dict(path=str(path.relative_to(root)), iteration=state['iteration'], policy_version=state['policy_version'],
                  actor_updates=state['actor_updates'], critic_updates=state['critic_updates'], session_id=state.get('session_id'),
                  optimizer_attempts=state['budget']['used']['optimizer_attempts'], **storage)
    require(rows is not None or _counter_contract(identity)=='legacy_not_recorded', 'New checkpoint counters require real archived rollout rows')
    result['execution_counters'] = _execution_counters(state, identity, rows)
    del saved
    return result


def audit_evaluation(root, path, data_lookup, dataset_identity, *, loaded_checkpoint=None):
    report = read_json(path)
    require(report.get('status')=='passed' and report.get('updates_performed') is False and report.get('training_buffer_used') is False, 'Evaluation optimized weights or did not pass')
    require(report.get('actor_gradients_unchanged') is True and report.get('rng_restored') is True and all(report['networks_unchanged'].values()), 'Evaluation changed network/gradient/RNG state')
    require(report['network_fingerprints_before']==report['network_fingerprints_after'], 'Evaluation network fingerprints differ')
    require(bool(report['actor_identity'].get('checkpoint')) and bool(report['actor_identity'].get('sha256')), 'Evaluation does not name the actually loaded checkpoint')
    if loaded_checkpoint is not None:
        require(str(Path(report['actor_identity']['checkpoint']).resolve())==loaded_checkpoint['path']
                and report['actor_identity']['sha256']==loaded_checkpoint['sha256'],
                'V2 evaluation does not use the declared weights-only checkpoint')
    plan = read_json(resolve(root, report['selection_path'], path.parent))
    require(plan['catalog_identity']==dataset_identity and plan['split'] in ('val', 'test'), 'Evaluation pool identity differs')
    unsigned = {key: value for key, value in plan.items() if key!='plan_sha256'}
    expected = hashlib.sha256(json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    require(expected==plan['plan_sha256']==report['plan_sha256'], 'Evaluation plan digest differs')
    require(len(plan['seeds'])>=2, 'Stage10 evaluation acceptance requires multiple explicit seeds')
    expected_pool = sum(dataset_identity['sample_counts'][plan['split']].values())
    require(plan['complete_pool_count']==expected_pool, 'Evaluation selection was not based on the complete held-out pool')
    if plan['requested_eval_count']=='all':
        require(plan['selected_sample_count']==expected_pool, 'all evaluation omitted held-out paired samples')
    require(len(plan['tasks'])==plan['selected_sample_count']*len(plan['seeds'])==report['requested_task_count'], 'Evaluation Cartesian task count differs')
    episodes = []
    require(len(report['episode_manifests'])==len(plan['tasks'])==report['completed_episode_count'], 'Evaluation stopped before its planned episodes')
    for reference, task in zip(report['episode_manifests'], plan['tasks']):
        require(task['sample']['row']['split']==plan['split'], 'Evaluation includes a different split')
        audited = data_lookup[(task['dataset'], plan['split'], str(task['sample_id']))]
        require(task['sample']['manifest_sha256']==audited['manifest_sha256'], 'Evaluation paired sample differs from full manifest')
        episode_path = resolve(root, reference['path'], path.parent)
        require(sha256(episode_path)==reference['sha256'], 'Evaluation episode SHA differs')
        episode = read_json(episode_path)
        require(episode.get('transition_valid') is True and episode['task_id']==task['task_id']==reference['task_id'], 'Evaluation episode invalid or out of order')
        require(episode['seed']==task['seed'] and episode['sample_id']==task['sample_id'] and episode['dataset']==task['dataset'], 'Evaluation episode task identity differs')
        close(episode['reward_sum'], sum(item['reward_sum'] for item in episode['decisions']), 'Evaluation episode reward')
        episodes.append(episode)
    require(aggregate_evaluation(episodes)==report['aggregate'], 'Evaluation aggregate cannot be independently reproduced')
    return dict(episodes=len(episodes), seeds=plan['seeds'], distinct_paired_samples=plan['selected_sample_count'],
                unique_audio_count=plan['selected_unique_audio_count'], split=plan['split'], complete_pool=expected_pool)


def audit_evaluation_restore(session, identity):
    """v2 独立评估只核权重来源；不要求或解析多 rank 完整恢复的执行/RNG状态。"""
    resume = session.get('resume') or {}
    require(resume.get('restore_mode')=='weights_only' and resume.get('restored_full_state') is False
            and resume.get('training_resume') is False, 'V2 evaluation falsely claims a full training resume')
    path = Path(resume['checkpoint']).resolve()
    require(path.is_file() and sha256(path)==resume['sha256'], 'V2 evaluation source checkpoint SHA differs')
    saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    require(saved.get('version')=='genmo.closedloop.stage10.full_state.v2'
            and saved.get('identity')==identity and {'actor', 'critic', 'state'}.issubset(saved),
            'V2 evaluation checkpoint model identity differs')
    require(saved['state']['iteration']==session['initial_iteration'], 'V2 evaluation loaded another model iteration')
    return dict(path=str(path), sha256=resume['sha256'], restore_mode='weights_only',
                optimizer_restored=False, rng_restored=False,
                rank_execution_state_scope='Not interpreted as evaluation full-state resume')


class Audit:
    def __init__(self, allow_incomplete):
        self.allow_incomplete, self.checks = allow_incomplete, []
    def check(self, name, function):
        try:
            result = function()
            self.checks.append({'name': name, 'status': 'passed', 'details': result})
            return result
        except FileNotFoundError as error:
            self.checks.append({'name': name, 'status': 'not_run' if self.allow_incomplete else 'failed', 'error': str(error)})
        except Exception as error:
            self.checks.append({'name': name, 'status': 'failed', 'error': f'{type(error).__name__}: {error}'})
        return None


def _session_evidence(root, directory):
    """读取真实启动日志；只有末尾未写完的 JSON 行可作为中断证据保留。"""
    session_id = directory.name
    summary_path = directory/'summary.json'
    summary = read_json(summary_path) if summary_path.exists() else None
    if summary is not None:
        require(summary.get('session_id')==session_id, 'Session directory identity differs')
    metrics_path = root/'metrics'/f'{session_id}.jsonl'
    records, trailing_partial = [], False
    if metrics_path.exists():
        lines = metrics_path.read_bytes().splitlines(keepends=True)
        for index, line in enumerate(lines):
            try:
                record = json.loads(line)
            except (ValueError, UnicodeError):
                require(index==len(lines)-1 and not line.endswith(b'\n'), 'Malformed complete metrics record')
                trailing_partial = True
                continue
            require(record.get('session_id')==session_id, 'Metrics session identity differs')
            timestamp = datetime.fromisoformat(record['time_utc'])
            require(timestamp.tzinfo is not None, 'Metrics timestamp must include its timezone')
            records.append((record, timestamp))
    starts = [record for record, _ in records if record.get('event')=='session_start']
    require(len(starts)<=1, 'Duplicated session_start evidence')
    start = starts[0] if starts else None
    if start is not None and summary is not None:
        require(start['initial_iteration']==summary.get('initial_iteration') and start.get('resume')==summary.get('resume'),
                'Session summary differs from durable start/resume evidence')
    context = summary
    if context is None and start is not None:
        context = dict(session_id=session_id, mode=start['mode'], initial_iteration=start['initial_iteration'],
                       resume=start.get('resume'), data_audit=str((directory/'data_audit.json').relative_to(root)))
    return dict(session_id=session_id, directory=directory, summary_path=summary_path, summary=summary,
                context=context, start=start, records=records, trailing_partial=trailing_partial,
                first_time=records[0][1] if records else None, last_time=records[-1][1] if records else None)


def _completion(evidence, *, recovered=False):
    """历史中断允许没有结束标记；已经存在的标记始终校验，不能掩盖改写。"""
    summary = evidence['summary']
    marker = evidence['directory']/'completion.json'
    if summary is None:
        require(not marker.exists(), 'Completion exists without its bound session summary')
        if not recovered:
            raise FileNotFoundError(evidence['summary_path'])
        return {'completion': 'missing_after_recovered_interruption'}
    completion = read_json(marker) if marker.exists() else None
    if completion is not None:
        require(completion.get('schema')=='genmo.closedloop.stage10.session_completion.v1', 'Wrong session completion schema')
        require(completion.get('summary_sha256')==sha256(evidence['summary_path']), 'Session summary differs from completion marker')
        require(completion.get('status')==summary.get('status') and completion.get('exit_code')==summary.get('exit_code'),
                'Session completion status/exit code differs from summary')
    elif not recovered:
        raise FileNotFoundError(marker)
    if not recovered:
        require(summary.get('status')=='passed' and summary.get('exit_code')==0, 'Session did not exit successfully')
        if (summary.get('resume') or {}).get('training_resume') and summary.get('final_iteration')==summary.get('initial_iteration'):
            require(summary.get('stop_reason')=='budget_exhausted' or summary.get('stopped_on_signal') is True,
                    'Resume without an accepted update requires an explicit normal budget/signal stop')
        require(summary['source_unchanged'].get('unchanged') is True and summary.get('original_assets_unchanged') is True, 'Session source/assets changed')
        gmt = summary['worker_shutdown']['gmt']
        require(gmt.get('policy_unchanged') is True and gmt.get('runtime_parameters_unchanged') is True
                and gmt.get('process_exit_code')==0 and not gmt.get('forced_shutdown') and not gmt.get('close_error'), 'Worker did not close with frozen-state proof')
    else:
        # 恢复可处理进程死亡，不可把已知源文件或资产改变解释成普通中断。
        if 'source_unchanged' in summary:
            require(summary['source_unchanged'].get('unchanged') is True, 'Recovered session reports changed sources')
        if 'original_assets_unchanged' in summary:
            require(summary['original_assets_unchanged'] is True, 'Recovered session reports changed original assets')
    if 'budget' in summary:
        budget_check(summary['budget'])
    return dict(session_id=evidence['session_id'], completion='verified' if completion else 'missing_after_recovered_interruption',
                recovered=recovered, status=summary.get('status'), initial_iteration=summary.get('initial_iteration'),
                final_iteration=summary.get('final_iteration'))


def _select_publication_chain(root, sessions):
    """不依赖目录排序/mtime：只沿 latest、同 session 连续轮次和实际 resume 选链。"""
    latest = read_json(root/'latest.json')
    require(latest.get('schema')=='genmo.closedloop.stage10.checkpoint_publication.v1', 'Wrong latest publication schema')
    latest_path = resolve(root, latest['publication'])
    require(read_json(latest_path)=={key: value for key, value in latest.items() if key!='publication'}, 'Latest and immutable publication disagree')
    publications, inventory = {}, []
    for path in sorted((root/'checkpoints/publications').glob('*.json')):
        relative = str(path.relative_to(root))
        try:
            value = read_json(path)
            require(value.get('schema')=='genmo.closedloop.stage10.checkpoint_publication.v1', 'Wrong publication schema')
            integer(value['iteration'], 'published iteration', 1)
            require(isinstance(value['session_id'], str), 'Missing publication session')
            resolve(root, value['path'])
            publications[path.resolve()] = value
            inventory.append(dict(path=relative, iteration=value['iteration'], session_id=value['session_id'],
                                  checkpoint=value['path'], sha256=value['sha256'],
                                  iteration_summary=value.get('metadata', {}).get('iteration_summary')))
        except Exception as error:
            inventory.append(dict(path=relative, error=f'{type(error).__name__}: {error}'))
    require(latest_path in publications, 'Latest publication is not an immutable publication in this run')
    selected, contexts, visited = [], {}, set()
    current = latest_path
    while current is not None:
        require(current not in visited, 'Cyclic checkpoint recovery chain')
        visited.add(current)
        publication = publications[current]
        owner = publication['session_id']
        require(owner in sessions and sessions[owner]['context'] is not None, 'Published checkpoint lacks session summary or durable session_start')
        context = sessions[owner]['context']
        require(context.get('mode')=='train', 'Training publication belongs to a different mode')
        initial = integer(context['initial_iteration'], 'session initial iteration')
        index = publication['iteration']
        require(index>initial, 'Published update did not advance its restored state')
        summary_path = resolve(root, publication['metadata']['iteration_summary'])
        require(summary_path.is_relative_to(root/'sessions'/owner), 'Published iteration summary belongs to another session')
        summary = read_json(summary_path)
        require(summary['iteration']==index, 'Published iteration summary number differs')
        if context.get('status')=='passed':
            require(str(summary_path.relative_to(root)) in context.get('iterations', []), 'Successful session omitted its published iteration')
        selected.append((summary, context, publication, current))
        contexts[owner] = context
        if index>initial+1:
            matches = [path for path, value in publications.items() if value['session_id']==owner and value['iteration']==index-1]
            require(len(matches)==1, 'Same-session accepted sequence is missing or ambiguous')
            current = matches[0]
            continue
        resume = context.get('resume')
        if initial==0:
            if resume is not None:
                require(resume.get('training_resume') is True and resume.get('restored_full_state') is True
                        and resume.get('old_buffer_discarded') is True and resume.get('initial_iteration')==0, 'Invalid initial-state resume')
                path = resolve(root, resume['checkpoint'])
                require(sha256(path)==resume['sha256'], 'Initial resume checkpoint SHA differs')
                saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
                require(saved.get('version')=='genmo.closedloop.stage9.full_state.v1'
                        and saved['identity']==read_json(root/'run.json')['identity']
                        and saved['state']['iteration']==saved['state']['policy_version']==0
                        and saved['state'].get('buffer_size')==0 and saved['state'].get('pending_plan') is False,
                        'Initial resume did not load a complete iteration-zero checkpoint')
            current = None
        else:
            require(resume and resume.get('training_resume') is True and resume.get('restored_full_state') is True
                    and resume.get('old_buffer_discarded') is True and resume.get('initial_iteration')==initial,
                    'Recovery link lacks full-state resume evidence')
            path = resolve(root, resume['checkpoint'])
            matches = [key for key, value in publications.items() if value['iteration']==initial
                       and resolve(root, value['path'])==path and value['sha256']==resume['sha256']]
            require(len(matches)==1, 'Resume does not identify exactly one previous publication')
            require(publications[matches[0]]['session_id']!=owner, 'Resume did not create a new run session')
            current = matches[0]
    selected.reverse()
    return dict(latest=latest, selected=selected, contexts=contexts,
                unselected_publications=[value for value in inventory if resolve(root, value['path']) not in visited])


def _audit_training(root, run, audit, result, minimum_iterations, require_resume):
    identity = run['identity']
    result['execution_counter_contract'] = _counter_contract(identity)
    sessions = {}
    directories = set((root/'sessions').glob('*'))
    directories.update(root/'sessions'/path.stem for path in (root/'metrics').glob('*.jsonl'))
    for directory in sorted(directories):
        evidence = audit.check(f'session_json:{directory.name}', lambda directory=directory: _session_evidence(root, directory))
        if evidence is not None:
            sessions[directory.name] = evidence
            audit.checks[-1]['details'] = dict(session_id=directory.name,
                summary_present=evidence['summary'] is not None, durable_start_present=evidence['start'] is not None,
                metrics_records=len(evidence['records']), trailing_partial_metrics=evidence['trailing_partial'])
    chain = audit.check('canonical_publication_chain', lambda: _select_publication_chain(root, sessions))
    if chain is None:
        return
    # Path/对象仅供内部校验；输出只保留可复核的相对引用。
    audit.checks[-1]['details'] = dict(publications=[str(path.relative_to(root)) for _, _, _, path in chain['selected']],
                                     unselected_publications=chain['unselected_publications'])
    selected = chain['selected']
    canonical_ids = set(chain['contexts'])
    latest = chain['latest']
    # 已发布后立即在下一 session 因预算/信号正常停止，也能提供完整终态；不能要求其再更新。
    terminal_ids = {latest['session_id']}
    for name, evidence in sessions.items():
        context = evidence['context'] or {}
        resume = context.get('resume') or {}
        if (context.get('status')=='passed' and context.get('final_iteration')==latest['iteration']
                and context.get('initial_iteration')==latest['iteration'] and resume.get('training_resume') is True
                and (context.get('stop_reason')=='budget_exhausted' or context.get('stopped_on_signal') is True)
                and resume.get('restored_full_state') is True and resume.get('old_buffer_discarded') is True
                and resume.get('sha256')==latest['sha256']
                and resolve(root, resume.get('checkpoint', 'missing'))==resolve(root, latest['path'])):
            terminal_ids.add(name)
    history = []
    for name, evidence in sessions.items():
        summary = evidence['summary'] or {}
        completed = summary.get('status')=='passed' and summary.get('exit_code')==0 and (evidence['directory']/'completion.json').exists()
        successors = []
        if not completed:
            for other in canonical_ids | terminal_ids:
                if other==name:
                    continue
                candidate = sessions[other]
                context = candidate['context'] or {}
                resume = context.get('resume') or {}
                if not resume.get('training_resume'):
                    continue
                direct = any(publication['session_id']==name and publication['sha256']==resume.get('sha256')
                             and resolve(root, publication['path'])==resolve(root, resume['checkpoint'])
                             for _, _, publication, _ in selected)
                ordered = (evidence['last_time'] is not None and candidate['first_time'] is not None
                           and candidate['first_time']>evidence['last_time'])
                if direct or ordered:
                    successors.append(other)
        recovered = bool(successors)
        audit.check(f'session:{name}', lambda evidence=evidence, recovered=recovered: _completion(evidence, recovered=recovered))
        history.append(dict(session_id=name, status=summary.get('status', 'incomplete'), selected=name in canonical_ids,
                            recovered_by=sorted(successors), disposition='recovered_history' if recovered else 'completed' if completed else 'unrecovered',
                            trailing_partial_metrics=evidence['trailing_partial'],
                            error=summary.get('error'),
                            summary=str(evidence['summary_path'].relative_to(root)) if evidence['summary'] is not None else None))
    result['recovery_history'] = dict(sessions=history, unselected_publications=chain['unselected_publications'])
    data_lookup = None
    for name in sorted(canonical_ids | terminal_ids):
        context = sessions[name]['context']
        def check_data(context=context):
            data = read_json(resolve(root, context['data_audit']))
            lookup = audit_data(data)
            require(data['identity']==identity['dataset'] and data['data_content_sha256']==identity['data_content_sha256'], 'Data identity changed between sessions')
            return lookup
        lookup = audit.check(f'data:{name}', check_data)
        if lookup is not None:
            data_lookup = lookup
            audit.checks[-1]['details'] = dict(sample_count=len(lookup), full_manifest_scan=True)
    seen_paths, checkpoints = set(), {}
    last_budget, total_attempts, total_controls = None, 0, 0
    for summary, context, publication, _ in selected:
        index = summary['iteration']
        def check_iteration(summary=summary, context=context, publication=publication, index=index):
            nonlocal last_budget, total_attempts, total_controls
            require(summary.get('status')=='accepted' and summary['policy_version_after']==summary['policy_version_before']+1, 'Iteration has no accepted policy advancement')
            require(summary['policy_version_before']==index-1, 'Logical iteration and policy version differ')
            require(data_lookup is not None, 'No validated complete dataset evidence')
            with archived_execution(root, summary) as archive:
                rows, rollout = audit_rollout(root, summary, data_lookup, identity['training_contract'], seen_paths, identity=identity)
                targets = audit_targets(root, summary, rows, identity['training_contract'])
                if archive is not None:
                    rollout['execution_archive'] = archive
            update = audit_update(summary, identity['training_contract'])
            budget_check(summary['budget'], last_budget)
            checkpoint = audit_checkpoint(root, summary, identity, update, publication, rows)
            require(checkpoint['session_id']==context['session_id'], 'Checkpoint belongs to a different training session')
            counters = checkpoint['execution_counters']
            if counters['contract']=='verified' and index>1:
                require(index-1 in checkpoints, 'Previous checkpoint counter evidence failed verification')
                previous = checkpoints[index-1]['execution_counters']
                require(counters['first_archived_decision_id']==previous['decision'], 'Next rollout first decision does not continue the saved checkpoint')
                require(counters['episode_count']>=previous['episode_count'], 'Checkpoint episode counter rolled back')
            checkpoints[index] = checkpoint
            last_budget = summary['budget']
            total_attempts += update['optimizer_attempts']
            total_controls += rollout['control_steps']
            return dict(iteration=index, rollout=rollout, fixed_targets=targets, update=update, checkpoint=checkpoint)
        audit.check(f'iteration:{index}', check_iteration)
    retired_count = sum(row['storage']=='retired_metadata_only' for row in checkpoints.values())
    result['checkpoint_storage'] = dict(complete_checkpoints=len(checkpoints)-retired_count,
        intentionally_retired_checkpoints=retired_count, all_historical_weight_bytes_revalidated=retired_count==0,
        retired_scope='Original publication and retirement metadata only; deleted weight bytes cannot be revalidated')
    def continuity():
        indices = [summary['iteration'] for summary, _, _, _ in selected]
        require(len(indices)>=minimum_iterations, f'Need at least {minimum_iterations} accepted iterations')
        require(indices==list(range(1, latest['iteration']+1)), 'Accepted iteration sequence has gaps or duplicates')
        return dict(accepted_iterations=len(indices), fresh_policy_versions=[summary['policy_version_before'] for summary, _, _, _ in selected])
    audit.check('multiple_fresh_policy_iterations', continuity)
    def resumes():
        continued = []
        for context in chain['contexts'].values():
            resume = context.get('resume')
            if not resume or not resume.get('training_resume'):
                continue
            initial = integer(resume['initial_iteration'], 'resume initial iteration')
            following = [summary for summary, owner, _, _ in selected if owner is context]
            require(following and following[0]['iteration']==initial+1, 'Resume first update is not the next policy version')
            final = following[-1]['iteration']
            if context.get('status')=='passed':
                require(context.get('final_iteration')==final and final>initial, 'Resume did not perform a new accepted optimization')
            require(bool(resume.get('new_backend_session_id')), 'Resume physical worker identity is missing')
            new_worker = following[0]['gmt_frozen']['execution_journal'].get('backend_session_id')
            require(new_worker==resume['new_backend_session_id'], 'Resume physical worker differs from accepted execution')
            if initial:
                require(initial in checkpoints, 'Resume initial checkpoint failed validation')
                previous = next(summary for summary, _, _, _ in selected if summary['iteration']==initial)
                old_worker = previous['gmt_frozen']['execution_journal'].get('backend_session_id')
                require(old_worker and old_worker!=new_worker, 'Resume reused an old physical worker session')
                path = resolve(root, resume['checkpoint'])
                if path.is_file():
                    require(sha256(path)==resume['sha256'], 'Resumed checkpoint bytes changed')
                else:
                    require(checkpoints[initial]['storage']=='retired_metadata_only'
                            and checkpoints[initial]['sha256']==resume['sha256'],
                            'Retired resumed checkpoint lacks verified retirement evidence')
            record = dict(initial_iteration=initial, final_iteration=final, new_backend_session_id=new_worker)
            if _counter_contract(identity)!='legacy_not_recorded':
                if initial:
                    saved_counters = checkpoints[initial]['execution_counters']
                else:
                    saved = torch.load(resolve(root, resume['checkpoint']), map_location='cpu', weights_only=False, mmap=True)
                    saved_counters = _execution_counters(saved['state'], identity)
                    del saved
                require(initial+1 in checkpoints, 'First resumed checkpoint counters failed validation')
                next_counters = checkpoints[initial+1]['execution_counters']
                require(saved_counters['decision']==next_counters['first_archived_decision_id'], 'Resumed first decision differs from restored execution counter')
                record['execution_counters'] = dict(saved={key:saved_counters[key] for key in
                    ('decision', 'attempt', 'episode_count', 'latency_budget_s')},
                    next_first_decision_id=next_counters['first_archived_decision_id'], decision_continuous=True)
            else:
                record['execution_counter_contract'] = 'legacy_not_recorded'
            continued.append(record)
        if require_resume:
            require(continued, 'No complete-state resume followed by a real Actor update')
        return continued
    resume_result = audit.check('resume_then_optimize', resumes)
    if result['execution_counter_contract']!='legacy_not_recorded':
        result['execution_counter_contract'] = ('verified' if len(checkpoints)==len(selected) and resume_result is not None
                                                else 'required_not_verified')
    def check_latest():
        require(checkpoints and latest['iteration']==max(checkpoints), 'Latest pointer differs from final validated iteration')
        path = resolve(root, latest['path'])
        require(str(path.relative_to(root))==checkpoints[latest['iteration']]['path'] and path.stat().st_size==latest['size_bytes'] and sha256(path)==latest['sha256'], 'Latest checkpoint file differs from publication')
        budget_check(latest['budget'], last_budget)
        require(latest['budget']['used']['optimizer_attempts']>=total_attempts and latest['budget']['used']['control_steps']>=total_controls, 'Budget omitted actual optimizer/physical consumption')
        require(latest['budget']['used']['accepted_iterations']>=len(checkpoints), 'Accepted iteration budget is too small')
        return dict(iteration=latest['iteration'], optimizer_attempts=total_attempts, validated_rollout_controls=total_controls)
    audit.check('published_checkpoint_and_budget', check_latest)
    def persisted_budget():
        persisted = budget_check(read_json(root/'budget.json'))
        # 构造器在已有账本时只读，复核全部扩限文件 SHA/原因/链及单调消耗。
        TrainingBudget(root/'budget.json', persisted['limits'])
        for evidence in sessions.values():
            if evidence['summary'] and 'budget' in evidence['summary']:
                budget_check(persisted, evidence['summary']['budget'])
        for summary, _, _, _ in selected:
            budget_check(persisted, summary['budget'])
        budget_check(persisted, latest['budget'])
        return dict(used=persisted['used'], session_count=len(sessions))
    audit.check('persisted_budget_monotonic', persisted_budget)


def _finish_report(result, audit, minimum_iterations, require_resume):
    statuses = [item['status'] for item in audit.checks]
    result.update(status='failed' if 'failed' in statuses else 'incomplete' if 'not_run' in statuses else 'passed',
                  checks=audit.checks, passed_checks=statuses.count('passed'), failed_checks=statuses.count('failed'),
                  not_run_checks=statuses.count('not_run'), minimum_iterations=minimum_iterations, require_resume=require_resume)
    return result


def audit_run(run_dir, *, allow_incomplete=False, minimum_iterations=2, require_resume=True):
    root = Path(run_dir).resolve()
    audit = Audit(allow_incomplete)
    result = {'version': VERSION, 'run_dir': str(root), 'audit_scope': 'CPU artifacts and metadata; no network forward or physics replay'}
    def check_run():
        value = read_json(root/'run.json')
        require(value.get('schema') in ('genmo.closedloop.stage10.run.v1',
                'genmo.closedloop.stage10.run.v2'), 'Wrong Stage10 run schema')
        require(isinstance(value.get('identity'), dict) and {'dataset', 'data_content_sha256', 'training_contract'}.issubset(value['identity']), 'Missing Stage10 run identity')
        require(value.get('mode') in ('train', 'eval', 'preflight'), 'Unknown Stage10 run mode')
        return value
    run = audit.check('run_identity', check_run)
    if run is None:
        result.update(status='failed' if any(item['status']=='failed' for item in audit.checks) else 'incomplete', checks=audit.checks)
        return result
    identity = run['identity']
    if run['schema']=='genmo.closedloop.stage10.run.v2':
        result['version'] = 'genmo.closedloop.stage10.audit.v2'
        from tools.eval.audit_closedloop_stage10_v2 import audit_training_v2
        audit.check('parallel_training_recovery_audit', lambda: audit_training_v2(
            root, run, audit, result, minimum_iterations, require_resume))
        return _finish_report(result, audit, minimum_iterations, require_resume)
    if run['mode']=='train':
        audit.check('training_recovery_audit', lambda: _audit_training(root, run, audit, result, minimum_iterations, require_resume))
        return _finish_report(result, audit, minimum_iterations, require_resume)
    sessions = sorted((root/'sessions').glob('*/summary.json'))
    summaries, data_lookup, iteration_summaries, evaluation_checkpoints = [], None, [], {}
    for path in sessions:
        summary = audit.check(f'session_json:{path.parent.name}', lambda path=path: read_json(path))
        if summary is None:
            continue
        def check_session(summary=summary, path=path):
            require(summary.get('status')=='passed' and summary.get('exit_code')==0, 'Session did not exit successfully')
            completion = read_json(path.parent/'completion.json')
            require(completion.get('schema')=='genmo.closedloop.stage10.session_completion.v1', 'Wrong session completion schema')
            require(completion.get('summary_sha256')==sha256(path), 'Session summary differs from completion marker')
            require(completion.get('status')==summary['status'] and completion.get('exit_code')==summary['exit_code']==0,
                    'Session completion status/exit code differs from successful summary')
            require(summary['source_unchanged'].get('unchanged') is True and summary.get('original_assets_unchanged') is True, 'Session source/assets changed')
            if summary['mode']!='preflight':
                gmt = summary['worker_shutdown']['gmt']
                require(gmt.get('policy_unchanged') is True and gmt.get('runtime_parameters_unchanged') is True
                        and gmt.get('process_exit_code')==0 and not gmt.get('forced_shutdown') and not gmt.get('close_error'), 'Worker did not close with frozen-state proof')
            budget_check(summary['budget'])
            return dict(session_id=summary['session_id'], mode=summary['mode'], initial_iteration=summary.get('initial_iteration'),
                        final_iteration=summary.get('final_iteration'), completion_summary_sha256=completion['summary_sha256'])
        audit.check(f'session:{path.parent.name}', check_session)
        if run['mode']=='eval' and identity.get('stage10')=='genmo.closedloop.stage10.v2':
            loaded = audit.check(f'evaluation_weights_only:{path.parent.name}',
                lambda summary=summary: audit_evaluation_restore(summary, identity))
            if loaded is not None:
                evaluation_checkpoints[summary['session_id']] = loaded
        try:
            data = read_json(resolve(root, summary['data_audit']))
            lookup = audit_data(data)
            require(data['identity']==identity['dataset'] and data['data_content_sha256']==identity['data_content_sha256'], 'Data identity changed between sessions')
            data_lookup = lookup
            audit.checks.append(dict(name=f'data:{path.parent.name}', status='passed', details={'sample_count': len(lookup), 'full_manifest_scan': True}))
        except Exception as error:
            audit.checks.append(dict(name=f'data:{path.parent.name}', status='failed', error=f'{type(error).__name__}: {error}'))
        summaries.append(summary)
        for reference in summary.get('iterations', []):
            def load_iteration(reference=reference):
                value = read_json(resolve(root, reference))
                integer(value['iteration'], 'iteration', 1)
                return value
            iteration = audit.check(f'iteration_json:{reference}', load_iteration)
            if iteration is not None:
                iteration_summaries.append((iteration, summary))
    if not sessions:
        audit.check('sessions_present', lambda: (_ for _ in ()).throw(FileNotFoundError('No session summary has been published')))
    if data_lookup is not None:
        for session in summaries:
            for reference in session.get('evaluations', []):
                audit.check(f'evaluation:{reference}', lambda reference=reference, session=session: audit_evaluation(
                    root, resolve(root, reference), data_lookup, identity['dataset'],
                    loaded_checkpoint=evaluation_checkpoints.get(session['session_id'])))
    if run.get('mode')=='eval':
        audit.check('explicit_evaluation_present', lambda: require(any(session.get('evaluations') for session in summaries), 'No explicit-checkpoint evaluation was published') or {'present': True})
    else:
        iteration_summaries.sort(key=lambda pair: pair[0]['iteration'])
        seen_paths, last_budget, total_attempts, total_controls, checkpoints = set(), None, 0, 0, {}
        for summary, session in iteration_summaries:
            index = summary['iteration']
            def check_iteration(summary=summary, session=session, index=index):
                nonlocal last_budget, total_attempts, total_controls
                require(summary.get('status')=='accepted' and summary['policy_version_after']==summary['policy_version_before']+1, 'Iteration has no accepted policy advancement')
                require(summary['policy_version_before']==index-1, 'Logical iteration and policy version differ')
                if data_lookup is None:
                    raise ValueError('No validated complete dataset evidence')
                rows, rollout = audit_rollout(root, summary, data_lookup, identity['training_contract'], seen_paths, identity=identity)
                targets = audit_targets(root, summary, rows, identity['training_contract'])
                update = audit_update(summary, identity['training_contract'])
                budget_check(summary['budget'], last_budget)
                checkpoint = audit_checkpoint(root, summary, identity, update)
                require(checkpoint['session_id']==session['session_id'], 'Checkpoint belongs to a different training session')
                checkpoints[index] = checkpoint
                last_budget = summary['budget']
                total_attempts += update['optimizer_attempts']
                total_controls += rollout['control_steps']
                return dict(iteration=index, rollout=rollout, fixed_targets=targets, update=update, checkpoint=checkpoint)
            audit.check(f'iteration:{index}', check_iteration)
        def check_continuity():
            indices = [summary['iteration'] for summary, _ in iteration_summaries]
            require(len(indices)>=minimum_iterations, f'Need at least {minimum_iterations} accepted iterations')
            require(indices==list(range(1, max(indices)+1)), 'Accepted iteration sequence has gaps or duplicates')
            return dict(accepted_iterations=len(indices), fresh_policy_versions=[summary['policy_version_before'] for summary, _ in iteration_summaries])
        audit.check('multiple_fresh_policy_iterations', check_continuity)
        if require_resume:
            def check_resume():
                continued = []
                for session in summaries:
                    resume = session.get('resume')
                    if session.get('mode')!='train' or not resume or not resume.get('training_resume'):
                        continue
                    require(resume.get('restored_full_state') is True and resume.get('old_buffer_discarded') is True, 'Resume reused old Buffer or did not restore full state')
                    initial = integer(resume['initial_iteration'], 'resume initial iteration')
                    require(initial==session['initial_iteration'] and session['final_iteration']>initial, 'Resume did not perform a new accepted optimization')
                    require(initial in checkpoints and bool(resume.get('new_backend_session_id')), 'Resume initial checkpoint/session evidence is missing')
                    path = resolve(root, resume['checkpoint'])
                    require(str(path.relative_to(root))==checkpoints[initial]['path'] and sha256(path)==resume['sha256'], 'Resumed checkpoint differs from the previous accepted publication')
                    following = [summary for summary, owner in iteration_summaries if owner is session]
                    require(following and following[0]['iteration']==initial+1, 'Resume first update is not the next policy version')
                    # worker UUID 必须和旧物理session不同，而非只换上层run session。
                    previous_summary = next(summary for summary, _ in iteration_summaries if summary['iteration']==initial)
                    old_session = previous_summary['gmt_frozen'].get('execution_journal', {}).get('backend_session_id')
                    require(old_session and resume['new_backend_session_id']!=old_session, 'Resume reused an old physical worker session')
                    continued.append(dict(initial_iteration=initial, final_iteration=session['final_iteration'], new_backend_session_id=resume['new_backend_session_id']))
                require(continued, 'No complete-state resume followed by a real Actor update')
                return continued
            audit.check('resume_then_optimize', check_resume)
        def check_latest():
            latest = read_json(root/'latest.json')
            require(latest.get('schema')=='genmo.closedloop.stage10.checkpoint_publication.v1', 'Wrong latest publication schema')
            require(latest['iteration']==max(checkpoints), 'Latest pointer differs from final validated iteration')
            path = resolve(root, latest['path'])
            require(str(path.relative_to(root))==checkpoints[latest['iteration']]['path'] and path.stat().st_size==latest['size_bytes'] and sha256(path)==latest['sha256'], 'Latest checkpoint file differs from publication')
            publication = read_json(resolve(root, latest['publication']))
            require(publication=={key: value for key, value in latest.items() if key!='publication'}, 'Latest and immutable publication disagree')
            budget_check(latest['budget'], last_budget)
            require(latest['budget']['used']['optimizer_attempts']>=total_attempts and latest['budget']['used']['control_steps']>=total_controls, 'Budget omitted actual optimizer/physical consumption')
            require(latest['budget']['used']['accepted_iterations']>=len(checkpoints), 'Accepted iteration budget is too small')
            return dict(iteration=latest['iteration'], optimizer_attempts=total_attempts, validated_rollout_controls=total_controls)
        audit.check('published_checkpoint_and_budget', check_latest)
    def check_persisted_budget():
        persisted = budget_check(read_json(root/'budget.json'))
        for session in summaries:
            budget_check(persisted, session['budget'])
        for summary, _ in iteration_summaries:
            budget_check(persisted, summary['budget'])
        if (root/'latest.json').exists():
            budget_check(persisted, read_json(root/'latest.json')['budget'])
        return {'used': persisted['used'], 'session_count': len(summaries)}
    audit.check('persisted_budget_monotonic', check_persisted_budget)
    statuses = [item['status'] for item in audit.checks]
    result.update(status='failed' if 'failed' in statuses else 'incomplete' if 'not_run' in statuses else 'passed',
                  checks=audit.checks, passed_checks=statuses.count('passed'), failed_checks=statuses.count('failed'),
                  not_run_checks=statuses.count('not_run'))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--allow-incomplete', action='store_true')
    parser.add_argument('--minimum-iterations', type=int, default=2)
    parser.add_argument('--no-require-resume', action='store_true', help='Explicitly report a pre-resume partial acceptance scope')
    args = parser.parse_args(argv)
    if args.minimum_iterations<1:
        parser.error('--minimum-iterations must be >=1')
    report = audit_run(args.run_dir, allow_incomplete=args.allow_incomplete,
                       minimum_iterations=args.minimum_iterations, require_resume=not args.no_require_resume)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps({'status': report['status'], 'passed_checks': report.get('passed_checks', 0),
                      'failed_checks': report.get('failed_checks', 0)}, ensure_ascii=False))
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':
    raise SystemExit(main())
