"""第二阶段固定均衡小验证集的计划、汇总、曲线与模型选择工具。

本模块复用既有evaluation任务身份和奖励聚合，不创建仿真环境、优化器或训练任务。
默认从四个val来源分别固定选4条配对样本，与42/1729两个seed形成32个10秒任务；
各rank按配对样本索引分片，不改变task_id、音乐起点或噪声种子。汇总必须覆盖完整
固定计划，逐episode文件SHA、身份、实际奖励/时长和网络/RNG只读边界重新检查。
周期子集只作训练监视，不冒充完整val/test验收，也不声称第一阶段从未见过这些数据。

提供四来源等权回报及基线失败/时长保护的best observed/best saved选择，嵌套数值
指标的TensorBoard/JSONL记录，以及固定初始随机链上的高斯KL漂移诊断。所有权重
身份可来自内存模型指纹，不强求周期评估先写checkpoint。调用者负责独立环境、
独立物理预算、RNG隔离、原子发布报告和真实源码/奖励/执行合同的一致性。
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from gem.closedloop.dppo.evaluation import (
    _digest,
    _plain,
    _stats,
    aggregate_evaluation,
    build_evaluation_tasks,
)
from gem.closedloop.dppo.full_dataset import SOURCES
from .performance import profiled
from gem.closedloop.frozen_actor import stable_noise_seed

VERSION = 'genmo.closedloop.stage10.periodic_monitor.v1'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _plan_valid(plan):
    _require(plan.get('plan_sha256') == _digest({k: v for k, v in plan.items() if k != 'plan_sha256'}),
             'Periodic plan SHA differs')
    _require(plan.get('periodic_subset', {}).get('schema') == VERSION
             and plan.get('split') == 'val' and plan.get('training_buffer_allowed') is False,
             'Periodic evaluation requires an explicit held-out subset plan')
    tasks = plan['tasks']
    _require(len(tasks) == plan['task_count'] == plan['selected_sample_count'] * len(plan['seeds'])
             and len({t['task_id'] for t in tasks}) == len(tasks), 'Periodic tasks duplicate or omit seeds')
    for task in tasks:
        identity = {k: task[k] for k in ('dataset', 'sample_id', 'split', 'seed', 'music_start_frame')}
        _require(task['task_id'] == _digest(identity)[:24]
                 and task['noise_index_start'] == stable_noise_seed(int(task['seed']), _digest(identity)),
                 'Periodic task identity/noise differs')
    _require('periodic_partition' not in plan, 'Periodic helpers require the complete subset parent plan')
    nseed = len(plan['seeds'])
    _require(nseed > 0, 'Periodic plan has no seeds')
    for position in range(0, len(tasks), nseed):
        group = tasks[position:position+nseed]
        _require([t['seed'] for t in group] == plan['seeds']
                 and len({(t['dataset'], t['sample_id']) for t in group}) == 1,
                 'Periodic paired sample/seed ordering differs')
    count = plan['periodic_subset']['samples_per_source']*nseed
    _require(set(t['dataset'] for t in tasks) == set(SOURCES)
             and all(sum(t['dataset'] == source for t in tasks) == count for source in SOURCES),
             'Periodic parent does not contain the balanced four-source subset')


def build_balanced_plan(catalog, *, samples_per_source=4, seeds=(42, 1729), seconds=10., selection_seed=42):
    """在独立局部RNG中固定选样，复用全池任务的原task/noise身份。"""
    _require(type(samples_per_source) is int and samples_per_source > 0, 'samples_per_source must be positive')
    _require(set(catalog.samples['val']) == set(SOURCES), 'Periodic plan requires the four original sources')
    full = build_evaluation_tasks(catalog, split='val', eval_count='all', seeds=seeds,
        selection_seed=selection_seed, start_mode='beginning', episode_seconds=seconds)
    rng = np.random.default_rng(selection_seed)
    chosen = set()
    for source in sorted(SOURCES):
        samples = sorted(catalog.samples['val'][source], key=lambda s: str(s['row']['sample_id']))
        _require(len(samples) >= samples_per_source, f'Insufficient periodic val samples: {source}')
        for index in sorted(rng.choice(len(samples), size=samples_per_source, replace=False).tolist()):
            chosen.add((source, str(samples[index]['row']['sample_id'])))
    plan = copy.deepcopy(full)
    tasks = [t for t in plan['tasks'] if (t['dataset'], t['sample_id']) in chosen]
    samples = [t['sample'] for t in tasks[::len(full['seeds'])]]
    plan.update(tasks=tasks, selected_sample_count=len(chosen), task_count=len(tasks), requested_eval_count=len(chosen),
        selected_unique_group_count=len({s['group_id'] for s in samples}),
        selected_unique_audio_count=len({s['row']['source_audio_sha256'] for s in samples}),
        selection_rule='fixed_source_balanced_without_replacement',
        periodic_subset=dict(schema=VERSION, samples_per_source=samples_per_source,
            source_sample_counts=dict.fromkeys(sorted(SOURCES), samples_per_source),
            full_pool_parent_plan_sha256=full['plan_sha256'], full_heldout_acceptance=False))
    plan['plan_sha256'] = _digest({k: v for k, v in plan.items() if k != 'plan_sha256'})
    _plan_valid(plan)
    return plan


def shard_plan(plan, rank, world):
    """同一样本的所有seed由同一rank执行；旧evaluate_policy直接接受输出schema。"""
    _plan_valid(plan)
    _require(type(world) is int and world >= 1
             and type(rank) is int and 0 <= rank < world, 'Invalid periodic rank/world')
    child = copy.deepcopy(plan)
    nseed = len(plan['seeds'])
    indices = list(range(rank, plan['selected_sample_count'], world))
    tasks = [task for index in indices for task in plan['tasks'][index*nseed:(index+1)*nseed]]
    child.update(tasks=tasks, task_count=len(tasks), selected_sample_count=len(indices), requested_eval_count=len(indices),
        periodic_partition=dict(rank=rank, world=world, parent_plan_sha256=plan['plan_sha256'], sample_indices=indices))
    child['plan_sha256'] = _digest({k: v for k, v in child.items() if k != 'plan_sha256'})
    return child


def _read_rank_report(value):
    if isinstance(value, (str, Path)):
        path = Path(value).resolve(strict=True)
        return json.loads(path.read_text()), path.parent
    _require(isinstance(value, dict) and 'report' in value and 'root' in value,
             'Pass a report.json path or {report, root} for periodic evidence')
    return value['report'], Path(value['root']).resolve(strict=True)


def _load_episode(root, reference):
    path = (root/reference['path']).resolve(strict=True)
    _require(path.is_relative_to(root) and path.is_file() and _hash(path) == reference['sha256'],
             'Periodic episode path or SHA differs')
    episode = json.loads(path.read_text())
    _require(episode.get('transition_valid') is True and reference.get('transition_valid') is True
             and episode['task_id'] == reference['task_id'], 'Invalid periodic episode')
    return episode


def _episode_row(episode):
    overall = aggregate_evaluation([episode])['overall']
    return dict(task_id=episode['task_id'], dataset=episode['dataset'], seed=episode['seed'],
        reward=overall['reward_sum'], executed_seconds=overall['executed_seconds'],
        physical_failure=int(episode['physical_failure']), rejection_count=overall['rejection_count'])


def _verify_episode(episode, task, limit):
    """真实控制计数、奖励和执行时长不可只靠报告自报值交叉相等。"""
    for key in ('task_id', 'dataset', 'sample_id', 'split', 'seed', 'music_start_frame', 'noise_index_start'):
        _require(episode[key] == task[key], f'Periodic episode identity differs: {key}')
    _require(episode['episode_seconds_limit'] == limit, 'Periodic episode duration differs')
    total, seconds = 0., 0.
    for decision in episode['decisions']:
        controls = decision['controls']
        _require(len(controls) == decision['executed_control_steps']
                 and decision['executed_physics_steps'] == 4*len(controls), 'Periodic real control/physics counts differ')
        reward = sum(float(c['reward']) for c in controls)+float(decision['zero_step_event_reward'])
        _require(math.isfinite(reward) and math.isclose(reward, decision['reward_sum'], abs_tol=1e-8, rel_tol=1e-9),
                 'Periodic real reward differs')
        total += reward
        for control in controls:
            dt = float(control['dt_s'])
            _require(math.isfinite(dt) and dt > 0, 'Invalid periodic control duration')
            seconds += dt
    _require(math.isclose(total, episode['reward_sum'], abs_tol=1e-8, rel_tol=1e-9), 'Periodic episode reward differs')
    maximum = task['remaining_music_seconds'] if limit is None else min(task['remaining_music_seconds'], limit)
    _require(seconds <= maximum+.02000001, 'Periodic execution exceeds fixed task window')


def paired_summary(baseline, current):
    _require(baseline['plan_sha256'] == current['plan_sha256'], 'Periodic baseline plan differs')
    _require(baseline.get('evaluation_identity') == current.get('evaluation_identity'), 'Periodic execution/reward identity differs')
    before = {r['task_id']: r for r in baseline['episode_metrics']}
    after = {r['task_id']: r for r in current['episode_metrics']}
    _require(set(before) == set(after), 'Periodic baseline tasks differ')
    pairs = []
    for task_id, row in before.items():
        other = after[task_id]
        pairs.append(dict(task_id=task_id, dataset=row['dataset'], seed=row['seed'],
            reward_delta=other['reward']-row['reward'], duration_delta=other['executed_seconds']-row['executed_seconds'],
            new_failure=bool(other['physical_failure'] and not row['physical_failure']),
            failure_recovered=bool(row['physical_failure'] and not other['physical_failure'])))
    return dict(task_count=len(pairs), reward_delta=_stats([r['reward_delta'] for r in pairs]),
        duration_delta=_stats([r['duration_delta'] for r in pairs]), new_failures=sum(r['new_failure'] for r in pairs),
        failure_recovered=sum(r['failure_recovered'] for r in pairs), pairs=pairs)


def merge_periodic_reports(plan, rankreports, *, baseline=None, evaluation_identity=None):
    """逐文件核验并汇总全部固定任务；报告/模型可来自内存，不要求checkpoint SHA。"""
    _plan_valid(plan)
    reports = list(rankreports)
    _require(bool(reports), 'Periodic merge has no rank reports')
    gathered, origins, ranks, fingerprint, deterministic, declared_world = {}, {}, set(), None, None, None
    for value in reports:
        report, root = _read_rank_report(value)
        _require(report.get('status') == 'passed' and report.get('updates_performed') is False
                 and report.get('training_buffer_used') is False and report.get('rng_restored') is True
                 and report.get('actor_gradients_unchanged') is True
                 and bool(report.get('networks_unchanged')) and all(report['networks_unchanged'].values())
                 and report['network_fingerprints_before'] == report['network_fingerprints_after'],
                 'Periodic evaluation was incomplete or changed model/gradient/RNG')
        child_path = (root/report['selection_path']).resolve(strict=True)
        _require(child_path.is_relative_to(root), 'Periodic selection escapes report directory')
        child = json.loads(child_path.read_text())
        partition = child.get('periodic_partition', {})
        rank, world = partition.get('rank'), partition.get('world')
        _require(type(world) is int and world >= 1, 'Invalid periodic report world')
        if declared_world is None:
            declared_world = world
        active_ranks = set(range(min(world, plan['selected_sample_count'])))
        _require(child == shard_plan(plan, rank, world) and report['plan_sha256'] == child['plan_sha256']
                 and world == declared_world and len(reports) == len(active_ranks)
                 and rank in active_ranks and rank not in ranks, 'Periodic shard plan/rank differs')
        ranks.add(rank)
        actor = report['network_fingerprints_before']['actor']
        if fingerprint is None:
            fingerprint, deterministic = actor, report['deterministic']
        _require(actor == fingerprint and report['deterministic'] == deterministic
                 and report['episode_seconds_limit'] == plan['episode_seconds'], 'Periodic ranks used different policy/semantics')
        _require(report['requested_task_count'] == report['completed_episode_count'] == len(child['tasks'])
                 == len(report['episode_manifests']), 'Periodic shard omits tasks')
        episodes = []
        for task, reference in zip(child['tasks'], report['episode_manifests']):
            episode = _load_episode(root, reference)
            _verify_episode(episode, task, plan['episode_seconds'])
            _require(episode['task_id'] not in gathered, 'Duplicate periodic task')
            gathered[episode['task_id']] = episode
            origins[episode['task_id']] = dict(task_id=episode['task_id'], sha256=reference['sha256'],
                path=str((root/reference['path']).resolve()))
            episodes.append(episode)
        _require(aggregate_evaluation(episodes) == report['aggregate'], 'Periodic rank aggregate cannot be reproduced')
    _require(ranks == set(range(min(declared_world, plan['selected_sample_count']))), 'Periodic active rank omitted')
    _require(set(gathered) == {t['task_id'] for t in plan['tasks']}, 'Periodic merge omits fixed tasks')
    episodes = [gathered[t['task_id']] for t in plan['tasks']]
    rows = [_episode_row(e) for e in episodes]
    by_source = {}
    for source in sorted(SOURCES):
        source_rows = [r for r in rows if r['dataset'] == source]
        _require(bool(source_rows), 'Periodic source missing')
        by_source[source] = dict(task_count=len(source_rows), reward=_stats([r['reward'] for r in source_rows]),
            duration=_stats([r['executed_seconds'] for r in source_rows]),
            physical_failure_count=sum(r['physical_failure'] for r in source_rows))
    result = dict(schema=VERSION+'.merge', status='passed', mode='periodic_subset', plan=copy.deepcopy(plan),
        plan_sha256=plan['plan_sha256'], task_count=len(rows), episodes=episodes, episode_metrics=rows,
        source_episode_manifests=[origins[t['task_id']] for t in plan['tasks']],
        aggregate=aggregate_evaluation(episodes), by_source=by_source, evaluation_identity=_plain(evaluation_identity),
        actor_identity=dict(model_fingerprint=fingerprint), deterministic=deterministic,
        source_balanced_reward=sum(v['reward']['mean'] for v in by_source.values())/len(SOURCES),
        mean_executed_seconds=sum(r['executed_seconds'] for r in rows)/len(rows),
        physical_failure_count=sum(r['physical_failure'] for r in rows),
        assertions=dict(all_fixed_tasks_included=True, original_task_identity_and_noise_preserved=True,
            rank_models_unchanged=True, full_heldout_acceptance=False))
    if baseline is not None:
        result['paired_baseline'] = paired_summary(baseline, result)
    return result


def choose_best(current, baseline, *, best_observed=None, best_saved=None, saved=False, iteration=None):
    """先守住基线失败数/平均时长，再按四来源等权reward选；保存身份单独标记。"""
    paired = paired_summary(baseline, current)
    eligible = (current['physical_failure_count'] <= baseline['physical_failure_count']
                and current['mean_executed_seconds'] + 1e-9 >= baseline['mean_executed_seconds'])
    candidate = dict(iteration=iteration, plan_sha256=current['plan_sha256'], actor_identity=current['actor_identity'],
        score=current['source_balanced_reward'], physical_failure_count=current['physical_failure_count'],
        mean_executed_seconds=current['mean_executed_seconds'], saved=bool(saved),
        evaluation_identity=current.get('evaluation_identity'), session_id=current.get('session_id'))
    if eligible and (best_observed is None or candidate['score'] > best_observed['score']):
        best_observed = candidate
    if eligible and saved and (best_saved is None or candidate['score'] > best_saved['score']):
        best_saved = candidate
    return dict(eligible=eligible, best_observed=best_observed, best_saved=best_saved,
                paired_new_failures=paired['new_failures'])


def load_best_pointers(output_dir, current, *, state=None):
    """恢复时优先采用评估后持久指针，核合同/轮次并排除已作废的未保存训练尾部。

    完整checkpoint可先于同轮eval写入，所以checkpoint内best字段可能落后一轮。
    根指针不能引用未来轮次、不同评估计划/合同、缺失保存文件或superseded历史尾部；
    无效指针回退到同样经过核对的checkpoint state，而不把不同训练尝试混为最佳。
    """
    root, state = Path(output_dir), state or {}
    tails = [json.loads(path.read_text()) for path in sorted((root/'superseded_tails').glob('*.json'))]
    def valid(candidate, saved):
        if not isinstance(candidate, dict):
            return False
        index = candidate.get('iteration')
        if (type(index) is not int or not 0 <= index <= current['iteration']
                or candidate.get('plan_sha256') != current['plan_sha256']
                or candidate.get('evaluation_identity') != current.get('evaluation_identity')):
            return False
        for tail in tails:
            prior = tail.get('previous_accepted', {})
            if (tail['durable_iteration'] < index <= prior.get('iteration', -1)
                    and candidate.get('session_id') in (None, prior.get('session_id'))):
                return False
        if saved:
            descriptor = candidate.get('checkpoint')
            if candidate.get('saved') is not True or not isinstance(descriptor, dict) or descriptor.get('iteration') != index:
                return False
            path = (root/descriptor.get('path', '')).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file() or path.stat().st_size == 0:
                return False
        return True
    result = {}
    for key in ('best_observed', 'best_saved'):
        path = root/f'{key}.json'
        disk = json.loads(path.read_text()) if path.exists() else None
        result[key] = copy.deepcopy(disk if valid(disk, key == 'best_saved') else
                                    state.get(key) if valid(state.get(key), key == 'best_saved') else None)
    return result


def mark_saved_evaluation(current, baseline, checkpoint, *, actor_fingerprint, output_dir,
                          best_observed=None, best_saved=None):
    """只把已落盘且对应同一轮/同一Actor的评估候选绑定到完整checkpoint。

    checkpoint 必须来自完整保存边界的返回值；调用者传入该边界实际Actor指纹。
    本函数核对文件存在、路径位于run目录、轮次与评估指纹，保留发布器已经核验的
    文件SHA，不在每次周期评估再遍历整个checkpoint。尚未保存的100轮权重不能
    因为last_durable_checkpoint_iteration字段存在就被误标为best_saved。
    """
    root = Path(output_dir).resolve(strict=True)
    path = (root/checkpoint['path']).resolve(strict=True)
    _require(path.is_relative_to(root) and path.is_file() and path.stat().st_size > 0,
             'Saved evaluation checkpoint is not a durable run file')
    _require(checkpoint['iteration'] == current.get('iteration'), 'Saved checkpoint iteration differs from evaluation')
    _require(actor_fingerprint == current['actor_identity']['model_fingerprint'],
             'Saved checkpoint Actor differs from evaluated Actor')
    declared_actor = checkpoint.get('metadata', {}).get('actor_model_fingerprint')
    _require(declared_actor is None or declared_actor == actor_fingerprint,
             'Published checkpoint Actor fingerprint differs from evaluated Actor')
    sha = checkpoint.get('sha256', '')
    _require(isinstance(sha, str) and len(sha) == 64 and all(c in '0123456789abcdef' for c in sha),
             'Saved checkpoint requires its published file SHA256')
    choice = choose_best(current, baseline, best_observed=best_observed, best_saved=best_saved,
                         saved=True, iteration=current['iteration'])
    selected = choice['best_saved']
    if selected is not None and selected['iteration'] == current['iteration']:
        selected = copy.deepcopy(selected)
        selected['checkpoint'] = {key: checkpoint[key] for key in ('iteration', 'path', 'sha256')}
        choice['best_saved'] = selected
    return choice


def snapshot_critic_transition(row):
    """复制真实eval转移的价值输入/奖励边界；不保留去噪链，不改写执行返回对象。"""
    def context(value):
        return None if value is None else {key: tensor.detach().cpu().clone() for key, tensor in value.items()}
    return SimpleNamespace(context=context(row.context), next_context=context(row.next_context),
        identity=copy.deepcopy(row.identity), rewards=list(row.rewards),
        executed_control_steps=row.executed_control_steps, control_tick_begin=row.control_tick_begin,
        control_tick_end=row.control_tick_end, terminated=row.terminated, truncated=row.truncated,
        transition_valid=row.transition_valid, old_value=0., next_value=0.,
        metadata={key: copy.deepcopy(row.metadata[key]) for key in
                  ('remaining_music_seconds', 'next_remaining_music_seconds', 'event_reward') if key in row.metadata})


def build_fixed_critic_reference(transitions, critic, device, *, gamma_upper=.99, lambda_upper=.95):
    """用初始独立eval实际奖励建立固定GAE targets，行政截断采用初始Critic自举。

    targets来自既有fixed_targets，继续/终止/无效mask及实际控制时间折扣保持原合同。
    后续诊断只向这些固定targets做回归，不能解释为无偏真实价值误差或物理质量。
    """
    from .trainer import fixed_targets
    rows = [snapshot_critic_transition(row) for row in transitions]
    _require(bool(rows), 'No executed initial evaluation transitions for Critic reference')
    modes = {name: child.training for name, child in critic.named_modules()}
    try:
        targets = fixed_targets(rows, critic, device, gamma_upper=gamma_upper,
                                lambda_upper=lambda_upper, normalize=False)
    finally:
        for name, child in critic.named_modules():
            child.training = modes[name]
    records = [dict(context=row.context, remaining_music_seconds=row.metadata['remaining_music_seconds'],
                    target=float(targets['returns'][index]), initial_prediction=row.old_value,
                    identity=row.identity, control_tick_begin=row.control_tick_begin,
                    control_tick_end=row.control_tick_end)
               for index, row in enumerate(rows) if bool(targets['valid'][index])]
    _require(bool(records), 'Initial Critic reference has no valid executed transitions')
    digest = hashlib.sha256()
    digest.update(_digest([dict(target=r['target'], remaining_music_seconds=r['remaining_music_seconds'],
                               identity=r['identity'], control_tick_begin=r['control_tick_begin'],
                               control_tick_end=r['control_tick_end']) for r in records]).encode())
    for record in records:
        for key, value in sorted(record['context'].items()):
            tensor = value.contiguous()
            digest.update(str((key, str(tensor.dtype), tuple(tensor.shape))).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return dict(schema=VERSION+'.fixed_critic_reference', records=records,
                reference_sha256=digest.hexdigest(), gamma_upper=gamma_upper, lambda_upper=lambda_upper,
                diagnostic_scope='fixed_initial_eval_gae_targets_with_initial_critic_bootstrap')


def fixed_critic_diagnostic(critic, reference):
    """独立固定初始轨迹的MSE/EV；零target方差返回缺测而非伪造EV。"""
    device = next(critic.parameters()).device
    predictions, targets = [], []
    modes = {name: child.training for name, child in critic.named_modules()}
    try:
        critic.eval()
        with torch.no_grad():
            for record in reference['records']:
                context = {key: value.to(device) for key, value in record['context'].items()}
                predictions.append(float(critic(context, torch.tensor([record['remaining_music_seconds']], device=device))[0]))
                targets.append(float(record['target']))
    finally:
        for name, child in critic.named_modules():
            child.training = modes[name]
    stats = summarize_critic_predictions(predictions, targets)
    return dict(stats, predictions=predictions, targets=targets, reference_sha256=reference['reference_sha256'],
                diagnostic_scope=reference['diagnostic_scope'])


def summarize_critic_predictions(predictions, targets):
    """跨rank原样拼接独立诊断预测/targets后统一重算，避免平均各rank EV。"""
    p, t = np.asarray(predictions, dtype=np.float64), np.asarray(targets, dtype=np.float64)
    _require(p.ndim == 1 and p.shape == t.shape and p.size > 0 and np.isfinite(p).all() and np.isfinite(t).all(),
             'Invalid fixed Critic diagnostic predictions/targets')
    error, variance = p-t, float(t.var())
    return dict(transition_count=int(p.size), mse=float(np.square(error).mean()),
                explained_variance=None if variance <= 1e-12 else float(1.-error.var()/variance),
                target_variance=variance, mean_prediction=float(p.mean()), mean_target=float(t.mean()))


def numeric_metrics(value, prefix=''):
    """展开有限嵌套数值；列表、文本和缺测不伪造为零。"""
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            result.update(numeric_metrics(item, f'{prefix}/{key}' if prefix else str(key)))
    elif type(value) in (int, float, bool):
        _require(math.isfinite(value), f'Nonfinite monitor metric: {prefix}')
        result[prefix] = float(value)
    return result


@profiled('storage.monitor')
def log_metrics(writer, jsonl_path, step, metrics, *, prefix='', durable=True):
    """同一外层轮次写TensorBoard和JSONL；调用者持有唯一writer，不创建新训练进程。"""
    _require(type(step) is int and step >= 0, 'Monitor step must be nonnegative')
    values = numeric_metrics(metrics, prefix)
    if writer is not None:
        for tag, value in values.items():
            writer.add_scalar(tag, value, step)
        if durable:
            writer.flush()
    if jsonl_path is not None:
        path = Path(jsonl_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(step=step, metrics=values), ensure_ascii=False, allow_nan=False)+'\n')
            stream.flush()
            if durable:
                os.fsync(stream.fileno())
    return values


def load_reference_traces(paths, *, max_chains=4):
    """显式初始eval raw文件形成只读固定链；不冒充训练UpperTransition。"""
    _require(type(max_chains) is int and max_chains > 0, 'max_chains must be positive')
    if isinstance(paths, (str, Path)):
        path = Path(paths)
        paths = sorted(path.rglob('*.pt')) if path.is_dir() else [path]
    records = []
    for path in list(paths)[:max_chains]:
        path = Path(path).resolve(strict=True)
        payload = torch.load(path, map_location='cpu', weights_only=False)
        trace = payload['trace']
        _require(all(k in trace for k in ('conditions', 'chain', 'old_means', 'old_stds', 'free_mask')),
                 'Reference is not a stochastic initial-policy raw trace')
        records.append(dict(path=str(path), sha256=_hash(path), trace=trace))
    _require(bool(records), 'No fixed reference chain supplied')
    return dict(schema=VERSION+'.fixed_reference', records=records,
        reference_sha256=_digest([(r['path'], r['sha256']) for r in records]))


def fixed_chain_drift(policy, reference, *, include_values=False):
    """当前高斯相对初始固定链的联合KL；没有反向、优化或UpperTransition伪造。"""
    device = next(policy.actor.parameters()).device
    values = []
    modes = {name: child.training for name, child in policy.actor.named_modules()}
    try:
        policy.actor.eval()
        with torch.no_grad():
            for record in reference['records']:
                trace = record['trace']
                _require(trace['chain'].shape[1] == policy.steps+1, 'Fixed reference denoising steps differ')
                context = {k: v.to(device) for k, v in trace['conditions'].items()}
                prepared = policy.prepare_conditions(context) if hasattr(policy, 'prepare_conditions') else None
                for step in range(policy.steps):
                    kwargs = {} if prepared is None else dict(prepared=prepared)
                    current = policy.transition_parameters(context, trace['chain'][:, step].to(device), step, **kwargs)
                    old_mean, old_std = trace['old_means'][:, step].to(device).double(), trace['old_stds'][:, step].to(device).double()
                    mean, std = current['mean'].double(), current['std'].double()
                    terms = (std/old_std).log()+(old_std.square()+(old_mean-mean).square())/(2*std.square())-.5
                    mask = trace['free_mask'].to(device)
                    values.extend(terms.masked_fill(~mask, 0).sum((-2, -1)).cpu().tolist())
    finally:
        for name, child in policy.actor.named_modules():
            child.training = modes[name]
    stats = _stats(values)
    result = dict(reference_sha256=reference['reference_sha256'], internal_transition_count=len(values),
        mean_joint_kl=stats['mean'], p95_joint_kl=stats['p95'], max_joint_kl=stats['max'],
        diagnostic_scope='fixed_initial_chain_gaussian_drift_not_physics_quality')
    if include_values:
        result['joint_kl_values'] = values
    return result
