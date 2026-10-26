"""第二阶段固定均衡小验证集的计划、汇总、曲线与模型选择工具。

本模块复用既有evaluation任务身份和奖励聚合，不创建仿真环境、优化器或训练任务。
默认从四个val来源分别固定选4条配对样本，与42/1729两个seed形成32个10秒任务；
各rank按配对样本索引分片，不改变task_id、音乐起点或噪声种子。汇总必须覆盖完整
固定计划，逐episode文件SHA、身份、实际奖励/时长和网络/RNG只读边界重新检查。
周期子集只作训练监视，不冒充完整val/test验收，也不声称第一阶段从未见过这些数据。
显式training_progress计划也可固定选择当前train与原train交集，报告保留该用途，
与MC价值监督严格分开；更换音乐清单后必须重新建立同任务Stage1物理基线。

提供四来源等权回报及基线失败/时长保护的best observed/best saved选择，嵌套数值
指标的完整JSONL记录、显式白名单TensorBoard曲线，以及固定初始随机链上的高斯KL漂移诊断。所有权重
身份可来自内存模型指纹，不强求周期评估先写checkpoint。调用者负责独立环境、
独立物理预算、RNG隔离、原子发布报告和实际奖励/执行配置的一致性；源码只作记录。
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from collections.abc import Mapping
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
from gem.closedloop.dppo.music_tasks import SOURCE_PROBABILITIES
from .performance import profiled
from .execution_compatibility import same_execution,same_evaluation_execution
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
    training_progress=plan.get('evaluation_purpose')=='training_progress'
    _require(plan.get('periodic_subset', {}).get('schema') == VERSION
             and plan.get('split') in (('train',) if training_progress else ('val','test'))
             and plan.get('training_buffer_allowed') is False,
             'Periodic evaluation requires an explicit evaluation-only subset plan')
    tasks = plan['tasks']
    _require(len(tasks) == plan['task_count'] == plan['selected_sample_count'] * len(plan['seeds'])
             and len({t['task_id'] for t in tasks}) == len(tasks), 'Periodic tasks duplicate or omit seeds')
    for task in tasks:
        _require(task['split']==plan['split'] and task['sample']['row']['split']==plan['split'],
                 'Periodic evaluation sample split differs')
        if plan.get('pool_original_split') is not None:
            _require(task['sample']['row'].get('resplit_provenance',{}).get('original_split')==plan['pool_original_split'],
                     'Periodic evaluation original split differs')
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
    counts=plan['periodic_subset']['source_sample_counts']
    _require(set(t['dataset'] for t in tasks) == set(SOURCES)
             and all(sum(t['dataset'] == source for t in tasks) == counts[source]*nseed for source in SOURCES),
             'Periodic parent does not contain the balanced four-source subset')


def build_balanced_plan(catalog, *, samples_per_source=4, seeds=(42, 1729), seconds=10., selection_seed=42,
                        start_mode='beginning',unique_music=False,split='val',training_progress=False,original_split=None):
    """在独立局部RNG中固定选样，复用全池任务的原task/noise身份。"""
    all_music=samples_per_source=='all_unique'
    per_source=isinstance(samples_per_source,Mapping)
    _require(all_music and unique_music or type(samples_per_source) is int and samples_per_source > 0
        or per_source and set(samples_per_source)==set(SOURCES)
        and all(type(n) is int and n>0 for n in samples_per_source.values()),
        'samples_per_source must be positive, an explicit four-source count map, or all_unique with unique_music')
    _require(split in (('train',) if training_progress else ('val','test'))
             and set(catalog.samples[split]) == set(SOURCES), 'Periodic plan requires four explicit evaluation sources')
    from .evaluation import evaluation_samples
    samples_by_source=evaluation_samples(catalog,split,original_split)
    _require(start_mode in ('beginning','center','mixed'), 'Invalid periodic start_mode')
    _require(start_mode=='beginning' or seconds is not None, 'Complete song evaluation must begin at music start')
    full = build_evaluation_tasks(catalog, split=split, eval_count='all', seeds=seeds,
        selection_seed=selection_seed, start_mode='center' if start_mode=='center' else 'beginning', episode_seconds=seconds,
        training_progress=training_progress,original_split=original_split)
    rng = np.random.default_rng(selection_seed)
    chosen = set();source_counts={};seen_groups=set();seen_audio=set()
    for source in sorted(SOURCES):
        samples = sorted(samples_by_source[source], key=lambda s: str(s['row']['sample_id']))
        if unique_music:
            groups={}
            for sample in samples:groups.setdefault(sample['group_id'],[]).append(sample)
            samples=[min(group,key=lambda s:(-s['row']['num_frames'],s['row']['sample_id'])) for _,group in sorted(groups.items())]
            # 同一音频的不同动作片段不能充当新增的独立音乐。
            unique={}
            for sample in samples:unique.setdefault(sample['row']['source_audio_sha256'],sample)
            samples=[sample for sample in unique.values() if sample['group_id'] not in seen_groups
                     and sample['row']['source_audio_sha256'] not in seen_audio]
        count=len(samples) if all_music else samples_per_source[source] if per_source else samples_per_source
        _require(len(samples) >= count>0, f'Insufficient periodic evaluation samples: {source}')
        source_counts[source]=count
        for index in sorted(rng.choice(len(samples), size=count, replace=False).tolist()):
            sample=samples[index]
            chosen.add((source, str(sample['row']['sample_id'])))
            seen_groups.add(sample['group_id']);seen_audio.add(sample['row']['source_audio_sha256'])
    plan = copy.deepcopy(full)
    tasks = [t for t in plan['tasks'] if (t['dataset'], t['sample_id']) in chosen]
    if start_mode=='mixed':
        center=build_evaluation_tasks(catalog,split=split,eval_count='all',seeds=seeds,
            selection_seed=selection_seed,start_mode='center',episode_seconds=seconds,
            training_progress=training_progress,original_split=original_split)
        centered={(t['dataset'],t['sample_id'],t['seed']):t for t in center['tasks']}
        counts={source:0 for source in SOURCES};mixed=[]
        for index in range(0,len(tasks),len(seeds)):
            group=tasks[index:index+len(seeds)];source=group[0]['dataset']
            mixed.extend(centered[(t['dataset'],t['sample_id'],t['seed'])] if counts[source]%2 else t for t in group)
            counts[source]+=1
        tasks=mixed
    samples = [t['sample'] for t in tasks[::len(full['seeds'])]]
    plan.update(tasks=tasks, selected_sample_count=len(chosen), task_count=len(tasks), requested_eval_count=len(chosen),
        selected_unique_group_count=len({s['group_id'] for s in samples}),
        selected_unique_audio_count=len({s['row']['source_audio_sha256'] for s in samples}),
        selection_rule='fixed_source_balanced_unique_music' if unique_music else 'fixed_source_balanced_without_replacement',
        start_mode=start_mode,
        periodic_subset=dict(schema=VERSION, samples_per_source=samples_per_source,
            source_sample_counts=source_counts,
            full_pool_parent_plan_sha256=full['plan_sha256'], full_heldout_acceptance=False))
    if unique_music:plan['selection_unit']='unique_music_group_and_audio_per_source'
    if seconds is None:plan['complete_music_scope']='full_available_paired_sample_not_uncropped_original_recording'
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
    from .compact_evidence import read_episode
    episode = read_episode(path)
    _require(episode.get('transition_valid') is True and reference.get('transition_valid') is True
             and episode['task_id'] == reference['task_id'], 'Invalid periodic episode')
    return episode


def _episode_row(episode):
    overall = aggregate_evaluation([episode])['overall']
    controls=[r for d in episode['decisions'] for r in d['controls']]
    actual=sum(float(r.get('activity',{}).get('actual_activity_rad_s') or 0)*r['dt_s'] for r in controls)
    target=sum(float(r.get('activity',{}).get('target_activity_rad_s') or 0)*r['dt_s'] for r in controls)
    return dict(task_id=episode['task_id'], dataset=episode['dataset'], seed=episode['seed'],
        reward=overall['reward_sum'], executed_seconds=overall['executed_seconds'],
        activity_ratio=actual/target if target>0 else None,
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
    _require(same_evaluation_execution(baseline.get('evaluation_identity'), current.get('evaluation_identity')), 'Periodic execution/reward configuration differs')
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
    deltas = [r['reward_delta'] for r in pairs]
    tasks={t['task_id']:t for t in baseline.get('plan',{}).get('tasks',[])}
    groups=[(r['dataset'],tasks.get(r['task_id'],{}).get('sample',{}).get('group_id',r['task_id'])) for r in pairs]
    music = {}
    for key, pair in zip(groups, pairs):
        music.setdefault(key, []).append(pair['reward_delta'])
    means = [float(np.mean(values)) for values in music.values()]
    music_summary = dict(count=len(means), improved=sum(v > 0 for v in means),
        regressed=sum(v < 0 for v in means), unchanged=sum(v == 0 for v in means),
        improvement_fraction=sum(v > 0 for v in means)/len(means),
        reward_delta=dict(**_stats(means), p10=float(np.quantile(means, .1))),
        scope='one_vote_per_music_seeds_averaged',
        music=[dict(dataset=key[0], group_id=key[1], reward_delta=float(np.mean(values)), seed_count=len(values))
               for key, values in music.items()])
    return dict(task_count=len(pairs), reward_delta=_stats(deltas),
        music_summary=music_summary,
        duration_delta=_stats([r['duration_delta'] for r in pairs]), new_failures=sum(r['new_failure'] for r in pairs),
        failure_recovered=sum(r['failure_recovered'] for r in pairs), pairs=pairs, regression_alert=_regression_alert(deltas,groups=groups))


def _regression_alert(deltas, z=1.2815515655446004, *, groups=None):
    """来源等权、音乐成簇的单侧90%上界；这里只给单次证据，连续次数由checkpoint状态维护。"""
    n = len(deltas)
    if n < 2:
        return dict(available=False, upper90=None, triggered=False, scope='paired_mean_reward_delta_one_sided_90')
    mean = sum(deltas) / n
    sd = math.sqrt(sum((d-mean)**2 for d in deltas) / (n-1))
    upper = mean + z*sd/math.sqrt(n)
    scope='paired_mean_reward_delta_one_sided_90'
    if groups is not None:
        by_source={}
        for (source,group),delta in zip(groups,deltas):by_source.setdefault(source,{}).setdefault(group,[]).append(delta)
        rng=np.random.default_rng(20261014);estimates=[];means=[]
        for source,music in sorted(by_source.items()):
            sums=np.array([sum(v) for v in music.values()]);counts=np.array([len(v) for v in music.values()])
            indices=rng.integers(0,len(sums),size=(4096,len(sums)))
            estimates.append(sums[indices].sum(1)/counts[indices].sum(1));means.append(sums.sum()/counts.sum())
        mean=float(np.mean(means));distribution=np.mean(estimates,axis=0)
        upper=float(np.quantile(distribution,.90));sd=float(np.std(distribution,ddof=1))*math.sqrt(n)
        scope='source_balanced_music_cluster_bootstrap_one_sided_90'
    return dict(available=True, mean=mean, standard_error=sd/math.sqrt(n), upper90=upper, triggered=upper < 0.,
        scope=scope)


def advance_regression_alert(state, alert, iteration):
    """同一接受轮重复评估不重复累计；连续两次负上界才告警，状态随checkpoint恢复。"""
    previous=state.get('regression_monitor',{})
    if previous.get('iteration')==iteration:return previous
    count=previous.get('consecutive',0)+1 if alert.get('available') and alert.get('triggered') else 0
    result=dict(iteration=iteration,consecutive=count,triggered=count>=2,upper90=alert.get('upper90'))
    state['regression_monitor']=result
    return result


def merge_periodic_reports(plan, rankreports, *, baseline=None, evaluation_identity=None,
                           source_probabilities=SOURCE_PROBABILITIES):
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
        plan_sha256=plan['plan_sha256'], task_count=len(rows), episode_metrics=rows,
        source_episode_manifests=[origins[t['task_id']] for t in plan['tasks']],
        aggregate=aggregate_evaluation(episodes), by_source=by_source, evaluation_identity=_plain(evaluation_identity),
        actor_identity=dict(model_fingerprint=fingerprint), deterministic=deterministic,
        source_balanced_reward=sum(v['reward']['mean'] for v in by_source.values())/len(SOURCES),
        mean_executed_seconds=sum(r['executed_seconds'] for r in rows)/len(rows),
        physical_failure_count=sum(r['physical_failure'] for r in rows),
        assertions=dict(all_fixed_tasks_included=True, original_task_identity_and_noise_preserved=True,
            rank_models_unchanged=True, full_heldout_acceptance=False))
    probabilities = dict(zip(SOURCES, source_probabilities))
    _require(len(source_probabilities) == len(SOURCES) and
             all(math.isfinite(p) and p > 0 for p in probabilities.values()) and
             math.isclose(sum(probabilities.values()), 1.), 'Invalid evaluation source weights')
    result['training_source_weighted_reward'] = sum(probabilities[s]*by_source[s]['reward']['mean'] for s in SOURCES)
    result['training_source_weights'] = probabilities
    result['weighting_scope'] = 'source_mixture_only_fixed_music_panel_not_adaptive_training_distribution'
    if baseline is not None:
        result['paired_baseline'] = paired_summary(baseline, result)
    return result


def choose_best(current, baseline, *, best_observed=None, best_saved=None, saved=False, iteration=None):
    """先守住基线失败数/平均时长，再按四来源等权reward选；保存身份单独标记。"""
    paired = paired_summary(baseline, current)
    eligible = (current['physical_failure_count'] <= baseline['physical_failure_count']
                and current['mean_executed_seconds'] + 1e-9 >= baseline['mean_executed_seconds'])
    # 不做单个来源专项优化；保留整体活动度不得明显萎缩的选模保护。
    initial_activity=[r['activity_ratio'] for r in baseline['episode_metrics'] if r.get('activity_ratio') is not None]
    actual_activity=[r['activity_ratio'] for r in current['episode_metrics'] if r.get('activity_ratio') is not None]
    activity_ok=(not initial_activity or (len(actual_activity)==len(initial_activity)
        and sum(actual_activity)>=.8*sum(initial_activity)))
    eligible=eligible and activity_ok and current.get('physical_guard',{}).get('passed',True)
    candidate = dict(iteration=iteration, plan_sha256=current['plan_sha256'], actor_identity=current['actor_identity'],
        score=current['source_balanced_reward'], physical_failure_count=current['physical_failure_count'],
        mean_executed_seconds=current['mean_executed_seconds'], saved=bool(saved),
        evaluation_identity=current.get('evaluation_identity'), session_id=current.get('session_id'))
    if eligible and (best_observed is None or candidate['score'] > best_observed['score']):
        best_observed = candidate
    if eligible and saved and (best_saved is None or candidate['score'] > best_saved['score']):
        best_saved = candidate
    return dict(eligible=eligible, best_observed=best_observed, best_saved=best_saved,
                activity_retention_passed=activity_ok,paired_new_failures=paired['new_failures'])


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
                or not same_evaluation_execution(candidate.get('evaluation_identity'), current.get('evaluation_identity'))):
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
        metadata={**{key:copy.deepcopy(row.metadata[key]) for key in
                  ('remaining_music_seconds','next_remaining_music_seconds','event_reward') if key in row.metadata},
            'critic_context':context(row.metadata['critic_context']),
            'next_critic_context':context(row.metadata['next_critic_context'])})


def build_fixed_critic_reference(transitions, critic, device, *, gamma_upper=.99, lambda_upper=.95):
    """用初始独立eval实际奖励建立固定GAE targets，行政截断采用初始Critic自举。

    并行世界按控制完成顺序交错返回不同环境；先按backend/env/episode/policy分组、
    按真实控制起点排序，再调用原GAE。否则相邻列表项属于不同环境，会把连续音乐
    错切成一步TD，导致独立价值诊断偏乐观。这里只整理独立快照，不修改执行顺序、
    普通训练GAE或已经发布的固定参考；已有参考继续代表当时保存的原始标签。
    targets来自既有fixed_targets，继续/终止/无效mask及实际控制时间折扣保持原合同。
    后续诊断只向这些固定targets做回归，不能解释为无偏真实价值误差或物理质量。
    """
    from .trainer import fixed_targets
    rows = [snapshot_critic_transition(row) for row in transitions]
    _require(bool(rows), 'No executed initial evaluation transitions for Critic reference')
    episodes = {}
    for row in rows:
        identity = row.identity
        key = (identity['backend_session_id'], identity.get('env_id', 0),
               identity['episode_id'], identity['policy_version'])
        episodes.setdefault(key, []).append(row)
    rows = [row for episode in episodes.values()
            for row in sorted(episode, key=lambda item: item.control_tick_begin)]
    modes = {name: child.training for name, child in critic.named_modules()}
    try:
        targets = fixed_targets(rows, critic, device, gamma_upper=gamma_upper,
                                lambda_upper=lambda_upper, normalize=False)
    finally:
        for name, child in critic.named_modules():
            child.training = modes[name]
    records = [dict(context=row.context, critic_context=row.metadata['critic_context'], remaining_music_seconds=row.metadata['remaining_music_seconds'],
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
        for key, value in sorted(record['critic_context'].items()):
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
                context = {key: value.to(device) for key, value in record['critic_context'].items()}
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


def actor_minibatch_curves(records):
    """将逐去噪步骤映射到稳定曲线标签，保留原始审计列表与缺测语义。

    通用曲线展开器有意跳过任意列表，避免把全量链身份或大数组写成海量scalar。
    这里只显式展开每次Actor更新的有限20步统计；未抽中的步骤没有ratio/KL均值，
    仍为None而非零。Stratified8的访问分布与Full20不同，不在此重新加权或改数值。
    """
    return {f'step{index+1}': dict(record, denoising_steps={
        f'step{step["step"]}': dict(step) for step in record.get('per_denoising_step', [])})
        for index, record in enumerate(records)}


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
def log_metrics(writer, jsonl_path, step, metrics, *, prefix='', durable=True, dashboard_metrics=None):
    """JSONL保留审计数值；TensorBoard只写调用者明确聚合的白名单，横轴为接受迭代。"""
    _require(type(step) is int and step >= 0, 'Monitor step must be nonnegative')
    values = numeric_metrics(metrics, prefix)
    if writer is not None:
        if dashboard_metrics is None:
            raise ValueError('TensorBoard requires explicit compact dashboard_metrics')
        writer.write(prefix, step, dashboard_metrics)
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
                prepared = policy.prepare_conditions(context,include_frozen=False)
                for step in policy.trainable_steps:
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
    result = dict(gaussian_steps=list(policy.trainable_steps),frozen_identity=policy.assert_frozen_identity(),reference_sha256=reference['reference_sha256'], internal_transition_count=len(values),
        mean_joint_kl=stats['mean'], p95_joint_kl=stats['p95'], max_joint_kl=stats['max'],
        diagnostic_scope='fixed_initial_chain_gaussian_drift_not_physics_quality')
    if include_values:
        result['joint_kl_values'] = values
    return result
