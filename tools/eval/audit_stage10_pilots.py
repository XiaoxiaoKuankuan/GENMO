#!/usr/bin/env python3
"""只读导出和对照Stage10已完成pilot的真实证据。

--export-run可经SSH标准输入在服务器执行，只读取run目录：核验评估JSON与最终
checkpoint的SHA，逐控制步复算积分和四子步计数，提取活动度、奖励分项、末端K步
策略和完整20步KL。它不导入Isaac/Actor、不占GPU、不修改训练源码或原始产物。
同一run连续恢复时，按真实迭代顺序核验各会话及累计计数，再合并证据；每个
final评估保留所属会话的实际控制步，不能把早期final错标为整个run的终点。
--compare在本地按task身份配对，使用来源内音乐组成簇的bootstrap区间，
不把同曲不同种子视为独立音乐；分别报告普通训练轮耗时和含评估/启动的总耗时。
--run-evidence可传入两台服务器的只读证据JSON，使用其中已结束的启动日志核对总耗时；
没有启动日志时不以普通轮耗时之和冒充全程耗时。输入文件SHA随比较报告记录。
输出只用于实施进度与学习效果审计，不自动宣称G5通过或启动正式训练。
--compare-models按Stage1/P1/P2顺序比较完整歌曲导出：真实失败保留，行政截断
拒绝；按来源等权统计每秒回报、活动度，并单列实际机器人与GENMO参考关节边界。
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def scalars(value):
    return {k: v for k, v in value.items() if not isinstance(v, (dict, list))}


def checked_path(root, relative, expected_sha):
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or sha256(path) != expected_sha:
        raise ValueError(f'文件路径或SHA不匹配: {path}')
    return path


def joint_boundary_sample(component):
    """从真实控制步的原始关节与限位统计，区分实体与已消费参考，不使用混合罚分代替。"""
    raw=component['raw'];hard=raw['hard_limits_rad'];safe=raw['safe_limits_rad']
    tolerance=raw['numeric_tolerance_rad'];result={}
    for label,key in [('actual','actual_joint_position_rad'),('reference','consumed_reference_joint_position_rad')]:
        positions=raw[key]
        if not len(positions)==len(hard)==len(safe)==21:raise ValueError('关节边界证据维度不一致')
        excess=[max(lo-q,q-hi,0.) for q,(lo,hi) in zip(positions,hard)]
        result[label+'_hard_any']=any(v>tolerance for v in excess)
        result[label+'_soft_any']=any(q<lo-tolerance or q>hi+tolerance for q,(lo,hi) in zip(positions,safe))
        result[label+'_max_hard_excess_rad']=max(excess)
    return result


def episode_metrics(episode, task):
    """真实50Hz奖励已经积分，只求和；活动度用实际dt加权。"""
    for key in ('task_id', 'dataset', 'seed', 'music_start_frame', 'noise_index_start'):
        if episode[key] != task[key]:
            raise ValueError(f'任务身份不匹配: {key}')
    elapsed = reward = actual = target = gate = 0.
    controls_count = physics = 0
    components = collections.defaultdict(float)
    boundaries=collections.defaultdict(float);boundary_count=0
    for decision in episode['decisions']:
        controls = decision['controls']
        if len(controls) != decision['executed_control_steps'] or decision['executed_physics_steps'] != 4 * len(controls):
            raise ValueError('控制步/物理四子步计数不一致')
        decision_reward = sum(float(c['reward']) for c in controls) + float(decision['zero_step_event_reward'])
        if not math.isclose(decision_reward, decision['reward_sum'], abs_tol=1e-8, rel_tol=1e-9):
            raise ValueError('上层转移的真实积分奖励不一致')
        reward += decision_reward
        controls_count += len(controls)
        physics += decision['executed_physics_steps']
        for control in controls:
            dt = float(control['dt_s'])
            if not math.isclose(dt, .02, abs_tol=1e-10):
                raise ValueError('控制周期不是50Hz')
            rate_sum = sum(float(c['integrated_reward']) for c in control['components'].values())
            if not math.isclose(rate_sum, float(control['continuous_reward']), abs_tol=1e-8, rel_tol=1e-9):
                raise ValueError('逐控制步奖励分项和不一致')
            if not math.isclose(rate_sum + float(control['event_reward']), float(control['reward']), abs_tol=1e-8, rel_tol=1e-9):
                raise ValueError('控制步奖励与事件罚分不一致')
            elapsed += dt
            activity = control['activity']
            actual += float(activity['actual_activity_rad_s']) * dt
            target += float(activity['target_activity_rad_s']) * dt
            gate += float(activity['gate']) * dt
            for name, value in control['components'].items():
                components[name] += float(value['integrated_reward'])
            if 'raw' in control['components'].get('joint_limit',{}):
                values=joint_boundary_sample(control['components']['joint_limit']);boundary_count+=1
                for key,value in values.items():
                    boundaries[key]=(max(boundaries[key],value) if key.endswith('_rad') else boundaries[key]+float(value))
    if elapsed <= 0 or not math.isclose(reward, episode['reward_sum'], abs_tol=1e-8, rel_tol=1e-9):
        raise ValueError('episode真实奖励/时长不一致')
    sample = task['sample']
    return dict(task_id=task['task_id'], dataset=task['dataset'], sample_id=task['sample_id'],
        seed=task['seed'], music_group=sample['group_id'], audio_sha=sample['row'].get('source_audio_sha256'),
        music_start_frame=task['music_start_frame'], noise_index_start=task['noise_index_start'],
        reward=reward, seconds=elapsed, reward_per_second=reward / elapsed,
        actual_activity_rad_s=actual / elapsed, target_activity_rad_s=target / elapsed,
        activity_ratio=actual / target if target > 0 else None, activity_gate=gate / elapsed,
        components=dict(components), physical_failure=bool(episode['physical_failure']),
        truncated=bool(episode['truncated']), end_reason=episode['end_reason'],
        control_steps=controls_count, physics_steps=physics,
        joint_boundaries=(dict(available=True,control_samples=boundary_count,
            **{key if key.endswith('_rad') else key.replace('_any','_control_fraction'):
                float(value) if key.endswith('_rad') else value/boundary_count for key,value in boundaries.items()})
            if boundary_count==controls_count else dict(available=False,control_samples=boundary_count)),
        motion_diagnostic=episode.get('motion_diagnostic'),
        generated_reference_diagnostic=episode.get('generated_reference_diagnostic'))


def completed_sessions(root):
    """只合并同一run已正常结束且累计计数严格衔接的会话，缺失摘要不能被忽略。"""
    directories = [p for p in (root / 'sessions').iterdir() if p.is_dir()]
    if not directories or any(not (p / 'summary.json').is_file() for p in directories):
        raise ValueError('存在未完整结束或缺少摘要的会话')
    sessions = [(p / 'summary.json', load_json(p / 'summary.json')) for p in directories]
    sessions.sort(key=lambda item: (item[1]['initial_state']['iteration'], item[1]['final_state']['iteration']))
    counters = ('iteration', 'rollouts_completed', 'actor_updates', 'critic_updates', 'policy_version',
        'optimizer_attempts', 'training_control_steps', 'training_physics_steps', 'training_upper_transitions')
    for index, (path, summary) in enumerate(sessions):
        if (summary['status'] != 'passed' or summary['exit_code'] != 0 or not summary['source_unchanged'] or
                summary['session_id'] != path.parent.name or summary['run_id'] != sessions[0][1]['run_id']):
            raise ValueError('会话身份、退出状态或源码冻结核验失败')
        if index:
            previous = sessions[index - 1][1]['final_state']
            if any(previous[k] != summary['initial_state'][k] for k in counters):
                raise ValueError('连续恢复的累计计数不衔接')
            if summary['final_state']['iteration'] <= summary['initial_state']['iteration']:
                raise ValueError('恢复会话没有新的接受轮次，不能重复拼接')
    return sessions


def export_run(root, *, physical_boundaries=False):
    root = Path(root).resolve(strict=True)
    sessions = completed_sessions(root)
    summary_path, summary = sessions[-1]
    initial = sessions[0][1]
    session = summary_path.parent
    identity = load_json(root / 'training_identity.json')
    latest = load_json(root / 'latest.json')
    checkpoint = checked_path(root, latest['path'], latest['sha256'])
    if checkpoint.stat().st_size != latest['size_bytes']:
        raise ValueError('最终checkpoint字节数不一致')
    if any(latest[key] != summary['checkpoint_shutdown'][key] for key in ('path', 'sha256', 'iteration')):
        raise ValueError('最新checkpoint与最后会话的结束发布不一致')
    result = dict(schema='genmo.stage10.pilot_audit_export.v1', run_root=str(root),
        run_id=summary['run_id'], summary_sha256=sha256(summary_path),
        summary_status=summary['status'], exit_code=summary['exit_code'],
        source_unchanged=summary['source_unchanged'],
        world_shutdown={k: v for k, v in summary['world_shutdown'].items() if k != 'lane_journals'},
        initial_state=scalars(initial['initial_state']), final_state=scalars(summary['final_state']),
        imported_critic=initial['initial_state'].get('critic_initialization', {}).get('artifact'),
        latest=latest, checkpoint_sha_verified=True, identity=identity,
        resolved_config=(session / 'resolved_config.yaml').read_text(),
        evaluations=[], training=[], source_files={},
        sessions=[dict(session_id=s['session_id'], summary_sha256=sha256(p), status=s['status'],
            exit_code=s['exit_code'], source_unchanged=s['source_unchanged'],
            initial_state=scalars(s['initial_state']), final_state=scalars(s['final_state']),
            world_shutdown=s['world_shutdown'], evaluation_world_shutdown=s.get('evaluation_world_shutdown'))
            for p, s in sessions])
    artifacts = [(s, artifact) for _, s in sessions for artifact in s['evaluation_artifacts']]
    for owner, artifact in artifacts:
        path = checked_path(root, artifact['path'], artifact['sha256'])
        data = load_json(path)
        tasks = {row['task_id']: row for row in data['plan']['tasks']}
        episodes = [episode_metrics(row, tasks[row['task_id']]) for row in data['episodes']]
        if physical_boundaries:
            from tools.compare_stage10_physical import load_physical
            values,evidence=load_physical(data,extractor=lambda row:joint_boundary_sample(
                row['reward_primitives']['components']['joint_limit']))
            for episode in episodes:
                rows=values[episode['task_id']];count=len(rows['ticks'])
                if count!=episode['control_steps']:raise ValueError('权威关节边界控制步缺失')
                episode['joint_boundaries']=dict(available=True,control_samples=count,
                    **{key if key.endswith('_rad') else key.replace('_any','_control_fraction'):
                        float(array.max() if key.endswith('_rad') else array.mean())
                        for key,array in rows.items() if key!='ticks'})
            result.setdefault('physical_journal_evidence',[]).append(dict(label=artifact['label'],journals=evidence))
        if len(episodes) != len(tasks) or len({r['task_id'] for r in episodes}) != len(tasks):
            raise ValueError('评估任务缺失或重复')
        measured_controls = (owner['final_state']['training_control_steps'] if artifact['label'].startswith('final_')
            else owner['initial_state']['training_control_steps'] if artifact['label'] == 'initial' else None)
        result['evaluations'].append(dict(label=artifact['label'], iteration=artifact['iteration'],
            session_id=owner['session_id'], training_control_steps=measured_controls,
            file=str(path), sha256=artifact['sha256'], plan_sha256=data['plan_sha256'],
            evaluation_identity=data['evaluation_identity'], actor_identity=data['actor_identity'],
            assertions=data['assertions'], aggregate=data['aggregate'],
            source_balanced_reward=data['source_balanced_reward'], rows=episodes,
            fixed_critic_diagnostic=data.get('fixed_critic_diagnostic'),
            fresh_critic_diagnostic=data.get('fresh_critic_diagnostic'),
            stage1_retention=data.get('stage1_retention'), wall_seconds=data['wall_seconds']))
    for _, owner in sessions:
        path = root / 'metrics' / (owner['session_id'] + '.jsonl')
        result['source_files'][str(path)] = sha256(path)
        for line in path.open():
            row = json.loads(line)
            if row.get('mode') not in ('dppo_bc', 'kl_rejected_rollback', 'critic_warmup'):
                continue
            selected = {key: row.get(key) for key in ('label', 'mode', 'wall_seconds', 'actor_seconds',
                'critic_seconds', 'full20_kl', 'overhead_since_previous_record', 'physics_paused_during_optimization')}
            selected['state'] = scalars(row['state'])
            selected['session_id'] = owner['session_id']
            actor = row.get('actor') or {}
            selected['actor'] = {key: value for key, value in actor.items() if key not in ('steps', 'epoch_orders')}
            selected['actor']['steps'] = [{key: value for key, value in step.items()
                if key not in ('global_chain_indices', 'per_denoising_step')}
                for step in actor.get('steps', [])]
            selected['collection_by_rank'] = row.get('collection_by_rank')
            selected['performance_by_rank'] = row.get('performance_by_rank')
            selected['memory_by_rank'] = row.get('memory_by_rank')
            critic=row.get('critic') or {}
            selected['critic']={key:critic.get(key) for key in ('before','after','optimizer_steps','metric_scope')}
            result['training'].append(selected)
    accepted = [r['state']['iteration'] for r in result['training'] if r['mode'] == 'dppo_bc']
    if accepted != list(range(initial['initial_state']['iteration'] + 1, summary['final_state']['iteration'] + 1)):
        raise ValueError('接受轮次存在缺失、重复或顺序错误')
    result['mc_calibrations']=[dict(file=str(path),sha256=sha256(path),report=load_json(path))
        for p, _ in sessions for path in sorted((p.parent/'mc_calibration').glob('*.json'))]
    return result


def paired_statistics(before, after, metric, *, draws=20000, seed=20261014):
    """按来源等权计算配对均值；同一音乐的多个片段和种子一起重采样。"""
    import numpy as np
    left = {row['task_id']: row for row in before}
    right = {row['task_id']: row for row in after}
    if set(left) != set(right) or len(left) != len(before) or len(right) != len(after):
        raise ValueError('配对任务缺失或重复')
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for key, row in left.items():
        other = right[key]
        for field in ('dataset', 'sample_id', 'seed', 'music_group', 'audio_sha',
                      'music_start_frame', 'noise_index_start'):
            if row[field] != other[field]:
                raise ValueError(f'配对身份不一致: {field}')
        if row[metric] is None or other[metric] is None:
            raise ValueError(f'配对指标不完整: {metric}')
        groups[row['dataset']][row['music_group']].append(other[metric] - row[metric])
    rng = np.random.default_rng(seed)
    estimates, source_means, source_counts = [], {}, {}
    for source, music in sorted(groups.items()):
        values = list(music.values())
        sums = np.array([sum(v) for v in values], dtype=float)
        counts = np.array([len(v) for v in values], dtype=float)
        source_means[source] = float(sums.sum() / counts.sum())
        source_counts[source] = len(values)
        indices = rng.integers(0, len(values), size=(draws, len(values)))
        estimates.append(sums[indices].sum(1) / counts[indices].sum(1))
    bootstrap = np.mean(estimates, axis=0)
    return dict(mean=statistics.mean(source_means.values()),
        ci90=[float(v) for v in np.quantile(bootstrap, [.05, .95])],
        ci95=[float(v) for v in np.quantile(bootstrap, [.025, .975])],
        by_source=source_means, music_groups_by_source=source_counts, tasks=len(left),
        scope='source_balanced_paired_delta_music_cluster_bootstrap_not_independent_training_seeds',
        draws=draws, seed=seed)


def describe(values):
    values = list(values)
    return dict(mean=statistics.mean(values), median=statistics.median(values), min=min(values), max=max(values))


def summarize_run(data):
    accepted = [row for row in data['training'] if row['mode'] == 'dppo_bc']
    if not accepted:
        raise ValueError('没有接受的PPO轮次')
    baseline, final = data['evaluations'][0], data['evaluations'][-1]
    sources = sorted({row['dataset'] for row in baseline['rows']})
    source_changes = {}
    for source in sources:
        old = [row for row in baseline['rows'] if row['dataset'] == source]
        new = [row for row in final['rows'] if row['dataset'] == source]
        source_changes[source] = {}
        for metric in ('reward', 'reward_per_second', 'actual_activity_rad_s', 'activity_ratio', 'activity_gate'):
            a = statistics.mean(row[metric] for row in old)
            b = statistics.mean(row[metric] for row in new)
            source_changes[source][metric] = dict(initial=a, final=b, delta=b-a,
                relative_percent=100*(b/a-1) if a else None)
    for row in accepted:
        kl = row['full20_kl']
        if (kl['covered_chains'] <= 0 or kl['covered_internal_transitions'] != 20 * kl['covered_chains'] or
                [s['step'] for s in kl['per_denoising_step']] != list(range(20))):
            raise ValueError('最终KL没有覆盖完整20步')
    result = dict(run_id=data['run_id'], status=data['summary_status'], exit_code=data['exit_code'],
        source_unchanged=data['source_unchanged'], checkpoint_sha_verified=data['checkpoint_sha_verified'],
        accepted_iterations=len(accepted), rejected_rollouts=sum(row['mode']=='kl_rejected_rollback' for row in data['training']),
        final_state=data['final_state'], initial_state=data['initial_state'],
        physical_failures_final=sum(row['physical_failure'] for row in final['rows']),
        end_reasons_final=dict(collections.Counter(row['end_reason'] for row in final['rows'])),
        full20_coverage_passed=True, source_changes=source_changes,
        actor_steps=sum(row['actor']['optimizer_steps'] for row in accepted),
        early_stop_rounds=sum(row['actor'].get('early_stop_reason') is not None for row in accepted),
        lr_first_used=accepted[0]['actor']['steps'][0]['learning_rate'],
        lr_last_used=accepted[-1]['actor']['steps'][-1]['learning_rate'],
        lr_next=data['final_state']['selected_actor_lr'])
    getters = dict(iteration_wall=lambda r:r['wall_seconds'], actor=lambda r:r['actor_seconds'],
        critic=lambda r:r['critic_seconds'], final_kl=lambda r:r['full20_kl']['wall_seconds'],
        collection=lambda r:max(x['through_targets_and_budget_seconds'] for x in r['collection_by_rank']),
        mean_coordinate_kl=lambda r:r['full20_kl']['mean_coordinate_kl'],
        maximum_step_kl=lambda r:r['full20_kl']['maximum_step_mean_coordinate_kl'],
        objective_clip_fraction=lambda r:r['full20_kl']['objective_clip_fraction'])
    result['training_metrics'] = {key: describe(fn(row) for row in accepted) for key,fn in getters.items()}
    # 正式训练关闭重型梯度信噪比诊断；缺测不能填0，也不能阻止已有物理证据导出。
    cosines=[(row['actor'].get('gradient_snr') or {}).get('half_split_cosine_pooled') for row in accepted]
    cosines=[value for value in cosines if value is not None]
    result['training_metrics']['half_gradient_cosine']=describe(cosines) if cosines else None
    result['gradient_snr_observed_iterations']=len(cosines)
    result['paired_to_initial'] = {metric:paired_statistics(baseline['rows'], final['rows'], metric)
        for metric in ('reward', 'reward_per_second', 'actual_activity_rad_s', 'activity_ratio')}
    result['kl_first16_mean'] = describe(statistics.mean(step['mean_coordinate_kl']['mean']
        for step in row['full20_kl']['per_denoising_step'][:16]) for row in accepted)
    result['kl_last4_mean'] = describe(statistics.mean(step['mean_coordinate_kl']['mean']
        for step in row['full20_kl']['per_denoising_step'][16:]) for row in accepted)
    weighted_joint = [sum(row['full20_kl']['covered_chains'] *
        row['full20_kl']['per_denoising_step'][step]['joint_kl']['mean']
        for row in accepted) for step in range(20)]
    result['first16_joint_kl_share'] = (sum(weighted_joint[:16]) / sum(weighted_joint)
        if sum(weighted_joint) > 0 else None)
    return result


def evaluation_control_steps(data, evaluation):
    """用实际交互量标注评估曲线，不能用轮数乘名义102400控制步替代。

    周期标签在实际评估前固定了交互计数；final使用终态（包括末次MC），initial
    使用已导入预热的初态。其他标签只允许唯一的已记录接受轮对应，避免静默估算。
    """
    label=evaluation['label']
    if evaluation.get('training_control_steps') is not None:
        return int(evaluation['training_control_steps'])
    match=re.fullmatch(r'periodic_r\d+_c(\d+)_a\d+',label)
    if match:return int(match[1])
    if label=='initial':return int(data['initial_state']['training_control_steps'])
    if label.startswith('final_'):return int(data['final_state']['training_control_steps'])
    values={row['state']['training_control_steps'] for row in data['training']
        if row['mode']=='dppo_bc' and row['state']['iteration']==evaluation['iteration']}
    if len(values)!=1:raise ValueError(f'评估缺少唯一的实际交互计数: {label}')
    return int(next(iter(values)))


def add_launcher_timing(comparison, evidence):
    """从唯一的正常结束记录读取全程墙钟；拒绝重复会话或失败退出。"""
    if len(evidence) != len(comparison['runs']):
        raise ValueError('启动日志数量与pilot数量不一致')
    for name, source in zip(comparison['runs'], evidence):
        if 'wall_seconds' in source:
            if source.get('run_id')!=name or source.get('exit_code')!=0:
                raise ValueError(f'pilot启动日志身份或退出状态不匹配: {name}')
            total=float(source['wall_seconds'])
            if not math.isfinite(total) or total<=0:raise ValueError('启动墙钟必须为正的有限数')
        else:
            matches = re.findall(r'\] end (\S+) exit=(\d+) wall=(\d+)s', source['pilot_progress'])
            if len(matches) != 1 or matches[0][0] != name or matches[0][1] != '0':
                raise ValueError(f'pilot启动日志身份、会话数或退出状态不匹配: {name}')
            total = int(matches[0][2])
        result = comparison['runs'][name]
        result['end_to_end_seconds'] = total
        result['amortized_total_seconds_per_accepted_iteration'] = total / result['accepted_iterations']
        result['end_to_end_scope'] = 'launcher_start_to_exit_excludes_previously_imported_mc_preheat'


def compare_runs(first, second):
    """策略学习率可以不同；评估任务、物理和采样合同必须一致。"""
    one, two = first['evaluations'][0], second['evaluations'][0]
    for data in (first, second):
        if data['summary_status'] != 'passed' or data['exit_code'] != 0 or not data['source_unchanged']:
            raise ValueError('不能把未完整退出的pilot当成正式对照')
        for evaluation in data['evaluations']:
            if evaluation['plan_sha256'] != one['plan_sha256']:
                raise ValueError('固定评估计划发生变化')
            for key in ('kernel', 'timing_contract', 'execution_profile', 'critical_latency_budgets', 'episode_seconds', 'physics'):
                if evaluation['evaluation_identity'][key] != one['evaluation_identity'][key]:
                    raise ValueError(f'物理评估合同不一致: {key}')
    initial_pair = paired_statistics(one['rows'], two['rows'], 'reward')
    return dict(schema='genmo.stage10.pilot_comparison.v1',
        scope='one_training_seed_per_setting_source_balanced_music_cluster_paired_evaluation',
        evaluation_tasks=len(one['rows']),
        independent_music_groups=len({(r['dataset'],r['music_group']) for r in one['rows']}),
        initial_pair=initial_pair, runs={d['run_id']:summarize_run(d) for d in (first,second)},
        second_minus_first={metric:paired_statistics(first['evaluations'][-1]['rows'],
            second['evaluations'][-1]['rows'], metric)
            for metric in ('reward', 'reward_per_second', 'actual_activity_rad_s', 'activity_ratio')},
        decision='No automatic promotion; inspect independent reward gains, source regressions and missing plan work.')


def summarize_complete_evaluation(evaluation):
    """完整episode统计保留早失败，不以执行更短的高每秒分数代替完成率。"""
    rows=evaluation['rows']
    if not rows or any(row['truncated'] or (row['end_reason']!='music_end' and not row['physical_failure']) for row in rows):
        raise ValueError('完整歌曲对照不接受行政截断或基础设施异常')
    if any(row['end_reason']=='music_end' and row['physical_failure'] for row in rows):
        raise ValueError('真实失败标志和终止原因不一致')
    if any(not row.get('joint_boundaries',{}).get('available') for row in rows):
        raise ValueError('完整歌曲对照缺少权威实际/参考关节边界')
    def metrics(part):
        result={key:statistics.mean(row[key] for row in part) for key in
            ('reward','reward_per_second','seconds','actual_activity_rad_s','activity_ratio')}
        result.update(tasks=len(part),completed=sum(row['end_reason']=='music_end' for row in part),
            physical_failures=sum(row['physical_failure'] for row in part),
            control_steps=sum(row['control_steps'] for row in part))
        result['completion_rate']=result['completed']/result['tasks']
        for label in ('actual','reference'):
            for level in ('soft','hard'):
                key=f'{label}_{level}_control_fraction'
                result[key]=sum(row['joint_boundaries'][key]*row['control_steps'] for row in part)/result['control_steps']
            key=label+'_max_hard_excess_rad'
            result[key]=max(row['joint_boundaries'][key] for row in part)
        return result
    sources={source:metrics([row for row in rows if row['dataset']==source])
        for source in sorted({row['dataset'] for row in rows})}
    overall=metrics(rows)
    balanced={key:statistics.mean(values[key] for values in sources.values()) for key in
        ('reward','reward_per_second','seconds','actual_activity_rad_s','activity_ratio','completion_rate',
         'actual_hard_control_fraction','reference_hard_control_fraction')}
    return dict(overall=overall,by_source=sources,source_balanced=balanced,
        end_reasons=dict(collections.Counter(row['end_reason'] for row in rows)),
        independent_music_groups=len({(row['dataset'],row['music_group']) for row in rows}),
        boundary_scope='fraction_of_actual_controls_with_any_joint_beyond_limit_not_fraction_of_joints')


def compare_models(stage1,first,second):
    """三个实际权重在同一完整歌曲计划上的独立复核，不进行参数搜索或自动晋升。"""
    values={'Stage1':stage1,'P1':first,'P2':second}
    evaluations={name:data['evaluations'][-1] for name,data in values.items()}
    baseline=evaluations['Stage1'];reference_identity=baseline['evaluation_identity']
    for name,data in values.items():
        if data['summary_status']!='passed' or data['exit_code']!=0 or not data['source_unchanged'] or not data['checkpoint_sha_verified']:
            raise ValueError(f'完整歌曲会话没有正确结束或权重未核验: {name}')
        evaluation=evaluations[name]
        if evaluation['plan_sha256']!=baseline['plan_sha256']:
            raise ValueError('完整歌曲任务计划不一致')
        for key in ('kernel','timing_contract','execution_profile','critical_latency_budgets','episode_seconds','physics'):
            if evaluation['evaluation_identity'][key]!=reference_identity[key]:
                raise ValueError(f'完整歌曲物理或行为合同不一致: {key}')
        for key in ('assets','source_files','physical_configuration','reward','generation_horizon'):
            if data['identity'][key]!=stage1['identity'][key]:
                raise ValueError(f'完整歌曲执行身份不一致: {key}')
    summaries={name:summarize_complete_evaluation(e) for name,e in evaluations.items()}
    metrics=('reward','reward_per_second','seconds','actual_activity_rad_s','activity_ratio','physical_failure')
    pairs={name:{metric:paired_statistics(baseline['rows'],evaluation['rows'],metric)
        for metric in metrics} for name,evaluation in evaluations.items() if name!='Stage1'}
    return dict(schema='genmo.stage10.complete_model_comparison.v1',models=summaries,
        paired_to_stage1=pairs,second_minus_first={metric:paired_statistics(
            evaluations['P1']['rows'],evaluations['P2']['rows'],metric) for metric in metrics},
        evaluation_plan_sha256=baseline['plan_sha256'],
        actors={name:evaluation['actor_identity'] for name,evaluation in evaluations.items()},
        scope='complete_available_paired_music_samples_test_only_not_unseen_Stage1_pretraining_certificate')


def plot_model_comparison(comparison,output):
    """实际失败、奖励、动作和边界同时展示，不能只挑总回报画图。"""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt,font_manager
    font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    names=list(comparison['models']);colors=['#607d8b','#2875b9','#db7a22']
    fig,axes=plt.subplots(2,3,figsize=(16,9),layout='constrained')
    models=comparison['models'];sources=list(models['Stage1']['by_source']);x=np.arange(len(sources))
    for ax,key,title,unit in [
        (axes[0,0],'completion_rate','完整可用歌曲完成率','%'),
        (axes[0,1],'reward_per_second','来源等权：每秒物理回报','回报 / 秒'),
        (axes[0,2],'actual_max_hard_excess_rad','实际机器人：最大硬边界超量','弧度')]:
        vals=[models[name]['overall' if key!='reward_per_second' else 'source_balanced'][key] for name in names]
        if key=='completion_rate':vals=[100*v for v in vals]
        bars=ax.bar(names,vals,color=colors)
        labels=[f"{models[name]['overall']['completed']}/{models[name]['overall']['tasks']}" for name in names] if key=='completion_rate' else [f'{v:.5f}' for v in vals]
        ax.bar_label(bars,labels=labels,padding=3,fontsize=9);ax.set(title=title,ylabel=unit);ax.margins(y=.2)
    axes[0,0].set_ylim(0,110)
    for i,name in enumerate(names):
        by=models[name]['by_source'];initial=models['Stage1']['by_source']
        axes[1,0].bar(x+(i-1)*.25,[100*(by[s]['reward_per_second']/initial[s]['reward_per_second']-1) for s in sources],.25,label=name,color=colors[i])
        axes[1,1].bar(x+(i-1)*.25,[100*(by[s]['actual_activity_rad_s']/initial[s]['actual_activity_rad_s']-1) for s in sources],.25,label=name,color=colors[i])
        axes[1,2].bar(x+(i-1)*.25,[100*by[s]['actual_hard_control_fraction'] for s in sources],.25,label=name,color=colors[i])
    for ax,title,unit in [(axes[1,0],'各来源每秒回报相对Stage1变化','%'),
                         (axes[1,1],'各来源实际活动度相对Stage1变化','%'),
                         (axes[1,2],'实际机器人：触硬边界控制步比例','%')]:
        ax.set(title=title,ylabel=unit);ax.set_xticks(x,sources,rotation=15);ax.axhline(0,color='black',lw=.7);ax.legend(fontsize=8)
    for ax in axes.flat:ax.grid(axis='y',alpha=.2)
    count=models['Stage1']['overall']['tasks'];groups=models['Stage1']['independent_music_groups']
    title=comparison.get('plot_title','Stage1 / 固定学习率P1 / 自适应学习率P2：同条件完整物理测试')
    fig.suptitle(f'{title}\n{groups}个音乐组、{count}个任务；真实失败全部保留；参考边界另见JSON',fontsize=14)
    fig.savefig(output/'complete_song_comparison.png',dpi=160);fig.savefig(output/'complete_song_comparison.pdf');plt.close(fig)


def plot_comparison(first, second, comparison, output):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt, font_manager
    font = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2,2,figsize=(13,8.5),layout='constrained')
    colors=['#2875b9','#db7a22']
    for index,data in enumerate((first,second)):
        label=data['run_id'];color=colors[index]
        evaluations=data['evaluations'];baseline=evaluations[0]['source_balanced_reward']
        axes[0,0].plot([evaluation_control_steps(data,e)/1e6 for e in evaluations],
            [e['source_balanced_reward']-baseline for e in evaluations],'-o',label=label,color=color)
        rows=[r for r in data['training'] if r['mode']=='dppo_bc']
        iterations=[r['state']['iteration'] for r in rows]
        axes[0,1].plot(iterations,[r['actor']['steps'][0]['learning_rate']/1e-7 for r in rows],label=label,color=color)
        axes[1,0].plot(iterations,[r['full20_kl']['maximum_step_mean_coordinate_kl'] for r in rows],label=label,color=color)
        rejected=[r for r in data['training'] if r['mode']=='kl_rejected_rollback']
        if rejected:
            axes[1,0].scatter([r['state']['iteration'] for r in rejected],
                [r['full20_kl']['maximum_step_mean_coordinate_kl'] for r in rejected],
                marker='x',s=65,color=color,zorder=5,label=f'{label}拒绝尝试（不增加轮数）')
        changes=comparison['runs'][label]['source_changes'];sources=sorted(changes)
        axes[1,1].bar(np.arange(len(sources))+(index-.5)*.36,
            [changes[s]['actual_activity_rad_s']['relative_percent'] for s in sources],
            width=.36,label=label,color=color)
    axes[0,0].axhline(0,color='black',lw=.8);axes[0,0].set(title='固定评估：相对各自初始Actor的回报变化',xlabel='实际训练控制步 / 百万（含MC预热与校准）',ylabel='积分回报差（越高越好）')
    axes[0,1].set(title='实际用于更新的Actor学习率',xlabel='接受的PPO轮数',ylabel='学习率 / 1e-7')
    target=second['identity']['training']['kl_target_coordinate']
    limit=second['identity']['kl']['maximum_step_mean_coordinate']
    axes[1,0].axhline(target,color='gray',ls='--',label=f'P2目标{target:g}')
    axes[1,0].axhline(limit,color='red',ls=':',label=f'硬上限{limit:g}')
    axes[1,0].set(title='完整20步检查中的最大单步坐标KL',xlabel='接受的PPO轮数',ylabel='KL')
    axes[1,1].axhline(0,color='black',lw=.8)
    axes[1,1].set_xticks(np.arange(len(sources)),sources,rotation=12)
    axes[1,1].set(title='最终评估的实际活动度变化',ylabel='相对初始值变化（%）')
    for ax in axes.flat:ax.legend(fontsize=9);ax.grid(axis='y',alpha=.2)
    counts=[comparison['runs'][d['run_id']]['accepted_iterations'] for d in (first,second)]
    fig.suptitle(f'固定学习率 vs 自适应学习率｜接受轮数 {counts[0]} / {counts[1]}\n'
        f'{comparison["evaluation_tasks"]}个固定任务，{comparison["independent_music_groups"]}个音乐组；短评估不能代表完整歌曲',fontsize=13)
    fig.savefig(output/'pilot_comparison.png',dpi=160)
    fig.savefig(output/'pilot_comparison.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--export-run', type=Path)
    mode.add_argument('--compare', nargs=2, type=Path, metavar=('P1_JSON','P2_JSON'))
    mode.add_argument('--compare-models',nargs=3,type=Path,metavar=('STAGE1_JSON','P1_JSON','P2_JSON'))
    parser.add_argument('--run-evidence', nargs=2, type=Path, metavar=('P1_EVIDENCE','P2_EVIDENCE'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--physical-boundaries',action='store_true',help='只读核验完整权威journal，提取实际与参考软/硬关节边界')
    args = parser.parse_args()
    if args.export_run:
        if args.run_evidence:parser.error('--run-evidence只适用于--compare')
        json.dump(export_run(args.export_run,physical_boundaries=args.physical_boundaries), sys.stdout, ensure_ascii=False, allow_nan=False)
        sys.stdout.write('\n')
    else:
        if args.physical_boundaries:parser.error('--physical-boundaries只适用于--export-run')
        if args.output is None:parser.error('--compare需要--output')
        args.output.mkdir(parents=True,exist_ok=True)
        if args.compare_models:
            if args.run_evidence:parser.error('--run-evidence只适用于训练pilot的--compare')
            target=args.output/'complete_song_comparison.json'
            if target.exists():raise FileExistsError(target)
            result=compare_models(*map(load_json,args.compare_models))
            result['input_exports']=[dict(path=str(p.resolve()),sha256=sha256(p)) for p in args.compare_models]
            plot_model_comparison(result,args.output)
            target.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print(target)
            return
        target=args.output/'comparison.json'
        if target.exists():raise FileExistsError(target)
        first,second=map(load_json,args.compare)
        result=compare_runs(first,second)
        result['input_exports']=[dict(path=str(p.resolve()),sha256=sha256(p)) for p in args.compare]
        if args.run_evidence:
            add_launcher_timing(result, list(map(load_json,args.run_evidence)))
            result['input_run_evidence']=[dict(path=str(p.resolve()),sha256=sha256(p)) for p in args.run_evidence]
        plot_comparison(first,second,result,args.output)
        target.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        print(target)


if __name__ == '__main__':
    main()
