#!/usr/bin/env python3
"""只读导出和对照Stage10已完成pilot的真实证据。

--export-run可经SSH标准输入在服务器执行，只读取run目录：核验评估JSON与最终
checkpoint的SHA，逐控制步复算积分和四子步计数，提取活动度、奖励分项、末端K步
策略和完整20步KL。它不导入Isaac/Actor、不占GPU、不修改训练源码或原始产物。
--compare在本地按task身份配对，使用来源内音乐组成簇的bootstrap区间，
不把同曲不同种子视为独立音乐；分别报告普通训练轮耗时和含评估/启动的总耗时。
--run-evidence可传入两台服务器的只读证据JSON，使用其中已结束的启动日志核对总耗时；
没有启动日志时不以普通轮耗时之和冒充全程耗时。输入文件SHA随比较报告记录。
输出只用于实施进度与学习效果审计，不自动宣称G5通过或启动正式训练。
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


def episode_metrics(episode, task):
    """真实50Hz奖励已经积分，只求和；活动度用实际dt加权。"""
    for key in ('task_id', 'dataset', 'seed', 'music_start_frame', 'noise_index_start'):
        if episode[key] != task[key]:
            raise ValueError(f'任务身份不匹配: {key}')
    elapsed = reward = actual = target = gate = 0.
    controls_count = physics = 0
    components = collections.defaultdict(float)
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
        motion_diagnostic=episode.get('motion_diagnostic'),
        generated_reference_diagnostic=episode.get('generated_reference_diagnostic'))


def export_run(root):
    root = Path(root).resolve(strict=True)
    sessions = sorted((root / 'sessions').glob('*/summary.json'))
    if len(sessions) != 1:
        raise ValueError('本工具要求一个完整pilot会话，不能静默拼接多次恢复')
    summary = load_json(sessions[0])
    session = sessions[0].parent
    identity = load_json(root / 'training_identity.json')
    latest = load_json(root / 'latest.json')
    checkpoint = checked_path(root, latest['path'], latest['sha256'])
    if checkpoint.stat().st_size != latest['size_bytes']:
        raise ValueError('最终checkpoint字节数不一致')
    result = dict(schema='genmo.stage10.pilot_audit_export.v1', run_root=str(root),
        run_id=summary['run_id'], summary_sha256=sha256(sessions[0]),
        summary_status=summary['status'], exit_code=summary['exit_code'],
        source_unchanged=summary['source_unchanged'],
        world_shutdown={k: v for k, v in summary['world_shutdown'].items() if k != 'lane_journals'},
        initial_state=scalars(summary['initial_state']), final_state=scalars(summary['final_state']),
        imported_critic=summary['initial_state'].get('critic_initialization', {}).get('artifact'),
        latest=latest, checkpoint_sha_verified=True, identity=identity,
        resolved_config=(session / 'resolved_config.yaml').read_text(),
        evaluations=[], training=[], source_files={})
    for artifact in summary['evaluation_artifacts']:
        path = checked_path(root, artifact['path'], artifact['sha256'])
        data = load_json(path)
        tasks = {row['task_id']: row for row in data['plan']['tasks']}
        episodes = [episode_metrics(row, tasks[row['task_id']]) for row in data['episodes']]
        if len(episodes) != len(tasks) or len({r['task_id'] for r in episodes}) != len(tasks):
            raise ValueError('评估任务缺失或重复')
        result['evaluations'].append(dict(label=artifact['label'], iteration=artifact['iteration'],
            file=str(path), sha256=artifact['sha256'], plan_sha256=data['plan_sha256'],
            evaluation_identity=data['evaluation_identity'], actor_identity=data['actor_identity'],
            assertions=data['assertions'], aggregate=data['aggregate'],
            source_balanced_reward=data['source_balanced_reward'], rows=episodes,
            fixed_critic_diagnostic=data.get('fixed_critic_diagnostic'),
            fresh_critic_diagnostic=data.get('fresh_critic_diagnostic'),
            stage1_retention=data.get('stage1_retention'), wall_seconds=data['wall_seconds']))
    for path in sorted((root / 'metrics').glob('*.jsonl')):
        result['source_files'][str(path)] = sha256(path)
        for line in path.open():
            row = json.loads(line)
            if row.get('mode') not in ('dppo_bc', 'kl_rejected_rollback', 'critic_warmup'):
                continue
            selected = {key: row.get(key) for key in ('label', 'mode', 'wall_seconds', 'actor_seconds',
                'critic_seconds', 'full20_kl', 'overhead_since_previous_record', 'physics_paused_during_optimization')}
            selected['state'] = scalars(row['state'])
            actor = row.get('actor') or {}
            selected['actor'] = {key: value for key, value in actor.items() if key not in ('steps', 'epoch_orders')}
            selected['actor']['steps'] = [{key: value for key, value in step.items()
                if key not in ('global_chain_indices', 'per_denoising_step')}
                for step in actor.get('steps', [])]
            selected['collection_by_rank'] = row.get('collection_by_rank')
            selected['performance_by_rank'] = row.get('performance_by_rank')
            result['training'].append(selected)
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
        if kl['covered_internal_transitions'] != 20 * kl['covered_chains']:
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
        objective_clip_fraction=lambda r:r['full20_kl']['objective_clip_fraction'],
        half_gradient_cosine=lambda r:r['actor']['gradient_snr']['half_split_cosine_pooled'])
    result['training_metrics'] = {key: describe(fn(row) for row in accepted) for key,fn in getters.items()}
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


def add_launcher_timing(comparison, evidence):
    """从唯一的正常结束记录读取全程墙钟；拒绝重复会话或失败退出。"""
    if len(evidence) != len(comparison['runs']):
        raise ValueError('启动日志数量与pilot数量不一致')
    for name, source in zip(comparison['runs'], evidence):
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
        scope='one_training_seed_per_setting_fixed_32_short_validation_tasks_12_music_groups',
        initial_pair=initial_pair, runs={d['run_id']:summarize_run(d) for d in (first,second)},
        second_minus_first={metric:paired_statistics(first['evaluations'][-1]['rows'],
            second['evaluations'][-1]['rows'], metric)
            for metric in ('reward', 'reward_per_second', 'actual_activity_rad_s', 'activity_ratio')},
        decision='No automatic promotion; inspect independent reward gains, source regressions and missing plan work.')


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
        axes[0,0].plot([e['iteration'] for e in evaluations],
            [e['source_balanced_reward']-baseline for e in evaluations],'-o',label=label,color=color)
        rows=[r for r in data['training'] if r['mode']=='dppo_bc']
        iterations=[r['state']['iteration'] for r in rows]
        axes[0,1].plot(iterations,[r['actor']['steps'][0]['learning_rate']/1e-7 for r in rows],label=label,color=color)
        axes[1,0].plot(iterations,[r['full20_kl']['maximum_step_mean_coordinate_kl'] for r in rows],label=label,color=color)
        changes=comparison['runs'][label]['source_changes'];sources=sorted(changes)
        axes[1,1].bar(np.arange(len(sources))+(index-.5)*.36,
            [changes[s]['actual_activity_rad_s']['relative_percent'] for s in sources],
            width=.36,label=label,color=color)
    axes[0,0].axhline(0,color='black',lw=.8);axes[0,0].set(title='固定评估：相对各自初始Actor的回报变化',xlabel='接受的PPO轮数',ylabel='积分回报差（越高越好）')
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
    fig.suptitle('P1固定学习率 vs P2自适应学习率｜各40轮、同一初始Actor和MC Critic\n32个9.6–10秒任务，12个音乐组；短评估不能代表完整歌曲',fontsize=13)
    fig.savefig(output/'pilot_comparison.png',dpi=160)
    fig.savefig(output/'pilot_comparison.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--export-run', type=Path)
    mode.add_argument('--compare', nargs=2, type=Path, metavar=('P1_JSON','P2_JSON'))
    parser.add_argument('--run-evidence', nargs=2, type=Path, metavar=('P1_EVIDENCE','P2_EVIDENCE'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.export_run:
        if args.run_evidence:parser.error('--run-evidence只适用于--compare')
        json.dump(export_run(args.export_run), sys.stdout, ensure_ascii=False, allow_nan=False)
        sys.stdout.write('\n')
    else:
        if args.output is None:parser.error('--compare需要--output')
        args.output.mkdir(parents=True,exist_ok=True)
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
