"""只读汇总2026-10-28旧模型配对物理评价与修复后八卡真实训练状态。

输入是修复任务根目录下已存在的旧模型zero/repeat/Adapter评价、正式launch manifest、
status、training.jsonl及固定评价JSON，不加载网络、不调用仿真或优化器。旧GPU7选择
独立parallel评价，其余选择v2；逐歌核验初态/xi/预算身份并计算方向统一的改善量与
单次重复绝对差，不能把该比例称统计显著性。训练数据量仅使用real_by_role.train，
与2048场景评价时的总物理推进和实际测量评价转移分开。输出必须使用全新报告目录，
不修改或覆盖任何输入模型、配置、日志；报告中的缺失与不完整状态保持显式。
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time


METRICS = {
    'full_horizon_tracking': 1, 'root_position_rmse_m': -1, 'joint_q_rmse_rad': -1,
    'joint_qd_rmse_rad_s': -1, 'body_orientation_rmse_rad': -1,
    'body_orientation_absolute_mean_rad': -1, 'jerk_rms': -1,
    'contact_foot_slip_mps': -1, 'music_f1': 1,
    **{name: 1 for name in ('body_pos', 'body_ori', 'body_linvel', 'body_angvel', 'root_pos', 'root_ori')},
}


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def metric(row, name):
    if name == 'music_f1':
        return row.get('music', {}).get('f1')
    return row.get(name, row.get('sm_terms', {}).get(name))


def compare(zero, repeat, adapter):
    maps = [{row['task']: row for row in report['episodes']} for report in (zero, repeat, adapter)]
    if any(set(value) != set(maps[0]) for value in maps[1:]):
        raise ValueError('zero/repeat/Adapter歌曲集合不一致')
    cases, aggregates = [], {name: [] for name in METRICS}
    for task, baseline in maps[0].items():
        repeated, current = maps[1][task], maps[2][task]
        for key in ('initial_state_sha256', 'physical_xi_sha256', 'evaluation_budget_steps'):
            if current[key] != baseline[key] or repeated[key] != baseline[key]:
                raise ValueError(f'配对身份不一致：{task} {key}')
        values = {}
        for name, direction in METRICS.items():
            a, b, c = (metric(row, name) for row in (baseline, repeated, current))
            if any(value is None for value in (a, b, c)):
                values[name] = None
                continue
            gain, noise = direction*(c-a), abs(b-a)
            values[name] = dict(zero=a, repeat=b, adapter=c, change=c-a, improvement=gain,
                repeat_abs=noise, gain_over_repeat=gain/noise if noise > 1e-12 else None)
            aggregates[name].append(values[name])
        cases.append(dict(task=task, zero_reason=baseline['reason'], adapter_reason=current['reason'],
            zero_steps=baseline['steps'], adapter_steps=current['steps'],
            initial_state_sha256=baseline['initial_state_sha256'], physical_xi_sha256=baseline['physical_xi_sha256'],
            evaluation_budget_steps=baseline['evaluation_budget_steps'], metrics=values))
    summary = {}
    for name, rows in aggregates.items():
        if not rows:
            summary[name] = None
            continue
        gain = statistics.mean(row['improvement'] for row in rows)
        noise = statistics.mean(row['repeat_abs'] for row in rows)
        summary[name] = dict(zero=statistics.mean(row['zero'] for row in rows),
            adapter=statistics.mean(row['adapter'] for row in rows), mean_improvement=gain,
            mean_absolute_repeat=noise, gain_over_mean_absolute_repeat=gain/noise if noise > 1e-12 else None,
            improved_songs=sum(row['improvement'] > 0 for row in rows), measured_songs=len(rows))
    return dict(cases=cases, metrics=summary, zero_completion_rate=zero['completion_rate'],
        adapter_completion_rate=adapter['completion_rate'],
        constraint_failures=adapter.get('constraint_failure_count', 0),
        noise_interpretation='one_repeat_not_a_confidence_interval; aggregate denominator is mean per-song absolute repeat difference')


def old_evaluations(root):
    groups, identities, manifests = [], {}, set()
    for gpu in range(8):
        source = root/('old_checkpoint_evaluation_parallel/model_gpu7' if gpu == 7
                       else f'old_checkpoint_evaluation_v2/model_gpu{gpu}')
        result = dict(gpu=gpu, source=str(source), completed=(source/'completed.json').exists(), splits={})
        manifest = read(source/'manifest.json')
        if manifest:
            result.update(checkpoint=manifest['checkpoint'], adapter_updates=manifest['checkpoint_updates'])
            manifests.add(manifest['config']['tasks'])
        for split in ('validation', 'evaluation'):
            reports = [read(source/f'{split}_{name}.json') for name in ('zero', 'repeat', 'adapter')]
            for name, report in zip(('zero', 'repeat', 'adapter'), reports):
                if report is None:
                    continue
                for row in report['episodes']:
                    signature = (row['initial_state_sha256'], row['physical_xi_sha256'], row['evaluation_budget_steps'])
                    identities.setdefault(row['task'], {}).setdefault(signature, []).append([gpu, name])
            if all(report is not None for report in reports) and (source/f'{split}_paired.json').exists():
                result['splits'][split] = compare(*reports)
        groups.append(result)
    identity_report = {task: dict(signature_count=len(values), consistent=len(values) == 1,
        signatures=[dict(initial_state_sha256=key[0], physical_xi_sha256=key[1], budget=key[2], observations=observations)
                    for key, observations in values.items()]) for task, values in identities.items()}
    assets = []
    for value in sorted(manifests):
        path = Path(value)
        if not path.exists():
            assets.append(dict(tasks_manifest=value, available_on_reporting_host=False))
            continue
        entry = dict(tasks_manifest=value, tasks_sha256=digest(path), nominal=[])
        tasks = read(path)
        for task in tasks:
            if task['split'] not in ('validation', 'evaluation'):
                continue
            cache = Path(task['nominal_reference'])
            entry['nominal'].append(dict(task=task['id'], path=str(cache),
                sha256=digest(cache) if cache.exists() else None, generation_contract=task.get('generation_contract')))
        assets.append(entry)
    return dict(groups=groups, cross_group_identity=identity_report, assets=assets,
                complete_groups=sum(group['completed'] for group in groups))


def training_rows(path):
    rows = []
    if path.exists():
        lines = path.read_text().splitlines()
        for index, line in enumerate(lines):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # 仅允许读取时正在写的最后半行，不将它伪装为完整更新。
                if index != len(lines)-1:
                    raise
    return rows


def summarize_evaluation(report, path, split):
    """保留实际评价合同和逐歌配对原值，不将256步前缀当作整歌效果。"""
    episodes = report.get('episodes', [])
    means = {}
    for name in METRICS:
        values = [metric(row, name) for row in episodes]
        valid = [value for value in values if value is not None]
        means[name] = dict(mean=statistics.mean(valid) if valid else None,
                          measured_songs=len(valid), total_songs=len(episodes))
    return dict(path=str(path), split=split,
        updates=report.get('adapter_updates'), zero=report.get('zero_adapter'),
        full_song=report.get('full_song_evaluation'), evaluation_scope=report.get('evaluation_scope'),
        max_steps=report.get('max_steps'), measured_transitions=report.get('measured_transitions'),
        total_scene_physics_transitions=report.get('physics_transitions'),
        measured_environments=report.get('measured_evaluation_envs'), scene_environments=report.get('num_envs'),
        completion_rate=report.get('completion_rate'), tracking=report.get('mean_full_horizon_tracking'),
        constraint_failure_count=report.get('constraint_failure_count'),
        constraint_failures=report.get('constraint_failures'),
        metric_means=means, episodes=episodes,
        paired_difference=report.get('paired_zero_difference'), paired_zero=report.get('paired_zero'),
        zero_baseline_key=report.get('zero_baseline_key'), wall_seconds=report.get('wall_seconds'),
        evidence=report.get('evidence'))


def formal_status(root):
    launch = read(root/'launch_manifest.json')
    if launch is None:
        return dict(root=str(root), launched=False, groups=[])
    groups = []
    for item in launch['launches']:
        output = Path(item['output'])
        status = read(output/'status.json') or {}
        rows = training_rows(output/'training.jsonl')
        updated = [row for row in rows if row.get('optimizer_step_completed')]
        partial = [row for row in updated if row.get('rejected_samples', 0)]
        timing = [row['policy_seconds'] for row in updated if 'policy_seconds' in row]
        latest = updated[-1] if updated else {}
        evaluation = []
        for split in ('validation', 'evaluation'):
            for path in sorted(output.glob(split+'_*.json')):
                report = read(path)
                evaluation.append(summarize_evaluation(report, path, split))
        pid = status.get('pid')
        process = Path(f'/proc/{pid}') if pid else None
        live = bool(process is not None and process.exists())
        try:
            cmdline = (process/'cmdline').read_bytes().replace(b'\x00', b' ').decode(errors='replace') if live else None
        except FileNotFoundError:
            # 只读期间进程可能自然退出，不复用过期PID证明运行。
            live, cmdline = False, None
        groups.append(dict(gpu=item['gpu'], session=item['session'], output=str(output), command=item['command'],
            pid=pid, process_present_on_reporting_host=live,
            process_matches_output=bool(cmdline and str(output) in cmdline), phase=status.get('phase'),
            adapter_updates=status.get('adapter_optimizer_updates', status.get('accepted_updates', 0)),
            wm_updates=status.get('wm_updates'),
            effective_train_transitions=status.get('real_by_role', {}).get('train',
                rows[-1].get('train_transitions', 0) if rows else 0),
            total_scene_physics_transitions=status.get('real_transitions'), real_by_role=status.get('real_by_role'),
            environment_count=status.get('num_envs'), peak_cuda_allocated_mib=status.get('peak_cuda_allocated_mib'),
            last_parameter_delta=latest.get('parameter_delta'), last_projection_gradient=latest.get('projection_gradient'),
            last_update_effect=latest.get('update_effect'), last_projection_solver=latest.get('projection_solver'),
            completed_Adam_records=len(updated),
            cumulative_accepted_BPTT_samples=sum(row.get('accepted_samples', 0) for row in updated),
            cumulative_rejected_BPTT_samples=sum(row.get('rejected_samples', 0) for row in updated),
            partial_batch_updates=[dict(adapter_updates=row.get('adapter_updates'),
                accepted_samples=row.get('accepted_samples'), rejected_samples=row.get('rejected_samples'),
                rejected_microbatches=row.get('rejected_microbatches'),
                parameter_delta=row.get('parameter_delta')) for row in partial],
            whole_batch_rejections=training_rows(output/'rejected_updates.jsonl'),
            first_own_Adam_update=updated[0].get('adapter_updates') if updated else None,
            first_own_Adam_record=updated[0] if updated else None,
            policy_seconds_first=timing[0] if timing else None,
            policy_seconds_last=timing[-1] if timing else None,
            policy_seconds_median=statistics.median(timing) if timing else None,
            latest_checkpoint=read(output/'latest_checkpoint.json'), best_checkpoint=read(output/'best_checkpoint.json'),
            zero_checkpoint=read(output/'zero_checkpoint.json'),
            evaluations=evaluation, failure_files=[str(path) for path in output.glob('failure_*.json')]))
    return dict(root=str(root), launched=True, source=launch['source'], groups=groups, skipped=launch.get('skipped', []),
                counter_boundary='effective_train_transitions excludes evaluation and unused scene environments')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--formal-root')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, output = Path(args.root), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    old = old_evaluations(root)
    formal = formal_status(Path(args.formal_root) if args.formal_root else root/'formal')
    result = dict(created_unix=time.time(), old_checkpoint_physics=old, formal=formal,
        warning='finite optimizer updates are not physical improvement; incomplete groups remain explicitly incomplete')
    (output/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    lines = ['# Stage2 修复任务实际结果汇总', '', f'旧模型完整评价：{old["complete_groups"]}/8。', '',
             '|模型|评价划分|Tracking变化|q RMSE变化|qd RMSE变化|根RMSE变化|完成率 零→Adapter|',
             '|---|---|---:|---:|---:|---:|---|']
    for group in old['groups']:
        for split, value in group['splits'].items():
            metrics = value['metrics']
            changes = [metrics[key]['adapter']-metrics[key]['zero'] if metrics[key] else None
                       for key in ('full_horizon_tracking', 'joint_q_rmse_rad', 'joint_qd_rmse_rad_s', 'root_position_rmse_m')]
            changes = ['待核验' if number is None else f'{number:+.6g}' for number in changes]
            lines.append('|'+ '|'.join([str(group['gpu']), split, *changes,
                f'{value["zero_completion_rate"]:.0%}→{value["adapter_completion_rate"]:.0%}'])+'|')
    lines += ['', '完整逐歌曲六项奖励、误差、重复差、初态与名义资产身份见 summary.json。', '',
              '收益/重复差只使用一次重复，不能解释为置信区间。旧模型评价不是新投影训练收益。', '',
              '|GPU|PID|实际训练转移|全部场景物理转移|Adapter更新|WM更新|当前阶段|',
              '|---|---:|---:|---:|---:|---:|---|']
    for group in formal['groups']:
        lines.append('|'+ '|'.join(str(group[key]) for key in ('gpu', 'pid', 'effective_train_transitions',
            'total_scene_physics_transitions', 'adapter_updates', 'wm_updates', 'phase'))+'|')
    (output/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(output=str(output), old_complete=old['complete_groups'], formal_groups=len(formal['groups'])), ensure_ascii=False))


if __name__ == '__main__':
    main()
