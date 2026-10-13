"""只读比较Stage10固定物理评估的真实执行、参考连续性及任务配对收益。

本工具读取两个完整32任务评估报告及其权威执行journal，不运行Actor、不加载
Isaac、不改写模型/轨迹/数据库。先核对固定音乐、种子、奖励、资产、部署时钟和
物理源码合同；算法对照要求实际随机核一致，显式noise模式只允许std日程变化。
每个journal按只读SQLite连接逐回复核验SHA，复用已有无损列式解码。按报告中的
session/env/episode/控制tick取得实际qpos、关节速度、GMT动作及参考，而不是
把最终高斯均值当作实际机器人动作。所有指标使用50Hz实际控制时间。

收益按同任务配对；置信区间以音乐为簇、同音乐两种子一起抽样，并在各来源内
分层bootstrap。仅为有限固定任务统计，不能代替新音乐泛化或多轮收敛。轨迹
长度不同时仅比较共同执行前缀，显式报告未配对控制步，不能隐藏失败或缩短执行。
输出必须是不存在的新JSON；输入只读，不创建训练恢复点，不设置质量通过门槛。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gem.closedloop.dppo.journal_codec import decode_payload
from gem.runtime.trajectory_blocks import expand_feedback


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def statistics(values):
    x = np.asarray(values, dtype=np.float64)
    if not x.size:
        return dict(count=0, rms=None, mean=None, p95_abs=None, max_abs=None)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite physical comparison input')
    return dict(count=x.size, rms=float(np.sqrt(np.mean(x*x))), mean=float(x.mean()),
                p95_abs=float(np.quantile(np.abs(x), .95)), max_abs=float(np.abs(x).max()))


def validate_pair(before, after, *, mode):
    if any(report.get('status') != 'passed' or report.get('task_count') != len(report['episodes'])
           for report in (before, after)):
        raise ValueError('Complete successful fixed evaluations required')
    if before['plan_sha256'] != after['plan_sha256'] or before['plan'] != after['plan']:
        raise ValueError('Fixed music, seeds or evaluation plan differ')
    a, b = (report['evaluation_identity'] for report in (before, after))
    for key in ('timing_contract', 'critical_latency_budgets', 'episode_seconds', 'physics'):
        if a[key] != b[key]:
            raise ValueError('Physical evaluation contract differs: ' + key)
    for key in ('assets', 'reward', 'timing', 'source_files'):
        if a['training_identity'][key] != b['training_identity'][key]:
            raise ValueError('Bound physical asset/source contract differs: ' + key)
    kernels = [dict(identity['kernel']) for identity in (a, b)]
    if mode == 'noise':
        for kernel in kernels:
            kernel.pop('effective_std_floors', None)
    if kernels[0] != kernels[1]:
        raise ValueError('Actual behavior kernel differs outside explicit comparison scope')
    left = {row['task_id']: row for row in before['episodes']}
    right = {row['task_id']: row for row in after['episodes']}
    if len(left) != len(before['episodes']) or set(left) != set(right):
        raise ValueError('Duplicate or missing physical evaluation task')
    return left, right


def load_physical(report):
    wanted = defaultdict(dict)
    ordered = {}
    for episode in report['episodes']:
        keys = []
        for decision in episode['decisions']:
            identity = decision['identity']
            journal = Path(decision['raw_sample_path']).parent.parent / 'execution_journal.sqlite'
            for control in decision['controls']:
                key = (identity['backend_session_id'], identity['env_id'], identity['episode_id'], control['tick'])
                if key in wanted[journal]:
                    raise ValueError('Repeated episode control in evaluation report')
                wanted[journal][key] = None
                keys.append((journal, key))
        ordered[episode['task_id']] = keys
    evidence = []
    for path, selected in wanted.items():
        before = (path.stat().st_size, path.stat().st_mtime_ns)
        replies = 0
        with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as connection:
            for digest, payload in connection.execute('SELECT sha256,payload FROM replies ORDER BY rowid'):
                body = decode_payload(payload, sha256=digest)
                body = body.get('result', body)
                replies += 1
                if not isinstance(body, dict):
                    continue
                for row in expand_feedback(body, columnar=True).get('trace', ()):
                    key = (row['backend_session_id'], row['env_id'], row['episode_id'], row['tick'])
                    if key not in selected:
                        continue
                    if selected[key] is not None:
                        raise ValueError('Repeated authoritative physical control')
                    if not row['state_valid'] or row['completed_physics_steps'] != 4:
                        raise ValueError('Matched control lacks four complete trusted physical substeps')
                    selected[key] = dict(qpos=np.asarray(row['actual_qpos'], dtype=np.float64).copy(),
                        velocity=np.asarray(row['actual_joint_vel'], dtype=np.float64).copy(),
                        action=np.asarray(row['applied_action'], dtype=np.float64).copy(),
                        reference=np.asarray(row['reference']['joint_pos'], dtype=np.float64).copy())
        if before != (path.stat().st_size, path.stat().st_mtime_ns) or any(v is None for v in selected.values()):
            raise ValueError('Missing physical control or evaluation journal changed during reading')
        evidence.append(dict(path=str(path), verified_reply_count=replies, matched_controls=len(selected),
                             size_bytes=before[0], sha256=sha256(path)))
    result = {}
    for task, keys in ordered.items():
        ticks = np.array([key[-1] for _, key in keys], dtype=np.int64)
        if len(ticks) > 1 and not np.all(np.diff(ticks) == 12):
            raise ValueError('Evaluation control timeline has gaps or repeated ticks')
        result[task] = dict(ticks=ticks, **{name: np.array([wanted[path][key][name] for path, key in keys])
                                          for name in ('qpos', 'velocity', 'action', 'reference')})
    return result, evidence


def compare(before, after, *, mode='algorithm'):
    left, right = validate_pair(before, after, mode=mode)
    physical_a, evidence_a = load_physical(before)
    physical_b, evidence_b = load_physical(after)
    differences = defaultdict(list)
    dynamics = {side: defaultdict(list) for side in ('before', 'after')}
    pairs, clusters, unmatched = [], defaultdict(lambda: defaultdict(list)), [0, 0]
    for task, episode in left.items():
        other = right[task]
        delta = other['reward_sum'] - episode['reward_sum']
        pairs.append(dict(task_id=task, dataset=episode['dataset'], sample_id=episode['sample_id'],
            seed=episode['seed'], reward_delta=delta, new_failure=other['physical_failure'] and not episode['physical_failure']))
        clusters[episode['dataset']][episode['sample_id']].append(delta)
        a, b = physical_a[task], physical_b[task]
        n = min(len(a['ticks']), len(b['ticks']))
        unmatched[0] += len(a['ticks']) - n
        unmatched[1] += len(b['ticks']) - n
        if n:
            qa, qb = a['qpos'][:n], b['qpos'][:n]
            differences['root_position_m'].append((qb[:, :3] - qa[:, :3]).ravel())
            differences['joint_position_rad'].append((qb[:, 7:] - qa[:, 7:]).ravel())
            unit_a = qa[:, 3:7] / np.linalg.norm(qa[:, 3:7], axis=1, keepdims=True)
            unit_b = qb[:, 3:7] / np.linalg.norm(qb[:, 3:7], axis=1, keepdims=True)
            unit_b *= np.where((unit_a * unit_b).sum(1, keepdims=True) < 0., -1., 1.)
            # 同一四元数及其反号必须得到精确零；atan2避免acos在零附近放大舍入。
            differences['root_orientation_rad'].append(4 * np.arctan2(
                np.linalg.norm(unit_b - unit_a, axis=1), np.linalg.norm(unit_b + unit_a, axis=1)))
            for key, unit in (('velocity', 'rad_s'), ('reference', 'rad'), ('action', 'gmt_action_units')):
                differences[key + '_' + unit].append((b[key][:n] - a[key][:n]).ravel())
        for side, values in (('before', a), ('after', b)):
            if len(values['ticks']) > 1:
                dynamics[side]['actual_joint_acceleration_rad_s2'].append((np.diff(values['velocity'], axis=0) / .02).ravel())
                dynamics[side]['reference_joint_step_rad'].append(np.diff(values['reference'], axis=0).ravel())
                dynamics[side]['reference_joint_velocity_rad_s'].append((np.diff(values['reference'], axis=0) / .02).ravel())
    # 同一音乐的两个seed不是独立音乐样本；按来源等权且在来源内重采音乐簇。
    rng = np.random.default_rng(12012)
    draws = []
    for source in sorted(clusters):
        means = np.array([np.mean(v) for v in clusters[source].values()])
        draws.append(means[rng.integers(0, len(means), size=(2000, len(means)))].mean(1))
    ci = np.quantile(np.stack(draws).mean(0), [.025, .975]).tolist()
    summarize = lambda groups: {name: statistics(np.concatenate(chunks)) for name, chunks in groups.items()}
    return dict(schema='genmo.stage10.paired_physical_evaluation.v3', comparison_mode=mode,
        task_count=len(pairs), music_cluster_count=sum(len(v) for v in clusters.values()), pairs=pairs,
        source_balanced_reward_before=before['source_balanced_reward'], source_balanced_reward_after=after['source_balanced_reward'],
        source_balanced_reward_delta=after['source_balanced_reward'] - before['source_balanced_reward'],
        reward_delta_cluster_bootstrap_95_percent=ci, bootstrap_repeats=2000,
        physical_failures=[before['physical_failure_count'], after['physical_failure_count']],
        unmatched_control_steps=unmatched, actual_physical_difference=summarize(differences),
        dynamics={side: summarize(groups) for side, groups in dynamics.items()},
        raw_tracking_errors={side: report['aggregate']['overall']['raw_errors'] for side, report in (('before', before), ('after', after))},
        stage1_retention={side: report.get('stage1_retention') for side, report in (('before', before), ('after', after))},
        behavior_kernels=[report['evaluation_identity']['kernel'] for report in (before, after)],
        journal_evidence=[evidence_a, evidence_b],
        scope='fixed_music_matched_actual_physics_common_execution_prefix_not_long_term_convergence')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--mode', choices=('algorithm', 'noise'), default='algorithm')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    inputs = [dict(path=str(p.resolve()), sha256=sha256(p)) for p in (args.before, args.after)]
    result = compare(json.loads(args.before.read_text()), json.loads(args.after.read_text()), mode=args.mode)
    if any(sha256(item['path']) != item['sha256'] for item in inputs):
        raise ValueError('Evaluation input report changed during reading')
    result['inputs'] = inputs
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({key: result[key] for key in ('task_count', 'source_balanced_reward_delta',
        'reward_delta_cluster_bootstrap_95_percent', 'physical_failures', 'unmatched_control_steps')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
