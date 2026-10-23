"""只读汇总 Stage2 已完成正式轮次的优化、回溯、Critic 与参考变化证据。

输入是运行目录 metrics/*.jsonl 的字节快照，兼容旧 trainable4_kl/backtracking
与当前 trainable_steps_kl/trust_region。只纳入含整轮 Actor、最终 KL 和已推进
iteration 的记录，忽略尚待整轮验收的 actor_minibatch 事件。“已发布”指整轮
metrics 已写出，不要求每轮都有 checkpoint；--max-iteration 可固定分析边界。
记录输入 SHA、排除数量和未完成 JSON 尾行；缺测保留 null，不伪装成零。

输出逐轮/逐步 LR、真实候选回溯、clip/active clip、全量目标、KL、耗时和 Critic
训练拟合，保留历史 objective/reduction 身份，不把旧代理目标混称联合概率。
最后可训练高斯均值变化与 reference_change.samples 的同初态同噪声最终参考
重生成分别命名；二者均不等于固定物理评估或实机效果。时间字段可能重叠，
只分别统计，不把 Actor 总时长与已包含的 KL/PPO 时长相加。

只向显式独立路径原子写入 JSON，不修改来源、模型、预算、采样或生产算法。
只依赖 Python 标准库，可在本地读取复制日志，不需要 GPU。
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics


def measured(values):
    """缺测不补零；非有限测量不能静默掩盖。"""
    result = [value for value in values if value is not None]
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for value in result):
        raise ValueError('数值诊断包含非有限值或非数值')
    return result


def describe(values):
    x = sorted(measured(values))
    if not x:
        return None
    position = (len(x) - 1) * .95
    low = int(position)
    p95 = x[low] + (x[min(low + 1, len(x) - 1)] - x[low]) * (position - low)
    return dict(count=len(x), min=x[0], p50=statistics.median(x), p95=p95,
                max=x[-1], mean=statistics.mean(x))


def measured_sum(values):
    values = measured(values)
    return sum(values) if values else None


def get(mapping, *keys):
    """按字段存在性回退，保留实际零值。"""
    return next((mapping[key] for key in keys if mapping.get(key) is not None), None)


def reference_summary(kl):
    report = kl.get('reference_change')
    if report is None:
        return None
    samples = report.get('samples')
    quantities = ('joint_rms_rad', 'maximum_joint_change_rad', 'normalized_rms',
                  'root_position_rms_m', 'root_orientation_rms_rad')
    protected = ([sample['protected_coordinates_unchanged'] for sample in samples
                  if sample.get('protected_coordinates_unchanged') is not None]
                 if samples is not None else [])
    return dict(scope=report.get('scope'), selection=report.get('selection'),
                covered_chains=report.get('covered_chains'),
                recorded_samples=len(samples) if samples is not None else None,
                wall_seconds=report.get('wall_seconds'),
                quantities={key: describe([sample.get(key) for sample in samples])
                            if samples is not None else None for key in quantities},
                protected_coordinates_checked=len(protected) if samples is not None else None,
                protected_coordinates_unchanged=all(protected) if protected else None)


def transaction_summary(transaction):
    if transaction is None:
        return None
    raw = transaction.get('attempts')
    trials, failures = [], Counter()
    for trial in raw or []:
        acceptance = trial.get('acceptance') or {}
        failures.update((acceptance.get('failures') or {}).keys())
        kl = trial.get('kl') or {}
        trials.append(dict(attempt=trial.get('attempt'), learning_rate=trial.get('learning_rate'),
                           passed=acceptance.get('passed'), failures=acceptance.get('failures'),
                           chain_joint_kl=kl.get('chain_joint_kl'), wall_seconds=trial.get('wall_seconds')))
    return dict(accepted=transaction.get('accepted'), attempt_count=transaction.get('attempt_count'),
                backtracks=transaction.get('backtracks'), initial_lr=transaction.get('initial_lr'),
                accepted_lr=transaction.get('accepted_lr'), selection=transaction.get('selection'),
                failure_counts=dict(failures) if raw is not None else None,
                accepted_candidate_seconds=measured_sum([t['wall_seconds'] for t in trials if t['passed'] is True]),
                rejected_candidate_seconds=measured_sum([t['wall_seconds'] for t in trials if t['passed'] is False]),
                attempts=trials if raw is not None else None)


def summarize_step(step):
    before = step.get('pre_step_chain') or {}
    fields = ('optimizer_step', 'learning_rate', 'learning_rates', 'global_chain_count',
              'total_gradient_norm', 'gradient_clip_factor', 'streaming_recompute_max_error',
              'ppo_loss', 'ppo_seconds', 'ppo_ratio_seconds', 'ppo_backward_seconds',
              'bc_seconds', 'post_step_kl_seconds', 'optimizer_transaction_seconds', 'wall_seconds',
              'full_gradient_direction', 'full_rollout_before', 'full_rollout_after', 'full_rollout_groups')
    return dict(**{key: step.get(key) for key in fields},
                ordinary_clip_fraction=get(before, 'clip_fraction') if before else step.get('clip_fraction'),
                active_clip_fraction=get(before, 'active_clip_fraction', 'objective_clip_fraction')
                if before else step.get('objective_clip_fraction'),
                pre_step_chain=step.get('pre_step_chain'),
                transaction=transaction_summary(get(step, 'trust_region', 'backtracking')))


def summarize_round(record, kl, kl_field):
    actor = record['actor']
    efficiency = record.get('efficiency') or {}
    timing = {key: efficiency.get(key) for key in ('environment_seconds', 'actor_total_seconds',
              'actor_ppo_seconds', 'bc_seconds', 'kl_seconds', 'critic_seconds', 'total_seconds')}
    for key, fallback in (('actor_total_seconds', 'actor_seconds'),
                          ('critic_seconds', 'critic_seconds'), ('total_seconds', 'wall_seconds')):
        if timing[key] is None:
            timing[key] = record.get(fallback)
    steps = [summarize_step(step) for step in actor.get('steps', [])]
    coordinate = kl.get('coordinate_kl') or {}
    critic = record.get('critic')
    critic_fit = ({key: critic.get(key) for key in ('before', 'after', 'optimizer_steps', 'epochs',
                  'global_sample_visits', 'critic_feature_contract')} if critic is not None else None)
    return dict(label=record.get('label'), iteration=record['state']['iteration'],
                session_id=record.get('session_id'), time_utc=record.get('time_utc'),
                kl_field=kl_field, objective=actor.get('objective'), objective_reduction=actor.get('objective_reduction'),
                optimizer_steps=actor.get('optimizer_steps'), planned=actor.get('planned_optimizer_steps'),
                accepted_optimizer_steps=get(actor, 'accepted_optimizer_steps', 'optimizer_steps'),
                attempted_optimizer_steps=actor.get('attempted_optimizer_steps'),
                rejected_step_candidates=actor.get('rejected_step_candidates'),
                accepted_learning_rates=[step['learning_rate'] for step in steps],
                nominal_learning_rate=actor.get('nominal_learning_rate'),
                last_accepted_learning_rate=actor.get('last_accepted_learning_rate'),
                active_clip_fraction=get(kl, 'active_clip_fraction', 'objective_clip_fraction'),
                ordinary_clip_fraction=kl.get('clip_fraction'), mean_coordinate_kl=kl.get('mean_coordinate_kl'),
                maximum_step_mean_coordinate_kl=kl.get('maximum_step_mean_coordinate_kl'),
                p95_coordinate_kl=coordinate.get('p95'), chain_joint_kl=kl.get('chain_joint_kl'),
                full_rollout_objective={key: kl.get(key) for key in ('objective_mean', 'behavior_objective_mean',
                    'objective_gain_mean', 'joint_score_gain_mean', 'ratio_is_joint_probability', 'denominator')},
                full_rollout_objective_history=actor.get('full_rollout_objective_history'),
                lr_control=actor.get('lr_control'), early_stop_reason=actor.get('early_stop_reason'),
                rejected_step=actor.get('rejected_step'), discarded_pending_minibatch=actor.get('discarded_pending_minibatch'),
                probability_passed=(record.get('probability') or {}).get('passed'), timing=timing,
                timing_scope=efficiency.get('timing_scope'), critic_training_fit=critic_fit,
                critic_scope='current_training_targets_not_independent_complete_MC',
                last_trainable_gaussian_mean_change=kl.get('final_reference_mean_change'),
                regenerated_final_reference=reference_summary(kl), steps=steps)


def read_rounds(source, maximum):
    rows, inputs, excluded = [], [], Counter()
    for path in sorted((source / 'metrics').glob('*.jsonl')):
        data = path.read_bytes()
        inputs.append(dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data)))
        lines = data.splitlines(keepends=True)
        for index, line in enumerate(lines):
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if index == len(lines) - 1 and not line.endswith(b'\n'):
                    excluded['incomplete_json_tail'] += 1
                    continue
                raise ValueError(f'历史 JSON 损坏：{path}:{index + 1}') from None
            actor = record.get('actor')
            kl_field = next((key for key in ('trainable_steps_kl', 'trainable4_kl')
                             if record.get(key) is not None), None)
            iteration = (record.get('state') or {}).get('iteration')
            if record.get('event') or not isinstance(actor, dict) or kl_field is None or iteration is None:
                excluded['not_completed_training_record'] += 1
                continue
            if ('pending' in (actor.get('acceptance_scope') or '') or
                    actor.get('accepted_optimizer_steps') == 0 and actor.get('optimizer_steps', 0) > 0):
                excluded['actor_not_finally_accepted'] += 1
                continue
            if maximum is not None and iteration > maximum:
                excluded['above_max_iteration'] += 1
                continue
            rows.append(summarize_round(record, record[kl_field], kl_field))
    rows.sort(key=lambda row: (row['iteration'], row['time_utc'] or '', row['session_id'] or ''))
    return rows, inputs, dict(excluded)


def build_report(source, maximum):
    rows, inputs, excluded = read_rounds(source, maximum)
    if not rows:
        raise ValueError('没有满足轮次边界的完整正式训练记录')
    attempts, failures, reasons = Counter(), Counter(), Counter()
    transactions, counted = [], []
    for row in rows:
        control = row['lr_control'] or {}
        if control.get('reason') is not None:
            reasons[control['reason']] += 1
        for step in row['steps']:
            transaction = step['transaction']
            if transaction is not None:
                transactions.append(transaction)
                if transaction['attempt_count'] is not None:
                    counted.append(transaction['attempt_count'])
                    attempts[str(transaction['attempt_count'])] += 1
        discarded = row['discarded_pending_minibatch'] or {}
        transaction = transaction_summary(get(discarded, 'trust_region', 'backtracking'))
        if transaction is not None:
            transactions.append(transaction)
    for transaction in transactions:
        failures.update(transaction['failure_counts'] or {})
    retry_steps = sum(count > 1 for count in counted) if counted else None
    active = measured([row['active_clip_fraction'] for row in rows])
    totals = {key: measured_sum([row['timing'][key] for row in rows]) for key in rows[0]['timing']}
    return dict(schema='genmo.stage2.published_history_summary.v2', source=str(source), inputs=inputs,
                max_iteration=maximum, excluded_records=excluded, rounds=len(rows),
                first_iteration=rows[0]['iteration'], last_iteration=rows[-1]['iteration'],
                accepted_adam_steps=measured_sum([row['accepted_optimizer_steps'] for row in rows]),
                attempted_optimizer_steps=measured_sum([row['attempted_optimizer_steps'] for row in rows]),
                rejected_step_candidates=measured_sum([row['rejected_step_candidates'] for row in rows]),
                accepted_lr=describe([lr for row in rows for lr in row['accepted_learning_rates']]),
                retry_steps=retry_steps, retry_step_fraction=retry_steps / len(counted) if counted else None,
                retry_measured_steps=len(counted), attempt_counts=dict(attempts), failure_counts=dict(failures),
                lr_control_reasons=dict(reasons), active_clip=describe(active),
                ordinary_clip=describe([row['ordinary_clip_fraction'] for row in rows]),
                rounds_with_nonzero_active_clip=sum(value > 0 for value in active) if active else None,
                mean_chain_joint_kl=describe([(row['chain_joint_kl'] or {}).get('mean') for row in rows]),
                max_chain_joint_kl=describe([(row['chain_joint_kl'] or {}).get('max') for row in rows]),
                kl_utilization=describe([(row['lr_control'] or {}).get('utilization') for row in rows]),
                full_rollout_objective_gain=describe([row['full_rollout_objective']['objective_gain_mean'] for row in rows]),
                timing_totals=totals,
                timing_by_round={key: describe([row['timing'][key] for row in rows]) for key in totals},
                accepted_candidate_seconds=measured_sum([t['accepted_candidate_seconds'] for t in transactions]),
                rejected_candidate_seconds=measured_sum([t['rejected_candidate_seconds'] for t in transactions]),
                timing_scope='fields_may_overlap_do_not_sum_actor_total_and_KL_or_PPO',
                incomplete_update_rounds=[row['label'] for row in rows
                    if row['optimizer_steps'] is not None and row['planned'] is not None
                    and row['optimizer_steps'] != row['planned']],
                rows=rows, scope='completed_metrics_records_not_checkpoint_or_physical_improvement_validation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-iteration', type=int, help='只纳入不超过该轮次的完整正式记录')
    args = parser.parse_args()
    if args.max_iteration is not None and args.max_iteration < 1:
        parser.error('--max-iteration 必须为正整数')
    source, output = args.source.resolve(strict=True), args.output.resolve()
    if output.is_relative_to(source):
        raise ValueError('诊断不得写入正式训练目录')
    report = build_report(source, args.max_iteration)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(output)
    print(json.dumps({key: value for key, value in report.items() if key not in ('rows', 'inputs')},
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__=='__main__':main()
