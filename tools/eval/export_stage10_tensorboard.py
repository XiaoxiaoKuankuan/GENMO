#!/usr/bin/env python3
"""把已完成 Stage10 正式运行重建为可直接查看的精简 TensorBoard。

输入为原运行目录；按完整 session 的初始/最终迭代串联普通 Actor 更新，读取原始
metrics JSONL，而不是从几百个旧 TensorBoard 标签猜测分母。固定验证只接受 session
summary 登记且 SHA 一致的报告，使用同一 Stage1 基线重算每秒回报、原始跟踪误差、
活动度和音乐组配对区间；独立完整 MC 另作 critic_mc run。不会加载 checkpoint、
启动 GPU/Isaac、修改原日志或更改模型，输出必须是不存在的独立目录。
发布前校验迭代连续性、任务身份和有限白名单，再输出指标定义、精简事件、来源
摘要和趋势小结。失败目录保留为 partial 名称，不能冒充已完成可交付视图。
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gem.closedloop.dppo.tensorboard_metrics import (
    DashboardWriter, DEFINITIONS, VERSION, evaluation_scalars, evaluation_view, mc_scalars, training_scalars,
)


def read_report(path, expected_sha=None):
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha is not None and digest != expected_sha:
        raise ValueError(f'Evaluation SHA mismatch: {path}')
    return json.loads(raw), digest


def export_run(run, output):
    run, output = Path(run).resolve(strict=True), Path(output).resolve()
    partial = output.with_name(output.name+'.partial')
    if output.exists() or partial.exists() or output == run:
        raise FileExistsError('A new independent output directory is required')
    sessions = []
    for path in (run/'sessions').glob('*/summary.json'):
        report, digest = read_report(path)
        if report.get('status') != 'passed' or report.get('exit_code') != 0:
            raise ValueError(f'Only completed sessions can be exported: {path}')
        sessions.append((report['initial_state']['iteration'], report['final_state']['iteration'], path, report, digest))
    sessions.sort(key=lambda row: (row[0], row[1]))
    if not sessions or sessions[0][0] != 0:
        raise ValueError('Need complete evidence starting with the Stage1 baseline')
    previous = 0
    for begin, end, path, _, _ in sessions:
        if begin != previous or end <= begin:
            raise ValueError(f'Session iteration gap or overlap: {path}')
        previous = end
    baseline_path = run/'evaluation_baseline.json'
    baseline_report, baseline_sha = read_report(baseline_path)
    baseline = evaluation_view(baseline_report)
    del baseline_report
    writer = DashboardWriter(partial)
    provenance = [dict(path=str(baseline_path), sha256=baseline_sha)]
    seen, evaluations, scalar_series = set(), [], set()
    blocks = defaultdict(list)
    old_tags = set()
    try:
        curves = run/'curves.jsonl'
        if curves.exists():
            with curves.open() as stream:
                for line in stream:
                    old_tags.update(json.loads(line)['metrics'])
        for begin, end, path, summary, summary_sha in sessions:
            provenance.append(dict(path=str(path), sha256=summary_sha))
            metrics_path = run/'metrics'/f'{summary["session_id"]}.jsonl'
            digest = hashlib.sha256()
            with metrics_path.open('rb') as stream:
                for raw in stream:
                    digest.update(raw)
                    record = json.loads(raw)
                    if record.get('mode') != 'dppo_bc':
                        continue
                    step = record['state']['iteration']
                    if not begin < step <= end or step in seen:
                        raise ValueError('Duplicate or out-of-session accepted iteration')
                    values = training_scalars(record)
                    writer.write('training', step, values,
                                 walltime=datetime.fromisoformat(record['time_utc'].replace('Z', '+00:00')).timestamp())
                    seen.add(step)
                    scalar_series.update(('training', tag) for tag in values)
                    blocks[(step-1)//100].append(values)
            provenance.append(dict(path=str(metrics_path), sha256=digest.hexdigest()))
            for artifact in summary['evaluation_artifacts']:
                evaluation_path = (run/artifact['path']).resolve(strict=True)
                if not evaluation_path.is_relative_to(run):
                    raise ValueError('Evaluation artifact escapes source run')
                report, digest = read_report(evaluation_path, artifact['sha256'])
                step = artifact['iteration']
                if report['iteration'] != step:
                    raise ValueError('Evaluation is labeled with the wrong checkpoint iteration')
                values = evaluation_scalars(report, baseline)
                writer.write('validation', step, values, walltime=evaluation_path.stat().st_mtime)
                scalar_series.update(('validation', tag) for tag in values)
                evaluations.append(dict(iteration=step, scalars=values))
                provenance.append(dict(path=str(evaluation_path), sha256=digest))
                del report
                print(f'[COMPACT] validation iteration={step}', flush=True)
            for mc_path in sorted((path.parent/'mc_calibration').glob('*.json')):
                report, digest = read_report(mc_path)
                step = report['behavior_version']
                if not begin <= step <= end:
                    raise ValueError('MC policy version is outside its session')
                values = mc_scalars(report)
                writer.write('critic_mc', step, values, walltime=mc_path.stat().st_mtime)
                scalar_series.update(('critic_mc', tag) for tag in values)
                provenance.append(dict(path=str(mc_path), sha256=digest))
        if seen != set(range(1, previous+1)):
            raise ValueError('Accepted iteration coverage is incomplete')
        if not {0, previous}.issubset({row['iteration'] for row in evaluations}):
            raise ValueError('Missing baseline or final evaluation')
        if len({row['iteration'] for row in evaluations}) != len(evaluations):
            raise ValueError('Duplicate evaluation iteration would hide a model identity')
        evaluations.sort(key=lambda row: row['iteration'])
        best = max(evaluations, key=lambda row: row['scalars']['Reward/total_per_second'])
        used_tags = {tag for _, tag in scalar_series}
        summary = dict(schema=VERSION+'.export', status='passed', source_run=str(run),
            accepted_iterations=len(seen), fixed_evaluations=len(evaluations),
            old_scalar_tags=len(old_tags), compact_scalar_tags=len(used_tags),
            scalar_series=len(scalar_series), groups=sorted({tag.split('/')[0] for tag in used_tags}),
            series_by_scope=dict(Counter(scope for scope, _ in scalar_series)),
            baseline=evaluations[0], final=evaluations[-1], highest_short_validation_rate=best,
            selection_note='曲线中的最高短验证每秒回报不代表整曲部署最优',
            training_blocks=[dict(begin=block*100+1, end=min(previous,(block+1)*100),
                iterations=len(rows), source_balanced_reward_per_second=sum(r['Reward/total_per_second'] for r in rows)/len(rows))
                for block, rows in sorted(blocks.items())],
            missing_historical_series='普通训练未逐控制步保存奖励分项到metrics；分项来自完整固定验证，不填零或用稀疏审计冒充全量。',
            sources=provenance)
        (partial/'export_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    finally:
        writer.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    partial.rename(output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = export_run(args.run_dir, args.output)
    print(json.dumps({key: summary[key] for key in (
        'status', 'accepted_iterations', 'fixed_evaluations', 'old_scalar_tags', 'compact_scalar_tags', 'groups')},
        ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
