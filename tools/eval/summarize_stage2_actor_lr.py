"""汇总统一Actor学习率实验的已核验物理结果，输出便于审阅的轻量证据。

逐项绑定扫描权重、实际加载SHA、运行中Actor指纹、完整物理分析和固定Critic离线
诊断。只有完整参数相同的候选共享物理结果；别名仍分别保留自己的提议LR与KL。
把每任务回报/时长/失败、三域关节限位、八项Tracking、活动与音乐诊断、BC能力
保持、实际墙钟和GPU监控一起保存，不按幸存任务筛选，也不以小loss代替物理收益。

默认要求全部物理/分析完成，--partial只生成明确标注的不完整阶段快照。每次输出
必须新建，原模型/日志/报告均只读。所有资源账本包括未计奖但真实推进的世界步骤，
与有奖励的机器人控制步分别列示；GPU显存是5秒轮询峰值，不是精确瞬时峰值。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gem.closedloop.dppo.budget import atomic_json
from gem.closedloop.dppo.training_checkpoint import file_sha256


def read_verified(path, expected):
    if file_sha256(path) != expected:
        raise ValueError('Published evidence changed: ' + str(path))
    return json.loads(Path(path).read_text())


def verify_actor(job, report, scan_case):
    """实际物理执行身份必须指向扫描后保存的那组权重，不能只比配置名称。"""
    initialization = report['evaluation_identity']['training_identity']['actor_initialization']
    if job['actor'] is None:
        if initialization['kind'] != 'source_stage1' or initialization['source_global_step'] != 595000:
            raise ValueError('Physical baseline is not source Stage1 s595000')
    else:
        expected = job['actor']
        if (report['actor_identity']['model_fingerprint'] != expected['actor_fingerprint']
                or initialization['path'] != expected['path'] or initialization['sha256'] != expected['sha256']
                or initialization['source_case_sha256'] != scan_case):
            raise ValueError('Evaluated Actor differs from the scanned candidate')
        if file_sha256(expected['path']) != expected['sha256']:
            raise ValueError('Evaluated Actor weights changed')


def gpu_peak(path):
    records = []
    with path.open() as stream:
        for row in csv.reader(stream):
            if len(row) != 4:
                raise ValueError('Incomplete GPU monitor record')
            records.append((int(row[1]), float(row[2]), float(row[3])))
    return dict(max_memory_GiB=max(v[1] for v in records) / 1024,
        per_gpu_GiB={str(i): max(v[1] for v in records if v[0] == i) / 1024 for i in range(8)},
        sampling_interval_seconds=5, includes_external_PhysX_allocations=True,
        scope='nvidia_smi_polled_peak_not_exact_instantaneous_peak')


def summarize(root, output, partial=False):
    if output.exists():
        raise FileExistsError(output)
    scan_path = root / 'fixed_scan/report.json'
    queue_path = root / 'physics/queue.json'
    analysis_path = root / 'physical_analysis/index.json'
    critic_path = root / 'critic_analysis/index.json'
    queue = json.loads(queue_path.read_text())
    scan = read_verified(scan_path, queue['scan_sha256'])
    index = json.loads(analysis_path.read_text())
    critic_index = json.loads(critic_path.read_text())
    complete = (queue['status'] == 'completed_all_distinct_survivors'
                and index['status'] == 'completed' and critic_index['status'] == 'completed')
    if not complete and not partial:
        raise ValueError('Physical, control and Critic analyses must all complete')
    analyses = {v['name']: v for v in index['attempts'] if v['status'] == 'passed'}
    critics = {v['name']: v for v in critic_index['attempts'] if v['status'] == 'passed'}
    physical = {v['name']: v for v in queue['attempts'] if v['status'] == 'passed'}
    cases = []
    for job in queue['jobs']:
        name = job['name']
        if any(name not in mapping for mapping in (analyses, critics, physical)):
            if partial:
                continue
            raise ValueError('Completed queue misses a candidate')
        a, c, p = analyses[name], critics[name], physical[name]
        report = read_verified(p['evidence']['report'], p['evidence']['report_sha256'])
        summary = read_verified(p['evidence']['summary'], p['evidence']['summary_sha256'])
        analysis = read_verified(a['analysis'], a['sha256'])
        critic = read_verified(c['path'], c['sha256'])
        verify_actor(job, report, scan['source_case_sha256'])
        if (analysis['actor'] != report['actor_identity'] or critic['actor'] != report['actor_identity']
                or critic['source_report_sha256'] != p['evidence']['report_sha256']
                or summary['status'] != 'passed' or not summary['source_unchanged']):
            raise ValueError('Analysis is not bound to the completed physical run')
        if analysis['yaw']['controls'] != analysis['aggregate']['executed_control_steps']:
            raise ValueError('Actual counted controls differ')
        retention = analysis['retention']
        tasks = [{k: v for k, v in row.items() if k != 'generated'} for row in analysis['tasks']]
        cases.append(dict(name=name, aliases=job['aliases'], actor=analysis['actor'],
            aggregate=analysis['aggregate'], tracking_eight=analysis['tracking_eight_control_mean'],
            yaw={k: v for k, v in analysis['yaw'].items() if k in ('overall', 'by_task', 'controls', 'physics_substeps')},
            by_source=analysis['by_source'], tasks=tasks,
            retention={k: v for k, v in retention.items() if k != 'samples'}, paired=analysis.get('paired'),
            fixed_critic_mc={k: v for k, v in critic.items() if k in ('overall', 'by_source', 'episodes', 'scope', 'execution')},
            evaluation_wall_seconds=analysis['evaluation_seconds'], process_wall_seconds=p['elapsed_seconds'],
            gpu=gpu_peak(Path(p['run']).parent / 'gpu.csv'), budget=summary['final_state']['budget']['used'],
            evidence=dict(report=p['evidence'], analysis=dict(path=a['analysis'], sha256=a['sha256']),
                          critic=dict(path=c['path'], sha256=c['sha256']))))
    names = {alias: case['name'] for case in cases for alias in [case['name'], *case['aliases']]}
    candidates = []
    for candidate in scan['candidates']:
        trial = candidate['backtracking']
        accepted = trial['attempts'][-1]['diagnostics'] if trial['accepted'] else None
        candidates.append(dict(name=candidate['name'], objective=candidate['objective'],
            proposed_lr=candidate['actor_lr'], accepted_lr=trial['accepted_lr'], attempts=trial['attempt_count'],
            kl_multiplier=candidate['kl_multiplier'], physics_name=names.get(candidate['name']),
            final_kl={k: v for k, v in candidate['final_kl'].items()
                      if k not in ('frozen_identity', 'reference_change')},
            parameter_modules=None if accepted is None else accepted['parameters']['modules'],
            output_diagnostics=None if accepted is None else {k: v for k, v in accepted['outputs'].items() if k != 'samples'},
            gradient=candidate['gradient'],
            passed_fixed_prescreen=candidate['screening']['passed']))
    result = dict(schema='genmo.stage2.uniform_lr_compact_physical_review.v1',
        status='complete_physics_not_formal_promotion' if complete else 'partial_not_final_selection',
        sources=[dict(path=str(p), sha256=file_sha256(p)) for p in (scan_path, queue_path, analysis_path, critic_path)],
        implementation_sha256=file_sha256(__file__), cases=cases, candidates=candidates,
        completed_distinct_physical=len(cases), expected_distinct_physical=len(queue['jobs']),
        caveats=['same_16_music_groups_used_for_selection_not_independent_confirmation',
                 'full_available_paired_samples_not_uncropped_source_audio',
                 'fixed_critic_CPU_predictions_are_posthoc_not_online_GPU_predictions'])
    atomic_json(output, result)
    print(json.dumps(dict(output=str(output), distinct=len(cases), status=result['status'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    summarize(args.root, args.output, args.partial)
