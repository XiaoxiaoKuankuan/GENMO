"""随机音乐起点二十轮试验的只读验收与轻量汇总。

复用现有逐控制步物理审计、模型权重核验和优化器统计，检查同一运行的第0/4/8/
12/16/20轮，而不是套用旧八轮队列或旧奖励的结果。全部输入保持只读；输出必须
是新目录。核对完整训练状态、实际Adam次数、MC策略版本、每次评估的真实权重及
32任务完整终止，再给出各轮奖励、时长、活动、八项Tracking和三域关节限位。

最后checkpoint中的各rank采样器保留了真实执行时间区间。本工具跨rank求区间
并集，按歌曲全长统计前/中/后三分之一覆盖，避免把抽到的起点或重复执行次数
冒充唯一覆盖；一秒预热、MC及验证不会计入Actor覆盖。统计只证明本次单随机种子
有限实验，不自动宣称收敛，不启动GPU、物理环境或训练。原始证据和失败均保留。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch

from gem.closedloop.dppo.budget import atomic_json
from gem.closedloop.dppo.full_dataset import FullMusicCatalog, _merge_intervals, SOURCES
from gem.closedloop.dppo.training_checkpoint import file_sha256
from gem.closedloop.dppo.training_config import configuration
from tools.eval.analyze_stage2_actor_lr_online import optimizer_rows, verify_evaluation_weights
from tools.eval.analyze_stage2_actor_lr_physics import analyze
from tools.eval.run_stage2_actor_lr_physics import verify_summary
from tools.eval.summarize_stage2_actor_lr import read_verified
from gem.closedloop.dppo.trainer import load_actor


def execution_coverage(payload, catalog):
    """实际执行区间跨rank合并；统计中后段是否真实进入Actor采集。"""
    intervals = {}
    draws = dict(standing=0, paired_motion=0)
    for rank in payload['rank_states']:
        collector = rank['collector']
        if collector['schema'] != 'genmo.stage10.inprocess_collector.time_sampling.v1':
            raise ValueError('Missing current random-time collector identity')
        sampler = collector['time_pool']
        draws['standing'] += sampler['standing_draws']
        draws['paired_motion'] += sampler['motion_draws']
        for state in collector['states']:
            if state is None:
                continue
            for sample, spans in state['sampler']['intervals'].items():
                for begin, end in spans:
                    intervals[sample] = _merge_intervals(intervals.get(sample, []), begin, end)
    sources = {}
    for source in SOURCES:
        length = [0., 0., 0.]
        covered = [0., 0., 0.]
        seen = 0
        for sample in catalog.samples['train'][source]:
            duration = sample['row']['num_frames'] * 20
            spans = intervals.get(source + '/' + sample['row']['sample_id'], [])
            seen += bool(spans)
            for i in range(3):
                left, right = duration * i / 3, duration * (i + 1) / 3
                length[i] += right - left
                covered[i] += sum(max(0., min(b, right) - max(a, left)) for a, b in spans)
        sources[source] = dict(unique_samples=seen, total_samples=len(catalog.samples['train'][source]),
            covered_seconds_by_third=[v/600 for v in covered], source_seconds_by_third=[v/600 for v in length],
            unique_coverage_by_third=[a/b for a, b in zip(covered, length)],
            total_unique_seconds=sum(covered)/600, total_unique_fraction=sum(covered)/sum(length))
    return dict(scope='global_union_of_real_Actor_controls_excluding_warmup_MC_BC_evaluation',
        thirds=['beginning', 'middle', 'end'], issued_task_counts=draws, by_source=sources)


def gpu_memory(path):
    """五秒轮询含PhysX外部分配，数值不冒充CUDA精确瞬时峰值。"""
    peaks = {}
    with path.open() as stream:
        for row in csv.DictReader(stream):
            values = {k.strip(): v.strip() for k, v in row.items()}
            rank = values['index']
            value = float(values['memory.used [MiB]'].split()[0]) / 1024
            peaks[rank] = max(peaks.get(rank, 0.), value)
    return dict(per_gpu_GiB=peaks, sample_interval_seconds=5, includes_external_PhysX=True)


def review(run, destination):
    evidence = verify_summary(run)
    summary = read_verified(evidence['summary'], evidence['summary_sha256'])
    session = Path(evidence['summary']).parent
    config = configuration(session/'resolved_config.yaml')
    stage, final = config['stage10'], summary['final_state']
    if (stage['run_control']['max_training_iterations'] != 20
            or stage['training']['actor_lr'] != 3e-7
            or stage['training']['actor_objective'] != 'joint_score_pg'
            or stage['actor_experiment']['kl_multiplier'] != 3
            or any(final[k] != 20 for k in ('iteration', 'actor_updates', 'policy_version'))
            or not final['mc_preheat_complete']):
        raise ValueError('Expected the authorized complete twenty-round random-start experiment')
    metrics = run/'metrics'/f'{session.name}.jsonl'
    rows = optimizer_rows([json.loads(line) for line in metrics.read_text().splitlines()])
    accepted = [r for r in rows if r['mode'] == 'dppo_bc']
    if [r['iteration'] for r in accepted] != list(range(1, 21)):
        raise ValueError('Missing or duplicated accepted Actor iterations')
    attempts = sum(row['backtracking']['attempt_count'] for row in rows)
    if attempts != final['optimizer_attempts']:
        raise ValueError('Actual Adam attempts disagree with resource accounting')
    expected = {0, 4, 8, 12, 16, 20}
    if {a['iteration'] for a in summary['evaluation_artifacts']} != expected:
        raise ValueError('Fixed complete-song evaluation coverage differs')
    preheat = json.loads((session/'mc_calibration/preheat.json').read_text())
    if not preheat['quality_gate_passed'] or stage['critic']['preheat_artifact'] is not None:
        raise ValueError('Expected fresh complete MC initialization under this run identity')
    calibrations = []
    for step in sorted(expected - {0}):
        path = session/'mc_calibration'/f'policy_{step}.json'
        mc = json.loads(path.read_text())
        if mc['behavior_version'] != step or mc['fit'] or mc['after']['episodes'] != 32:
            raise ValueError('MC diagnostics must belong to the current fixed behavior')
        calibrations.append(dict(iteration=step, path=str(path), sha256=file_sha256(path),
            metrics=mc['after'], wall_seconds=mc['wall_seconds']))
    destination.mkdir(parents=True, exist_ok=False)
    config['runtime']['genmo_device'] = 'cpu'
    actor, _, _ = load_actor(config)
    evaluations = []
    for artifact in summary['evaluation_artifacts']:
        report_path = run/artifact['path']
        report = read_verified(report_path, artifact['sha256'])
        weights = verify_evaluation_weights(run, artifact, report, actor)
        analysis = analyze(report_path, destination/artifact['label'],
            run/'evaluation_baseline.json' if artifact['iteration'] else None)
        if (analysis['aggregate']['valid_episode_count'] != 32
                or analysis['aggregate']['administrative_truncation_count']):
            raise ValueError('Incomplete or administratively truncated physical comparison')
        evaluations.append(dict(iteration=artifact['iteration'], label=artifact['label'], weights=weights,
            report=dict(path=str(report_path), sha256=artifact['sha256']),
            aggregate=analysis['aggregate'], paired=analysis.get('paired'), by_source=analysis['by_source'],
            tracking_eight=analysis['tracking_eight_control_mean'], yaw=analysis['yaw']['overall'],
            tasks=[{k:v for k,v in t.items() if k!='generated'} for t in analysis['tasks']],
            retention={k:v for k,v in analysis['retention'].items() if k!='samples'},
            wall_seconds=analysis['evaluation_seconds']))
    del actor
    pointer = json.loads((run/'latest.json').read_text())
    path = run/pointer['path']
    if file_sha256(path) != pointer['sha256']:
        raise ValueError('Final checkpoint changed')
    payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    coverage = execution_coverage(payload, FullMusicCatalog(config['paths']['data_root']))
    result = dict(schema='genmo.stage2.random20_review.v1', status='completed', evidence=evidence,
        config=stage, initial_state=summary['initial_state'], final_state=final, rows=rows,
        accepted_optimizer_steps=20, optimizer_attempts=attempts,
        rejected_adam_attempts=attempts-20, evaluations=evaluations,
        preheat=preheat, independent_current_policy_MC=calibrations, sampling_coverage=coverage,
        gpu=gpu_memory(run.parent/'gpu.csv'), final_checkpoint=pointer,
        metrics_sha256=file_sha256(metrics), implementation_sha256=file_sha256(__file__),
        scope='single_training_seed_16_music_groups_32_full_available_paired_tasks_not_formal_promotion')
    atomic_json(destination/'review.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    review(args.run.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
