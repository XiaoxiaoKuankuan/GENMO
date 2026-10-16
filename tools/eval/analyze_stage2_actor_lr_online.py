"""只读汇总高学习率八轮在线实验，逐次核对实际Adam、整曲物理与保存权重。

本工具跟随既有有限实验队列，只使用CPU；不创建仿真、优化器或新的训练数据。
复用完整物理分析器核验第0/4/8轮的全部50Hz奖励、四子步、三域限位及FineDance
轨迹；另读取当前策略自己的完整MC诊断，不能用初始GAE或旧MC标签冒充价值真值。
每份评估必须找到对应的实际checkpoint，并重新加载Actor核验参数和buffer指纹。

逐轮保留提议/接受LR、全部回溯尝试、实际全局链分母、PPO/BC贡献、参数与输出
变化、全链四步KL、峰值显存及墙钟。统计包含被拒绝的Adam计算，不能把八次接受
更新误报成只计算了八次。原模型和报告只读，输出只允许新建；队列失败也留下明确
的不完整结果。完成本工具仅代表证据汇总完成，不代表新方案获准正式长期训练。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch

from gem.closedloop.dppo.budget import atomic_json
from gem.closedloop.dppo.evaluation import _model_fingerprint
from gem.closedloop.dppo.trainer import load_actor
from gem.closedloop.dppo.training_checkpoint import file_sha256, SCHEMA
from gem.closedloop.dppo.training_config import configuration
from tools.eval.analyze_stage2_actor_lr_physics import analyze
from tools.eval.run_stage2_actor_lr_online import verify_online
from tools.eval.summarize_stage2_actor_lr import gpu_peak, read_verified


def optimizer_rows(records):
    """按真实有效链归一化及实际Adam尝试计数，不依赖预估每卡配额。"""
    result = []
    for row in records:
        if row.get('mode') not in ('dppo_bc', 'kl_rejected_rollback'):
            continue
        actor = row['actor']
        step = actor['steps'][0] if actor['steps'] else actor['rejected_step']
        transaction = step['backtracking']
        chains = step['global_chain_count']
        scale = 4 if step['objective'] == 'coordinate_dppo' else 1
        if (not 0 < chains <= 16384 or step['denominator'] != chains * scale
                or step['internal_sample_visits'] != chains * 4
                or step['bc_global_samples'] != 1024
                or transaction['attempt_count'] != len(transaction['attempts'])
                or not row['physics_paused_during_optimization']):
            raise ValueError('Online full-batch denominator, BC, attempt count or physical pause differs')
        if row['mode'] == 'dppo_bc' and (not transaction['accepted'] or actor['optimizer_steps'] != 1):
            raise ValueError('Accepted round lacks exactly one accepted Adam update')
        result.append(dict(iteration=row['state']['iteration'], mode=row['mode'],
            training_control_steps=row['state']['training_control_steps'],
            gradient={k: v for k, v in step.items() if k != 'backtracking'},
            backtracking=transaction, advantage=actor['advantage_diagnostics'],
            final_kl=row['trainable4_kl'], collection_by_rank=row['collection_by_rank'],
            wall_seconds=row['wall_seconds'], actor_seconds=row['actor_seconds'],
            critic_seconds=row['critic_seconds'], critic=row['critic'],
            memory_by_rank=row['memory_by_rank'], performance_by_rank=row['performance_by_rank'],
            overhead_since_previous_record=row['overhead_since_previous_record']))
    return result


def verify_evaluation_weights(run, artifact, report, actor):
    paths = list((run / 'checkpoints').glob('*_evaluated_' + artifact['label'] + '.pt'))
    if len(paths) != 1:
        raise ValueError('Evaluation needs exactly one corresponding actual weight snapshot')
    path = paths[0]
    payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if (payload['schema'] != SCHEMA or payload['state']['iteration'] != artifact['iteration']
            or payload['identity'] != report['evaluation_identity']['training_identity']):
        raise ValueError('Evaluation checkpoint training identity or iteration differs')
    actor.load_state_dict(payload['actor'], strict=True)
    fingerprint = _model_fingerprint(actor)
    if fingerprint != payload['actor_fingerprint'] or fingerprint != report['actor_identity']['model_fingerprint']:
        raise ValueError('Saved Actor parameters differ from actual evaluated Actor')
    return dict(path=str(path), sha256=file_sha256(path), actor_fingerprint=fingerprint,
                iteration=artifact['iteration'], actual_weights_reloaded_on_cpu=True)


def analyze_job(job, output):
    evidence = verify_online(job['run'])
    if evidence != job['evidence']:
        raise ValueError('Completed online evidence changed after queue publication')
    run = Path(job['run'])
    summary = read_verified(evidence['summary'], evidence['summary_sha256'])
    session = Path(evidence['summary']).parent
    config = configuration(session / 'resolved_config.yaml')
    config['runtime']['genmo_device'] = 'cpu'
    actor, _, _ = load_actor(config)
    metrics = run / 'metrics' / (session.name + '.jsonl')
    rows = optimizer_rows([json.loads(line) for line in metrics.read_text().splitlines()])
    accepted = [r for r in rows if r['mode'] == 'dppo_bc']
    if [r['iteration'] for r in accepted] != list(range(1, 9)):
        raise ValueError('Finite online trace must contain all eight accepted updates in order')
    attempts = sum(r['backtracking']['attempt_count'] for r in rows)
    if attempts != summary['final_state']['optimizer_attempts']:
        raise ValueError('Optimizer attempts are missing from actual compute accounting')
    output.mkdir(parents=True, exist_ok=False)
    evaluations = []
    first_by_iteration = {}
    baseline = run / 'evaluation_baseline.json'
    for artifact in summary['evaluation_artifacts']:
        report_path = run / artifact['path']
        report = read_verified(report_path, artifact['sha256'])
        weights = verify_evaluation_weights(run, artifact, report, actor)
        iteration = artifact['iteration']
        if iteration not in (0, 4, 8):
            raise ValueError('Unexpected evaluation iteration in the finite experiment')
        if iteration in first_by_iteration:
            old = first_by_iteration[iteration]
            if report['actor_identity'] != old['actor_identity']:
                raise ValueError('Repeated evaluation at one Actor version changed its weight identity')
            def physical_aggregate(value):
                aggregate = {k: v for k, v in value['aggregate']['overall'].items() if k != 'latency_seconds'}
                aggregate['raw_errors'] = {k: v for k, v in aggregate['raw_errors'].items() if k != 'control_tick'}
                return aggregate
            if physical_aggregate(report) == physical_aggregate(old):
                evaluations.append(dict(iteration=iteration, label=artifact['label'], weights=weights,
                    report=dict(path=str(report_path), sha256=artifact['sha256']),
                    exact_repeat_of=old['label'],
                    scope='same_physical_aggregate_excluding_wall_latency_and_absolute_tick'))
                continue
            # 同版本重评若确有物理差异，也保留完整分析；不能把运行差异当作缺失记录跳过。
        first_by_iteration.setdefault(iteration, dict(report, label=artifact['label']))
        result = analyze(report_path, output / artifact['label'], baseline if iteration else None)
        analysis_path = output / artifact['label'] / 'analysis.json'
        evaluations.append(dict(iteration=iteration, label=artifact['label'], weights=weights,
            analysis=dict(path=str(analysis_path), sha256=file_sha256(analysis_path)),
            aggregate=result['aggregate'], paired=result.get('paired'),
            tracking_eight=result['tracking_eight_control_mean'],
            yaw=result['yaw']['overall'], by_source=result['by_source'],
            tasks=[{k: v for k, v in t.items() if k != 'generated'} for t in result['tasks']],
            retention={k: v for k, v in result['retention'].items() if k != 'samples'},
            evaluation_wall_seconds=result['evaluation_seconds']))
    del actor
    value = dict(name=job['name'], evidence=evidence, rows=rows, evaluations=evaluations,
        accepted_optimizer_steps=len(accepted), attempted_optimizer_steps=attempts,
        rejected_optimizer_attempts=attempts-len(accepted),
        process_wall_seconds=job['elapsed_seconds'], gpu=gpu_peak(run.parent / 'gpu.csv'),
        initial_state=summary['initial_state'], final_state=summary['final_state'],
        current_policy_mc=evidence['mc'], metrics_source=dict(path=str(metrics), sha256=file_sha256(metrics)))
    atomic_json(output / 'online_analysis.json', value)
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queue', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--wait', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(schema='genmo.stage2.uniform_lr_online_review.v1', status='waiting', cases=[],
        queue=str(args.queue), implementation_sha256=file_sha256(__file__),
        scope='eight_step_finite_online_evidence_not_formal_training_promotion')
    completed = set()
    try:
        while True:
            if not args.queue.exists():
                if not args.wait:
                    raise FileNotFoundError(args.queue)
                time.sleep(30)
                continue
            raw = args.queue.read_bytes()
            queue = json.loads(raw)
            result.update(status='running', queue_sha256=hashlib.sha256(raw).hexdigest(), plan=queue['plan'])
            for job in queue['attempts']:
                if job['status'] != 'passed' or job['name'] in completed:
                    continue
                value = analyze_job(job, args.output / job['name'])
                result['cases'].append(value)
                completed.add(job['name'])
                atomic_json(args.output / 'review.json', result)
                print(json.dumps(dict(name=job['name'], accepted=value['accepted_optimizer_steps'],
                    attempts=value['attempted_optimizer_steps'])), flush=True)
            if queue['status'] != 'running':
                result.update(status='completed' if queue['status'] == 'completed_selected_finite_experiments'
                              else 'incomplete_online_queue', queue_status=queue['status'],
                              failures=[j for j in queue['attempts'] if j['status'] != 'passed'])
                break
            if not args.wait:
                result['status'] = 'partial'
                break
            atomic_json(args.output / 'review.json', result)
            time.sleep(30)
    except BaseException as exc:
        result.update(status='analysis_failed', error=repr(exc))
        raise
    finally:
        atomic_json(args.output / 'review.json', result)


if __name__ == '__main__':
    main()
