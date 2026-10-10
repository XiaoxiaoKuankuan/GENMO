"""只读汇总Stage10有限验收的真实训练、固定评估和有界归档指标。

读取正式训练器发布的JSONL、逐轮归档清单与独立评估报告，不加载模型、不使用
GPU、不生成或修补训练数据。主体、包含封存/归档入队的外层墙钟和结束归档排空
分别统计；同会话固定工作量诊断轮显式排除普通轮P50/P95，首次采集/编译轮单列。
采样使用八rank中最慢的持久化完成时间，禁止把更短的collector本体冒充全采样。

此工具只整理已有证据，不替代SHA回放、恢复审计、学习效果验收或多随机种子
实验。容量按实际归档字节计算500轮投影并明确标注估计，不把1.883TB规划值
当成已写入量。输入只读，报告必须写入新的路径，防止覆盖原模型/日志/验收结果。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def distribution(values):
    ordered=sorted(float(x) for x in values)
    def percentile(q):
        if not ordered:return None
        position=(len(ordered)-1)*q
        low=int(position);high=min(low+1,len(ordered)-1)
        return ordered[low]+(ordered[high]-ordered[low])*(position-low)
    return dict(n=len(ordered),p50=percentile(.5),p95=percentile(.95),
        minimum=min(ordered) if ordered else None,maximum=max(ordered) if ordered else None,
        mean=sum(ordered)/len(ordered) if ordered else None)


def summarize(run):
    records=[json.loads(line) for path in sorted((run/'metrics').glob('*.jsonl'))
             for line in path.read_text().splitlines()]
    accepted=sorted((r for r in records if r.get('event')=='iteration_accepted'),
                    key=lambda r:(r.get('time_utc',''),r['session_id'],r['iteration']))
    outer={(r['session_id'],r['iteration']):r for r in records if r.get('event')=='iteration_walltime'}
    seen=set();iterations=[]
    for r in accepted:
        session=r['session_id'];cold=session not in seen;seen.add(session)
        collectors=r['collectors'];actor=r['actor'];kl=r['kl'];critic=r['critic']
        row=dict(session_id=session,iteration=r['iteration'],initial_round_of_session=cold,
            diagnostic=bool(r.get('fixed_work_probe')),real_chains=len(r['global_manifest']),
            sample_seconds=max(c['through_persistence_seconds'] for c in collectors),
            collector_body_seconds=max(c['seconds'] for c in collectors),
            core_seconds=r['seconds'],outer_seconds=outer.get((session,r['iteration']),{}).get('seconds'),
            **r['timings'],actor_steps=actor['optimizer_steps'],bc_global_samples=actor['bc_global_samples'],
            internal_visits=actor['applied_internal_sample_visits'],coverage=actor['applied_unique_chain_fraction'],
            planned_internal_visits=actor['planned_internal_sample_visits'],
            internal_visit_fraction=actor['applied_internal_sample_visits']/actor['planned_internal_sample_visits'],
            discarded_pending_minibatch=actor.get('discarded_pending_minibatch'),
            selected_denoising_steps=actor['denoising_steps_per_chain'],full_denoising_steps=actor['full_denoising_steps'],
            final_joint_kl=kl['mean_joint_kl'],final_kl_fresh_internal_forwards=kl['fresh_internal_forwards'],
            actual_output_change=kl.get('effective_mean_change'),per_step_kl=kl['per_denoising_step'],
            physical_failures=r['physical_failures'],executed_seconds=r['executed_seconds'],
            reward_per_second=r['reward_per_second'],transitions_per_second=r['transitions_per_second'],
            trained_unique_chains_per_second=len(r['global_manifest'])*actor['applied_unique_chain_fraction']/r['seconds'],
            applied_internal_visits_per_second=actor['applied_internal_sample_visits']/r['seconds'],
            boundary_wait_controls=sum(c.get('administrative_drain_controls',0) for c in collectors),
            boundary_wait_reward=sum(c.get('administrative_drain_reward',0.) for c in collectors),
            generation_batch_histogram={},
            actor_step_diagnostics=[{key:step.get(key) for key in (
                'optimizer_step','epoch','internal_transitions','full_internal_transitions',
                'denoising_sampling','learning_rates','pre_minibatch_mean_joint_kl',
                'mean_ratio','clip_fraction','objective_clipped_fraction','per_denoising_step',
                'ppo_only_gradient_norm','total_gradient_norm','gradient_contributions',
                'grad_clip_factor','parameter_update_observation','bc')} for step in actor['steps']],
            actor_early_stop_reason=actor.get('early_stop_reason'),
            gae_global_by_rank=[dict(rank=c['rank'],**c.get('gae_global',{})) for c in collectors],
            gae_fragment_lengths=[length for c in collectors for length in c.get('gae_contiguous_fragment_lengths',[])],
            boundary_wait_by_rank=[dict(rank=c['rank'],environments=c.get('boundary_wait_by_environment',[]))
                for c in collectors],
            first_gmt_consumption_by_rank=[dict(rank=c['rank'],references=c.get('new_reference_first_gmt_input',[]))
                for c in collectors],
            collection_phases_by_rank=[{key:c.get(key) for key in (
                'rank','control_steps','seconds','local_compute_seconds','through_persistence_seconds',
                'world_timing','server_rpc_transport_timing','metric_processing_seconds',
                'journal_and_budget_close_with_rank_wait_seconds',
                'snapshot_seconds','learning_data_persistence_seconds','synchronization_wait_seconds',
                'generation_timing_totals','generation_pipeline_totals','generation_batch_wall_seconds',
                'terminated_transitions','truncated_transitions','termination_reasons')} for c in collectors],
            columnar_reward_timing_by_rank=[dict(rank=c['rank'],
                contract=c.get('columnar_reward_timing_contract','legacy_object_counter_delta_may_reset'),
                usable_for_breakdown=c.get('columnar_reward_timing_contract')=='per_consume_call_across_episode_replacements.v2'
                    and all(v>=0 for v in c.get('columnar_reward_cpu_timings',{}).values()),
                seconds=c.get('columnar_reward_cpu_timings',{})) for c in collectors],
            critic={k:critic.get(k) for k in ('initial_mse','mse','initial_explained_variance','explained_variance')},
            tensor_cache_by_rank=r['tensor_cache_by_rank'],budget_used=r['budget']['used'],
            probability_check=r['probability_check'])
        for c in collectors:
            for size,count in c.get('effective_generation_batch_histogram',{}).items():
                row['generation_batch_histogram'][size]=row['generation_batch_histogram'].get(size,0)+count
        iterations.append(row)
    hot=[r for r in iterations if not r['initial_round_of_session'] and not r['diagnostic']]
    ordinary={key:distribution(r[key] for r in hot if r[key] is not None) for key in
        ('sample_seconds','actor_seconds','kl_seconds','core_seconds','outer_seconds','transitions_per_second',
         'trained_unique_chains_per_second','applied_internal_visits_per_second')}
    archived=[r for r in records if r.get('event')=='execution_archived']
    completions=[r for r in records if r.get('event')=='execution_archive_completed']
    curves=[json.loads(line) for line in (run/'curves.jsonl').read_text().splitlines()]
    queue=[(key,value) for record in curves for key,value in record['metrics'].items()
           if key.startswith('archive_queue/')]
    compressed=[r['archive_size_bytes'] for r in archived];raw=[r['original_size_bytes'] for r in archived]
    archive=dict(completed=len(completions),failed=sum(r['status']!='passed' for r in completions),
        raw_bytes=sum(raw),compressed_bytes=sum(compressed),raw_per_round=distribution(raw),
        compressed_per_round=distribution(compressed),
        seconds=distribution(r['total_seconds'] for r in completions),
        stage_seconds={key:distribution(r['stage_seconds'].get(key,0.) for r in completions)
            for key in sorted({k for r in completions for k in r['stage_seconds']})},
        peak_pending=max((v for k,v in queue if k.endswith('/peak_pending_count')),default=None),
        submit_wall_seconds=distribution(v for k,v in queue if k.endswith('/submit_wall_seconds')),
        shutdown_drain_seconds=[r['seconds'] for r in records if r.get('event')=='archive_drain'],
        projected_500_compressed_bytes=sum(compressed)/len(compressed)*500 if compressed else None,
        projection_scope='observed_mean_times_500_excludes_checkpoints_eval_margin_not_actual_written_bytes')
    evaluations=[]
    for path in sorted(run.glob('sessions/*/evaluations/*.json')):
        value=json.loads(path.read_text())
        evaluations.append(dict(path=str(path),status=value.get('status'),
            aggregate=value.get('aggregate'),by_source=value.get('by_source'),
            source_balanced_reward=value.get('source_balanced_reward'),
            mean_executed_seconds=value.get('mean_executed_seconds'),
            physical_failure_count=value.get('physical_failure_count'),
            fixed_actor_drift=value.get('fixed_actor_drift'),fixed_critic_diagnostic=value.get('fixed_critic_diagnostic'),
            paired_baseline=value.get('paired_baseline'),episode_metrics=value.get('episode_metrics'),
            evaluation_identity=value.get('evaluation_identity'),iteration=value.get('iteration'),
            session_id=value.get('session_id'),
            wall_seconds=value.get('wall_seconds'),
            task_count=value.get('task_count'),plan_sha256=value.get('plan_sha256')))
    return dict(schema='stage10.pipeline_measurements.v1',run=str(run),iterations=iterations,
        warm_ordinary=ordinary,archive=archive,evaluations=evaluations,
        acceptance_not_inferred_from_timing=True,raw_records_unchanged=True,
        nested_phase_seconds_must_not_be_added=True,
        percentile_scope='observed_finite_rounds_linear_interpolation_not_confidence_interval')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=summarize(args.run.resolve(strict=True))
    with args.output.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(warm_ordinary=result['warm_ordinary'],archive=result['archive']),ensure_ascii=False))


if __name__=='__main__':main()
