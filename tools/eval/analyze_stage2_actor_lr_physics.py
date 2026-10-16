"""只读核验统一高学习率实验的完整歌曲、MC价值误差和实际跟踪轨迹。

输入单次32任务评估报告及可选Stage1配对报告；复用已有来源SHA、完整任务和奖励
积分核验，不重新定义奖励。八项Tracking按每个真实控制步的GMT平方误差及原std
计算分项均值，避免用exp(平均误差)冒充平均Tracking。三个关节限位域、完成率、
总回报、时长和每秒回报同时保留，不能把早失败后的幸存区间高分当成收益。

固定Critic参考文件仅使用采样时实际预测值；重新从完整episode积分奖励计算真实
MC标签，拒绝行政截断，绝不使用文件中的GAE target冒充MC。逐条对应session、
episode和decision，不混用Actor行为。原始权威world journal逐记录验证SHA并提取
实际yaw误差；每一条已计入奖励的50Hz控制必须找到唯一的真实四子步记录。
FineDance075/138另存真实机器人、GMT消费参考及生成窗口索引，供失败轨迹比较。
只写独立分析目录，不修改原始证据、训练状态或任何模型参数。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from gem.closedloop.dppo.budget import atomic_json
from gem.closedloop.dppo.journal_codec import decode_payload
from gem.closedloop.dppo.rewards import TRACKING_STD, TRACKING_ERROR_FIELDS
from gem.closedloop.dppo.training_checkpoint import file_sha256
from gem.runtime.trajectory_blocks import expand_feedback
from tools.eval.compare_closedloop_stage10 import _load_periodic,compare_evaluations
from tools.eval.stage10_group_comparison import control_mc_returns


def identity_key(value):
    return tuple(value[k] for k in ('backend_session_id','episode_id','decision_id'))


def mc_predictions(report_path, episodes, expected_report):
    reference_path=Path(report_path).parent/'fixed_diagnostic_reference.json'
    if not reference_path.is_file():
        return dict(available=False,reason='no_matching_initial_prediction_snapshot')
    reference=json.loads(reference_path.read_text())
    if reference['evaluation_identity']!=expected_report['evaluation_identity']:
        raise ValueError('MC prediction snapshot belongs to a different execution identity')
    if reference['initial_iteration']!=expected_report['iteration']:
        return dict(available=False,reason='old_prediction_snapshot_cannot_label_new_actor_behavior')
    predictions={}
    for rank in reference['ranks']:
        info=rank.get('critic')
        if info is None:continue
        if file_sha256(info['path'])!=info['sha256']:
            raise ValueError('Critic observation/prediction snapshot bytes changed')
        values=torch.load(info['path'],map_location='cpu',weights_only=False,mmap=True)
        for record in values['records']:
            key=identity_key(record['identity'])
            if key in predictions:raise ValueError('Duplicate causal prediction identity')
            predictions[key]=float(record['initial_prediction'])
    pairs=[];excluded=[]
    for episode in episodes:
        decisions=episode['decisions']
        if episode['truncated'] or not decisions or not decisions[-1]['terminated']:
            excluded.append(episode['task_id']);continue
        target=control_mc_returns([[v['reward'] for v in d['controls']] for d in decisions],
                                 event_rewards=[d['zero_step_event_reward'] for d in decisions])
        for decision,value in zip(decisions,target):
            key=identity_key(decision['identity'])
            if key not in predictions:
                raise ValueError('A complete MC state lacks its causal initial Critic prediction')
            pairs.append(dict(task_id=episode['task_id'],source=episode['dataset'],
                decision=decision['decision_index'],prediction=predictions[key],mc_return=float(value)))
    def summarize(rows):
        p=np.asarray([r['prediction'] for r in rows]);y=np.asarray([r['mc_return'] for r in rows])
        if not len(rows):return dict(states=0,mse=None,explained_variance=None)
        error=p-y;var=float(np.var(y))
        return dict(states=len(rows),mse=float(np.mean(error**2)),bias=float(np.mean(error)),
                    explained_variance=1-float(np.var(error))/var if var>1e-12 else None)
    return dict(available=True,overall=summarize(pairs),
        by_source={s:summarize([r for r in pairs if r['source']==s]) for s in sorted({r['source'] for r in pairs})},
        excluded_administrative_or_incomplete_tasks=excluded,rows=pairs,
        scope='same_behavior_complete_MC_no_bootstrap_no_extra_dt_not_stored_GAE_targets')


def physical_controls(episodes, destination):
    tasks={e['task_id']:e for e in episodes};wanted={};journals=defaultdict(set)
    for episode in episodes:
        for decision in episode['decisions']:
            identity=decision['identity'];key=(identity['backend_session_id'],identity['episode_id'])
            for control in decision['controls']:
                token=(*key,int(control['tick']))
                if token in wanted:raise ValueError('Duplicate counted physical control')
                wanted[token]=episode['task_id']
            journals[decision['authoritative_journal_path']].add(key)
    found={};traces=defaultdict(list);fingerprints=[]
    for path,identities in journals.items():
        seen_mutations=set()
        connection=sqlite3.connect('file:'+str(Path(path).resolve())+'?mode=ro',uri=True)
        try:
            for digest,payload in connection.execute('SELECT sha256,payload FROM replies ORDER BY rowid'):
                reply=decode_payload(payload,sha256=digest)
                for lane in reply.get('result',{}).get('replies',[]):
                    envelope=lane.get('result',{})
                    if envelope.get('operation')!='advance':continue
                    mutation=(envelope['backend_session_id'],envelope['mutation_seq'])
                    if mutation in seen_mutations:continue
                    seen_mutations.add(mutation)
                    body=envelope['result']
                    if (envelope['backend_session_id'],body.get('episode_id')) not in identities:continue
                    for row in expand_feedback(body).get('trace',[]):
                        token=(row['backend_session_id'],row['episode_id'],int(row['tick']))
                        task=wanted.get(token)
                        if task is None:continue
                        if token in found or row['completed_physics_steps']!=4 or not row['state_valid']:
                            raise ValueError('Duplicate or incomplete real four-substep control')
                        yaw=float(row['errors']['yaw_error_rad'])
                        if not np.isfinite(yaw):raise ValueError('Nonfinite actual yaw error')
                        found[token]=yaw
                        e=tasks[task]
                        if e['dataset']=='FineDance' and e['sample_id'] in ('075','138'):
                            traces[task].append(dict(tick=token[2],yaw=yaw,
                                actual=row['actual_joint_pos_gmt'],reference=row['reference']['joint_pos'],
                                actual_qpos=row['actual_qpos'],reference_quat=row['motion_tracking']['reference']['anchor_quat_w'],
                                actual_quat=row['motion_tracking']['actual']['anchor_quat_w']))
        finally:connection.close()
        fingerprints.append(dict(path=path,sha256=file_sha256(path)))
    if set(found)!=set(wanted):
        raise ValueError(f'Missing actual physical controls: {len(found)} / {len(wanted)}')
    yaw_by_task=defaultdict(list)
    for token,value in found.items():yaw_by_task[wanted[token]].append(value)
    saved=[]
    for task,records in traces.items():
        records.sort(key=lambda r:r['tick'])
        arrays={k:np.asarray([r[k] for r in records]) for k in records[0]}
        if len(records)>1 and not np.all(np.diff(arrays['tick'])==12):
            raise ValueError('FineDance trajectory loses real 50Hz continuity')
        path=destination/(task+'.npz')
        with path.open('xb') as stream:np.savez_compressed(stream,**arrays)
        saved.append(dict(task_id=task,sample=tasks[task]['sample_id'],seed=tasks[task]['seed'],
            path=str(path),sha256=file_sha256(path),scope='real_Isaac_actual_and_consumed_reference_50Hz',
            generated_plans=[dict(decision=d['decision_index'],raw_sample_path=d['raw_sample_path'],
                                 limits=d['generated_joint_limit_diagnostics']) for d in tasks[task]['decisions']]))
    def stats(values):
        return dict(mean=float(np.mean(values)),p95=float(np.quantile(values,.95)),max=float(np.max(values))) if values else None
    return dict(controls=len(found),physics_substeps=4*len(found),overall=stats(list(found.values())),
                by_task={k:stats(v) for k,v in yaw_by_task.items()},focused_trajectories=saved,journals=fingerprints)


def music_cluster_interval(comparison,metric='episode_reward',seed=713):
    groups=defaultdict(list)
    for row in comparison['pairs']:
        groups[(row['dataset'],row['sample_id'])].append(row['delta'][metric])
    means=np.asarray([np.mean(v) for v in groups.values()])
    rng=np.random.default_rng(seed)
    samples=means[rng.integers(0,len(means),(10000,len(means)))].mean(1)
    return dict(independent_music=len(means),paired_mean=float(means.mean()),
                lower95=float(np.quantile(samples,.025)),upper95=float(np.quantile(samples,.975)),
                bootstrap='resample_music_groups_keep_two_seeds_together_not_32_independent_songs')


def analyze(report_path,output,baseline=None):
    output.mkdir(parents=True,exist_ok=False)
    report,plan,episodes,source=_load_periodic(report_path)
    if plan['task_count']!=32 or plan['selected_sample_count']!=16 or plan['episode_seconds'] is not None:
        raise ValueError('Expected 16 independent music tasks times two seeds through real termination')
    controls=[v for e in episodes for d in e['decisions'] for v in d['controls']]
    scores={name:float(np.mean([np.exp(-r['raw_errors'][TRACKING_ERROR_FIELDS[name]]/std**2)
                               for r in controls])) for name,std in TRACKING_STD.items()}
    overall=report['aggregate']['overall']
    summaries=list(Path(report_path).parent.glob('sessions/*/summary.json'))
    initial_state=(json.loads(summaries[0].read_text()).get('initial_state',{}) if len(summaries)==1 else {})
    initialized=bool(initial_state.get('critic_initialization') or initial_state.get('mc_preheat_complete'))
    # evaluate-only只加载Actor，原入口不会导入MC Critic。不能把随机Value Head的
    # 误差用于批评已预热Critic；有限在线实验另有当前策略的完整MC校准。
    mc=(mc_predictions(report_path,episodes,report) if initialized else dict(available=False,
        reason='pure_actor_evaluation_did_not_initialize_preheated_critic',
        required_evidence='finite_online_current_policy_complete_MC_calibration'))
    tasks=[]
    for e in episodes:
        actual=e.get('motion_diagnostic',{});generated=e.get('generated_reference_diagnostic',{})
        tasks.append(dict(task_id=e['task_id'],source=e['dataset'],sample=e['sample_id'],seed=e['seed'],
            music_seconds=e['full_music_seconds'],reward=e['reward_sum'],reason=e['end_reason'],
            physical_failure=e['physical_failure'],truncated=e['truncated'],
            controls=sum(d['executed_control_steps'] for d in e['decisions']),
            motion=actual,generated=generated))
    result=dict(schema='genmo.stage2.uniform_lr_complete_physical_analysis.v1',source=source,
        actor=report['actor_identity'],plan_sha256=plan['plan_sha256'],aggregate=overall,
        tracking_eight_control_mean=scores,by_source=report['by_source'],tasks=tasks,
        retention=report.get('stage1_retention'),evaluation_seconds=report.get('wall_seconds'),
        mc=mc,yaw=physical_controls(episodes,output),
        analysis_implementation_sha256=file_sha256(__file__),
        claim_scope='fixed_16_music_32_full_available_paired_episodes_not_full_dataset_acceptance')
    if baseline:
        comparison=compare_evaluations(baseline,report_path,mode='periodic_subset')
        atomic_json(output/'paired.json',comparison)
        result['paired']=dict(failures=comparison['overall']['failure_pair_counts'],
            reward=music_cluster_interval(comparison),seconds=music_cluster_interval(comparison,'executed_seconds'),
            task_metrics=comparison['overall']['task_weighted_metrics'])
    atomic_json(output/'analysis.json',result)
    print(json.dumps(dict(output=str(output),completion=overall['music_completion_count'],
        reward=overall['reward_sum'],seconds=overall['executed_seconds'],mc=result['mc'].get('overall'),
        yaw=result['yaw']['overall']),ensure_ascii=False),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    inputs=parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--report',type=Path)
    inputs.add_argument('--queue',type=Path,help='只读跟随有限物理队列，CPU分析与后续GPU任务并行')
    parser.add_argument('--baseline',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();torch.set_num_threads(2)
    if args.report:
        if args.resume:raise ValueError('Single analysis output is immutable; choose a fresh directory')
        analyze(args.report,args.output,args.baseline)
        return
    if args.baseline:raise ValueError('Queue baseline is its identified Stage1 task')
    if args.resume:
        index=json.loads((args.output/'index.json').read_text())
        if index['queue']!=str(args.queue.resolve()):raise ValueError('Different physics queue')
    else:
        args.output.mkdir(parents=True,exist_ok=False)
        index=dict(schema='genmo.stage2.uniform_lr_physical_analysis_queue.v1',
                   queue=str(args.queue.resolve()),attempts=[],status='running')
    while True:
        queue=json.loads(args.queue.read_text())
        passed=[r for r in queue['attempts'] if r.get('status')=='passed']
        baseline=next((r['evidence']['report'] for r in passed if r['name']=='stage1'),None)
        for job in passed:
            previous=[r for r in index['attempts'] if r['name']==job['name']]
            if previous and previous[-1]['status']=='passed':
                if file_sha256(previous[-1]['analysis'])!=previous[-1]['sha256']:
                    raise ValueError('Published analysis bytes changed')
                continue
            destination=args.output/job['name']/f'attempt{len(previous)+1:02d}'
            item=dict(name=job['name'],aliases=job['aliases'],status='running',started=time.time())
            index['attempts'].append(item);atomic_json(args.output/'index.json',index)
            try:
                result=analyze(Path(job['evidence']['report']),destination,
                               None if job['name']=='stage1' else Path(baseline))
                path=destination/'analysis.json'
                item.update(status='passed',analysis=str(path),sha256=file_sha256(path),
                            aggregate=result['aggregate'],paired=result.get('paired'))
            except BaseException as exc:
                item.update(status='failed',error=repr(exc));index['status']='analysis_failed'
                raise
            finally:
                item['elapsed_seconds']=time.time()-item['started']
                atomic_json(args.output/'index.json',index)
        if queue['status'] in ('completed_all_distinct_survivors','interrupted','stopped_with_failure'):
            index['status']='completed' if queue['status']=='completed_all_distinct_survivors' else 'physics_queue_incomplete'
            atomic_json(args.output/'index.json',index)
            return
        time.sleep(30)


if __name__=='__main__':main()
