"""从原Stage1重新初始化的独立50更新音乐漂移实验。

加载原未更新Actor而不导入旧AdamW/Critic，保持冻结Stage1/SM和真实墙钟异步采集。
每个候选用当前模型真实采集的多个时间anchor；全曲状态持续推进，不从低速启动段
反复重置。固定8曲真实Newton评价在0/10/25/50执行，延迟事件与旧基线配对。
0版本仅在逐张量核对原始Actor相同时复用已经保存的8曲完整物理证据，明确记录来源。
独立目录保存全部候选、完整训练checkpoint和评价；负结果不删除、不提前退出。
支持域/数值异常仍留证据；最大尝试次数是本次诊断计算预算，不是质量早停。
"""
import gc
import json
import os
from pathlib import Path
import time
import torch
from omegaconf import OmegaConf

from .application import SupervisionStream,build_stage1_losses,task_inputs,behavior_config,provenance
from .checkpoint import save_checkpoint,capture_rng
from .formal import run_evaluation_processes,write_json
from .replay import RecentReplay
from .trainer import Stage2Trainer
from .validation import ModelSupport,UnsupportedModelState
from .sample_distribution import describe_sample


def run_experiment(collector,actor,critic,engine,original,cfg,tasks,output,device):
    settings=cfg['diagnostic'];source=Path(settings['source_run'])
    ck=torch.load(source/'checkpoints/initial_actor.pt',map_location='cpu',weights_only=False,mmap=True)
    actor.load_state_dict(ck['actor'],strict=True)
    assert all(torch.equal(v.detach().cpu(),ck['actor'][k]) for k,v in actor.state_dict().items())
    del ck;actor.eval()
    trainer=Stage2Trainer(actor,critic,engine,build_stage1_losses(actor.stage1,original).to(device),cfg['training'])
    supervision=SupervisionStream(cfg['stage1']['data_config'],cfg['training']['supervision_batch'],42,device)
    replay=RecentReplay(cfg['training']['replay_capacity'])
    engine.support_guard=ModelSupport(json.loads((source/'state_support_contract.json').read_text()),device=device)
    collector.anchor_horizon=16;collector.record_controls=True;collector.record_contacts=True
    collector.timing_mode='wall_clock_async';collector.noise_seed=42
    config_path=output/'experiment.yaml';OmegaConf.save(OmegaConf.create(cfg),config_path)
    (output/'checkpoints').mkdir();(output/'evaluations').mkdir()
    baseline=source/'evaluations/fixed_delay_replay/initial_genmo_baseline/physical_summary.json'
    write_json(output/'evaluations/baseline_reuse.json',{'actor_tensor_identity_verified':True,
        'source':str(baseline),'same_music_initial_seed_noise_and_fixed_delays':True,
        'fresh_simulation':False,'reason':'复用逐张量相同初始模型已有8曲完整Newton轨迹；新增episode初态变换在配对初态为恒等'})
    behavior=behavior_config(cfg,actor,engine.adapter);source_info=provenance(cfg)
    task_cursor=0;collector.state=None;failed_episodes=[];evaluation_rows=[]
    # 每次完成更新后旧请求不再作为当前Actor的优化样本；实际历史与有效参考照常连续。
    def save(update):
        path=output/'checkpoints'/f'actor_{update:06}.pt'
        save_checkpoint(path,trainer,replay,behavior_config=behavior,
            runtime_state={'supervision':supervision.state_dict(),'task_cursor':task_cursor,
                'collector_restart_requires_new_episode':True,'experiment_attempt_budget':settings['maximum_attempts']},provenance=source_info)
        return path
    save(0)
    while trainer.accepted_actor_updates<50 and trainer.update_index<settings['maximum_attempts']:
        collector.actor_version=trainer.accepted_actor_updates
        collector.ready_anchors.clear();samples=[];intervals=[];sampling=[]
        while len(samples)<cfg['training']['actor_microbatch']:
            if collector.state is None or collector.terminal:
                if collector.state is not None:
                    failed_episodes.append({'task':collector.state.task_id,'end_tick':collector.state.tick,
                        'actor_version':collector.actor_version,'terminal':collector.terminal})
                task=tasks[task_cursor%len(tasks)];task_cursor+=1
                music,initial=task_inputs(task,collector.adapter,device);collector.reset(music,initial)
            collected=collector.collect(cfg['training']['collect_decisions']);intervals.extend(collected)
            available=list(collector.ready_anchors);collector.ready_anchors.clear()
            # 同一批尽量跨本次真实采集区间；没有伪造高速标签，也不放宽WM支持域。
            available.sort(key=lambda x:x.state.tick)
            if len(available)>cfg['training']['actor_microbatch']:
                indices=torch.linspace(0,len(available)-1,cfg['training']['actor_microbatch']).round().long().tolist()
                available=[available[i] for i in indices]
            for sample in available:
                if sample.generation.request.actor_version!=trainer.accepted_actor_updates:continue
                row=describe_sample(sample,engine.adapter)
                try:
                    with torch.no_grad():engine.rollout_anchor(sample,plan_override=sample.generation.plan,support_check_only=True)
                    samples.append(sample);row['support']='legal'
                except UnsupportedModelState as error:
                    row.update(support='outside',error=str(error),details=getattr(error,'details',None))
                sampling.append(row)
            if len(intervals)>100 and not samples:
                # 保留失败采集证据并进入下一首；不把静止样本代替失败动作，不更改边界。
                failed_episodes.append({'task':collector.state.task_id,'end_tick':collector.state.tick,'reason':'100_decisions_no_legal_anchor'})
                collector.terminal=True;intervals=[]
        samples=samples[:cfg['training']['actor_microbatch']]
        result=trainer.update_anchors(samples,supervision.next(),physical_intervals=intervals)
        result['training_anchor_ticks']=[s.state.tick for s in samples]
        result['training_tasks']=[s.state.task_id for s in samples]
        result['sampling']=sampling
        with (output/'training.jsonl').open('a') as f:f.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n')
        print(json.dumps({k:result[k] for k in ['update','accepted_actor_updates','accepted_execution_feedback_update','original_music_execution_loss','episode_music_execution_loss','update_wall_seconds']},ensure_ascii=False),flush=True)
        update=trainer.accepted_actor_updates
        write_json(output/'status.json',{'accepted':update,'attempts':trainer.update_index,'task_cursor':task_cursor,'failed_episodes':failed_episodes,'evaluations':evaluation_rows,'quality_exit_enabled':False})
        if result['accepted_execution_feedback_update'] and update in (10,25,50):
            path=save(update)
            eval_cfg=json.loads(json.dumps(cfg));eval_cfg['evaluation']['timing_mode']='fixed_delay_replay'
            eval_path=output/'evaluations'/f'actor_{update:06}';eval_config=output/f'evaluation_{update:06}.yaml'
            OmegaConf.save(OmegaConf.create(eval_cfg),eval_config)
            if collector.worker.future is not None:raise RuntimeError("候选更新边界仍有在途请求，不能覆盖")
            torch.cuda.empty_cache();gc.collect()
            run_evaluation_processes(eval_cfg,eval_config,path,eval_path,'full',os.environ.copy())
            summary=json.loads((eval_path/'physical_summary.json').read_text())
            evaluation_rows.append({'update':update,'completed':summary['completed'],'episodes':summary['episodes'],
                'dance_loss':sum(r['physical_metrics_mean']['dance_loss'] for r in summary['physical_results'])/len(summary['physical_results']),
                'physical_summary':str(eval_path/'physical_summary.json')})
            write_json(output/'evaluation_progress.json',evaluation_rows)
        # 长episode保存全部生成器事件会占用内存，但作为真实诊断证据保留；不清除有效缓存。
        del samples,intervals;gc.collect()
    save(trainer.accepted_actor_updates) if trainer.accepted_actor_updates not in (0,10,25,50) else None
    write_json(output/'completed.json',{'accepted_updates':trainer.accepted_actor_updates,'attempts':trainer.update_index,
        'evaluations':evaluation_rows,'large_formal_training_started':False,'quality_exit_enabled':False})
