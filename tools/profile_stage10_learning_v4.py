"""服务器1八卡、完整160条真实机器人条件的有界离线PPO计算对照，不发布策略。

输入来自封存真实rollout的160份条件与固定优势。首先在新数值身份下以同一初始
模型和显式独立种子实际采样新链，之后所有候选固定这些x_k/x_next/old probability，
绝不改写旧行为概率为学习重算值。新链是离线数值校准，不声称这些新动作已执行。
每个optimizer minibatch固定80条完整链/1600内部转移，2epoch最多4次真实Adam。
BC由另外的回归覆盖，本工具明确排除BC和GMT；软停止仅在标记离线计算基准时关闭，
最终全量KL始终实际执行，但不会发布任何权重或冒充正式接受轮。输出记录输入SHA、
计算量、微批、PPO及KL调用、参数/Adam差异和冻结不变性。临时更新后恢复初始状态。
必须使用torchrun八个rank，每卡仅持有20条真实条件，梯度按全局1600内部样本SUM。
同时覆盖一次Adam数值对照、四次更新离线定额测速和保留原软停止的真实更新流程；
三者分开报告，定额结果绝不冒充通过硬KL的正式轮。每轮输出最慢rank的墙钟时间。
"""
import argparse
import copy
import json
import os
from datetime import timedelta
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from gem.closedloop.training import build_stage1_actor
from gem.closedloop.dppo.policy import DPPODiffusionPolicy
from gem.closedloop.dppo.dual_collector import split_trace
from gem.closedloop.dppo.tensor_cache import RolloutTensorCache
from gem.closedloop.dppo.updater_v2 import actor_update_v2, analytic_kl_local, probability_check_local, balanced_epoch_order
from gem.closedloop.dppo.distributed_runtime import DistributedCollectives
from tools.train_closedloop_stage10_8gpu import _available_gpus
from gem.closedloop.dppo.parallel_support import cpu_snapshot
from gem.closedloop.dppo.performance import PhaseProfiler,activate,deactivate
from gem.closedloop.dppo.run_management import file_sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('stage1-config','weights','assets','rollout','output'):
        parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    rank,world,local_rank=(int(os.environ.get(k,'-1')) for k in ('RANK','WORLD_SIZE','LOCAL_RANK'))
    if world!=8 or rank!=local_rank:raise ValueError('Use single-node torchrun with exactly eight GPUs')
    dist.init_process_group('gloo',timeout=timedelta(minutes=15))
    startup=[None]
    if rank==0:
        try:
            if args.output.exists():raise FileExistsError(args.output)
            startup[0]=dict(devices=_available_gpus())
        except Exception as error:startup[0]=dict(error=str(error))
    dist.broadcast_object_list(startup,0)
    if 'error' in startup[0]:raise RuntimeError(startup[0]['error'])
    args.device=f'cuda:{local_rank}'
    torch.cuda.set_device(local_rank)
    group=dist.new_group(backend='nccl',timeout=timedelta(minutes=10))
    collective=DistributedCollectives(rank,world,tensor_group=group,device=args.device,measure_gradient_communication=True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    config=OmegaConf.create(json.loads(args.stage1_config.read_text()))
    config.endecoder.stats_path=str(args.assets/'qpos30_train_stats.json')
    config.endecoder.kinematics_path=str(args.assets/'bumi_kinematics_robot_retargeter_fe934_v1.json')
    data=OmegaConf.create(dict(qpos30_stats=dict(path=config.endecoder.stats_path),
        dataset_defaults=dict(kinematics_path=config.endecoder.kinematics_path),sample_contract=dict(history_steps=50),datasets={}))
    actor=build_stage1_actor(config,data)
    saved=torch.load(args.weights,map_location='cpu',mmap=True,weights_only=False)['actor']
    actor.load_state_dict(saved,strict=True); actor=actor.float().eval().to(args.device)
    policy=DPPODiffusionPolicy(actor,cfg_batch=True,numerical_layout='sample_matrix_bmm_fp32.v1',defer_checks=True)
    source=torch.load(args.rollout,map_location='cpu',weights_only=False)
    if len(source['traces'])!=160: raise ValueError('Benchmark requires exactly 160 real contexts')
    rows=[]
    start=time.perf_counter()
    for i in range(rank*20,rank*20+20,2):
        batch={k:torch.cat([t['conditions'][k] for t in source['traces'][i:i+2]]).to(args.device)
               for k in source['traces'][i]['conditions']}
        trace=policy.sample_rollout(batch,generator=[torch.Generator(device=args.device).manual_seed(14000+j) for j in (i,i+1)])
        for j in range(2):
            single=cpu_snapshot(split_trace(trace,j,2))
            rows.append(SimpleNamespace(context=single['conditions'],chain=single['chain'][0],old_log_prob=single['old_log_probs'][0],
                free_mask=single['free_mask'][0],identity=dict(policy_version=0),transition_valid=True,
                metadata=dict(sampler_trace=single,remaining_music_seconds=10.)))
        if rank==0:print(json.dumps(dict(sampled_per_rank=i+2)),flush=True)
    sampling=time.perf_counter()-start
    targets=dict(advantages=source['targets'][sorted(source['targets'])[rank]]['advantages'])
    local_manifest=[dict(owner_rank=rank,local_index=i,valid=True,has_free=bool(row.free_mask.any())) for i,row in enumerate(rows)]
    manifest=[item for shard in collective.all_gather_object(local_manifest) for item in shard]
    orders=[balanced_epoch_order(list(range(160)),manifest,torch.Generator().manual_seed(61+e)) for e in range(2)]
    cache=RolloutTensorCache(rows,targets,args.device)
    report=dict(scope='eight_gpu_global160_local20_offline_no_BC_no_GMT_no_publication',
        new_chain_scope='actual_new_kernel_sampling_on_frozen_real_robot_conditions_not_executed',
        device=torch.cuda.get_device_name(),input_sha256=file_sha256(args.rollout) if rank==0 else None,
        weights_sha256=file_sha256(args.weights) if rank==0 else None,
        new_chain_sampling_seconds=sampling,results=[])
    references={}
    candidates=[(micro,'post_step_full',False,'one_step_numerical',0) for micro in (2,4,8,16,32)]
    for repeat in range(args.repeats):
        candidates += [(m,mode,detail,scope,repeat) for m,mode,detail,scope in (
            (2,'post_step_full',False,'four_step_offline'),(4,'post_step_full',False,'four_step_offline'),
            (8,'post_step_full',False,'four_step_offline'),(16,'post_step_full',False,'four_step_offline'),
            (32,'post_step_full',False,'four_step_offline'),(16,'pre_step_plus_final',False,'four_step_offline'),
            (16,'post_step_full',True,'four_step_offline'),(16,'post_step_full',False,'soft_stop_enabled'),
            (16,'pre_step_plus_final',False,'soft_stop_enabled'))]
    for micro,mode,detailed,scope,repeat in candidates:
        actor.load_state_dict(saved); actor.zero_grad(set_to_none=True)
        optimizer=torch.optim.AdamW([p for p in actor.parameters() if p.requires_grad],lr=5e-9,weight_decay=0.)
        profile=PhaseProfiler(args.device,rank,detailed=detailed); token=activate(profile)
        try:
            check=probability_check_local(policy,rows,denoising_microbatch=micro,tensor_cache=cache,
                distributed=collective,global_manifest=manifest)
            sink={}
            torch.cuda.synchronize();collective.barrier();start=time.perf_counter()
            update=actor_update_v2(policy,optimizer,rows,targets,actor_minibatch_internal_transitions=1600,
                denoising_microbatch=micro,epoch_orders=orders,soft_kl_limit=.015 if scope=='soft_stop_enabled' else None,
                max_optimizer_steps=1 if scope=='one_step_numerical' else 4,tensor_cache=cache,
                distributed=collective,global_manifest=manifest,
                gradient_diagnostics=False,kl_check_mode=mode,kl_cache_sink=sink)
            torch.cuda.synchronize(); actor_end=time.perf_counter()
            final=analytic_kl_local(policy,rows,denoising_microbatch=micro,tensor_cache=cache,reuse_cache=sink.get('cache'),
                distributed=collective,global_manifest=manifest)
            torch.cuda.synchronize(); end=time.perf_counter()
            weights=cpu_snapshot(actor.state_dict()) if rank==0 else {}
            state=cpu_snapshot(optimizer.state_dict()) if rank==0 else {}
            delta=None
            reference=references.get(scope)
            if rank==0 and reference is None:references[scope]=(weights,state)
            elif rank==0:
                delta=dict(parameter_max_abs=max(float((v.double()-reference[0][k].double()).abs().max()) for k,v in weights.items()),
                    optimizer_max_abs=max(float((v.double()-reference[1]['state'][i][k].double()).abs().max())
                        for i,s in state['state'].items() for k,v in s.items() if torch.is_tensor(v)))
            frozen_equal=all(torch.equal(p.detach().cpu(),saved[n]) for n,p in actor.named_parameters() if not p.requires_grad)
            timing_by_rank=collective.all_gather_object(dict(rank=rank,actor_seconds=actor_end-start,final_kl_seconds=end-actor_end,
                learning_seconds=end-start,performance=profile.report(),communication=collective.collect_gradient_timings(synchronize=True)))
            item=dict(microbatch=micro,kl_mode=mode,detailed_profiler=detailed,optimizer_steps=update['optimizer_steps'],
                scope=scope,repeat=repeat,hard_kl_would_accept=final['mean_joint_kl']<=.03,
                internal_transitions_per_update=[s['internal_transitions'] for s in update['steps']],
                probability_check=check,actor_seconds=max(t['actor_seconds'] for t in timing_by_rank),
                final_kl_seconds=max(t['final_kl_seconds'] for t in timing_by_rank),
                learning_seconds=max(t['learning_seconds'] for t in timing_by_rank),final_kl=final,update=update,difference_from_micro2=delta,
                frozen_equal=all(collective.all_gather_object(frozen_equal)),ranks=timing_by_rank,
                padding_rows=0,local_upper_per_minibatch=10,global_internal_per_minibatch=1600)
            report['results'].append(item)
            if rank==0:
                args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
                print(json.dumps({k:item[k] for k in ('scope','repeat','microbatch','kl_mode','optimizer_steps','actor_seconds','final_kl_seconds','difference_from_micro2')}),flush=True)
            del weights,state,optimizer
        finally:
            deactivate(token)
    actor.load_state_dict(saved); cache.close()
    dist.destroy_process_group(group);dist.destroy_process_group()


if __name__=='__main__': main()
