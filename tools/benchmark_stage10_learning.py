"""服务器1八卡：同一真实Stage10 rollout的完整梯度、Adam、KL及吞吐对照。

输入必须来自collect-only或明确capture的真实Isaac闭环，并核验学习状态、rank
状态、rollout的SHA和实际行为核。不会改写old statistics，不重新采样物理任务。
若原Adam为空，先在此离线副本执行一次真实梯度更新以建立非空动量，再把模型、
Adam、BC/RNG和权重控制器冻结为全部候选的共同起点；此步不是训练接受轮次。

数值轮保存全部未裁剪合成梯度，核对每个模块方向、参数、Adam、末端输出及全20步
KL。计时轮不复制完整梯度，预热后比较同一完整工作量而不是单microbatch延迟。
本离线对照明确关闭软停止以保持更新覆盖一致，最终仍报告原硬KL是否通过；不把
超限的离线副本发布为正式策略。不同精度的迁移KL与自身概率检查分开，BF16和
编译候选即使更快也必须另有真实闭环效果，不能由此脚本自动进入正式训练。

显式profile选项在常规计时完成后额外运行一次固定工作量，只在rank0采集一个
预热后的optimizer step的真实CPU/CUDA trace。输出kernel区间并集、NCCL区间与
主机同步API等待，不把包含CPU发射空隙的CUDA事件时间当成GPU实际计算占用。
这次有测量扰动的运行不进入P50/P95，也不发布训练权重。

显式calibrate-actor-rates模式从同一原始Actor/Adam/BC/RNG出发有限扫描学习率，
只用真实旧链标定完整PPO+BC更新，保存全部梯度比例、clipping和Full20 KL；失败
候选不会成为训练策略。critic-reference模式用同一真实固定目标比较预训练时序
Critic与旧MLP，按完整episode留出，并另测独立物理评估轨迹。冻结特征可以缓存，
可训练层每步重算；固定GAE误差不冒充无偏真实价值误差或下一批在线泛化能力。
"""
from __future__ import annotations
import argparse
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time
from contextlib import nullcontext
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.distributed as dist
from gem.closedloop.dppo.training_checkpoint import file_sha256,immutable_cpu_snapshot
from gem.closedloop.dppo.budget import atomic_json
from gem.closedloop.dppo.diagnostic_rollout import load_rollout
from gem.closedloop.dppo.distributed_runtime import DistributedCollectives
from gem.closedloop.dppo.gradient_merge import BCGradientController
from gem.closedloop.dppo.parallel_support import capture_local_rng,restore_local_rng,local_call,root_call
from gem.closedloop.dppo.dense_policy import DenseDiffusionPolicy
from gem.closedloop.dppo.retention_anchor import Stage1RetentionAnchor
from gem.closedloop.dppo.tensor_cache import RolloutTensorCache
from gem.closedloop.dppo.trainer import load_actor,trainable_actor_parameters
from gem.closedloop.dppo.optimization_support import probability_check_local
from gem.closedloop.dppo.updater import actor_update,final_kl,check_kl
from gem.closedloop.dppo.learning_diagnostics import module_directions,terminal_outputs
from gem.closedloop.dppo.learning_diagnostics import compare_named


def calibrate_actor(a,d,source,rank_state,policy,optimizer,anchor,generators,cache,rows,targets,manifest,kwargs,stage):
    """共享全部实际采样输入与随机起点；扫描结果只供选择新的独立训练配置。"""
    if not 1<=len(a.calibrate_actor_rates)<=6 or any(not 0<x<1e-3 for x in a.calibrate_actor_rates):
        raise ValueError('Finite Actor learning-rate calibration requires 1..6 positive bounded candidates')
    report=dict(schema='genmo.stage10.real_data_actor_calibration',source_case=str(a.case),
        source_case_sha256=file_sha256(a.case/'case.json'),real_global_chains=len(manifest),
        shared_initial_adam_nonempty=bool(source['actor_optimizer']['state']),candidates=[],
        scope='fixed_real_behavior_full_combined_update_not_accepted_physical_training',
        old_statistics_not_rewritten=True,soft_stop_disabled_for_equal_work=True,
        objective=kwargs['objective_reduction'],clip=kwargs['clip'],kl_limits=stage['kl'])
    for lr in a.calibrate_actor_rates:
        policy.actor.load_state_dict(source['actor']);optimizer.load_state_dict(copy.deepcopy(source['actor_optimizer']))
        for group in optimizer.param_groups:group['lr']=lr
        anchor.load_state_dict(rank_state['bc']);restore_local_rng(rank_state['rng'],generators)
        settings=dict(stage['bc_gradient'],initial=a.calibrate_bc_initial)
        if a.calibrate_bc_maximum is not None:settings['maximum']=a.calibrate_bc_maximum
        controller=BCGradientController(**settings)
        def observe(step):
            if d.rank==0:
                print(json.dumps(dict(phase='calibrate_actor',lr=lr,step=step['optimizer_step'],
                    ppo_loss=step['ppo_loss'],coordinate_kl=step['pre_step_mean_coordinate_kl'],
                    bc_weight=step['gradient_contributions']['bc_weight']),ensure_ascii=False),flush=True)
        d.barrier();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        result=actor_update(policy,optimizer,rows,targets,cache,manifest,controller,
            microbatch=a.microbatches[0],step_observer=observe,**kwargs)
        torch.cuda.synchronize();seconds=max(d.all_gather_object(time.perf_counter()-started))
        kl=final_kl(policy,rows,cache,manifest,distributed=d,microbatch=a.kl_microbatches[0],
            clip=kwargs['clip'],objective_reduction=kwargs['objective_reduction'])
        candidate=dict(actor_lr=lr,bc_initial=a.calibrate_bc_initial,
            update=result,full20_kl=kl,hard_kl=check_kl(kl,stage['kl']),actor_seconds=seconds,
            peak_memory_by_rank=d.all_gather_object(torch.cuda.max_memory_allocated()))
        # 同一预热后真实rollout可比较三个更新目标，再各自做独立物理评估；训练交互完全相同。
        if a.save_evaluation_actor:
            soft=source['config']['stage9']['kl_soft_stop_coordinate']
            admissible=(candidate['hard_kl']['passed'] and result['optimizer_steps']==result['planned_optimizer_steps']
                and all(step['pre_step_mean_coordinate_kl']<soft for step in result['steps']))
            candidate['evaluation_publication_admissible']=admissible
            if admissible:
                def save_actor():
                    path=a.output/f'actor_{lr:g}_{kwargs["objective_reduction"]}_K{a.sampled_steps}.pt'
                    payload=dict(schema='genmo.closedloop.stage10.actor_evaluation',evaluation_only=True,
                        complete_update_and_soft_kl_passed=True,
                        actor=immutable_cpu_snapshot(policy.actor.state_dict()),kernel=copy.deepcopy(policy.kernel_config),
                        source_identity=source['identity'],source_case=str(a.case),source_case_sha256=report['source_case_sha256'],
                        update=dict(actor_lr=lr,bc_initial=a.calibrate_bc_initial,clip=kwargs['clip'],
                            objective=kwargs['objective_reduction'],sampled_steps=a.sampled_steps,
                            optimizer_steps=result['optimizer_steps'],planned_optimizer_steps=result['planned_optimizer_steps'],
                            real_global_chains=len(manifest)),
                        full20_kl=kl,hard_kl=candidate['hard_kl'])
                    temporary=path.with_suffix('.partial')
                    with temporary.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
                    os.link(temporary,path);temporary.unlink()
                    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
                    try:os.fsync(fd)
                    finally:os.close(fd)
                    return dict(path=str(path),sha256=file_sha256(path),size_bytes=path.stat().st_size,
                        schema=payload['schema'],evaluation_only=True)
                candidate['evaluation_actor']=root_call(d,save_actor)
        report['candidates'].append(candidate)
        root_call(d,lambda:atomic_json(a.output/'report.json',report))
    return report


def compare_critics(a,d,actor,train,cache,rows,targets,manifest,stage):
    """两种Critic使用同一固定监督与episode切分，独立评估不参与拟合和参数选择。"""
    from gem.closedloop.dppo.pretrained_critic import PretrainedConditionCritic
    from gem.closedloop.dppo.critic import UpperCritic
    from gem.closedloop.dppo.periodic_monitor import fixed_critic_diagnostic,summarize_critic_predictions
    from gem.closedloop.dppo.updater import critic_update
    from gem.closedloop.dppo.rollback_audit import exact_state_digest
    reference_manifest=json.loads(a.critic_reference.read_text())
    descriptor=next(r['critic'] for r in reference_manifest['ranks'] if r['rank']==d.rank)
    if file_sha256(descriptor['path'])!=descriptor['sha256']:raise ValueError('Independent Critic reference SHA changed')
    reference=torch.load(descriptor['path'],map_location='cpu',weights_only=False)
    if reference['reference_sha256']!=descriptor['reference_sha256']:raise ValueError('Critic reference identity mismatch')
    keys=[(r.identity['backend_session_id'],r.identity['env_id'],r.identity['episode_id']) for r in rows]
    episodes=list(dict.fromkeys(keys));held=set(episodes[::4])
    train_ids=[i for i,key in enumerate(keys) if key not in held];held_ids=[i for i,key in enumerate(keys) if key in held]
    if not train_ids or not held_ids:raise ValueError('At least two independent episodes per rank required')
    owners=d.all_gather_object(train_ids)
    fit_manifest=[dict(m,valid=m['valid'] and m['local_index'] in owners[m['owner_rank']]) for m in manifest]
    base_actor=exact_state_digest(actor.state_dict())
    def independent(critic):
        local=fixed_critic_diagnostic(critic,reference)
        parts=d.all_gather_object(local)
        return dict(summarize_critic_predictions([x for p in parts for x in p['predictions']],
            [x for p in parts for x in p['targets']]),scope=reference['diagnostic_scope'])
    @torch.no_grad()
    def metrics(critic,indices):
        chosen=[rows[i] for i in indices]
        pred=critic(cache.context(chosen),cache.get('remaining_music_seconds',chosen)).double()
        y=torch.as_tensor(targets['returns'],device=d.device)[indices].double();err=y-pred
        vector=torch.stack((y.new_tensor(len(y)),y.sum(),y.square().sum(),err.sum(),err.square().sum()))
        n,s,s2,e,e2=d.sum_tensor(vector).cpu().tolist();var=max(0.,s2/n-(s/n)**2)
        return dict(count=int(n),mse=e2/n,explained_variance=None if var<=1e-12 else 1-(e2/n-(e/n)**2)/var)
    @torch.no_grad()
    def order_sensitivity(critic):
        chosen=[rows[i] for i in held_ids[:32]];context=cache.context(chosen)
        remaining=cache.get('remaining_music_seconds',chosen);base=critic(context,remaining)
        results={}
        for name,field,mask_key in [('music','music_features','music_valid'),('prefix','known_qpos30','known_qpos30_mask')]:
            changed={k:v.clone() for k,v in context.items()}
            mask=context[mask_key].any(-1) if context[mask_key].ndim==3 else context[mask_key]
            eligible=[]
            for i in range(len(chosen)):
                idx=mask[i].nonzero().flatten();eligible.append(len(idx)>1)
                changed[field][i,idx]=context[field][i,idx.flip(0)]
            delta=(critic(changed,remaining)-base).abs()[torch.tensor(eligible,device=d.device)]
            parts=d.all_gather_object(delta.cpu().tolist());values=[x for part in parts for x in part]
            results[name]=dict(count=len(values),mean_abs=float(np.mean(values)),max_abs=max(values),
                changed_fraction=sum(x>1e-7 for x in values)/len(values),scope='reverse_values_keep_original_times_and_masks')
        return results
    report=dict(schema='genmo.stage10.real_data_critic_comparison',source_case=str(a.case),
        source_case_sha256=file_sha256(a.case/'case.json'),independent_reference=str(a.critic_reference),
        reference_sha256=file_sha256(a.critic_reference),fit_chains=sum(len(x) for x in owners),
        held_out_chains=len(manifest)-sum(len(x) for x in owners),split='entire_episodes_every_fourth_per_rank',
        targets='unchanged_initial_critic_bootstrapped_GAE_not_unbiased_ground_truth',cases=[])
    for kind in ('pretrained_temporal','legacy_mlp_control'):
        torch.manual_seed(42)
        critic=(PretrainedConditionCritic(actor,dimension=stage['critic']['dimension'],layers=stage['critic']['layers'],
            heads=stage['critic']['heads'],freeze_encoders=True) if kind=='pretrained_temporal' else
            UpperCritic(qpos_mean=actor.endecoder.mean,qpos_std=actor.endecoder.std,
                proprio_scales=tuple(train.model.proprio_scales))).to(d.device)
        d.broadcast_module(critic)
        optimizer=torch.optim.AdamW([p for p in critic.parameters() if p.requires_grad],lr=stage['critic']['learning_rate'],weight_decay=0.)
        generator=torch.Generator().manual_seed(20261012)
        item=dict(architecture=kind,initial=dict(fit=metrics(critic,train_ids),held_out=metrics(critic,held_ids),
            independent=independent(critic)),checkpoints=[])
        completed=0
        for epochs in a.critic_epochs:
            if not completed<epochs<=64:raise ValueError('Critic epoch endpoints must increase within 64')
            started=time.perf_counter()
            fit=critic_update(critic,optimizer,rows,targets,cache,fit_manifest,distributed=d,generator=generator,
                epochs=epochs-completed,batch=stage['critic']['global_batch'])
            item['checkpoints'].append(dict(epochs=epochs,fit=fit,held_out=metrics(critic,held_ids),
                independent=independent(critic),order_sensitivity=order_sensitivity(critic),
                wall_seconds=max(d.all_gather_object(time.perf_counter()-started))))
            completed=epochs
            if d.rank==0:print(json.dumps(dict(phase='critic_comparison',architecture=kind,epochs=epochs,
                held_out=item['checkpoints'][-1]['held_out'],independent=item['checkpoints'][-1]['independent'])),flush=True)
        report['cases'].append(item)
        report['actor_unchanged']=exact_state_digest(actor.state_dict())==base_actor
        if not report['actor_unchanged'] or any(p.grad is not None for p in actor.parameters()):
            raise RuntimeError('Critic comparison changed the Actor or its gradients')
        root_call(d,lambda:atomic_json(a.output/'report.json',report))
        del critic,optimizer
    return report


def recommend_compute_profile(report,device_bytes):
    """按已验证数值、实际P50与显存余量选择计算微批；不替代LR或物理效果标定。"""
    eligible=[]
    for item in report['actor']:
        if item['p50'] is None:continue
        check=item['diagnostic']
        numerical=all(x['passed'] for x in check['gradients']) and all(
            check[key]['passed'] for key in ('parameters','adam','terminal_outputs'))
        peak=max(x['allocated'] for x in item['memory'])
        if numerical and peak<=.8*device_bytes:
            eligible.append(dict(variant=item['variant'],actor_microbatch=item['microbatch'],
                p50_seconds=item['p50'],peak_allocated_bytes=peak))
    selected=min(eligible,key=lambda item:item['p50_seconds']) if eligible else None
    kl=min(report['kl'],key=lambda item:item['p50']) if report['kl'] else None
    return dict(schema='genmo.stage10.measured_compute_recommendation',selected=selected,
        kl_microbatch=None if kl is None else kl['microbatch'],eligible=eligible,
        device_total_bytes=device_bytes,maximum_allocated_fraction=.8,
        selection='fastest_measured_warmed_full_work_with_original_numerical_checks',
        evidence='fixed_real_case_nonempty_Adam_not_production_PhysX_memory_or_learning_quality',
        actor_lr_and_BC_require_separate_calibration=True)


def summarize_gpu_trace(path):
    """汇总真实profiler区间；不同类别可能重叠，不能直接相加得到墙钟。"""
    events=json.loads(path.read_text())['traceEvents']
    spans={name:[] for name in ('kernels','compute_kernels','nccl_kernels','host_sync_api')}
    names={}
    for e in events:
        if e.get('ph')!='X' or not e.get('dur'):continue
        interval=(float(e['ts']),float(e['ts'])+float(e['dur']))
        name=e.get('name','');category=e.get('cat','')
        if category=='kernel':
            spans['kernels'].append(interval)
            spans['nccl_kernels' if 'nccl' in name.lower() else 'compute_kernels'].append(interval)
            names[name]=names.get(name,0.)+e['dur']/1e6
        if category in ('cuda_runtime','cuda_driver') and 'Synchronize' in name:
            spans['host_sync_api'].append(interval)
    def union(intervals):
        total=0.;end=None
        for begin,stop in sorted(intervals):
            total+=stop-begin if end is None or begin>=end else max(0.,stop-end)
            end=max(stop,end) if end is not None else stop
        return total/1e6
    steps=[e['dur']/1e6 for e in events if e.get('ph')=='X' and e.get('name','').startswith('ProfilerStep#')]
    if not spans['kernels'] or not steps:raise RuntimeError('Profiler trace lacks actual CUDA kernels or active step')
    return dict(trace_sha256=file_sha256(path),active_step_wall_seconds=sum(steps),
        interval_union_seconds={k:union(v) for k,v in spans.items()},
        kernel_event_count=len(spans['kernels']),top_kernel_sum_seconds=sorted(names.items(),key=lambda item:-item[1])[:20],
        scope='rank0_one_warmed_optimizer_step_extra_run_excluded_from_throughput',
        intervals_may_overlap=True,host_sync_is_blocking_API_time_not_all_CPU_idle_time=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--microbatches',type=int,nargs='+',default=[64,128,256])
    p.add_argument('--kl-microbatches',type=int,nargs='+',default=[256,512,1024])
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--sampled-steps',type=int,choices=[8,20],default=8)
    p.add_argument('--variants',nargs='+',choices=['reference','bf16_cudnn','compiled_fp32','stable_bmm','wide_tile','weight_bf16x3','weight_bf16x6'],default=['reference'])
    p.add_argument('--profile',action='store_true',help='另测一个预热后的真实CPU/CUDA优化步，不纳入吞吐计时')
    p.add_argument('--diagnostic-only',action='store_true',help='每候选只做一次完整梯度/Adam/全20步KL比较，带梯度拷贝，不作为吞吐计时')
    p.add_argument('--calibrate-actor-rates',type=float,nargs='+',help='从同一起点有限标定完整PPO+BC学习率，不保存候选权重')
    p.add_argument('--calibrate-bc-initial',type=float,default=.01)
    p.add_argument('--calibrate-bc-maximum',type=float)
    p.add_argument('--save-evaluation-actor',action='store_true',help='仅将通过中间软线及最终硬KL的完整标定权重发布为物理评估输入，不是训练恢复点')
    p.add_argument('--objective',choices=['free_coordinate_mean','joint_sum'],help='明确的离线目标对照，不改行为核')
    p.add_argument('--ppo-clip',type=float,help='仅固定数据对照的显式clip候选')
    p.add_argument('--critic-reference',type=Path,help='独立真实物理评估的fixed_diagnostic_reference.json')
    p.add_argument('--critic-epochs',type=int,nargs='+',default=[4,16,32])
    a=p.parse_args();rank=int(os.environ['LOCAL_RANK'])
    if a.diagnostic_only and a.profile:p.error('Numerical diagnostics and independent performance profiling are separate modes')
    if a.save_evaluation_actor and not a.calibrate_actor_rates:p.error('Evaluation Actor publication requires explicit finite calibration')
    if int(os.environ['WORLD_SIZE'])!=8 or not 1<=a.repeats<=5:raise ValueError('Eight GPUs and 1..5 finite repeats required')
    for k,v in dict(NCCL_CUMEM_HOST_ENABLE='0',NCCL_IB_DISABLE='1',NCCL_SOCKET_IFNAME='lo',TORCH_NCCL_BLOCKING_WAIT='1').items():os.environ.setdefault(k,v)
    torch.cuda.set_device(rank);torch.set_num_threads(4)
    dist.init_process_group('gloo',timeout=timedelta(minutes=40))
    group=dist.new_group(backend='nccl',timeout=timedelta(minutes=20))
    d=DistributedCollectives(rank,8,tensor_group=group,device=f'cuda:{rank}')
    root_call(d,lambda:a.output.mkdir(parents=True,exist_ok=False))
    case=json.loads((a.case/'case.json').read_text())
    if case['schema']!='genmo.stage10.fixed_learning_case' or len(case['ranks'])!=8:raise ValueError('Complete eight-rank 当前 case required')
    if file_sha256(a.case/'learning_state.pt')!=case['learning_state_sha256']:raise ValueError('Learning state SHA changed')
    r=a.case/f'rank{rank:02d}';record=next(x for x in case['ranks'] if x['rank']==rank)
    if file_sha256(r/'rank_state.pt')!=record['rank_state_sha256'] or file_sha256(r/'manifest.json')!=record['rollout_manifest_sha256']:
        raise ValueError('Rank input SHA changed')
    inputs=local_call(d,lambda:load_rollout(r));rows=inputs['transitions'];targets=inputs['targets'];manifest=inputs['global_manifest']
    if len(manifest)!=case['real_global_chains'] or case['real_global_chains']<1024:raise ValueError('Small diagnostic data cannot replace large real acceptance')
    source=torch.load(a.case/'learning_state.pt',map_location='cpu',weights_only=False,mmap=True)
    rank_state=torch.load(r/'rank_state.pt',map_location='cpu',weights_only=False)
    config=copy.deepcopy(source['config']);settings=config['stage9'];stage=config['stage10']
    if a.objective is not None:settings['objective_logprob_reduction']=a.objective
    if a.ppo_clip is not None:settings['ppo_clip']=a.ppo_clip
    config['runtime']['genmo_device']=str(d.device)
    actor,train,loading=local_call(d,lambda:load_actor(config));actor.load_state_dict(source['actor'])
    if a.critic_reference is not None:
        if a.calibrate_actor_rates:p.error('Choose Actor calibration or Critic comparison')
        # 与训练入口一致，在修改Actor算子类别前复制Stage1原生条件编码器。
        cache=RolloutTensorCache(rows,targets,d.device,max_device_bytes=8*1024**3)
        compare_critics(a,d,actor,train,cache,rows,targets,manifest,stage)
        cache.close();dist.destroy_process_group(group);dist.destroy_process_group();return
    from gem.closedloop.dppo.run_control import build_policy
    if any(v!='reference' for v in a.variants):
        p.error('Numerical candidates require freshly sampled data from their own actual kernel; benchmark reference within that declared case')
    policy=build_policy(actor,settings,stage['precision'])
    if policy.kernel_config!=source['kernel'] or inputs['kernel']!=source['kernel']:raise ValueError('Baseline behavior numerical contract changed')
    optimizer=torch.optim.AdamW(trainable_actor_parameters(actor),lr=settings['actor_lr'],weight_decay=0.)
    optimizer.load_state_dict(copy.deepcopy(source['actor_optimizer']))
    bc_config=copy.deepcopy(config);bc_config['stage9'].update(bc_batch=settings['bc_batch']//8,seed=42+1000003*rank)
    anchor=Stage1RetentionAnchor(bc_config,actor,train,source_global_step=loading['source_global_step'])
    anchor.load_state_dict(rank_state['bc'])
    controller=BCGradientController(**stage['bc_gradient']);controller.load_state_dict(source['controller'])
    generators=dict(actor=torch.Generator(),critic=torch.Generator())
    restore_local_rng(rank_state['rng'],generators)
    cache=RolloutTensorCache(rows,targets,d.device,max_device_bytes=8*1024**3)
    probability=probability_check_local(policy,rows,global_manifest=manifest,distributed=d,
        denoising_microbatch=512,tensor_cache=cache)
    kwargs=dict(distributed=d,anchor=anchor,generator=generators['actor'],ppo_epochs=2,
        minibatch_chains=stage['scale']['actor_minibatch_chains'],sampled_steps=a.sampled_steps,
        clip=settings['ppo_clip'],gamma_denoising=settings['gamma_denoising'],soft_kl=None,
        kl_reduction=stage['kl']['reduction'],objective_reduction=settings['objective_logprob_reduction'],
        grad_clip_norm=settings['grad_clip_norm'],bc_samples_per_rollout=settings['bc_samples_per_rollout'])
    if a.calibrate_actor_rates:
        calibrate_actor(a,d,source,rank_state,policy,optimizer,anchor,generators,cache,rows,targets,manifest,kwargs,stage)
        cache.close();dist.destroy_process_group(group);dist.destroy_process_group();return
    offline_warm=None
    if not optimizer.state:
        offline_warm=actor_update(policy,optimizer,rows,targets,cache,manifest,controller,
            **dict(kwargs,ppo_epochs=1,minibatch_chains=len(manifest),microbatch=128))
    base=immutable_cpu_snapshot(actor.state_dict());adam=immutable_cpu_snapshot(optimizer.state_dict())
    bc_base=immutable_cpu_snapshot(anchor.state_dict());controller_base=copy.deepcopy(controller.state_dict())
    rng_base=capture_local_rng(generators)
    if not adam['state']:raise ValueError('Nonempty Adam state required')
    report=dict(schema='genmo.stage10.fixed_compute_comparison',source_case=str(a.case),
        source_case_sha256=file_sha256(a.case/'case.json'),real_global_chains=len(manifest),probability=probability,
        offline_warmup=offline_warm,all_candidates_same_nonempty_adam=True,old_statistics_not_rewritten=True,
        soft_stop_disabled_only_for_fixed_work=True,sampled_steps=a.sampled_steps,actor=[],kl=[])
    def publish():
        root_call(d,lambda:atomic_json(a.output/'report.json',report))
    publish()
    def restore():
        actor.load_state_dict(base);optimizer.load_state_dict(copy.deepcopy(adam));anchor.load_state_dict(bc_base)
        controller.load_state_dict(controller_base);restore_local_rng(rng_base,generators)
        actor.zero_grad(set_to_none=True)
    baseline_outputs=terminal_outputs(policy,rows,d.device);reference=None
    for variant in a.variants:
        if variant!='reference':
            from gem.closedloop.dppo.dense_policy import configure_dense_actor
            options=dict(precision='bf16_backbone' if variant=='bf16_cudnn' else stage['precision']['mode'],
                attention='sdpa_cudnn' if variant=='bf16_cudnn' else stage['precision']['attention'],
                linear_execution='stable_bmm' if variant=='stable_bmm' else
                    'stable_tf32x3_wide' if variant=='wide_tile' else stage['precision']['linear_execution'],
                weight_gradient='cublas_'+variant.removeprefix('weight_') if variant.startswith('weight_') else
                    stage['precision'].get('weight_gradient','direct_joint_gemm'))
            if hasattr(actor.denoiser,'_stage10_eager_forward'):raise ValueError('Compiled variant must be tested last or in a separate process')
            policy.numerical_execution.update(configure_dense_actor(actor,**options));policy.precision=options['precision']
            if variant=='compiled_fp32':
                from gem.closedloop.dppo.compiled_forward import install_shared_forward
                install_shared_forward(policy)
        for micro in a.microbatches:
            samples=[];diagnostic=None
            for repeat in range(1 if a.diagnostic_only else a.repeats+2):
                restore();captured=[]
                def observer(model,step):
                    if rank==0:captured.append({n:None if p.grad is None else p.grad.detach().cpu().clone()
                        for n,p in model.named_parameters()})
                d.barrier();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                result=actor_update(policy,optimizer,rows,targets,cache,manifest,controller,microbatch=micro,
                    gradient_observer=observer if repeat==0 else None,**kwargs)
                torch.cuda.synchronize();elapsed=max(d.all_gather_object(time.perf_counter()-start))
                planned=2*((sum(m['valid'] and m['has_free'] for m in manifest)+kwargs['minibatch_chains']-1)//kwargs['minibatch_chains'])
                if result['optimizer_steps']!=planned or result['internal_sample_visits']!=result['planned_internal_visits']:
                    raise ValueError('Candidate changed actual complete optimizer/data work')
                if repeat>=2:samples.append(elapsed)
                if rank==0:print(json.dumps(dict(phase='actor_compute',variant=variant,microbatch=micro,
                    repeat=repeat,seconds=elapsed)),flush=True)
                if repeat==0:
                    kl=final_kl(policy,rows,cache,manifest,distributed=d,microbatch=512,
                        clip=settings['ppo_clip'],objective_reduction=settings['objective_logprob_reduction'])
                    outputs=terminal_outputs(policy,rows,d.device)
                    def compare():
                        nonlocal reference
                        weights=immutable_cpu_snapshot(actor.state_dict())
                        state={f'{i}/{k}':v.cpu().clone() for i,x in optimizer.state_dict()['state'].items()
                            for k,v in x.items() if torch.is_tensor(v)}
                        current=dict(gradients=captured,weights=weights,adam=state,outputs=outputs,kl=kl)
                        if reference is None:reference=current
                        return dict(gradients=[compare_named(x,y,atol=3e-5,rtol=2e-4) for x,y in zip(captured,reference['gradients'])],
                            module_directions=[module_directions(x,y) for x,y in zip(captured,reference['gradients'])],
                            parameters=compare_named(weights,reference['weights'],atol=1e-7,rtol=0.),
                            adam=compare_named(state,reference['adam'],atol=1e-7,rtol=2e-4),
                            terminal_outputs=compare_named({'mean':outputs},{'mean':reference['outputs']},atol=1e-7,rtol=0.),
                            terminal_max_change_from_base=float((outputs-baseline_outputs).abs().max()),
                            full20_kl=kl,hard_kl=check_kl(kl,stage['kl']),
                            precision_contract_change=variant!='reference')
                    diagnostic=root_call(d,compare)
                memory=d.all_gather_object(dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.memory_reserved()))
            report['actor'].append(dict(variant=variant,microbatch=micro,samples=samples,p50=float(np.percentile(samples,50)) if samples else None,
                p95=float(np.percentile(samples,95)) if samples else None,diagnostic=diagnostic,
                numerical_diagnostic_only=a.diagnostic_only,actual_optimizer_steps=result['optimizer_steps'],
                actual_internal_visits=result['internal_sample_visits'],actual_BC_samples=result['actual_global_bc_samples'],memory=memory))
            publish()
    for micro in ([] if a.diagnostic_only else a.kl_microbatches):
        samples=[];restore()
        for repeat in range(a.repeats+2):
            d.barrier();torch.cuda.synchronize();start=time.perf_counter()
            kl=final_kl(policy,rows,cache,manifest,distributed=d,microbatch=micro,
                clip=settings['ppo_clip'],objective_reduction=settings['objective_logprob_reduction'])
            seconds=max(d.all_gather_object(time.perf_counter()-start))
            if repeat>=2:samples.append(seconds)
        report['kl'].append(dict(microbatch=micro,samples=samples,p50=float(np.percentile(samples,50)),
            p95=float(np.percentile(samples,95)),report=kl,numerical_execution=policy.numerical_execution))
        publish()
    if a.profile:
        restore();d.barrier()
        trace=a.output/'rank00_optimizer_trace.json'
        profiler=(torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=True,
            schedule=torch.profiler.schedule(wait=0,warmup=1,active=1,repeat=1),
            on_trace_ready=lambda p:p.export_chrome_trace(str(trace))) if rank==0 else nullcontext())
        with profiler as active:
            actor_update(policy,optimizer,rows,targets,cache,manifest,controller,microbatch=a.microbatches[-1],
                step_observer=(lambda result:active.step()) if rank==0 else None,**kwargs)
        d.barrier()
        report['profiler']=root_call(d,lambda:summarize_gpu_trace(trace));publish()
    report['compute_recommendation']=recommend_compute_profile(report,torch.cuda.get_device_properties(d.device).total_memory)
    root_call(d,lambda:atomic_json(a.output/'compute_recommendation.json',report['compute_recommendation']))
    report['cache']=d.all_gather_object(cache.report());publish();cache.close()
    dist.destroy_process_group(group);dist.destroy_process_group()


if __name__=='__main__':main()
