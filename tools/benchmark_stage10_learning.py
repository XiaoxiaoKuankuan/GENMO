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

cross-kernel模式允许数值候选在同一stable采样数据和同一起点上执行同样的8步更新：
除原逐张量对照外，报告更新前跨核数值KL底噪、更新后相对reference的超额KL、参数增量
与reference的余弦和clip比例。它只回答候选学习更新是否与主核等价及其吞吐，不替代候选
在自身采样数据上的零更新检查，也不自动改变训练数值核。

heldout-surrogate模式是A0学习信号离线检验：按(rank,环境槽)整条环境流把链分成拟合半和
留出半，BC权重固定为0，只用拟合半执行标准PPO更新，再在留出半计算优势加权的代理比值
增益 mean(A*(r-1))。同一流程对拟合半优势做全局置换得到零分布，报告z值。不保存权重。
"""
from __future__ import annotations
import argparse
import copy
from datetime import timedelta
import json
import math
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
from gem.closedloop.dppo.collectives import DistributedCollectives
from gem.closedloop.dppo.gradient_merge import BCGradientController
from gem.closedloop.dppo.parallel_support import capture_local_rng,restore_local_rng,local_call,root_call
from gem.closedloop.dppo.retention_anchor import Stage1RetentionAnchor
from gem.closedloop.dppo.tensor_cache import RolloutTensorCache
from gem.closedloop.dppo.trainer import load_actor,trainable_actor_parameters
from gem.closedloop.dppo.optimization_support import probability_check_local
from gem.closedloop.dppo.updater import actor_update,final_kl,check_kl
from gem.closedloop.dppo.learning_diagnostics import module_directions,terminal_outputs
from gem.closedloop.dppo.learning_diagnostics import compare_named


def _set_advantages(cache,values):
    cache.host['advantages']=values.to(cache.host['advantages'].dtype).clone()
    cache.blocks.clear()


@torch.no_grad()
def heldout_gain(policy,rows,cache,manifest,d,*,microbatch,gamma_denoising):
    """留出链全部20步的优势加权代理比值增益；r=1时恒为0，只读当前Actor和旧统计。"""
    from gem.closedloop.dppo.execution_checks import policy_phase
    from gem.closedloop.dppo.gaussian_probability import gaussian_statistics
    from gem.closedloop.dppo.optimization_support import _owned,_parameters,_old_kernel
    from gem.closedloop.dppo.tensor_cache import ConditionGraphCache
    from gem.closedloop.dppo.updater import _eligible
    owned=_owned(_eligible(manifest),manifest,d);local=[rows[i] for _,i in owned]
    totals=torch.zeros(3,dtype=torch.float64,device=d.device)
    with policy_phase(policy):
        conditions=ConditionGraphCache(policy,cache);conditions.prime(local)
        pairs=[(i,s) for i in range(len(local)) for s in policy.trainable_steps]
        for offset in range(0,len(pairs),microbatch):
            part=pairs[offset:offset+microbatch];samples=[local[i] for i,_ in part];steps=[s for _,s in part]
            params,mask=_parameters(policy,samples,steps,d.device,cache,conditions)
            old_mean,old_std=_old_kernel(samples,steps,d.device,cache)
            stats=gaussian_statistics(cache.get('chain',samples,[s+1 for s in steps]),old_mean,old_std,
                params['mean'],params['std'],mask)
            ratio=stats['mean_coordinate_log_ratio'].exp()
            advantage=cache.get('advantages',samples).to(ratio)*cache.denoising_discounts(steps,20,gamma_denoising).to(ratio)
            valid=stats['free_count']>0
            totals+=torch.stack(((advantage*(ratio-1)).masked_fill(~valid,0.).sum(),valid.sum().double(),
                (ratio-1).abs().masked_fill(~valid,0.).sum()))
    totals=d.sum_tensor(totals);gain,count,change=totals.cpu().tolist()
    return dict(gain_per_internal_transition=gain/max(count,1.),internal_transitions=int(count),
        mean_abs_ratio_change=change/max(count,1.))


def heldout_surrogate_test(a,d,policy,optimizer,anchor,generators,cache,rows,manifest,kwargs,settings):
    import hashlib
    half=lambda slot:int(hashlib.sha256(f'{a.heldout_seed}:{d.rank}:{slot}'.encode()).hexdigest(),16)%2
    halves=d.all_gather_object([half(row.metadata.get('collector_env_slot',i)) for i,row in enumerate(rows)])
    fit=[dict(m,has_free=m['has_free'] and halves[m['owner_rank']][m['local_index']]==0) for m in manifest]
    held=[dict(m,has_free=m['has_free'] and halves[m['owner_rank']][m['local_index']]==1) for m in manifest]
    controller=BCGradientController(initial=0.,minimum=0.,mode='fixed')
    base=immutable_cpu_snapshot(policy.actor.state_dict());adam=immutable_cpu_snapshot(optimizer.state_dict())
    bc_base=immutable_cpu_snapshot(anchor.state_dict());rng_base=capture_local_rng(generators)
    true_advantages=cache.host['advantages'].clone()
    local_fit=[m['local_index'] for m in fit if m['owner_rank']==d.rank and m['valid'] and m['has_free']]
    def restore():
        policy.actor.load_state_dict(base);optimizer.load_state_dict(copy.deepcopy(adam))
        anchor.load_state_dict(bc_base);restore_local_rng(rng_base,generators);policy.actor.zero_grad(set_to_none=True)
    def evaluate(permutation):
        restore()
        if permutation is not None:
            values=true_advantages[local_fit].double().tolist()
            pool=[v for part in d.all_gather_object(values) for v in part]
            order=torch.randperm(len(pool),generator=torch.Generator().manual_seed(10007*a.heldout_seed+permutation))
            offset=sum(len(x) for x in d.all_gather_object(values)[:d.rank])
            permuted=true_advantages.clone()
            if local_fit:permuted[local_fit]=torch.tensor(pool,dtype=permuted.dtype)[order][offset:offset+len(local_fit)]
            _set_advantages(cache,permuted)
        update=actor_update(policy,optimizer,rows,None,cache,fit,controller,microbatch=settings['denoising_microbatch'],**kwargs)
        _set_advantages(cache,true_advantages)
        result=heldout_gain(policy,rows,cache,held,d,microbatch=128,gamma_denoising=settings['gamma_denoising'])
        result.update(optimizer_steps=update['optimizer_steps'],permutation=permutation)
        if d.rank==0:print(json.dumps(dict(phase='heldout_surrogate',**result)),flush=True)
        return result
    restore()
    zero=heldout_gain(policy,rows,cache,held,d,microbatch=128,gamma_denoising=settings['gamma_denoising'])
    real=evaluate(None);nulls=[evaluate(k) for k in range(a.heldout_surrogate)]
    values=np.array([x['gain_per_internal_transition'] for x in nulls])
    z=None if len(values)<2 or values.std(ddof=1)<=0 else float((real['gain_per_internal_transition']-values.mean())/values.std(ddof=1))
    report=dict(schema='genmo.stage10.heldout_surrogate_test',source_case=str(a.case),seed=a.heldout_seed,
        fit_chains=sum(m['valid'] and m['has_free'] for m in fit),heldout_chains=sum(m['valid'] and m['has_free'] for m in held),
        split='sha256(seed,rank,collector_env_slot)_whole_environment_streams',bc_weight=0.,
        before_update=zero,real=real,null=nulls,null_mean=float(values.mean()) if len(values) else None,
        null_std=float(values.std(ddof=1)) if len(values)>1 else None,z=z,
        interpretation='z>=2 suggests PPO update generalizes advantage signal to held-out environment streams')
    root_call(d,lambda:atomic_json(a.output/'heldout_surrogate.json',report))
    restore()
    return report


def calibrate_actor(a,d,source,rank_state,policy,optimizer,anchor,generators,cache,rows,targets,manifest,kwargs,stage):
    """共享全部实际采样输入与随机起点；扫描结果只供选择新的独立训练配置。"""
    if not 1<=len(a.calibrate_actor_rates)<=6 or any(not 0<x<1e-3 for x in a.calibrate_actor_rates):
        raise ValueError('Finite Actor learning-rate calibration requires 1..6 positive bounded candidates')
    from gem.closedloop.dppo.kl_calibration import digest
    report=dict(schema='genmo.stage2.real_data_four_step_actor_calibration.v1',source_case=str(a.case),
        kernel_sha256=digest(policy.kernel_config),reward_sha256=digest(source['identity']['reward']),
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
            update=result,trainable4_kl=kl,hard_kl=check_kl(kl,stage['kl']),actor_seconds=seconds,
            peak_memory_by_rank=d.all_gather_object(torch.cuda.max_memory_allocated()))
        # 同一预热后真实rollout可比较三个更新目标，再各自做独立物理评估；训练交互完全相同。
        if a.save_evaluation_actor:
            joint_units=stage['kl']['reduction']=='joint_sum'
            soft=a.soft_kl_joint if joint_units else source['config']['stage9']['kl_soft_stop_coordinate']
            soft_field='pre_step_mean_joint_kl' if joint_units else 'pre_step_mean_coordinate_kl'
            admissible=(candidate['hard_kl']['passed'] and result['optimizer_steps']==result['planned_optimizer_steps']
                and all(step[soft_field]<soft for step in result['steps']))
            candidate['publication_soft_kl']=dict(field=soft_field,limit=soft)
            candidate['evaluation_publication_admissible']=admissible
            if admissible:
                def save_actor():
                    path=a.output/f'actor_{lr:g}_{kwargs["objective_reduction"]}_K{a.sampled_steps}.pt'
                    payload=dict(schema='genmo.stage2.asymmetric_actor_evaluation.v1',evaluation_only=True,
                        complete_update_and_soft_kl_passed=True,
                        actor=immutable_cpu_snapshot(policy.actor.state_dict()),kernel=copy.deepcopy(policy.kernel_config),
                        source_identity=source['identity'],source_case=str(a.case),source_case_sha256=report['source_case_sha256'],
                        update=dict(actor_lr=lr,bc_initial=a.calibrate_bc_initial,clip=kwargs['clip'],
                            objective=kwargs['objective_reduction'],sampled_steps=a.sampled_steps,
                            optimizer_steps=result['optimizer_steps'],planned_optimizer_steps=result['planned_optimizer_steps'],
                            real_global_chains=len(manifest),kl_limits=copy.deepcopy(stage['kl']),
                            publication_soft_kl=candidate['publication_soft_kl']),
                        trainable4_kl=kl,hard_kl=candidate['hard_kl'])
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
    p.add_argument('--sampled-steps',type=int,choices=[4],default=4)
    p.add_argument('--variants',nargs='+',choices=['reference','bf16_cudnn','compiled_fp32','stable_bmm','wide_tile','weight_bf16x3','weight_bf16x6'],default=['reference'])
    p.add_argument('--profile',action='store_true',help='另测一个预热后的真实CPU/CUDA优化步，不纳入吞吐计时')
    p.add_argument('--profile-only',action='store_true',help='只在明确指定微批做独立CUDA trace，不重跑完整吞吐扫描')
    p.add_argument('--diagnostic-only',action='store_true',help='每候选只做一次完整梯度/Adam/全20步KL比较，带梯度拷贝，不作为吞吐计时')
    p.add_argument('--calibrate-actor-rates',type=float,nargs='+',help='从同一起点有限标定完整PPO+BC学习率，不保存候选权重')
    p.add_argument('--calibrate-bc-initial',type=float,default=.01)
    p.add_argument('--calibrate-bc-maximum',type=float)
    p.add_argument('--save-evaluation-actor',action='store_true',help='仅将通过中间软线及最终硬KL的完整标定权重发布为物理评估输入，不是训练恢复点')
    p.add_argument('--objective',choices=['free_coordinate_mean','joint_sum'],help='明确的离线目标对照，不改行为核')
    p.add_argument('--ppo-clip',type=float,help='仅固定数据对照的显式clip候选')
    p.add_argument('--mean-joint-kl-limit',type=float,help='严格joint对照的真实joint KL上限，不能借用平均坐标门槛')
    p.add_argument('--soft-kl-joint',type=float,help='严格joint对照的逐minibatch发布门槛，单位为真实joint KL')
    p.add_argument('--cross-kernel',action='store_true',help='允许数值候选在同一stable采样数据上做更新等价对照')
    p.add_argument('--heldout-surrogate',type=int,default=0,help='A0留出代理增益检验的零分布置换次数')
    p.add_argument('--heldout-seed',type=int,default=0)
    a=p.parse_args();rank=int(os.environ['LOCAL_RANK'])
    if a.diagnostic_only and a.profile:p.error('Numerical diagnostics and independent performance profiling are separate modes')
    if a.profile_only:
        if a.diagnostic_only or a.calibrate_actor_rates:p.error('Profile-only is separate from numerical or rate calibration')
        a.profile=True
    if a.save_evaluation_actor and not a.calibrate_actor_rates:p.error('Evaluation Actor publication requires explicit finite calibration')
    if a.mean_joint_kl_limit is not None or a.soft_kl_joint is not None:
        if (a.objective!='joint_sum' or a.mean_joint_kl_limit is None or a.soft_kl_joint is None
                or not all(math.isfinite(x) for x in (a.soft_kl_joint,a.mean_joint_kl_limit))
                or not 0<a.soft_kl_joint<a.mean_joint_kl_limit):
            p.error('Strict joint comparison requires explicit joint objective and 0 < joint soft KL < joint hard KL')
    if int(os.environ['WORLD_SIZE'])!=8 or not 1<=a.repeats<=5:raise ValueError('Eight GPUs and 1..5 finite repeats required')
    for k,v in dict(NCCL_CUMEM_HOST_ENABLE='0',NCCL_IB_DISABLE='1',NCCL_SOCKET_IFNAME='lo',TORCH_NCCL_BLOCKING_WAIT='1').items():os.environ.setdefault(k,v)
    torch.cuda.set_device(rank);torch.set_num_threads(4)
    dist.init_process_group('gloo',timeout=timedelta(minutes=40))
    group=dist.new_group(backend='nccl',timeout=timedelta(minutes=20))
    d=DistributedCollectives(rank,8,tensor_group=group,device=f'cuda:{rank}')
    root_call(d,lambda:a.output.mkdir(parents=True,exist_ok=False))
    case=json.loads((a.case/'case.json').read_text())
    if case['schema']!='genmo.stage2.asymmetric_fixed_learning_case.v1' or len(case['ranks'])!=8:raise ValueError('Complete eight-rank 当前 case required')
    if file_sha256(a.case/'learning_state.pt')!=case['learning_state_sha256']:raise ValueError('Learning state SHA changed')
    r=a.case/f'rank{rank:02d}';record=next(x for x in case['ranks'] if x['rank']==rank)
    if file_sha256(r/'rank_state.pt')!=record['rank_state_sha256'] or file_sha256(r/'manifest.json')!=record['rollout_manifest_sha256']:
        raise ValueError('Rank input SHA changed')
    inputs=local_call(d,lambda:load_rollout(r));rows=inputs['transitions'];targets=inputs['targets'];manifest=inputs['global_manifest']
    if len(manifest)!=case['real_global_chains'] or case['real_global_chains']<1024:raise ValueError('Small diagnostic data cannot replace large real acceptance')
    source=torch.load(a.case/'learning_state.pt',map_location='cpu',weights_only=False,mmap=True)
    rank_state=torch.load(r/'rank_state.pt',map_location='cpu',weights_only=False)
    config=copy.deepcopy(source['config']);settings=config['stage9'];stage=config['stage10']
    if a.mean_joint_kl_limit is not None:
        stage['kl'].update(reduction='joint_sum',mean_joint=a.mean_joint_kl_limit)
    if a.objective is not None:settings['objective_logprob_reduction']=a.objective
    if a.ppo_clip is not None:settings['ppo_clip']=a.ppo_clip
    config['runtime']['genmo_device']=str(d.device)
    actor,train,loading=local_call(d,lambda:load_actor(config))
    from gem.closedloop.dppo.run_control import build_policy
    if any(v!='reference' for v in a.variants) and not a.cross_kernel:
        p.error('Numerical candidates require freshly sampled data from their own actual kernel; use --cross-kernel for stable-sampled update equivalence')
    if a.cross_kernel and a.variants[0]!='reference':p.error('--cross-kernel compares every candidate against a first reference run')
    policy=build_policy(actor,settings,stage['precision'])
    # 冻结前段必须先从Stage1初始化；恢复只覆盖可训练Actor。
    actor.load_state_dict(source['actor'])
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
    if a.heldout_surrogate:
        root_call(d,lambda:a.output.mkdir(parents=True,exist_ok=True))
        heldout_surrogate_test(a,d,policy,optimizer,anchor,generators,cache,rows,manifest,kwargs,settings)
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
    for variant in ([] if a.profile_only else a.variants):
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
                pre_kl=None
                if a.cross_kernel and repeat==0:
                    pre_kl=final_kl(policy,rows,cache,manifest,distributed=d,microbatch=512,
                        clip=settings['ppo_clip'],objective_reduction=settings['objective_logprob_reduction'])
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
                        delta={k:(weights[k].double()-base[k].double()).reshape(-1) for k in weights if weights[k].is_floating_point()}
                        clip=float(np.mean([x['clip_fraction'] for x in result['steps']]))
                        current=dict(gradients=captured,weights=weights,adam=state,outputs=outputs,kl=kl,delta=delta,clip=clip)
                        if reference is None:reference=current
                        cross=None
                        if a.cross_kernel:
                            dot=sum(float(delta[k]@reference['delta'][k]) for k in delta)
                            norm=math.sqrt(sum(float(v@v) for v in delta.values())*sum(float(v@v) for v in reference['delta'].values()))
                            cross=dict(pre_update_mean_coordinate_kl=pre_kl['mean_coordinate_kl'],
                                pre_update_maximum_step_mean_coordinate_kl=pre_kl['maximum_step_mean_coordinate_kl'],
                                post_update_mean_coordinate_kl=kl['mean_coordinate_kl'],
                                excess_mean_coordinate_kl_vs_reference=kl['mean_coordinate_kl']-reference['kl']['mean_coordinate_kl'],
                                parameter_delta_cosine_vs_reference=dot/norm if norm>0 else None,
                                clip_fraction_mean=clip,reference_clip_fraction_mean=reference['clip'],
                                scope='stable_sampled_old_statistics;candidate_learning_kernel')
                        return dict(cross_kernel=cross,gradients=[compare_named(x,y,atol=3e-5,rtol=2e-4) for x,y in zip(captured,reference['gradients'])],
                            module_directions=[module_directions(x,y) for x,y in zip(captured,reference['gradients'])],
                            parameters=compare_named(weights,reference['weights'],atol=1e-7,rtol=0.),
                            adam=compare_named(state,reference['adam'],atol=1e-7,rtol=2e-4),
                            terminal_outputs=compare_named({'mean':outputs},{'mean':reference['outputs']},atol=1e-7,rtol=0.),
                            terminal_max_change_from_base=float((outputs-baseline_outputs).abs().max()),
                            trainable4_kl=kl,hard_kl=check_kl(kl,stage['kl']),
                            precision_contract_change=variant!='reference')
                    diagnostic=root_call(d,compare)
                memory=d.all_gather_object(dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.memory_reserved()))
            report['actor'].append(dict(variant=variant,microbatch=micro,samples=samples,p50=float(np.percentile(samples,50)) if samples else None,
                p95=float(np.percentile(samples,95)) if samples else None,diagnostic=diagnostic,
                numerical_diagnostic_only=a.diagnostic_only,actual_optimizer_steps=result['optimizer_steps'],
                actual_internal_visits=result['internal_sample_visits'],actual_BC_samples=result['actual_global_bc_samples'],memory=memory))
            publish()
    for micro in ([] if a.diagnostic_only or a.profile_only else a.kl_microbatches):
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
        recommendation=recommend_compute_profile(report,torch.cuda.get_device_properties(d.device).total_memory)
        chosen=recommendation['selected']
        profile_micro=chosen['actor_microbatch'] if chosen else a.microbatches[0]
        trace=a.output/'rank00_optimizer_trace.json'
        profiler=(torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA],record_shapes=True,profile_memory=True,
            schedule=torch.profiler.schedule(wait=0,warmup=1,active=1,repeat=1),
            on_trace_ready=lambda p:p.export_chrome_trace(str(trace))) if rank==0 else nullcontext())
        with profiler as active:
            actor_update(policy,optimizer,rows,targets,cache,manifest,controller,microbatch=profile_micro,
                step_observer=(lambda result:active.step()) if rank==0 else None,**kwargs)
        d.barrier()
        report['profiler']=root_call(d,lambda:summarize_gpu_trace(trace))
        report['profiler']['microbatch']=profile_micro
        if a.profile_only:
            report['profile_final_trainable4_kl']=final_kl(policy,rows,cache,manifest,distributed=d,
                microbatch=a.kl_microbatches[0],clip=settings['ppo_clip'],objective_reduction=settings['objective_logprob_reduction'])
        publish()
    report['compute_recommendation']=recommend_compute_profile(report,torch.cuda.get_device_properties(d.device).total_memory)
    root_call(d,lambda:atomic_json(a.output/'compute_recommendation.json',report['compute_recommendation']))
    report['cache']=d.all_gather_object(cache.report());publish();cache.close()
    dist.destroy_process_group(group);dist.destroy_process_group()


if __name__=='__main__':main()
