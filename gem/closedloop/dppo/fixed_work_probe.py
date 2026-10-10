"""同一真实采集会话中的八卡固定工作量数值与吞吐验收。

只有显式有限诊断配置会在第二轮调用本模块。此时第一轮已经完成真实更新，Adam
非空；第二轮2048条链尚未更新，行为策略、编译对象、旧概率和优势均保持原身份。
对Actor微批128/256/512与KL微批256/512/1024串行比较，始终覆盖完整20步及两个
epoch，不把进程间重新编译形成的另一套算子当作原行为策略，不改写old数据。

每次候选恢复同一Actor、Adam、BC和随机状态。诊断中的optimizer尝试真实扣预算，
参数不发布，所有候选结束后逐字节核验恢复，再执行本轮正常训练事务。固定工作量
对照明确禁用本诊断副本的软停止，最终全量硬KL仍记录接受与否；普通训练规则不变。
本轮总墙钟包含诊断，禁止作为普通轮性能。完整梯度/参数/Adam复制仅发生在数值轮，
性能轮经一次预热后统计三次最慢rank墙钟；源码与报告纳入同一run的审计身份。
"""
from __future__ import annotations

import copy
import gc
import time
import numpy as np
import torch

from .parallel_support import (broadcast_state, capture_local_rng, restore_local_rng,
                              local_call, root_call, cpu_snapshot)
from .budget import atomic_json
from .rollback_audit import compare_recovered_state, release_failed_computation
from .updater_v2 import actor_update_v2, analytic_kl_local, probability_check_local
from .performance import PhaseProfiler, activate, deactivate
from .run_management import file_sha256


@torch.no_grad()
def _terminal_outputs(policy, rows, cache):
    from .execution_checks import policy_phase
    from .tensor_cache import ConditionGraphCache
    from .updater_v2 import _parameters
    result=[]
    with policy_phase(policy):
        conditions=ConditionGraphCache(policy,cache);conditions.prime(rows)
        for step in (18,19):
            values=[]
            for start in range(0,len(rows),128):
                selected=rows[start:start+128]
                parameters,_=_parameters(policy,selected,[step]*len(selected),cache.device,cache,conditions)
                values.append(parameters['mean'].cpu())
            result.append(torch.cat(values))
        conditions.validate_inputs()
    return torch.stack(result)


def run_fixed_work_probe(c, rows, targets, manifest, backup, bc_before, rng):
    from tools.verify_stage10_saved_learning import compare_named
    from tools.verify_stage10_gradient_reductions import module_directions
    from tools.stage10_step_gradient_report import step_contributions
    d=c.distributed;actor=c.actor;optimizer=c.actor_optimizer;cache=c.tensor_cache
    if len(manifest)!=2048 or c.settings['ppo_epochs']!=2 or c.settings['denoising_steps']!=20:
        raise ValueError('In-session probe requires exactly 2048 real chains, Full20 and two epochs')
    if not optimizer.state:raise ValueError('In-session probe requires nonempty actual Adam state')
    selected=[i for i,m in enumerate(manifest) if m['valid'] and m['has_free']]
    if len(selected)!=2048:raise ValueError('Fixed work cannot silently discard invalid or fully known chains')
    directory=c.session/'fixed_work_probe'
    root_call(d,lambda:directory.mkdir(exist_ok=False))
    state=broadcast_state({k:backup[k] for k in ('actor','actor_optimizer')} if d.rank==0 else None,d)
    old_grad=cpu_snapshot({name:p.grad for name,p in actor.named_parameters()})
    report=dict(schema='stage10.fixed_work_same_session.v1',global_real_chains=2048,behavior_policy_version=c.state['policy_version'],
        numerical_contract=copy.deepcopy(c.policy.kernel_config),nonempty_adam=True,probabilities={},actor=[],kl=[],
        output_scope='all_local_chains_last_two_sensitive_steps_and_full20_KL',
        soft_stop_disabled_only_for_fixed_work_diagnostic=True,ordinary_round=False,
        actual_optimization_attempts_charged=True,old_data_rewritten=False)
    started=time.perf_counter()
    orders=[[selected[i] for i in torch.randperm(len(selected),generator=torch.Generator().manual_seed(445+epoch)).tolist()]
            for epoch in range(2)]
    def save():root_call(d,lambda:atomic_json(directory/'report.json',report))
    attempts=0
    def reserve():
        nonlocal attempts
        root_call(d,lambda:c.budget.reserve('fixed_work_probe',optimizer_attempts=1))
        attempts+=1
    def restore():
        actor.zero_grad(set_to_none=True)
        actor.load_state_dict(state['actor']);optimizer.load_state_dict(copy.deepcopy(state['actor_optimizer']))
        if c.bc is not None:c.bc.load_state_dict(copy.deepcopy(bc_before))
        restore_local_rng(rng,c.generators);actor.zero_grad(set_to_none=True)
    kwargs=dict(global_manifest=manifest,distributed=d,actor_minibatch_internal_transitions=2048*20,
        ppo_epochs=2,epoch_orders=orders,max_optimizer_steps=2,soft_kl_limit=None,
        gradient_diagnostics=True,gradient_module_details=False,tensor_cache=cache,
        kl_check_mode='pre_step_plus_final',bc=c.bc,bc_weight=c.settings['bc_weight'],
        clip=c.settings['ppo_clip'],gamma_denoising=c.settings['gamma_denoising'],
        grad_clip_norm=c.settings['grad_clip_norm'],
        reserve_attempt=reserve)
    reference=None;output_reference=None;recovery=None
    try:
        for micro in (128,256,512,1024):
            report['probabilities'][str(micro)]=probability_check_local(c.policy,rows,global_manifest=manifest,
                distributed=d,denoising_microbatch=micro,tensor_cache=cache)
            save()
        base_outputs=_terminal_outputs(c.policy,rows,cache)
        for micro in (128,256,512):
            times=[];diagnostic=None;unavailable=None
            for repeat in range(5):
                local_call(d,restore);captured=[]
                def observe(model,step):
                    if d.rank==0:captured.append(cpu_snapshot({name:p.grad for name,p in model.named_parameters()}))
                d.barrier();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
                profiler=PhaseProfiler(d.device,d.rank,detailed=True) if repeat==0 else None
                token=activate(profiler) if profiler is not None else None
                try:
                    update=actor_update_v2(c.policy,optimizer,rows,targets,denoising_microbatch=micro,
                        gradient_observer=observe if repeat==0 else None,**kwargs)
                except Exception as failure:
                    details=release_failed_computation(failure)
                    actor.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
                    failures=d.all_gather_object(details)
                    # 仅八卡明确的 CUDA OOM 允许继续有限候选比较，其他异常终止并整轮恢复。
                    if not all(item['cuda_oom'] for item in failures):raise
                    unavailable=dict(reason='cuda_out_of_memory',repeat=repeat,failures_by_rank=failures,
                        partial_work_charged=True,full_work_completed=False)
                finally:
                    if token is not None:deactivate(token)
                if unavailable is not None:
                    local_call(d,restore)
                    memory=d.all_gather_object(dict(allocated_peak_bytes=torch.cuda.max_memory_allocated(),
                        reserved_bytes=torch.cuda.memory_reserved()))
                    break
                torch.cuda.synchronize();seconds=time.perf_counter()-begin
                memory=d.all_gather_object(dict(seconds=seconds,allocated_peak_bytes=torch.cuda.max_memory_allocated(),
                    reserved_bytes=torch.cuda.memory_reserved()))
                if update['optimizer_steps']!=2 or update['applied_internal_sample_visits']!=81920:
                    raise ValueError('Fixed-work candidate changed update count or complete coverage')
                if repeat>=2:times.append(max(item['seconds'] for item in memory))
                if repeat==0:
                    profiles=d.all_gather_object(profiler.report())
                    kl=analytic_kl_local(c.policy,rows,global_manifest=manifest,distributed=d,
                        denoising_microbatch=256,tensor_cache=cache)
                    outputs=_terminal_outputs(c.policy,rows,cache)
                    if output_reference is None:output_reference=outputs.clone()
                    output_reports=d.all_gather_object(dict(rank=d.rank,
                        comparison=compare_named({'mean':outputs},{'mean':output_reference},atol=1e-7,rtol=0.),
                        changed_elements=int((outputs!=base_outputs).sum()),max_change=float((outputs-base_outputs).abs().max())))
                    def compare():
                        nonlocal reference
                        weights=cpu_snapshot(actor.state_dict())
                        adam={f'{i}/{key}':value.detach().cpu().clone() for i,entry in optimizer.state_dict()['state'].items()
                              for key,value in entry.items() if torch.is_tensor(value)}
                        if reference is None:reference=(captured,weights,adam,kl)
                        grad=[compare_named(x,y,atol=3e-5,rtol=2e-4) for x,y in zip(captured,reference[0])]
                        params=compare_named(weights,reference[1],atol=1e-7,rtol=0.)
                        moment=compare_named(adam,reference[2],atol=1e-7,rtol=2e-4)
                        return dict(gradients=grad,modules=[module_directions(x,y) for x,y in zip(captured,reference[0])],
                            parameters=params,adam=moment,full_kl=kl,full_kl_abs_difference=abs(kl['mean_joint_kl']-reference[3]['mean_joint_kl']),
                            terminal_outputs_by_rank=output_reports,hard_kl_accepted=kl['mean_joint_kl']<=c.settings['kl_stop_joint'],
                            numerical_passed=all(g['passed'] for g in grad) and params['passed'] and moment['passed'] and
                                all(item['comparison']['passed'] for item in output_reports))
                    diagnostic=root_call(d,compare)
                    diagnostic['phase_profiles_by_rank']=profiles
                del captured
            if unavailable is not None:
                report['actor'].append(dict(microbatch=micro,status='unavailable',p50=None,p95=None,
                    samples=times,diagnostic=diagnostic,memory_ranks=memory,**unavailable))
                save()
                if d.rank==0:print(f'[FIXED_WORK] Actor B={micro} unavailable: CUDA OOM; state restored',flush=True)
                continue
            report['actor'].append(dict(microbatch=micro,status='completed',p50=float(np.percentile(times,50)),p95=float(np.percentile(times,95)),
                samples=times,diagnostic=diagnostic,memory_ranks=memory,optimizer_steps=2,
                internal_sample_visits=81920,global_bc_samples=256))
            if d.rank==0:print(f'[FIXED_WORK] Actor B={micro} P50={np.percentile(times,50):.3f}s accepted={diagnostic["numerical_passed"]}',flush=True)
            save()
        # 所有KL候选及逐步梯度使用参考B128更新后的同一权重，不能用零更新KL冒充最终KL。
        updated=broadcast_state(reference[1] if d.rank==0 else None,d)
        actor.load_state_dict(updated)
        del updated
        report['per_step_gradient']=step_contributions(c.policy,rows,targets,cache,d,
            clip=c.settings['ppo_clip'],gamma_denoising=c.settings['gamma_denoising']);save()
        reference_kl=None
        for micro in (256,512,1024):
            times=[]
            for repeat in range(5):
                d.barrier();torch.cuda.synchronize();begin=time.perf_counter()
                profiler=PhaseProfiler(d.device,d.rank,detailed=True) if repeat==0 else None
                token=activate(profiler) if profiler is not None else None
                try:
                    kl=analytic_kl_local(c.policy,rows,global_manifest=manifest,distributed=d,
                        denoising_microbatch=micro,tensor_cache=cache)
                finally:
                    if token is not None:deactivate(token)
                torch.cuda.synchronize();seconds=max(d.all_gather_object(time.perf_counter()-begin))
                if repeat==0:profiles=d.all_gather_object(profiler.report())
                if repeat>=2:times.append(seconds)
                if kl['fresh_internal_forwards']!=40960:raise ValueError('KL candidate omitted complete steps')
            if reference_kl is None:reference_kl=kl
            report['kl'].append(dict(microbatch=micro,p50=float(np.percentile(times,50)),p95=float(np.percentile(times,95)),
                samples=times,report=kl,phase_profiles_by_rank=profiles,
                mean_difference=abs(kl['mean_joint_kl']-reference_kl['mean_joint_kl'])))
            save()
    except Exception as failure:
        report['failure']=release_failed_computation(failure)
        actor.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
        raise
    finally:
        local_call(d,restore)
        for name,p in actor.named_parameters():p.grad=None if old_grad[name] is None else old_grad[name].to(p.device)
        expected=dict(state,bc=bc_before,rng=rng)
        actual=dict(actor=actor.state_dict(),actor_optimizer=optimizer.state_dict(),
            bc=None if c.bc is None else c.bc.state_dict(),rng=capture_local_rng(c.generators))
        recovery=d.all_gather_object(local_call(d,lambda:compare_recovered_state(expected,actual)))
        report.update(recovery_by_rank=recovery,optimizer_attempt_count=attempts,
            seconds_including_restores=max(d.all_gather_object(time.perf_counter()-started)))
        save()
        if not all(item['identical'] for item in recovery):raise RuntimeError('Fixed-work diagnostic failed exact restoration')
    digest=root_call(d,lambda:file_sha256(directory/'report.json'))
    return dict(report=str(directory/'report.json'),report_sha256=digest,
        optimizer_attempts=attempts,seconds=report['seconds_including_restores'],
        scope='finite_diagnostic_included_in_outer_wall_not_an_ordinary_round',restored_exactly=True)
