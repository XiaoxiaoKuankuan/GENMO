"""逐步自由坐标平均代理Clipped PPO、独立真实联合KL和原非对称Critic更新。

每条链15～18步分别用FP64 joint_log_ratio/D执行符号相关裁剪，四步共享同一GAE。
两遍流式计算按4*N_global和各步D准确归一化，条件图在每个Adam版本重建。
PPO/BC梯度SUM后统一裁剪，不额外平均world size；正式BC关闭时完全不计算BC。

每个minibatch记录真实参数、同噪声最终输出及联合概率变化，坐标KL或裁剪饱和可
提前停止。新rollout从名义LR试探，剩余试探步长过小时保留此前
接受更新并结束剩余小批次；丢弃步骤的BC游标、控制器与RNG同时恢复。最后仍将
优化器LR恢复为名义值，Adam动量不清零。所有旧rollout链的四步KL仍须通过；Actor/Critic/Adam/BC/RNG回滚由
training统一管理。BC设置每步全局样本下限，监控实际样本量及成本。保持Critic
结构、价值目标、物理时钟、原奖励和真实采样核，不把数值通过称为物理收益。
"""
from __future__ import annotations

import math
import time
import copy
from collections import Counter
import torch
from .execution_checks import policy_phase
from .gradient_merge import merge_global_gradients, take_gradients
from .tensor_cache import ConditionGraphCache
from .optimization_support import _owned, _parameters, _old_kernel, balanced_epoch_order, _local_phase


def _eligible(manifest,actor=True):
    return [i for i,m in enumerate(manifest) if m['valid'] and (m['has_free'] or not actor)]


def _actor_update(policy,optimizer,rows,targets,cache,manifest,controller,*,distributed,anchor,
                    generator,ppo_epochs=1,minibatch_chains=2048,microbatch=128,sampled_steps=4,
                    clip=.01,gamma_denoising=1.,soft_kl=.0009,kl_reduction='free_coordinate_mean',
                    objective_reduction='free_coordinate_mean',grad_clip_norm=1.,reserve_attempt=None,
                    gradient_observer=None,step_observer=None,bc_samples_per_rollout=2048,
                    bc_min_global_batch=1,snr=False,step_strategy='intermediate4',detailed_diagnostics=False,
                    max_active_clip=.95,trust_region=None,kl_limits=None,max_optimizer_steps=None,candidate_observer=None,
                    bc_enabled=True,audit_full_rollout_objective=False,full_rollout_observer=None,
                    audit_full_gradient_direction=False):
    """四步独立代理比值及裁剪；两遍微批反传与全局SUM严格保持同一目标。"""
    from .chain_ppo import evaluate_chain_records,from_records,backward_chain_surrogate
    from .gaussian_probability import stepwise_diagnostics,distribution
    from .actor_backtracking import backtracked_step
    from .actor_optimizer import validate_actor_optimizer,actor_base_lr,set_actor_base_lr,actor_learning_rates
    from .actor_step_diagnostics import StepDiagnostics,output_probe,compare_probes
    from .parallel_support import local_call
    from .parallel_support import capture_local_rng,restore_local_rng
    if step_strategy!='intermediate4' or sampled_steps!=4 or tuple(policy.trainable_steps)!=(15,16,17,18):
        raise ValueError('PPO must optimize exactly stochastic transitions 15..18')
    if (policy.steps!=20 or ppo_epochs<1 or minibatch_chains<1 or microbatch<1 or
            gamma_denoising!=1. or kl_reduction!='free_coordinate_mean' or objective_reduction!='free_coordinate_mean'):
        raise ValueError('Stepwise PPO requires coordinate-mean proxies and one undiscounted upper advantage')
    validate_actor_optimizer(policy.actor,optimizer)
    policy._accepted_kl_receipt=None
    selected=_eligible(manifest);reports=[];orders=[];stop_reason=None;covered=set();discarded=None
    if not selected:raise ValueError('No free valid behavior chain remains for PPO')
    planned_steps=ppo_epochs*math.ceil(len(selected)/minibatch_chains)
    if max_optimizer_steps is not None and (type(max_optimizer_steps)is not int or max_optimizer_steps<1):
        raise ValueError('Explicit diagnostic optimizer limit must be positive')
    bc_total=max(bc_samples_per_rollout,planned_steps*bc_min_global_batch) if bc_enabled else 0
    bc_batches=[bc_total//planned_steps+(i<bc_total%planned_steps) for i in range(planned_steps)]
    actor=policy.actor;device=next(actor.parameters()).device;sketch={} if snr else None
    def collect_records(local,owned,count):
        with _local_phase(distributed,'full_chain_ratio_forward'),policy_phase(policy):
            records=evaluate_chain_records(policy,local,cache,microbatch=microbatch)
            adv=cache.get('advantages',local).double() if local else torch.zeros(0,device=device,dtype=torch.float64)
        packed=torch.cat((records.reshape(len(local),32),adv[:,None]),1)
        full=distributed.gather_rows(packed,[position for position,_ in owned],count)
        return records,full[:,:32].reshape(count,4,8),full[:,32]
    objective_aware=bool((trust_region or {}).get('objective_aware',False))
    audit_full_rollout_objective=audit_full_rollout_objective or objective_aware or audit_full_gradient_direction
    full_current=full_adv=full_before=None
    objective_history=[];source_names=[]
    selected_position={index:position for position,index in enumerate(selected)}
    if audit_full_rollout_objective:
        all_owned=_owned(selected,manifest,distributed)
        _,full_current,full_adv=collect_records([rows[i] for _,i in all_owned],all_owned,len(selected))
        full_before=stepwise_diagnostics(from_records(full_current),full_adv,clip=clip)
        labels=[(position,rows[i].metadata.get('training_task',{}).get('dataset','unknown')) for position,i in all_owned]
        names={position:name for shard in distributed.all_gather_object(labels) for position,name in shard}
        source_names=[names[i] for i in range(len(selected))]
        objective_history.append(dict(stage='before_update',**full_before))
        if full_rollout_observer is not None:full_rollout_observer('before_update',full_current,full_adv,source_names)
    for epoch in range(ppo_epochs):
        order=balanced_epoch_order(selected,manifest,generator) if distributed.rank==0 else None
        order=distributed.broadcast_object(order)
        if detailed_diagnostics:orders.append(order)
        for begin in range(0,len(order),minibatch_chains):
            indices=order[begin:begin+minibatch_chains]
            accepted_full_current=full_current
            owned=_owned(indices,manifest,distributed);local=[rows[i] for _,i in owned]
            optimizer.zero_grad(set_to_none=True);actor.eval();started=time.perf_counter()
            if audit_full_rollout_objective:
                positions=[selected_position[i] for i in indices]
                full=full_current[positions];adv=full_adv[positions]
                records=full[[position for position,_ in owned]]
                step_full_before=stepwise_diagnostics(from_records(full_current),full_adv,clip=clip)
            else:
                records,full,adv=collect_records(local,owned,len(indices))
                step_full_before=None
            ratio_seconds=max(distributed.all_gather_object(time.perf_counter()-started))
            statistics=from_records(full)
            pre=stepwise_diagnostics(statistics,adv,clip=clip)
            if soft_kl is not None and pre['mean_coordinate_kl']>=soft_kl:stop_reason='pre_step_mean_coordinate_kl'
            elif pre['active_clip_fraction']>=max_active_clip:stop_reason='active_clip_saturation'
            if stop_reason:
                discarded=dict(epoch=epoch,global_chain_count=len(indices),diagnostics=pre,
                    wall_seconds=time.perf_counter()-started,optimizer_step_executed=False)
                break
            # 仅保存轻量学习状态；若本小批次未执行有效更新，BC游标/控制器和RNG也恢复。
            pending_control=copy.deepcopy(controller.state_dict()) if trust_region is not None else None
            pending_anchor=copy.deepcopy(anchor.state_dict()) if trust_region is not None and bc_enabled else None
            pending_rng=capture_local_rng({'actor':generator}) if trust_region is not None else None
            probe_before=local_call(distributed,lambda:output_probe(policy,local,cache,max_chains=2))
            gradient_started=time.perf_counter()
            with _local_phase(distributed,'full_chain_streamed_backward'),policy_phase(policy):
                backward=backward_chain_surrogate(policy,local,cache,records,clip=clip,
                    global_chain_count=len(indices),microbatch=microbatch)
            ppo=take_gradients(actor)
            ppo_seconds=max(distributed.all_gather_object(time.perf_counter()-gradient_started))
            bc_started=time.perf_counter()
            global_bc=bc_batches[len(reports)]
            if bc_enabled:
              with _local_phase(distributed,'stage2_bc_local_backward'):
                from .dense_policy import dense_precision_scope
                anchor.batch_size=global_bc//distributed.world_size+(distributed.rank<global_bc%distributed.world_size)
                with dense_precision_scope(policy.precision):
                    bc_report=anchor.backward(actor,weight=anchor.batch_size/global_bc)
                bc=take_gradients(actor)
            else:
                bc_report=dict(loss=0.,batch_size=0,samples=[])
                bc={}
            bc_seconds=max(distributed.all_gather_object(time.perf_counter()-bc_started))
            observation=local_call(distributed,lambda:StepDiagnostics(actor) if distributed.rank==0 else None)
            contributions=merge_global_gradients(actor,ppo,bc,controller,distributed=distributed,half_split=snr,sketch=sketch,
                component_observer=observation.capture_component if (detailed_diagnostics or audit_full_rollout_objective) and observation is not None else None,
                bc_enabled=bc_enabled)
            if gradient_observer is not None:gradient_observer(actor,len(reports))
            direction=None
            if audit_full_gradient_direction:
                from .optimizer_direction import FullRolloutDirection
                all_local=[rows[i] for _,i in all_owned]
                all_records=full_current[[position for position,_ in all_owned]]
                def full_direction_backward():
                    # 先完成各rank局部反向并同步异常，再进入梯度SUM；否则单卡异常
                    # 会使其他卡停在NCCL，而出错卡已进入控制组异常同步。
                    with _local_phase(distributed,'diagnostic_full_rollout_gradient'),policy_phase(policy):
                        backward_chain_surrogate(policy,all_local,cache,all_records,clip=clip,
                            global_chain_count=len(selected),microbatch=microbatch)
                direction=FullRolloutDirection.capture(actor,distributed,full_direction_backward)
            with _local_phase(distributed,'stage2_actor_clip'):
                total_norm=float(torch.nn.utils.clip_grad_norm_([p for p in actor.parameters() if p.requires_grad],
                    grad_clip_norm,error_if_nonfinite=True))
            clip_factor=min(1.,grad_clip_norm/(total_norm+1e-6))
            # 只在rank0保留可训练参数前值来测实际FP32更新，不复制Adam或冻结网络。
            transaction_started=time.perf_counter();kl_elapsed=0.
            post_records=None;post=None;transaction=None
            if trust_region is not None:
                # 目标感知与KL-only对照都检查全部真实有效旧链，包含尚未用于梯度的链。
                # 数值/坐标门限和显式可选联合保护各用自己的量纲，最后仍完整复核。
                guarded=set(selected)
                guard_indices=[i for i in selected if i in guarded]
                guard_owned=_owned(guard_indices,manifest,distributed)
                guard_local=[rows[i] for _,i in guard_owned]
                limits={k:v*trust_region['margin'] for k,v in kl_limits.items()
                    if (k.startswith('chain_joint_') or k in ('mean_coordinate','maximum_step_mean_coordinate','p95_coordinate'))
                    and v is not None}
                if soft_kl is not None:
                    limits['mean_coordinate']=min(limits.get('mean_coordinate',float('inf')),soft_kl*trust_region['margin'])
                limits['maximum_active_clip']=max_active_clip
                if objective_aware:
                    limits['objective_reference']=step_full_before['objective_mean']
                    limits['objective_tolerance']=trust_region.get('objective_tolerance',1e-10)
                set_actor_base_lr(optimizer,min(trust_region['maximum_lr'],
                    actor_base_lr(optimizer)*trust_region['growth']))
                candidate_cache={}
                def evaluate_candidate():
                    nonlocal post_records,post,kl_elapsed,full_current
                    measured=time.perf_counter()
                    if audit_full_rollout_objective:
                        _,candidate_records,candidate_adv=collect_records([rows[i] for _,i in all_owned],all_owned,len(selected))
                        full_current=candidate_records
                        positions=[selected_position[i] for i in guard_indices]
                        post_records=candidate_records[positions];post_adv=candidate_adv[positions]
                    else:
                        _,post_records,post_adv=collect_records(guard_local,guard_owned,len(guard_indices))
                    post=stepwise_diagnostics(from_records(post_records),post_adv,clip=clip)
                    if objective_aware:candidate_cache[actor_base_lr(optimizer)]=(full_current,post_records,post)
                    kl_elapsed+=time.perf_counter()-measured
                    return post
                transaction=backtracked_step(actor,optimizer,controller,evaluate=evaluate_candidate,
                    limits=limits,distributed=distributed,generators={'actor':generator},
                    max_attempts=trust_region['max_attempts'],reserve_attempt=reserve_attempt,
                    observe=candidate_observer,minimum_lr=trust_region['maximum_lr']*
                        trust_region.get('minimum_trial_lr_ratio',.0625),
                    select_best=objective_aware,search_attempts=trust_region.get('objective_search_attempts',3))
                if transaction['accepted'] and objective_aware:
                    full_current,post_records,post=candidate_cache[transaction['accepted_lr']]
                if not transaction['accepted']:
                    if transaction['finite_trust_region_exhausted']:
                        controller.load_state_dict(pending_control)
                        if bc_enabled:anchor.load_state_dict(pending_anchor)
                        full_current=accepted_full_current
                        restore_local_rng(pending_rng,{'actor':generator})
                        optimizer.zero_grad(set_to_none=True)
                        stop_reason='rollout_no_useful_trust_region_step'
                        discarded=dict(epoch=epoch,global_chain_count=len(indices),diagnostics=pre,
                            trust_region=transaction,optimizer_step_executed=bool(transaction['attempt_count']),
                            accepted_optimizer_steps=0,pending_BC_and_RNG_restored=True,
                            wall_seconds=max(distributed.all_gather_object(time.perf_counter()-started)))
                        del observation,direction
                        break
                    import json
                    raise FloatingPointError('Stepwise PPO numerical trust-region failure; whole iteration rollback: '+json.dumps(transaction))
            else:
                if reserve_attempt is not None:reserve_attempt()
                with _local_phase(distributed,'stage2_actor_optimizer_step'):optimizer.step()
            transaction_seconds=max(distributed.all_gather_object(time.perf_counter()-transaction_started))
            parameter_report=local_call(distributed,lambda:observation.finish(optimizer,contributions['bc_weight'],clip_factor)
                if observation is not None else None)
            parameter_report=distributed.broadcast_object(parameter_report)
            direction_report=local_call(distributed,lambda:direction.finish(observation.before) if direction is not None else None)
            direction_report=distributed.broadcast_object(direction_report)
            del direction
            del observation
            probe_after=local_call(distributed,lambda:output_probe(policy,local,cache,max_chains=2))
            output_change=compare_probes(probe_before,probe_after,distributed)
            if post is None:
                measured=time.perf_counter()
                if audit_full_rollout_objective:
                    _,full_current,post_adv=collect_records([rows[i] for _,i in all_owned],all_owned,len(selected))
                    post_records=full_current
                else:
                    _,post_records,post_adv=collect_records(local,owned,len(indices))
                post=stepwise_diagnostics(from_records(post_records),post_adv,clip=clip)
                kl_elapsed+=time.perf_counter()-measured
            kl_seconds=max(distributed.all_gather_object(kl_elapsed))
            local_sources=Counter(meta['dataset_id'] for sample in bc_report.get('samples',[]) for meta in sample)
            shards=distributed.all_gather_object(dict(batch=bc_report['batch_size'],loss=bc_report['loss'],sources=dict(local_sources)))
            sources=Counter()
            for shard in shards:sources.update(shard['sources'])
            covered.update(indices)
            full_after=None;objective_groups=None
            if audit_full_rollout_objective:
                full_after=stepwise_diagnostics(from_records(full_current),full_adv,clip=clip)
                from .objective_audit import grouped_objective
                objective_groups=grouped_objective(from_records(full_current),full_adv,source_names,clip=clip,
                    before_statistics=from_records(accepted_full_current))
                objective_history.append(dict(stage='accepted_Adam',optimizer_step=len(reports)+1,**full_after))
                if full_rollout_observer is not None:full_rollout_observer('accepted_Adam',full_current,full_adv,source_names)
            report=dict(objective='stepwise_coordinate_mean_clipped_ppo',ratio_is_joint_probability=False,
                denominator='4_times_global_valid_behavior_chain_count',ppo_clip_applied=True,epoch=epoch,
                optimizer_step=len(reports)+1,global_chain_count=len(indices),sampled_steps=4,
                denoising_steps=list(policy.trainable_steps),internal_sample_visits=len(indices)*4,
                full_objective_transitions=len(indices)*4,ppo_loss=float(distributed.sum_tensor(backward['loss'])),
                streaming_recompute_max_error=max(distributed.all_gather_object(float(backward['recompute_max_error']))),
                clip_fraction=pre['clip_fraction'],objective_clip_fraction=pre['active_clip_fraction'],
                proxy_clip_fraction=post['proxy_clip_fraction'],proxy_active_clip_fraction=post['proxy_active_clip_fraction'],
                mean_proxy_ratio=post['mean_proxy_ratio'],proxy_ratio_p95=post['proxy_ratio_p95'],proxy_ratio_p99=post['proxy_ratio_p99'],
                pre_step_chain=pre,post_step_chain=post,
                full_rollout_before=step_full_before,full_rollout_after=full_after,full_rollout_groups=objective_groups,
                pre_step_mean_joint_kl=pre['chain_joint_kl']['mean'],
                pre_step_mean_coordinate_kl=float(statistics['mean_coordinate_kl'].mean()),
                per_denoising_step=[dict(step=s+15,joint_kl=distribution(post_records[:,s,0]),
                    mean_coordinate_kl=distribution(post_records[:,s,1])) for s in range(4)],
                gradient_contributions=contributions,total_gradient_norm=total_norm,gradient_clip_factor=clip_factor,
                parameter_update=parameter_report,full_gradient_direction=direction_report,
                output_change=output_change,trust_region=transaction,
                candidate_checked_chains=len(guard_indices) if trust_region is not None else len(indices),
                candidate_check_scope='full_rollout' if trust_region is not None else 'current_minibatch',
                bc_global_samples=sum(s['batch'] for s in shards),bc_samples_by_source=dict(sources),
                bc_loss=sum(s['batch']*s['loss'] for s in shards)/global_bc if global_bc else 0.,
                bc_enabled=bc_enabled,
                learning_rate=actor_base_lr(optimizer),learning_rates=actor_learning_rates(optimizer),
                ppo_ratio_seconds=ratio_seconds,
                ppo_backward_seconds=ppo_seconds,ppo_seconds=ratio_seconds+ppo_seconds,
                bc_seconds=bc_seconds,post_step_kl_seconds=kl_seconds,optimizer_transaction_seconds=transaction_seconds,
                wall_seconds=max(distributed.all_gather_object(time.perf_counter()-started)))
            reports.append(report)
            if step_observer is not None:step_observer(report)
            # 回溯覆盖全部有效链；接受与否还要独立整轮目标与KL审查。
            if soft_kl is not None and post['mean_coordinate_kl']>=soft_kl:stop_reason='post_step_mean_coordinate_kl'
            elif post['active_clip_fraction']>=max_active_clip:stop_reason='post_step_active_clip_saturation'
            if trust_region is not None and stop_reason is None:
                usage=kl_budget_fraction(post,limits)
                report['rollout_kl_budget_fraction']=usage
                # 接近额度只是诊断，不能单独证明没有有意义的更新空间；下一小批次
                # 继续真实试探，只有有限回溯仍拒绝且步长耗尽才保留前序更新并结束。
            if stop_reason:break
            if max_optimizer_steps is not None and len(reports)>=max_optimizer_steps:
                stop_reason='explicit_diagnostic_optimizer_limit';break
        if stop_reason:break
    discarded_attempts=((discarded or {}).get('trust_region') or {}).get('attempt_count',0)
    return dict(objective='stepwise_coordinate_mean_clipped_ppo',ppo_clip_applied=True,optimizer_steps=len(reports),
        objective_aware=objective_aware,objective_tolerance=(trust_region or {}).get('objective_tolerance',1e-10),
        attempted_optimizer_steps=sum((r['trust_region'] or {}).get('attempt_count',1) for r in reports)+discarded_attempts,
        rejected_step_candidates=sum((r['trust_region'] or {}).get('backtracks',0) for r in reports)+discarded_attempts,
        executed_optimizer_steps=sum((r['trust_region'] or {}).get('attempt_count',1) for r in reports)+discarded_attempts,
        minibatch_accepted_steps=len(reports),executed_step_scope='actual_Adam_calls_including_rejected_candidates',
        accepted_optimizer_steps=None,acceptance_scope='pending_full_rollout_stepwise_objective_and_KL',
        planned_optimizer_steps=planned_steps,skipped_optimizer_steps=planned_steps-len(reports),
        steps=reports,epoch_orders=orders,early_stop_reason=stop_reason,discarded_pending_minibatch=discarded,
        gradient_snr=None,_ppo_gradient_sketch=sketch,epoch_order_rng_contract='fixed_four_steps_shuffled_complete_chains',
        full_rollout_objective_history=objective_history,bc_enabled=bc_enabled,
        eligible_chains=len(selected),applied_unique_chains=len(covered),applied_chain_fraction=len(covered)/len(selected),
        internal_sample_visits=sum(r['internal_sample_visits'] for r in reports),
        actual_global_bc_samples=sum(r['bc_global_samples'] for r in reports),planned_global_bc_samples=bc_total,
        bc_min_global_batch=bc_min_global_batch,
        bc_samples_by_source={source:sum(r['bc_samples_by_source'].get(source,0) for r in reports)
            for source in sorted({source for r in reports for source in r['bc_samples_by_source']})},
        planned_internal_visits=len(selected)*4*ppo_epochs,objective_reduction='free_coordinate_mean',
        proxy_is_joint_probability=False,old_statistics_fixed=True,final_trainable4_kl_pending=True)


def actor_update(policy,optimizer,*args,trust_region=None,**kwargs):
    """每次新rollout独立试探LR；Adam动量不清空，临时回溯LR不传给下一批。

    四步独立代理目标与模块倍率保持不变。接受步骤的实际LR记录在steps中，优化器
    离开本函数时恢复共同名义LR，checkpoint不会把本批剩余额度误作长期LR。
    正常结束剩余小批次仍须通过调用方完整rollout最终KL；异常继续由整轮事务恢复。
    """
    from .actor_optimizer import actor_base_lr,set_actor_base_lr,actor_learning_rates
    incoming=actor_learning_rates(optimizer)
    nominal=None if trust_region is None else trust_region['maximum_lr']
    if nominal is not None:set_actor_base_lr(optimizer,nominal)
    try:
        result=_actor_update(policy,optimizer,*args,trust_region=trust_region,**kwargs)
        last_accepted=(result.get('steps') or [{}])[-1].get('learning_rate')
        discarded_trials=(((result.get('discarded_pending_minibatch') or {}).get('trust_region') or {})
            .get('attempts') or [])
        last_trial=(discarded_trials[-1]['learning_rate'] if discarded_trials else
            last_accepted if last_accepted is not None else actor_base_lr(optimizer))
        result.update(nominal_learning_rate=nominal,incoming_optimizer_learning_rates=incoming,
            last_accepted_learning_rate=last_accepted,last_trial_learning_rate=last_trial,
            rollout_learning_rate_reset=nominal is not None)
        return result
    finally:
        if nominal is not None:set_actor_base_lr(optimizer,nominal)


@torch.no_grad()
def final_kl(policy,rows,cache,manifest,*,distributed,microbatch=256,clip=.01,objective_reduction='free_coordinate_mean'):
    """所有有效链的真实联合KL独立保留；PPO目标与裁剪按四个内部转移计算。"""
    from .chain_ppo import evaluate_chain_records,from_records
    from .gaussian_probability import stepwise_diagnostics,distribution
    if objective_reduction!='free_coordinate_mean':raise ValueError('Final PPO diagnostics require independent coordinate-mean steps')
    selected=_eligible(manifest);owned=_owned(selected,manifest,distributed)
    local=[rows[i] for _,i in owned];device=next(policy.actor.parameters()).device
    start=time.perf_counter()
    with _local_phase(distributed,'stage2_final_trainable4_kl'),policy_phase(policy):
        records=evaluate_chain_records(policy,local,cache,microbatch=microbatch,physical_changes=True)
        adv=cache.get('advantages',local).double() if local else torch.zeros(0,device=device,dtype=torch.float64)
    packed=torch.cat((records.reshape(len(local),32),adv[:,None]),1)
    gathered=distributed.gather_rows(packed,[p for p,_ in owned],len(selected))
    full=gathered[:,:32].reshape(-1,4,8);adv=gathered[:,32]
    joint,coordinate,logratio,counts,shift=[full[:,:,i] for i in range(5)]
    chain=stepwise_diagnostics(from_records(full),adv,clip=clip)
    frozen_identity=policy.assert_frozen_identity()
    identities=distributed.all_gather_object(frozen_identity)
    if any(v!=identities[0] for v in identities):raise RuntimeError('Frozen prior identity differs across ranks')
    kl_seconds=max(distributed.all_gather_object(time.perf_counter()-start))
    reference_change=reference_change_diagnostics(policy,local,cache,distributed)
    chain.update(mean_joint_kl=float(joint.mean()),mean_coordinate_kl=float(coordinate.mean()),
        joint_kl=distribution(joint),coordinate_kl=distribution(coordinate),
        maximum_step_mean_coordinate_kl=float(coordinate.mean(0).max()),
        per_denoising_step=[dict(chain['per_denoising_step'][s],
            mean_normalized_mean_change_rms=float(shift[:,s].mean())) for s in range(4)],
        covered_chains=len(selected),covered_internal_transitions=len(selected)*4,
        gaussian_kl_steps=list(policy.trainable_steps),statistical_denominator='one_complete_chain_four_joint_Gaussian_transitions',
        frozen_identity=frozen_identity,
        terminal_two_step_joint_kl_fraction=float(joint[:,-2:].sum()/joint.sum()) if joint.sum()>0 else None,
        joint_log_ratio_range=[chain['chain_joint_log_ratio']['min'],chain['chain_joint_log_ratio']['max']],
        effective_mean_change=dict(mean_rms=float(shift.mean()),max_rms=float(shift.max()),
            unchanged_internal_fraction=float((shift==0).double().mean()),scope='new_means_on_fixed_old_sampling_path'),
        objective_clip_fraction=chain['active_clip_fraction'],
        final_reference_mean_change=dict(root_feature_rms_m=distribution(full[:,-1,5]),
            rotation6d_rms_unitless=distribution(full[:,-1,6]),joint_rms_rad=distribution(full[:,-1,7]),
            root_semantics='heading_local_xy_increments_and_root_height_not_world_xyz',
            scope='denormalized_index18_Gaussian_mean_not_final_reference'),
        kl_scope='old_rollout_conditional_chain_KL_estimate_not_future_physics_safety',
        p95_scope='sum_four_joint_KLs_per_complete_chain',reference_change=reference_change,
        kl_only_wall_seconds=kl_seconds,wall_seconds=max(distributed.all_gather_object(time.perf_counter()-start)))
    return chain


def _quaternion_angle(a,b):
    a=torch.nn.functional.normalize(a.double(),dim=-1)
    b=torch.nn.functional.normalize(b.double(),dim=-1)
    return 2*torch.acos((a*b).sum(-1).abs().clamp(max=1.))


@torch.no_grad()
def reference_change_diagnostics(policy,rows,cache,distributed,max_local_chains=16):
    started=time.perf_counter()
    selected=rows[:max_local_chains]
    changes=[]
    with _local_phase(distributed,'stage2_updated_reference_diagnostic'),policy_phase(policy):
        if selected:
            context=cache.context(selected);prepared=policy.prepare_conditions(context)
            state=cache.get('chain',selected,[15]*len(selected))
            for step in policy.trainable_steps:
                params=policy.transition_parameters(context,state,step,prepared=prepared)
                state=policy.actor._constrain(params['mean']+params['std']*
                    cache.get('sampling_noise',selected,[step]*len(selected)),prepared['adapted'])
            state=policy.transition_parameters(context,state,19,prepared=prepared)['mean']
            before=cache.get('chain',selected,[20]*len(selected))
            free=cache.get('free_mask',selected)
            delta=(state.double()-before.double())*policy.actor.endecoder.std.double()
            physical_before=policy.actor.endecoder.denormalize(before)
            physical_after=policy.actor.endecoder.denormalize(state)
            q0=policy.actor.endecoder.codec.decode_to_canonical_qpos(physical_before)
            q1=policy.actor.endecoder.codec.decode_to_canonical_qpos(physical_after)
            mask=context['future_valid']
            for i in range(len(selected)):
                joint=free[i,:,9:]
                changes.append(dict(normalized_rms=float((state[i]-before[i]).square()[free[i]].mean().sqrt()),
                    joint_rms_rad=float(delta[i,:,9:].square()[joint].mean().sqrt()) if joint.any() else None,
                    root_position_rms_m=float((q1[i,:,:3]-q0[i,:,:3]).square()[mask[i]].mean().sqrt()),
                    root_orientation_rms_rad=float(_quaternion_angle(q0[i,:,3:7],q1[i,:,3:7])[mask[i]].square().mean().sqrt()),
                    maximum_joint_change_rad=float(delta[i,:,9:].abs()[joint].max()) if joint.any() else None,
                    protected_coordinates_unchanged=bool(torch.equal(state[i][~free[i]],before[i][~free[i]]))))
    all_rows=[r for shard in distributed.all_gather_object(changes) for r in shard]
    return dict(scope='regenerated_final_deterministic_reference_same_initial_state_and_saved_innovations_not_physics',
        covered_chains=len(all_rows),selection='first_up_to_16_eligible_chains_per_rank',samples=all_rows,
        wall_seconds=max(distributed.all_gather_object(time.perf_counter()-started)))


def check_kl(report,limits):
    fields=dict(mean_coordinate='mean_coordinate_kl',mean_joint='mean_joint_kl',
        maximum_step_mean_coordinate='maximum_step_mean_coordinate_kl')
    failures={}
    # 即使联合尾部门限显式未标定，也不能关闭真实KL的数值检查。
    for name in ('mean_coordinate_kl','maximum_step_mean_coordinate_kl','mean_joint_kl'):
        if name in report and not math.isfinite(report[name]):
            failures['nonfinite_'+name]=dict(value=report[name],limit='finite_required')
    for family in ('coordinate_kl','joint_kl','chain_joint_kl'):
        for name,value in report.get(family,{}).items():
            if not math.isfinite(value):failures['nonfinite_'+family+'_'+name]=dict(value=value,limit='finite_required')
    for key,field in fields.items():
        bound=limits.get(key)
        if bound is not None and (not math.isfinite(report[field]) or report[field]>bound):
            failures[key]=dict(value=report[field],limit=bound)
    p95=limits.get('p95_coordinate')
    if p95 is not None and (not math.isfinite(report['coordinate_kl']['p95']) or report['coordinate_kl']['p95']>p95):
        failures['p95_coordinate']=dict(value=report['coordinate_kl']['p95'],limit=p95)
    return dict(passed=not failures,failures=failures,units='explicit_joint_or_coordinate_never_reuse_one_threshold')


def kl_budget_fraction(report,limits):
    """仅汇总显式启用的同量纲门限；未标定联合保护不冒用旧数字。"""
    fractions=[report[field]/limits[key] for key,field in (
        ('mean_coordinate','mean_coordinate_kl'),('maximum_step_mean_coordinate','maximum_step_mean_coordinate_kl'))
        if limits.get(key) is not None]
    if limits.get('p95_coordinate') is not None:fractions.append(report['coordinate_kl']['p95']/limits['p95_coordinate'])
    fractions.extend(report['chain_joint_kl'][name]/limits['chain_joint_'+name] for name in ('mean','p95','p99','max')
        if limits.get('chain_joint_'+name) is not None)
    return max(fractions,default=0.)


def critic_update(critic,optimizer,rows,targets,cache,manifest,*,distributed,generator,epochs=4,batch=1024,grad_clip=1.):
    critic.train()
    selected=_eligible(manifest,False);owned=_owned(selected,manifest,distributed);local_indices=[i for _,i in owned]
    local_rows=[rows[i] for i in local_indices]
    cached_features=hasattr(critic,'encode_frozen') and getattr(critic,'cache_frozen_conditions',False)
    with _local_phase(distributed,'stage10_critic_frozen_encoding'):
        encoded=(critic.encode_frozen(cache.critic_context(local_rows)) if cached_features else cache.critic_context(local_rows)) if local_rows else {}
    lookup={i:j for j,i in enumerate(local_indices)}
    target=torch.as_tensor(targets['returns'],device=cache.device,dtype=torch.float32)
    def forward(indices):
        idx=torch.tensor([lookup[i] for i in indices],device=cache.device)
        features={k:v.index_select(0,idx) for k,v in encoded.items()}
        remaining=cache.get('remaining_music_seconds',[rows[i] for i in indices])
        return critic.forward_encoded(features,remaining) if cached_features else critic(features,remaining)
    @torch.no_grad()
    def metrics():
        with _local_phase(distributed,'stage10_critic_value_metrics'):
            totals=torch.zeros(6,device=cache.device,dtype=torch.float64)
            if local_indices:
                expected=target[local_indices].double();error=expected-forward(local_indices).double()
                totals=torch.stack((expected.new_tensor(len(expected)),expected.sum(),expected.square().sum(),error.sum(),error.square().sum(),error.abs().sum()))
        n,s,s2,e,e2,absolute=distributed.sum_tensor(totals).cpu().tolist()
        variance=max(0.,s2/n-(s/n)**2);error_variance=max(0.,e2/n-(e/n)**2)
        return dict(mse=e2/n,mae=absolute/n,bias=e/n,target_variance=variance,
            explained_variance=None if variance<=1e-12 else 1-error_variance/variance,count=int(n))
    initial=metrics();losses=[];norms=[]
    for epoch in range(epochs):
        order=balanced_epoch_order(selected,manifest,generator) if distributed.rank==0 else None
        order=distributed.broadcast_object(order)
        for offset in range(0,len(order),batch):
            indices=order[offset:offset+batch];local=[i for _,i in _owned(indices,manifest,distributed)]
            optimizer.zero_grad(set_to_none=True)
            with _local_phase(distributed,'stage10_critic_trainable_head'):
                loss=torch.zeros((),device=cache.device)
                if local:
                    loss=.5*(forward(local)-target[local]).square().sum()/len(indices)
                    loss.backward()
            distributed.sum_gradients(critic)
            with _local_phase(distributed,'stage10_critic_clip_and_step'):
                norms.append(torch.nn.utils.clip_grad_norm_([p for p in critic.parameters() if p.requires_grad],grad_clip,error_if_nonfinite=True).detach())
                optimizer.step()
            losses.append(loss.detach())
    losses=distributed.sum_tensor(torch.stack(losses)).cpu().tolist() if losses else []
    norms=torch.stack(norms).cpu().tolist() if norms else []
    final=metrics()
    return dict(before=initial,after=final,optimizer_steps=len(losses),epochs=epochs,losses=losses,gradient_norms=norms,
        frozen_encoder_passes=int(cached_features and bool(local_rows)),feature_bytes=sum(v.numel()*v.element_size() for v in encoded.values()),
        critic_feature_contract='frozen_music_prefix_cached_trainable_history_recomputed' if cached_features else 'recompute_trainable_encoders_each_step',
        global_sample_visits=epochs*len(selected),metric_scope='fixed_rollout_bootstrap_targets_next_batch_before_update_is_separate')
