"""完整四步联合Clipped PPO与Stage1 BC、整轮联合KL和非对称Critic更新。

每条链15～18步自由坐标的FP64 log-ratio先求和，只执行一次符号相关PPO裁剪，
对应同一个上层GAE。两遍流式计算保持完整链梯度，条件图在每个Adam版本重建。
全局实际链数是唯一分母；PPO/BC梯度SUM后统一裁剪，不额外平均四步或world size。

每个minibatch记录真实参数、同噪声最终输出及联合概率变化，链KL/ESS/裁剪饱和可
提前停止。最终所有旧rollout链的四步KL仍须通过；Actor/Critic/Adam/BC/RNG回滚由
training统一管理。BC设置每步全局样本下限，监控实际样本量及成本。保持Critic
结构、价值目标、物理时钟、原奖励和真实采样核，不把数值通过称为物理收益。
"""
from __future__ import annotations

import math
import time
from collections import Counter
import torch
from .execution_checks import policy_phase
from .gradient_merge import merge_global_gradients, take_gradients
from .gaussian_probability import gaussian_statistics, ppo_surrogate
from .tensor_cache import ConditionGraphCache
from .optimization_support import _owned, _parameters, _old_kernel, balanced_epoch_order, _local_phase


def _eligible(manifest,actor=True):
    return [i for i,m in enumerate(manifest) if m['valid'] and (m['has_free'] or not actor)]


def actor_update(policy,optimizer,rows,targets,cache,manifest,controller,*,distributed,anchor,
                    generator,ppo_epochs=1,minibatch_chains=2048,microbatch=128,sampled_steps=4,
                    clip=.10,gamma_denoising=1.,soft_kl=.10,kl_reduction='chain_joint',
                    objective_reduction='joint_sum',grad_clip_norm=1.,reserve_attempt=None,
                    gradient_observer=None,step_observer=None,bc_samples_per_rollout=2048,
                    bc_min_global_batch=1,snr=False,step_strategy='intermediate4',detailed_diagnostics=False,
                    max_active_clip=.95,min_ess_fraction=.05,trust_region=None,kl_limits=None):
    """每条完整链一个联合比值；两遍微批反传与全局SUM严格保持同一目标。"""
    from .chain_ppo import evaluate_chain_records,from_records,backward_chain_surrogate
    from .gaussian_probability import chain_diagnostics,distribution
    from .actor_backtracking import uniform_actor_optimizer,backtracked_step
    from .actor_step_diagnostics import StepDiagnostics,output_probe,compare_probes
    from .parallel_support import local_call
    if step_strategy!='intermediate4' or sampled_steps!=4 or tuple(policy.trainable_steps)!=(15,16,17,18):
        raise ValueError('PPO must optimize exactly stochastic transitions 15..18')
    if (policy.steps!=20 or ppo_epochs<1 or minibatch_chains<1 or microbatch<1 or
            gamma_denoising!=1. or kl_reduction!='chain_joint' or objective_reduction!='joint_sum'):
        raise ValueError('Full-chain PPO requires joint-sum, chain KL and one undiscounted upper advantage')
    uniform_actor_optimizer(policy.actor,optimizer)
    policy._accepted_kl_receipt=None
    selected=_eligible(manifest);reports=[];orders=[];stop_reason=None;covered=set();discarded=None
    if not selected:raise ValueError('No free valid behavior chain remains for PPO')
    planned_steps=ppo_epochs*math.ceil(len(selected)/minibatch_chains)
    bc_total=max(bc_samples_per_rollout,planned_steps*bc_min_global_batch)
    bc_batches=[bc_total//planned_steps+(i<bc_total%planned_steps) for i in range(planned_steps)]
    actor=policy.actor;device=next(actor.parameters()).device;sketch={} if snr else None
    def collect_records(local,owned,count):
        with _local_phase(distributed,'full_chain_ratio_forward'),policy_phase(policy):
            records=evaluate_chain_records(policy,local,cache,microbatch=microbatch)
            adv=cache.get('advantages',local).double() if local else torch.zeros(0,device=device,dtype=torch.float64)
        packed=torch.cat((records.reshape(len(local),32),adv[:,None]),1)
        full=distributed.gather_rows(packed,[position for position,_ in owned],count)
        return records,full[:,:32].reshape(count,4,8),full[:,32]
    for epoch in range(ppo_epochs):
        order=balanced_epoch_order(selected,manifest,generator) if distributed.rank==0 else None
        order=distributed.broadcast_object(order)
        if detailed_diagnostics:orders.append(order)
        for begin in range(0,len(order),minibatch_chains):
            indices=order[begin:begin+minibatch_chains]
            owned=_owned(indices,manifest,distributed);local=[rows[i] for _,i in owned]
            optimizer.zero_grad(set_to_none=True);actor.eval();started=time.perf_counter()
            records,full,adv=collect_records(local,owned,len(indices))
            ratio_seconds=max(distributed.all_gather_object(time.perf_counter()-started))
            statistics=from_records(full)
            pre=chain_diagnostics(statistics,adv,clip=clip)
            if soft_kl is not None and pre['chain_joint_kl']['mean']>=soft_kl:stop_reason='pre_step_chain_joint_kl'
            elif pre['active_clip_fraction']>=max_active_clip:stop_reason='active_clip_saturation'
            elif pre['importance_weight_ess_fraction']<min_ess_fraction:stop_reason='importance_weight_ess_collapse'
            if stop_reason:
                discarded=dict(epoch=epoch,global_chain_count=len(indices),diagnostics=pre,
                    wall_seconds=time.perf_counter()-started,optimizer_step_executed=False)
                break
            probe_before=local_call(distributed,lambda:output_probe(policy,local,cache,max_chains=2))
            gradient_started=time.perf_counter()
            with _local_phase(distributed,'full_chain_streamed_backward'),policy_phase(policy):
                backward=backward_chain_surrogate(policy,local,cache,records,clip=clip,
                    global_chain_count=len(indices),microbatch=microbatch)
            ppo=take_gradients(actor)
            ppo_seconds=max(distributed.all_gather_object(time.perf_counter()-gradient_started))
            bc_started=time.perf_counter()
            with _local_phase(distributed,'stage2_bc_local_backward'):
                from .dense_policy import dense_precision_scope
                global_bc=bc_batches[len(reports)]
                anchor.batch_size=global_bc//distributed.world_size+(distributed.rank<global_bc%distributed.world_size)
                with dense_precision_scope(policy.precision):
                    bc_report=anchor.backward(actor,weight=anchor.batch_size/global_bc)
                bc=take_gradients(actor)
            bc_seconds=max(distributed.all_gather_object(time.perf_counter()-bc_started))
            contributions=merge_global_gradients(actor,ppo,bc,controller,distributed=distributed,half_split=snr,sketch=sketch)
            if gradient_observer is not None:gradient_observer(actor,len(reports))
            with _local_phase(distributed,'stage2_actor_clip'):
                total_norm=float(torch.nn.utils.clip_grad_norm_([p for p in actor.parameters() if p.requires_grad],
                    grad_clip_norm,error_if_nonfinite=True))
            clip_factor=min(1.,grad_clip_norm/(total_norm+1e-6))
            # 只在rank0保留可训练参数前值来测实际FP32更新，不复制Adam或冻结网络。
            observation=local_call(distributed,lambda:StepDiagnostics(actor) if distributed.rank==0 else None)
            transaction_started=time.perf_counter();kl_elapsed=0.
            post_records=None;post=None;transaction=None
            if trust_region is not None:
                # 更新还会影响先前批次；累计已用链全部参与候选检查，避免只在新批次
                # ESS正常却破坏前面链的权重。最终仍检查包含未用链在内的完整rollout。
                guarded=covered.union(indices)
                guard_indices=[i for i in selected if i in guarded]
                guard_owned=_owned(guard_indices,manifest,distributed)
                guard_local=[rows[i] for _,i in guard_owned]
                limits={k:v*trust_region['margin'] for k,v in kl_limits.items()
                    if k.startswith('chain_joint_')}
                limits['chain_joint_mean']=min(limits['chain_joint_mean'],soft_kl*trust_region['margin'])
                limits.update(minimum_ess=min_ess_fraction,maximum_active_clip=max_active_clip)
                optimizer.param_groups[0]['lr']=min(trust_region['maximum_lr'],
                    optimizer.param_groups[0]['lr']*trust_region['growth'])
                def evaluate_candidate():
                    nonlocal post_records,post,kl_elapsed
                    measured=time.perf_counter()
                    _,post_records,post_adv=collect_records(guard_local,guard_owned,len(guard_indices))
                    post=chain_diagnostics(from_records(post_records),post_adv,clip=clip)
                    kl_elapsed+=time.perf_counter()-measured
                    return post
                transaction=backtracked_step(actor,optimizer,controller,evaluate=evaluate_candidate,
                    limits=limits,distributed=distributed,generators={'actor':generator},
                    max_attempts=trust_region['max_attempts'],reserve_attempt=reserve_attempt)
                if not transaction['accepted']:
                    import json
                    raise FloatingPointError('Full-chain trust-region exhausted; whole iteration rollback: '+json.dumps(transaction))
            else:
                if reserve_attempt is not None:reserve_attempt()
                with _local_phase(distributed,'stage2_actor_optimizer_step'):optimizer.step()
            transaction_seconds=max(distributed.all_gather_object(time.perf_counter()-transaction_started))
            parameter_report=local_call(distributed,lambda:observation.finish(optimizer,contributions['bc_weight'],clip_factor)
                if observation is not None else None)
            parameter_report=distributed.broadcast_object(parameter_report)
            del observation
            probe_after=local_call(distributed,lambda:output_probe(policy,local,cache,max_chains=2))
            output_change=compare_probes(probe_before,probe_after,distributed)
            if post is None:
                measured=time.perf_counter()
                _,post_records,post_adv=collect_records(local,owned,len(indices))
                post=chain_diagnostics(from_records(post_records),post_adv,clip=clip)
                kl_elapsed+=time.perf_counter()-measured
            kl_seconds=max(distributed.all_gather_object(kl_elapsed))
            local_sources=Counter(meta['dataset_id'] for sample in bc_report.get('samples',[]) for meta in sample)
            shards=distributed.all_gather_object(dict(batch=bc_report['batch_size'],loss=bc_report['loss'],sources=dict(local_sources)))
            sources=Counter()
            for shard in shards:sources.update(shard['sources'])
            covered.update(indices)
            report=dict(objective='full_chain_clipped_ppo',ppo_clip_applied=True,epoch=epoch,
                optimizer_step=len(reports)+1,global_chain_count=len(indices),sampled_steps=4,
                denoising_steps=list(policy.trainable_steps),internal_sample_visits=len(indices)*4,
                full_objective_transitions=len(indices)*4,ppo_loss=float(distributed.sum_tensor(backward['loss'])),
                streaming_recompute_max_error=max(distributed.all_gather_object(float(backward['recompute_max_error']))),
                clip_fraction=pre['clip_fraction'],objective_clip_fraction=pre['active_clip_fraction'],
                pre_step_chain=pre,post_step_chain=post,
                pre_step_mean_joint_kl=pre['chain_joint_kl']['mean'],
                pre_step_mean_coordinate_kl=float(statistics['mean_coordinate_kl'].mean()),
                per_denoising_step=[dict(step=s+15,joint_kl=distribution(post_records[:,s,0]),
                    mean_coordinate_kl=distribution(post_records[:,s,1])) for s in range(4)],
                gradient_contributions=contributions,total_gradient_norm=total_norm,gradient_clip_factor=clip_factor,
                parameter_update=parameter_report,output_change=output_change,trust_region=transaction,
                candidate_checked_chains=len(guard_indices) if trust_region is not None else len(indices),
                candidate_check_scope='all_previously_used_plus_current_chains' if trust_region is not None else 'current_minibatch',
                bc_global_samples=sum(s['batch'] for s in shards),bc_samples_by_source=dict(sources),
                bc_loss=sum(s['batch']*s['loss'] for s in shards)/global_bc,
                learning_rate=optimizer.param_groups[0]['lr'],ppo_ratio_seconds=ratio_seconds,
                ppo_backward_seconds=ppo_seconds,ppo_seconds=ratio_seconds+ppo_seconds,
                bc_seconds=bc_seconds,post_step_kl_seconds=kl_seconds,optimizer_transaction_seconds=transaction_seconds,
                wall_seconds=max(distributed.all_gather_object(time.perf_counter()-started)))
            reports.append(report)
            if step_observer is not None:step_observer(report)
            # 轻量minibatch检查只覆盖当前完整链；接受与否还要整轮所有链KL审查。
            if soft_kl is not None and post['chain_joint_kl']['mean']>=soft_kl:stop_reason='post_step_chain_joint_kl'
            elif post['active_clip_fraction']>=max_active_clip:stop_reason='post_step_active_clip_saturation'
            elif post['importance_weight_ess_fraction']<min_ess_fraction:stop_reason='post_step_importance_weight_ess_collapse'
            if stop_reason:break
        if stop_reason:break
    return dict(objective='full_chain_clipped_ppo',ppo_clip_applied=True,optimizer_steps=len(reports),
        attempted_optimizer_steps=sum((r['trust_region'] or {}).get('attempt_count',1) for r in reports),
        rejected_step_candidates=sum((r['trust_region'] or {}).get('backtracks',0) for r in reports),
        executed_optimizer_steps=sum((r['trust_region'] or {}).get('attempt_count',1) for r in reports),
        minibatch_accepted_steps=len(reports),executed_step_scope='actual_Adam_calls_including_rejected_candidates',
        accepted_optimizer_steps=None,acceptance_scope='pending_full_rollout_joint_KL',
        planned_optimizer_steps=planned_steps,skipped_optimizer_steps=planned_steps-len(reports),
        steps=reports,epoch_orders=orders,early_stop_reason=stop_reason,discarded_pending_minibatch=discarded,
        gradient_snr=None,_ppo_gradient_sketch=sketch,epoch_order_rng_contract='fixed_four_steps_shuffled_complete_chains',
        eligible_chains=len(selected),applied_unique_chains=len(covered),applied_chain_fraction=len(covered)/len(selected),
        internal_sample_visits=sum(r['internal_sample_visits'] for r in reports),
        actual_global_bc_samples=sum(r['bc_global_samples'] for r in reports),planned_global_bc_samples=bc_total,
        bc_min_global_batch=bc_min_global_batch,
        bc_samples_by_source={source:sum(r['bc_samples_by_source'].get(source,0) for r in reports)
            for source in sorted({source for r in reports for source in r['bc_samples_by_source']})},
        planned_internal_visits=len(selected)*4*ppo_epochs,objective_reduction='joint_sum',
        proxy_is_joint_probability=True,old_statistics_fixed=True,final_trainable4_kl_pending=True)


@torch.no_grad()
def final_kl(policy,rows,cache,manifest,*,distributed,microbatch=256,clip=.10,objective_reduction='joint_sum'):
    """所有有效链的真实联合KL；四步各自诊断，clip/ESS一律按完整链统计。"""
    from .chain_ppo import evaluate_chain_records,from_records
    from .gaussian_probability import chain_diagnostics,distribution
    if objective_reduction!='joint_sum':raise ValueError('Final PPO diagnostics require complete-chain joint ratio')
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
    chain=chain_diagnostics(from_records(full),adv,clip=clip)
    frozen_identity=policy.assert_frozen_identity()
    identities=distributed.all_gather_object(frozen_identity)
    if any(v!=identities[0] for v in identities):raise RuntimeError('Frozen prior identity differs across ranks')
    kl_seconds=max(distributed.all_gather_object(time.perf_counter()-start))
    reference_change=reference_change_diagnostics(policy,local,cache,distributed)
    return dict(**chain,mean_joint_kl=float(joint.mean()),mean_coordinate_kl=float(coordinate.mean()),
        joint_kl=distribution(joint),coordinate_kl=distribution(coordinate),
        maximum_step_mean_coordinate_kl=float(coordinate.mean(0).max()),
        per_denoising_step=[dict(step=s+15,joint_kl=distribution(joint[:,s]),
            mean_coordinate_kl=distribution(coordinate[:,s]),joint_log_ratio=distribution(logratio[:,s]),
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
    for key,field in fields.items():
        bound=limits.get(key)
        if bound is not None and (not math.isfinite(report[field]) or report[field]>bound):
            failures[key]=dict(value=report[field],limit=bound)
    p95=limits.get('p95_coordinate')
    if p95 is not None and (not math.isfinite(report['coordinate_kl']['p95']) or report['coordinate_kl']['p95']>p95):
        failures['p95_coordinate']=dict(value=report['coordinate_kl']['p95'],limit=p95)
    return dict(passed=not failures,failures=failures,units='explicit_joint_or_coordinate_never_reuse_one_threshold')


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
