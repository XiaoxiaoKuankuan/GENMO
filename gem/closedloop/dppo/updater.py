"""四步Clipped DPPO＋Stage1 BC、完整KL与非对称Critic更新。

每epoch打乱真实有效行为链，按全局minibatch分片，15～18四个随机步骤全部参与。
以采样时真实旧均值/方差计算自由坐标平均log-ratio及PPO裁剪；局部loss分母为
全局实际链数×4，PPO/BC梯度一次SUM后统一裁剪，没有额外world_size平均。
旧行为统计及GAE整轮固定；Actor条件图每次optimizer更新后重建。BC总量按实际
计划更新次数分配；本步回溯耗尽时只恢复本步，并明确记录已经接受的BC数量。

每次optimizer候选都对原行为rollout的全部链、四步计算无梯度KL；失败只缩小
当前步长，梯度只计算一次，先前通过的更新和独立Critic不回滚。最近完整KL绑定
模型/rollout身份后复用，参考生成诊断只在最终模型计算。Critic只缓存冻结音乐/
前缀及历史输入，可训练历史GRU
每个optimizer step重新建图。物理时钟、奖励和原冻结去噪身份保持原合同。
"""
from __future__ import annotations

import math
import time
import copy
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
                    generator,ppo_epochs=1,minibatch_chains=4096,microbatch=128,sampled_steps=4,
                    clip=.01,gamma_denoising=.99,kl_reduction='free_coordinate_mean',
                    objective_reduction='free_coordinate_mean',grad_clip_norm=1.,reserve_attempt=None,
                    gradient_observer=None,step_observer=None,bc_samples_per_rollout=1024,snr=False,
                    step_strategy='intermediate4',detailed_diagnostics=False,
                    backtracking,kl_limits,kl_microbatch=128,minimum_lr=5e-8):
    if step_strategy!='intermediate4' or sampled_steps!=4 or tuple(policy.trainable_steps)!=(15,16,17,18):
        raise ValueError('PPO must optimize exactly indices 15 to 18')
    if policy.steps!=20 or ppo_epochs<1 or minibatch_chains<1 or microbatch<1:
        raise ValueError('Invalid PPO iteration scale')
    if kl_reduction not in ('joint_sum','free_coordinate_mean'):raise ValueError('Unknown KL control units')
    from .actor_backtracking import uniform_actor_optimizer,backtracked_step,StepSnapshot
    from .parallel_support import capture_local_rng,restore_local_rng
    from .kl_reuse import seal_accepted_kl
    uniform_actor_optimizer(policy.actor,optimizer)
    policy._accepted_kl_receipt=None
    selected=_eligible(manifest);reports=[];orders=[];stop_reason=None;covered=set()
    pending_metrics=[];pending_bc=[]
    if not selected:raise ValueError('No free valid behavior chain remains for PPO')
    planned_steps=ppo_epochs*math.ceil(len(selected)/minibatch_chains)
    bc_batches=(None if bc_samples_per_rollout is None else [
        bc_samples_per_rollout//planned_steps+(i<bc_samples_per_rollout%planned_steps) for i in range(planned_steps)])
    actor=policy.actor;device=next(actor.parameters()).device
    workspace=StepSnapshot()
    attempts=0;rejected_step=None;last_kl=None;computed_bc=0;candidate_kl_seconds=0.
    def evaluate_kl():
        return final_kl(policy,rows,cache,manifest,distributed=distributed,microbatch=kl_microbatch,
                        clip=clip,objective_reduction=objective_reduction,include_reference=False)
    sketch={} if snr else None
    for epoch in range(ppo_epochs):
        order=balanced_epoch_order(selected,manifest,generator) if distributed.rank==0 else None
        order=distributed.broadcast_object(order)
        if detailed_diagnostics:orders.append(order)
        for begin in range(0,len(order),minibatch_chains):
            before_control=copy.deepcopy(controller.state_dict())
            before_anchor=copy.deepcopy(anchor.state_dict())
            before_rng=capture_local_rng(dict(actor=generator))
            indices=order[begin:begin+minibatch_chains]
            owned=_owned(indices,manifest,distributed)
            pairs=[(i,s) for _,i in owned for s in policy.trainable_steps]
            optimizer.zero_grad(set_to_none=True);actor.eval()
            summary=torch.zeros(7,device=device,dtype=torch.float64)
            step_metrics=torch.zeros(20,6,device=device,dtype=torch.float64)
            start=time.perf_counter()
            with _local_phase(distributed,'stage10_actor_local_forward_backward'),policy_phase(policy):
                conditions=ConditionGraphCache(policy,cache,include_frozen=False);conditions.prime([rows[i] for _,i in owned])
                for offset in range(0,len(pairs),microbatch):
                    chunk=pairs[offset:offset+microbatch];samples=[rows[i] for i,_ in chunk];steps=[s for _,s in chunk]
                    params,mask=_parameters(policy,samples,steps,device,cache,conditions)
                    old_mean,old_std=_old_kernel(samples,steps,device,cache)
                    observed=cache.get('chain',samples,[s+1 for s in steps])
                    stats=gaussian_statistics(observed,old_mean,old_std,params['mean'],params['std'],mask)
                    advantage=cache.get('advantages',samples)*cache.denoising_discounts(steps,20,gamma_denoising)
                    proxy=ppo_surrogate(stats,advantage,clip=clip,reduction=objective_reduction)
                    loss=-proxy['objective'].sum()/(len(indices)*4)
                    loss.backward()
                    with torch.no_grad():
                        summary+=torch.stack((loss.detach(),proxy['ratio'].sum(),proxy['clipped'].double().sum(),
                            proxy['objective_clipped'].double().sum(),stats['joint_kl'].sum(),stats['mean_coordinate_kl'].sum(),
                            proxy['ratio'].new_tensor(len(chunk))))
                        step_metrics.index_add_(0,cache.step_indices(steps),torch.stack((stats['joint_kl'],
                            stats['mean_coordinate_kl'],proxy['ratio'].detach(),proxy['clipped'].double(),
                            proxy['objective_clipped'].double(),torch.ones_like(stats['joint_kl'])),1))
                    del loss,params,stats,proxy
                conditions.backward()
            combined=distributed.sum_tensor(torch.cat((summary,step_metrics.flatten())))
            summary,step_metrics=combined[:7],combined[7:].reshape(20,6)
            values=summary.cpu().tolist();visits=values[-1]
            ppo=take_gradients(actor)
            with _local_phase(distributed,'stage10_bc_local_forward_backward'):
                from .dense_policy import dense_precision_scope
                global_bc=None if bc_batches is None else bc_batches[len(reports)]
                if global_bc is not None:
                    anchor.batch_size=global_bc//distributed.world_size+(distributed.rank<global_bc%distributed.world_size)
                bc_weight=1/distributed.world_size if global_bc is None else (anchor.batch_size/global_bc if global_bc else 0.)
                with dense_precision_scope(policy.precision):bc_report=anchor.backward(actor,weight=bc_weight)
                bc=take_gradients(actor)
            computed_bc+=global_bc if global_bc is not None else bc_report["batch_size"]*distributed.world_size
            contributions=merge_global_gradients(actor,ppo,bc,controller,distributed=distributed,half_split=snr,sketch=sketch)
            if gradient_observer is not None:gradient_observer(actor,len(reports))
            with _local_phase(distributed,'stage10_actor_combined_clip'):
                total_norm=float(torch.nn.utils.clip_grad_norm_([p for p in actor.parameters() if p.requires_grad],
                    grad_clip_norm,error_if_nonfinite=True))
            with _local_phase(distributed,'stage10_actor_optimizer_step'):
                transaction=backtracked_step(actor,optimizer,controller,evaluate=evaluate_kl,
                    limits=kl_limits,distributed=distributed,generators=dict(actor=generator),
                    max_attempts=backtracking['max_attempts'],factor=backtracking['factor'],
                    minimum_lr=minimum_lr,workspace=workspace,reserve_attempt=reserve_attempt)
            attempts+=transaction['attempt_count']
            candidate_kl_seconds+=sum(t['kl']['kl_only_wall_seconds'] for t in transaction['attempts'])
            if not transaction['accepted']:
                def restore_sampling():
                    controller.load_state_dict(before_control)
                    anchor.load_state_dict(before_anchor)
                    restore_local_rng(before_rng,dict(actor=generator))
                    optimizer.zero_grad(set_to_none=True)
                from .parallel_support import local_call
                local_call(distributed,restore_sampling)
                stop_reason='optimizer_step_backtracking_exhausted'
                rejected_step=dict(epoch=epoch,global_chain_count=len(indices),backtracking=transaction,
                    retained_optimizer_steps=len(reports),wall_seconds=time.perf_counter()-start)
                # load_state_dict改变Tensor版本，恢复后的模型重新全量检查再封存，绝不盲复用。
                last_kl=evaluate_kl()
                break
            last_kl=transaction['attempts'][-1]['kl']
            # 后续minibatch不会复用旧条件图，old statistics则保持同一份rollout。
            local_sources=Counter(meta['dataset_id'] for sample in bc_report.get('samples',[]) for meta in sample)
            pending_bc.append(dict(batch=bc_report['batch_size'],loss=bc_report['loss'],sources=dict(local_sources)))
            pending_metrics.append(step_metrics)
            covered.update(indices)
            reports.append(dict(objective='clipped_dppo',ppo_clip_applied=True,
                epoch=epoch,optimizer_step=len(reports)+1,global_chain_count=len(indices),
                sampled_steps=sampled_steps,denoising_steps=list(policy.trainable_steps),
                internal_sample_visits=int(visits),full_objective_transitions=len(indices)*4,
                ppo_loss=values[0],mean_proxy_ratio=values[1]/visits,clip_fraction=values[2]/visits,
                objective_clip_fraction=values[3]/visits,pre_step_mean_joint_kl=values[4]/visits,
                pre_step_mean_coordinate_kl=values[5]/visits,
                gradient_contributions=contributions,total_gradient_norm=total_norm,
                gradient_clip_factor=min(1.,grad_clip_norm/(total_norm+1e-6)),
                backtracking=transaction,
                learning_rate=optimizer.param_groups[0]['lr'],wall_seconds=time.perf_counter()-start))
            if step_observer is not None:step_observer(reports[-1])
        if stop_reason:break
    if reports:
        shards=distributed.all_gather_object(pending_bc)
        if any(len(shard)!=len(reports) for shard in shards):raise RuntimeError('Rank optimizer/log counts differ')
        for index,(report,metrics) in enumerate(zip(reports,torch.stack(pending_metrics).cpu().tolist())):
            source_counts=Counter();bc_shards=[s[index] for s in shards]
            for shard in bc_shards:source_counts.update(shard['sources'])
            total=sum(s['batch'] for s in bc_shards)
            report.update(bc_global_samples=total,bc_samples_by_source=dict(source_counts),
                bc_loss=sum(s['batch']*s['loss'] for s in bc_shards)/max(total,1),
                per_denoising_step=[dict(step=s,visits=int(n),mean_joint_kl=joint/n if n else None,
                    mean_coordinate_kl=coordinate/n if n else None,mean_proxy_ratio=ratio/n if n else None,
                    clip_fraction=clipped/n if n else None,objective_clip_fraction=active/n if n else None)
                    for s,(joint,coordinate,ratio,clipped,active,n) in enumerate(metrics)])
    snr_report=None
    if snr:
        halves=[r['gradient_contributions'].get('half_split') for r in reports]
        halves=[h for h in halves if h is not None]
        first2=sum(h['first_half_square'] for h in halves);second2=sum(h['second_half_square'] for h in halves)
        dot=sum(h['dot'] for h in halves);cosines=[h['cosine'] for h in halves if h['cosine'] is not None]
        snr_report=dict(half_split_cosine_per_step=[h['cosine'] for h in halves],
            half_split_cosine_mean=sum(cosines)/len(cosines) if cosines else None,
            half_split_cosine_pooled=None if first2<=0 or second2<=0 else dot/math.sqrt(first2*second2),
            sketch_elements=sum(v.numel() for v in sketch.values()),
            scope='independent_half_rank_PPO_gradients_per_optimizer_step;sketch_is_iteration_sum_for_consecutive_cosine')
    if last_kl is not None:
        policy._accepted_kl_receipt=seal_accepted_kl(policy,cache,manifest,last_kl,
            clip=clip,reduction=objective_reduction)
    return dict(objective='clipped_dppo',ppo_clip_applied=True,
        optimizer_steps=len(reports),attempted_optimizer_steps=attempts,steps=reports,epoch_orders=orders,early_stop_reason=stop_reason,
        complete_accepted_kl_available=last_kl is not None,rejected_step=rejected_step,
        retained_previous_steps=True,step_rejected=False,computed_global_bc_samples=computed_bc,
        candidate_kl_seconds=candidate_kl_seconds,kl_timing_is_part_of_actor_wall=True,
        gradient_snr=snr_report,_ppo_gradient_sketch=sketch,
        epoch_order_rng_contract='fixed_four_steps_shuffled_chains',
        eligible_chains=len(selected),applied_unique_chains=len(covered),applied_chain_fraction=len(covered)/len(selected),
        planned_optimizer_steps=ppo_epochs*math.ceil(len(selected)/minibatch_chains),
        internal_sample_visits=sum(r['internal_sample_visits'] for r in reports),
        actual_global_bc_samples=sum(r['bc_global_samples'] for r in reports),
        bc_samples_by_source={source:sum(r['bc_samples_by_source'].get(source,0) for r in reports)
            for source in sorted({source for r in reports for source in r['bc_samples_by_source']})},
        planned_internal_visits=len(selected)*sampled_steps*ppo_epochs,
        objective_reduction=objective_reduction,proxy_is_joint_probability=objective_reduction=='joint_sum',
        old_statistics_fixed=True,final_trainable4_kl_pending=True)


@torch.no_grad()
def final_kl(policy,rows,cache,manifest,*,distributed,microbatch=256,clip=.05,
             objective_reduction='free_coordinate_mean',include_reference=True):
    selected=_eligible(manifest);owned=_owned(selected,manifest,distributed)
    local=[rows[i] for _,i in owned];device=next(policy.actor.parameters()).device
    columns=9
    records=torch.zeros(len(local),4,columns,device=device,dtype=torch.float64)
    start=time.perf_counter()
    with _local_phase(distributed,'stage2_final_trainable4_kl'),policy_phase(policy):
        conditions=ConditionGraphCache(policy,cache,include_frozen=False);conditions.prime(local)
        pairs=[(i,s) for i in range(len(local)) for s in policy.trainable_steps]
        for offset in range(0,len(pairs),microbatch):
            part=pairs[offset:offset+microbatch];samples=[local[i] for i,_ in part];steps=[s for _,s in part]
            params,mask=_parameters(policy,samples,steps,device,cache,conditions)
            old_mean,old_std=_old_kernel(samples,steps,device,cache)
            stats=gaussian_statistics(cache.get('chain',samples,[s+1 for s in steps]),old_mean,old_std,
                params['mean'],params['std'],mask)
            proxy=ppo_surrogate(stats,cache.get('advantages',samples),clip=clip,reduction=objective_reduction)
            shift=((params['mean'].double()-old_mean).square().masked_fill(~mask,0.).sum((-2,-1))/stats['free_count']).sqrt()
            physical_delta=(params['mean'].double()-old_mean)*policy.actor.endecoder.std.double()
            physical=[]
            for section in (slice(0,3),slice(3,9),slice(9,30)):
                part_mask=mask[:,:,section]
                physical.append((physical_delta[:,:,section].square().masked_fill(~part_mask,0.).sum((-2,-1))/
                    part_mask.sum((-2,-1)).clamp_min(1)).sqrt())
            i=torch.tensor([i for i,_ in part],device=device);s=cache.step_indices(steps)
            records[i,s-15]=torch.stack((stats['joint_kl'],stats['mean_coordinate_kl'],proxy['ratio'],
                stats['joint_log_ratio'],shift,*physical,proxy['objective_clipped'].double()),1)
        conditions.validate_inputs()
    full=distributed.gather_rows(records.reshape(len(local),4*columns),[p for p,_ in owned],len(selected)).reshape(-1,4,columns)
    joint,coordinate,ratio,logratio,shift=[full[:,:,i] for i in range(5)]
    frozen_identity=policy.assert_frozen_identity()
    identities=distributed.all_gather_object(frozen_identity)
    if any(v!=identities[0] for v in identities):raise RuntimeError('Frozen prior identity differs across ranks')
    def quantile(x):return dict(mean=float(x.mean()),p95=float(torch.quantile(x.flatten(),.95)),max=float(x.max()))
    step_reports=[]
    for s in range(4):
        step_reports.append(dict(step=s+15,joint_kl=quantile(joint[:,s]),mean_coordinate_kl=quantile(coordinate[:,s]),
            proxy_ratio=quantile(ratio[:,s]),clip_fraction=float(((ratio[:,s]<1-clip)|(ratio[:,s]>1+clip)).double().mean()),
            objective_clip_fraction=float(full[:,s,8].mean()),
            mean_normalized_mean_change_rms=float(shift[:,s].mean())))
    kl_seconds=max(distributed.all_gather_object(time.perf_counter()-start))
    reference_change=reference_change_diagnostics(policy,local,cache,distributed) if include_reference else None
    return dict(mean_joint_kl=float(joint.mean()),mean_coordinate_kl=float(coordinate.mean()),
        clip_fraction=float(((ratio<1-clip)|(ratio>1+clip)).double().mean()),
        joint_kl=quantile(joint),coordinate_kl=quantile(coordinate),chain_joint_kl=quantile(joint.sum(1)),
        maximum_step_mean_coordinate_kl=float(coordinate.mean(0).max()),
        per_denoising_step=step_reports,covered_chains=len(selected),covered_internal_transitions=len(selected)*4,
        gaussian_kl_steps=list(policy.trainable_steps),statistical_denominator="all_valid_chains_times_four_trainable_steps",
        frozen_identity=frozen_identity,
        terminal_two_step_joint_kl_fraction=float(joint[:,-2:].sum()/joint.sum()) if joint.sum()>0 else None,
        joint_log_ratio_range=[float(logratio.min()),float(logratio.max())],
        effective_mean_change=dict(mean_rms=float(shift.mean()),max_rms=float(shift.max()),
            unchanged_internal_fraction=float((shift==0).double().mean()),
            scope='final_policy_means_on_old_sampling_path_not_resampled_physical_trajectory'),
        objective_clip_fraction=float(full[:,:,8].mean()),
        final_reference_mean_change=dict(root_feature_rms_m=quantile(full[:,-1,5]),
            rotation6d_rms_unitless=quantile(full[:,-1,6]),joint_rms_rad=quantile(full[:,-1,7]),
            root_semantics='heading_local_xy_increments_and_root_height_not_world_xyz',
            scope='denormalized_index18_gaussian_mean_not_final_deterministic_reference'),
        kl_scope='four_trainable_old_path_gaussians_not_deterministic_final_action_density',
        p95_scope='four_trainable_steps_per_chain_coordinate_means_only',
        reference_change=reference_change,kl_only_wall_seconds=kl_seconds,
        wall_seconds=max(distributed.all_gather_object(time.perf_counter()-start)))


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
