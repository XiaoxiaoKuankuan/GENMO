"""固定行为四步DPPO＋BC、可训练历史Critic和完整四步KL审查。

20步行为仅索引15..18参与Actor优化；每个epoch打乱完整链，四步全部参与，不使用
旧K8抽样权重或把冻结零KL加入分母。旧均值/标准差/联合概率、价值目标与优势整轮
固定。自由有效坐标平均log-ratio是代理目标，原joint概率比及KL分别保存。
当前optimizer minibatch的条件只编码一次，PPO与原Stage1 BC梯度统一SUM、裁剪及
Adam更新。结束后无梯度检查全部有效链的四步KL，冻结部分检查不可变身份。
少量同噪声重生成只诊断最终参考变化，单独计时，不冒充实际物理执行或完整KL覆盖。
失败可回滚模型/优化器/BC/RNG，已发生的物理和资源消耗不能回滚。

Critic只缓存冻结音乐/前缀特征及历史输入，历史GRU、时序层和Value Head每次优化
重新建图；当前/下一批误差与独立MC验证分开。归约使用全局实际有效分母，尾批和
空rank参与同一通信序列。本模块不推进物理，也不保存checkpoint。
"""
from __future__ import annotations

import math
import time
from collections import Counter

import torch

from .denoising_sampling import chain_steps, plan_record
from .execution_checks import policy_phase
from .gradient_merge import merge_global_gradients, take_gradients
from .gaussian_probability import gaussian_statistics, ppo_surrogate
from .tensor_cache import ConditionGraphCache
from .optimization_support import _owned, _parameters, _old_kernel, balanced_epoch_order, _local_phase


def _eligible(manifest,actor=True):
    return [i for i,m in enumerate(manifest) if m['valid'] and (m['has_free'] or not actor)]


def actor_update(policy,optimizer,rows,targets,cache,manifest,controller,*,distributed,anchor,
                    generator,ppo_epochs=2,minibatch_chains=1024,microbatch=128,sampled_steps=4,
                    clip=.05,gamma_denoising=.99,soft_kl=.002,kl_reduction='free_coordinate_mean',
                    objective_reduction='free_coordinate_mean',grad_clip_norm=1.,reserve_attempt=None,
                    gradient_observer=None,step_observer=None,bc_samples_per_rollout=None,snr=False,step_strategy='intermediate4'):
    if step_strategy!='intermediate4' or sampled_steps!=4 or tuple(policy.trainable_steps)!=(15,16,17,18):
        raise ValueError('PPO must optimize exactly indices 15 to 18')
    if policy.steps!=20 or ppo_epochs<1 or minibatch_chains<1 or microbatch<1:
        raise ValueError('Invalid PPO iteration scale')
    if kl_reduction not in ('joint_sum','free_coordinate_mean'):raise ValueError('Unknown KL control units')
    selected=_eligible(manifest);reports=[];orders=[];stop_reason=None
    if not selected:raise ValueError('No free valid behavior chain remains for PPO')
    planned_steps=ppo_epochs*math.ceil(len(selected)/minibatch_chains)
    bc_batches=(None if bc_samples_per_rollout is None else [
        bc_samples_per_rollout//planned_steps+(i<bc_samples_per_rollout%planned_steps) for i in range(planned_steps)])
    actor=policy.actor;device=next(actor.parameters()).device
    sketch={} if snr else None
    for epoch in range(ppo_epochs):
        order=balanced_epoch_order(selected,manifest,generator) if distributed.rank==0 else None
        order=distributed.broadcast_object(order);orders.append(order)
        for begin in range(0,len(order),minibatch_chains):
            indices=order[begin:begin+minibatch_chains]
            owned=_owned(indices,manifest,distributed)
            seed=(int(torch.randint(0,2**63-1,(1,),generator=generator)) if distributed.rank==0 else None)
            seed=distributed.broadcast_object(seed)
            pairs=[(i,s) for position,i in owned for s in chain_steps(seed,indices[position],20,sampled_steps,step_strategy)]
            optimizer.zero_grad(set_to_none=True);actor.eval()
            summary=torch.zeros(7,device=device,dtype=torch.float64)
            step_metrics=torch.zeros(20,6,device=device,dtype=torch.float64)
            start=time.perf_counter()
            with _local_phase(distributed,'stage10_actor_local_forward_backward'),policy_phase(policy):
                conditions=ConditionGraphCache(policy,cache);conditions.prime([rows[i] for _,i in owned])
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
            summary=distributed.sum_tensor(summary);step_metrics=distributed.sum_tensor(step_metrics)
            values=summary.cpu().tolist();visits=values[-1]
            measured=values[4 if kl_reduction=='joint_sum' else 5]/visits
            if soft_kl is not None and measured>=soft_kl:
                optimizer.zero_grad(set_to_none=True);stop_reason='pre_step_soft_kl'
                discarded=dict(epoch=epoch,indices=indices,mean_kl=measured,kl_units=kl_reduction,
                    sample_visits=int(visits),wall_seconds=time.perf_counter()-start,optimizer_step_executed=False)
                break
            ppo=take_gradients(actor)
            with _local_phase(distributed,'stage10_bc_local_forward_backward'):
                from .dense_policy import dense_precision_scope
                global_bc=None if bc_batches is None else bc_batches[len(reports)]
                if global_bc is not None:
                    anchor.batch_size=global_bc//distributed.world_size+(distributed.rank<global_bc%distributed.world_size)
                bc_weight=1/distributed.world_size if global_bc is None else (anchor.batch_size/global_bc if global_bc else 0.)
                with dense_precision_scope(policy.precision):bc_report=anchor.backward(actor,weight=bc_weight)
                bc=take_gradients(actor)
            contributions=merge_global_gradients(actor,ppo,bc,controller,distributed=distributed,half_split=snr,sketch=sketch)
            if gradient_observer is not None:gradient_observer(actor,len(reports))
            with _local_phase(distributed,'stage10_actor_combined_clip'):
                total_norm=float(torch.nn.utils.clip_grad_norm_([p for p in actor.parameters() if p.requires_grad],
                    grad_clip_norm,error_if_nonfinite=True))
            if reserve_attempt is not None:reserve_attempt()
            with _local_phase(distributed,'stage10_actor_optimizer_step'):
                optimizer.step()
            # 后续minibatch不会复用旧条件图，old statistics则保持同一份rollout。
            local_sources=Counter(meta['dataset_id'] for sample in bc_report.get('samples',[]) for meta in sample)
            bc_shards=distributed.all_gather_object(dict(batch=bc_report['batch_size'],loss=bc_report['loss'],sources=dict(local_sources)))
            source_counts=Counter()
            for shard in bc_shards:source_counts.update(shard['sources'])
            per_step=[]
            for s,(joint,coordinate,ratio,clipped,active,n) in enumerate(step_metrics.cpu().tolist()):
                per_step.append(dict(step=s,visits=int(n),mean_joint_kl=joint/n if n else None,
                    mean_coordinate_kl=coordinate/n if n else None,mean_proxy_ratio=ratio/n if n else None,
                    clip_fraction=clipped/n if n else None,objective_clip_fraction=active/n if n else None))
            reports.append(dict(epoch=epoch,optimizer_step=len(reports)+1,global_chain_indices=indices,
                sampled_steps=sampled_steps,denoising_sampling=plan_record(seed,indices,20,sampled_steps,step_strategy),
                internal_sample_visits=int(visits),full_objective_transitions=len(indices)*4,
                ppo_loss=values[0],mean_proxy_ratio=values[1]/visits,clip_fraction=values[2]/visits,
                objective_clip_fraction=values[3]/visits,pre_step_mean_joint_kl=values[4]/visits,
                pre_step_mean_coordinate_kl=values[5]/visits,per_denoising_step=per_step,
                gradient_contributions=contributions,total_gradient_norm=total_norm,
                gradient_clip_factor=min(1.,grad_clip_norm/(total_norm+1e-6)),
                bc_global_samples=sum(x['batch'] for x in bc_shards),
                bc_samples_by_source=dict(source_counts),
                bc_loss=sum(x['batch']*x['loss'] for x in bc_shards)/sum(x['batch'] for x in bc_shards),
                learning_rate=optimizer.param_groups[0]['lr'],wall_seconds=time.perf_counter()-start))
            if step_observer is not None:step_observer(reports[-1])
        if stop_reason:break
    covered={i for report in reports for i in report['global_chain_indices']}
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
    return dict(optimizer_steps=len(reports),steps=reports,epoch_orders=orders,early_stop_reason=stop_reason,
        gradient_snr=snr_report,_ppo_gradient_sketch=sketch,
        epoch_order_rng_contract='fixed_four_steps_shuffled_chains',
        discarded_pending_minibatch=discarded if stop_reason else None,
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
def final_kl(policy,rows,cache,manifest,*,distributed,microbatch=256,clip=.05,objective_reduction='free_coordinate_mean'):
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
    reference_change=reference_change_diagnostics(policy,local,cache,distributed)
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
                norms.append(float(torch.nn.utils.clip_grad_norm_([p for p in critic.parameters() if p.requires_grad],grad_clip,error_if_nonfinite=True)))
                optimizer.step()
            losses.append(float(distributed.sum_tensor(loss.detach())))
    final=metrics()
    return dict(before=initial,after=final,optimizer_steps=len(losses),epochs=epochs,losses=losses,gradient_norms=norms,
        frozen_encoder_passes=int(cached_features and bool(local_rows)),feature_bytes=sum(v.numel()*v.element_size() for v in encoded.values()),
        critic_feature_contract='frozen_music_prefix_cached_trainable_history_recomputed' if cached_features else 'recompute_trainable_encoders_each_step',
        global_sample_visits=epochs*len(selected),metric_scope='fixed_rollout_bootstrap_targets_next_batch_before_update_is_separate')
