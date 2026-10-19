"""当前四个可训练高斯步骤的完整KL审查和独立非对称Critic训练。

Actor只由full_batch_actor执行一次真实全局joint-score＋BC更新；本模块覆盖所有
有效链的15～18步，保留坐标及joint-chain KL、裁剪诊断、最终参考变化与冻结
身份。Critic使用实际全局分母和可训练历史GRU，不跨optimizer复用可训练特征。
旧随机K8/Full20或坐标PPO优化入口已删除，不能通过参数切换回退役算法。
"""
from __future__ import annotations

import math
import time
import torch
from .execution_checks import policy_phase
from .gaussian_probability import gaussian_statistics, ppo_surrogate
from .tensor_cache import ConditionGraphCache
from .optimization_support import _owned, _parameters, _old_kernel, balanced_epoch_order, _local_phase


def _eligible(manifest,actor=True):
    return [i for i,m in enumerate(manifest) if m['valid'] and (m['has_free'] or not actor)]


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
