"""Stage10 v3：固定行为数据、K8代理PPO、预训练Critic缓存与最终Full20验收。

rollout旧均值/方差/联合概率、价值、returns和全局标准化advantage整轮固定。
每个epoch重新对完整链打乱并无放回抽样去噪步骤；N/K逆概率权重乘回完整20步
目标，去噪折扣保留。后续minibatch使用已经更新的参数，PPO clipping有实际意义。
主代理目标采用自由有效坐标平均log-ratio，真实joint统计同时记录且不被改写。

条件在当前optimizer minibatch的唯一链上只编码一次，临时条件梯度最后回传。
BC继续使用原Stage1损失；局部PPO和BC统一配对SUM、统一裁剪、Adam真实更新。
当前前向的真实KL只决定待更新步骤是否停止；循环结束必须用最终参数重新计算
所有有效链全部20步的KL，失败由调用方恢复整轮模型/优化器/BC/RNG/控制状态。
物理世界与资源消耗不能回滚。本模块不保存checkpoint、不推进物理或改变奖励。

Critic冻结特征仅编码一次并跨epoch缓存，时序投影/Transformer/Value Head每步
重算。当前批拟合前后指标分开，新批更新前指标不能由同批低MSE代替。所有归约
使用实际全局有效分母，尾批和某rank没有局部样本仍参与同一通信序列。
"""
from __future__ import annotations

import math
import time

import torch

from .denoising_sampling import chain_steps, plan_record
from .execution_checks import policy_phase
from .gradient_merge_v3 import merge_global_gradients, take_gradients
from .probability_v3 import gaussian_statistics, ppo_surrogate
from .tensor_cache import ConditionGraphCache
from .updater_v2 import _owned, _parameters, _old_kernel, balanced_epoch_order, _local_phase


def _eligible(manifest,actor=True):
    return [i for i,m in enumerate(manifest) if m['valid'] and (m['has_free'] or not actor)]


def actor_update_v3(policy,optimizer,rows,targets,cache,manifest,controller,*,distributed,anchor,
                    generator,ppo_epochs=2,minibatch_chains=1024,microbatch=128,sampled_steps=8,
                    clip=.05,gamma_denoising=.99,soft_kl=.002,kl_reduction='free_coordinate_mean',
                    objective_reduction='free_coordinate_mean',grad_clip_norm=1.,reserve_attempt=None,
                    gradient_observer=None,step_observer=None):
    if sampled_steps not in (8,20) or policy.steps!=20 or ppo_epochs<1 or minibatch_chains<1 or microbatch<1:
        raise ValueError('v3 requires Full20 behavior and explicit Uniform8/Full20 optimization')
    if kl_reduction not in ('joint_sum','free_coordinate_mean'):raise ValueError('Unknown KL control units')
    selected=_eligible(manifest);reports=[];orders=[];stop_reason=None
    if not selected:raise ValueError('No free valid behavior chain remains for PPO')
    actor=policy.actor;device=next(actor.parameters()).device
    for epoch in range(ppo_epochs):
        order=balanced_epoch_order(selected,manifest,generator) if distributed.rank==0 else None
        order=distributed.broadcast_object(order);orders.append(order)
        for begin in range(0,len(order),minibatch_chains):
            indices=order[begin:begin+minibatch_chains]
            owned=_owned(indices,manifest,distributed)
            seed=(int(torch.randint(0,2**63-1,(1,),generator=generator)) if distributed.rank==0 and sampled_steps<20 else None)
            seed=distributed.broadcast_object(seed)
            pairs=[(i,s) for position,i in owned for s in chain_steps(seed,indices[position],20,sampled_steps)]
            optimizer.zero_grad(set_to_none=True);actor.eval()
            summary=torch.zeros(7,device=device,dtype=torch.float64)
            step_metrics=torch.zeros(20,6,device=device,dtype=torch.float64)
            start=time.perf_counter()
            with _local_phase(distributed,'v3_actor_local_forward_backward'),policy_phase(policy):
                conditions=ConditionGraphCache(policy,cache);conditions.prime([rows[i] for _,i in owned])
                for offset in range(0,len(pairs),microbatch):
                    chunk=pairs[offset:offset+microbatch];samples=[rows[i] for i,_ in chunk];steps=[s for _,s in chunk]
                    params,mask=_parameters(policy,samples,steps,device,cache,conditions)
                    old_mean,old_std=_old_kernel(samples,steps,device,cache)
                    observed=cache.get('chain',samples,[s+1 for s in steps])
                    stats=gaussian_statistics(observed,old_mean,old_std,params['mean'],params['std'],mask)
                    advantage=cache.get('advantages',samples)*cache.denoising_discounts(steps,20,gamma_denoising)
                    proxy=ppo_surrogate(stats,advantage,clip=clip,reduction=objective_reduction)
                    importance=20/sampled_steps
                    loss=-(proxy['objective']*importance).sum()/(len(indices)*20)
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
            with _local_phase(distributed,'v3_bc_local_forward_backward'):
                from .policy_v3 import dense_precision_scope
                with dense_precision_scope(policy.precision):bc_report=anchor.backward(actor,weight=1/distributed.world_size)
                bc=take_gradients(actor)
            contributions=merge_global_gradients(actor,ppo,bc,controller,distributed=distributed)
            if gradient_observer is not None:gradient_observer(actor,len(reports))
            with _local_phase(distributed,'v3_actor_combined_clip'):
                total_norm=float(torch.nn.utils.clip_grad_norm_([p for p in actor.parameters() if p.requires_grad],
                    grad_clip_norm,error_if_nonfinite=True))
            if reserve_attempt is not None:reserve_attempt()
            with _local_phase(distributed,'v3_actor_optimizer_step'):
                optimizer.step()
            # 后续minibatch不会复用旧条件图，old statistics则保持同一份rollout。
            bc_shards=distributed.all_gather_object(dict(batch=bc_report['batch_size'],loss=bc_report['loss']))
            per_step=[]
            for s,(joint,coordinate,ratio,clipped,active,n) in enumerate(step_metrics.cpu().tolist()):
                per_step.append(dict(step=s,visits=int(n),mean_joint_kl=joint/n if n else None,
                    mean_coordinate_kl=coordinate/n if n else None,mean_proxy_ratio=ratio/n if n else None,
                    clip_fraction=clipped/n if n else None,objective_clip_fraction=active/n if n else None))
            reports.append(dict(epoch=epoch,optimizer_step=len(reports)+1,global_chain_indices=indices,
                sampled_steps=sampled_steps,denoising_sampling=plan_record(seed,indices,20,sampled_steps),
                internal_sample_visits=int(visits),full_objective_transitions=len(indices)*20,
                ppo_loss=values[0],mean_proxy_ratio=values[1]/visits,clip_fraction=values[2]/visits,
                objective_clip_fraction=values[3]/visits,pre_step_mean_joint_kl=values[4]/visits,
                pre_step_mean_coordinate_kl=values[5]/visits,per_denoising_step=per_step,
                gradient_contributions=contributions,total_gradient_norm=total_norm,
                gradient_clip_factor=min(1.,grad_clip_norm/(total_norm+1e-6)),
                bc_global_samples=sum(x['batch'] for x in bc_shards),
                bc_loss=sum(x['batch']*x['loss'] for x in bc_shards)/sum(x['batch'] for x in bc_shards),
                learning_rate=optimizer.param_groups[0]['lr'],wall_seconds=time.perf_counter()-start))
            if step_observer is not None:step_observer(reports[-1])
        if stop_reason:break
    covered={i for report in reports for i in report['global_chain_indices']}
    return dict(optimizer_steps=len(reports),steps=reports,epoch_orders=orders,early_stop_reason=stop_reason,
        discarded_pending_minibatch=discarded if stop_reason else None,
        eligible_chains=len(selected),applied_unique_chains=len(covered),applied_chain_fraction=len(covered)/len(selected),
        planned_optimizer_steps=ppo_epochs*math.ceil(len(selected)/minibatch_chains),
        internal_sample_visits=sum(r['internal_sample_visits'] for r in reports),
        actual_global_bc_samples=sum(r['bc_global_samples'] for r in reports),
        planned_internal_visits=len(selected)*sampled_steps*ppo_epochs,
        objective_reduction=objective_reduction,proxy_is_joint_probability=objective_reduction=='joint_sum',
        old_statistics_fixed=True,final_full20_kl_pending=True)


@torch.no_grad()
def final_kl_v3(policy,rows,cache,manifest,*,distributed,microbatch=256,clip=.05,objective_reduction='free_coordinate_mean'):
    selected=_eligible(manifest);owned=_owned(selected,manifest,distributed)
    local=[rows[i] for _,i in owned];device=next(policy.actor.parameters()).device
    columns=8
    records=torch.zeros(len(local),20,columns,device=device,dtype=torch.float64)
    start=time.perf_counter()
    with _local_phase(distributed,'v3_final_full20_kl'),policy_phase(policy):
        conditions=ConditionGraphCache(policy,cache);conditions.prime(local)
        pairs=[(i,s) for i in range(len(local)) for s in range(20)]
        for offset in range(0,len(pairs),microbatch):
            part=pairs[offset:offset+microbatch];samples=[local[i] for i,_ in part];steps=[s for _,s in part]
            params,mask=_parameters(policy,samples,steps,device,cache,conditions)
            old_mean,old_std=_old_kernel(samples,steps,device,cache)
            stats=gaussian_statistics(cache.get('chain',samples,[s+1 for s in steps]),old_mean,old_std,
                params['mean'],params['std'],mask)
            proxy=ppo_surrogate(stats,torch.ones(len(part),device=device),clip=clip,reduction=objective_reduction)
            shift=((params['mean'].double()-old_mean).square().masked_fill(~mask,0.).sum((-2,-1))/stats['free_count']).sqrt()
            physical_delta=(params['mean'].double()-old_mean)*policy.actor.endecoder.std.double()
            physical=[]
            for section in (slice(0,3),slice(3,9),slice(9,30)):
                part_mask=mask[:,:,section]
                physical.append((physical_delta[:,:,section].square().masked_fill(~part_mask,0.).sum((-2,-1))/
                    part_mask.sum((-2,-1)).clamp_min(1)).sqrt())
            i=torch.tensor([i for i,_ in part],device=device);s=cache.step_indices(steps)
            records[i,s]=torch.stack((stats['joint_kl'],stats['mean_coordinate_kl'],proxy['ratio'],
                stats['joint_log_ratio'],shift,*physical),1)
        conditions.validate_inputs()
    full=distributed.gather_rows(records.reshape(len(local),20*columns),[p for p,_ in owned],len(selected)).reshape(-1,20,columns)
    joint,coordinate,ratio,logratio,shift=[full[:,:,i] for i in range(5)]
    def quantile(x):return dict(mean=float(x.mean()),p95=float(torch.quantile(x.flatten(),.95)),max=float(x.max()))
    step_reports=[]
    for s in range(20):
        step_reports.append(dict(step=s,joint_kl=quantile(joint[:,s]),mean_coordinate_kl=quantile(coordinate[:,s]),
            proxy_ratio=quantile(ratio[:,s]),clip_fraction=float(((ratio[:,s]<1-clip)|(ratio[:,s]>1+clip)).double().mean()),
            mean_normalized_mean_change_rms=float(shift[:,s].mean())))
    return dict(mean_joint_kl=float(joint.mean()),mean_coordinate_kl=float(coordinate.mean()),
        joint_kl=quantile(joint),coordinate_kl=quantile(coordinate),chain_joint_kl=quantile(joint.sum(1)),
        maximum_step_mean_coordinate_kl=float(coordinate.mean(0).max()),
        per_denoising_step=step_reports,covered_chains=len(selected),covered_internal_transitions=len(selected)*20,
        terminal_two_step_joint_kl_fraction=float(joint[:,-2:].sum()/joint.sum()) if joint.sum()>0 else None,
        joint_log_ratio_range=[float(logratio.min()),float(logratio.max())],
        effective_mean_change=dict(mean_rms=float(shift.mean()),max_rms=float(shift.max()),
            unchanged_internal_fraction=float((shift==0).double().mean()),
            scope='final_policy_means_on_old_sampling_path_not_resampled_physical_trajectory'),
        final_reference_mean_change=dict(root_xyz_rms_m=quantile(full[:,-1,5]),
            rotation6d_rms_unitless=quantile(full[:,-1,6]),joint_rms_rad=quantile(full[:,-1,7]),
            scope='decoded_final_gaussian_mean_not_actual_robot_execution'),
        kl_scope='old_path_conditional_gaussian_not_final_action_marginal',
        p95_scope='distribution_of_per_internal_free_coordinate_average_over_real_chains',
        wall_seconds=max(distributed.all_gather_object(time.perf_counter()-start)))


def check_kl_v3(report,limits):
    fields=dict(mean_coordinate='mean_coordinate_kl',mean_joint='mean_joint_kl',
        maximum_step_mean_coordinate='maximum_step_mean_coordinate_kl')
    failures={}
    for key,field in fields.items():
        bound=limits.get(key)
        if bound is not None and (not math.isfinite(report[field]) or report[field]>bound):
            failures[key]=dict(value=report[field],limit=bound)
    p95=limits.get('p95_coordinate')
    if p95 is not None and report['coordinate_kl']['p95']>p95:
        failures['p95_coordinate']=dict(value=report['coordinate_kl']['p95'],limit=p95)
    return dict(passed=not failures,failures=failures,units='explicit_joint_or_coordinate_never_reuse_one_threshold')


def critic_update_v3(critic,optimizer,rows,targets,cache,manifest,*,distributed,generator,epochs=4,batch=1024,grad_clip=1.):
    critic.train()
    selected=_eligible(manifest,False);owned=_owned(selected,manifest,distributed);local_indices=[i for _,i in owned]
    local_rows=[rows[i] for i in local_indices]
    cached_features=hasattr(critic,'encode_frozen') and getattr(critic,'freeze_encoders',False)
    with _local_phase(distributed,'v3_critic_frozen_encoding'):
        encoded=critic.encode_frozen(cache.context(local_rows)) if cached_features else cache.context(local_rows)
    lookup={i:j for j,i in enumerate(local_indices)}
    target=torch.as_tensor(targets['returns'],device=cache.device,dtype=torch.float32)
    def forward(indices):
        idx=torch.tensor([lookup[i] for i in indices],device=cache.device)
        features={k:v.index_select(0,idx) for k,v in encoded.items()}
        remaining=cache.get('remaining_music_seconds',[rows[i] for i in indices])
        return critic.forward_encoded(features,remaining) if cached_features else critic(features,remaining)
    @torch.no_grad()
    def metrics():
        with _local_phase(distributed,'v3_critic_value_metrics'):
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
            with _local_phase(distributed,'v3_critic_trainable_head'):
                loss=torch.zeros((),device=cache.device)
                if local:
                    loss=.5*(forward(local)-target[local]).square().sum()/len(indices)
                    loss.backward()
            distributed.sum_gradients(critic)
            with _local_phase(distributed,'v3_critic_clip_and_step'):
                norms.append(float(torch.nn.utils.clip_grad_norm_([p for p in critic.parameters() if p.requires_grad],grad_clip,error_if_nonfinite=True)))
                optimizer.step()
            losses.append(float(distributed.sum_tensor(loss.detach())))
    final=metrics()
    return dict(before=initial,after=final,optimizer_steps=len(losses),epochs=epochs,losses=losses,gradient_norms=norms,
        frozen_encoder_passes=1 if cached_features else 0,feature_bytes=sum(v.numel()*v.element_size() for v in encoded.values()),
        critic_feature_contract='independent_pretrained_frozen_cache' if cached_features else 'recompute_trainable_encoders_each_step',
        global_sample_visits=epochs*len(selected),metric_scope='fixed_rollout_bootstrap_targets_next_batch_before_update_is_separate')
