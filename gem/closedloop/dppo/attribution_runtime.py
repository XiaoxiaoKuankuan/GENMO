"""真实rollout的分块概率归因与固定噪声输出对照运行接口。

本模块从生产ConditionGraphCache和同一个transition_parameters取均值/实际标准差，
在完整旧链状态上重算，不再实现第二套扩散采样器。逐微批把坐标FP64结果搬到CPU，
用于离线精确统计；真实创新噪声与独立Normal密度同时核验。元数据包含行为身份、
实际前缀、来源、音乐时间及每条链的步骤/帧/分组贡献，以便追查最高权重尾部样本。

固定噪声探针复用output_probe，覆盖普通样本和高KL/高ratio样本，保存四步状态、
冻结确定性末步输入及最终输出的差异。所有输出明确为反事实生成参考，不能当作
机器人实际轨迹。可训练范围隔离只应用真实同一次Adam的参数增量，不修改采样核。
"""
from __future__ import annotations

import torch

from .actor_step_diagnostics import output_probe
from .execution_checks import policy_phase
from .optimization_support import _parameters,_old_kernel
from .probability_attribution import coordinate_statistics,feature_groups,quantiles
from .tensor_cache import ConditionGraphCache


def row_metadata(row,advantage,rank):
    task=row.metadata.get('training_task',{})
    remaining=row.metadata.get('remaining_music_seconds')
    frames=row.metadata.get('full_music_num_frames')
    music_frame=(frames-round(remaining*30)) if frames is not None and remaining is not None else task.get('music_start_frame')
    return dict(identity=dict(row.identity),rank=rank,source=task.get('dataset','unknown'),
        sample_id=task.get('sample_id'),music_frame=music_frame,
        music_time_bin='unknown' if music_frame is None else str(int(music_frame)//300),
        music_start_frame=row.metadata.get('music_start_frame'),
        decision_time_seconds=float(row.context['decision_time'].reshape(-1)[0]),
        prefix_frames=int((row.context['known_qpos30_mask'].all(-1)&row.context['future_valid']).sum()),
        advantage=float(advantage),compact_execution_available='compact_execution_provenance' in row.metadata)


@torch.no_grad()
def collect_coordinates(policy,rows,cache,*,rank=0,microbatch=128):
    if not rows:raise ValueError('Coordinate attribution requires real valid rows')
    n=len(rows);frames=rows[0].free_mask.shape[0];device=next(policy.actor.parameters()).device
    names=('kl','log_ratio','mean_delta','standardized_mean_delta','residual','actual_std','std_log_change','sampling_noise_error')
    count=len(policy.trainable_steps)
    detail={name:torch.zeros(n,count,frames,30,dtype=torch.float64) for name in names}
    detail['mask']=torch.zeros(n,count,frames,30,dtype=torch.bool)
    old_error=torch.zeros(n,count,dtype=torch.float64)
    with policy_phase(policy):
        conditions=ConditionGraphCache(policy,cache,include_frozen=False);conditions.prime(rows)
        pairs=[(i,step) for i in range(n) for step in policy.trainable_steps]
        for offset in range(0,len(pairs),microbatch):
            chunk=pairs[offset:offset+microbatch];samples=[rows[i] for i,_ in chunk];steps=[s for _,s in chunk]
            params,mask=_parameters(policy,samples,steps,device,cache,conditions)
            old_mean,old_std=_old_kernel(samples,steps,device,cache)
            value=cache.get('chain',samples,[s+1 for s in steps])
            values=coordinate_statistics(value,old_mean,old_std,params['mean'],params['std'],mask)
            noise=cache.get('sampling_noise',samples,steps).double()
            values['sampling_noise_error']=torch.where(mask,(values['residual']-noise).abs(),0.)
            index=torch.tensor([i for i,_ in chunk]);step=torch.tensor([s-policy.trainable_steps[0] for s in steps])
            for name,tensor in values.items():detail[name][index,step]=tensor.cpu()
            reference=torch.distributions.Normal(old_mean.double(),old_std.double()).log_prob(value.double()).masked_fill(~mask,0.).sum((-2,-1))
            old_error[index,step]=(reference-cache.get('old_log_prob',samples,steps)).abs().cpu()
        conditions.validate_inputs()
    joint_order=policy.actor.endecoder.kinematics.joint_order
    groups=feature_groups(joint_order)
    metadata=[]
    for i,row in enumerate(rows):
        item=row_metadata(row,cache.get('advantages',[row])[0],rank)
        item.update(kl_by_step=detail['kl'][i].sum((1,2)).tolist(),
            log_ratio_by_step=detail['log_ratio'][i].sum((1,2)).tolist(),
            kl_by_frame=detail['kl'][i].sum((0,2)).tolist(),
            log_ratio_by_frame=detail['log_ratio'][i].sum((0,2)).tolist(),
            kl_by_feature={name:float(detail['kl'][i,:,:,indices].sum()) for name,indices in groups.items()},
            log_ratio_by_feature={name:float(detail['log_ratio'][i,:,:,indices].sum()) for name,indices in groups.items()},
            old_log_probability_abs_error_by_step=old_error[i].tolist())
        metadata.append(item)
    return detail,metadata


def probe_difference(before,after):
    """逐帧及末步映射的变化；qpos为codec解码的canonical坐标，非世界机器人状态。"""
    if before is None:return []
    result=[]
    from .updater import _quaternion_angle
    for i,identity in enumerate(before['identities']):
        mask=before['mask'][i];valid=mask.any(-1);delta=(after['final'][i]-before['final'][i]).double()
        pre=(after['pre_final'][i]-before['pre_final'][i]).double()
        joints=(after['qpos'][i,:,7:]-before['qpos'][i,:,7:]).double()
        def rms(value,select):return float(value[select].square().mean().sqrt()) if select.any() else None
        frame=[]
        for t in range(len(mask)):
            frame.append(dict(frame=t,free_coordinates=int(mask[t].sum()),
                normalized_rms=rms(delta[t],mask[t]),joint_rms_rad=rms(joints[t],mask[t,9:])))
        before_rms=rms(pre,mask);after_rms=rms(delta,mask)
        result.append(dict(identity=identity,source=before['sources'][i],advantage=float(before['advantages'][i]),
            final_normalized_rms=after_rms,pre_final_normalized_rms=before_rms,
            deterministic_final_gain=after_rms/before_rms if before_rms else None,
            moving_state_delta_rms=[rms((after['moving_states'][i,k]-before['moving_states'][i,k]).double(),mask) for k in range(before['moving_states'].shape[1])],
            old_path_mean_delta_rms=[rms((after['mean'][i,k]-before['mean'][i,k]).double(),mask) for k in range(before['mean'].shape[1])],
            canonical_root_position_rms_m=rms((after['qpos'][i,:,:3]-before['qpos'][i,:,:3]).double(),valid),
            canonical_root_orientation_rms_rad=rms(_quaternion_angle(before['qpos'][i,:,3:7],after['qpos'][i,:,3:7]),valid),
            final_joint_rms_rad=rms(joints,mask[:,9:]),
            contact_rms=float((after['contact'][i]-before['contact'][i]).double().square().mean().sqrt()),
            protected_unchanged=bool(torch.equal(after['final'][i][~mask],before['final'][i][~mask])),by_frame=frame))
    return result


def probe_summary(samples):
    keys=('final_normalized_rms','pre_final_normalized_rms','deterministic_final_gain','canonical_root_position_rms_m',
        'canonical_root_orientation_rms_rad','final_joint_rms_rad','contact_rms')
    return dict(samples=samples,covered_chains=len(samples),
        distributions={key:quantiles([s[key] for s in samples if s[key] is not None]) for key in keys},
        scope='same_real_conditions_and_innovations_counterfactual_reference_not_PhysX_motion')
