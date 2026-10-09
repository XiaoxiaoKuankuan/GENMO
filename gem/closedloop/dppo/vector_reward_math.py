"""在GPU上按环境维计算原第二阶段连续奖励，不改变活动和音乐的因果定义。

输入直接来自GPU物理、Tracking诊断、实际关节状态和FP64参考库，计算Tracking、
Stable、Alive、Cmd、Torque、Contact和JointLimit七项。环境维从头至尾保留；仅
循环四个物理子步和两只脚，不逐机器人调用NumPy。中间数值统一FP64，对齐原奖励
把物理FP32输入转为NumPy float64后的计算；完整raw/normalized证据随执行边界转移。

音乐离散极小值和配对活动窗继续由原ExecutionReward按每个环境因果计算，避免
GPU归约舍入改变音乐节拍。此模块不产生优化器、不计算优势、不改变权重；无效
数值、关节限位或物理子步会使设备断言失败。配置SHA和env/tick绑定由上层核对。
启用前必须与原标量奖励在真实八卡保存证据上逐项比较，不允许放宽概率门槛。
"""
from __future__ import annotations
import hashlib
import json
import torch
from .rewards import resolve_reward_config, TRACKING_ERROR_FIELDS

VECTOR_REWARD_VERSION='stage10.gpu_vector_continuous_reward.v1'


def reward_config_sha(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


class VectorRewardMath:
    def __init__(self, config, num_envs, device):
        self.config=resolve_reward_config(config)
        self.config_sha=reward_config_sha(self.config)
        self.previous=torch.zeros(num_envs,21,dtype=torch.float64,device=device)
        self.previous_valid=torch.zeros(num_envs,dtype=torch.bool,device=device)

    def reset(self, ids):
        self.previous[ids]=0
        self.previous_valid[ids]=False

    @torch.no_grad()
    def compute(self, *, tracking, errors, substeps, actual_joint_pos, reference_joint_pos,
                terminated, active, ticks):
        if len(substeps)!=4:raise ValueError('GPU reward requires four true physical substeps')
        cfg=self.config;last=substeps[-1];checks=[]
        def number(value):
            value=value.double()
            checks.append(torch.isfinite(value).all())
            return value
        def checked(flag):checks.append(flag.all())
        def term(score,raw,normalized=None,valid=None):
            checked(torch.isfinite(score)&(score>=0)&(score<=1.+1e-12))
            return dict(score=score,valid=torch.ones_like(active) if valid is None else valid,
                raw=raw,normalized={} if normalized is None else normalized)
        scores={};normalized={};raw={}
        for name,item in tracking['terms'].items():
            error=number(item['error']);checked(error>=0)
            normalized[name]=error/cfg['tracking']['std'][name]**2
            scores[name]=torch.exp(-normalized[name]);raw[TRACKING_ERROR_FIELDS[name]]=error
        raw.update(source='frozen_gmt_current_motion_command',objective='gmt.motion_tracking.v1',
            anchor_body_name=tracking['anchor_body_name'],body_names=tracking['body_names'],
            joint_names=tracking['joint_names'],terms=tracking['terms'],control_tick=ticks)
        track=sum(cfg['tracking']['weights'][name]*score for name,score in scores.items())
        output={'track':term(track,raw,dict(error_over_std_squared=normalized,scores=scores,
            std=cfg['tracking']['std'],weights=cfg['tracking']['weights']))}
        raw={};normalized={};scores={}
        for name,key,scale in (('root_height','root_height_error_m','root_height_m'),
                               ('non_yaw','non_yaw_orientation_error_rad','non_yaw_rad')):
            raw[key]=number(errors[key]);normalized[name]=raw[key]/cfg['scales'][scale]
            scores[name]=torch.exp(-normalized[name].abs().clamp_max(1e150).square())
        output['stable']=term(sum(cfg['stable_mix'][name]*score for name,score in scores.items()),raw,
            dict(errors_over_scale=normalized,scores=scores))
        output['alive']=term((~terminated).double(),dict(complete_control_interval=True,
            execution_valid=True,execution_failed=terminated))
        target=number(last['joint_position_target']);limit=number(last['joint_velocity_limits_rad_s'])
        checked(limit>0)
        rate=torch.where(self.previous_valid[:,None],(target-self.previous)/cfg['dt'],0.)
        normalized=rate.abs()/limit;cost=normalized.square().clamp(0.,1.)
        output['cmd']=term(cost.mean(-1),dict(joint_position_target_rad=target,
            previous_joint_position_target_rad=self.previous.clone(),cmd_rate_rad_s=rate,
            joint_velocity_limits_rad_s=limit,first_step=~self.previous_valid),
            dict(absolute_rate_over_limit=normalized,per_joint_cost=cost),valid=self.previous_valid.clone())
        self.previous.copy_(torch.where(active[:,None],target,self.previous))
        self.previous_valid|=active
        torque_costs=[];ratios=[];torque_details=[]
        slides=[];bads=[];contact_details=[]
        for index,d in enumerate(substeps):
            checked((d['physics_tick']==ticks-12+3*(index+1))|~active)
            torque=number(d['pd_torque_estimate_nm']);limit=number(d['joint_effort_limits_nm']);checked(limit>0)
            ratio=torque.abs()/limit
            cost=((ratio-cfg['torque']['free_ratio'])/(1.-cfg['torque']['free_ratio'])).clamp(0.,1.).square()
            value=cfg['torque']['mean_weight']*cost.mean(-1)+cfg['torque']['max_weight']*cost.amax(-1)
            ratios.append(ratio);torque_costs.append(value)
            torque_details.append(dict(physics_tick=d['physics_tick'],time_s=d['physics_tick'].double()/600.,
                pd_torque_estimate_nm=torque,joint_effort_limits_nm=limit,per_joint_cost=cost,cost=value))
            feet,names=d['foot_body_names'],d['contact_body_names'];allowed=d['allowed_contact_body_names']
            if not set(feet)<=set(allowed)<=set(names):raise ValueError('Invalid fixed GPU contact binding')
            threshold=d['undesired_contact_force_threshold_n'];support_threshold=d['foot_contact_force_threshold_n']
            if threshold<0 or support_threshold<0:raise ValueError('Negative contact force threshold')
            force=number(d['foot_net_contact_forces_w_n']);net=number(d['net_contact_forces_w_n'])
            force_norm=torch.linalg.vector_norm(force,dim=-1)
            contact=force_norm>support_threshold
            foot_costs=[];supported=[];speed_raw={};selected_raw={}
            for foot_index,name in enumerate(feet):
                speed=number(d['foot_support_sphere_tangent_speed_m_s'][name])
                clearance=number(d['foot_support_sphere_clearance_m'][name]);checked(speed>=0)
                selected=contact[:,foot_index,None]&(clearance<=cfg['contact']['support_clearance_m'])
                count=selected.sum(-1);supported.append(count>0)
                costs=(speed/cfg['scales']['slide_m_s']).square().clamp(0.,1.)
                foot_costs.append(torch.where(selected,costs,0.).sum(-1)/count.clamp_min(1))
                speed_raw[name]=speed;selected_raw[name]=selected
            support_count=torch.stack(supported,-1).sum(-1)
            slide=torch.stack(foot_costs,-1).sum(-1)/support_count.clamp_min(1)
            bad_names=[name for name in names if name not in allowed]
            indices=[names.index(name) for name in bad_names]
            bad=(torch.linalg.vector_norm(net,dim=-1)[:,indices]>threshold).any(-1).double()
            slides.append(slide);bads.append(bad)
            contact_details.append(dict(physics_tick=d['physics_tick'],contact_force_semantics='net_all_colliders_proxy',
                foot_net_contact_forces_w_n=force,net_contact_forces_w_n=net,
                contact_body_names=names,allowed_contact_body_names=allowed,undesired_contact_body_names=bad_names,
                undesired_contact_force_threshold_n=threshold,foot_contact_force_threshold_n=support_threshold,
                slide_velocity_semantics=d.get('slide_velocity_semantics','support_sphere_center_tangent_velocity_proxy'),
                slide_proxy=speed_raw,support_selected=selected_raw,supported_foot_count=support_count,
                slide_cost=slide,bad_contact_cost=bad,peak_foot_net_force_n=force_norm.amax(-1)))
        output['torque']=term(torch.stack(torque_costs,-1).mean(-1),dict(sampling='four_physics_substeps',
            torque_semantics='implicit_PD_torque_estimate_not_hardware_torque',samples=torque_details,
            max_torque_ratio=torch.stack(ratios,1).flatten(1).amax(-1)),
            dict(absolute_torque_over_effort_limit=torch.stack(ratios,1),substep_costs=torch.stack(torque_costs,-1)))
        slide,bad=torch.stack(slides,-1).mean(-1),torch.stack(bads,-1).mean(-1)
        output['contact']=term(cfg['contact']['slide_weight']*slide+cfg['contact']['bad_contact_weight']*bad,
            dict(sampling='four_physics_substeps',samples=contact_details,impact_reward_weight=0.),
            dict(slide=slide,bad_contact=bad,substep_slide_costs=torch.stack(slides,-1),substep_bad_contact_costs=torch.stack(bads,-1)))
        hard=number(last['joint_pos_limits_rad']);checked(hard[...,1]>hard[...,0])
        if last.get('soft_joint_pos_limits_rad') is None:
            margin=(hard[...,1]-hard[...,0])*cfg['joint_limit']['hard_range_margin_fraction']
            safe=torch.stack((hard[...,0]+margin,hard[...,1]-margin),-1);source='hard_range_inner_margin'
        else:safe=number(last['soft_joint_pos_limits_rad']);source='backend_soft_joint_position_limits'
        original=safe;tolerance=cfg['joint_limit']['numeric_tolerance_rad']
        checked((safe[...,0]>=hard[...,0]-tolerance)&(safe[...,1]<=hard[...,1]+tolerance)&(safe[...,1]>=safe[...,0]))
        safe=torch.minimum(torch.maximum(safe,hard[...,:1]),hard[...,1:])
        actual,reference=number(actual_joint_pos),number(reference_joint_pos)
        def joint_cost(position):
            lw,uw=safe[...,0]-hard[...,0],hard[...,1]-safe[...,1]
            lower=torch.where(lw>0,(safe[...,0]-position)/torch.where(lw>0,lw,1.),0.)
            upper=torch.where(uw>0,(position-safe[...,1])/torch.where(uw>0,uw,1.),0.)
            lower=torch.where((lw==0)&(position<=hard[...,0]),1.,lower)
            upper=torch.where((uw==0)&(position>=hard[...,1]),1.,upper)
            return torch.maximum(lower,upper).clamp(0.,1.)
        ac,rc=joint_cost(actual),joint_cost(reference)
        output['joint_limit']=term(cfg['joint_limit']['actual_weight']*ac.amax(-1)+cfg['joint_limit']['reference_weight']*rc.amax(-1),
            dict(actual_joint_position_rad=actual,consumed_reference_joint_position_rad=reference,
                reference_tick=ticks,hard_limits_rad=hard,original_safe_limits_rad=original,safe_limits_rad=safe,
                safe_limits_source=source,numeric_tolerance_rad=tolerance,numeric_boundary_clamped=(safe!=original).flatten(1).any(-1),
                numeric_boundary_note='only floating-point soft-limit excess within configured tolerance is clipped to hard limits'),
            dict(actual_per_joint_cost=ac,reference_per_joint_cost=rc))
        torch._assert_async(torch.stack(checks).all(),'Invalid GPU batched reward input or score')
        return dict(schema=VECTOR_REWARD_VERSION,config_sha256=self.config_sha,
            env_id=last['env_index'],tick=ticks.clone(),components=output)
