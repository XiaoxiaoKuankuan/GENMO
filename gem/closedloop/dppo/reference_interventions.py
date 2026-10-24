"""固定Stage1世界参考上的连续、因果、仅自由区的诊断干预。

输入是零扰动门槛采集的逐计划世界qpos、当前真实已提交前缀以及音乐绝对时间。
高度/姿态/关节幅度使用全音乐时间的平滑起效函数；相邻计划用已提交前缀的参考
偏移及一二阶差分约束五次桥接，避免每个生成块重新起落。这里的前缀是已提交参考，
不是机器人执行后的状态，不作为Actor监督目标。30→50Hz使用原生产重采样实现，
检查关节边界、根姿态、全身FK及足部速度加速度。拒绝不能伪装成合法改善。
仅显式实验C调用，不进入正常RoboFFT采样或Actor更新；原prepare/commit合同仍执行。
"""
from __future__ import annotations
from contextlib import contextmanager
import copy
from pathlib import Path
import sys
import numpy as np
import torch
from scipy.spatial.transform import Rotation


VARIANTS=('height_plus','height_minus','yaw_plus','yaw_minus','pitch_plus','roll_plus','leg_minus','smooth05')


def rotation_from(q):
    return Rotation.from_quat(np.asarray(q)[...,[1,2,3,0]])


def smoothstep(t):
    u=np.clip((np.asarray(t)-.5)/2.,0.,1.)
    return u**3*(10-15*u+6*u*u)


def pose_delta(base,changed):
    return np.concatenate((changed[:,:3]-base[:,:3],
        (rotation_from(changed[:,3:7])*rotation_from(base[:,3:7]).inv()).as_rotvec(),changed[:,7:]-base[:,7:]),-1)


def apply_delta(base,delta):
    q=np.asarray(base,dtype=np.float64).copy();q[:,:3]+=delta[:,:3];q[:,7:]+=delta[:,6:]
    quat=(Rotation.from_rotvec(delta[:,3:6])*rotation_from(base[:,3:7])).as_quat()
    q[:,3:7]=quat[:,[3,0,1,2]]
    return q


def bridge_delta(desired,base,prefix,frames=12):
    """五次Hermite连接已提交参考的偏移/速度/加速度与期望自由区；不改前缀。"""
    n=len(prefix)
    if n<3 or n+frames+2>=len(base):raise ValueError('Insufficient known/free support for C2 bridge')
    known=pose_delta(base[:n],prefix);out=desired.copy();out[:n]=known
    start=n-1;end=start+frames
    value=known[-1];velocity=1.5*known[-1]-2*known[-2]+.5*known[-3]
    acceleration=known[-1]-2*known[-2]+known[-3]
    target=desired[end];tv=(desired[end+1]-desired[end-1])/2;ta=desired[end+1]-2*desired[end]+desired[end-1]
    T=float(frames);c0=value;c1=velocity*T;c2=acceleration*T*T/2
    rhs=np.stack((target-c0-c1-c2,tv*T-c1-2*c2,ta*T*T-2*c2))
    tail=np.linalg.solve(np.array([[1,1,1],[3,4,5],[6,12,20]],dtype=float),rhs)
    u=np.arange(1,frames+1,dtype=float)[:,None]/T
    out[n:end+1]=c0+c1*u+c2*u*u+tail[0]*u**3+tail[1]*u**4+tail[2]*u**5
    return out


def desired_delta(base,tick,variant,standing,joint_names):
    delta=np.zeros((len(base),27));gate=smoothstep((tick+np.arange(len(base))*20-600)/600.)
    if variant.startswith('height'):delta[:,2]=gate*(.005 if variant.endswith('plus') else -.005)
    elif variant.startswith(('yaw','pitch','roll')):
        axis={'roll':3,'pitch':4,'yaw':5}[variant.split('_')[0]]
        delta[:,axis]=gate*(.01 if variant.endswith('plus') else -.01)
    elif variant.startswith('leg'):
        ids=[i for i,name in enumerate(joint_names) if any(s in name for s in ('leg_','hip','knee','ankle'))]
        delta[:,np.array(ids)+6]=-.01*gate[:,None]*(base[:,np.array(ids)+7]-standing[np.array(ids)+7])
    elif variant.startswith('smooth'):
        smooth=base[:,7:].copy();smooth[2:-2]=(base[:-4,7:]+4*base[1:-3,7:]+6*base[2:-2,7:]+4*base[3:-1,7:]+base[4:,7:])/16
        delta[:,6:]=.05*gate[:,None]*(smooth-base[:,7:])
    elif variant!='identity':raise ValueError('Unknown legal reference intervention')
    return delta


def check_proposal(base,new,free,kin,resample):
    """检查源30Hz与实际50Hz增量，包含所有FK刚体/足部以及旋转连续性。"""
    limits=kin.joint_limits.detach().cpu().numpy();free=np.asarray(free,dtype=bool)
    free_joint=free[:,9:]
    excess=np.maximum(limits[:,0]-new[:,7:],new[:,7:]-limits[:,1])
    base_excess=np.maximum(limits[:,0]-base[:,7:],base[:,7:]-limits[:,1])
    new_hard=int(((excess>1e-5)&free_joint).sum())
    metrics=dict(free_hard_limit_coordinates=new_hard,
        inherited_prefix_limit_coordinates=int(((excess>1e-5)&~free_joint).sum()),
        worsened_limit_coordinates=int(((excess>np.maximum(base_excess,0)+1e-5)&free_joint).sum()))
    active=np.flatnonzero(free.any(-1))
    if not len(active):return True,metrics
    stop=int(active[-1])+1
    # 音乐无效尾部不属于可消费参考，差分不能跨过真实有效端点。
    for hz,a,b in ((30,base[:stop],new[:stop]),(50,resample(base[:stop]),resample(new[:stop]))):
        delta=pose_delta(a,b);v=np.diff(delta,axis=0)*hz;acc=np.diff(v,axis=0)*hz
        for label,sl in (('root_position',slice(0,3)),('root_orientation',slice(3,6)),('joint',slice(6,None))):
            metrics[f'{label}_velocity_delta_{hz}']=float(np.abs(v[:,sl]).max())
            metrics[f'{label}_acceleration_delta_{hz}']=float(np.abs(acc[:,sl]).max())
        device=kin.joint_limits.device
        with torch.no_grad():
            fa=kin.forward_body_positions(torch.as_tensor(a,dtype=torch.float32,device=device))
            fb=kin.forward_body_positions(torch.as_tensor(b,dtype=torch.float32,device=device))
            d=(fb-fa).cpu().numpy()
        if not np.isfinite(d).all():raise ValueError('Nonfinite reference FK')
        bv=np.diff(d,axis=0)*hz;ba=np.diff(bv,axis=0)*hz
        metrics[f'body_velocity_delta_{hz}']=float(np.abs(bv).max());metrics[f'body_acceleration_delta_{hz}']=float(np.abs(ba).max())
    legal=(new_hard==0 and all(metrics[f'joint_velocity_delta_{hz}']<=.6 and
        metrics[f'joint_acceleration_delta_{hz}']<=20 and metrics[f'body_velocity_delta_{hz}']<=.25 and
        metrics[f'body_acceleration_delta_{hz}']<=12 and metrics[f'root_orientation_velocity_delta_{hz}']<=.25 and
        metrics[f'root_orientation_acceleration_delta_{hz}']<=3 for hz in (30,50)))
    return legal,metrics


@contextmanager
def fixed_world_intervention(c,bank,variant,audit):
    from . import world_collector
    from .joint_limits import generated_boundaries_tensors,generated_boundary_reports
    sys.path.insert(0,str(Path(c.config['paths']['gmt_repo'])/'scripts/motion'))
    from bumi4340_reference_kinematics import resample
    original=world_collector.generate_batch;policy=c.policy;kin=policy.actor.endecoder.kinematics
    standing=kin.make_standing_qpos().cpu().numpy();limits=kin.joint_limits.detach().cpu().numpy()
    def generate(current_policy,packets,**kwargs):
        outputs,timing=original(current_policy,packets,**kwargs)
        for item,packet in zip(outputs,packets):
            generated=item['generated'];trace=item['trace'];seed=int(packet['seed'])
            if seed not in bank:
                if not audit.get('allow_unpaired_extension',False):raise ValueError('No matched Stage1 plan; counterfactual extrapolation forbidden')
                audit['unpaired_extension_plans']=audit.get('unpaired_extension_plans',0)+1
                base=np.asarray(generated['qpos_world'],dtype=np.float64)
            else:base=np.asarray(bank[seed]['qpos_world'],dtype=np.float64)
            prefix=np.asarray(packet['reservation']['source_qpos'],dtype=np.float64)
            free=trace['free_mask'][0].detach().cpu().numpy();n=len(prefix)
            if variant=='identity':
                if not np.array_equal(np.asarray(generated['qpos_world']),base):
                    raise ValueError('Zero replay generated reference differs from Stage1 bank')
                audit['plans']+=1;continue
            desired=desired_delta(base,int(generated['decision_tick']),variant,standing,kin.joint_order)
            accepted=False
            for factor in (1.,.5,.25,.125,0.):
                delta=bridge_delta(desired*factor,base,prefix)
                proposal=apply_delta(base,delta)
                # 自由关节投影到硬限位；已提交超限前缀只记录，绝不修改。
                proposal[:,7:]=np.where(free[:,9:],np.clip(proposal[:,7:],limits[:,0]+1e-6,limits[:,1]-1e-6),proposal[:,7:])
                proposal[:n]=prefix
                proposal[~free.any(-1)]=np.asarray(generated['qpos_world'])[~free.any(-1)]
                proposal[:n]=prefix
                legal,metrics=check_proposal(base,proposal,free,kin,resample)
                if legal:accepted=True;break
            audit['plans']+=1
            if not accepted:
                audit['invalid_plans']=audit.get('invalid_plans',0)+1
                raise ValueError('Reference intervention rejected before submission: '+repr(metrics))
            audit['free_limit_projection_coordinates']=audit.get('free_limit_projection_coordinates',0)+int(
                (((base[:,7:]<limits[:,0])|(base[:,7:]>limits[:,1]))&free[:,9:]).sum())
            audit['minimum_amplitude_factor']=min(audit.get('minimum_amplitude_factor',1.),factor)
            for name,value in metrics.items():audit[name]=max(audit.get(name,0),value)
            if not np.array_equal(proposal[:n],prefix):raise AssertionError('Committed source prefix modified')
            device=trace['qpos30'].device;codec=policy.actor.endecoder.codec
            q=torch.as_tensor(proposal,dtype=torch.float32,device=device)[None]
            anchor=codec.build_canonical_anchor(q)
            components,_,_=codec.canonicalize(q,kin.forward_body_positions(q),anchor=anchor)
            physical=codec.assemble_features(components)
            physical=torch.where(trace['free_mask'],physical,trace['qpos30'])
            trace['qpos30']=physical;trace['qpos']=codec.decode_to_canonical_qpos(physical)
            trace['normalized']=(physical-policy.actor.endecoder.mean.to(physical))/policy.actor.endecoder.std.to(physical)
            trace['normalized']=torch.where(trace['conditions']['future_valid'][...,None],trace['normalized'],0.)
            trace['action_x0']=trace['normalized'].clone()
            boundary=generated_boundaries_tensors(q,trace['free_mask'][:,:,9:],[packet['snapshot']],kin.joint_order)
            generated.update(qpos_world=proposal,qpos30=physical[0].cpu().numpy().copy(),
                joint_limit_diagnostics=generated_boundary_reports({k:v.cpu() for k,v in boundary.items()})[0])
            block,position=item['_shared_raw_block']
            for field in ('qpos','qpos30','normalized','action_x0','contact'):block['trace'][field][position:position+1]=trace[field]
        return outputs,timing
    world_collector.generate_batch=generate
    try:yield
    finally:world_collector.generate_batch=original
