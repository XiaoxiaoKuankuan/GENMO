"""真实参考/实际关节边界与完整新生成窗口的独立诊断。

硬限位及软限位来自同一机器人运行时、GMT具名关节顺序。记录每关节越界量、比例、
名称与连续50Hz越界时长；连续计数严格按同一episode的已执行控制步更新。
完整60帧诊断仅检查可修改且有效的关节坐标，使用实际生成的21关节角按名称重排。
该完整窗口结果只做诊断，绝不再次累加到50Hz奖励，也不裁剪动作或已承诺参考。
"""
from __future__ import annotations
import numpy as np
from gem.closedloop.contracts import GMT_EXPECTED_JOINT_ORDER


def boundary_metrics(position,limits,names,mask=None):
    q=np.asarray(position,dtype=np.float64);bounds=np.asarray(limits,dtype=np.float64)
    if bounds.shape!=(21,2) or q.shape[-1]!=21 or (bounds[:,1]<=bounds[:,0]).any():
        raise ValueError('Joint boundaries require ordered finite [21,2] limits')
    if not np.isfinite(q).all() or not np.isfinite(bounds).all():raise ValueError('Nonfinite joint boundary input')
    valid=np.ones(q.shape,dtype=bool) if mask is None else np.asarray(mask,dtype=bool)
    if valid.shape!=q.shape:raise ValueError('Generation joint mask differs from actual qpos')
    excess=np.maximum(np.maximum(bounds[:,0]-q,q-bounds[:,1]),0.)
    excess=np.where(valid,excess,0.);violation=(excess>1e-6)&valid
    per_joint=excess if q.ndim==1 else excess.max(axis=0)
    return dict(max_excess_rad=float(excess.max()),violation_fraction=float(violation.sum()/max(1,valid.sum())),
        valid_coordinates=int(valid.sum()),per_joint_max_excess_rad=per_joint.tolist(),
        violating_joint_names=[name for name,bad in zip(names,violation if q.ndim==1 else violation.any(axis=0)) if bad],
        violation_mask=violation.tolist())


def executed_boundaries(row,previous_runs):
    d=row['physical_diagnostics'];names=d['joint_names']
    if tuple(names)!=GMT_EXPECTED_JOINT_ORDER:raise ValueError('Runtime joint-limit order differs from GMT')
    hard=np.asarray(d['joint_pos_limits_rad'],dtype=np.float64)
    soft=np.asarray(d['soft_joint_pos_limits_rad'],dtype=np.float64)
    if (soft[:,0]<hard[:,0]-1e-6).any() or (soft[:,1]>hard[:,1]+1e-6).any():
        raise ValueError('Runtime soft limits lie outside hard limits')
    runs=np.asarray(previous_runs,dtype=np.int64).copy();result={}
    for side,position in enumerate((row['actual_joint_pos_gmt'],row['reference']['joint_pos'])):
        record={}
        for kind,limits in enumerate((hard,soft)):
            metric=boundary_metrics(position,limits,names)
            runs[side,kind]=np.where(metric['violation_mask'],runs[side,kind]+1,0)
            metric['consecutive_seconds']=(runs[side,kind]*.02).tolist()
            metric['maximum_consecutive_seconds']=float(runs[side,kind].max()*.02)
            record[('hard','soft')[kind]]=metric
        result[('actual','consumed_reference')[side]]=record
    return dict(schema='genmo.joint_boundary_diagnostics.v1',tick=int(row['tick']),joint_names=list(names),
        scope='actual_control_end_and_current_consumed_reference',**result),runs


def generated_boundaries(qpos,free_joint_mask,snapshot,source_joint_names):
    names=snapshot['joint_names']
    if tuple(names)!=GMT_EXPECTED_JOINT_ORDER:raise ValueError('Generated reference boundary joint order mismatch')
    indices=[list(source_joint_names).index(name) for name in names]
    position=np.asarray(qpos)[:,7:][:,indices];mask=np.asarray(free_joint_mask)[:,indices]
    return dict(scope='all_modifiable_valid_generated_coordinates_not_execution_reward',reward_weight=0.,
        hard=boundary_metrics(position,snapshot['joint_hard_limits_rad'],names,mask),
        soft=boundary_metrics(position,snapshot['joint_soft_limits_rad'],names,mask))
