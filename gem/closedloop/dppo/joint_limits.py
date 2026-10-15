"""真实参考/实际关节边界与完整新生成窗口的独立诊断。

硬限位及软限位来自同一机器人运行时、GMT具名关节顺序。记录每关节越界量、比例、
名称与连续50Hz越界时长；连续计数严格按同一episode的已执行控制步更新。
完整60帧诊断仅检查可修改且有效的关节坐标，使用实际生成的21关节角按名称重排。
该完整窗口结果只做诊断，绝不再次累加到50Hz奖励，也不裁剪动作或已承诺参考。
在线生成使用设备批量归约后随必需参考输出合并传输；保留NumPy逐任务函数作为
独立诊断参照，二者阈值、FP64计算和完整逐坐标证据相同，不逐环境同步CUDA。
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


def generated_boundaries_tensors(qpos,free_joint_mask,snapshots,source_joint_names,*,check_finite=True):
    """批量计算[B,60/120,21]合法性；只返回设备张量，不逐环境读GPU标量。

    名称和权威限位在CPU元数据上核验，全部越界计算在qpos所在设备按FP64执行。
    调用方已经检查完整生成输出有限时可关闭重复有限性同步；独立调用默认检查。
    """
    import torch
    if qpos.ndim!=3 or qpos.shape[-1]!=28 or not len(qpos) or len(snapshots)!=len(qpos):
        raise ValueError('Batched generated qpos requires nonempty [B,T,28] and one snapshot per row')
    if free_joint_mask.shape!=(*qpos.shape[:2],21) or free_joint_mask.device!=qpos.device:
        raise ValueError('Generation joint mask differs from actual qpos')
    names=GMT_EXPECTED_JOINT_ORDER
    if len(source_joint_names)!=21 or set(source_joint_names)!=set(names):
        raise ValueError('Generated source joint names must be a permutation of GMT joints')
    if any(tuple(s['joint_names'])!=names for s in snapshots):
        raise ValueError('Generated reference boundary joint order mismatch')
    bounds=np.asarray([[s['joint_hard_limits_rad'],s['joint_soft_limits_rad']] for s in snapshots],dtype=np.float64)
    if bounds.shape!=(len(qpos),2,21,2) or not np.isfinite(bounds).all() or (bounds[...,1]<=bounds[...,0]).any():
        raise ValueError('Joint boundaries require ordered finite [21,2] limits')
    indices=torch.tensor([list(source_joint_names).index(n) for n in names],device=qpos.device)
    position=qpos[...,7:].index_select(-1,indices).double()
    if check_finite and not bool(torch.isfinite(position).all()):
        raise ValueError('Nonfinite joint boundary input')
    valid=free_joint_mask.index_select(-1,indices).bool()
    limits=torch.as_tensor(bounds,device=qpos.device)
    excess=torch.maximum(limits[:,:,None,:,0]-position[:,None],position[:,None]-limits[:,:,None,:,1]).clamp_min_(0.)
    excess=excess.masked_fill(~valid[:,None],0.)
    violation=(excess>1e-6)&valid[:,None]
    count=valid.sum((-2,-1))
    return dict(max_excess_rad=excess.amax((-2,-1)),
        violation_fraction=violation.sum((-2,-1)).double()/count.clamp_min(1)[:,None],
        valid_coordinates=count,per_joint_max_excess_rad=excess.amax(-2),violation_mask=violation)


def generated_boundary_reports(host_metrics):
    """合并D2H结束后恢复既有证据结构；这里禁止隐式CUDA标量读取。"""
    if any(v.device.type!='cpu' for v in host_metrics.values()):
        raise ValueError('Batch boundary reports require one completed host transfer')
    arrays={k:v.numpy() for k,v in host_metrics.items()};records=[]
    for i,count in enumerate(arrays['valid_coordinates']):
        report=dict(scope='all_modifiable_valid_generated_coordinates_not_execution_reward',reward_weight=0.)
        for j,kind in enumerate(('hard','soft')):
            violation=arrays['violation_mask'][i,j]
            report[kind]=dict(max_excess_rad=float(arrays['max_excess_rad'][i,j]),
                violation_fraction=float(arrays['violation_fraction'][i,j]),valid_coordinates=int(count),
                per_joint_max_excess_rad=arrays['per_joint_max_excess_rad'][i,j].tolist(),
                violating_joint_names=[n for n,bad in zip(GMT_EXPECTED_JOINT_ORDER,violation.any(0)) if bad],
                violation_mask=violation.tolist())
        records.append(report)
    return records


def summarize_boundaries(records,*,generated=()):
    """普通训练仅持久化聚合数值/关节名称；原始逐控制步证据仍留到本轮更新结束。"""
    result=dict(executed_control_samples=len(records),generated_proposals=len(generated),
        generated_scope='all_modifiable_valid_coordinates_not_added_to_reward')
    for side in ('actual','consumed_reference'):
        result[side]={}
        cost='actual_cost' if side=='actual' else 'reference_cost'
        values=[r[cost] for r in records]
        result[side]['cost_mean']=float(np.mean(values)) if values else None
        result[side]['cost_max']=max(values) if values else None
        for kind in ('hard','soft'):
            metrics=[r[side][kind] for r in records]
            result[side][kind]=dict(max_excess_rad=max((m['max_excess_rad'] for m in metrics),default=None),
                violation_fraction=float(np.mean([m['violation_fraction'] for m in metrics])) if metrics else None,
                maximum_consecutive_seconds=max((m['maximum_consecutive_seconds'] for m in metrics),default=None),
                violating_joint_names=sorted({n for m in metrics for n in m['violating_joint_names']}))
    result['generated']={}
    for kind in ('hard','soft'):
        metrics=[r[kind] for r in generated];count=sum(m['valid_coordinates'] for m in metrics)
        result['generated'][kind]=dict(valid_coordinates=count,
            max_excess_rad=max((m['max_excess_rad'] for m in metrics),default=None),
            violation_fraction=sum(m['violation_fraction']*m['valid_coordinates'] for m in metrics)/count if count else None,
            violating_joint_names=sorted({n for m in metrics for n in m['violating_joint_names']}))
    return result
