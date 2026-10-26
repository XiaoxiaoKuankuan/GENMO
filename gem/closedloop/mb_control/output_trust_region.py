"""GENMO真实输出空间的候选AdamW更新、验收与事务回滚。

输入为已经由真实接管anchor得到的有限执行梯度，以及固定请求/预测条件/前缀/噪声的
重生成检查函数。每个候选从相同Actor、AdamW全部状态和RNG开始，仅缩小候选学习率。
只有物理参考变化、冻结教师累计偏差、活动度和同anchor舞蹈损失同时合格才保留更新。
拒绝时恢复参数、buffer、一二阶矩、step和随机数，不增加任何训练计数。该模块本身
不声称WM验收等于真实物理收益；真正Newton候选对照由独立审计执行。
"""

import copy
import math
import time
import torch
from supermimic.rotations import quat_to_mat,so3_log
from .checkpoint import capture_rng,restore_rng


def rms(x): return float(x.float().square().mean().sqrt())


def motion_distance(a,b,mask):
    a,b=a[:,mask],b[:,mask]
    return {"joint_rms_rad":rms(a[...,7:]-b[...,7:]),"root_rms_m":rms(a[...,:3]-b[...,:3]),
        "orientation_rms_rad":rms(so3_log(quat_to_mat(b[...,3:7]).transpose(-1,-2)@quat_to_mat(a[...,3:7])))}


def activity_metrics(a,b,mask):
    a,b=a[:,mask,7:],b[:,mask,7:]
    if a.shape[1]<3:raise ValueError("信任域活动度需要至少三个真实目标支持点")
    da,db=torch.diff(a,dim=1)*30,torch.diff(b,dim=1)*30
    ar,br=(a.amax(1)-a.amin(1)).mean(),(b.amax(1)-b.amin(1)).mean()
    av,bv=da.square().mean().sqrt(),db.square().mean().sqrt()
    return {"range_ratio":float(ar/br) if br>.05 else None,"speed_ratio":float(av/bv) if bv>.2 else None,
        "velocity_phase_cosine":float(torch.nn.functional.cosine_similarity(da.flatten(),db.flatten(),dim=0)) if bv>.2 else None,
        "teacher_range_rad":float(br),"teacher_speed_rad_s":float(bv),
        "candidate_range_rad":float(ar),"candidate_speed_rad_s":float(av)}


def check_outputs(before,after,local,global_target,mask,*,prefix,config,unaligned_global_target=None):
    if not torch.isfinite(after).all():return ["nonfinite_candidate"],{}
    all_free=torch.arange(after.shape[1],device=after.device)>=prefix
    step=motion_distance(after,before,all_free)
    local_drift=motion_distance(after,local,all_free)
    global_drift=motion_distance(after,global_target,mask)
    activity=activity_metrics(after,global_target,mask)
    local_activity=activity_metrics(after,local,all_free)
    reasons=[]
    for field,limit,reason in (("joint_rms_rad","joint_step_rad","joint_step_exceeded"),
                              ("root_rms_m","root_step_m","root_step_exceeded"),
                              ("orientation_rms_rad","orientation_step_rad","orientation_step_exceeded")):
        if step[field]>config[limit]:reasons.append(reason)
    if not torch.equal(after[:,:prefix],before[:,:prefix]):reasons.append("known_prefix_modified")
    if local_drift["joint_rms_rad"]>config["local_joint_drift_rad"]:reasons.append("local_teacher_drift")
    if (global_drift["joint_rms_rad"]>config["global_joint_drift_rad"] or
        global_drift["root_rms_m"]>config["global_root_drift_m"] or
        global_drift["orientation_rms_rad"]>config["global_orientation_drift_rad"]):reasons.append("global_teacher_drift")
    for stats in (activity,local_activity):
        if any(stats[k] is not None and stats[k]<config["minimum_activity_ratio"] for k in ("range_ratio","speed_ratio")):
            reasons.append("motion_activity_collapse")
    if activity["velocity_phase_cosine"] is not None and activity["velocity_phase_cosine"]<config["minimum_phase_cosine"]:
        reasons.append("global_teacher_drift")
    return sorted(set(reasons)),{"step":step,"local":local_drift,"global":global_drift,
        "rms_definition":"coordinate RMS over time and components; orientation uses SO(3) log vector",
        "global_activity":activity,"local_activity":local_activity,
        "global_absolute_unaligned":motion_distance(after,unaligned_global_target,mask) if unaligned_global_target is not None else None,
        "global_alignment":"fixed_recorded_reference_anchor_not_candidate_pose"}


def candidate_adamw_step(actor,optimizer,evaluate,*,config,execution_gradient_valid,gradient_components=None):
    """evaluate返回(reasons,metrics)；计数只由调用方在accepted=True之后增加。"""
    if not execution_gradient_valid:
        return {"accepted":False,"attempts":[],"rejection_reasons":["missing_finite_execution_gradient"]}
    maximum=int(config.get("maximum_attempts",3));shrink=float(config.get("lr_shrink",.25))
    if not 1<=maximum<=3 or not 0<shrink<1:raise ValueError("候选尝试最多3次且学习率严格缩小")
    model_state={n:t.detach().clone() for n,t in actor.state_dict().items()}
    opt_state=copy.deepcopy(optimizer.state_dict());rng=capture_rng()
    named=[(n,p) for n,p in actor.named_parameters() if p.requires_grad]
    gradients=[None if p.grad is None else p.grad.detach().clone() for _,p in named]
    if not any(g is not None and bool(g.abs().max()>0) for g in gradients) or any(g is not None and not torch.isfinite(g).all() for g in gradients):
        raise FloatingPointError("候选更新需要有限非零执行组合梯度")
    base_lr=[g["lr"] for g in optimizer.param_groups];attempts=[];accepted=False
    started=time.perf_counter()
    try:
        for attempt in range(maximum):
            actor.load_state_dict(model_state,strict=True);optimizer.load_state_dict(copy.deepcopy(opt_state));restore_rng(rng)
            for (_,p),g in zip(named,gradients):p.grad=None if g is None else g.clone()
            for group,lr in zip(optimizer.param_groups,base_lr):group["lr"]=lr*shrink**attempt
            t=time.perf_counter();optimizer.step()
            finite=all(torch.isfinite(v).all() for v in actor.state_dict().values())
            with torch.no_grad():
                reasons,metrics=evaluate() if finite else (["nonfinite_candidate"],{})
                deltas=[p.detach()-model_state[n] for n,p in named]
                influences={key:sum(float((g*d).sum()) for g,d in zip(gs,deltas) if g is not None)
                    for key,gs in (gradient_components or {}).items()}
                delta_l2=math.sqrt(sum(float(d.float().square().sum()) for d in deltas))
            if not math.isfinite(delta_l2) or delta_l2<=0:reasons=list(reasons)+["nonfinite_candidate"]
            attempts.append({"attempt":attempt+1,"lr_scale":shrink**attempt,"seconds":time.perf_counter()-t,
                "accepted":not reasons,"reasons":sorted(set(reasons)),"metrics":metrics,
                "parameter_delta_l2":delta_l2,"raw_gradient_dot_actual_parameter_delta":influences})
            if not reasons:
                accepted=True;break
    finally:
        if not accepted:
            actor.load_state_dict(model_state,strict=True);optimizer.load_state_dict(opt_state)
        for group,lr in zip(optimizer.param_groups,base_lr):group["lr"]=lr
        restore_rng(rng)
        optimizer.zero_grad(set_to_none=True)
    return {"accepted":accepted,"attempts":attempts,"seconds":time.perf_counter()-started,
        "rejection_reasons":[] if accepted else sorted({r for a in attempts for r in a["reasons"]}),
        "rollback_complete":not accepted,"optimizer_moments_restored_on_rejection":not accepted}
