"""固定音乐内容保持的逐时间戳物理评价，不用自参考跟踪高分替代舞蹈质量。

输入真实执行轨迹、当时消费参考与冻结在线教师，按关节名称比较上/下肢、根部相对
位移/yaw、速度、幅度和相位。一次性初态SE(2)对齐与绝对世界误差分别输出。提前
终止保留完整时长/失败，成对评价只另外报告共同有效区间，不把缺失尾部视作零误差。
固定延迟回放属于离线事件对照，真实墙钟异步仍保留，报告中显式区分。
"""
import numpy as np
import torch
from supermimic.rotations import quat_to_mat,yaw_mat,so3_log


def _rms(x):return float(x.float().square().mean().sqrt()) if x.numel() else None


def paired_delivery_context(before,after):
    """逐请求核对实际外生交付事件；墙钟运行只报告差异，不伪称噪声身份相同。"""
    def events(row):
        return {e["request_tick"]:(e.get("actual_ready_tick"),e.get("commit_tick"),
            e.get("rejection_tick"),e.get("accepted")) for e in row.get("events",[])}
    a,b=events(before),events(after);common=set(a)&set(b)
    mismatches=sorted(t for t in common if a[t]!=b[t])
    same_requests=set(a)==set(b)
    mode=before.get("timing_mode","wall_clock_async")
    if mode=="fixed_delay_replay" and mismatches:
        raise ValueError("固定延迟对照的共同请求实际交付事件不一致，不能作为配对回放")
    return {"timing_mode":mode,"common_requests":len(common),
        "different_delivery_request_ticks":mismatches,
        "baseline_only_request_ticks":sorted(set(a)-set(b)),
        "current_only_request_ticks":sorted(set(b)-set(a)),
        "exogenous_events_paired":mode=="fixed_delay_replay" and not mismatches,
        "same_request_noise_identity":same_requests and before.get("noise_seed")==after.get("noise_seed")
            and before.get("noise_rule")==after.get("noise_rule"),
        "missing_tail_is_not_a_success":not same_requests}


def trajectory_content(state,reference_q,teacher_qpos,adapter):
    if len(state)<2:
        return {"status":"insufficient_real_control_steps","steps":len(state)}
    target_q=teacher_qpos[:,7:][:,adapter.to_sm]
    actual_q=state[:,13:34]
    result={"generated_reference_joint_rms_rad":_rms(reference_q-target_q),
        "executed_joint_rms_rad":_rms(actual_q-target_q),
        "absolute_root_rms_m":_rms(state[:,:3]-teacher_qpos[:,:3])}
    R=quat_to_mat(state[:,3:7]);T=quat_to_mat(teacher_qpos[:,3:7])
    delta=yaw_mat(R[:1])@yaw_mat(T[:1]).transpose(-1,-2)
    target_relative=(delta@(teacher_qpos[:,:3]-teacher_qpos[:1,:3])[...,None]).squeeze(-1)
    result["relative_root_displacement_rms_m"]=_rms((state[:,:3]-state[:1,:3])-target_relative)
    result["relative_root_yaw_rms_rad"]=_rms(so3_log(yaw_mat(T[:1]@T.transpose(-1,-2))@yaw_mat(R@R[:1].transpose(-1,-2)))[:,2])
    for group,indices in (("upper",[i for i,n in enumerate(adapter.bundle.spec.joint_names) if "arm" in n or "elbow" in n]),
                          ("lower",[i for i,n in enumerate(adapter.bundle.spec.joint_names) if any(k in n for k in ("leg","knee","ankle"))])):
        a,b=actual_q[:,indices],target_q[:,indices];ref=reference_q[:,indices]
        da,db=torch.diff(a,dim=0)*50,torch.diff(b,dim=0)*50
        ar=(a.amax(0)-a.amin(0)).mean();br=(b.amax(0)-b.amin(0)).mean()
        ac=a-a.mean(0);bc=b-b.mean(0)
        correlation=float(torch.nn.functional.cosine_similarity(ac.flatten(),bc.flatten(),dim=0)) if float(bc.norm())>1e-7 else None
        result[group]={"joint_error_rms_rad":_rms(a-b),"reference_error_rms_rad":_rms(ref-b),
            "actual_speed_rad_s":_rms(da),"teacher_speed_rad_s":_rms(db),
            "actual_range_rad":float(ar),"teacher_range_rad":float(br),
            "range_ratio":float(ar/br) if br>.001 else None,
            "speed_ratio":_rms(da)/_rms(db) if _rms(db)>.001 else None,
            "motion_content_cosine":correlation,
            "velocity_phase_cosine":float(torch.nn.functional.cosine_similarity(da.flatten(),db.flatten(),dim=0)) if _rms(db)>.001 else None}
    return result


def common_window(before,after):
    paths=[r.get("trajectory_artifact") for r in (before,after)]
    common=min(before["physical_seconds"],after["physical_seconds"])
    result={"seconds":common,"baseline_missing_tail_seconds":before["music_seconds"]-before["physical_seconds"] if "music_seconds" in before else None,
        "current_missing_tail_seconds":after["music_seconds"]-after["physical_seconds"] if "music_seconds" in after else None,
        "baseline_failed":not before["complete"],"current_failed":not after["complete"],
        "timing_modes":[r.get("timing_mode","wall_clock_async") for r in (before,after)]}
    if all(paths):
        a,b=[np.load(p,allow_pickle=False) for p in paths]
        ticks,ia,ib=np.intersect1d(a["ticks"],b["ticks"],return_indices=True)
        result["paired_ticks"]=len(ticks)
        result["joint_trajectory_difference_rms_rad"]=float(np.sqrt(np.mean((a["state"][ia,13:34]-b["state"][ib,13:34])**2))) if len(ticks) else None
        if len(ticks) and all("fixed_stage1_teacher_world_qpos28" in x for x in (a,b)):
            result["music_joint_errors"]={}
            for label,data,index in (("baseline",a,ia),("current",b,ib)):
                order=[list(data["fixed_stage1_teacher_joint_names"]).index(n) for n in data["joint_names"]]
                target=data["fixed_stage1_teacher_world_qpos28"][index,7:][:,order]
                result["music_joint_errors"][label]={
                    "executed_rms_rad":float(np.sqrt(np.mean((data["state"][index,13:34]-target)**2))),
                    "reference_rms_rad":float(np.sqrt(np.mean((data["ref_q"][index]-target)**2)))}
        a.close();b.close()
    return result


@torch.no_grad()
def model_physics_gap(collector, horizons=(16,32), maximum_anchors=2):
    """同一真实接管anchor、计划和覆盖事件，对照H步WM与已保存Newton轨迹。

    教师SE(2)只在anchor计算一次。真实侧直接逐步重算同合同损失，不能拿全曲世界
    误差与模型对齐误差相减。这里不更新Actor，也不把未来真实轨迹送入条件。
    """
    from .music_repair_audit import anchors_from_episode
    from .imagination import PhysicalAnchorImagination
    from .dance_losses import AnchorAlignment
    result=[]
    for horizon in horizons:
        samples=anchors_from_episode(collector.controls,tuple(collector.generations.values()),horizon)
        selected=samples[::max(1,len(samples)//maximum_anchors)][:maximum_anchors]
        for sample in selected:
            engine=PhysicalAnchorImagination(collector.actor,collector.adapter,collector.scheduler,collector.rewards,
                horizon_steps=horizon,gamma=collector.gamma,forecast_enabled=collector.worker.forecast_enabled)
            predicted=[]
            interval,_,_=engine.rollout_anchor(sample,plan_override=sample.generation.plan,
                on_step=lambda z,_:predicted.append(z.state.detach()))
            target=sample.state.teacher_reference
            alignment=AnchorAlignment.at_anchor(sample.state.state,target.read_current_future(sample.state.tick,0)[0])
            real={k:0. for k in ("dance_loss","current_reference_loss","action_rate_cost","joint_limit_cost")}
            for i,step in enumerate(sample.transitions):
                reference=step.start.reference.read_current_future(step.end.tick,0)[0]
                _,metrics=collector.rewards(step.end.state,step.end.previous_action,step.start.previous_action,reference,
                    teacher_reference=target.read_current_future(step.end.tick,0)[0],teacher_alignment=alignment)
                for k in real:real[k]+=float(metrics[k].mean())*collector.gamma**i*.02
            actual=torch.cat([s.end.state for s in sample.transitions]);predicted=torch.cat(predicted)
            model={k:float(interval.event["objective_terms"][k].mean()) for k in real}
            result.append({"anchor_tick":sample.state.tick,"horizon":horizon,"model":model,"physical":real,
                "model_minus_physical":{k:model[k]-real[k] for k in real},
                "state55_rms":_rms(predicted-actual),"joint_rms_rad":_rms(predicted[:,13:34]-actual[:,13:34]),
                "root_rms_m":_rms(predicted[:,:3]-actual[:,:3]),"alignment":"fixed_once_at_real_anchor"})
    return result
