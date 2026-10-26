"""真实在线教师、候选信任域和Newton候选执行的联合审计。

输入独立冻结Stage1在线采集文件，恢复原始请求、噪声和真实首次接管anchor。先检查
固定噪声重生成与独立教师一致性，运行真实GENMO全20步梯度及AdamW候选回滚，再从
同一真实物理前史重放候选前后参考。保留完整请求、参考、模型/真实轨迹和接触。
这里的有限诊断更新不会自动发布或替代正式训练，单例成功不代表音乐质量提升。
"""
from dataclasses import replace
import copy
import json
from pathlib import Path
import time
import torch
from .contracts import PhysicalAnchor
from .checkpoint import unpack_value,pack_value,capture_rng,restore_rng
from .application import task_inputs,SupervisionStream,build_stage1_losses
from .trainer import Stage2Trainer
from .online_teacher import online_base_identity,identity_digest
from .dance_losses import AnchorAlignment
from .output_trust_region import motion_distance
from .short_audit import phase_labels,teacher_target_contract


def attach_teacher(z,teacher):
    pending=z.pending_request
    if pending is not None:pending=replace(pending,snapshot=attach_teacher(pending.snapshot,teacher))
    return replace(z,teacher_reference=teacher,pending_request=pending)


def online_episode(cfg,engine,task,device,*,allow_partial_for_real_mc=False):
    music,initial=task_inputs(task,engine.adapter,device)
    if allow_partial_for_real_mc:
        from .online_teacher import load_online_timeline
        target=load_online_timeline(engine.teacher,music,initial,allow_partial_for_real_mc=True)
    else:target=engine.teacher.timeline(music,initial)
    key=identity_digest(online_base_identity(engine.teacher,music,initial))
    item=json.loads(Path(cfg["teacher"]["online_manifest"]).read_text())["timelines"][key]
    payload=torch.load(item["path"],map_location="cpu",weights_only=False)
    memo={};controls=unpack_value(payload["controls"],engine.adapter,device,memo)
    if allow_partial_for_real_mc and not controls[-1].terminal:
        from .returns import close_real_mc_episode
        # 旧构建文件的零控制步真实终止未写回最后一个控制步；只连接已记录的终点。
        if controls[-1].end.tick!=int(payload["ticks"][-1]):raise ValueError("真实终止时刻与最后控制步不一致")
        event=(unpack_value(payload["terminal_event"],engine.adapter,device,memo) if "terminal_event" in payload else
            replace(controls[-1],start=controls[-1].end,steps=0,terminal=True,reason=payload["reason"]))
        completed=list(controls);close_real_mc_episode(completed,event);controls=tuple(completed)
    controls=tuple(replace(x,start=attach_teacher(x.start,target),end=attach_teacher(x.end,target)) for x in controls)
    generations=unpack_value(payload["generations"],engine.adapter,device,memo)
    generations=tuple(replace(g,request=replace(g.request,snapshot=attach_teacher(g.request.snapshot,target))) for g in generations)
    return payload,initial,controls,generations,target


def anchors_from_episode(controls,generations,horizon):
    by_version={g.reference_version:g for g in generations if g.accepted};seen=set();samples=[]
    for i,step in enumerate(controls):
        ref=step.start.reference;support=ref.dependency_support(step.start.tick)
        versions=set(ref.source_versions[(support-ref.lower)//20].tolist())
        for version in versions-seen:
            if version not in by_version or i+horizon>len(controls):continue
            window=controls[i:i+horizon]
            consumed=set()
            for x in window:
                cache=x.start.reference;s=cache.dependency_support(x.start.tick)
                consumed.update(s[cache.source_versions[(s-cache.lower)//20]==version].tolist())
            samples.append(PhysicalAnchor(by_version[version],step.start,tuple(window),tuple(sorted(consumed)),tuple(sorted(seen))))
        seen.update(versions)
    return samples


def restore_physical_prefix(collector,controls,initial,tick,tolerance=1e-4):
    collector.env.reset(torch.arange(1,device=initial.device),initial)
    z=controls[0].start.detached()
    with torch.no_grad():
        for step in controls:
            if step.start.tick>=tick:break
            z=replace(z,reference=step.start.reference)
            z=collector.adapter.refresh_current_observation(z)
            _,_,goal=z.reference.read_current_future(z.tick)
            action=collector.adapter.bundle.act(z.sm_history,goal)
            state=collector.env.step_physics(action)
            z=collector.adapter.advance(z,state,action,origin="real")
    expected=next(x.start.state for x in controls if x.start.tick==tick)
    error=float((z.state-expected).abs().max())
    if error>tolerance:raise RuntimeError(f"真实前史恢复误差{error}，候选比较不配对")
    return error


def run_music_repair_audit(collector,actor,critic,engine,original,cfg,tasks,output,device):
    output=Path(output)
    losses=build_stage1_losses(actor.stage1,original).to(device)
    trainer=Stage2Trainer(actor,critic,engine,losses,cfg["training"])
    supervised=SupervisionStream(cfg["stage1"]["data_config"],1,cfg["training"]["seed"],device)
    initial_actor=copy.deepcopy(actor.state_dict());initial_opt=copy.deepcopy(trainer.actor_optimizer.state_dict())
    reports=[];calibration=[]
    for task in tasks:
        payload,initial,controls,generations,target=online_episode(cfg,engine,task,device)
        samples=anchors_from_episode(controls,generations,engine.horizon_steps)
        limit=cfg.get("repair_audit",{}).get("anchors_per_task",2)
        selected=samples[::max(1,len(samples)//limit)][:limit]
        for sample in selected:
            actor.load_state_dict(initial_actor);trainer.actor_optimizer.load_state_dict(copy.deepcopy(initial_opt))
            trainer.accepted_actor_updates=0;trainer.invalid_actor_attempts=0
            request=sample.generation.request;p=request.ticket.prefix_frames
            with torch.no_grad():
                before=actor.generate(request.snapshot,request.ticket,sample.generation.forecast,request.noise)
                repeated=actor.generate(request.snapshot,request.ticket,sample.generation.forecast,request.noise)
                local=engine.teacher.local(request)
                alternative=engine.teacher.generate(request.snapshot,request.ticket,-request.noise)
            mask=torch.arange(60,device=device)>=p
            calibration.append({"task":task["id"],"request_tick":request.ticket.request_tick,
                "same_noise_max_abs":float((before.world_qpos-repeated.world_qpos).abs().max()),
                "stage1_parity_max_abs":float((before.world_qpos-local.world_qpos).abs().max()),
                "different_noise_motion_distance":motion_distance(before.world_qpos,alternative.world_qpos,mask),
                "different_noise_is_not_single_update_allowance":True})
            prefix_error=restore_physical_prefix(collector,controls,initial,sample.state.tick)
            physical_before=[];contacts_before=[]
            def record_before(z,reward):
                physical_before.append(z.state.detach().cpu());contacts_before.append(collector.env.contact_events())
            with torch.no_grad():
                real_before,_,_=engine.rollout_anchor(sample,plan_override=before,transition_fn=collector.env.step_physics,on_step=record_before)
            row=trainer.update_anchors([sample],supervised.next(),physical_intervals=list(sample.transitions))
            with torch.no_grad():after=actor.generate(request.snapshot,request.ticket,sample.generation.forecast,request.noise)
            restore_physical_prefix(collector,controls,initial,sample.state.tick)
            physical_after=[];contacts_after=[]
            def record_after(z,reward):
                physical_after.append(z.state.detach().cpu());contacts_after.append(collector.env.contact_events())
            with torch.no_grad():
                real_after,_,_=engine.rollout_anchor(sample,plan_override=after,transition_fn=collector.env.step_physics,on_step=record_after)
            a=torch.cat(physical_before);b=torch.cat(physical_after)
            row.update(task=task["id"],anchor_tick=sample.state.tick,prefix_restore_max_abs=prefix_error,
                physical_steps_before=real_before.steps,physical_steps_after=real_after.steps,
                physical_complete_horizon=real_before.steps==real_after.steps==engine.horizon_steps,
                physical_reason_before=real_before.reason,physical_reason_after=real_after.reason,
                physical_joint_limits_before={k:float(v.mean()) for k,v in real_before.event["objective_terms"].items()
                    if k.startswith(("joint_soft_","joint_hard_","joint_near_"))},
                physical_joint_limits_after={k:float(v.mean()) for k,v in real_after.event["objective_terms"].items()
                    if k.startswith(("joint_soft_","joint_hard_","joint_near_"))},
                physical_dance_before=float(real_before.event["objective_terms"]["dance_loss"].mean()),
                physical_dance_after=float(real_after.event["objective_terms"]["dance_loss"].mean()),
                physical_speed_before=float(a[:,34:55].square().mean().sqrt()),physical_speed_after=float(b[:,34:55].square().mean().sqrt()),
                physical_joint_range_before=float((a[:,13:34].amax(0)-a[:,13:34].amin(0)).mean()),
                physical_joint_range_after=float((b[:,13:34].amax(0)-b[:,13:34].amin(0)).mean()),
                contact_events_changed=contacts_before!=contacts_after)
            path=output/f"candidate_{len(reports):03d}.pt"
            torch.save({"sample":pack_value(sample),"before_reference":before.world_qpos.cpu(),"after_reference":after.world_qpos.cpu(),
                "physical_before":a,"physical_after":b,"contacts_before":contacts_before,"contacts_after":contacts_after,
                "actor_after":actor.state_dict(),"report":row},path)
            if row["accepted_execution_feedback_update"]:
                torch.save({"schema":"genmo.stage2.inference_export.v1","actor":actor.state_dict(),
                    "frozen_sha256":engine.adapter.bundle.metadata["sha256"],"model_contract":cfg["model"],
                    "accepted_actor_updates":trainer.accepted_actor_updates,
                    "scope":"independent_single_candidate_not_formal_or_resumable",
                    "teacher_target_contract":teacher_target_contract(cfg),"source_candidate":str(path)},
                    output/f"candidate_{len(reports):03d}_inference.pt")
            reports.append(row)
            with (output/"candidate_reports.jsonl").open("a") as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
            print(json.dumps({"event":"music_repair_candidate","task":task["id"],"accepted":row["accepted_execution_feedback_update"],
                "before":row["physical_dance_before"],"after":row["physical_dance_after"]}),flush=True)
    actor.load_state_dict(initial_actor);engine.teacher.assert_frozen()
    summary={"schema":"music_repair_real_candidates.v1","horizon":engine.horizon_steps,"calibration":calibration,
        "candidates":reports,"teacher_target_contract":teacher_target_contract(cfg),"reward_contract":cfg["rewards"],
        "no_formal_training_started":True,"passed":bool(reports) and all(c["same_noise_max_abs"]<=1e-6 and c["stage1_parity_max_abs"]<=1e-5 for c in calibration),
        "physical_quality_improvement_claim":False}
    (output/"music_repair_audit.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))
    return summary
