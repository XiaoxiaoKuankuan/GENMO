"""动作坍缩的同请求反事实审计，保留生成参考、模型轨迹及真实Newton轨迹。

先用原Stage1真实滚动收集请求及接管前史。各检查点随后复用完全相同的音乐、真实
历史、前缀、预测上下文和DDIM初始噪声，只有被比较的网络参数/未来分支不同。
物理分支从episode初态按同一已记录参考闭环重放，核验接管状态，再执行H=32步。
之后的参考接管固定为原行为，用于单请求因果比较，不冒充各策略完整闭环训练收益。
输出未经重锚、平滑或删帧的参考及状态，低跟踪误差不能替代动作保持判断。
"""

from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import torch

from .application import seed_all, task_inputs
from .checkpoint import pack_value
from .evaluation import motion_metrics
from .imagination import PhysicalAnchorImagination


def audit_collapse(collector, tasks, cfg, output, device):
    output=Path(output);settings=cfg["collapse_audit"]
    original={n:v.detach().cpu().clone() for n,v in collector.actor.state_dict().items()}
    variants=[("stage1",original,False)]
    for label,path in settings["checkpoints"].items():
        ck=torch.load(path,map_location="cpu",weights_only=False,mmap=True)
        for future in (False,True):variants.append((f"{label}_{'future_on' if future else 'future_off'}",ck["actor"],future))
    collector.anchor_horizon=32;collector.record_controls=True;collector.record_contacts=True
    rows=[]
    for task in tasks:
        collector.actor.load_state_dict(original,strict=True)
        seed=task.get("seeds",[42])[0];seed_all(seed)
        music,initial=task_inputs(task,collector.adapter,device);collector.reset(music,initial)
        controls=[];samples=[]
        while not collector.terminal and collector.state.tick<settings.get("collection_seconds",16)*600:
            collector.collect(1);controls.extend(collector.take_controls())
            samples.extend(collector.ready_anchors);collector.ready_anchors.clear()
        if not samples:raise RuntimeError("坍缩复现没有真实完整32步接管样本")
        by_tick={x.start.tick:x for x in controls}
        indices=np.unique(np.linspace(0,len(samples)-1,min(settings.get("anchors_per_task",3),len(samples)),dtype=int))
        def restore(tick):
            collector.env.reset(torch.arange(1,device=initial.device),initial)
            z=controls[0].start.detached()
            with torch.no_grad():
                for t in range(0,tick,12):
                    z=replace(z,reference=by_tick[t].start.reference)
                    z=collector.adapter.refresh_current_observation(z)
                    _,_,goal=z.reference.read_current_future(t)
                    action=collector.adapter.bundle.act(z.sm_history,goal)
                    z=collector.adapter.advance(z,collector.env.step_physics(action),action,origin="real")
            error=float((z.state-by_tick[tick].start.state).abs().max())
            if error>1e-4:raise RuntimeError(f"同请求物理起点恢复误差{error}，不能当作配对证据")
            return error
        for index in indices:
            sample=samples[index];request=sample.generation.request;p=request.ticket.prefix_frames
            key=f"{task['id'].replace('/','_')}_request{request.ticket.request_tick}"
            torch.save({"sample":pack_value(sample),"initial":initial.cpu(),"task":task,
                "real_prefix":pack_value(tuple(controls[:sample.state.tick//12]))},output/f"{key}_request.pt")
            for label,weights,future in variants:
                collector.actor.load_state_dict(weights,strict=True)
                context=sample.generation.forecast
                if not future:context=replace(context,valid=torch.zeros_like(context.valid))
                with torch.no_grad():plan=collector.actor.generate(request.snapshot,request.ticket,context,request.noise)
                engine=PhysicalAnchorImagination(collector.actor,collector.adapter,collector.scheduler,collector.rewards,
                    horizon_steps=32,gamma=collector.gamma)
                traces={};returns={};restore_error=0.
                for physical in (False,True):
                    if physical:restore_error=restore(sample.state.tick)
                    trace=[]
                    with torch.no_grad():
                        interval,_,_=engine.rollout_anchor(sample,plan_override=plan,
                            transition_fn=collector.env.step_physics if physical else None,
                            on_step=lambda z,r:trace.append((z.tick,z.state.detach().cpu())))
                    name="physical" if physical else "model"
                    traces[name]=trace;returns[name]=float(interval.reward.mean())
                q=plan.world_qpos[:,p:,7:].detach().cpu();speed=torch.diff(q,dim=1)*30
                row={"task":task["id"],"source":task.get("source"),"variant":label,
                    "request_tick":request.ticket.request_tick,"anchor_tick":sample.state.tick,
                    "prefix_frames":p,"reference_joint_range_rad":float((q.amax(1)-q.amin(1)).mean()),
                    "reference_joint_speed_rms_rad_s":float(speed.square().mean().sqrt()),
                    "physical_motion":motion_metrics(traces["physical"],music),
                    "model_reward":returns["model"],"physical_reward":returns["physical"],
                    "reset_state_max_abs":restore_error,"physical_control_steps":len(traces["physical"]),
                    "scope":"same_request_single_intervention_with_recorded_continuation"}
                arrays={"generated_world_qpos28":plan.world_qpos.cpu().numpy(),"generated_contact":plan.contact.cpu().numpy(),
                    "ddim_noise":request.noise.cpu().numpy(),"prefix_world_qpos28":request.snapshot.reference.source_at(
                        request.ticket.request_tick+torch.arange(p,device=device)*20).cpu().numpy()}
                for name,trace in traces.items():
                    arrays[f"{name}_ticks"]=np.asarray([x[0] for x in trace]);states=torch.cat([x[1] for x in trace])
                    arrays[f"{name}_state55"]=states.numpy()
                    from supermimic.mdp import unflatten_state
                    with torch.no_grad():body=collector.adapter.runtime.kin.forward(*unflatten_state(states.to(device)),all_bodies=True)[0]
                    arrays[f"{name}_body_position"]=body.cpu().numpy()
                np.savez_compressed(output/f"{key}_{label}.npz",**arrays)
                rows.append(row)
                with (output/"paired_requests.jsonl").open("a") as stream:stream.write(json.dumps(row,ensure_ascii=False)+"\n")
                print(json.dumps({"event":"collapse_same_request","task":task["id"],"variant":label,
                    "request_tick":request.ticket.request_tick,"reference_speed":row["reference_joint_speed_rms_rad_s"]}),flush=True)
    collector.actor.load_state_dict(original,strict=True)
    (output/"paired_request_summary.json").write_text(json.dumps({"rows":rows,"same_conditions_and_noise":True,
        "physical_backend":"isaaclab3_newton_mujoco_warp","not_full_policy_rollout":True},ensure_ascii=False,indent=2))
