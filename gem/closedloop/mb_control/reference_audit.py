"""仅参考保持项改变时，重算保存的真实物理方向证据与新组合导数。

固定执行奖励、SM/WM资产、调度和教师目标后，参考正则不改变任何WM/物理状态递推。
本模块复用原真实审计中逐分量执行导数和模型中心差分，重新对新local/global参考
损失求导并组合；真实侧从原Newton正负扰动逐步状态批量重算dance、track和限位。
每条物理分量差分还与原记录核对。它不是新Newton采集或重新运行所有WM导数；报告
明确说明复用范围，且需要另用完整重算路径检查数值等价，不能替代实际参数候选测试。
只支持上层action rate权重0的当前合同，非零时必须走完整路径重算真实动作。
"""

import copy
import json
from pathlib import Path
import torch


def recheck_reference_only(cfg,shards,output,device):
    from supermimic.motion import ReferenceBatch
    from .application import build_components,task_inputs
    from .checkpoint import unpack_value
    from .imagination import replace_plan_version
    from .dance_losses import (reference_objectives,execution_loss,AnchorAlignment,
        consumed_music_ticks,consumed_music_support)
    from .short_audit import perturb_plan,teacher_target_contract,phase_labels,summarize_short
    from .online_teacher import same_frozen_timeline_identity
    from .validation import combine_envelopes
    if cfg["rewards"].get("action_rate_weight")!=0.:
        raise ValueError("参考项专用重算只支持action rate=0；否则需要完整执行动作重算")
    _,adapter,_,rewards,_,_,_=build_components(cfg,device)
    before=adapter.bundle.frozen_snapshot();output=Path(output);output.mkdir(parents=True,exist_ok=False)
    weights={"dance_loss":1.,"current_reference_loss":cfg["rewards"].get("track_weight",.1),
        "action_rate_cost":0.,"joint_limit_cost":cfg["rewards"].get("joint_limit_weight",2.),
        "local_prior":cfg["training"].get("local_prior_weight",2.),"global_prior":cfg["training"].get("global_prior_weight",.2)}
    cases=[];reports=[];criteria=None;maximum_physical_parity=0.
    for directory in map(Path,shards):
        source=json.loads((directory/"short_physics_audit.json").read_text())
        recorded=json.loads((directory/"manifest.json").read_text())["config"]
        if any(cfg[k]!=recorded[k] for k in ("model","clock","rewards","critic")):
            raise ValueError("执行/采样/奖励合同改变，不能复用原执行分量导数")
        if source["frozen_sha256"]!=adapter.bundle.metadata["sha256"] or source["physics_determinism"]!=cfg["execution"]["physics_determinism"]:
            raise ValueError("原执行资产或物理确定性合同改变")
        if criteria is not None and criteria!=source["criteria"]:raise ValueError("原扰动协议不一致")
        criteria=source["criteria"];reports.append(source)
        for original in source["cases"]:
            artifact=torch.load(original["artifact"],map_location="cpu",weights_only=False)
            sample=unpack_value(artifact["sample"],adapter,device,{})
            music,initial=task_inputs(artifact["task"],adapter,device)
            fixed=rewards.teacher.timeline(music,initial)
            if not same_frozen_timeline_identity(fixed.identity,sample.state.teacher_reference.identity):
                raise ValueError("冻结音乐目标改变，需要重新运行物理和WM执行梯度")
            local=rewards.teacher.local(sample.generation.request);case=copy.deepcopy(original)
            first=sample.transitions[0].event["contacts"]
            # 原报告的汇总阶段可能来自H32后半段，不能并入H16覆盖。
            case["phases_by_horizon"]={str(h):phase_labels(sample,first,None,adapter,h)
                for h in criteria["horizons"]}
            case["phases"]=sorted(set(x for xs in case["phases_by_horizon"].values() for x in xs))
            case["phase_scope"]="real_states_and_contacts_over_each_H_window"
            for horizon in criteria["horizons"]:
                old=original["horizons"][str(horizon)]
                traces=torch.load(old["trace_artifact"],map_location="cpu",weights_only=False)
                steps=sample.transitions[:horizon];support=consumed_music_support(sample,horizon);ticks=consumed_music_ticks(sample,horizon)
                full_ticks=torch.tensor([s.end.tick for s in steps],device=device)
                target=ReferenceBatch(*(v.detach().flatten(0,1) for v in fixed.reference_at_ticks(full_ticks)))
                alignment=AnchorAlignment.at_anchor(sample.state.state,fixed.read_current_future(sample.state.tick,0)[0])
                target=alignment.reference(target)
                def priors(plan):
                    a,b,_=reference_objectives(plan,local,fixed,adapter,anchor_state=sample.state.state,
                        anchor_tick=sample.state.tick,scales=cfg["rewards"].get("dance_scales"),
                        global_support_ticks=support,global_control_ticks=ticks,
                        global_reference_records=steps,global_reference_version=sample.generation.reference_version)
                    return {"local_prior":a,"global_prior":b}
                @torch.no_grad()
                def physical(plan,states):
                    states=states.to(sample.state.state)
                    if len(states)!=horizon:raise ValueError("参考项专用重算不能补齐提前终止轨迹")
                    refs=[replace_plan_version(s.start.reference,plan,sample.generation.reference_version).read_current_future(s.end.tick,0)[0] for s in steps]
                    ref=ReferenceBatch(*(torch.cat(x) for x in zip(*refs)))
                    dance,_=execution_loss(states,target,adapter.runtime.kin,cfg["rewards"].get("dance_scales"))
                    track,_=execution_loss(states,ref,adapter.runtime.kin,cfg["rewards"].get("dance_scales"))
                    q=states[:,13:34];mdp=adapter.runtime.mdp
                    joint=(torch.relu(q-mdp.soft_hi)+torch.relu(mdp.soft_lo-q)).sum(-1)
                    discount=cfg["critic"]["gamma_per_control"]**torch.arange(horizon,device=device)*.02
                    terms={k:(x*discount).sum() for k,x in (("dance_loss",dance),("current_reference_loss",track),("joint_limit_cost",joint))}
                    terms.update(priors(plan))
                    return -sum(weights[k]*x for k,x in terms.items()),terms
                base_plan=sample.generation.plan
                real,real_parts=physical(base_plan,traces["physical"])
                repeated,repeat_parts=physical(base_plan,traces["repeat_physical"])
                model_parts=dict(old["model_loss_components"])
                model_parts.update({k:float(x.detach()) for k,x in priors(base_plan).items()})
                model=-sum(weights[k]*model_parts[k] for k in weights)
                for k in ("dance_loss","current_reference_loss","joint_limit_cost"):
                    if abs(float(real_parts[k])-old["physical_loss_components"][k])>1e-6:
                        raise ValueError("真实执行分量重算不等价")
                floor=max(criteria["minimum_resolvable_return_delta"],abs(float(real-repeated))*criteria["noise_multiplier"])
                rows=[];derivatives={}
                for old_row in old["directions"]:
                    coordinate=old_row["coordinate"];epsilon=old_row["epsilon"];probe=old_row.get("probe_label",old_row["direction"])
                    if coordinate not in derivatives:
                        amplitude=torch.zeros((),device=device,requires_grad=True)
                        parts=priors(perturb_plan(base_plan,coordinate,amplitude))
                        derivatives[coordinate]={k:float(torch.autograd.grad(-x,amplitude,retain_graph=True)[0]) for k,x in parts.items()}
                    grads=dict(old_row["component_model_derivatives"],**derivatives[coordinate])
                    negative_plan=perturb_plan(base_plan,coordinate,-epsilon);positive_plan=perturb_plan(base_plan,coordinate,epsilon)
                    trace=next(x for x in traces["perturbations"] if x["probe"]==probe and x["epsilon"]==epsilon)
                    negative,np=physical(negative_plan,trace["negative_states"]);positive,pp=physical(positive_plan,trace["positive_states"])
                    mfd=dict(old_row["component_model_centered_difference"])
                    pfd=dict(old_row["component_physical_centered_difference"])
                    for k in np:
                        measured=-float(pp[k]-np[k])/(2*epsilon)
                        if k not in ("local_prior","global_prior"):
                            error=abs(measured-pfd[k]);maximum_physical_parity=max(maximum_physical_parity,error)
                            if error>5e-5:raise ValueError("原真实分量中心差分与逐步状态重算不等价")
                        else:mfd[k]=measured
                        pfd[k]=measured
                    derivative=sum(weights[k]*grads[k] for k in weights);fd=sum(weights[k]*mfd[k] for k in weights)
                    delta=float(positive-negative);slope=delta/(2*epsilon);determinate=abs(delta)>floor
                    rows.append(dict(old_row,model_autograd=derivative,model_centered_difference=fd,
                        autograd_fd_passed=abs(derivative-fd)<=criteria["autograd_fd_atol"]+criteria["autograd_fd_rtol"]*abs(fd),
                        physical_centered_difference=slope,physical_return_delta_plus_minus=delta,noise_floor=floor,
                        status="determinate" if determinate else "indeterminate_noise_or_early_termination",
                        direction_agrees=derivative*slope>0 if determinate else None,
                        optimized_physical_delta=float((positive if derivative>0 else negative)-real),
                        component_model_derivatives=grads,component_weights=weights,
                        component_model_centered_difference=mfd,component_physical_centered_difference=pfd))
                case["horizons"][str(horizon)]=dict(old,model_return=model,physical_return=float(real),physical_repeat_return=float(repeated),
                    model_loss_components=model_parts,physical_loss_components=dict(old["physical_loss_components"],**{k:float(x) for k,x in real_parts.items()}),directions=rows)
            cases.append(case)
            with (output/"short_cases.jsonl").open("a") as f:f.write(json.dumps(case,ensure_ascii=False,allow_nan=False)+"\n")
            print(json.dumps({"event":"reference_objective_relabelled","task":case["task"],"tick":case["anchor_tick"]}),flush=True)
    adapter.bundle.assert_unchanged(before);rewards.teacher.assert_frozen()
    summary=summarize_short(cases,criteria,numerical_rule=criteria.get("numerical_rule","all_scales"))
    summary.update(schema="physical_anchor_short_bptt.v1",training_method="physical_anchor_short_bptt",cases=cases,
        frozen_sha256=adapter.bundle.metadata["sha256"],frozen_unchanged=True,forecast_enabled=True,source_hash_gate=False,
        physics_determinism=cfg["execution"]["physics_determinism"],criteria=criteria,reward_contract=cfg["rewards"],
        teacher_identity=rewards.teacher.identity,teacher_seed=cfg["teacher"]["seed"],teacher_target_contract=teacher_target_contract(cfg),
        validated_state_envelope=combine_envelopes([x["validated_state_envelope"] for x in reports]),
        coverage={"physics":"nominal_flat_newton_mujoco_warp","tasks":[t for r in reports for t in r["coverage"]["tasks"]]},
        task_identities={k:v for r in reports for k,v in r["task_identities"].items()},
        physical_evidence="new_reference_objective_on_saved_real_Newton_states",
        model_evidence="unchanged_execution_component_derivatives_reused_new_reference_derivatives_recomputed",
        maximum_real_component_slope_parity_error=maximum_physical_parity,
        source_physics_reports=[str(Path(p)/"short_physics_audit.json") for p in shards])
    (output/"short_physics_audit.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))
    (output/"manifest.json").write_text(json.dumps({"config":cfg,"recomputed":"reference_regularizers_only","source_reports":summary["source_physics_reports"]},ensure_ascii=False,indent=2))
    return summary
