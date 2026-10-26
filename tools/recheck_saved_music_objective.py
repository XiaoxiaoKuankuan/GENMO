"""在保留的真实Newton扰动轨迹上重新核对音乐Actor目标的物理方向。

仅允许SM资产、执行计划、调度、执行奖励与物理设置不变而参考保持Loss改变的情形。
重新运行冻结WM及新目标求导；真实侧逐步读取先前实际Newton产生的原始状态，使用
相同冻结SM与参考重算动作和新目标。完整原始文件、扰动幅度、接触和来源报告保留，
报告明确标注为真实轨迹重算，不冒充一次新物理采集，也不允许替代真实参数候选测试。
如果执行奖励、教师轨迹或物理资产改变必须重新跑Newton入口，禁止借此重标通过。
"""
import argparse,copy,json,sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True);p.add_argument("--shards",nargs="+",required=True)
    p.add_argument("--output",required=True);p.add_argument("--device",default="cuda:0")
    p.add_argument("--reuse-execution-derivatives",action="store_true",help="仅参考正则改变：复用已验证的原执行分量导数，重算新参考导数及真实逐步损失")
    args=p.parse_args();sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from omegaconf import OmegaConf
    cfg=OmegaConf.to_container(OmegaConf.load(args.config),resolve=True);sys.path.insert(0,cfg["supermimic"]["repo"])
    if args.reuse_execution_derivatives:
        from gem.closedloop.mb_control.reference_audit import recheck_reference_only
        recheck_reference_only(cfg,args.shards,args.output,args.device)
        return
    import torch
    from gem.closedloop.mb_control.application import build_components,task_inputs
    from gem.closedloop.mb_control.checkpoint import unpack_value
    from gem.closedloop.mb_control.imagination import PhysicalAnchorImagination
    from gem.closedloop.mb_control.short_audit import perturb_plan,music_actor_objective,summarize_short,teacher_target_contract,phase_labels
    from gem.closedloop.mb_control.validation import combine_envelopes
    from gem.closedloop.mb_control.online_teacher import same_frozen_timeline_identity
    from gem.closedloop.mb_control.audit_metrics import trajectory_metrics
    actor,adapter,scheduler,rewards,_,_,_=build_components(cfg,args.device)
    frozen_before=adapter.bundle.frozen_snapshot()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    all_cases=[];reports=[];criteria=None
    for directory in map(Path,args.shards):
        source=json.loads((directory/"short_physics_audit.json").read_text());reports.append(source)
        recorded=json.loads((directory/"manifest.json").read_text())["config"]
        for key in ("clock","rewards","critic"):
            if cfg[key]!=recorded[key]:raise ValueError("执行奖励/调度改变，不能复用真实物理轨迹")
        if source["frozen_sha256"]!=adapter.bundle.metadata["sha256"] or source["physics_determinism"]!=cfg["execution"]["physics_determinism"]:
            raise ValueError("原始轨迹的SM或物理合同不同")
        if criteria is not None and criteria!=source["criteria"]:raise ValueError("原始扰动协议不一致")
        criteria=source["criteria"]
        for original_case in source["cases"]:
            artifact=torch.load(original_case["artifact"],map_location="cpu",weights_only=False)
            sample=unpack_value(artifact["sample"],adapter,args.device,{})
            music,initial=task_inputs(artifact["task"],adapter,args.device)
            frozen=rewards.teacher.timeline(music,initial)
            if not same_frozen_timeline_identity(sample.state.teacher_reference.identity,frozen.identity):
                raise ValueError("同一音乐的冻结真实消费目标改变，必须重做物理对照")
            local=rewards.teacher.local(sample.generation.request)
            case=copy.deepcopy(original_case)
            first_contact=sample.transitions[0].event["contacts"]
            case["phases_by_horizon"]={str(h):phase_labels(sample,first_contact,None,adapter,h)
                for h in criteria["horizons"]}
            case["phases"]=sorted(set(x for values in case["phases_by_horizon"].values() for x in values))
            case["phase_scope"]="real_states_and_contacts_over_each_H_window"
            for horizon in criteria["horizons"]:
                old=original_case["horizons"][str(horizon)]
                traces=torch.load(old["trace_artifact"],map_location="cpu",weights_only=False)
                engine=PhysicalAnchorImagination(actor,adapter,scheduler,rewards,horizon_steps=horizon,
                    gamma=cfg["critic"]["gamma_per_control"],forecast_enabled=True)
                def run(coordinate=0,amplitude=0.,real_states=None):
                    candidate=perturb_plan(sample.generation.plan,coordinate,amplitude)
                    states=[];cursor=0
                    def recorded_transition(action):
                        nonlocal cursor
                        if cursor>=len(real_states):raise ValueError("真实物理轨迹提前耗尽")
                        state=real_states[cursor:cursor+1].to(action);cursor+=1;return state
                    with torch.set_grad_enabled(real_states is None and torch.is_tensor(amplitude) and amplitude.requires_grad):
                        interval,_,_=engine.rollout_anchor(sample,plan_override=candidate,
                            transition_fn=recorded_transition if real_states is not None else None,
                            on_step=lambda z,_:states.append(z.state.detach().cpu()))
                        value,terms,weights=music_actor_objective(interval,candidate,local,sample,adapter,cfg,horizon)
                    return value,terms,weights,torch.cat(states),interval.steps
                model,model_parts,_,mt,_=run()
                real,real_parts,_,rt,rs=run(real_states=traces["physical"])
                repeat,_,_,rr,repeated_steps=run(real_states=traces["repeat_physical"])
                floor=max(criteria["minimum_resolvable_return_delta"],abs(float(real-repeat))*criteria["noise_multiplier"])
                rows=[]
                for old_row in old["directions"]:
                    probe=old_row.get("probe_label",old_row["direction"]);coordinate=old_row["coordinate"];eps=old_row["epsilon"]
                    data=next(t for t in traces["perturbations"] if t["probe"]==probe and t["epsilon"]==eps)
                    amplitude=torch.zeros((),device=args.device,requires_grad=True)
                    value,parts,weights,_,_=run(coordinate,amplitude)
                    derivative=float(torch.autograd.grad(value,amplitude,retain_graph=True)[0])
                    derivatives={k:float(torch.autograd.grad(-parts[k].mean(),amplitude,retain_graph=True)[0]) for k in weights}
                    minus,mp,_,_,_=run(coordinate,-eps);plus,pp,_,_,_=run(coordinate,eps)
                    negative,np,_,_,ns=run(coordinate,-eps,data["negative_states"])
                    positive,sp,_,_,ps=run(coordinate,eps,data["positive_states"])
                    fd=float((plus-minus)/(2*eps));delta=float(positive-negative);slope=delta/(2*eps)
                    determinate=abs(delta)>floor and rs==ns==ps==horizon
                    row=dict(old_row,model_autograd=derivative,model_centered_difference=fd,
                        autograd_fd_passed=abs(derivative-fd)<=criteria["autograd_fd_atol"]+criteria["autograd_fd_rtol"]*abs(fd),
                        physical_centered_difference=slope,physical_return_delta_plus_minus=delta,noise_floor=floor,
                        status="determinate" if determinate else "indeterminate_noise_or_early_termination",
                        direction_agrees=derivative*slope>0 if determinate else None,
                        optimized_physical_delta=float((positive if derivative>0 else negative)-real),
                        component_model_derivatives=derivatives,component_weights=weights,
                        component_model_centered_difference={k:-float((pp[k]-mp[k]).mean())/(2*eps) for k in weights},
                        component_physical_centered_difference={k:-float((sp[k]-np[k]).mean())/(2*eps) for k in weights})
                    rows.append(row)
                stats=trajectory_metrics(mt,rt,adapter.runtime.kin,limits=criteria["state_rmse_limits"],start_tick=sample.state.tick)
                case["horizons"][str(horizon)]=dict(old,statistics=stats,model_return=float(model),physical_return=float(real),
                    physical_repeat_return=float(repeat),model_loss_components={k:float(v.mean()) for k,v in model_parts.items()},
                    physical_loss_components={k:float(v.mean()) for k,v in real_parts.items()},directions=rows)
            all_cases.append(case)
            with (output/"short_cases.jsonl").open("a") as f:f.write(json.dumps(case,ensure_ascii=False,allow_nan=False)+"\n")
            print(json.dumps({"event":"saved_newton_objective_rechecked","task":case["task"],"tick":case["anchor_tick"]}),flush=True)
    summary=summarize_short(all_cases,criteria,numerical_rule=criteria.get("numerical_rule","all_scales"))
    adapter.bundle.assert_unchanged(frozen_before);rewards.teacher.assert_frozen()
    summary.update(schema="physical_anchor_short_bptt.v1",training_method="physical_anchor_short_bptt",cases=all_cases,
        frozen_sha256=adapter.bundle.metadata["sha256"],frozen_unchanged=True,forecast_enabled=True,source_hash_gate=False,
        physics_determinism=cfg["execution"]["physics_determinism"],reward_contract=cfg["rewards"],criteria=criteria,
        teacher_identity=rewards.teacher.identity,teacher_seed=cfg["teacher"]["seed"],teacher_target_contract=teacher_target_contract(cfg),
        validated_state_envelope=combine_envelopes([r["validated_state_envelope"] for r in reports]),
        coverage={"physics":"nominal_flat_newton_mujoco_warp","tasks":[t for r in reports for t in r["coverage"]["tasks"]]},
        task_identities={k:v for r in reports for k,v in r["task_identities"].items()},
        physical_evidence="recomputed_objective_on_preserved_real_newton_perturbations_not_new_simulation",
        source_physics_reports=[str(Path(p)/"short_physics_audit.json") for p in args.shards])
    (output/"short_physics_audit.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=="__main__":main()
