"""冻结音乐教师目标下的GENMO更新与逐目标梯度审计。

输入为真实首次接管anchor、原Stage1配对监督及同环境真实控制序列。执行目标由
固定音乐教师、学生当前参考辅助项和原物理成本构成；局部教师在完整20步DDIM后
约束物理轨迹，覆盖FutureEncoder及Stage1主干。逐项累计真实参数梯度，再报告范数
和方向余弦后统一裁剪、更新。教师/pi/wm不创建优化器，不把非零梯度或模型回报
宣称为物理改进。终点价值默认权重为零，仅在显式配置且新奖励Critic验证通过后使用。
"""

from collections import Counter
import math
import time
import torch

from .dance_losses import reference_prior
from .execution_audit import state_probe
from .sample_distribution import describe_sample
from .trainer import frozen_parameters
from .validation import UnsupportedModelState


def _norm(gs):
    values = [g.float().square().sum() for g in gs if g is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.


def _cosine(a, b, norm_a, norm_b):
    dot = sum(float((x.float()*y.float()).sum()) for x,y in zip(a,b) if x is not None and y is not None)
    return dot/max(1e-20, norm_a*norm_b) if norm_a and norm_b else None


def _legacy_update_teacher_anchors(trainer, samples, supervised_batch, physical_intervals, physical_interval_groups):
    started = time.perf_counter()
    actor, engine, config = trainer.actor, trainer.imagination, trainer.config
    teacher = engine.teacher
    if teacher is None or not physical_intervals or not samples:
        raise ValueError("教师训练需要真实接管样本、真实控制序列和独立冻结Stage1")
    value_weight = float(config.get("terminal_value_weight", 0.))
    if value_weight and not trainer.critic_ready:
        raise ValueError("终点价值必须来自新奖励已验证Critic")
    actor.eval();trainer.critic.eval()
    trainer.actor_optimizer.zero_grad(set_to_none=True)
    trainer.critic_optimizer.zero_grad(set_to_none=True)
    params = [p for p in actor.parameters() if p.requires_grad]
    names = [n for n,p in actor.named_parameters() if p.requires_grad]
    groups = physical_interval_groups if physical_interval_groups is not None else [physical_intervals]
    targets = [pair for group in groups if group for pair in trainer._targets(group)]
    targets = targets[::max(1,len(targets)//16)]
    components = ("dance", "track", "prior", "physical", "terminal_value", "bc")
    gradients = {name: [None]*len(params) for name in components}
    sums = {name: 0. for name in components}
    raw_sums, sample_rows, rejected, execution_norms = {}, [], [], []
    comparisons, valid = [], 0
    for sample in samples:
        description = describe_sample(sample, engine.adapter)
        if sample.generation.request.actor_version != trainer.accepted_actor_updates:
            rejected.append("stale_actor_request")
            sample_rows.append(dict(description, accepted=False, rejection="stale_actor_request"))
            continue
        try:
            local = teacher.local(sample.generation.request)
            with frozen_parameters(trainer.critic):
                interval, plan, noise = engine.rollout_anchor(sample)
                terms = interval.event["objective_terms"]
                prior, prior_parts = reference_prior(plan, local, engine.adapter,
                    scales=engine.rewards.config.get("dance_scales"), global_reference=sample.state.teacher_reference)
                probe = torch.autograd.grad(state_probe(interval.end.state), noise, retain_graph=True, allow_unused=True)[0]
                if probe is None or not torch.isfinite(probe).all() or not float(probe.norm()) > 0:
                    raise FloatingPointError("教师目标真实执行路径断开或非有限")
                execution_norms.append(float(probe.detach().norm()))
                tail = (interval.continuation*trainer.critic(interval.end)).mean() if value_weight and not interval.terminal else prior.new_zeros(())
                losses = {"dance": terms["dance_loss"].mean(),
                    "track": engine.rewards.config.get("track_weight", .1)*terms["current_reference_loss"].mean(),
                    "prior": config.get("prior_weight", 2.)*prior,
                    "physical": terms["physical_cost"].mean(), "terminal_value": -value_weight*tail}
                for name, loss in losses.items():
                    sums[name] += float(loss.detach())
                    if not loss.requires_grad:
                        continue
                    gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                    for i,g in enumerate(gs):
                        if g is not None:
                            if not torch.isfinite(g).all(): raise FloatingPointError(f"{name}梯度非有限")
                            if gradients[name][i] is None: gradients[name][i] = g.detach()
                            else: gradients[name][i].add_(g.detach())
                    del gs
                for name, value in {**terms, "local_reference_prior":prior,
                                    "fixed_timeline_joint_deviation_rms":prior_parts["fixed_timeline_joint_deviation_rms"]}.items():
                    raw_sums[name] = raw_sums.get(name, 0.)+float(value.detach().mean())
                comparisons.append((sample.generation, plan.world_qpos.detach(), local, sample.state.teacher_reference))
                valid += 1
                sample_rows.append(dict(description, accepted=True, rejection=None))
            del interval, plan, noise, prior, terms, losses, probe, tail
        except UnsupportedModelState as error:
            rejected.append(str(error))
            sample_rows.append(dict(description, accepted=False, rejection="outside_validated_world_model_support"))
    for name in components:
        if valid:
            sums[name] /= valid
            gradients[name] = [g.div_(valid) if g is not None else None for g in gradients[name]]
    raw_sums = {k:v/max(1,valid) for k,v in raw_sums.items()}
    norms = {name:_norm(gs) for name,gs in gradients.items()}
    accepted = valid > 0 and norms["dance"] > 0
    actor_norm = delta = bc_value = 0.
    output_changes, cumulative, future_bc_norm = [], [], 0.
    if accepted:
        out = actor.original_supervision(supervised_batch)
        bc, _ = trainer.stage1_losses(supervised_batch, out["pred_x_start"], out["static_conf_logits"],
                                      global_step=config.get("stage1_supervision_step",595000))
        bc_value = float(bc.detach());sums["bc"] = config["bc_weight"]*bc_value
        gradients["bc"] = [g.detach() if g is not None else None for g in torch.autograd.grad(
            config["bc_weight"]*bc, params, allow_unused=True)]
        norms["bc"] = _norm(gradients["bc"])
        future_bc_norm = _norm([g for n,g in zip(names,gradients["bc"]) if n.startswith("future_encoder.")])
    cosines = {f"{a}__{b}":_cosine(gradients[a],gradients[b],norms[a],norms[b]) for i,a in enumerate(components) for b in components[i+1:]}
    future_norms = {name:_norm([g for n,g in zip(names,gs) if n.startswith("future_encoder.")]) for name,gs in gradients.items()}
    model_squared = 0.
    for i in range(len(params)):
        values = [gradients[name][i] for name in components if name != "bc" and gradients[name][i] is not None]
        if values:
            combined = sum(values[1:], values[0].clone())
            model_squared += float(combined.float().square().sum())
    if accepted:
        before = [p.detach().clone() for p in params]
        for i,p in enumerate(params):
            values = [gradients[name][i] for name in components if gradients[name][i] is not None]
            p.grad = sum(values[1:], values[0].clone()) if values else None
        actor_norm = float(torch.nn.utils.clip_grad_norm_(params,config.get("grad_clip",1.),error_if_nonfinite=True))
        trainer.actor_optimizer.step()
        delta = float(torch.stack([(p.detach()-old).float().square().sum() for p,old in zip(params,before)]).sum().sqrt())
        if not math.isfinite(delta) or delta <= 0: raise FloatingPointError("教师Actor没有有效参数更新")
        del before
        trainer.actor_optimizer.zero_grad(set_to_none=True)
        trainer.accepted_actor_updates += 1
        with torch.no_grad():
            for record,old,local,fixed in comparisons:
                r=record.request;p=r.ticket.prefix_frames
                after=actor.generate(r.snapshot,r.ticket,record.forecast,r.noise).world_qpos
                output_changes.append(float((after[:,p:,7:]-old[:,p:,7:]).square().mean().sqrt()))
                cumulative.append(float((after[:,p:,7:]-local.world_qpos[:,p:,7:]).square().mean().sqrt()))
    else:
        trainer.invalid_actor_attempts += 1
    del gradients
    critic_loss,critic_norm=trainer.fit_critic(targets)
    engine.adapter.bundle.assert_unchanged(trainer.frozen_identity);teacher.assert_frozen()
    trainer.update_index += 1
    counts=Counter((r["accepted"],r["source"],r["speed_bin"],r["amplitude_bin"],r["contact"],r["motion_class"]) for r in sample_rows)
    return {"update":trainer.update_index,"training_method":"frozen_stage1_music_teacher_short_bptt",
        "accepted_actor_updates":trainer.accepted_actor_updates,"accepted_execution_feedback_update":accepted,
        "original_music_execution_loss":raw_sums.get("dance_loss"),
        "current_reference_tracking_loss":raw_sums.get("current_reference_loss"),
        "reference_preservation_loss":raw_sums.get("local_reference_prior"),"original_stage1_loss":bc_value,
        "loss_components_weighted":sums,"gradient_norms_weighted":norms,"gradient_cosines":cosines,
        "future_encoder_gradient_norms":future_norms,"future_encoder_stage1_bc_gradient":future_bc_norm,
        "physical_cost":raw_sums.get("physical_cost"),"terminal_value_weight":value_weight,
        "model_return":-(sums["dance"]+sums["track"]+sums["physical"]),
        "model_gradient_norm_before_bc":math.sqrt(model_squared),
        "weighted_bc_gradient_norm":norms["bc"],"actor_grad_norm":actor_norm,
        "actor_parameter_delta_l2":delta,"fixed_noise_reference_joint_change_rms_rad":sum(output_changes)/len(output_changes) if output_changes else None,
        "cumulative_stage1_joint_deviation_rms_rad":sum(cumulative)/len(cumulative) if cumulative else None,
        "fixed_timeline_joint_deviation_rms_rad":raw_sums.get("fixed_timeline_joint_deviation_rms"),
        "valid_execution_rollouts":valid,"execution_state_noise_gradient_norms":execution_norms,
        "horizon_steps":engine.horizon_steps,"rejection_reasons":rejected,"sample_distribution":sample_rows,
        "sample_distribution_counts":[{"accepted":k[0],"source":k[1],"speed":k[2],"amplitude":k[3],"contact":k[4],"motion_class":k[5],"count":v} for k,v in counts.items()],
        "critic_mse":critic_loss,"critic_grad_norm":critic_norm,"critic_updates":trainer.critic_updates,
        "physical_reward_dt_sum":sum(float(x.reward.mean()) for x in physical_intervals),
        "frozen_parameters_and_buffers_unchanged":True,"frozen_teacher_unchanged":True,
        "bc_only_update":False,"update_wall_seconds":time.perf_counter()-started,
        "evidence":"teacher_model_gradients_not_physical_quality"}


def parameter_module(name):
    if name.startswith("future_encoder."): return "future_encoder"
    if any(x in name for x in ("history", "prefix", "condition_adapter", "proprio", "known_")): return "history_prefix"
    if any(x in name for x in ("output", "final_layer", "linear_out", "motion_out", "static_conf")): return "diffusion_output"
    return "stage1_backbone"


def update_teacher_anchors(trainer,samples,supervised_batch,physical_intervals,physical_interval_groups=None):
    if not trainer.config.get("output_trust_region",{}).get("enabled",False):
        return _legacy_update_teacher_anchors(trainer,samples,supervised_batch,physical_intervals,physical_interval_groups)
    return guarded_teacher_update(trainer,samples,supervised_batch,physical_intervals,physical_interval_groups)


def guarded_teacher_update(trainer,samples,supervised_batch,physical_intervals,physical_interval_groups):
    """新合同：逐分量原始梯度、独立双教师约束、完整AdamW候选验收；旧实验保持原入口。"""
    from .dance_losses import reference_objectives,AnchorAlignment,consumed_music_support,consumed_music_ticks
    from .output_trust_region import candidate_adamw_step,check_outputs
    from .checkpoint import capture_rng,restore_rng
    from .sample_distribution import coverage_summary
    started=time.perf_counter();actor=trainer.actor;engine=trainer.imagination;config=trainer.config
    trust=config["output_trust_region"];teacher=engine.teacher
    if not samples or not physical_intervals or teacher is None:raise ValueError("缺少真实执行反馈与独立教师")
    if config.get("terminal_value_weight",0.)!=0:raise ValueError("本轮新目标Critic尚未通过真实持续改善及终点价值消融，lambdaV必须为0")
    if not config.get("defer_critic",False):
        raise ValueError("新在线音乐合同只允许独立完整MC拟合Critic；旧区间TD没有固定anchor目标上下文，不能复用")
    actor.eval();trainer.critic.eval();trainer.actor_optimizer.zero_grad(set_to_none=True)
    named=[(n,p) for n,p in actor.named_parameters() if p.requires_grad];names=[n for n,_ in named];params=[p for _,p in named]
    keys=("dance","track","local","global","bc","action_rate","joint_limit","terminal_value")
    weights={"dance":1.,"track":engine.rewards.config.get("track_weight",.1),
        "local":config.get("local_prior_weight",2.),"global":config.get("global_prior_weight",.2),
        "bc":config["bc_weight"],"action_rate":engine.rewards.config.get("action_rate_weight",0.),
        "joint_limit":engine.rewards.config.get("joint_limit_weight",2.),"terminal_value":0.}
    gradients={k:[None]*len(params) for k in keys};losses_before={k:0. for k in keys}
    fixed=[];rows=[];rejected=[];execution_norms=[];diagnostics={}

    def losses(sample,interval,plan,local):
        terms=interval.event["objective_terms"]
        lp,gp,parts=reference_objectives(plan,local,sample.state.teacher_reference,engine.adapter,
            anchor_state=sample.state.state,anchor_tick=sample.state.tick,scales=engine.rewards.config.get("dance_scales"),
            global_support_ticks=consumed_music_support(sample,engine.horizon_steps),
            global_control_ticks=consumed_music_ticks(sample,engine.horizon_steps),
            global_reference_records=sample.transitions[:engine.horizon_steps],global_reference_version=sample.generation.reference_version)
        return {"dance":terms["dance_loss"].mean(),"track":terms["current_reference_loss"].mean(),
            "local":lp,"global":gp,"action_rate":terms["action_rate_cost"].mean(),
            "joint_limit":terms["joint_limit_cost"].mean(),"terminal_value":lp.new_zeros(())}

    for sample in samples:
        description=describe_sample(sample,engine.adapter)
        if sample.generation.request.actor_version!=trainer.accepted_actor_updates:
            rows.append(dict(description,accepted=False,stage="D_drawn",rejection="stale_actor_request"));continue
        try:
            local=teacher.local(sample.generation.request)
            interval,plan,noise=engine.rollout_anchor(sample)
            probe=torch.autograd.grad(state_probe(interval.end.state),noise,retain_graph=True,allow_unused=True)[0]
            if probe is None or not torch.isfinite(probe).all() or float(probe.norm())<=0 or interval.steps!=engine.horizon_steps:
                raise FloatingPointError("候选更新缺少有限完整真实anchor执行路径导数")
            terms=losses(sample,interval,plan,local)
            for k,v in terms.items():
                losses_before[k]+=float(v.detach())
                if v.requires_grad:
                    gs=torch.autograd.grad(v,params,retain_graph=True,allow_unused=True)
                    for i,g in enumerate(gs):
                        if g is not None:
                            if not torch.isfinite(g).all():raise FloatingPointError(k+"梯度非有限")
                            gradients[k][i]=g.detach().clone() if gradients[k][i] is None else gradients[k][i]+g.detach()
                    del gs
            for k,v in interval.event["objective_terms"].items():diagnostics[k]=diagnostics.get(k,0.)+float(v.detach().mean())
            clock=plan.ticket.request_tick+torch.arange(60,device=plan.world_qpos.device)*20
            fixed_ref=sample.state.teacher_reference
            mask=(torch.arange(60,device=clock.device)>=plan.ticket.prefix_frames)&(clock>=sample.state.tick)&(clock<=fixed_ref.upper)
            mask=mask & torch.isin(clock,consumed_music_support(sample,engine.horizon_steps))
            recorded_anchor=sample.state.reference.read_current_future(sample.state.tick,future_steps=0)[0]
            alignment=AnchorAlignment.at_reference_anchor(recorded_anchor,fixed_ref.read_current_future(sample.state.tick,future_steps=0)[0])
            absolute_target=fixed_ref.source_at(clock.clamp(fixed_ref.lower,fixed_ref.upper))
            target=alignment.poses(absolute_target)
            fixed.append((sample,local,plan.world_qpos.detach(),target,mask,float(terms["dance"].detach()),absolute_target))
            execution_norms.append(float(probe.detach().norm()))
            rows.append(dict(description,accepted=False,stage="D_drawn",gradient_valid=True,rejection=None))
            del interval,plan,noise,terms,probe
        except UnsupportedModelState as error:
            rejected.append("outside_validated_world_model_support")
            rows.append(dict(description,accepted=False,stage="C_outside_support",rejection=str(error),
                             support_details=getattr(error,"details",None)))
    valid=len(fixed)
    for k in keys:
        losses_before[k]/=max(1,valid)
        gradients[k]=[g/max(1,valid) if g is not None else None for g in gradients[k]]
    diagnostics={k:v/max(1,valid) for k,v in diagnostics.items()}
    bc_rng=capture_rng()
    if valid:
        prediction=actor.original_supervision(supervised_batch)
        bc,_=trainer.stage1_losses(supervised_batch,prediction["pred_x_start"],prediction["static_conf_logits"],
                                  global_step=config.get("stage1_supervision_step",595000))
        losses_before["bc"]=float(bc.detach())
        gradients["bc"]=[g.detach() if g is not None else None for g in torch.autograd.grad(bc,params,allow_unused=True)]
        del prediction,bc
    norms={k:_norm(v) for k,v in gradients.items()}
    cosine={f"{a}__{b}":_cosine(gradients[a],gradients[b],norms[a],norms[b]) for i,a in enumerate(keys) for b in keys[i+1:]}
    modules={m:[i for i,n in enumerate(names) if parameter_module(n)==m] for m in ("stage1_backbone","history_prefix","future_encoder","diffusion_output")}
    module_metrics={m:{k:{"raw_norm":_norm([gradients[k][i] for i in indices]),
        "weighted_norm":weights[k]*_norm([gradients[k][i] for i in indices]),
        "cosine_with_dance":_cosine([gradients[k][i] for i in indices],[gradients["dance"][i] for i in indices],
            _norm([gradients[k][i] for i in indices]),_norm([gradients["dance"][i] for i in indices]))}
        for k in keys} for m,indices in modules.items()}
    projected={k:[None if g is None else g*weights[k] for g in gradients[k]] for k in keys}
    projection_log=[]
    if config.get("auxiliary_gradient_projection",False) and norms["dance"]:
        for k in ("track","local","global","bc","action_rate"):
            dot=sum(float((a*b).sum()) for a,b in zip(projected[k],gradients["dance"]) if a is not None and b is not None)
            if dot<0:
                coefficient=dot/norms["dance"]**2
                projected[k]=[(a if a is not None else torch.zeros_like(b))-coefficient*b if b is not None else a
                              for a,b in zip(projected[k],gradients["dance"])]
                projection_log.append({"term":k,"removed_coefficient":coefficient})
    for i,p in enumerate(params):
        values=[projected[k][i] for k in keys if projected[k][i] is not None]
        p.grad=sum(values[1:],values[0].clone()) if values else None
    grad_norm=float(torch.nn.utils.clip_grad_norm_(params,config.get("grad_clip",1.),error_if_nonfinite=True)) if valid else 0.
    del projected
    model_components={k:v for k,v in gradients.items()}
    # 各模块对真实AdamW参数位移的影响也使用原始梯度，不拿SGD方向替代AdamW。
    for module,indices in modules.items():
        selected=set(indices)
        for k in keys:model_components[module+"/"+k]=[g if i in selected else None for i,g in enumerate(gradients[k])]

    def evaluate():
        reasons=[];outputs=[];after={k:0. for k in keys};joint_diagnostics={}
        for sample,local,before,target,mask,anchor_dance,absolute_target in fixed:
            try:
                interval,plan,_=engine.rollout_anchor(sample)
                values=losses(sample,interval,plan,local)
                for k,v in values.items():after[k]+=float(v)/valid
                rs,metrics=check_outputs(before,plan.world_qpos,local.world_qpos,target,mask,
                    prefix=plan.ticket.prefix_frames,config=trust,unaligned_global_target=absolute_target)
                if float(values["dance"])>anchor_dance+trust["dance_worsening_atol"]+trust["dance_worsening_rtol"]*abs(anchor_dance):
                    reasons.append("candidate_dance_loss_worse")
                reasons.extend(rs);outputs.append(metrics)
                for k,v in interval.event["objective_terms"].items():
                    if k.startswith(("joint_soft_","joint_hard_","joint_near_")):
                        joint_diagnostics[k]=joint_diagnostics.get(k,0.)+float(v.mean())/valid
            except UnsupportedModelState as error:
                reasons.append("outside_validated_world_model_support");outputs.append({"support_details":getattr(error,"details",None)})
            except (FloatingPointError,ValueError) as error:
                reasons.append("nonfinite_candidate");outputs.append({"error":str(error)})
        restore_rng(bc_rng)
        prediction=actor.original_supervision(supervised_batch)
        bc,_=trainer.stage1_losses(supervised_batch,prediction["pred_x_start"],prediction["static_conf_logits"],
            global_step=config.get("stage1_supervision_step",595000));after["bc"]=float(bc)
        if any(not math.isfinite(v) for v in after.values()):reasons.append("nonfinite_candidate")
        if after["dance"]>losses_before["dance"]+trust["dance_worsening_atol"]+trust["dance_worsening_rtol"]*abs(losses_before["dance"]):
            reasons.append("candidate_dance_loss_worse")
        return sorted(set(reasons)),{"outputs":outputs,"losses_before":losses_before,"losses_after":after,
            "loss_delta":{k:after[k]-losses_before[k] for k in keys},"joint_limit_diagnostics_after":joint_diagnostics}

    verdict=candidate_adamw_step(actor,trainer.actor_optimizer,evaluate,config=trust,
        execution_gradient_valid=valid>0 and math.isfinite(norms["dance"]) and norms["dance"]>0,
        gradient_components=model_components)
    for attempt in verdict["attempts"]:
        attempt["weighted_gradient_dot_actual_parameter_delta"]={k:weights[k.rsplit("/",1)[-1]]*v
            for k,v in attempt["raw_gradient_dot_actual_parameter_delta"].items()}
    accepted=verdict["accepted"]
    if accepted:
        trainer.accepted_actor_updates+=1
        for row in rows:
            if row.get("gradient_valid"):row.update(accepted=True,stage="E_effective_update")
    else:
        trainer.invalid_actor_attempts+=1;rejected+=verdict["rejection_reasons"]
    critic_loss=critic_norm=None
    if not config.get("defer_critic",False):
        if not trainer.critic_ready:raise ValueError("新奖励Critic未验证，不能在线继续拟合旧价值")
        groups=physical_interval_groups if physical_interval_groups is not None else [physical_intervals]
        targets=[pair for group in groups for pair in trainer._targets(group)]
        critic_loss,critic_norm=trainer.fit_critic(targets[::max(1,len(targets)//16)])
    engine.adapter.bundle.assert_unchanged(trainer.frozen_identity);teacher.assert_frozen();trainer.update_index+=1
    last=verdict["attempts"][-1] if verdict["attempts"] else {}
    return {"update":trainer.update_index,"accepted_actor_updates":trainer.accepted_actor_updates,
        "training_method":"online_stage1_teacher_trust_region_short_bptt",
        "accepted_execution_feedback_update":accepted,"loss_components_raw":losses_before,
        "loss_components_weighted":{k:weights[k]*v for k,v in losses_before.items()},"loss_weights":weights,
        "gradient_norms_raw":norms,"gradient_norms_weighted":{k:weights[k]*norms[k] for k in keys},
        "action_rate_scale_audit":{"dimensions":21,"normalization":"sum_unchanged",
            "raw_sum_gradient_norm":norms["action_rate"],"raw_mean_gradient_norm":norms["action_rate"]/21,
            "old_weighted_sum_gradient_norm":.1*norms["action_rate"],
            "current_weighted_sum_gradient_norm":weights["action_rate"]*norms["action_rate"]},
        "gradient_cosines":cosine,"module_gradient_metrics":module_metrics,
        "future_encoder_gradient_norms":{k:module_metrics["future_encoder"][k]["weighted_norm"] for k in keys},
        "future_encoder_stage1_bc_gradient":module_metrics["future_encoder"]["bc"]["raw_norm"],
        "parameter_module_counts":{m:len(indices) for m,indices in modules.items()},
        "auxiliary_projection":projection_log,"output_trust_region":verdict,
        "original_music_execution_loss":losses_before["dance"],"current_reference_tracking_loss":losses_before["track"],
        "local_preservation_loss":losses_before["local"],"global_preservation_loss":losses_before["global"],
        "original_stage1_loss":losses_before["bc"],"execution_state_noise_gradient_norms":execution_norms,
        "valid_execution_rollouts":valid,"actor_grad_norm":grad_norm,
        "actor_parameter_delta_l2":last.get("parameter_delta_l2",0.) if accepted else 0.,
        "sample_distribution":rows,"effective_sample_coverage":coverage_summary([r for r in rows if r["accepted"]]),
        "rejection_reasons":rejected,"raw_execution_diagnostics":diagnostics,"horizon_steps":engine.horizon_steps,
        "terminal_value_weight":0.,"critic_deferred":config.get("defer_critic",False),
        "critic_mse":critic_loss,"critic_grad_norm":critic_norm,"critic_updates":trainer.critic_updates,
        "frozen_parameters_and_buffers_unchanged":True,"frozen_teacher_unchanged":True,"bc_only_update":False,
        "update_wall_seconds":time.perf_counter()-started,"evidence":"candidate_model_acceptance_not_physical_quality"}
