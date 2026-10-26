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


def update_teacher_anchors(trainer, samples, supervised_batch, physical_intervals, physical_interval_groups):
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
