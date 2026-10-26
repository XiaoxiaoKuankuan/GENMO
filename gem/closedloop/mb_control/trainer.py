"""仅GENMO与独立上层Critic两个优化器的模型路径导数训练器。

正式模式在初始GENMO固定期间使用真实完整MC预热并独立验证价值；短时Actor目标为
真实接管起点的H步模型回报、冻结参数但输入可微的终点价值以及原Stage1配对监督。
原多区间长链保留研究对照。Critic targets在任何更新前固定并detach，短时模式只用
各物理环境各自的真实控制轨迹回归。冻结pi/wm逐张量核对，绝不创建其优化器。
"""

from contextlib import contextmanager
import time

import torch

from .returns import actor_return, lambda_targets


def gradient_norm(parameters):
    values = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.


@contextmanager
def frozen_parameters(module):
    flags = [p.requires_grad for p in module.parameters()]
    try:
        module.requires_grad_(False)
        yield
    finally:
        for parameter, enabled in zip(module.parameters(), flags):
            parameter.requires_grad_(enabled)


class Stage2Trainer:
    def __init__(self, actor, critic, imagination, stage1_losses, config):
        self.actor, self.critic, self.imagination, self.stage1_losses = actor, critic, imagination, stage1_losses
        self.config, self.update_index = dict(config), 0
        self.accepted_actor_updates = self.critic_updates = self.invalid_actor_attempts = 0
        self.critic_ready = False
        self.critic_quality = {}
        self.actor_optimizer = torch.optim.AdamW([
            {"params": [p for p in actor.stage1.parameters() if p.requires_grad], "lr": config["genmo_lr"]},
            {"params": actor.future_encoder.parameters(), "lr": config["forecast_lr"]}], weight_decay=config.get("weight_decay", .01))
        self.critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=config["critic_lr"])
        groups = [set(p.untyped_storage().data_ptr() for p in module.parameters())
                  for module in (actor, critic, imagination.adapter.bundle)]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
            raise ValueError("GENMO、Critic、冻结模型不能共享参数存储")
        if config["warmup_updates"] < 1 or config["bc_weight"] <= 0:
            raise ValueError("必须预热独立Critic并保留原Stage1监督")
        self.frozen_identity = imagination.adapter.bundle.frozen_snapshot()

    def counters(self):
        return {name: getattr(self, name) for name in ("update_index", "accepted_actor_updates", "critic_updates",
                "invalid_actor_attempts", "critic_ready", "critic_quality")}

    def restore_counters(self, values):
        for name in self.counters():
            setattr(self, name, values[name])

    def fit_critic(self, training_pairs):
        """只拟合已固定的真实MC或合法bootstrap目标，不更新生成器。"""
        if not training_pairs:
            raise ValueError("Critic训练目标不能为空")
        self.critic.train()
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss = 0.
        for z, target in training_pairs:
            loss = torch.nn.functional.mse_loss(self.critic(z), target.detach())
            (loss / len(training_pairs)).backward()
            critic_loss += float(loss.detach()) / len(training_pairs)
        norm = float(torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.get("grad_clip", 1.), error_if_nonfinite=True))
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad(set_to_none=True)
        self.critic_updates += 1
        return critic_loss, norm

    def _targets(self, intervals):
        with torch.no_grad():
            values = [torch.zeros_like(x.reward) if x.terminal else self.critic(x.end.detached()) for x in intervals]
            targets = lambda_targets(intervals, values, self.config.get("lambda_per_half_second", .95))
        return [(x.start.detached(), y) for x, y in zip(intervals, targets)]

    def update_anchors(self, samples, supervised_batch, *, physical_intervals, physical_interval_groups=None):
        """仅真实、当前Actor版本、完整H步的首次接管样本可计作有效执行反馈。"""
        from .execution_audit import state_probe
        from .validation import UnsupportedModelState
        started = time.perf_counter()
        if not self.critic_ready or not physical_intervals or not samples:
            raise ValueError("短时Actor更新需要已验收Critic和新鲜真实控制轨迹")
        self.actor.eval()
        self.critic.eval()
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        # 只用真实控制轨迹回归；不把WM自己的终点当成价值验证真值。
        groups = physical_interval_groups if physical_interval_groups is not None else [physical_intervals]
        targets = [pair for group in groups if group for pair in self._targets(group)]
        targets = targets[::max(1, len(targets)//16)]
        objectives, execution_norms, rejected = [], [], []
        comparison, valid = None, 0
        for sample in samples:
            if sample.generation.request.actor_version != self.accepted_actor_updates:
                rejected.append("stale_actor_request")
                continue
            try:
                with frozen_parameters(self.critic):
                    interval, plan, noise = self.imagination.rollout_anchor(sample)
                    value = torch.zeros_like(interval.reward) if interval.terminal else self.critic(interval.end)
                    objective = actor_return([interval], value).mean()
                    probe = state_probe(interval.end.state)
                    grad = torch.autograd.grad(probe, noise, retain_graph=True, allow_unused=True)[0]
                    norm = 0. if grad is None else float(grad.detach().norm())
                    if grad is not None and not torch.isfinite(grad).all():
                        raise FloatingPointError("短时执行状态梯度非有限")
                    if norm <= 0 or interval.steps != self.imagination.horizon_steps:
                        rejected.append("disconnected_or_incomplete_short_execution")
                        continue
                    (-objective/len(samples)).backward()
                    objectives.append(float(objective.detach()))
                    execution_norms.append(norm)
                    valid += 1
                    if comparison is None:
                        comparison = (sample.generation, plan.world_qpos.detach().clone())
            except UnsupportedModelState as error:
                rejected.append(str(error))
        params = [p for p in self.actor.parameters() if p.requires_grad]
        model_norm = gradient_norm(params)
        if not torch.isfinite(torch.tensor(model_norm)):
            raise FloatingPointError("短时执行目标参数梯度非有限")
        accepted = valid > 0 and model_norm > 0
        bc_value = bc_norm = actor_norm = delta = 0.
        output_change = None
        if accepted:
            before = [p.detach().clone() for p in params]
            model_grads = [p.grad.detach().clone() if p.grad is not None else None for p in params]
            out = self.actor.original_supervision(supervised_batch)
            bc, _ = self.stage1_losses(supervised_batch, out["pred_x_start"], out["static_conf_logits"],
                global_step=self.config.get("stage1_supervision_step", 595000))
            (self.config["bc_weight"]*bc).backward()
            bc_value = float(bc.detach())
            differences = [(p.grad.detach()-(old if old is not None else 0.)).float().square().sum()
                           for p, old in zip(params, model_grads) if p.grad is not None]
            bc_norm = float(torch.stack(differences).sum().sqrt())
            del model_grads
            actor_norm = float(torch.nn.utils.clip_grad_norm_(params, self.config.get("grad_clip", 1.), error_if_nonfinite=True))
            self.actor_optimizer.step()
            delta = float(torch.stack([(p.detach()-old).float().square().sum() for p, old in zip(params, before)]).sum().sqrt())
            del before
            if not 0 < delta < float("inf"):
                raise FloatingPointError("Actor没有产生有限非零参数更新")
            self.accepted_actor_updates += 1
            self.actor_optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                record, old_plan = comparison
                request = record.request
                after = self.actor.generate(request.snapshot, request.ticket, record.forecast, request.noise).world_qpos
                output_change = float((after[:, request.ticket.prefix_frames:, 7:]-old_plan[:, request.ticket.prefix_frames:, 7:]).square().mean().sqrt())
        else:
            self.invalid_actor_attempts += 1
            self.actor_optimizer.zero_grad(set_to_none=True)
        loss, critic_norm = self.fit_critic(targets)
        self.imagination.adapter.bundle.assert_unchanged(self.frozen_identity)
        self.update_index += 1
        return {"update": self.update_index, "training_method": "physical_anchor_short_bptt",
                "accepted_actor_updates": self.accepted_actor_updates, "accepted_execution_feedback_update": accepted,
                "model_return": sum(objectives)/len(objectives) if objectives else None,
                "model_gradient_norm_before_bc": model_norm, "weighted_bc_gradient_norm": bc_norm,
                "original_stage1_loss": bc_value, "actor_grad_norm": actor_norm, "critic_grad_norm": critic_norm,
                "critic_mse": loss, "critic_updates": self.critic_updates, "bc_only_update": False,
                "execution_state_noise_gradient_norms": execution_norms, "valid_execution_rollouts": valid,
                "real_anchor_ticks": [s.state.tick for s in samples],
                "real_anchor_identities": [{"collector": s.collector_id, "episode": s.episode_index,
                    "tick": s.state.tick, "decision": s.generation.request.ticket.decision_id} for s in samples],
                "horizon_steps": self.imagination.horizon_steps, "actor_parameter_delta_l2": delta,
                "fixed_noise_reference_joint_change_rms_rad": output_change,
                "physical_reward_dt_sum": sum(float(x.reward.mean()) for x in physical_intervals),
                "frozen_parameters_and_buffers_unchanged": True, "rejection_reasons": rejected,
                "update_wall_seconds": time.perf_counter()-started,
                "evidence": "real_anchor_model_gradient_separate_from_physical_quality"}

    def update(self, initial_states, supervised_batch, actual_latencies, *, physical_intervals):
        from .execution_audit import state_probe
        from .validation import UnsupportedModelState
        from .forecast import forecast_rollout
        from .contracts import ForecastContext
        started = time.perf_counter()
        if not initial_states or len(initial_states) != len(actual_latencies):
            raise ValueError("真实快照与外生延迟批次必须非空且对齐")
        if self.config.get("require_critic_quality", False) and not self.critic_ready:
            raise ValueError("正式Actor更新要求独立真实回报验证通过的Critic")
        # 有界smoke也不能靠“已经拟合两次”解锁未经验证的终点价值。
        # 未加载真实质量通过状态时，仅允许Critic管线更新；Actor保持固定。
        warming = not self.critic_ready
        if not physical_intervals:
            raise ValueError("每轮需要当前策略真实采集，不能只优化WM自反馈")
        self.actor.eval()
        self.critic.eval()
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        training_pairs = self._targets(physical_intervals)
        returns = []
        graph_rollouts, accepted_commits, valid_rollouts = 0, 0, 0
        execution_norms, rejected = [], []
        comparison = None
        for z, latencies in zip(initial_states, actual_latencies):
            noises = [torch.randn(len(z.state), 60, 30, device=z.state.device)
                      for _ in range(self.imagination.decisions)]
            noises[0].requires_grad_(not warming)
            with torch.set_grad_enabled(not warming), frozen_parameters(self.critic):
                plans = []
                try:
                    intervals = self.imagination.rollout(z, noises, latencies, on_plan=plans.append)
                except UnsupportedModelState as error:
                    rejected.append(str(error))
                    continue
                training_pairs.extend(self._targets(intervals))
                final = intervals[-1]
                terminal_value = torch.zeros_like(final.reward) if final.terminal or warming else self.critic(final.end)
                objective = actor_return(intervals, terminal_value).mean()
                returns.append(float(objective.detach()))
                commits = sum(x.event.get("accepted", False) for x in intervals)
                accepted_commits += commits
                graph_rollouts += int(objective.requires_grad)
                if not warming and objective.requires_grad:
                    # 无BC、无Critic、无直接参考评分的状态探针只做连通性门禁，
                    # 不是额外Actor奖励，也不代表真实物理方向已经验证。
                    probe = state_probe(final.end.state)
                    execution_grad = (torch.autograd.grad(probe, noises[0], retain_graph=True, allow_unused=True)[0]
                                      if probe.requires_grad else None)
                    execution_norm = float(execution_grad.detach().norm()) if execution_grad is not None else 0.
                    execution_norms.append(execution_norm)
                    if execution_grad is not None and not torch.isfinite(execution_grad).all():
                        raise FloatingPointError("执行状态梯度包含NaN/Inf")
                    if commits < 2 or execution_norm <= 0:
                        rejected.append("insufficient_commits_or_disconnected_execution")
                        continue
                    with torch.profiler.record_function("mb/actor_bptt_backward"):
                        (-objective / len(initial_states)).backward()
                    valid_rollouts += 1
                    if comparison is None:
                        comparison = (z.detached(), plans[0].ticket, noises[0].detach(), plans[0].world_qpos.detach())
        bc_value, actor_norm = 0., 0.
        model_norm = gradient_norm(self.actor.parameters())
        if not torch.isfinite(torch.tensor(model_norm)):
            raise FloatingPointError("模型执行目标梯度包含NaN/Inf")
        accepted = not warming and valid_rollouts > 0 and model_norm > 0
        parameter_delta = 0.
        reference_change = None
        if accepted:
            before = [p.detach().clone() for p in self.actor.parameters()]
            output = self.actor.original_supervision(supervised_batch)
            bc, _ = self.stage1_losses(supervised_batch, output["pred_x_start"], output["static_conf_logits"],
                                      global_step=self.config.get("stage1_supervision_step", 595000))
            with torch.profiler.record_function("mb/stage1_supervision_backward"):
                (self.config["bc_weight"] * bc).backward()
            bc_value = float(bc.detach())
            actor_norm = float(torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.get("grad_clip", 1.), error_if_nonfinite=True))
            self.actor_optimizer.step()
            parameter_delta = float(torch.stack([(p.detach() - old).float().square().sum()
                                    for p, old in zip(self.actor.parameters(), before)]).sum().sqrt())
            del before
            if parameter_delta <= 0 or not torch.isfinite(torch.tensor(parameter_delta)):
                raise FloatingPointError("Actor优化没有产生有限非零参数变化")
            self.accepted_actor_updates += 1
            self.actor_optimizer.zero_grad(set_to_none=True)
            if comparison is not None:
                with torch.no_grad():
                    z0, ticket0, noise0, pose0 = comparison
                    context0 = (forecast_rollout(z0, self.imagination.adapter, ticket0.first_influence_tick).context
                                if self.imagination.forecast_enabled else ForecastContext.invalid(z0.state, z0.tick))
                    after = self.actor.generate(z0, ticket0, context0, noise0).world_qpos
                    reference_change = float((after[:, ticket0.prefix_frames:, 7:] - pose0[:, ticket0.prefix_frames:, 7:]).square().mean().sqrt())
        elif not warming:
            self.invalid_actor_attempts += 1
            self.actor_optimizer.zero_grad(set_to_none=True)
        # 目标在更新之前已固定。这里只反传detach状态，不共享Actor图。
        critic_loss, critic_norm = self.fit_critic(training_pairs)
        self.imagination.adapter.bundle.assert_unchanged(self.frozen_identity)
        self.update_index += 1
        return {"update": self.update_index, "critic_warmup": warming, "model_return": sum(returns) / len(returns) if returns else None,
                "accepted_actor_updates": self.accepted_actor_updates, "accepted_execution_feedback_update": accepted,
                "critic_updates": self.critic_updates, "invalid_actor_attempts": self.invalid_actor_attempts,
                "bc_only_update": False, "model_gradient_norm_before_bc": model_norm,
                "execution_state_noise_gradient_norms": execution_norms, "valid_execution_rollouts": valid_rollouts,
                "actor_parameter_delta_l2": parameter_delta, "rejection_reasons": rejected,
                "fixed_noise_reference_joint_change_rms_rad": reference_change,
                "frozen_parameters_and_buffers_unchanged": True,
                "physical_reward_dt_sum": sum(float(x.reward.mean()) for x in physical_intervals),
                "original_stage1_loss": bc_value, "critic_mse": critic_loss,
                "actor_grad_norm": actor_norm, "critic_grad_norm": critic_norm,
                "imagined_accepted_commits": accepted_commits, "execution_graph_rollouts": graph_rollouts,
                "update_wall_seconds": time.perf_counter() - started,
                "evidence": "model_objective_and_separate_physical_collection_not_quality_gain"}
