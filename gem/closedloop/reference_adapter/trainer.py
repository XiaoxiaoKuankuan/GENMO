"""独立三优化器训练器：真实WM监督、冻结模型BPTT、SM lambda-return Critic。

先用零Adapter真实完整episode预热价值；每轮先拟合独立WM，再锁定model_version完成
Adapter和Critic更新。batch64通过逐真实起点梯度累积实现，物理仍为单环境；没有旧
GENMO Trainer、BC、PPO或扩散概率。报告分项值和首个合法轨迹的参数梯度贡献，物理
收益必须由独立真实评价确定，不能以此处loss或模型回报代替。
"""

import copy
import time

import torch

from .critic import lambda_returns, calibration
from .dynamics import frozen_parameters
from .features import critic_features
from .quality import critic_gate, prediction_gate


def component_grad_norm(value, parameters):
    if not value.requires_grad:
        return 0.
    gradients = torch.autograd.grad(value.mean(), parameters, retain_graph=True, allow_unused=True)
    return float(sum((g.detach().square().sum() for g in gradients if g is not None), value.new_zeros(())).sqrt())


class AdapterTrainer:
    def __init__(self, engine, replay, cfg):
        self.engine, self.replay, self.config = engine, replay, dict(cfg)
        self.adapter, self.critic, self.dynamics = engine.step.adapter, engine.critic, engine.dynamics
        self.adapter_optimizer = torch.optim.Adam(self.adapter.parameters(), lr=cfg.get("adapter_lr", 1e-4))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.get("critic_lr", 1e-4))
        self.wm_optimizer = torch.optim.Adam(self.dynamics.model.net.parameters(), lr=cfg.get("wm_lr", 3e-5))
        self.update_index, self.critic_updates, self.real_steps = 0, 0, 0
        self.critic_ready = False
        self.frozen_identity = engine.step.bundle.frozen_snapshot()
        self.warmup_report = None
        self.physical_gate = None
        self.audited_wm = None
        self.attempted_updates = 0
        self.imagined_steps = 0
        self._rollback = None

    def features(self, z):
        return critic_features(z, self.adapter, self.engine.step.bundle, self.engine.step.runtime.kin)

    def _fit_value(self, x, target):
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss = (self.critic(x.detach())-target.detach()).square().mean()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.get("grad_clip_norm", 1.), error_if_nonfinite=True)
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad(set_to_none=True)
        self.critic_updates += 1
        return float(loss.detach()), float(norm)

    @torch.no_grad()
    def real_targets(self, episodes):
        features, targets, near_failure = [], [], []
        gamma = self.engine.gamma
        for episode in episodes:
            if (not episode or not episode[-1].terminated.all() or episode[-1].reason not in ("music_end", "physical_failure")
                    or episode[0].start.tick != episode[0].start.episode_start_tick):
                raise ValueError("真实MC必须从reset开始且物理/音乐真实终止；参考耗尽和行政截断不构成完整MC")
            ret = torch.zeros_like(episode[-1].reward)
            returns = []
            for row in reversed(episode):
                ret = row.reward + gamma*(~row.terminated)*ret
                returns.append(ret)
            features.extend(self.features(row.start) for row in episode)
            targets.extend(reversed(returns))
            near_failure.extend([episode[-1].reason == "physical_failure" and i >= len(episode)-16 for i in range(len(episode))])
        if not features:
            raise ValueError("缺少独立真实完整episode")
        x, y = torch.cat(features), torch.cat(targets)
        return x, y, torch.tensor(near_failure, device=y.device)

    def warmup_dataset(self):
        """只在零策略完整回放固定后构造一次；各预热轮复用同一真实标签。"""
        train = self.replay.complete_episodes("train")
        validation = self.replay.complete_episodes("validation")
        if any(r.policy_version != 0 for e in train+validation for r in e):
            raise ValueError("首次Critic预热必须使用零Adapter真实数据")
        x, y, _ = self.real_targets(train)
        xv, yv, near = self.real_targets(validation)
        return x, y, xv, yv, near

    def warmup_real(self, epochs=20, *, dataset=None):
        x, y, xv, yv, near = self.warmup_dataset() if dataset is None else dataset
        best, best_state = float("inf"), None
        batch = self.config.get("critic_batch_size", 256)
        for _ in range(epochs):
            order = torch.randperm(len(y), device=y.device)
            for ids in order.split(batch):
                self._fit_value(x[ids], y[ids])
            with torch.no_grad():
                score = float((self.critic(xv)-yv).square().mean())
            if score < best:
                best = score
                best_state = (copy.deepcopy(self.critic.state_dict()), copy.deepcopy(self.critic_optimizer.state_dict()))
        if best_state is not None:
            self.critic.load_state_dict(best_state[0])
            self.critic_optimizer.load_state_dict(best_state[1])
        with torch.no_grad():
            report = calibration(self.critic(xv), yv, near, constant_value=y.mean())
        gate = critic_gate(report, self.config.get("critic_quality"), train_samples=len(y))
        self.critic_ready = gate["passed"]
        self.warmup_report = report | {"epochs": epochs, "train_samples": len(y), "quality_gate": gate}
        return self.warmup_report

    @torch.no_grad()
    def check_real_calibration(self, episodes):
        x, y, near = self.real_targets(episodes)
        constant = self.warmup_report["constant_value"] if self.warmup_report else None
        return calibration(self.critic(x), y, near, constant_value=constant)

    def register_physical_gate(self, report):
        if not report.get("passed") or not report.get("tasks"):
            raise RuntimeError("当前Adapter真实物理方向门槛未通过")
        self.physical_gate = copy.deepcopy(report) | {"accepted_update": self.update_index,
                                                      "registered_unix": time.time(), "model_version": self.dynamics.model_version}
        self.audited_wm = {k: p.detach().clone() for k, p in self.dynamics.named_parameters()}

    def validate_wm(self):
        batch = self.config.get("wm_validation_batch_size", 32)
        report = self.dynamics.validate_real(self.replay.sample_windows(batch, 16, "validation", latest_only=True))
        gate = prediction_gate(report, self.config.get("wm_quality"))
        return {"errors": report, "quality_gate": gate}

    def rollback_last_update(self):
        if self._rollback is not None:
            for name, state in self._rollback.items():
                getattr(self, name).load_state_dict(state)
            self.update_index -= 1
            self._rollback = None

    def update(self, model_gate=None):
        if not self.critic_ready:
            raise RuntimeError("Critic独立真实MC质量门槛未通过")
        cfg, started = self.config, time.perf_counter()
        batch, window = cfg.get("batch_size", 64), cfg.get("wm_fit_window", 8)
        self.attempted_updates += 1
        report = self.dynamics.fit(self.replay.sample_windows(batch, window), self.wm_optimizer, cfg.get("grad_clip_norm", 1.))
        validation = self.validate_wm()
        if not validation["quality_gate"]["passed"]:
            raise RuntimeError(f"WM独立真实1/8/16步误差门槛失败: {validation}")
        if model_gate is not None:
            self.register_physical_gate(model_gate())
        if self.physical_gate is None or self.update_index-self.physical_gate["accepted_update"] >= cfg.get("small_audit_every", 25):
            raise RuntimeError("缺少有效当前分布物理方向审计或超过允许更新间隔")
        drift = sum((p.detach()-self.audited_wm[k].to(p)).square().sum() for k, p in self.dynamics.named_parameters())
        scale = sum(p.square().sum() for p in self.audited_wm.values()).clamp_min(1e-12)
        drift = float((drift/scale).sqrt())
        if drift > cfg.get("max_wm_relative_drift", .005):
            raise RuntimeError(f"WM距物理核验版本漂移{drift}超过门槛，需提前真实方向审计")
        starts = self.replay.sample_starts(batch)
        parameters = tuple(self.adapter.parameters())
        with torch.no_grad():
            before = self.engine.step.propose(starts[0])
        self._rollback = None
        snapshot = {name: copy.deepcopy(getattr(self, name).state_dict()) for name in
                    ("adapter", "adapter_optimizer", "critic", "critic_optimizer")}
        xs, targets, returns = [], [], []
        raw, weighted, contributions, execution_norms = {}, {}, {}, []
        self.adapter_optimizer.zero_grad(set_to_none=True)
        microbatch = max(1, cfg.get("backward_microbatch", 1))
        pending, imagined = [], 0
        try:
            with self.dynamics.adapter_phase() as version, frozen_parameters(self.critic):
                for i, start in enumerate(starts):
                    trajectory = self.engine.rollout(start)
                    imagined += len(trajectory.rows)
                    reward = torch.stack([r.reward for r in trajectory.rows])
                    cont = torch.stack([r.continuation for r in trajectory.rows])
                    targets.append(lambda_returns(reward, cont, trajectory.values, self.engine.gamma, cfg.get("lambda_return_per_control", .95)))
                    xs.append(trajectory.features[:-1].detach())
                    # 只检查真实执行任务项，不让幅值/平滑正则替代执行链梯度。
                    execution = sum(self.engine.gamma**j*(r.raw["track"]+r.weighted["music"]) for j, r in enumerate(trajectory.rows))
                    execution_norms.append(component_grad_norm(execution, parameters))
                    if i == 0:
                        contributions = {name: component_grad_norm(value, parameters) for name, value in trajectory.component_objectives.items()}
                        g = torch.autograd.grad(execution.mean(), trajectory.rows[0].action, retain_graph=True, allow_unused=True)[0]
                        contributions["action_input"] = float(g.norm()) if g is not None else 0.
                        bias_grad = torch.autograd.grad(execution.mean(), self.adapter.net[-1].bias, retain_graph=True)[0].reshape(10, 27)
                        contributions["root_output"] = float(bias_grad[:, :6].norm())
                        contributions["joint_output"] = float(bias_grad[:, 6:].norm())
                    for name, value in trajectory.component_objectives.items():
                        weighted.setdefault(name, []).append(float(value.detach().mean()))
                    for row in trajectory.rows:
                        for name, value in row.raw.items():
                            if isinstance(value, torch.Tensor):
                                raw.setdefault(name, []).append(float(value.detach().float().mean()))
                    returns.append(float(trajectory.objective.detach().mean()))
                    pending.append(-trajectory.objective.mean()/len(starts))
                    if len(pending) == microbatch or i == len(starts)-1:
                        torch.stack(pending).sum().backward()
                        pending.clear()
                if max(execution_norms) <= cfg.get("min_execution_gradient", 1e-9):
                    raise RuntimeError("只有正则梯度或执行反馈梯度为零，拒绝计数")
                norm = torch.nn.utils.clip_grad_norm_(parameters, cfg.get("grad_clip_norm", 1.), error_if_nonfinite=True)
                self.adapter_optimizer.step()
                self.adapter_optimizer.zero_grad(set_to_none=True)
                if self.dynamics.model_version != version:
                    raise RuntimeError("Adapter计算图期间WM版本改变")
            if not all(torch.isfinite(p).all() for p in parameters):
                raise FloatingPointError("Adapter更新后参数非有限")
            with torch.no_grad():
                after = self.engine.step.propose(starts[0])
                residual_change = after.corrected_reference.plan.residual-before.corrected_reference.plan.residual
                action_change = after.action-before.action
                parameter_delta = sum((v-snapshot["adapter"][k]).square().sum() for k, v in self.adapter.state_dict().items() if v.is_floating_point()).sqrt()
            if parameter_delta <= 0 or residual_change.abs().max() <= 0 or action_change.abs().max() <= 0:
                raise RuntimeError("参数/同状态参考/部署控制没有有效改变，拒绝计数")
            loss, critic_norm = self._fit_value(torch.cat(xs).flatten(0, 1), torch.cat(targets).flatten())
            calibration_report = self.check_real_calibration(self.replay.complete_episodes("validation"))
            calibration_quality = critic_gate(calibration_report, cfg.get("critic_quality"))
            if not calibration_quality["passed"]:
                raise RuntimeError(f"在线Critic真实MC质量退化: {calibration_quality}")
            self.engine.step.bundle.assert_unchanged(self.frozen_identity)
        except Exception:
            for name, state in snapshot.items():
                getattr(self, name).load_state_dict(state)
            self.adapter_optimizer.zero_grad(set_to_none=True)
            raise
        self.update_index += 1
        self._rollback = snapshot
        self.imagined_steps += imagined
        return {**{k: float(v) if isinstance(v, torch.Tensor) else v for k, v in report.items()},
                "update": self.update_index, "accepted_adapter_updates": self.update_index,
                "attempted_updates": self.attempted_updates, "model_return": sum(returns)/len(returns),
                "adapter_grad_norm": float(norm), "execution_gradient_max": max(execution_norms),
                "parameter_delta_l2": float(parameter_delta), "same_state_reference_root_rms": float(residual_change[..., :6].square().mean().sqrt()),
                "same_state_reference_joint_rms": float(residual_change[..., 6:].square().mean().sqrt()),
                "same_state_action_rms": float(action_change.square().mean().sqrt()),
                "critic_mse": loss, "critic_grad_norm": critic_norm, "real_critic_calibration": calibration_report,
                "wm_validation": validation, "wm_relative_drift_since_audit": drift,
                "unweighted_step_terms": {k: sum(v)/len(v) for k, v in raw.items()},
                "discounted_weighted_components": {k: sum(v)/len(v) for k, v in weighted.items()},
                "component_parameter_gradient_norm": contributions,
                "gradient_contribution_scope": "first_real_start_current_update_weighted_discounted_objective",
                "physical_quality_passed": self.physical_gate["passed"],
                "physical_gate_source": {k: v for k, v in self.physical_gate.items() if k != "tasks"},
                "physical_gate_scope": "last_measured_direction_audit_plus_current_real_prediction_checks",
                "real_steps": self.real_steps, "imagined_steps": imagined, "update_wall_seconds": time.perf_counter()-started,
                "batch_implementation": f"sequential_forward_graph_microbatch_{microbatch}"}

    def state_dict(self):
        return dict(update_index=self.update_index, critic_updates=self.critic_updates, real_steps=self.real_steps,
                    critic_ready=self.critic_ready, warmup_report=self.warmup_report,
                    model_version=self.dynamics.model_version, physical_gate=self.physical_gate, audited_wm=self.audited_wm,
                    attempted_updates=self.attempted_updates, imagined_steps=self.imagined_steps)

    def load_state_dict(self, value):
        self.update_index, self.critic_updates, self.real_steps = value["update_index"], value["critic_updates"], value["real_steps"]
        self.critic_ready, self.warmup_report = value["critic_ready"], value["warmup_report"]
        self.dynamics.model_version = value["model_version"]

        self.physical_gate, self.audited_wm = value.get("physical_gate"), value.get("audited_wm")
        self.attempted_updates, self.imagined_steps = value.get("attempted_updates", 0), value.get("imagined_steps", 0)
