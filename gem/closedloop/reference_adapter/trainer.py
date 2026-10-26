"""独立三优化器训练器：真实WM监督、冻结模型BPTT、SM lambda-return Critic。

先用零Adapter真实完整episode预热价值；每轮先拟合独立WM，再锁定model_version完成
Adapter和Critic更新。batch64通过逐真实起点梯度累积实现，物理仍为单环境；没有旧
GENMO Trainer、BC、PPO或扩散概率。报告分项值和首个合法轨迹的参数梯度贡献，物理
收益必须由独立真实评价确定，不能以此处loss或模型回报代替。
"""

import torch

from .critic import lambda_returns, calibration
from .dynamics import frozen_parameters
from .features import critic_features


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
            if not episode or not episode[-1].terminated.all():
                raise ValueError("真实MC必须有明确的物理/音乐/参考终止，不能把行政截断设零价值")
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

    def warmup_real(self, epochs=20):
        train = self.replay.complete_episodes("train")
        validation = self.replay.complete_episodes("validation")
        if any(r.policy_version != 0 for e in train+validation for r in e):
            raise ValueError("首次Critic预热必须使用零Adapter真实数据")
        x, y, _ = self.real_targets(train)
        xv, yv, near = self.real_targets(validation)
        for _ in range(epochs):
            self._fit_value(x, y)
        with torch.no_grad():
            report = calibration(self.critic(xv), yv, near)
        self.critic_ready = True
        self.warmup_report = report | {"epochs": epochs, "train_samples": len(y), "physical_quality_passed": False}
        return self.warmup_report

    @torch.no_grad()
    def check_real_calibration(self, episodes):
        x, y, near = self.real_targets(episodes)
        return calibration(self.critic(x), y, near)

    def update(self, model_gate=None):
        if not self.critic_ready:
            raise RuntimeError("先采集零Adapter真实数据并完成Critic预热")
        cfg = self.config
        batch, window = cfg.get("batch_size", 64), cfg.get("wm_fit_window", 8)
        # WM和验证都来自真实窗口，正式轮次不使用想象状态标签。
        report = self.dynamics.fit(self.replay.sample_windows(batch, window), self.wm_optimizer, cfg.get("grad_clip_norm", 1.))
        with torch.no_grad():
            _, validation = self.dynamics.supervised_loss(self.replay.sample_windows(batch, window, "validation"))
        report.update({"validation_"+k: float(v) for k, v in validation.items()})
        if model_gate is not None:
            gate = model_gate()
            if not gate["passed"]:
                raise RuntimeError("当前WM/Adapter分布未通过真实物理方向审计；停止Adapter更新")
        starts = self.replay.sample_starts(batch)
        self.adapter_optimizer.zero_grad(set_to_none=True)
        xs, targets, returns = [], [], []
        raw, weighted, gradient_contributions = {}, {}, {}
        parameters = tuple(self.adapter.parameters())
        with self.dynamics.adapter_phase() as version:
            with frozen_parameters(self.critic):
                for i, start in enumerate(starts):
                    trajectory = self.engine.rollout(start)
                    reward = torch.stack([r.reward for r in trajectory.rows])
                    cont = torch.stack([r.continuation for r in trajectory.rows])
                    targets.append(lambda_returns(reward, cont, trajectory.values, self.engine.gamma, cfg.get("lambda_return_per_control", .95)))
                    xs.append(trajectory.features[:-1].detach())
                    if i == 0:
                        gradient_contributions = {name: component_grad_norm(value, parameters)
                                                  for name, value in trajectory.component_objectives.items()}
                    for name, value in trajectory.component_objectives.items():
                        weighted.setdefault(name, []).append(float(value.detach().mean()))
                    for row in trajectory.rows:
                        for name, value in row.raw.items():
                            if isinstance(value, torch.Tensor):
                                raw.setdefault(name, []).append(float(value.detach().float().mean()))
                    returns.append(float(trajectory.objective.detach().mean()))
                    (-trajectory.objective.mean()/len(starts)).backward()
            norm = torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), cfg.get("grad_clip_norm", 1.), error_if_nonfinite=True)
            self.adapter_optimizer.step()
            self.adapter_optimizer.zero_grad(set_to_none=True)
            # 目标取自同一更新前轨迹；Critic只拟合已detach输入，不共享Adapter图。
            loss, critic_norm = self._fit_value(torch.cat(xs).flatten(0, 1), torch.cat(targets).flatten())
            if self.dynamics.model_version != version:
                raise RuntimeError("Adapter/Critic阶段WM版本改变")
        self.engine.step.bundle.assert_unchanged(self.frozen_identity)
        self.update_index += 1
        return {**{k: float(v) if isinstance(v, torch.Tensor) else v for k, v in report.items()},
                "update": self.update_index, "model_return": sum(returns)/len(returns), "adapter_grad_norm": float(norm),
                "critic_mse": loss, "critic_grad_norm": critic_norm,
                "unweighted_step_terms": {k: sum(v)/len(v) for k, v in raw.items()},
                "discounted_weighted_components": {k: sum(v)/len(v) for k, v in weighted.items()},
                "component_parameter_gradient_norm": gradient_contributions,
                "gradient_contribution_scope": "first_real_start_current_update_weighted_discounted_objective",
                "physical_quality_passed": False, "batch_implementation": "sequential_real_starts_gradient_accumulation"}

    def state_dict(self):
        return dict(update_index=self.update_index, critic_updates=self.critic_updates, real_steps=self.real_steps,
                    critic_ready=self.critic_ready, warmup_report=self.warmup_report,
                    model_version=self.dynamics.model_version)

    def load_state_dict(self, value):
        self.update_index, self.critic_updates, self.real_steps = value["update_index"], value["critic_updates"], value["real_steps"]
        self.critic_ready, self.warmup_report = value["critic_ready"], value["warmup_report"]
        self.dynamics.model_version = value["model_version"]
