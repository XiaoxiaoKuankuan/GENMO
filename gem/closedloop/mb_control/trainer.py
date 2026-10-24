"""仅GENMO与独立上层Critic两个优化器的模型路径导数训练器。

初始GENMO固定期间用当前策略真实/想象轨迹预热价值；随后Actor目标为多区间回报加
冻结参数但输入可微的终点价值，以及原Stage1配对监督。Critic targets在任何更新前
固定并detach，价值回归不向GENMO累计梯度。冻结pi/wm逐张量核对，绝不创建其优化器。
"""

from contextlib import contextmanager
import time

import torch

from .returns import actor_return, lambda_targets


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

    def _targets(self, intervals):
        with torch.no_grad():
            values = [torch.zeros_like(x.reward) if x.terminal else self.critic(x.end.detached()) for x in intervals]
            targets = lambda_targets(intervals, values, self.config.get("lambda_per_half_second", .95))
        return [(x.start.detached(), y) for x, y in zip(intervals, targets)]

    def update(self, initial_states, supervised_batch, actual_latencies, *, physical_intervals):
        started = time.perf_counter()
        warming = self.update_index < self.config["warmup_updates"]
        if not physical_intervals:
            raise ValueError("每轮需要当前策略真实采集，不能只优化WM自反馈")
        self.actor.eval()
        self.critic.eval()
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        training_pairs = self._targets(physical_intervals)
        returns = []
        for z, latencies in zip(initial_states, actual_latencies):
            noises = [torch.randn(len(z.state), 60, 30, device=z.state.device)
                      for _ in range(self.imagination.decisions)]
            with torch.set_grad_enabled(not warming), frozen_parameters(self.critic):
                intervals = self.imagination.rollout(z, noises, latencies)
                training_pairs.extend(self._targets(intervals))
                final = intervals[-1]
                terminal_value = torch.zeros_like(final.reward) if final.terminal or warming else self.critic(final.end)
                objective = actor_return(intervals, terminal_value).mean()
                returns.append(float(objective.detach()))
                if not warming:
                    (-objective / len(initial_states)).backward()
        bc_value, actor_norm = 0., 0.
        if not warming:
            output = self.actor.original_supervision(supervised_batch)
            bc, _ = self.stage1_losses(supervised_batch, output["pred_x_start"], output["static_conf_logits"],
                                      global_step=self.config.get("stage1_supervision_step", 595000))
            (self.config["bc_weight"] * bc).backward()
            bc_value = float(bc.detach())
            actor_norm = float(torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.get("grad_clip", 1.), error_if_nonfinite=True))
            self.actor_optimizer.step()
            self.actor_optimizer.zero_grad(set_to_none=True)
        # 目标在更新之前已固定。这里只反传detach状态，不共享Actor图。
        critic_loss = 0.
        self.critic.train()
        for z, target in training_pairs:
            loss = torch.nn.functional.mse_loss(self.critic(z), target)
            (loss / len(training_pairs)).backward()
            critic_loss += float(loss.detach()) / len(training_pairs)
        critic_norm = float(torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.get("grad_clip", 1.), error_if_nonfinite=True))
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad(set_to_none=True)
        self.imagination.adapter.bundle.assert_unchanged(self.frozen_identity)
        self.update_index += 1
        return {"update": self.update_index, "critic_warmup": warming, "model_return": sum(returns) / len(returns),
                "physical_reward_dt_sum": sum(float(x.reward.mean()) for x in physical_intervals),
                "original_stage1_loss": bc_value, "critic_mse": critic_loss,
                "actor_grad_norm": actor_norm, "critic_grad_norm": critic_norm,
                "update_wall_seconds": time.perf_counter() - started,
                "evidence": "model_objective_and_separate_physical_collection_not_quality_gain"}
