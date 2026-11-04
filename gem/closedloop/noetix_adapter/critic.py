"""为原生 noetix 参考 Adapter 接入独立 Critic 和真实转移 TD 监督。

Adapter 的扩展目标含姿态支持点和音乐，不能直接喂给预训练的606维目标Critic。
本模块在原生BPTT入口提取未修正的名义目标600维及当前锚点朝向6维，保留状态
输入梯度；Actor和WM在策略反传阶段仍由原生实现冻结。Critic以指定底座权重
初始化，优化器从零开始，学习率独立于Adapter的衰减。

预热和正式训练均可使用真实Replay的一步转移拟合TD：奖励和终止标签来自
物理仿真，真正终止不bootstrap，时间截断保留价值。正式阶段另沿用原生想象
lambda-return拟合。TD误差只表示自举拟合情况，不宣称完整MC校准或整歌收益。
实际Critic Adam次数由优化器step回调统计，并由外层训练入口保存恢复。
"""

import torch

from supermimic.imagination.policy_trainer import PolicyTrainer
from supermimic.world_model.wm_trainer import reference_at


def nominal_goal(packed):
    """扩展参考恢复为原生606维名义目标，支持任意前导批次维。"""
    return torch.cat((packed[..., :600], packed[..., -6:]), dim=-1)


def td_target(reward, done, next_value, gamma):
    return reward + gamma * (1.0 - done) * next_value


class AdapterPolicyTrainer(PolicyTrainer):
    """保留原生想象更新，补名义Critic输入、真实TD及独立学习率。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.critic_steps = 0
        self.real_critic_steps = 0
        self.critic_opt.register_step_post_hook(self._count_critic_step)

    def _count_critic_step(self, optimizer, args, kwargs):
        self.critic_steps += 1

    def _critic_inputs(self, ro):
        feat, goals, err = super()._critic_inputs(ro)
        return feat, nominal_goal(goals), err

    def set_lr(self, lr):
        # 原生set_lr会同时覆盖Critic LR；这里保持两种优化器独立。
        for group in self.opt.param_groups:
            group['lr'] = lr

    def update_real_critic(self, buffer, batch_size=2048, batches=2):
        """只更新Critic，不产生Adapter Adam步，也不更新Actor/WM权重。"""
        losses, gradients, values, targets = [], [], [], []
        for _ in range(batches):
            with torch.no_grad():
                batch = buffer.sample_windows(batch_size, 1)
                frames = torch.stack((batch.frame[0], torch.minimum(
                    batch.frame[0] + 1, self.lib.last_frame_of(batch.frame[0]))))
                state = batch.states.flatten(0, 1)
                reference = reference_at(self.lib, frames.flatten())
                base = self.lib.library.goal_window(frames.flatten(), self.cfg.future_frames,
                    self.cfg.goal_features, goal_root_height=self.policy.cfg.goal_root_height)
                anchor = self.wm.heads.anchor_index
                goal = self.step_fn.goal(state, base, reference.body_pos_w[:, anchor],
                                         reference.body_quat_w[:, anchor])
                feat = self.wm.spec.canonical(state)
                xi = self.wm.spec.normalize_xi(batch.xi).repeat(2, 1)
                heads = self.wm.heads
                err = (heads.critic_error_features(state, reference) if self.critic.err_dim > 4
                       else heads.root_error_features(state, reference))
                self.critic.update_input_norm(feat, goal, xi, err)
                value = self.critic(feat, goal, xi, err).reshape(2, batch_size)
                target = td_target(batch.reward[0], batch.done[0], value[1], self.cfg.gamma)
            pred = self.critic(feat[:batch_size], goal[:batch_size], xi[:batch_size], err[:batch_size])
            loss = torch.nn.functional.mse_loss(pred, target)
            self.critic_opt.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(self.critic.parameters(),
                self.cfg.critic_grad_clip, error_if_nonfinite=True)
            self.critic_opt.step()
            self.real_critic_steps += 1
            losses.append(loss.detach())
            gradients.append(grad.detach())
            values.append(pred.detach().mean())
            targets.append(target.mean())
        return {key: float(torch.stack(items).mean()) for key, items in (
            ('real_td_loss', losses), ('real_td_grad_norm', gradients),
            ('real_value_mean', values), ('real_target_mean', targets))}
