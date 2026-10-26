"""真实物理与想象共用的八项跟踪奖励，同时独立报告原SM六项。

身体/足部/腿关节和尺度来自显式迁移配置，按名称映射全部FK身体，不能默认使用SM的
12体子集。位置/姿态相对根XY与yaw对齐只用于原跟踪指标，不重锚参考缓存。根误差
仍是世界误差；contact2不屏蔽代价。所有奖励乘dt一次，音乐代理默认不提供。
"""

import math

import torch

from supermimic.mdp import unflatten_state
from supermimic.rotations import so3_log, yaw_mat

TERM_NAMES = ("anchor_pos", "anchor_ori", "body_pos", "body_ori", "body_lin_vel",
              "body_ang_vel", "foot_height", "leg_joint_pos")
WEIGHTS = (.5, .5, 1., 1., 1., 1., .8, .5)


class TrackingRewards:
    def __init__(self, runtime, config):
        self.runtime, self.config = runtime, dict(config)
        if config.get("mode") not in ("sm_six", "upper_eight", "teacher_music"):
            raise ValueError("奖励必须显式选择sm_six或upper_eight")
        self.mode = config["mode"]
        self.scales = config["scales"]
        if set(self.scales) != set(TERM_NAMES) or any(not math.isfinite(v) or v <= 0 for v in self.scales.values()):
            raise ValueError("八项尺度必须完整、有限、为正")
        if self.scales["anchor_pos"] != .3 or self.scales["anchor_ori"] != .4:
            raise ValueError("首版根尺度保持0.3m / 0.4rad")
        kin = runtime.kin
        def select(names, available):
            if not names or len(set(names)) != len(names) or not set(names) <= set(available):
                raise ValueError(f"奖励名称集合不存在或重复: {names}")
            return tuple(available.index(n) for n in names)
        self.bodies = select(config["body_names"], kin.body_names)
        self.feet = select(config["foot_names"], kin.body_names)
        self.legs = select(config["leg_joint_names"], list(kin.joint_names))
        if config.get("music_surrogate_enabled", False):
            raise ValueError("音乐代理未验证，首版仅原Stage1监督与独立音乐评价")

    def __call__(self, state, action, previous_action, reference, *, teacher_reference=None):
        sm_reward, sm_terms = self.runtime.reward(state, action, previous_action, reference)
        p, R, v, w, q, qd = unflatten_state(state)
        kin, ref = self.runtime.kin, reference
        bp, bR, bv, bw = kin.forward(p, R, v, w, q, qd, all_bodies=True)
        rp, rR, rv, rw = kin.forward(ref.root_p, ref.root_R, ref.root_v, ref.root_w, ref.q, ref.qd, all_bodies=True)
        delta_R = yaw_mat(R @ ref.root_R.transpose(-1, -2))
        anchor = torch.cat((p[..., :2], ref.root_p[..., 2:3]), -1)
        aligned_p = anchor[:, None] + (delta_R[:, None] @ (rp - ref.root_p[:, None]).unsqueeze(-1)).squeeze(-1)
        aligned_R = delta_R[:, None] @ rR
        angle_sq = lambda a, b: so3_log(a.transpose(-1, -2) @ b).square().sum(-1)
        e = {
            "anchor_pos": (p - ref.root_p).square().sum(-1),
            "anchor_ori": angle_sq(ref.root_R, R),
            "body_pos": (bp[:, self.bodies] - aligned_p[:, self.bodies]).square().sum(-1).mean(-1),
            "body_ori": angle_sq(aligned_R[:, self.bodies], bR[:, self.bodies]).mean(-1),
            "body_lin_vel": (bv[:, self.bodies] - rv[:, self.bodies]).square().sum(-1).mean(-1),
            "body_ang_vel": (bw[:, self.bodies] - rw[:, self.bodies]).square().sum(-1).mean(-1),
            "foot_height": (bp[:, self.feet, 2] - aligned_p[:, self.feet, 2]).square().mean(-1),
            "leg_joint_pos": (q[:, self.legs] - ref.q[:, self.legs]).square().mean(-1),
        }
        scores = {key: torch.exp(-value / self.scales[key] ** 2) for key, value in e.items()}
        tracking = sum(weight * scores[key] for key, weight in zip(TERM_NAMES, WEIGHTS)) / 6.3
        rate = (action - previous_action).square().sum(-1)
        limits = (torch.relu(q - self.runtime.mdp.soft_hi) + torch.relu(self.runtime.mdp.soft_lo - q)).sum(-1)
        root_cost_weight = float(self.config.get("root_huber_weight", 0.))
        root_norm = torch.stack((e["anchor_pos"] / .3**2, e["anchor_ori"] / .4**2), -1)
        root_cost = torch.where(root_norm < 1, .5 * root_norm,
                                torch.sqrt(root_norm.clamp_min(1e-12)) - .5).sum(-1)
        upper = (tracking - self.config.get("action_rate_weight", .1) * rate -
                 self.config.get("joint_limit_weight", 2.) * limits - root_cost_weight * root_cost) * .02
        metrics = {"sm_six_reward_dt": sm_reward, "upper_eight_reward_dt": upper,
                   "upper_eight_tracking": tracking, "joint_speed_sq": qd.square().mean(-1),
                   "root_xy_error_sq": (p[:, :2] - ref.root_p[:, :2]).square().sum(-1),
                   "action_rate": rate, "joint_limit": limits, "root_huber": root_cost,
                   **{f"error/{k}": value for k, value in e.items()},
                   **{f"score/{k}": value for k, value in scores.items()}}
        if self.mode == "teacher_music":
            if teacher_reference is None:
                raise ValueError("音乐执行主目标缺少固定原Stage1时间线，不能退回学生自参考奖励")
            from .dance_losses import execution_loss
            dance, dance_groups = execution_loss(state, teacher_reference, kin, self.config.get("dance_scales"))
            track, track_groups = execution_loss(state, reference, kin, self.config.get("dance_scales"))
            physical = self.config.get("action_rate_weight", .1)*rate + self.config.get("joint_limit_weight", 2.)*limits
            music_reward = -(dance + self.config.get("track_weight", .1)*track + physical)*.02
            metrics.update(dance_loss=dance, current_reference_loss=track, physical_cost=physical,
                teacher_music_reward_dt=music_reward,
                **{f"dance/{k}":v for k,v in dance_groups.items()},
                **{f"current_reference/{k}":v for k,v in track_groups.items()})
            return music_reward, metrics
        return (sm_reward if self.mode == "sm_six" else upper), metrics
