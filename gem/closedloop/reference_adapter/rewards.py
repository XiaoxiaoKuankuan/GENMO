"""原SM奖励的直接复用与独立附加成本。

runtime.reward只接收执行终态和名义下一参考，返回的SM奖励已经包含dt；本模块仅对
新增音乐/残差/活动成本乘一次dt。六项诊断用原MDP的身体集合、坐标对齐和尺度，
用于报告而不替换原函数。硬/软终止同样以名义参考计算，修正目标不参与降低门槛。
"""

import torch

from supermimic.mdp import unflatten_state
from supermimic.rotations import yaw_mat, rot_angle_sq
from .contracts import DT
from .reference import overlap_cost

DEFAULT_WEIGHTS = dict(music=.2, magnitude=.02, smooth=.01, overlap=.02, reference_limit=.1, activity=.05)


def sm_diagnostics(runtime, state, nominal):
    p, R, v, w, q, qd = unflatten_state(state)
    bp, bR, bv, bw = runtime.kin.forward(p, R, v, w, q, qd)
    dR = yaw_mat(R @ nominal.root_R.transpose(-1, -2))
    anchor = torch.cat((p[..., :2], nominal.root_p[..., 2:3]), -1)
    target_p = anchor.unsqueeze(-2) + (dR.unsqueeze(-3) @ (nominal.body_p - nominal.root_p.unsqueeze(-2)).unsqueeze(-1)).squeeze(-1)
    errors = dict(body_pos=(target_p-bp).square().sum(-1).mean(-1),
                  body_ori=rot_angle_sq(dR.unsqueeze(-3) @ nominal.body_R, bR).mean(-1),
                  body_linvel=(nominal.body_v-bv).square().sum(-1).mean(-1),
                  body_angvel=(nominal.body_w-bw).square().sum(-1).mean(-1),
                  root_pos=(nominal.root_p-p).square().sum(-1), root_ori=rot_angle_sq(nominal.root_R, R))
    return {**{f"error_{k}": v for k, v in errors.items()},
            **{k: torch.exp(-v / getattr(runtime.mdp.cfg, k+"_std")**2) for k, v in errors.items()}}


class Rewards:
    def __init__(self, runtime, music, weights=None):
        self.runtime, self.music = runtime, music
        self.weights = DEFAULT_WEIGHTS | (weights or {})
        if set(self.weights) != set(DEFAULT_WEIGHTS):
            raise ValueError("未知附加奖励名称")

    def __call__(self, z, corrected, state_next, action, nominal_next, bounds):
        r_sm, original = self.runtime.reward(state_next, action, z.previous_action, nominal_next, diagnostics=True)
        memory = self.music.push(z.music_state, state_next[..., 34:], nominal_next.qd, z.tick+12)
        memory, r_music, activity, music_info = self.music.settle(memory, z.audio)
        normalized = corrected.plan.residual / bounds
        q = corrected.future.q
        costs = dict(magnitude=normalized.square().mean((1, 2)),
                     smooth=torch.diff(normalized, dim=1).square().mean((1, 2)),
                     overlap=overlap_cost(corrected.plan, z.previous_plan, bounds, z.heading),
                     reference_limit=(torch.relu(q-self.runtime.mdp.soft_hi)+torch.relu(self.runtime.mdp.soft_lo-q)).sum(-1).mean(-1),
                     activity=activity)
        raw = {"sm": r_sm, "music": r_music, **costs, **original,
               **music_info,
               "halo_max_abs": corrected.halo_residual.abs().amax(-1),
               "target_qd_max_abs": corrected.future.qd.abs().amax((1, 2)),
               "residual_saturation": (normalized.abs() >= .98).to(q).mean((1, 2))}
        weighted = {"sm": r_sm, "music": DT*self.weights["music"]*r_music,
                    **{k: -DT*self.weights[k]*v for k, v in costs.items()}}
        return sum(weighted.values()), raw, weighted, memory
