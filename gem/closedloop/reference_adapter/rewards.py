"""原SM奖励的直接复用与独立附加成本。

runtime.reward只接收执行终态和名义下一参考，返回的SM奖励已经包含dt；本模块仅对
新增音乐/残差/活动成本乘一次dt。六项诊断用原MDP的身体集合、坐标对齐和尺度，
用于报告而不替换原函数。硬/软终止同样以名义参考计算，修正目标不参与降低门槛。
"""

import torch
from dataclasses import asdict, replace
from math import isfinite

from supermimic.mdp import unflatten_state
from supermimic.rotations import yaw_mat, rot_angle_sq
from .contracts import DT
from .reference import overlap_cost

DEFAULT_WEIGHTS = dict(music=1e-4, magnitude=.02, smooth=.01, overlap=.02, reference_limit=.1, activity=.05)


def configure_sm_mdp(runtime, overrides):
    """入口一次性覆盖用户指定的误差尺度和终止阈值，仍调用SM权威公式。

    RewardCfg在原MDP构造器中是共享默认对象，必须replace为本运行独立实例，
    不能原地修改并污染另一个零基线/旧标准运行。六项权重、速度尺度、Actor及其
    normalizer均不改变；hard/soft termination自动使用同一高度和重力投影阈值。
    """
    allowed = {'root_pos_std', 'root_ori_std', 'body_pos_std', 'body_ori_std', 'term_height', 'term_tilt'}
    if set(overrides)-allowed or any(not isfinite(v) or v <= 0 for v in overrides.values()):
        raise ValueError('SM误差尺度/终止阈值必须为已声明字段的有限正数')
    runtime.mdp.cfg = replace(runtime.mdp.cfg, **overrides)


def sm_mdp_contract(runtime):
    """报告实际生效参数，禁止把不同尺度/终止标准的评分混成配对提升。"""
    return dict(parameters=asdict(runtime.mdp.cfg), dt=DT,
        weights=dict(body_pos=1., body_ori=1., body_linvel=1., body_angvel=1., root_pos=.5, root_ori=.5),
        formula='sum(weight*exp(-squared_error/std**2))*dt',
        termination='abs(z-z_nominal)>term_height OR abs(R[2,2]-R_nominal[2,2])>term_tilt',
        target='original_GENMO_nominal_reference', body_alignment='original_SM_root_xy_yaw')


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
