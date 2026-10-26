"""每一步重新执行Adapter、冻结Actor和独立WM的闭环BPTT。

reference_steps固定10，展开horizon独立设置为16。仅detach真实起点；参考锚点、上一
计划、执行状态、Actor历史及音乐统计在内部保持图。累计软存活按SM公式只乘一次；
滚动边界停止并以当前已知计划bootstrap，记录给定名义计划条件下的局部策略边界。
"""

from dataclasses import dataclass

import torch

from .features import critic_features


@dataclass
class ImaginedTrajectory:
    rows: list
    features: torch.Tensor
    values: torch.Tensor
    objective: torch.Tensor
    component_objectives: dict
    model_version: int
    boundary: str


class Imagination:
    def __init__(self, step, dynamics, critic, *, bptt_steps=16, gamma=.99):
        if bptt_steps < 1:
            raise ValueError("BPTT步数必须为正，独立于10帧参考")
        self.step, self.dynamics, self.critic = step, dynamics, critic
        self.bptt_steps, self.gamma = bptt_steps, gamma

    def rollout(self, initial, *, intervention=None):
        if initial.origin != "real":
            raise ValueError("想象必须从真实采样起点开始")
        z = initial.detached()
        feature = lambda state: critic_features(state, self.step.adapter, self.step.bundle, self.step.runtime.kin)
        xs, rows = [feature(z)], []
        survival = z.executed_state.new_ones(len(z.executed_state))
        components, boundary = {}, "bptt_truncation"
        version = self.dynamics.model_version
        for k in range(self.bptt_steps):
            if z.mode == "rolling" and (z.tick+144 > z.nominal_reference.upper or
                    (z.plan_boundary_tick is not None and z.tick >= z.plan_boundary_tick)):
                boundary = "known_nominal_plan_boundary_bootstrap"
                break
            proposal = self.step.propose(z, **(intervention if k == 0 and intervention is not None else {}))
            next_state = self.dynamics.step(z.executed_state, proposal.action, z.xi)
            row = self.step.advance(z, proposal, next_state, origin="imagined")
            for name, value in row.weighted.items():
                components[name] = components.get(name, 0.) + self.gamma**k * survival * value
            survival = survival * row.continuation
            rows.append(row)
            z = row.end
            xs.append(feature(z))
            if bool(row.terminated.all()) or row.truncated:
                boundary = row.reason if bool(row.terminated.all()) else "known_nominal_plan_boundary_bootstrap"
                break
        if not rows:
            raise ValueError("真实起点没有合法想象控制步")
        xs = torch.stack(xs)
        values = self.critic(xs)
        components["terminal_value"] = self.gamma**len(rows) * survival * values[-1]
        if self.dynamics.model_version != version:
            raise RuntimeError("存图期间WM版本改变")
        return ImaginedTrajectory(rows, xs, values, sum(components.values()), components, version, boundary)
