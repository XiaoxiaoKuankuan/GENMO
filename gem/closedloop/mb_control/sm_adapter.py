"""GENMO 到权威 SM 接口的薄适配，仅按关节名称构造双向排列。

同一个 bundle 提供 pi/wm，runtime 提供75维观测、FK和SM六项奖励。模块不复制网络、
PD或机器人常量；GENMO历史采用原proprio48定义且明确标记真实/想象来源。
"""

from dataclasses import replace

import torch

from supermimic.bridge.runtime import SuperMimicRuntime
from supermimic.mdp import unflatten_state
from gem.closedloop.contracts import GMT_EXPECTED_JOINT_ORDER


class SMAdapter:
    def __init__(self, bundle, genmo_joint_names, device="cpu"):
        self.bundle = bundle
        self.runtime = SuperMimicRuntime(bundle, device)
        native = tuple(genmo_joint_names)
        policy = bundle.spec.joint_names
        if len(set(native)) != 21 or set(native) != set(policy):
            raise ValueError("GENMO与SM关节名称不是一一对应")
        self.to_sm = tuple(native.index(name) for name in policy)
        self.to_genmo = tuple(policy.index(name) for name in native)
        # 这是已训练Stage1的历史字段顺序，不是引入旧控制后端。
        self.to_history = tuple(policy.index(name) for name in GMT_EXPECTED_JOINT_ORDER)

    def proprio(self, state, xi):
        _, R, _, w, q, qd = unflatten_state(state)
        # 与原proprio48一致：gravity、body angular velocity、相对active default的q、qd。
        relative_q = q - self.runtime.mdp.default_q - xi[..., 1:22]
        return torch.cat((-R[..., 2, :], (R.transpose(-1, -2) @ w.unsqueeze(-1)).squeeze(-1),
                          relative_q[..., self.to_history], qd[..., self.to_history]), -1)

    def refresh_current_observation(self, z):
        current, _, _ = z.reference.read_current_future(z.tick)
        obs = self.runtime.observation(z.state, z.previous_action, z.xi, current)
        return replace(z, sm_history=torch.cat((z.sm_history[:, :-1], obs[:, None]), 1))

    def advance(self, z, state, action, *, origin, trim_reference=True):
        tick = z.tick + 12
        current, _, _ = z.reference.read_current_future(tick)
        obs = self.runtime.observation(state, action, z.xi, current)
        return replace(z, tick=tick, state=state, previous_action=action,
                       sm_history=torch.cat((z.sm_history[:, 1:], obs[:, None]), 1),
                       genmo_history=torch.cat((z.genmo_history[:, 1:], self.proprio(state, z.xi)[:, None]), 1),
                       genmo_valid=torch.cat((z.genmo_valid[:, 1:], torch.ones_like(z.genmo_valid[:, :1])), 1),
                       reference=z.reference.trim(tick) if trim_reference else z.reference, history_origin=origin)
