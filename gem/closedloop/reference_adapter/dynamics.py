"""由冻结原bundle深拷贝得到的独立可监督动力学模型。

只开放WM.net参数；原in/out normalizer统计保持固定。监督标签必须来自真实且连续的
物理转移，使用原WorldModel.targets/state_error定义；多步闭环预测对照真实标签。
Adapter阶段冻结参数但不关闭输入autograd，想象显式调用本实例step，绝不回落原bundle。
"""

from contextlib import contextmanager
import copy

import torch
from torch import nn

from supermimic.mdp import flatten_state, unflatten_state, normalize_xi


@contextmanager
def frozen_parameters(*modules):
    snapshots = [(p, p.requires_grad) for module in modules for p in module.parameters()]
    for p, _ in snapshots:
        p.grad = None
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, enabled in snapshots:
            p.requires_grad_(enabled)
            p.grad = None


class DynamicsModel(nn.Module):
    def __init__(self, bundle):
        super().__init__()
        self.model = copy.deepcopy(bundle.world_model)
        self.model.requires_grad_(False)
        self.model.net.requires_grad_(True)
        self.model.eval()
        self.model_version = 0
        self._graph_active = False
        # 与SM train.py同一误差归一化下限，非PD/机器人常量。
        floors = [3e-3]*3 + [5e-3]*3 + [2e-2]*3 + [5e-2]*3 + [3e-3]*21 + [5e-2]*21
        self.register_buffer("error_floor", torch.tensor(floors, device=next(self.model.parameters()).device))

    def step(self, state, action, xi):
        return flatten_state(self.model.step(unflatten_state(state), action, normalize_xi(xi)))

    @contextmanager
    def adapter_phase(self):
        if self._graph_active:
            raise RuntimeError("不能嵌套WM版本屏障")
        self._graph_active = True
        try:
            with frozen_parameters(self):
                yield self.model_version
        finally:
            self._graph_active = False

    def supervised_loss(self, windows, *, checkpoint_steps=False):
        from torch.utils.checkpoint import checkpoint
        if windows.get("origin") != "real":
            raise ValueError("WM禁止使用想象状态作为真实标签")
        s, a, xi, target, mask = (windows[k] for k in ("state", "action", "xi", "next_state", "mask"))
        if mask.shape != a.shape[:2] or not bool(mask[:, 0].all()):
            raise ValueError("WM窗口起点必须有效")
        state = s[:, 0]
        std = torch.maximum(self.model.out_norm.std(), self.error_floor)
        errors, one_step = [], []
        for t in range(a.shape[1]):
            state = (checkpoint(self.step, state, a[:, t], xi[:, t], use_reentrant=False)
                     if checkpoint_steps else self.step(state, a[:, t], xi[:, t]))
            errors.append((self.model.state_error(unflatten_state(state), unflatten_state(target[:, t])) / std).square().mean(-1))
            # 单步targets只报告，避免为不参与损失的诊断额外保存整个长窗口的图。
            with torch.no_grad():
                predicted = self.model.net(self.model.in_norm(self.model.inputs(unflatten_state(s[:, t]), a[:, t], normalize_xi(xi[:, t]))))
                expected = self.model.out_norm(self.model.targets(unflatten_state(s[:, t]), unflatten_state(target[:, t])))
                one_step.append((predicted-expected).square().mean(-1))
        error = torch.stack(errors, 1)
        one = torch.stack(one_step, 1)
        # 沿用SM的多步state_error主损失；单步targets误差另列，不改变归一化。
        loss = (error * mask).sum() / mask.sum().clamp_min(1)
        return loss, {"wm_state_error": loss.detach(), "wm_targets_mse": (one*mask).sum().detach()/mask.sum().clamp_min(1),
                      "wm_last_error": (error[:, -1]*mask[:, -1]).sum().detach()/mask[:, -1].sum().clamp_min(1)}

    def fit(self, windows, optimizer, grad_clip=1.):
        if self._graph_active:
            raise RuntimeError("Adapter计算图存活时禁止更新WM参数")
        optimizer.zero_grad(set_to_none=True)
        loss, report = self.supervised_loss(windows)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(self.model.net.parameters(), grad_clip, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        self.model_version += 1
        return report | {"wm_grad_norm": float(norm), "model_version": self.model_version}

    @torch.no_grad()
    def validate_real(self, windows, *, horizons=(1, 4, 8, 16, 32, 64), kin=None):
        """按实际连续窗口报告H1至H64，不外推缺少的真实标签。"""
        from .audit import _state_error
        if windows.get("origin") != "real" or windows["action"].shape[1] < 16:
            raise ValueError("WM门槛需要至少完整真实16步窗口")
        horizons = tuple(h for h in horizons if h <= windows['action'].shape[1])
        state, reports = windows["state"][:, 0], {}
        for i in range(max(horizons)):
            state = self.step(state, windows["action"][:, i], windows["xi"][:, i])
            if i+1 in horizons:
                reports[str(i+1)] = _state_error(state, windows["next_state"][:, i])
                reports[str(i+1)]['root_angular_velocity_rad_s'] = float((state[:, 10:13]-windows['next_state'][:, i, 10:13]).square().mean().sqrt())
                if kin is not None:
                    a = kin.forward(*unflatten_state(state))[3]
                    b = kin.forward(*unflatten_state(windows['next_state'][:, i]))[3]
                    reports[str(i+1)]['body_angular_velocity_rad_s'] = float((a-b).square().mean().sqrt())
        return reports
