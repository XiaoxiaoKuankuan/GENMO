"""增强状态新Critic与按50Hz步定义的SM lambda-return。

网络独立新建512/256两隐藏层，不导入旧GENMO价值参数；输入含状态、xi、实际历史、
原参考、残差/音乐记忆及调度时钟。目标从更新前轨迹stop-gradient构造；真实终止
令continuation为0，行政/BPTT截断保留bootstrap。独立真实MC仅做预热及校准。
"""

import torch
from torch import nn
from supermimic.networks import Critic as SuperMimicCritic


class AlternatingCritic(SuperMimicCritic):
    """原SM归一化MLP加可微回报上界；不改变2835维特权输入或Adapter观测。

    当前奖励除Tracking外均非正，因此每步最大Tracking/(1-gamma)是真实回报上界。
    使用upper-softplus(upper-raw)保持负值无界，避免硬clamp或错误直通梯度；原始
    网络输出在零附近时导数接近1。末层零权重及解析bias使初始价值为零。归一化、
    网络和上界均进入state_dict，只有独立Critic优化器更新该网络。
    """
    def __init__(self, dimension, upper_bound):
        import math
        super().__init__(dimension)
        if not math.isfinite(upper_bound) or upper_bound <= 0:
            raise ValueError('Critic真实回报上界必须为有限正数')
        self.register_buffer('value_upper_bound', torch.tensor(float(upper_bound)))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -math.log(-math.expm1(-upper_bound)))

    def forward(self, x):
        raw = super().forward(x)
        return self.value_upper_bound-torch.nn.functional.softplus(self.value_upper_bound-raw)


class AdapterCritic(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dimension, 512), nn.SiLU(), nn.Linear(512, 256), nn.SiLU(), nn.Linear(256, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def lambda_returns(reward, continuation, values, gamma=.99, lam=.95):
    if values.shape[0] != reward.shape[0]+1 or continuation.shape != reward.shape:
        raise ValueError("lambda-return需要H奖励/H存活/H+1价值")
    result = []
    with torch.no_grad():
        target = values[-1]
        for t in range(len(reward)-1, -1, -1):
            target = reward[t] + gamma*continuation[t]*((1-lam)*values[t+1]+lam*target)
            result.append(target)
    return torch.stack(result[::-1])


def calibration(predicted, actual, near_failure=None, *, constant_value=None):
    error = (predicted-actual).square()
    variance = actual.var(unbiased=False)
    ev = float(1-(predicted-actual).var(unbiased=False)/variance) if float(variance) > 1e-12 else None
    constant_value = actual.mean() if constant_value is None else constant_value
    baseline = (actual-constant_value).square()
    return {"mse": float(error.mean()), "explained_variance": ev,
            "constant_baseline_mse": float(baseline.mean()), "constant_value": float(constant_value),
            "near_failure_mse": float(error[near_failure].mean()) if near_failure is not None and near_failure.any() else None,
            "near_failure_constant_mse": float(baseline[near_failure].mean()) if near_failure is not None and near_failure.any() else None,
            "near_failure_samples": int(near_failure.sum()) if near_failure is not None else 0,
            "samples": actual.numel(), "target": "independent_real_discounted_return"}
