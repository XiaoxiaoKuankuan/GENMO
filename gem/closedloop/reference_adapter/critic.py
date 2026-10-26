"""增强状态新Critic与按50Hz步定义的SM lambda-return。

网络独立新建512/256两隐藏层，不导入旧GENMO价值参数；输入含状态、xi、实际历史、
原参考、残差/音乐记忆及调度时钟。目标从更新前轨迹stop-gradient构造；真实终止
令continuation为0，行政/BPTT截断保留bootstrap。独立真实MC仅做预热及校准。
"""

import torch
from torch import nn


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
