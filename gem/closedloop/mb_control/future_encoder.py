"""58维独立未来条件编码，末层零初始化以保持现有Stage1输出。

状态误差与提前量输入58→128，再加入每动作帧相对预测时刻的时间特征，最后投影到
原latent_dim。invalid分支严格为零，时间特征也不旁路零投影；音乐CFG两支共享输出。
这不是未来真实历史，不接收监督target或未来实测延迟。
"""

import torch
from torch import nn


class FutureStateEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.input = nn.Linear(58, 128)
        self.time = nn.Linear(1, 128, bias=False)
        self.output = nn.Linear(128, latent_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, context, future_times):
        if context.features.shape != (len(future_times), 58) or context.valid.dtype != torch.bool or context.valid.shape != (len(future_times),):
            raise ValueError("ForecastContext需要[B,58]及bool valid")
        if context.forecast_tick < context.request_tick:
            raise ValueError("预测时间早于请求")
        features = torch.where(context.valid[:, None], context.features, 0.)
        relative = (future_times - context.forecast_tick / 600).to(features)
        hidden = self.input(features)[:, None] + self.time(relative[..., None])
        output = self.output(torch.nn.functional.silu(hidden))
        return torch.where(context.valid[:, None, None], output, 0.)
