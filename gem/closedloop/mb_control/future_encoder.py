"""58维独立未来条件编码，末层零初始化以保持现有Stage1输出。

状态误差与提前量输入58→128，再加入每动作帧相对预测时刻的时间特征，最后投影到
原latent_dim。invalid分支严格为零，时间特征也不旁路零投影；音乐CFG两支共享输出。
这不是未来真实历史，不接收监督target或未来实测延迟。批量训练接口显式接受每个样本
各自的请求/预测时刻，合批只合并网络计算，不共享任何样本的时钟或valid。
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
        # 在线采样仍沿用单次请求的标量时刻，训练使用下面的独立时刻向量接口。
        request_ticks = future_times.new_full((len(future_times),), context.request_tick)
        forecast_ticks = future_times.new_full((len(future_times),), context.forecast_tick)
        return self.forward_batch(context.features, context.valid, request_ticks, forecast_ticks,
                                  future_times)

    def forward_batch(self, features, valid, request_ticks, forecast_ticks, future_times):
        b = len(future_times)
        if features.shape != (b, 58) or valid.dtype != torch.bool or valid.shape != (b,):
            raise ValueError("ForecastContext需要[B,58]及bool valid")
        if request_ticks.shape != (b,) or forecast_ticks.shape != (b,):
            raise ValueError("每个ForecastContext必须保留独立的request_tick和forecast_tick")
        if bool((forecast_ticks < request_ticks).any()):
            raise ValueError("预测时间早于请求")
        features = torch.where(valid[:, None], features, 0.)
        relative = (future_times - forecast_ticks[:, None].to(future_times) / 600).to(features)
        hidden = self.input(features)[:, None] + self.time(relative[..., None])
        output = self.output(torch.nn.functional.silu(hidden))
        return torch.where(valid[:, None, None], output, 0.)
