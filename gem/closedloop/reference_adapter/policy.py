"""严格三隐藏层、零初始化的十帧参考策略。

输入由InputSpec具名打包；512/512/256三个SiLU隐藏层直接产生270维，逐帧有界映射为
根平移、旋转向量及RobotSpec序21关节。末层零初始化保障零Adapter，旋转额外限制
范数；没有插值、广播、循环网络或在线WM预瞄。Actor仍执行一个21维归一化控制动作。
"""

import torch
from torch import nn

from .contracts import InputSpec


class ReferenceAdapter(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.input_spec = InputSpec(tuple(spec.joint_names))
        self.net = nn.Sequential(nn.Linear(self.input_spec.dimension, 512), nn.SiLU(),
                                 nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, 256), nn.SiLU(),
                                 nn.Linear(256, 270))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        bounds = [.03, .03, .02, .05, .05, .05]
        bounds += [.05 if "ankle" in name else .10 for name in spec.joint_names]
        self.register_buffer("bounds", torch.tensor(bounds))

    def forward(self, fields):
        x = self.input_spec.flatten(fields)
        d = self.net(x).reshape(len(x), 10, 27).tanh() * self.bounds
        rotation = d[..., 3:6]
        rotation = rotation / (rotation.norm(dim=-1, keepdim=True) / .05).clamp_min(1.)
        return torch.cat((d[..., :3], rotation, d[..., 6:]), -1)
