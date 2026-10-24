"""现有Stage1 Actor的轻量封装，只新增独立未来条件编码。

输入仍为十字段条件与单独ForecastContext，输出仍为60帧qpos30/contact2。监督调用
直接沿用原Stage1 training_forward，未来分支等价valid=false贡献零，不改配对库。
GENMO在BPTT中使用eval关闭dropout但保留参数梯度；部署在外层显式no_grad。
"""

import torch
from torch import nn

from .future_encoder import FutureStateEncoder
from .sampler import sample_differentiable
from .contracts import GeneratedPlan
from .online_conditions import build_conditions


class ForecastActor(nn.Module):
    def __init__(self, stage1, *, activation_checkpointing=True, guidance_scale=2.5):
        super().__init__()
        self.stage1 = stage1
        self.future_encoder = FutureStateEncoder(stage1.denoiser.latent_dim)
        self.activation_checkpointing = activation_checkpointing
        self.guidance_scale = guidance_scale

    def sample_differentiable(self, conditions, context, noise):
        self.eval()
        return sample_differentiable(self, conditions, context, noise, guidance_scale=self.guidance_scale,
                                     activation_checkpointing=self.activation_checkpointing)

    def generate(self, state, ticket, context, noise):
        with torch.profiler.record_function("mb/conditions"):
            conditions, anchor = build_conditions(state, ticket, self.stage1.endecoder.codec)
        with torch.profiler.record_function("mb/ddim20_and_decode"):
            output = self.sample_differentiable(conditions, context, noise)
        with torch.profiler.record_function("mb/reference_anchor"):
            world = self.stage1.endecoder.codec.apply_world_anchor(output["qpos28"], anchor)
        return GeneratedPlan(ticket, world, output["contact"])

    def original_supervision(self, batch):
        mode = self.stage1.training
        try:
            self.stage1.train(True)
            return self.stage1.training_forward(batch)
        finally:
            self.stage1.train(mode)

    @torch.no_grad()
    def sample(self, conditions, context, noise):
        return self.sample_differentiable(conditions, context, noise)
