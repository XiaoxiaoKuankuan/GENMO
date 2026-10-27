"""统一监督和自由采样的JointPhysicsActor，未来条件同时进入两条梯度路径。

监督只复用Stage1的归一化、加噪、逐坐标约束及Transformer，增加独立FutureEncoder残差。
音乐dropout不改变历史、前缀或未来条件；无缓存的context贡献严格为零。随机噪声、等级和
实际mask随输出返回用于配对评价。完整自由生成复用原20步DDIM，临时eval后恢复每个子模块
原状态；既不持有SM/WM，也不读取contact/地面标签作为条件。
"""

from contextlib import contextmanager

import torch
from torch import nn

from gem.closedloop.contracts import validate_stage1_training_batch
from gem.closedloop.mb_control.contracts import ForecastContext
from gem.closedloop.mb_control.future_encoder import FutureStateEncoder


@contextmanager
def sampling_mode(module):
    states = [(child, child.training) for child in module.modules()]
    try:
        module.eval()
        yield
    finally:
        for child, training in states:
            child.training = training


class JointPhysicsActor(nn.Module):
    def __init__(self, stage1):
        super().__init__()
        self.stage1 = stage1
        self.future_encoder = FutureStateEncoder(stage1.denoiser.latent_dim)
        self.batch_cfg = True
        self.fast_forecast = False
        self.initialization = "static_config_random_no_genmo_checkpoint"

    def future_residual(self, batch, contexts):
        # 原ForecastContext的时刻是标量；逐样本调用支持一个batch内的不同全局时刻。
        if contexts is None:
            contexts = [ForecastContext.invalid(batch["music_features"][i:i+1],
                        round(float(batch["decision_time"][i]) * 600)) for i in range(len(batch["decision_time"]))]
        if len(contexts) != len(batch["decision_time"]):
            raise ValueError("ForecastContext必须按样本对应")
        residuals = []
        for i, context in enumerate(contexts):
            if abs(float(batch["decision_time"][i]) * 600 - context.request_tick) > 1e-5:
                raise ValueError("Stage1秒时间与600Hz请求时间不一致")
            residuals.append(self.future_encoder(context, batch["future_times"][i:i+1]))
        return torch.cat(residuals)

    def denoise_supervised(self, batch, forecast_context=None, timesteps=None, noise=None,
                           drop_music_mask=None):
        base = self.stage1
        validate_stage1_training_batch(batch, history_steps=50)
        if batch["target_qpos30"].shape[1:] != (60, 30):
            raise ValueError("新流程实际监督窗口必须为60帧")
        adapted = base.adapt_conditions(batch)
        valid = batch["target_qpos30_valid"]
        target = torch.where(valid, base.endecoder.normalize(
            torch.where(valid, batch["target_qpos30"], 0.).to(adapted["known_x"])), 0.)
        b = len(target)
        timesteps = torch.randint(1000, (b,), device=target.device) if timesteps is None else timesteps
        if timesteps.shape != (b,) or timesteps.dtype != torch.long or bool(((timesteps < 0) | (timesteps >= 1000)).any()):
            raise ValueError("噪声等级必须是完整1000步日程内的int64[B]")
        noise = torch.randn_like(target) if noise is None else noise.to(target)
        if noise.shape != target.shape or not torch.isfinite(noise).all():
            raise ValueError("noise必须是有限[B,60,30]")
        xt = base.train_diffusion.q_sample(target, timesteps, noise=noise)
        xt = base._constrain(torch.where(valid, xt, noise), adapted)
        if drop_music_mask is None:
            drop_music_mask = torch.rand(b, device=target.device) < (base.music_mask_prob if self.training else 0.)
        condition = base.encode_conditions(adapted, drop_music_mask)
        future = self.future_residual(batch, forecast_context)
        condition = condition + torch.where(adapted["future_valid"][..., None], future, 0.)
        output = base._denoise(xt, timesteps, adapted, condition)
        return {**output, "target_x_start": target.detach(), "xt": xt, "timesteps": timesteps,
                "noise": noise, "music_dropout_mask": drop_music_mask,
                "known_mask": adapted["known_mask"], "unknown_mask": valid & ~adapted["known_mask"]}

    def forward(self, batch, forecast_context=None, physics_inputs=None, future_inputs=None, **kwargs):
        data = self.denoise_supervised(batch, forecast_context, **kwargs)
        future = self.denoise_supervised(**future_inputs) if future_inputs is not None else None
        physics = None
        if physics_inputs is not None:
            # 冻结网络随机dropout，但保持学生参数和输入的物理导数。
            with sampling_mode(self):
                physics = self.denoise_supervised(**physics_inputs)
        if future_inputs is not None:
            return data, physics, future
        return (data, physics) if physics_inputs is not None else data

    def sample_differentiable(self, conditions, context, noise):
        from gem.closedloop.mb_control.sampler import sample_differentiable
        with sampling_mode(self):
            return sample_differentiable(self, conditions, context, noise,
                guidance_scale=2.5, activation_checkpointing=False)

    def generate(self, state, ticket, context, noise):
        from gem.closedloop.mb_control.online_conditions import build_conditions
        from gem.closedloop.mb_control.contracts import GeneratedPlan
        conditions, anchor = build_conditions(state, ticket, self.stage1.endecoder.codec)
        out = self.sample_differentiable(conditions, context, noise)
        return GeneratedPlan(ticket, self.stage1.endecoder.codec.apply_world_anchor(out["qpos28"], anchor), out["contact"])
