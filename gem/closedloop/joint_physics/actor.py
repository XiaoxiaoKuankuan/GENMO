"""统一监督和自由采样的JointPhysicsActor，未来条件同时进入两条梯度路径。

监督只复用Stage1的归一化、加噪、逐坐标约束及Transformer，增加独立FutureEncoder残差。
音乐dropout不改变历史、前缀或未来条件；无缓存的context贡献严格为零。随机噪声、等级和
实际mask随输出返回用于配对评价。完整自由生成复用原20步DDIM，临时eval后恢复每个子模块
原状态；既不持有SM/WM，也不读取contact/地面标签作为条件。
bf16_mixed只覆盖条件编码和Transformer；归一化、扩散加噪及对外输出保持FP32，
不会改变参数存储dtype。FutureEncoder将不同请求时刻的样本一次合批，避免逐样本GPU同步。
"""

from contextlib import contextmanager

import torch
from torch import nn

from gem.closedloop.contracts import validate_stage1_training_batch
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
        self.precision = "fp32"
        self.initialization = "static_config_random_no_genmo_checkpoint"

    def network_autocast(self, device):
        if self.precision not in ("fp32", "bf16_mixed"):
            raise ValueError("JointPhysicsActor precision仅支持fp32或bf16_mixed")
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                              enabled=self.precision == "bf16_mixed")

    def future_residual(self, batch, contexts):
        times = batch["future_times"]
        b = len(batch["decision_time"])
        if contexts is None:
            features = batch["music_features"].new_zeros((b, 58))
            valid = torch.zeros(b, dtype=torch.bool, device=times.device)
            request_ticks = (batch["decision_time"] * 600).round()
            forecast_ticks = request_ticks
        else:
            if len(contexts) != b:
                raise ValueError("ForecastContext必须按样本对应")
            if any(c.features.shape != (1, 58) or c.valid.shape != (1,) for c in contexts):
                raise ValueError("训练ForecastContext必须逐样本保存[1,58]及[1]valid")
            # Python只整理已有CPU时刻元数据；网络和CUDA时刻校验均按整批执行。
            features = torch.cat([c.features for c in contexts], dim=0)
            valid = torch.cat([c.valid for c in contexts], dim=0)
            request_ticks = times.new_tensor([c.request_tick for c in contexts])
            forecast_ticks = times.new_tensor([c.forecast_tick for c in contexts])
            if bool(((batch["decision_time"] * 600 - request_ticks).abs() > 1e-5).any()):
                raise ValueError("Stage1秒时间与600Hz请求时间不一致")
        return self.future_encoder.forward_batch(features, valid, request_ticks, forecast_ticks, times)

    def denoise_supervised(self, batch, forecast_context=None, timesteps=None, noise=None,
                           drop_music_mask=None):
        # 即使调用者误套全局autocast，物理表示、归一化和扩散日程也保持FP32。
        with torch.autocast(device_type=batch["target_qpos30"].device.type, enabled=False):
            return self._denoise_supervised_fp32(batch, forecast_context, timesteps, noise,
                                                drop_music_mask)

    def _denoise_supervised_fp32(self, batch, forecast_context, timesteps, noise, drop_music_mask):
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
        with self.network_autocast(target.device):
            condition = base.encode_conditions(adapted, drop_music_mask)
            future = self.future_residual(batch, forecast_context)
            condition = condition + torch.where(adapted["future_valid"][..., None], future, 0.)
            output = base._denoise(xt, timesteps, adapted, condition)
        # float()保留输入/参数导数，供FP32 Stage1损失、codec、FK和SM/WM执行链使用。
        output = {name: value.float() for name, value in output.items()}
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
