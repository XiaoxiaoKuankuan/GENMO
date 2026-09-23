"""BUMI Stage 1 可复用条件 Actor：音乐、因果 proprio48 历史和 physical-qpos30 前缀。

本模块只负责条件适配、原 GENMO Transformer 的残差条件和条件扩散，不持有优化器或
监督损失。第一阶段训练和后续上层 Actor 可复用同一实例。外部字段直接沿用第 3 步契约，
physical qpos30 仅通过 BumiEndecoder.normalize 映射到网络 x0 域；未知占位在标准化前后
均由逐坐标 mask 隔离。历史采用独立固定物理尺度和带相对时间的掩码 GRU，不读取任何
数据集统计或 GMT normalizer。默认尺度 1 表示显式保留各字段的原物理单位，不冒充统计量。

训练和 DDIM 每一步都将已知坐标保持为干净 x0，只有未知部分参与扩散。CFG 只切换音乐，
历史和前缀残差只计算一次并共用于两个分支。contact2 沿用原独立 head，永不作为条件。
最终 physical 输出再次按输入前缀组合，避免 normalize/denormalize 舍入改变承诺值；
qpos28 与 FK 继续使用原 codec，历史不拼入 120 帧动作时间轴。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from gem.closedloop.contracts import (
    PROPRIO_CONTRACT_VERSION,
    STAGE1_CONDITION_KEYS,
    STAGE1_CONTRACT_VERSION,
    validate_stage1_condition_batch,
    validate_stage1_training_batch,
)
from gem.diffusion_utils import gaussian_diffusion as gd
from gem.diffusion_utils.respace import SpacedDiffusion, space_timesteps
from gem.robots.bumi.endecoder import BumiEndecoder

STAGE1_ACTOR_INTERFACE_VERSION = "genmo.bumi_closedloop.actor.v1"


class FixedProprioNormalizer(nn.Module):
    """按四个物理字段除以显式尺度；不是 empirical mean/std，也不拟合样本。"""

    def __init__(self, scales: Sequence[float]) -> None:
        super().__init__()
        if len(scales) != 4 or any(not math.isfinite(s) or s <= 0 for s in scales):
            raise ValueError("proprio_scales requires four finite positive physical scales")
        scale = torch.cat([torch.full((n,), float(s)) for n, s in zip((3, 3, 21, 21), scales)])
        if not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
            raise ValueError("proprio_scales must remain finite and positive in float32")
        self.register_buffer("scale", scale, persistent=False)

    def forward(self, values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return torch.where(valid[..., None], values, 0.0) / self.scale.to(values)


class TemporalHistoryEncoder(nn.Module):
    """顺序处理有效历史与 decision-relative 秒数，无效槽跳过更新，全空历史严格为零。"""

    def __init__(self, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.cell = nn.GRUCell(49, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, latent_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, history, valid, relative_times):
        state = history.new_zeros((history.shape[0], self.cell.hidden_size))
        # 时间与数值同时隔离；即使残差投影已经训练，无效槽也不改变下一有效状态。
        features = torch.cat((history, relative_times[..., None]), dim=-1)
        features = torch.where(valid[..., None], features, 0.0)
        for index in range(history.shape[1]):
            candidate = self.cell(features[:, index], state)
            state = torch.where(valid[:, index, None], candidate, state)
        residual = self.out_proj(state)
        return torch.where(valid.any(dim=1, keepdim=True), residual, 0.0)


class PrefixEncoder(nn.Module):
    """逐帧读取 30 维 normalized 前缀及 30 维坐标 mask，以轻量投影提供残差。"""

    def __init__(self, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(60, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, latent_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, known_x, known_mask):
        features = torch.cat((torch.where(known_mask, known_x, 0.0), known_mask.to(known_x)), -1)
        residual = self.out_proj(torch.nn.functional.silu(self.input_proj(features)))
        return torch.where(known_mask.any(dim=-1, keepdim=True), residual, 0.0)


class Stage1Actor(nn.Module):
    """保留原音乐 MLP、存在标志 MLP、NetworkEncoderRoPE 和 qpos30/contact2 heads。"""

    def __init__(
        self,
        endecoder: BumiEndecoder,
        denoiser: nn.Module,
        music_embedder: nn.Module,
        cond_exists_embedder: nn.Module | None = None,
        history_hidden_dim: int = 128,
        history_steps: int = 50,
        proprio_scales: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
        music_mask_prob: float = 0.1,
        diffusion_steps: int = 1000,
        noise_schedule: str = "cosine",
    ) -> None:
        super().__init__()
        if not isinstance(endecoder, BumiEndecoder):
            raise TypeError("Stage1Actor requires the existing BumiEndecoder")
        if history_steps <= 0 or history_hidden_dim <= 0:
            raise ValueError("history_steps and history_hidden_dim must be positive")
        if not 0.0 <= music_mask_prob <= 1.0:
            raise ValueError("music_mask_prob must lie in [0,1]")
        if int(diffusion_steps) != 1000:
            raise ValueError("Stage 1 preserves the original 1000-step GENMO diffusion schedule")
        if (
            denoiser.output_dim != 30
            or denoiser.max_len != 120
            or denoiser.encode_text
            or denoiser.input_remove_condition
            or denoiser.input_remove_global
            or denoiser.avgbeta
            or not isinstance(denoiser.static_conf_head, nn.Module)
            or denoiser.static_conf_head.fc2.out_features != 2
            or denoiser.add_cond_linear.in_features != 30 + denoiser.latent_dim
        ):
            raise ValueError("Stage1Actor requires native music qpos30/contact2 NetworkEncoderRoPE")
        self.endecoder = endecoder
        self.denoiser = denoiser
        self.music_embedder = music_embedder
        self.cond_exists_embedder = cond_exists_embedder
        self.history_steps = int(history_steps)
        self.music_mask_prob = float(music_mask_prob)
        self.proprio_normalizer = FixedProprioNormalizer(proprio_scales)
        self.history_encoder = TemporalHistoryEncoder(history_hidden_dim, denoiser.latent_dim)
        self.prefix_encoder = PrefixEncoder(history_hidden_dim, denoiser.latent_dim)
        self.diffusion_steps = int(diffusion_steps)
        self.noise_schedule = str(noise_schedule)
        self.train_diffusion = self._diffusion(self.diffusion_steps)
        self.interface_config = {
            "interface_version": STAGE1_ACTOR_INTERFACE_VERSION,
            "condition_contract_version": STAGE1_CONTRACT_VERSION,
            "proprio_contract_version": PROPRIO_CONTRACT_VERSION,
            "history_steps": self.history_steps,
            "history_hidden_dim": int(history_hidden_dim),
            "proprio_normalization": "fixed_physical_scale_no_empirical_statistics",
            "proprio_scales": list(map(float, proprio_scales)),
            "music_mask_prob": self.music_mask_prob,
            "diffusion_steps": self.diffusion_steps,
            "noise_schedule": self.noise_schedule,
            "known_policy": "clean_x0_every_step_coordinate_mask",
            "cfg_policy": "music_only_dropout_shared_history_and_prefix",
            "motion_frames": 120,
            "motion_fps": 30,
            "contact_condition": False,
            # heads/dropout 等不改变张量形状，必须独立绑定，避免 strict=True 仍加载错架构。
            "backbone": {
                "class": type(denoiser).__name__,
                "latent_dim": int(denoiser.latent_dim),
                "num_layers": int(denoiser.num_layers),
                "num_heads": int(denoiser.num_heads),
                "dropout": float(denoiser.dropout),
                "allow_autoregressive": bool(denoiser.allow_autoregressive),
            },
            "condition_modules": {
                "music": self._module_interface(music_embedder),
                "exists": self._module_interface(cond_exists_embedder),
            },
        }

    @staticmethod
    def _module_interface(module):
        """绑定无参数激活和 dropout，补充 state_dict 无法表达的音乐编码行为。"""
        if module is None:
            return None
        result = []
        for name, item in module.named_modules():
            entry = {"name": name, "class": type(item).__name__}
            if isinstance(item, nn.Dropout):
                entry["probability"] = float(item.p)
            if isinstance(item, nn.GELU):
                entry["approximate"] = item.approximate
            if isinstance(item, nn.LayerNorm):
                entry["eps"] = float(item.eps)
            result.append(entry)
        return result

    def _diffusion(self, steps: int) -> SpacedDiffusion:
        return SpacedDiffusion(
            use_timesteps=space_timesteps(self.diffusion_steps, str(steps)),
            betas=gd.get_named_beta_schedule(self.noise_schedule, self.diffusion_steps, 1.0),
            model_mean_type=gd.ModelMeanType.START_X,
            model_var_type=gd.ModelVarType.FIXED_SMALL,
            loss_type=gd.LossType.MSE,
            rescale_timesteps=False,
        )

    def adapt_conditions(self, conditions: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        """只选择既有十个条件字段；不查询 target、contact 标签或训练 metadata。"""
        selected = {key: conditions[key] for key in STAGE1_CONDITION_KEYS}
        validate_stage1_condition_batch(selected, history_steps=self.history_steps)
        dtype = next(self.denoiser.parameters()).dtype
        future_valid = selected["future_valid"]
        known_mask = selected["known_qpos30_mask"]
        physical = selected["known_qpos30"].to(dtype=dtype)
        # 先隔离任意有限未知占位，normalize 后再遮罩非零 mean 导致的未知数值。
        known_x = self.endecoder.normalize(torch.where(known_mask, physical, 0.0))
        known_x = torch.where(known_mask, known_x, 0.0)
        music_valid = selected["music_valid"] & future_valid
        music_embed = torch.where(
            music_valid[..., None], selected["music_features"].to(dtype=dtype), 0.0
        )
        history_valid = selected["proprio_history_valid"]
        history = self.proprio_normalizer(
            selected["proprio_history"].to(dtype=dtype), history_valid
        )
        relative = selected["proprio_history_times"] - selected["decision_time"][:, None]
        return {
            "music_embed": music_embed,
            "music_valid": music_valid,
            "known_physical": selected["known_qpos30"],
            "known_x": known_x,
            "known_mask": known_mask,
            "history": history,
            "history_valid": history_valid,
            "history_relative_times": torch.where(history_valid, relative, 0.0).to(dtype),
            "future_valid": future_valid,
            "future_relative_times": (
                selected["future_times"] - selected["decision_time"][:, None]
            ).to(dtype),
        }

    def _residual(self, adapted):
        history = self.history_encoder(
            adapted["history"], adapted["history_valid"], adapted["history_relative_times"]
        )
        prefix = self.prefix_encoder(adapted["known_x"], adapted["known_mask"])
        return torch.where(adapted["future_valid"][..., None], history[:, None] + prefix, 0.0)

    def _music_condition(self, adapted, drop_music=None):
        valid = adapted["music_valid"]
        music = self.music_embedder(adapted["music_embed"])
        music = torch.where(valid[..., None], music, 0.0)
        if drop_music is not None:
            if drop_music.shape != (music.shape[0],) or drop_music.dtype != torch.bool:
                raise ValueError("drop_music must be bool [B]")
            valid = valid & ~drop_music[:, None]
            music = torch.where(valid[..., None], music, 0.0)
        if self.cond_exists_embedder is not None:
            music = self.cond_exists_embedder(torch.cat((music, valid[..., None].to(music)), -1))
        return torch.where(adapted["future_valid"][..., None], music, 0.0)

    def encode_conditions(self, adapted, drop_music=None):
        return self._music_condition(adapted, drop_music) + self._residual(adapted)

    @staticmethod
    def _constrain(value, adapted):
        value = torch.where(adapted["known_mask"], adapted["known_x"], value)
        return torch.where(adapted["future_valid"][..., None], value, 0.0)

    def _denoise(self, xt, timesteps, adapted, f_cond):
        xt = self._constrain(xt, adapted)
        # 全 padding 条件也允许推理：开放一个零 sentinel key，随后把该样本全部输出屏蔽。
        # 原 Transformer 不接受全 mask softmax，长度 clamp 不会影响任何有效样本。
        length = adapted["future_valid"].sum(dim=1).clamp_min(1)
        output = self.denoiser(
            xt,
            timesteps,
            y={"f_cond": f_cond, "length": length},
            inputs={},
        )
        return {
            "pred_x_start": self._constrain(output["pred_x_start"], adapted),
            "static_conf_logits": torch.where(
                adapted["future_valid"][..., None], output["static_conf_logits"], 0.0
            ),
        }

    def training_forward(self, batch, timesteps=None, noise=None):
        validate_stage1_training_batch(batch, history_steps=self.history_steps)
        adapted = self.adapt_conditions(batch)
        valid = batch["target_qpos30_valid"]
        physical = torch.where(valid, batch["target_qpos30"], 0.0).to(adapted["known_x"])
        target_x = torch.where(valid, self.endecoder.normalize(physical), 0.0)
        if timesteps is None:
            timesteps = torch.randint(
                self.diffusion_steps, (target_x.shape[0],), device=target_x.device
            )
        if timesteps.shape != (target_x.shape[0],) or timesteps.dtype != torch.long:
            raise ValueError("timesteps must be int64 [B]")
        if bool(((timesteps < 0) | (timesteps >= self.diffusion_steps)).any()):
            raise ValueError("timesteps outside diffusion schedule")
        noise = torch.randn_like(target_x) if noise is None else noise.to(target_x)
        if noise.shape != target_x.shape or not bool(torch.isfinite(noise).all()):
            raise ValueError("noise must be finite [B,120,30]")
        xt = self.train_diffusion.q_sample(target_x, timesteps, noise=noise)
        # 无 halo 的未知末端位移没有 x0 标签；其输入独立于占位标签，仍可被模型生成。
        xt = torch.where(valid, xt, noise)
        xt = self._constrain(xt, adapted)
        drop_music = torch.zeros(target_x.shape[0], device=target_x.device, dtype=torch.bool)
        if self.training and self.music_mask_prob:
            drop_music = (
                torch.rand(target_x.shape[0], device=target_x.device) < self.music_mask_prob
            )
        output = self._denoise(xt, timesteps, adapted, self.encode_conditions(adapted, drop_music))
        return {**output, "target_x_start": target_x, "xt": xt, "music_dropout_mask": drop_music}

    def forward(self, batch, timesteps=None, noise=None):
        """监督去噪的 nn.Module 入口；纯条件推理显式调用 sample。"""
        return self.training_forward(batch, timesteps=timesteps, noise=noise)

    @torch.no_grad()
    def sample(self, conditions, steps=20, guidance_scale=2.5, noise=None, return_trace=False):
        if self.training:
            raise RuntimeError("call actor.eval() before sampling")
        if isinstance(steps, bool) or int(steps) != steps or not 2 <= steps <= self.diffusion_steps:
            raise ValueError("DDIM steps must be an integer in [2,1000]")
        if not math.isfinite(guidance_scale):
            raise ValueError("guidance_scale must be finite")
        adapted = self.adapt_conditions(conditions)
        xt = torch.randn_like(adapted["known_x"]) if noise is None else noise.to(adapted["known_x"])
        if xt.shape != adapted["known_x"].shape or not bool(torch.isfinite(xt).all()):
            raise ValueError("noise must be finite [B,120,30]")
        xt = self._constrain(xt, adapted)
        residual = self._residual(adapted)
        conditional = self._music_condition(adapted) + residual
        drop_all = torch.ones(xt.shape[0], dtype=torch.bool, device=xt.device)
        unconditional = self._music_condition(adapted, drop_all) + residual

        def denoise_guided(value, timestep, **_kwargs):
            cond = self._denoise(value, timestep, adapted, conditional)
            if guidance_scale == 1.0:
                return cond
            uncond = self._denoise(value, timestep, adapted, unconditional)
            # 沿用原 CFG 的 head 插值；两个分支的唯一条件差异是音乐可见性。
            guided = uncond["pred_x_start"] + guidance_scale * (
                cond["pred_x_start"] - uncond["pred_x_start"]
            )
            logits = uncond["static_conf_logits"] + guidance_scale * (
                cond["static_conf_logits"] - uncond["static_conf_logits"]
            )
            return {
                "pred_x_start": self._constrain(guided, adapted),
                "static_conf_logits": logits,
            }

        diffusion = self._diffusion(int(steps))
        trace = []
        for index in range(diffusion.num_timesteps - 1, -1, -1):
            t = torch.full((xt.shape[0],), index, dtype=torch.long, device=xt.device)
            output = diffusion.ddim_sample(
                denoise_guided, xt, t, clip_denoised=False, model_kwargs={"y": {}}, eta=0.0
            )
            xt = self._constrain(output["sample"], adapted)
            if return_trace:
                trace.append(xt.clone())
        physical = self.endecoder.denormalize(xt)
        # 物理空间逐坐标精确回写，不以浮点标准化往返误差改变输入承诺。
        physical = torch.where(adapted["known_mask"], adapted["known_physical"], physical)
        physical = torch.where(adapted["future_valid"][..., None], physical, 0.0)
        qpos = self.endecoder.codec.decode_to_canonical_qpos(physical)
        qpos = torch.where(adapted["future_valid"][..., None], qpos, 0.0)
        logits = output["static_conf_logits"]
        contact = torch.where(adapted["future_valid"][..., None], logits.sigmoid(), 0.0)
        result = {
            "qpos30": physical,
            "contact": contact,
            "contact_logits": logits,
            "normalized": xt,
            "qpos": qpos,
        }
        if return_trace:
            result["trace"] = trace
        return result
