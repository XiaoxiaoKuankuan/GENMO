"""第九步 DPPO 随机扩散策略：复用原 Stage1Actor，显式记录与重算内部决策概率。

本模块不复制 Transformer、不持有优化器，也不改变原确定性 Actor.sample 评估入口。
训练策略在原 1000 步 cosine/x0 预测日程中抽取默认 20 个时间点，以 music-only CFG
计算 DDIM 均值；默认 eta=0.1，实际采样标准差具有 0.001 下界，最后一步同样加入
真实高斯噪声。均值中的方向系数使用未加下界的基础 sigma，下界只改变实际采样分布。

概率随机变量始终是解码前的 normalized qpos30。已承诺逐坐标前缀每步固定为干净 x0，
padding 每步固定为零，两者均不进入高斯概率；contact2 仍是确定性辅助 head，不伪造
其动作概率。模型与链使用 FP32，联合概率在 FP64 中对全部自由坐标求和，不取平均、
不裁剪逐坐标概率，也不对物理解码后的 qpos 或执行轨迹套用高斯。

收集与概率重算都显式采用 eval 模式并关闭 autocast；eval 不会关闭参数梯度。采样只
保存脱离计算图的条件快照和完整链，重算把前后状态视为固定数据，支持逐去噪步或混合
去噪步 microbatch，避免让梯度穿过历史采样、GMT 或仿真。全 padding 条件可收集为
零自由度诊断样本，概率为零；是否排除 Actor 更新由 rollout buffer/trainer 决定。
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from typing import Any

import torch

from gem.closedloop.actor import Stage1Actor
from gem.closedloop.contracts import STAGE1_CONDITION_KEYS

DPPO_KERNEL_VERSION = "genmo.bumi_closedloop.stochastic_ddim_joint_sum.v1"


def masked_joint_log_prob(
    value: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    free_mask: torch.Tensor,
) -> torch.Tensor:
    """在自由 normalized 坐标上计算对角 Gaussian 的 FP64 联合 log probability。"""
    if value.shape != mean.shape or free_mask.shape != mean.shape or mean.ndim != 3:
        raise ValueError("value, mean and free_mask must have matching [B,T,D] shapes")
    if free_mask.dtype != torch.bool:
        raise TypeError("free_mask must be boolean")
    if std.shape not in ((mean.shape[0], 1, 1), mean.shape):
        raise ValueError("std must be [B,1,1] or match mean")
    if not bool(torch.isfinite(std).all()) or bool((std <= 0).any()):
        raise ValueError("Gaussian std must be finite and strictly positive")
    # 先在相减前隔离确定性坐标，避免它们影响概率或产生无意义的大数。
    observed = torch.where(free_mask, value.detach(), 0.0).to(torch.float64)
    expected = torch.where(free_mask, mean, 0.0).to(torch.float64)
    scale = std.to(torch.float64)
    terms = -0.5 * ((observed - expected) / scale).square() - scale.log()
    terms = terms - 0.5 * math.log(2.0 * math.pi)
    return torch.where(free_mask, terms, 0.0).sum(dim=(-2, -1))


class DPPODiffusionPolicy:
    """Stage1Actor 的随机核适配器；step_index 是实际去噪顺序 0..K-1。"""

    def __init__(
        self,
        actor: Stage1Actor,
        *,
        steps: int = 20,
        eta: float = 0.1,
        std_floor: float = 0.001,
        guidance_scale: float = 2.5,
        cfg_batch: bool = False,
        std_schedule: list[float] | tuple[float, ...] | None = None,
    ) -> None:
        if not isinstance(actor, Stage1Actor):
            raise TypeError("DPPO policy requires the existing Stage1Actor")
        if isinstance(steps, bool) or not isinstance(steps, int) or not 2 <= steps <= 1000:
            raise ValueError("steps must be an integer in [2,1000]")
        if not math.isfinite(eta) or not 0.0 <= eta <= 1.0:
            raise ValueError("eta must lie in [0,1]")
        if not math.isfinite(std_floor) or std_floor <= 0.0:
            raise ValueError("std_floor must be finite and strictly positive")
        if not math.isfinite(guidance_scale):
            raise ValueError("guidance_scale must be finite")
        if type(cfg_batch) is not bool:
            raise TypeError("cfg_batch must be boolean")
        if std_schedule is not None and (len(std_schedule) != steps or any(
            isinstance(value, bool) or not math.isfinite(value) or value < std_floor for value in std_schedule
        )):
            raise ValueError("std_schedule must contain one finite floor >= std_floor per denoising step")
        self.actor = actor
        self.steps = steps
        self.eta = float(eta)
        self.std_floor = float(std_floor)
        self.guidance_scale = float(guidance_scale)
        self._cfg_batch = cfg_batch
        self.std_schedule = None if std_schedule is None else tuple(float(v) for v in std_schedule)
        self._device_coefficients = {}
        self.last_sample_timing = {}
        self.diffusion = actor._diffusion(steps)
        self.timestep_map = tuple(reversed(self.diffusion.timestep_map))
        self.kernel_config = {
            "version": DPPO_KERNEL_VERSION,
            "steps": steps,
            "eta": self.eta,
            "std_floor": self.std_floor,
            "guidance_scale": self.guidance_scale,
            "prediction_type": "x0",
            "noise_schedule": actor.noise_schedule,
            "base_diffusion_steps": actor.diffusion_steps,
            "mean_uses_base_std": True,
            "final_step_noise": True,
            "log_prob_reduction": "joint_sum_fp64",
            "network_dtype": "float32",
            "cfg_policy": "music_only_shared_history_and_prefix",
            "timestep_map": list(self.timestep_map),
        }
        if cfg_batch or self.std_schedule is not None:
            self.kernel_config.update(version="genmo.bumi_closedloop.stochastic_ddim_joint_sum.v2",
                cfg_forward="batched" if cfg_batch else "separate",
                effective_std_floors=list(self.std_schedule or (self.std_floor,) * steps))
        self._prepare_actor()

    @property
    def cfg_batch(self):
        return self._cfg_batch

    @cfg_batch.setter
    def cfg_batch(self, enabled):
        """数值验收可回退CFG双前向；执行方式随随机核身份一同更新。"""
        if type(enabled) is not bool:
            raise TypeError("cfg_batch must be boolean")
        self._cfg_batch = enabled
        if hasattr(self, 'kernel_config'):
            self.kernel_config.update(version="genmo.bumi_closedloop.stochastic_ddim_joint_sum.v2",
                cfg_forward="batched" if enabled else "separate",
                effective_std_floors=list(self.std_schedule or (self.std_floor,) * self.steps))

    @staticmethod
    def _input_signature(conditions):
        return tuple((key, value.data_ptr(), value._version, tuple(value.shape), value.dtype, value.device)
                     for key, value in sorted(conditions.items()))

    def _parameter_signature(self):
        # 原地optimizer.step/load_state_dict均改变Tensor版本，无需把权重复制到CPU求hash。
        return tuple((id(p), p._version, p.requires_grad) for p in self.actor.parameters())

    def prepare_conditions(self, conditions):
        """单链/单微批条件准备；保留encoder梯度，禁止跨参数更新或grad模式复用。

        采样在外层no_grad内调用一次；训练在每个microbatch内创建有梯度图并只
        backward一次。这里detach的是环境输入，而不是可学习历史/前缀/音乐编码。
        """
        self._prepare_actor()
        selected = self._conditions(conditions)
        with torch.autocast(device_type=selected['known_qpos30'].device.type, enabled=False):
            adapted = self.actor.adapt_conditions(selected)
            residual = self.actor._residual(adapted)
            conditional = self.actor._music_condition(adapted) + residual
            unconditional = None
            if self.guidance_scale != 1.:
                drop = torch.ones(conditional.shape[0], dtype=torch.bool, device=conditional.device)
                unconditional = self.actor._music_condition(adapted, drop) + residual
        return dict(adapted=adapted, conditional=conditional, unconditional=unconditional,
            owner=id(self), inputs=self._input_signature(selected), parameters=self._parameter_signature(),
            grad_enabled=torch.is_grad_enabled())

    def _coefficients(self, device):
        device = torch.device(device)
        if device not in self._device_coefficients:
            alpha = torch.as_tensor(self.diffusion.alphas_cumprod, device=device, dtype=torch.float32)
            previous = torch.as_tensor(self.diffusion.alphas_cumprod_prev, device=device, dtype=torch.float32)
            base = self.eta * ((1. - previous) / (1. - alpha)).sqrt()
            base = base * (1. - alpha / previous).clamp_min(0.).sqrt()
            self._device_coefficients[device] = dict(alpha=alpha, previous=previous, base=base,
                timesteps=torch.as_tensor(self.timestep_map, dtype=torch.long, device=device),
                floors=torch.tensor(self.std_schedule or (self.std_floor,) * self.steps,
                                    dtype=torch.float32, device=device))
        return self._device_coefficients[device]

    def _prepare_actor(self) -> None:
        if any(parameter.dtype != torch.float32 for parameter in self.actor.parameters()):
            raise TypeError("DPPO actor parameters must be float32")
        self.actor.eval()

    @staticmethod
    def _conditions(conditions: Mapping[str, torch.Tensor], *, clone: bool = False) -> dict:
        if set(conditions) != set(STAGE1_CONDITION_KEYS):
            raise ValueError("DPPO conditions must contain exactly STAGE1_CONDITION_KEYS")
        if any(not isinstance(value, torch.Tensor) for value in conditions.values()):
            raise TypeError("DPPO conditions must be tensors")
        return {key: (conditions[key].detach().clone() if clone else conditions[key].detach())
                for key in STAGE1_CONDITION_KEYS}

    def _step_indices(self, step_index: int | torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        if isinstance(step_index, int) and not isinstance(step_index, bool):
            result = torch.full((value.shape[0],), step_index, dtype=torch.long, device=value.device)
        elif isinstance(step_index, torch.Tensor) and step_index.dtype == torch.long:
            if step_index.shape != (value.shape[0],):
                raise ValueError("step_index tensor must be int64 [B]")
            result = step_index.to(value.device)
        else:
            raise TypeError("step_index must be an integer or int64 [B]")
        if bool(((result < 0) | (result >= self.steps)).any()):
            raise ValueError("step_index outside denoising chain")
        return result

    @staticmethod
    def _state(value: torch.Tensor, reference: torch.Tensor, name: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor) or value.shape != reference.shape:
            raise ValueError(f"{name} must have shape [B,120,30]")
        if value.dtype != torch.float32 or value.device != reference.device:
            raise TypeError(f"{name} must be float32 on the conditions device")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite")
        return value.detach()

    def transition_parameters(
        self,
        conditions: Mapping[str, torch.Tensor],
        x_k: torch.Tensor,
        step_index: int | torch.Tensor,
        *, prepared: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        """返回带参数梯度的 Gaussian 参数；输入 x_k 固定，允许每样本不同去噪步。"""
        self._prepare_actor()
        selected = self._conditions(conditions)
        with torch.autocast(device_type=selected["known_qpos30"].device.type, enabled=False):
            if prepared is None:
                prepared = self.prepare_conditions(conditions)
            elif (prepared.get('owner') != id(self)
                    or prepared.get('inputs') != self._input_signature(selected)
                    or prepared.get('parameters') != self._parameter_signature()
                    or prepared.get('grad_enabled') != torch.is_grad_enabled()):
                raise ValueError('Prepared conditions differ from inputs, Actor version or grad mode')
            adapted = prepared['adapted']
            state = self._state(x_k, adapted["known_x"], "x_k")
            state = self.actor._constrain(state, adapted)
            indices = self._step_indices(step_index, state)
            coefficients = self._coefficients(state.device)
            original_t = coefficients['timesteps'][indices]
            conditional, unconditional = prepared['conditional'], prepared['unconditional']
            if self.cfg_batch and unconditional is not None:
                combined = {key: torch.cat((value, value), dim=0) for key, value in adapted.items()}
                output = self.actor._denoise(torch.cat((state, state), dim=0),
                    torch.cat((original_t, original_t), dim=0), combined,
                    torch.cat((conditional, unconditional), dim=0))
                conditional_output = {key: value[:state.shape[0]] for key, value in output.items()}
                unconditional_output = {key: value[state.shape[0]:] for key, value in output.items()}
            else:
                conditional_output = self.actor._denoise(state, original_t, adapted, conditional)
                unconditional_output = None
            prediction = conditional_output["pred_x_start"]
            logits = conditional_output["static_conf_logits"]
            if self.guidance_scale != 1.0:
                output = (unconditional_output if unconditional_output is not None else
                          self.actor._denoise(state, original_t, adapted, unconditional))
                prediction = output["pred_x_start"] + self.guidance_scale * (
                    prediction - output["pred_x_start"])
                logits = output["static_conf_logits"] + self.guidance_scale * (
                    logits - output["static_conf_logits"])
            prediction = self.actor._constrain(prediction, adapted)
            spaced_t = self.steps - 1 - indices
            alpha = coefficients['alpha'][spaced_t, None, None]
            previous_alpha = coefficients['previous'][spaced_t, None, None]
            base_std = coefficients['base'][spaced_t, None, None]
            epsilon = (state - alpha.sqrt() * prediction) / (1.0 - alpha).sqrt()
            direction = (1.0 - previous_alpha - base_std.square()).clamp_min(0.0).sqrt()
            mean = previous_alpha.sqrt() * prediction + direction * epsilon
            mean = self.actor._constrain(mean, adapted)
            std = torch.maximum(base_std, coefficients['floors'][indices, None, None])
            if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()):
                raise FloatingPointError("DPPO transition parameters are nonfinite")
            return {
                "mean": mean,
                "std": std,
                "base_std": base_std,
                "free_mask": adapted["future_valid"][..., None] & ~adapted["known_mask"],
                "pred_x_start": prediction,
                "contact_logits": logits,
            }

    def evaluate_log_probs(
        self,
        conditions: Mapping[str, torch.Tensor],
        x_k: torch.Tensor,
        x_next: torch.Tensor,
        step_index: int | torch.Tensor,
        *, prepared: dict | None = None,
    ) -> torch.Tensor:
        """重算一个或混合内部决策的 log probability，返回 FP64 [B]。"""
        parameters = self.transition_parameters(conditions, x_k, step_index, prepared=prepared)
        observed = self._state(x_next, parameters["mean"], "x_next")
        return masked_joint_log_prob(observed, parameters["mean"], parameters["std"],
                                     parameters["free_mask"])

    @torch.no_grad()
    def sample_rollout(
        self,
        conditions: Mapping[str, torch.Tensor],
        *,
        generator: torch.Generator | None = None,
        initial_noise: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """采样完整随机链和只读条件副本；概率在最终物理解码之前记录。"""
        self._prepare_actor()
        snapshot = self._conditions(conditions, clone=True)
        with torch.autocast(device_type=snapshot["known_qpos30"].device.type, enabled=False):
            device = snapshot['known_qpos30'].device
            def synchronize():
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
            synchronize()
            beginning = time.perf_counter()
            prepared = self.prepare_conditions(snapshot)
            adapted = prepared['adapted']
            synchronize()
            conditioned = time.perf_counter()
            template = adapted["known_x"]
            noise = (torch.randn(template.shape, device=template.device, dtype=torch.float32,
                                 generator=generator) if initial_noise is None
                     else self._state(initial_noise, template, "initial_noise"))
            state = self.actor._constrain(noise, adapted)
            chain, means, stds, probabilities = [state.clone()], [], [], []
            for index in range(self.steps):
                parameters = self.transition_parameters(snapshot, state, index, prepared=prepared)
                noise = torch.randn(state.shape, device=state.device, dtype=torch.float32, generator=generator)
                following = self.actor._constrain(parameters["mean"] + parameters["std"] * noise, adapted)
                probabilities.append(masked_joint_log_prob(following, parameters["mean"],
                                     parameters["std"], parameters["free_mask"]))
                means.append(parameters["mean"].clone())
                stds.append(parameters["std"].clone())
                chain.append(following.clone())
                state = following
            synchronize()
            denoised = time.perf_counter()
            physical = self.actor.endecoder.denormalize(state)
            physical = torch.where(adapted["known_mask"], adapted["known_physical"], physical)
            physical = torch.where(adapted["future_valid"][..., None], physical, 0.0)
            qpos = self.actor.endecoder.codec.decode_to_canonical_qpos(physical)
            qpos = torch.where(adapted["future_valid"][..., None], qpos, 0.0)
            logits = parameters["contact_logits"]
            synchronize()
            self.last_sample_timing = dict(condition_prepare_seconds=conditioned-beginning,
                denoising_seconds=denoised-conditioned, decode_seconds=time.perf_counter()-denoised)
            return {
                "chain": torch.stack(chain, dim=1),
                "old_log_probs": torch.stack(probabilities, dim=1),
                "old_means": torch.stack(means, dim=1),
                "old_stds": torch.stack(stds, dim=1),
                "free_mask": parameters["free_mask"].clone(),
                "conditions": snapshot,
                "timestep_map": torch.as_tensor(self.timestep_map, dtype=torch.long, device=state.device),
                "kernel_config": {**self.kernel_config, "timestep_map": list(self.timestep_map)},
                "normalized": state.clone(),
                "qpos30": physical,
                "qpos": qpos,
                "contact_logits": logits,
                "contact": torch.where(adapted["future_valid"][..., None], logits.sigmoid(), 0.0),
            }
