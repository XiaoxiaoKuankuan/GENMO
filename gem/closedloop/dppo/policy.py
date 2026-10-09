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
from .performance import profiled
from .execution_checks import require_tensor, checked_policy_phase

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
    require_tensor(torch.isfinite(std).all() & (std > 0).all(),
                   "Gaussian std must be finite and strictly positive")
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
        execution_batch_size: int | None = None,
        numerical_layout: str = 'legacy_step_lane',
        defer_checks: bool = False,
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
        if execution_batch_size is not None and (type(execution_batch_size) is not int or execution_batch_size < 1):
            raise ValueError('execution_batch_size must be a positive integer or None')
        if std_schedule is not None and (len(std_schedule) != steps or any(
            isinstance(value, bool) or not math.isfinite(value) or value < std_floor for value in std_schedule
        )):
            raise ValueError("std_schedule must contain one finite floor >= std_floor per denoising step")
        self.actor = actor
        self.defer_checks = defer_checks
        self._phase_signature = None
        self.numerical_layout = numerical_layout
        if numerical_layout not in ('legacy_step_lane', 'sample_matrix_bmm_fp32.v1'):
            raise ValueError('Unknown numerical execution layout')
        if numerical_layout != 'legacy_step_lane':
            from .batch_execution import set_sample_linear
            set_sample_linear(actor.denoiser, True)
        self.steps = steps
        self.eta = float(eta)
        self.std_floor = float(std_floor)
        self.guidance_scale = float(guidance_scale)
        self._cfg_batch = cfg_batch
        self._execution_batch_size = None
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
        self.execution_batch_size = execution_batch_size
        self._prepare_actor()

    @property
    def execution_batch_size(self):
        return self._execution_batch_size

    @execution_batch_size.setter
    def execution_batch_size(self, size):
        if size is not None and (type(size) is not int or size < 1):
            raise ValueError('Execution batch shape must be positive or None')
        self._execution_batch_size = size
        if hasattr(self, 'kernel_config'):
            if self.numerical_layout != 'legacy_step_lane':
                self.kernel_config.update(version='genmo.bumi_closedloop.stochastic_ddim_joint_sum.v4',
                    execution_contract=self.numerical_layout, execution_batch_size=None,
                    condition_batch_size=1, network_batch='effective_rows_cfg_expanded_no_padding')
                return
            if size is None:
                self.kernel_config.pop('execution_contract', None)
                self.kernel_config.pop('execution_batch_size', None)
            else:
                self.kernel_config.update(version='genmo.bumi_closedloop.stochastic_ddim_joint_sum.v3',
                    execution_contract='fixed_shape_step_lane_single_condition_fp32.v1', execution_batch_size=size)

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
            if self.execution_batch_size is not None or self.numerical_layout != 'legacy_step_lane':
                self.execution_batch_size = self.execution_batch_size

    @staticmethod
    def _input_signature(conditions):
        return tuple((key, value.data_ptr(), value._version, tuple(value.shape), value.dtype, value.device)
                     for key, value in sorted(conditions.items()))

    def _parameter_signature(self):
        if self._phase_signature is not None:
            return self._phase_signature
        # 原地optimizer.step/load_state_dict均改变Tensor版本，无需把权重复制到CPU求hash。
        return tuple((id(p), p._version, p.requires_grad) for p in self.actor.parameters())

    @profiled('policy.condition_encoding', gpu=True)
    def prepare_conditions(self, conditions):
        """单链/单微批条件准备；保留encoder梯度，禁止跨参数更新或grad模式复用。

        采样在外层no_grad内调用一次；训练在每个microbatch内创建有梯度图并只
        backward一次。这里detach的是环境输入，而不是可学习历史/前缀/音乐编码。
        """
        self._prepare_actor()
        selected = self._conditions(conditions)
        # 新执行身份固定条件编码的batch=1，避免同链条件在训练合批后切换GRU/GEMM数值路径。
        # 真实学习使用ConditionGraphCache，仅对每条唯一链运行一次这里的编码。
        count = selected['known_qpos30'].shape[0]
        if (self.execution_batch_size is not None or self.numerical_layout != 'legacy_step_lane') and count > 1:
            entries = [self.prepare_conditions({key: value[i:i+1] for key, value in selected.items()})
                       for i in range(count)]
            result = dict(entries[0])
            result['adapted'] = {key: torch.cat([entry['adapted'][key] for entry in entries]) for key in entries[0]['adapted']}
            for name in ('conditional', 'unconditional'):
                result[name] = None if entries[0][name] is None else torch.cat([entry[name] for entry in entries])
            result['inputs'] = self._input_signature(selected)
            return result
        with torch.autocast(device_type=selected['known_qpos30'].device.type, enabled=False):
            adapted = self.actor.adapt_conditions(selected)
            conditional, unconditional = self._encode_prepared(adapted)
        return dict(adapted=adapted, conditional=conditional, unconditional=unconditional,
            owner=id(self), inputs=self._input_signature(selected), parameters=self._parameter_signature(),
            grad_enabled=torch.is_grad_enabled())

    def _encode_prepared(self, adapted):
        """纯张量条件编码；采样图可捕获此段，校验和有梯度条件缓存留在调用边界。"""
        residual = self.actor._residual(adapted)
        conditional = self.actor._music_condition(adapted) + residual
        unconditional = None
        if self.guidance_scale != 1.:
            empty = torch.zeros_like(residual)
            if self.actor.cond_exists_embedder is not None:
                flag = empty.new_zeros((*empty.shape[:-1], 1))
                empty = self.actor.cond_exists_embedder(torch.cat((empty, flag), -1))
            unconditional = torch.where(adapted['future_valid'][..., None], empty, 0.) + residual
        return conditional, unconditional

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
        if self._phase_signature is not None:
            return
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
        require_tensor(((result >= 0) & (result < self.steps)).all(), "step_index outside denoising chain")
        return result

    @staticmethod
    def _state(value: torch.Tensor, reference: torch.Tensor, name: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor) or value.shape != reference.shape:
            raise ValueError(f"{name} must have shape [B,120,30]")
        if value.dtype != torch.float32 or value.device != reference.device:
            raise TypeError(f"{name} must be float32 on the conditions device")
        require_tensor(torch.isfinite(value).all(), f"{name} must be finite")
        return value.detach()

    @profiled('policy.transition', gpu=True)
    def transition_parameters(
        self,
        conditions: Mapping[str, torch.Tensor],
        x_k: torch.Tensor,
        step_index: int | torch.Tensor,
        *, prepared: dict | None = None, diagnostics: bool = False,
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
            actual_batch = state.shape[0]
            output_lanes = None
            if self.execution_batch_size is not None and self.numerical_layout == 'legacy_step_lane':
                if actual_batch > self.execution_batch_size:
                    raise ValueError('Microbatch exceeds the fixed numerical execution shape')
                # 固定形状还不够：部分GEMM在同批不同行存在ULP差异。每个去噪步固定
                # 到step % B的位置，使单链采样与20步微批重算使用相同的计算行。
                output_lanes = indices.remainder(self.execution_batch_size)
                if torch.unique(output_lanes).numel() != actual_batch:
                    raise ValueError('Fixed-shape microbatch needs distinct denoising step lanes')
                selection = torch.zeros(self.execution_batch_size, device=state.device, dtype=torch.long)
                selection[output_lanes] = torch.arange(actual_batch, device=state.device)
                state, indices, original_t = state[selection], indices[selection], original_t[selection]
                adapted = {key: value[selection] for key, value in adapted.items()}
                conditional = conditional[selection]
                unconditional = None if unconditional is None else unconditional[selection]
            if self.cfg_batch and unconditional is not None:
                # 同一条采样链20步共享条件，CFG固定输入只拼接一次。学习微批各自
                # 持有prepared，不跨optimizer复用；cat仍保留条件编码器的梯度。
                if output_lanes is None:
                    signature = tuple((id(v), v._version) for v in
                        (*adapted.values(), conditional, unconditional))
                    cached = prepared.get('_cfg_inputs')
                    if cached is None or cached[0] != signature:
                        cached = (signature, {key: torch.cat((value, value), dim=0)
                                  for key, value in adapted.items()},
                                  torch.cat((conditional, unconditional), dim=0))
                        prepared['_cfg_inputs'] = cached
                    combined, cfg_condition = cached[1:]
                else:
                    combined = {key: torch.cat((value, value), dim=0) for key, value in adapted.items()}
                    cfg_condition = torch.cat((conditional, unconditional), dim=0)
                output = self.actor._denoise(torch.cat((state, state), dim=0),
                    torch.cat((original_t, original_t), dim=0), combined,
                    cfg_condition)
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
            require_tensor(torch.isfinite(mean).all() & torch.isfinite(std).all(),
                           "DPPO transition parameters are nonfinite", FloatingPointError)
            result = {
                "mean": mean,
                "std": std,
                "base_std": base_std,
                "free_mask": adapted["future_valid"][..., None] & ~adapted["known_mask"],
                "pred_x_start": prediction,
                "contact_logits": logits,
            }
            if diagnostics:
                result.update(conditional_encoding=conditional,
                    unconditional_encoding=unconditional,
                    conditional_prediction=conditional_output['pred_x_start'],
                    unconditional_prediction=(output['pred_x_start'] if self.guidance_scale != 1. else None))
            return {key: None if value is None else (value[:actual_batch] if output_lanes is None else value[output_lanes])
                    for key, value in result.items()}

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
    @checked_policy_phase
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
            generators = generator if isinstance(generator, (list, tuple)) else None
            batch_size = snapshot['known_qpos30'].shape[0]
            if generators is not None and (len(generators) != batch_size or
                    any(not isinstance(item, torch.Generator) for item in generators) or
                    len({id(item) for item in generators}) != len(generators)):
                raise ValueError('Batched environments require one distinct generator per sample')
            def draw(shape):
                if generators is None:
                    return torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
                return torch.cat([torch.randn((1,*shape[1:]), device=device, dtype=torch.float32, generator=item)
                                  for item in generators])
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
            noise = (draw(template.shape) if initial_noise is None
                     else self._state(initial_noise, template, "initial_noise"))
            state = self.actor._constrain(noise, adapted)
            chain, means, stds, probabilities = [state.clone()], [], [], []
            for index in range(self.steps):
                parameters = self.transition_parameters(snapshot, state, index, prepared=prepared)
                noise = draw(state.shape)
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
