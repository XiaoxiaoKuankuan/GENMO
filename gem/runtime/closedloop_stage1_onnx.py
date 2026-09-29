"""Stage 1 条件音乐生成模型的固定形状 ONNX 去噪接口与 DDIM 运行器。

导出图包含音乐编码、50 帧因果 proprio48 历史 GRU、逐坐标动作前缀、原 Transformer
以及 CFG 双分支；两个分支只在音乐条件上不同。公开 batch=1、动作长120帧、运动30Hz。
时间输入为历史相对决策时刻的秒数，调用前仍通过既有条件契约检查绝对时间和有效 mask。
图输出 normalized qpos30 的 x0 及独立 contact2 logits，不包含优化器、GT、DDIM循环、
qpos解码、FK、GMT或物理仿真。Python运行器复用训练Actor的扩散调度与最终前缀精确回写。
该接口与旧纯音乐ONNX不兼容，不能把旧五输入运行器直接用于这个模型。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from gem.closedloop.contracts import validate_stage1_condition_batch

INPUT_NAMES = [
    "noisy_motion",
    "diffusion_timestep",
    "music_features",
    "music_valid",
    "proprio_history",
    "proprio_history_valid",
    "history_relative_times",
    "known_qpos30",
    "known_qpos30_mask",
    "future_valid",
    "guidance_scale",
]
OUTPUT_NAMES = ["pred_motion", "pred_foot_contact_logits"]
CONTRACT_VERSION = "genmo.bumi_closedloop.stage1_onnx.v1"


class Stage1GuidedDenoiser(nn.Module):
    """将真实 Stage1 条件模块和两个 CFG 分支合并成一次 Transformer 调用。"""

    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    def forward(
        self,
        noisy_motion,
        diffusion_timestep,
        music_features,
        music_valid,
        proprio_history,
        proprio_history_valid,
        history_relative_times,
        known_qpos30,
        known_qpos30_mask,
        future_valid,
        guidance_scale,
    ):
        actor = self.actor
        known_x = actor.endecoder.normalize(torch.where(known_qpos30_mask, known_qpos30, 0.0))
        known_x = torch.where(known_qpos30_mask, known_x, 0.0)
        valid_music = music_valid & future_valid
        adapted = {
            "known_x": known_x,
            "known_mask": known_qpos30_mask,
            "future_valid": future_valid,
            "history": actor.proprio_normalizer(proprio_history, proprio_history_valid),
            "history_valid": proprio_history_valid,
            "history_relative_times": torch.where(
                proprio_history_valid, history_relative_times, 0.0
            ),
            "music_embed": torch.where(valid_music[..., None], music_features, 0.0),
            "music_valid": valid_music,
        }
        residual = actor._residual(adapted)
        conditional = actor._music_condition(adapted) + residual
        unconditional = (
            actor._music_condition(
                adapted, torch.ones(1, dtype=torch.bool, device=noisy_motion.device)
            )
            + residual
        )
        noisy = actor._constrain(noisy_motion, adapted)
        length = future_valid.sum(dim=1).clamp_min(1)
        output = actor.denoiser(
            torch.cat((noisy, noisy), 0),
            torch.cat((diffusion_timestep, diffusion_timestep), 0),
            y={
                "f_cond": torch.cat((conditional, unconditional), 0),
                "length": torch.cat((length, length), 0),
            },
            inputs={},
        )
        scale = guidance_scale.reshape(1, 1, 1)
        motion = output["pred_x_start"]
        logits = output["static_conf_logits"]
        guided = motion[1:2] + scale * (motion[:1] - motion[1:2])
        contact = logits[1:2] + scale * (logits[:1] - logits[1:2])
        return actor._constrain(guided, adapted), torch.where(future_valid[..., None], contact, 0.0)


def make_inputs(conditions, noise, timestep, guidance_scale):
    """校验标准条件并将时间转为图所需的相对秒数，监督字段不进入图。"""
    validate_stage1_condition_batch(conditions, history_steps=50)
    return (
        noise,
        timestep,
        conditions["music_features"].float(),
        conditions["music_valid"],
        conditions["proprio_history"].float(),
        conditions["proprio_history_valid"],
        (conditions["proprio_history_times"] - conditions["decision_time"][:, None]).float(),
        conditions["known_qpos30"].float(),
        conditions["known_qpos30_mask"],
        conditions["future_valid"],
        noise.new_tensor([guidance_scale]),
    )


class Stage1OnnxSampler:
    """使用已导出的真实图完成DDIM；Actor只提供既有调度、编解码和mask语义。"""

    def __init__(self, path: str | Path, actor, *, device="cuda"):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        providers = (
            [("CUDAExecutionProvider", {"device_id": 0, "use_tf32": 0}), "CPUExecutionProvider"]
            if str(device).startswith("cuda")
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
        if (
            str(device).startswith("cuda")
            and self.session.get_providers()[0] != "CUDAExecutionProvider"
        ):
            raise RuntimeError("请求 CUDA ONNX，但 CUDA provider 未成功加载")
        if [i.name for i in self.session.get_inputs()] != INPUT_NAMES:
            raise RuntimeError("Stage1 ONNX 输入顺序或名称与版本契约不符")
        self.actor = actor

    @torch.no_grad()
    def sample(self, conditions, *, steps=20, guidance_scale=2.5, noise=None, return_trace=False):
        actor = self.actor
        adapted = actor.adapt_conditions(conditions)
        xt = torch.randn_like(adapted["known_x"]) if noise is None else noise.to(adapted["known_x"])
        inputs = make_inputs(
            conditions, xt, torch.zeros(1, dtype=torch.long, device=xt.device), guidance_scale
        )
        feed = {key: value.detach().cpu().numpy() for key, value in zip(INPUT_NAMES, inputs)}
        xt = actor._constrain(xt, adapted)

        def denoise(value, timestep, **_kwargs):
            feed["noisy_motion"] = value.detach().cpu().numpy()
            feed["diffusion_timestep"] = timestep.detach().cpu().numpy()
            motion, contact = self.session.run(OUTPUT_NAMES, feed)
            if not np.isfinite(motion).all() or not np.isfinite(contact).all():
                raise FloatingPointError("ONNX 产生非有限输出")
            return {
                "pred_x_start": torch.from_numpy(motion).to(value),
                "static_conf_logits": torch.from_numpy(contact).to(value),
            }

        diffusion = actor._diffusion(int(steps))
        trace = []
        for index in range(diffusion.num_timesteps - 1, -1, -1):
            timestep = torch.full((1,), index, dtype=torch.long, device=xt.device)
            output = diffusion.ddim_sample(
                denoise, xt, timestep, clip_denoised=False, model_kwargs={"y": {}}, eta=0.0
            )
            xt = actor._constrain(output["sample"], adapted)
            if return_trace:
                trace.append(xt.clone())
        physical = actor.endecoder.denormalize(xt)
        physical = torch.where(adapted["known_mask"], adapted["known_physical"], physical)
        physical = torch.where(adapted["future_valid"][..., None], physical, 0.0)
        qpos = actor.endecoder.codec.decode_to_canonical_qpos(physical)
        qpos = torch.where(adapted["future_valid"][..., None], qpos, 0.0)
        logits = output["static_conf_logits"]
        result = {
            "normalized": xt,
            "qpos30": physical,
            "qpos": qpos,
            "contact_logits": logits,
            "contact": torch.where(adapted["future_valid"][..., None], logits.sigmoid(), 0.0),
        }
        if return_trace:
            result["trace"] = trace
        return result
