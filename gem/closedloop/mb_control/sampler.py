"""同一当前GENMO网络的完整20步DDIM eta=0重参数化采样。

grad和部署no-grad调用同一核心，并复用Stage1原1000步日程与DDIM公式。初始噪声由
调用方提供，每请求重新抽样，整次反传固定。每步known逐坐标where，CFG仅删除音乐。
可选非重入checkpoint只重算去噪网络，不截断状态/条件/解码梯度，无概率比或混合网络。
"""

import torch
from torch.utils.checkpoint import checkpoint


def sample_differentiable(actor, conditions, context, initial_noise, *, guidance_scale=2.5,
                          activation_checkpointing=True):
    base = actor.stage1
    if base.training:
        raise RuntimeError("采样需要eval模式；eval不关闭autograd")
    adapted = base.adapt_conditions(conditions)
    if initial_noise.shape != (len(adapted["known_x"]), 60, 30):
        raise ValueError("采样固定60帧qpos30，初始噪声必须为[B,60,30]")
    value = base._constrain(initial_noise.to(adapted["known_x"]), adapted)
    residual = base._residual(adapted) + actor.future_encoder(context, conditions["future_times"])
    conditional = base._music_condition(adapted) + residual
    unconditional = base._music_condition(adapted, torch.ones(len(value), dtype=torch.bool, device=value.device)) + residual

    def guided_impl(x, t, c, u):
        cond = base._denoise(x, t, adapted, c)
        if guidance_scale == 1.0:
            return cond
        uncond = base._denoise(x, t, adapted, u)
        return {"pred_x_start": base._constrain(uncond["pred_x_start"] + guidance_scale *
                                                 (cond["pred_x_start"] - uncond["pred_x_start"]), adapted),
                "static_conf_logits": uncond["static_conf_logits"] + guidance_scale *
                                      (cond["static_conf_logits"] - uncond["static_conf_logits"])}

    def guided(x, t, **kwargs):
        if activation_checkpointing and torch.is_grad_enabled():
            return checkpoint(guided_impl, x, t, conditional, unconditional, use_reentrant=False)
        return guided_impl(x, t, conditional, unconditional)

    diffusion = base._diffusion(20)
    for index in reversed(range(20)):
        t = torch.full((len(value),), index, dtype=torch.long, device=value.device)
        out = diffusion.ddim_sample(guided, value, t, clip_denoised=False, model_kwargs={"y": {}}, eta=0.)
        value = base._constrain(out["sample"], adapted)
    physical = base.endecoder.denormalize(value)
    physical = torch.where(adapted["known_mask"], adapted["known_physical"], physical)
    physical = torch.where(adapted["future_valid"][..., None], physical, 0.)
    qpos = base.endecoder.codec.decode_to_canonical_qpos(physical)
    return {"qpos30": physical, "qpos28": qpos,
            "contact": torch.where(adapted["future_valid"][..., None], out["static_conf_logits"].sigmoid(), 0.),
            "contact_logits": out["static_conf_logits"]}
