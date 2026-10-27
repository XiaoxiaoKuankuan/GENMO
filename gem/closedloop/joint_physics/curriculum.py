"""单run内数据主导的物理课程与真实扩散SNR筛选。

监督噪声始终均匀覆盖1000步；筛选仅控制执行子批，不改变BC。权重从新run step0递增，
drop音乐的样本保留完整监督但不施加单动作执行目标。所有拒绝原因和有效率由训练器记录。
"""

import torch


def physics_weights(step, cfg):
    if not cfg["enabled"] or step < cfg["start_step"]:
        return 0., 0.
    fraction = min(1., (step-cfg["start_step"]+1)/max(1, cfg["warmup_steps"]))
    return fraction*cfg["lambda_exec"], fraction*cfg["lambda_safety"]


def physics_selection(output, diffusion, cfg):
    alpha = torch.as_tensor(diffusion.alphas_cumprod, device=output["timesteps"].device)
    alpha = alpha[output["timesteps"]]
    snr = alpha/(1-alpha)
    eligible = (snr >= cfg["snr_min"]) & (snr <= cfg["snr_max"]) & ~output["music_dropout_mask"]
    return eligible, snr
