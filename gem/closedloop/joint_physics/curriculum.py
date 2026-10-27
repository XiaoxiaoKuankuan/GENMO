"""单run内数据主导的物理课程与真实扩散SNR筛选。

监督噪声始终均匀覆盖1000步；筛选仅控制执行子批，不改变BC。权重从新run step0递增，
drop音乐的样本保留完整监督但不施加单动作执行目标。所有拒绝原因和有效率由训练器记录。
"""

import torch
from collections import deque


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


def physics_noise_levels(diffusion, cfg, report=None):
    """由实际1000步日程计算SNR，物理噪声只取当前认证实际覆盖的等级。"""
    alpha = torch.as_tensor(diffusion.alphas_cumprod)
    snr = alpha/(1-alpha)
    indices = ((snr >= cfg["snr_min"]) & (snr <= cfg["snr_max"])).nonzero().flatten()
    if report is not None:
        audited = report.get("audited_timesteps", [])
        indices = indices[torch.tensor([int(k) in audited for k in indices], dtype=torch.bool)]
    if not len(indices):
        raise ValueError("物理SNR区间与真实审计噪声等级没有交集")
    return indices


class Readiness:
    """步数下限加最近学生有效率/音乐/类型覆盖；只对已认证真实学生窗口计数。"""
    def __init__(self, cfg):
        self.cfg = cfg
        self.rows = deque(maxlen=cfg["readiness_window"])
        self.activated_step = None

    def observe(self, rows, step):
        self.rows.extend({"valid": r["reason"] == "execution_gradient_completed", "source": r["source"],
                          "types": r.get("tags", [])} for r in rows)
        valid = [r for r in self.rows if r["valid"]]
        sources = {r["source"] for r in valid}
        types = {t for r in valid for t in r["types"]}
        rate = len(valid)/max(1,len(self.rows))
        if (self.activated_step is None and step >= self.cfg["start_step"] and
            len(self.rows) >= self.cfg["readiness_min_samples"] and rate >= self.cfg["readiness_min_rate"] and
            len(sources) >= self.cfg["readiness_min_sources"] and len(types) >= self.cfg["readiness_min_types"]):
            self.activated_step = step
        return {"activated_step": self.activated_step, "recent_samples": len(self.rows),
                "valid_rate": rate, "sources": sorted(sources), "types": sorted(types)}

    def weight(self, step):
        if self.activated_step is None or not self.cfg["enabled"]:
            return 0.
        return self.cfg["lambda_exec"]*min(1., (step-self.activated_step+1)/max(1,self.cfg["warmup_steps"]))

    def state_dict(self):
        return {"rows": list(self.rows), "activated_step": self.activated_step}

    def load_state_dict(self, value):
        self.rows.extend(value["rows"])
        self.activated_step = value["activated_step"]
