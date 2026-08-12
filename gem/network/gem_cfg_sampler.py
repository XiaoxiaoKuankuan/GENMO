# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Classifier-free guidance 采样包装器。

训练阶段会随机 mask 条件；推理采样时这个包装器分别跑有条件和无条件分支，
再用 guidance scale 放大条件信号。它主要服务完整扩散模型的生成/补全路径。
"""

from copy import deepcopy

import torch


# A wrapper model for Classifier-free guidance **SAMPLING** only
# https://arxiv.org/abs/2207.12598
class ClassifierFreeSampleModel:
    def __init__(self, model, mask_localpose=False):
        self.model = model  # model is the actual model to run
        self.mask_localpose = mask_localpose

    def __call__(self, x, timesteps, y=None, **kwargs):
        y_uncond = deepcopy(y)
        # Text is optional for specialist denoisers.  Generalist checkpoints
        # keep the exact legacy behaviour when the key is present.
        if "encoded_text" in y:
            y_uncond["encoded_text"] = torch.zeros_like(y["encoded_text"])
        y_uncond["f_cond"] = y["f_uncond"]
        if "multi_text_data" in y:
            y_uncond["multi_text_data"]["text_embed"] = torch.zeros_like(
                y["multi_text_data"]["text_embed"]
            )
        if self.mask_localpose:
            y_uncond["f_cond"] = y["f_empty"]
            x_start = y["pred_x_start"].clone()
            localpose_idx = self.model.denoiser.denoiser3d.endecoder.obs_indices_dict["body_pose"]
            x_uncond = x.clone()
            x_uncond[:, :, localpose_idx] = x_start[:, :, localpose_idx]
        else:
            x_uncond = x

        out = self.model(x, timesteps, y, **kwargs)
        out_uncond = self.model(x_uncond, timesteps, y_uncond, **kwargs)
        outputs = dict()
        for k in out:
            outputs[k] = out_uncond[k] + y["scale"] * (out[k] - out_uncond[k])
        return outputs

    def parameters(self):
        return self.model.parameters()

    def named_parameters(self):
        return self.model.named_parameters()
