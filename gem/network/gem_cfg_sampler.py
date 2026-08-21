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
            # Parameterized heads may be disabled (for example BUMI has
            # pred_cam_dim=0). Preserve a mutually absent optional output
            # instead of trying to apply tensor CFG arithmetic to ``None``.
            if out[k] is None and out_uncond[k] is None:
                outputs[k] = None
            elif isinstance(out[k], torch.Tensor) and isinstance(
                out_uncond[k], torch.Tensor
            ):
                outputs[k] = out_uncond[k] + y["scale"] * (out[k] - out_uncond[k])
            else:
                raise TypeError(
                    f"CFG output {k!r} must be tensors in both branches or None in both; "
                    f"got {type(out[k]).__name__}/{type(out_uncond[k]).__name__}"
                )
        return outputs

    def parameters(self):
        return self.model.parameters()

    def named_parameters(self):
        return self.model.named_parameters()
