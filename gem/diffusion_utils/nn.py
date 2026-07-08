# Copyright (c) 2021 OpenAI
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND LicenseRef-NVIDIA-OneWay-Noncommercial
# This code is derived from https://github.com/openai/guided-diffusion
"""
扩散损失中的张量归约工具。

GaussianDiffusion 用这些函数把每个样本除 batch 维之外的维度求和/平均，从而得到
按样本统计的训练 loss。

Various utilities for neural networks.
"""


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def sum_flat(tensor):
    """
    Take the sum over all non-batch dimensions.
    """
    return tensor.sum(dim=list(range(1, len(tensor.shape))))
