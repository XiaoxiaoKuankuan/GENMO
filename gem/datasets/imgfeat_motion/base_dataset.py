# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""带图像特征的运动数据集基类。

BEDLAM、H36M 和 3DPW 等视频数据集会继承这个基类。子类负责加载原始数据并实现
样本处理逻辑，基类统一提供 `__len__` 和 `__getitem__`，让 DataLoader 能直接
取到 GEM 训练所需的字典样本。
"""

from torch.utils import data


class ImgfeatMotionDatasetBase(data.Dataset):
    def __init__(self):
        super().__init__()
        self._load_dataset()
        self._get_idx2meta()  # -> Set self.idx2meta

    def __len__(self):
        return len(self.idx2meta)

    def _load_dataset(self):
        raise NotImplementedError

    def _get_idx2meta(self):
        raise NotImplementedError

    def _load_data(self, idx):
        raise NotImplementedError

    def _process_data(self, data, idx):
        raise NotImplementedError

    def __getitem__(self, idx):
        data = self._load_data(idx)
        data = self._process_data(data, idx)
        return data
