"""检验既有EDGE35音乐文件的格式读取，不生成或重建任何配对库。

临时张量仅验证.pt原始Tensor与.npy的读入一致性，并拒绝含糊字典或错误维度。
此测试属于数据接口回归，不代表真实音乐质量、物理执行或训练收益。
"""

import numpy as np
import pytest
import torch


def test_original_torch_music_and_numpy_are_identical(tmp_path):
    from gem.closedloop.mb_control.application import read_music_features
    features = torch.arange(70, dtype=torch.float32).reshape(2, 35)
    torch.save(features, tmp_path / "music.pt")
    np.save(tmp_path / "music.npy", features.numpy())
    torch.testing.assert_close(read_music_features(tmp_path / "music.pt"), features)
    torch.testing.assert_close(read_music_features(tmp_path / "music.npy"), features)
    torch.save({"features": features}, tmp_path / "ambiguous.pt")
    with pytest.raises(ValueError):
        read_music_features(tmp_path / "ambiguous.pt")
