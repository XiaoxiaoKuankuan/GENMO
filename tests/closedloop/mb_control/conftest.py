"""新Stage2合同测试夹具；只构造明确的张量状态，不伪造真实物理结果。

SM源码路径来自显式环境变量或配套独立工作树。无Isaac导入；数值夹具只用于缓存、
时间和价值网络，真实权重/物理测试必须另提供实际资产，本轮所有测试均未执行。
"""

from pathlib import Path
import os
import sys

import pytest
import torch

repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo))
sys.path.insert(0, os.getenv("SUPERMIMIC_REPO", str(repo.parent / "SuperMimic")))


@pytest.fixture
def tensor_state():
    from gem.closedloop.mb_control.contracts import ClosedLoopState
    from gem.closedloop.mb_control.reference_buffer import ReferenceBuffer
    qpos = torch.zeros(1, 60, 28)
    qpos[..., 2], qpos[..., 3] = .65, 1.
    ticks = torch.arange(60) * 20
    cache = ReferenceBuffer(ticks, qpos, torch.zeros(1, 60, 2), torch.zeros_like(ticks), None, tuple(range(21)))
    state = torch.zeros(1, 55)
    state[:, 2], state[:, 3] = .65, 1.
    xi = torch.zeros(1, 25)
    xi[:, 0] = 1.
    return ClosedLoopState(0, state, xi, torch.zeros(1, 21), torch.zeros(1, 10, 75),
                           torch.zeros(1, 50, 48), torch.ones(1, 50, dtype=torch.bool), cache,
                           torch.zeros(1, 300, 35), 0, 6000)
