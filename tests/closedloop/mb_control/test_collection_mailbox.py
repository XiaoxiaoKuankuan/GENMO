"""IPC轮次替换与正式编号checkpoint不可覆盖的存储合同回归。

验证通信邮箱可以发布两次完整张量，而SM编号保存仍明确拒绝覆盖。只测试文件协议，
真实并行、机器人动作和训练收益必须由独立Newton验收给出，不能由本测试推断。
"""

import pytest
import torch

from gem.closedloop.mb_control.parallel_collection import publish_tensor
from supermimic.bridge.bundle import atomic_save


def test_ipc_replaces_but_numbered_checkpoint_remains_protected(tmp_path):
    mailbox, numbered = tmp_path/"response.pt", tmp_path/"actor_numbered.pt"
    publish_tensor({"id": 1, "value": torch.tensor([1.])}, mailbox)
    publish_tensor({"id": 2, "value": torch.tensor([2.])}, mailbox)
    loaded = torch.load(mailbox, weights_only=False)
    assert loaded["id"] == 2 and loaded["value"].item() == 2
    atomic_save({"id": 1}, numbered)
    with pytest.raises(FileExistsError):
        atomic_save({"id": 2}, numbered)
    assert torch.load(numbered, weights_only=False)["id"] == 1
    assert not list(tmp_path.glob(".ipc-*")) and not list(tmp_path.glob(".genmo-*"))
