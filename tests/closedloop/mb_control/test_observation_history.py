"""使用真实冻结SM资产检验50Hz观测历史的追加与同tick替换合同。

输入为明确的名义张量状态，观测通过权威MDP和实际FK生成；不运行假物理，也不把
本测试当成真实执行验收。refresh必须仅替换最后一槽，advance每个控制时刻只追加
一次；相同状态序列在real与imagined标签下应产生完全相同的75维历史。
"""

from dataclasses import replace
import os

import pytest
import torch


def test_history_is_appended_once_and_refreshed_in_place():
    checkpoint = os.getenv("SM_BRIDGE_CHECKPOINT")
    if not checkpoint:
        pytest.skip("需要实际固定SM资产")
    from supermimic.bridge.bundle import load_bundle
    from gem.closedloop.mb_control.sm_adapter import SMAdapter
    from gem.closedloop.mb_control.gradient_audit import tensor_fixture
    bundle = load_bundle(checkpoint, robot_asset_dir=os.getenv("SM_ROBOT_ASSET_DIR"))
    adapter = SMAdapter(bundle, tuple(reversed(bundle.spec.joint_names)))
    initial = torch.zeros(1, 55)
    initial[:, 2], initial[:, 3] = bundle.spec.spawn_height, 1.
    initial[:, 13:34] = torch.tensor(bundle.spec.default_joint_pos)
    z = tensor_fixture(adapter, torch.zeros(1, 200, 35), initial)
    # 不相同的历史槽可以发现误滚动；使用真实MDP输出作为当前槽。
    tagged = torch.arange(9.).view(1, 9, 1).expand(1, 9, 75)
    z = replace(z, sm_history=torch.cat((tagged, z.sm_history[:, -1:]), 1))
    refreshed = adapter.refresh_current_observation(adapter.refresh_current_observation(z))
    torch.testing.assert_close(refreshed.sm_history, z.sm_history, atol=0, rtol=0)
    action = torch.full((1, 21), .02)
    next_state = bundle.step_model(initial, action, z.xi)
    real = adapter.advance(refreshed, next_state, action, origin="real")
    imagined = adapter.advance(refreshed, next_state, action, origin="imagined")
    assert real.tick == 12 and imagined.tick == 12
    torch.testing.assert_close(real.sm_history[:, :-1], z.sm_history[:, 1:], atol=0, rtol=0)
    torch.testing.assert_close(real.sm_history, imagined.sm_history, atol=0, rtol=0)
    torch.testing.assert_close(adapter.refresh_current_observation(real).sm_history, real.sm_history, atol=0, rtol=0)
    shifted = real.reference.world_qpos + torch.nn.functional.one_hot(torch.tensor(2), 28) * .01
    changed = replace(real, reference=replace(real.reference, world_qpos=shifted))
    changed = adapter.refresh_current_observation(changed)
    torch.testing.assert_close(changed.sm_history[:, :-1], real.sm_history[:, :-1], atol=0, rtol=0)
    current, _, _ = changed.reference.read_current_future(changed.tick)
    expected = adapter.runtime.observation(next_state, action, changed.xi, current)
    torch.testing.assert_close(changed.sm_history[:, -1], expected, atol=0, rtol=0)
