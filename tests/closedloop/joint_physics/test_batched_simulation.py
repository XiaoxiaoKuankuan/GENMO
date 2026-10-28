"""批量接触统计的物理语义单测，不以张量夹具冒充真实Newton运行。

使用生产同一Warp kernel在CPU执行，检查worldid隔离、旧bridge活动接触判定、脚部名称映射
预计算的使用方式、无效容量尾部及越界诊断；另核对容量和求解迭代告警必须分开。
真实多环境吞吐、显存及solver状态由独立采集入口在服务器验证。
"""

import torch

from gem.closedloop.joint_physics.batched_simulation import health_failure_mask, reduce_contacts_tensor


def test_contact_worldid_and_old_active_contact_semantics():
    # geom0为地面，geom1左脚，geom2右脚，geom3躯干。世界编号刻意与geom不同。
    geom = torch.tensor([[0, 1], [0, 2], [1, 3], [0, 2], [-1, 2], [0, 1], [999, 999]], dtype=torch.int32)
    world = torch.tensor([2, 0, 2, 1, 1, 1, 999], dtype=torch.int32)
    distance = torch.tensor([-.01, -.02, -.01, .1, -.01, -.01, -.01])
    margin = torch.zeros(7)
    foot = torch.tensor([[0, 0], [1, 0], [0, 1], [0, 0]], dtype=torch.int32)
    output, invalid = reduce_contacts_tensor(torch.tensor([6], dtype=torch.int32), geom, world,
                                             distance, margin, foot, 3)
    expected = torch.tensor([[1, 0, 1], [1, 1, 0], [2, 1, 0]], dtype=torch.int32)
    torch.testing.assert_close(output, expected)
    assert invalid.item() == 0  # 第7条容量尾部未初始化数据不能污染结果。
    single, _ = reduce_contacts_tensor(torch.tensor([6], dtype=torch.int32), geom, torch.zeros_like(world),
                                       distance, margin, foot, 1)
    assert single.tolist() == [[4, 1, 1]]  # 与原单环境contact_events的any语义一致。


def test_invalid_active_indices_and_inactive_contacts():
    geom = torch.tensor([[0, 4], [0, 1], [0, 1]], dtype=torch.int32)
    output, invalid = reduce_contacts_tensor(torch.tensor([3], dtype=torch.int32), geom,
        torch.tensor([0, 99, 99], dtype=torch.int32), torch.tensor([-.1, -.1, 0.]), torch.zeros(3),
        torch.tensor([[0, 0], [1, 0]], dtype=torch.int32), 2)
    assert output.sum() == 0
    assert invalid.item() == 2  # dist==margin不属于活动接触，不误报第三条。


def test_solver_capacity_and_iteration_status_are_distinct():
    health = torch.tensor([[512, 500, 1024, 1024], [0, 501, 10, 10],
                           [0, 1, 1025, 10], [8, 1, 10, 10], [0, 1, 10, 1025]], dtype=torch.int32)
    mask = health_failure_mask(health, torch.zeros(1, dtype=torch.int32), contact_capacity=1024,
                              constraint_capacity=500, capacity_mask=511 | 2048)
    assert mask.tolist() == [False, True, True, True, True]
    assert health_failure_mask(health[:1], torch.ones(1, dtype=torch.int32), contact_capacity=1024,
                               constraint_capacity=500, capacity_mask=511 | 2048).item()
