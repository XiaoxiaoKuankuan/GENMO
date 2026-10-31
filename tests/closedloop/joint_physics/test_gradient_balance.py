"""物理梯度占比课程和校准步DDP同步的必要回归测试。

用可解析的小网络检查：全局梯度先平均再求范数、零物理样本rank、全局零样本、
no_sync校准与正常DDP交替、AdamW实际更新一致以及未用参数保持None。独立验证
30%定义、平滑进入、权重上限和完整控制器恢复。小网络只证明归约与状态协议，
真实GENMO/H32/BF16八卡启动结果另记录，不能替代Newton效果评价。
"""

import copy
from pathlib import Path

import pytest
import torch
from torch import distributed as dist
from torch import multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from gem.closedloop.joint_physics.checkpoint import check_restore_contract
from gem.closedloop.joint_physics.gradient_balance import (
    GradientBalance, average_parameter_gradients, global_gradient_measurement,
)


def balance_config():
    return dict(enabled=True, target_share=.3, maximum_measured_share=.4, ramp_steps=20000,
                every_steps=100, ema_beta=.9, max_change_factor=1.2, min_weight=.0001, max_weight=5.)


def test_share_ramp_and_exact_controller_restore():
    controller = GradientBalance(balance_config(), .02, 90000)
    measurement = dict(supervised_norm=2.4, execution_norm=.8, cosine=.02, modules={})
    assert controller.observe(90000, measurement, 32) == pytest.approx(.02)
    assert controller.metrics['measured_share'] == pytest.approx(.016/2.416)
    for step in range(90100, 100001, 100):
        controller.observe(step, measurement, 32)
    restored = GradientBalance(balance_config(), .02)
    restored.load_state_dict(controller.state_dict())
    for step in range(100100, 110101, 100):
        assert controller.observe(step, measurement, 32) == restored.observe(step, measurement, 32)
        assert controller.state_dict() == restored.state_dict()
    assert controller.metrics['measured_share'] == pytest.approx(.3)
    assert controller.weight == pytest.approx((.3/.7)*3)
    assert not controller.due(110101)
    assert controller.due(110200)
    weight = controller.weight
    controller.observe(110200, {**measurement, 'execution_norm': 0.}, 0)
    assert controller.weight == weight
    assert controller.metrics['status'] == 'no_nonzero_global_gradient'
    controller.observe(110300, {**measurement, 'execution_norm': 800.}, 32)
    assert controller.metrics['measured_share'] <= .400000001
    with pytest.raises(FloatingPointError):
        controller.observe(110400, {**measurement, 'execution_norm': float('nan')}, 32)


def test_contract_migration_does_not_relax_other_fields():
    old = {'physics': {'lambda_exec': .02, 'horizon': 32}, 'training': {'max_steps': 200000}, 'asset': 'fixed'}
    new = copy.deepcopy(old)
    new['physics']['gradient_balance'] = balance_config()
    check_restore_contract(old, new, continue_objective=True)
    with pytest.raises(ValueError):
        check_restore_contract(old, new)
    bad = copy.deepcopy(new)
    bad['physics']['horizon'] = 16
    with pytest.raises(ValueError):
        check_restore_contract(old, bad, continue_objective=True)
    with pytest.raises(ValueError):
        check_restore_contract(new, new, continue_objective=True)
    check_restore_contract(new, new)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Linear(2, 1)
        self.unused = torch.nn.Parameter(torch.ones(3))

    def forward(self, x):
        return self.shared(x)


def _losses(model, rank, empty=False):
    x = torch.tensor([[1., 2.], [3., -1.]]) + rank
    pred = model(x)
    supervised = (pred-(1.+rank)).square().mean()
    # rank0有2个物理样本，rank1为0；world/global_count=1。
    execution = (pred+.3).square().sum() if rank == 0 and not empty else pred.sum()*0.
    return supervised, execution


def _worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://'+rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(42)
        actor = Tiny()
        reference = copy.deepcopy(actor)
        ddp = DistributedDataParallel(actor, find_unused_parameters=True, gradient_as_bucket_view=True)
        optimizer = torch.optim.AdamW(actor.parameters(), lr=.01)
        ref_optimizer = torch.optim.AdamW(reference.parameters(), lr=.01)
        observations = []
        for index in range(5):
            optimizer.zero_grad(set_to_none=True)
            ref_optimizer.zero_grad(set_to_none=True)
            calibration = index % 2 == 0
            empty = index == 2
            d0, p0 = _losses(reference, 0, empty)
            d1, _ = _losses(reference, 1, empty)
            global_data = (d0+d1)/2
            global_physics = p0/2
            expected = global_gradient_measurement(global_data, global_physics, reference, 1, ('shared',))
            if calibration:
                with ddp.no_sync():
                    data, physics = _losses(ddp, rank, empty)
                    measured = global_gradient_measurement(data, physics, actor, 2, ('shared',))
                    (data+.7*physics).backward()
                    average_parameter_gradients(actor, 2)
                for key in ('supervised_norm', 'execution_norm', 'cosine'):
                    assert measured[key] == pytest.approx(expected[key], rel=1e-5, abs=1e-7)
                assert measured['modules']['shared'] == pytest.approx(expected['modules']['shared'], rel=1e-5, abs=1e-7)
                observations.append(measured)
            else:
                data, physics = _losses(ddp, rank, empty)
                (data+.7*physics).backward()
            (global_data+.7*global_physics).backward()
            for actual, target in zip(actor.parameters(), reference.parameters()):
                if target.grad is None:
                    assert actual.grad is None
                else:
                    torch.testing.assert_close(actual.grad, target.grad)
            optimizer.step()
            ref_optimizer.step()
            for actual, target in zip(actor.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, target)
        torch.save(observations, Path(directory)/f'rank{rank}.pt')
    finally:
        dist.destroy_process_group()


def test_two_rank_calibration_and_regular_backward(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path/'rendezvous'), str(tmp_path)), nprocs=2, join=True)
    a, b = [torch.load(tmp_path/f'rank{rank}.pt', weights_only=False) for rank in (0, 1)]
    assert a == b
    assert a[1]['execution_norm'] == 0.
