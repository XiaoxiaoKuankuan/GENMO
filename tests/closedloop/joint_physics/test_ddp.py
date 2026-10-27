"""真实双进程Gloo/DDP的非均匀有效物理样本归约回归。

rank0有两个有效样本而rank1为零，结果与单进程全局样本均值梯度比较；随后两rank都零
样本仍同步完成backward。这里只验证分布式归约和死锁边界，线性模型不作为GENMO/物理
有效性证据；完整架构DDP有界运行由服务器CLI另外验收。
"""

from pathlib import Path

import torch
from torch import distributed as dist
from torch import multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from gem.closedloop.joint_physics.trainer import distributed_execution_scale


class _TwoInputs(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Linear(1, 1, bias=False)

    def forward(self, data, physical):
        return self.shared(data), self.shared(physical)


def _worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://"+rendezvous, rank=rank, world_size=2)
    try:
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(.7)
        ddp = DistributedDataParallel(model)
        results = []
        for empty in (False, True):
            ddp.zero_grad(set_to_none=True)
            output = ddp(torch.tensor([[1.], [3.]]) if rank == 0 else torch.tensor([[2.]]))
            active = rank == 0 and not empty
            loss = (output-torch.tensor([[0.], [1.]])).square().sum() if active else output.sum()*0
            count, scale = distributed_execution_scale(2 if active else 0, "cpu", 2)
            (scale*loss).backward()
            results.append((float(count), model.weight.grad.clone()))
        torch.save(results, Path(directory)/f"rank{rank}.pt")
        shared = _TwoInputs()
        with torch.no_grad():shared.shared.weight.fill_(.7)
        double = DistributedDataParallel(shared, find_unused_parameters=True)
        data, rejected = double(torch.tensor([[1.+rank]]), torch.tensor([[float('nan')]]))
        assert not torch.isfinite(rejected).all()
        # 被拒绝前向完全不接入loss；参数均由同一个网络的数据分支同步反传。
        data.square().mean().backward()
        assert torch.isfinite(shared.shared.weight.grad).all()
        torch.save(shared.shared.weight.grad, Path(directory)/f"nonfinite_rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_two_rank_counts_include_zero_and_match_global_mean(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path/"rendezvous"), str(tmp_path)), nprocs=2, join=True)
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(.7)
    (model(torch.tensor([[1.], [3.]]))-torch.tensor([[0.], [1.]])).square().mean().backward()
    for rank in (0, 1):
        results = torch.load(tmp_path/f"rank{rank}.pt", weights_only=False)
        assert results[0][0] == 2 and results[1][0] == 0
        torch.testing.assert_close(results[0][1], model.weight.grad)
        assert torch.equal(results[1][1], torch.zeros_like(model.weight.grad))
        torch.testing.assert_close(torch.load(tmp_path/f"nonfinite_rank{rank}.pt",weights_only=True),torch.tensor([[3.5]]))
