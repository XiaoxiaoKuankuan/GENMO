#!/usr/bin/env python3
"""在正式训练前验证每张 GPU 的计算和完整 DDP 拓扑的大张量通信。

通过 torchrun 启动，每个 rank 先绑定 LOCAL_RANK，再独立运行矩阵乘法并核对结果；
随后初始化明确设备的 NCCL，分别用 32 MiB、256 MiB float32 张量验证 all-reduce
和 rank0 broadcast。所有缓冲在本进程退出时释放，不产生 checkpoint 或训练目录。
该脚本是运行就绪检查，不证明模型训练收敛或生成质量；必须在用户授权的空闲 GPU 上运行。
"""

import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    a = torch.ones((1024, 1024), device=device)
    assert torch.equal(a @ a, torch.full_like(a, 1024))
    torch.cuda.synchronize()
    print(json.dumps({"rank": rank, "device": local_rank, "independent_cuda": "PASS"}), flush=True)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=120))
    try:
        for size_mib in (32, 256):
            tensor = torch.full((size_mib * 1024 * 1024 // 4,), float(rank + 1), device=device)
            dist.all_reduce(tensor)
            assert bool((tensor == world_size * (world_size + 1) / 2).all())
            tensor.fill_(3.0 if rank == 0 else -1.0)
            dist.broadcast(tensor, src=0)
            assert bool((tensor == 3.0).all())
            del tensor
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "world_size": world_size,
                            "size_mib": size_mib,
                            "all_reduce": "PASS",
                            "broadcast": "PASS",
                        }
                    ),
                    flush=True,
                )
        dist.barrier(device_ids=[local_rank])
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
