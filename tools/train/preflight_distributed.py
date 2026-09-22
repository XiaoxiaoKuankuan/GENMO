#!/usr/bin/env python3
"""在正式训练前验证每张 GPU 的计算和完整 DDP 拓扑的大张量通信。

通过 torchrun 启动，每个 rank 先绑定 LOCAL_RANK，再独立运行矩阵乘法并核对结果；
随后初始化明确设备的 NCCL，分别用 32 MiB、256 MiB float32 张量验证 all-reduce
和 rank0 broadcast。所有缓冲在本进程退出时释放，不产生 checkpoint 或训练目录。
可选 --bumi-batch-size 使用正式四库 release、冻结 T5、完整 BUMI 网络和真实 AdamW
实测前向/反向/优化后的显存及吞吐，辅助选择大显存 batch；不写训练 checkpoint。
该脚本是运行就绪检查，不证明模型训练收敛或生成质量；必须在用户授权的空闲 GPU 上运行。
"""

import argparse
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.distributed as dist


def benchmark_bumi(args, device, rank, world_size):
    """保持正式表示、损失和文本，测量含优化器状态的实际训练开销。"""
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from torch.utils._pytree import tree_map

    os.environ["BUMI_TEXT_DATA_ROOT"] = str(Path(args.bumi_data_root).resolve(strict=True))
    os.environ["BUMI_TEXT_STATS_PATH"] = str(Path(args.bumi_stats).resolve(strict=True))
    os.environ["BUMI_TEXT_ONLINE_T5"] = "true"
    os.environ["BUMI_T5_MODEL_PATH"] = str(Path(args.t5_path).resolve(strict=True))
    with initialize_config_dir(
        version_base="1.3", config_dir=str(Path(__file__).resolve().parents[2] / "configs")
    ):
        cfg = compose(config_name="train")
    cfg.data.loader_opts.train.batch_size = args.bumi_batch_size
    cfg.data.loader_opts.train.num_workers = args.workers
    dm = instantiate(cfg.data, _recursive_=False)
    dm._trainer = SimpleNamespace(
        global_rank=rank, world_size=world_size, accumulate_grad_batches=1
    )
    model = instantiate(cfg.model, _recursive_=False).to(device).train()
    model._trainer = SimpleNamespace(datamodule=dm)
    # 完整优化器状态从第一次 optimizer.step 创建；冻结 T5 不进入 AdamW。
    configured = model.configure_optimizers()
    optimizer = configured[0][0] if isinstance(configured, tuple) else configured
    loader = iter(dm.train_dataloader())
    durations, losses = [], []
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(args.steps):
        start = time.perf_counter()
        batch = tree_map(lambda v: v.to(device) if isinstance(v, torch.Tensor) else v, next(loader))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model.prepare_batch(batch, "diffusion")
            model.create_condition_mask(batch, train=True)
            result = model.pipeline(batch, train=True, mode="diffusion", global_step=10000)
        loss = result["loss"]
        if not torch.isfinite(loss):
            raise ValueError("batch 预检 loss 非有限")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        durations.append(elapsed)
        losses.append(float(loss.detach()))
        print(
            json.dumps(
                dict(
                    stage="bumi_batch",
                    rank=rank,
                    step=step,
                    batch=args.bumi_batch_size,
                    seconds=elapsed,
                    loss=losses[-1],
                    grad_norm=float(norm),
                    peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                    peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20,
                )
            ),
            flush=True,
        )
        del batch, result, loss
    encoder = model.online_text_encoder
    print(
        json.dumps(
            dict(
                stage="bumi_batch_complete",
                rank=rank,
                batch=args.bumi_batch_size,
                steady_seconds=sum(durations[1:]) / len(durations[1:]),
                peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20,
                t5_cache_hits=encoder.hits,
                t5_cache_misses=encoder.misses,
                parameters=sum(p.numel() for p in model.parameters()),
            )
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bumi-batch-size", type=int)
    parser.add_argument("--bumi-data-root", type=Path)
    parser.add_argument("--bumi-stats", type=Path)
    parser.add_argument("--t5-path", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()
    if args.bumi_batch_size and (
        args.steps < 2 or not all((args.bumi_data_root, args.bumi_stats, args.t5_path))
    ):
        parser.error("BUMI batch 预检需要真实 release/stats/T5 路径及至少两个优化步骤")
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
        if args.bumi_batch_size:
            benchmark_bumi(args, device, rank, world_size)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
