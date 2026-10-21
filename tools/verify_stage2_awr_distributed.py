"""Server-2真实Stage1网络的单卡/八卡梯度与更新等价验收。

读取冻结s595000及原Stage1真实标签，固定t和noise，用相同加权去噪/完整BC损失比较
单卡逐样本参考与八卡尾批、空rank、不同microbatch的SUM梯度。这里只证明计算等价，
不采集PhysX、不产生RL学习结论；不更改源checkpoint，报告写用户指定独立证据路径。
配置关闭BC时只将真实Stage1样本用作固定输入，不执行BC前反向；时间/来源因子与
ActorTrainer使用同一实现，在不同rank切分前固定全局分母和权重表。
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/closedloop/stage2_awr_rewardstd_nobc.yaml")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--gmt-repo", type=Path)
    args = parser.parse_args()
    import torch
    import torch.distributed as dist
    from gem.closedloop.awr.config import load_config, physical_config, preflight
    from gem.closedloop.awr.distributed import Distributed
    from gem.closedloop.awr.losses import weighted_denoising
    from gem.closedloop.awr.objective import timestep_weights, source_weights
    from gem.closedloop.awr.io import atomic_json
    from gem.closedloop.common.supervision import load_actor, Stage1Supervision
    from gem.closedloop.stage1_dataset import collate_stage1_training_samples
    from gem.closedloop.contracts import STAGE1_CONDITION_KEYS
    rank = int(os.environ["RANK"])
    if int(os.environ["WORLD_SIZE"]) != 8:
        parser.error("This acceptance requires exactly eight ranks")
    dist.init_process_group("gloo", timeout=timedelta(minutes=30))
    group = None
    started = time.perf_counter()
    try:
        config = load_config(args.config); config["paths"]["genmo_repo"] = str(ROOT)
        if args.gmt_repo is not None:
            config["paths"]["gmt_repo"] = str(args.gmt_repo.resolve())
        d = Distributed(f"cuda:{rank}")
        evidence = d.root(lambda: preflight(config))
        torch.cuda.set_device(rank); torch.set_num_threads(4); torch.manual_seed(812)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
        group = dist.new_group(backend="nccl", timeout=timedelta(minutes=30))
        d.tensor_group = group
        c = physical_config(config, run_id="awr_distributed_acceptance", rank=rank, device=d.device)
        actor, train, report = load_actor(c)
        d.sync_model(actor)
        bc = Stage1Supervision(c, actor, train, source_step=report["source_global_step"])
        def make_batch():
            import numpy as np
            import random
            np.random.seed(812); random.seed(812); torch.manual_seed(812)
            return collate_stage1_training_samples([bc.datasets[i % 4][i] for i in range(9)])
        batch = d.root(make_batch)
        actor.eval()
        generator = torch.Generator().manual_seed(813)
        noise = torch.randn(batch["target_qpos30"].shape, generator=generator)
        times = torch.randint(actor.diffusion_steps, (9,), generator=generator)
        time_table = timestep_weights(actor.train_diffusion.alphas_cumprod,
                                     config["awr"]["actor"].get("timestep_weighting"))
        bc_enabled = config["awr"]["actor"].get("bc_enabled", True)
        parameters = [p for p in actor.parameters() if p.requires_grad]
        reports = []
        def backward(ids, n_awr, n_bc, micro):
            for offset in range(0, len(ids), micro):
                indices = ids[offset:offset+micro]
                context = {k: batch[k][indices].to(d.device) for k in STAGE1_CONDITION_KEYS}
                target = batch["target_qpos30"][indices].to(d.device)
                valid = batch["target_qpos30_valid"][indices].to(d.device)
                out = actor.denoising_forward(context, target, valid,
                                               timesteps=times[indices].to(d.device), noise=noise[indices].to(d.device),
                                               guidance_scale=config["awr"]["actor"]["denoising_guidance_scale"])
                weights = torch.tensor([1.+i/3 for i in indices], device=d.device)
                balance = source_weights([str(i % 4) for i in range(n_awr)],
                                         config["awr"]["actor"].get("source_balance", "none"))
                weights = weights*time_table[times[indices]].to(d.device)*balance[indices].to(d.device)
                loss, _ = weighted_denoising(out["pred_x_start"], out["target_x_start"],
                    valid & ~context["known_qpos30_mask"], weights, global_count=n_awr)
                loss.backward()
            for index in range(rank if n_awr != -1 else 0, n_bc, d.world_size if n_awr != -1 else 1):
                single = {k: v[index:index+1].to(d.device) if torch.is_tensor(v) else
                          v[index:index+1] if isinstance(v, (list, tuple)) else v for k, v in batch.items()}
                if "B" in single: single["B"] = 1
                out = actor.training_forward(single, timesteps=times[index:index+1].to(d.device),
                                              noise=noise[index:index+1].to(d.device))
                loss, _ = bc.losses(single, out["pred_x_start"], out["static_conf_logits"],
                                   global_step=report["source_global_step"])
                (.1*loss/n_bc).backward()
        for n, bc_count, micro in ((9, 3, 2), (3, 1, 1)):
            if not bc_enabled:
                bc_count = 0
            actor.zero_grad(set_to_none=True)
            # 参考AWR和BC分别在rank0遍历全部真实样本，避免任何拓扑相关分母。
            reference = None
            if rank == 0:
                backward(list(range(n)), n, 0, 1)
                for index in range(bc_count):
                    single = {k: v[index:index+1].to(d.device) if torch.is_tensor(v) else
                              v[index:index+1] if isinstance(v, (list, tuple)) else v for k, v in batch.items()}
                    if "B" in single: single["B"] = 1
                    out = actor.training_forward(single, timesteps=times[index:index+1].to(d.device),
                                                  noise=noise[index:index+1].to(d.device))
                    loss, _ = bc.losses(single, out["pred_x_start"], out["static_conf_logits"],
                                       global_step=report["source_global_step"])
                    (.1*loss/bc_count).backward()
                reference = [None if p.grad is None else p.grad.detach().cpu().clone() for p in parameters]
            d.gather(True)
            actor.zero_grad(set_to_none=True)
            backward(list(range(rank, n, 8)), n, bc_count, micro)
            d.gradients(actor)
            def compare():
                squared, norm, maximum = 0., 0., 0.
                for expected, p in zip(reference, parameters):
                    if (expected is None) != (p.grad is None):
                        raise ValueError("Distributed unused-parameter state differs")
                    if expected is None: continue
                    delta = p.grad.detach().cpu().double()-expected.double()
                    squared += float(delta.square().sum()); norm += float(expected.double().square().sum())
                    maximum = max(maximum, float(delta.abs().max()))
                relative = (squared/max(norm, 1e-30))**.5
                if not torch.isfinite(torch.tensor(relative)) or relative > 2e-5:
                    raise ValueError(f"Eight-GPU gradient equivalence failed: {relative}")
                return dict(awr_samples=n, bc_samples=bc_count, microbatch=micro,
                            relative_gradient_l2=relative, maximum_absolute_gradient_error=maximum)
            result = d.root(compare)
            # 对所有参数执行相同SGD验证通信后的更新一致性，源权重始终只读。
            torch.optim.SGD(parameters, lr=1e-4).step()
            result["updated_parameter_identity"] = d.assert_models_equal(actor)
            reports.append(result)
        peaks = d.gather(torch.cuda.max_memory_allocated(d.device))
        d.root(lambda: atomic_json(args.report, dict(status="passed", real_stage1=True, real_stage1_bc=bc_enabled,
            bc_enabled=bc_enabled, timestep_weighting=config["awr"]["actor"].get("timestep_weighting"),
            source_balance=config["awr"]["actor"].get("source_balance", "none"),
            distributed_backend="nccl", world_size=8, cases=reports, peak_memory_bytes=peaks,
            wall_seconds=time.perf_counter()-started, preflight=evidence)))
        if rank == 0: print("真实Stage1八卡梯度和更新一致性验收通过", flush=True)
    finally:
        if group is not None: dist.destroy_process_group(group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
