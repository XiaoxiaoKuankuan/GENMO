"""GENMO–GMT独立AWR入口：预检、完整宏迭代、停止恢复及物理评价。

训练仅在独立feature/stage2-awr工作树执行，先核验空闲八卡与冻结资产，再创建NCCL；
PhysX由每阶段专属GMT子进程承载。默认先导只运行配置中1个宏迭代，包含同Replay
自蒸馏对照。stop写本运行停止标志并排空，resume沿用保存身份；不停止其他任务。
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("preflight", "run", "evaluate", "resume", "stop", "ablate"))
    parser.add_argument("--config", type=Path, default=ROOT/"configs/closedloop/stage2_awr_group_credit.yaml")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--profile", choices=("smoke", "pilot"))
    parser.add_argument("--genmo-repo", type=Path, default=ROOT)
    parser.add_argument("--gmt-repo", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="evaluate专用，本运行checkpoint的JSON引用")
    parser.add_argument("--all-unique", action="store_true")
    args = parser.parse_args()
    from gem.closedloop.awr.config import load_config, preflight
    from gem.closedloop.awr.io import atomic_json
    if args.operation != "preflight" and args.run_dir is None:
        parser.error("--run-dir is required")
    if args.operation == "stop":
        if not (args.run_dir/"state.json").is_file():
            parser.error("No AWR run state exists at this exact path")
        atomic_json(args.run_dir/"STOP", dict(reason="user_requested_graceful_stop"))
        print("已请求停止发新任务；当前Episode和完整优化步将先完成。", flush=True)
        return 0
    resume = args.operation in ("resume", "evaluate")
    if resume:
        config = json.loads((args.run_dir/"request.json").read_text())
    else:
        config = load_config(args.config, profile=args.profile)
        config["paths"]["genmo_repo"] = str(args.genmo_repo.resolve())
        if args.gmt_repo is not None:
            config["paths"]["gmt_repo"] = str(args.gmt_repo.resolve())
    if args.operation == "preflight":
        print(json.dumps(preflight(config), ensure_ascii=False, indent=2))
        return 0
    for key, value in dict(NCCL_CUMEM_HOST_ENABLE="0", NCCL_IB_DISABLE="1", NCCL_SOCKET_IFNAME="lo",
                           TORCH_NCCL_BLOCKING_WAIT="1", PYTHONDONTWRITEBYTECODE="1").items():
        os.environ.setdefault(key, value)
    rank, world, local = [int(os.environ.get(k, "-1")) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK")]
    if world != 8 or rank != local or os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1,2,3,4,5,6,7":
        parser.error("Use single-node torchrun --nproc_per_node=8 with CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7")
    import torch
    import torch.distributed as dist
    from gem.closedloop.awr.distributed import Distributed
    dist.init_process_group("gloo", timeout=timedelta(hours=2))
    tensor_group = None
    resource_monitor = None
    try:
        d = Distributed(device=f"cuda:{local}")
        audit = d.root(lambda: preflight(config))
        torch.cuda.set_device(local); torch.set_num_threads(config["runtime"]["torch_threads"])
        torch.manual_seed(config["awr"]["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        tensor_group = dist.new_group(backend="nccl", timeout=timedelta(hours=1))
        d.tensor_group = tensor_group
        if float(d.sum(torch.tensor(rank+1., device=d.device))) != 36.:
            raise RuntimeError("Eight-GPU SUM validation failed")
        os.environ.setdefault("ISAACLAB_PATH", config["runtime"]["isaaclab_root"])
        libraries = config["runtime"]["sysroot_library"]
        os.environ["LD_LIBRARY_PATH"] = libraries+os.pathsep+os.environ.get("LD_LIBRARY_PATH", "")
        from gem.closedloop.awr.runtime import AWRRun
        run = AWRRun(config, args.run_dir, d, audit, resume=resume)
        from gem.closedloop.awr.resources import GPUResourceMonitor
        if rank == 0:
            resource_monitor = GPUResourceMonitor(args.run_dir).start()
        if args.operation == "evaluate":
            try:
                reference = json.loads(args.checkpoint.read_text()) if args.checkpoint else None
                run.manual_evaluate(reference, all_unique=args.all_unique)
                return 0
            finally:
                if run.replay: run.replay.close()
        return run.run()
    finally:
        if resource_monitor is not None:
            resource_monitor.close()
        if tensor_group is not None: dist.destroy_process_group(tensor_group)
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
