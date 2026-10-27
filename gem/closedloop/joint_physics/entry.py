"""独立数据准备、训练、评价和导数审计CLI。

读取本方法配置后才加载原SM bridge；只有显式真实物理命令才启动Newton。训练默认smoke且
限制步数，train/torchrun由操作者显式启动。所有输出独占新目录；恢复仍写新目录。命令行
和文档共享--set覆盖语义，不启动后台长期训练、不修改已有任务。
"""

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys


def main(kind):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/closedloop/joint_physics_from_scratch.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--output")
    parser.add_argument("--torch-device")
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train" if kind == "prepare" else "val")
    parser.add_argument("--max-sources", type=int)
    parser.add_argument("--max-control-steps", type=int)
    parser.add_argument("--newton", action="store_true")
    parser.add_argument("--interface-probe", action="store_true", help="仅GT接口/方向审计，绝不算学生训练成功")
    parser.add_argument("--complete-validation", help="对已有本方法原始Newton报告补充已记录目标合同和真实前瞻检查，不覆盖原报告")
    parser.add_argument("--mode", choices=("smoke", "train") if kind == "train" else ("reconstruction", "free_generation", "closed_loop"),
                        default="smoke" if kind == "train" else "reconstruction")
    preliminary, _ = parser.parse_known_args()
    from .config import read_config, build_adapter, seed_all
    cfg = read_config(preliminary.config, preliminary.set)
    sys.path.insert(0, str(Path(cfg["supermimic"]["repo"]).resolve()))
    physical = kind in ("prepare", "evaluate") or (kind == "gradients" and preliminary.newton)
    if physical:
        from isaaclab.app import add_launcher_args, launch_simulation
        add_launcher_args(parser)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(cfg["runtime"]["cpu_threads"])
    seed_all(cfg["training"]["seed"])
    device = args.torch_device or cfg["runtime"]["device"]
    output = args.output or cfg["runtime"]["output"] or f"outputs/joint_physics/{kind}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    if kind == "train":
        if args.mode == "train" and cfg["physics"]["enabled"] and (not cfg["data"]["sidecar"] or not cfg["physics"]["validation_report"]):
            raise ValueError("正式联合训练需真实sidecar及本方法验证报告，不能长期只有BC却称联合训练")
        from torch import distributed as dist
        from .trainer import JointTrainer
        world_size, rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))
        if world_size > 1:
            local_rank = int(os.environ["LOCAL_RANK"])
            device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
            if device != "cpu":
                torch.cuda.set_device(local_rank)
            dist.init_process_group("nccl" if device != "cpu" else "gloo")
            names = [output if rank == 0 else None]
            dist.broadcast_object_list(names, src=0)
            output = names[0]
        stop = args.max_steps if args.max_steps is not None else (8 if args.mode == "smoke" else cfg["training"]["max_steps"])
        if stop < 1 or (args.mode == "smoke" and stop > cfg["training"]["smoke_max_steps"]):
            raise ValueError("smoke必须在显式有界预算内")
        trainer = JointTrainer(cfg, device, rank=rank, world_size=world_size,
                               resume=args.resume or cfg["initialization"].get("resume_checkpoint"))
        if stop <= trainer.step:
            raise ValueError("max-steps是新run绝对global_step上限，必须大于恢复步数")
        try:
            trainer.run(output, stop)
        finally:
            if world_size > 1:
                dist.destroy_process_group()
        return
    adapter = build_adapter(cfg, device)
    if args.max_sources is not None:
        cfg["prepare"]["max_sources"] = cfg["evaluation"]["max_sources"] = args.max_sources
    if args.max_control_steps is not None:
        cfg["prepare"]["max_control_steps"] = cfg["evaluation"]["max_control_steps"] = args.max_control_steps
    def execute(env):
        if kind == "prepare":
            from .prepare_rollouts import prepare
            return prepare(cfg, args.split, output, adapter, env)
        if kind == "gradients":
            from .evaluation import audit_gradients, complete_validation
            if args.complete_validation:
                return complete_validation(cfg, args.complete_validation, output, adapter)
            return audit_gradients(cfg, args.checkpoint, output, adapter, env, args.interface_probe)
        from .evaluation import evaluate
        return evaluate(cfg, args.checkpoint, output, adapter, env, args.mode)
    if not physical:
        execute(None)
        return
    from supermimic.env import make_sim
    from supermimic.bridge.streaming_env import StreamingPhysicalEnv
    from isaaclab.utils import replace
    sim_cfg = make_sim(device, args)
    sim_cfg = replace(sim_cfg, physics=replace(sim_cfg.physics, class_type=None,
        solver_cfg=replace(sim_cfg.physics.solver_cfg, disable_sensors=True),
        deterministic_mode=cfg["physics"]["determinism"]))
    with launch_simulation(sim_cfg, args):
        env = StreamingPhysicalEnv(sim_cfg, 1, adapter.bundle)
        execute(env)
