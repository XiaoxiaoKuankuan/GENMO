"""新联合训练完整断点，禁止旧Stage2权重或优化器冒充恢复。

保存同一GENMO/AdamW/scheduler、课程、每rank数据/噪声/窗口游标、全部RNG及实际资产身份。
保存到新编号文件，原子发布且不覆盖负结果。恢复先核对schema与张量/数据/物理合同，源码
提交仅记录。EMA首版不启用，既无生成教师也无第二个优化器。
"""

import os
from pathlib import Path
import random

import numpy as np
import torch

from .config import SCHEMA

CHECKPOINT_SCHEMA = SCHEMA + ".checkpoint.v1"


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f".partial-{os.getpid()}")
    created = False
    try:
        with temporary.open("xb") as handle:
            created = True
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)  # 目标已存在时拒绝，不覆盖旧checkpoint。
    finally:
        if created:
            temporary.unlink(missing_ok=True)


def read_checkpoint(path):
    value = torch.load(path, map_location="cpu", weights_only=False)
    required = {"actor", "optimizer", "scheduler", "global_step", "curriculum", "rank_states", "contract", "counters", "config"}
    if value.get("schema") != CHECKPOINT_SCHEMA or not required <= value.keys():
        raise ValueError("只接受本联合流程完整checkpoint；旧Stage2不是等价恢复状态")
    return value


def save_trainer(trainer, path):
    from torch import distributed as dist
    rank_state = {"data": trainer.stream.state_dict(), "noise": trainer.noise_rng.get_state(),
                  "window": trainer.window_rng.bit_generator.state, "rng": rng_state()}
    states = [None] * trainer.world_size
    if trainer.world_size > 1:
        dist.all_gather_object(states, rank_state)
    else:
        states[0] = rank_state
    if trainer.rank == 0:
        atomic_save({"schema": CHECKPOINT_SCHEMA, "actor": trainer.actor.state_dict(),
            "optimizer": trainer.optimizer.state_dict(), "scheduler": trainer.scheduler.state_dict(),
            "global_step": trainer.step, "curriculum": {"step": trainer.step, "physics": trainer.cfg["physics"]},
            "rank_states": states, "contract": trainer.contract, "counters": trainer.counters,
            "config": trainer.cfg, "initialization": trainer.actor.initialization}, path)
    if trainer.world_size > 1:
        dist.barrier()


def restore_trainer(trainer, path):
    value = read_checkpoint(path)
    if value["contract"] != trainer.contract or len(value["rank_states"]) != trainer.world_size:
        raise ValueError("恢复的架构/数据/资产/物理/采样预算或DDP布局不一致")
    trainer.actor.load_state_dict(value["actor"], strict=True)
    trainer.optimizer.load_state_dict(value["optimizer"])
    trainer.scheduler.load_state_dict(value["scheduler"])
    trainer.step, trainer.counters = value["global_step"], value["counters"]
    state = value["rank_states"][trainer.rank]
    trainer.stream.load_state_dict(state["data"])
    trainer.noise_rng.set_state(state["noise"].cpu())
    trainer.window_rng.bit_generator.state = state["window"]
    restore_rng(state["rng"])
