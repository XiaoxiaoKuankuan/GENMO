"""新联合训练完整断点，禁止旧Stage2权重或优化器冒充恢复。

保存同一GENMO/AdamW/scheduler、课程、每rank数据/噪声/窗口游标、全部RNG及实际资产身份。
保存到新编号文件，原子发布且不覆盖负结果。恢复先核对schema与张量/数据/物理合同，源码
提交仅记录。EMA首版不启用，既无生成教师也无第二个优化器。
"""

import copy
import os
from pathlib import Path
import random

import numpy as np
import torch

from .config import SCHEMA

CHECKPOINT_SCHEMA = SCHEMA + ".checkpoint.v3.formal_prefix_future"


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "cpu": torch.get_rng_state(),
            "cuda_current": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"].cpu())
    if state.get("cuda_current") is not None:
        torch.cuda.set_rng_state(state["cuda_current"].cpu())
    elif state.get("cuda"):
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
    value = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    required = {"actor", "optimizer", "scheduler", "global_step", "curriculum", "rank_states", "contract", "counters", "config"}
    if value.get("schema") not in (CHECKPOINT_SCHEMA, SCHEMA+".checkpoint.v2.independent_physics", SCHEMA+".checkpoint.v1") or not required <= value.keys():
        raise ValueError("只接受本联合流程完整checkpoint；旧Stage2不是等价恢复状态")
    return value


def save_trainer(trainer, path):
    from torch import distributed as dist
    if any(p.dtype != torch.float32 for p in trainer.actor.parameters()):
        raise ValueError('完整断点要求GENMO主参数FP32')
    if any(value.is_floating_point() and value.dtype != torch.float32
           for state in trainer.optimizer.state.values() for value in state.values()
           if isinstance(value,torch.Tensor)):
        raise ValueError('AdamW状态必须保持FP32')
    rank_state = {"data": trainer.stream.state_dict(), "noise": trainer.noise_rng.get_state(),
                  "window": trainer.window_rng.bit_generator.state, "rng": rng_state(),
                  "physics_data": trainer.physics_stream.state_dict() if trainer.physics_stream is not None else None,
                  "physics_noise": trainer.physics_noise_rng.get_state(),
                  "future_data": trainer.future_stream.state_dict() if trainer.future_stream is not None else None,
                  "future_noise": trainer.future_noise_rng.get_state()}
    states = [None] * trainer.world_size
    if trainer.world_size > 1:
        dist.all_gather_object(states, rank_state)
    else:
        states[0] = rank_state
    if trainer.rank == 0:
        atomic_save({"schema": CHECKPOINT_SCHEMA, "actor": trainer.actor.state_dict(),
            "optimizer": trainer.optimizer.state_dict(), "scheduler": trainer.scheduler.state_dict(),
            "global_step": trainer.step, "curriculum": {"step": trainer.step, "physics": trainer.cfg["physics"],
                                                        "readiness": trainer.readiness.state_dict()},
            "rank_states": states, "contract": trainer.contract, "counters": trainer.counters,
            "config": trainer.cfg, "sidecar_snapshot": trainer.store.snapshot() if trainer.store else None,
            "gradient_balance": trainer.gradient_balance.state_dict() if trainer.gradient_balance else None,
            "continuation": trainer.continuation,
            "precision_contract": {"network_autocast": trainer.cfg['training']['precision'],
                                   "parameters": "float32", "optimizer_states": "float32", "physics": "float32"},
            "initialization": trainer.actor.initialization}, path)
    if trainer.world_size > 1:
        dist.barrier()


def check_restore_contract(previous, current, *, continue_objective=False):
    """显式续训只允许增加梯度权重课程，其他资产/数学/采样合同仍必须完全一致。"""
    previous, current = copy.deepcopy(previous), copy.deepcopy(current)
    if continue_objective:
        if 'gradient_balance' in previous['physics'] or not current['physics'].get('gradient_balance', {}).get('enabled'):
            raise ValueError('目标切换仅支持固定权重训练接入新的梯度课程；已有课程请严格resume')
        current['physics'].pop('gradient_balance')
    if previous != current:
        raise ValueError('恢复的架构/数据/资产/物理/采样预算不一致')


def restore_trainer(trainer, path, value=None, *, continue_objective=False):
    value = read_checkpoint(path) if value is None else value
    if value["schema"] != CHECKPOINT_SCHEMA:
        raise ValueError("独立物理抽样只恢复本算法完整状态；旧共用噪声checkpoint仅可评价")
    check_restore_contract(value['contract'], trainer.contract, continue_objective=continue_objective)
    if len(value["rank_states"]) != trainer.world_size:
        raise ValueError('恢复的DDP布局不一致')
    if trainer.gradient_balance and not continue_objective and not value.get('gradient_balance'):
        raise ValueError('严格恢复缺少完整梯度权重控制器状态')
    trainer.actor.load_state_dict(value["actor"], strict=True)
    trainer.optimizer.load_state_dict(value["optimizer"])
    trainer.scheduler.load_state_dict(value["scheduler"])
    trainer.step, trainer.counters = value["global_step"], value["counters"]
    state = value["rank_states"][trainer.rank]
    trainer.stream.load_state_dict(state["data"])
    trainer.noise_rng.set_state(state["noise"].cpu())
    trainer.window_rng.bit_generator.state = state["window"]
    trainer.physics_noise_rng.set_state(state["physics_noise"].cpu())
    if trainer.physics_stream is not None:
        trainer.physics_stream.load_state_dict(state["physics_data"])
    trainer.future_noise_rng.set_state(state["future_noise"].cpu())
    if trainer.future_stream is not None:
        trainer.future_stream.load_state_dict(state["future_data"])
    trainer.readiness.load_state_dict(value["curriculum"]["readiness"])
    if continue_objective:
        trainer.gradient_balance.start_step = trainer.step
        trainer.continuation = {'parent_checkpoint': str(Path(path).resolve()), 'parent_global_step': trainer.step,
            'change': 'fixed_lambda_to_global_parameter_gradient_share',
            'preserved': 'actor_adamw_scheduler_all_rank_rng_and_samplers',
            'equivalent_objective_resume': False}
    else:
        trainer.continuation = value.get('continuation')
        if trainer.gradient_balance:
            trainer.gradient_balance.load_state_dict(value['gradient_balance'])
    restore_rng(state["rng"])
