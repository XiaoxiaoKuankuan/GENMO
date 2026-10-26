"""新Stage2训练状态的原子保存与恢复，只持久化两个优化器。

保存GENMO/未来编码、独立Critic、优化器、CPU/CUDA/NumPy/Python随机状态、配对数据
游标、延迟采样与近期真实回放。参考仅保存紧凑张量，加载时重新绑定同一权威FK。
源码版本只记录；冻结资产内容、采样/物理合同需要匹配。没有保存物理求解器，因此
恢复后采集必须显式新episode，不把qpos/回放恢复宣传为物理精确续跑。
"""

from dataclasses import fields, is_dataclass
import random

import numpy as np
import torch

from supermimic.bridge.bundle import atomic_save
from .contracts import ClosedLoopState
from .reference_buffer import ReferenceBuffer

SCHEMA = "genmo.supermimic.bptt.training.v3"


def optimizer_layout(module, optimizer):
    names = {id(p): name for name, p in module.named_parameters()}
    return [[names[id(p)] for p in group["params"]] for group in optimizer.param_groups]


def pack_tensor(value, memo):
    """一次保存内复用相同张量视图的CPU副本，尤其是所有时刻共享的整首音乐。

    不按数值近似合并不同状态；键包含存储、偏移、形状、步幅、类型和设备。
    memo仅在一次同步保存内有效，不能跨Actor更新或物理推进复用。
    """
    key = (str(value.device), value.untyped_storage().data_ptr(), value.storage_offset(),
           tuple(value.shape), tuple(value.stride()), value.dtype)
    if key not in memo:
        memo[key] = value.detach().cpu()
    return memo[key]


def pack_state(state, memo=None):
    memo = {} if memo is None else memo
    values = {f.name: getattr(state, f.name) for f in fields(state) if f.name not in ("reference", "teacher_reference")}
    values = {k: pack_tensor(v, memo) if isinstance(v, torch.Tensor) else v for k, v in values.items()}
    values["pending_request"] = pack_value(state.pending_request, memo)
    cache = state.reference
    values["reference"] = {k: pack_tensor(getattr(cache, k), memo) for k in
                           ("ticks", "world_qpos", "contact", "source_versions")}
    values["reference"]["version"] = cache.version
    teacher = state.teacher_reference
    values["teacher_reference"] = None if teacher is None else {
        k: pack_tensor(getattr(teacher, k), memo) for k in ("ticks", "world_qpos", "contact", "source_versions")}
    return values


def unpack_tensor(value, device, memo):
    key = (str(device), str(value.device), value.untyped_storage().data_ptr(), value.storage_offset(),
           tuple(value.shape), tuple(value.stride()), value.dtype)
    if key not in memo:
        memo[key] = value.to(device)
    return memo[key]


def unpack_state(payload, adapter, device, memo=None):
    memo = {} if memo is None else memo
    data = {k: unpack_tensor(v, device, memo) if isinstance(v, torch.Tensor) else v
            for k, v in payload.items() if k not in ("reference", "teacher_reference")}
    ref = {k: unpack_tensor(v, device, memo) if isinstance(v, torch.Tensor) else v for k, v in payload["reference"].items()}
    data["reference"] = ReferenceBuffer(**ref, kin=adapter.runtime.kin, to_sm=adapter.to_sm)
    teacher = payload.get("teacher_reference")
    data["teacher_reference"] = None if teacher is None else ReferenceBuffer(
        **{k: unpack_tensor(v, device, memo) for k, v in teacher.items()}, kin=adapter.runtime.kin, to_sm=adapter.to_sm)
    data["pending_request"] = unpack_value(data.get("pending_request"), adapter, device, memo)
    return ClosedLoopState(**data)


def pack_value(value, memo=None):
    """显式数据合同序列化，不把FK、求解器或线程pickle进训练检查点。"""
    memo = {} if memo is None else memo
    if isinstance(value, ClosedLoopState):
        return {"kind": "ClosedLoopState", "data": pack_state(value, memo)}
    if isinstance(value, torch.Tensor):
        return pack_tensor(value, memo)
    if is_dataclass(value):
        return {"kind": type(value).__name__, "data": {f.name: pack_value(getattr(value, f.name), memo) for f in fields(value)}}
    if isinstance(value, tuple):
        return {"kind": "tuple", "data": [pack_value(v, memo) for v in value]}
    if isinstance(value, dict):
        return {k: pack_value(v, memo) for k, v in value.items()}
    return value


def unpack_value(value, adapter, device, memo=None):
    from .contracts import PendingGeneration, GenerationRecord, PhysicalAnchor, RequestTicket, GeneratedPlan, ForecastContext
    from .returns import DecisionInterval
    memo = {} if memo is None else memo
    if isinstance(value, torch.Tensor):
        return unpack_tensor(value, device, memo)
    if not isinstance(value, dict):
        return value
    if "kind" not in value:
        return {k: unpack_value(v, adapter, device, memo) for k, v in value.items()}
    if value["kind"] == "ClosedLoopState":
        return unpack_state(value["data"], adapter, device, memo)
    if value["kind"] == "tuple":
        return tuple(unpack_value(v, adapter, device, memo) for v in value["data"])
    registry = {c.__name__: c for c in (PendingGeneration, GenerationRecord, PhysicalAnchor, RequestTicket,
                                       GeneratedPlan, ForecastContext, DecisionInterval)}
    return registry[value["kind"]](**{k: unpack_value(v, adapter, device, memo) for k, v in value["data"].items()})


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def restore_rng(value):
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"].cpu())
    if value["cuda"] is not None:
        torch.cuda.set_rng_state_all([v.cpu() for v in value["cuda"]])


def save_checkpoint(path, trainer, replay, *, behavior_config, runtime_state, provenance):
    trainer.imagination.adapter.bundle.assert_unchanged(trainer.frozen_identity)
    replay_payload = replay.state_dict()
    memo = {}
    replay_payload["items"] = [pack_value(z, memo) for z in replay_payload["items"]]
    atomic_save({"schema": SCHEMA, "actor": trainer.actor.state_dict(), "critic": trainer.critic.state_dict(),
                 "genmo_optimizer": trainer.actor_optimizer.state_dict(),
                 "upper_critic_optimizer": trainer.critic_optimizer.state_dict(),
                 "optimizer_layout": {"genmo": optimizer_layout(trainer.actor, trainer.actor_optimizer),
                                      "critic": optimizer_layout(trainer.critic, trainer.critic_optimizer)},
                 "update": trainer.update_index, "counters": trainer.counters(), "rng": capture_rng(), "replay": replay_payload,
                 "runtime_state": runtime_state, "behavior_config": behavior_config,
                 "frozen_sha256": trainer.imagination.adapter.bundle.metadata["sha256"],
                 "provenance": provenance, "physical_solver_state_saved": False}, path)


def load_checkpoint(path, trainer, replay, *, behavior_config, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("schema") != SCHEMA:
        raise ValueError("仅接受新BPTT完整状态，不接受旧策略/价值/优化器")
    saved_behavior = dict(ck["behavior_config"])
    # 本字段新增之前实现只有一个物理环境；显式默认1不是源码版本迁移门禁。
    saved_behavior.setdefault("physical_collector_count", 1)
    if saved_behavior != behavior_config:
        raise ValueError("实际采样、奖励、时序或优化合同改变，不能伪装等价续训")
    if ck["frozen_sha256"] != trainer.imagination.adapter.bundle.metadata["sha256"]:
        raise ValueError("续训必须使用同一固定pi/wm资产")
    expected_layout = {"genmo": optimizer_layout(trainer.actor, trainer.actor_optimizer),
                       "critic": optimizer_layout(trainer.critic, trainer.critic_optimizer)}
    if ck.get("optimizer_layout") != expected_layout:
        raise ValueError("优化器参数名称/分组改变；不能仅按位置加载动量")
    trainer.actor.load_state_dict(ck["actor"], strict=True)
    trainer.critic.load_state_dict(ck["critic"], strict=True)
    trainer.actor_optimizer.load_state_dict(ck["genmo_optimizer"])
    trainer.critic_optimizer.load_state_dict(ck["upper_critic_optimizer"])
    trainer.restore_counters(ck["counters"])
    payload = dict(ck["replay"])
    memo = {}
    payload["items"] = [unpack_value(z, trainer.imagination.adapter, device, memo) for z in payload["items"]]
    replay.load_state_dict(payload)
    restore_rng(ck["rng"])
    return {**ck["runtime_state"], "requires_new_physical_episode": True}
