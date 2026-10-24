"""新Stage2训练状态的原子保存与恢复，只持久化两个优化器。

保存GENMO/未来编码、独立Critic、优化器、CPU/CUDA/NumPy/Python随机状态、配对数据
游标、延迟采样与近期真实回放。参考仅保存紧凑张量，加载时重新绑定同一权威FK。
源码版本只记录；冻结资产内容、采样/物理合同需要匹配。没有保存物理求解器，因此
恢复后采集必须显式新episode，不把qpos/回放恢复宣传为物理精确续跑。
"""

from dataclasses import fields
import random

import numpy as np
import torch

from supermimic.bridge.bundle import atomic_save
from .contracts import ClosedLoopState
from .reference_buffer import ReferenceBuffer

SCHEMA = "genmo.supermimic.bptt.training.v2"


def optimizer_layout(module, optimizer):
    names = {id(p): name for name, p in module.named_parameters()}
    return [[names[id(p)] for p in group["params"]] for group in optimizer.param_groups]


def pack_state(state):
    values = {f.name: getattr(state, f.name) for f in fields(state) if f.name != "reference"}
    values = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in values.items()}
    cache = state.reference
    values["reference"] = {k: getattr(cache, k).detach().cpu() for k in
                           ("ticks", "world_qpos", "contact", "source_versions")}
    values["reference"]["version"] = cache.version
    return values


def unpack_state(payload, adapter, device):
    data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in payload.items() if k != "reference"}
    ref = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in payload["reference"].items()}
    data["reference"] = ReferenceBuffer(**ref, kin=adapter.runtime.kin, to_sm=adapter.to_sm)
    return ClosedLoopState(**data)


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
    replay_payload["items"] = [pack_state(z) for z in replay_payload["items"]]
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
    if ck["behavior_config"] != behavior_config:
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
    payload["items"] = [unpack_state(z, trainer.imagination.adapter, device) for z in payload["items"]]
    replay.load_state_dict(payload)
    restore_rng(ck["rng"])
    return {**ck["runtime_state"], "requires_new_physical_episode": True}
