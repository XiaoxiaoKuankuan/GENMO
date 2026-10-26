"""Adapter完整状态的原子、不覆盖checkpoint与严格语义恢复。

保存Adapter/Critic/独立WM、三优化器、固定normalizer、RNG、真实回放、模型版本及
采集器残差/音乐/时钟。源码SHA仅作溯源；恢复核验张量与行为合同，不要求源码相等。
求解器隐藏状态不在checkpoint中，恢复保留学习状态和历史证据，但真实采集从新episode
开始；不把保存的state55冒充完整物理恢复。编号文件使用独占链接，禁止覆盖原资产。
"""

import random

import numpy as np
import torch

from supermimic.bridge.bundle import atomic_save
from .contracts import map_tensors

SCHEMA = "genmo.supermimic.reference_adapter.full.v2"


def shared_cpu_state(value):
    """同一只读名义/音频张量只复制一次，避免长曲回放的二次方内存膨胀。"""
    cpu_cache = {}
    def cpu(tensor):
        key = (tensor.untyped_storage().data_ptr(), tensor.storage_offset(), tuple(tensor.shape), tensor.stride(), tensor.dtype, tensor.device)
        if key not in cpu_cache:
            cpu_cache[key] = tensor.detach().cpu()
        return cpu_cache[key]
    return map_tensors(value, cpu)


def save_checkpoint(path, trainer, *, collector=None, behavior=None, provenance=None, cursor=None):
    if trainer.dynamics._graph_active:
        raise RuntimeError("只能在WM版本屏障之外保存完整checkpoint")
    payload = dict(schema=SCHEMA, adapter=trainer.adapter.state_dict(), critic=trainer.critic.state_dict(),
                   dynamics=trainer.dynamics.state_dict(), adapter_optimizer=trainer.adapter_optimizer.state_dict(),
                   critic_optimizer=trainer.critic_optimizer.state_dict(), wm_optimizer=trainer.wm_optimizer.state_dict(),
                   frozen_bundle=trainer.engine.step.bundle.state_dict(), input_spec=trainer.adapter.input_spec,
                   trainer=trainer.state_dict(), replay=trainer.replay.state_dict(),
                   collector_state=collector.state if collector else None,
                   collector_steps=collector.steps if collector else 0,
                   collector_resets=collector.reset_count if collector else 0,
                   music_s0=trainer.engine.step.rewards.music.s0,
                   behavior=behavior, provenance=provenance, cursor=cursor,
                   rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                            cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
                   physics_resume="new_episode_required_solver_hidden_state_not_saved")
    atomic_save(shared_cpu_state(payload), path)


def load_checkpoint(path, trainer, *, behavior=None, device="cpu"):
    value = torch.load(path, map_location=device, weights_only=False)
    if value.get("schema") != SCHEMA or value["input_spec"] != trainer.adapter.input_spec:
        raise ValueError("不是兼容的十帧Adapter完整checkpoint")
    if behavior is not None and value["behavior"] != behavior:
        raise ValueError("实际奖励/采样/音乐/物理行为合同不匹配；源码版本不参与此检查")
    s0 = trainer.engine.step.rewards.music.s0
    if not torch.equal(s0, value["music_s0"].to(s0)):
        raise ValueError("固定音乐基线s0不匹配，不能以新归一化冒充完整恢复")
    current = trainer.engine.step.bundle.state_dict()
    if current.keys() != value["frozen_bundle"].keys() or any(not torch.equal(v, value["frozen_bundle"][k].to(v)) for k, v in current.items()):
        raise ValueError("冻结Actor/原WM/normalizer不匹配，不能伪装同一执行合同")
    for name in ("adapter", "critic", "dynamics", "adapter_optimizer", "critic_optimizer", "wm_optimizer"):
        getattr(trainer, name).load_state_dict(value[name])
    trainer.replay.load_state_dict(value["replay"])
    trainer.load_state_dict(value["trainer"])
    trainer.engine.step.rewards.music.s0 = value["music_s0"].detach().clone()
    random.setstate(value["rng"]["python"])
    np.random.set_state(value["rng"]["numpy"])
    torch.set_rng_state(value["rng"]["torch"].cpu())
    if value["rng"]["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in value["rng"]["cuda"]])
    return {"collector_state_for_audit": value["collector_state"], "collector_steps": value["collector_steps"],
            "collector_resets": value["collector_resets"],
            "cursor": value["cursor"], "physics_resume": value["physics_resume"]}
