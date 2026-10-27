"""原数据时间轴、真实sidecar与局部执行窗口的独立合同。

Stage1帧索引显式乘20映射到600Hz；50Hz控制步乘12，两者只在3源帧/5控制步公共网格
直接配对。缓存记录来源和split，任何缺失、错位、旧计划越权或不完整物理窗口都拒绝执行
梯度，但调用者仍可保留监督。这里不把GT动作当成state55，也不把多个局部窗称为完整BPTT。
"""

from dataclasses import dataclass
import hashlib
import json

import torch

SIDECAR_SCHEMA = "genmo.joint_physics.sidecar.v2.offline_requests"
RAW_SIDECAR_SCHEMA = "genmo.joint_physics.sidecar.v1"


class PhysicsRejected(ValueError):
    def __init__(self, reason, details=None):
        super().__init__(reason)
        self.reason = reason
        self.details = details or {}


def source_key(meta):
    return json.dumps([meta["dataset_id"], meta["split"], meta["sample_id"]], ensure_ascii=False, separators=(",", ":"))


def frame_tick(frame):
    if int(frame) != frame or frame < 0:
        raise ValueError("Stage1全局源帧必须为非负整数")
    return int(frame) * 20


def digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def execution_contract(cfg):
    return {"prediction_mode": cfg["physics"]["prediction_mode"], "loss": cfg["execution_loss"],
            "target": "same_source_data_fixed_episode_world", "start": "real_before_first_candidate_influence",
            "reference": "original_codec_reference_buffer_velocity_halo_v1"}


@dataclass(frozen=True)
class Window:
    request: object
    start: object
    ticket: object
    anchor: torch.Tensor
    target: object
    horizon: int
    source_id: str
    provenance: dict

    def validate(self):
        if self.start.history_origin != "real" or self.start.tick != self.ticket.first_influence_tick:
            raise PhysicsRejected("not_real_first_influence_start")
        if self.request.tick != self.ticket.request_tick or self.request.tick % 60:
            raise PhysicsRejected("time_grid_mismatch")
        if self.provenance["old_plan_committed_tick"] > self.request.tick:
            raise PhysicsRejected("future_plan_leak")
        if self.start.tick + self.horizon * 12 > self.provenance["last_real_tick"]:
            raise PhysicsRejected("incomplete_real_window")
        if self.provenance["split"] not in ("train", "val", "test"):
            raise PhysicsRejected("unknown_split")
