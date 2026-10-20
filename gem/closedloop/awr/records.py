"""AWR独立动作、反馈与完整Episode契约。

一条记录绑定生成前条件、原始canonical目标、真实提交证据及50Hz奖励区间。
完整Episode使用backend_session_id/env_id/episode_id联合身份，内部只允许一个
行为版本。被拒绝参考仍保留物理反馈与事件，却不成为Actor监督。生成Contact仅
是诊断输出；本契约没有旧概率、Gaussian链、旧均值/std或任何PPO必填字段。
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
import math
from typing import Any

import torch

from gem.closedloop.contracts import validate_stage1_condition_batch
from gem.closedloop.common.tracking_context import validate_tracking_context
from .io import cpu_copy, digest


@dataclass
class SamplingResult:
    conditions: dict
    qpos30: torch.Tensor
    target_valid: torch.Tensor
    contact: torch.Tensor
    root_anchor: Any
    sampler: dict
    seed: int


@dataclass
class PhysicalDecisionRecord:
    identity: dict
    control_tick_begin: int
    control_tick_end: int
    executed_control_steps: int
    rewards: torch.Tensor
    terminated: bool
    truncated: bool
    reason: str | None
    execution: dict = field(default_factory=dict)


@dataclass
class AWRTransition(PhysicalDecisionRecord):
    context: dict = field(default_factory=dict)
    next_context: dict | None = None
    target_qpos30: torch.Tensor | None = None
    target_valid: torch.Tensor | None = None
    free_mask: torch.Tensor | None = None
    contact: torch.Tensor | None = None
    actor_valid: bool = False
    metadata: dict = field(default_factory=dict)
    transition_valid: bool = True

    @property
    def executed_physics_steps(self):
        return self.executed_control_steps * 4

    @property
    def key(self):
        return tuple(str(self.identity[k]) for k in ("backend_session_id", "env_id", "episode_id"))

    @property
    def sample_id(self):
        return digest({k: self.identity[k] for k in
                       ("backend_session_id", "env_id", "episode_id", "decision_id")})

    def validate(self):
        for key in ("backend_session_id", "env_id", "episode_id", "decision_id",
                    "plan_id", "policy_version", "actor_fingerprint"):
            if key not in self.identity:
                raise ValueError(f"Missing AWR identity: {key}")
        for key in ("control_tick_begin", "control_tick_end", "executed_control_steps"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 0:
                raise ValueError(f"Invalid physical integer: {key}")
        if self.control_tick_end - self.control_tick_begin != 12 * self.executed_control_steps:
            raise ValueError("AWR control duration differs from the 600Hz clock")
        if self.rewards.shape != (self.executed_control_steps,) or not torch.isfinite(self.rewards).all():
            raise ValueError("AWR requires finite actual 50Hz reward samples")
        if not self.transition_valid:
            raise ValueError("Invalid physical data cannot enter replay")
        validate_stage1_condition_batch(self.context, history_steps=self.context["proprio_history"].shape[1])
        validate_tracking_context(self.metadata["critic_context"])
        if self.context["future_valid"].shape != (1, 60):
            raise ValueError("AWR v1 requires one 60-frame reference per record")
        if self.target_qpos30.shape != (60, 30) or self.target_valid.shape != (60, 30):
            raise ValueError("Invalid canonical AWR target shape")
        if self.target_valid.dtype != torch.bool or self.free_mask.dtype != torch.bool:
            raise ValueError("AWR validity must be boolean")
        known = self.context["known_qpos30_mask"][0]
        future = self.context["future_valid"][0, :, None]
        if (self.target_valid & ~future).any() or (known & ~self.target_valid).any():
            raise ValueError("AWR target validity contradicts condition masks")
        if not torch.equal(self.free_mask, self.target_valid & ~known):
            raise ValueError("AWR free coordinates do not match original conditions")
        if self.metadata.get("data_split") != "train":
            raise ValueError("Held-out data cannot enter training replay")
        if not math.isfinite(float(self.metadata.get("event_reward", 0.))):
            raise ValueError("Nonfinite event reward")
        if self.actor_valid:
            if not self.execution.get("committed") or not self.execution.get("observed_by_gmt"):
                raise ValueError("Unexecuted reference cannot receive an AWR action label")
            if not self.execution.get("target_matches_commit"):
                raise ValueError("Generated reference differs from submitted free coordinates")
            if not self.free_mask.any() or not torch.isfinite(self.target_qpos30).all():
                raise ValueError("AWR target has no finite free coordinates")
            if not torch.equal(self.target_qpos30[known], self.context["known_qpos30"][0][known]):
                raise ValueError("Known prefix changed in original generated target")
        return self

    def snapshot(self):
        return AWRTransition(**cpu_copy(self.to_dict())).validate()

    def to_dict(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass
class AWRCompleteEpisode:
    rows: list[AWRTransition]
    source: str
    group_id: str
    audio_sha256: str
    contract: dict
    task: dict
    returns: torch.Tensor | None = None

    @property
    def episode_id(self):
        return digest(list(self.rows[0].key))

    def validate(self):
        if not self.rows or not self.source or not self.group_id or not self.audio_sha256:
            raise ValueError("Complete AWR episode needs rows and music identities")
        seen, first = set(), self.rows[0]
        previous = first.control_tick_begin
        for index, row in enumerate(self.rows):
            row.validate()
            if row.key != first.key or row.identity["policy_version"] != first.identity["policy_version"]:
                raise ValueError("Episode crosses physical or behavior identities")
            if row.identity["actor_fingerprint"] != first.identity["actor_fingerprint"]:
                raise ValueError("Actor changed inside complete episode")
            if row.sample_id in seen or row.control_tick_begin != previous:
                raise ValueError("Duplicate decision or discontinuous episode")
            if row.truncated or row.metadata.get("physical_episode_administrative_truncation"):
                raise ValueError("Administrative/incomplete episode has no full MC label")
            if index < len(self.rows)-1 and row.terminated:
                raise ValueError("Data follows a terminal state")
            seen.add(row.sample_id)
            previous = row.control_tick_end
        if not self.rows[-1].terminated:
            raise ValueError("Episode has no true terminal")
        if self.returns is not None:
            if self.returns.shape != (len(self.rows),) or not torch.isfinite(self.returns).all():
                raise ValueError("Invalid complete MC return vector")
        return self

    def to_dict(self):
        return {**{f.name: getattr(self, f.name) for f in fields(self) if f.name != "rows"},
                "rows": [r.to_dict() for r in self.rows]}

    @classmethod
    def from_dict(cls, value):
        return cls(**dict(value, rows=[AWRTransition(**r) for r in value["rows"]])).validate()
