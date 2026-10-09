"""第九步上层执行转移、独立 CPU 快照和可确认的执行日志。

本模块只保存数据，不调用 Actor、GMT 或物理推进。每条转移同时保留决策条件、完整
归一化去噪链、逐去噪步旧联合概率、真实控制步奖励及两类结束标记。写入 Buffer 时
递归 detach、搬到 CPU 并复制，避免下一次历史更新、参考窗口回收或 autograd 图改变
已经采集的数据；采样链必须是 FP32，概率保留 FP64，不能用量化节省存储。

StepJournal 使用 SQLite 的事务和 FULL 同步保存完整执行回复。session/episode/请求
身份是唯一键：完全相同的重发只返回 False，身份相同而内容不同则拒绝。调用方只有
在事务成功后才可以确认后端回复；SQLite 的崩溃恢复不需要把全部轨迹载入内存。
无效回复中的非有限值以显式标记持久化，不能悄悄补零或把故障记录冒充正常训练样本。
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Mapping

import numpy as np
import torch

from gem.closedloop.contracts import validate_stage1_condition_batch


def cpu_snapshot(value: Any) -> Any:
    """递归复制 Tensor/NumPy/容器，所有 Tensor 与计算图和源缓冲断开。"""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Mapping):
        return {key: cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(item) for item in value)
    if isinstance(value, list):
        return [cpu_snapshot(item) for item in value]
    return copy.deepcopy(value)


@dataclass
class UpperTransition:
    """一次真实决策至下一真实决策的转移；metadata 保存采样核与事件原始证据。"""
    identity: dict
    context: dict
    next_context: dict | None
    chain: torch.Tensor | None
    old_log_prob: torch.Tensor | None
    free_mask: torch.Tensor | None
    rewards: torch.Tensor
    old_value: float
    next_value: float
    control_tick_begin: int
    control_tick_end: int
    executed_control_steps: int
    executed_physics_steps: int
    terminated: bool = False
    truncated: bool = False
    transition_valid: bool = True
    reason: str | None = None
    metadata: dict = field(default_factory=dict)

    def snapshot(self) -> "UpperTransition":
        result = UpperTransition(**{key: cpu_snapshot(value) for key, value in vars(self).items()})
        result.rewards = torch.as_tensor(result.rewards, dtype=torch.float64)
        if result.chain is not None:
            result.chain = torch.as_tensor(result.chain)
            if result.chain.ndim == 4 and result.chain.shape[0] == 1:
                result.chain = result.chain[0]
        for key in ("old_log_prob", "free_mask"):
            value = getattr(result, key)
            if value is not None:
                value = torch.as_tensor(value)
                expected_ndim = 1 if key == "old_log_prob" else 2
                if value.ndim == expected_ndim + 1 and value.shape[0] == 1:
                    value = value[0]
                setattr(result, key, value)
        result.validate()
        return result

    def validate(self) -> None:
        for name in ("control_tick_begin", "control_tick_end", "executed_control_steps", "executed_physics_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.rewards.ndim != 1 or len(self.rewards) != self.executed_control_steps:
            raise ValueError("rewards must contain exactly the actually completed control steps")
        if not self.transition_valid:
            return  # 故障证据允许缺链或部分物理子步，禁止进入正常训练路径。
        if self.control_tick_end - self.control_tick_begin != 12 * self.executed_control_steps:
            raise ValueError("valid transition tick range does not match actual control steps")
        if self.executed_physics_steps != 4 * self.executed_control_steps:
            raise ValueError("partial physics advancement cannot be a valid training transition")
        validate_stage1_condition_batch(self.context, history_steps=50)
        if self.context["decision_time"].shape != (1,):
            raise ValueError("Stage9 buffer requires single-environment conditions")
        if self.next_context is not None:
            validate_stage1_condition_batch(self.next_context, history_steps=50)
        if self.chain is None or self.chain.dtype != torch.float32 or self.chain.ndim != 3:
            raise ValueError("valid transition requires a FP32 [K+1,120,30] chain")
        if self.chain.shape[1:] != (120, 30) or self.chain.shape[0] < 2:
            raise ValueError("invalid denoising chain shape")
        if self.old_log_prob is None or self.old_log_prob.shape != (self.chain.shape[0] - 1,):
            raise ValueError("old_log_prob must be [K]")
        if self.old_log_prob.dtype != torch.float64:
            raise ValueError("old joint log probabilities must preserve FP64 precision")
        if self.free_mask is None or self.free_mask.dtype != torch.bool or self.free_mask.shape != (120, 30):
            raise ValueError("free_mask must be bool [120,30]")
        expected_free = self.context["future_valid"][0, :, None] & ~self.context["known_qpos30_mask"][0]
        if not torch.equal(self.free_mask, expected_free):
            raise ValueError("stored free_mask differs from the sampled condition mask")
        for name in ("chain", "old_log_prob", "rewards"):
            if not torch.isfinite(getattr(self, name)).all():
                raise ValueError(f"nonfinite {name} in valid transition")
        if not all(math.isfinite(float(value)) for value in (self.old_value, self.next_value)):
            raise ValueError("nonfinite value prediction")
        if not self.terminated and self.next_context is None:
            raise ValueError("nonterminal valid transition requires trustworthy next conditions")
        required = ("run_id", "backend_session_id", "episode_id", "decision_id", "policy_version")
        if any(key not in self.identity for key in required):
            raise ValueError(f"transition identity requires {required}")
        if self.executed_control_steps == 0 and not (self.terminated or self.truncated):
            raise ValueError("nonterminal zero-step transitions would permit an infinite loop")


class RolloutBuffer:
    """单一策略版本的有界在策略 Buffer；无效证据不进入 valid_transitions。"""
    def __init__(self, capacity: int = 64):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self._transitions: list[UpperTransition] = []
        self._policy_version = None

    def append(self, transition: UpperTransition) -> None:
        if len(self._transitions) >= self.capacity:
            raise OverflowError("rollout buffer is full")
        snapshot = transition.snapshot()
        version = snapshot.identity.get("policy_version")
        if self._transitions and version != self._policy_version:
            raise ValueError("cannot mix policy versions in an on-policy rollout")
        self._policy_version = version
        self._transitions.append(snapshot)

    @property
    def transitions(self) -> tuple[UpperTransition, ...]:
        return tuple(self._transitions)

    @property
    def valid_transitions(self) -> tuple[UpperTransition, ...]:
        return tuple(item for item in self._transitions if item.transition_valid)

    def __len__(self):
        return len(self._transitions)

    def clear(self) -> None:
        self._transitions.clear()
        self._policy_version = None


def _journal_value(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return {"__array__": value.dtype.str, "shape": list(value.shape), "values": _journal_value(value.tolist())}
    if isinstance(value, np.generic):
        return _journal_value(value.item())
    if isinstance(value, Mapping):
        return {str(key): _journal_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_journal_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"__nonfinite__": repr(value)}
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass(frozen=True)
class EncodedJournalReply:
    """一次规范化后的不可变执行回复；复用同一UTF-8字节容量、SHA与SQL内容。"""
    identity: str
    payload: bytes
    sha256: str
    format: str = 'json.v1'


class StepJournal:
    """同步落盘的幂等执行回复日志；成功返回后调用方才可发送后端 ACK。"""
    def __init__(self, path: str | Path, *, format='json.v1'):
        from .journal_codec import FORMAT
        if format not in ('json.v1', FORMAT):
            raise ValueError('Unknown journal format')
        self.format = format
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("CREATE TABLE IF NOT EXISTS replies (identity TEXT PRIMARY KEY, sha256 TEXT NOT NULL, payload TEXT NOT NULL)")
        self.connection.execute('CREATE TABLE IF NOT EXISTS journal_format (format TEXT NOT NULL)')
        saved = self.connection.execute('SELECT format FROM journal_format').fetchall()
        if saved and saved != [(format,)]:
            self.connection.close()
            raise ValueError('Journal format cannot change on reopening')
        if not saved:
            if format != 'json.v1' and self.connection.execute('SELECT COUNT(*) FROM replies').fetchone()[0]:
                self.connection.close()
                raise ValueError('Cannot append binary replies to legacy journal')
            self.connection.execute('INSERT INTO journal_format VALUES (?)', (format,))
        self.connection.commit()

    @staticmethod
    def encode_result(result: Mapping[str, Any], *, format='json.v1') -> EncodedJournalReply:
        body = result.get("result", result)
        if not isinstance(body, Mapping):
            body = result
        session = result.get("backend_session_id", body.get("backend_session_id"))
        sequence = result.get("mutation_seq", body.get("mutation_seq", body.get("advance_id")))
        if session is None or sequence is None:
            raise ValueError("durable execution reply requires session and mutation/request identity")
        # ack.v2 的序号在整个 worker session 单调增长，不能通过改变 episode 绕过冲突检测。
        identity = json.dumps([session, sequence], separators=(",", ":"))
        from .journal_codec import FORMAT, encode_binary
        if format == FORMAT:
            payload = encode_binary(result)
        elif format == 'json.v1':
            payload = json.dumps(_journal_value(result), ensure_ascii=False, sort_keys=True,
                                 separators=(',', ':'), allow_nan=False).encode('utf-8')
        else:
            raise ValueError('Unknown journal format')
        return EncodedJournalReply(identity, payload, hashlib.sha256(payload).hexdigest(), format)

    def append_result(self, result: Mapping[str, Any]) -> bool:
        return self.append_encoded(self.encode_result(result, format=self.format))

    def append_encoded(self, encoded: EncodedJournalReply) -> bool:
        if not isinstance(encoded, EncodedJournalReply):
            raise TypeError('Journal requires a canonical immutable encoded reply')
        if encoded.format != self.format:
            raise ValueError('Encoded journal format differs from storage contract')
        identity, digest = encoded.identity, encoded.sha256
        with self.connection:
            existing = self.connection.execute("SELECT sha256 FROM replies WHERE identity=?", (identity,)).fetchone()
            if existing:
                if existing[0] != digest:
                    raise ValueError("same execution identity returned different payload")
                return False
            self.connection.execute("INSERT INTO replies VALUES (?,?,?)", (identity, digest, encoded.payload.decode('utf-8') if self.format == 'json.v1' else sqlite3.Binary(encoded.payload)))
        return True

    def __len__(self):
        return self.connection.execute("SELECT COUNT(*) FROM replies").fetchone()[0]

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
