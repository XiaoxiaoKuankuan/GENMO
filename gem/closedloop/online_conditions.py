"""将 Isaac 闭环的真实历史、音乐和旧参考转换成现有 Stage1 十字段条件。

本模块仅处理在线推理条件，不读取示范动作或监督 target，也不创建第二套模型 batch。
所有外部时间使用 600 Hz 整数 tick：30 Hz 音乐/参考相邻点为 20 tick，50 Hz 实际
观测相邻点为 12 tick。历史严格来自已消费的实际控制状态，读取本模块不会推进历史。

前缀使用旧计划的世界 qpos28，一次编码得到窗口共同的 XY/yaw anchor 与 physical
qpos30；最后一个前缀点缺少下一点，因此 root XY 位移两坐标始终未知。网络归一化和
独立 proprio 物理尺度都交给现有 Actor。P=0 时必须显式提供旧参考 anchor，禁止用实际
机器人位置重锚掩盖跟踪漂移。输出时间使用 float64，避免长 episode 的秒数精度破坏
30/50 Hz 校验；运动/音乐保持 float32，mask 保持 bool。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from gem.closedloop.contracts import (
    MOTION_WINDOW_FRAMES,
    MUSIC_FEATURE_DIM,
    PROPRIO_DIM,
    STAGE1_TARGET_KEYS,
    validate_stage1_condition_batch,
)
from gem.robots.bumi.feature_codec import BumiMotionFeatureCodec

CLOCK_HZ = 600
SOURCE_TICKS = 20
CONTROL_TICKS = 12


def _integer(value: Any, name: str) -> int:
    """拒绝静默截断小数 tick、bool 和其他非整数元数据。"""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _tensor(value: Any, dtype: torch.dtype | None = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=dtype).clone()
    return torch.as_tensor(np.array(value, copy=True), dtype=dtype)


class OnlineConditionBuilder:
    """只读取快照，不修改后端缓存；构造 batch=1 的原始 physical 条件。"""

    def __init__(self, codec: BumiMotionFeatureCodec, history_steps: int = 50) -> None:
        if not isinstance(codec, BumiMotionFeatureCodec):
            raise TypeError("codec must be the existing BumiMotionFeatureCodec")
        self.codec = codec
        self.history_steps = _integer(history_steps, "history_steps")
        if self.history_steps <= 0:
            raise ValueError("history_steps must be positive")

    def build(
        self,
        snapshot: Mapping[str, Any],
        reservation: Mapping[str, Any],
        music_features: Any,
        music_start_tick: int = 600,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """将一次请求时刻的快照与前缀预留转换为条件及执行元数据。

        reservation 的 source_qpos/source_ticks 只包含 P 个已知世界姿态；P=0 时提供
        reference_anchor_qpos[28]。允许元数据放在 reservation.meta 或顶层，顶层优先；
        大数组不会复制进元数据，监督字段在任一条件输入出现时直接拒绝。
        """
        for name, data in (("snapshot", snapshot), ("reservation", reservation)):
            forbidden = set(data) & set(STAGE1_TARGET_KEYS)
            if forbidden:
                raise ValueError(f"{name} contains supervision fields: {sorted(forbidden)}")
        tick = _integer(snapshot["tick"], "snapshot.tick")
        meta = dict(reservation.get("meta", {}))
        arrays = {"meta", "source_qpos", "source_ticks", "reference_anchor_qpos"}
        meta.update({key: value for key, value in reservation.items() if key not in arrays})
        decision_tick = _integer(meta.get("decision_tick", tick), "decision_tick")
        if decision_tick != tick:
            raise ValueError("snapshot tick must equal the frozen request decision_tick")
        if set(meta) & set(STAGE1_TARGET_KEYS):
            raise ValueError("reservation metadata contains supervision fields")
        for key in ("env_id", "episode_id"):
            if key in meta and key in snapshot and meta[key] != snapshot[key]:
                raise ValueError(f"snapshot/reservation {key} mismatch")
            if key in snapshot:
                meta.setdefault(key, snapshot[key])
        for key in ("seed", "decision_id"):
            if key in snapshot:
                meta.setdefault(key, snapshot[key])

        prefix = _integer(meta["prefix_frames"], "prefix_frames")
        if not 0 <= prefix < MOTION_WINDOW_FRAMES:
            raise ValueError("prefix_frames must be in [0,119] and leave a generation horizon")
        source = _tensor(reservation["source_qpos"], torch.float32)
        source_ticks = _tensor(reservation["source_ticks"])
        if source.shape != (prefix, 28) or source_ticks.shape != (prefix,):
            raise ValueError("reserved source must have shapes [P,28] and [P]")
        if source_ticks.dtype not in (torch.int32, torch.int64):
            raise TypeError("source_ticks must be integer ticks")
        expected_ticks = decision_tick + torch.arange(prefix, dtype=torch.int64) * SOURCE_TICKS
        if not torch.equal(source_ticks.to(torch.int64), expected_ticks):
            raise ValueError("reserved source ticks must start at decision_tick and advance by 20")
        if not torch.isfinite(source).all():
            raise ValueError("reserved source contains NaN/Inf")

        known = torch.zeros((1, MOTION_WINDOW_FRAMES, 30), dtype=torch.float32)
        known_mask = torch.zeros_like(known, dtype=torch.bool)
        if prefix:
            encoded = self.codec.encode(source)
            anchor = encoded.anchor
            known[0, :prefix] = encoded.physical_features
            known_mask[0, :prefix] = True
            known_mask[0, prefix - 1, :2] = False
        else:
            if "reference_anchor_qpos" not in reservation:
                raise ValueError("P=0 requires reference_anchor_qpos from the old reference")
            anchor_qpos = _tensor(reservation["reference_anchor_qpos"], torch.float32)
            if anchor_qpos.shape != (28,):
                raise ValueError("reference_anchor_qpos must have shape [28]")
            anchor = self.codec.build_canonical_anchor(anchor_qpos[None])
        known = torch.where(known_mask, known, 0.0)

        history, history_valid, history_ticks = self._history(snapshot, decision_tick)
        future_ticks = decision_tick + torch.arange(MOTION_WINDOW_FRAMES, dtype=torch.int64) * SOURCE_TICKS
        music_start_tick = _integer(music_start_tick, "music_start_tick")
        if (decision_tick - music_start_tick) % SOURCE_TICKS:
            raise ValueError("decision and music start must share the 30 Hz source grid")
        music = _tensor(music_features, torch.float32)
        if music.ndim != 2 or music.shape[1] != MUSIC_FEATURE_DIM or not torch.isfinite(music).all():
            raise ValueError("music_features must be finite float [N,35] at 30 Hz")
        indices = (future_ticks - music_start_tick) // SOURCE_TICKS
        music_valid = (indices >= 0) & (indices < len(music))
        window = torch.zeros((MOTION_WINDOW_FRAMES, MUSIC_FEATURE_DIM), dtype=torch.float32)
        window[music_valid] = music[indices[music_valid]]
        conditions = {
            "music_features": window[None],
            "music_valid": music_valid[None],
            "proprio_history": history[None],
            "proprio_history_valid": history_valid[None],
            "proprio_history_times": history_ticks[None].to(torch.float64) / CLOCK_HZ,
            "known_qpos30": known,
            "known_qpos30_mask": known_mask,
            # 推理计划在音乐结尾之外仍有状态点，只有音乐条件被显式标为不可见。
            "future_valid": torch.ones((1, MOTION_WINDOW_FRAMES), dtype=torch.bool),
            "future_times": future_ticks[None].to(torch.float64) / CLOCK_HZ,
            "decision_time": torch.tensor([decision_tick / CLOCK_HZ], dtype=torch.float64),
        }
        validate_stage1_condition_batch(conditions, history_steps=self.history_steps)
        position = anchor.position_w.reshape(3).detach().cpu()
        quaternion = anchor.heading_quat_wxyz.reshape(4).detach().cpu()
        yaw = float(anchor.yaw.reshape(()).detach().cpu())
        meta.update({
            "decision_tick": decision_tick,
            "prefix_frames": prefix,
            "world_anchor": [*position.tolist(), yaw],
            "anchor_pos": position.numpy().copy(),
            "anchor_quat": quaternion.numpy().copy(),
            "source_ticks": future_ticks.numpy().copy(),
            "music_start_tick": music_start_tick,
        })
        return conditions, meta

    def _history(self, snapshot: Mapping[str, Any], tick: int):
        """将已有历史放入固定因果网格；无效槽归零，不插值或伪造有效历史。"""
        values = _tensor(snapshot["history_values"], torch.float32)
        valid = _tensor(snapshot["history_valid"])
        ticks = _tensor(snapshot["history_ticks"])
        count = len(values)
        if values.shape != (count, PROPRIO_DIM) or valid.shape != (count,) or ticks.shape != (count,):
            raise ValueError("history must have shapes [N,48], [N], [N]")
        if valid.dtype != torch.bool or ticks.dtype not in (torch.int32, torch.int64):
            raise TypeError("history_valid must be bool and history_ticks must be integers")
        if count > self.history_steps:
            raise ValueError("snapshot history exceeds configured H")
        if count > 1 and not bool(((ticks[1:] - ticks[:-1]) == CONTROL_TICKS).all()):
            raise ValueError("history ticks must be unique, ordered, and advance by 12")
        if bool((ticks[valid] > tick).any()):
            raise ValueError("valid history cannot be later than the request")
        if not torch.isfinite(values[valid]).all():
            raise ValueError("valid history contains NaN/Inf")
        target_ticks = tick - torch.arange(self.history_steps - 1, -1, -1, dtype=torch.int64) * CONTROL_TICKS
        result = torch.zeros((self.history_steps, PROPRIO_DIM), dtype=torch.float32)
        result_valid = torch.zeros(self.history_steps, dtype=torch.bool)
        if count:
            offsets = ticks.to(torch.int64) - target_ticks[0]
            if bool((offsets % CONTROL_TICKS != 0).any()):
                raise ValueError("history ticks and decision tick must share the 50 Hz control grid")
            slots = offsets // CONTROL_TICKS
            inside = (slots >= 0) & (slots < self.history_steps)
            if bool((valid & ~inside).any()):
                raise ValueError("valid history falls outside the configured causal window")
            keep = inside & valid
            result[slots[keep]] = values[keep]
            result_valid[slots[keep]] = True
        return result, result_valid, target_ticks
