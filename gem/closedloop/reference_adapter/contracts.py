"""Adapter 的具名输入、姿态、计划和增强状态合同。

全部时间使用绝对 600Hz 整数 tick；控制一步为12tick，音乐一帧为20tick。物理状态
仅供 WM/Critic 和可观测字段提取；InputSpec 显式排除 xi、世界根位置/速度和未来真值。
不可变 dataclass 用 replace 传递计算图，只有真实采样起点和回放入库允许 detach。
"""

from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any

import torch

CLOCK_HZ, CONTROL_TICKS, MUSIC_TICKS = 600, 12, 20
DT, REFERENCE_STEPS, RESIDUAL_DIM = .02, 10, 27


@dataclass(frozen=True)
class InputField:
    name: str
    shape: tuple[int, ...]
    units: str
    normalization: str


@dataclass(frozen=True)
class InputSpec:
    joint_names: tuple[str, ...]
    fields: tuple[InputField, ...] = (
        InputField("control_history", (10, 75), "SM原obs75，最旧到最新", "固定Actor.obs_norm"),
        InputField("nominal_goal", (570,), "SM原Goal57x10", "固定Actor.goal_norm"),
        InputField("observable_error", (48,), "q rad / qd rad/s / IMU相对rot6d", "qd乘0.1"),
        InputField("previous_plan", (10, 27), "名义参考相对残差，RobotSpec序", "逐分量除固定上限"),
        InputField("previous_corrected_plan", (10, 27), "上次修正物理姿态相对当前名义计划", "逐分量除固定上限"),
        InputField("plan_times", (10,), "相对当前控制时刻秒数", "秒"),
        InputField("plan_valid", (10,), "有效mask", "0或1"),
        InputField("committed_anchors", (2, 27), "左支持/当前相对名义姿态", "逐分量除固定上限"),
        InputField("music", (16, 35), "当前及未来15个EDGE35原帧", "原训练特征，不在线拟合"),
        InputField("music_valid", (16,), "音频有效mask", "0或1"),
        InputField("music_times", (16,), "相对当前时刻秒数", "秒"),
        InputField("music_memory", (20, 4), "执行能量/执行活动/名义活动/mask", "固定s0"),
    )

    @property
    def dimension(self):
        import math
        return sum(math.prod(f.shape) for f in self.fields)

    def flatten(self, values):
        if set(values) != {f.name for f in self.fields}:
            raise ValueError("InputSpec字段缺失或出现未授权的额外输入")
        batch = len(values[self.fields[0].name])
        for f in self.fields:
            x = values[f.name]
            if x.shape != (batch, *f.shape) or not torch.isfinite(x).all():
                raise ValueError(f"输入字段{f.name}形状或有限性错误")
        return torch.cat([values[f.name].reshape(batch, -1) for f in self.fields], -1)


@dataclass(frozen=True)
class Pose:
    p: torch.Tensor
    R: torch.Tensor
    q: torch.Tensor

    def select(self, index):
        return Pose(*(getattr(self, name)[index] for name in ("p", "R", "q")))


@dataclass(frozen=True)
class ResidualPlan:
    ticks: torch.Tensor
    residual: torch.Tensor
    corrected_pose: Pose
    valid: torch.Tensor
    nominal_version: int


@dataclass(frozen=True)
class AdapterState:
    tick: int
    executed_state: torch.Tensor
    xi: torch.Tensor
    previous_action: torch.Tensor
    control_history: torch.Tensor
    nominal_reference: Any
    left_pose: Pose
    current_pose: Pose
    previous_plan: ResidualPlan | None
    heading: torch.Tensor
    audio: Any
    music_state: Any
    episode_id: str
    song_id: str
    split: str = "train"
    origin: str = "real"
    mode: str = "fixed"
    known_at_tick: int = 0
    plan_boundary_tick: int | None = None
    scheduling: Any = None

    def validate(self):
        b = len(self.executed_state)
        for name, shape in (("executed_state", (b, 55)), ("xi", (b, 25)),
                            ("previous_action", (b, 21)), ("control_history", (b, 10, 75))):
            x = getattr(self, name)
            if x.shape != shape or not torch.isfinite(x).all():
                raise ValueError(f"增强状态{name}合同错误")
        if self.tick % CONTROL_TICKS or self.known_at_tick > self.tick:
            raise ValueError("时钟错误或使用尚未可见的名义计划")
        if self.origin not in ("real", "imagined") or self.mode not in ("fixed", "rolling"):
            raise ValueError("未知状态来源或参考模式")
        if self.split not in ("train", "validation"):
            raise ValueError("必须按歌曲指定train/validation划分")

    def detached(self):
        return map_tensors(self, lambda x: x.detach())


def map_tensors(value, fn):
    """保留名义/音频共享对象的只读语义；不复制FK或隐式求解器状态。"""
    from gem.closedloop.mb_control.reference_buffer import ReferenceBuffer
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, ReferenceBuffer):
        return replace(value, ticks=fn(value.ticks), world_qpos=fn(value.world_qpos),
                       contact=fn(value.contact), source_versions=fn(value.source_versions),
                       kin=None, prepared_reference=None, prepared_source=None, prepared_start=None)
    if is_dataclass(value):
        return replace(value, **{f.name: map_tensors(getattr(value, f.name), fn) for f in fields(value)})
    if isinstance(value, dict):
        return {k: map_tensors(v, fn) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(map_tensors(v, fn) for v in value)
    return value
