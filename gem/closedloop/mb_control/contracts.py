"""600Hz时序与完整闭环状态的显式数据合同。

真实快照含两套历史、有效参考与halo、音乐、原始物理xi和可见延迟历史。
ForecastContext 单独保存预测误差，绝不写入实测历史。只有真实采集边界允许detach；
想象中使用不可变replace传递张量图。首版批次共享时钟，pending决策不进入Critic。
"""

from dataclasses import dataclass, replace
from typing import Any

import torch

CLOCK_HZ, SOURCE_TICKS, CONTROL_TICKS, PHYSICS_TICKS, REQUEST_GRID = 600, 20, 12, 3, 60
MOTION_FRAMES, SM_HISTORY, GENMO_HISTORY = 60, 10, 50


@dataclass(frozen=True)
class ForecastContext:
    features: torch.Tensor  # [B,58]，最后一维为提前秒数。
    valid: torch.Tensor
    request_tick: int
    forecast_tick: int

    @classmethod
    def invalid(cls, like, tick):
        return cls(like.new_zeros((len(like), 58)),
                   torch.zeros(len(like), device=like.device, dtype=torch.bool), tick, tick)


@dataclass(frozen=True)
class ClosedLoopState:
    tick: int
    state: torch.Tensor
    xi: torch.Tensor
    previous_action: torch.Tensor
    sm_history: torch.Tensor
    genmo_history: torch.Tensor
    genmo_valid: torch.Tensor
    reference: Any
    music: torch.Tensor
    music_start_tick: int
    music_end_tick: int
    latency_history: tuple[int, ...] = ()
    protected_until: int = 0
    decision_id: int = 0
    consecutive_timeouts: int = 0
    pending: bool = False
    history_origin: str = "real"

    def validate(self, *, decision=False):
        b = len(self.state)
        for name, shape in (("state", (b, 55)), ("xi", (b, 25)),
                            ("previous_action", (b, 21)), ("sm_history", (b, 10, 75)),
                            ("genmo_history", (b, 50, 48)), ("genmo_valid", (b, 50))):
            value = getattr(self, name)
            if value.shape != shape:
                raise ValueError(f"{name} 需要 {shape}，实际 {tuple(value.shape)}")
        if self.tick % CONTROL_TICKS or self.history_origin not in ("real", "imagined"):
            raise ValueError("时钟或历史来源错误")
        if decision and (self.pending or self.tick % REQUEST_GRID):
            raise ValueError("价值/生成状态必须位于无pending公共网格边界")
        if self.music.ndim != 3 or self.music.shape[0] != b or self.music.shape[-1] != 35:
            raise ValueError("音乐必须为[B,N,35]")
        if any(t < 0 for t in self.latency_history):
            raise ValueError("历史延迟不能为负")

    def detached(self):
        names = ("state", "xi", "previous_action", "sm_history", "genmo_history", "genmo_valid")
        return replace(self, **{k: getattr(self, k).detach().clone() for k in names},
                       music=self.music.detach(), reference=self.reference.detached())


@dataclass(frozen=True)
class RequestTicket:
    decision_id: int
    request_tick: int
    predicted_ready_tick: int
    first_mutable_source_tick: int
    first_influence_tick: int
    prefix_frames: int
    base_version: int


@dataclass(frozen=True)
class GeneratedPlan:
    ticket: RequestTicket
    world_qpos: torch.Tensor  # [B,60,28]，原GENMO关节序。
    contact: torch.Tensor


class ReferenceExhausted(RuntimeError):
    """明确终止：没有足够真实支持点，不外推或重复假参考。"""
