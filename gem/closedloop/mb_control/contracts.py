"""600Hz时序与完整闭环状态的显式数据合同。

真实快照含两套历史、有效参考与halo、音乐、原始物理xi和可见延迟历史。
ForecastContext 单独保存预测误差，绝不写入实测历史。只有真实采集边界允许detach；
想象中使用不可变replace传递张量图。Critic支持50Hz任意时刻；在途请求显式保存当时
可见的请求快照、预约及噪声，不能用未来实测完成时间补充其条件。
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
    pending_request: Any = None

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
            raise ValueError("生成请求必须位于无pending公共网格边界")
        if self.pending != (self.pending_request is not None):
            raise ValueError("在途标志必须有完整请求语义，不能只有pending布尔值")
        if self.pending_request is not None:
            if self.pending_request.ticket.request_tick > self.tick or self.pending_request.snapshot.pending:
                raise ValueError("在途请求不能引用未来状态或递归在途请求")
        if self.music.ndim != 3 or self.music.shape[0] != b or self.music.shape[-1] != 35:
            raise ValueError("音乐必须为[B,N,35]")
        if any(t < 0 for t in self.latency_history):
            raise ValueError("历史延迟不能为负")

    def detached(self):
        names = ("state", "xi", "previous_action", "sm_history", "genmo_history", "genmo_valid")
        return replace(self, **{k: getattr(self, k).detach().clone() for k in names},
                       music=self.music.detach(), reference=self.reference.detached(),
                       pending_request=self.pending_request.detached() if self.pending_request is not None else None)


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


@dataclass(frozen=True)
class PendingGeneration:
    snapshot: ClosedLoopState
    ticket: RequestTicket
    noise: torch.Tensor
    actor_version: int

    def detached(self):
        return replace(self, snapshot=self.snapshot.detached(), noise=self.noise.detach().clone())


@dataclass(frozen=True)
class GenerationRecord:
    request: PendingGeneration
    forecast: ForecastContext
    plan: GeneratedPlan
    actual_ready_tick: int
    commit_tick: int | None
    reference_version: int | None
    accepted: bool


@dataclass(frozen=True)
class PhysicalAnchor:
    """一次真实首次消费及其连续短轨迹；实际延迟仅为执行证据，不进入GENMO条件。"""
    generation: GenerationRecord
    state: ClosedLoopState
    transitions: tuple
    consumed_source_ticks: tuple[int, ...]
    previously_consumed_versions: tuple[int, ...]
    collector_id: int = 0
    episode_index: int = 0

    def validate(self, horizon):
        g = self.generation
        if not g.accepted or g.reference_version is None or g.commit_tick is None:
            raise ValueError("被拒绝或无实际接管的请求不是物理训练样本")
        self.state.validate()
        if self.state.history_origin != "real" or self.state.tick < g.commit_tick:
            raise ValueError("短时模型起点必须是接管后的真实状态")
        if g.reference_version in self.previously_consumed_versions or not self.consumed_source_ticks:
            raise ValueError("必须记录新计划第一次影响动作之前的状态，不能重复或伪造消费")
        if len(self.transitions) < horizon:
            raise ValueError("没有足够真实连续控制步支持该anchor")
        for index, step in enumerate(self.transitions[:horizon]):
            if step.steps != 1 or step.start.tick != self.state.tick+index*12 or step.end.tick != step.start.tick+12:
                raise ValueError("anchor控制轨迹缺失或时间戳不连续")
            if step.start.history_origin != "real":
                raise ValueError("物理样本不能来自WM")
        if g.request.snapshot.tick != g.request.ticket.request_tick or g.forecast.request_tick != g.request.ticket.request_tick:
            raise ValueError("重生成条件必须属于原请求时刻")


class ReferenceExhausted(RuntimeError):
    """明确终止：没有足够真实支持点，不外推或重复假参考。"""
