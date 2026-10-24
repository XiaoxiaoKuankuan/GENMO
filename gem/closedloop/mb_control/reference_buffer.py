"""不可变30Hz世界参考缓存与准确的50Hz依赖支持集。

保存GENMO原生关节序qpos28，解码后的世界anchor不随实际/预测根重置。读取时按名称
映射SM、SLERP及中心差分得到当前与未来10帧。支持集包含实际插值端点和速度halo；
提交使用cat/where创建新缓存，保留历史图，不做原地写或detach。60点只支持59/30秒。
"""

from dataclasses import dataclass, replace

import torch

from supermimic.bridge.reference import build_reference, current_future, derivative_ticks, slerp
from .contracts import CONTROL_TICKS, SOURCE_TICKS, ReferenceExhausted


@dataclass(frozen=True)
class ReferenceBuffer:
    ticks: torch.Tensor
    world_qpos: torch.Tensor
    contact: torch.Tensor
    source_versions: torch.Tensor
    kin: object
    to_sm: tuple[int, ...]
    version: int = 0

    def __post_init__(self):
        n = self.ticks.numel()
        if n < 2 or self.world_qpos.shape[1:] != (n, 28) or self.contact.shape != (*self.world_qpos.shape[:2], 2):
            raise ValueError("缓存需要至少两个30Hz姿态点及contact2")
        if self.ticks.dtype != torch.long or not bool(((self.ticks[1:] - self.ticks[:-1]) == 20).all()):
            raise ValueError("源时刻必须连续递增20tick")
        if tuple(sorted(self.to_sm)) != tuple(range(21)):
            raise ValueError("关节映射必须为21维唯一排列")
        if self.source_versions.shape != self.ticks.shape or self.source_versions.dtype != torch.long:
            raise ValueError("每个参考源点需要整数来源版本")
        if not torch.isfinite(self.world_qpos).all() or not torch.isfinite(self.contact).all():
            raise ValueError("动态参考包含非有限数值")
        if bool((self.world_qpos[..., 3:7].norm(dim=-1) < 1e-8).any()):
            raise ValueError("动态参考包含零四元数")

    @property
    def lower(self):
        return int(self.ticks[0])

    @property
    def upper(self):
        return int(self.ticks[-1])

    def _indices(self, ticks):
        if bool(((ticks < self.lower) | (ticks > self.upper)).any()):
            raise ReferenceExhausted(f"读取范围超出真实参考支持[{self.lower},{self.upper}]")
        offset = ticks - self.lower
        lo = (offset // SOURCE_TICKS).long()
        hi = (lo + 1).clamp_max(len(self.ticks) - 1)
        fraction = (offset % SOURCE_TICKS).to(self.world_qpos) / SOURCE_TICKS
        return lo, hi, fraction

    def source_at(self, ticks):
        lo, hi, a = self._indices(ticks)
        left, right = self.world_qpos[:, lo], self.world_qpos[:, hi]
        weight = a[None, :, None]
        return torch.cat(((1 - weight) * left[..., :3] + weight * right[..., :3],
                          slerp(left[..., 3:7], right[..., 3:7], a[None]),
                          (1 - weight) * left[..., 7:] + weight * right[..., 7:]), -1)

    def pose_at(self, ticks):
        pose = self.source_at(ticks)
        return pose[..., :3], pose[..., 3:7], pose[..., 7:][..., self.to_sm]

    def read_current_future(self, control_tick, future_steps=10):
        ticks = control_tick + torch.arange(future_steps + 1, device=self.ticks.device) * CONTROL_TICKS
        # 对正常滚动保留完整右halo，禁止尾部用单侧差分冒充完整目标。
        if int(ticks[-1]) + CONTROL_TICKS > self.upper:
            raise ReferenceExhausted("未来目标或右速度halo已耗尽")
        ref = build_reference(self.pose_at, ticks, self.lower, self.upper, self.kin)
        return current_future(ref)

    def dependency_support(self, control_tick, future_steps=10):
        ticks = control_tick + torch.arange(future_steps + 1, device=self.ticks.device) * CONTROL_TICKS
        if int(ticks[-1]) + CONTROL_TICKS > self.upper:
            raise ReferenceExhausted("支持集不足，不能计算影响时刻")
        left, right = derivative_ticks(ticks, self.lower, self.upper)
        all_ticks = torch.cat((ticks, left, right))
        lo, hi, fraction = self._indices(all_ticks)
        # 恰落源点时右插值权重为0，因此不算结构依赖。
        indices = torch.unique(torch.cat((lo, hi[fraction != 0])))
        return self.ticks[indices]

    def first_influence(self, commit_tick, first_mutable_tick):
        tick = ((commit_tick + 11) // 12) * 12
        while tick <= self.upper:
            if bool((self.dependency_support(tick) >= first_mutable_tick).any()):
                return tick
            tick += 12
        raise ReferenceExhausted("找不到新参考影响时刻")

    def first_center_consumed(self, commit_tick, first_mutable_tick):
        for tick in range(((commit_tick + 11) // 12) * 12, self.upper + 1, 12):
            source = self.ticks.new_tensor([tick])
            lo, hi, fraction = self._indices(source)
            if bool((self.ticks[lo] >= first_mutable_tick).any() or
                    ((self.ticks[hi] >= first_mutable_tick) & (fraction != 0)).any()):
                return tick
        raise ReferenceExhausted("没有可消费的新参考中心支持点")

    def merge_plan(self, plan, commit_tick, protection):
        ticket = plan.ticket
        if self.version != ticket.base_version:
            raise ValueError("重复提交或请求期间参考版本改变")
        if commit_tick < ticket.request_tick or protection >= ticket.first_mutable_source_tick:
            raise ValueError("晚到计划侵犯已消费支持点")
        if plan.world_qpos.shape != (self.world_qpos.shape[0], 60, 28):
            raise ValueError("生成计划必须保留60帧qpos28")
        keep = self.ticks < ticket.first_mutable_source_tick
        start = ticket.prefix_frames
        new_ticks = ticket.request_tick + torch.arange(start, 60, device=self.ticks.device) * 20
        if not bool(keep.any()) or int(self.ticks[keep][-1]) + 20 != int(new_ticks[0]):
            raise ReferenceExhausted("合并计划与旧参考之间有空洞")
        return replace(self, ticks=torch.cat((self.ticks[keep], new_ticks)),
                       world_qpos=torch.cat((self.world_qpos[:, keep], plan.world_qpos[:, start:]), 1),
                       contact=torch.cat((self.contact[:, keep], plan.contact[:, start:]), 1),
                       source_versions=torch.cat((self.source_versions[keep], torch.full_like(new_ticks, self.version + 1))),
                       version=self.version + 1)

    def trim(self, control_tick):
        # 留足最早当前速度差分的左右30Hz插值端点。
        first = ((control_tick - 12 - self.lower) // 20) * 20 + self.lower
        keep = self.ticks >= max(self.lower, first)
        return replace(self, ticks=self.ticks[keep], world_qpos=self.world_qpos[:, keep],
                       contact=self.contact[:, keep], source_versions=self.source_versions[keep])

    def detached(self):
        return replace(self, world_qpos=self.world_qpos.detach().clone(), contact=self.contact.detach().clone(),
                       ticks=self.ticks.clone(), source_versions=self.source_versions.clone())
