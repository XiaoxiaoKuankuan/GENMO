"""真实批量采集的原生参考分页缓存，减少每控制步重复 FK 和逐歌曲目标装配。

每个槽位接收一首原始动作，只通过原 ReferenceBuffer/SM bridge 完整计算一次参考，
随后整曲解析参考保存在 CPU；GPU 仅保留 chunk_steps+11 个50Hz时刻的窗口。各槽位独立
记录歌曲、页起点和年龄，批量 gather 当前及未来10点后调用原 current_future 生成 SM goal。
槽位结束后 release 释放整首 CPU 缓存，新歌曲使用独立初态，不拼接 reset 前后的轨迹。

完整解析缓存只是一种计算复用，不改变当时可用信息。每次 read 必须再次核对原
teacher-forced append-only计划的提交版本、可见上界和速度halo；未来未提交的源帧即使已
离线计算也不能进入当前控制。页尾未填充区域仅为无效存储，不能通过复制末姿态延长参考。
无效槽位读取一个合法槽位的有限参考，供整批SM计算占位，采集器仍负责掩蔽动作和记录。
本模块不生成FutureContext、不运行WM、不改变SM/FK/旋转/插值与速度计算公式。
"""

from __future__ import annotations

import numpy as np
import torch

from supermimic.bridge.reference import current_future
from supermimic.mdp import flatten_state
from supermimic.motion import ReferenceBatch
from gem.closedloop.mb_control.contracts import ReferenceExhausted
from gem.closedloop.mb_control.reference_buffer import ReferenceBuffer


class ReferencePageCache:
    """独立年龄的环境槽位参考池；initial返回[1,55]，read返回整批N个参考/goal。"""

    def __init__(self, num_envs, chunk_steps, adapter, device):
        if num_envs < 1 or chunk_steps < 1:
            raise ValueError("参考页环境数量与页长必须为正")
        self.num_envs, self.chunk_steps = int(num_envs), int(chunk_steps)
        self.page_frames = self.chunk_steps + 11
        self.adapter, self.device = adapter, torch.device(device)
        self._full = [None] * self.num_envs
        self._episodes = [None] * self.num_envs
        self._starts = np.full(self.num_envs, -1, dtype=np.int64)
        self._pages = None
        self._slots = torch.arange(self.num_envs, device=self.device)
        self._offsets = torch.arange(11, device=self.device)
        self.page_uploads = 0

    @torch.no_grad()
    def assign(self, slot, episode):
        """每首只用原ReferenceBuffer构造一次；不会把真实未来状态写入缓存。"""
        slot = int(slot)
        if not 0 <= slot < self.num_envs:
            raise IndexError("参考槽位越界")
        if self._episodes[slot] is not None:
            raise ValueError("复用参考槽位前必须release，不能覆盖仍在执行的歌曲")
        old, period = int(episode["old_plan_frames"]), int(episode["old_plan_period_ticks"])
        if old < 60 or period <= 0 or period % 60:
            raise ValueError("旧参考计划的长度/提交时钟不符合原合同")
        qpos = episode["reference_qpos"].to(self.device, dtype=torch.float32)
        n = len(qpos)
        ticks = torch.arange(n, device=self.device, dtype=torch.long) * 20
        ref = ReferenceBuffer(ticks, qpos[None], qpos.new_zeros((1, n, 2)),
            episode["source_versions"].to(self.device), self.adapter.runtime.kin, self.adapter.to_sm)
        ref.read_current_future(0)
        full = ReferenceBatch(*(value[0].detach().cpu().contiguous() for value in ref.prepared_reference))
        if self._pages is None:
            self._pages = ReferenceBatch(*(torch.zeros((self.num_envs, self.page_frames, *value.shape[1:]),
                                                       dtype=value.dtype, device=self.device) for value in full))
        elif any(a.shape[2:] != b.shape[1:] for a, b in zip(self._pages, full)):
            raise ValueError("新参考身体/关节字段形状与当前资产不一致")
        self._full[slot], self._episodes[slot], self._starts[slot] = full, episode, -1

    def release(self, slot):
        slot = int(slot)
        self._full[slot], self._episodes[slot], self._starts[slot] = None, None, -1

    def initial(self, slot):
        full = self._full[int(slot)]
        if full is None:
            raise ValueError("未分配参考的槽位没有初态")
        return flatten_state(tuple(value[:1].to(self.device) for value in full[:6]))

    def _upload(self, slots, starts):
        indices = torch.as_tensor(slots, dtype=torch.long, device=self.device)
        # 同次缺页按字段合批传输，避免每个slot分别启动13次CPU→GPU小拷贝。
        for field, destination in enumerate(self._pages):
            copies = []
            for slot, start in zip(slots, starts):
                values = self._full[int(slot)][field][int(start):int(start)+self.page_frames]
                if len(values) < self.page_frames:
                    # 仅填充无效存储为零，read的真实支持检查保证这些位置不会被消费。
                    padded = values.new_zeros((self.page_frames, *values.shape[1:]))
                    padded[:len(values)].copy_(values)
                    values = padded
                copies.append(values)
            destination[indices] = torch.stack(copies).to(self.device)
        self._starts[slots] = starts
        self.page_uploads += len(slots)

    @torch.no_grad()
    def read(self, ages_cpu, active_cpu, episodes_slots):
        """独立槽位age批量读取；时间/已提交参考不足直接拒绝，绝不扩展支持。"""
        ages = np.asarray(ages_cpu, dtype=np.int64)
        active = np.asarray(active_cpu, dtype=np.bool_)
        if ages.shape != (self.num_envs,) or active.shape != (self.num_envs,):
            raise ValueError("独立环境年龄/有效掩码形状错误")
        live = np.flatnonzero(active)
        if len(live) == 0:
            raise ReferenceExhausted("没有合法活动参考；采集器应填充新歌曲或结束")
        for slot in live:
            episode = self._episodes[slot]
            if episode is None or episodes_slots[slot] is not episode:
                raise ValueError("槽位来源身份改变或已释放，禁止串用歌曲参考")
            tick = int(ages[slot]) * 12
            if tick < 0:
                raise ReferenceExhausted("控制年龄不能为负")
            old, period = int(episode["old_plan_frames"]), int(episode["old_plan_period_ticks"])
            visible = min(len(episode["reference_qpos"])-1, old-1 + (tick//period)*(period//20)) * 20
            if tick + 144 > visible:
                raise ReferenceExhausted(f"槽位{slot}的已提交参考/速度halo不足: tick={tick},upper={visible}")
            if int(ages[slot])+10 >= len(self._full[slot].root_p):
                raise ReferenceExhausted("页读取超出真实源动作支持")
        starts = (ages[live] // self.chunk_steps) * self.chunk_steps
        missing = self._starts[live] != starts
        if missing.any():
            self._upload(live[missing], starts[missing])
        # 无效slot复用第一个活动slot的解析参考，只作有限占位，不生成其真实轨迹。
        selected_slots = np.where(active, np.arange(self.num_envs), int(live[0]))
        selected_ages = ages[selected_slots]
        row = torch.as_tensor(selected_slots, dtype=torch.long, device=self.device)
        offsets = torch.as_tensor(selected_ages-self._starts[selected_slots], dtype=torch.long, device=self.device)
        time = offsets[:, None] + self._offsets[None]
        window = ReferenceBatch(*(field[row[:, None], time] for field in self._pages))
        return current_future(window)

    def statistics(self):
        return {"page_uploads": int(self.page_uploads), "page_frames": self.page_frames,
            "resident_cpu_bytes": sum(value.numel()*value.element_size()
                for ref in self._full if ref is not None for value in ref),
            "resident_gpu_bytes": 0 if self._pages is None else sum(value.numel()*value.element_size() for value in self._pages)}
