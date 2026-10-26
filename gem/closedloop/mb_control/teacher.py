"""冻结原 Stage1 的音乐动作教师，以及与学生前缀无关的固定音乐参考。

教师从 s595000 原始检查点重新构造独立网络，严格加载原网络、统计与编解码合同，
从不复制正在优化的学生，也不加入任何优化器。局部标签使用真实请求的音乐、历史、
已知前缀及相同 DDIM 初始噪声；标签计算无梯度，学生的完整 DDIM 路径仍可微。
新目标优先加载原Stage1+冻结SM真实在线执行时实际消费的参考，不随学生更新改写。
旧独立运动学时间线保留为研究对照，使用教师自身的因果历史和前缀，明确标记
imagined，不会写入学生真实历史。固定参考本身不是学生执行或训练收益证据。
整曲目标在 episode 开始前按音乐内容缓存；局部标签只在训练侧按请求缓存。
GenerationWorker 不调用教师，部署时不增加教师推理延迟。
"""

from collections import OrderedDict
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from omegaconf import OmegaConf

from gem.closedloop.checkpoint import load_stage1_checkpoint
from gem.closedloop.training import build_stage1_actor
from supermimic.bridge.bundle import atomic_save, file_digest
from supermimic.rotations import mat_to_quat
from .contracts import GeneratedPlan
from .online_conditions import build_conditions
from .reference_buffer import ReferenceBuffer


def tensor_digest(*values):
    digest = hashlib.sha256()
    for value in values:
        a = value.detach().cpu().contiguous()
        digest.update(str((tuple(a.shape), a.dtype)).encode())
        digest.update(a.numpy().tobytes())
    return digest.hexdigest()


class FrozenStage1Teacher(nn.Module):
    def __init__(self, cfg, adapter, scheduler, device):
        super().__init__()
        self.adapter, self.scheduler = adapter, scheduler
        self.config = dict(cfg["teacher"])
        self.physics_determinism = cfg.get("execution", {}).get("physics_determinism", "not_guaranteed")
        path = Path(cfg["stage1"]["checkpoint"])
        ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        original = OmegaConf.create(ck["config"])
        original.endecoder.stats_path = str(Path(cfg["stage1"]["stats"]).resolve())
        original.endecoder.kinematics_path = str(Path(cfg["stage1"]["kinematics"]).resolve())
        data = OmegaConf.create({"datasets": {}, "sample_contract": {"history_steps": 50},
            "qpos30_stats": {"path": original.endecoder.stats_path},
            "dataset_defaults": {"kinematics_path": original.endecoder.kinematics_path}})
        # 教师构造和缓存不得改变学生采样的随机序列。
        with torch.random.fork_rng(devices=[]):
            self.stage1 = build_stage1_actor(original, data)
            load_stage1_checkpoint(self.stage1, ck)
        self.stage1.to(device).requires_grad_(False).eval()
        self.identity = {"schema": "frozen_stage1_music_teacher.v1", "checkpoint": str(path),
            "checkpoint_sha256": file_digest(path), "stats_sha256": file_digest(cfg["stage1"]["stats"]),
            "kinematics_sha256": file_digest(cfg["stage1"]["kinematics"]),
            "ddim_steps": 20, "eta": 0., "cfg_scale": 2.5,
            "timeline_history": "teacher_own_causal_kinematics_not_student_real_history"}
        if self.config.get("target_mode") == "online_consumed_reference_v1":
            self.identity.update(target_definition="online_consumed_reference_v1",
                timeline_history="frozen_original_stage1_real_online_reference",
                request_noise_rule="seed_plus_request_tick_grid_times_1009.v1",
                online_seed=self.config.get("online_seed",42))
        self.cache_dir = Path(self.config["cache_dir"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.local_cache = OrderedDict()
        self.timeline_cache = OrderedDict()
        self._versions = {n: v._version for n, v in self.stage1.state_dict(keep_vars=True).items()}

    def train(self, mode=True):
        super().train(False)
        return self

    def assert_frozen(self):
        if any(p.requires_grad or p.grad is not None for p in self.stage1.parameters()):
            raise RuntimeError("冻结音乐教师出现可训练参数或梯度")
        if any(v._version != self._versions[n] for n, v in self.stage1.state_dict(keep_vars=True).items()):
            raise RuntimeError("冻结音乐教师参数或统计被原地修改")

    @torch.no_grad()
    def generate(self, snapshot, ticket, noise):
        conditions, anchor = build_conditions(snapshot, ticket, self.stage1.endecoder.codec)
        # 原DDIM即使eta=0仍会抽取乘零噪声。隔离这些随机数，使缓存冷/热以及
        # 教师查询次数都不改变学生下一次真实请求的噪声；调用发生于训练侧无worker并发时。
        devices = [noise.device.index] if noise.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            out = self.stage1.sample(conditions, steps=20, guidance_scale=2.5, noise=noise)
        world = self.stage1.endecoder.codec.apply_world_anchor(out["qpos"], anchor)
        return GeneratedPlan(ticket, world.detach(), out["contact"].detach())

    @torch.no_grad()
    def local(self, request):
        key = tensor_digest(request.noise, request.snapshot.genmo_history,
            request.snapshot.genmo_valid, request.snapshot.reference.world_qpos,
            request.snapshot.reference.ticks, request.snapshot.music)
        key = (request.ticket, request.snapshot.music_start_tick, request.snapshot.music_end_tick, key)
        if key not in self.local_cache:
            plan = self.generate(request.snapshot, request.ticket, request.noise)
            self.local_cache[key] = (plan.world_qpos.cpu(), plan.contact.cpu())
            while len(self.local_cache) > self.config.get("local_cache_size", 256):
                self.local_cache.popitem(last=False)
        self.local_cache.move_to_end(key)
        q, c = self.local_cache[key]
        return GeneratedPlan(request.ticket, q.to(request.noise), c.to(request.noise))

    def _buffer(self, q, contact):
        ticks = torch.arange(q.shape[1], device=q.device, dtype=torch.long)*20
        return ReferenceBuffer(ticks, q, contact, torch.zeros_like(ticks),
            self.adapter.runtime.kin, self.adapter.to_sm)

    @torch.no_grad()
    def timeline(self, music, initial):
        if self.config.get("target_mode") == "online_consumed_reference_v1":
            from .online_teacher import load_online_timeline
            key=tensor_digest(music,initial)
            if key not in self.timeline_cache:
                self.timeline_cache[key]=load_online_timeline(self,music,initial)
                while len(self.timeline_cache)>self.config.get("timeline_memory_cache_size",4):
                    self.timeline_cache.popitem(last=False)
            self.timeline_cache.move_to_end(key)
            return self.timeline_cache[key]
        return self.independent_timeline(music,initial)

    @torch.no_grad()
    def independent_timeline(self, music, initial):
        key = tensor_digest(music, initial)
        identity = dict(self.identity, music_initial_sha256=key, seed=self.config.get("seed", 73421),
                        request_interval_ticks=300, prefix_clock=dict(self.scheduler.__dict__))
        filename = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        if filename in self.timeline_cache:
            self.timeline_cache.move_to_end(filename)
            return self.timeline_cache[filename]
        path = self.cache_dir/f"{filename}.pt"
        if path.is_file():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload["identity"] != identity:
                raise ValueError("固定教师参考资产或音乐合同不匹配")
            q, contact = payload["world_qpos"].to(initial), payload["contact"].to(initial)
        else:
            from .gradient_audit import tensor_fixture
            z = replace(tensor_fixture(self.adapter, music, initial), history_origin="imagined")
            rng = torch.Generator(device=initial.device).manual_seed(identity["seed"])
            pieces, contacts = [], []
            # 额外一个请求为音乐末端提供真实生成的速度 halo，不重复末帧伪造未来。
            while z.tick <= z.music_end_tick+300:
                ticket = self.scheduler.reserve(z)
                noise = torch.randn(1, 60, 30, generator=rng, device=initial.device)
                plan = self.generate(z, ticket, noise)
                cache = z.reference.merge_plan(plan, ticket.request_tick, ticket.first_mutable_source_tick-20)
                z = replace(z, reference=cache, decision_id=z.decision_id+1)
                source_ticks = z.tick + torch.arange(15, device=initial.device)*20
                pieces.append(cache.source_at(source_ticks))
                idx = ((source_ticks-cache.lower)//20).long()
                contacts.append(cache.contact[:, idx])
                for _ in range(25):
                    reference, _, _ = z.reference.read_current_future(z.tick+12)
                    state = torch.cat((reference.root_p, mat_to_quat(reference.root_R), reference.root_v,
                        reference.root_w, reference.q, reference.qd), -1)
                    z = self.adapter.advance(z, state, z.previous_action, origin="imagined")
            tail_ticks = torch.arange(z.tick, z.reference.upper+1, 20, device=initial.device)
            pieces.append(z.reference.source_at(tail_ticks))
            contacts.append(z.reference.contact[:, ((tail_ticks-z.reference.lower)//20).long()])
            q, contact = torch.cat(pieces, 1), torch.cat(contacts, 1)
            payload = {"identity": identity, "world_qpos": q.cpu(), "contact": contact.cpu(),
                "evidence": "independent_frozen_stage1_generated_target_not_physical_execution"}
            # 不同采集器可能同时准备同一歌曲；已完成的相同缓存可直接复用。
            if not path.exists():
                try:
                    atomic_save(payload, path)
                except FileExistsError:
                    pass
        result = self._buffer(q, contact)
        self.timeline_cache[filename] = result
        while len(self.timeline_cache) > self.config.get("timeline_memory_cache_size", 4):
            self.timeline_cache.popitem(last=False)
        self.assert_frozen()
        return result
