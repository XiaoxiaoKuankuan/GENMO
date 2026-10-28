"""多环境真实物理 sidecar 生产、独立槽位调度和原子分片索引。

每个 GPU 仅创建一个 Newton 环境批次。每个槽位独立拥有来源、时钟、历史与终止原因，
完成或失败后立即交给下一首；不会把 reset 前后的轨迹连接成成功片段。SM Actor 每控制步
执行一次整批前向，原生执行器按 50Hz 控制、200Hz 物理推进，参考/FK 算法仍来自原 bridge。
真实状态、动作及 GPU 归约接触先缓冲固定时间块，再统一拷回 CPU；不是每步复制接触列表。

完成的歌曲批量生成离线 FutureContext，写入不可变 v2 分片，再原子发布每 worker 索引及
公共索引。来源由确定性清单静态分配到各 GPU，已验证发布的来源不重复处理。半成品不进入
训练索引；数据、资产、可见旧计划、失败与时间合同保持原样。不创建 GENMO 或训练优化器。
显存同时报告PyTorch分配器峰值和块边界CUDA设备总占用采样峰值；后者覆盖Newton/Warp，
也包含同卡其他进程/驱动占用，不等于本进程的精确瞬时峰值。旧peak_memory_bytes仅保留
为Torch峰值的兼容字段，不能解释为全部VRAM。
"""

from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np
import torch

from .contracts import SIDECAR_SCHEMA, digest
from .data import SourceCatalog
from .offline import cache_requests_batched, source_provenance
from .prepare_rollouts import (asset_contract, preparation_lock, preparation_sources,
                              publish_index, source_episode)


class PreparationMemoryMonitor:
    """仅在初始化、时间块和Future前后查询显存，不在每个控制步同步CUDA。"""

    def __init__(self, device):
        self.device = torch.device(device)
        self.torch_peak, self.device_peak, self.device_last, self.device_total = 0, 0, 0, 0
        self.sample_stages = Counter()

    def sample(self, stage):
        if self.device.type == "cuda":
            free, total = torch.cuda.mem_get_info(self.device)
            self.device_last, self.device_total = int(total-free), int(total)
            self.device_peak = max(self.device_peak, self.device_last)
            self.torch_peak = max(self.torch_peak, int(torch.cuda.max_memory_allocated(self.device)))
        self.sample_stages[str(stage)] += 1

    def report(self):
        return {"peak_memory_bytes": self.torch_peak,
            "torch_peak_memory_allocated_bytes": self.torch_peak,
            "sampled_device_memory_used_bytes": self.device_peak,
            "last_sampled_device_memory_used_bytes": self.device_last,
            "device_memory_capacity_bytes": self.device_total,
            "memory_sample_stages": dict(self.sample_stages),
            "memory_statistics_contract": {
                "peak_memory_bytes": "compatibility_alias_of_torch_peak_memory_allocated_bytes",
                "torch_peak_memory_allocated_bytes": "pytorch_allocator_peak_only_excludes_newton_warp_allocations",
                "sampled_device_memory_used_bytes": "maximum_sampled_cuda_total_minus_free_including_other_processes_and_driver",
                "sampling": "collector_initialization_first_fill_trace_block_boundaries_before_after_offline_future_final",
                "device_value_is_exact_process_peak": False,
                "per_control_step_cuda_query": False}}


def empty_index(cfg, catalog, adapter, split):
    return {"schema": SIDECAR_SCHEMA, "split": split, "assets": asset_contract(cfg, adapter),
            "dataset": catalog.identity, "sources": {}, "shards": {}, "episodes": [],
            "sample_contract": cfg["data"]["sample_contract"], "source_versions": source_provenance(cfg)}


def verify_index(index, expected, directory):
    for key in ("schema", "split", "assets", "sample_contract", "dataset"):
        if index[key] != expected[key]:
            raise ValueError(f"批量预处理身份不匹配: {key}")
    for name, sha in index["shards"].items():
        if digest(Path(directory)/name) != sha:
            raise ValueError(f"已发布分片损坏: {name}")


@contextmanager
def merge_lock(root):
    with (Path(root)/".merge.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def merge_worker(root, worker_dir, worker_index):
    """同一主索引的并发追加持文件锁；仅引用已经 fsync 的不可变分片。"""
    root, worker_dir = Path(root), Path(worker_dir)
    prefix = worker_dir.relative_to(root)
    with merge_lock(root):
        index = json.loads((root/"index.json").read_text())
        for key in ("schema", "split", "assets", "sample_contract", "dataset"):
            if index[key] != worker_index[key]:
                raise ValueError(f"worker合并合同不同: {key}")
        for name, sha in worker_index["shards"].items():
            target = str(prefix/name)
            if target in index["shards"] and index["shards"][target] != sha:
                raise ValueError("已发布分片禁止改写")
            index["shards"][target] = sha
        added = set()
        for source, location in worker_index["sources"].items():
            location = {**location, "shard": str(prefix/location["shard"])}
            if source in index["sources"] and index["sources"][source] != location:
                raise ValueError(f"不同worker重复来源: {source}")
            if source not in index["sources"]:
                added.add(source)
                index["sources"][source] = location
        index["episodes"].extend(item for item in worker_index["episodes"] if item["source_key"] in added)
        index.update(published_sources=len(index["sources"]), last_publish_unix_seconds=time.time())
        publish_index(root, index)
    return len(added)


class TraceBlock:
    """整块搬回 CPU，按每步记录的 episode 编号拆分，允许块内槽位多次换歌。"""

    def __init__(self, steps, n, device):
        self.capacity, self.n, self.device = steps, n, device
        self.values, self.ids, self.actions = {}, [], []

    def append(self, ids, state_values, action_mask):
        k = len(self.ids)
        for name, value in state_values.items():
            if name not in self.values:
                self.values[name] = torch.empty((self.capacity, *value.shape), dtype=value.dtype, device=value.device)
            self.values[name][k].copy_(value)
        self.ids.append(list(ids))
        self.actions.append(np.asarray(action_mask).copy())

    def full(self):
        return len(self.ids) >= self.capacity

    def flush(self, records):
        if not self.ids:
            return
        size = len(self.ids)
        cpu = {name: value[:size].cpu() for name, value in self.values.items()}
        ids, actions = np.asarray(self.ids), np.asarray(self.actions)
        for episode_id in np.unique(ids):
            if episode_id < 0:
                continue
            record = records[int(episode_id)]
            t, slot = np.where(ids == episode_id)
            action_rows = actions[t, slot]
            for name, tensor in cpu.items():
                selection = tensor[t, slot]
                if name in ("action", "contacts"):
                    selection = selection[torch.from_numpy(action_rows)]
                record["chunks"].setdefault(name, []).append(selection)
        self.ids.clear()
        self.actions.clear()


def finish_episode(record, cfg, contact_contract):
    episode = record["episode"]
    for key, chunks in record["chunks"].items():
        episode[key] = torch.cat(chunks, 0)
    count = len(episode["state"])
    ticks = torch.arange(count)*12
    period, old = cfg["prepare"]["old_plan_period_ticks"], cfg["prepare"]["old_plan_frames"]
    versions = ticks//period
    episode["visible_upper"] = ((old-1+versions*(period//20))*20).clamp_max((episode["source_frames"]-1)*20)
    episode["plan_version"], episode["committed_tick"] = versions, versions*period
    episode["initial_state"] = episode["state"][0].clone()
    episode["xi"] = record["xi"]
    episode["contact_encoding"] = contact_contract
    contacts = episode.pop("contacts").tolist()
    episode["contacts"] = [{"source": "mujoco_warp_solver_active_contacts", "active_count": c[0],
                            "left_foot": bool(c[1]), "right_foot": bool(c[2])} for c in contacts]
    episode.update(termination_reason=record["reason"], last_tick=record["last_tick"],
                   control_steps=len(episode["action"]), wall_seconds=time.perf_counter()-record["started"],
                   frame_origin_tick=0, crop_anchor="fixed_episode_world_xy_yaw; one_floor_z_translation")
    return episode


@torch.no_grad()
def collect_batched(cfg, adapter, env, sources, emit, *, chunk_steps=64, forecast_batch_size=512):
    """固定 GPU 槽位池持续补位；emit 只接收完整真实轨迹与已完成前瞻。"""
    from .batched_reference import ReferencePageCache
    n, device = env.N, torch.device(env.device)
    pages = ReferencePageCache(n, max(64, chunk_steps), adapter, device)
    slots, ages = np.full(n, -1, dtype=np.int64), np.zeros(n, dtype=np.int64)
    records, next_id, done = {}, 0, deque()
    history = torch.zeros(n, 10, 75, device=device)
    block = TraceBlock(chunk_steps, n, device)
    memory = PreparationMemoryMonitor(device)
    memory.sample("collector_initialization")
    total_steps, total_requests, total_sources = 0, 0, 0
    start = time.perf_counter()
    workers = min(8, int(cfg["runtime"]["cpu_threads"]))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="sidecar-source") as pool:
        iterator, pending = iter(sources), deque()
        def schedule():
            while len(pending) < n:
                item = next(iterator, None)
                if item is None:
                    break
                dataset, row = item
                pending.append((dataset, row, pool.submit(source_episode, dataset, row, cfg)))
        schedule()
        def fill():
            nonlocal next_id
            reset_ids, initials = [], []
            for slot in np.flatnonzero(slots < 0):
                if not pending:
                    break
                dataset, row, future = pending.popleft()
                episode = future.result()
                if (episode["source_frames"]-1)*20 < 144:
                    raise ValueError(f"真实初始参考不足以执行控制: {episode['source_key']}")
                pages.assign(int(slot), episode)
                initials.append(pages.initial(int(slot)))
                reset_ids.append(int(slot))
                records[next_id] = {"episode": episode, "dataset": dataset, "row": row, "chunks": {},
                                    "started": time.perf_counter()}
                slots[slot], ages[slot] = next_id, 0
                next_id += 1
            if reset_ids:
                ids = torch.tensor(reset_ids, device=device)
                env.reset(ids, torch.cat(initials))
                initial = env.observe()
                xis = initial["xi"][ids].cpu()
                for i, slot in enumerate(reset_ids):
                    records[int(slots[slot])]["xi"] = xis[i]
                history[ids] = 0
            schedule()

        def end(slot, reason):
            identifier = int(slots[slot])
            records[identifier].update(reason=reason, last_tick=int(ages[slot])*12)
            done.append(identifier)
            slots[slot] = -1
            pages.release(int(slot))

        def publish_done():
            nonlocal total_sources, total_requests
            memory.sample("trace_block_before_flush")
            block.flush(records)
            memory.sample("trace_block_after_flush")
            completed = []
            while done:
                identifier = done.popleft()
                record = records.pop(identifier)
                episode = finish_episode(record, cfg, env.contact_contract())
                completed.append((episode, record["dataset"], record["row"]))
            if not completed:
                return
            memory.sample("offline_future_before")
            summaries = cache_requests_batched(completed, adapter, batch_size=forecast_batch_size)
            memory.sample("offline_future_after")
            from .requests import static_request
            for (episode, _, _), summary in zip(completed, summaries):
                summary["legal_windows"] = {str(h): sum(static_request(episode, i, h)[1] is None
                    for i in range(len(episode["offline_requests"]["frames"]))) for h in (16, 32, 64)}
                total_sources += 1
                total_requests += int(episode["offline_requests"]["valid"].sum())
                emit(episode, summary)
            elapsed = time.perf_counter()-start
            print(json.dumps({"batched_progress": True, "sources": total_sources,
                "control_steps": total_steps, "valid_future_requests": total_requests,
                "source_per_second": total_sources/elapsed, "control_steps_per_second": total_steps/elapsed,
                "future_requests_per_second": total_requests/elapsed, "wall_seconds": elapsed,
                "num_envs": n, **memory.report()}), flush=True)

        fill()
        memory.sample("initial_slots_filled")
        while (slots >= 0).any():
            for slot in np.flatnonzero(slots >= 0):
                episode = records[int(slots[slot])]["episode"]
                tick = int(ages[slot])*12
                visible = min(episode["source_frames"]-1, cfg["prepare"]["old_plan_frames"]-1 +
                              tick//cfg["prepare"]["old_plan_period_ticks"]*(cfg["prepare"]["old_plan_period_ticks"]//20))*20
                if tick+144 > visible:
                    end(slot, "source_or_reference_end")
            if not (slots >= 0).any():
                publish_done()
                fill()
                continue
            active = slots >= 0
            current, _, goal = pages.read(ages, active, [records[int(i)]["episode"] if i>=0 else None for i in slots])
            observed = env.observe()
            obs = adapter.runtime.observation(observed["state"], observed["previous_action"], observed["xi"], current)
            history = torch.cat((history[:, 1:], obs[:, None]), 1)
            fresh = torch.as_tensor((ages==0)&active, device=device)
            history[fresh] = obs[fresh, None].expand(-1, 10, -1)
            terminated = adapter.runtime.termination(observed["state"], current)
            finite_state = torch.isfinite(observed["state"]).all(-1)
            action = adapter.bundle.act(history, goal)
            finite_action = torch.isfinite(action).all(-1)
            flags = torch.stack((terminated, finite_state, finite_action), -1).cpu().numpy()
            executing = active.copy()
            reasons = {}
            for slot in np.flatnonzero(active):
                if not flags[slot, 1]:
                    reasons[slot] = "nonfinite_real_state"
                elif not flags[slot, 2]:
                    reasons[slot] = "nonfinite_sm_action"
                elif ages[slot] and flags[slot, 0]:
                    reasons[slot] = "sm_physical_termination"
                elif cfg["prepare"]["max_control_steps"] is not None and ages[slot] >= cfg["prepare"]["max_control_steps"]:
                    reasons[slot] = "explicit_control_budget"
                if slot in reasons:
                    executing[slot] = False
            # 已结束槽位只得到零控制，下一次分配才显式reset；这些推进不计入轨迹。
            masked_action = torch.where(torch.as_tensor(executing, device=device)[:, None], action, 0.)
            env.step_physics(masked_action)
            block.append(slots, {"state": observed["state"], "observation": obs,
                "proprio": adapter.proprio(observed["state"], observed["xi"]),
                "previous_action": observed["previous_action"], "action": action,
                "contacts": env.contact_events_tensor()}, executing)
            total_steps += int(executing.sum())
            ages[executing] += 1
            for slot, reason in reasons.items():
                end(slot, reason)
            if block.full():
                env.assert_solver_healthy()
                publish_done()
            fill()
        env.assert_solver_healthy()
        publish_done()
    elapsed = time.perf_counter()-start
    memory.sample("collector_final")
    return {"sources": total_sources, "control_steps": total_steps, "valid_future_requests": total_requests,
            "wall_seconds": elapsed, "source_per_second": total_sources/elapsed,
            "control_steps_per_second": total_steps/elapsed, "future_requests_per_second": total_requests/elapsed,
            "num_envs": n, "solver_health": env.solver_health_tensor().cpu().tolist(),
            "solver_allocations": env.allocation_summary(),
            **memory.report(),
            "reference_cache": pages.statistics()}


def prepare_worker(cfg, split, root, adapter, env, rank, world_size, *, chunk_steps=64, forecast_batch_size=512):
    """由协调器冻结来源清单，各worker独立分片，主索引只做原子追加。"""
    root = Path(root)
    directory = root/f"worker_{rank:02d}"
    directory.mkdir(exist_ok=True)
    catalog = SourceCatalog(cfg, split)
    expected = empty_index(cfg, catalog, adapter, split)
    manifest = json.loads((root/"batch_manifest.json").read_text())
    if manifest["world_size"] != world_size:
        raise ValueError("断点续处理必须保留worker分配数量")
    assigned = set(manifest["assignments"][str(rank)])
    with preparation_lock(directory):
        index = json.loads((directory/"index.json").read_text()) if (directory/"index.json").exists() else expected
        verify_index(index, expected, directory)
        # worker已发布而主索引尚未合并时崩溃，恢复先补发布，再决定跳过哪些来源。
        merge_worker(root, directory, index)
        sources = []
        for dataset, row in preparation_sources(catalog, cfg["prepare"]):
            key = json.dumps([dataset.dataset_name, dataset.split, dataset.rows[row]["sample_id"]], ensure_ascii=False, separators=(",", ":"))
            if key in assigned and key not in index["sources"]:
                sources.append((dataset, row))
        pending = []
        def flush():
            if not pending:
                return
            name = f"shard_{len(index['shards']):05d}_{uuid.uuid4().hex[:8]}.pt"
            with (directory/name).open("xb") as handle:
                torch.save([e for e, _ in pending], handle)
                handle.flush()
                os.fsync(handle.fileno())
            index["shards"][name] = digest(directory/name)
            for i, (episode, summary) in enumerate(pending):
                index["sources"][episode["source_key"]] = {"shard": name, "episode": i}
                item = {key: episode[key] for key in ("source_key", "control_steps", "termination_reason", "wall_seconds")}
                item["offline_preprocessing"] = summary
                index["episodes"].append(item)
            pending.clear()
            publish_index(directory, index)
            merge_worker(root, directory, index)
        def emit(episode, summary):
            pending.append((episode, summary))
            if len(pending) >= cfg["prepare"]["shard_episodes"]:
                flush()
        frozen = adapter.bundle.frozen_snapshot()
        result = collect_batched(cfg, adapter, env, sources, emit, chunk_steps=chunk_steps,
                                 forecast_batch_size=forecast_batch_size)
        flush()
        adapter.bundle.assert_unchanged(frozen)
        result["frozen_unchanged"] = True
        result["legal_windows"] = {str(h): sum(item["offline_preprocessing"]["legal_windows"][str(h)]
                                              for item in index["episodes"]) for h in (16,32,64)}
        (directory/"summary.json").write_text(json.dumps(result, indent=2))
        return result
