"""原数据动作由冻结SM连续跟踪的Newton附属缓存生成器。

每source只reset一次，从原时间零点连续推进50Hz物理；终止立即保存，绝不复位拼接。
保存每步真实state55/obs75/proprio48/action与接触，历史按索引还原而非重复整首数据。
旧teacher-forced计划以120源帧启动，按已记录时间只扩展可见后缀；请求时可见范围和版本
独立存储。音乐只记录文件身份，不在每个anchor重复保存。分片只新建、不覆盖。
增量生产与八卡训练分属独立进程：单写者文件锁防止重复采集；完整分片先落盘，再原子发布
索引，读者只看到完整版本。可先初始化空索引并按数据来源交错扩充；断点跳过已验证来源，
保留失败和提前终止，不为获得有效窗口重置拼接，也不在训练进程中重新执行Newton。
"""

import json
from contextlib import contextmanager
import fcntl
from functools import lru_cache
import os
from pathlib import Path
import time
import uuid

import torch

from supermimic.mdp import flatten_state
from gem.closedloop.mb_control.reference_buffer import ReferenceBuffer
from .config import ROOT
from .contracts import SIDECAR_SCHEMA, digest, source_key
from .data import SourceCatalog


def asset_contract(cfg, adapter):
    return {"sm_wm_sha256": adapter.bundle.metadata["sha256"],
        "stats_sha256": digest(cfg["stage1"]["endecoder"]["stats_path"]),
        "fk_sha256": digest(ROOT / cfg["stage1"]["endecoder"]["kinematics_path"]),
        "robot": adapter.bundle.metadata["robot"], "time": {"clock": 600, "source": 30, "control": 50, "physics": 200},
        "backend": cfg["physics"]["backend"], "determinism": cfg["physics"]["determinism"],
        "observation_noise": False}


def collect_source(dataset, row_index, cfg, adapter, env):
    sequence = dataset.reader.load_aligned_sequence(dataset.rows[row_index])
    sample = dataset.get_window(row_index, 0)
    meta = sample["meta"]
    qpos = sequence["qpos"].clone()
    ground = dataset._ground_supervision(sequence)
    qpos[:, 2] -= ground["ground_height_world_m"]
    n = len(qpos)
    old_frames, period = cfg["prepare"]["old_plan_frames"], cfg["prepare"]["old_plan_period_ticks"]
    if old_frames < 60 or period % 60:
        raise ValueError("旧计划可见范围不足或提交不在公共网格")
    # 每次只公布原先未知的新后缀，已公布源点的版本保持不变。
    source_versions = torch.div((torch.arange(n)-old_frames).clamp_min(0)*20, period, rounding_mode="floor")+1
    source_versions[:old_frames] = 0
    episode = {"source_key": source_key(meta), "split": dataset.split, "dataset_id": dataset.dataset_name,
        "sample_id": meta["sample_id"], "source_frames": n, "motion_path": meta["motion_path"],
        "music_feature_path": meta["music_feature_path"], "motion_sha256": digest(meta["motion_path"]),
        "music_sha256": digest(meta["music_feature_path"]), "ground_height_world_m": ground["ground_height_world_m"],
        "ground_metadata": ground, "reference_qpos": qpos, "source_versions": source_versions,
        "old_plan_origin": "teacher_forced_append_only_precommitted", "reset_count": 1,
        "state_origin": "continuous_newton_observation_not_data_reference", "source_crop": [0, n]}
    device = env.device
    @lru_cache(maxsize=2)
    def cache_for_version(version):
        upper = min(n-1, old_frames-1 + version*(period//20))
        ticks = torch.arange(upper+1, device=device)*20
        ref = ReferenceBuffer(ticks, qpos[:upper+1].to(device)[None], torch.zeros(1, upper+1, 2, device=device),
            source_versions[:upper+1].to(device), adapter.runtime.kin, adapter.to_sm, version=version)
        return ref, version*period
    ref, _ = cache_for_version(0)
    initial, _, _ = ref.read_current_future(0)
    initial_state = flatten_state((initial.root_p, initial.root_R, initial.root_v, initial.root_w, initial.q, initial.qd))
    env.reset(torch.arange(1, device=device), initial_state)
    initial_obs = env.observe()
    episode["xi"] = initial_obs["xi"][0].cpu()
    episode["initial_state"] = initial_obs["state"][0].cpu()
    fields = {key: [] for key in ("state", "observation", "proprio", "previous_action", "action", "visible_upper", "plan_version", "committed_tick", "contacts")}
    tick, started = 0, time.perf_counter()
    history = None
    with torch.no_grad():
        while True:
            ref, committed = cache_for_version(tick // period)
            if tick+144 > ref.upper:
                reason = "source_or_reference_end"
                break
            observed = env.observe()
            current, _, goal = ref.read_current_future(tick)
            obs = adapter.runtime.observation(observed["state"], observed["previous_action"], observed["xi"], current)
            history = obs[:, None].expand(-1, 10, -1).clone() if history is None else torch.cat((history[:, 1:], obs[:, None]), 1)
            for key, val in (("state", observed["state"][0]), ("observation", obs[0]),
                             ("proprio", adapter.proprio(observed["state"], observed["xi"])[0]),
                             ("previous_action", observed["previous_action"][0])):
                fields[key].append(val.cpu())
            fields["visible_upper"].append(ref.upper)
            fields["plan_version"].append(ref.version)
            fields["committed_tick"].append(committed)
            if tick and bool(adapter.runtime.termination(observed["state"], current).any()):
                reason = "sm_physical_termination"
                break
            if cfg["prepare"]["max_control_steps"] is not None and tick//12 >= cfg["prepare"]["max_control_steps"]:
                reason = "explicit_control_budget"
                break
            action = adapter.bundle.act(history, goal)
            env.step_physics(action)
            fields["action"].append(action[0].cpu())
            fields["contacts"].append(env.contact_events())
            tick += 12
    for key, values in fields.items():
        episode[key] = (values if key == "contacts" else torch.stack(values) if values and isinstance(values[0], torch.Tensor)
                        else torch.tensor(values))
    episode.update(termination_reason=reason, last_tick=tick, control_steps=len(fields["action"]),
                   wall_seconds=time.perf_counter()-started, frame_origin_tick=0,
                   crop_anchor="fixed_episode_world_xy_yaw; one_floor_z_translation")
    return episode


@contextmanager
def preparation_lock(output):
    """同一目录仅允许一个生产者；锁文件持久保留，进程退出由内核释放锁。"""
    with (Path(output)/".prepare.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("该sidecar目录已有采集进程，禁止并发写入或重复启动") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def publish_index(output, index):
    """先持久化临时索引，再原子替换；训练不会读到半个JSON或未完成分片。"""
    output = Path(output)
    temporary = output/f".index_{uuid.uuid4().hex}.partial"
    try:
        with temporary.open("x") as handle:
            json.dump(index, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(output/"index.json")
        descriptor = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def preparation_sources(catalog, options):
    """固定seed打乱每个来源并轮流取样，防止增量预算全部消耗在第一种数据上。"""
    if not options.get("balanced_sources", False):
        yield from catalog.sources()
        return
    import numpy as np
    generators = []
    seed = int(options.get("source_seed", 42))
    for i, (dataset, rows) in enumerate(zip(catalog.datasets, catalog.rows)):
        permutation = np.random.default_rng(seed+i*100003).permutation(rows).tolist()
        generators.append((dataset, iter(permutation)))
    while generators:
        active = []
        for dataset, iterator in generators:
            row = next(iterator, None)
            if row is not None:
                yield dataset, row
                active.append((dataset, iterator))
        generators = active


def prepare(cfg, split, output, adapter, env, *, initialize_only=False):
    """创建或继续独立v2目录，持锁覆盖身份检查、真实采集及原子发布全过程。"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=cfg["prepare"].get("resume", False))
    with preparation_lock(output):
        return _prepare_locked(cfg, split, output, adapter, env, initialize_only)


def _prepare_locked(cfg, split, output, adapter, env, initialize_only):
    from .offline import cache_requests, preprocessing_codec, source_provenance
    catalog = SourceCatalog(cfg, split)
    output = Path(output)
    resume = cfg["prepare"].get("resume", False)
    started = time.perf_counter()
    index = {"schema": SIDECAR_SCHEMA, "split": split, "assets": asset_contract(cfg, adapter),
             "dataset": catalog.identity, "sources": {}, "shards": {}, "episodes": [],
             "sample_contract": cfg["data"]["sample_contract"], "source_versions": source_provenance(cfg)}
    if resume and (output/"index.json").exists():
        previous = json.loads((output/"index.json").read_text())
        if any(previous[k] != index[k] for k in ("schema", "split", "assets", "sample_contract", "dataset")):
            raise ValueError("离线断点续处理身份不同，必须使用新目录")
        for shard,sha in previous["shards"].items():
            if digest(output/shard) != sha:
                raise ValueError("已保存分片身份改变，禁止重复采集或覆盖")
        index = previous
    if initialize_only:
        if not (output/"index.json").exists():
            publish_index(output, index)
        return index
    if env is None:
        raise ValueError("真实增量采集必须提供Newton环境；空索引初始化才可不启动环境")
    skip = set(index["sources"])
    for directory in cfg["prepare"].get("exclude_sidecars", []):
        from .data import SidecarStore
        existing = SidecarStore(directory, split, index["assets"], sample_contract=cfg["data"]["sample_contract"])
        skip.update(existing.index["sources"])
    codec = preprocessing_codec(cfg)
    pending = []
    frozen = adapter.bundle.frozen_snapshot()
    def flush():
        adapter.bundle.assert_unchanged(frozen)
        name = f"shard_{len(index['shards']):05d}_{uuid.uuid4().hex[:8]}.pt"
        with (output/name).open("xb") as file:
            torch.save(pending, file)
            file.flush()
            os.fsync(file.fileno())
        index["shards"][name] = digest(output/name)
        for i, episode in enumerate(pending):
            index["sources"][episode["source_key"]] = {"shard": name, "episode": i}
        pending.clear()
        index["published_sources"] = len(index["sources"])
        index["last_publish_unix_seconds"] = time.time()
        publish_index(output, index)
    from collections import Counter
    per_dataset = Counter(json.loads(key)[0] for key in index['sources'])
    for dataset, row in preparation_sources(catalog, cfg["prepare"]):
        limit = cfg["prepare"].get("max_sources")
        if limit is not None and len(index["episodes"]) >= limit:
            break
        key = json.dumps([dataset.dataset_name, dataset.split, dataset.rows[row]["sample_id"]], ensure_ascii=False, separators=(",", ":"))
        if key in skip:
            continue
        per_limit = cfg["prepare"].get("max_sources_per_dataset")
        if per_limit is not None and per_dataset.get(dataset.dataset_name,0) >= per_limit:
            continue
        per_dataset[dataset.dataset_name] = per_dataset.get(dataset.dataset_name,0)+1
        episode = collect_source(dataset, row, cfg, adapter, env)
        preprocessing = cache_requests(episode, dataset, row, adapter, codec)
        from .requests import static_request
        preprocessing["legal_windows"] = {str(horizon): sum(
            static_request(episode, request_index, horizon)[1] is None
            for request_index in range(len(episode["offline_requests"]["frames"])))
            for horizon in (16, 32, 64)}
        pending.append(episode)
        index["episodes"].append({k: episode[k] for k in ("source_key", "control_steps", "termination_reason", "wall_seconds")})
        index["episodes"][-1]["offline_preprocessing"] = preprocessing
        print(json.dumps(index["episodes"][-1], ensure_ascii=False), flush=True)
        if len(pending) >= cfg["prepare"]["shard_episodes"]:
            flush()
    if pending:
        flush()
    adapter.bundle.assert_unchanged(frozen)
    index.update(wall_seconds=time.perf_counter()-started, frozen_unchanged=True)
    publish_index(output, index)
    return index
