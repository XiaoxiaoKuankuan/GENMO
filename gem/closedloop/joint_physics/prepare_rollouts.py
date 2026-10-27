"""原数据动作由冻结SM连续跟踪的Newton附属缓存生成器。

每source只reset一次，从原时间零点连续推进50Hz物理；终止立即保存，绝不复位拼接。
保存每步真实state55/obs75/proprio48/action与接触，历史按索引还原而非重复整首数据。
旧teacher-forced计划以120源帧启动，按已记录时间只扩展可见后缀；请求时可见范围和版本
独立存储。音乐只记录文件身份，不在每个anchor重复保存。分片只新建、不覆盖。
"""

import json
from pathlib import Path
import time

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
    def cache_at(tick):
        version = tick // period
        upper = min(n-1, old_frames-1 + version*(period//20))
        ticks = torch.arange(upper+1, device=device)*20
        ref = ReferenceBuffer(ticks, qpos[:upper+1].to(device)[None], torch.zeros(1, upper+1, 2, device=device),
            source_versions[:upper+1].to(device), adapter.runtime.kin, adapter.to_sm, version=version)
        return ref, version*period
    ref, _ = cache_at(0)
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
            ref, committed = cache_at(tick)
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


def prepare(cfg, split, output, adapter, env):
    catalog = SourceCatalog(cfg, split)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    index = {"schema": SIDECAR_SCHEMA, "split": split, "assets": asset_contract(cfg, adapter),
             "dataset": catalog.identity, "sources": {}, "shards": {}, "episodes": []}
    pending = []
    frozen = adapter.bundle.frozen_snapshot()
    def flush():
        name = f"shard_{len(index['shards']):05d}.pt"
        with (output/name).open("xb") as file:
            torch.save(pending, file)
        index["shards"][name] = digest(output/name)
        for i, episode in enumerate(pending):
            index["sources"][episode["source_key"]] = {"shard": name, "episode": i}
        pending.clear()
    for dataset, row in catalog.sources():
        limit = cfg["prepare"].get("max_sources")
        if limit is not None and len(index["episodes"]) >= limit:
            break
        episode = collect_source(dataset, row, cfg, adapter, env)
        pending.append(episode)
        index["episodes"].append({k: episode[k] for k in ("source_key", "control_steps", "termination_reason", "wall_seconds")})
        print(json.dumps(index["episodes"][-1], ensure_ascii=False), flush=True)
        if len(pending) >= cfg["prepare"]["shard_episodes"]:
            flush()
    if pending:
        flush()
    adapter.bundle.assert_unchanged(frozen)
    index.update(wall_seconds=time.perf_counter()-started, frozen_unchanged=True)
    (output/"index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2))
    return index
