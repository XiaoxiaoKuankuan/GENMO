"""一次性物理请求预处理与持久化，训练只随机读取结果。

输入是连续Newton实测episode和原配对数据索引；对每个公共网格请求，仅从当时真实历史和
已提交旧计划运行冻结SM/WM前瞻，保存58维条件、预测状态序列、源时间和前缀身份。实测
未来仅保留为独立执行起点/验证标签，不参与前瞻。紧凑偏移表避免为每个anchor重复整曲。
支持将既有v1真实缓存离线转换到新v2目录，不再次仿真、不改旧资产、不在训练时回填。
批量入口按相同前瞻步数和起始边界分组，将不同歌曲/请求平移至相对时钟，共享一次SM/WM
批量前向；每行的参考、真实历史、原始绝对时刻和版本仍完全独立。CPU只装配静态请求
元数据，不再为请求构造完整Stage1标签/FK；GPU状态轨迹按批一次返回CPU。
"""

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import json
import subprocess
import time

import torch

from gem.closedloop.mb_control.contracts import ClosedLoopState, ReferenceExhausted
from gem.closedloop.mb_control.forecast import forecast_rollout
from gem.closedloop.mb_control.reference_buffer import ReferenceBuffer
from gem.robots.bumi.endecoder import BumiEndecoder
from .config import ROOT
from .contracts import PhysicsRejected, SIDECAR_SCHEMA, digest
from .data import SourceCatalog, SidecarStore


def preprocessing_codec(cfg):
    options = dict(cfg["stage1"]["endecoder"])
    options["kinematics_path"] = str(ROOT/options["kinematics_path"])
    return BumiEndecoder(**options).codec


def source_provenance(cfg):
    result = {}
    for name, path in (("genmo", ROOT), ("supermimic", cfg["supermimic"]["repo"])):
        value = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
        result[name] = value.stdout.strip() if value.returncode == 0 else "unavailable"
    return result


@lru_cache(maxsize=4096)
def _request_geometry(prefix, at_source_start, visible_upper_relative):
    """仅在CPU复用权威ReferenceBuffer时间支持公式，不读取姿态或未来实测。"""
    request_tick = 0 if at_source_start else 60
    upper = int(visible_upper_relative)
    if upper < 20:
        raise ReferenceExhausted("旧计划不足两个源点")
    ticks = torch.arange(upper // 20 + 1) * 20
    pose = torch.zeros(1, len(ticks), 28)
    pose[..., 3] = 1.
    temporal = ReferenceBuffer(ticks, pose, torch.zeros(1, len(ticks), 2),
                               torch.zeros_like(ticks), None, tuple(range(21)))
    mutable = request_tick + int(prefix)*20
    protection = int(temporal.dependency_support(request_tick-12).max()) if request_tick else -20
    if mutable <= protection:
        raise PhysicsRejected("protected_reference_support")
    first = temporal.first_influence(request_tick, mutable)
    if request_tick+(int(prefix)-1)*20 > upper:
        raise ReferenceExhausted("已知前缀不在请求时的已提交范围内")
    # 最后状态构造58维条件仍需要完整的未来goal和速度halo，不能隐式补尾。
    if first+132 > upper:
        raise ReferenceExhausted("前瞻终点的旧参考或速度halo不完整")
    return request_tick, first, ((first+132+19)//20)*20


@dataclass(frozen=True)
class _OfflineRequest:
    episode_index: int
    index: int
    frame: int
    prefix: int
    base_tick: int
    relative_request_tick: int
    relative_forecast_tick: int
    reference_upper: int


def _request_metadata(episode, dataset, row, episode_index):
    frames = torch.arange(0, int(episode["source_frames"]), 3)
    n = len(frames)
    values = {"frames": frames, "prefix_frames": torch.zeros(n, dtype=torch.long),
        "request_ticks": frames*20, "forecast_ticks": torch.zeros(n, dtype=torch.long),
        "features": torch.zeros(n, 58), "valid": torch.zeros(n, dtype=torch.bool),
        "reference_versions": torch.full((n,), -1, dtype=torch.long),
        "visible_uppers": torch.full((n,), -1, dtype=torch.long),
        "committed_ticks": torch.full((n,), -1, dtype=torch.long),
        "reasons": ["unprocessed"]*n}
    requests = []
    sample_id = str(dataset.rows[row]["sample_id"])
    if sample_id != episode["sample_id"] or dataset.split != episode["split"]:
        raise ValueError("离线请求的数据来源或split不匹配")
    if int(episode.get("frame_origin_tick", 0)) != 0:
        raise ValueError("离线请求只接受明确的原source零起点时间轴")
    for index, frame in enumerate(frames.tolist()):
        future_count = min(int(dataset.motion_frames), int(episode["source_frames"])-frame)
        prefix = min(dataset._requested_prefix_frames(sample_id, frame), max(future_count-1, 0))
        values["prefix_frames"][index] = prefix
        tick, state_index = frame*20, frame*20//12
        try:
            if state_index >= len(episode["state"]):
                raise PhysicsRejected("missing_real_time")
            if prefix < 1:
                raise PhysicsRejected("insufficient_known_prefix")
            upper = int(episode["visible_upper"][state_index])
            committed = int(episode["committed_tick"][state_index])
            version = int(episode["plan_version"][state_index])
            values["reference_versions"][index] = version
            values["visible_uppers"][index] = upper
            values["committed_ticks"][index] = committed
            if committed > tick:
                raise PhysicsRejected("future_plan_leak")
            if episode["split"] not in ("train", "val", "test"):
                raise PhysicsRejected("unknown_split")
            if upper % 20 or upper >= len(episode["reference_qpos"])*20:
                raise ValueError("旧计划可见范围不是合法源时刻")
            base = 0 if tick == 0 else tick-60
            relative, forecast, reference_upper = _request_geometry(prefix, tick == 0, upper-base)
            if base+forecast >= len(episode["state"])*12:
                raise PhysicsRejected("missing_real_time")
            if base+forecast+12 > (len(episode["state"])-1)*12:
                raise PhysicsRejected("incomplete_real_window")
            requests.append(_OfflineRequest(episode_index, index, frame, prefix, base,
                                             relative, forecast, reference_upper))
        except (PhysicsRejected, ReferenceExhausted) as error:
            values["reasons"][index] = getattr(error, "reason", "offline_forecast_incomplete_or_nonfinite")
    return values, requests


def _stack_request_state(requests, episodes, adapter):
    """每批只搬运请求时及以前的真实状态和当时可见的旧计划；不读取未来真实状态值。"""
    device = adapter.runtime.kin.device
    fields = defaultdict(list)
    first = requests[0]
    for request in requests:
        episode = episodes[request.episode_index]
        i = request.frame*20//12
        start = request.base_tick//20
        end = start+request.reference_upper//20+1
        fields["qpos"].append(episode["reference_qpos"][start:end])
        fields["state"].append(episode["state"][i])
        fields["xi"].append(episode["xi"])
        fields["previous_action"].append(episode["previous_action"][i])
        fields["sm_history"].append(episode["observation"][torch.arange(i-9, i+1).clamp_min(0)])
        history_indices = torch.arange(i-49, i+1)
        valid = history_indices >= 0
        fields["genmo_history"].append(torch.where(valid[:, None], episode["proprio"][history_indices.clamp_min(0)], 0.))
        fields["genmo_valid"].append(valid)
    tensors = {name: torch.stack(items).to(device) for name, items in fields.items()}
    ticks = torch.arange(first.reference_upper//20+1, device=device)*20
    # 此内部缓存处于相对批量时钟，无计划提交操作；零版本只标记计算用不可变切片。
    # 每个真实请求的绝对tick/旧版本/已提交范围在offline_requests独立持久化。
    reference = ReferenceBuffer(ticks, tensors["qpos"], tensors["qpos"].new_zeros((len(requests), len(ticks), 2)),
                                torch.zeros_like(ticks), adapter.runtime.kin, adapter.to_sm)
    return ClosedLoopState(first.relative_request_tick, tensors["state"], tensors["xi"],
        tensors["previous_action"], tensors["sm_history"], tensors["genmo_history"],
        tensors["genmo_valid"], reference, tensors["state"].new_zeros((len(requests), 1, 35)),
        0, first.reference_upper, history_origin="real")


@torch.no_grad()
def cache_requests_batched(episodes_with_dataset_row, adapter, codec=None, *, batch_size=512):
    """跨歌曲真实批量前瞻，输入为[(episode,dataset,row),...]，就地补齐v2并返回逐曲统计。"""
    if int(batch_size) < 1:
        raise ValueError("前瞻batch_size必须为正整数")
    started = time.perf_counter()
    entries = list(episodes_with_dataset_row)
    episodes = [entry[0] for entry in entries]
    results, traces, groups = [], [], defaultdict(list)
    for episode_index, (episode, dataset, row) in enumerate(entries):
        values, requests = _request_metadata(episode, dataset, row, episode_index)
        results.append(values)
        traces.append({})
        for request in requests:
            groups[(request.relative_request_tick, request.relative_forecast_tick, request.reference_upper)].append(request)
    batches, transition_batches, maximum_batch = 0, 0, 0
    for (_, forecast_tick, _), requests in groups.items():
        for offset in range(0, len(requests), int(batch_size)):
            chunk = requests[offset:offset+int(batch_size)]
            state = _stack_request_state(chunk, episodes, adapter)
            trace = []
            prediction = forecast_rollout(state, adapter, forecast_tick,
                on_step=lambda value: trace.append(value.state), reuse_current_observation=True,
                allow_nonfinite_context=True)
            if not trace:
                trace.append(prediction.predicted.state)
            states = torch.stack(trace, 1)
            features = prediction.context.features
            finite = torch.isfinite(states).all(dim=(1, 2)) & torch.isfinite(features).all(dim=1)
            # 每个完整批次一次搬回轨迹和特征，不在控制步内逐请求.cpu()/同步。
            cpu_states, cpu_features, cpu_finite = states.cpu(), features.cpu(), finite.cpu()
            for position, request in enumerate(chunk):
                values = results[request.episode_index]
                if bool(cpu_finite[position]):
                    values["valid"][request.index] = True
                    values["features"][request.index] = cpu_features[position]
                    values["forecast_ticks"][request.index] = request.base_tick+forecast_tick
                    values["reasons"][request.index] = "available"
                    traces[request.episode_index][request.index] = cpu_states[position].clone()
                else:
                    values["reasons"][request.index] = "offline_forecast_incomplete_or_nonfinite"
            batches += 1
            transition_batches += prediction.control_steps
            maximum_batch = max(maximum_batch, len(chunk))
    elapsed = time.perf_counter()-started
    total_valid = sum(int(value["valid"].sum()) for value in results)
    device = torch.device(adapter.runtime.kin.device)
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    summaries = []
    for episode, values, trace in zip(episodes, results, traces):
        offsets, ordered = [0], []
        for index in range(len(values["frames"])):
            item = trace.get(index)
            if item is not None:
                ordered.append(item)
            offsets.append(offsets[-1]+(len(item) if item is not None else 0))
        values["predicted_states"] = torch.cat(ordered) if ordered else torch.empty(0, 55)
        values["state_offsets"] = torch.tensor(offsets, dtype=torch.long)
        values.update(origin="offline_frozen_old_plan_forecast_not_future_measurement",
                      wall_seconds=elapsed, forecast_backend="batched_relative_clock_v1")
        episode["offline_requests"] = values
        summaries.append({"source": episode["source_key"], "request_count": len(values["frames"]),
            "valid_requests": int(values["valid"].sum()), "wall_seconds": elapsed,
            "batch_shared_wall_seconds": elapsed, "batch_sources": len(episodes),
            "batch_valid_requests": total_valid, "valid_requests_per_second": total_valid/max(elapsed, 1e-9),
            "forecast_batches": batches, "wm_transition_batches": transition_batches,
            "maximum_batch_size": maximum_batch, "peak_cuda_memory_bytes": peak_memory})
    return summaries


def cache_requests(episode, dataset, row, adapter, codec=None, *, batch_size=512):
    """保持单曲调用签名；内部对多个请求批量执行，不再逐请求调用WM。"""
    return cache_requests_batched([(episode, dataset, row)], adapter, codec, batch_size=batch_size)[0]


def prepare_from_sidecar(cfg, split, raw, output, adapter):
    from .prepare_rollouts import asset_contract
    catalog = SourceCatalog(cfg, split)
    store = SidecarStore(raw, split, asset_contract(cfg, adapter), allow_raw=True)
    if store.index["schema"] == SIDECAR_SCHEMA:
        raise ValueError("该缓存已完成一次性预处理，训练直接读取；不要重复生成")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    index = {"schema": SIDECAR_SCHEMA, "split": split, "assets": store.index["assets"],
        "dataset": catalog.identity, "sources": {}, "shards": {}, "episodes": [],
        "sample_contract": cfg["data"]["sample_contract"], "source_versions": source_provenance(cfg),
        "raw_index_sha256": store.identity, "raw_source": str(Path(raw).resolve())}
    codec, pending = preprocessing_codec(cfg), []
    frozen, started = adapter.bundle.frozen_snapshot(), time.perf_counter()
    def flush():
        name = f"shard_{len(index['shards']):05d}.pt"
        with (output/name).open("xb") as file:
            torch.save(pending, file)
        index["shards"][name] = digest(output/name)
        for i, episode in enumerate(pending):
            index["sources"][episode["source_key"]] = {"shard": name, "episode": i}
        pending.clear()
    for dataset, row in catalog.sources():
        if cfg["prepare"]["max_sources"] is not None and len(index["episodes"]) >= cfg["prepare"]["max_sources"]:
            break
        metadata = dataset.get_window(row, 0)["meta"]
        episode = dict(store.episode(metadata))
        info = cache_requests(episode, dataset, row, adapter, codec)
        pending.append(episode)
        index["episodes"].append(info)
        print(json.dumps(info, ensure_ascii=False), flush=True)
        if len(pending) >= cfg["prepare"]["shard_episodes"]:
            flush()
    if pending:
        flush()
    adapter.bundle.assert_unchanged(frozen)
    index.update(wall_seconds=time.perf_counter()-started, frozen_unchanged=True,
                 physics_replayed=False, training_does_not_preprocess=True)
    (output/"index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2))
    return index
