"""一次性物理请求预处理与持久化，训练只随机读取结果。

输入是连续Newton实测episode和原配对数据索引；对每个公共网格请求，仅从当时真实历史和
已提交旧计划运行冻结SM/WM前瞻，保存58维条件、预测状态序列、源时间和前缀身份。实测
未来仅保留为独立执行起点/验证标签，不参与前瞻。紧凑偏移表避免为每个anchor重复整曲。
支持将既有v1真实缓存离线转换到新v2目录，不再次仿真、不改旧资产、不在训练时回填。
"""

from pathlib import Path
import json
import subprocess
import time

import torch

from gem.closedloop.stage1_dataset import collate_stage1_training_samples
from gem.closedloop.training import batch_to_device
from gem.closedloop.mb_control.contracts import ReferenceExhausted
from gem.closedloop.mb_control.forecast import forecast_rollout
from gem.robots.bumi.endecoder import BumiEndecoder
from .config import ROOT
from .contracts import PhysicsRejected, SIDECAR_SCHEMA, digest
from .data import SourceCatalog, SidecarStore
from .reference import make_window


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


@torch.no_grad()
def cache_requests(episode, dataset, row, adapter, codec):
    started = time.perf_counter()
    frames = torch.arange(0, episode["source_frames"], 3)
    n = len(frames)
    values = {"frames": frames, "prefix_frames": torch.zeros(n, dtype=torch.long),
        "forecast_ticks": torch.zeros(n, dtype=torch.long), "features": torch.zeros(n, 58),
        "valid": torch.zeros(n, dtype=torch.bool), "reasons": [], "state_offsets": [0]}
    states = []
    device = adapter.runtime.kin.device
    for i, frame in enumerate(frames.tolist()):
        sample = batch_to_device(collate_stage1_training_samples([dataset.get_window(row, frame)]), device)
        values["prefix_frames"][i] = sample["prefix_frames"][0].cpu()
        trace = []
        try:
            window, _ = make_window(episode, sample, adapter, codec, 1)
            prediction = forecast_rollout(window.request, adapter, window.start.tick,
                                           on_step=lambda state: trace.append(state.state.cpu()))
            if not trace:
                trace.append(prediction.predicted.state.cpu())
            values["features"][i] = prediction.context.features[0].cpu()
            values["forecast_ticks"][i] = prediction.context.forecast_tick
            values["valid"][i] = True
            values["reasons"].append("available")
            states.extend(trace)
        except (PhysicsRejected, ReferenceExhausted, FloatingPointError) as error:
            values["reasons"].append(getattr(error, "reason", "offline_forecast_incomplete_or_nonfinite"))
        values["state_offsets"].append(values["state_offsets"][-1] + (len(trace) if values["valid"][i] else 0))
    values["predicted_states"] = torch.cat(states) if states else torch.empty(0, 55)
    values["state_offsets"] = torch.tensor(values["state_offsets"], dtype=torch.long)
    values["origin"] = "offline_frozen_old_plan_forecast_not_future_measurement"
    values["wall_seconds"] = time.perf_counter()-started
    episode["offline_requests"] = values
    return {"source": episode["source_key"], "request_count": n,
            "valid_requests": int(values["valid"].sum()), "wall_seconds": values["wall_seconds"]}


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
