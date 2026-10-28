"""增量sidecar生产与训练热读取边界的轻量回归。

使用临时目录和小型模拟episode验证单写者互斥、原子索引发布、四来源交错顺序，以及
断点后只采集新增source、不覆盖已发布分片。这里不构造Newton或声称真实物理通过；
实际冻结资产、真实窗口数和GPU开销由服务器的有界增量生产记录验证。
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gem.closedloop.joint_physics import offline, prepare_rollouts
from gem.closedloop.joint_physics.data import SidecarStore


def test_single_writer_and_atomic_index(tmp_path):
    with prepare_rollouts.preparation_lock(tmp_path):
        with pytest.raises(RuntimeError, match="已有采集进程"):
            with prepare_rollouts.preparation_lock(tmp_path):
                pass
    with prepare_rollouts.preparation_lock(tmp_path):
        prepare_rollouts.publish_index(tmp_path, {"sources": {"first": 1}})
    assert json.loads((tmp_path/"index.json").read_text()) == {"sources": {"first": 1}}
    assert not list(tmp_path.glob("*.partial"))


def test_balanced_order_is_repeatable_and_covers_all_rows():
    catalog = SimpleNamespace(datasets=["a", "b", "c", "d"], rows=[list(range(5))]*4)
    options = {"balanced_sources": True, "source_seed": 42}
    first = list(prepare_rollouts.preparation_sources(catalog, options))
    assert first == list(prepare_rollouts.preparation_sources(catalog, options))
    assert [dataset for dataset, _ in first[:4]] == ["a", "b", "c", "d"]
    assert len(set(first)) == 20
    assert first != list(prepare_rollouts.preparation_sources(catalog, dict(options, source_seed=43)))


def test_incremental_resume_keeps_old_shards_and_skips_existing_sources(tmp_path, monkeypatch):
    datasets = [SimpleNamespace(dataset_name=name, split="train", rows=[{"sample_id": str(i)} for i in range(2)])
                for name in ("a", "b")]
    catalog = SimpleNamespace(datasets=datasets, rows=[[0, 1], [0, 1]], identity={"split": "train"})
    monkeypatch.setattr(prepare_rollouts, "SourceCatalog", lambda *_: catalog)
    monkeypatch.setattr(prepare_rollouts, "asset_contract", lambda *_: {"sm": "frozen"})
    monkeypatch.setattr(offline, "source_provenance", lambda *_: {"test": "fixture"})
    monkeypatch.setattr(offline, "preprocessing_codec", lambda *_: None)
    def cache(episode, *_):
        episode["offline_requests"] = {"frames": torch.empty(0, dtype=torch.long)}
        return {"valid_requests": 0}
    monkeypatch.setattr(offline, "cache_requests", cache)
    collected = []
    def collect(dataset, row, *_):
        key = json.dumps([dataset.dataset_name, "train", str(row)], ensure_ascii=False, separators=(",", ":"))
        collected.append(key)
        return {"source_key": key, "control_steps": 0, "termination_reason": "fixture", "wall_seconds": 0.0}
    monkeypatch.setattr(prepare_rollouts, "collect_source", collect)
    adapter = SimpleNamespace(bundle=SimpleNamespace(frozen_snapshot=lambda: {}, assert_unchanged=lambda _: None))
    cfg = {"data": {"sample_contract": {"prefix": "same"}},
           "prepare": {"resume": False, "balanced_sources": True, "shard_episodes": 1, "max_sources": 2}}
    destination = tmp_path/"train_increment"
    initial = prepare_rollouts.prepare(cfg, "train", destination, adapter, None, initialize_only=True)
    assert len(initial["sources"]) == 0 and not collected
    store = SidecarStore(destination, "train", {"sm": "frozen"})
    cfg["prepare"]["resume"] = True
    first = prepare_rollouts.prepare(cfg, "train", destination, adapter, object())
    hashes = dict(first["shards"])
    assert store.refresh() and len(store.index["sources"]) == 2
    cfg["prepare"]["max_sources"] = 4
    second = prepare_rollouts.prepare(cfg, "train", destination, adapter, object())
    assert len(collected) == len(set(collected)) == 4
    assert len(second["sources"]) == 4
    assert all(second["shards"][name] == sha for name, sha in hashes.items())
    assert store.refresh() and len(store.index["sources"]) == 4
    assert not list(Path(destination).glob("*.partial"))
