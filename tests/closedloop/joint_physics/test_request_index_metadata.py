"""物理请求索引初始化的轻量metadata路径回归。

在临时目录保存原文件身份和小型真实缓存格式，确保构建请求索引不会调用完整Stage1
get_window；实际随机抽样仍调用原入口，并保持source ID、前缀、RNG及完整恢复状态。
使用原resolve_contract_path检查越界路径，使用原SidecarStore核对原文件SHA；测试不
加载GENMO/SM权重，不构造Newton，也不把CPU索引检查宣称为训练或物理验收。
"""

import json
from types import SimpleNamespace

import pytest
import torch

from gem.closedloop.joint_physics import requests
from gem.closedloop.joint_physics.contracts import SIDECAR_SCHEMA, digest, source_key
from gem.closedloop.joint_physics.data import SidecarStore


def _fixture(tmp_path):
    motion, music = tmp_path/"motion.pt", tmp_path/"music.pt"
    motion.write_bytes(b"unchanged original motion identity")
    music.write_bytes(b"unchanged original music identity")
    # 原Dataset会把manifest sample_id转str；数字fixture防止索引路径遗漏相同规范化。
    row = {"sample_id": 123, "motion_path": motion.name, "music_feature_path": music.name}
    meta = {"dataset_id": "fixture", "split": "train", "sample_id": "123",
            "motion_path": str(motion), "music_feature_path": str(music)}
    key = source_key(meta)
    state = torch.zeros(100, 55)
    state[:, 3] = 1.
    episode = {"source_key": key, "split": "train", "source_frames": 120,
        "state": state, "action": torch.zeros(99, 21),
        "motion_sha256": digest(motion), "music_sha256": digest(music),
        "contacts": [{"left_foot": True, "right_foot": True}]*99,
        "offline_requests": {"frames": torch.tensor([0]), "prefix_frames": torch.tensor([12]),
            "valid": torch.tensor([True]), "forecast_ticks": torch.tensor([108]), "reasons": ["available"]}}
    torch.save([episode], tmp_path/"shard.pt")
    (tmp_path/"index.json").write_text(json.dumps({"schema": SIDECAR_SCHEMA, "split": "train",
        "sources": {key: {"shard": "shard.pt", "episode": 0}}, "shards": {"shard.pt": digest(tmp_path/"shard.pt")}}))
    calls = []
    def get_window(index, frame):
        calls.append((index, frame))
        return {"meta": dict(meta), "prefix_frames": torch.tensor(12)}
    dataset = SimpleNamespace(dataset_name="fixture", split="train", rows=[row],
                              reader=SimpleNamespace(root=tmp_path), get_window=get_window)
    catalog = SimpleNamespace(sources=lambda: iter([(dataset, 0)]), identity={"fixture": "same"})
    store = SidecarStore(tmp_path, "train")
    return catalog, store, dataset, calls, key


def test_index_uses_metadata_and_next_still_builds_original_window(tmp_path, monkeypatch):
    catalog, store, _, calls, key = _fixture(tmp_path)
    names = tuple(f"joint_{i}" for i in range(21))
    stream = requests.PhysicsRequestStream(catalog, store, names, 32, 42, historical_mode="diagnostic")
    assert calls == []
    assert stream.keys == [key]
    assert key in store.verified  # 路径优化不跳过原数据身份检查。
    monkeypatch.setattr(requests, "collate_joint_samples", lambda samples: {"samples": samples})
    monkeypatch.setattr(requests, "batch_to_device", lambda batch, _device: batch)
    before = stream.state_dict()
    batch, records = stream.next(2, "cpu")
    assert calls == [(0, 0), (0, 0)]
    assert len(batch["samples"]) == 2 and records[0]["source"] == key
    assert records[0]["draw"] == 0 and records[1]["draw"] == 1
    restored = requests.PhysicsRequestStream(catalog, store, names, 32, 42, historical_mode="diagnostic")
    assert len(calls) == 2
    restored.load_state_dict(before)
    _, replayed = restored.next(2, "cpu")
    assert replayed == records
    assert restored.state_dict() == stream.state_dict()


def test_metadata_path_keeps_original_root_escape_rejection(tmp_path):
    catalog, store, dataset, calls, _ = _fixture(tmp_path)
    dataset.rows[0]["motion_path"] = "../outside.pt"
    with pytest.raises(ValueError, match="escapes dataset root"):
        requests.PhysicsRequestStream(catalog, store, tuple(f"joint_{i}" for i in range(21)),
                                      32, 42, historical_mode="diagnostic")
    assert calls == []


def test_persistent_request_index_preserves_sampling_and_restore(tmp_path, monkeypatch):
    catalog, store, _, _, _ = _fixture(tmp_path)
    names = tuple(f'joint_{i}' for i in range(21))
    old = requests.PhysicsRequestStream(catalog,store,names,32,42,historical_mode='diagnostic')
    cached = requests.PhysicsRequestStream(catalog,store,names,32,42,historical_mode='diagnostic',cache_directory=tmp_path/'requests')
    monkeypatch.setattr(requests,'collate_joint_samples',lambda samples:{'samples':samples})
    monkeypatch.setattr(requests,'batch_to_device',lambda batch,device:batch)
    assert old.next(3,'cpu')[1] == cached.next(3,'cpu')[1]
    assert old.state_dict() == cached.state_dict()
    state = cached.state_dict()
    monkeypatch.setattr(store,'episode',lambda meta: (_ for _ in ()).throw(AssertionError('不应重复扫描分片')))
    restored = requests.PhysicsRequestStream(catalog,store,names,32,42,historical_mode='diagnostic',cache_directory=tmp_path/'requests')
    restored.load_state_dict(state)
    assert old.next(4,'cpu')[1] == restored.next(4,'cpu')[1]
