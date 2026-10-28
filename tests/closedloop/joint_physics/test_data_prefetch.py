"""大批CPU装配、整曲缓存和精确恢复的回归测试。

用原Stage1测试夹具生成临时配对数据，逐字段比较串行未缓存路径与多线程缓存预取路径。
断点在下一批已排队时保存，恢复必须从尚未消费的同一窗口继续；另外覆盖诊断改变batch
大小、rank独立采样及原文件发生变化时的合同失败。不运行GPU、SM、WM或Newton，不能
把这些数据装配测试当作BF16或正式八卡数值验收。
"""

import copy
import os

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from gem.closedloop.joint_physics import data
from tests.closedloop.test_stage1_dataset import _write_dataset, KINEMATICS_PATH


@pytest.fixture
def stream_config(tmp_path, monkeypatch):
    root = _write_dataset(tmp_path / "data", length=180)
    dc = OmegaConf.create({"datasets": {"train": {"fixture": {
        "_target_": "gem.closedloop.stage1_dataset.BumiClosedLoopStage1Dataset",
        "root": str(root), "dataset_name": "test_bumi", "split": "train",
        "kinematics_path": str(KINEMATICS_PATH), "motion_frames": 60,
        "prefix_min_frames": 12, "prefix_max_frames": 24, "history_steps": 50,
        "joint_limit_tolerance": .0001}}}, "train_sampling_reference": {"test_bumi": 1.}})
    monkeypatch.setattr(data, "data_config", lambda cfg: dc)
    return {"training": {"seed": 42}, "data": {"loading": {
        "workers": 2, "prefetch_batches": 1, "sequence_cache_mb": 8}}}


def assert_same_batch(first, second):
    assert first.keys() == second.keys()
    for key in first:
        if isinstance(first[key], torch.Tensor):
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
        else:
            assert first[key] == second[key]


def test_prefetch_matches_original_and_restores_unconsumed_batch(stream_config):
    serial_cfg = copy.deepcopy(stream_config)
    serial_cfg["data"]["loading"] = {}
    serial = data.JointDataStream(serial_cfg)
    parallel = data.JointDataStream(stream_config)
    restored = data.JointDataStream(stream_config)
    try:
        global_rng = copy.deepcopy(np.random.get_state())
        for count in (8, 8, 1, 7):
            assert_same_batch(serial.next(count, "cpu"), parallel.next(count, "cpu"))
        state = parallel.state_dict()
        assert state == serial.state_dict()
        assert parallel._pending is not None
        restored.load_state_dict(state)
        for count in (7, 7, 3):
            expected = serial.next(count, "cpu")
            assert_same_batch(expected, parallel.next(count, "cpu"))
            assert_same_batch(expected, restored.next(count, "cpu"))
        assert parallel.state_dict() == restored.state_dict()
        observed_rng = np.random.get_state()
        assert global_rng[0] == observed_rng[0]
        assert np.array_equal(global_rng[1], observed_rng[1])
        assert global_rng[2:] == observed_rng[2:]
    finally:
        serial.close()
        parallel.close()
        restored.close()


def test_rank_independent_stream_and_restore_existing_queue(stream_config):
    rank0 = data.JointDataStream(stream_config, rank=0, world_size=8)
    rank1 = data.JointDataStream(stream_config, rank=1, world_size=8)
    try:
        zero, one = rank0.next(8, "cpu"), rank1.next(8, "cpu")
        assert [m["decision_frame"] for m in zero["meta"]] != [m["decision_frame"] for m in one["meta"]]
        state = rank1.state_dict()
        expected = rank1.next(8, "cpu")
        rank1.load_state_dict(state)
        assert_same_batch(expected, rank1.next(8, "cpu"))
        with pytest.raises(ValueError, match="rank布局"):
            rank0.load_state_dict(state)
    finally:
        rank0.close()
        rank1.close()


def test_cached_source_reuses_reader_and_rejects_changed_files(stream_config, monkeypatch):
    catalog = data.SourceCatalog(stream_config, "train")
    dataset = catalog.datasets[0]
    calls = []
    original = dataset.reader.reader.load_aligned_sequence
    def counted(row):
        calls.append(row["sample_id"])
        return original(row)
    monkeypatch.setattr(dataset.reader.reader, "load_aligned_sequence", counted)
    first = dataset.get_window(0, start_frame=3)
    dataset.get_window(0, start_frame=18)
    assert len(calls) == 1
    assert 0 < dataset.reader.bytes <= dataset.reader.max_bytes
    path = first["meta"]["motion_path"]
    stat = os.stat(path)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    with pytest.raises(ValueError, match="原音乐/动作文件发生改变"):
        dataset.get_window(0, start_frame=30)


def test_cache_budget_is_bounded(stream_config):
    cfg = copy.deepcopy(stream_config)
    cfg["data"]["loading"]["sequence_cache_mb"] = .0001
    dataset = data.SourceCatalog(cfg, "train").datasets[0]
    dataset.get_window(0, start_frame=3)
    assert dataset.reader.bytes <= dataset.reader.max_bytes
    assert not dataset.reader.cache  # 整曲超过预算时仍正确装配，但不持久占用缓存。


def test_window_targets_match_original_full_fk_codec(stream_config, monkeypatch):
    """去掉未使用FK后，首帧、中段及尾部halo不足窗口的每个条件和标签仍逐位相同。"""
    dataset = data.SourceCatalog(stream_config, "train").datasets[0]
    fast = dataset.codec.encode_condition_features
    def original_full_fk(qpos):
        encoded = dataset.codec.encode(qpos)
        return encoded.physical_features, encoded.anchor
    for frame in (0, 3, 75, 177):
        monkeypatch.setattr(dataset.codec, "encode_condition_features", original_full_fk)
        expected = dataset.get_window(0, start_frame=frame)
        monkeypatch.setattr(dataset.codec, "encode_condition_features", fast)
        actual = dataset.get_window(0, start_frame=frame)
        assert_same_batch(expected, actual)
