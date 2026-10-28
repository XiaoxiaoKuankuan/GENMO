"""多环境离线采集的槽位换歌、终止和原子索引合同测试。

只在CPU建立小型真实轨迹格式张量：验证一个时间块内同槽位先后属于不同歌曲时，状态、
动作和接触不会串歌；终止状态可以保留但其后动作不能保存；独立worker分片合并幂等且
拒绝同source冲突。压缩真实接触字典继续适用于原运动类别统计。测试不冒充Newton验收。
"""

import json
import time

import numpy as np
import pytest
import torch

from gem.closedloop.joint_physics.batched_prepare import TraceBlock, finish_episode, merge_worker
from gem.closedloop.joint_physics.prepare_rollouts import publish_index
from gem.closedloop.joint_physics.requests import motion_tags


def _record(source):
    return {"episode": {"source_key": source, "source_frames": 120}, "chunks": {},
            "xi": torch.zeros(25), "reason": "source_or_reference_end", "last_tick": 24,
            "started": time.perf_counter()}


def _values(stamp):
    state = torch.zeros(2, 55)
    state[:, 0], state[:, 3] = torch.tensor([stamp, stamp+100.]), 1.
    return {"state": state, "action": torch.full((2, 21), float(stamp)),
            "contacts": torch.tensor([[1, 1, 0], [2, 1, 1]], dtype=torch.int32),
            "observation": torch.full((2, 75), float(stamp)),
            "proprio": torch.full((2, 48), float(stamp)),
            "previous_action": torch.zeros(2, 21)}


def test_trace_block_reused_slot_and_terminal_state_do_not_mix_sources():
    records = {i: _record(str(i)) for i in range(3)}
    block = TraceBlock(4, 2, torch.device("cpu"))
    block.append([0, 1], _values(0), np.array([True, True]))
    # source0在slot0结束并立即换成source2；source1保留终止state但本步不执行action。
    block.append([2, 1], _values(1), np.array([True, False]))
    block.append([2, -1], _values(2), np.array([True, False]))
    block.flush(records)
    assert records[0]["chunks"]["state"][0][:, 0].tolist() == [0.]
    assert records[1]["chunks"]["state"][0][:, 0].tolist() == [100., 101.]
    assert records[2]["chunks"]["state"][0][:, 0].tolist() == [1., 2.]
    assert records[1]["chunks"]["action"][0].shape == (1, 21)
    assert records[1]["chunks"]["contacts"][0].shape == (1, 3)
    block.append([2, -1], _values(3), np.array([True, False]))
    block.flush(records)
    cfg = {"prepare": {"old_plan_period_ticks": 300, "old_plan_frames": 60}}
    result = finish_episode(records[2], cfg, {"encoding": "world_reduced_int32_v1"})
    assert result["state"][:, 0].tolist() == [1., 2., 3.]
    assert result["control_steps"] == 3
    assert result["contacts"] == [
        {"source": "mujoco_warp_solver_active_contacts", "active_count": 1,
         "left_foot": True, "right_foot": False}]*3
    tags = motion_tags(result["state"], tuple(f"joint_{i}" for i in range(21)), result["contacts"])
    assert "single_support" in tags
    assert "double_support" not in tags


def _index():
    return {"schema": "test", "split": "train", "assets": {"sm": "frozen"},
            "sample_contract": {"prefix": "same"}, "dataset": {"id": "same"},
            "shards": {}, "sources": {}, "episodes": []}


def test_worker_atomic_merge_is_idempotent_and_rejects_source_collision(tmp_path):
    publish_index(tmp_path, _index())
    first = tmp_path/"worker_00"
    second = tmp_path/"worker_01"
    first.mkdir()
    second.mkdir()
    worker = _index()
    worker.update(shards={"one.pt": "hash1"}, sources={"song": {"shard": "one.pt", "episode": 0}},
                  episodes=[{"source_key": "song", "control_steps": 2}])
    assert merge_worker(tmp_path, first, worker) == 1
    assert merge_worker(tmp_path, first, worker) == 0
    before = (tmp_path/"index.json").read_text()
    index = json.loads(before)
    assert index["sources"]["song"]["shard"] == "worker_00/one.pt"
    assert len(index["episodes"]) == 1
    with pytest.raises(ValueError, match="重复来源"):
        merge_worker(tmp_path, second, worker)
    assert (tmp_path/"index.json").read_text() == before
    assert not list(tmp_path.glob("*.partial"))
