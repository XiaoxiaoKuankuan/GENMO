"""验证显式追加采集的预算、不可变原计划与失败留存。

纯CPU使用真实EpisodeReplay的二进制提交、校验和快照，物理驱动由明确的合成
成功/预热失败替代。测试数量预留、失败不删除、行政截断不能自动替换，以及
已封存采集阶段幂等恢复；不将这些测试称为真实PhysX验收。
"""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gem.closedloop.awr.distributed import Distributed
from gem.closedloop.awr.io import atomic_json, read_tree, write_tree
from gem.closedloop.awr.replay import EpisodeReplay


def module():
    path = Path(__file__).resolve().parents[3]/"tools"/"complete_stage2_awr_collection.py"
    spec = importlib.util.spec_from_file_location("complete_stage2_awr_collection", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_attempt_reservations_bound_interrupted_batches():
    tool = module()
    tasks = [dict(task_id=str(i)) for i in range(26)]
    assert len(tool.next_wave(tasks, [], 504, 512, 26)) == 8
    # 8个预留中只有3个完成，另外5个即使中断也占预算，不能重新调度。
    wave = tool.next_wave(tasks, [str(i) for i in range(8)], 507, 512, 26)
    assert [t["task_id"] for t in wave] == [str(i) for i in range(8, 13)]
    assert tool.next_wave(tasks, [str(i) for i in range(26)], 511, 512, 26) == []
    assert tool.next_wave(tasks, [], 512, 512, 26) == []
    with pytest.raises(ValueError, match="Duplicate"):
        tool.next_wave(tasks, ["1", "1"], 510, 512, 26)


def test_original_admin_or_unattempted_tasks_cannot_be_silently_replaced():
    tool = module()
    tasks = [dict(task_id="ok"), dict(task_id="fail")]
    plan = dict(tasks=tasks, plan_sha256=tool.sha(tasks))
    outcomes = [dict(task_id="ok", complete=True, reason="music_end"),
                dict(task_id="fail", complete=False, reason="administrative", physical_failure=False)]
    with pytest.raises(ValueError, match="administrative"):
        tool.initial_coverage(plan, outcomes, {"ok"})
    with pytest.raises(ValueError, match="exactly once"):
        tool.initial_coverage(plan, outcomes[:1], {"ok"})
    outcomes[1].update(reason="physical_warmup_failure", physical_failure=True)
    assert len(tool.initial_coverage(plan, outcomes, {"ok"})) == 2


@pytest.mark.parametrize("limit", (1, 2))
def test_real_replay_seal_preserves_failure_and_counts_extra_attempts(tmp_path, monkeypatch, episode_factory, limit):
    tool = module()
    actor, _, make = episode_factory
    root = tmp_path/"run"
    root.mkdir()
    plan_tasks = [dict(task_id="original_ok"), dict(task_id="original_fail")]
    plan = dict(tasks=plan_tasks, plan_sha256=tool.sha(plan_tasks))
    plan_path = root/"artifacts"/"macro000_tasks.json"
    atomic_json(plan_path, plan)
    original_bytes = plan_path.read_bytes()
    replay = EpisodeReplay(root/"replay", capacity=2, contract={"reward": "test"})
    state = dict(macro=0, request_sha256="request", source_commits={"genmo": "original"},
                 artifacts={"macro000_tasks": str(plan_path)})
    run = SimpleNamespace(root=root, d=Distributed(), state=state,
        config={"runtime": {"num_envs": 2}, "awr": {"actor_version": 0}},
        runtime_config={"awr": {"actor_version": 0}}, actor=actor,
        replay=replay, policy=None, builder=None, contract={"reward": "test"}, catalog=None)
    run.publish = lambda **values: state.update(values)
    run.artifact = lambda key, value: state["artifacts"].update({key: value})
    def artifact(key, factory):
        path = root/"artifacts"/(key+".json")
        value = factory()
        atomic_json(path, value)
        run.artifact(key, str(path))
        return value
    run.json_artifact = artifact
    def recover(macro):
        result = []
        for value in tool.receipts(root, 0):
            replay.register(value)
            result.append(value["task_id"])
        return result
    run._recover_publications = recover
    actor_fingerprint = run.d.assert_models_equal(actor)
    initial = make(episode_id="original_ok")
    for row in initial.rows:
        row.identity["actor_fingerprint"] = actor_fingerprint
    publication = replay.publish(initial)
    publication.update(task_id="original_ok", macro=0, published_unix_ns=1)
    atomic_json(root/"replay"/"publications"/(initial.episode_id+".json"), publication)
    original_execution = root/"macro000"/"collection"/"original"/"rank00"/"execution"
    original_outcomes = [dict(task_id="original_ok", complete=True, reason="music_end"),
                         dict(task_id="original_fail", complete=False,
                              reason="physical_warmup_failure", physical_failure=True)]
    atomic_json(original_execution/"outcomes.json", original_outcomes)
    write_tree(original_execution/"excluded"/"original_fail",
               dict(snapshot={"terminated": True, "truncated": False, "tick": 24, "reason": "test_failure"}))
    extra = [dict(task_id="extra_fail", rank=0), dict(task_id="extra_ok", rank=0)]
    monkeypatch.setattr(tool, "supplemental_candidates", lambda *a: extra[:limit])
    calls = []
    class Collector:
        def __init__(self, *args, **kwargs):
            self.path = args[5]
        def collect(self, tasks, catalog, contract, *, on_episode, should_stop):
            assert len(tasks) == 1
            task = tasks[0]
            calls.append(task["task_id"])
            if task["task_id"] == "extra_fail":
                outcome = dict(task_id="extra_fail", complete=False,
                               reason="physical_warmup_failure", physical_failure=True)
                write_tree(self.path/"excluded"/"extra_fail",
                           dict(snapshot={"terminated": True, "truncated": False, "tick": 24, "reason": "test_failure"}))
            else:
                episode = make(episode_id="extra_ok", group="g1")
                for row in episode.rows:
                    row.identity["actor_fingerprint"] = actor_fingerprint
                outcome = dict(task_id="extra_ok", complete=True, reason="music_end",
                               publication=on_episode(episode))
            atomic_json(self.path/"outcomes.json", [outcome])
            return dict(tasks=[outcome], seconds=1.)
        def close(self):
            pass
    monkeypatch.setattr("gem.closedloop.awr.collection.CompleteCollector", Collector)
    monkeypatch.setattr("gem.closedloop.awr.world.open_world", lambda *a, **kw: SimpleNamespace(world=None))
    monkeypatch.setattr("gem.closedloop.awr.world.close_isolated_world", lambda *a: None)
    monkeypatch.setattr("gem.closedloop.awr.world.budget_for_tasks", lambda *a: None)
    args = SimpleNamespace(max_extra_attempts=limit, authorization="explicit test attempt budget")
    try:
        if limit == 1:
            with pytest.raises(RuntimeError, match="budget exhausted"):
                tool.complete_collection(run, args, {"sha256": "reviewed_tool"})
            assert calls == ["extra_fail"] and "macro000_replay" not in state["artifacts"]
            assert (root/"macro000"/"supplement"/"exhausted.json").exists()
            assert plan_path.read_bytes() == original_bytes
            return
        assert tool.complete_collection(run, args, {"sha256": "reviewed_tool"})
        assert calls == ["extra_fail", "extra_ok"]
        assert plan_path.read_bytes() == original_bytes
        accepted = json.loads(Path(state["artifacts"]["macro000_collection_amendment"]).read_text())
        assert accepted["original_complete"] == 1
        assert accepted["complete_episodes"] == 2
        assert accepted["reserved_extra_attempts"] == accepted["confirmed_extra_attempts"] == 2
        assert accepted["excluded_tasks"] == 2 and len(accepted["failure_evidence"]) == 2
        assert accepted["failed_warmup_seconds"] == .08
        assert accepted["complete_episode_initialization_seconds"] == 2.
        snapshot = replay.load_snapshot(state["artifacts"]["macro000_replay"])
        assert len(snapshot.episode_ids) == 2
        assert tool.complete_collection(run, args, {"sha256": "reviewed_tool"})
        assert calls == ["extra_fail", "extra_ok"]
        # 接续索引不允许坏数组流入训练：生产读取始终再次校验完整二进制。
        episode_path = root/"replay"/"episodes"/snapshot.episode_ids[0]
        array_path = next(episode_path.glob("*.npy"))
        data = bytearray(array_path.read_bytes())
        data[-1] ^= 1
        array_path.write_bytes(data)
        with pytest.raises(ValueError, match="checksum"):
            replay.read(snapshot.episode_ids[0], training=True)
    finally:
        replay.close()
