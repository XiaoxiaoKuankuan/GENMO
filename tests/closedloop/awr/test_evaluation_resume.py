"""完整音乐评价的停止、完整任务登记与恢复调度回归。

使用实际评价入口、二进制分片和账本，只把物理世界替换成可控续体：任务执行中请求
停止仍完成该Episode，之后不发新任务；恢复必须保留失败、固定任务索引和slot。
此测试证明工程调度/校验语义，不声称CPU替身等同真实GMT/PhysX或学习效果。
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gem.closedloop.awr.checkpoint import GracefulStop
from gem.closedloop.awr.distributed import Distributed
from gem.closedloop.awr.evaluation_progress import EvaluationLedger, record_attempt_seconds
from gem.closedloop.awr.io import write_tree


def evidence(path, task, *, failed=False, truncated=False):
    episode = dict(task_id=task, transition_valid=True, truncated=truncated,
        physical_failure=failed, end_reason="yaw" if failed else "music_end",
        reward_trace=np.array([1., -5. if failed else 2.]))
    sha = write_tree(path, episode)
    return dict(path=str(path), sha256=sha, task_id=task)


def test_ledger_preserves_failure_and_checks_identity_arrays_and_duplicates(tmp_path):
    identity = dict(actor="fixed", plan="fixed")
    path = tmp_path/"progress.json"
    ledger = EvaluationLedger(path, identity, ["a", "b"])
    ref = evidence(tmp_path/"failed", "a", failed=True)
    ledger.append(ref)
    restored = EvaluationLedger(path, identity, ["a", "b"])
    assert restored.pending([{"task_id": "a"}, {"task_id": "b"}]) == [{"task_id": "b"}]
    with pytest.raises(ValueError, match="Repeated"):
        restored.append(ref)
    with pytest.raises(ValueError, match="identity"):
        EvaluationLedger(path, dict(actor="different", plan="fixed"), ["a", "b"])
    array = next((tmp_path/"failed").glob("*.npy"))
    with array.open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"\xff")
    with pytest.raises(ValueError, match="checksum"):
        EvaluationLedger(path, identity, ["a", "b"])


def test_ledger_rejects_administrative_tail_and_foreign_task(tmp_path):
    ledger = EvaluationLedger(tmp_path/"ledger.json", {}, ["a"])
    with pytest.raises(ValueError, match="complete"):
        ledger.append(evidence(tmp_path/"partial", "a", truncated=True))
    with pytest.raises(ValueError, match="foreign"):
        ledger.append(evidence(tmp_path/"foreign", "other"))
    assert not ledger.references and not ledger.path.exists()


def test_attempt_seconds_keep_previous_cost_and_reject_reuse(tmp_path):
    assert record_attempt_seconds(tmp_path, {}, "first", 3.5) == 3.5
    assert record_attempt_seconds(tmp_path, {}, "second", 2.5) == 6.
    with pytest.raises(ValueError, match="reused"):
        record_attempt_seconds(tmp_path, {}, "first", 100.)
    with pytest.raises(ValueError, match="nonnegative"):
        record_attempt_seconds(tmp_path, {}, "invalid", float("nan"))


def test_evaluation_drains_then_resumes_exact_remaining_tasks(tmp_path, monkeypatch):
    from gem.closedloop.awr import evaluation
    actor = torch.nn.Linear(2, 1)
    d = Distributed()
    policy = SimpleNamespace(actor=actor, kernel_config={"steps": 20})
    fingerprint = d.assert_models_equal(actor)
    stopped = [False]
    calls, worlds, closed = [], [], []
    tasks = [dict(task_id=str(i), dataset="source", sample_id=str(i), seed=7,
                  remaining_music_seconds=30.-i, sample={"group_id": str(i)}) for i in range(4)]
    plan = dict(tasks=tasks, plan_sha256="fixed-plan")
    config = dict(runtime={}, evaluation={"envs_per_rank": 2})

    def open_world(*args, **kwargs):
        worlds.append(kwargs["num_envs"])
        return SimpleNamespace(world=object())

    class Driver:
        def __init__(self, *args, num_envs, **kwargs):
            states = [SimpleNamespace(resource=SimpleNamespace(env=i)) for i in range(num_envs)]
            self.collector = SimpleNamespace(states=states, _drive=self.drive)
        def drive(self, flows):
            # 每个slot交错推进；两个已开始的Episode都必须排空。
            while flows:
                for slot in list(flows):
                    try:
                        next(flows[slot])
                    except StopIteration:
                        del flows[slot]
        def close(self):
            closed.append("driver")

    def flow(env, task, index, *args, **kwargs):
        calls.append((task["task_id"], index, env))
        yield "active_episode"
        if task["task_id"] == "0":
            stopped[0] = True
        episode = dict(task_id=task["task_id"], dataset=task["dataset"], seed=task["seed"],
            transition_valid=True, truncated=False, physical_failure=task["task_id"] == "0",
            end_reason="yaw" if task["task_id"] == "0" else "music_end", reward_sum=2.,
            decisions=[dict(executed_control_steps=2)], generated_reference_diagnostic={},
            motion_diagnostic={})
        return episode, None

    def retire(*args):
        yield "retire"

    monkeypatch.setattr(evaluation, "open_world", open_world)
    monkeypatch.setattr(evaluation, "close_isolated_world", lambda _: closed.append("world"))
    monkeypatch.setattr(evaluation, "CompleteCollector", Driver)
    monkeypatch.setattr(evaluation, "budget_for_tasks", lambda *args: {})
    monkeypatch.setattr(evaluation, "evaluation_episode_flow", flow)
    monkeypatch.setattr(evaluation, "backend_operation", retire)
    monkeypatch.setattr(evaluation, "aggregate_evaluation",
        lambda episodes: dict(overall={}, by_source={"source": {"reward_per_executed_second": 1.}}))
    args = (config, policy, None, None, plan)
    kwargs = dict(fingerprint=fingerprint, physics_identity="frozen",
                  progress_path=tmp_path/"ledger", should_stop=lambda: stopped[0])
    before = torch.get_rng_state().clone()
    with pytest.raises(GracefulStop):
        evaluation.evaluate(*args, tmp_path/"attempt1", d, **kwargs)
    assert calls == [("0", 0, 0), ("1", 1, 1)]
    interrupted = json.loads((tmp_path/"attempt1/interrupted.json").read_text())
    assert not interrupted["eligible_for_checkpoint_selection"]
    assert set(interrupted["completed_task_ids"]) == {"0", "1"}
    assert not (tmp_path/"attempt1/report.json").exists()
    stopped[0] = False
    report = evaluation.evaluate(*args, tmp_path/"attempt2", d, **kwargs)
    # 未完成任务2原属于slot1、任务3原属于slot0，不能恢复时重新排队。
    assert calls == [("0", 0, 0), ("1", 1, 1), ("3", 3, 0), ("2", 2, 1)]
    assert report["failure_reasons"] == {"yaw": 1}
    assert report["recovered_complete_tasks"] == 2
    assert len(report["task_summaries"]) == 4 and closed == ["driver", "world"]*2
    torch.testing.assert_close(before, torch.get_rng_state(), atol=0, rtol=0)
    # 已全部登记时发布报告只读取证据，不再创建物理世界。
    again = evaluation.evaluate(*args, tmp_path/"attempt3", d, **kwargs)
    assert len(worlds) == 2 and again["recovered_complete_tasks"] == 4


def test_manual_evaluation_preserves_completed_training_and_stable_resume_key():
    from gem.closedloop.awr.runtime import AWRRun
    run = AWRRun.__new__(AWRRun)
    run.state = dict(status="complete", phase="complete", finished_unix=123.)
    run.d, run.actor = Distributed(), torch.nn.Linear(2, 1)
    run.fixed_plan = run.full_plan = {"plan_sha256": "fixed"}
    run.publish = lambda **kwargs: run.state.update(kwargs)
    run.prepare = lambda: run.publish(status="running", phase="data_audit")
    keys = []
    def evaluate(name, plan):
        keys.append(name)
        if len(keys) == 1:
            raise GracefulStop()
        return {"complete": True}
    run.eval_actor = evaluate
    assert run.manual_evaluate() is None
    assert run.state["status"] == "complete" and run.state["manual_evaluation_status"] == "stopped"
    assert run.manual_evaluate() == {"complete": True}
    assert keys[0] == keys[1]
    assert run.state["status"] == run.state["phase"] == run.state["manual_evaluation_status"] == "complete"
    assert run.state["finished_unix"] == 123.
    run.replay = None
    run.prepare = lambda: pytest.fail("Completed training must not restart prepare/physical evaluation")
    assert run.run() == 0 and run.state["finished_unix"] == 123.
