"""验证AWR验收汇总对未完成运行及实际源码身份的保真性。

使用独立临时目录构造阶段产物，确认汇总不补造未进行的评价、不更改状态或模型，
并保留真实失败与成对比较的否定结果。仅测试报告语义，不冒充物理或学习验收。
"""
import importlib.util
import json
from pathlib import Path
import threading

import pytest


def load_tool(name):
    module_path = Path(__file__).resolve().parents[3] / "tools" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_independent_audit_uses_rewards_not_stored_mc_labels(episode_factory):
    import numpy as np
    module = load_tool("audit_stage2_awr_replay")
    _, _, make = episode_factory
    episode = make(lengths=(2, 0), rewards=((1., 2.), ()), event=-5.)
    episode.returns.fill_(9999.)
    gamma = .99 ** (1/25)
    np.testing.assert_allclose(module.independent_mc(episode.rows),
                               [1+2*gamma-5*gamma**2, -5.], rtol=1e-14, atol=1e-14)


def test_partial_report_preserves_failures_and_run_identity(tmp_path):
    module = load_tool("report_stage2_awr")
    compare = tmp_path / "comparison.json"
    compare.write_text(json.dumps(dict(balanced_return_delta=5., acceptable=False,
        improvement_supported=False, new_paired_failures=["failed_task"], paired=[{"task_id": "failed_task"}])))
    state = dict(status="running", phase="actor_awr", macro=0, created_unix=0.,
        source_commits={"genmo": "actual_running_commit"},
        artifacts={"macro000_self_distillation_comparison": str(compare)})
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state))
    original = path.read_bytes()
    result = module.summarize(tmp_path)
    assert result["source_commits"] == state["source_commits"]
    assert result["evaluations"] == {} and result["training"] == {}
    assert not result["comparisons"]["macro000_self_distillation_comparison"]["acceptable"]
    assert result["comparisons"]["macro000_self_distillation_comparison"]["new_paired_failures"] == ["failed_task"]
    assert path.read_bytes() == original


def test_resource_monitor_preserves_query_failure_and_sampled_scope(tmp_path, monkeypatch):
    from gem.closedloop.awr.resources import GPUResourceMonitor
    monitor = GPUResourceMonitor(tmp_path, interval_seconds=.001)
    sampled = threading.Event()
    calls = []
    def query():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("test missing device query")
        sampled.set()
        return [dict(index=i, memory_used_mib=100+i, utilization_percent=25) for i in range(8)]
    monkeypatch.setattr(monitor, "sample", query)
    monitor.start()
    try:
        assert sampled.wait(2)
    finally:
        report = monitor.close()
    assert "lower_bound" in report["measurement"]
    assert report["sampled_peak_memory_mib"]["7"] == 107
    assert report["query_errors"][0]["error"] == "test missing device query"
    assert len(list((tmp_path / "resources").glob("*.summary.json"))) == 1


def test_report_preserves_explicit_collection_amendment(tmp_path):
    module = load_tool("report_stage2_awr")
    approval = tmp_path/"amendment.json"
    approval.write_text(json.dumps(dict(schema="genmo.awr.collection_amendment.v1",
        original_tasks=512, reserved_extra_attempts=8, total_confirmed_attempts=520,
        complete_episodes=512, excluded_tasks=8, full_outcomes=[{"task_id": "failed"}])))
    (tmp_path/"state.json").write_text(json.dumps(dict(created_unix=0.,
        artifacts={"macro000_collection_amendment": str(approval)})))
    value = module.summarize(tmp_path)["collection"]["macro000_collection_amendment"]
    assert value["total_confirmed_attempts"] == 520 and value["excluded_tasks"] == 8
    assert value["complete_episodes"] == 512 and "full_outcomes" not in value


def test_report_distinguishes_parallel_generation_cost_from_wall_time(tmp_path):
    module = load_tool("report_stage2_awr")
    path = tmp_path/"collection.json"
    rank = dict(seconds=3., tasks=[dict(complete=True, reason="music_end", states=2,
        actor_samples=2, executed_seconds=1.)], generation_batches=[
        dict(generation_seconds=2., effective_rows=2)], timing={"active_control_steps": 100},
        actual_budget={"used": {"control_steps": 100}})
    path.write_text(json.dumps(dict(seconds=3., ranks=[rank, rank])))
    (tmp_path/"state.json").write_text(json.dumps(dict(created_unix=0.,
        artifacts={"macro000_collection_a": str(path)})))
    value = module.summarize(tmp_path)["collection"]["macro000_collection_a"]
    assert value["seconds"] == 3.
    assert value["generation"]["sum_over_ranks_seconds"] == 4.
    assert value["generation"]["mean_batch_seconds"] == 2.
    assert value["generation"]["effective_rows"] == 4


def test_resource_monitor_rejects_missing_gpu(monkeypatch, tmp_path):
    from gem.closedloop.awr.resources import GPUResourceMonitor
    monkeypatch.setattr("subprocess.check_output", lambda *args, **kwargs: "0, 100, 25\n")
    with pytest.raises(ValueError, match="eight distinct"):
        GPUResourceMonitor(tmp_path).sample()


def test_attempt_costs_keep_failed_work_and_open_attempt_explicit(tmp_path):
    module = load_tool("report_stage2_awr")
    root = tmp_path/"pilot"
    root.mkdir()
    def launch(name, pid, start, end=None, code=None):
        (tmp_path/(name+".launch.json")).write_text(json.dumps(dict(pid=pid, started_unix=start,
            command=["bash", "run_stage2_awr_server2.sh", "resume", "--run-dir", str(root)])))
        if end is not None:
            (tmp_path/(name+".exit.json")).write_text(json.dumps(dict(pid=pid, finished_unix=end, exit_code=code)))
    launch("pilot", 1, 0., 100., 1)
    launch("pilot.resume", 2, 110., 160., 0)
    launch("pilot.check", 3, 170.)
    # 相似前缀但不同运行名不能混入本运行的成本。
    (tmp_path/"pilot_other.launch.json").write_text("{}")
    value = module.attempt_costs(root, now=200.)
    assert value["completed_attempt_seconds"] == 150.
    assert value["all_attempt_seconds_including_open_estimates"] == 180.
    assert value["failed_completed_attempts"] == 1 and value["open_attempts"] == 1
    assert value["reserved_gpu_hours"] == pytest.approx(8*180/3600)
    assert not value["overlapping_attempts"]
    assert next(a for a in value["attempts"] if a["pid"] == 3)["elapsed_is_provisional"]


def test_raw_gpu_samples_keep_interrupted_attempt_and_partial_tail(tmp_path):
    module = load_tool("report_stage2_awr")
    directory = tmp_path/"resources"
    directory.mkdir()
    sample = lambda timestamp, memory: dict(unix=timestamp,
        gpus=[dict(index=0, memory_used_mib=memory, utilization_percent=20)])
    (directory/"gpu_failed.jsonl").write_text(
        json.dumps(sample(1., 100))+"\n"+json.dumps(sample(4., 300))+"\n"+'{"unix":')
    value = module.gpu_sample_audit(tmp_path)[0]
    assert not value["summary_present"]
    assert value["sampled_peak_memory_mib"]["0"] == 300
    assert value["maximum_sample_gap_seconds"] == 3.
    assert value["incomplete_trailing_lines"] == 1
    assert value["sample_counts"]["0"] == 2


def test_attempt_costs_preserve_legacy_exit_without_inventing_pid(tmp_path):
    module = load_tool("report_stage2_awr")
    root = tmp_path/"pilot"
    root.mkdir()
    (tmp_path/"pilot.launch.json").write_text(json.dumps(dict(pid=1, started_unix=0.,
        command=["bash", "run_stage2_awr_server2.sh", "run", "--run-dir", str(root)])))
    exit_path = tmp_path/"pilot.exit.json"
    exit_path.write_text(json.dumps(dict(finished_unix=10., exit_code=1)))
    original = exit_path.read_bytes()
    value = module.attempt_costs(root)["attempts"][0]
    assert value["elapsed_seconds"] == 10. and value["legacy_exit_without_pid"]
    assert not value["exit_pid_verified"] and exit_path.read_bytes() == original
    exit_path.write_text(json.dumps(dict(pid=2, finished_unix=10., exit_code=1)))
    with pytest.raises(ValueError, match="PID differs"):
        module.attempt_costs(root)


def test_weight_concentration_reports_source_mass_without_reweighting():
    module = load_tool("report_stage2_awr")
    statistics = dict(weight=dict(count=4, mean=1., std=.75**.5),
        by_source=dict(A=dict(weight=dict(count=3, mean=.5)),
                       B=dict(weight=dict(count=1, mean=2.5))))
    before = json.dumps(statistics, sort_keys=True)
    value = module.weight_concentration(statistics)
    assert value["effective_sample_size"] == pytest.approx(4/1.75)
    assert value["by_source"]["A"] == dict(sample_fraction=.75, weight_mass_fraction=.375)
    assert value["by_source"]["B"] == dict(sample_fraction=.25, weight_mass_fraction=.625)
    assert not value["weights_changed"] and not value["sampling_policy_changed"]
    assert json.dumps(statistics, sort_keys=True) == before
