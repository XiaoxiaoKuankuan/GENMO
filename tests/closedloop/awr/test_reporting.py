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


def test_resource_monitor_rejects_missing_gpu(monkeypatch, tmp_path):
    from gem.closedloop.awr.resources import GPUResourceMonitor
    monkeypatch.setattr("subprocess.check_output", lambda *args, **kwargs: "0, 100, 25\n")
    with pytest.raises(ValueError, match="eight distinct"):
        GPUResourceMonitor(tmp_path).sample()
