"""验证正式AWR五轮共享补采额度与恢复边界。

使用真实JSON计划和波次预留，检查跨轮累计、失败/中断不退还、恢复上限不重置、
未封存上一轮和篡改计划拒绝，以及正常停止后原预热失败不被重新派发。
不启动GPU或物理世界；全部产物位于pytest临时目录。另核验正式配置仅扩展已授权
宏迭代/补采预算，保留原Actor、Critic、奖励、评价和默认单轮配置。
"""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gem.closedloop.awr.collection_budget import (
    allocate_collection_budget, update_collection_budget, finished_warmup_tasks)
from gem.closedloop.awr.config import load_config
from gem.closedloop.awr.io import atomic_json, digest
from gem.closedloop.awr.runtime import AWRRun
from gem.closedloop.awr.distributed import Distributed
from tools.launch_stage2_awr_formal5_server2 import validate_formal_budget


def state(macro=0):
    return dict(macro=macro, request_sha256="request", source_commits={"genmo": "fixed"},
                artifacts={})


OPTIONS = dict(total_limit=26, authorization="明确授权：五轮全运行共用26次额外尝试")


def reserve(root, current, count):
    limit = allocate_collection_budget(root, current, **OPTIONS)
    macro = current["macro"]
    path = root/f"macro{macro:03d}"/"supplement"
    tasks = [dict(task_id=f"{macro}_{i}", rank=i % 8) for i in range(limit)]
    identity = dict(max_extra_attempts=limit, authorization=OPTIONS["authorization"],
                    request_sha256=current["request_sha256"], source_commits=current["source_commits"])
    plan = dict(tasks=tasks, plan_sha256=digest(tasks), identity=identity)
    atomic_json(path/"plan.json", plan)
    # 预留并不表示成功，也可以模拟全部中断而仍消耗这些次数。
    atomic_json(path/"wave000.json", dict(index=0, plan_sha256=plan["plan_sha256"],
                                         task_ids=[v["task_id"] for v in tasks[:count]], tasks=tasks[:count]))
    return limit


def test_budget_shared_across_macros_and_resume_does_not_reset(tmp_path):
    current = state()
    assert reserve(tmp_path, current, 8) == 26
    assert allocate_collection_budget(tmp_path, current, **OPTIONS) == 26
    current["artifacts"]["macro000_collection_amendment"] = "sealed"
    current["macro"] = 1
    assert reserve(tmp_path, current, 7) == 18
    report = update_collection_budget(tmp_path, current, **OPTIONS)
    assert report["reserved_attempts_total"] == 15 and report["remaining_attempts"] == 11
    assert allocate_collection_budget(tmp_path, current, **OPTIONS) == 18
    current["artifacts"]["macro001_collection_amendment"] = "sealed"
    current["macro"] = 2
    assert reserve(tmp_path, current, 11) == 11
    current["artifacts"]["macro002_collection_amendment"] = "sealed"
    current["macro"] = 3
    with pytest.raises(RuntimeError, match="exhausted"):
        allocate_collection_budget(tmp_path, current, **OPTIONS)
    report = json.loads((tmp_path/"collection_budget.json").read_text())
    assert report["reserved_attempts_total"] == 26 and report["remaining_attempts"] == 0


def test_unsealed_previous_macro_and_changed_authorization_refused(tmp_path):
    current = state()
    reserve(tmp_path, current, 2)
    current["macro"] = 1
    with pytest.raises(ValueError, match="not sealed"):
        allocate_collection_budget(tmp_path, current, **OPTIONS)
    with pytest.raises(ValueError, match="identity changed"):
        update_collection_budget(tmp_path, current, total_limit=27, authorization=OPTIONS["authorization"])
    with pytest.raises(ValueError, match="identity changed"):
        update_collection_budget(tmp_path, current, total_limit=26, authorization="different approval")


def test_duplicate_or_mutated_wave_cannot_reclaim_attempts(tmp_path):
    current = state()
    reserve(tmp_path, current, 2)
    path = tmp_path/"macro000"/"supplement"/"wave000.json"
    wave = json.loads(path.read_text())
    wave["index"] = 1
    atomic_json(path.with_name("wave001.json"), wave)
    with pytest.raises(ValueError, match="Duplicate"):
        update_collection_budget(tmp_path, current, **OPTIONS)
    path.with_name("wave001.json").unlink()
    wave["index"] = 0
    wave["tasks"][0]["rank"] = 99
    atomic_json(path, wave)
    with pytest.raises(ValueError, match="changed"):
        update_collection_budget(tmp_path, current, **OPTIONS)


def test_external_unaccounted_plan_refused(tmp_path):
    directory = tmp_path/"macro000"/"supplement"
    atomic_json(directory/"plan.json", {"unaccounted": True})
    with pytest.raises(ValueError, match="no run-wide"):
        allocate_collection_budget(tmp_path, state(), **OPTIONS)


def test_warmup_failures_skipped_on_resume_without_counting_as_complete(tmp_path):
    path = tmp_path/"macro000"/"collection"/"attempt"/"rank00"/"execution"/"outcomes.json"
    outcomes = [dict(task_id="warmup", complete=False, reason="physical_warmup_failure", physical_failure=True),
                dict(task_id="admin", complete=False, reason="administrative", physical_failure=False),
                dict(task_id="valid", complete=True, reason="physical_failure", physical_failure=True)]
    atomic_json(path, outcomes)
    assert finished_warmup_tasks(tmp_path, 0, {"warmup", "admin", "valid"}) == ["warmup"]
    atomic_json(path, outcomes+[outcomes[0]])
    with pytest.raises(ValueError, match="more than once"):
        finished_warmup_tasks(tmp_path, 0, {"warmup"})


def test_runtime_passes_shared_budget_and_records_reserved_attempts(tmp_path, monkeypatch):
    from tools import complete_stage2_awr_collection as tool
    current = state()
    run = SimpleNamespace(root=tmp_path, state=current, d=Distributed(),
        config={"awr": {"collection_completion": dict(
            max_extra_attempts_total=26, authorization=OPTIONS["authorization"])}},
        replay=SimpleNamespace(load_snapshot=lambda key: key))
    def complete(actual, args, identity):
        assert actual is run and args.max_extra_attempts == 26
        assert identity["git_commit"] == "fixed" and len(identity["sha256"]) == 64
        reserve(tmp_path, current, 8)
        current["artifacts"]["macro000_replay"] = "sealed_snapshot"
        return True
    monkeypatch.setattr(tool, "complete_collection", complete)
    assert AWRRun.complete_collection_with_budget(run) == "sealed_snapshot"
    assert json.loads((tmp_path/"collection_budget.json").read_text())["reserved_attempts_total"] == 8


def test_formal_config_only_extends_explicit_budgets():
    root = Path(__file__).resolve().parents[3]
    base = load_config(root/"configs/closedloop/stage2_awr.yaml")
    formal = load_config(root/"configs/closedloop/stage2_awr_formal5.yaml")
    assert validate_formal_budget(formal)["max_training_collection_attempts"] == 1050
    with pytest.raises(ValueError, match="five-macro"):
        validate_formal_budget(base)
    normalized = copy.deepcopy(formal)
    normalized["awr"]["max_macro_iterations"] = 1
    del normalized["awr"]["collection_completion"]
    assert normalized == base
