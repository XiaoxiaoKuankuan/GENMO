"""跨宏迭代管理用户明确授权的预热失败补采总额度。

输入为固定运行身份、当前宏迭代和各轮既有补采计划/尝试预留；输出为当前轮可以交给
原补采实现的额度，并原子记录整次运行已消耗的次数。只有主rank调用本模块。
预留即扣次数，包括尚未产生结果的中断尝试；额度不随宏迭代或进程恢复重置。
本模块不生成音乐、不修改Episode、不删除失败、不更改原固定任务配额。实际原计划
覆盖、预热失败快照、行为Actor和新增数据校验仍由原complete_collection实现负责。
"""
from __future__ import annotations

import json
from pathlib import Path

from .io import atomic_json, digest


def _read_budget(root, state, *, total_limit, authorization):
    if type(total_limit) is not int or total_limit < 1 or not authorization.strip():
        raise ValueError("Explicit positive run-wide supplemental budget and authorization required")
    root = Path(root)
    path = root / "collection_budget.json"
    identity = dict(request_sha256=state["request_sha256"],
                    source_commits=state["source_commits"],
                    max_extra_attempts_total=total_limit, authorization=authorization)
    value = (json.loads(path.read_text()) if path.exists()
             else dict(schema="genmo.awr.run_collection_budget.v1", identity=identity, allocations={}))
    if value["identity"] != identity:
        raise ValueError("Run-wide supplemental budget/authorization identity changed")
    allocations = value["allocations"]
    # 不能忽略另一个入口创建的补采计划而重新获得额度。
    for plan_path in root.glob("macro*/supplement/plan.json"):
        macro = str(int(plan_path.parents[1].name.removeprefix("macro")))
        if macro not in allocations:
            raise ValueError("Supplemental plan has no run-wide budget allocation")
    all_reserved = set()
    for key, grant in allocations.items():
        if type(grant["limit"]) is not int or not 0 < grant["limit"] <= total_limit:
            raise ValueError("Invalid supplemental allocation")
        directory = root / f"macro{int(key):03d}" / "supplement"
        plans = directory / "plan.json"
        waves = sorted(directory.glob("wave*.json"))
        if not plans.exists():
            if waves:
                raise ValueError("Supplemental reservations exist without their immutable plan")
            grant["reserved_attempts"] = 0
            continue
        plan = json.loads(plans.read_text())
        if digest(plan["tasks"]) != plan["plan_sha256"]:
            raise ValueError("Supplemental plan checksum changed")
        for name, expected in (("max_extra_attempts", grant["limit"]),
                               ("authorization", authorization),
                               ("request_sha256", state["request_sha256"]),
                               ("source_commits", state["source_commits"])):
            if plan["identity"][name] != expected:
                raise ValueError("Supplemental plan exceeds its run-wide allocation identity")
        candidates = {task["task_id"]: task for task in plan["tasks"]}
        if len(candidates) != len(plan["tasks"]) or len(candidates) != grant["limit"]:
            raise ValueError("Supplemental candidates do not match their allocation")
        reserved = set()
        for index, wave_path in enumerate(waves):
            wave = json.loads(wave_path.read_text())
            ids = wave["task_ids"]
            if (wave["index"] != index or wave["plan_sha256"] != plan["plan_sha256"]
                    or ids != [task["task_id"] for task in wave["tasks"]]
                    or len(ids) != len(set(ids)) or reserved.intersection(ids)
                    or not set(ids) <= set(candidates)
                    or any(task != candidates[task["task_id"]] for task in wave["tasks"])):
                raise ValueError("Duplicate, changed or unplanned supplemental reservation")
            reserved.update(ids)
        if len(reserved) > grant["limit"] or all_reserved.intersection(reserved):
            raise ValueError("Supplemental attempt allocation exceeded or reused across macros")
        all_reserved.update(reserved)
        grant["reserved_attempts"] = len(reserved)
    if len(all_reserved) > total_limit:
        raise ValueError("Run-wide supplemental attempt budget exceeded")
    value.update(reserved_attempts_total=len(all_reserved),
                 remaining_attempts=total_limit-len(all_reserved))
    return path, value


def update_collection_budget(root, state, *, total_limit, authorization):
    """从不可变尝试预留重算已用额度；不根据成功数量退还失败或中断尝试。"""
    path, value = _read_budget(root, state, total_limit=total_limit, authorization=authorization)
    atomic_json(path, value)
    return value


def allocate_collection_budget(root, state, *, total_limit, authorization):
    """为当前轮固定一个不超过全运行剩余额度的上限，恢复沿用同一个上限。"""
    path, value = _read_budget(root, state, total_limit=total_limit, authorization=authorization)
    macro = state["macro"]
    if type(macro) is not int or macro < 0:
        raise ValueError("Invalid AWR macro iteration")
    key = str(macro)
    previous = value["allocations"]
    if any(int(k) > macro for k in previous):
        raise ValueError("Cannot return to a supplemental allocation from an earlier macro")
    for k in previous:
        if int(k) < macro and f"macro{int(k):03d}_collection_amendment" not in state["artifacts"]:
            raise ValueError("Earlier supplemental collection is not sealed")
    if key not in previous:
        if value["remaining_attempts"] == 0:
            atomic_json(path, value)
            raise RuntimeError("Run-wide extra-attempt budget exhausted; all failures retained")
        previous[key] = dict(limit=value["remaining_attempts"], reserved_attempts=0,
                             reserved_before_allocation=value["reserved_attempts_total"])
    atomic_json(path, value)
    return previous[key]["limit"]


def finished_warmup_tasks(root, macro, planned_ids):
    """正常停止恢复不重新执行已经留有真实预热失败结果的原任务。"""
    finished = set()
    for path in sorted((Path(root)/f"macro{macro:03d}"/"collection").rglob("execution/outcomes.json")):
        for outcome in json.loads(path.read_text()):
            if (outcome["task_id"] in planned_ids and not outcome["complete"]
                    and outcome["reason"] == "physical_warmup_failure"
                    and outcome.get("physical_failure") is True):
                if outcome["task_id"] in finished:
                    raise ValueError("Original warmup failure was attempted more than once")
                finished.add(outcome["task_id"])
    return sorted(finished)
