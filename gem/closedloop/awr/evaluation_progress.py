"""完整音乐评价的逐任务恢复账本。

每个rank只登记已经原子保存、自然结束或真实失败的完整评价Episode。恢复时核验
模型/物理/采样器/计划身份、全部数组SHA和任务归属，再跳过已完成任务；不会拼接
物理中间状态，不筛掉失败，也不能用未完成评价选择模型。各rank账本独立原子写入，
不同评价尝试的墙钟另存，正常停止先排空已开始Episode再返回阶段边界。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .io import atomic_json, read_tree


class EvaluationLedger:
    def __init__(self, path, identity, task_ids):
        self.path, self.identity = Path(path), identity
        self.task_ids = set(task_ids)
        self.references = {}
        if self.path.exists():
            value = json.loads(self.path.read_text())
            if (value["schema"] != "genmo.awr.evaluation_progress.v1"
                    or value["identity"] != identity or set(value["task_ids"]) != self.task_ids):
                raise ValueError("Evaluation resume changed Actor/physics/plan/rank identity")
            for reference in value["references"]:
                self._validate(reference)
                self.references[reference["task_id"]] = reference

    def _check_task(self, reference):
        task = reference["task_id"]
        if task not in self.task_ids or task in self.references:
            raise ValueError("Repeated or foreign evaluation task")

    def _validate(self, reference):
        self._check_task(reference)
        folder = Path(reference["path"])
        manifest = json.loads((folder/"manifest.json").read_text())
        if manifest["sha256"] != reference["sha256"]:
            raise ValueError("Completed evaluation shard identity changed")
        episode = read_tree(folder)
        if (episode["task_id"] != reference["task_id"]
                or not episode["transition_valid"] or episode["truncated"]
                or not episode.get("end_reason")):
            raise ValueError("Evaluation resume requires complete valid task evidence")

    def append(self, reference):
        self._validate(reference)
        updated = dict(self.references)
        updated[reference["task_id"]] = dict(reference)
        atomic_json(self.path, dict(schema="genmo.awr.evaluation_progress.v1",
            identity=self.identity, task_ids=sorted(self.task_ids),
            references=list(updated.values())))
        self.references = updated

    def pending(self, tasks):
        return [task for task in tasks if task["task_id"] not in self.references]


def record_attempt_seconds(directory, identity, attempt, seconds):
    """正常结束/停止的尝试成本累加；强制中断的未知阶段时间仍由外层启动账本记录。"""
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Evaluation timing must be finite and nonnegative")
    path = Path(directory)/"attempt_seconds.json"
    value = json.loads(path.read_text()) if path.exists() else dict(identity=identity, attempts={})
    if value["identity"] != identity or attempt in value["attempts"]:
        raise ValueError("Evaluation timing attempt was reused or changed identity")
    value["attempts"][attempt] = float(seconds)
    atomic_json(path, value)
    return sum(value["attempts"].values())
