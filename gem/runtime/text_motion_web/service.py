"""网页任务协调、持久化与工作进程恢复。

HTTP 进程仅管理任务和状态；独立 spawn 进程占用 GPU。锁保证跨 HTTP 请求只接受
一个活动任务，进程退出会标记失败并允许下次重建。每次阶段更新原子保存任务 JSON；
重启从独立任务记录重建历史索引，不自动重跑中断任务。输出根目录文件锁阻止多服务争用。
已结束历史最多保留 60 条，按创建时间淘汰最旧任务及其全部产物；进行中任务不参与清理。
淘汰目录先原子移到独立回收目录再删除，重启继续清理，避免中途退出让旧历史重新出现。
"""

from __future__ import annotations

import copy
import fcntl
import json
import logging
import multiprocessing
import queue
import re
import shutil
import threading
import time
from pathlib import Path
from uuid import uuid4

from .models import ModelRegistry
from .storage import FIXED, HISTORY_LIMIT, TERMINAL, atomic_json, now
from .worker import worker_main


class BusyError(ValueError):
    pass


class JobService:
    def __init__(self, root: Path, *, registry=None, worker_target=worker_main, discover=True):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.file_lock = (self.root / ".service.lock").open("a")
        try:
            fcntl.flock(self.file_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file_lock.close()
            raise RuntimeError("此输出目录已有工作台运行，请使用该服务或选择另一目录") from None
        self.lock = threading.RLock()
        self.jobs = {}
        self.active_id = None
        self.closed = threading.Event()
        self.registry = registry or ModelRegistry(self.root)
        self.context = multiprocessing.get_context("spawn")
        self.worker_target = worker_target
        self.process = self.commands = self.events = None
        self.recovery_errors = []
        self.retention_errors = []
        for record in sorted((self.root / "tasks").glob("*/task.json")):
            try:
                self._task_directory(record.parent.name)
                if record.is_symlink():
                    raise ValueError("任务记录不能是软链接")
                job = json.loads(record.read_text())
                if job["id"] != record.parent.name:
                    raise ValueError("任务 ID 与目录不一致")
                if job["status"] not in TERMINAL:
                    job.update(
                        status="failed",
                        failed_stage=job["status"],
                        error="服务已重启，之前的任务被中断；可复用参数重新生成",
                        completed_at=now(),
                    )
                    atomic_json(record, job)
                self.jobs[job["id"]] = job
            except (KeyError, ValueError, OSError) as exc:
                self.recovery_errors.append(f"{record}: {exc}")
        self._prune_history()
        if discover:
            self.registry.start()
        self.monitor = threading.Thread(
            target=self._monitor, name="motion-task-monitor", daemon=True
        )
        self.monitor.start()

    def _save_history(self):
        atomic_json(self.root / "history.json", {"jobs": list(self.jobs.values())})

    def _save(self, job):
        atomic_json(self.root / "tasks" / job["id"] / "task.json", job)
        if job["status"] in TERMINAL:
            self._prune_history()
        else:
            self._save_history()

    def _task_directory(self, job_id):
        """只信任本站固定 UUID 目录，不按任务 JSON 中的路径字段删除文件。"""
        if not isinstance(job_id, str) or not re.fullmatch(r"[a-f0-9]{32}", job_id):
            raise ValueError("任务 ID 必须是 32 位十六进制字符")
        tasks = self.root / "tasks"
        directory = tasks / job_id
        if tasks.is_symlink() or directory.is_symlink() or directory.resolve().parent != tasks:
            raise ValueError("任务目录不能通过软链接指向其他位置")
        return directory

    def _expired_directory(self):
        directory = self.root / ".expired_tasks"
        if directory.is_symlink() or directory.resolve().parent != self.root:
            raise ValueError("历史回收目录不能指向其他位置")
        return directory

    def _retention_error(self, exc):
        message = f"历史自动清理失败，已保留未清理的文件，下次任务结束或重启时重试：{exc}"
        if message not in self.retention_errors:
            self.retention_errors.append(message)
            logging.getLogger(__name__).warning(message)

    def _clean_expired(self):
        """仅删除已经原子移出的任务；失败时保留目录供下一次重试。"""
        try:
            directory = self._expired_directory()
            entries = list(directory.iterdir()) if directory.exists() else []
        except (OSError, ValueError) as exc:
            self._retention_error(exc)
            return
        for entry in entries:
            try:
                if not re.fullmatch(r"[a-f0-9]{32}", entry.name) or entry.is_symlink():
                    raise ValueError(f"拒绝清理非本站回收目录：{entry}")
                # shutil.rmtree 不跟随子目录中的软链接，外部文件不会被递归删除。
                shutil.rmtree(entry)
            except (OSError, ValueError) as exc:
                self._retention_error(exc)

    def _prune_history(self):
        """新任务结束或服务启动时执行；正常情况下最多保留 60 条已结束记录。"""
        with self.lock:
            self.retention_errors = []
            self._clean_expired()
            finished = sorted(
                (job for job in self.jobs.values()
                 if job["status"] in TERMINAL and job["id"] != self.active_id),
                key=lambda job: (job["created_at"], job["id"]), reverse=True,
            )
            for job in finished[HISTORY_LIMIT:]:
                try:
                    source = self._task_directory(job["id"])
                    if source.exists():
                        expired = self._expired_directory()
                        expired.mkdir(exist_ok=True)
                        target = expired / job["id"]
                        if target.exists() or target.is_symlink():
                            raise ValueError(f"回收目录冲突，保留原任务：{target}")
                        source.rename(target)
                    del self.jobs[job["id"]]
                except (OSError, ValueError) as exc:
                    self._retention_error(exc)
            self._save_history()
            self._clean_expired()

    def _start_worker(self):
        if self.process is not None and self.process.is_alive():
            return
        if self.process is not None:
            self.process.join(timeout=0)
            self.commands.close()
            self.events.close()
        self.commands = self.context.Queue()
        self.events = self.context.Queue()
        self.process = self.context.Process(
            target=self.worker_target, args=(self.commands, self.events), name="genmo-web-gpu"
        )
        self.process.start()

    def submit(self, payload):
        if not isinstance(payload, dict) or set(payload) != {
            "model_id",
            "prompt",
            "num_frames",
            "ddim_steps",
        }:
            raise ValueError("生成请求仅接受 model_id、prompt、num_frames、ddim_steps 四个字段")
        if not isinstance(payload["prompt"], str) or not payload["prompt"].strip():
            raise ValueError("Prompt 去除首尾空白后不能为空")
        for key, lower, upper in [("num_frames", 1, 900), ("ddim_steps", 2, 1000)]:
            if type(payload[key]) is not int or not lower <= payload[key] <= upper:
                raise ValueError(f"{key} 必须是 {lower}–{upper} 之间的整数")
        with self.lock:
            if self.active_id or self.closed.is_set():
                raise BusyError("已有任务正在生成，请等待完成后再提交")
        # 校验可能读大文件，不持有任务状态锁，确保轮询及时响应。
        model = self.registry.get(payload["model_id"])
        with self.lock:
            if self.active_id or self.closed.is_set():
                raise BusyError("已有任务正在生成，请等待完成后再提交")
            job_id = uuid4().hex
            job = dict(
                copy.deepcopy(payload),
                id=job_id,
                model=model,
                fixed=FIXED.copy(),
                status="queued",
                created_at=now(),
                started_epoch=time.time(),
                elapsed_seconds=0,
                task_dir=str(self.root / "tasks" / job_id),
            )
            self.jobs[job_id] = job
            self.active_id = job_id
            self._save(job)
            try:
                self._start_worker()
                self.commands.put(copy.deepcopy(job))
            except Exception as exc:
                self._update(
                    dict(id=job_id, status="failed", failed_stage="loading", error=str(exc))
                )
            return self._public(job)

    def _update(self, event):
        with self.lock:
            job = self.jobs.get(event["id"])
            if job is None or job["status"] in TERMINAL:
                return
            job.update(event)
            if job["status"] in TERMINAL:
                job["completed_at"] = now()
                if self.active_id == job["id"]:
                    self.active_id = None
            self._save(job)

    def _monitor(self):
        while not self.closed.wait(0.15):
            with self.lock:
                events, process = self.events, self.process
                if events is None:
                    continue
                try:
                    while True:
                        self._update(events.get_nowait())
                except queue.Empty:
                    pass
                if process is not None and not process.is_alive() and self.active_id:
                    job = self.jobs[self.active_id]
                    self._update(
                        dict(
                            id=self.active_id,
                            status="failed",
                            failed_stage=job["status"],
                            error=f"工作进程已退出（退出码 {process.exitcode}），可重新生成",
                            elapsed_seconds=time.time() - job["started_epoch"],
                        )
                    )

    def _public(self, job):
        result = copy.deepcopy(job)
        if result["status"] not in TERMINAL:
            result["elapsed_seconds"] = time.time() - result["started_epoch"]
        if result["status"] == "done":
            result["video_url"] = f"/api/jobs/{job['id']}/video"
            result["thumbnail_url"] = f"/api/jobs/{job['id']}/thumbnail"
        return result

    def get(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError(job_id)
            return self._public(self.jobs[job_id])

    def history(self):
        with self.lock:
            return {
                "jobs": [
                    self._public(j)
                    for j in sorted(self.jobs.values(), key=lambda j: j["created_at"], reverse=True)
                ],
                "active_id": self.active_id,
                "history_limit": HISTORY_LIMIT,
                "recovery_errors": self.recovery_errors + self.retention_errors,
            }

    def media_path(self, job_id, filename):
        job = self.get(job_id)
        if job["status"] != "done":
            raise KeyError(job_id)
        path = (Path(job["output_dir"]) / filename).resolve()
        task_root = (self.root / "tasks" / job_id).resolve()
        if not path.is_relative_to(task_root) or not path.is_file():
            raise KeyError(job_id)
        return path

    def close(self):
        if self.closed.is_set():
            return
        self.closed.set()
        self.monitor.join(timeout=3)
        self.registry.close()
        with self.lock:
            if self.process is not None:
                if self.process.is_alive():
                    self.commands.put(None)
                    self.process.join(timeout=5)
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(timeout=5)
                self.commands.close()
                self.events.close()
            if self.active_id:
                self._update(
                    dict(
                        id=self.active_id,
                        status="failed",
                        failed_stage=self.jobs[self.active_id]["status"],
                        error="服务已关闭，任务被中断；可复用参数重试",
                    )
                )
            fcntl.flock(self.file_lock, fcntl.LOCK_UN)
            self.file_lock.close()
