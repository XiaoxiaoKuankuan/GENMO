"""独立GPU物理采集进程与正式训练之间的有界、同步版本数据交接。

每个进程独占一个Newton环境、GENMO实例和冻结同源SM，不共享机器人历史或调度。
主进程先原子发布当前Actor权重，子进程加载后继续其真实episode并采集；主进程可
同时推进自己的物理环境。全部生成线程收束、数据返回后，主进程才允许Actor更新。
IPC仅保存紧凑真实数据和独立RNG，不保存物理求解器；恢复始终新episode。每环境的
Critic回报递推必须独立，不能跨进程拼成一条轨迹。子进程不创建任何优化器；父进程
结束后退出，异常只报告，不自动重启质量失败，也不访问其他训练进程。
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

import torch

from supermimic.bridge.bundle import atomic_save
from .application import seed_all, task_inputs
from .checkpoint import capture_rng, restore_rng, pack_value, unpack_value


def publish(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False))
    os.replace(temporary, path)


def publish_tensor(payload, destination):
    """仅用于已消费IPC邮箱和当前采集权重；不用于编号训练checkpoint。

    SM的atomic_save故意禁止覆盖有效编号产物，通信邮箱则需要完整替换上一轮。
    先写同目录临时文件并fsync，最后一次os.replace发布，读取方不会看到半份数据。
    """
    destination = Path(destination)
    descriptor, temporary = tempfile.mkstemp(prefix=".ipc-", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def run_collector_service(args, cfg, collector):
    directory = Path(args.rpc_directory)
    if not directory.is_dir() or args.parent_pid is None:
        raise ValueError("物理采集服务必须由明确的父进程和独立IPC目录启动")
    collector.collector_id = args.collector_id
    collector.anchor_horizon = cfg["imagination"]["horizon_steps"]
    collector.record_controls = True
    collector.record_contacts = True
    if not isinstance(collector.anchor_horizon, int):
        raise ValueError("并行物理采集需要已解析的真实验证短时域")
    frozen = collector.adapter.bundle.frozen_snapshot()
    seed_all(cfg["training"]["seed"]+100000*args.collector_id)
    publish(directory/"ready.json", {"pid": os.getpid(), "collector_id": args.collector_id,
        "physical_environments": collector.env.N, "frozen_sha256": collector.adapter.bundle.metadata["sha256"]})
    last_request, actor_version = -1, -1
    try:
        while os.getppid() == args.parent_pid:
            request_path = directory/"request.json"
            if not request_path.is_file():
                time.sleep(.05)
                continue
            request = json.loads(request_path.read_text())
            if request["id"] == last_request:
                time.sleep(.05)
                continue
            last_request = request["id"]
            if request["command"] == "stop":
                return
            if request["command"] != "collect" or request["actor_version"] < actor_version:
                raise ValueError("非法物理采集命令或Actor版本回退")
            if request["actor_version"] != actor_version:
                weights = torch.load(request["weights"], map_location="cpu", weights_only=False)
                if (weights["schema"] != "genmo.collection.actor.v1" or weights["actor_version"] != request["actor_version"] or
                        weights["frozen_sha256"] != collector.adapter.bundle.metadata["sha256"]):
                    raise ValueError("采集Actor与当前版本或冻结资产不一致")
                collector.actor.load_state_dict(weights["actor"], strict=True)
                actor_version = request["actor_version"]
            if request.get("rng_path"):
                restore_rng(torch.load(request["rng_path"], map_location="cpu", weights_only=False))
            if collector.state is None or collector.terminal:
                if not request.get("task"):
                    raise ValueError("独立物理episode缺少显式任务")
                music, initial = task_inputs(request["task"], collector.adapter, args.torch_device)
                collector.reset(music, initial)
            elif request.get("task") is not None:
                raise ValueError("不能在尚未终止的真实episode中途更换任务")
            collector.actor_version = actor_version
            started = time.perf_counter()
            collector.collect(request["decisions"])
            controls = collector.take_controls()
            anchors = tuple(collector.ready_anchors)
            collector.ready_anchors.clear()
            if collector.worker.future is not None:
                raise RuntimeError("物理采集返回前必须收束全部生成")
            collector.adapter.bundle.assert_unchanged(frozen)
            memo = {}
            payload = {"id": last_request, "actor_version": actor_version, "collector_id": args.collector_id,
                "anchors": pack_value(anchors, memo), "controls": pack_value(tuple(controls), memo),
                "rng": capture_rng(), "terminal": collector.terminal, "tick": collector.state.tick,
                "physical_steps": len(controls), "seconds": time.perf_counter()-started,
                "started_monotonic": started, "ended_monotonic": time.perf_counter(),
                "frozen_unchanged": True, "worker_future_finished": True}
            publish_tensor(payload, directory/"response.pt")
            publish(directory/"response.json", {"id": last_request, "actor_version": actor_version,
                "collector_id": args.collector_id, "real_controls": len(controls), "real_anchors": len(anchors),
                "terminal": collector.terminal})
    except BaseException as error:
        publish(directory/"failure.json", {"request_id": last_request, "error": repr(error), "traceback": traceback.format_exc()})
        raise


class RemotePhysicalCollector:
    """父进程侧：只协调自己的子进程，不修改或停止任何外部训练。"""

    def __init__(self, cfg, config_path, directory, gpu, collector_id, *, restored_rng=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.collector_id, self.needs_task = collector_id, True
        self.sequence, self.request_id = 0, None
        self.rng = restored_rng
        self.restore_pending = restored_rng is not None
        self.timeout = cfg.get("execution", {}).get("collector_timeout_seconds", 600)
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE="1")
        command = [sys.executable, "-u", "tools/collect_stage2_supermimic_worker.py", "--config", str(config_path),
            "--output", str(self.directory/"runtime"), "--rpc-directory", str(self.directory),
            "--parent-pid", str(os.getpid()), "--collector-id", str(collector_id)]
        self.log = (self.directory/"worker.log").open("w")
        self.process = subprocess.Popen(command, env=environment, stdout=self.log, stderr=subprocess.STDOUT)
        publish(self.directory/"launch.json", {"pid": self.process.pid, "gpu": gpu, "command": command})

    def _wait(self, name, predicate=lambda value: True):
        started = time.monotonic()
        while time.monotonic()-started < self.timeout:
            failure = self.directory/"failure.json"
            if failure.is_file():
                raise RuntimeError(f"独立采集器{self.collector_id}失败：{failure.read_text()}")
            if self.process.poll() is not None:
                raise RuntimeError(f"独立采集器{self.collector_id}退出{self.process.returncode}；日志{self.directory/'worker.log'}")
            path = self.directory/name
            if path.is_file():
                value = json.loads(path.read_text())
                if predicate(value):
                    return value
            time.sleep(.05)
        raise TimeoutError(f"独立采集器{self.collector_id}等待{name}超过{self.timeout}秒")

    def wait_ready(self, frozen_sha256):
        ready = self._wait("ready.json")
        if ready["physical_environments"] != 1 or ready["frozen_sha256"] != frozen_sha256:
            raise ValueError("独立物理采集器的真实环境数或冻结模型不匹配")

    def submit(self, weights, actor_version, decisions, task=None):
        if self.request_id is not None:
            raise RuntimeError("上次采集尚未收束，禁止覆盖IPC请求")
        if self.needs_task != (task is not None):
            raise ValueError("新episode任务分配与独立采集器状态不一致")
        self.sequence += 1
        self.request_id = self.sequence
        rng_path = None
        if self.restore_pending:
            rng_path = str(self.directory/"restore_rng.pt")
            atomic_save(self.rng, rng_path)
            self.restore_pending = False
        publish(self.directory/"request.json", {"id": self.sequence, "command": "collect", "weights": str(weights),
            "actor_version": actor_version, "decisions": decisions, "task": task, "rng_path": rng_path})

    def receive(self, adapter, device):
        self._wait("response.json", lambda value: value["id"] == self.request_id)
        response = torch.load(self.directory/"response.pt", map_location="cpu", weights_only=False)
        if response["id"] != self.request_id or not response["frozen_unchanged"] or not response["worker_future_finished"]:
            raise RuntimeError("物理采集响应不完整或被冻结参数改变")
        memo = {}
        response["anchors"] = unpack_value(response["anchors"], adapter, device, memo)
        response["controls"] = unpack_value(response["controls"], adapter, device, memo)
        self.rng, self.needs_task, self.request_id = response["rng"], response["terminal"], None
        return response

    def close(self):
        if self.process.poll() is None:
            publish(self.directory/"request.json", {"id": self.sequence+1, "command": "stop"})
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=20)
        self.log.close()
