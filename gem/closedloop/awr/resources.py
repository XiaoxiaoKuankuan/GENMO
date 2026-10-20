"""管理Replay进程句柄预算并记录独占八卡作业的GPU资源。

各rank在加载Replay之前按完整Episode容量提高自己的文件句柄软限制，保留硬限制，
避免大量只读mmap在512集规模耗尽默认1024句柄；不能满足预算则提前明确拒绝。
主rank在Actor初始化后启动独立采样线程，每秒只读nvidia-smi，将各卡显存和利用率
写入本次尝试的JSONL；结束发布独立摘要。峰值明确为离散采样的整卡显存下界，包含
驱动及物理工作进程，不能冒充PyTorch allocator的精确峰值。查询失败单独留证，
不停止或修改任何进程；恢复会新建尝试文件，不覆盖已保存资源记录。
"""
from __future__ import annotations

import json
from pathlib import Path
import resource
import subprocess
import threading
import time
import uuid

from .io import atomic_json


def prepare_file_handles(episode_capacity):
    """为每个rank常驻的Replay mmap预留句柄，仅提高当前进程软限制。

    每Episode预留32种dtype数组句柄，另留1024个通信/SQLite/加载句柄，最少4096。
    不修改系统或用户硬限制；硬限制不足时在读取任何Episode前显式失败。
    """
    capacity = int(episode_capacity)
    if capacity < 1 or capacity != episode_capacity:
        raise ValueError("Replay episode capacity must be a positive integer")
    required = max(4096, 1024+32*capacity)
    before, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard != resource.RLIM_INFINITY and required > hard:
        raise RuntimeError(f"AWR Replay needs RLIMIT_NOFILE >= {required}; hard limit is {hard}")
    if before != resource.RLIM_INFINITY and before < required:
        resource.setrlimit(resource.RLIMIT_NOFILE, (required, hard))
    after, actual_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if after != resource.RLIM_INFINITY and after < required:
        raise RuntimeError("AWR file handle budget was not applied")
    return dict(episode_capacity=capacity, required_soft=required, previous_soft=before,
                actual_soft=after, hard=actual_hard, hard_limit_changed=False,
                dtype_handles_per_episode_budget=32, other_handle_reserve=1024)


class GPUResourceMonitor:
    def __init__(self, run_dir, *, interval_seconds=1.):
        self.directory = Path(run_dir) / "resources"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.name = "gpu_" + uuid.uuid4().hex[:12]
        self.interval = interval_seconds
        self.stop_event = threading.Event()
        self.thread = None
        self.started = time.time()
        self.peaks, self.utilization, self.counts, self.errors = {}, {}, {}, []

    def sample(self):
        text = subprocess.check_output(["nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits"], text=True, timeout=5)
        rows = []
        for line in text.strip().splitlines():
            index, memory, utilization = [int(v.strip()) for v in line.split(",")]
            rows.append(dict(index=index, memory_used_mib=memory, utilization_percent=utilization))
        if sorted(row["index"] for row in rows) != list(range(8)):
            raise ValueError("Expected eight distinct GPU resource samples")
        return rows

    def _run(self):
        with (self.directory / (self.name + ".jsonl")).open("x") as stream:
            while not self.stop_event.is_set():
                entry = dict(unix=time.time())
                try:
                    entry["gpus"] = self.sample()
                    for row in entry["gpus"]:
                        key = str(row["index"])
                        self.peaks[key] = max(self.peaks.get(key, 0), row["memory_used_mib"])
                        self.utilization[key] = self.utilization.get(key, 0) + row["utilization_percent"]
                        self.counts[key] = self.counts.get(key, 0) + 1
                except Exception as error:
                    entry["error"] = str(error)
                    self.errors.append(dict(unix=entry["unix"], error=str(error)))
                stream.write(json.dumps(entry) + "\n")
                stream.flush()
                self.stop_event.wait(self.interval)

    def start(self):
        self.thread = threading.Thread(target=self._run, name="awr_gpu_resources", daemon=True)
        self.thread.start()
        return self

    def close(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=7)
        if self.thread is not None and self.thread.is_alive():
            raise RuntimeError("GPU resource query did not finish within its timeout")
        report = dict(schema="genmo.awr.gpu_resources.v1",
            scope="after_actor_initialization_through_this_run_attempt",
            measurement="sampled_total_device_memory_lower_bound_not_allocator_peak",
            interval_seconds=self.interval, started_unix=self.started, finished_unix=time.time(),
            sampled_peak_memory_mib=self.peaks, sample_counts=self.counts,
            mean_utilization_percent={k: v/self.counts[k] for k, v in self.utilization.items()},
            query_errors=self.errors)
        atomic_json(self.directory / (self.name + ".summary.json"), report)
        return report
