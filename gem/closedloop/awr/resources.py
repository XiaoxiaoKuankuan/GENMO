"""记录独占八卡作业中Actor与GMT物理进程共同占用的GPU资源。

主rank在Actor初始化后启动独立采样线程，每秒只读nvidia-smi，将各卡显存和利用率
写入本次尝试的JSONL；结束发布独立摘要。峰值明确为离散采样的整卡显存下界，包含
驱动及物理工作进程，不能冒充PyTorch allocator的精确峰值。查询失败单独留证，
不停止或修改任何进程；恢复会新建尝试文件，不覆盖已保存资源记录。
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import threading
import time
import uuid

from .io import atomic_json


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
