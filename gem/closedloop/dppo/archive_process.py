"""Stage10 v2 已封存执行证据的独立单 CPU 归档进程入口。

本进程通过 stdin/stdout 的逐行 JSON 接收当前运行中一个已封存轮次的目录，
复用 LongRunMaintenance 原有压缩、逐成员 SHA 校验、原子包/清单发布和原件
回收逻辑。不会创建 RunManager、获取或继承训练写锁，不加载训练模型、优化器、
随机状态或物理环境，不进行 GPU 操作；真正的运行配额 reservation 保留在父进程，
这里仅校验路径及实际文件系统剩余空间，避免每轮扫描整个历史运行目录。

结果回传原清单、真实阶段计时、文件账本更新路径和待父进程写入的 metrics。
异常仅回传结构化错误，保留既有归档崩溃窗口供后续幂等恢复。stdin EOF 表示父
调度已经完成 drain，runner 正常退出。Linux 在任何重模块导入前设置父进程死亡
SIGTERM，并复核父 PID，避免设置期间父已退出的竞争及旧子进程与恢复并发写入。
这个文件只由 ArchiveProcessClient 用新的 Python subprocess 启动，不 fork CUDA。
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback
from types import SimpleNamespace


def _follow_parent():
    expected = int(os.environ['GENMO_ARCHIVE_PARENT_PID'])
    if sys.platform.startswith('linux'):
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
            raise OSError(ctypes.get_errno(), 'Cannot bind archive process lifetime to parent')
    if os.getppid() != expected:
        raise RuntimeError('Archive parent exited before lifetime binding')


class LocalArchiveGuard:
    """不持有全run账本或历史扫描；父进程已为整个事务预占全局空间。"""
    def __init__(self, run_dir, min_free_bytes):
        self.run_dir = Path(run_dir).resolve()
        self.min_free_bytes = min_free_bytes
        self.touched = set()

    def _path(self, path):
        path = Path(path)
        if path.is_symlink():
            raise ValueError('Archive evidence cannot be a symlink')
        resolved = path.resolve()
        if not resolved.is_relative_to(self.run_dir):
            raise ValueError('Archive evidence escapes the run directory')
        return resolved

    def check(self, required_bytes=0):
        if shutil.disk_usage(self.run_dir).free-required_bytes < self.min_free_bytes:
            raise RuntimeError('Archive filesystem free space would fall below reserve')

    @contextmanager
    def reservation(self, required_bytes):
        self.check(required_bytes)
        yield

    def account_file(self, path):
        self.touched.add(str(self._path(path)))


def main():
    _follow_parent()
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from gem.closedloop.dppo.long_run import LongRunMaintenance
    for line in sys.stdin:
        request = json.loads(line)
        response = dict(request_id=request.get('request_id'), pid=os.getpid(), status='failed')
        started, timings, metrics, guard = time.perf_counter(), {}, [], None
        try:
            if request.get('operation') != 'archive':
                raise ValueError('Unknown archive process operation')
            guard = LocalArchiveGuard(request['run_dir'], request['min_free_bytes'])
            directory = guard._path(request['directory'])
            maintenance = LongRunMaintenance.__new__(LongRunMaintenance)
            maintenance.guard = guard
            level=request.get('compression_level',6)
            if type(level) is not int or not 1<=level<=9:raise ValueError('Invalid lossless compression level')
            maintenance.compression_level=level
            maintenance.manager = SimpleNamespace(run_dir=guard.run_dir, append_metrics=metrics.append)
            result = maintenance._archive_sealed(directory, timings=timings)
            response.update(status='passed', result=result)
        except BaseException as error:
            response['error'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        response.update(stage_seconds=timings, total_seconds=time.perf_counter()-started,
                        metrics=metrics, accounted_paths=[] if guard is None else sorted(guard.touched))
        print(json.dumps(response, allow_nan=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
