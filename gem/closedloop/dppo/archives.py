"""训练执行证据的有界调度与独立单后台归档进程。

该模块只调度已经完成封存的 iteration 目录，不读取或修改模型参数、优化器、
训练 Buffer 与物理环境。真正的压缩、逐成员 SHA 校验、原子清单发布和原件回收
由调用方传入的函数完成，正式 v2 回调通过 ArchiveProcessClient 交给单独 Python
子进程执行。子进程由 subprocess 新启动，不继承 CUDA 上下文或运行锁文件描述符，
清除 CUDA 可见设备及 torchrun 通信环境，只交换小型 JSON 请求和结果。
容量同时包含正在压缩和等待压缩的任务，默认最多四轮；
队列满时生产者等待，从而把磁盘压力显式传递给训练入口。

同一目录正在排队时不会重复入队。任一任务失败后停止执行后续压缩任务，保留
原目录供恢复重试，并把首个异常交给 submit、drain 或 close 的调用方。close 会
先等待全部已入队任务退出，再终止线程；调用方必须在释放 RunManager 写锁前调用。
"""
from __future__ import annotations

import queue
import json
import math
import os
import select
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


class ArchiveWorkerError(RuntimeError):
    """后台异常必须在训练主线程可见，不能静默丢失归档失败。"""


class ArchiveProcessClient:
    """一个调度线程独占一个CPU子进程；协议错误/进程死亡均停止后续任务。"""
    def __init__(self, *, response_timeout_s=3600.):
        if not math.isfinite(response_timeout_s) or response_timeout_s <= 0:
            raise ValueError('Archive process response timeout must be positive and finite')
        environment = os.environ.copy()
        for key in list(environment):
            if (key in {'RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'GROUP_RANK',
                        'ROLE_RANK', 'ROLE_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'}
                    or key.startswith(('TORCHELASTIC_', 'NCCL_'))):
                environment.pop(key)
        environment.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
            GENMO_ARCHIVE_PARENT_PID=str(os.getpid()))
        self._stderr = tempfile.TemporaryFile(mode='w+b')
        self._process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).with_name('archive_process.py'))],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr, bufsize=0,
            env=environment, close_fds=True, start_new_session=True)
        self._closed, self._request_id = False, 0
        self.response_timeout_s, self._response_buffer = response_timeout_s, b''
        self._stop_mutex, self._cancelled = threading.RLock(), threading.Event()

    @property
    def pid(self):
        return self._process.pid

    def _failure(self, message):
        # 必须先确认子进程停止，调用方才能回收空间预留并释放运行写锁。
        self.abort()
        self._stderr.flush()
        self._stderr.seek(0, os.SEEK_END)
        self._stderr.seek(max(0, self._stderr.tell()-4096))
        detail = self._stderr.read().decode('utf-8', errors='replace')
        return ArchiveWorkerError(f'{message}; archive process pid={self.pid}, '
                                  f'exitcode={self._process.poll()}; stderr={detail}')

    def abort(self):
        """异常/外部截止路径可跨线程取消；返回前确认进程退出并完成wait回收。"""
        self._cancelled.set()
        with self._stop_mutex:
            if self._process.poll() is None:
                self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()

    def archive(self, directory, *, run_dir, min_free_bytes, compression_level=6):
        if self._closed:
            raise RuntimeError('Archive process is closed')
        self._request_id += 1
        request = dict(operation='archive', request_id=self._request_id, directory=str(directory),
                       run_dir=str(run_dir), min_free_bytes=min_free_bytes,compression_level=compression_level)
        try:
            pending = memoryview((json.dumps(request, allow_nan=False)+'\n').encode())
            while pending:
                written = self._process.stdin.write(pending)
                if not written:
                    raise self._failure('Archive process request pipe stopped accepting bytes')
                pending = pending[written:]
            self._process.stdin.flush()
            deadline = time.monotonic()+self.response_timeout_s
            while b'\n' not in self._response_buffer:
                if self._cancelled.is_set():
                    raise self._failure('Archive process was cancelled')
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise self._failure('Archive process response timed out; stop for idempotent recovery')
                ready, _, _ = select.select([self._process.stdout], [], [], min(.1, remaining))
                if ready:
                    block = os.read(self._process.stdout.fileno(), 65536)
                    if not block:
                        raise self._failure('Archive process exited without a complete response')
                    self._response_buffer += block
            line, _, self._response_buffer = self._response_buffer.partition(b'\n')
        except (BrokenPipeError, OSError) as error:
            raise self._failure('Archive process IPC failed') from error
        if not line:
            raise self._failure('Archive process exited without a response')
        try:
            response = json.loads(line)
        except ValueError as error:
            raise self._failure('Archive process returned malformed JSON') from error
        if (not isinstance(response, dict) or response.get('request_id') != self._request_id
                or response.get('pid') != self.pid):
            raise self._failure('Archive process response identity differs')
        if response.get('status') != 'passed':
            self.abort()
            failure = ArchiveWorkerError(f"Archive subprocess failed: {response.get('error')}")
            failure.response = response
            raise failure
        return response

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            # EOF让空闲runner正常退出；close只能在调度线程drain/join以后调用。
            try:
                self._process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=5)
        finally:
            self._process.stdout.close()
            self._stderr.close()


class BoundedArchiveWorker:
    """包含在途任务的严格有界队列；完整 checkpoint 保存不经过此线程。"""
    def __init__(self, archive, *, max_pending=4, on_close=None):
        if type(max_pending) is not int or max_pending < 1:
            raise ValueError('Archive queue capacity must be a positive integer')
        self.archive, self.max_pending = archive, max_pending
        self.on_close = on_close
        self._queue = queue.Queue(maxsize=max_pending)
        self._credits = threading.BoundedSemaphore(max_pending)
        self._mutex = threading.RLock()
        self._pending = set()
        self._peak_pending_count = 0
        self._failure = None
        self._closed = False
        self._thread = threading.Thread(target=self._run, name='genmo-execution-archive', daemon=False)
        self._thread.start()

    def _raise_failure(self):
        with self._mutex:
            failure = self._failure
        if failure is not None:
            raise ArchiveWorkerError('Execution archive worker failed; sealed originals are retained') from failure

    @property
    def pending_count(self):
        with self._mutex:
            return len(self._pending)

    @property
    def peak_pending_count(self):
        with self._mutex:return self._peak_pending_count

    def submit(self, directory):
        directory = Path(directory).resolve()
        self._raise_failure()
        with self._mutex:
            if self._closed:
                raise RuntimeError('Archive worker is closed')
            if directory in self._pending:
                return False
        while not self._credits.acquire(timeout=0.05):
            self._raise_failure()
            with self._mutex:
                if self._closed:
                    raise RuntimeError('Archive worker is closed')
        submitted = False
        try:
            self._raise_failure()
            with self._mutex:
                if self._closed:
                    raise RuntimeError('Archive worker is closed')
                if directory in self._pending:
                    return False
                self._pending.add(directory)
                self._peak_pending_count=max(self._peak_pending_count,len(self._pending))
                self._queue.put_nowait(directory)
                submitted = True
                return True
        finally:
            if not submitted:
                self._credits.release()

    def _run(self):
        while True:
            directory = self._queue.get()
            try:
                if directory is None:
                    # Linux PDEATHSIG也跟随创建者线程退出；先由该线程回收子进程。
                    try:
                        if self.on_close is not None:
                            self.on_close()
                    except BaseException as error:
                        with self._mutex:
                            self._failure = self._failure or error
                    return
                with self._mutex:
                    failed = self._failure is not None
                if not failed:
                    try:
                        self.archive(directory)
                    except BaseException as error:
                        with self._mutex:
                            self._failure = self._failure or error
            finally:
                if directory is not None:
                    with self._mutex:
                        self._pending.remove(directory)
                    self._credits.release()
                self._queue.task_done()

    def drain(self):
        self._queue.join()
        self._raise_failure()

    def close(self):
        with self._mutex:
            if self._closed:
                self._raise_failure()
                return
            self._closed = True
        self._queue.put(None)
        self._queue.join()
        self._thread.join()
        self._raise_failure()
