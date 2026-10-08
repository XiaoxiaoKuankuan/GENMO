"""训练执行证据的单工作线程、有界异步归档调度器。

该模块只调度已经完成封存的 iteration 目录，不读取或修改模型参数、优化器、
训练 Buffer 与物理环境。真正的压缩、逐成员 SHA 校验、原子清单发布和原件回收
由调用方传入的函数完成。容量同时包含正在压缩和等待压缩的任务，默认最多四轮；
队列满时生产者等待，从而把磁盘压力显式传递给训练入口。

同一目录正在排队时不会重复入队。任一任务失败后停止执行后续压缩任务，保留
原目录供恢复重试，并把首个异常交给 submit、drain 或 close 的调用方。close 会
先等待全部已入队任务退出，再终止线程；调用方必须在释放 RunManager 写锁前调用。
"""
from __future__ import annotations

import queue
import threading
from pathlib import Path


class ArchiveWorkerError(RuntimeError):
    """后台异常必须在训练主线程可见，不能静默丢失归档失败。"""


class BoundedArchiveWorker:
    """包含在途任务的严格有界队列；完整 checkpoint 保存不经过此线程。"""
    def __init__(self, archive, *, max_pending=4):
        if type(max_pending) is not int or max_pending < 1:
            raise ValueError('Archive queue capacity must be a positive integer')
        self.archive, self.max_pending = archive, max_pending
        self._queue = queue.Queue(maxsize=max_pending)
        self._credits = threading.BoundedSemaphore(max_pending)
        self._mutex = threading.RLock()
        self._pending = set()
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
