"""执行证据的显式有界gzip压缩候选，不改变tar成员或可靠发布协议。

默认python路径继续使用原tarfile+gzip实现；显式pigz路径仅把相同tar字节流送入
已安装的pigz，限制为最多四个压缩线程。输出仍是标准gzip，既有回读、逐成员SHA、
磁盘同步、原子发布及恢复代码保持权威；本模块不删除源文件或发布压缩包。

压缩进程只接受固定参数，不经过shell。管道提供有界背压，标准错误进入临时文件，
避免错误输出堵塞管道。生成tar、压缩退出或收尾超时任一失败都会抛出异常；只清理
本调用拥有的压缩子进程。调用方仍须核验临时包并fsync，才能按原协议发布与回收。
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile


def validate_compression(backend, threads):
    if backend not in ('python', 'pigz'):
        raise ValueError('archive_compression_backend must be python or pigz')
    if type(threads) is not int or not 1 <= threads <= 4:
        raise ValueError('archive_compression_threads must be an integer from 1 to 4')
    if backend == 'python' and threads != 1:
        raise ValueError('Python gzip requires one compression thread')


@contextmanager
def compressed_tar(path, *, level=6, backend='python', threads=1):
    validate_compression(backend, threads)
    if type(level) is not int or not 1 <= level <= 9:
        raise ValueError('Invalid gzip compression level')
    if backend == 'python':
        with tarfile.open(path, 'w:gz', compresslevel=level) as archive:
            yield archive
        return
    executable = shutil.which('pigz')
    if executable is None:
        raise FileNotFoundError('Explicit pigz archive backend is unavailable')
    with open(path, 'wb') as output, tempfile.TemporaryFile() as errors:
        # 新的轻量Python进程先绑定父死亡信号再exec pigz，不在多线程父进程中使用preexec_fn。
        process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()),
            str(os.getpid()), executable, str(level), str(threads)],
            stdin=subprocess.PIPE, stdout=output, stderr=errors)
        try:
            with tarfile.open(fileobj=process.stdin, mode='w|') as archive:
                yield archive
            process.stdin.close()
            code = process.wait(timeout=120)
            if code:
                errors.seek(0)
                detail = errors.read(16384).decode('utf-8', errors='replace')
                raise RuntimeError(f'pigz failed with exit code {code}: {detail}')
            output.flush()
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if not process.stdin.closed:
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass  # 保留原始压缩/生成失败，不被清理时同一个断管覆盖。


def _pigz_entry():
    expected, executable, level, threads = sys.argv[1:]
    validate_compression('pigz', int(threads))
    if not 1 <= int(level) <= 9:
        raise ValueError('Invalid internal compression level')
    if sys.platform.startswith('linux'):
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), 'Cannot bind pigz lifetime to archive worker')
    if os.getppid() != int(expected):
        raise RuntimeError('Archive worker exited before compression lifetime binding')
    os.execv(executable, [executable, '-n', f'-{level}', '-p', threads, '-c'])


if __name__ == '__main__':
    _pigz_entry()
