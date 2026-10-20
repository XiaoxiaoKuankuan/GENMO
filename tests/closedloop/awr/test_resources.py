"""验证Replay mmap的进程句柄预算和分布式错误可见性。

在独立子进程先降低软限制，确认预算恢复后可同时保留超过原限制的真实npy mmap；
硬限制不足必须在读取前拒绝，且不修改父进程/操作系统限制。所有文件由pytest临时
目录管理。另检查同步异常在集合通信之前写入stderr，不依赖真实GPU或物理世界。
"""
import os
import subprocess
import sys

import pytest

from gem.closedloop.awr.distributed import Distributed


def test_replay_mmap_handle_budget(tmp_path):
    script = """
import json, resource, sys
from pathlib import Path
import numpy as np
from gem.closedloop.awr.resources import prepare_file_handles
root=Path(sys.argv[1])
old, hard=resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (64, hard))
report=prepare_file_handles(512)
np.save(root/'array.npy', np.arange(4))
views=[np.load(root/'array.npy', mmap_mode='c') for _ in range(200)]
assert sum(int(v[3]) for v in views)==600
assert report['actual_soft']>=17408 and report['hard']==hard
assert report['previous_soft']==64
print(json.dumps(report))
"""
    result = subprocess.run([sys.executable, "-B", "-c", script, str(tmp_path)],
                            text=True, capture_output=True, env=os.environ.copy())
    assert result.returncode == 0, result.stderr
    assert '"hard_limit_changed": false' in result.stdout


def test_insufficient_hard_limit_fails_before_loading():
    script = """
import resource
from gem.closedloop.awr.resources import prepare_file_handles
resource.setrlimit(resource.RLIMIT_NOFILE, (64,64))
try:
    prepare_file_handles(512)
except RuntimeError as exc:
    assert 'hard limit is 64' in str(exc)
    assert resource.getrlimit(resource.RLIMIT_NOFILE)==(64,64)
else:
    raise AssertionError('Insufficient hard limit accepted')
"""
    result = subprocess.run([sys.executable, "-B", "-c", script],
                            text=True, capture_output=True, env=os.environ.copy())
    assert result.returncode == 0, result.stderr


def test_local_failure_visible_before_collective(capsys):
    d = Distributed()
    def gather(error):
        output = capsys.readouterr().err
        assert "Too many open files" in output and "Traceback" in output
        return [error]
    d.gather = gather
    with pytest.raises(RuntimeError, match="synchronized failure"):
        d.local(lambda: (_ for _ in ()).throw(OSError(24, "Too many open files")))
