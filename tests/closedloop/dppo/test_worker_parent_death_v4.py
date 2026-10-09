"""Linux 下训练 rank 突然退出时，独立 GMT worker 必须跟随退出的真实进程回归。

使用生产 Workers.start 和 RPC 协议启动一个明确忽略 SIGTERM 的轻量工作进程，
再 SIGKILL 其父进程，核对 PDEATHSIG 可终止并回收直属子进程。测试不运行物理、
不占用 GPU；只在服务器1执行。临时 subreaper 防止测试孤儿变成未回收的僵尸，
finally 精确回收本测试 PID 并恢复原 subreaper 设置，不操作任何训练任务。
"""
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


def test_worker_health_ignores_headless_warning_but_detects_fatal_gpu_creation(tmp_path):
    from tools.eval.run_closedloop_baseline import fatal_worker_start_error
    path=tmp_path/'worker.log'
    path.write_text('[Warning] [carb.windowing-glfw.plugin] GLFW initialization failed.\n')
    assert fatal_worker_start_error(path) is None
    fatal='[Error] [omni.gpu_foundation_factory.plugin] Failed to create any GPU devices, including compatibility mode.'
    path.write_text('earlier startup data\n'*2000+fatal+'\n')
    assert fatal_worker_start_error(path)==fatal


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux PDEATHSIG contract')
def test_rank_death_kills_worker_even_when_sigterm_is_ignored(tmp_path):
    worker = tmp_path/'worker.py'
    worker.write_text('''"""测试专用RPC worker，显式忽略SIGTERM以模拟Isaac的信号处理。"""
import os, signal, sys
from gem.runtime.closedloop_protocol import RpcServer
signal.signal(signal.SIGTERM, signal.SIG_IGN)
class Handler:
    def hello(self): return dict(pid=os.getpid())
RpcServer(sys.argv[1], Handler()).serve()
''')
    ready = tmp_path/'ready.json'
    code = '''from pathlib import Path
import json, sys, time
from tools.eval.run_closedloop_baseline import Workers
root=Path(sys.argv[1])
w=Workers(dict(runtime=dict(torch_threads=1,worker_start_timeout_s=30,rpc_timeout_s=5)),root)
socket=Path(w.temp.name)/'worker.sock'
client=w.start('gmt',[sys.executable,'-B',str(root/'worker.py'),str(socket)],Path.cwd(),socket,strip_distributed=True)
(root/'ready.json').write_text(json.dumps(dict(pid=w.entries[0]['proc'].pid,temp=w.temp.name)))
time.sleep(120)
'''
    libc = ctypes.CDLL(None)
    previous = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(previous), 0, 0, 0) == 0
    assert libc.prctl(36, 1, 0, 0, 0) == 0
    parent = subprocess.Popen([sys.executable,'-B','-c',code,str(tmp_path)],
                              stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    child, reaped, temporary = None, False, None
    try:
        deadline = time.monotonic()+40
        while not ready.exists() and time.monotonic()<deadline and parent.poll() is None:
            time.sleep(.02)
        assert ready.exists(), 'Test worker did not reach RPC readiness'
        identity = json.loads(ready.read_text())
        child, temporary = identity['pid'], Path(identity['temp'])
        os.kill(child,signal.SIGTERM)
        time.sleep(.1)
        assert Path(f'/proc/{child}').exists()
        parent.kill();parent.wait(timeout=5)
        deadline = time.monotonic()+5
        while time.monotonic()<deadline:
            found,status = os.waitpid(child,os.WNOHANG)
            if found == child:
                reaped = True
                assert os.WIFSIGNALED(status) and os.WTERMSIG(status)==signal.SIGKILL
                break
            time.sleep(.02)
        assert reaped, 'Worker survived the death of its owning rank'
    finally:
        if parent.poll() is None:
            parent.kill();parent.wait(timeout=5)
        if child is not None and not reaped:
            os.kill(child,signal.SIGKILL);os.waitpid(child,0)
        if temporary is not None:
            # 仅删除从本测试的 Workers 返回的已退出实例的 socket/空目录。
            (temporary/'worker.sock').unlink(missing_ok=True)
            temporary.rmdir()
        assert libc.prctl(36,previous.value,0,0,0)==0
