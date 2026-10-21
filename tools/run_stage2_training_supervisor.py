"""用户明确授权的Stage2正式八卡训练后台恢复入口，持续运行至累计轮次上限。

数据流：启动原八卡训练脚本，首次可在新目录继承指定完整checkpoint；目录已有
latest时只恢复同一实验。子进程真实输出追加到同一正式日志，每次退出记录退出码、
最新完整checkpoint轮次和时间。意外退出或尚未达到上限就正常结束时，等待30秒后
从最新完整状态重启；不重置Adam，不更改学习率、概率、奖励、任务或任何数值校验。

本工具只能由明确长训授权启动，不停止其他GPU作业，不删除锁文件、证据或模型。
不以ESS、奖励涨跌、新增失败或短训提升作为停止条件；OOM和数值异常仍由原训练器
如实报错并回滚，恢复行为不能掩盖错误或保证物理收益。SIGTERM/SIGINT显式终止
监督器和当前子任务，不在用户要求停止后重新拉起。状态文件可区分后台存活与轮次
真实推进，不能仅凭监督器PID宣称训练正常。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gem.closedloop.dppo.budget import atomic_json


def training_command(config,output,limit,initialization=None):
    command=['bash',str(ROOT/'scripts/train_stage10_8gpu_server1.sh'),'--config',str(config),
        '--output-dir',str(output),'--stop-after-iteration',str(limit)]
    if (output/'latest.json').is_file():command+=['--resume','latest']
    elif initialization is not None:command+=['--initialize-from-checkpoint',str(initialization)]
    return command


def latest_iteration(output):
    path=output/'latest.json'
    return None if not path.is_file() else int(json.loads(path.read_text())['iteration'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--status-dir',type=Path,required=True)
    parser.add_argument('--stop-after-iteration',type=int,default=200)
    parser.add_argument('--initialize-from-checkpoint',type=Path)
    args=parser.parse_args()
    status=args.status_dir.resolve();status.mkdir(parents=True,exist_ok=True)
    output=args.output_dir.resolve();stopping=False;child=None;attempt=0
    def request_stop(signum,frame):
        nonlocal stopping
        stopping=True
        if child is not None and child.poll() is None:child.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    while not stopping:
        iteration=latest_iteration(output)
        if iteration is not None and iteration>=args.stop_after_iteration:
            atomic_json(status/'supervisor.json',dict(status='complete',iteration=iteration,
                completed_at=time.time(),pid=os.getpid(),attempts=attempt))
            return 0
        attempt+=1
        command=training_command(args.config.resolve(),output,args.stop_after_iteration,args.initialize_from_checkpoint)
        started=time.time()
        with (status/'formal.log').open('ab',buffering=0) as log:
            log.write((json.dumps(dict(supervisor_attempt=attempt,command=command,time=started))+'\n').encode())
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            atomic_json(status/'supervisor.json',dict(status='running',pid=os.getpid(),child_pid=child.pid,
                attempt=attempt,started_at=started,iteration_at_start=iteration,command=command,
                requested_iteration_limit=args.stop_after_iteration))
            while child.poll() is None:time.sleep(1.)
        record=dict(attempt=attempt,exit_code=child.returncode,wall_seconds=time.time()-started,
            iteration=latest_iteration(output),ended_at=time.time(),explicit_stop=stopping)
        atomic_json(status/f'formal_attempt_{attempt:03d}.exit.json',record)
        if not stopping:
            for _ in range(30):
                if stopping:break
                time.sleep(1.)
    atomic_json(status/'supervisor.json',dict(status='explicitly_stopped',pid=os.getpid(),
        iteration=latest_iteration(output),stopped_at=time.time(),attempts=attempt))
    return 0


if __name__=='__main__':raise SystemExit(main())
