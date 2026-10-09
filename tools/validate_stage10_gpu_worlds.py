"""仅服务器1八张空闲GPU上的有限Isaac向量环境验收协调器。

协调器本身不创建CUDA上下文，每张GPU启动一个独立Isaac解释器、一个共享物理场景，
场景内机器人数量由num-envs指定。八个进程使用不同输出目录和USD转换目录，剥离
torchrun环境变量以防Isaac误判分布式设备；所有子进程均在自己的进程组，超时或
任一失败只清理本次创建的组，不影响其他训练。开始前检查八卡空闲，禁止覆盖结果。
测试报告保留各rank的实际退出码、来源commit、GPU占用和物理验收结果；基础验收
不等同于完整DPPO闭环，后续必须进行真实160转移更新、KL及恢复测试。
"""
from __future__ import annotations
import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.train_closedloop_stage10_8gpu import _available_gpus


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--gmt-repo',type=Path,required=True)
    parser.add_argument('--isaac-python',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--num-envs',type=int,default=8)
    parser.add_argument('--deadline-seconds',type=int,default=600)
    parser.add_argument('--benchmark-repeats',type=int,default=0)
    parser.add_argument('--diagnostic-limit',type=int,default=32)
    args=parser.parse_args()
    for name in ('config','gmt_repo','isaac_python','output'):
        setattr(args,name,getattr(args,name).expanduser().resolve())
    devices=_available_gpus()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    entries=[]; started=time.monotonic(); report={}
    try:
        for rank in range(8):
            env=os.environ.copy()
            for key in list(env):
                if key in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','GROUP_RANK','ROLE_RANK',
                           'ROLE_WORLD_SIZE','MASTER_ADDR','MASTER_PORT') or key.startswith('TORCHELASTIC_'):
                    env.pop(key)
            env.pop('CUDA_VISIBLE_DEVICES',None)
            env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONDONTWRITEBYTECODE='1',
                OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                ISAACLAB_PATH='/data0/user/liwei/closedloop_stage8_runtime/IsaacLab')
            env['LD_LIBRARY_PATH']='/data0/user/liwei/closedloop_stage8_runtime/sysroot/usr/lib/x86_64-linux-gnu:'+env.get('LD_LIBRARY_PATH','')
            log=(args.output/f'rank{rank:02d}.log').open('w')
            command=[str(args.isaac_python),'-B',str(args.gmt_repo/'scripts/rsl_rl/check_frozen_vector_world.py'),
                '--config',str(args.config),'--output',str(args.output/f'rank{rank:02d}'),
                '--genmo-repo',str(ROOT),'--rank',str(rank),'--num-envs',str(args.num_envs),
                '--benchmark-repeats',str(args.benchmark_repeats),'--diagnostic-limit',str(args.diagnostic_limit),'--headless']
            owner=os.getpid()
            def parent_guard():
                if ctypes.CDLL(None).prctl(1,signal.SIGKILL)!=0:os._exit(125)
                if os.getppid()!=owner:os._exit(125)
            proc=subprocess.Popen(command,cwd=args.gmt_repo,env=env,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,preexec_fn=parent_guard)
            entries.append((rank,proc,log))
        while any(proc.poll() is None for _,proc,_ in entries):
            failed=[rank for rank,proc,_ in entries if proc.poll() not in (None,0)]
            if failed: raise RuntimeError(f'Isaac validation workers failed: {failed}')
            if time.monotonic()-started>args.deadline_seconds: raise TimeoutError('Finite vector validation deadline')
            time.sleep(.5)
        failed=[rank for rank,proc,_ in entries if proc.returncode!=0]
        if failed: raise RuntimeError(f'Isaac workers failed: {failed}')
        records=[json.loads((args.output/f'rank{rank:02d}/report.json').read_text()) for rank in range(8)]
        if any(record['identity'].get('gpu_uuid')!=devices[rank][1].removeprefix('GPU-') for rank,record in enumerate(records)):
            raise AssertionError('GPU world UUID differs from allocated physical device')
        report=dict(status='passed',ranks=records)
    except BaseException as error:
        report=dict(status='failed',error=dict(type=type(error).__name__,message=str(error)))
        raise
    finally:
        for rank,proc,log in entries:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try: proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=15)
            log.close()
        report.update(num_envs_per_gpu=args.num_envs,devices=devices,seconds=time.monotonic()-started,
            scope='eight_GPU_vector_environment_foundations_not_DPPO',
            exit_codes={str(rank):proc.returncode for rank,proc,_ in entries},
            source_commits={name:subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
                            for name,repo in (('GENMO',ROOT),('GMT',args.gmt_repo))})
        (args.output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__': main()
