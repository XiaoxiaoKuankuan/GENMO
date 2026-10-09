"""服务器1独占八卡、逐规模运行的GPU多环境有限吞吐矩阵。

collection模式调用已验收采集器，每rank始终20条、全局160条真实上层转移，可显式
执行原Critic/DPPO/BC/完整KL；不同N的真实物理步和终止数分别报告，不把时序差异
掩饰为等工作量提速。capacity模式只测冻结GMT/GPU PhysX的真实多机器人容量，
支持512/1024，但不称为160条GENMO训练；固定20条/rank时超过20个环境不会增加
有效采样并行度。必须先完成N8真实训练/恢复验收，再由用户授权的本任务使用本工具。

全部结果位于显式新目录。各规模顺序运行，不重叠占卡；已有任务由下级入口拒绝。
每个子测试有限超时，保留退出码和失败证据；任一失败默认停止矩阵。硬件遥测每秒
读取八卡利用率和显存，单独保存；此进程不加载模型、不改变正式配置、预算或模型。
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('collection','capacity'),required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--gmt-repo',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--environments',type=int,nargs='+',required=True)
    p.add_argument('--data-audit',type=Path)
    p.add_argument('--updates',action='store_true')
    p.add_argument('--rounds',type=int,default=2)
    p.add_argument('--continue-on-failure',action='store_true')
    a=p.parse_args()
    if any(n<1 or n>1024 for n in a.environments) or len(set(a.environments))!=len(a.environments):
        raise ValueError('Explicit distinct environment counts 1..1024 required')
    if a.mode=='collection' and max(a.environments)>32:
        raise ValueError('More than 32 allocated environments is capacity-only for the fixed local20 workload')
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',PYTHONDONTWRITEBYTECODE='1')
    report=dict(scope=a.mode,fixed_global_transitions=160 if a.mode=='collection' else None,
        requested_envs=a.environments,updates=a.updates,cases=[])
    for count in a.environments:
        directory=a.output/f'n{count:04d}'
        if a.mode=='collection':
            command=['bash',str(ROOT/'scripts/validate_stage10_runtime_v4_server1.sh'),'vector-collection',
                '--config',str(a.config),'--gmt-repo',str(a.gmt_repo),'--output',str(directory),
                '--num-envs',str(count),'--rounds',str(a.rounds)]
            if a.data_audit:command+=['--data-audit',str(a.data_audit)]
            if a.updates:command+=['--updates']
        else:
            command=[sys.executable,'-B',str(ROOT/'tools/validate_stage10_gpu_worlds.py'),
                '--config',str(a.config),'--gmt-repo',str(a.gmt_repo),'--output',str(directory),
                '--isaac-python','/data0/user/liwei/closedloop_stage8_runtime/env_isaaclab/bin/python',
                '--num-envs',str(count),'--deadline-seconds','900','--benchmark-repeats','3']
        started=time.perf_counter()
        with (a.output/f'n{count:04d}.log').open('w') as log, (a.output/f'n{count:04d}_gpu.jsonl').open('w') as telemetry:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            try:
                while process.poll() is None:
                    values=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total',
                        '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
                    telemetry.write(json.dumps(dict(seconds=time.perf_counter()-started,csv=values.stdout,
                        exit_code=values.returncode))+'\n');telemetry.flush()
                    time.sleep(1.)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:process.wait(timeout=30)
                    except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
        entry=dict(environments_per_gpu=count,exit_code=process.returncode,wall_seconds=time.perf_counter()-started,
            output=str(directory),log=str(a.output/f'n{count:04d}.log'),
            active_training_env_limit=min(count,20) if a.mode=='collection' else None)
        report['cases'].append(entry)
        (a.output/'matrix.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(entry,ensure_ascii=False),flush=True)
        if process.returncode and not a.continue_on_failure:break
    report['status']='passed' if len(report['cases'])==len(a.environments) and all(c['exit_code']==0 for c in report['cases']) else 'failed'
    (a.output/'matrix.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
