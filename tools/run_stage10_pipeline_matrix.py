"""服务器1顺序执行显式选择的Stage10有限对照，不启动正式或长期训练。

每个案例使用独立不存在的输出目录，严格串行调用既有八卡验收入口；每次启动
前要求正好八张GPU且无计算进程，发现其他作业立即停止，不杀作业。三个K方案
按累计4096真实上层转移比较：1024档四轮、2048档两轮；物理步骤、实际更新、BC
与边界等待仍按真实记录，不声称不同rollout规模的优化轨迹等价。

环境与optimizer minibatch案例各两轮。所有任务沿用编译FP32、原学习率/随机核/
部署profile及完整32任务初始/结束验证，KL拒绝保留原失败并停止该案例，不自动
降学习率或重试。后续独立案例只有GPU再次空闲才能开始。报告保留每个真实返回码、
开始结束时间和墙钟，不因剩余案例成功覆盖先前失败。源码工作区不得在运行中修改。
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
CASES={
    'env32':('stage10_2048_env32_decisions8.yaml',2,()),
    'env16':('stage10_2048_env16_decisions16.yaml',2,()),
    'optimizer1024':('stage10_2048_optimizer1024.yaml',2,()),
    'optimizer512':('stage10_2048_optimizer512.yaml',2,()),
    'full1024':('stage10_8gpu_server1_scale1024.yaml',4,()),
    'uniform1024':('stage10_8gpu_server1_scale1024.yaml',4,('--denoising-samples','8','--sampling-strategy','uniform')),
    'stratified1024':('stage10_8gpu_server1_scale1024.yaml',4,('--denoising-samples','8','--sampling-strategy','stratified8')),
    'full2048':('stage10_8gpu_server1_scale2048.yaml',2,()),
    'uniform2048':('stage10_8gpu_server1_scale2048.yaml',2,('--denoising-samples','8','--sampling-strategy','uniform')),
    'stratified2048':('stage10_8gpu_server1_scale2048.yaml',2,('--denoising-samples','8','--sampling-strategy','stratified8')),
}


def idle_eight():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory',
        '--format=csv,noheader'],text=True).strip()
    if len(rows)!=8 or processes:raise RuntimeError(f'Eight idle GPUs required; count={len(rows)}, processes={processes}')
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',nargs='+',choices=tuple(CASES),required=True)
    parser.add_argument('--gmt-repo',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if len(set(args.cases))!=len(args.cases):raise ValueError('Duplicate finite cases are not allowed')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    report=dict(schema='stage10.finite_pipeline_matrix.v1',cases=[],status='running')
    def save():
        temporary=output/'report.tmp'
        with temporary.open('w') as stream:
            json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,output/'report.json')
    save()
    try:
        for name in args.cases:
            devices=idle_eight();config,rounds,extra=CASES[name]
            row=dict(case=name,rounds=rounds,devices=devices,started_utc=datetime.now(timezone.utc).isoformat(),status='running')
            report['cases'].append(row);save();started=time.perf_counter()
            command=[sys.executable,'-B',str(ROOT/'tools/validate_stage10_scale.py'),
                '--config',str(ROOT/'configs/closedloop'/config),'--gmt-repo',str(args.gmt_repo.resolve()),
                '--output',str(output/name),'--precision-mode','fp32_fast','--rounds',str(rounds),
                '--deadline-seconds','3600',*extra]
            with (output/f'{name}.launch.log').open('x') as log:
                completed=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            accepted=output/name/'run/accepted.json'
            count=json.loads(accepted.read_text())['iteration'] if accepted.exists() else 0
            row.update(exit_code=completed.returncode,accepted_rounds=count,
                status='passed' if not completed.returncode and count==rounds else 'failed',
                seconds=time.perf_counter()-started,ended_utc=datetime.now(timezone.utc).isoformat())
            save();print(json.dumps(row,ensure_ascii=False),flush=True)
        report['status']='passed' if all(r['status']=='passed' for r in report['cases']) else 'failed'
    except BaseException as error:
        report.update(status='stopped',error=f'{type(error).__name__}: {error}')
        raise
    finally:save()
    return int(report['status']!='passed')


if __name__=='__main__':raise SystemExit(main())
