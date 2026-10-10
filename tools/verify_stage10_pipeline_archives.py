"""服务器1八rank分担有限训练产物的只读完整审计，不执行新训练或物理采样。

每个run只交给一个rank，沿用正式Stage10审计器复核原归档成员SHA、固定rollout、
逐环境GAE、完整KL、BC/优化器次数、冻结GMT、checkpoint与恢复链。新汇总同时保留
原始转移的episode、终止和截断计数，供环境布局对照使用，不从配置猜测真实执行。

必须八卡空闲、与性能基准错开，入口使用Gloo协调CPU审计并绑定对应GPU；不把这些
只读哈希/数值检查冒充额外GPU推理。每个run的完整结果及失败保留，要求恢复的run
必须显式指定。原模型、日志和证据全部只读；审计器自有临时解包由原上下文清理。
输出目录必须不存在，不覆盖之前的审计结果，任何失败都使入口返回非零。
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time
import traceback

import torch
import torch.distributed as dist

from gem.closedloop.dppo.budget import atomic_json
from tools.run_stage10_pipeline_matrix import idle_eight
from tools.eval.audit_closedloop_stage10 import audit_run
from tools.report_stage10_pipeline import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--require-resume-run', action='append', type=Path, default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if int(os.environ['WORLD_SIZE']) != 8:
        raise ValueError('Requires Server1 eight ranks')
    rank = int(os.environ['LOCAL_RANK'])
    runs = [path.resolve(strict=True) for path in args.runs]
    require_resume = {path.resolve(strict=True) for path in args.require_resume_run}
    if len(set(runs)) != len(runs) or not require_resume.issubset(runs):
        raise ValueError('Duplicate runs or unmatched resume requirement')
    torch.set_num_threads(1)
    dist.init_process_group('gloo', timeout=timedelta(minutes=90))
    status = [None]
    if rank == 0:
        try:
            idle_eight()
            args.output.mkdir(parents=True, exist_ok=False)
        except Exception as error:
            status[0] = f'{type(error).__name__}: {error}'
    dist.broadcast_object_list(status, src=0)
    if status[0]:
        dist.destroy_process_group()
        raise RuntimeError(status[0])
    torch.cuda.set_device(rank)
    local = []
    try:
        for index in range(rank, len(runs), 8):
            run = runs[index]
            started = time.perf_counter()
            print(f'[AUDIT rank={rank}] begin {run}', flush=True)
            try:
                report = audit_run(run, minimum_iterations=2, require_resume=run in require_resume)
            except Exception as error:
                report = dict(status='failed', error=f'{type(error).__name__}: {error}',
                              traceback=traceback.format_exc())
            destination = args.output/f'run_{index:02d}.json'
            atomic_json(destination, report)
            measurements_path = args.output/f'run_{index:02d}_measurements.json'
            measurements_error = None
            try:
                atomic_json(measurements_path, summarize(run))
            except Exception as error:
                measurements_error = f'{type(error).__name__}: {error}'
            record = dict(run=str(run), rank=rank, report=str(destination), status=report['status'],
                require_resume=run in require_resume, seconds=time.perf_counter()-started,
                failed_checks=report.get('failed_checks'),measurements=str(measurements_path),
                measurements_error=measurements_error)
            if measurements_error is not None:
                record['status'] = 'failed'
            local.append(record)
            print(json.dumps(record), flush=True)
        gathered = [None]*8
        dist.all_gather_object(gathered, local)
        records = [row for shard in gathered for row in shard]
        if rank == 0:
            atomic_json(args.output/'report.json', dict(schema='stage10.pipeline_artifact_audit.v1',
                status='passed' if all(row['status']=='passed' for row in records) else 'failed',
                read_only=True, new_physical_steps=0, new_optimizer_steps=0, runs=records))
        return int(any(row['status']!='passed' for row in records))
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    raise SystemExit(main())
