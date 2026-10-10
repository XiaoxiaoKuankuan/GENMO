"""在服务器1重放十份已封存的真实八卡执行证据，测量无损归档持续吞吐。

源run只读，先逐包/逐成员核验SHA再解包到全新测试目录；不反序列化训练对象，
不修改任何物理字段、浮点字节序或符号零。归档阶段直接复用生产单CPU进程、
四任务有界队列、FULL证据封存、压缩后回读SHA、原子发布及安全回收临时原件。
源归档始终保留，结束再次确认源包SHA，没有新物理采集或Actor更新。

准备数据的解包/写入时间独立报告，随后按明确周期（默认100秒）提交十轮真实
文件，以实际压缩和磁盘速度测试持续能力及背压。测试不是训练提速结果，不能
把后台积压隐去；报告提交等待、队列峰值、完整排空、实际字节和所有错误。
必须与GPU性能基准错开，启动前要求服务器1八卡空闲，磁盘预留100GiB。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import threading
import time

from gem.closedloop.dppo.archive_store import archive_path
from gem.closedloop.dppo.archives import ArchiveProcessClient,BoundedArchiveWorker
from gem.closedloop.dppo.budget import atomic_json
from gem.closedloop.dppo.run_management import file_sha256
from tools.run_stage10_pipeline_matrix import idle_eight
from tools.report_stage10_pipeline import distribution


def prepare(source_run,destination):
    manifests=sorted(source_run.glob('sessions/*/iterations/*/archive_manifest.json'))
    if len(manifests)!=10:raise ValueError('Requires exactly ten completed source iterations')
    records=[];started=time.perf_counter()
    raw=sum(json.loads(p.read_text())['original_size_bytes'] for p in manifests)
    if shutil.disk_usage(destination).free<2*raw+100*1024**3:
        raise OSError('Insufficient bounded replay space plus 100 GiB reserve')
    for number,path in enumerate(manifests,1):
        manifest=json.loads(path.read_text());source=archive_path(source_run,path.parent)
        if file_sha256(source)!=manifest['archive_sha256']:raise ValueError('Source archive SHA mismatch')
        target=destination/'sessions/replay/iterations'/f'{number:06d}'
        target.mkdir(parents=True,exist_ok=False)
        for name in ('summary.json','seal_manifest.json'):shutil.copyfile(path.parent/name,target/name)
        seal=json.loads((target/'seal_manifest.json').read_text())
        if (manifest['members']!=seal['members'] or file_sha256(target/'seal_manifest.json')!=manifest['seal_sha256']
                or file_sha256(target/'summary.json')!=seal['summary_sha256']):raise ValueError('Source seal mismatch')
        members=manifest['members']
        with tarfile.open(source,'r|gz') as archive:
            for member,expected in zip(archive,members,strict=True):
                relative=Path(member.name)
                if (relative.is_absolute() or '..' in relative.parts or not member.isfile()
                        or member.name!=expected['path'] or member.size!=expected['size_bytes']):
                    raise ValueError('Unsafe or inconsistent source member')
                output=target/relative;output.parent.mkdir(parents=True,exist_ok=True)
                digest=hashlib.sha256();count=0
                with archive.extractfile(member) as inp,output.open('xb') as out:
                    while chunk:=inp.read(1024*1024):out.write(chunk);digest.update(chunk);count+=len(chunk)
                    out.flush();os.fsync(out.fileno())
                if count!=expected['size_bytes'] or digest.hexdigest()!=expected['sha256']:
                    raise ValueError('Copied member bytes differ')
        records.append(dict(source=str(source),source_sha256=manifest['archive_sha256'],directory=str(target),
            source_iteration=seal['iteration'],raw_bytes=manifest['original_size_bytes'],member_count=len(members)))
    return records,time.perf_counter()-started


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--level',type=int,choices=range(1,10),required=True)
    parser.add_argument('--producer-seconds',type=float,default=100.)
    args=parser.parse_args()
    if not 90<=args.producer_seconds<=160:raise ValueError('Finite throughput test period must be 90..160 seconds')
    idle_eight();output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    run=output/'run';run.mkdir()
    inputs,preparation=prepare(args.source_run.resolve(strict=True),run)
    report=dict(schema='stage10.real_evidence_archive_replay.v1',status='running',inputs=inputs,
        source_read_only=True,physical_execution_steps=0,optimizer_steps=0,preparation_seconds=preparation,
        producer_seconds=args.producer_seconds,compression_level=args.level,queue_capacity=4,enqueues=[],archives=[])
    atomic_json(output/'report.json',report)
    client=None;mutex=threading.Lock()
    def archive(directory):
        nonlocal client
        if client is None:client=ArchiveProcessClient()
        result=client.archive(directory,run_dir=run,min_free_bytes=100*1024**3,compression_level=args.level)
        with mutex:report['archives'].append(result)
    def close():
        if client is not None:client.close()
    worker=BoundedArchiveWorker(archive,max_pending=4,on_close=close);started=time.perf_counter()
    try:
        for index,record in enumerate(inputs):
            target=started+index*args.producer_seconds
            while time.perf_counter()<target:time.sleep(min(1.,target-time.perf_counter()))
            begin=time.perf_counter();worker.submit(record['directory'])
            report['enqueues'].append(dict(index=index,scheduled_seconds=index*args.producer_seconds,
                ready_seconds=begin-started,submit_wall_seconds=time.perf_counter()-begin,pending=worker.pending_count))
            with mutex:atomic_json(output/'report.json',report)
        drain=time.perf_counter();worker.drain();report['drain_seconds']=time.perf_counter()-drain
        report.update(status='passed',total_production_and_drain_seconds=time.perf_counter()-started,
            peak_pending=worker.peak_pending_count,
            archive_seconds=distribution(r['total_seconds'] for r in report['archives']))
        if len(report['archives'])!=10:raise RuntimeError('Incomplete archive replay')
        for record,result in zip(inputs,report['archives'],strict=True):
            if result['result']['original_size_bytes']!=record['raw_bytes']:raise ValueError('Archive workload changed')
            if file_sha256(record['source'])!=record['source_sha256']:raise ValueError('Source archive changed')
        report['source_sha_rechecked']=True
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        try:worker.close()
        finally:atomic_json(output/'report.json',report)
    print(json.dumps({k:report[k] for k in ('status','peak_pending','archive_seconds','drain_seconds')}))


if __name__=='__main__':main()
