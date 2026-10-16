"""逐个执行高学习率扫描幸存权重的32任务真实Isaac配对评估。

本工具只接收已完成的服务器2固定数据扫描，不改变训练源码、采样核、奖励或门禁。
先评估原始Stage1，再评估所有通过预筛的不同Actor权重；只有整个参数指纹完全相同
的候选才共用一次物理结果，候选别名和各自KL限制均保留。每次启动前检查GPU空闲、
磁盘余量和配置/权重SHA；始终串行启动八卡进程，不能与其他Isaac任务竞争GPU。

所有子任务强制evaluate-only，不提供长期训练入口。物理失败属于有效评估结果，
进程失败或来源身份变化则停止队列，保留日志、退出码和已完成证据。可用--resume
继续已有队列：成功任务先严格检查摘要和输入SHA；失败任务使用新attempt目录，
绝不覆盖原checkpoint、日志或现场。SIGINT/SIGTERM只转发给本工具创建的进程组。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gem.closedloop.dppo.budget import atomic_json


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def evaluation_jobs(scan):
    if scan.get('status') != 'completed_fixed_data_scan_not_physical_acceptance':
        raise ValueError('Only a complete identified fixed-data scan may enter physics')
    candidates = scan['candidates']
    if len(candidates) != 30 or not scan['same_stage1_actor'] or not scan['same_empty_adam']:
        raise ValueError('Expected all 30 comparable scan candidates')
    baseline = next(c for c in candidates if c['name'] == 'coordinate_dppo_lr5e-08_kl1')
    jobs = [dict(name='stage1', config=baseline['path'], config_sha256=baseline['sha256'],
                 actor=None, aliases=['stage1_s595000'])]
    by_fingerprint = {}
    for candidate in candidates:
        if not candidate['screening']['passed']:
            continue
        if not candidate['backtracking']['accepted'] or not candidate['final_acceptance']['passed']:
            raise ValueError('A screened-in candidate lacks complete final KL acceptance')
        actor = candidate['evaluation_actor']
        fingerprint = actor['actor_fingerprint']
        if fingerprint in by_fingerprint:
            by_fingerprint[fingerprint]['aliases'].append(candidate['name'])
            continue
        job = dict(name=candidate['name'], config=candidate['path'], config_sha256=candidate['sha256'],
                   actor=actor, aliases=[candidate['name']])
        by_fingerprint[fingerprint] = job
        jobs.append(job)
    return jobs


def verify_inputs(job):
    if sha(job['config']) != job['config_sha256']:
        raise ValueError('Candidate YAML changed after the scan')
    if job['actor'] and sha(job['actor']['path']) != job['actor']['sha256']:
        raise ValueError('Candidate Actor bytes changed after the scan')


def verify_summary(directory):
    summaries = list(Path(directory).glob('sessions/*/summary.json'))
    if len(summaries) != 1:
        raise ValueError('Expected exactly one complete evaluation session')
    summary = json.loads(summaries[0].read_text())
    if summary.get('status') != 'passed' or summary.get('exit_code') != 0 or not summary.get('source_unchanged'):
        raise ValueError('Physical evaluation did not finish with unchanged source identity')
    report = Path(directory) / 'evaluation_baseline.json'
    if not report.is_file():
        raise ValueError('Missing complete physical evaluation report')
    return dict(summary=str(summaries[0]), summary_sha256=sha(summaries[0]),
                report=str(report), report_sha256=sha(report))


def gpu_monitor(destination, stopped):
    with destination.open('x') as stream:
        while not stopped.is_set():
            result = subprocess.run(['nvidia-smi', '--query-gpu=timestamp,index,memory.used,utilization.gpu',
                '--format=csv,noheader,nounits'], capture_output=True, text=True)
            stream.write(result.stdout)
            stream.flush()
            stopped.wait(5)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    scan = json.loads(args.scan.read_text())
    jobs = evaluation_jobs(scan)
    if not args.execute:
        print(json.dumps(dict(jobs=jobs, executions=len(jobs), evaluation_only=True), ensure_ascii=False, indent=2))
        return 0
    if args.resume:
        record = json.loads((args.output/'queue.json').read_text())
        if record['scan_sha256'] != sha(args.scan) or record['jobs'] != jobs:
            raise ValueError('Cannot resume a queue with different inputs')
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        record = dict(schema='genmo.stage2.uniform_lr_physics_queue.v1', scan=str(args.scan),
            scan_sha256=sha(args.scan), jobs=jobs, attempts=[], status='running',
            deduplication='identical_complete_actor_parameter_fingerprint_only')
        atomic_json(args.output/'queue.json', record)
    active = None
    interrupted = False
    def stop(signum, frame):
        nonlocal interrupted
        interrupted = True
        if active is not None and active.poll() is None:
            os.killpg(active.pid, signal.SIGINT)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    for job in jobs:
        verify_inputs(job)
        previous = [a for a in record['attempts'] if a['name'] == job['name']]
        if previous and previous[-1].get('status') == 'passed':
            if verify_summary(previous[-1]['run']) != previous[-1]['evidence']:
                raise ValueError('Completed physical evidence changed')
            continue
        if interrupted:
            break
        busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
                                        '--format=csv,noheader'], text=True).strip()
        if busy:
            raise RuntimeError('GPU occupied; refusing to launch evaluation: '+busy)
        if shutil.disk_usage(args.output).free < 150*1024**3:
            raise RuntimeError('Insufficient evidence disk reserve; stopping before launch')
        directory = args.output/job['name']/f'attempt{len(previous)+1:02d}'
        directory.mkdir(parents=True, exist_ok=False)
        command = ['bash', str(ROOT/'scripts/train_stage10_8gpu_server1.sh'), '--config', job['config'],
                   '--output-dir', str(directory/'run'), '--evaluate-only', '--stop-after-iteration', '1']
        if job['actor']:
            command += ['--evaluation-actor', job['actor']['path']]
        item = dict(name=job['name'], aliases=job['aliases'], command=command, run=str(directory/'run'),
                    started=time.time(), status='running')
        record['attempts'].append(item)
        atomic_json(args.output/'queue.json', record)
        stopped = threading.Event()
        monitor = threading.Thread(target=gpu_monitor, args=(directory/'gpu.csv', stopped), daemon=True)
        monitor.start()
        try:
            with (directory/'console.log').open('x') as stream:
                active = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                          start_new_session=True)
                item['pid'] = active.pid
                atomic_json(args.output/'queue.json', record)
                code = active.wait()
            item.update(exit_code=code, elapsed_seconds=time.time()-item['started'])
            if code:
                raise RuntimeError(f'Evaluation child exited {code}')
            item['evidence'] = verify_summary(directory/'run')
            item['status'] = 'passed'
        except BaseException as exc:
            item.update(status='failed', error=repr(exc), elapsed_seconds=time.time()-item['started'])
            record['status'] = 'stopped_with_failure'
            raise
        finally:
            stopped.set()
            monitor.join(timeout=10)
            active = None
            atomic_json(args.output/'queue.json', record)
        print(json.dumps(dict(name=job['name'], status=item['status'], seconds=item['elapsed_seconds']),
                         ensure_ascii=False), flush=True)
    record['status'] = 'interrupted' if interrupted else 'completed_all_distinct_survivors'
    atomic_json(args.output/'queue.json', record)
    return 130 if interrupted else 0


if __name__ == '__main__':
    raise SystemExit(main())
