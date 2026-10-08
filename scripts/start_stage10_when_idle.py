"""服务器1八卡正式训练的空闲等待、有限验收和一次性启动入口。

本工具承接已明确授权的10000外层轮、每500轮保存任务，只在全部八张GPU没有
计算进程且显存低于512MiB时启动。等待期仅使用CPU，每30秒检查一次，连续三次
空闲才继续，最长等待24小时；不终止其他用户/其他任务的进程，不允许共享GPU。

启动前固定两个仓库的提交、正式配置SHA及新输出目录。先在独立acceptance目录
完成第一轮正常保存、退出、恢复至第二轮，验收最多两轮/八次Actor更新，60分钟
总上限。独立审计必须全部通过，并核实实际microbatch=2、CFG合批和严格概率门槛。
随后按正式32任务参考数据重新核算双盘容量，再次确认GPU空闲和代码身份后，才
执行正式10000轮命令。任何检查失败都保留日志并停止，不重试、不修改学习率/KL。

job目录保存request.json、status.json、逐阶段控制台和原子退出状态；fcntl排他锁
防止两个启动器重复操作同一任务。等待时创建cancel文件可取消尚未开始的阶段。
正式训练由现有训练器负责500轮checkpoint、真实预算、故障回滚和受控退出，本工具
只监督自己创建的进程组，发生明确超时时才终止该进程组。短测原始证据保留供审计。
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import yaml


def write_json(path, value):
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def gpu_snapshot():
    def query(argument):
        return subprocess.check_output(['nvidia-smi', argument, '--format=csv,noheader,nounits'],
                                       text=True, timeout=20).strip()
    pids = sorted({int(row) for row in query('--query-compute-apps=pid').splitlines() if row.strip()})
    devices = [tuple(map(int, row.split(','))) for row in query('--query-gpu=index,memory.used').splitlines()]
    if sorted(index for index, _ in devices) != list(range(8)):
        raise RuntimeError('Expected exactly eight server1 GPUs')
    return dict(idle=not pids and all(memory < 512 for _, memory in devices),
                compute_pids=pids, memory_mib={str(index): memory for index, memory in devices})


def verify_sources(request):
    for key in ('genmo', 'gmt'):
        repository = request[key+'_repo']
        head = subprocess.check_output(['git', '-C', repository, 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', repository, 'status', '--porcelain'], text=True)
        if head != request[key+'_head'] or dirty.strip():
            raise RuntimeError(f'{key} source changed or worktree is dirty; explicit review required')
    if hashlib.sha256(Path(request['config']).read_bytes()).hexdigest() != request['config_sha256']:
        raise RuntimeError('Formal configuration changed while waiting')


def acceptance_gate(root, audit):
    if audit.get('status') != 'passed' or audit.get('failed_checks') != 0 or audit.get('not_run_checks') != 0:
        raise RuntimeError('Independent eight-GPU acceptance audit did not pass all required checks')
    summaries = [json.loads(p.read_text()) for p in root.glob('sessions/*/summary.json')]
    if len(summaries) != 2 or any(row['status'] != 'passed' for row in summaries):
        raise RuntimeError('Both pre-resume and resumed acceptance sessions must pass')
    for row in summaries:
        profile = row['final_state']['execution_profile']
        if (profile.get('microbatch') != 2 or profile.get('cfg_batch') is not True
                or profile.get('probability_tolerances') != dict(logprob=1e-4, ratio=1e-3, gaussian=1e-8)):
            raise RuntimeError('Actual batching/CFG or strict probability gates differ from the accepted contract')
    if sorted(row['final_state']['iteration'] for row in summaries) != [1, 2]:
        raise RuntimeError('Acceptance did not complete exactly one round followed by resume to round two')


class Launcher:
    def __init__(self, job, request):
        self.job, self.request = job, request
        self.wait_deadline = time.monotonic()+request['max_wait_seconds']

    def status(self, phase, **fields):
        value = dict(phase=phase, checked_at=datetime.now(timezone.utc).isoformat(),
                     supervisor_pid=os.getpid(), **fields)
        write_json(self.job/'status.json', value)
        print(json.dumps(value, ensure_ascii=False), flush=True)

    def wait_idle(self, phase, *, deadline=None, stable=3):
        deadline = self.wait_deadline if deadline is None else deadline
        count = 0
        while True:
            if (self.job/'cancel').exists():
                raise RuntimeError('User cancelled the pending launch')
            if time.monotonic() >= deadline:
                raise TimeoutError('GPU wait deadline reached; no unrelated process was stopped')
            snapshot = gpu_snapshot()
            count = count+1 if snapshot['idle'] else 0
            self.status(phase, consecutive_idle_checks=count, gpu=snapshot)
            if count >= stable:
                return
            time.sleep(min(30, max(0., deadline-time.monotonic())))

    def command(self, phase, command, *, timeout, environment=None):
        if (self.job/'cancel').exists():
            raise RuntimeError('User cancelled before starting the next phase')
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
                   PYTHONPATH=self.request['genmo_repo'], OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
        env.update(environment or {})
        with (self.job/f'{phase}.console.log').open('xb') as log:
            process = subprocess.Popen(command, cwd=self.request['genmo_repo'], env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, close_fds=True)
            started = time.monotonic()
            self.status(phase, child_pid=process.pid, command=command)
            try:
                code = process.wait(timeout=timeout)
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=45)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                raise
            result = dict(exit_code=code, wall_seconds=time.monotonic()-started, child_pid=process.pid)
            write_json(self.job/f'{phase}.result.json', result)
            if code != 0:
                raise RuntimeError(f'{phase} failed with exit code {code}; see its console log')
            return result

    def execute(self):
        r = self.request
        self.wait_idle('waiting_for_gpu')
        verify_sources(r)
        # new目录仅在获得全部GPU后创建，排队阶段不伪造一个已开始的正式run。
        config = yaml.safe_load(Path(r['config']).read_text())
        test = copy.deepcopy(config)
        stage = test['stage10']
        stage['limits'] = dict(accepted_iterations=2, optimizer_attempts=8, generations=3000,
                               control_steps=1000000, physics_steps=4000000)
        stage['run_control']['max_walltime_seconds'] = 3600
        stage['evaluation'].update(eval_count=4, samples_per_source=1)
        stage['storage']['max_run_bytes'] = 100*2**30
        stage['storage']['archive_secondary'].update(max_bytes=10*2**30,
            root=str(Path(stage['storage']['archive_secondary']['root'])/('acceptance_'+self.job.name)))
        test_config = self.job/'acceptance.yaml'
        with test_config.open('x') as stream:
            yaml.safe_dump(test, stream, sort_keys=False, allow_unicode=True)
        acceptance = self.job/'acceptance'
        if acceptance.exists() or Path(r['formal_run']).exists():
            raise FileExistsError('Both acceptance and formal output directories must be new')
        training = ['bash', 'scripts/train_stage10_8gpu_server1.sh', '--output-dir', str(acceptance)]
        gpu_env = dict(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', STAGE10_8GPU_CONFIG=str(test_config))
        deadline = time.monotonic()+3600
        self.command('acceptance_first', training+['--stop-after-iteration', '1'],
                     timeout=max(1, deadline-time.monotonic()), environment=gpu_env)
        self.wait_idle('waiting_acceptance_resume', deadline=min(deadline, time.monotonic()+180), stable=1)
        self.command('acceptance_resume', training+['--stop-after-iteration', '2', '--resume', 'latest'],
                     timeout=max(1, deadline-time.monotonic()), environment=gpu_env)
        python = config['paths']['genmo_python']
        self.command('acceptance_audit', [python, '-B', 'tools/eval/audit_closedloop_stage10.py',
            '--run-dir', str(acceptance), '--output', str(self.job/'acceptance_audit.json')], timeout=300)
        acceptance_gate(acceptance, json.loads((self.job/'acceptance_audit.json').read_text()))
        self.wait_idle('waiting_formal_gpu')
        verify_sources(r)
        self.command('capacity_recheck', [python, '-B', 'tools/eval/plan_closedloop_stage10_capacity.py',
            '--reference-run', r['capacity_reference'], '--target-run', r['formal_run'],
            '--config', r['config'], '--output', str(self.job/'capacity_recheck.json')], timeout=300)
        verify_sources(r)
        self.command('formal_training', ['bash', 'scripts/train_stage10_8gpu_server1.sh', '--output-dir',
            r['formal_run'], '--stop-after-iteration', '10000'],
            timeout=config['stage10']['run_control']['max_walltime_seconds']+1800,
            environment=dict(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', STAGE10_8GPU_CONFIG=r['config']))
        latest = json.loads((Path(r['formal_run'])/'latest.json').read_text())
        if latest['iteration'] != 10000:
            raise RuntimeError(f"Formal training stopped before the target: durable iteration={latest['iteration']}")
        self.status('completed', formal_run=r['formal_run'], durable_iteration=10000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-dir', type=Path, required=True)
    parser.add_argument('--formal-run', type=Path, required=True)
    parser.add_argument('--capacity-reference', type=Path, required=True)
    parser.add_argument('--max-wait-seconds', type=int, default=86400)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    config_path = repo/'configs/closedloop/stage10_8gpu_server1_v2.yaml'
    config = yaml.safe_load(config_path.read_text())
    storage, stage = config['stage10']['storage'], config['stage10']
    if (args.max_wait_seconds <= 0 or stage['limits']['accepted_iterations'] != 10000
            or storage['checkpoint_every_iterations'] != 500 or storage['checkpoint_keep_every'] != 500):
        raise ValueError('This explicit launcher requires 10000 outer rounds and 500-round saving')
    args.job_dir.mkdir(parents=True, exist_ok=True)
    with (args.job_dir/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        request_path = args.job_dir/'request.json'
        if request_path.exists() or args.formal_run.exists():
            raise FileExistsError('Do not repeat an existing launch request or overwrite a formal run')
        request = dict(genmo_repo=str(repo), gmt_repo=config['paths']['gmt_repo'], config=str(config_path),
            config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(), formal_run=str(args.formal_run.resolve()),
            capacity_reference=str(args.capacity_reference.resolve()), max_wait_seconds=args.max_wait_seconds)
        for key in ('genmo', 'gmt'):
            request[key+'_head'] = subprocess.check_output(['git', '-C', request[key+'_repo'],
                'rev-parse', 'HEAD'], text=True).strip()
        verify_sources(request)
        write_json(request_path, request)
        launcher = Launcher(args.job_dir.resolve(), request)
        try:
            launcher.execute()
        except BaseException as error:
            launcher.status('stopped', error_type=type(error).__name__, error=str(error))
            raise


if __name__ == '__main__':
    main()
