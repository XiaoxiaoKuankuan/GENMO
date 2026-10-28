"""实际启动八个独立SuperMimic交替训练进程，不使用DDP/NCCL或方向验收。

启动器在Server-1读取实时GPU计算占用，只启动空闲卡；每卡独立配置、tmux、日志、
PID、Replay和checkpoint。先创建唯一时间戳根目录和固定提交的只读运行工作树，
旧实验不覆盖、其他进程不终止。完整命令及跳过资源原因写入launch_manifest.json，
训练进程从零Adapter/原WM开始，预算和实验矩阵读取显式配置目录。默认占用卡跳过；
可显式提交每卡等待任务，在GPU计算进程自然退出后启动，不终止或抢占原任务。
等待进程不加载CUDA，状态与训练状态分别落盘；仅真正启动的子进程拥有训练PID，
排队不冒充采集或Adapter更新。训练异常不自动重启，保留完整失败现场。
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import yaml


def _atomic_json(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    os.replace(temp, path)


def _wait_and_run(spec_path, poll_seconds=30.):
    """无CUDA等待器：逐卡互斥并检查真实计算PID，空闲后仅启动一次独立训练。"""
    import fcntl
    spec = json.loads(Path(spec_path).read_text())
    root, gpu = Path(spec['root']), spec['gpu']
    status_path = root/f'queue_gpu{gpu}.json'
    log_path, exit_path = root/f'gpu{gpu}.log', root/f'gpu{gpu}.exit'
    if log_path.exists() or exit_path.exists():
        raise FileExistsError('该组已启动过；等待器不覆盖日志或自动重启')
    lock_dir = Path(spec.get('lock_dir', f'/tmp/genmo_stage2_gpu_leases_{os.getuid()}'))
    lock_dir.mkdir(parents=True, exist_ok=True)
    state = dict(gpu=gpu, uuid=spec['uuid'], queue_pid=os.getpid(), training_pid=None,
                 submitted_unix=time.time(), command=spec['command'], output=spec['output'])

    def publish(phase, **fields):
        state.update(phase=phase, updated_unix=time.time(), **fields)
        _atomic_json(status_path, state)

    try:
        # 互斥锁只约束本启动器；其它任务仍依赖nvidia-smi真实占用检查，不发停止信号。
        with (lock_dir/(spec['uuid']+'.lock')).open('a') as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    publish('waiting_for_gpu_lease')
                    time.sleep(poll_seconds)
            while True:
                try:
                    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
                        '--format=csv,noheader'], text=True)
                    uuids = {int(row.split(',')[0]): row.split(',')[1].strip()
                             for row in devices.splitlines()}
                    if uuids.get(gpu) != spec['uuid']:
                        raise RuntimeError('GPU索引/UUID改变，拒绝在错误设备启动')
                    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                        '--format=csv,noheader'], text=True)
                except subprocess.CalledProcessError as error:
                    publish('waiting_gpu_query_error', error=str(error))
                    time.sleep(poll_seconds)
                    continue
                blocking = [row.strip() for row in processes.splitlines()
                            if row.split(',')[0].strip() == spec['uuid']]
                if not blocking:
                    break
                publish('waiting_for_gpu', blocking_compute_processes=blocking)
                time.sleep(poll_seconds)
            with log_path.open('x') as log:
                child = subprocess.Popen(spec['command'], cwd=spec['cwd'], stdout=log,
                                         stderr=subprocess.STDOUT)
                publish('training', training_pid=child.pid, started_unix=time.time(),
                        blocking_compute_processes=[])
                code = child.wait()
            exit_path.write_text(str(code)+'\n')
            publish('completed' if code == 0 else 'failed', exit_code=code)
            return code
    except BaseException as error:
        publish('queue_error', error=repr(error))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--session-prefix', required=True)
    p.add_argument('--gpus', default='0,1,2,3,4,5,6,7', help='本次明确启动的GPU索引；已有计算任务仍跳过')
    p.add_argument('--resume-root', help='逐GPU读取同组latest_checkpoint，不共享模型或Replay；输出仍须新目录')
    p.add_argument('--python', default='/data0/user/liwei/envs/genmo_adapter_runtime_20261026/bin/python')
    p.add_argument('--isaaclab', default='/data0/user/liwei/GENMO_assets/adapter_20261026/IsaacLab')
    p.add_argument('--inductor-cache', default='/data0/user/liwei/GENMO_assets/adapter_20261026/inductor_cache')
    p.add_argument('--config-dir', default='configs/closedloop/stage2_alternating',
                   help='相对仓库或绝对路径；读取gpu0.yaml到gpu7.yaml，运行时各自冻结')
    p.add_argument('--wait-for-free-gpus', action='store_true', help='占用卡建立无CUDA等待任务，空闲后自动启动')
    p.add_argument('--poll-seconds', type=float, default=30.)
    args = p.parse_args()
    requested_gpus = [int(value) for value in args.gpus.split(',')]
    if not requested_gpus or len(set(requested_gpus)) != len(requested_gpus) or any(gpu not in range(8) for gpu in requested_gpus):
        raise ValueError('GPU列表必须是0到7的不重复索引')
    repo = Path(__file__).resolve().parents[1]
    config_dir = Path(args.config_dir)
    if not config_dir.is_absolute():
        config_dir = repo/config_dir
    if args.poll_seconds <= 0:
        raise ValueError('轮询间隔必须为正')
    for gpu in requested_gpus:
        if not (config_dir/f'gpu{gpu}.yaml').is_file():
            raise FileNotFoundError(config_dir/f'gpu{gpu}.yaml')
    source = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    runtime = root/'source_genmo'
    subprocess.run(['git', '-C', str(repo), 'worktree', 'add', '--detach', str(runtime), source], check=True)
    original = yaml.safe_load((config_dir/f'gpu{requested_gpus[0]}.yaml').read_text())
    sm_repo = Path(original['supermimic']['repo'])
    sm_source = subprocess.check_output(['git', '-C', str(sm_repo), 'rev-parse', 'HEAD'], text=True).strip()
    runtime_sm = root/'source_supermimic'
    subprocess.run(['git', '-C', str(sm_repo), 'worktree', 'add', '--detach', str(runtime_sm), sm_source], check=True)
    gpu_lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    uuids = {int(line.split(',')[0]): line.split(',')[1].strip() for line in gpu_lines.splitlines()}
    manifest = dict(source=source, supermimic_source=sm_source, runtime_genmo=str(runtime),
                    runtime_supermimic=str(runtime_sm), created_unix=time.time(), launches=[], skipped=[],
                    independent=True, ddp=False, nccl=False, formal=True,
                    config_dir=str(config_dir), wait_for_free_gpus=args.wait_for_free_gpus)
    for gpu in requested_gpus:
        config = yaml.safe_load((config_dir/f'gpu{gpu}.yaml').read_text())
        config['supermimic']['repo'] = str(runtime_sm)
        path = root/f'config_gpu{gpu}.yaml'
        path.write_text('# 独立正式运行配置；冻结资产只读，在线模型/回放均独立。\n'+yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
        busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
        occupied = gpu in uuids and any(line.split(',')[0].strip() == uuids[gpu] for line in busy.splitlines())
        if gpu not in uuids or (occupied and not args.wait_for_free_gpus):
            manifest['skipped'].append(dict(gpu=gpu, reason='GPU不存在或已有计算进程，未抢占', occupancy=busy.strip()))
            continue
        session = f'{args.session_prefix}_g{gpu}'
        if subprocess.run(['tmux', 'has-session', '-t', session], capture_output=True).returncode == 0:
            manifest['skipped'].append(dict(gpu=gpu, reason='同名tmux已存在，未覆盖', session=session))
            continue
        command = ['env', f'CUDA_VISIBLE_DEVICES={gpu}', 'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
            'PYTHONDONTWRITEBYTECODE=1', 'TORCHINDUCTOR_COMPILE_THREADS=1',
            f'TORCHINDUCTOR_CACHE_DIR={args.inductor_cache}', f'ISAACLAB_PATH={args.isaaclab}',
            args.python, '-B', '-u', str(runtime/'tools/train_stage2_alternating.py'),
            '--config', str(path), '--output', str(root/f'gpu{gpu}'), '--visualizer', 'none', '--deterministic']
        if args.resume_root:
            pointer = json.loads((Path(args.resume_root)/f'gpu{gpu}/latest_checkpoint.json').read_text())
            resume = Path(pointer['path'])
            if not resume.is_file():
                raise FileNotFoundError(f'GPU{gpu}自己的完整checkpoint不存在：{resume}')
            command += ['--resume', str(resume)]
        runner = root/f'run_gpu{gpu}.sh'
        if args.wait_for_free_gpus:
            spec_path = root/f'wait_gpu{gpu}.json'
            spec_path.write_text(json.dumps(dict(root=str(root), gpu=gpu, uuid=uuids[gpu],
                cwd=str(runtime), output=str(root/f'gpu{gpu}'), command=command), ensure_ascii=False, indent=2))
            wait_command = [args.python, '-B', '-u', str(runtime/'tools/launch_stage2_alternating.py'),
                            '--wait-spec', str(spec_path), '--poll-seconds', str(args.poll_seconds)]
            runner.write_text('#!/usr/bin/env bash\n# 无CUDA等待器；既有任务自然退出后仅启动本组一次。\nset -eu\n'+
                'exec '+shlex.join(wait_command)+' > '+shlex.quote(str(root/f'queue_gpu{gpu}.log'))+' 2>&1\n')
        else:
            runner.write_text('#!/usr/bin/env bash\n# 本组独立正式训练；不自动重启，不停止其他任务。\nset -u\n'+
                'cd '+shlex.quote(str(runtime))+' || exit 3\n'+shlex.join(command)+' > '+shlex.quote(str(root/f'gpu{gpu}.log'))+' 2>&1\n'+
                'training_exit=$?\nprintf "%s\\n" "$training_exit" > '+shlex.quote(str(root/f'gpu{gpu}.exit'))+'\nexit "$training_exit"\n')
        subprocess.run(['tmux', 'new-session', '-d', '-s', session, 'bash', str(runner)], check=True)
        manifest['launches'].append(dict(gpu=gpu, uuid=uuids[gpu], session=session, command=command,
            output=str(root/f'gpu{gpu}'), config=str(path), runner=str(runner), num_envs=config['vector']['num_envs'],
            horizon=config['bptt_steps'], critic_enabled=config['alternating']['critic_enabled'],
            action_noise=config['alternating']['action_noise'], root_orientation=config['alternating'].get('root_orientation', True),
            experiment=config['alternating'].get('experiment_label'), target_updates=config['vector']['target_accepted_updates'],
            submission_mode='wait_until_free' if args.wait_for_free_gpus else 'immediate',
            initially_occupied=occupied, adapter_lr=config['training']['adapter_lr']))
        temp = root/'launch_manifest.tmp'
        temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        os.replace(temp, root/'launch_manifest.json')
        time.sleep(1)
    (root/'launch_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    if '--wait-spec' in sys.argv:
        worker_parser = argparse.ArgumentParser(description='单卡正式训练资源等待器，不加载CUDA')
        worker_parser.add_argument('--wait-spec', required=True)
        worker_parser.add_argument('--poll-seconds', type=float, default=30.)
        worker_args = worker_parser.parse_args()
        if worker_args.poll_seconds <= 0:
            raise ValueError('轮询间隔必须为正')
        raise SystemExit(_wait_and_run(worker_args.wait_spec, worker_args.poll_seconds))
    else:
        main()
