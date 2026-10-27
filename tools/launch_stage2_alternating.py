"""实际启动八个独立SuperMimic交替训练进程，不使用DDP/NCCL或方向验收。

启动器在Server-1读取实时GPU计算占用，只启动空闲卡；每卡独立配置、tmux、日志、
PID、Replay和checkpoint。先创建唯一时间戳根目录和固定提交的只读运行工作树，
旧实验不覆盖、其他进程不终止。完整命令及跳过资源原因写入launch_manifest.json，
训练进程从零Adapter/原WM开始100个WM-only循环后直接正式BPTT更新。
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--session-prefix', required=True)
    p.add_argument('--resume-root', help='逐GPU读取同组latest_checkpoint，不共享模型或Replay；输出仍须新目录')
    p.add_argument('--python', default='/data0/user/liwei/envs/genmo_adapter_runtime_20261026/bin/python')
    p.add_argument('--isaaclab', default='/data0/user/liwei/GENMO_assets/adapter_20261026/IsaacLab')
    p.add_argument('--inductor-cache', default='/data0/user/liwei/GENMO_assets/adapter_20261026/inductor_cache')
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    source = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    runtime = root/'source_genmo'
    subprocess.run(['git', '-C', str(repo), 'worktree', 'add', '--detach', str(runtime), source], check=True)
    original = yaml.safe_load((repo/'configs/closedloop/stage2_alternating/gpu0.yaml').read_text())
    sm_repo = Path(original['supermimic']['repo'])
    sm_source = subprocess.check_output(['git', '-C', str(sm_repo), 'rev-parse', 'HEAD'], text=True).strip()
    runtime_sm = root/'source_supermimic'
    subprocess.run(['git', '-C', str(sm_repo), 'worktree', 'add', '--detach', str(runtime_sm), sm_source], check=True)
    gpu_lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    uuids = {int(line.split(',')[0]): line.split(',')[1].strip() for line in gpu_lines.splitlines()}
    manifest = dict(source=source, supermimic_source=sm_source, runtime_genmo=str(runtime),
                    runtime_supermimic=str(runtime_sm), created_unix=time.time(), launches=[], skipped=[],
                    independent=True, ddp=False, nccl=False, formal=True)
    for gpu in range(8):
        config = yaml.safe_load((repo/f'configs/closedloop/stage2_alternating/gpu{gpu}.yaml').read_text())
        config['supermimic']['repo'] = str(runtime_sm)
        path = root/f'config_gpu{gpu}.yaml'
        path.write_text('# 独立正式运行配置；冻结资产只读，在线模型/回放均独立。\n'+yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
        busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
        if gpu not in uuids or any(line.split(',')[0].strip() == uuids[gpu] for line in busy.splitlines()):
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
        runner.write_text('#!/usr/bin/env bash\n# 本组独立正式训练；不自动重启，不停止其他任务。\nset -u\n'+
            'cd '+shlex.quote(str(runtime))+' || exit 3\n'+shlex.join(command)+' > '+shlex.quote(str(root/f'gpu{gpu}.log'))+' 2>&1\n'+
            'training_exit=$?\nprintf "%s\\n" "$training_exit" > '+shlex.quote(str(root/f'gpu{gpu}.exit'))+'\nexit "$training_exit"\n')
        subprocess.run(['tmux', 'new-session', '-d', '-s', session, 'bash', str(runner)], check=True)
        manifest['launches'].append(dict(gpu=gpu, uuid=uuids[gpu], session=session, command=command,
            output=str(root/f'gpu{gpu}'), config=str(path), runner=str(runner), num_envs=config['vector']['num_envs']))
        temp = root/'launch_manifest.tmp'
        temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        os.replace(temp, root/'launch_manifest.json')
        time.sleep(1)
    (root/'launch_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
