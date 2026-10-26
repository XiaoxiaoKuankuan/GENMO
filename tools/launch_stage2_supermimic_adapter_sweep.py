"""在Server-1逐个启动八个独立单GPU多环境实验，不创建任何跨GPU通信组。

先读取真实压力报告并检查八张卡没有计算进程，再给每张卡独立输出目录、tmux会话、
配置和命令；所有组共享只读冻结资产，在线Replay/模型/优化器互不共享。已有目录
或会话拒绝覆盖，不终止其他任务。训练进程自己做真实WM/Critic预热和方向验收。
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
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True); p.add_argument('--pressure-root', required=True)
    p.add_argument('--python', required=True); p.add_argument('--num-envs', type=int, default=1024)
    p.add_argument('--session-prefix', required=True)
    p.add_argument('--isaaclab', required=True); p.add_argument('--inductor-cache', required=True)
    args=p.parse_args()
    root=Path(args.root).resolve(); pressure=Path(args.pressure_root).resolve()
    repo=Path(__file__).resolve().parents[1]
    if args.num_envs not in (512, 1024, 2048):
        raise ValueError('正式环境数只接受已验证的512/1024/2048')
    reports=[]
    for path in pressure.glob(f'pressure{args.num_envs}*/pressure.json'):
        value=json.loads(path.read_text())
        if value.get('passed') and value['num_envs']==args.num_envs:
            reports.append((path,value))
    if not reports:
        raise ValueError('缺少所选环境数的真实物理/WM/BPTT显存压力证据')
    gpu_rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines()
    if len(gpu_rows)!=8:
        raise ValueError('本启动器要求现场可见8张物理GPU')
    occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).strip()
    if occupied:
        raise RuntimeError('检测到既有GPU计算进程，保留现场，不抢占：'+occupied)
    root.mkdir(parents=True, exist_ok=False)
    source=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    # 运行代码固定在独立detached worktree，后续更新主分支记录不会热修改训练源码。
    runtime_repo=root/'source_genmo'
    subprocess.run(['git','-C',str(repo),'worktree','add','--detach',str(runtime_repo),source],check=True)
    original_cfg=yaml.safe_load((repo/'configs/closedloop/adapter_sweep_20261026/gpu0.yaml').read_text())
    sm_repo=Path(original_cfg['supermimic']['repo'])
    sm_source=subprocess.check_output(['git','-C',str(sm_repo),'rev-parse','HEAD'],text=True).strip()
    runtime_sm=root/'source_supermimic'
    subprocess.run(['git','-C',str(sm_repo),'worktree','add','--detach',str(runtime_sm),sm_source],check=True)
    launches=[]
    for gpu in range(8):
        session=f'{args.session_prefix}_g{gpu}'
        exists=subprocess.run(['tmux','has-session','-t',session],capture_output=True).returncode==0
        if exists:
            raise RuntimeError('不覆盖已有tmux会话：'+session)
        output=root/f'gpu{gpu}'
        original=repo/f'configs/closedloop/adapter_sweep_20261026/gpu{gpu}.yaml'
        config=root/f'config_gpu{gpu}.yaml'
        resolved=yaml.safe_load(original.read_text())
        resolved['supermimic']['repo']=str(runtime_sm)
        resolved['vector']['num_envs']=args.num_envs
        with config.open('x') as f:
            f.write('# 正式运行配置：固定发布源码，独立单GPU模型/优化器/Replay；冻结资产只读。\n')
            f.write(yaml.safe_dump(resolved,allow_unicode=True,sort_keys=False))
        command=['env',f'CUDA_VISIBLE_DEVICES={gpu}','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1',
            'PYTHONDONTWRITEBYTECODE=1','TORCHINDUCTOR_COMPILE_THREADS=1',
            f'TORCHINDUCTOR_CACHE_DIR={args.inductor_cache}',f'ISAACLAB_PATH={args.isaaclab}',
            args.python,'-B','-u',str(runtime_repo/'tools/train_stage2_supermimic_adapter_vector.py'),
            '--config',str(config),'--output',str(output),'--mode','train','--num-envs',str(args.num_envs),
            '--visualizer','none','--deterministic']
        runner=root/f'run_gpu{gpu}.sh'
        content='#!/usr/bin/env bash\n# 单GPU独立实验；不自动重启、不停止其他组。\nset -u\n'
        content+='cd '+shlex.quote(str(runtime_repo))+' || exit 3\n'
        content+=shlex.join(command)+' > '+shlex.quote(str(root/f'gpu{gpu}.log'))+' 2>&1\n'
        content+='adapter_exit=$?\nprintf "%s\\n" "$adapter_exit" > '+shlex.quote(str(root/f'gpu{gpu}_exit_code.txt'))+'\nexit "$adapter_exit"\n'
        with runner.open('x') as f:f.write(content)
        subprocess.run(['tmux','new-session','-d','-s',session,'bash',str(runner)],check=True)
        launches.append(dict(gpu=gpu,session=session,directory=str(output),command=command,runner=str(runner),num_envs=args.num_envs))
        partial=dict(unix=time.time(),source=source,supermimic_source=sm_source,launches=launches,
            runtime_genmo=str(runtime_repo),runtime_supermimic=str(runtime_sm),launch_complete=len(launches)==8)
        temporary=root/'launch_manifest.tmp'
        temporary.write_text(json.dumps(partial,ensure_ascii=False,indent=2))
        os.replace(temporary,root/'launch_manifest.json')
        time.sleep(1)
    manifest=dict(partial,pressure_evidence=str(reports[-1][0]),
        independent=True,ddp=False,torchrun=False,nccl_allreduce=False)
    with (root/'launch_manifest.json').open('w') as f:json.dump(manifest,f,ensure_ascii=False,indent=2)
    reporter=[args.python,'-B','-u',str(runtime_repo/'tools/report_stage2_supermimic_adapter_sweep.py'),'--root',str(root),'--watch']
    subprocess.run(['tmux','new-session','-d','-s',args.session_prefix+'_report',shlex.join(reporter)],check=True)
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
