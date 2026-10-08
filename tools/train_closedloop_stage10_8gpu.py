"""服务器单机八卡同步 Stage10 入口，默认八个独立冻结 GMT 采集器。

v2 每卡采集二十条真实转移，使用固定 5e-9 学习率的多 minibatch PPO，整轮 KL
验收通过后才发布下一轮策略。各卡持有独立物理环境、随机数和音乐游标，共享同步
更新的模型。显式传入 v1 配置仍使用下面描述的历史单采集流程，不能交叉完整恢复。

由 torchrun 创建八个 rank，首先在建立 CUDA 上下文前核验全部可见 GPU 空闲及
既有资产，然后初始化 Gloo 控制通信与 NCCL 梯度通信。rank 0 复用已验收的完整
数据 Stage10 入口：实际 GMT 采集、固定回报、唯一预算、checkpoint 和日志。
其他七张卡只参与同一模型的 Actor/Critic 梯度计算，所有 rank 同步更新同一组
参数并逐轮核验一致性。默认只运行一轮启动验收，不自动启动长期训练。

支持单机八卡新 run 和 --resume latest；主进程沿用既有checkpoint身份、预算、
采样器和随机状态校验，再把模型及完整优化器同步到其余rank。不接受跨 Stage9
初始化和单卡入口伪装多卡。本脚本不读取 200 首音乐名单，
仍由原训练入口审计完整数据集。失败时以非零状态交给 torchrun 终止整个作业。
"""
from __future__ import annotations

import argparse
import csv
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist

from gem.closedloop.dppo.distributed_runtime import DistributedCollectives, DistributedLearner
from tools import train_closedloop_stage10 as training


def _available_gpus():
    visible = [value.strip() for value in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if value.strip()]
    if len(visible) != 8 or len(set(visible)) != 8:
        raise ValueError('Eight distinct GPUs must be explicitly exposed in CUDA_VISIBLE_DEVICES')
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free',
                                     '--format=csv,noheader,nounits'], text=True)
    devices = [[v.strip() for v in row] for row in csv.reader(output.splitlines())]
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                     '--format=csv,noheader,nounits'], text=True)
    active_uuids = {row[0].strip() for row in csv.reader(active.splitlines()) if row}
    selected = []
    for name in visible:
        matches = [row for row in devices if row[0] == name or row[1] == name]
        if len(matches) != 1 or matches[0][1] in active_uuids or int(matches[0][2]) < 8192:
            raise RuntimeError(f'GPU {name} is missing, occupied, or has less than 8 GiB available')
        selected.append(matches[0])
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/closedloop/stage10_8gpu_server1_v2.yaml')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--stop-after-iteration', type=int, default=1)
    parser.add_argument('--resume', help='Resume latest or the latest published checkpoint in this run')
    args = parser.parse_args(argv)
    rank, world, local_rank = (int(os.environ.get(key, '-1')) for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
    if world != 8 or not 0 <= rank < world or local_rank != rank:
        parser.error('Use single-node torchrun --standalone --nproc_per_node=8')
    config = training.configuration(args.config)
    parallel_v2 = config['stage10']['version'] == training.VERSION_V2
    if config['stage10'].get('distributed') != dict(world_size=8, backend='nccl', collection='all_ranks' if parallel_v2 else 'rank0'):
        parser.error('Expected explicit 8 GPU synchronous learner configuration')
    if not 1 <= args.stop_after_iteration <= config['stage10']['limits']['accepted_iterations']:
        parser.error('Stop iteration must fit the configured budget')
    # Gloo不占用CUDA，在rank0完成资源核验之前其他rank不创建CUDA上下文。
    dist.init_process_group('gloo', timeout=timedelta(minutes=30))
    startup = [None]
    if rank == 0:
        try:
            devices = _available_gpus()
            check = training.runtime_preflight(config, check_gpu=True)
            if not check['ready']:
                raise RuntimeError('Runtime assets failed preflight')
            startup[0] = dict(ok=True, check=check, devices=devices)
        except Exception as exc:
            startup[0] = dict(ok=False, error=f'{type(exc).__name__}: {exc}')
    dist.broadcast_object_list(startup, src=0)
    if not startup[0]['ok']:
        raise RuntimeError(startup[0]['error'])
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(config['runtime']['torch_threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    tensor_group = dist.new_group(backend='nccl', timeout=timedelta(minutes=15))
    collectives = DistributedCollectives(rank, world, tensor_group=tensor_group, device=f'cuda:{local_rank}')
    proof = collectives.sum_tensor(torch.tensor(float(rank + 1), device=f'cuda:{local_rank}'))
    if float(proof) != 36.:
        raise RuntimeError('Eight GPU NCCL all-reduce failed')
    if rank == 0:
        print('[DISTRIBUTED] eight GPU NCCL all_reduce=36; shared Actor/Critic training', flush=True)
    if parallel_v2:
        from gem.closedloop.dppo.parallel_training import run_parallel
        code = run_parallel(args, config, collectives, startup[0]['check'])
        # v2 返回前已经汇总各 rank 的退出状态；KL 拒绝也正常释放通信组。
        # 失败路径不再增加 barrier，避免把故障退出变为新的同步等待。
        if not code:
            dist.barrier()
        dist.destroy_process_group(tensor_group)
        dist.destroy_process_group()
        return code
    learner = DistributedLearner(collectives, args.output_dir, preflight=startup[0]['check'])
    if rank == 0:
        arguments = ['--config', str(args.config), '--mode', 'train', '--output-dir', str(args.output_dir),
                     '--stop-after-iteration', str(args.stop_after_iteration)]
        if args.resume:
            arguments.extend(['--resume', args.resume])
        code = training.main(arguments, learner=learner)
        if code:
            # 不在可能已失配的collective上广播stop，非零退出由elastic统一终止。
            return code
        learner.close()
    else:
        learner.worker_loop()
    dist.barrier()
    dist.destroy_process_group(tensor_group)
    dist.destroy_process_group()
    if rank == 0:
        print(json.dumps(dict(status='passed', world_size=world, shared_model=True,
                              requested_stop_iteration=args.stop_after_iteration, resumed=bool(args.resume))), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
