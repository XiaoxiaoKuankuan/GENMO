"""服务器1八卡大规模DPPO有限端到端验收启动器，不启动长期训练。

从4096/8192/16384生产候选配置构建独立路径配置，只覆盖仓库路径、显式数值模式、
显式算子候选和本次输出位置。每环境决策、真实全局链数、PPO epochs、Actor/Critic/BC工作量、
概率与KL门槛保持配置值。最多运行两轮，完整使用正式训练器的初始化验证、末轮
验证、断点、预算和有界归档机制；不使用旧160条小测试代替完整验收。

输出目录必须不存在，训练写入其run子目录，配置/控制日志独立保存。恢复测试显式
指向同一已发布latest，不能恢复不同数值合同。启动前由正式入口检查八卡空闲，
子进程超时受控终止，未完成工作保持失败，不将后台背压隐藏在计时之外。
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]


def select_numerical_contract(config, precision_mode, numerical_variant, *, resume):
    """一次确定实际候选，恢复时只核对原合同，不用CLI默认值覆盖断点身份。"""
    performance = config['stage10']['performance']
    variant = (performance.get('numerical_variant', 'default')
               if numerical_variant is None else numerical_variant)
    if resume:
        if (performance['precision_mode'] != precision_mode or
                performance.get('numerical_variant', 'default') != variant):
            raise ValueError('Resume cannot change numerical execution contract')
    else:
        performance.update(precision_mode=precision_mode, numerical_variant=variant)
    if variant in ('blocked64_fp32_gemm', 'compiled_blocked64_fp32') and precision_mode != 'fp32_fast':
        raise ValueError('Blocked FP32 numerical variant requires fp32_fast')
    if variant.startswith('compensated_bf16') and precision_mode != 'bf16_backbone_candidate':
        raise ValueError('Compensated BF16 numerical variant requires bf16_backbone_candidate')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--gmt-repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--precision-mode', choices=('fp32_reference','fp32_fast','tf32_candidate','bf16_backbone_candidate'), required=True)
    parser.add_argument('--numerical-variant', choices=('default','blocked64_fp32_gemm',
        'compiled_blocked64_fp32','compensated_bf16x3','compensated_bf16x6'),
        help='显式新数值合同；省略时保留所选配置，恢复时不能修改')
    parser.add_argument('--rounds', type=int, choices=(1,2), default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--profile-world', action='store_true',help='只用于Python外围定位，该轮不作无profiler吞吐结果')
    parser.add_argument('--deadline-seconds', type=int, default=3600)
    args = parser.parse_args()
    if not 300 <= args.deadline_seconds <= 7200: raise ValueError('Finite deadline must be 300..7200 seconds')
    output = args.output.resolve()
    if args.resume:
        if not (output/'run/latest.json').is_file(): raise ValueError('Resume requires a published complete checkpoint')
        config = yaml.safe_load((output/'config.yaml').read_text())
        select_numerical_contract(config, args.precision_mode, args.numerical_variant, resume=True)
    else:
        output.mkdir(parents=True, exist_ok=False)
        config = yaml.safe_load(args.config.read_text())
        if config['runtime'].get('num_envs') != 64 or 'scale' not in config['stage10']:
            raise ValueError('Full-scale acceptance requires 64 active environments and derived scale')
        config['paths'].update(genmo_repo=str(ROOT), gmt_repo=str(args.gmt_repo.resolve()),
            kinematics=str(ROOT/'configs/bumi/bumi_kinematics_robot_retargeter_fe934_v1.json'),
            compat_profile=str(args.gmt_repo.resolve()/'configs/sim2sim/model_135000_stage2.json'))
        select_numerical_contract(config, args.precision_mode, args.numerical_variant, resume=False)
        config['stage10']['performance']['python_world_profile']=args.profile_world
        # 临时归档与正式目录隔离，容量上限保持原授权。
        config['stage10']['storage']['archive_secondary']['root'] = str(output/'execution_archives')
        (output/'config.yaml').write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        NCCL_CUMEM_HOST_ENABLE='0', NCCL_IB_DISABLE='1', NCCL_SOCKET_IFNAME='lo', TORCH_NCCL_BLOCKING_WAIT='1')
    environment['PYTHONPATH'] = str(ROOT)+(':'+environment['PYTHONPATH'] if environment.get('PYTHONPATH') else '')
    environment['LD_LIBRARY_PATH'] = '/data0/user/liwei/closedloop_stage8_runtime/sysroot/usr/lib/x86_64-linux-gnu'+(
        ':'+environment['LD_LIBRARY_PATH'] if environment.get('LD_LIBRARY_PATH') else '')
    command = ['timeout','--signal=TERM','--kill-after=60',str(args.deadline_seconds),sys.executable,'-B','-m',
        'torch.distributed.run','--standalone','--nproc_per_node=8','tools/train_closedloop_stage10_8gpu.py',
        '--config',str(output/'config.yaml'),'--output-dir',str(output/'run'),
        '--stop-after-iteration',str(args.rounds)]
    if args.resume: command += ['--resume','latest']
    with (output/('resume.console.log' if args.resume else 'console.log')).open('x') as log:
        completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    return completed.returncode


if __name__ == '__main__': raise SystemExit(main())
