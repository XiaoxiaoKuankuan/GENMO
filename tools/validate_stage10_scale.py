"""服务器1八卡大规模DPPO有限端到端验收启动器，不启动长期训练。

从1024/2048/4096/8192有限候选配置构建独立路径配置，只覆盖仓库路径、显式数值模式、
显式算子候选和本次输出位置。每环境决策、真实全局链数、PPO epochs、Actor/Critic/BC工作量、
概率与KL门槛保持配置值。显式限定为1至10轮，完整使用正式训练器的初始化验证、末轮
验证、断点、预算和有界归档机制；不使用旧160条小测试代替完整验收。

输出目录必须不存在，训练写入其run子目录，配置/控制日志独立保存。恢复测试显式
指向同一已发布latest，不能恢复不同数值合同。启动前由正式入口检查八卡空闲，
子进程超时受控终止，未完成工作保持失败，不将后台背压隐藏在计时之外。
"""
from __future__ import annotations
import argparse
import math
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


def select_learning_experiment(config,*,steps=None,actor_lr=None,sensitive_output_fp64=False,resume=False,
                              sampling_strategy=None):
    """显式算法对照入口；缺省完整20步，恢复只核对，不自动调参或更改行为链。"""
    training=config['stage10']['training'];performance=config['stage10']['performance']
    from gem.closedloop.dppo.denoising_sampling import validate_step_count
    if steps is not None:
        validate_step_count(training['denoising_steps'],steps)
        if resume and training.get('denoising_steps_per_chain',20)!=steps:
            raise ValueError('Resume cannot change timestep sampling objective')
        if not resume:training['denoising_steps_per_chain']=steps
    if sampling_strategy is not None:
        from gem.closedloop.dppo.denoising_sampling import validate_sampling
        validate_sampling(training['denoising_steps'],training.get('denoising_steps_per_chain'),sampling_strategy)
        if resume and training.get('denoising_sampling_strategy','uniform')!=sampling_strategy:
            raise ValueError('Resume cannot change denoising sampling strategy')
        if not resume:training['denoising_sampling_strategy']=sampling_strategy
    if actor_lr is not None:
        if not math.isfinite(actor_lr) or actor_lr<=0:raise ValueError('Explicit experiment lr must be finite and positive')
        if resume and training['actor_lr']!=actor_lr:raise ValueError('Resume cannot change experiment learning rate')
        if not resume:training['actor_lr']=actor_lr
    if sensitive_output_fp64:
        requested={'denoiser.final_layer.fc2':'joint_gemm_fp64'}
        if resume and performance.get('weight_reduction_overrides')!=requested:
            raise ValueError('Resume cannot change sensitive gradient precision')
        if not resume:performance['weight_reduction_overrides']=requested


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--gmt-repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--precision-mode', choices=('fp32_reference','fp32_fast','tf32_candidate','bf16_backbone_candidate'), required=True)
    parser.add_argument('--numerical-variant', choices=('default','blocked64_fp32_gemm',
        'compiled_blocked64_fp32','compensated_bf16x3','compensated_bf16x6'),
        help='显式新数值合同；省略时保留所选配置，恢复时不能修改')
    parser.add_argument('--rounds', type=int, choices=range(1,11), default=1,
                        help='有界验收1至10轮；10轮用于持续归档背压测试，不启动长期训练')
    parser.add_argument('--sampling-strategy',choices=('uniform','stratified8'))
    parser.add_argument('--archive-compression-level',type=int,choices=range(1,10),
                        help='独立无损归档吞吐对照；不删除字段，不跳过回读SHA')
    parser.add_argument('--archive-compression-backend',choices=('python','pigz'))
    parser.add_argument('--archive-compression-threads',type=int,choices=range(1,5))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--fixed-work-probe',action='store_true',help='第二轮实际行为策略上的有界微批矩阵，不是普通轮计时')
    parser.add_argument('--denoising-samples',type=int,choices=(4,8,20),
        help='显式独立PPO目标对照；行为采样及最终完整KL仍20步，缺省不改配置')
    parser.add_argument('--actor-lr',type=float,help='有限校准的显式固定值；不进行扫描或自动回退')
    parser.add_argument('--sensitive-output-fp64',action='store_true')
    parser.add_argument('--profile-world', action='store_true',help='只用于Python外围定位，该轮不作无profiler吞吐结果')
    parser.add_argument('--deadline-seconds', type=int, default=3600)
    args = parser.parse_args()
    if args.fixed_work_probe and (args.resume or args.rounds!=2):
        raise ValueError('Fixed-work probe only permits a fresh finite two-round run')
    if not 300 <= args.deadline_seconds <= 7200: raise ValueError('Finite deadline must be 300..7200 seconds')
    output = args.output.resolve()
    if args.resume:
        if not (output/'run/latest.json').is_file(): raise ValueError('Resume requires a published complete checkpoint')
        config = yaml.safe_load((output/'config.yaml').read_text())
        select_numerical_contract(config, args.precision_mode, args.numerical_variant, resume=True)
        select_learning_experiment(config,steps=args.denoising_samples,actor_lr=args.actor_lr,
            sensitive_output_fp64=args.sensitive_output_fp64,resume=True,sampling_strategy=args.sampling_strategy)
    else:
        output.mkdir(parents=True, exist_ok=False)
        config = yaml.safe_load(args.config.read_text())
        if config['runtime'].get('num_envs') not in (16,32,64) or 'scale' not in config['stage10']:
            raise ValueError('Acceptance requires an explicit 16/32/64 environment derived-scale configuration')
        config['paths'].update(genmo_repo=str(ROOT), gmt_repo=str(args.gmt_repo.resolve()),
            kinematics=str(ROOT/'configs/bumi/bumi_kinematics_robot_retargeter_fe934_v1.json'),
            compat_profile=str(args.gmt_repo.resolve()/'configs/sim2sim/model_135000_stage2.json'))
        select_numerical_contract(config, args.precision_mode, args.numerical_variant, resume=False)
        select_learning_experiment(config,steps=args.denoising_samples,actor_lr=args.actor_lr,
            sensitive_output_fp64=args.sensitive_output_fp64,resume=False,sampling_strategy=args.sampling_strategy)
        config['stage10']['performance']['python_world_profile']=args.profile_world
        if args.fixed_work_probe:config['stage10']['performance']['fixed_work_probe_iteration']=2
        # 临时归档与正式目录隔离，容量上限保持原授权。
        config['stage10']['storage']['archive_secondary']['root'] = str(output/'execution_archives')
        if args.archive_compression_level is not None:
            config['stage10']['storage']['archive_compression_level']=args.archive_compression_level
        for name in ('archive_compression_backend','archive_compression_threads'):
            if getattr(args,name) is not None:config['stage10']['storage'][name]=getattr(args,name)
        (output/'config.yaml').write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    if args.resume and args.archive_compression_level is not None and config['stage10']['storage'].get(
            'archive_compression_level',6)!=args.archive_compression_level:
        raise ValueError('Resume cannot change bound compression policy')
    for name,default in (('archive_compression_backend','python'),('archive_compression_threads',1)):
        if args.resume and getattr(args,name) is not None and config['stage10']['storage'].get(name,default)!=getattr(args,name):
            raise ValueError('Resume cannot change bound compression policy')
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
