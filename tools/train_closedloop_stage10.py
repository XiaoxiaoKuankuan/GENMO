"""服务器1当前 Stage10 八卡 Isaac 同进程训练入口，默认只做有限训练。

先用Gloo在任何CUDA/Isaac上下文创建前核验八张卡空闲、模型SHA和独立输出目录，
再为每rank启动一个Isaac App和对应GPU世界，NCCL同步DPPO/BC/Critic梯度。
Actor初始化s595000，冻结原生GMT140500；本入口直接维护当前训练实现和恢复格式。
新配置最多500轮预算不自动开启，必须显式指定stop-after-iteration。正常结束
保存一致checkpoint、运行固定评估、关闭物理世界和通信组；异常返回真实非零码。

GENMO和Isaac必须使用同一Python3.11测试运行环境，不能混用3.10二进制包。
App初始化期间暂时隐藏分布式环境变量，避免Kit误启动额外分布式资源；GPU编号
仍明确按CUDA_VISIBLE_DEVICES全局0..7与local_rank匹配，启动后恢复通信身份。
本入口不停止已有作业、不覆盖任何旧run，也不自动重启失败任务。
"""
from __future__ import annotations
import argparse
from datetime import timedelta
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/closedloop/stage10_server1.yaml')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--stop-after-iteration',type=int,default=2)
    parser.add_argument('--warmup-rollouts',type=int)
    parser.add_argument('--resume',choices=['latest'])
    parser.add_argument('--strict-evidence',action='store_true')
    parser.add_argument('--calibrate-only',action='store_true',help='只测新模型单机器人关键路径，不更新参数')
    parser.add_argument('--evaluate-only',action='store_true',help='新实验只做固定物理评估，用于实际噪声核标定，不采训练rollout')
    parser.add_argument('--evaluation-actor',type=Path,help='已通过固定真实数据标定的独立Actor权重，只允许evaluate-only使用')
    parser.add_argument('--critic-preheat-only',action='store_true',help='固定Actor，完成完整MC/TD价值对照后结束，不进入Actor更新')
    parser.add_argument('--collect-only',action='store_true',help='只采一轮完整真实rollout供计算对照，不更新Actor/Critic')
    parser.add_argument('--capture-validation-iteration',type=int,help='明确保存指定轮更新前的真实学习张量及Adam用于固定数据对照')
    parser.add_argument('--genmo-repo',type=Path,default=ROOT)
    parser.add_argument('--gmt-repo',type=Path)
    args=parser.parse_args()
    # 单机八卡使用已有验收的通信环境，不依赖当前shell的隐式NCCL默认值。
    for key,value in dict(NCCL_CUMEM_HOST_ENABLE='0',NCCL_IB_DISABLE='1',
                         NCCL_SOCKET_IFNAME='lo',TORCH_NCCL_BLOCKING_WAIT='1').items():
        os.environ.setdefault(key,value)
    import torch
    import torch.distributed as dist
    from gem.closedloop.dppo.training_config import configuration
    config=configuration(args.config,genmo_repo=args.genmo_repo,gmt_repo=args.gmt_repo)
    if args.calibrate_only:
        config['runtime'].update(num_envs=1,generation_batch=1,phase_contract='single_request_calibration.v1')
    if sum((args.calibrate_only,args.collect_only,args.evaluate_only,args.critic_preheat_only))>1:parser.error('Choose only one finite diagnostic phase')
    if args.evaluate_only and args.resume:parser.error('Noise calibration requires a fresh explicit experiment identity')
    if args.evaluation_actor and not args.evaluate_only:parser.error('Fixed-data Actor candidates are evaluation-only, not training recovery points')
    if args.collect_only:args.capture_validation_iteration=1;args.warmup_rollouts=0
    rank,world,local=(int(os.environ.get(k,'-1')) for k in ['RANK','WORLD_SIZE','LOCAL_RANK'])
    if world!=8 or rank!=local or not 0<=rank<8 or os.environ.get('CUDA_VISIBLE_DEVICES')!='0,1,2,3,4,5,6,7':
        parser.error('Use one-node torchrun with eight GPUs explicitly visible in global order')
    from gem.closedloop.dppo.run_control import validate_stopping_iteration
    try:validate_stopping_iteration(args.stop_after_iteration,config['stage10'])
    except ValueError as error:parser.error(str(error))
    dist.init_process_group('gloo',timeout=timedelta(minutes=30))
    preflight=[None]
    if rank==0:
        try:
            busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
            if busy:raise RuntimeError('Eight GPU resource check found an existing job: '+busy)
            if args.output_dir.exists() and not args.resume:raise FileExistsError(args.output_dir)
            from gem.closedloop.dppo.training_checkpoint import file_sha256
            for key,expected in config['stage10']['assets'].items():
                if file_sha256(config['paths'][key])!=expected:raise ValueError('Asset SHA mismatch: '+key)
            preflight[0]=dict(ok=True)
        except BaseException as e:preflight[0]=dict(ok=False,error=repr(e))
    dist.broadcast_object_list(preflight,src=0)
    if not preflight[0]['ok']:raise RuntimeError(preflight[0]['error'])
    gmt=Path(config['paths']['gmt_repo'])
    for path in [gmt/'rsl_rl',gmt/'source/NoetixRobot',gmt/'scripts',gmt/'scripts/motion',gmt/'scripts/rsl_rl']:
        sys.path.insert(0,str(path))
    os.environ.setdefault('ISAACLAB_PATH',config['runtime']['isaaclab_root'])
    libraries=config['runtime']['sysroot_library']
    if libraries not in os.environ.get('LD_LIBRARY_PATH','').split(os.pathsep):
        os.environ['LD_LIBRARY_PATH']=libraries+os.pathsep+os.environ.get('LD_LIBRARY_PATH','')
    from isaaclab.app import AppLauncher
    app_parser=argparse.ArgumentParser(add_help=False);AppLauncher.add_app_launcher_args(app_parser)
    app_args=app_parser.parse_args([]);app_args.headless=True;app_args.device=f'cuda:{local}'
    from vector_app_resources import configure_vector_app
    configure_vector_app(app_args)
    removed={k:os.environ.pop(k) for k in list(os.environ)
        if k in ['RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','GROUP_RANK','ROLE_RANK',
                 'ROLE_WORLD_SIZE','MASTER_ADDR','MASTER_PORT'] or k.startswith('TORCHELASTIC_')}
    visible_devices=os.environ['CUDA_VISIBLE_DEVICES']
    try:app=AppLauncher(app_args).app
    finally:
        os.environ.update(removed)
        # Kit启动器会清理可见卡变量；恢复原八卡声明供隔离评估正确绑定全局GPU。
        os.environ['CUDA_VISIBLE_DEVICES']=visible_devices
    code=1;tensor_group=None
    try:
        import NoetixRobot.tasks
        torch.cuda.set_device(local);torch.set_num_threads(config['runtime']['torch_threads'])
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        tensor_group=dist.new_group(backend='nccl',timeout=timedelta(minutes=15))
        from gem.closedloop.dppo.collectives import DistributedCollectives
        distributed=DistributedCollectives(rank,world,tensor_group=tensor_group,device=f'cuda:{local}')
        if float(distributed.sum_tensor(torch.tensor(rank+1.,device=distributed.device)))!=36.:
            raise RuntimeError('Eight-GPU SUM proof failed')
        from gem.closedloop.dppo.training import run
        code=run(args,config,distributed)
    finally:
        if tensor_group is not None:dist.destroy_process_group(tensor_group)
        dist.destroy_process_group()
        import omni.kit.app
        omni.kit.app.get_app().post_quit(code);app.close()
    return code


if __name__=='__main__':raise SystemExit(main())
