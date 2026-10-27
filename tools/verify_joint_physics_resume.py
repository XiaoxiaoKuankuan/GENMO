"""真实完整checkpoint下一批次和下一次联合更新的精确恢复验收。

加载指定新流程checkpoint推进一步并保存到独立目录，然后比较继续推进与从该保存点
重新构造训练器后的下一次更新。逐张量核对GENMO、AdamW、scheduler、数据/物理来源时间
噪声游标、课程和全部RNG；默认至多三个更新，显式对齐物理探针时至多probe_every+2个
有界更新，保证比较步骤实际消费物理请求与噪声游标，不覆盖输入checkpoint或正式日志。
验证是否真的含物理更新独立报告，零物理样本时不能把恢复一致称为联合训练成功。
"""
import argparse
import gc
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def equal(a,b,path='root'):
    import numpy as np
    import torch
    if isinstance(a,torch.Tensor):
        if not torch.equal(a,b):raise AssertionError(path)
    elif isinstance(a,np.ndarray):
        if not np.array_equal(a,b):raise AssertionError(path)
    elif isinstance(a,dict):
        if a.keys()!=b.keys():raise AssertionError(path+'.keys')
        for k in a:equal(a[k],b[k],path+'.'+str(k))
    elif isinstance(a,(tuple,list)):
        if len(a)!=len(b):raise AssertionError(path+'.length')
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+f'[{i}]')
    elif a!=b:raise AssertionError(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--align-physics-probe',action='store_true',
                        help='保存点对齐下一次独立物理抽样，真实核对物理来源/时间/噪声恢复')
    args=parser.parse_args()
    import torch
    from gem.closedloop.joint_physics.checkpoint import read_checkpoint,save_trainer
    from gem.closedloop.joint_physics.trainer import JointTrainer
    original=read_checkpoint(args.checkpoint)
    cfg=original['config']
    del original
    torch.set_num_threads(cfg['runtime']['cpu_threads'])
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    trainer=JointTrainer(cfg,args.device,resume=args.checkpoint)
    initial_step=trainer.step
    trainer.update()
    if args.align_physics_probe:
        if trainer.physics_stream is None or not trainer.physics_stream.keys:
            raise ValueError('物理游标恢复验收需要非空合法真实请求池')
        interval=max(1,int(cfg['physics']['probe_every']))
        while trainer.step % interval:
            trainer.update()
    boundary_step=trainer.step
    save_trainer(trainer,output/'boundary.pt')
    expected=trainer.update();save_trainer(trainer,output/'continued.pt')
    if trainer.adapter is not None:trainer.adapter.bundle.assert_unchanged(trainer.frozen)
    del trainer;gc.collect()
    torch.cuda.empty_cache()
    restored=JointTrainer(cfg,args.device,resume=output/'boundary.pt')
    observed=restored.update();save_trainer(restored,output/'restored.pt')
    if restored.adapter is not None:restored.adapter.bundle.assert_unchanged(restored.frozen)
    keys=('sample_ids','timesteps','music_dropout_mask','physics_sample_ids','physics_timesteps','data_loss','execution_loss','rejections','physics_effective_updates')
    for key in keys:equal(expected[key],observed[key],key)
    if args.align_physics_probe and not observed['physics_sample_ids']:
        raise AssertionError('比较步骤没有实际消耗物理请求，不能声称物理游标恢复通过')
    left,right=read_checkpoint(output/'continued.pt'),read_checkpoint(output/'restored.pt')
    for key in ('actor','optimizer','scheduler','rank_states','curriculum','counters'):equal(left[key],right[key],key)
    (output/'verification.json').write_text(json.dumps({'passed':True,'compared_step':observed['step'],
        'initial_step':initial_step,'boundary_step':boundary_step,
        'compared_physics_requests':len(observed['physics_sample_ids']),
        'compared_physics_timesteps':observed['physics_timesteps'],
        'physics_effective_updates':observed['physics_effective_updates'],'next_batch_and_noise_exact':True,
        'model_optimizer_scheduler_rng_sampler_exact':True,'frozen_unchanged':True},indent=2))


if __name__=='__main__':main()
