"""从完整八卡断点读取抽样游标，在CPU顺序计算一次等规模全局梯度诊断。

每rank使用原256数据样本、4个Future样本、4个真实物理请求，复用全部Stage1损失、
Future辅助权重和H32执行链，最后按8rank平均及全局有效物理样本数汇总。原批量不拆成
微批，避免带mask/分位数的损失归约发生变化。不创建AdamW、不改参数、不启动Newton。
数据/噪声/请求来自断点独立RNG状态；CPU神经网络dropout使用固定新种子，因此不是
CUDA运行的逐位精确重放，也不包含AdamW预条件后的实际参数更新。所有结果明确此边界。
同时测量对应物理请求的真实缓存与原数据参考基线，用于区别旧跟踪误差和学生新增误差。
这是只读低优先级CPU任务，绝不占用正在训练的8张GPU。
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import torch

from tools.audit_joint_physics_plateau import gradient_stats, scalar


def accumulate(total, gradients, scale):
    """CPU逐rank累加与DDP平均同一标量系数；允许某参数无当前分支梯度。"""
    for i, gradient in enumerate(gradients):
        if gradient is None:
            continue
        if total[i] is None:
            total[i]=gradient.detach().float().clone().mul_(scale)
        else:
            total[i].add_(gradient.detach().float(),alpha=scale)


def run(args):
    from gem.closedloop.joint_physics.config import read_config,build_actor,build_adapter,seed_all
    cfg=read_config(args.config);sys.path.insert(0,cfg['supermimic']['repo'])
    from gem.closedloop.joint_physics.checkpoint import read_checkpoint
    from gem.closedloop.joint_physics.contracts import execution_contract,PhysicsRejected
    from gem.closedloop.joint_physics.data import JointDataStream,SidecarStore
    from gem.closedloop.joint_physics.reference import attach_supervised_sidecars,attach_sidecars
    from gem.closedloop.joint_physics.requests import PhysicsRequestStream
    from gem.closedloop.joint_physics.prepare_rollouts import asset_contract
    from gem.closedloop.joint_physics.execution import ValidatedSupport,DiagnosticSupport,execute_window
    from gem.closedloop.joint_physics.losses import execution_terms
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    seed_all(args.seed)
    cfg['data']['loading'].update(workers=0,prefetch_batches=0,pin_memory=False,sequence_cache_mb=4096)
    checkpoint=read_checkpoint(args.checkpoint)
    actor,losses=build_actor(cfg,'cpu');actor.load_state_dict(checkpoint['actor'],strict=True)
    adapter=build_adapter(cfg,'cpu');frozen=adapter.bundle.frozen_snapshot();assets=asset_contract(cfg,adapter)
    store=SidecarStore(cfg['data']['sidecar'],'train',assets,sample_contract=cfg['data']['sample_contract'],cache_bytes=2*1024**3)
    support=ValidatedSupport(cfg['physics']['validation_report'],assets,32,'cpu',execution_contract(cfg))
    named=[(n,p) for n,p in actor.named_parameters() if p.requires_grad];params=[p for _,p in named]
    gd,gf,gp=([None]*len(params) for _ in range(3));world=len(checkpoint['rank_states'])
    valid=0;rows=[];step=checkpoint['global_step']
    def emit(row):
        rows.append(row)
        with (output/'rows.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        print(json.dumps(row,ensure_ascii=False),flush=True)
    for rank,state in enumerate(checkpoint['rank_states']):
        stream=JointDataStream(cfg,rank=rank,world_size=world);stream.load_state_dict(state['data'])
        seed_all(args.seed+rank)
        try:
            b=cfg['training']['batch_size'];generator=torch.Generator().set_state(state['noise'])
            data=stream.next(b,'cpu')
            data,contexts,_=attach_supervised_sidecars(data,store,adapter,actor.stage1.endecoder.codec,32,'cpu')
            actor.train()
            out=actor.denoise_supervised(data,contexts,torch.randint(1000,(b,),generator=generator),
                torch.randn(b,60,30,generator=generator),torch.rand(b,generator=generator)<actor.stage1.music_mask_prob)
            loss,_=losses(data,out['pred_x_start'],out['static_conf_logits'],step)
            accumulate(gd,torch.autograd.grad(loss,params,allow_unused=True),1/world)
            emit({'rank':rank,'kind':'data','batch':b,'loss':scalar(loss),'elapsed':time.perf_counter()-started})
            del out,loss,data,contexts
            future=PhysicsRequestStream(stream.catalog,store,adapter.bundle.spec.joint_names,1,
                cfg['training']['seed']+100000,rank,historical_mode='diagnostic',cache_directory=cfg['data']['loading']['request_cache'])
            future.load_state_dict(state['future_data']);fb=cfg['future_supervision']['batch_size']
            fbatch,_=future.next(fb,'cpu');fbatch,fc,_=attach_supervised_sidecars(fbatch,store,adapter,actor.stage1.endecoder.codec,1,'cpu')
            generator=torch.Generator().set_state(state['future_noise'])
            out=actor.denoise_supervised(fbatch,fc,torch.randint(1000,(fb,),generator=generator),
                torch.randn(fb,60,30,generator=generator),torch.rand(fb,generator=generator)<actor.stage1.music_mask_prob)
            loss,_=losses(fbatch,out['pred_x_start'],out['static_conf_logits'],step)
            accumulate(gf,torch.autograd.grad(loss,params,allow_unused=True),1/world)
            emit({'rank':rank,'kind':'future','batch':fb,'loss':scalar(loss)})
            del out,loss,fbatch,fc,future
            physical=PhysicsRequestStream(stream.catalog,store,adapter.bundle.spec.joint_names,32,cfg['training']['seed'],rank,
                support,historical_mode='diagnostic',noise_levels=cfg['physics']['noise_levels'],cache_directory=cfg['data']['loading']['request_cache'])
            physical.load_state_dict(state['physics_data']);pb=cfg['physics']['subbatch']
            batch,meta=physical.next(pb,'cpu');batch,contexts,windows,reasons=attach_sidecars(batch,store,adapter,actor.stage1.endecoder.codec,32)
            generator=torch.Generator().set_state(state['physics_noise'])
            levels=torch.tensor([m['timesteps'][int(torch.randint(len(m['timesteps']),(1,),generator=generator))] for m in meta])
            noise=torch.randn(pb,60,30,generator=generator);actor.eval()
            out=actor.denoise_supervised(batch,contexts,levels,noise,torch.zeros(pb,dtype=torch.bool))
            total=out['pred_x_start'].sum()*0;accepted=0
            for i,window in enumerate(windows):
                try:
                    if window is None:raise PhysicsRejected(reasons[i])
                    result=execute_window(window,out['pred_x_start'][i:i+1],out['static_conf_logits'][i:i+1],actor,adapter,cfg['execution_loss'],
                        support=DiagnosticSupport(support,meta[i]['tags']),training_policy=cfg['physics']['training_safety'])
                    total=total+result['loss'];accepted+=1
                    with torch.no_grad():
                        adapted=actor.stage1.adapt_conditions(batch)
                        gt=actor.stage1._constrain(out['target_x_start'],adapted)
                        base=execute_window(window,gt[i:i+1],out['static_conf_logits'][i:i+1].detach(),actor,adapter,cfg['execution_loss'],
                            support=DiagnosticSupport(support,meta[i]['tags']),training_policy=cfg['physics']['training_safety'])
                        episode=store.episode(batch['meta'][i]);start=window.start.tick//12;real=[]
                        for j in range(32):
                            ref,_,_=window.target.read_current_future(window.start.tick+(j+1)*12,0)
                            real.append(execution_terms(episode['state'][start+j+1:start+j+2],ref,adapter.runtime.kin,cfg['execution_loss'])[0])
                    emit({'rank':rank,'kind':'physics','source':meta[i]['source'],'frame':meta[i]['frame'],'k':int(levels[i]),
                        'loss':scalar(result['loss']),'gt_reference_loss':scalar(base['loss']),
                        'real_tracking_loss':scalar(torch.stack(real).mean()),
                        'terms':{k:scalar(v) for k,v in result['terms'].items()},'coverage':result['coverage']})
                except PhysicsRejected as error:emit({'rank':rank,'kind':'rejected','source':meta[i]['source'],'reason':error.reason})
            accumulate(gp,torch.autograd.grad(total,params,allow_unused=True),1.)
            valid+=accepted
            del out,total,batch,contexts,windows,physical
        finally:stream.close()
    for g in gp:
        if g is not None:g.div_(max(1,valid))
    gs=[None if a is None and b is None else ((a if a is not None else torch.zeros_like(b))+
        cfg['future_supervision']['weight']*(b if b is not None else torch.zeros_like(a))) for a,b in zip(gd,gf)]
    summary={'checkpoint':args.checkpoint,'step':step,'world_size':world,'data_samples':world*cfg['training']['batch_size'],
        'future_samples':world*cfg['future_supervision']['batch_size'],'physics_samples':valid,
        'data_vs_execution':gradient_stats(named,gd,gp,cfg['physics']['lambda_exec']),
        'data_future_vs_execution':gradient_stats(named,gs,gp,cfg['physics']['lambda_exec']),
        'seconds':time.perf_counter()-started,'network_precision':actor.precision,'device':'cpu',
        'evidence':'same_checkpoint_sampler_states_and_global_loss_normalization; CPU_dropout_and_BF16_not_bitwise_CUDA_replay',
        'optimizer_step_performed':False,'frozen_unchanged':True}
    adapter.bundle.assert_unchanged(frozen)
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--threads',type=int,default=8);p.add_argument('--seed',type=int,default=20261031)
    run(p.parse_args())
