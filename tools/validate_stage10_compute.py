"""服务器1真实八卡的 Stage10 算法和数值执行有限验收入口。

本脚本只运行命令行明确指定的有限检查，使用独立输出目录，输出完整JSON与
各rank计时，不启动长期训练。math阶段用CUDA核对直接高斯比值、联合KL、mask、
PPO clip及抽样归一化；policy阶段加载真实s595000和四来源Stage1条件，核对20步
行为采样与不同batch/置换/梯度模式重算；critic阶段检查预训练副本、因果输入、
时序敏感性和独立梯度。它不把示范条件数值测试称为真实物理rollout验收。

候选FP32/TF32/BF16以及compile都保留独立结果，旧1e-4/1e-3/1e-8概率门槛不变。
失败以非零退出或明确candidate_failed报告保留，不能通过放宽检查进入生产。
全量梯度/Adam及真实闭环由后续同数据入口核验，本入口不代替这些验收。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from gem.closedloop.contracts import STAGE1_CONDITION_KEYS
from gem.closedloop.dppo.gaussian_probability import gaussian_statistics, ppo_surrogate
from gem.closedloop.dppo.policy import masked_joint_log_prob
from gem.closedloop.dppo.dense_policy import DenseDiffusionPolicy
from gem.closedloop.dppo.pretrained_critic import PretrainedConditionCritic
from gem.closedloop.dppo.trainer import load_actor
from gem.closedloop.stage1_dataset import BumiClosedLoopStage1Dataset, collate_stage1_training_samples


def math_checks(device,frames=120):
    torch.manual_seed(12010)
    x=torch.randn(16,frames,30,device=device);old=torch.randn_like(x)
    new=(old+.01*torch.randn_like(old)).requires_grad_(True)
    sigma=torch.full((16,1,1),.03,device=device);current=sigma*1.003
    mask=torch.rand_like(x)>.15;mask[0]=False
    stats=gaussian_statistics(x,old,sigma,new,current,mask)
    reference=masked_joint_log_prob(x,new,current,mask)-masked_joint_log_prob(x,old,sigma,mask)
    torch.testing.assert_close(stats['joint_log_ratio'],reference,atol=1e-8,rtol=1e-10)
    kl=torch.distributions.kl_divergence(torch.distributions.Normal(old.double(),sigma.double()),
        torch.distributions.Normal(new.double(),current.double())).masked_fill(~mask,0.).sum((-2,-1))
    torch.testing.assert_close(stats['joint_kl'],kl,atol=1e-8,rtol=1e-10)
    stats['joint_log_ratio'].sum().backward();gradient=new.grad.clone();new.grad=None
    reference.sum().backward();torch.testing.assert_close(new.grad,gradient,atol=3e-5,rtol=2e-4)
    zero=gaussian_statistics(x,old,sigma,old,sigma,mask)
    assert torch.equal(zero['joint_log_ratio'],torch.zeros_like(zero['joint_log_ratio']))
    assert torch.equal(zero['joint_kl'],torch.zeros_like(zero['joint_kl']))
    controlled=dict(joint_log_ratio=torch.tensor([.2,-.2,.2,-.2],device=device,dtype=torch.float64),
        mean_coordinate_log_ratio=torch.tensor([.2,-.2,.2,-.2],device=device,dtype=torch.float64),
        free_count=torch.ones(4,device=device))
    result=ppo_surrogate(controlled,torch.tensor([1.,-1.,-1.,1.],device=device),clip=.1)
    assert result['objective_clipped'].tolist()==[True,True,False,False]
    # 穷举小总体的所有无放回子集，验证N/K权重与完整20步分母的等价期望。
    import itertools
    contributions=torch.arange(5,dtype=torch.float64)
    sampled=torch.stack([contributions[list(c)].sum()*5/2/5 for c in itertools.combinations(range(5),2)])
    assert torch.equal(sampled.mean(),contributions.mean())
    return dict(passed=True,gaussian_max_error=float((stats['joint_log_ratio']-reference).abs().max()),
        clip_branch_checked=True,without_replacement_loss_expectation_checked=True)


def load_real_conditions(actor,config,device,count,frames=120):
    data=OmegaConf.load(ROOT/'configs/closedloop/stage1_dataset_server1_fourset_90505_v1.yaml')
    data.server1_example.root=config['paths']['data_root']
    data.dataset_defaults.kinematics_path=config['paths']['kinematics']
    data.sample_contract.update(prefix_min_frames=6,prefix_max_frames=30,prefix_zero_probability=.15)
    samples=[]
    for entry in data.datasets.train.values():
        opts=OmegaConf.to_container(entry,resolve=True);opts.pop('_target_');opts['motion_frames']=frames
        ds=BumiClosedLoopStage1Dataset(**opts)
        for i in range((count+3)//4):samples.append(ds[(7*i+3)%len(ds)])
    batch=collate_stage1_training_samples(samples[:count])
    return {k:batch[k].to(device) for k in STAGE1_CONDITION_KEYS}


def probability_checks(policy,context):
    device=context['known_qpos30'].device
    generators=[torch.Generator(device=device).manual_seed(1700+i) for i in range(len(context['decision_time']))]
    start=time.perf_counter();trace=policy.sample_rollout(context,generator=generators)
    result=dict(generation_seconds=time.perf_counter()-start,variants=[],behavior_requires_grad=False)
    assert not any(trace[k].requires_grad for k in ['chain','old_means','old_stds','old_log_probs'])
    batch=len(trace['chain']);flat=batch*20
    for micro in [1,4,8,16,32,64,128]:
        maxima=torch.zeros(3,device=device,dtype=torch.float64)
        order=torch.randperm(flat,device=device,generator=torch.Generator(device=device).manual_seed(42))
        start=time.perf_counter()
        for begin in range(0,flat,micro):
            ids=order[begin:begin+micro];rows=ids//20;steps=ids%20
            conditions={k:v[rows] for k,v in context.items()}
            # 有梯度与无梯度版本分别检查，不能用采样时的detached网络输出替代重算。
            for grad in [False,True]:
                with torch.set_grad_enabled(grad):
                    params=policy.transition_parameters(conditions,trace['chain'][rows,steps],steps)
                    joint=masked_joint_log_prob(trace['chain'][rows,steps+1],params['mean'],params['std'],params['free_mask'])
                    old=trace['old_log_probs'][rows,steps]
                    independent=torch.distributions.Normal(trace['old_means'][rows,steps].double(),
                        trace['old_stds'][rows,steps].double()).log_prob(trace['chain'][rows,steps+1].double())
                    independent=independent.masked_fill(~params['free_mask'],0.).sum((-2,-1))
                    error=torch.stack(((joint-old).abs().max(),((joint-old).exp()-1).abs().max(),(independent-old).abs().max()))
                    maxima=torch.maximum(maxima,error.detach())
                    del params,joint
        torch.cuda.synchronize(device)
        record=dict(microbatch=micro,max_errors=maxima.cpu().tolist(),seconds=time.perf_counter()-start)
        record['passed']=bool((maxima<=maxima.new_tensor([1e-4,1e-3,1e-8])).all())
        result['variants'].append(record)
        if not record['passed']:
            result['passed']=False;return result
    result['passed']=True
    return result


def critic_checks(actor,context):
    device=context['known_qpos30'].device
    critic=PretrainedConditionCritic(actor).to(device)
    assert not any(p.requires_grad for m in critic.encoder_modules() for p in m.parameters())
    for name in ['music_embedder','cond_exists_embedder','history_encoder','prefix_encoder']:
        a,c=getattr(actor,name),getattr(critic,name)
        if a is not None:
            for p,q in zip(a.parameters(),c.parameters()):assert p.data_ptr()!=q.data_ptr() and torch.equal(p,q)
    remaining=torch.full((len(context['decision_time']),),20.,device=device)
    critic.train();features=critic.encode_frozen(context)
    before=critic.forward_encoded(features,remaining)
    expected=critic(context,remaining)
    torch.testing.assert_close(before,expected,atol=1e-7,rtol=1e-6)
    before.square().mean().backward()
    assert all(p.grad is None for p in actor.parameters())
    assert all(p.grad is None for m in critic.encoder_modules() for p in m.parameters())
    assert any(p.grad is not None for p in critic.temporal.parameters())
    shuffled=dict(context);shuffled['music_features']=context['music_features'].flip(1)
    with torch.no_grad():delta=(critic(shuffled,remaining)-expected).abs()
    try:critic({**context,'future_reward':remaining},remaining)
    except ValueError:pass
    else:raise AssertionError('Critic accepted future privileged input')
    return dict(passed=True,independent_parameters=True,frozen_encoders=True,
        music_order_max_value_change=float(delta.max()),cached_features_equal=True,actor_gradient_isolated=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['math','policy','critic'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--asset-root',type=Path,default=Path('/data0/user/liwei/closedloop_stage10_v3_assets/20261012'))
    parser.add_argument('--precision',choices=['native_fp32','tf32','bf16_backbone'],default='native_fp32')
    parser.add_argument('--attention',default='sdpa_math')
    parser.add_argument('--compile',action='store_true')
    parser.add_argument('--linear-execution',default='native_gemm')
    parser.add_argument('--chains',type=int,default=8)
    parser.add_argument('--frames',type=int,choices=[60,120],default=60)
    args=parser.parse_args();rank=int(os.environ['RANK']);device=torch.device('cuda:'+os.environ['LOCAL_RANK'])
    torch.cuda.set_device(device);torch.set_num_threads(4);torch.manual_seed(42)
    args.output.mkdir(parents=True,exist_ok=True)
    record=dict(phase=args.phase,rank=rank,precision=args.precision,attention=args.attention,compile=args.compile,
        linear_execution=args.linear_execution,generation_horizon_frames=args.frames,
        conditions_scope='four_source_real_Stage1_demonstration_not_physical_rollout')
    try:
        if args.phase=='math':result=math_checks(device,args.frames)
        else:
            assets=args.asset_root/'stage1_s595000'
            config=dict(paths=dict(checkpoint=str(assets/'s595000.pt'),stats=str(assets/'qpos30_train_stats.json'),
                kinematics=str(assets/'bumi_kinematics_robot_retargeter_fe934_v1.json'),
                data_root='/data0/user/liwei/datasets/bumi_music_umr70_mine_pass_90505_v1'),runtime=dict(genmo_device=str(device)))
            actor,_,loading=load_actor(config)
            context=load_real_conditions(actor,config,device,args.chains,args.frames)
            if args.phase=='critic':result=critic_checks(actor,context)
            else:
                policy=DenseDiffusionPolicy(actor,std_schedule=[.001]*16+[.02,.025,.03,.03],
                    precision=args.precision,attention=args.attention,compile_denoiser=args.compile,
                    linear_execution=args.linear_execution,generation_frames=args.frames)
                actual_lengths=set()
                hooks=[b.register_forward_pre_hook(lambda _,a:actual_lengths.add(a[0].shape[1])) for b in actor.denoiser.blocks]
                result=probability_checks(policy,context)
                for h in hooks:h.remove()
                if actual_lengths!={args.frames}:raise AssertionError('Transformer computed a different horizon')
                result['actual_Transformer_lengths']=sorted(actual_lengths)
                result['pretrained_maximum_frames']=actor.denoiser.max_len
            record['source_global_step']=loading['source_global_step']
        record.update(result)
    except BaseException as error:
        import traceback
        record.update(passed=False,error=repr(error),traceback=traceback.format_exc())
    record['peak_gpu_allocated_bytes']=torch.cuda.max_memory_allocated(device)
    (args.output/f'rank{rank:02d}.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(record,ensure_ascii=False),flush=True)
    return 0 if record.get('passed') else 1


if __name__=='__main__':raise SystemExit(main())
