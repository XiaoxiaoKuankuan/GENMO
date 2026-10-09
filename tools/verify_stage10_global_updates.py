"""服务器1真实八卡NCCL下的PPO、分布式BC与完整Critic epochs数学验收。

使用可手工重算的小高斯模型隔离八卡SUM/全局均值缩放，比较八次真实SGD更新与
单GPU完整数据参考，覆盖仅BC参数、全局未用参数、固定old概率、每步128监督样本
及Critic四个无放回epoch。SGD保留尺度差异，避免Adam首步的归一化掩盖多乘/除8。
另验证当前前向KL软停止确实不执行待更新步骤、不消费BC或优化尝试预算。

这是通信与算法回归，不是真实GENMO吞吐、机器人反馈或8192闭环验收。所有rank
绑定对应GPU，输出仅写入新的测试目录，不修改训练权重、旧证据或运行任务。
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import sys
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from tests.closedloop.dppo.test_updater_v2 import BatchGaussianPolicy, rows
from tests.closedloop.dppo.test_distributed_training import SmallCritic
from gem.closedloop.dppo.distributed_runtime import DistributedCollectives
from gem.closedloop.dppo.updater_v2 import actor_update_v2, critic_update_local, balanced_epoch_order


class ShardedAnchor:
    def __init__(self, device, rank=None):
        data = torch.linspace(-.7,1.1,128,device=device)
        self.data = data if rank is None else data[rank*16:(rank+1)*16]
        self.calls = 0; self.distributed_global_mean = rank is not None
        self.generator = torch.Generator().manual_seed(491+(rank or 0))
    def backward(self, actor, weight):
        self.calls += 1
        loss = (actor.denoiser.weight[0,0]*self.data+actor.bc_only).square().mean()
        (weight*loss).backward()
        return dict(loss=float(loss.detach()),batch_size=len(self.data),bc_update_steps=self.calls)
    def state_dict(self):
        return dict(calls=self.calls,generator=self.generator.get_state())
    def load_state_dict(self,value):
        self.calls=value['calls'];self.generator.set_state(value['generator'])


def compare(actual, expected):
    for name,value in actual.state_dict().items():
        torch.testing.assert_close(value,expected.state_dict()[name],atol=2e-7,rtol=2e-6)


def check_rank_local_rollback(output, collective, policy, optimizer, critic, critic_optimizer, anchor, local):
    """真实广播回滚，故障注入仅替代昂贵网络；保留八个不同BC游标与已消耗预算。"""
    from gem.closedloop.dppo import parallel_training as training
    from gem.closedloop.dppo.rollback_audit import exact_state_digest
    from gem.closedloop.dppo.run_management import TrainingBudget
    from tools.train_closedloop_stage10 import configuration
    config=configuration(Path(__file__).resolve().parents[1]/'configs/closedloop/stage10_8gpu_server1_scale8192.yaml')
    budget=(TrainingBudget(output/'rollback_budget.json',dict(accepted_iterations=2,optimizer_attempts=8,
        generations=20,control_steps=100,physics_steps=400)) if collective.rank==0 else None)
    before=exact_state_digest(anchor.state_dict())
    fingerprints=collective.all_gather_object(before)
    assert len(set(fingerprints))==8
    context=SimpleNamespace(distributed=collective,initial_iteration=0,stage=config['stage10'],
        settings=config['stage9'],policy=policy,actor=policy.actor,critic=critic,
        actor_optimizer=optimizer,critic_optimizer=critic_optimizer,bc=anchor,budget=budget,session=output,
        profile={'microbatch':7},tensor_cache=None,
        generators={name:torch.Generator().manual_seed(82+collective.rank) for name in ('actor','critic')})
    def mutate(module,opt):
        opt.zero_grad(set_to_none=True)
        sum(parameter.square().sum() for parameter in module.parameters() if parameter.requires_grad).backward()
        opt.step()
    def critic_step(*args,**kwargs):
        mutate(critic,critic_optimizer);return dict(optimizer_steps=1)
    def actor_step(*args,**kwargs):
        kwargs['reserve_attempt']();mutate(policy.actor,optimizer)
        anchor.calls+=1;torch.rand(7,generator=anchor.generator)
        torch.rand(3,device=collective.device)
        return dict(optimizer_steps=1)
    with patch.multiple(training,probability_check_local=lambda *a,**k:dict(passed=True),
            critic_update_local=critic_step,actor_update_v2=actor_step,
            analytic_kl_local=lambda *a,**k:dict(mean_joint_kl=.04,per_denoising_step=[dict(mean_joint_kl=.04)]*20)):
        try:training._update_cached(context,SimpleNamespace(transitions=local),{},[],1)
        except RuntimeError as error:assert 'Final whole-rollout KL rejected' in str(error)
        else:raise AssertionError('Injected final KL rejection did not reject')
    assert exact_state_digest(anchor.state_dict())==before
    if collective.rank==0:
        assert budget.state_dict()['used']['optimizer_attempts']==1
        rejected=json.loads((output/'rejected_000001.json').read_text())
        assert rejected['rolled_back'] and len(rejected['rollback_audit_by_rank'])==8
    return dict(distinct_rank_BC_streams=8,all_local_BC_states_restored=True,
        actor_critic_optimizer_rng_exact_audit=True,spent_attempts_not_refunded=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); rank=int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE'])!=8: raise ValueError('Requires eight Server1 GPU ranks')
    torch.cuda.set_device(rank); torch.set_num_threads(1); device=f'cuda:{rank}'
    dist.init_process_group('gloo',timeout=timedelta(minutes=5))
    group=dist.new_group(backend='nccl',timeout=timedelta(minutes=5))
    collectives=DistributedCollectives(rank,8,device=device,tensor_group=group)
    if rank==0:args.output.mkdir(parents=True,exist_ok=False)
    dist.barrier()
    policy=BatchGaussianPolicy(); all_rows=rows(policy,32)
    old=torch.stack([row.old_log_prob.clone() for row in all_rows])
    policy.actor.to(device); reference=copy.deepcopy(policy)
    local=all_rows[rank*4:(rank+1)*4]
    advantages=torch.linspace(-1.1,1.3,32,dtype=torch.float64)
    manifest=[dict(owner_rank=i//4,local_index=i%4,valid=True,has_free=True) for i in range(32)]
    orders=[balanced_epoch_order(list(range(32)),manifest,torch.Generator().manual_seed(19+e)) for e in range(2)]
    anchor,reference_anchor=ShardedAnchor(device,rank),ShardedAnchor(device)
    optimizer=torch.optim.SGD(policy.actor.parameters(),lr=.0002,momentum=.7)
    reference_optimizer=torch.optim.SGD(reference.actor.parameters(),lr=.0002,momentum=.7)
    arguments=dict(ppo_epochs=2,epoch_orders=orders,actor_minibatch_internal_transitions=16,
        max_optimizer_steps=8,denoising_microbatch=7,bc_weight=.1,grad_clip_norm=1000.,
        soft_kl_limit=None,kl_check_mode='pre_step_plus_final',verify_initial_probability=True)
    expected=actor_update_v2(reference,reference_optimizer,all_rows,dict(advantages=advantages),bc=reference_anchor,**arguments)
    actual=actor_update_v2(policy,optimizer,local,dict(advantages=advantages[rank*4:(rank+1)*4]),
        bc=anchor,global_manifest=manifest,distributed=collectives,**arguments)
    compare(policy.actor,reference.actor)
    assert actual['optimizer_steps']==8 and actual['bc_global_samples']==1024 and anchor.calls==8
    assert torch.equal(old,torch.stack([row.old_log_prob for row in all_rows]))
    critic,expected_critic=SmallCritic().to(device),SmallCritic().to(device)
    returns=torch.linspace(-.8,1.2,32,device=device)
    critic_optimizer=torch.optim.SGD(critic.parameters(),lr=.0001,momentum=.5)
    expected_optimizer=torch.optim.SGD(expected_critic.parameters(),lr=.0001,momentum=.5)
    generator=torch.Generator().manual_seed(94)
    for _ in range(4):
        order=balanced_epoch_order(list(range(32)),manifest,generator)
        for start in range(0,32,8):
            indices=order[start:start+8]; selected=[all_rows[i] for i in indices]
            context={key:torch.cat([row.context[key] for row in selected]).to(device) for key in selected[0].context}
            remaining=torch.tensor([row.metadata['remaining_music_seconds'] for row in selected],device=device)
            expected_optimizer.zero_grad(set_to_none=True)
            loss=.5*(expected_critic(context,remaining)-returns[indices]).square().mean();loss.backward()
            torch.nn.utils.clip_grad_norm_(expected_critic.parameters(),1000.)
            expected_optimizer.step()
    fitted=critic_update_local(critic,critic_optimizer,local,dict(returns=returns[rank*4:(rank+1)*4]),
        global_manifest=manifest,distributed=collectives,epochs=4,batch_size=8,grad_clip_norm=1000.,
        generator=torch.Generator().manual_seed(94))
    compare(critic,expected_critic)
    assert fitted['optimizer_steps']==16 and fitted['global_sample_visits']==128
    before=copy.deepcopy(policy.actor.state_dict()); attempts=[]; before_bc=anchor.calls
    stopped=actor_update_v2(policy,optimizer,local,dict(advantages=advantages[rank*4:(rank+1)*4]),
        global_manifest=manifest,distributed=collectives,bc=anchor,reserve_attempt=lambda:attempts.append(True),
        **dict(arguments,verify_initial_probability=False,soft_kl_limit=1e-14))
    assert stopped['optimizer_steps']==0 and not attempts and anchor.calls==before_bc
    for name,value in policy.actor.state_dict().items():torch.testing.assert_close(value,before[name],atol=0,rtol=0)
    rollback=check_rank_local_rollback(args.output,collectives,policy,optimizer,critic,critic_optimizer,anchor,local)
    output=dict(rank=rank,status='passed',actor_steps=actual['optimizer_steps'],bc_samples=actual['bc_global_samples'],
        critic_steps=fitted['optimizer_steps'],critic_sample_visits=fitted['global_sample_visits'],
        pre_step_soft_stop_without_parameter_BC_or_budget_update=True,
        rollback=rollback,
        scope='eight_GPU_NCCL_small_model_global_mean_correctness_not_real_rollout_throughput')
    (args.output/f'rank{rank:02d}.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output),flush=True)
    dist.destroy_process_group(group);dist.destroy_process_group()


if __name__=='__main__':main()
