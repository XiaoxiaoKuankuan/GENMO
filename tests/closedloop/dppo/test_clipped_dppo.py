"""完整链Clipped PPO真实小网络、变长尾批和分布式归约回归。

复用Stage1 Actor、15+4+1行为采样、真实高斯概率及生产更新器，验证5条链按2/2/1
更新三次、每次四步、旧行为统计不变及BC总数。两进程场景故意让rank1没有任何
PPO样本但仍贡献BC；与单进程相同非空Adam初态对照，检测重复world平均或空rank
漏同步。另验证soft KL只丢弃尚未执行的更新。此测试不声称真实物理收益。
"""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.closedloop.test_stage1_actor import actor_factory, _activate_branches, _conditions
from tests.closedloop.dppo.test_asymmetric_critic import tracking_context
from gem.closedloop.contracts import STAGE1_CONDITION_KEYS
from gem.closedloop.dppo.frozen_prefix import FrozenPrefixPolicy
from gem.closedloop.dppo.rollout_helpers import split_trace
from gem.closedloop.dppo.tensor_cache import RolloutTensorCache
from gem.closedloop.dppo.collectives import DistributedCollectives
from gem.closedloop.dppo.gradient_merge import BCGradientController
from gem.closedloop.dppo.trainer import trainable_actor_parameters
from gem.closedloop.dppo.updater import actor_update


class Anchor:
    def __init__(self):self.batch_size=0
    def state_dict(self):return dict(batch_size=self.batch_size)
    def load_state_dict(self,state):self.batch_size=state['batch_size']

    def backward(self,actor,weight):
        loss=sum(p.square().mean() for p in actor.parameters() if p.requires_grad)
        if self.batch_size:(loss*weight).backward()
        return dict(batch_size=self.batch_size,loss=float(loss.detach()),
            samples=[[{'dataset_id':'fixture'}]*self.batch_size])


def case(actor_factory):
    actor,batch=actor_factory(motion_frames=60);_activate_branches(actor)
    policy=FrozenPrefixPolicy(actor,std_schedule=[.001]*15+[.04,.06,.08,.1,0.],generation_frames=60)
    privileged=tracking_context(_conditions(batch));context={k:privileged[k] for k in STAGE1_CONDITION_KEYS}
    trace=policy.sample_rollout(context,generator=torch.Generator().manual_seed(17))
    rows=[]
    for j in range(5):
        i=j%2
        rows.append(SimpleNamespace(context={k:v[i:i+1].clone() for k,v in context.items()},identity={},
            chain=trace['chain'][i].clone(),old_log_prob=trace['old_log_probs'][i].clone(),free_mask=trace['free_mask'][i].clone(),
            metadata=dict(critic_context={k:v[i:i+1].clone() for k,v in privileged.items()},
                sampler_trace=split_trace(trace,i,2),remaining_music_seconds=5.)))
    targets=dict(advantages=torch.tensor([1.,-1.,.4,-.3,.2]),valid=torch.ones(5,dtype=torch.bool))
    return policy,rows,targets


def run(rank,world,actor,rows,targets,directory,soft_kl=None,adaptive=False,grouped=False):
    torch.set_num_threads(1)
    # spawn会通过共享内存传递CPU Tensor；每个rank必须独占参数，避免重复原位Adam。
    actor=copy.deepcopy(actor)
    if actor.train_diffusion is None:
        actor.train_diffusion=actor._diffusion(actor.diffusion_steps)
    policy=FrozenPrefixPolicy(actor,std_schedule=[.001]*15+[.04,.06,.08,.1,0.],generation_frames=60)
    dist.init_process_group('gloo',init_method=(Path(directory)/'gloo').as_uri(),rank=rank,world_size=world)
    manifest=[dict(owner_rank=0,local_index=i,valid=True,has_free=True) for i in range(len(rows))]
    local=rows if rank==0 else []
    local_targets=targets if rank==0 else {k:v[:0] for k,v in targets.items()}
    cache=RolloutTensorCache(local,local_targets,'cpu')
    from gem.closedloop.dppo.actor_optimizer import make_actor_optimizer
    options=dict(actor_lr=1e-7,actor_trainable_scope='all')
    if grouped:options['actor_lr_multipliers']=dict(transformer=.1,qpos_output=1.,other=.1,transformer_overrides={})
    opt=make_actor_optimizer(policy.actor,options)
    # 真实非空Adam初态：零梯度step只建立动量，不改变Actor及old statistics。
    for p in trainable_actor_parameters(policy.actor):p.grad=torch.zeros_like(p)
    opt.step();opt.zero_grad(set_to_none=True)
    before={k:cache.host[k].clone() for k in ('chain','old_log_prob','old_means','old_stds') if k in cache.host}
    attempts=[]
    try:
        report=actor_update(policy,opt,local,local_targets,cache,manifest,
            BCGradientController(initialize_from_gradient=True),distributed=DistributedCollectives(rank,world),
            anchor=Anchor(),generator=torch.Generator().manual_seed(4),minibatch_chains=2,microbatch=3,
            bc_samples_per_rollout=7,soft_kl=soft_kl,reserve_attempt=lambda:attempts.append(1),
            detailed_diagnostics=True,
            trust_region=dict(maximum_lr=1e-7,growth=1.25,margin=.8,max_attempts=8) if adaptive else None,
            kl_limits=dict(chain_joint_mean=.2,chain_joint_p95=.6,chain_joint_p99=.9,chain_joint_max=2.) if adaptive else None)
        assert len(attempts)==report['attempted_optimizer_steps']
        assert all(torch.equal(v,cache.host[k]) for k,v in before.items())
        assert policy.assert_frozen_identity()['checked']
        torch.save(dict(actor=policy.actor.state_dict(),optimizer=opt.state_dict(),report=report),
                   Path(directory)/f'rank{rank}.pt')
    finally:
        cache.close();dist.destroy_process_group()


@pytest.mark.parametrize('grouped',[False,True])
def test_multistep_tail_empty_rank_bc_and_nonempty_adam(tmp_path,actor_factory,grouped):
    policy,rows,targets=case(actor_factory)
    initial=copy.deepcopy(policy.actor.state_dict())
    one=tmp_path/'one';two=tmp_path/'two';one.mkdir();two.mkdir()
    run(0,1,copy.deepcopy(policy.actor),rows,targets,str(one),grouped=grouped)
    child_actor=copy.deepcopy(policy.actor)
    # 扩散调度器含不可pickle的lambda；子进程按原1000步配置重建，模型/行为张量不变。
    child_actor.train_diffusion=None
    mp.start_processes(run,args=(2,child_actor,rows,targets,str(two),None,False,grouped),nprocs=2,start_method='spawn')
    expected=torch.load(one/'rank0.pt',weights_only=False)
    results=[torch.load(two/f'rank{i}.pt',weights_only=False) for i in range(2)]
    for result in [expected,*results]:
        r=result['report']
        assert r['optimizer_steps']==3 and r['actual_global_bc_samples']==7
        assert r['internal_sample_visits']==20 and r['applied_unique_chains']==5
        assert [v['global_chain_count'] for v in r['steps']]==[2,2,1]
        assert [v['bc_global_samples'] for v in r['steps']]==[3,2,2]
        assert sorted(r['epoch_orders'][0])==list(range(5))
        assert all(v['denoising_steps']==[15,16,17,18] for v in r['steps'])
        assert all(float(s['step'])==4 for s in result['optimizer']['state'].values())
        if grouped:
            assert len(result['optimizer']['param_groups'])>1
            for group in result['optimizer']['param_groups']:
                assert group['lr']==pytest.approx(1e-7*group['lr_multiplier'])
    assert any(not torch.equal(initial[k],expected['actor'][k]) for k in initial)
    for result in results:
        for key,value in expected['actor'].items():
            torch.testing.assert_close(value,result['actor'][key],rtol=1e-5,atol=1e-7)
        for a,b in zip(expected['report']['steps'],result['report']['steps']):
            assert a['ppo_loss']==pytest.approx(b['ppo_loss'],rel=1e-6,abs=1e-8)
            assert a['gradient_contributions']['all']['ppo_norm']==pytest.approx(b['gradient_contributions']['all']['ppo_norm'],rel=1e-6)


def test_later_minibatches_recheck_earlier_chain_weights(tmp_path,actor_factory):
    policy,rows,targets=case(actor_factory)
    run(0,1,policy.actor,rows,targets,str(tmp_path),soft_kl=.1,adaptive=True)
    report=torch.load(tmp_path/'rank0.pt',weights_only=False)['report']
    assert report['optimizer_steps']==3
    assert [s['candidate_checked_chains'] for s in report['steps']]==[2,4,5]
    assert report['steps'][-1]['post_step_chain']['chain_joint_kl']['mean']<=.08
    assert all(s['trust_region']['accepted'] for s in report['steps'])


def test_soft_kl_stops_before_pending_optimizer_and_bc(tmp_path,actor_factory):
    policy,rows,targets=case(actor_factory)
    run(0,1,policy.actor,rows,targets,str(tmp_path),soft_kl=1e-12)
    report=torch.load(tmp_path/'rank0.pt',weights_only=False)['report']
    assert report['optimizer_steps']==1 and report['actual_global_bc_samples']==3
    assert report['early_stop_reason']=='post_step_chain_joint_kl'
    assert report['skipped_optimizer_steps']==2
    assert report['steps'][0]['post_step_chain']['chain_joint_kl']['mean']>1e-12
