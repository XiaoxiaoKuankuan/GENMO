"""Clipped DPPO真实小网络、变长尾批和分布式归约回归。

复用Stage1 Actor、15+4+1行为采样、真实高斯概率及生产更新器，验证5条链按2/2/1
更新三次、每次四步、旧行为统计不变及BC总数。两进程场景故意让rank1没有任何
PPO样本但仍贡献BC；与单进程相同非空Adam初态对照，检测重复world平均或空rank
漏同步。另验证第三步超限只恢复该步，保留前两步和BC游标，完整KL可复用。此测试不声称真实物理收益。
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
    def __init__(self):self.batch_size=0;self.calls=0
    def state_dict(self):return dict(batch_size=self.batch_size,calls=self.calls)
    def load_state_dict(self,s):self.batch_size=s["batch_size"];self.calls=s["calls"]

    def backward(self,actor,weight):
        self.calls+=1
        torch.rand(2)
        loss=sum(p.square().mean() for p in actor.parameters() if p.requires_grad)
        if self.batch_size:(loss*weight).backward()
        return dict(batch_size=self.batch_size,loss=float(loss.detach()),
            samples=[[{'dataset_id':'fixture'}]*self.batch_size])


def case(actor_factory, frozen_prefix_steps=15):
    actor,batch=actor_factory(motion_frames=60);_activate_branches(actor)
    policy=FrozenPrefixPolicy(actor,frozen_prefix_steps=frozen_prefix_steps,std_schedule=[.001]*15+[.04,.06,.08,.1,0.],generation_frames=60)
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


def run(rank,world,actor,rows,targets,directory,scenario=None):
    torch.set_num_threads(1)
    policy=FrozenPrefixPolicy(actor,std_schedule=[.001]*15+[.04,.06,.08,.1,0.],generation_frames=60)
    dist.init_process_group('gloo',init_method=(Path(directory)/'gloo').as_uri(),rank=rank,world_size=world)
    manifest=[dict(owner_rank=0,local_index=i,valid=True,has_free=True) for i in range(len(rows))]
    local=rows if rank==0 else []
    local_targets=targets if rank==0 else {k:v[:0] for k,v in targets.items()}
    cache=RolloutTensorCache(local,local_targets,'cpu')
    opt=torch.optim.AdamW(trainable_actor_parameters(policy.actor),lr=1e-6,weight_decay=0.)
    # 真实非空Adam初态：零梯度step只建立动量，不改变Actor及old statistics。
    for p in trainable_actor_parameters(policy.actor):p.grad=torch.zeros_like(p)
    opt.step();opt.zero_grad(set_to_none=True)
    before={k:cache.host[k].clone() for k in ('chain','old_log_prob','old_means','old_stds') if k in cache.host}
    attempts=[];states=[];anchor=Anchor()
    import gem.closedloop.dppo.updater as module
    original_kl=module.final_kl;checks=[]
    def audit(*args,**kwargs):
        result=original_kl(*args,**kwargs);checks.append(1)
        if scenario:
            # 合成门槛只控制分支；底层仍执行真实全链高斯检查。
            fail=(3<=len(checks)<=5) if scenario=='reject' else len(checks)==3
            value=2. if fail else .1
            result['mean_coordinate_kl']=value
            result['maximum_step_mean_coordinate_kl']=value
            result['coordinate_kl']['p95']=value
        return result
    module.final_kl=audit
    def observe(report):
        states.append(copy.deepcopy(dict(actor=actor.state_dict(),optimizer=opt.state_dict(),
            anchor=anchor.state_dict(),rng=torch.get_rng_state())))
    try:
        report=actor_update(policy,opt,local,local_targets,cache,manifest,
            BCGradientController(initialize_from_gradient=True),distributed=DistributedCollectives(rank,world),
            anchor=anchor,generator=torch.Generator().manual_seed(4),minibatch_chains=2,microbatch=3,
            bc_samples_per_rollout=7,reserve_attempt=lambda:attempts.append(1),
            backtracking={'max_attempts':3,'factor':.5},kl_limits={'mean_coordinate':1.,
            'maximum_step_mean_coordinate':1.,'p95_coordinate':1.},kl_microbatch=3,step_observer=observe,
            detailed_diagnostics=True)
        assert len(attempts)==report['attempted_optimizer_steps']
        assert all(torch.equal(v,cache.host[k]) for k,v in before.items())
        assert policy.assert_frozen_identity()['checked']
        from gem.closedloop.dppo.kl_reuse import reusable_accepted_kl
        reused=reusable_accepted_kl(report,policy,cache,manifest,DistributedCollectives(rank,world),
            clip=.01,reduction='free_coordinate_mean')
        assert reused['covered_chains']==5 and reused['covered_internal_transitions']==20
        torch.save(dict(actor=policy.actor.state_dict(),optimizer=opt.state_dict(),report=report,
            states=states,anchor=anchor.state_dict(),rng=torch.get_rng_state(),kl_checks=len(checks)),
                   Path(directory)/f'rank{rank}.pt')
    finally:
        module.final_kl=original_kl
        cache.close();dist.destroy_process_group()


def test_multistep_tail_empty_rank_bc_and_nonempty_adam(tmp_path,actor_factory):
    policy,rows,targets=case(actor_factory)
    initial=copy.deepcopy(policy.actor.state_dict())
    one=tmp_path/'one';two=tmp_path/'two';one.mkdir();two.mkdir()
    run(0,1,copy.deepcopy(policy.actor),rows,targets,str(one))
    mp.start_processes(run,args=(2,copy.deepcopy(policy.actor),rows,targets,str(two)),nprocs=2,start_method='fork')
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
    assert any(not torch.equal(initial[k],expected['actor'][k]) for k in initial)
    for result in results:
        for key,value in expected['actor'].items():
            torch.testing.assert_close(value,result['actor'][key],rtol=1e-5,atol=1e-7)
        for a,b in zip(expected['report']['steps'],result['report']['steps']):
            assert a['ppo_loss']==pytest.approx(b['ppo_loss'],rel=1e-6,abs=1e-8)
            assert a['gradient_contributions']['all']['ppo_norm']==pytest.approx(b['gradient_contributions']['all']['ppo_norm'],rel=1e-6)


def test_third_step_rejection_preserves_accepted_steps_adam_bc_and_rng(tmp_path,actor_factory):
    from tests.closedloop.dppo.test_actor_backtracking import equal_state
    policy,rows,targets=case(actor_factory)
    run(0,1,policy.actor,rows,targets,str(tmp_path),scenario='reject')
    result=torch.load(tmp_path/'rank0.pt',weights_only=False);r=result['report']
    assert r['optimizer_steps']==2 and r['actual_global_bc_samples']==5
    assert r['attempted_optimizer_steps']==5 and result['kl_checks']==6
    assert r['early_stop_reason']=='optimizer_step_backtracking_exhausted'
    assert result['anchor']['calls']==2
    for k in ('actor','optimizer','anchor','rng'):
        assert equal_state(result[k],result['states'][1][k])


def test_third_step_backtracking_accepts_smaller_lr_without_repeating_bc(tmp_path,actor_factory):
    policy,rows,targets=case(actor_factory)
    run(0,1,policy.actor,rows,targets,str(tmp_path),scenario='retry')
    result=torch.load(tmp_path/'rank0.pt',weights_only=False);r=result['report']
    assert r['optimizer_steps']==3 and r['actual_global_bc_samples']==7
    assert r['attempted_optimizer_steps']==4 and result['kl_checks']==4
    assert result['anchor']['calls']==3 and r['steps'][-1]['learning_rate']==5e-7
    assert all(float(s['step'])==4 for s in result['optimizer']['state'].values())
