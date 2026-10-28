"""RoboFFT-D 数学、真实小型 Stage1Actor、固定 MC 和 compact 数据流验收。

使用仓库现有 Bumi 统计量/机器人夹具与小型真实 Transformer，全部输出由 pytest
临时目录管理。纯数学测试独立重建官方 MC 聚合、clamp、negdisc 和 SNR 等价公式；
策略测试验证同 CFG、条件梯度、前缀/padding、紧凑采样和单遍/流式反传一致。
本文件不使用生产权重、不连接 Isaac，也不将 CPU 合同验证冒充机器人跟踪质量。
"""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from tests.closedloop.test_stage1_actor import actor_factory, _activate_branches, _conditions
from tests.closedloop.dppo.test_asymmetric_critic import tracking_context
from gem.closedloop.robofft import ALGORITHM
from gem.closedloop.robofft.policy import RoboFFTDiffusionPolicy
from gem.closedloop.robofft.surrogate import action_objective, delta_log_snr, epsilon_loss, q_sample
from gem.closedloop.robofft.updater import prepare_mc, objective_batch, actor_update, diagnostics
from gem.closedloop.dppo.tensor_cache import RolloutTensorCache
from gem.closedloop.dppo.rollout_helpers import split_trace
from gem.closedloop.dppo.training_config import configuration

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def settings():
    return configuration(ROOT/'tests/closedloop/fixtures/robofft_contract.yaml')['stage9']


def make_policy(factory, *, starts=(45, 47), prefix=6):
    actor, batch = factory(motion_frames=60, starts=starts, prefix=prefix)
    _activate_branches(actor)
    t = settings()
    policy = RoboFFTDiffusionPolicy(actor, frozen_prefix_steps=9, generation_frames=60,
        std_schedule=t['std_schedule'], robofft=t['robofft'])
    return policy, _conditions(batch)


def make_rows(policy, conditions):
    trace = policy.sample_rollout(conditions, generator=torch.Generator().manual_seed(919))
    privileged = tracking_context(conditions)
    rows = []
    for i in range(len(trace['normalized'])):
        rows.append(NS(context={k:v[i:i+1] for k,v in conditions.items()}, identity={},
            algorithm=ALGORITHM, chain=None, old_log_prob=None, action_x0=trace['action_x0'][i].clone(),
            free_mask=trace['free_mask'][i], mc_record=None,
            metadata=dict(critic_context={k:v[i:i+1] for k,v in privileged.items()},
                sampler_trace=split_trace(trace,i,len(trace['normalized'])),remaining_music_seconds=5.)))
    targets=dict(advantages=torch.tensor([1., -.7][:len(rows)],dtype=torch.float64),
                 advantages_normalized=True,advantage_normalization_scope='global_valid_upper_transitions')
    cache = RolloutTensorCache(rows,targets,'cpu')
    manifest=[dict(valid=True,has_free=True,owner_rank=0,local_index=i) for i in range(len(rows))]
    return rows, targets, cache, manifest


def local_distributed():
    return NS(rank=0,world_size=1,device=torch.device('cpu'),all_gather_object=lambda x:[x],
        broadcast_object=lambda x:x,sum_tensor=lambda x:x,sum_gradients=lambda model:None)


def gradients(actor):
    return {n:(torch.zeros_like(p) if p.grad is None else p.grad.clone())
            for n,p in actor.named_parameters() if p.requires_grad}


def test_official_mc_order_clamp_negdisc_and_gradient():
    new=torch.tensor([[0.,3.,70.,0.],[4.,0.,0.,5.]],dtype=torch.float64,requires_grad=True)
    old=torch.tensor([[1.,0.,0.,60.],[0.,5.,13.,0.]],dtype=torch.float64)
    advantage=torch.tensor([2.,-1.],dtype=torch.float64)
    got=action_objective(new,old,advantage,clip=.01,beta_neg=.125,logprob_normalizer=.5,logratio_alpha=.1)
    independent=[];ratios=[]
    for i in range(2):
        logs=[torch.clamp(-.5*new[i,k]+.5*old[i,k],-10,10)*.1 for k in range(4)]
        ratio=torch.exp(sum(logs)/4);ratios.append(ratio)
        loss=torch.maximum(-advantage[i]*ratio,-advantage[i]*ratio.clamp(.99,1.01))
        independent.append(loss*(.125 if advantage[i]<0 else 1.))
    torch.testing.assert_close(got['loss'],torch.stack(independent),atol=0,rtol=0)
    torch.testing.assert_close(got['ratio'],torch.stack(ratios),atol=0,rtol=0)
    torch.testing.assert_close(torch.autograd.grad(got['loss'].sum(),new,retain_graph=True)[0],
                               torch.autograd.grad(torch.stack(independent).sum(),new)[0])
    assert got['logratio_clamp_fraction'].tolist()==[.5,0.]
    assert not torch.allclose(got['ratio'], ((old-new).mul(.5).clamp(-10,10)*.1).exp().mean(-1))
    equal=action_objective(new,new.detach(),advantage)
    assert torch.equal(equal['ratio'],torch.ones(2,dtype=torch.float64))
    assert torch.equal(equal['surrogate_kl'],torch.zeros(2,dtype=torch.float64))
    assert torch.autograd.grad(equal['loss'].sum(),new)[0].abs().sum()>0
    undiscounted=action_objective(new,new.detach(),advantage,beta_neg=1.)
    assert equal['loss'][1]==undiscounted['loss'][1]*.125


def test_epsilon_and_snr_x0_value_and_gradient_agree():
    rng=torch.Generator().manual_seed(30)
    x0=torch.randn(3,7,30,generator=rng,dtype=torch.float64)
    eps=torch.randn(3,7,30,generator=rng,dtype=torch.float64)
    alpha=torch.tensor([.99,.7,.2],dtype=torch.float64)[:,None,None]
    pred=torch.randn(3,7,30,generator=rng,dtype=torch.float64,requires_grad=True)
    mask=torch.rand(3,7,30,generator=rng)>.2
    weights=delta_log_snr(alpha[:,0,0],torch.tensor([.9999,.8,.4],dtype=torch.float64))
    xt=alpha.sqrt()*x0+(1-alpha).sqrt()*eps
    epsilon=epsilon_loss(xt,pred,eps,alpha,weights,mask)
    equivalent=.5*weights*((pred-x0).square()*alpha/(1-alpha)).masked_fill(~mask,0).mean(-1).sum(-1)
    torch.testing.assert_close(epsilon,equivalent,atol=1e-11,rtol=1e-12)
    torch.testing.assert_close(torch.autograd.grad(epsilon.sum(),pred,retain_graph=True)[0],
                               torch.autograd.grad(equivalent.sum(),pred)[0],atol=1e-12,rtol=1e-12)
    high=torch.tensor(.99999,dtype=torch.float64)
    noisy=q_sample(x0,eps,high,torch.zeros_like(x0),torch.zeros_like(mask),torch.ones(3,7,dtype=torch.bool))
    torch.testing.assert_close(noisy,high.sqrt()*x0+(1-high).sqrt()*eps,atol=0,rtol=0)
    assert delta_log_snr(high,torch.tensor(1.,dtype=torch.float64))==0


def test_compact_same_rng_final_x0_without_gaussian_calls(actor_factory,monkeypatch):
    policy,context=make_policy(actor_factory)
    full=policy.sample_rollout(context,store_chain=True,generator=torch.Generator().manual_seed(72))
    def forbidden(*args,**kwargs):pytest.fail('Gaussian probability called by compact sampling')
    monkeypatch.setattr('gem.closedloop.dppo.policy.masked_joint_log_prob',forbidden)
    rng=torch.Generator().manual_seed(72)
    compact=policy.sample_rollout(context,generator=rng)
    for k in ('normalized','action_x0','qpos','qpos30','contact'):
        torch.testing.assert_close(compact[k],full[k],atol=0,rtol=0)
    assert not {'chain','old_log_probs','old_means','old_stds','sampling_noise'} & compact.keys()
    assert torch.equal(compact['action_x0'],full['chain'][:,-1])
    assert policy.trainable_steps==tuple(range(9,19))
    assert all(not p.requires_grad for p in policy.frozen_policy.actor.parameters())
    rng2=torch.Generator().manual_seed(72)
    # 原完整链已禁用 Gaussian oracle，此处直接保存的采样终态 RNG 单独重跑。
    monkeypatch.undo()
    policy.sample_rollout(context,store_chain=True,generator=rng2)
    assert torch.equal(rng.get_state(),rng2.get_state())


def test_cfg_eval_mask_and_condition_gradients(actor_factory):
    policy,context=make_policy(actor_factory,starts=(150,))
    for module in policy.actor.modules():
        if isinstance(module,torch.nn.Dropout):module.p=.6
    policy.actor.train()
    trace=policy.sample_rollout(context,generator=torch.Generator().manual_seed(41))
    mc=policy.draw_mc(trace['action_x0'],torch.Generator().manual_seed(99))
    loss=policy.losses(context,trace['action_x0'],**mc)
    repeat=policy.losses(context,trace['action_x0'],**mc)
    torch.testing.assert_close(loss,repeat,atol=0,rtol=0)
    assert not policy.actor.training
    changed=mc['eps'].clone();changed.masked_fill_(~trace['free_mask'][:,None],100.)
    torch.testing.assert_close(loss,policy.losses(context,trace['action_x0'],changed,
        mc['sampling_indices'],mc['original_timesteps']),atol=0,rtol=0)
    loss.sum().backward()
    names=['history_encoder','prefix_encoder','music']
    grads=gradients(policy.actor)
    for name in names:
        assert sum(float(v.abs().sum()) for k,v in grads.items() if name in k)>0,name
    assert all(p.grad is None for p in policy.frozen_policy.actor.parameters())
    # 原采样与新代理共享同一个 _predict_prepared；独立手工 CFG 校验。
    with torch.no_grad():
        prepared=policy.prepare_conditions(context,include_frozen=False)
        state=policy.actor._constrain(torch.randn_like(trace['action_x0']),prepared['adapted'])
        t=torch.tensor([policy.timestep_map[12]])
        predicted=policy._predict_prepared(prepared,state,t)['pred_x_start']
        a=policy.actor._denoise(state,t,prepared['adapted'],prepared['conditional'])['pred_x_start']
        b=policy.actor._denoise(state,t,prepared['adapted'],prepared['unconditional'])['pred_x_start']
        expected=policy.actor._constrain(b+2.5*(a-b),prepared['adapted'])
        torch.testing.assert_close(predicted,expected,atol=2e-6,rtol=2e-6)


def test_fixed_mc_reorder_streaming_and_old_immutability(actor_factory,tmp_path):
    policy,context=make_policy(actor_factory)
    rows,targets,cache,manifest=make_rows(policy,context)
    rng=torch.Generator().manual_seed(123)
    environment_rng=torch.random.get_rng_state().clone()
    prepare_mc(policy,rows,cache,rng,microbatch=1)
    assert torch.equal(environment_rng,torch.random.get_rng_state())
    old=torch.stack([r.mc_record.old_loss for r in rows])
    records=[r.mc_record for r in rows]
    mc={key:torch.stack([getattr(r,key) for r in records]) for key in ('eps','sampling_indices','original_timesteps')}
    full=policy.losses(context,cache.get('action_x0',rows),**mc)
    objective=action_objective(full,old,targets['advantages'])
    torch.testing.assert_close(objective['ratio'],torch.ones(2,dtype=torch.float64),atol=2e-6,rtol=0)
    objective['loss'].mean().backward();one=gradients(policy.actor)
    policy.actor.zero_grad(set_to_none=True)
    for row in reversed(rows):
        objective_batch(policy,[row],cache,clip=.01,global_count=2,backward=True)
    streamed=gradients(policy.actor)
    for name in one:
        torch.testing.assert_close(streamed[name],one[name],atol=2e-5,rtol=2e-4,msg=name)
    assert sum(float(x.abs().sum()) for x in streamed.values())>0
    assert torch.equal(old,torch.stack([r.mc_record.old_loss for r in rows]))
    with pytest.raises(ValueError,match='already fixed'):prepare_mc(policy,rows,cache,rng)
    path=tmp_path/'mc.pt';torch.save(records,path)
    restored=torch.load(path,weights_only=False)
    for value in restored:value.validate()
    assert torch.equal(restored[0].eps,records[0].eps)
    # CPU RNG 可恢复；改变 action microbatch 不改变逐动作 MC 抽样。
    replay_rng=torch.Generator().manual_seed(123)
    replay=policy.draw_mc(cache.get('action_x0',rows),replay_rng)
    assert torch.equal(mc['eps'],replay['eps']) and torch.equal(rng.get_state(),replay_rng.get_state())
    records[0].old_loss.add_(1)
    with pytest.raises(ValueError,match='modified'):records[0].validate()
    cache.close()


def test_action_update_changes_output_no_gaussian_dependency(actor_factory,monkeypatch):
    policy,context=make_policy(actor_factory)
    rows,targets,cache,manifest=make_rows(policy,context)
    prepare_mc(policy,rows,cache,torch.Generator().manual_seed(333))
    frozen=copy.deepcopy(policy.frozen_policy.actor.state_dict())
    old=[r.mc_record.old_loss.clone() for r in rows]
    def forbidden(*args,**kwargs):pytest.fail('Old Gaussian PPO/KL was called')
    for module,name in [('gem.closedloop.dppo.policy','masked_joint_log_prob'),
        ('gem.closedloop.dppo.updater','actor_update'),('gem.closedloop.dppo.updater','final_kl'),
        ('gem.closedloop.dppo.actor_backtracking','backtracked_step'),
        ('gem.closedloop.dppo.chain_ppo','evaluate_chain_records')]:
        monkeypatch.setattr(module+'.'+name,forbidden)
    from gem.closedloop.dppo.actor_optimizer import make_actor_optimizer
    optimizer=make_actor_optimizer(policy.actor,settings())
    report=actor_update(policy,optimizer,rows,targets,cache,manifest,distributed=local_distributed(),
        generator=torch.Generator().manual_seed(72),ppo_epochs=2,minibatch_actions=3,microbatch=1)
    assert report['optimizer_steps']==2 and abs(report['initial_surrogate']['surrogate_kl'])<1e-12
    assert report['reference_change'][0]['normalized_change_rms']>0
    assert report['steps'][0]['learning_rate']==1e-5
    for row,previous in zip(rows,old):assert torch.equal(row.mc_record.old_loss,previous)
    for k,v in frozen.items():assert torch.equal(policy.frozen_policy.actor.state_dict()[k],v)
    assert report['final_surrogate']['gaussian_evaluations']==0
    cache.close()




def test_fixed_evaluation_uses_compact_surrogate_not_gaussian(actor_factory,tmp_path,monkeypatch):
    from gem.closedloop.robofft.evaluation import make_reference,load_references,fixed_action_drift
    from gem.closedloop.dppo.tensorboard_metrics import checked_scalars,dashboard_scalars
    policy,context=make_policy(actor_factory,starts=(45,))
    def forbidden(*args,**kwargs):pytest.fail('Gaussian diagnostics used in RoboFFT evaluation')
    monkeypatch.setattr('gem.closedloop.dppo.policy.masked_joint_log_prob',forbidden)
    monkeypatch.setattr('gem.closedloop.dppo.periodic_monitor.fixed_chain_drift',forbidden)
    trace=policy.sample_rollout(context,generator=torch.Generator().manual_seed(2026))
    record=make_reference(policy,trace,seed=61)
    torch.save(record,tmp_path/'reference.pt')
    reference=load_references(tmp_path)
    diagnostic=fixed_action_drift(policy,reference)
    assert diagnostic['surrogate_kl_values']==[0.]
    assert diagnostic['reference_changes'][0]['normalized_change_rms']==0.
    with torch.no_grad():policy.actor.denoiser.final_layer.fc2.weight.add_(.001)
    diagnostic=fixed_action_drift(policy,reference)
    assert diagnostic['reference_changes'][0]['normalized_change_rms']>0
    assert diagnostic['surrogate_kl_values'][0]>0
    assert 'joint_kl_values' not in diagnostic
    values={'Policy/surrogate_kl':diagnostic['surrogate_kl_values'][0]}
    assert dashboard_scalars('training',checked_scalars(values))==values
