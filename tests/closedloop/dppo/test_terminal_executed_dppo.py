"""最后五步及实际执行源帧PPO的数学、来源和真实小网络验收。

真实Stage1小网络验证10步采样的最后5步含索引9均有正方差、可重算概率和梯度，
冻结前5步不随Adam变化；最终参考诊断不能多调用一次末步。合成权威50Hz控制
记录验证30Hz插值、复制前缀回溯、跨批旧来源、未消费远景、缺失/重复记录拒绝。
独立Normal完整图与生产分块反向比较执行掩码的loss及梯度，远景极端值不进入
概率/KL，原始采样mask保持不变。另检查新配置身份和Actor专属优势标准化。
所有测试仅CPU，模型夹具和Gloo文件只写pytest临时目录，不冒充八卡或物理改善。
"""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from gem.closedloop.dppo.executed_action_mask import CONTRACT,bind_executed_action_masks,action_mask
from gem.closedloop.dppo.execution_provenance import SCHEMA
from tests.closedloop.test_stage1_actor import actor_factory,_activate_branches,_conditions


def provenance_rows():
    plans={'A':dict(decision_tick=600,commit_tick=600,parent_plan_id='bootstrap:e',
                   prefix_frames=6,generation_frames=30,policy_version=0),
           'B':dict(decision_tick=660,commit_tick=660,parent_plan_id='A',
                   prefix_frames=6,generation_frames=30,policy_version=0)}
    rows=[]
    for name,begin,end in [('A',600,660),('B',660,756)]:
        controls=[]
        for tick in range(begin,end,12):
            ticks=list(range(tick-120,tick+121,12))
            indices=[1 if t>=720 else 0 for t in ticks]
            controls.append(dict(control_tick_begin=tick,control_tick_end=tick+12,
                consumed_reference_ticks=ticks,consumed_plan_indices=indices,
                reference_tick=tick+12,reference_plan_id='B' if tick+12>=720 else 'bootstrap:e',
                gmt_reference_version=1,reward_reference_version=1))
        free=torch.ones(30,30,dtype=torch.bool);free[:6]=False
        rows.append(NS(identity=dict(backend_session_id='s',env_id=0,episode_id='e',
            policy_version=0,plan_id=name),free_mask=free,control_tick_begin=begin,
            control_tick_end=end,executed_control_steps=(end-begin)//12,
            metadata=dict(compact_execution_provenance=dict(schema=SCHEMA,plans=copy.deepcopy(plans),
                steps=controls,plan_dictionary=['bootstrap:e','B']))))
    return rows


def test_real_centers_interpolation_and_copied_prefix_credit_original_decision():
    rows=provenance_rows();before=[r.free_mask.clone() for r in rows]
    report=bind_executed_action_masks(rows)
    assert rows[0].metadata['ppo_execution_frames']==[6,7,8]
    assert torch.equal(action_mask(rows[0])[6:9],before[0][6:9])
    assert int(action_mask(rows[0]).sum())==90
    assert not action_mask(rows[1]).any()  # B只在GMT远景出现；实际中心仍是从A复制来的帧。
    assert report['actor_transitions']==1 and report['controls']==13
    assert report['coordinate_fraction']==pytest.approx(90/1440)
    for row,mask in zip(rows,before):assert torch.equal(row.free_mask,mask)


def test_previous_rollout_owner_is_not_relabelled_as_current_action():
    row=provenance_rows()[1]
    report=bind_executed_action_masks([row])
    assert report['previous_rollout_source_support']>0 and not action_mask(row).any()


def test_cumulative_root_includes_prior_displacement_and_heading_only():
    rows=provenance_rows()
    for row in rows:
        row.free_mask[3:6]=True
        row.metadata['compact_execution_provenance']['plans']['A']['prefix_frames']=3
    bind_executed_action_masks(rows)
    mask=action_mask(rows[0])
    assert mask[3:6,:2].all() and mask[3:6,3:9].all()
    assert not mask[3:6,2].any() and not mask[3:6,9:].any()
    assert not mask[9:].any()


@pytest.mark.parametrize('failure',['missing_record','missing_control','missing_parent','cycle','uncommitted','version'])
def test_missing_or_conflicting_actual_provenance_is_rejected(failure):
    rows=provenance_rows()
    for row in rows:
        rec=row.metadata['compact_execution_provenance']
        if failure=='missing_record':row.metadata.clear()
        elif failure=='missing_control':rec['steps'].pop()
        elif failure=='missing_parent':del rec['plans']['A']
        elif failure=='cycle':rec['plans']['B']['parent_plan_id']='B'
        elif failure=='uncommitted':rec['plans']['B']['commit_tick']=999
        elif failure=='version':rec['steps'][0]['reward_reference_version']=2
    with pytest.raises(ValueError):bind_executed_action_masks(rows)


def terminal_policy(actor_factory):
    from gem.closedloop.dppo.frozen_prefix import FrozenPrefixPolicy
    actor,batch=actor_factory(motion_frames=60);_activate_branches(actor)
    context=_conditions(batch)
    for key in ('known_qpos30','known_qpos30_mask','future_valid','future_times'):
        context[key]=context[key][:,:30].clone()
    policy=FrozenPrefixPolicy(actor,steps=10,frozen_prefix_steps=5,
        std_schedule=[.001]*5+[.04,.06,.08,.1,.02],generation_frames=30,music_context_frames=60)
    return policy,context


def test_actual_final_gaussian_is_trainable_and_frozen_prefix_is_unchanged(actor_factory):
    from gem.closedloop.dppo.policy import masked_joint_log_prob
    policy,c=terminal_policy(actor_factory)
    trace=policy.sample_rollout(c,generator=torch.Generator().manual_seed(17))
    assert policy.trainable_steps==tuple(range(5,10)) and policy.deterministic_steps==()
    assert trace['old_log_probs'].shape==(2,10) and trace['sampling_noise'].shape==(2,10,30,30)
    assert trace['deterministic_final_mean'] is None
    with torch.no_grad():
        before=policy.transition_parameters(c,trace['chain'][:,4],4)['mean'].clone()
        last=policy.transition_parameters(c,trace['chain'][:,9],9)
        torch.testing.assert_close(last['std'],torch.full_like(last['std'],.02))
        replay=masked_joint_log_prob(trace['chain'][:,10],last['mean'],last['std'],last['free_mask'])
        torch.testing.assert_close(replay,trace['old_log_probs'][:,9],rtol=1e-7,atol=2e-3)
        assert not torch.equal(last['mean'],trace['chain'][:,10])
        old_last=last['mean'].clone()
    optimizer=torch.optim.AdamW(policy.actor.parameters(),lr=1e-5)
    for step in policy.trainable_steps:
        optimizer.zero_grad(set_to_none=True)
        parameters=policy.transition_parameters(c,trace['chain'][:,step],step)
        parameters['mean'].square().mean().backward()
        assert any(p.grad is not None and bool(p.grad.abs().sum()) for p in policy.actor.parameters())
        optimizer.step()
    policy.assert_frozen_identity()
    assert all(p.grad is None and not p.requires_grad for p in policy.frozen_policy.actor.parameters())
    with torch.no_grad():
        torch.testing.assert_close(policy.transition_parameters(c,trace['chain'][:,4],4)['mean'],before,rtol=0,atol=0)
        assert not torch.equal(policy.transition_parameters(c,trace['chain'][:,9],9)['mean'],old_last)


def test_executed_mask_matches_independent_normal_gradient_and_ignores_remote_coordinates():
    from tests.closedloop.dppo.test_condition_chunk_backward import analytic_case
    from gem.closedloop.dppo.tensor_cache import RolloutTensorCache
    from gem.closedloop.dppo.chain_ppo import evaluate_chain_records,backward_chain_surrogate
    from gem.closedloop.dppo.execution_checks import policy_phase
    policy,rows,targets=analytic_case(count=5,chains=3)
    for row in rows:
        mask=torch.zeros_like(row.free_mask);mask[0,0]=True
        row.metadata['ppo_action_mask']=mask
    cache=RolloutTensorCache(rows,targets,'cpu')
    features=torch.cat([r.context['feature'] for r in rows]);masks=torch.stack([action_mask(r) for r in rows])
    terms=[]
    for step in policy.trainable_steps:
        mean=policy.mean(features,torch.full((len(rows),),step))
        observed=torch.stack([r.chain[step+1] for r in rows])
        current=torch.distributions.Normal(mean,torch.ones_like(mean))
        old=torch.distributions.Normal(torch.zeros_like(mean),torch.ones_like(mean))
        terms.append((current.log_prob(observed)-old.log_prob(observed)).masked_fill(~masks,0.).sum((-2,-1)))
    ratio=torch.stack(terms,1).exp();adv=targets['advantages'][:,None]
    loss=-torch.minimum(ratio*adv,ratio.clamp(.9,1.1)*adv).mean();loss.backward()
    expected={n:p.grad.clone() for n,p in policy.actor.named_parameters()}
    policy.actor.zero_grad(set_to_none=True)
    with policy_phase(policy):
        records=evaluate_chain_records(policy,rows,cache,microbatch=3)
        result=backward_chain_surrogate(policy,rows,cache,records,clip=.1,global_chain_count=3,microbatch=2,condition_chunk_chains=2)
    # 与已有独立Normal回归共用loss容差；梯度仍使用更严格的双精度对照。
    torch.testing.assert_close(result['loss'],loss.detach(),rtol=1e-8,atol=1e-9)
    for n,p in policy.actor.named_parameters():torch.testing.assert_close(p.grad,expected[n],rtol=1e-10,atol=1e-12)
    assert bool((records[:,:,3]==1).all())
    cache.close()
    for row in rows:
        row.chain[:,1,:]=1e10
        row.metadata['sampler_trace']['old_means'][:,:,1,:]=-1e10
    changed=RolloutTensorCache(rows,targets,'cpu')
    with policy_phase(policy):after=evaluate_chain_records(policy,rows,changed,microbatch=3)
    torch.testing.assert_close(after,records,rtol=0,atol=0)
    changed.close()


def test_actor_normalization_excludes_unconsumed_actions_but_keeps_critic_targets():
    from gem.closedloop.dppo.returns import normalize_advantages_global
    t=dict(advantages_raw=torch.tensor([1.,3.,1000.]),valid=torch.ones(3,dtype=torch.bool),
           actor_valid=torch.tensor([True,True,False]),returns=torch.tensor([4.,5.,6.]))
    result=normalize_advantages_global(t)
    torch.testing.assert_close(result['advantages'],torch.tensor([-1.,1.,0.],dtype=torch.float64))
    assert result['advantage_global_count']==2 and result['valid'].all()
    assert torch.equal(result['returns'],t['returns'])


def test_terminal_executed_configuration_is_explicit_and_bound():
    from gem.closedloop.dppo.training_config import configuration
    from gem.closedloop.dppo.actor_optimization import optimization_binding
    root=Path(__file__).resolve().parents[3]
    c=configuration(root/'configs/closedloop/stage2_terminal5_executed_server2.yaml')
    t=c['stage9'];binding=optimization_binding(c['stage10'])
    assert t['ppo_step_strategy']=='terminal5' and t['ppo_action_scope']==CONTRACT
    assert t['frozen_prefix_steps']==5 and t['std_schedule'][-1]>0
    assert binding['trainable_gaussian_steps']==[5,6,7,8,9] and binding['ppo_action_scope']==CONTRACT


def test_terminal_actual_update_and_final_reference_replay(actor_factory,tmp_path):
    import torch.distributed as dist
    from gem.closedloop.dppo.collectives import DistributedCollectives
    from gem.closedloop.dppo.continuation import DisabledBC
    from gem.closedloop.dppo.gradient_merge import BCGradientController
    from gem.closedloop.dppo.tensor_cache import RolloutTensorCache
    from gem.closedloop.dppo.rollout_helpers import split_trace
    from gem.closedloop.dppo.optimization_support import probability_check_local
    from gem.closedloop.dppo.updater import actor_update,final_kl
    from tests.closedloop.dppo.test_asymmetric_critic import tracking_context
    policy,c=terminal_policy(actor_factory);priv=tracking_context(c)
    policy.kernel_config['actor_probability_scope']=CONTRACT
    trace=policy.sample_rollout(c,generator=torch.Generator().manual_seed(17))
    rows=[]
    for i in range(2):
        mask=torch.zeros_like(trace['free_mask'][i])
        frames=trace['free_mask'][i].any(-1).nonzero().flatten()[:3]
        mask[frames]=trace['free_mask'][i][frames]
        rows.append(NS(context={k:v[i:i+1].clone() for k,v in c.items()},identity={},transition_valid=True,
            chain=trace['chain'][i],old_log_prob=trace['old_log_probs'][i],free_mask=trace['free_mask'][i],
            metadata=dict(critic_context={k:v[i:i+1].clone() for k,v in priv.items()},
                sampler_trace=split_trace(trace,i,2),remaining_music_seconds=3.,
                ppo_action_mask=mask,ppo_action_scope=CONTRACT)))
    targets=dict(advantages=torch.tensor([1.,-1.]),valid=torch.ones(2,dtype=torch.bool))
    cache=RolloutTensorCache(rows,targets,'cpu')
    manifest=[dict(owner_rank=0,local_index=i,valid=True,has_free=True) for i in range(2)]
    dist.init_process_group('gloo',init_method=(tmp_path/'gloo').as_uri(),rank=0,world_size=1)
    distributed=DistributedCollectives(0,1,device='cpu')
    try:
        check=probability_check_local(policy,rows,global_manifest=manifest,distributed=distributed,
            denoising_microbatch=2,tensor_cache=cache)
        assert check['passed'] and check['checked_steps']==list(range(10))
        before=final_kl(policy,rows,cache,manifest,distributed=distributed,microbatch=3)
        assert before['gaussian_kl_steps']==[5,6,7,8,9]
        assert max(s['normalized_rms'] for s in before['reference_change']['samples'])<1e-6
        opt=torch.optim.AdamW((p for p in policy.actor.parameters() if p.requires_grad),lr=1e-9,weight_decay=0.)
        report=actor_update(policy,opt,rows,targets,cache,manifest,BCGradientController(),distributed=distributed,
            anchor=DisabledBC(),generator=torch.Generator().manual_seed(4),minibatch_chains=2,
            microbatch=3,sampled_steps=5,step_strategy='terminal5',bc_enabled=False,
            bc_samples_per_rollout=0,audit_full_rollout_objective=True)
        assert report['optimizer_steps']==1 and report['steps'][0]['denoising_steps']==[5,6,7,8,9]
        assert opt.state
        after=final_kl(policy,rows,cache,manifest,distributed=distributed,microbatch=3)
        assert after['covered_internal_transitions']==10 and after['frozen_identity']['deterministic_frozen_steps']==[]
        assert all(s['protected_coordinates_unchanged'] for s in after['reference_change']['samples'])
        assert all(p.grad is None for p in policy.frozen_policy.actor.parameters())
    finally:
        cache.close();dist.destroy_process_group()


def test_new_behavior_missing_execution_mask_cannot_fall_back_to_full_horizon():
    row=NS(free_mask=torch.ones(3,30,dtype=torch.bool),metadata={
        'sampler_trace':{'kernel_config':{'actor_probability_scope':CONTRACT}}})
    with pytest.raises(ValueError,match='never full-horizon'):action_mask(row)


@pytest.mark.parametrize('terminal',[False,True])
def test_real_transition_persistence_uses_declared_gaussian_steps(actor_factory,tmp_path,terminal):
    from gem.closedloop.dppo.buffer import UpperTransition
    from gem.closedloop.dppo.diagnostic_rollout import persist_rollout,load_rollout
    from gem.closedloop.dppo.rollout_helpers import split_trace
    from gem.closedloop.dppo.frozen_prefix import FrozenPrefixPolicy
    from tests.closedloop.dppo.test_asymmetric_critic import tracking_context
    policy,c=terminal_policy(actor_factory)
    if not terminal:
        policy=FrozenPrefixPolicy(policy.actor,steps=10,frozen_prefix_steps=4,
            std_schedule=[.001]*4+[.02,.04,.06,.08,.1,0.],generation_frames=30,music_context_frames=60)
    else:policy.kernel_config['actor_probability_scope']=CONTRACT
    privileged=tracking_context(c)
    c={k:privileged[k] for k in c}
    trace=policy.sample_rollout(c,generator=torch.Generator().manual_seed(17))
    row=UpperTransition(identity=dict(run_id='test',backend_session_id='test',episode_id='test',
        decision_id=0,policy_version=0),context={k:v[:1] for k,v in c.items()},next_context=None,
        chain=trace['chain'][0],old_log_prob=trace['old_log_probs'][0],free_mask=trace['free_mask'][0],
        rewards=torch.ones(5,dtype=torch.float64),old_value=0.,next_value=0.,control_tick_begin=0,
        control_tick_end=60,executed_control_steps=5,executed_physics_steps=20,terminated=True,
        metadata=dict(critic_context={k:v[:1] for k,v in privileged.items()},
            sampler_trace=split_trace(trace,0,2)))
    if terminal:row.metadata.update(ppo_action_mask=row.free_mask.clone(),ppo_action_scope=CONTRACT)
    row.validate()
    manifest=[dict(owner_rank=0,local_index=0,valid=True,has_free=True)]
    persist_rollout(tmp_path/'case',[row],dict(advantages=torch.ones(1)),manifest,
        kernel=policy.kernel_config,identity={},full_execution=False)
    restored=load_rollout(tmp_path/'case')['transitions'][0]
    assert restored.old_log_prob.shape==((10 if terminal else 9),)
    torch.testing.assert_close(restored.old_log_prob,row.old_log_prob,rtol=0,atol=0)
    torch.testing.assert_close(action_mask(restored),action_mask(row),rtol=0,atol=0)
    wrong=copy.deepcopy(row);wrong.old_log_prob=wrong.old_log_prob[:-1]
    with pytest.raises(ValueError,match='declared stochastic'):wrong.validate()
    wrong=copy.deepcopy(row);wrong.metadata['sampler_trace']['probability_steps']=torch.arange(8)
    with pytest.raises(ValueError,match='step indices'):wrong.validate()
    wrong=copy.deepcopy(row);wrong.metadata['sampler_trace']['kernel_config']['deterministic_steps']=[0]
    with pytest.raises(ValueError,match='step identity'):wrong.validate()
