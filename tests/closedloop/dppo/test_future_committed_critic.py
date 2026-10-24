"""验证独立未来参考Critic的因果合同、旧权重映射、缓存训练和Actor隔离。

使用小型真实Stage1网络验证零投影初始化与旧Critic严格等值；优化后未来参考能够
影响预测且可训练编码器每次重建梯度。故意注入错时缓存、错误策略版本及未知到达
泄漏必须拒绝。本文件只验证实现数学与边界，不把合成数据loss当作MC或物理收益。
"""
import copy
from types import SimpleNamespace as NS
import pytest
import torch
from tests.closedloop.test_stage1_actor import actor_factory,_conditions
from tests.closedloop.dppo.test_asymmetric_critic import tracking_context
from gem.closedloop.dppo.pretrained_critic import PretrainedConditionCritic,initialize_future_critic_shared
from gem.closedloop.dppo.tracking_context import validate_tracking_context
from gem.closedloop.dppo.returns import validate_value_state_boundaries
from gem.closedloop.dppo.training_checkpoint import schema_for_identity


def future_context(batch):
    c=tracking_context(_conditions(batch));n=len(c['decision_time'])
    c.update(committed_reference=torch.rand(n,41,38),committed_valid=torch.ones(n,41,dtype=torch.bool),
        committed_times=c['decision_time'][:,None]+torch.arange(10,51,dtype=torch.float64)[None]/50,
        committed_metadata=torch.zeros(n,10),committed_metadata_valid=torch.ones(n,10,dtype=torch.bool),
        committed_snapshot_tick=(c['decision_time']*600).round().long(),committed_policy_version=torch.zeros(n,dtype=torch.int64))
    c['committed_metadata_valid'][:,6:8]=False
    return validate_tracking_context(c)


def test_exact_legacy_initialization_future_learning_and_actor_isolation(actor_factory):
    actor,batch=actor_factory(motion_frames=60);before=copy.deepcopy(actor.state_dict())
    old=PretrainedConditionCritic(actor,dimension=32,layers=1,heads=4)
    new=PretrainedConditionCritic(actor,dimension=32,layers=1,heads=4,future_reference=True)
    initialize_future_critic_shared(new,old.state_dict());c=future_context(batch)
    old.eval();new.eval()
    torch.testing.assert_close(new(c,[4.,5.]),old(c,[4.,5.]),atol=0,rtol=0)
    features=new.encode_frozen(c);optimizer=torch.optim.Adam(new.parameters(),lr=1e-3)
    for _ in range(3):
        optimizer.zero_grad();loss=(new.forward_encoded(features,[4.,5.])-torch.tensor([2.,-2.])).square().mean()
        loss.backward();optimizer.step()
    assert new.future_reference_encoder.frames[0].weight.grad.abs().sum()>0
    changed={k:v.clone() for k,v in c.items()};changed['committed_reference']*=.5
    assert not torch.equal(new(c,[4.,5.]),new(changed,[4.,5.]))
    assert all(torch.equal(v,actor.state_dict()[k]) for k,v in before.items())
    assert all(p.grad is None for p in actor.parameters())
    with pytest.raises(ValueError):new(tracking_context(_conditions(batch)),[4.,5.])


@pytest.mark.parametrize('fault',['timestamp','future_time','arrival','padding','policy'])
def test_future_cache_rejects_wrong_time_version_and_leak(actor_factory,fault):
    _,batch=actor_factory(motion_frames=60);c=future_context(batch)
    if fault=='timestamp':c['committed_snapshot_tick']+=12
    elif fault=='future_time':c['committed_times']+=.02
    elif fault=='arrival':c['committed_metadata_valid'][:,6]=True
    elif fault=='padding':c['committed_valid'][:,0]=False
    else:c['committed_policy_version']+=1
    row=NS(metadata={'critic_context':c},control_tick_begin=int((c['decision_time'][0]*600).round()),
           control_tick_end=9999,identity={'policy_version':0})
    with pytest.raises(ValueError):validate_value_state_boundaries(row)


def test_future_checkpoint_identity_is_separate():
    old={'critic':{'type':'asymmetric_tracking_temporal'}}
    new={'critic':{'type':'asymmetric_tracking_committed_temporal'}}
    assert schema_for_identity(old)!=schema_for_identity(new)


def test_current_and_bootstrap_use_separate_visible_cache(actor_factory):
    _,batch=actor_factory(motion_frames=60);c={k:v[:1].clone() for k,v in future_context(batch).items()}
    nxt={k:v.clone() for k,v in c.items()}
    for key in ('decision_time','proprio_history_times','tracking_history_times','tracking_reference_times','future_times','committed_times'):
        nxt[key]+=.3
    nxt['committed_snapshot_tick']+=180
    row=NS(metadata={'critic_context':c,'next_critic_context':nxt},control_tick_begin=int(c['committed_snapshot_tick'][0]),
        control_tick_end=int(nxt['committed_snapshot_tick'][0]),identity={'policy_version':0})
    validate_value_state_boundaries(row)
    row.metadata['next_critic_context']=c
    with pytest.raises(ValueError,match='another time'):validate_value_state_boundaries(row)


def test_future_preheat_schema_cannot_be_replaced_by_legacy(tmp_path):
    from tests.closedloop.dppo.test_critic_preheat_initialization import fixture,load
    from gem.closedloop.dppo.critic_preheat import SCHEMA_PREHEAT
    actor,critic,optimizer,identity,data=fixture(tmp_path)
    identity['critic']['type']=data['architecture']['type']='asymmetric_tracking_committed_temporal'
    data['schema']=SCHEMA_PREHEAT+'.future_committed'
    assert load(tmp_path,actor,critic,optimizer,identity,data)['actor_unchanged']
    before=copy.deepcopy(critic.state_dict());data['schema']=SCHEMA_PREHEAT
    with pytest.raises(ValueError):load(tmp_path,actor,critic,optimizer,identity,data)
    assert all(torch.equal(v,critic.state_dict()[k]) for k,v in before.items())


def test_future_full_checkpoint_preserves_adam_and_requires_robofft_rng(actor_factory,tmp_path):
    from gem.closedloop.dppo.training_checkpoint import load_training_checkpoint,immutable_cpu_snapshot
    from gem.closedloop.dppo.gradient_merge import BCGradientController
    from gem.closedloop.robofft import OBJECTIVE
    actor,batch=actor_factory(motion_frames=60)
    critic=PretrainedConditionCritic(actor,dimension=32,layers=1,heads=4,future_reference=True)
    ao=torch.optim.AdamW(actor.parameters(),lr=1e-6);co=torch.optim.AdamW(critic.parameters(),lr=1e-4)
    critic(future_context(batch),[4.,5.]).square().mean().backward();co.step();co.zero_grad(set_to_none=True)
    controller=BCGradientController();rng=torch.Generator().manual_seed(42)
    identity=dict(critic={'type':'asymmetric_tracking_committed_temporal'},training={'actor_objective':OBJECTIVE})
    payload=immutable_cpu_snapshot(dict(schema=schema_for_identity(identity),identity=identity,
        actor=actor.state_dict(),critic=critic.state_dict(),actor_optimizer=ao.state_dict(),critic_optimizer=co.state_dict(),
        bc_controller=controller.state_dict(),rank_states=[{'rng':{'generators':{'robofft_mc':rng.get_state()}}}]))
    path=tmp_path/'future.pt';torch.save(payload,path)
    with torch.no_grad():critic.value_token.add_(1.)
    co.state.clear()
    loaded=load_training_checkpoint(path,identity=identity,actor=actor,critic=critic,actor_optimizer=ao,critic_optimizer=co,controller=controller)
    assert co.state and all(torch.equal(v,critic.state_dict()[k]) for k,v in payload['critic'].items())
    expected=torch.rand(8,generator=rng);rng.set_state(loaded['rank_states'][0]['rng']['generators']['robofft_mc'])
    torch.testing.assert_close(expected,torch.rand(8,generator=rng),atol=0,rtol=0)
    payload['rank_states'][0]['rng']['generators'].clear();torch.save(payload,path)
    with pytest.raises(ValueError,match='complete independent MC RNG'):
        load_training_checkpoint(path,identity=identity,actor=actor,critic=critic,actor_optimizer=ao,critic_optimizer=co,controller=controller)
