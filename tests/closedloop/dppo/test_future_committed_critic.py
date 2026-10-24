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
