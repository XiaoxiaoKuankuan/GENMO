"""验证用户要求的三轮未来参考Critic短训练，不采集新的完整音乐MC轨迹。

真实小型Stage1/旧Critic checkpoint进入新Critic后，核对Actor、Actor Adam、
共享Critic参数以及初始Value逐位保持；新增分支零投影且Critic Adam重新建立。
故意改变价值契约或破坏零初始化时必须在任何参数写入之前拒绝。配置测试验证
累计三轮、20秒独立物理世界、禁用完整MC；这些数学测试不证明物理跟踪收益。
"""
import copy
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from tests.closedloop.test_stage1_actor import actor_factory
from tests.closedloop.dppo.test_future_committed_critic import future_context
from gem.closedloop.dppo.pretrained_critic import PretrainedConditionCritic
from gem.closedloop.dppo.training import _validate_future_critic_release
from gem.closedloop.dppo.training_config import configuration
from gem.closedloop.robofft import CHECKPOINT_SCHEMA, OBJECTIVE
from gem.closedloop.robofft.checkpoint import initialize_optimization_variant

ROOT=Path(__file__).resolve().parents[3]


def trial_config():
    return configuration(ROOT/'tests/closedloop/fixtures/future_critic_contract.yaml')


def test_trial_is_explicit_bounded_and_has_no_full_mc():
    c=trial_config();args=NS(future_critic_trial=True,stop_after_iteration=3,initialize_robofft_variant='parent.pt')
    _validate_future_critic_release(args,c)
    assert c['stage10']['critic']['warmup_rollouts']==0
    assert c['runtime']['future_reference_context'] and c['runtime']['paired_state_audit']
    e=c['stage10']['evaluation']
    assert sum(e['samples_per_source'].values())==8 and len(e['seeds'])==e['envs_per_rank']==2
    for section,key,value in [('critic','complete_mc',dict(c['stage10']['critic']['complete_mc'],enabled=True)),
                              ('run_control','max_training_iterations',4),('limits','accepted_iterations',4),
                              ('evaluation','episode_seconds',None),('evaluation','persistent_world',True)]:
        broken=copy.deepcopy(c);broken['stage10'][section][key]=value
        with pytest.raises(ValueError,match='at most three'): _validate_future_critic_release(args,broken)
    for changes in (dict(stop_after_iteration=4),dict(initialize_robofft_variant=None)):
        with pytest.raises(ValueError,match='at most three'):
            _validate_future_critic_release(NS(**{**vars(args),**changes}),c)


def test_old_checkpoint_preserves_initial_value_and_actor_adam(actor_factory,tmp_path):
    actor,batch=actor_factory(motion_frames=60)
    old=PretrainedConditionCritic(actor,dimension=32,layers=1,heads=4)
    new=PretrainedConditionCritic(actor,dimension=32,layers=1,heads=4,future_reference=True)
    ao=torch.optim.AdamW(actor.parameters(),lr=1e-6)
    # 为一个真实Actor参数建立非空动量，避免空优化器造成假阳性。
    parameter=next(p for p in actor.parameters() if p.requires_grad)
    parameter.grad=torch.ones_like(parameter);ao.step();ao.zero_grad(set_to_none=True)
    co=torch.optim.AdamW(new.parameters(),lr=1e-4)
    source=dict(training=dict(actor_objective=OBJECTIVE,gamma_upper=.95,robofft=dict(mc_samples=16,beta_neg=.125)),
        runtime=dict(num_envs=128,generation_batch=128),
        critic=dict(type='asymmetric_tracking_temporal',dimension=32,complete_mc=dict(enabled=True)))
    payload=dict(schema=CHECKPOINT_SCHEMA,identity=source,actor=copy.deepcopy(actor.state_dict()),
        critic=copy.deepcopy(old.state_dict()),actor_optimizer=copy.deepcopy(ao.state_dict()),
        state=dict(iteration=28,actor_updates=111,mc_preheat_complete=True))
    path=tmp_path/'legacy.pt';torch.save(payload,path)
    target=copy.deepcopy(source);target['critic']['type']='asymmetric_tracking_committed_temporal'
    target['critic']['complete_mc']['enabled']=False
    target['runtime'].update(future_reference_context=True,paired_state_audit=True,num_envs=16,generation_batch=16)
    kw=dict(identity=target,actor=actor,critic=new,actor_optimizer=ao,critic_optimizer=co,future_critic_trial=True)
    before=copy.deepcopy(actor.state_dict())
    with torch.no_grad():new.future_reference_encoder.projection.weight.add_(.01)
    with pytest.raises(ValueError,match='exact zero'):initialize_optimization_variant(path,**kw)
    assert all(torch.equal(v,actor.state_dict()[k]) for k,v in before.items())
    with torch.no_grad():new.future_reference_encoder.projection.weight.zero_()
    target['training']['gamma_upper']=.99
    with pytest.raises(ValueError,match='protected contract'):initialize_optimization_variant(path,**kw)
    target['training']['gamma_upper']=.95
    with torch.no_grad():parameter.add_(1.)
    ao.state.clear()
    record=initialize_optimization_variant(path,**kw)
    old.eval();new.eval();context=future_context(batch)
    torch.testing.assert_close(new(context,[4.,5.]),old(context,[4.,5.]),atol=0,rtol=0)
    assert all(torch.equal(v,actor.state_dict()[k]) for k,v in payload['actor'].items())
    assert not co.state and ao.state_dict()['state']
    for index,state in payload['actor_optimizer']['state'].items():
        for key,value in state.items():
            torch.testing.assert_close(value,ao.state_dict()['state'][index][key],atol=0,rtol=0)
    assert record['critic_shared_weights_preserved'] and record['future_branch_zero_initialized']
    assert not record['critic_and_adam_preserved'] and not record['independent_new_mc_validation']
