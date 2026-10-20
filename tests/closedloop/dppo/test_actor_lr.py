"""跨轮统一学习率、逐步回溯下限及checkpoint恢复的回归。

使用显式KL统计验证P95或最大单步超近门槛时不会被平均值掩盖；低KL必须连续出现，
完整步骤且无回溯才允许加速。真实小网络/非空Adam测试回溯预算与恢复；checkpoint
同时恢复学习率控制器，损坏的控制器必须在模型被修改前拒绝。这里不宣称物理收益。
"""
import copy
import pytest
import torch

from gem.closedloop.dppo.actor_lr import ActorLRController, kl_utilization
from gem.closedloop.dppo.actor_backtracking import backtracked_step, StepSnapshot
from tests.closedloop.dppo.test_actor_backtracking import setup, Collective, kl, equal_state

CONFIG=dict(minimum=5e-8,maximum=1e-6,decrease_factor=2/3,increase_factor=1.2,
            low_utilization=.4,high_utilization=.8,low_rounds=3)
LIMITS=dict(mean_coordinate=.0009,maximum_step_mean_coordinate=.0018,p95_coordinate=.0021)


def report(u):
    return dict(mean_coordinate_kl=u*.0009,maximum_step_mean_coordinate_kl=u*.0018,
                coordinate_kl={'p95':u*.0021})


def test_three_complete_low_rounds_increase_then_backtrack_decreases():
    c=ActorLRController(3e-7,CONFIG)
    for _ in range(2):
        c.update(report(.2),LIMITS,accepted=4,planned=4,attempts=4)
        assert c.base_lr==3e-7
    c.update(report(.2),LIMITS,accepted=4,planned=4,attempts=4)
    assert c.base_lr==pytest.approx(3.6e-7)
    c.update(report(.2),LIMITS,accepted=4,planned=4,attempts=5)
    assert c.base_lr==pytest.approx(2.4e-7) and c.low_streak==0
    for _ in range(20):c.update(report(0),LIMITS,accepted=0,planned=4,attempts=1)
    assert c.base_lr==5e-8


def test_tail_kl_and_partial_updates_cannot_accelerate():
    c=ActorLRController(3e-7,CONFIG);r=report(.1);r['coordinate_kl']['p95']=.002
    assert kl_utilization(r,LIMITS)==pytest.approx(.002/.0021)
    c.update(r,LIMITS,accepted=4,planned=4,attempts=4)
    assert c.base_lr==pytest.approx(2e-7)
    c.update(report(.1),LIMITS,accepted=2,planned=4,attempts=2)
    assert c.low_streak==0 and c.base_lr<2e-7
    with pytest.raises(FloatingPointError):kl_utilization(report(float('nan')),LIMITS)


def test_minimum_lr_limits_attempts_and_snapshot_storage_is_reused():
    actor,opt,control=setup();opt.param_groups[0]['lr']=7.5e-8
    workspace=StepSnapshot();workspace.capture(actor,opt)
    pointer=workspace.model['weight'].data_ptr()
    original=copy.deepcopy((actor.state_dict(),opt.state_dict()))
    calls=[]
    result=backtracked_step(actor,opt,control,evaluate=lambda:kl(.1),
        limits={'mean_coordinate':.001},distributed=Collective(),max_attempts=3,
        minimum_lr=5e-8,workspace=workspace,reserve_attempt=lambda:calls.append(1))
    assert not result['accepted'] and len(calls)==1
    assert workspace.model['weight'].data_ptr()==pointer
    assert equal_state(original,(actor.state_dict(),opt.state_dict()))


def test_controller_roundtrip_and_corrupt_state_rejected():
    c=ActorLRController(3e-7,CONFIG)
    c.update(report(.1),LIMITS,accepted=4,planned=4,attempts=4)
    saved=c.state_dict();other=ActorLRController(3e-7,CONFIG);other.load_state_dict(saved)
    assert other.state_dict()==saved
    for k,v in [('base_lr',float('nan')),('low_streak',3),('rounds',-1)]:
        bad=copy.deepcopy(saved);bad[k]=v
        with pytest.raises(ValueError):other.load_state_dict(bad)
        assert other.state_dict()==saved


def test_checkpoint_restores_lr_state_and_rejects_corruption_before_model_write(tmp_path):
    from gem.closedloop.dppo.training_checkpoint import SCHEMA,load_training_checkpoint
    actor,opt,bc=setup();critic,co,_=setup();lr=ActorLRController(3e-7,CONFIG)
    lr.update(report(.1),LIMITS,accepted=4,planned=4,attempts=4)
    payload=copy.deepcopy(dict(schema=SCHEMA,identity={'case':'step_kl'},
        actor=actor.state_dict(),critic=critic.state_dict(),actor_optimizer=opt.state_dict(),
        critic_optimizer=co.state_dict(),bc_controller=bc.state_dict(),scheduler=lr.state_dict()))
    path=tmp_path/'state.pt';torch.save(payload,path)
    restored=ActorLRController(3e-7,CONFIG)
    with torch.no_grad():actor.weight.add_(.2)
    load_training_checkpoint(path,identity=payload['identity'],actor=actor,critic=critic,
        actor_optimizer=opt,critic_optimizer=co,controller=bc,lr_controller=restored)
    assert equal_state(actor.state_dict(),payload['actor'])
    assert restored.state_dict()==lr.state_dict()
    payload['scheduler']['base_lr']=float('nan');torch.save(payload,path)
    with torch.no_grad():actor.weight.add_(.3)
    before=copy.deepcopy(actor.state_dict())
    with pytest.raises(ValueError):
        load_training_checkpoint(path,identity=payload['identity'],actor=actor,critic=critic,
            actor_optimizer=opt,critic_optimizer=co,controller=bc,lr_controller=restored)
    assert equal_state(actor.state_dict(),before)
