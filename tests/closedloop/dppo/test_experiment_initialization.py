"""完整学习状态继承和零Actor更新继续采样的CPU回归验收。

输入为临时小网络、非空Adam及可解析预算账本，测试新实验仅更换有效批量时保留
Actor/Critic、两组Adam与进度；奖励或学习率变化不能借此静默通过。另验证有限回溯
无合格步时继续下一物理批次，但不伪造Actor更新/版本，不吞掉真正联合KL异常。
所有产物位于pytest临时目录，不启动Isaac或长训，不把协议替身当作物理收益。
"""
import copy
import pytest
import torch
from gem.closedloop.dppo.gradient_merge import BCGradientController
from gem.closedloop.dppo.training_checkpoint import SCHEMA,load_training_checkpoint
from gem.closedloop.dppo.run_management import TrainingBudget
from gem.closedloop.dppo.budget_ledger import IncrementalBudget,replay_budget
from gem.closedloop.dppo.training import _final_actor_acceptance,_advance_training_iteration


def test_new_batch_preserves_nonempty_adam_and_rejects_behavior_changes(tmp_path):
    actor=torch.nn.Linear(2,1);critic=torch.nn.Linear(2,1)
    ao=torch.optim.AdamW(actor.parameters(),lr=1e-6);co=torch.optim.AdamW(critic.parameters(),lr=1e-4)
    for model,opt in ((actor,ao),(critic,co)):
        model(torch.ones(1,2)).sum().backward();opt.step();opt.zero_grad()
    bc=BCGradientController()
    identity=dict(scale={'actor_minibatch_chains':4096},training={'actor_lr':1e-6,'bc_batch':512,
        'actor_minibatch_internal_transitions':16384,'max_actor_optimizer_steps':4},
        reward={'version':'same'},kernel={'eta':.1},evaluation={'every':3},
        kl={'calibration':{'optimization_sha256':'old'}})
    payload=copy.deepcopy(dict(schema=SCHEMA,identity=identity,actor=actor.state_dict(),critic=critic.state_dict(),
        actor_optimizer=ao.state_dict(),critic_optimizer=co.state_dict(),bc_controller=bc.state_dict(),
        state={'iteration':6,'actor_updates':24}))
    path=tmp_path/'checkpoint.pt';torch.save(payload,path)
    target=copy.deepcopy(identity);target['scale']['actor_minibatch_chains']=8192
    target['training'].update(bc_batch=1024,actor_minibatch_internal_transitions=32768,max_actor_optimizer_steps=2)
    target['evaluation']['every']=50;target['kl']['calibration']['optimization_sha256']='new'
    with pytest.raises(ValueError,match='configuration'):
        load_training_checkpoint(path,identity=target,actor=actor,critic=critic,actor_optimizer=ao,critic_optimizer=co,controller=bc)
    restored=load_training_checkpoint(path,identity=target,actor=actor,critic=critic,
        actor_optimizer=ao,critic_optimizer=co,controller=bc,new_experiment=True)
    assert restored['state']['iteration']==6 and ao.state and co.state
    for wrong in ('reward','training'):
        changed=copy.deepcopy(target)
        changed[wrong].update(version='different') if wrong=='reward' else changed['training'].update(actor_lr=3e-6)
        before=actor.weight.detach().clone()
        with pytest.raises(ValueError,match='configuration'):
            load_training_checkpoint(path,identity=changed,actor=actor,critic=critic,
                actor_optimizer=ao,critic_optimizer=co,controller=bc,new_experiment=True)
        assert torch.equal(actor.weight,before)
    for key,value in ao.state_dict()['state'].items():
        for name,item in value.items():
            if torch.is_tensor(item):assert torch.equal(item,payload['actor_optimizer']['state'][key][name])


def test_new_experiment_keeps_spent_budget_without_reset(tmp_path):
    limits=dict(accepted_iterations=200,optimizer_attempts=100,generations=1000,control_steps=1000,physics_steps=4000)
    parent=TrainingBudget(tmp_path/'parent.json',limits)
    parent.reserve('old',accepted_iterations=6,optimizer_attempts=24)
    child=TrainingBudget(tmp_path/'child.json',limits)
    child.initialize_from_checkpoint(parent.state_dict())
    assert child.state_dict()==parent.state_dict()
    child.reserve('new',accepted_iterations=1)
    assert parent.state_dict()['used']['accepted_iterations']==6
    assert child.state_dict()['used']['accepted_iterations']==7
    with pytest.raises(ValueError,match='unused'):child.initialize_from_checkpoint(parent.state_dict())


def test_production_sqlite_budget_inherits_by_transaction_and_replays(tmp_path):
    limits=dict(accepted_iterations=200,optimizer_attempts=100,generations=1000,control_steps=1000,physics_steps=4000)
    parent=IncrementalBudget(tmp_path/'parent.json',limits)
    parent.reserve('old',optimizer_attempts=24)
    parent.accept_iteration('old',identity='parent/1')
    child=IncrementalBudget(tmp_path/'child.json',limits)
    descriptor=(tmp_path/'child.json').read_bytes()
    child.initialize_from_checkpoint(parent.state_dict())
    expected=parent.state_dict();expected.pop('ledger')
    actual=child.state_dict();actual.pop('ledger')
    assert actual==expected
    assert (tmp_path/'child.json').read_bytes()==descriptor
    child.accept_iteration('new',identity='child/1')
    assert child.state['used']['accepted_iterations']==2
    assert parent.state['used']['accepted_iterations']==1
    child.close()
    reopened=IncrementalBudget(tmp_path/'child.json',limits)
    assert reopened.state['accepted_identities']==['parent/1','child/1']
    assert reopened.state['used']['optimizer_attempts']==24
    with pytest.raises(ValueError,match='unused'):reopened.initialize_from_checkpoint(expected)
    reopened.close();parent.close()


def test_no_usable_actor_step_is_normal_but_kl_failure_still_visible():
    limits=dict(chain_joint_mean=.2,chain_joint_p95=.6,chain_joint_p99=.9,chain_joint_max=2.)
    kl=dict(chain_joint_kl=dict(mean=0.,p95=0.,p99=0.,max=0.))
    actor=dict(optimizer_steps=0,steps=[],early_stop_reason='rollout_no_useful_trust_region_step')
    result=_final_actor_acceptance(actor,kl,limits)
    assert result['passed'] and result['continue_with_new_physical_rollout']
    state=dict(iteration=6,actor_updates=24,policy_version=6)
    _advance_training_iteration(state,actor)
    assert state==dict(iteration=7,actor_updates=24,policy_version=6,actor_update_skipped_rollouts=1)
    kl['chain_joint_kl']['max']=3.
    assert not _final_actor_acceptance(actor,kl,limits)['passed']
    with pytest.raises(RuntimeError,match='explicit normal'):
        _final_actor_acceptance(dict(actor,early_stop_reason=None),kl,limits)


def test_supervisor_recovers_same_experiment_instead_of_reinitializing(tmp_path):
    import json
    from tools.run_stage2_training_supervisor import training_command,latest_iteration
    init=tmp_path/'parent.pt';config=tmp_path/'config.yaml'
    first=training_command(config,tmp_path,200,init)
    assert first[-2:]==['--initialize-from-checkpoint',str(init)]
    (tmp_path/'latest.json').write_text(json.dumps({'iteration':9}))
    resumed=training_command(config,tmp_path,200,init)
    assert resumed[-2:]==['--resume','latest'] and '--initialize-from-checkpoint' not in resumed
    assert latest_iteration(tmp_path)==9
