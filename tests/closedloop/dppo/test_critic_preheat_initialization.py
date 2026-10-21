"""Critic初始化成果的身份隔离、拒绝原子性和预热后评估边界验证。

只在临时目录构造小型冻结编码器与非空Adam，不读取正式模型或推进PhysX。
验证同一固定Actor的独立价值成果可加载，同时不修改Actor、不迁移Actor优化器或
运行时钟；源码修改允许复用，随机核、物理配置、冻结编码器、验证质量或价值目标gamma/lambda改变必须在修改
Critic前拒绝。
另外检查预热恢复需要真正初始评估，Actor已更新时不能补造Stage1基线。
4/8秒窗口只允许共享完整纯MC初始化；任何含TD的成果必须匹配真实TD切片时长。
"""
import copy
from types import SimpleNamespace
import pytest
import torch

from gem.closedloop.dppo.critic_preheat import (
    MATCHED, ARCHITECTURE, SCHEMA_PREHEAT, _value_targets, _supervision, initialize_critic)
from gem.closedloop.dppo.evaluation import _model_fingerprint
from gem.closedloop.dppo.training import _requires_initial_evaluation
from gem.closedloop.dppo.training_checkpoint import file_sha256


def fixture(tmp_path):
    actor = torch.nn.Linear(2,2)
    critic = torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.Linear(2,1))
    critic[0].requires_grad_(False)
    optimizer = torch.optim.AdamW(critic[1].parameters(), lr=1e-4)
    critic(torch.ones(3,2)).sum().backward();optimizer.step();critic.zero_grad(set_to_none=True)
    identity = {k:{'identity':k} for k in MATCHED}
    identity.update(critic={**{k:k for k in ARCHITECTURE},'complete_mc':{'mix_candidates':[0.,.5,1.]}},
        training={'gamma_upper':.99,'lambda_upper':.95},ppo_rollout={'world_seconds':4.},
        source_files={'gmt':{'physics.py':'fixed'}})
    for name in ['actor.py','contracts.py','online_conditions.py','training.py','evaluation_music.py']:
        p=tmp_path/'gem/closedloop'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('value = 1\n')
    data=dict(schema=SCHEMA_PREHEAT,actor_fingerprint=_model_fingerprint(actor),
        matched_identity={k:identity[k] for k in MATCHED},architecture={k:identity['critic'][k] for k in ARCHITECTURE},
        value_targets=_value_targets(identity),
        physical_sources=identity['source_files']['gmt'],behavior_sources={'actor.py':'historical_source'},
        critic=copy.deepcopy(critic.state_dict()),critic_optimizer=copy.deepcopy(optimizer.state_dict()),
        selected_updates=1,prior_mc_control_steps=500,prior_mc_physics_steps=2000,source={'initialization_only':True},
        validation=dict(selected_mc_ratio=1.,quality_gate_passed=True,
            relative_validation_mse_reduction=.7,after={'explained_variance':.6}))
    data['supervision']=_supervision(identity,data['validation'])
    return actor,critic,optimizer,identity,data


def load(tmp_path, actor, critic, optimizer, identity, data):
    path=tmp_path/'preheat.pt';torch.save(data,path)
    return initialize_critic(dict(path=str(path),sha256=file_sha256(path)),identity=identity,
        actor=actor,critic=critic,optimizer=optimizer,genmo_repo=tmp_path,
        quality_gate=dict(minimum_relative_mse_reduction=.3,minimum_explained_variance=.2))


def test_preheated_value_and_nonempty_adam_restore_without_actor_change(tmp_path):
    actor,critic,optimizer,identity,data=fixture(tmp_path)
    fingerprint=_model_fingerprint(actor)
    with torch.no_grad():critic[1].weight.add_(2.)
    optimizer.state.clear()
    result=load(tmp_path,actor,critic,optimizer,identity,data)
    torch.testing.assert_close(critic[1].weight,data['critic']['1.weight'],rtol=0,atol=0)
    assert optimizer.state and result['prior_mc_control_steps']==500
    assert _model_fingerprint(actor)==fingerprint and result['fresh_PhysX']


@pytest.mark.parametrize('changed',['kernel','physics','encoder','quality','gamma','lambda','legacy'])
def test_mismatch_rejected_before_value_or_adam_mutation(tmp_path,changed):
    actor,critic,optimizer,identity,data=fixture(tmp_path)
    if changed=='kernel':data['matched_identity']['kernel']={'different':True}
    if changed=='physics':data['matched_identity']['physical_configuration']={'control_hz':100}
    if changed=='encoder':data['critic']['0.weight'].add_(1.)
    if changed=='quality':data['validation']['after']['explained_variance']=.1
    if changed=='gamma':identity['training']['gamma_upper']=.97
    if changed=='lambda':identity['training']['lambda_upper']=.8
    if changed=='legacy':data.pop('value_targets')
    before=copy.deepcopy(critic.state_dict());step=next(iter(optimizer.state.values()))['step'].clone()
    with pytest.raises(ValueError):load(tmp_path,actor,critic,optimizer,identity,data)
    for key,value in critic.state_dict().items():torch.testing.assert_close(value,before[key],rtol=0,atol=0)
    torch.testing.assert_close(next(iter(optimizer.state.values()))['step'],step,rtol=0,atol=0)


def test_preheat_resume_builds_initial_reference_and_never_invents_it_after_actor_update():
    args=SimpleNamespace(collect_only=False,critic_preheat_only=False)
    state=dict(actor_updates=0,policy_version=0)
    assert _requires_initial_evaluation(args,state,False)
    assert not _requires_initial_evaluation(args,state,True)
    args.critic_preheat_only=True;assert not _requires_initial_evaluation(args,state,False)
    args.critic_preheat_only=False;state['actor_updates']=1
    with pytest.raises(ValueError,match='cannot invent'):_requires_initial_evaluation(args,state,False)


@pytest.mark.parametrize('ratio',[0.,.5,1.])
@pytest.mark.parametrize('seconds',[4.,8.])
def test_only_complete_mc_supervision_is_independent_of_rollout_window(tmp_path,ratio,seconds):
    actor,critic,optimizer,identity,data=fixture(tmp_path)
    data['validation']['selected_mc_ratio']=ratio
    data['supervision']=_supervision(identity,data['validation'])
    identity['ppo_rollout']['world_seconds']=seconds
    before=copy.deepcopy(critic.state_dict())
    step=next(iter(optimizer.state.values()))['step'].clone()
    if ratio==1. or seconds==4.:
        result=load(tmp_path,actor,critic,optimizer,identity,data)
        assert result['supervision']==data['supervision']
    else:
        with pytest.raises(ValueError,match='TD labels'):
            load(tmp_path,actor,critic,optimizer,identity,data)
        for key,value in critic.state_dict().items():torch.testing.assert_close(value,before[key],rtol=0,atol=0)
        torch.testing.assert_close(next(iter(optimizer.state.values()))['step'],step,rtol=0,atol=0)


def test_source_edit_does_not_block_preheat_or_overwrite_record(tmp_path):
    actor,critic,optimizer,identity,data=fixture(tmp_path)
    identity['source_files']={'genmo':{'actor.py':'new'},'gmt':{'physics.py':'new'}}
    (tmp_path/'gem/closedloop/actor.py').write_text('value = 2\n')
    assert load(tmp_path,actor,critic,optimizer,identity,data)['actor_unchanged']
    assert data['physical_sources']=={'physics.py':'fixed'}
    assert data['behavior_sources']=={'actor.py':'historical_source'}


def test_mc_real_interaction_clock_excludes_retired_slot_physics_but_budget_keeps_it():
    from gem.closedloop.dppo.mc_calibration import mc_execution_counts
    assert mc_execution_counts({'control_steps':100,'physics_steps':400},
        {'actual_GMT_controls':70,'allocated_physics_steps':400})==[70,280,400]
    assert mc_execution_counts({'control_steps':0,'physics_steps':0})==[0,0,0]
    assert mc_execution_counts({'control_steps':70,'physics_steps':280})==[70,280,280]
    with pytest.raises(ValueError,match='counts disagree'):
        mc_execution_counts({'control_steps':100,'physics_steps':400},
            {'actual_GMT_controls':101,'allocated_physics_steps':400})
