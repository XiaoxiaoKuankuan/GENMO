"""Critic初始化成果的身份隔离、拒绝原子性和预热后评估边界验证。

只在临时目录构造小型冻结编码器与非空Adam，不读取正式模型或推进PhysX。
验证同一固定Actor的独立价值成果可加载，同时不修改Actor、不迁移Actor优化器或
运行时钟；随机核、物理源码、冻结编码器或验证质量改变必须在修改Critic前拒绝。
另外检查预热恢复需要真正初始评估，Actor已更新时不能补造Stage1基线。
"""
import copy
from types import SimpleNamespace
import pytest
import torch

from gem.closedloop.dppo.critic_preheat import (
    MATCHED, ARCHITECTURE, SCHEMA_PREHEAT, _critical_sources, initialize_critic)
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
    identity.update(critic={k:k for k in ARCHITECTURE},source_files={'gmt':{'physics.py':'fixed'}})
    for name in ['actor.py','contracts.py','online_conditions.py','training.py']:
        p=tmp_path/'gem/closedloop'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('value = 1\n')
    data=dict(schema=SCHEMA_PREHEAT,actor_fingerprint=_model_fingerprint(actor),
        matched_identity={k:identity[k] for k in MATCHED},architecture=identity['critic'],
        physical_sources=identity['source_files']['gmt'],behavior_sources=_critical_sources(tmp_path),
        critic=copy.deepcopy(critic.state_dict()),critic_optimizer=copy.deepcopy(optimizer.state_dict()),
        selected_updates=1,prior_mc_control_steps=500,prior_mc_physics_steps=2000,source={'initialization_only':True},
        validation=dict(quality_gate_passed=True,relative_validation_mse_reduction=.7,after={'explained_variance':.6}))
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


@pytest.mark.parametrize('changed',['kernel','physics','encoder','quality','code'])
def test_mismatch_rejected_before_value_or_adam_mutation(tmp_path,changed):
    actor,critic,optimizer,identity,data=fixture(tmp_path)
    if changed=='kernel':data['matched_identity']['kernel']={'different':True}
    if changed=='physics':data['physical_sources']={'physics.py':'changed'}
    if changed=='encoder':data['critic']['0.weight'].add_(1.)
    if changed=='quality':data['validation']['after']['explained_variance']=.1
    if changed=='code':(tmp_path/'gem/closedloop/actor.py').write_text('value = 2\n')
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
