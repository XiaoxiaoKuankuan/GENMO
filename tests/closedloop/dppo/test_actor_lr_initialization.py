"""显式Actor学习率迁移的完整学习状态与拒绝边界回归。

数据流为真实CPU小网络完成数次AdamW更新，保存当前完整checkpoint，再通过生产
加载器进入新配置。分别覆盖单组与固定倍率多组优化器：仅组lr允许变化，网络、
Adam矩/步数、Critic Adam、BC和历史训练状态须原样保留；相同后续梯度必须得到与
手动load_state_dict再set_actor_base_lr完全一致的Adam更新，不用空优化器伪证恢复。

同时覆盖未授权gamma/reward/kernel/KL/Critic/trust改动、损坏模型/Adam/控制器在
写入前拒绝；新checkpoint可按新合同普通resume，旧LR合同拒绝。直接执行真实CLI
的非法参数组合，验证在Torch/Isaac启动前退出；监督器首次传迁移开关，latest出现
后仅resume。训练state辅助路径保留MC完成状态、RNG/collector及预算，轮次10到11。
所有checkpoint和CLI临时输出仅使用pytest临时目录，禁止CUDA/物理训练，不代表
八卡、真实模型或物理收益已验收；最终由正式固定数据候选结果决定是否采用新LR。
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from gem.closedloop.dppo.actor_optimizer import actor_base_lr, set_actor_base_lr
from gem.closedloop.dppo.gradient_merge import BCGradientController
from gem.closedloop.dppo.parallel_support import capture_local_rng, restore_local_rng
from gem.closedloop.dppo.training_checkpoint import SCHEMA, load_training_checkpoint

ROOT=Path(__file__).resolve().parents[3]


def _equal(left,right):
    if isinstance(left,torch.Tensor):
        assert left.dtype==right.dtype and left.shape==right.shape
        assert torch.equal(left,right)
    elif isinstance(left,np.ndarray):
        np.testing.assert_array_equal(left,right)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left:_equal(left[key],right[key])
    elif isinstance(left,(list,tuple)):
        assert type(left) is type(right) and len(left)==len(right)
        for a,b in zip(left,right):_equal(a,b)
    else:assert left==right


def _networks(lr,grouped=True):
    actor=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Tanh(),torch.nn.Linear(4,2))
    critic=torch.nn.Linear(3,1)
    if grouped:
        groups=[]
        for prefix,name,multiplier in (('0.','body',.25),('2.','head',1.)):
            entries=[(n,p) for n,p in actor.named_parameters() if n.startswith(prefix)]
            groups.append(dict(params=[p for _,p in entries],param_names=[n for n,_ in entries],
                name=name,lr_multiplier=multiplier,lr=lr*multiplier))
        ao=torch.optim.AdamW(groups,lr=lr,weight_decay=0.,amsgrad=True)
    else:ao=torch.optim.AdamW(actor.parameters(),lr=lr,weight_decay=0.,amsgrad=True)
    co=torch.optim.AdamW(critic.parameters(),lr=1e-4,weight_decay=.01)
    return dict(actor=actor,critic=critic,actor_optimizer=ao,critic_optimizer=co,
                controller=BCGradientController())


def _identity(lr):
    return dict(training=dict(actor_lr=lr,trust_region=dict(maximum_lr=lr,growth=1.25,margin=.8,
        max_attempts=10,minimum_trial_lr_ratio=1/512),gamma_upper=.95,ppo_clip=.1,
        actor_trainable_scope='all',actor_minibatch_internal_transitions=40960,
        max_actor_optimizer_steps=10,bc_batch=0),scale=dict(actor_minibatch_chains=8192),
        evaluation={'seed':42},critic=dict(learning_rate=1e-4),kernel={'eta':.1},
        reward={'version':'unchanged'},kl={'chain_joint_mean':10.,'calibration':{'optimization_sha256':'old'}},
        physical_episode={'safety_limit_seconds':20.})


def _snapshot(objects):
    return copy.deepcopy({key:value.state_dict() for key,value in objects.items()})


def _fixture(tmp_path,grouped=True):
    source=_networks(3e-6,grouped)
    x=torch.arange(12,dtype=torch.float32).reshape(4,3)/10
    for _ in range(3):
        for name in ('actor','critic'):
            optimizer=source[name+'_optimizer']
            source[name](x).square().sum().backward()
            optimizer.step();optimizer.zero_grad(set_to_none=True)
    source['controller'].update(2.,3.)
    # 故意保存较小实际组LR，证明审计不会将配置nominal误当旧Adam实际LR。
    set_actor_base_lr(source['actor_optimizer'],1.5e-6)
    generators={'actor':torch.Generator().manual_seed(123)}
    payload=dict(schema=SCHEMA,identity=_identity(3e-6),**_snapshot(source),
        state=dict(iteration=10,actor_updates=48,critic_updates=7680,policy_version=10,
            mc_preheat_complete=True,mc_quality_passed=True,next_mc_calibration_control_steps=100000000,
            selected_actor_lr=3e-6,last_accepted_actor_lr=1.5e-6,best_observed={'old':True}),
        budget={'used':{'optimizer_attempts':135}},
        rank_states=[dict(rng=capture_local_rng(generators),bc={'seen':15},collector={'cursor':31})])
    payload['bc_controller']=payload.pop('controller')
    path=tmp_path/'parent.pt';torch.save(payload,path)
    target=_networks(3e-7,grouped)
    identity=_identity(3e-7)
    identity['scale']['actor_minibatch_chains']=81920
    identity['training'].update(actor_minibatch_internal_transitions=409600,max_actor_optimizer_steps=1)
    identity['kl']['calibration']['optimization_sha256']='new'
    return path,payload,target,identity


@pytest.mark.parametrize('grouped',[False,True])
def test_nonempty_adam_only_changes_lr_and_next_update_matches_manual_reference(tmp_path,grouped):
    path,payload,target,identity=_fixture(tmp_path,grouped)
    rng=capture_local_rng()
    loaded=load_training_checkpoint(path,identity=identity,**target,new_experiment=True,actor_lr_transfer=True)
    _equal(capture_local_rng(),rng)
    for name in ('actor','critic','critic_optimizer'):_equal(target[name].state_dict(),payload[name])
    _equal(target['controller'].state_dict(),payload['bc_controller'])
    actual=copy.deepcopy(target['actor_optimizer'].state_dict())
    expected=copy.deepcopy(payload['actor_optimizer'])
    assert expected['state'] and all(v['step']==3 for v in expected['state'].values())
    for group in expected['param_groups']:group['lr']=3e-7*group.get('lr_multiplier',1.)
    _equal(actual,expected)
    _equal(loaded['state'],payload['state']);_equal(loaded['rank_states'],payload['rank_states'])
    _equal(loaded['budget'],payload['budget'])
    audit=loaded['actor_lr_transfer']
    assert audit['previous_actor_base_lr']==3e-6 and audit['previous_optimizer_base_lr']==1.5e-6
    assert audit['actor_base_lr']==3e-7 and audit['actor_optimizer_changed_fields']==['param_groups.lr']
    assert audit['actor_learning_rates']==({'body':7.5e-8,'head':3e-7} if grouped else {'uniform':3e-7})
    manual=_networks(3e-7,grouped)
    manual['actor'].load_state_dict(payload['actor'])
    manual['actor_optimizer'].load_state_dict(payload['actor_optimizer'])
    set_actor_base_lr(manual['actor_optimizer'],3e-7)
    for index,(a,b) in enumerate(zip(target['actor'].parameters(),manual['actor'].parameters())):
        grad=torch.arange(a.numel(),dtype=a.dtype).reshape_as(a)/100+index+.1
        a.grad=grad.clone();b.grad=grad.clone()
    target['actor_optimizer'].step();manual['actor_optimizer'].step()
    _equal(target['actor'].state_dict(),manual['actor'].state_dict())
    _equal(target['actor_optimizer'].state_dict(),manual['actor_optimizer'].state_dict())


@pytest.mark.parametrize('options',[{},dict(new_experiment=True),dict(actor_lr_transfer=True),
    dict(new_experiment=True,actor_lr_transfer=True,fragment_transfer=True)])
def test_lr_change_requires_exclusive_explicit_new_experiment_before_any_mutation(tmp_path,options):
    path,_,target,identity=_fixture(tmp_path)
    before=_snapshot(target);rng=capture_local_rng()
    with pytest.raises(ValueError):load_training_checkpoint(path,identity=identity,**target,**options)
    _equal(_snapshot(target),before);_equal(capture_local_rng(),rng)


@pytest.mark.parametrize('keys,value',[
    (('training','gamma_upper'),.99),(('training','ppo_clip'),.2),
    (('training','trust_region','growth'),1.1),(('training','trust_region','margin'),.7),
    (('training','trust_region','max_attempts'),11),
    (('training','trust_region','minimum_trial_lr_ratio'),.1),
    (('training','actor_lr_multipliers'),{'other':.5}),
    (('critic','learning_rate'),2e-4),(('kernel','eta'),.2),
    (('reward','version'),'changed'),(('kl','chain_joint_mean'),4.),
    (('physical_episode','safety_limit_seconds'),10.),
])
def test_other_contract_changes_are_rejected_before_mutation(tmp_path,keys,value):
    path,_,target,identity=_fixture(tmp_path)
    cursor=identity
    for key in keys[:-1]:cursor=cursor[key]
    cursor[keys[-1]]=value
    before=_snapshot(target)
    with pytest.raises(ValueError):
        load_training_checkpoint(path,identity=identity,**target,new_experiment=True,actor_lr_transfer=True)
    _equal(_snapshot(target),before)


@pytest.mark.parametrize('side',['previous','current'])
@pytest.mark.parametrize('value',[0.,-1.,float('nan'),float('inf'),True,1e-7])
def test_both_lr_contracts_require_equal_positive_finite_base_and_maximum(tmp_path,side,value):
    path,payload,target,identity=_fixture(tmp_path)
    bad=payload['identity'] if side=='previous' else identity
    bad['training']['actor_lr']=value
    torch.save(payload,path)
    before=_snapshot(target)
    with pytest.raises(ValueError):
        load_training_checkpoint(path,identity=identity,**target,new_experiment=True,actor_lr_transfer=True)
    _equal(_snapshot(target),before)


@pytest.mark.parametrize('damage',['actor','critic','actor_moment','critic_moment','group_name',
    'group_multiplier','parameter_names','controller','saved_lr'])
def test_corrupt_snapshot_rejected_before_any_network_or_optimizer_write(tmp_path,damage):
    path,payload,target,identity=_fixture(tmp_path)
    if damage in ('actor','critic'):next(iter(payload[damage].values())).flatten()[0]=float('nan')
    elif damage.endswith('_moment'):
        optimizer=payload[damage.split('_')[0]+'_optimizer']
        next(iter(optimizer['state'].values()))['exp_avg'].flatten()[0]=float('nan')
    elif damage=='group_name':payload['actor_optimizer']['param_groups'][0]['name']='unexpected'
    elif damage=='group_multiplier':payload['actor_optimizer']['param_groups'][0]['lr_multiplier']=.5
    elif damage=='parameter_names':payload['actor_optimizer']['param_groups'][0]['param_names'].reverse()
    elif damage=='controller':payload['bc_controller']['updates']=-1
    else:payload['actor_optimizer']['param_groups'][0]['lr']=float('nan')
    torch.save(payload,path);before=_snapshot(target)
    with pytest.raises(ValueError):
        load_training_checkpoint(path,identity=identity,**target,new_experiment=True,actor_lr_transfer=True)
    _equal(_snapshot(target),before)


def test_new_snapshot_resumes_only_new_contract_and_preserves_training_progress(tmp_path):
    from gem.closedloop.dppo.training import (
        _advance_training_iteration,_checkpoint_initialization_identity,_initialized_experiment_state,
    )
    path,payload,target,identity=_fixture(tmp_path)
    loaded=load_training_checkpoint(path,identity=identity,**target,new_experiment=True,actor_lr_transfer=True)
    record=dict(checkpoint=str(path),sha256='source-sha',actor_fingerprint='actor-hash',source_iteration=10,
                **loaded['actor_lr_transfer'])
    initialization=_checkpoint_initialization_identity(record)
    assert initialization['kind']=='explicit_checkpoint_actor_lr'
    assert initialization['transfer']=='actor_learning_rate_only.v1'
    assert initialization['actor_learning_rates']['actor_base_lr']==3e-7
    state=_initialized_experiment_state(loaded,record)
    assert state['mc_preheat_complete'] and state['mc_quality_passed']
    assert state['next_mc_calibration_control_steps']==100000000
    assert state['selected_actor_lr']==3e-7 and state['last_accepted_actor_lr']==1.5e-6
    assert (state['iteration'],state['actor_updates'],state['critic_updates'],state['policy_version'])==(10,48,7680,10)
    assert not any(state.get(k) for k in ('requires_fragment_value_validation','requires_stage1_rebaseline',
                                        'requires_transfer_start_evaluation'))
    identity['actor_initialization']=initialization
    saved=dict(loaded,identity=identity,state=state,actor=target['actor'].state_dict(),
        critic=target['critic'].state_dict(),actor_optimizer=target['actor_optimizer'].state_dict(),
        critic_optimizer=target['critic_optimizer'].state_dict(),bc_controller=target['controller'].state_dict())
    saved.pop('actor_lr_transfer')  # 正式快照从当前网络/state构造，临时加载审计不作为顶层字段保存。
    child=tmp_path/'experiment_initialized.pt';torch.save(saved,child)
    resumed=_networks(3e-7)
    got=load_training_checkpoint(child,identity=identity,**resumed)
    _equal(_snapshot(resumed),_snapshot(target));_equal(got['state'],state)
    # 使用正式RNG恢复接口验证显式生成器游标；恢复后再还原测试进程，避免污染其他测试。
    rng=capture_local_rng();generators={'actor':torch.Generator()}
    try:
        restore_local_rng(got['rank_states'][0]['rng'],generators)
        _equal(capture_local_rng(generators),payload['rank_states'][0]['rng'])
    finally:restore_local_rng(rng)
    old=copy.deepcopy(identity);old['training']['actor_lr']=3e-6;old['training']['trust_region']['maximum_lr']=3e-6
    before=_snapshot(resumed)
    with pytest.raises(ValueError):load_training_checkpoint(child,identity=old,**resumed)
    _equal(_snapshot(resumed),before)
    _advance_training_iteration(state,dict(optimizer_steps=1,steps=[{'learning_rate':3e-7}]))
    assert (state['iteration'],state['actor_updates'],state['policy_version'])==(11,49,11)


def test_ordinary_batch_initialization_still_preserves_saved_learning_rate(tmp_path):
    path,payload,target,identity=_fixture(tmp_path)
    identity['training']['actor_lr']=3e-6
    identity['training']['trust_region']['maximum_lr']=3e-6
    loaded=load_training_checkpoint(path,identity=identity,**target,new_experiment=True)
    assert 'actor_lr_transfer' not in loaded
    assert actor_base_lr(target['actor_optimizer'])==1.5e-6
    _equal(target['actor_optimizer'].state_dict(),payload['actor_optimizer'])


def test_supervisor_passes_lr_transfer_once_and_keeps_default_disabled(tmp_path):
    from tools.run_stage2_training_supervisor import training_command
    config=tmp_path/'config.yaml';source=tmp_path/'source.pt'
    normal=training_command(config,tmp_path,15,source)
    assert '--initialize-actor-lr' not in normal
    first=training_command(config,tmp_path,15,source,actor_lr_transfer=True)
    assert first[-3:]==['--initialize-from-checkpoint',str(source),'--initialize-actor-lr']
    (tmp_path/'latest.json').write_text(json.dumps({'iteration':10}))
    resumed=training_command(config,tmp_path,15,source,actor_lr_transfer=True)
    assert resumed[-2:]==['--resume','latest'] and '--initialize-actor-lr' not in resumed
    assert '--initialize-from-checkpoint' not in resumed
    for source_value,fragments in ((None,False),(source,True)):
        with pytest.raises(ValueError):training_command(config,tmp_path,15,source_value,
            actor_lr_transfer=True,random_fragments=fragments)


@pytest.mark.parametrize('extra',[[],['--initialize-from-checkpoint','source.pt','--resume','latest'],
    ['--initialize-from-checkpoint','source.pt','--initialize-random-fragments'],
    ['--initialize-from-checkpoint','source.pt','--initialize-denoising-reward']])
def test_training_cli_invalid_lr_transfer_fails_before_cuda_or_output_creation(tmp_path,extra):
    output=tmp_path/'never_created'
    result=subprocess.run([sys.executable,'-B',str(ROOT/'tools/train_closedloop_stage10.py'),
        '--output-dir',str(output),'--initialize-actor-lr',*extra],capture_output=True,text=True,
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=''),timeout=30)
    assert result.returncode==2
    assert ('explicit checkpoint initialization' in result.stderr or
            'requires a new training experiment' in result.stderr)
    assert not output.exists()


@pytest.mark.parametrize('extra',[[],['--initialize-from-checkpoint','source.pt','--initialize-random-fragments']])
def test_supervisor_cli_rejects_invalid_lr_transfer_before_starting_child(tmp_path,extra):
    output=tmp_path/'never_created';status=tmp_path/'no_status'
    result=subprocess.run([sys.executable,'-B',str(ROOT/'tools/run_stage2_training_supervisor.py'),
        '--config',str(tmp_path/'absent.yaml'),'--output-dir',str(output),'--status-dir',str(status),
        '--initialize-actor-lr',*extra],capture_output=True,text=True,
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=''),timeout=30)
    assert result.returncode==2 and 'explicit checkpoint initialization' in result.stderr
    assert not output.exists() and not status.exists()
