"""20秒随机运动片段、重置前价值自举与显式实验继承的CPU验收。

输入为具有真实50Hz计数契约的上层环境替身、解析奖励、小型多参数组Adam和临时
checkpoint。测试从接管后19.5秒开始等待生成的跨界动作，证明物理推进精确停在
20秒、未提交票据释放、失败优先、价值使用旧episode末态且GAE不流入重置episode。
独立手算以0.5秒为单位的gamma变换，并核验旧Actor/Adam的所有参数组、动量和RNG
记录不因新gamma被重置。对真正改变奖励或采样核的载入仍应拒绝，普通resume不能
隐式改目标。另验证新实验重新建立冻结Stage1基线、固定价值参考及累计进度。
所有文件位于pytest临时目录；不启动Isaac，不把替身状态机结果当成物理收益。
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from gem.closedloop.dppo.gradient_merge import BCGradientController
from gem.closedloop.dppo.runtime import episode_boundary_summary
from gem.closedloop.dppo.trainer import fixed_targets
from gem.closedloop.dppo.training_checkpoint import SCHEMA,load_training_checkpoint
from gem.closedloop.dppo.training_config import configuration
from gem.closedloop.dppo.training import _fragment_value_initialization
from tests.closedloop.dppo.test_delayed_credit import transition
from tests.closedloop.dppo.test_env_adapter import adapter
from tests.closedloop.dppo.test_data_learning import actual_step


@pytest.mark.parametrize('failure',[False,True])
def test_exact_twenty_seconds_including_pending_generation_and_failure_priority(tmp_path,failure):
    env,backend=adapter(tmp_path,elapsed=.75)
    backend.tick=600+int(19.5*600)
    env.snapshot=backend.snapshot()
    env.music_end_tick=600+120*600
    env.soft_end_tick=600+20*600
    # 从19.5秒开始的替身也必须先补齐真实契约要求的0.5秒因果活动度历史。
    last=(backend.tick-612)//12
    for index in range(last-24,last+1):env.reward.evaluate_step(actual_step(index))
    if failure:backend.fail_at=env.soft_end_tick
    row=env.step()
    assert row.control_tick_begin==12300 and row.control_tick_end==12600
    assert row.executed_control_steps==25 and row.executed_physics_steps==100
    assert backend.tick==12600
    assert not any(name=='commit_plan' for name,_ in backend.calls)
    assert any(name=='discard_plan' for name,_ in backend.calls)
    assert row.metadata['physical_episode_elapsed_seconds']==20.
    assert row.terminated is failure and row.truncated is not failure
    assert (row.next_context is None) is failure
    if not failure:
        assert row.metadata['time_limit_bootstrap']
        assert row.metadata['next_critic_context']['decision_time'].item()==21.
        summary=episode_boundary_summary([row])
        assert summary['time_limit_episodes']==summary['bootstrap_time_limit_episodes']==1
        assert summary['elapsed_seconds_min']==summary['elapsed_seconds_max']==20.
    else:
        assert not row.metadata['time_limit_bootstrap']
        assert not row.metadata['physical_time_limit_reached']
        assert row.metadata['event_penalty_total']==-5.


def test_new_episode_never_supplies_timeout_value_and_discount_is_physical_time():
    a=transition(12300,reward=1.);a.truncated=True;a.next_value=4.
    b=transition(600,episode='reset',terminal=True,reward=1000.)
    out=fixed_targets([a,b],None,'cpu',reuse_values=True,critic_version=0,
        gamma_upper=.95,lambda_upper=.98)
    assert out['returns'][0].item()==pytest.approx(1.+.95*4.)
    assert out['returns'][1].item()==1000.
    a.terminated=True
    out=fixed_targets([a,b],None,'cpu',reuse_values=True,critic_version=0,
        gamma_upper=.95,lambda_upper=.98)
    assert out['returns'][0].item()==1.
    # 两个20ms控制区间，不能误当成一个0.5秒决策。
    a.terminated=False;a.executed_control_steps=2;a.rewards=torch.tensor([.2,.3])
    a.control_tick_end=a.control_tick_begin+24
    out=fixed_targets([a],None,'cpu',reuse_values=True,critic_version=0,
        gamma_upper=.95,lambda_upper=.98)
    assert out['returns'][0].item()==pytest.approx(.2+.95**(1/25)*.3+.95**(2/25)*4.,abs=2e-8)


@pytest.mark.parametrize('old_gamma,gamma,preheated,requires_fit',[
    (.99,.95,True,True),(.99,.99,True,False),(.99,.99,False,True)])
def test_only_changed_value_target_requires_unconditional_refit(old_gamma,gamma,preheated,requires_fit):
    state=dict(iteration=85,actor_updates=182,mc_preheat_complete=preheated)
    original=copy.deepcopy(state)
    result=_fragment_value_initialization(state,old_gamma,gamma)
    assert result['mc_preheat_complete'] is not requires_fit
    assert result['requires_fragment_value_validation'] is not requires_fit
    assert result['requires_stage1_rebaseline'] and result['requires_transfer_start_evaluation']
    assert state==original


def test_fragment_transfer_preserves_all_adam_groups_and_rejects_other_behavior_changes(tmp_path):
    actor=torch.nn.Sequential(torch.nn.Linear(2,3),torch.nn.Linear(3,1))
    critic=torch.nn.Linear(2,1)
    ao=torch.optim.AdamW([dict(params=list(actor[0].parameters()),lr=1e-7,name='transformer',lr_scale=.1),
        dict(params=list(actor[1].parameters()),lr=1e-6,name='output',lr_scale=1.)])
    co=torch.optim.AdamW(critic.parameters(),lr=1e-4)
    for model,opt in ((actor,ao),(critic,co)):
        model(torch.ones(1,2)).sum().backward();opt.step();opt.zero_grad()
    bc=BCGradientController()
    identity=dict(training={'gamma_upper':.99,'lambda_upper':.98,'actor_lr':1e-6},
        physical_episode={'safety_limit_seconds':None,'termination':'music_end_or_physical_failure'},
        dataset={'standing_start_probability':.2,'random_start':True},
        critic={'dimension':256,'preheat_artifact':{'path':'old'}},
        kernel={'eta':.1},reward={'version':'v3'})
    payload=copy.deepcopy(dict(schema=SCHEMA,identity=identity,actor=actor.state_dict(),critic=critic.state_dict(),
        actor_optimizer=ao.state_dict(),critic_optimizer=co.state_dict(),bc_controller=bc.state_dict(),
        state={'iteration':85,'actor_updates':182},rank_states=[{'rng':torch.get_rng_state()}]))
    path=tmp_path/'parent.pt';torch.save(payload,path)
    target=copy.deepcopy(identity)
    target['training']['gamma_upper']=.95
    target['physical_episode']=dict(safety_limit_seconds=20.,termination='music_end_or_physical_failure_or_time_limit',
        time_limit_bootstrap=True)
    target['dataset']['standing_start_probability']=0.
    target['critic']['preheat_artifact']=None
    kwargs=dict(identity=target,actor=actor,critic=critic,actor_optimizer=ao,critic_optimizer=co,controller=bc)
    with pytest.raises(ValueError):load_training_checkpoint(path,**kwargs,new_experiment=True)
    with pytest.raises(ValueError):load_training_checkpoint(path,**kwargs,fragment_transfer=True)
    ao.state.clear()
    restored=load_training_checkpoint(path,**kwargs,new_experiment=True,fragment_transfer=True)
    assert restored['state']['iteration']==85 and restored['state']['actor_updates']==182
    assert torch.equal(restored['rank_states'][0]['rng'],payload['rank_states'][0]['rng'])
    assert ao.state_dict()['param_groups']==payload['actor_optimizer']['param_groups']
    for key,state in ao.state_dict()['state'].items():
        for name,value in state.items():
            torch.testing.assert_close(value,payload['actor_optimizer']['state'][key][name],rtol=0,atol=0)
    for group,key,value in [('reward','version','other'),('kernel','eta',.2),('training','lambda_upper',.5),
        ('training','actor_lr',3e-6),('critic','dimension',512)]:
        changed=copy.deepcopy(target);changed[group][key]=value
        before=actor[0].weight.detach().clone()
        with pytest.raises(ValueError):load_training_checkpoint(path,**dict(kwargs,identity=changed),
            new_experiment=True,fragment_transfer=True)
        assert torch.equal(before,actor[0].weight)


@pytest.mark.parametrize('gamma',[.95,.99])
def test_fragment_configuration_keeps_transformer_kernel_and_full_music_evaluation(gamma):
    root=Path(__file__).resolve().parents[3]
    base=configuration(root/'configs/closedloop/stage2_batch_8192.yaml')
    config=configuration(root/f'configs/closedloop/stage2_random20s_gamma{round(gamma*100):03d}.yaml')
    stage=config['stage10']
    assert stage['physical_episode']['safety_limit_seconds']==20.
    assert stage['physical_episode']['time_limit_bootstrap']
    assert stage['dataset']['standing_start_probability']==0.
    assert stage['training']['gamma_upper']==gamma and stage['training']['lambda_upper']==.98
    assert stage['training']['actor_trainable_scope']=='all' and not stage['training']['bc_enabled']
    assert stage['training']['std_schedule']==base['stage10']['training']['std_schedule']
    assert stage['reward']==base['stage10']['reward'] and config['timing']==base['timing']
    assert stage['evaluation']['episode_seconds'] is None
    assert stage['scale']['actor_minibatch_chains']==8192 and stage['ppo_rollout']['world_seconds']==8.


def test_supervisor_uses_fragment_transfer_once_then_exact_resume(tmp_path):
    from tools.run_stage2_training_supervisor import training_command
    config=tmp_path/'config.yaml';parent=tmp_path/'parent.pt'
    first=training_command(config,tmp_path,200,parent,random_fragments=True)
    assert '--initialize-random-fragments' in first
    (tmp_path/'latest.json').write_text(json.dumps({'iteration':85}))
    resumed=training_command(config,tmp_path,200,parent,random_fragments=True)
    assert resumed[-2:]==['--resume','latest']
    assert '--initialize-random-fragments' not in resumed


def test_new_discount_rebuilds_stage1_reference_without_resetting_current_actor(tmp_path,monkeypatch):
    from gem.closedloop.dppo.evaluation_rebaseline import ensure_training_progress_baseline
    from tests.closedloop.dppo.test_training_music_monitor import train_catalog,train_plan
    catalog=train_catalog(tmp_path);actor=torch.nn.Linear(2,2)
    frozen=copy.deepcopy(actor).requires_grad_(False)
    options=dict(split='train',purpose='training_progress',original_split='train',samples_per_source=4,
        seeds=[42,1729],episode_seconds=None,selection_seed=42,unique_music=True)
    state=dict(iteration=85,actor_updates=182,policy_version=85,requires_stage1_rebaseline=True)
    c=NS(output=tmp_path,session=tmp_path/'session',catalog=catalog,stage={'evaluation':options},state=state,
        actor=actor,policy=NS(actor=actor,frozen_policy=NS(actor=frozen)),writer=None,generators={},
        identity={'source_stage1_step':595000},distributed=NS(rank=0,broadcast_object=lambda x:x),
        budget=NS(state_dict=lambda:{'spent':123}))
    before=actor.weight.detach().clone()
    def evaluate(view,label,**kwargs):
        assert view.actor is frozen and c.policy.actor is actor
        assert view.state['actor_updates']==0 and label=='initial'
        plan=train_plan(catalog)
        (view.output/'fixed_diagnostic_reference.json').write_text(json.dumps({'gamma':.95,'new_reference':True}))
        view._periodic_fixed_reference=('new_actor_reference','new_gamma_value_reference',{})
        return {'plan_sha256':plan['plan_sha256']}
    monkeypatch.setattr('gem.closedloop.dppo.evaluation_runtime._evaluate',evaluate)
    change=ensure_training_progress_baseline(c)
    assert change['reason']=='new_fragment_value_target' and change['archive'] is None
    assert change['MC_preheat_repeated'] and not change['optimizer_reset']
    assert state['iteration']==85 and state['actor_updates']==182
    assert 'requires_stage1_rebaseline' not in state
    assert c._periodic_fixed_reference[1]=='new_gamma_value_reference'
    assert json.loads((tmp_path/'fixed_diagnostic_reference.json').read_text())['gamma']==.95
    torch.testing.assert_close(actor.weight,before,rtol=0,atol=0)
