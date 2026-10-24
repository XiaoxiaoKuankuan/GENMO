"""验证200轮正式训练的更新约束及恢复语义，不启动GPU或物理模拟。

用实际RoboFFT小模型验证符号相关裁剪与Autograd一致、被拒绝epoch的Actor和
非空Adam逐位恢复；用在线Critic真实前向检查整段留出、固定标签限制及最佳状态
恢复。另检查同结构未来Critic导入不伪造MC验收、正式配置边界和监督器续训入口。
所有生成checkpoint位于pytest临时目录；测试通过只代表实现合同，不证明物理收益。
"""
import copy
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
import torch

from tests.closedloop.test_stage1_actor import actor_factory
from tests.closedloop.dppo.test_robofft import cpu_threads, make_policy, make_rows, local_distributed, settings
from gem.closedloop.robofft.surrogate import action_objective
from gem.closedloop.robofft import updater, OBJECTIVE
from gem.closedloop.dppo.actor_optimizer import make_actor_optimizer
from gem.closedloop.dppo.learning_guards import (capture_learning, split_episodes, physical_guard,
                                                validate_formal_configuration)
from gem.closedloop.dppo.training_config import configuration
from gem.closedloop.dppo.training_checkpoint import schema_for_identity
from gem.closedloop.robofft.checkpoint import initialize_optimization_variant

ROOT=Path(__file__).resolve().parents[3]


def assert_tree_equal(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,atol=0,rtol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:assert_tree_equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_tree_equal(x,y)
    else:assert a==b


def test_actual_ppo_clipped_zero_gradient_depends_on_advantage_sign():
    ratio=torch.tensor([1.03,.97,1.03,.97,1.,1.03],dtype=torch.float64)
    loss=(-ratio.log()/.1)[:,None].requires_grad_()
    advantage=torch.tensor([1.,1.,-1.,-1.,1.,0.])
    report=action_objective(loss,torch.zeros_like(loss),advantage,beta_neg=.5)
    grad,=torch.autograd.grad(report['loss'].sum(),loss)
    assert report['ratio_clip_fraction'].tolist()==[1,1,1,1,0,1]
    assert report['ppo_zero_gradient_fraction'].tolist()==[1,0,0,1,0,0]
    assert (grad[:,0]==0).tolist()==[True,False,False,True,False,True]
    assert report['zero_advantage_fraction'].tolist()==[0,0,0,0,0,1]


def test_evaluation_uses_current_policy_version_instead_of_last_collector(tmp_path,monkeypatch):
    from tests.closedloop.dppo.test_parallel_periodic_evaluation import context,IndependentEnv
    from gem.closedloop.dppo import evaluation_runtime
    c,_=context(tmp_path,monkeypatch)
    c.state.update(iteration=3,policy_version=3)
    c.env.iteration=2;c.env.policy_version=2
    seen=[]
    class Observed(IndependentEnv):
        def reset_task(self,*args,**kwargs):
            seen.append((self.policy_version,self.iteration))
            return super().reset_task(*args,**kwargs)
    monkeypatch.setattr(evaluation_runtime,'UpperEnvironment',Observed)
    evaluation_runtime._evaluate(c,'initial')
    assert seen and set(seen)=={(3,3)}


def test_epoch_rejection_restores_actor_and_nonempty_adam(actor_factory):
    policy,context=make_policy(actor_factory)
    rows,targets,cache,manifest=make_rows(policy,context)
    opt=make_actor_optimizer(policy.actor,settings())
    p=next(p for p in policy.actor.parameters() if p.requires_grad)
    p.grad=torch.ones_like(p);opt.step();opt.zero_grad(set_to_none=True)
    updater.prepare_mc(policy,rows,cache,torch.Generator().manual_seed(19))
    before=capture_learning(policy.actor,opt)
    policy.robofft['update_guard']=dict(max_ppo_zero_gradient_fraction=1.,
        max_joint_change_rms_rad=1e-20,max_near_joint_change_rms_rad=1e-20,max_normalized_change_rms=1e-20)
    attempts=[]
    result=updater.actor_update(policy,opt,rows,targets,cache,manifest,distributed=local_distributed(),
        generator=torch.Generator().manual_seed(33),ppo_epochs=3,minibatch_actions=2,microbatch=2,
        reserve_attempt=lambda:attempts.append(1))
    assert result['optimizer_steps']==0 and result['attempted_optimizer_steps']==len(attempts)==1
    assert result['stop_reason']=='epoch_guard_rollback_actor_and_adam'
    assert not result['epoch_guard'][0]['passed']
    assert_tree_equal(before,capture_learning(policy.actor,opt))
    assert result['final_surrogate']['ratio']==pytest.approx(1.,abs=1e-6)
    cache.close()


def test_episode_holdout_keeps_all_adjacent_states_together():
    manifest=[dict(owner_rank=i%2,local_index=i//2) for i in range(24)]
    ids={(m['owner_rank'],m['local_index']):dict(backend_session_id='a',env_id=m['owner_rank'],
         episode_id=m['local_index']//3) for m in manifest}
    train,val=split_episodes(list(range(24)),manifest,ids,.2)
    key=lambda i:(manifest[i]['owner_rank'],ids[(manifest[i]['owner_rank'],manifest[i]['local_index'])]['episode_id'])
    assert not {key(i) for i in train}&{key(i) for i in val}
    assert sorted(train+val)==list(range(24))


def test_critic_fixed_reference_rejection_restores_best_adam(actor_factory):
    from gem.closedloop.dppo.pretrained_critic import PretrainedConditionCritic
    from gem.closedloop.dppo.tensor_cache import RolloutTensorCache
    from gem.closedloop.dppo.updater import critic_update
    policy,context=make_policy(actor_factory)
    original,targets,cache,_=make_rows(policy,context);cache.close()
    rows=[copy.deepcopy(original[i%2]) for i in range(12)]
    critic=PretrainedConditionCritic(policy.actor,dimension=32,layers=1,heads=4)
    optimizer=torch.optim.Adam(critic.parameters(),lr=.01)
    p=next(p for p in critic.parameters() if p.requires_grad)
    p.grad=torch.ones_like(p);optimizer.step();optimizer.zero_grad(set_to_none=True)
    records=[]
    for i,row in enumerate(rows):
        row.identity=dict(backend_session_id='test',episode_id=i//2,env_id=0)
        with torch.no_grad():value=float(critic(row.metadata['critic_context'],[5.])[0])
        records.append(dict(critic_context=row.metadata['critic_context'],remaining_music_seconds=5.,
                            target=value,initial_prediction=value))
    targets=dict(targets,advantages=torch.ones(12),returns=torch.full((12,),20.))
    cache=RolloutTensorCache(rows,targets,'cpu')
    manifest=[dict(valid=True,has_free=True,owner_rank=0,local_index=i) for i in range(12)]
    before=capture_learning(critic,optimizer)
    result=critic_update(critic,optimizer,rows,targets,cache,manifest,distributed=local_distributed(),
        generator=torch.Generator().manual_seed(2),epochs=5,batch=8,
        validation_guard=dict(holdout_fraction=.2,max_fixed_mse_increase=.05,patience_epochs=2),
        fixed_reference=dict(records=records))
    assert result['optimizer_steps']==0 and result['attempted_optimizer_steps']>0
    assert result['selected_epoch']==0 and result['epochs']==2
    assert result['training_actions']+result['validation_actions']==12
    assert_tree_equal(before,capture_learning(critic,optimizer))
    cache.close()


def test_formal_configuration_and_initialization_do_not_invent_mc_validation(tmp_path):
    from gem.closedloop.dppo.training import _validate_future_critic_release
    c=configuration(ROOT/'configs/closedloop/stage2_robofft_future_critic_formal_server1.yaml')
    validate_formal_configuration(c)
    _validate_future_critic_release(NS(future_critic_formal=True,initialize_robofft_variant='a.pt',stop_after_iteration=200),c)
    assert not c['stage10']['critic']['complete_mc']['enabled']
    broken=copy.deepcopy(c);broken['stage10']['evaluation']['persistent_world']=True
    with pytest.raises(ValueError,match='fresh'):validate_formal_configuration(broken)
    actor,critic=torch.nn.Linear(2,2),torch.nn.Linear(2,1)
    ao,co=torch.optim.Adam(actor.parameters()),torch.optim.Adam(critic.parameters())
    for model,opt in ((actor,ao),(critic,co)):
        model(torch.ones(1,2)).sum().backward();opt.step();opt.zero_grad(set_to_none=True)
    identity=dict(critic=dict(type='asymmetric_tracking_committed_temporal',complete_mc=dict(enabled=False)),
        training=dict(actor_objective=OBJECTIVE,robofft=dict(mc_samples=16)),runtime={})
    source=dict(identity=identity,schema=schema_for_identity(identity),actor=copy.deepcopy(actor.state_dict()),
        critic=copy.deepcopy(critic.state_dict()),actor_optimizer=copy.deepcopy(ao.state_dict()),
        critic_optimizer=copy.deepcopy(co.state_dict()),state=dict(iteration=2,actor_updates=40,mc_preheat_complete=False))
    p=tmp_path/'source.pt';torch.save(source,p)
    target=copy.deepcopy(identity)
    target['critic']['validation_guard']=c['stage10']['critic']['validation_guard']
    target['training']['robofft']['update_guard']=c['stage9']['robofft']['update_guard']
    record=initialize_optimization_variant(p,identity=target,actor=actor,critic=critic,actor_optimizer=ao,
        critic_optimizer=co,future_critic_formal=True)
    assert not record['inherited_mc_preheat_for_unchanged_value_contract']
    assert record['critic_and_adam_preserved'] and not record['future_branch_zero_initialized']
    assert_tree_equal(source['critic_optimizer'],co.state_dict())
    target['training']['robofft']['mc_samples']=8
    with pytest.raises(ValueError,match='unchanged MC'):initialize_optimization_variant(p,identity=target,
        actor=actor,critic=critic,actor_optimizer=ao,critic_optimizer=co,future_critic_formal=True)


def test_physical_regression_and_supervisor_resume_are_explicit(tmp_path):
    from tools.run_stage2_training_supervisor import training_command
    baseline=dict(source_balanced_reward=40.)
    current=dict(source_balanced_reward=39.,paired_baseline=dict(new_failures=0,
        regression_alert=dict(triggered=True,upper90=-.2)))
    guard=dict(max_new_failures=0,max_relative_reward_drop=.005)
    assert not physical_guard(current,baseline,guard)['passed']
    current['paired_baseline']['regression_alert']['triggered']=False
    assert physical_guard(current,baseline,guard)['passed']
    current['paired_baseline']['new_failures']=1
    assert not physical_guard(current,baseline,guard)['passed']
    first=training_command('x.yaml',tmp_path,200,robofft_variant='a.pt',future_critic_formal=True)
    assert '--future-critic-formal' in first and '--initialize-robofft-variant' in first
    (tmp_path/'latest.json').write_text('{}')
    resumed=training_command('x.yaml',tmp_path,200,robofft_variant='a.pt',future_critic_formal=True)
    assert '--future-critic-formal' in resumed and '--resume' in resumed
    assert '--initialize-robofft-variant' not in resumed


def test_higher_reward_with_failed_physical_guard_cannot_become_best():
    from gem.closedloop.dppo.periodic_monitor import choose_best
    base=dict(plan_sha256='a',source_balanced_reward=40.,physical_failure_count=0,mean_executed_seconds=20.,
              actor_identity={},episode_metrics=[dict(task_id='one',dataset='a',seed=42,reward=40.,
                  physical_failure=False,executed_seconds=20.)])
    current=copy.deepcopy(base);current['source_balanced_reward']=41.
    current['episode_metrics'][0]['reward']=41.;current['physical_guard']=dict(passed=False)
    best=choose_best(current,base,saved=True,iteration=5)
    assert not best['eligible'] and best['best_saved'] is None and best['best_observed'] is None


def test_physical_stop_restores_two_models_and_adams_without_refunding_budget(tmp_path,monkeypatch):
    import json
    from gem.closedloop.dppo import training
    from gem.closedloop.dppo.learning_guards import rollback_quality_stop
    from gem.closedloop.dppo.training_checkpoint import file_sha256
    from tools import run_stage2_training_supervisor as supervisor
    actor,critic=torch.nn.Linear(2,2),torch.nn.Linear(2,1)
    ao,co=torch.optim.Adam(actor.parameters()),torch.optim.Adam(critic.parameters())
    for model,opt in ((actor,ao),(critic,co)):
        model(torch.ones(1,2)).sum().backward();opt.step();opt.zero_grad(set_to_none=True)
    saved=dict(actor=copy.deepcopy(actor.state_dict()),critic=copy.deepcopy(critic.state_dict()),
               actor_optimizer=copy.deepcopy(ao.state_dict()),critic_optimizer=copy.deepcopy(co.state_dict()))
    checkpoint=tmp_path/'safe.pt';torch.save(saved,checkpoint)
    (tmp_path/'quality_last_safe.json').write_text(json.dumps(dict(path='safe.pt',sha256=file_sha256(checkpoint),iteration=5)))
    for model,opt in ((actor,ao),(critic,co)):
        model(torch.ones(1,2)).sum().backward();opt.step();opt.zero_grad(set_to_none=True)
    state=dict(iteration=10,policy_version=10,optimizer_attempts=80,training_control_steps=10000,
                quality_stop=dict(reasons=['new_physical_failure']))
    c=NS(output=tmp_path,actor=actor,critic=critic,actor_optimizer=ao,critic_optimizer=co,
          state=state,distributed=local_distributed())
    publications=[]
    monkeypatch.setattr(training,'_poll_checkpoint',lambda *a,**kw:None)
    monkeypatch.setattr(training,'_snapshot',lambda c,reason,**kw:publications.append(reason))
    rollback_quality_stop(c,{})
    for key,value in saved.items():assert_tree_equal(value,getattr(c,key).state_dict())
    assert state['iteration']==10 and state['optimizer_attempts']==80 and state['training_control_steps']==10000
    assert state['policy_version']==11 and publications==['quality_rollback_controlled_stop']
    assert (tmp_path/'quality_stop.json').is_file()
    monkeypatch.setattr('sys.argv',['supervisor','--config','unused.yaml','--output-dir',str(tmp_path),
                                  '--status-dir',str(tmp_path/'status'),'--future-critic-formal'])
    assert supervisor.main()==0
    assert json.loads((tmp_path/'status/supervisor.json').read_text())['status']=='quality_stopped'
