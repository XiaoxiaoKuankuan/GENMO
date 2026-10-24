"""执行相关 RoboFFT 变体的数学、来源账本和固定 MC 集成验收。

独立手算时间加权 epsilon 损失及梯度，验证全一权重退化、整体缩放不变、自由
坐标质量守恒和前缀/padding 零梯度。真实小型 Stage1Actor 验证同一 MC 的旧新
比值、流式反传、时间权重不可变及序列化。合成合法600Hz/50Hz窗口检验前缀
跨计划来源和截断范围的预测标签。warm-start 只用小型网络检查学习状态与门禁。
测试不连接物理服务器，不把代理梯度一致性当作跟踪质量改善。
"""
from __future__ import annotations

import copy
from types import SimpleNamespace as NS
import pytest
import torch

from tests.closedloop.dppo.test_robofft import (actor_factory, cpu_threads, make_policy, make_rows,
    gradients, settings, ROOT)
from gem.closedloop.robofft.execution_weights import (VERSION, normalize_frame_weights,
    predicted_weights, weights_for_rows)
from gem.closedloop.robofft.surrogate import epsilon_loss, action_objective
from gem.closedloop.robofft.updater import prepare_mc, objective_batch
from gem.closedloop.dppo.execution_provenance import SCHEMA

WEIGHTS=dict(version=VERSION,near_weight=1.,middle_weight=.3,far_weight=.1,
             tail_seconds=.5,lookahead_ticks=120,frame_ticks=20)


def test_mass_all_one_scaling_padding_and_exact_gradient():
    torch.manual_seed(48)
    x=torch.randn(2,60,30,dtype=torch.float64)
    eps=torch.randn_like(x);pred=torch.randn_like(x,requires_grad=True)
    free=torch.ones_like(x,dtype=torch.bool);free[:,:10]=False;free[1,54:]=False;free[0,13,:6]=False
    raw=torch.tensor([[1.]*25+[.3]*15+[.1]*20]*2)
    w=normalize_frame_weights(raw,free)
    torch.testing.assert_close(w,normalize_frame_weights(raw*7,free),atol=1e-7,rtol=1e-7)
    torch.testing.assert_close((w*free.sum(-1)).sum(-1),free.sum((1,2)).double())
    assert not w.requires_grad and not w[~free.any(-1)].any()
    a=torch.tensor([.5,.8],dtype=torch.float64)[:,None,None];snr=torch.tensor([.3,.7])
    original=epsilon_loss(x,pred,eps,a,snr,free)
    ones=epsilon_loss(x,pred,eps,a,snr,free,normalize_frame_weights(torch.ones(2,60),free))
    torch.testing.assert_close(original,ones,atol=0,rtol=0)
    got=epsilon_loss(x,pred,eps,a,snr,free,w)
    residual=((x-a.sqrt()*pred)/(1-a).sqrt()-eps).masked_fill(~free,0)
    independent=.5*snr*(residual.square()*w[:,:,None]).sum((1,2))/30
    torch.testing.assert_close(got,independent)
    grad=torch.autograd.grad(got.sum(),pred,retain_graph=True)[0]
    torch.testing.assert_close(grad,torch.autograd.grad(independent.sum(),pred)[0])
    assert not grad[~free].any()
    # 相同误差在近期/远期的梯度之比严格为10，归一不改变相对权重。
    assert (w[0,20]/w[0,50]).item()==pytest.approx(10.)
    with pytest.raises(ValueError):normalize_frame_weights(torch.zeros(2,60),free)
    with pytest.raises(ValueError):normalize_frame_weights(-raw,free)
    assert not normalize_frame_weights(raw,torch.zeros_like(free)).any()


def test_prediction_uses_prefix_and_actual_arrival_not_fixed_fifteen():
    free=torch.ones(60,30,dtype=torch.bool);free[:10]=False
    early=predicted_weights(free,decision_tick=600,arrival_tick=744,settings=WEIGHTS)
    late=predicted_weights(free,decision_tick=600,arrival_tick=924,settings=WEIGHTS)
    assert early[24]==1 and late[32]==1 and early[32]==.3
    row=NS(identity={},metadata={},free_mask=free,control_tick_begin=600,control_tick_end=780)
    weights,audits=weights_for_rows([row],WEIGHTS)
    assert audits[0]['scope'].startswith('predicted') and not audits[0]['source_lineage_complete']
    assert not weights[0][:10].any()


def test_observed_prefix_lineage_and_lookahead_only_are_not_current():
    plans=dict(p=dict(decision_tick=0,commit_tick=144,prefix_frames=10,parent_plan_id='bootstrap:x',generation_frames=60),
               q=dict(decision_tick=180,commit_tick=324,prefix_frames=10,parent_plan_id='p',generation_frames=60))
    steps=[]
    for begin in range(324,805,12):
        ticks=list(range(begin-120,begin+121,12))
        steps.append(dict(control_tick_begin=begin,control_tick_end=begin+12,
            consumed_reference_ticks=ticks,consumed_plan_ids=['q']*21,
            reference_tick=begin+12,reference_plan_id='q'))
    free=torch.ones(60,30,dtype=torch.bool);free[:10]=False
    row=NS(identity=dict(backend_session_id='x',env_id=0,episode_id=1,plan_id='p'),
        metadata=dict(compact_execution_provenance=dict(schema=SCHEMA,plans=plans,steps=steps)),
        free_mask=free,control_tick_begin=0,control_tick_end=180)
    weights,audits=weights_for_rows([row],WEIGHTS);a=audits[0]
    assert a['scope'].startswith('observed') and a['source_lineage_complete']
    assert 10 in a['window_generated_frames'] and 10 not in a['current_generated_frames']
    assert 17 in a['current_generated_frames'] and 55 in a['unobserved_free_frames']
    assert 55 in a['unobserved_free_frames_overwritten_by_committed_child']
    assert weights[0][10]/weights[0][55]==pytest.approx(10.)
    assert a['normalized_weight_mass']==pytest.approx(1500.)
    # 未提交 -> 已提交的账本推进可以合并；固定来源冲突必须拒绝。
    earlier=copy.deepcopy(row);earlier.metadata['compact_execution_provenance']['steps']=[]
    earlier.metadata['compact_execution_provenance']['plans']['p']['commit_tick']=None
    weights_for_rows([earlier,row],WEIGHTS)
    earlier.metadata['compact_execution_provenance']['plans']['p']['decision_tick']=20
    with pytest.raises(ValueError,match='Conflicting'):weights_for_rows([earlier,row],WEIGHTS)


@pytest.mark.parametrize('chunk',[1,4,16])
def test_weighted_fixed_mc_streamed_gradient_and_mutation(actor_factory,tmp_path,chunk):
    policy,context=make_policy(actor_factory)
    policy.robofft.update(execution_weighting=WEIGHTS,mc_microbatch=chunk,beta_neg=.5)
    rows,targets,cache,manifest=make_rows(policy,context)
    prepare_mc(policy,rows,cache,torch.Generator().manual_seed(123),microbatch=2)
    records=[r.mc_record for r in rows]
    old=torch.stack([r.old_loss for r in records])
    mc={k:torch.stack([getattr(r,k) for r in records]) for k in ('eps','sampling_indices','original_timesteps','time_weights')}
    full=policy.losses(context,cache.get('action_x0',rows),**mc)
    objective=action_objective(full,old,targets['advantages'],beta_neg=.5)
    torch.testing.assert_close(objective['ratio'],torch.ones(2,dtype=torch.float64),atol=3e-6,rtol=0)
    objective['loss'].mean().backward();expected=gradients(policy.actor)
    policy.actor.zero_grad(set_to_none=True)
    for row in reversed(rows):objective_batch(policy,[row],cache,clip=.01,global_count=2,backward=True)
    for name,value in gradients(policy.actor).items():
        torch.testing.assert_close(value,expected[name],atol=3e-5,rtol=3e-4,msg=name)
    assert sum(float(x.abs().sum()) for x in expected.values())>0
    path=tmp_path/'weighted_mc.pt';torch.save(records,path)
    restored=torch.load(path,weights_only=False)
    torch.testing.assert_close(restored[0].time_weights,records[0].time_weights)
    records[0].time_weights.add_(.1)
    with pytest.raises(ValueError,match='modified'):records[0].validate()
    cache.close()


def test_variant_warm_start_preserves_adam_but_not_progress_and_rejects_value_change(tmp_path):
    from gem.closedloop.robofft import CHECKPOINT_SCHEMA
    from gem.closedloop.robofft.checkpoint import initialize_optimization_variant
    actor=torch.nn.Linear(2,2);critic=torch.nn.Linear(2,1)
    ao=torch.optim.Adam(actor.parameters(),lr=1e-6);co=torch.optim.Adam(critic.parameters(),lr=1e-4)
    actor(torch.ones(1,2)).sum().backward();ao.step()
    critic(torch.ones(1,2)).sum().backward();co.step()
    identity=dict(training=dict(gamma_upper=.95,robofft=dict(mc_samples=16,beta_neg=.125)),
        runtime=dict(num_envs=128,generation_batch=128))
    payload=dict(schema=CHECKPOINT_SCHEMA,identity=identity,actor=actor.state_dict(),critic=critic.state_dict(),
        actor_optimizer=ao.state_dict(),critic_optimizer=co.state_dict(),
        state=dict(iteration=27,actor_updates=107,mc_preheat_complete=True))
    path=tmp_path/'source.pt';torch.save(payload,path)
    new=copy.deepcopy(identity);new['training'].update(ppo_epochs=10)
    new['training']['robofft'].update(beta_neg=.5,execution_weighting=WEIGHTS,mc_microbatch=4)
    new['runtime'].update(num_envs=16,generation_batch=16)
    ao.param_groups[0]['lr']=3e-6
    args=dict(identity=new,actor=actor,critic=critic,actor_optimizer=ao,critic_optimizer=co)
    record=initialize_optimization_variant(path,**args)
    assert record['source_iteration']==27 and record['counters_restarted'] and not record['rng_restored']
    assert ao.param_groups[0]['lr']==3e-6 and ao.state_dict()['state'][0]['step']==1
    new['training']['gamma_upper']=.99
    before=copy.deepcopy(actor.state_dict())
    with pytest.raises(ValueError,match='protected contract'):initialize_optimization_variant(path,**args)
    for k,v in before.items():assert torch.equal(v,actor.state_dict()[k])


def test_ten_and_twenty_epochs_are_explicit_robofft_configurations(tmp_path):
    import yaml
    from gem.closedloop.dppo.training_config import configuration
    config=yaml.safe_load((ROOT/'configs/closedloop/stage2_robofft_execution_weighted_server1.yaml').read_text())
    for epochs in (10,20):
        config['stage10']['training']['ppo_epochs']=epochs
        path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(config))
        resolved=configuration(path)
        assert resolved['stage9']['ppo_epochs']==epochs
        assert resolved['stage10']['derived']['maximum_actor_steps']>=epochs


def test_variant_supervisor_only_initializes_first_launch(tmp_path):
    from tools.run_stage2_training_supervisor import training_command
    first=training_command('new.yaml',tmp_path,3,robofft_variant='parent.pt')
    assert first[-2:]==['--initialize-robofft-variant','parent.pt']
    (tmp_path/'latest.json').write_text('{}')
    resumed=training_command('new.yaml',tmp_path,3,robofft_variant='parent.pt')
    assert resumed[-2:]==['--resume','latest'] and '--initialize-robofft-variant' not in resumed
    with pytest.raises(ValueError):training_command('new.yaml',tmp_path,3,'old.pt',robofft_variant='parent.pt')


def test_weighted_objective_requires_explicit_distinct_schema(tmp_path):
    import yaml
    from gem.closedloop.dppo.training_config import configuration
    from gem.closedloop.robofft.configuration import optimization_binding, EXECUTION_SCHEMA
    c=yaml.safe_load((ROOT/'configs/closedloop/stage2_robofft_execution_weighted_server1.yaml').read_text())
    p=tmp_path/'variant.yaml';p.write_text(yaml.safe_dump(c))
    assert optimization_binding(configuration(p)['stage10'])['schema']==EXECUTION_SCHEMA
    c['stage10']['actor_optimization']['schema']='genmo.closedloop.stage2.robofft_d.v1'
    p.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError,match='Explicit RoboFFT'):configuration(p)
    c['stage10']['actor_optimization']['schema']=EXECUTION_SCHEMA
    c['stage10']['training']['robofft'].pop('execution_weighting')
    p.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError,match='Explicit RoboFFT'):configuration(p)


def test_plan_auditor_recomputes_cached_weights_and_rejects_changed_provenance(actor_factory):
    from tools.audit_robofft_execution_weights import audit_rows
    policy,context=make_policy(actor_factory)
    policy.robofft.update(execution_weighting=WEIGHTS,mc_microbatch=4)
    rows,targets,cache,_=make_rows(policy,context)
    prepare_mc(policy,rows,cache,torch.Generator().manual_seed(58))
    entries=audit_rows(rows,WEIGHTS,rank=0)
    assert len(entries)==2 and len(entries[0]['normalized_frame_weights'])==60
    assert entries[0]['audit']['normalized_weight_mass']==entries[0]['audit']['free_coordinate_count']
    rows[0].metadata['execution_weighting']['near_free_frames']+=1
    with pytest.raises(ValueError,match='Cached per-plan'):audit_rows(rows,WEIGHTS,rank=0)
    cache.close()
