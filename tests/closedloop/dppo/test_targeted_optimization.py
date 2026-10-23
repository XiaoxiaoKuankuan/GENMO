"""第二阶段限时优化的奖励、BC关闭与完整目标CPU回归。

使用真实八项GMT标量/向量奖励夹具，验证来源std不改、新评分不重复平方、旧v2仍
可复现且新旧配置不可混用；使用真实小扩散网络比较完整rollout目标，确保BC关闭
不调用监督反向、不修改控制器与监督RNG。独立小Adam验证目标下降拒绝与恢复。
只在pytest临时目录产生数据，不执行正式训练或宣称真实物理收益。
"""
import copy
import math

import pytest
import torch

from gem.closedloop.dppo.rewards import (ExecutionReward,TRACKING_STD,STAGE2_REWARD_STD,REWARD_STD_VERSION,
    ROOT_RELAXED_REWARD_VERSION,ROOT_RELAXED_REWARD_STD)
from gem.closedloop.dppo.vector_reward_math import VectorRewardMath
from gem.closedloop.dppo.vector_reward_adapter import VectorExecutionReward,compare_reward_evidence
from gem.closedloop.dppo.gradient_merge import BCGradientController
from gem.closedloop.dppo.actor_backtracking import backtracked_step,check_candidate_kl
from gem.closedloop.dppo.collectives import DistributedCollectives
from gem.closedloop.dppo.tensor_cache import RolloutTensorCache
from gem.closedloop.dppo.updater import actor_update,final_kl
from tests.closedloop.dppo.test_clipped_dppo import case,Anchor
from tests.closedloop.dppo.test_stage2_reward import device_packet
from tests.closedloop.dppo.test_actor_backtracking import Collective,equal_state
from tests.closedloop.test_stage1_actor import actor_factory
from test_data_learning import actual_step,target_activity_fixture


@pytest.mark.parametrize('version,expected_std',[(REWARD_STD_VERSION,STAGE2_REWARD_STD),
    (ROOT_RELAXED_REWARD_VERSION,ROOT_RELAXED_REWARD_STD)])
def test_new_reward_std_preserves_source_and_scalar_vector_formula(version,expected_std):
    scalar=ExecutionReward({'version':version},target_activity=target_activity_fixture)
    vector=VectorExecutionReward({'version':version},target_activity=target_activity_fixture)
    engine=VectorRewardMath(scalar.config,1,'cpu')
    import numpy as np
    scalar.seed_previous_target(np.zeros(21));vector.seed_previous_target(np.zeros(21));engine.previous_valid.fill_(True)
    row=actual_step(0,speed=1.);row['env_id']=0
    for name,item in row['motion_tracking']['terms'].items():
        item['error']=.04
        item['source_score']=math.exp(-.04/TRACKING_STD[name]**2)
    for substep in row['physics_substeps']:
        d=substep['physical_diagnostics']
        for field in ('foot_support_sphere_tangent_speed_m_s','foot_support_sphere_clearance_m'):
            import numpy as np
            d[field]={n:np.zeros(1) for n in d['foot_body_names']}
    row['physical_diagnostics']=copy.deepcopy(row['physics_substeps'][-1]['physical_diagnostics'])
    row['reward_primitives']=device_packet(row,engine)
    expected=scalar.evaluate_step(row);observed=vector.evaluate_step(row)
    assert expected['transition_valid'],expected['errors']
    compare_reward_evidence(observed,expected)
    scores=expected['components']['track']['normalized']['scores']
    assert scores['anchor_pos']==pytest.approx(math.exp(-.04/expected_std['anchor_pos']**2))
    assert scores['anchor_ori']==pytest.approx(math.exp(-.04/expected_std['anchor_ori']**2))
    assert scores['body_pos']==pytest.approx(math.exp(-.04/.15**2))
    assert scalar.config['tracking']['source_std']==TRACKING_STD
    assert scalar.config['tracking']['reward_std']==expected_std


def test_root_relaxation_changes_only_two_scales_and_rejects_cross_version_mix():
    old=ExecutionReward({'version':REWARD_STD_VERSION}).config
    new=ExecutionReward({'version':ROOT_RELAXED_REWARD_VERSION}).config
    assert {k for k in STAGE2_REWARD_STD if STAGE2_REWARD_STD[k]!=ROOT_RELAXED_REWARD_STD[k]}=={'anchor_pos','anchor_ori'}
    restored=copy.deepcopy(new);restored['version']=old['version']
    restored['tracking']['reward_std']=old['tracking']['reward_std']
    assert restored==old
    for version,std in ((REWARD_STD_VERSION,ROOT_RELAXED_REWARD_STD),(ROOT_RELAXED_REWARD_VERSION,STAGE2_REWARD_STD)):
        with pytest.raises(ValueError,match='explicit reward version'):
            ExecutionReward({'version':version,'tracking':{'reward_std':std}})


def test_reward_versions_reject_ambiguous_or_changed_source_std():
    assert ExecutionReward().config['tracking']['std']==TRACKING_STD
    with pytest.raises(ValueError):ExecutionReward({'tracking':{'std':STAGE2_REWARD_STD}})
    with pytest.raises(ValueError):ExecutionReward({'version':REWARD_STD_VERSION,'tracking':{'std':TRACKING_STD}})
    bad=copy.deepcopy(TRACKING_STD);bad['anchor_pos']=.2
    with pytest.raises(ValueError):ExecutionReward({'version':REWARD_STD_VERSION,'tracking':{'source_std':bad}})


def test_bc_disabled_has_zero_calls_and_same_full_objective_population(actor_factory,tmp_path):
    import torch.distributed as dist
    dist.init_process_group('gloo',init_method=(tmp_path/'gloo').as_uri(),rank=0,world_size=1)
    policy,rows,targets=case(actor_factory)
    opt=torch.optim.AdamW([p for p in policy.actor.parameters() if p.requires_grad],lr=1e-8,weight_decay=0.)
    anchor=Anchor();anchor.backward=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('BC called'))
    controller=BCGradientController();before=copy.deepcopy(controller.state_dict())
    distributed=DistributedCollectives(0,1,device='cpu')
    manifest=[dict(owner_rank=0,local_index=i,valid=True,has_free=True) for i in range(len(rows))]
    cache=RolloutTensorCache(rows,targets,'cpu')
    observed=[]
    try:
        result=actor_update(policy,opt,rows,targets,cache,manifest,controller,distributed=distributed,
            anchor=anchor,generator=torch.Generator().manual_seed(73),minibatch_chains=2,microbatch=3,
            soft_kl=None,max_active_clip=1.,bc_enabled=False,audit_full_rollout_objective=True,
            full_rollout_observer=lambda stage,records,adv,sources:observed.append((stage,records.clone())))
        assert result['actual_global_bc_samples']==0 and result['planned_global_bc_samples']==0
        assert equal_state(controller.state_dict(),before)
        reference=final_kl(policy,rows,cache,manifest,distributed=distributed,microbatch=3,clip=.01)
        assert result['full_rollout_objective_history'][-1]['objective_mean']==pytest.approx(reference['objective_mean'],abs=1e-10)
        assert all(s['full_rollout_after']['complete_chain_count']==len(rows) for s in result['steps'])
        assert len(observed)==1+result['optimizer_steps']
        assert all(r.shape[:2]==(len(rows),4) for _,r in observed)
        assert all(s['full_rollout_groups']['all']['gain_from_previous_same_membership']==pytest.approx(
            s['full_rollout_after']['objective_mean']-s['full_rollout_before']['objective_mean'],abs=1e-12)
            for s in result['steps'])
        assert result['steps'][0]['parameter_update']['gradient_projection_measured']
        assert result['steps'][0]['parameter_update']['modules']['all']['cosine_update_negative_ppo_gradient'] is not None
    finally:
        cache.close();dist.destroy_process_group()


def test_objective_aware_rejects_bad_gain_and_restores_nonempty_adam():
    actor=torch.nn.Linear(2,1);opt=torch.optim.AdamW(actor.parameters(),lr=1e-6,weight_decay=0.)
    for p in actor.parameters():p.grad=torch.ones_like(p)
    opt.step();control=BCGradientController()
    before=copy.deepcopy((actor.state_dict(),opt.state_dict(),control.state_dict()))
    def evaluate():return dict(chain_joint_kl={'mean':0.,'p95':0.,'max':0.},objective_mean=-.1)
    result=backtracked_step(actor,opt,control,evaluate=evaluate,limits={'objective_reference':0.},
        distributed=Collective(),max_attempts=2)
    assert not result['accepted'] and result['finite_trust_region_exhausted']
    assert equal_state((actor.state_dict(),opt.state_dict(),control.state_dict()),before)
    assert check_candidate_kl(evaluate(),{})['passed']


def test_objective_search_keeps_best_trial_and_single_adam_history():
    actor=torch.nn.Linear(2,1);opt=torch.optim.AdamW(actor.parameters(),lr=1e-6,weight_decay=0.)
    for p in actor.parameters():p.grad=torch.ones_like(p)
    opt.step();controller=BCGradientController()
    expected=copy.deepcopy(actor);expected_opt=torch.optim.AdamW(expected.parameters(),lr=1e-6,weight_decay=0.)
    expected_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
    for p in expected.parameters():p.grad=torch.ones_like(p)
    expected_opt.param_groups[0]['lr']=5e-7;expected_opt.step()
    result=backtracked_step(actor,opt,controller,
        evaluate=lambda:dict(objective_mean={1e-6:.01,5e-7:.03,2.5e-7:.02}[opt.param_groups[0]['lr']]),
        limits={'objective_reference':0.},distributed=Collective(),max_attempts=3,select_best=True,search_attempts=3)
    assert result['accepted_lr']==5e-7 and result['selected_attempt']==2
    # 验证恢复的是最佳候选自己的参数与Adam，而不是最后一次试探。
    assert all(float(s['step'])==2 for s in opt.state.values())
    assert equal_state(actor.state_dict(),expected.state_dict())
    assert equal_state(opt.state_dict(),expected_opt.state_dict())


def test_deterministic_display_preserves_frozen_hybrid_policy(actor_factory):
    policy,rows,_=case(actor_factory)
    context=rows[0].context
    a=policy.sample_rollout(context,generator=torch.Generator().manual_seed(49),deterministic_transitions=True)
    b=policy.sample_rollout(context,generator=torch.Generator().manual_seed(49),deterministic_transitions=True)
    torch.testing.assert_close(a['chain'],b['chain'],rtol=0,atol=0)
    assert not a['sampling_noise'].count_nonzero()
    with torch.no_grad():
        expected=policy.transition_parameters(context,a['chain'][:,19],19)['mean']
    torch.testing.assert_close(a['chain'][:,20],expected,rtol=0,atol=0)


def test_temporal_gate_preserves_initial_chain_and_shared_probability_mean(actor_factory):
    from gem.closedloop.dppo.temporal_residual_policy import TemporalResidualPolicy,residual_gate
    policy,rows,_=case(actor_factory)
    gated=TemporalResidualPolicy(copy.deepcopy(policy.actor),std_schedule=policy.std_schedule,
        generation_frames=60,temporal_residual=dict(focus_free_frames=8,taper_frames=5,evidence='synthetic_test_only'))
    context=rows[0].context
    old=policy.sample_rollout(context,generator=torch.Generator().manual_seed(53))
    new=gated.sample_rollout(context,generator=torch.Generator().manual_seed(53))
    for field in ('chain','qpos30','contact','old_means','old_stds','old_log_probs'):
        if field in old:torch.testing.assert_close(new[field],old[field],atol=0,rtol=0)
    with torch.no_grad():
        for name,parameter in gated.actor.named_parameters():
            if name.startswith('denoiser.final_layer.'):parameter.add_(1e-4)
        prepared=gated.prepare_conditions(context)
        result=gated._transition_prepared(prepared,new['chain'][:,15],15)
        prior=gated._frozen_transition(prepared,new['chain'][:,15],15,False)
        gate=residual_gate(prepared['adapted']['known_mask'],prepared['adapted']['future_valid'],
            focus_free_frames=8,taper_frames=5).expand_as(result['mean'])
        torch.testing.assert_close(result['mean'][gate==0],prior['mean'][gate==0],atol=0,rtol=0)
        # 全部自由坐标依旧参与概率，门控没有缩小free_mask。
        assert torch.equal(result['free_mask'],prior['free_mask'])
        assert bool((result['mean'][gate>0]!=prior['mean'][gate>0]).any())
    # 梯度条件缓存原先不计算冻结特征；此候选仍必须正确提供无梯度冻结特征。
    from gem.closedloop.dppo.tensor_cache import ConditionGraphCache
    cache=ConditionGraphCache(gated,include_frozen=False)
    cache.prime(rows[:1]);out=cache.transition(rows[:1],rows[0].chain[15:16],15)
    out['mean'].sum().backward();cache.backward()
    assert any(p.grad is not None for p in gated.actor.parameters() if p.requires_grad)
    assert all(p.grad is None for p in gated.frozen_policy.actor.parameters())


def test_comparison_restores_nonempty_adam_controller_bc_rng_exactly():
    from tools.compare_stage2_optimizer_batches import restore_experiment
    from gem.closedloop.dppo.parallel_support import capture_local_rng
    from gem.closedloop.dppo.rollback_audit import exact_state_digest
    actor=torch.nn.Linear(2,1);opt=torch.optim.AdamW(actor.parameters(),lr=1e-6,weight_decay=0.)
    for p in actor.parameters():p.grad=torch.ones_like(p)
    opt.step();controller=BCGradientController();anchor=Anchor();generator=torch.Generator().manual_seed(65)
    initial=dict(actor=copy.deepcopy(actor.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),
        controller=copy.deepcopy(controller.state_dict()),bc=copy.deepcopy(anchor.state_dict()),
        rng=capture_local_rng({'actor':generator}))
    digest=exact_state_digest(initial)
    opt.step();anchor.batch_size=12;torch.randn(8,generator=generator)
    restore_experiment(actor,opt,controller,anchor,initial,generator)
    assert digest==exact_state_digest(dict(actor=actor.state_dict(),optimizer=opt.state_dict(),
        controller=controller.state_dict(),bc=anchor.state_dict(),rng=capture_local_rng({'actor':generator})))


def test_execution_audit_traces_original_prefix_across_rollouts(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from gem.closedloop.dppo.execution_provenance import SCHEMA
    from tools.analyze_stage2_execution_credit import episode_analysis
    plans={'p0':dict(decision_tick=0,commit_tick=12,prefix_frames=0,parent_plan_id='bootstrap:x',generation_frames=60),
        'p1':dict(decision_tick=300,commit_tick=468,prefix_frames=17,parent_plan_id='p0',generation_frames=60)}
    entries=[]
    for case,plan,begin in [('first','p0',600),('second','p1',624)]:
        ticks=list(range(begin-120,begin+121,12))
        record=dict(schema=SCHEMA,plans=plans,steps=[dict(control_tick_begin=begin,control_tick_end=begin+12,
            consumed_reference_ticks=ticks,consumed_plan_ids=['p0' if t<=612 else 'p1' for t in ticks],
            reference_tick=begin+12,reference_plan_id='p0' if begin+12<=612 else 'p1')])
        row=SimpleNamespace(identity=dict(plan_id=plan,policy_version=0),
            metadata=dict(compact_execution_provenance=record,training_task={'dataset':'fixture'}),
            control_tick_begin=begin,control_tick_end=begin+12,rewards=torch.tensor([.1]),
            old_value=1.,next_value=1.,executed_control_steps=1,terminated=False,truncated=True,transition_valid=True)
        entries.append((case,row,torch.tensor(1.)))
    p,v,r=episode_analysis(('backend','env','episode'),entries)
    first=next(item for item in p if item['plan_id']=='p0')
    assert 31 in first['cross_rollout_source_frames']
    assert all(item['lambda1_type']=='right_censored_bootstrapped_not_full_MC' for item in v)
    assert not r['continuous_control_range']
    import tools.analyze_stage2_reference_consumption as tool
    roots=[tmp_path/'first',tmp_path/'second']
    for root in roots:(root/'rank00').mkdir(parents=True)
    for _,row,_ in entries:row.free_mask=torch.ones(60,30,dtype=torch.bool)
    monkeypatch.setattr(tool,'load_rollout',lambda path:{'transitions':[entries[0 if path.parent.name=='first' else 1][1]]})
    meta=dict(identity=entries[0][1].identity,source='fixture',music_frame=0,prefix_frames=0,
        kl_by_frame=[1.]*60,log_ratio_by_frame=[0.]*60)
    a=tool.associate(roots[0],{'metadata':[meta]})
    b=tool.associate(roots[0],{'metadata':[meta]},execution_cases=[roots[1]])
    assert 31 not in a['chains'][0]['current_generated_frames']
    assert 31 in b['chains'][0]['current_generated_frames']


def test_full_probability_reuse_requires_same_completed_case(tmp_path):
    import json
    from tools.compare_stage2_optimizer_batches import producer_full_probability
    case=tmp_path/'session/fixed_data_validation/iteration000001';case.mkdir(parents=True)
    assert producer_full_probability(case) is None
    path=case.parents[1]/'summary.json'
    document=dict(status='passed',exit_code=0,fixed_data_cases=[dict(fixed_data_case=str(case),
        full_zero_update_probability=dict(passed=True,checked_steps=list(range(19))))])
    path.write_text(json.dumps(document));assert producer_full_probability(case)['audit']['passed']
    document['fixed_data_cases'][0]['full_zero_update_probability']['checked_steps']=[15,16,17,18]
    path.write_text(json.dumps(document));assert producer_full_probability(case) is None
    document['fixed_data_cases'][0]['fixed_data_case']=str(case.parent/'iteration000002')
    path.write_text(json.dumps(document));assert producer_full_probability(case) is None


@pytest.mark.skipif(not torch.cuda.is_available(),reason='需服务器1空闲GPU验证CPU统计/GPU反向边界')
def test_cpu_full_records_match_gpu_streamed_gradients(actor_factory):
    from gem.closedloop.dppo.frozen_prefix import FrozenPrefixPolicy
    from gem.closedloop.dppo.chain_ppo import evaluate_chain_records,backward_chain_surrogate
    from gem.closedloop.dppo.execution_checks import policy_phase
    original,rows,targets=case(actor_factory)
    actor=copy.deepcopy(original.actor).to('cuda:0')
    policy=FrozenPrefixPolicy(actor,std_schedule=original.std_schedule,generation_frames=60)
    cache=RolloutTensorCache(rows,targets,'cuda:0')
    try:
        with torch.no_grad(),policy_phase(policy):records=evaluate_chain_records(policy,rows,cache,microbatch=3)
        with policy_phase(policy):
            backward_chain_surrogate(policy,rows,cache,records,clip=.1,global_chain_count=len(rows),microbatch=3)
        expected={name:p.grad.clone() for name,p in actor.named_parameters() if p.grad is not None}
        actor.zero_grad(set_to_none=True)
        with policy_phase(policy):
            backward_chain_surrogate(policy,rows,cache,records.cpu(),clip=.1,global_chain_count=len(rows),microbatch=3)
        for name,p in actor.named_parameters():
            if name in expected:torch.testing.assert_close(p.grad,expected[name],atol=0,rtol=0)
    finally:cache.close()
