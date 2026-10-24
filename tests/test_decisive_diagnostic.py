"""有限诊断的独立数学测试：裁剪导数、模块范数和已知前缀窗。

这些测试以Autograd直接验证符号相关的零梯度区域，检查模块匹配不把随机方向
变成真实梯度，并检查已知前缀与末帧过渡。它们不替代真实GPU行为链或物理评估。
"""
import torch
from gem.closedloop.dppo.decisive_diagnostic import clip_masks,matched_direction,module_stats,smooth_free_window
from gem.closedloop.robofft.surrogate import action_objective


def test_modeled_deployment_compares_real_ticks_not_training_wall_clock():
    import copy
    from gem.closedloop.dppo.decisive_diagnostic import deployed_clock_evidence
    a=dict(prefix_frames=12,measured_arrival_tick=744,actual_decision_interval_ticks=180,
        generation_schedule='continuous_rolling.v1',reference_commit={'effective_tick':744},events=[],
        compact_execution_provenance={'plans':{'plan':{'generation_wall_seconds':1.4,'commit_tick':744}},'steps':[]},
        latency_seconds=1.4)
    b=copy.deepcopy(a);b['latency_seconds']=.9;b['compact_execution_provenance']['plans']['plan']['generation_wall_seconds']=.9
    assert deployed_clock_evidence(a)==deployed_clock_evidence(b)
    b['measured_arrival_tick']+=12
    assert deployed_clock_evidence(a)!=deployed_clock_evidence(b)


def test_actual_clip_matches_autograd_for_both_signs():
    ratios=torch.tensor([.97,.97,1.03,1.03,1.,1.],dtype=torch.float64)
    adv=torch.tensor([1.,-1.,1.,-1.,1.,-1.],dtype=torch.float64)
    new=(-ratios.log()/.1)[:,None].repeat(1,16).requires_grad_()
    result=action_objective(new,torch.zeros_like(new),adv,beta_neg=.5)
    result['loss'].sum().backward()
    masks=clip_masks(ratios,adv)
    assert torch.equal(new.grad.abs().sum(1)==0,masks['active'])
    assert masks['outside'].sum()==4 and masks['active'].sum()==2
    assert masks['outside_active_gradient'].sum()==2


def test_nonzero_score_mean_and_beta_create_advantage_independent_term():
    score=torch.tensor([[2.,3.],[3.,4.],[4.,2.],[1.,3.]],dtype=torch.float64)
    adv=torch.tensor([-2.,-1.,1.,2.],dtype=torch.float64)
    weight=torch.where(adv<0,.5*adv,adv)
    gradient=(weight[:,None]*score).mean(0)
    covariance=((weight-weight.mean())[:,None]*(score-score.mean(0))).mean(0)
    torch.testing.assert_close(gradient,covariance+weight.mean()*score.mean(0))
    assert weight.mean()>0 and torch.linalg.vector_norm(weight.mean()*score.mean(0))>0


def test_module_matching_preserves_direction_and_exact_norm():
    a={'denoiser.blocks.0.weight':torch.tensor([3.,4.]),'denoiser.final_layer.weight':torch.tensor([0.,2.])}
    matched=matched_direction(a,{'transformer_00':.1,'qpos_output':.2})
    stats=module_stats(a,matched)
    assert abs(stats['transformer_00']['second_norm']-.1)<1e-7
    assert abs(stats['qpos_output']['second_norm']-.2)<1e-7
    assert abs(stats['transformer_00']['cosine']-1)<1e-7


def test_smooth_window_zero_on_prefix_and_boundaries():
    mask=torch.zeros(2,60,30,dtype=torch.bool);mask[0,11:]=True;mask[1,20:52]=True
    window=smooth_free_window(mask)
    assert (window[0,:12]==0).all() and window[0,-1]==0
    assert (window[1,:21]==0).all() and (window[1,51:]==0).all()
    assert window.min()>=0 and window.max()<=1 and window[0,20]==1


def test_complete_mc_independent_discount_and_terminal_event():
    from tools.analyze_decisive_credit import discounted_complete
    gamma=.95;g=gamma**(1/25)
    rows=[dict(controls=[dict(tick=12,dt_s=.02,reward=1.),dict(tick=24,dt_s=.02,reward=2.)],
               executed_control_steps=2,zero_step_event_reward=0.),
          dict(controls=[dict(tick=36,dt_s=.02,reward=3.)],executed_control_steps=1,zero_step_event_reward=-5.)]
    result=discounted_complete(rows,gamma)
    assert abs(result[0]-(1+2*g-2*g*g))<1e-12 and result[1]==-2


def test_complete_mc_rejects_missing_control_interval():
    import pytest
    from tools.analyze_decisive_credit import discounted_complete
    rows=[dict(controls=[dict(tick=12,dt_s=.02,reward=1.),dict(tick=36,dt_s=.02,reward=2.)],
               executed_control_steps=2,zero_step_event_reward=0.)]
    with pytest.raises(ValueError,match='Noncontiguous'):discounted_complete(rows,.95)


def test_consumed_reference_slerp_matches_half_turn_and_sign():
    import numpy as np
    from tools.summarize_decisive_physics import interpolate
    q=np.zeros((2,28));q[0,3]=1.;q[1,6]=-1.;q[1,0]=2.
    midpoint=interpolate(q,[(0,.5),(1,.5)])
    assert midpoint[0]==1. and abs(np.linalg.norm(midpoint[3:7])-1.)<1e-12
    assert abs(abs(midpoint[3])-2**-.5)<1e-12 and abs(abs(midpoint[6])-2**-.5)<1e-12


def test_consumed_reference_derivatives_do_not_cross_time_gaps():
    import numpy as np
    from tools.summarize_decisive_physics import reference_continuity
    class Kinematics:
        joint_limits=torch.tensor([[-1.,1.]]*21)
        def forward_body_positions(self,q):return q[:,:3,None].transpose(1,2)
    before={};after={}
    for i,tick in enumerate([0,12,24,60,72,84]):
        q=np.zeros(28);q[3]=1.;before[('task',tick)]=q.copy();q[7]=[0,.01,.02,.5,.51,.52][i];after[('task',tick)]=q
    result=reference_continuity(before,after,Kinematics())
    assert result['new_limit_coordinate_count']==0
    assert abs(result['maxima']['joint_velocity_delta_max']-.5)<1e-12
    assert result['maxima']['joint_acceleration_delta_max']<1e-10


def test_reward_reference_origin_tracks_copied_prefix():
    from tools.trace_decisive_plan_credit import origin_resolver
    plans={'one':dict(decision_tick=600,prefix_frames=10,parent_plan_id='bootstrap:one'),
           'two':dict(decision_tick=780,prefix_frames=10,parent_plan_id='one'),
           'three':dict(decision_tick=960,prefix_frames=10,parent_plan_id='two')}
    origin=origin_resolver(plans)
    assert origin('three',0)=='one' and origin('three',1)=='two' and origin('three',10)=='three'
    assert origin('one',2)=='bootstrap'
