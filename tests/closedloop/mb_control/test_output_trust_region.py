"""候选更新事务的CPU单元检查，不充当GENMO/SM物理或训练收益证据。

用小型可微参数验证真实AdamW的接受、回滚、一二阶矩/step/RNG恢复、重复固定输出、
累计教师偏差和活动度拒绝。真实20步DDIM与Newton候选对照另由服务器实资产审计完成。
"""
import copy
import torch
from gem.closedloop.mb_control.output_trust_region import candidate_adamw_step,check_outputs


def equal_tree(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    return a==b


def setup(lr=.001):
    actor=torch.nn.Linear(2,2);optimizer=torch.optim.AdamW(actor.parameters(),lr=lr)
    actor(torch.ones(1,2)).square().sum().backward();optimizer.step();optimizer.zero_grad()
    actor(torch.ones(1,2)).sum().backward()
    return actor,optimizer


def test_small_real_adamw_update_accepted():
    actor,opt=setup();before=copy.deepcopy(actor.state_dict())
    r=candidate_adamw_step(actor,opt,lambda:([],{}),config={},execution_gradient_valid=True)
    assert r['accepted'] and not equal_tree(before,actor.state_dict())


def test_rejection_restores_every_parameter_adam_state_and_rng():
    actor,opt=setup();before=copy.deepcopy(actor.state_dict());state=copy.deepcopy(opt.state_dict());rng=torch.get_rng_state().clone()
    def reject():
        torch.rand(20)
        return ['joint_step_exceeded'],{}
    r=candidate_adamw_step(actor,opt,reject,config={},execution_gradient_valid=True)
    assert not r['accepted'] and len(r['attempts'])==3
    assert equal_tree(before,actor.state_dict()) and equal_tree(state,opt.state_dict())
    assert torch.equal(rng,torch.get_rng_state())


def test_backtracking_keeps_only_accepted_moments_and_step():
    actor,opt=setup(lr=.1);before=copy.deepcopy(actor.state_dict());oldstep=next(iter(opt.state.values()))['step'].clone()
    def check():
        change=max(float((p-before[n]).abs().max()) for n,p in actor.state_dict().items())
        return (['joint_step_exceeded'] if change>.015 else []),{'change':change}
    r=candidate_adamw_step(actor,opt,check,config={},execution_gradient_valid=True)
    assert r['accepted'] and len(r['attempts'])>=2
    assert all(torch.equal(s['step'],oldstep+1) for s in opt.state.values())


def test_missing_execution_gradient_never_updates():
    actor,opt=setup();before=copy.deepcopy(actor.state_dict())
    r=candidate_adamw_step(actor,opt,lambda:([],{}),config={},execution_gradient_valid=False)
    assert not r['accepted'] and equal_tree(before,actor.state_dict())


def poses(scale=1.,offset=0.):
    p=torch.zeros(1,60,28);p[:,:,3]=1
    p[:,:,7:]=scale*torch.sin(torch.linspace(0,6.28,60))[None,:,None]+offset
    return p


CFG={'joint_step_rad':.015,'root_step_m':.005,'orientation_step_rad':.01,
     'local_joint_drift_rad':.05,'global_joint_drift_rad':.05,'global_root_drift_m':.15,
     'global_orientation_drift_rad':.25,'minimum_activity_ratio':.65,'minimum_phase_cosine':-.2}


def test_large_output_change_rejected_and_known_prefix_exact():
    p=poses();a=p.clone();a[:,10:,7:]+=.1;mask=torch.arange(60)>=10
    reasons,_=check_outputs(p,a,p,p,mask,prefix=10,config=CFG)
    assert 'joint_step_exceeded' in reasons and 'global_teacher_drift' in reasons


def test_many_small_steps_cannot_escape_frozen_teacher():
    teacher=poses();p=teacher.clone();mask=torch.arange(60)>=10;accepted=0
    for _ in range(20):
        a=p.clone();a[:,10:,7:]+=.01
        reasons,_=check_outputs(p,a,teacher,teacher,mask,prefix=10,config=CFG)
        if not reasons:p=a;accepted+=1
    assert 1<=accepted<=5 and accepted<20


def test_static_candidate_is_rejected_without_relaxing_activity():
    p=poses();a=p.clone();a[:,10:,7:]=0;mask=torch.arange(60)>=10
    reasons,_=check_outputs(p,a,p,p,mask,prefix=10,config=CFG)
    assert 'motion_activity_collapse' in reasons


def test_fixed_request_noise_regeneration_and_rejected_count():
    actor,opt=setup();request=torch.randn(1,2);noise=torch.randn(1,2)
    def generate():return actor(request)+noise
    before=generate().detach().clone();accepted_actor_updates=0
    verdict=candidate_adamw_step(actor,opt,lambda:(['candidate_dance_loss_worse'],{}),config={},execution_gradient_valid=True)
    accepted_actor_updates+=int(verdict['accepted'])
    assert accepted_actor_updates==0 and torch.equal(before,generate().detach())
