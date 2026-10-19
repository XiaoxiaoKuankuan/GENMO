"""统一高学习率实验的CPU/CUDA事务、score梯度和FP32舍入回归。

小型可解析网络验证非空Adam每次回溯从同一起点执行、完全拒绝和异常恢复、随机
状态与BC控制器一致、冻结参数排除、joint-score无指数以及有效链分母。合成KL
只检查控制逻辑，绝不作为真实策略或物理安全验收证据。CUDA用例必须在明确空闲
GPU上显式执行，验证非空Adam字节等价及实际CUDA随机状态恢复；无设备时明确跳过。
"""
import copy
from types import SimpleNamespace
import pytest
import torch
from gem.closedloop.dppo.actor_backtracking import backtracked_step, uniform_actor_optimizer, check_candidate_kl
from gem.closedloop.dppo.actor_step_diagnostics import StepDiagnostics
from gem.closedloop.dppo.full_batch_actor import joint_score_objective
from gem.closedloop.dppo.gaussian_probability import gaussian_statistics, chain_statistics, chain_diagnostics
from gem.closedloop.dppo.gradient_merge import BCGradientController
from tests.closedloop.test_stage1_actor import actor_factory


class Collective:
    def all_gather_object(self, value):return [value]


def kl(value):
    return dict(mean_coordinate_kl=value, maximum_step_mean_coordinate_kl=value,
        coordinate_kl={'p95':value}, chain_joint_kl={'mean':100*value,'p95':120*value,'max':150*value})


def setup(device='cpu'):
    if device=='cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA transaction regression requires an explicitly free GPU')
    actor=torch.nn.Linear(2,1,device=device)
    optimizer=torch.optim.AdamW(actor.parameters(),lr=1e-6,weight_decay=0.)
    for p in actor.parameters():p.grad=torch.ones_like(p)
    optimizer.step()
    for p in actor.parameters():p.grad=torch.full_like(p,.3)
    return actor,optimizer,BCGradientController()


def equal_state(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal_state(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal_state(x,y) for x,y in zip(a,b))
    return a==b


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_backtrack_nonempty_adam_matches_one_direct_step_and_counts_attempts(device):
    actor,opt,controller=setup(device)
    reference=copy.deepcopy(actor)
    expected=torch.optim.AdamW(reference.parameters(),lr=1e-6,weight_decay=0.)
    expected.load_state_dict(copy.deepcopy(opt.state_dict()))
    for p in reference.parameters():p.grad=torch.full_like(p,.3)
    expected.param_groups[0]['lr']=2.5e-7;expected.step()
    read_rng=torch.cuda.get_rng_state if device=='cuda' else torch.get_rng_state
    attempts=[];rng=read_rng().clone()
    def evaluate():
        torch.rand(4,device=device)
        return kl(opt.param_groups[0]['lr'])
    result=backtracked_step(actor,opt,controller,evaluate=evaluate,
        limits={'mean_coordinate':3e-7},distributed=Collective(),max_attempts=4,
        reserve_attempt=lambda:attempts.append(1))
    assert result['accepted_lr']==2.5e-7 and len(attempts)==3
    assert equal_state(actor.state_dict(),reference.state_dict())
    assert equal_state(opt.state_dict(),expected.state_dict())
    assert torch.equal(read_rng(),rng)
    assert all(float(s['step'])==2 for s in opt.state.values())


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_all_rejected_restores_model_adam_bc_and_rng_without_refunding(device):
    actor,opt,controller=setup(device)
    before=copy.deepcopy((actor.state_dict(),opt.state_dict(),controller.state_dict()))
    read_rng=torch.cuda.get_rng_state if device=='cuda' else torch.get_rng_state
    rng=read_rng().clone();count=[]
    def evaluate():
        torch.rand(5,device=device)
        return kl(.1)
    result=backtracked_step(actor,opt,controller,evaluate=evaluate,limits={'mean_coordinate':.001},
        distributed=Collective(),max_attempts=3,reserve_attempt=lambda:count.append(1))
    assert not result['accepted'] and len(count)==3
    assert equal_state(before,(actor.state_dict(),opt.state_dict(),controller.state_dict()))
    assert torch.equal(rng,read_rng())


def test_exception_restores_and_multiple_groups_or_frozen_parameter_rejected():
    actor,opt,c=setup();original=copy.deepcopy(actor.state_dict())
    def fail():raise RuntimeError('synthetic audit failure')
    with pytest.raises(RuntimeError,match='synthetic'):
        backtracked_step(actor,opt,c,evaluate=fail,limits={},distributed=Collective())
    assert equal_state(original,actor.state_dict())
    actor.bias.requires_grad_(False)
    with pytest.raises(ValueError,match='frozen'):uniform_actor_optimizer(actor,opt)
    groups=torch.optim.AdamW([{'params':[actor.weight]},{'params':[actor.bias]}])
    with pytest.raises(ValueError,match='one uniform'):uniform_actor_optimizer(actor,groups)


def test_joint_score_exact_gradient_without_exp_or_free_coordinate_division():
    old=torch.zeros(3,2,4,dtype=torch.float64)
    new=old.clone().requires_grad_(True)
    value=torch.arange(24,dtype=torch.float64).reshape_as(old)/10
    mask=torch.ones_like(old,dtype=torch.bool);mask[0,0]=False
    std=torch.full((3,1,1),.1,dtype=torch.float64)
    advantages=torch.tensor([1.,-.2,.3],dtype=torch.float64)
    stats=gaussian_statistics(value,old,std,new,std,mask)
    loss=-joint_score_objective(stats,advantages).sum()/3
    loss.backward()
    expected=-value/std.square()*advantages[:,None,None]/3
    torch.testing.assert_close(new.grad,expected.masked_fill(~mask,0.),rtol=1e-12,atol=1e-12)
    huge={'joint_log_ratio':torch.tensor([1e6,-1e6]),'free_count':torch.ones(2)}
    assert torch.isfinite(joint_score_objective(huge,torch.ones(2))).all()


def test_fp32_rounding_diagnostic_identifies_real_zero_updates():
    actor=torch.nn.Linear(2,1,bias=False)
    with torch.no_grad():actor.weight.fill_(1.)
    opt=torch.optim.AdamW(actor.parameters(),lr=1e-9,weight_decay=0.)
    obs=StepDiagnostics(actor)
    actor.weight.grad=torch.ones_like(actor.weight)
    opt.step()
    result=obs.finish(opt,.1,1.)['modules']['all']
    assert result['changed_fraction']==0.
    assert result['fp32_rounded_to_zero_fraction']==1.
    assert result['adam_subtraction_fp64_delta_rms']>0


def test_joint_gate_is_additional_and_bc_first_step_uses_measured_norm():
    assert not check_candidate_kl(kl(.001),{'mean_coordinate':.002,'chain_joint_max':.1})['passed']
    c=BCGradientController(initialize_from_gradient=True,target_ratio=.1)
    assert c.update(2.,4.)==pytest.approx(.05)
    before=c.value
    assert c.update(100.,4.)<=2*before


def test_configuration_binds_every_actual_optimizer_choice(tmp_path):
    from pathlib import Path
    import yaml
    from gem.closedloop.dppo.training_config import configuration
    source=Path(__file__).resolve().parents[3]/'configs/closedloop/stage2_full_chain.yaml'
    assert configuration(source)['stage10']['derived']['maximum_actor_steps']==8
    c=yaml.safe_load(source.read_text())
    c['stage10']['kl']['calibration']=configuration(source)['stage10']['kl']['calibration']
    c['stage10']['training']['actor_lr']=3e-6
    c['stage10']['training']['trust_region']['maximum_lr']=3e-6
    p=tmp_path/'changed.yaml';p.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError,match='binding'):configuration(p)
    c=yaml.safe_load(source.read_text())
    c['stage10']['training']['actor_objective']='coordinate_dppo'
    p.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError,match='full-chain'):configuration(p)


def test_numerical_candidate_rejected_then_retried_without_extra_adam_time():
    actor,opt,c=setup();count=[]
    def evaluate():
        count.append(opt.param_groups[0]['lr'])
        if len(count)==1:raise FloatingPointError('explicit ratio underflow')
        return kl(0.)
    result=backtracked_step(actor,opt,c,evaluate=evaluate,limits={},distributed=Collective())
    assert result['accepted_lr']==5e-7 and result['attempt_count']==2
    assert result['attempts'][0]['acceptance']['failures']['numerical']=='explicit ratio underflow'
    assert all(float(s['step'])==2 for s in opt.state.values())


def test_concentrated_finite_ratios_do_not_reject_valid_joint_kl():
    # 构造一条罕见但有限的旧高斯样本；其大ratio已被正优势PPO裁剪。
    # 权重集中本身不拒绝更新；真正超出联合KL或active clip上限时仍拒绝。
    old=torch.zeros(32,4,1,dtype=torch.float64)
    value=old.clone();value[-1]=25.
    sigma=torch.ones_like(old);mask=torch.ones_like(old,dtype=torch.bool)
    flat=lambda tensor:tensor.reshape(128,1,1)
    steps=gaussian_statistics(flat(value),flat(old),flat(sigma),flat(old+.1),flat(sigma),flat(mask))
    chain=chain_statistics(*(steps[key].reshape(32,4) for key in ('joint_log_ratio','joint_kl','free_count')))
    report=chain_diagnostics(chain,torch.ones(32),clip=.1)
    assert report['chain_joint_ratio']['max']>1e4
    assert report['chain_joint_kl']['mean']==pytest.approx(.02)
    assert report['active_clip_fraction']==pytest.approx(1/32)
    limits=dict(chain_joint_mean=.1,chain_joint_p95=.6,chain_joint_p99=.9,chain_joint_max=2.,maximum_active_clip=.95)
    assert check_candidate_kl(report,limits)['passed']
    actor,opt,c=setup()
    accepted=backtracked_step(actor,opt,c,evaluate=lambda:report,limits=limits,distributed=Collective())
    assert accepted['accepted'] and accepted['attempt_count']==1
    assert accepted['accepted_lr']==1e-6
    joint_failure=check_candidate_kl(report,{**limits,'chain_joint_mean':.01})
    assert set(joint_failure['failures'])=={'chain_joint_mean'}
    clip_failure=check_candidate_kl(report,{**limits,'maximum_active_clip':.01})
    assert set(clip_failure['failures'])=={'active_clip_fraction'}


def test_real_policy_full_batch_one_step_and_no_old_chain_mutation(tmp_path,actor_factory):
    from tests.closedloop.test_stage1_actor import _activate_branches,_conditions
    from tests.closedloop.dppo.test_asymmetric_critic import tracking_context
    from gem.closedloop.contracts import STAGE1_CONDITION_KEYS
    from gem.closedloop.dppo.frozen_prefix import FrozenPrefixPolicy
    from gem.closedloop.dppo.rollout_helpers import split_trace
    from gem.closedloop.dppo.tensor_cache import RolloutTensorCache
    from gem.closedloop.dppo.collectives import DistributedCollectives
    from gem.closedloop.dppo.full_batch_actor import full_batch_actor_update
    from gem.closedloop.dppo.trainer import trainable_actor_parameters
    import torch.distributed as dist
    class Anchor:
        def __init__(self):self.calls=0;self.batch_size=2
        def state_dict(self):return {'calls':self.calls,'batch_size':self.batch_size}
        def load_state_dict(self,s):self.calls=s['calls'];self.batch_size=s['batch_size']
        def backward(self,actor,weight):
            self.calls+=1
            loss=sum(p.square().mean() for p in actor.parameters() if p.requires_grad)
            (loss*weight).backward()
            return dict(batch_size=self.batch_size,loss=float(loss.detach()),samples=[[{'dataset_id':'fixture'}]*self.batch_size])
    actor,batch=actor_factory(motion_frames=60);_activate_branches(actor)
    policy=FrozenPrefixPolicy(actor,std_schedule=[.001]*15+[.04,.06,.08,.1,0.],generation_frames=60)
    privileged=tracking_context(_conditions(batch));context={k:privileged[k] for k in STAGE1_CONDITION_KEYS}
    trace=policy.sample_rollout(context,generator=torch.Generator().manual_seed(17))
    rows=[SimpleNamespace(context={k:v[i:i+1].clone() for k,v in context.items()},identity={},
        chain=trace['chain'][i].clone(),old_log_prob=trace['old_log_probs'][i].clone(),free_mask=trace['free_mask'][i].clone(),
        metadata=dict(critic_context={k:v[i:i+1].clone() for k,v in privileged.items()},
            sampler_trace=split_trace(trace,i,2),remaining_music_seconds=5.)) for i in range(2)]
    original=torch.stack([r.chain.clone() for r in rows])
    targets=dict(advantages=torch.tensor([1.,-1.]),advantages_raw=torch.tensor([2.,-2.]),valid=torch.ones(2,dtype=torch.bool))
    manifest=[dict(owner_rank=0,local_index=i,valid=True,has_free=True) for i in range(2)]
    dist.init_process_group('gloo',init_method=(tmp_path/'gloo').as_uri(),rank=0,world_size=1)
    cache=RolloutTensorCache(rows,targets,'cpu')
    optimizer=torch.optim.AdamW(trainable_actor_parameters(actor),lr=1e-6,weight_decay=0.)
    try:
        report=full_batch_actor_update(policy,optimizer,rows,targets,cache,manifest,
            BCGradientController(initialize_from_gradient=True),distributed=DistributedCollectives(0,1),
            anchor=Anchor(),generator=torch.Generator().manual_seed(4),
            backtracking={'max_attempts':2,'factor':.5},kl_limits={'mean_coordinate':1.},
            microbatch=3,kl_microbatch=3,bc_samples_per_rollout=4)
        assert report['optimizer_steps']==1 and report['attempted_optimizer_steps']==1
        assert report['actual_global_bc_samples']==4 and report['internal_sample_visits']==8
        assert report['steps'][0]['denominator']==2
        assert torch.equal(original,torch.stack([r.chain for r in rows]))
        assert policy.assert_frozen_identity()['checked']
    finally:
        cache.close();dist.destroy_process_group()
