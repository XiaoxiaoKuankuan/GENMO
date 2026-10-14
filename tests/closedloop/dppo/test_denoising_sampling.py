"""服务器1执行的去噪时间步抽样损失、随机恢复与全量KL边界测试。

用真实联合高斯PPO公式和可控小策略验证：全步模式不增加RNG消耗，抽样身份与rank
分片无关，重复种子精确重建；轮转无放回子集平均梯度等于原完整目标。4/8步只减少
内部loss样本，old链和概率不变、链minibatch和BC频率不变，最终KL仍覆盖所有20步。
这不是把抽样梯度等同单次全梯度：单次抽样方差需由真实rollout诊断单独报告。
"""
import copy
import pytest
import torch
from gem.closedloop.dppo.denoising_sampling import chain_steps,plan_record
from gem.closedloop.dppo.policy import masked_joint_log_prob
from tests.closedloop.dppo.test_optimization_support import BatchGaussianPolicy, rows, Anchor


class TwentySteps(BatchGaussianPolicy):
    steps=20
    timestep_map=tuple(range(999,0,-50))


def samples(policy):
    result=rows(BatchGaussianPolicy(),2)
    generator=torch.Generator().manual_seed(135)
    for row in result:
        chain=torch.zeros(21,120,30);means=[];stds=[];prob=[]
        with torch.no_grad():
            for step in range(20):
                parameters=policy.transition_parameters(row.context,chain[step:step+1],step)
                chain[step+1]=parameters['mean'][0]+.7*torch.randn(120,30,generator=generator)
                means.append(parameters['mean']);stds.append(parameters['std'])
                prob.append(masked_joint_log_prob(chain[step+1:step+2],parameters['mean'],parameters['std'],row.free_mask[None])[0])
        row.chain=chain;row.old_log_prob=torch.stack(prob)
        row.metadata['sampler_trace'].update(kernel_config=policy.kernel_config,timestep_map=torch.tensor(policy.timestep_map),
            old_means=torch.stack(means,1),old_stds=torch.stack(stds,1))
    return result


def test_plan_is_independent_of_owner_and_full_steps_do_not_need_seed():
    for count in (4,8,20):
        entire={i:chain_steps(91,i,20,count) for i in range(33)}
        merged={i:chain_steps(91,i,20,count) for rank in range(8) for i in range(rank,33,8)}
        assert entire==merged
        assert all(len(set(value))==count for value in entire.values())
        report=plan_record(91 if count<20 else None,list(range(33)),20,count)
        assert sum(report['step_histogram'])==33*count and report['inverse_probability_weight']==20/count
    assert chain_steps(None,0,20,20)==list(range(20))


def test_stratified_terminal_steps_and_exact_unbiased_weights():
    from gem.closedloop.dppo.denoising_sampling import validate_sampling
    import json
    for seed in range(20):
        selected=chain_steps(seed,13,20,8,'stratified8')
        assert len(set(selected))==8 and selected[-2:]==[18,19]
        plan=plan_record(seed,[13],20,8,'stratified8')
        assert plan==json.loads(json.dumps(plan))
        assert plan['inverse_probability_weight']==[3.]*18+[1.,1.]
        assert plan['selected_chain_steps']==[[13,selected]]
    # 枚举均匀六步的循环覆盖，三倍权重准确恢复每个原始步骤及其去噪折扣。
    values=torch.arange(1,21,dtype=torch.float64)*.99**torch.arange(19,-1,-1,dtype=torch.float64)
    estimates=[]
    for offset in range(18):
        steps=[(offset+j)%18 for j in range(6)]
        estimates.append((3*values[steps].sum()+values[18:].sum())/20)
    torch.testing.assert_close(torch.stack(estimates).mean(),values.mean(),atol=1e-14,rtol=1e-14)
    with pytest.raises(ValueError):validate_sampling(20,4,'stratified8')










@pytest.mark.parametrize('count',[4,8,20])
def test_independent_auditor_reconstructs_sampling_not_just_total_count(count):
    from tools.eval.audit_closedloop_stage10_v2 import _step_sampling
    indices=[0,19,31];seed=731 if count<20 else None
    contract=dict(denoising_steps=20,denoising_steps_per_chain=count)
    step=dict(internal_transitions=len(indices)*count,full_internal_transitions=len(indices)*20,
              denoising_sampling=plan_record(seed,indices,20,count))
    assert _step_sampling(step,indices,contract)==count
    for key,value in [('inverse_probability_weight',1.01),('selection_sha256','changed'),
                      ('full_20_step_hard_kl_required',False)]:
        bad=copy.deepcopy(step);bad['denoising_sampling'][key]=value
        with pytest.raises((ValueError,AssertionError)):
            _step_sampling(bad,indices,contract)
    bad=copy.deepcopy(step);bad['internal_transitions']+=1
    with pytest.raises((ValueError,AssertionError)):_step_sampling(bad,indices,contract)
    if count==20:
        assert _step_sampling(dict(internal_transitions=60),indices,dict(denoising_steps=20))==20


def test_terminal_steps_strategy_is_deterministic_and_marks_objective_scope():
    assert chain_steps(None,5,20,4,'last')==[16,17,18,19]
    assert chain_steps(7,9,20,4,'last')==chain_steps(11,2,20,4,'last')
    record=plan_record(None,[0,1,2],20,4,'last')
    assert record['contract']=='dppo.terminal_steps_objective_only.v1'
    assert record['inclusion_probability']==[0.]*16+[1.]*4 and record['step_histogram'][16:]==[3]*4
    assert record['objective_scope']=='terminal_K_steps_mean_not_full20_estimate'
    with pytest.raises(ValueError,match='Unknown'):chain_steps(None,0,20,4,'tail')


def test_terminal_strategy_actor_update_uses_only_last_steps(monkeypatch, tmp_path):
    import torch.distributed as dist
    from gem.closedloop.dppo import updater
    from gem.closedloop.dppo.collectives import DistributedCollectives
    from gem.closedloop.dppo.gradient_merge import BCGradientController
    from gem.closedloop.dppo.tensor_cache import RolloutTensorCache

    class NoConditionEncoder:
        bank = None
        def __init__(self, *args): pass
        def prime(self, rows): pass
        def backward(self): pass

    monkeypatch.setattr(updater, 'ConditionGraphCache', NoConditionEncoder)
    policy = TwentySteps(); policy.precision = 'native_fp32'
    data = samples(policy)
    targets = dict(advantages=torch.tensor([1., -.6]))
    manifest = [dict(owner_rank=0, local_index=i, valid=True, has_free=True) for i in range(2)]
    dist.init_process_group('gloo', init_method=(tmp_path / 'one_rank').as_uri(), rank=0, world_size=1)
    cache = RolloutTensorCache(data, targets, 'cpu')
    try:
        report = updater.actor_update(policy, torch.optim.AdamW(policy.actor.parameters(), lr=1e-5),
            data, targets, cache, manifest, BCGradientController(initial=0., minimum=0., mode='fixed'),
            distributed=DistributedCollectives(0, 1), anchor=Anchor(), generator=torch.Generator().manual_seed(3),
            sampled_steps=4, minibatch_chains=2, microbatch=5, soft_kl=None, step_strategy='last')
    finally:
        cache.close()
        dist.destroy_process_group()
    for step in report['steps']:
        assert [s['step'] for s in step['per_denoising_step'] if s['visits']] == [16, 17, 18, 19]
        assert step['denoising_sampling']['strategy'] == 'last'
    assert report['internal_sample_visits'] == 2 * 4 * 2
