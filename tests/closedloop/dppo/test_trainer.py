"""第九步 Actor 更新、联合概率诊断及原 Stage1 监督保持的 CPU 验收。

使用真实小型 Stage1Actor 和显式固定 rollout 验证 PPO-only 梯度、独立高斯解析 KL、
逐去噪步日志及 contact 辅助头隔离。监督测试直接调用原 Stage1BumiLosses，检查 BC
次数不随去噪步数膨胀、独立 warmup/RNG 可恢复，以及监督之后 Actor 回到 eval。
全部数据来自测试夹具并写入受控 pytest 临时目录，不使用正式 train/val 数据、不启动
GPU 或仿真，不以这些单步数值测试代替真实 checkpoint/执行数据验收。
"""
from __future__ import annotations

import copy
import math
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gem.closedloop.dppo.policy import DPPODiffusionPolicy
from gem.closedloop.dppo.trainer import SupervisedAnchor, actor_update, analytic_kl, probability_check
from gem.closedloop.frozen_actor import _fingerprint
from gem.closedloop.losses import Stage1BumiLosses
from tests.closedloop.test_stage1_actor import actor_factory, _activate_branches, _conditions
from tests.closedloop.test_stage1_losses import _weights


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _row(trace):
    return SimpleNamespace(context=trace['conditions'], chain=trace['chain'][0],
        old_log_prob=trace['old_log_probs'][0], free_mask=trace['free_mask'][0],
        metadata={'sampler_trace': trace}, transition_valid=True)


def _rollouts(actor, batch):
    policy = DPPODiffusionPolicy(actor, steps=3)
    rows = [_row(policy.sample_rollout(_conditions(batch), generator=torch.Generator().manual_seed(seed)))
            for seed in (14, 29)]
    return policy, rows


def test_ppo_only_updates_real_actor_without_contact_or_buffer_changes(actor_factory):
    actor, batch = actor_factory(starts=(45,))
    _activate_branches(actor)
    policy, rows = _rollouts(actor, batch)
    check = probability_check(policy, rows)
    assert check['passed'] and check['clip_fraction'] == 0
    assert check['ratio_quantiles']['p50'] == 1
    assert len(check['per_denoising_step']) == 3
    old_contact = _fingerprint(actor.denoiser.static_conf_head)
    old_buffers = {name:value.clone() for name,value in actor.named_buffers()}
    old_chain = rows[0].chain.clone()
    report = actor_update(policy, torch.optim.AdamW(actor.parameters(), lr=1e-6, weight_decay=.1),
                          rows, {'advantages': torch.tensor([1., -1.]), 'valid': torch.ones(2, dtype=torch.bool)},
                          bc=None, gamma_denoising=.9)
    assert report['parameters_changed'] and report['ppo_only_gradient_norm'] > 0
    assert report['bc'] is None and len(report['per_denoising_step']) == 3
    assert report['per_denoising_step'][0]['denoising_discount'] == pytest.approx(.81)
    assert report['per_denoising_step'][-1]['denoising_discount'] == 1
    assert _fingerprint(actor.denoiser.static_conf_head) == old_contact
    assert all(parameter.grad is None for parameter in actor.denoiser.static_conf_head.parameters())
    for name,value in actor.named_buffers():
        torch.testing.assert_close(value, old_buffers[name], rtol=0, atol=0)
    torch.testing.assert_close(rows[0].chain, old_chain, rtol=0, atol=0)
    kl = analytic_kl(policy, rows)
    assert kl['mean_joint_kl'] > 0 and math.isfinite(kl['mean_joint_kl'])
    assert kl['mean_chain_joint_kl'] == pytest.approx(3*kl['mean_joint_kl'])


def test_analytic_kl_matches_independent_normal_oracle(actor_factory):
    actor, batch = actor_factory(starts=(45,))
    policy, rows = _rollouts(actor, batch)
    assert analytic_kl(policy, rows)['max_joint_kl'] == 0
    with torch.no_grad():
        next(actor.denoiser.parameters()).add_(.0003)
    oracle=[]
    for item in rows:
        trace=item.metadata['sampler_trace']
        for index in range(3):
            parameters=policy.transition_parameters(item.context,item.chain[index:index+1],index)
            old=torch.distributions.Normal(trace['old_means'][:,index].double(),trace['old_stds'][:,index].double())
            new=torch.distributions.Normal(parameters['mean'].double(),parameters['std'].double())
            divergence=torch.distributions.kl_divergence(old,new)
            oracle.append(float(divergence.masked_select(item.free_mask[None]).sum()))
    report=analytic_kl(policy,rows)
    assert report['mean_joint_kl'] == pytest.approx(sum(oracle)/len(oracle), rel=1e-9, abs=1e-10)


@pytest.mark.parametrize('field', ['kernel', 'mask', 'old_probability'])
def test_probability_gate_rejects_tampered_contract_and_nan(actor_factory, field):
    actor, batch = actor_factory(starts=(45,))
    policy, rows = _rollouts(actor, batch)
    if field=='kernel':
        rows[0].metadata['sampler_trace']['kernel_config']['eta']=.9
    elif field=='mask':
        rows[0].free_mask=~rows[0].free_mask
    else:
        rows[0].old_log_prob[0]=float('nan')
    with pytest.raises((ValueError,FloatingPointError)):
        probability_check(policy,rows)
    with pytest.raises(ValueError,match='at least one'):
        probability_check(policy,[])


def test_clipped_objective_and_ratio_diagnostics_match_scalar_calculation(actor_factory):
    _, batch = actor_factory(starts=(45,), prefix=0)
    actor=torch.nn.Linear(1,1,bias=False)
    with torch.no_grad():
        actor.weight.zero_()
    class ScalarPolicy:
        steps=2
        timestep_map=(999,0)
        kernel_config={'test':'scalar'}
        def __init__(self):
            self.actor=actor
        def evaluate_log_probs(self, context, first, second, step):
            return self.actor.weight.flatten().double()*(step+1)
    policy=ScalarPolicy()
    ratios=[[1.005,1.2],[.5,1.005]]
    conditions=_conditions(batch)
    rows=[]
    for values in ratios:
        rows.append(SimpleNamespace(transition_valid=True,context=conditions,chain=torch.zeros(3,120,30),
            free_mask=torch.ones(120,30,dtype=torch.bool),old_log_prob=-torch.tensor(values,dtype=torch.float64).log(),
            metadata={'sampler_trace':{'kernel_config':policy.kernel_config,'timestep_map':torch.tensor([999,0])}}))
    advantages=[1.,-2.]
    expected=0.
    for advantage,values in zip(advantages,ratios):
        for step,ratio in enumerate(values):
            a=advantage*(.9**(1-step))
            expected-=min(ratio*a,min(max(ratio,.99),1.01)*a)/4
    result=actor_update(policy,torch.optim.SGD(actor.parameters(),lr=.001),rows,
                        {'advantages':torch.tensor(advantages)},bc=None,gamma_denoising=.9)
    assert result['ppo_loss'] == pytest.approx(expected)
    assert result['clip_fraction'] == .5
    assert result['ratio_quantiles']['min'] == .5
    assert result['ratio_quantiles']['max'] == 1.2


def _anchor(actor,batch):
    sample={key:(value[0] if key!='meta' else copy.deepcopy(value[0])) for key,value in batch.items() if key!='B'}
    class Dataset:
        def __len__(self):
            return 1
        def __getitem__(self,index):
            result=copy.deepcopy(sample)
            result['meta']['rng_probe']=[float(np.random.random()),random.random()]
            return result
    anchor=SupervisedAnchor.__new__(SupervisedAnchor)
    anchor.datasets=[Dataset() for _ in range(4)]
    anchor.losses=Stage1BumiLosses(actor.endecoder,_weights(),auxiliary_warmup_steps=10)
    anchor.generator=torch.Generator().manual_seed(181)
    anchor.torch_rng=torch.Generator().manual_seed(182).get_state()
    anchor.numpy_rng=np.random.RandomState(183).get_state()
    anchor.python_rng=random.Random(184).getstate()
    anchor.cuda_rng=None
    anchor.batch_size=2
    anchor.bc_update_steps=3
    return anchor


def test_original_stage1_supervision_rng_warmup_and_gradients_restore(actor_factory):
    actor,batch=actor_factory(starts=(45,))
    anchor=_anchor(actor,batch)
    saved=copy.deepcopy(anchor.state_dict())
    torch_before=torch.get_rng_state().clone()
    numpy_before=np.random.get_state()
    python_before=random.getstate()
    first=anchor.backward(actor,weight=.1)
    first_grads={name:parameter.grad.clone() for name,parameter in actor.named_parameters() if parameter.grad is not None}
    assert first['batch_size']==2 and first['bc_update_steps']==4
    assert first['warmup_step']==3 and first['warmup_origin']=='independent_bc_updates_from_zero'
    assert all(row['auxiliary_warmup_factor']==pytest.approx(.3) for row in first['warmup_factors'])
    assert all(value>0 for value in first['weighted_gradient_norm_per_microbatch'])
    assert any(parameter.grad is not None and parameter.grad.abs().sum()>0 for parameter in actor.denoiser.static_conf_head.parameters())
    torch.testing.assert_close(torch.get_rng_state(),torch_before,rtol=0,atol=0)
    np.testing.assert_array_equal(np.random.get_state()[1],numpy_before[1])
    assert random.getstate()==python_before
    assert not actor.training
    actor.zero_grad(set_to_none=True)
    anchor.load_state_dict(saved)
    second=anchor.backward(actor,weight=.1)
    assert first['loss']==second['loss'] and first['samples']==second['samples']
    for name,parameter in actor.named_parameters():
        if name in first_grads:
            torch.testing.assert_close(parameter.grad,first_grads[name],rtol=0,atol=0)
