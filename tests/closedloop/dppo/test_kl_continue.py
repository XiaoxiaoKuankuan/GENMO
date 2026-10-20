"""KL超限撤销后继续采样及有效模型持久化的CPU回归。

用真实小网络、非空Adam与RNG调用生产整轮快照/恢复函数，核验有限KL超限后学习
状态逐字节恢复，而物理进度与实际优化尝试不回退；再执行新数据更新，证明恢复后
优化器仍可工作。维护测试覆盖第一次拒绝同步补存、连续拒绝不重复写大模型、
拒绝不触发旧里程碑，以及滚动保留不会删除重要模型。非有限KL仍必须报错。
这里不启动Isaac或GPU，不把CPU控制流回归当作物理学习质量证明。
"""
import copy
import json
import random
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from gem.closedloop.dppo import training
from gem.closedloop.dppo.rollback_audit import compare_recovered_state
from gem.closedloop.dppo.training_checkpoint import AsyncCheckpointWriter, SCHEMA


class Collective:
    rank=0
    def all_gather_object(self,value):return [value]
    def broadcast_object(self,value):return value


class State:
    def __init__(self):self.value={'cursor':7}
    def state_dict(self):return copy.deepcopy(self.value)
    def load_state_dict(self,value):self.value=copy.deepcopy(value)


def update(module,optimizer,target):
    optimizer.zero_grad(set_to_none=True)
    (module(torch.ones(2,3))-target).square().mean().backward()
    optimizer.step()


def context(tmp_path):
    actor=torch.nn.Linear(3,1);critic=torch.nn.Linear(3,1)
    ao=torch.optim.AdamW(actor.parameters(),lr=3e-7)
    co=torch.optim.AdamW(critic.parameters(),lr=1e-4)
    update(actor,ao,1.);update(critic,co,2.)
    return NS(actor=actor,critic=critic,actor_optimizer=ao,critic_optimizer=co,
        bc=State(),controller=State(),generators={'actor':torch.Generator().manual_seed(42)},
        session=tmp_path,distributed=Collective(),overhead={},
        state=dict(iteration=2,policy_version=2,actor_updates=5,critic_updates=40,
            optimizer_attempts=5,rollouts_completed=2,training_control_steps=100,
            next_evaluation_control_steps=100000,next_mc_calibration_control_steps=100000),
        stage=dict(evaluation=dict(every_actor_iterations=100,every_control_steps=100000),
            critic=dict(complete_mc=dict(enabled=True,every_actor_iterations=100,every_control_steps=100000)),
            storage=dict(checkpoint_interval_iterations=20,checkpoint_interval_seconds=1800)))


def test_kl_rejection_restores_nonempty_adam_and_rng_but_allows_next_update(tmp_path):
    c=context(tmp_path);saved=training._rollback_snapshot(c)
    update(c.actor,c.actor_optimizer,4.);update(c.critic,c.critic_optimizer,4.)
    c.bc.value['cursor']=99;c.controller.value['cursor']=99
    random.random();np.random.rand();torch.rand(2,generator=c.generators['actor'])
    c.state.update(critic_updates=43,optimizer_attempts=8,training_control_steps=200)
    acceptance=dict(passed=False,failures={'p95_coordinate':dict(value=.00272,limit=.0021)})
    result=training._rollback_kl_rejection(c,saved,kl={'p95':.00272},acceptance=acceptance,
        actor={'optimizer_steps':3},label='train_000003')
    assert compare_recovered_state(saved,training._rollback_snapshot(c))['identical']
    assert result['action']=='resample_and_continue' and result['training_state_restored']
    assert c.state['iteration']==2 and c.state['policy_version']==2 and c.state['rejected_rollouts']==1
    assert c.state['optimizer_attempts']==8 and c.state['training_control_steps']==200
    assert not (tmp_path/'anomalies').exists()
    assert json.loads((tmp_path/'train_000003_rejected.json').read_text())['retained_accepted_iteration']==2
    before=copy.deepcopy(c.actor.state_dict())
    update(c.actor,c.actor_optimizer,-2.)
    assert any(not torch.equal(v,c.actor.state_dict()[k]) for k,v in before.items())
    assert c.actor_optimizer.state_dict()['state']


@pytest.mark.parametrize('value',[float('nan'),float('inf')])
def test_nonfinite_kl_is_not_ordinary_rejection(tmp_path,value):
    c=context(tmp_path)
    with pytest.raises(FloatingPointError):
        training._rollback_kl_rejection(c,training._rollback_snapshot(c),kl={},
            acceptance={'failures':{'mean':{'value':value}}},actor={},label='bad')
    assert not list(tmp_path.glob('*rejected.json'))


def test_first_rejection_saves_valid_state_once_and_new_accepted_state_saves_again(tmp_path,monkeypatch):
    c=context(tmp_path);calls=[]
    monkeypatch.setattr(training.time,'perf_counter',lambda:100.)
    def snapshot(c,reason,*,wait=False):
        calls.append((reason,wait,training._accepted_version(c)))
        c.last_saved_accepted_version=training._accepted_version(c)
    monkeypatch.setattr(training,'_snapshot',snapshot)
    last=training._maintain_after_iteration(c,{},is_warmup=False,rejected=True,last_save=99.)
    assert calls==[('kl_rejected_rollback',True,(2,5,40,2))]
    c.state['rollouts_completed']+=1;c.state['training_control_steps']+=100
    training._maintain_after_iteration(c,{},is_warmup=False,rejected=True,last_save=last)
    assert len(calls)==1
    c.state.update(iteration=3,actor_updates=7,policy_version=3)
    training._maintain_after_iteration(c,{},is_warmup=False,rejected=True,last_save=last)
    assert len(calls)==2


def test_rollback_checkpoints_share_rolling_retention_preserving_milestones(tmp_path):
    writer=AsyncCheckpointWriter(tmp_path,keep_last=2);paths=[]
    try:
        for i,reason in enumerate(['iteration_000020','kl_rejected_rollback','periodic','kl_rejected_rollback','periodic+kl_rejected_rollback']):
            payload=dict(schema=SCHEMA,state=dict(iteration=i,rollouts_completed=i,actor_updates=i,critic_updates=i),
                actor={'weight':torch.tensor([float(i)])},actor_fingerprint=str(i))
            writer.submit(payload,reason=reason);paths.append(tmp_path/writer.poll(wait=True)['path'])
        assert paths[0].exists() and paths[-1].exists() and paths[-2].exists()
        assert not paths[1].exists() and not paths[2].exists()
        assert torch.load(paths[-1],weights_only=False)['state']['iteration']==4
    finally:writer.close()
