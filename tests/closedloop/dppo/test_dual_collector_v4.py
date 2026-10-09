"""双环境ready队列原型的独立状态、因果请求与故障恢复CPU验收。

环境替身每一步都等待实际生成返回才推进状态，记录重置/游标/独立噪声种子和
线程归属；真实小型Actor另验证双生成器合批与独立单链一致。测试不会自动启动
额外Isaac进程。它证明调度与随机性边界，不能作为双真实物理环境吞吐验收。
"""
from types import SimpleNamespace as NS
import copy
import threading
import time
import pytest
import torch
from gem.closedloop.dppo.dual_collector import DualEnvironmentCollector,split_trace
from gem.closedloop.dppo.policy import DPPODiffusionPolicy
from tests.closedloop.test_stage1_actor import actor_factory,_conditions,_activate_branches


class Policy:
    numerical_layout='sample_matrix_bmm_fp32.v1'
    steps=20
    guidance_scale=2.5
    cfg_batch=True
    kernel_config={}
    last_sample_timing={}
    actor=torch.nn.Linear(1,1)
    def _parameter_signature(self): return tuple(p._version for p in self.actor.parameters())
    def sample_rollout(self,context,*,generator):
        assert len({g.initial_seed() for g in generator})==len(generator)
        return dict(normalized=context['state'].clone(),conditions=context,timestep_map=torch.arange(20),kernel_config={})


def factory(resources, fail_slot=None):
    def create(slot,policy):
        owner=threading.get_ident()
        class Sampler:
            def __init__(self): self.draw=0;self.steps=0
            def next_task(self):
                self.draw+=1
                return dict(sample=dict(dataset='Mine',row=dict(split='train',sample_id=f'env{slot}-{self.draw}'),manifest_sha256='hash'),
                    music=None,music_start_frame=0,task_index=self.draw)
            def record_execution(self,task,steps):self.steps+=steps
            def state_dict(self):return dict(draw=self.draw,steps=self.steps)
            def load_state_dict(self,s):self.__dict__.update(s)
        class Env:
            def __init__(self):
                self.backend=NS(session_id=f'{slot}-{time.perf_counter_ns()}')
                self.config=dict(stage9=dict(seed=17+slot))
                self.episode_count=self.decision=self.attempt=self.iteration=0
                self.latency_budget_s=.5
                self.resets=0;self.snapshot=dict(tick=0)
            def reset_task(self,*a,**kw):
                assert threading.get_ident()==owner
                self.resets+=1;self.episode_count+=1
            def step(self):
                assert threading.get_ident()==owner
                self.attempt+=1
                if fail_slot==slot:raise RuntimeError('injected worker failure')
                value=torch.tensor([[slot*100+self.decision]],dtype=torch.float32)
                result=policy.sample_rollout(dict(state=value),generator=torch.Generator().manual_seed(slot*10000+self.attempt))
                assert torch.equal(result['normalized'],value)
                self.decision+=1;self.snapshot['tick']+=300
                return NS(identity=dict(policy_version=self.policy_version,backend_session_id=self.backend.session_id),
                    metadata={},transition_valid=True,executed_control_steps=25,terminated=False,truncated=False,reason=None)
        r=NS(env=Env(),sampler=Sampler(),close=lambda:None)
        resources.append(r)
        return r
    return create


def test_two_real_requests_distinct_noise_and_persistent_episode():
    resources=[]
    c=DualEnvironmentCollector(Policy(),factory(resources),batch_wait_seconds=.05,timeout_seconds=5)
    try:
        for version in (0,1):
            fragments,report=c.collect(count_per_rank=20,policy_version=version)
            assert report['fragment_lengths']==[10,10]
            assert report['total_transitions']==20
            assert any(b['effective_rows']==2 for b in report['batches'])
            assert all(b['padding_rows']==0 for b in report['batches'])
            assert all(f[-1].truncated for f in fragments)
        assert [r.env.resets for r in resources]==[1,1]
        assert [r.sampler.steps for r in resources]==[500,500]
        saved=c.state_dict()
    finally:c.close()
    restored=[]
    d=DualEnvironmentCollector(Policy(),factory(restored),batch_wait_seconds=.05,timeout_seconds=5)
    try:
        boundaries=d.load_state_dict(saved)
        assert all(not b['budget_refunded'] for b in boundaries)
        assert [r.sampler.steps for r in restored]==[500,500]
        d.collect(count_per_rank=20,policy_version=2)
        assert [r.env.attempt for r in restored]==[30,30]
    finally:d.close()


def test_worker_failure_unblocks_peer_without_optimizer():
    c=DualEnvironmentCollector(Policy(),factory([],fail_slot=1),timeout_seconds=2)
    try:
        with pytest.raises(RuntimeError,match='injected'):
            c.collect(policy_version=0)
    finally:c.close()


def test_two_generators_actual_actor_matches_independent_paths(actor_factory):
    torch.set_num_threads(1)
    actor,batch=actor_factory(starts=(45,47));_activate_branches(actor)
    policy=DPPODiffusionPolicy(actor,cfg_batch=True,numerical_layout='sample_matrix_bmm_fp32.v1')
    context=_conditions(batch)
    combined=policy.sample_rollout(context,generator=[torch.Generator().manual_seed(s) for s in (181,182)])
    for i,seed in enumerate((181,182)):
        single=policy.sample_rollout({k:v[i:i+1] for k,v in context.items()},generator=torch.Generator().manual_seed(seed))
        actual=split_trace(combined,i,2)
        torch.testing.assert_close(actual['chain'],single['chain'],atol=0,rtol=0)
        torch.testing.assert_close(actual['old_log_probs'],single['old_log_probs'],atol=1e-8,rtol=0)
