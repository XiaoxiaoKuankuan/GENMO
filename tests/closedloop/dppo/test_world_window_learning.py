"""统一世界窗口的变长与空片段学习边界验收。

使用真实Buffer/GAE/cache接口验证空环境与空rank不构造伪动作，不调用Critic，
不跨Actor版本拼接优势；固定世界时间的配置没有每环境配额，BC总量可以按实际
optimizer尾批分配而无需整除世界规模。本测试不替代真实PhysX和八卡梯度验收。
"""
from pathlib import Path
import pytest
import torch
import yaml
from gem.closedloop.dppo.memory_execution import DeviceRolloutBuffer
from gem.closedloop.dppo.rollout_helpers import fixed_fragment_targets
from gem.closedloop.dppo.tensor_cache import RolloutTensorCache
from gem.closedloop.dppo.training_config import configuration


def test_empty_local_world_buffer_has_no_virtual_transition_or_critic_call():
    class Critic:
        def __call__(self,*args):raise AssertionError('empty rank must not call Critic')
    buffer=DeviceRolloutBuffer([[],[]],None,'cpu')
    rows,targets=fixed_fragment_targets([[],[]],Critic(),'cpu',critic_version=3)
    assert len(buffer)==len(rows)==0 and targets['returns'].shape==(0,)
    cache=RolloutTensorCache(rows,targets,'cpu')
    assert cache.report()['bytes']==0 and cache.report()['upper_transitions']==0
    cache.close();buffer.clear()


def test_current_config_accepts_nondivisible_bc_total_without_environment_quota(tmp_path):
    root=Path(__file__).resolve().parents[3]
    config=yaml.safe_load((root/'configs/closedloop/stage2_ddim10_last5_motion30_music30_klrelaxed_server2.yaml').read_text())
    config['stage10']['training']['bc_enabled']=True
    config['stage10']['training']['bc_samples_per_rollout']=517
    # 单个优化步确保517不被每步BC下限扩成多个批次；检验非整除配额而非关闭BC。
    config['stage10']['scale']['actor_minibatch_chains']=81920
    config['stage10']['kl']['calibration']['optimization_sha256']='auto'
    path=tmp_path/'candidate.yaml';path.write_text(yaml.safe_dump(config))
    resolved=configuration(path)
    assert 'decisions_per_environment' not in resolved['stage10']['scale']
    assert resolved['stage10']['training']['bc_samples_per_rollout']==517
    config['stage10']['scale']['decisions_per_environment']=8
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError,match='quotas'):configuration(path)



def test_reset_warmup_seeds_only_final_actual_pd_target_across_world_windows():
    from types import SimpleNamespace
    import numpy as np
    from gem.closedloop.dppo.env_adapter import UpperEnvironment
    from gem.closedloop.dppo.rewards import ExecutionReward
    env=object.__new__(UpperEnvironment)
    env.snapshot=dict(tick=0,done=False);env.rollout_window_reached=False;env._warmup_last_target=None
    env.reward=ExecutionReward(dict(music_reward_enabled=False,music_weight=0.),None)
    def advance(count):
        env.snapshot['tick']+=12*count
        env.rollout_window_reached=env.snapshot['tick']==300
        if False:yield
        return [dict(joint_position_target=np.full(21,env.snapshot['tick']))]
    env.advance_flow=advance
    flow=env.continue_warmup_flow()
    with pytest.raises(StopIteration):next(flow)
    assert env.snapshot['tick']==300 and env.reward._previous_target is None
    env.rollout_window_reached=False
    with pytest.raises(StopIteration):next(env.continue_warmup_flow())
    np.testing.assert_array_equal(env.reward._previous_target,np.full(21,600))
    assert env.reward._last_tick is None


def test_current_world_restore_preserves_counters_and_sampler_but_starts_new_physics():
    from types import SimpleNamespace as NS
    from gem.closedloop.dppo.world_collector import WorldEnvironmentCollector
    class Sampler:
        def __init__(self):self.cursor=0
        def state_dict(self):return {'cursor':self.cursor}
        def load_state_dict(self,value):self.cursor=value['cursor']
    def factory(slot,policy):
        env=NS(policy_version=0,iteration=0,decision=0,attempt=0,episode_count=0,
            latency_budget_s=.32,snapshot={'episode_id':'new','tick':600},config={'stage9':{'seed':42+slot}})
        return NS(env=env,sampler=Sampler(),budget=None)
    policy=NS(actor=torch.nn.Linear(2,1),steps=20,guidance_scale=2.5,numerical_layout='native_dense')
    collector=WorldEnvironmentCollector(policy,factory,object(),num_envs=2,generation_batch=2)
    resource=factory(0,None);resource.env.budget=NS(state_dict=lambda:{'used':{'control_steps':25}})
    resource.env.decision=8;resource.env.attempt=9;resource.env.episode_count=2
    resource.env.policy_version=7;resource.env.iteration=7;resource.env.snapshot={'episode_id':'old','tick':3000}
    resource.sampler.cursor=91
    collector.states[0]=NS(resource=resource,task={'old_physical_episode':True},transitions=17)
    saved=collector.state_dict()
    assert saved['schema']=='genmo.gpu_vector_collector.boundary.v2'
    assert saved['states'][1] is None
    restored=WorldEnvironmentCollector(policy,factory,object(),num_envs=2,generation_batch=2)
    restored.load_state_dict(saved)
    state=restored.states[0]
    assert state.task is None and state.transitions==17 and state.resource.sampler.cursor==91
    assert (state.resource.env.decision,state.resource.env.attempt,state.resource.env.policy_version)==(8,9,7)
    assert state.resource.env.snapshot=={'episode_id':'new','tick':600}
    assert restored.states[1] is None
    saved['schema']='genmo.gpu_vector_collector.boundary.v1'
    empty=WorldEnvironmentCollector(policy,lambda *args:pytest.fail('old identity must fail before constructing resources'),
        object(),num_envs=2,generation_batch=2)
    with pytest.raises(ValueError,match='contract'):empty.load_state_dict(saved)


def test_root_budget_rejects_legacy_json_instead_of_implicitly_migrating(tmp_path):
    import json
    from gem.closedloop.dppo.budget_ledger import IncrementalBudget
    path=tmp_path/'budget.json'
    previous={'schema':'genmo.closedloop.stage10.budget.v1','used':{'control_steps':17}}
    path.write_text(json.dumps(previous))
    limits=dict(accepted_iterations=1,optimizer_attempts=1,generations=20,control_steps=100,physics_steps=400)
    with pytest.raises(ValueError,match='current SQLite'):IncrementalBudget(path,limits)
    assert json.loads(path.read_text())==previous and not path.with_suffix('.sqlite').exists()
