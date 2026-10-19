"""按累计 PPO 轮数停止、严格恢复身份与局部正常终止的回归验收。

测试直接调用正式配置解析、checkpoint 加载、GAE 和世界环境续体；只使用临时
小模型与可计数协议替身，不启动 Isaac，不读取或修改正式权重。验证删除总时限
后本分支4轮有限上限、资源限制及固定随机核仍生效，旧身份在模型改变前拒绝，非空 Adam
和 BC 控制器可以恢复。64 个环境中一个在站姿启动时物理终止，只重置该环境；
它不生成虚假 PPO 转移，其他环境保持有效，真实执行完整性错误不被吞掉。
"""
from pathlib import Path
from types import SimpleNamespace as NS
import copy
import pytest
import torch
import yaml

from gem.closedloop.dppo.training_config import configuration
from gem.closedloop.dppo.run_control import validate_stopping_iteration
from gem.closedloop.dppo.training_checkpoint import SCHEMA, load_training_checkpoint
from gem.closedloop.dppo.gradient_merge import BCGradientController
from gem.closedloop.dppo.returns import compute_gae, normalize_advantages_global
from gem.closedloop.dppo.env_adapter import WarmupEpisodeEnded, ExecutionIntegrityError
from gem.closedloop.dppo.world_collector import WorldEnvironmentCollector

CONFIG=Path(__file__).resolve().parents[3]/'configs/closedloop/stage2_credit_server2.yaml'


def test_production_parameters_and_cumulative_iteration_ceiling():
    config=configuration(CONFIG); stage=config['stage10']; train=stage['training']
    assert train['actor_lr']==3e-7 and train['critic_lr']==1e-4
    assert (train['eta'],train['ppo_steps_per_chain'],train['ppo_clip'])==(.1,4,.01)
    assert train['std_schedule']==[.001]*15+[.04,.06,.08,.10,0.]
    assert stage['bc_gradient']['target_ratio']==.1 and stage['kl']['mean_coordinate']==.0009
    assert stage['run_control']=={'max_training_iterations':200}
    assert validate_stopping_iteration(200,stage)==200
    for wrong in (0,201,True,4.0):
        with pytest.raises(ValueError):validate_stopping_iteration(wrong,stage)
    assert train['kl_rejection']=='abort'
    assert stage['limits']==dict(accepted_iterations=2000,optimizer_attempts=18000,
        generations=40000000,control_steps=1100000000,physics_steps=4400000000)


@pytest.mark.parametrize('section,key,value',[
    ('run_control','max_walltime_seconds',172800),
    ('run_control','max_training_iterations',2001),
    ('runtime','vector_boundary_wait_contract','bounded_reference_wait.v1'),
])
def test_retired_or_over_budget_settings_are_rejected(tmp_path,section,key,value):
    config=yaml.safe_load(CONFIG.read_text())
    target=config['runtime'] if section=='runtime' else config['stage10'][section]
    target[key]=value
    path=tmp_path/'invalid.yaml';path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):configuration(path)


def test_gae_is_raw_by_default_and_normalized_once_after_global_collection():
    targets=compute_gae([[1.],[3.]],[0.,0.],[0.,0.],[1,1],[False,False],[False,False])
    assert not targets['advantages_normalized']
    torch.testing.assert_close(targets['advantages'],torch.tensor([1.,3.],dtype=torch.float64))
    normalized=normalize_advantages_global(targets)
    torch.testing.assert_close(normalized['advantages'],torch.tensor([-1.,1.],dtype=torch.float64))
    torch.testing.assert_close(normalized['returns'],targets['returns'])


def test_world_collector_does_not_silently_accept_retired_boundary_wait():
    with pytest.raises(TypeError,match='boundary_wait'):
        WorldEnvironmentCollector(None,None,None,boundary_wait_max_control_steps=100)


def test_new_checkpoint_retains_adam_bc_progress_and_rejects_old_schema(tmp_path):
    actor=torch.nn.Linear(2,1);critic=torch.nn.Linear(2,1)
    actor_optimizer=torch.optim.AdamW(actor.parameters(),lr=1e-7)
    critic_optimizer=torch.optim.AdamW(critic.parameters(),lr=1e-4)
    for model,optimizer in ((actor,actor_optimizer),(critic,critic_optimizer)):
        model(torch.ones(2,2)).sum().backward();optimizer.step();optimizer.zero_grad()
    controller=BCGradientController();controller.update(2.,3.)
    identity={'run_control':{'max_training_iterations':500}}
    payload=copy.deepcopy(dict(schema=SCHEMA,identity=identity,actor=actor.state_dict(),critic=critic.state_dict(),
        actor_optimizer=actor_optimizer.state_dict(),critic_optimizer=critic_optimizer.state_dict(),
        bc_controller=controller.state_dict(),state={'iteration':17,'actor_updates':102},
        rank_states=[{'rng':torch.get_rng_state().clone(),'bc':{'samples_seen':1024}}]))
    path=tmp_path/'checkpoint.pt';torch.save(payload,path)
    with torch.no_grad():actor.weight.add_(5.)
    actor_optimizer.state.clear();controller.value=.5
    restored=load_training_checkpoint(path,identity=identity,actor=actor,critic=critic,
        actor_optimizer=actor_optimizer,critic_optimizer=critic_optimizer,controller=controller)
    assert restored['state']['iteration']==17 and restored['rank_states'][0]['bc']['samples_seen']==1024
    assert actor_optimizer.state and controller.state_dict()==payload['bc_controller']
    torch.testing.assert_close(actor.weight,payload['actor']['weight'],rtol=0,atol=0)
    payload['schema']='genmo.closedloop.stage10.continuous_horizon.checkpoint.v1'
    payload['actor']['weight'].add_(4.);torch.save(payload,path)
    before=actor.weight.detach().clone()
    with pytest.raises(ValueError,match='schema'):
        load_training_checkpoint(path,identity=identity,actor=actor,critic=critic,
            actor_optimizer=actor_optimizer,critic_optimizer=critic_optimizer,controller=controller)
    torch.testing.assert_close(actor.weight,before,rtol=0,atol=0)


def test_one_startup_failure_does_not_invalidate_other_63_environments():
    collector=object.__new__(WorldEnvironmentCollector);collector.world_timing={};collector.states=[]
    task={'sample':{'dataset':'test','row':{'split':'train','sample_id':'sample'},'manifest_sha256':'fixed'},
        'music':None,'music_start_frame':0}
    class Env:
        def __init__(self,slot):
            self.slot=slot;self.resets=0;self.rollout_window_reached=False
            self.config={'stage9':{'seed':42}};self.episode_count=0;self.snapshot={'tick':600}
        def reset_task_flow(self,*args,**kwargs):
            if False:yield
            self.resets+=1;self.episode_count+=1
            if self.slot==0 and self.resets==1:raise WarmupEpisodeEnded('normal physical failure')
        def step_flow(self):
            if False:yield
            self.rollout_window_reached=True
            return NS(transition_valid=True,identity={'policy_version':3},metadata={},
                executed_control_steps=25,terminated=False,truncated=False,reason=None)
    for slot in range(64):
        env=Env(slot);sampler=NS(next_task=lambda:copy.deepcopy(task),record_execution=lambda *args:None)
        collector.states.append(NS(resource=NS(env=env,sampler=sampler),task=None,transitions=0))
    for slot in range(64):
        flow=collector._window_flow(slot,3)
        assert next(flow).method=='join_fragment'
        with pytest.raises(StopIteration) as finished:flow.send(None)
        assert len(finished.value.value)==1 and finished.value.value[0].transition_valid
    assert collector.world_timing['warmup_physical_failures']==1
    assert [s.resource.env.resets for s in collector.states]==[2]+[1]*63


def test_world_flow_does_not_swallow_incomplete_execution():
    collector=object.__new__(WorldEnvironmentCollector);collector.world_timing={}
    def reset(*args,**kwargs):
        if False:yield
        raise ExecutionIntegrityError('missing physical substep')
    env=NS(policy_version=0,rollout_window_reached=False,config={'stage9':{'seed':1}},episode_count=0,reset_task_flow=reset)
    sampler=NS(next_task=lambda:{'sample':{'row':{'split':'train'}},'music':None,'music_start_frame':0})
    collector.states=[NS(resource=NS(env=env,sampler=sampler),task=None)]
    flow=collector._window_flow(0,0);next(flow)
    with pytest.raises(ExecutionIntegrityError):flow.send(None)
