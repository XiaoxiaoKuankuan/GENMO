"""第十步恢复执行计数、显式配置身份及受控预算扩展的 CPU 生命周期回归。

复用已有小网络/虚拟物理worker夹具，让真实入口执行采集、优化、checkpoint发布与
恢复。噪声一致性检查调用真实 UpperEnvironment.generate 和显式 torch.Generator，
只用小型采样替身替代网络推理，不复制噪声种子的实现作为期望值。另验证校准后的
decision进入initial及逐轮状态，缺字段旧状态在worker启动前拒绝，使用另一配置路径
改变seed/时序/运行时也不能绕过身份校验。预算和存储调整保持模型契约不变，但扩限
必须显式说明原因并保留旧消耗。所有文件使用pytest临时目录，不连接服务器或GPU。
"""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

import test_stage10_lifecycle as fixtures
from gem.closedloop.dppo.env_adapter import UpperEnvironment
from tools import train_closedloop_stage10 as entry


lifecycle = fixtures.lifecycle


def probe_real_generation(config, state, output):
    """只替换生成内容和backend回复，实际请求、计数、显式seed与采样generator走原代码。"""
    actor = fixtures.TinyActor()
    actor.endecoder.codec = SimpleNamespace(apply_world_anchor=lambda value, anchor:value)
    def sample(batch, *, generator):
        return dict(qpos=torch.zeros(1,120,28), qpos30=torch.zeros(1,120,30),
                    contact=torch.zeros(1,120,2), actual_noise=torch.randn(16,generator=generator))
    policy = SimpleNamespace(actor=actor, sample_rollout=sample)
    def call(method, **kwargs):
        assert method in ('reserve_prefix','prepare_plan')
        return dict(prepared_plan_id='synthetic-no-physics')
    builder = SimpleNamespace(build=lambda *a,**k:({},dict(world_anchor=np.zeros(7))))
    budget = SimpleNamespace(reserve=lambda *a,**k:None)
    env = UpperEnvironment(config,SimpleNamespace(call=call),builder,policy,budget,output)
    entry.restore_execution_state(env,state)
    env.snapshot = dict(env_id=0,episode_id='new-physical-session',tick=600)
    env.music = np.zeros((120,35),dtype=np.float32)
    result = env.generate()
    return result['seed'],result['trace']['actual_noise']


def test_resumed_next_actual_generation_has_same_seed_and_noise(lifecycle,tmp_path,monkeypatch):
    observed=[]
    original=fixtures.SyntheticEnvironment.step
    def step(env):
        state=dict(iteration=env.iteration,policy_version=env.policy_version,**entry.capture_execution_state(env))
        observed.append(probe_real_generation(env.config,state,tmp_path/f'noise_{len(observed)}'))
        return original(env)
    monkeypatch.setattr(fixtures.SyntheticEnvironment,'step',step)
    output=tmp_path/'run'
    assert fixtures.run(lifecycle,output,stop=1)==0
    checkpoint,_=fixtures.latest(output)
    assert checkpoint['state']['decision']==2
    uninterrupted=copy.deepcopy(checkpoint['state'])
    uninterrupted['episode_count']+=1  # 下一任务的正常reset；物理session身份不进入显式噪声key。
    expected=probe_real_generation(lifecycle.environments[-1].config,uninterrupted,tmp_path/'expected')
    assert fixtures.run(lifecycle,output,stop=2,resume='latest')==0
    assert observed[2][0]==expected[0]
    torch.testing.assert_close(observed[2][1],expected[1],rtol=0,atol=0)
    resumed,_=fixtures.latest(output)
    assert resumed['state']['decision']==resumed['state']['attempt']==4
    assert lifecycle.environments[-1].decision==4


def test_post_calibration_initial_and_iteration_states_keep_all_execution_counters(lifecycle,tmp_path,monkeypatch):
    def calibrate(env,*args):
        env.budget.reserve('calibration',generations=3)
        env.decision=env.attempt=3
        env.episode_count=1
        env.latency_budget_s=.24
        return dict(synthetic=True,latency_budget_s=.24)
    monkeypatch.setattr(entry,'calibrate',calibrate)
    output=tmp_path/'run'
    assert fixtures.run(lifecycle,output,stop=1)==0
    initial=torch.load(output/'checkpoints/initial.pt',map_location='cpu',weights_only=False)
    assert {key:initial['state'][key] for key in ('decision','attempt','episode_count','latency_budget_s')} == dict(
        decision=3,attempt=3,episode_count=1,latency_budget_s=.24)
    first,_=fixtures.latest(output)
    assert first['state']['decision']==first['state']['attempt']==5
    assert fixtures.run(lifecycle,output,stop=2,resume='latest')==0
    second,_=fixtures.latest(output)
    assert second['state']['decision']==second['state']['attempt']==7
    assert second['state']['episode_count']==3 and second['state']['latency_budget_s']==.24


def test_legacy_checkpoint_without_decision_refuses_before_worker(lifecycle,tmp_path):
    output=tmp_path/'run'
    lifecycle.failure_on_actor_call=1
    assert fixtures.run(lifecycle,output,stop=1)==1
    path=output/'checkpoints/initial.pt'
    payload=torch.load(path,weights_only=False,map_location='cpu')
    del payload['state']['decision']
    torch.save(payload,path)
    starts=lifecycle.worker_starts
    lifecycle.failure_on_actor_call=None
    assert fixtures.run(lifecycle,output,stop=1,resume=path)==1
    assert lifecycle.worker_starts==starts and not (output/'latest.json').exists()
    summaries=[json.loads(p.read_text()) for p in output.glob('sessions/*/summary.json')]
    assert any('execution state: decision' in item.get('error',{}).get('message','') for item in summaries)


@pytest.mark.parametrize('keys,value',[(('stage10','seed'),43), (('timing','latency_guard_s'),.08),
    (('runtime','torch_threads'),2),(('runtime','physics_device'),'different-device'),
    (('runtime','headless'),False),(('model','history_steps'),49),
    (('diagnostics','record_original_threshold_crossings'),False),
    (('paths','isaac_python'),'/different/isaac/python')])
def test_alternate_config_seed_and_execution_changes_refuse_before_worker(lifecycle,tmp_path,keys,value):
    output=tmp_path/'run'
    assert fixtures.run(lifecycle,output,stop=1)==0
    previous=(output/'latest.json').read_bytes()
    config=copy.deepcopy(lifecycle.config)
    config[keys[0]][keys[1]]=value
    alternate=tmp_path/'different_location'/'formal.yaml'
    alternate.parent.mkdir()
    alternate.write_text(yaml.safe_dump(config))
    lifecycle.config_path=alternate
    assert fixtures.run(lifecycle,output,stop=2,resume='latest')==1
    assert lifecycle.worker_starts==1 and (output/'latest.json').read_bytes()==previous


def test_capacity_config_changes_require_explicit_extension_and_preserve_state(lifecycle,tmp_path):
    output=tmp_path/'run'
    assert fixtures.run(lifecycle,output,stop=1)==0
    before,_=fixtures.latest(output)
    config=copy.deepcopy(lifecycle.config)
    config['stage10']['limits'].update(accepted_iterations=6,optimizer_attempts=18,generations=120,
                                       control_steps=240,physics_steps=960)
    config['stage10']['storage']['max_run_bytes']*=2
    path=tmp_path/'formal.yaml';path.write_text(yaml.safe_dump(config))
    lifecycle.config_path=path
    with pytest.raises(ValueError,match='budget'):
        fixtures.run(lifecycle,output,stop=2,resume='latest')
    assert lifecycle.worker_starts==1
    assert entry.main(['--config',str(path),'--mode','train','--output-dir',str(output),'--resume','latest',
        '--stop-after-iteration','2','--extend-budget-reason','有限正式预算扩展'])==0
    after,_=fixtures.latest(output)
    assert after['identity']==before['identity']
    assert after['state']['decision']==4 and after['state']['actor_updates']==2
    assert after['state']['budget']['used']['accepted_iterations']==2
    assert after['state']['budget']['limits']['accepted_iterations']==6
    assert len(after['state']['budget']['limit_extensions'])==1
    report=next(json.loads(p.read_text()) for p in output.glob('sessions/*/summary.json')
                if json.loads(p.read_text()).get('budget_extension'))
    assert report['input_config_sha256']==entry.sha256_file(path)
    resolved=next(p for p in output.glob('sessions/*/resolved_config.yaml')
                  if entry.sha256_file(p)==report['resolved_config_sha256'])
    assert resolved.is_file() and report['budget_extension']


def test_exact_candidate_budget_check_stops_before_new_collection(lifecycle,tmp_path):
    config=copy.deepcopy(lifecycle.config)
    config['stage10']['limits']['optimizer_attempts']=4
    config['stage10']['training']['actor_lr_candidates']=[5e-7,1e-6]
    lifecycle.config_path.write_text(yaml.safe_dump(config))
    output=tmp_path/'run'
    assert fixtures.run(lifecycle,output,stop=4)==0
    checkpoint,_=fixtures.latest(output)
    # 夹具每轮只消费一次optimizer尝试；剩余1次不够真实两候选计划，不能再采一轮。
    assert checkpoint['state']['iteration']==3 and lifecycle.actor_calls==3
    report=json.loads(next(output.glob('sessions/*/summary.json')).read_text())
    assert report['stop_reason']=='budget_exhausted'
    assert report['budget_stop_details']['exhausted']['optimizer_attempts']['remaining']==1
    assert len(list(output.glob('sessions/*/iterations/*')))==3


def test_failed_generation_spend_is_not_reused_but_decision_is_restored():
    env=SimpleNamespace()
    state=dict(decision=7,attempt=9,episode_count=2,latency_budget_s=.3,iteration=1,policy_version=1)
    entry.restore_execution_state(env,state,spent_generations=12)
    assert env.decision==7 and env.attempt==12 and env.episode_count==2


@pytest.mark.parametrize('value',[None,True,-1,1.5])
def test_invalid_explicit_decision_never_becomes_zero(value):
    state=dict(decision=value,attempt=2,episode_count=1,latency_budget_s=.3)
    with pytest.raises(ValueError,match='decision'):
        entry.validate_execution_state(state)
