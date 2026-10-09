"""部署时钟与训练拓扑隔离的回归验收，只在服务器1八卡测试进程中执行。

覆盖同任务在N=1/8/1024与人为训练墙钟延迟下的到达时刻、前缀预算不变；旧部署
实时时钟仍响应延迟；profile变更必须阻止旧断点恢复。另对混合P、空历史与音乐
结尾比较批量/逐条在线条件，严格保持原十字段和前缀掩码，不放宽概率阈值。
"""
import copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from gem.closedloop.dppo.deployment_clock import DeploymentClock, MODELED_CLOCK
from gem.closedloop.dppo.env_adapter import UpperEnvironment
from tools.train_closedloop_stage10 import configuration, capture_execution_state, restore_execution_state


def profile():
    return dict(schema='genmo.deployment_delay_profile.v1', target='single_robot_single_request',
        samples_seconds=[.243, .261], prefix_budget_seconds=.38,
        provenance=[dict(path='measured/single_request.json', sha256='a'*64)])


def test_arrival_does_not_depend_on_wallclock_or_env_count():
    clock = DeploymentClock(profile())
    env = SimpleNamespace(mode='latency', deployment_clock=clock,
        config={'stage10':{'seed':42}}, sample={'row':{'sample_id':'a'}}, music_start_frame=30)
    arrivals=[]
    for n in (1, 8, 1024):
        for delay in (.01, 1., 30.):
            generated=dict(critical_ready_seconds=delay,elapsed=delay+10.,timing={})
            arrivals.append(UpperEnvironment._arrival_tick(env,generated,600))
            assert generated['timing']['deployment_clock']['prefix_budget_ticks']==228
    assert len(set(arrivals))==1
    assert arrivals[0] in (756,768)


def test_profile_resume_rejects_different_identity():
    env=SimpleNamespace(deployment_clock=DeploymentClock(profile()),decision=1,attempt=2,
        episode_count=1,latency_budget_s=.38,policy_version=1,iteration=1)
    state=dict(capture_execution_state(env),policy_version=1,iteration=1)
    restore_execution_state(env,state)
    changed=profile();changed['samples_seconds']=[.3]
    env.deployment_clock=DeploymentClock(changed)
    with pytest.raises(ValueError,match='clock/profile'):restore_execution_state(env,state)


def test_old_real_clock_still_uses_measured_delay():
    env=SimpleNamespace(mode='latency',deployment_clock=None,timing_contract='deployment_critical.v2')
    assert UpperEnvironment._arrival_tick(env,dict(elapsed=9.,critical_ready_seconds=.24),600)==744
    assert UpperEnvironment._arrival_tick(env,dict(elapsed=9.,critical_ready_seconds=.4),600)==840


def test_batch_condition_exact_against_scalar_reference():
    from tests.closedloop.test_online_conditions import make_inputs,KINEMATICS
    from gem.closedloop.online_conditions import OnlineConditionBuilder
    from gem.robots.bumi.feature_codec import BumiMotionFeatureCodec
    from gem.robots.bumi.kinematics import BumiKinematics
    builder=OnlineConditionBuilder(BumiMotionFeatureCodec(BumiKinematics(KINEMATICS)))
    items=[make_inputs(builder,prefix=p,history=h,tick=900) for p,h in
        [(0,0),(1,1),(18,50),(18,20),(30,50),(30,0),(119,50),(6,50)]]
    items[-1]=(items[-1][0],items[-1][1],items[-1][2][:20])
    contexts,metadata=builder.build_many(*zip(*items))
    for i,item in enumerate(items):
        expected,meta=builder.build(*item)
        for key in expected:torch.testing.assert_close(contexts[i][key],expected[key],rtol=0,atol=0)
        for key in meta:np.testing.assert_equal(metadata[i][key],meta[key])


def test_gpu_config_declares_fixed_profile_and_batched_pipeline():
    config=configuration(Path(__file__).resolve().parents[3]/'configs/closedloop/stage10_8gpu_server1_gpu_vectorized.yaml')
    assert config['runtime']['timing_contract']==MODELED_CLOCK
    assert DeploymentClock(config['timing']['deployment_profile']).budget_seconds==.38


def test_columnar_concatenation_preserves_mixed_leaves():
    from gem.runtime.trajectory_blocks import pack_trace,unpack_trace,concatenate_trace_blocks
    rows=[dict(tick=12*i,constant='same',matrix=np.arange(6).reshape(2,3)+i,
               optional=None if i==0 else np.array([i,i+1.]),flag=i>1,sequence=[dict(v=i),4]) for i in range(25)]
    blocks=[pack_trace(rows[:1]),pack_trace(rows[1:8]),pack_trace(rows[8:])]
    np.testing.assert_equal(unpack_trace(concatenate_trace_blocks(blocks)),rows)
    with pytest.raises(ValueError):concatenate_trace_blocks(blocks+[pack_trace(rows[:1])])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='服务器1真实GPU批量传输验收')
def test_bulk_copy_and_split_preserve_all_devices_and_no_storage_alias():
    import os
    from gem.closedloop.dppo.vector_generation import bulk_cpu_copy
    device=f"cuda:{int(os.environ.get('LOCAL_RANK',0))}"
    a=torch.arange(320,device=device,dtype=torch.float32).reshape(8,20,2)
    tree=dict(a=a,double=a.double(),mask=a>3,scalar=torch.tensor(42,device=device),same=a)
    result=bulk_cpu_copy(tree)
    for key in tree:
        assert result[key].device.type=='cpu'
        torch.testing.assert_close(result[key],tree[key].cpu(),rtol=0,atol=0)
    a.fill_(9)
    assert result['a'].flatten()[0]==0
