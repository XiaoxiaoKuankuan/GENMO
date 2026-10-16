"""随机音乐时间、配对运动初态及共享覆盖的因果回归测试。

所有清单和动作写入pytest临时目录；不读取正式模型、不启动物理或训练。测试证明
不同rank的时间bin不重复、同rank环境共享进度、恢复保持采样序列，以及未执行区间
不会被统计成覆盖。用解析运动检查音乐偏移、实际预热所需参考和水平坐标变换，
用既有环境替身核对训练reset数据流和评估隔离；这些不替代真实GPU物理验收。
"""
from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest

from gem.closedloop.dppo.full_dataset import FullMusicCatalog, FullMusicSampler, SOURCES
from gem.closedloop.dppo.motion_initialization import make_motion_initialization
from gem.closedloop.dppo.target_activity import load_paired_activity
from gem.closedloop.dppo.time_sampling import StratifiedTimePool, MOTION_RESET_CONTRACT, execution_coverage
from tests.closedloop.dppo.test_full_dataset import make_catalog_data


def pool(catalog, *, rank=0, world_size=1, standing=0):
    return StratifiedTimePool(catalog, seed=42, rank=rank, world_size=world_size,
        minimum_remaining_seconds=4., bin_seconds=1., standing_start_probability=standing)


def test_long_motion_later_positions_are_drawn_without_first_completing_the_prefix():
    catalog=SimpleNamespace(identity={'test':'251.4s'},samples={'train':{
        source:[{'row':{'num_frames':7542}}] for source in SOURCES}})
    ranks=[pool(catalog,rank=i,world_size=8) for i in range(8)]
    observed=[];starts=[]
    for sampler in ranks:
        for _ in sampler.shards['FineDance']:
            _,start,metadata=sampler.draw('FineDance')
            starts.append(start);observed.append(metadata['time_bin']['bin_id'])
            assert metadata['initialization']==MOTION_RESET_CONTRACT
    assert len(observed)==len(set(observed))==len(ranks[0].bins['FineDance'])
    assert min(starts)<30 and max(starts)>=246*30
    # 没有调用record_execution，后段已经能直接抽到，但不能称为执行覆盖。
    assert all(s.summary()['scope']=='issued_start_bins_not_executed_coverage' for s in ranks)


def test_lanes_share_pool_and_restore_exact_next_tasks(tmp_path):
    catalog=FullMusicCatalog(make_catalog_data(tmp_path,frames=300))
    shared=pool(catalog)
    lanes=[FullMusicSampler(catalog,seed=i,minimum_remaining_seconds=4,time_pool=shared) for i in range(2)]
    for i in range(12):
        task=lanes[i%2].next_task()
        lanes[i%2].record_execution(task,25)
    pool_state=shared.state_dict();lane_states=[s.state_dict() for s in lanes]
    expected=[lanes[i%2].next_task() for i in range(20)]
    shared.load_state_dict(pool_state)
    for sampler,state in zip(lanes,lane_states):sampler.load_state_dict(state)
    actual=[lanes[i%2].next_task() for i in range(20)]
    fields=('sample','music_start_frame','sampling','task_index')
    assert [{k:t[k] for k in fields} for t in actual]==[{k:t[k] for k in fields} for t in expected]
    assert len({(t['sample']['dataset'],t['sampling']['time_bin']['epoch'],
                 t['sampling']['time_bin']['bin_id']) for t in actual})==len(actual)


@pytest.mark.parametrize('fault',['identity','order','cursor','count','rng'])
def test_corrupt_pool_state_is_rejected_atomically(tmp_path,fault):
    sampler=pool(FullMusicCatalog(make_catalog_data(tmp_path)))
    sampler.draw('FineDance');before=sampler.state_dict();state=copy.deepcopy(before)
    if fault=='identity':state['identity']['world_size']=2
    elif fault=='order':state['orders']['Mine'][0]=-1
    elif fault=='cursor':state['cursors']['Mine']=-1
    elif fault=='count':state['motion_draws']+=1
    else:state['rng']={}
    with pytest.raises((ValueError,KeyError,TypeError)):sampler.load_state_dict(state)
    assert sampler.state_dict()==before


def test_standing_mix_retains_startup_without_consuming_time_bins(tmp_path):
    sampler=pool(FullMusicCatalog(make_catalog_data(tmp_path)),standing=.2)
    tasks=[sampler.draw('Mine') for _ in range(1000)]
    standing=[t for t in tasks if t[2]['initialization']=='standing']
    assert 150<len(standing)<250 and all(t[1]==0 and t[2]['time_bin'] is None for t in standing)
    assert sampler.standing_draws+sampler.motion_draws==1000
    sampler.load_state_dict(sampler.state_dict())


def test_execution_coverage_merges_actual_intervals_and_ignores_draw_only(tmp_path):
    catalog=FullMusicCatalog(make_catalog_data(tmp_path))
    lanes=[FullMusicSampler(catalog,seed=7,random_start=False) for _ in range(2)]
    a,b=[s.next_task() for s in lanes]
    assert a['sample']==b['sample']
    lanes[0].record_execution(a,25);lanes[1].record_execution(b,50)
    report=execution_coverage(catalog,lanes)['per_source'][a['sample']['dataset']]
    assert report['control_steps']==75 and report['covered_seconds']==1.0
    assert report['unique_executed_samples']==1 and report['fully_covered_samples']==0
    empty=FullMusicSampler(catalog);empty.next_task()
    assert all(v['covered_seconds']==0 for v in execution_coverage(catalog,[empty])['per_source'].values())


def test_coverage_denominator_uses_full_source_duration_on_different_control_phases(tmp_path):
    catalog=FullMusicCatalog(make_catalog_data(tmp_path,frames=301))
    sampler=FullMusicSampler(catalog,random_start=False)
    task=sampler.next_task()
    sampler.record_execution(task,301*50//30)
    report=execution_coverage(catalog,[sampler])['per_source'][task['sample']['dataset']]
    count=len(catalog.samples['train'][task['sample']['dataset']])
    assert report['source_seconds']==pytest.approx(count*301/30)
    assert report['covered_seconds']==10.02 and report['fully_covered_samples']==1
    assert report['unique_time_coverage']<=1


def test_runtime_checkpoint_restores_shared_cursor_once_and_rejects_old_schema(tmp_path):
    from gem.closedloop.dppo.runtime import InProcessTrainingRuntime
    catalog=FullMusicCatalog(make_catalog_data(tmp_path,frames=300))
    def runtime():
        result=InProcessTrainingRuntime.__new__(InProcessTrainingRuntime)
        result.num_envs=2;result.time_pool=pool(catalog)
        result.collector=SimpleNamespace(active=False,states=[None,None])
        result.c=SimpleNamespace(stage={'generation_horizon':{'frames':60},'ppo_rollout':{'world_seconds':4}})
        return result
    original=runtime()
    for _ in range(15):original.time_pool.draw('Mine')
    saved=original.checkpoint_state()
    expected=[original.time_pool.draw('Mine') for _ in range(20)]
    restored=runtime();restored.restore_logical(saved);restored.apply_restore()
    assert [restored.time_pool.draw('Mine') for _ in range(20)]==expected
    legacy=copy.deepcopy(saved);legacy['schema']='genmo.stage10.inprocess_collector.boundary'
    with pytest.raises(ValueError,match='restore'):restored.restore_logical(legacy)


@pytest.mark.parametrize('fails',[False,True])
def test_calibration_does_not_consume_training_time_bins(tmp_path,monkeypatch,fails):
    from gem.closedloop.dppo import runtime as module
    runtime=module.InProcessTrainingRuntime.__new__(module.InProcessTrainingRuntime)
    runtime.num_envs=1;runtime.time_pool=pool(FullMusicCatalog(make_catalog_data(tmp_path)))
    runtime.c=SimpleNamespace(distributed=None,identity={'assets':{}},
        policy=SimpleNamespace(numerical_execution={},kernel_config={}))
    runtime._open=lambda *args:None
    runtime._finish=lambda:({},None)
    def calibrate(**kwargs):
        runtime.time_pool.draw('FineDance')
        if fails:raise RuntimeError('diagnostic failed')
        return {}
    runtime.collector=SimpleNamespace(calibrate=calibrate)
    monkeypatch.setattr(module,'local_call',lambda distributed,function:function())
    before=runtime.time_pool.state_dict()
    if fails:
        with pytest.raises(RuntimeError,match='diagnostic failed'):runtime.calibrate()
    else:runtime.calibrate()
    assert runtime.time_pool.state_dict()==before


def test_motion_bootstrap_matches_music_phase_velocity_and_canonical_frame(tmp_path):
    catalog=FullMusicCatalog(make_catalog_data(tmp_path,frames=300))
    sample=catalog.samples['train']['Mine'][0]
    target,qpos=load_paired_activity(catalog.data_root,sample,music_start_frame=150,return_motion=True)
    t=np.arange(300)/30
    qpos[:,0]=10+t;qpos[:,1]=3+2*t;qpos[:,2]=.72
    yaw=math.pi/2
    qpos[:,3:7]=[math.cos(yaw/2),0,0,math.sin(yaw/2)]
    original=qpos.copy()
    reset=make_motion_initialization(qpos,target.source,start_frame=150,plan_frames=120)
    boot=reset['bootstrap_qpos']
    assert boot.shape==(126,28) and reset['music_start_tick']==600
    np.testing.assert_allclose(boot[6,:2],0,atol=1e-12)
    np.testing.assert_allclose(boot[36,:3],[2,-1,.72],atol=1e-12)
    np.testing.assert_allclose(boot[:,3:7],np.tile([1,0,0,0],(126,1)),atol=1e-12)
    np.testing.assert_allclose(boot[36,7:],qpos[150,7:])
    np.testing.assert_array_equal(qpos,original)
    assert target.source['music_start_frame']==reset['music_start_frame']==150


@pytest.mark.parametrize('start',[0,298])
def test_bootstrap_boundary_holds_are_explicit_and_do_not_move_music_start(tmp_path,start):
    catalog=FullMusicCatalog(make_catalog_data(tmp_path,frames=300))
    target,qpos=load_paired_activity(catalog.data_root,catalog.samples['train']['Mine'][0],
                                    music_start_frame=start,return_motion=True)
    reset=make_motion_initialization(qpos,target.source,start_frame=start,plan_frames=60)
    assert reset['music_start_frame']==start
    indices=start-30+np.arange(-6,60)
    assert reset['left_boundary_hold_frames']==int((indices<0).sum())
    assert reset['right_boundary_hold_frames']==int((indices>=300).sum())
    np.testing.assert_allclose(reset['bootstrap_qpos'][36,7:],qpos[start,7:])


def test_environment_passes_motion_reset_and_uses_real_warmup_flow(tmp_path,monkeypatch):
    from tests.closedloop.dppo.test_env_adapter import adapter
    env,backend=adapter(tmp_path,mode='paused')
    catalog=FullMusicCatalog(make_catalog_data(tmp_path,frames=300));sample=catalog.samples['train']['Mine'][0]
    env.config['stage9']['bc_data_root']=str(catalog.data_root)
    env.config['runtime']={'backend':'gpu_vectorized.v1'}
    env.config['stage10']={'dataset':{'initialization':MOTION_RESET_CONTRACT},'generation_horizon':{'frames':120}}
    original=backend.call;resets=[]
    def call(method,**payload):
        if method=='reset_episode':
            resets.append(payload);backend.tick=0
            packet=payload['episode_spec']['initialization']
            return dict(backend.snapshot(),reset_initialization=dict(contract=MOTION_RESET_CONTRACT,
                actual_state_verified=True,music_start_frame=packet['music_start_frame'],source=packet['source']))
        return original(method,**payload)
    backend.call=call
    env.reset_task(sample,catalog.load_music(sample),seed=42,phase='train',music_start_frame=150,
                   initialization=MOTION_RESET_CONTRACT)
    initialization=resets[0]['episode_spec']['initialization']
    assert initialization['music_start_frame']==150 and backend.tick==600
    assert env.music[0,0]==150 and env.music_end_tick==600+5*600
    assert env.snapshot['tick']==600
    before=len(resets)
    with pytest.raises(ValueError,match='explicit GPU Actor-training'):
        env.reset_task(sample,catalog.load_music(sample),seed=42,phase='evaluation',music_start_frame=150,
                       initialization=MOTION_RESET_CONTRACT)
    assert len(resets)==before


@pytest.mark.parametrize('fault',['missing','unverified','wrong_offset','wrong_source'])
def test_backend_ack_must_confirm_motion_initialization(tmp_path,monkeypatch,fault):
    from tests.closedloop.dppo.test_env_adapter import adapter
    env,backend=adapter(tmp_path,mode='paused')
    catalog=FullMusicCatalog(make_catalog_data(tmp_path));sample=catalog.samples['train']['Mine'][0]
    env.config['stage9']['bc_data_root']=str(catalog.data_root)
    env.config['runtime']={'backend':'gpu_vectorized.v1'}
    env.config['stage10']={'dataset':{'initialization':MOTION_RESET_CONTRACT},'generation_horizon':{'frames':120}}
    def bad_ack(method,**payload):
        packet=payload['episode_spec']['initialization']
        ack=dict(contract=MOTION_RESET_CONTRACT,actual_state_verified=True,
                 music_start_frame=packet['music_start_frame'],source=copy.deepcopy(packet['source']))
        if fault=='missing':ack={}
        elif fault=='unverified':ack['actual_state_verified']=False
        elif fault=='wrong_offset':ack['music_start_frame']+=1
        else:ack['source']['motion_file_sha256']='0'*64
        return dict(backend.snapshot(),reset_initialization=ack)
    backend.call=bad_ack
    with pytest.raises(RuntimeError,match='did not apply'):
        env.reset_task(sample,catalog.load_music(sample),seed=42,phase='train',initialization=MOTION_RESET_CONTRACT)
