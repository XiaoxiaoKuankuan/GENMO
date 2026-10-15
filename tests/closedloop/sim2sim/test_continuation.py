"""连续MuJoCo演示的停止策略与诊断真实性回归。

用最小后端替身驱动实际Sim2SimRuntime._episode循环，注入持续姿态误差和迟到
生成结果，验证动力学控制步继续、迟到计划不提交、严格模式仍会终止。替身仅
用于覆盖调度分支，真实接触动力学由用户授权的单曲有界验收另行验证。时钟测试
确认连续模式重新对齐节拍后，真实耗时仍包含延迟，不能虚报实时率。
所有输出仅写pytest临时目录，不加载GPU/Isaac，不写正式模型或数据。
"""
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
import queue
import threading

import numpy as np
import pytest

from gem.closedloop.sim2sim import runtime as module
from gem.closedloop.sim2sim.music import PreparedMusic,Track,AudioPlayer


class Timeline:
    reference_valid_end_tick=99999
    def window(self,tick):return {'origin_plan_ids':['generated:test']}
    def trim(self,tick):pass


class Backend:
    def __init__(self):
        self.tick=0;self.done=False;self.episode_id='test';self.calls=[];self.asset_hashes={}
        self.timeline=Timeline();self.data=SimpleNamespace(qpos=np.array([0.,0.,1.,1.,0.,0.,0.]))
        self.last_feed={k:np.zeros((1,1)) for k in ('policy','history_obs','command_window')}
    def reset_episode(self,episode):self.episode_id=episode
    def step_control(self,*,check_termination):
        self.calls.append(check_termination);self.tick+=12
        self.done=check_termination
        return dict(tick=self.tick,qpos=self.data.qpos.copy(),reference_qpos=self.data.qpos.copy(),
            joint_velocity=np.zeros(21),yaw_error_rad=0.,non_yaw_orientation_error_rad=1.,
            reference_remaining_seconds=10.,done=self.done,reason='physical_failure:non_yaw_orientation_error_rad')


def make_runtime(tmp_path,monkeypatch,continuous=True,pending=None):
    track=Track(Path('/test.wav'),'test')
    music=PreparedMusic(track,np.zeros((30,35),np.float32),.1,{},None,1000)
    monkeypatch.setattr(module,'prepare_music',lambda *a,**k:music)
    runtime=module.Sim2SimRuntime.__new__(module.Sim2SimRuntime)
    runtime.config=dict(paths={'output_root':str(tmp_path)},runtime={'audio':False,'warmup_control_steps':50,
        'auto_next':False,'lateness_limit_seconds':.1},termination={'non_yaw_orientation_error_rad':.6})
    runtime.tracks=[track];runtime.track_index=0;runtime.actor=SimpleNamespace(name='test',identity={'behavior_id':'test'})
    runtime._announced_music=None
    runtime.calibrations={'test':{'deadline_ticks':240}};runtime.episode_number=0;runtime.backend=Backend()
    runtime.commands=queue.Queue();runtime.stop=threading.Event();runtime.scripted_events=[]
    runtime.total_controls=0;runtime.max_seconds=.08;runtime.headless=True;runtime.record=False
    runtime.output=tmp_path;runtime.episodes=[];runtime.pending=pending;runtime.failure=None
    runtime.seed=42;runtime.continuous=continuous;runtime._publish=lambda **kwargs:None
    runtime._drain_generation=lambda:None
    captured=[]
    def finish(diagnostics,directory,**kwargs):
        Path(directory).mkdir()
        captured.append(diagnostics)
        return dict(simulated_seconds=len(diagnostics.records)*.02,real_time_factor=1.)
    monkeypatch.setattr(module.EpisodeDiagnostics,'finish',finish)
    return runtime,captured


def test_physical_error_warns_but_advances_all_control_steps(tmp_path,monkeypatch):
    runtime,captured=make_runtime(tmp_path,monkeypatch)
    _,reason=runtime._episode()
    assert reason=='administrative_time_limit'
    assert runtime.backend.calls==[False]*4
    assert captured[0].warnings['physical_failure:non_yaw_orientation_error_rad']==4


def test_strict_mode_retains_physical_termination(tmp_path,monkeypatch):
    runtime,_=make_runtime(tmp_path,monkeypatch,continuous=False)
    _,reason=runtime._episode()
    assert reason=='physical_failure:non_yaw_orientation_error_rad'
    assert runtime.backend.calls==[True]


def test_late_result_is_rejected_without_stopping_physics(tmp_path,monkeypatch):
    pending=Future();pending.set_result(dict(qpos_world=np.zeros((60,28)),decision_tick=0,prefix_frames=12,
        episode_id='unused',behavior_id='test',deadline_tick=-1,request_id='late',latency_seconds=.8,timing={}))
    runtime,captured=make_runtime(tmp_path,monkeypatch,pending=pending)
    runtime.backend.reset_episode=lambda episode:pending.result().update(episode_id=episode)
    _,reason=runtime._episode()
    assert reason=='administrative_time_limit' and len(runtime.backend.calls)==4
    assert captured[0].warnings['generation_deadline_missed']==1
    assert not captured[0].plans[0]['committed'] and not captured[0].requests


def test_clock_rebase_does_not_hide_real_elapsed_time():
    now=[0.];clock=module.LogicalClock(lambda:now[0]);now[0]=.5
    clock.rebase(.2)
    assert clock.elapsed()==pytest.approx(.3)
    assert clock.measured_elapsed()==pytest.approx(.5)


def test_explicit_audio_resync_is_visible_in_diagnostics():
    music=PreparedMusic(Track(Path('/test.wav'),'test'),np.zeros((30,35)),1.,{},np.zeros((1000,1)),1000)
    player=AudioPlayer(music,enabled=False);player.resync(.35)
    assert player.cursor==350 and player.snapshot()['resyncs']==1
