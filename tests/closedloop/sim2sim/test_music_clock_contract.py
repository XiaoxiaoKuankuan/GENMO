"""实时音乐/暂停与随机核身份的CPU契约测试。

测试明确区分完整音乐、配对片段与外部音频；暂停不推进样本或逻辑截止，恢复
保留同一剩余预算。缓存按内容/提取配置失效，篡改缓存不得静默被读入。
本文件不启动GPU、Isaac或声音设备，真实声音与动力学另在集中验收目录保存。
"""
from pathlib import Path
import json
import numpy as np
import pytest
import soundfile as sf
import torch
from gem.closedloop.sim2sim.music import Track,PreparedMusic,AudioPlayer,load_playlist,prepare_music,file_sha
from gem.closedloop.sim2sim.runtime import LogicalClock,settings_from_kernel


def test_pause_freezes_deadline_and_does_not_accumulate_wall_time():
    now=[10.]
    clock=LogicalClock(lambda:now[0]);now[0]=10.25;clock.pause()
    now[0]=1000.;assert clock.elapsed()==.25
    clock.resume();now[0]=1000.1
    assert clock.elapsed()==pytest.approx(.35)
    clock.pause();clock.pause();now[0]=1200.;clock.resume()
    assert clock.elapsed()==pytest.approx(.35)


def test_audio_pause_retains_sample_cursor_and_does_not_skip():
    pcm=np.arange(200,dtype=np.float32)[:,None]
    music=PreparedMusic(Track(Path('/unused.wav'),'test'),np.zeros((6,35)),.2,{},pcm,1000)
    player=AudioPlayer(music,enabled=False)
    out=np.empty((10,1),np.float32)
    player.update(0,True);player._callback(out,10,None,False)
    np.testing.assert_array_equal(out[:,0],np.arange(10))
    player.update(.01,False);player._callback(out,10,None,False)
    assert not out.any() and player.cursor==10
    player.update(.01,True);player._callback(out,10,None,False)
    np.testing.assert_array_equal(out[:,0],np.arange(10,20))


def test_manifest_retains_independent_music_identity(tmp_path):
    source=tmp_path/'AIST++';(source/'audio').mkdir(parents=True)
    sf.write(source/'audio/song.wav',np.zeros(1000),1000)
    manifest=tmp_path/'selection.json'
    manifest.write_text(json.dumps({'samples':[{'source':'AIST++','split':'test','row':{
        'audio_path':'audio/song.wav','music_group_id':'group','sequence_id':'motionA'}}]}))
    tracks=load_playlist(manifest,tmp_path)
    assert tracks[0].source=='AIST++' and tracks[0].split=='test' and tracks[0].group=='group'
    assert tracks[0].audio==source/'audio/song.wav'


def test_external_cache_identity_and_corruption_fail_closed(tmp_path,monkeypatch):
    from gem.utils import music_features
    calls=[]
    def extract(*args,**kwargs):
        calls.append(str(args[0]));return torch.zeros(30,35),{'test_fixture':True}
    monkeypatch.setattr(music_features,'extract_edge_baseline35',extract)
    audio=tmp_path/'song.wav';sf.write(audio,np.zeros(8000),8000)
    track=Track(audio,'external');cache=tmp_path/'cache'
    first=prepare_music(track,cache,decode_audio=False)
    second=prepare_music(track,cache,decode_audio=False)
    assert len(calls)==1 and first.identity==second.identity
    assert first.identity['target_activity_applicable'] is False
    cached=next(cache.glob('*.npz'))
    with np.load(cached) as f:values={k:f[k].copy() for k in f.files}
    values['features'][0,0]=1
    np.savez(cached,**values)
    with pytest.raises(ValueError,match='缓存内容SHA'):
        prepare_music(track,cache,decode_audio=False)


def test_partial_manifest_reextracts_full_song_without_pretending_full_pair(tmp_path,monkeypatch):
    from gem.utils import music_features
    monkeypatch.setattr(music_features,'extract_edge_baseline35',lambda *a,**kw:(torch.zeros(60,35),{}))
    audio=tmp_path/'song.wav';sf.write(audio,np.zeros(16000),8000)
    feature=tmp_path/'feature.pt';torch.save(torch.ones(30,35),feature)
    music=prepare_music(Track(audio,'partial',features=feature,audio_sha=file_sha(audio),feature_sha=file_sha(feature)),
                        tmp_path/'cache',decode_audio=False)
    assert music.features.shape==(60,35)
    assert music.identity['manifest_partial_feature_seconds']==1.
    assert music.identity['feature_selection']=='explicit_full_audio_reextract_for_partial_manifest'


def test_stage2_kernel_reconstruction_uses_declared_eta_cfg_and_horizon():
    kernel=dict(generation_horizon_frames=60,steps=20,eta=.17,guidance_scale=1.7,effective_std_floors=[.003]*20,
        numerical_execution=dict(precision='native_fp32',linear='stable_tf32x3',attention='sdpa_math',
            compiler=None,final_projection='float64_local',weight_gradient='direct_joint_gemm'))
    settings,precision=settings_from_kernel(kernel)
    assert settings==dict(generation_frames=60,denoising_steps=20,eta=.17,guidance_scale=1.7,std_schedule=[.003]*20)
    assert precision['linear_execution']=='stable_tf32x3' and precision['precise_output']
    kernel['numerical_execution']['compiler']='unverified'
    with pytest.raises(ValueError,match='compile'):settings_from_kernel(kernel)
