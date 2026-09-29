"""Stage8 同步视频的纯CPU验收，所有媒体夹具位于pytest临时目录。

用真实ffmpeg生成有限50Hz视频和PCM正弦音频，验证按逐控制步trace排除校准帧、
按真实warmup帧数延迟音乐，输出H264/AAC及正确帧数/时长。测试解码配音检查
静音前段与有声后段，覆盖启动失败全静音、输入SHA不符、断帧/跨episode拒绝及
成功后中间原片回收。本测试只证明媒体同步工具，不替代真实Isaac视频验收。
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import wave

import numpy as np
import pytest

from gem.closedloop.baseline_video import episode_video_mapping, mux_episode_video, mux_episode_videos


@pytest.fixture
def media(tmp_path):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('ffmpeg/ffprobe are required for real media verification')
    raw=tmp_path/'raw.mp4'
    subprocess.run(['ffmpeg','-v','error','-nostdin','-f','lavfi','-i','testsrc2=size=64x64:rate=50',
                    '-frames:v','40','-c:v','libx264','-threads','1','-pix_fmt','yuv420p',str(raw)],check=True)
    audio=tmp_path/'音乐.wav'
    samples=(np.sin(np.arange(48000)*2*np.pi*440/48000)*8000).astype('<i2')
    with wave.open(str(audio),'wb') as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(48000)
        stream.writeframes(samples.tobytes())
    trace=tmp_path/'trace.jsonl'
    write_trace(trace,frames=30,warmup=10,start=10)
    return raw,audio,trace,hashlib.sha256(audio.read_bytes()).hexdigest()


def write_trace(path,*,frames,warmup,start):
    rows=[{'env_id':0,'episode_id':'episode:2','tick':(i+1)*12,'control_tick_begin':i*12,
           'video_frame_index':start+i,'phase':'warmup' if i<warmup else 'music'} for i in range(frames)]
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))


def decode_audio(path):
    result=subprocess.run(['ffmpeg','-v','error','-i',str(path),'-map','0:a:0','-f','f32le',
                           '-ac','1','-ar','48000','pipe:1'],capture_output=True,check=True)
    return np.frombuffer(result.stdout,dtype='<f4')


def test_real_ffmpeg_mux_excludes_calibration_and_delays_music(media,tmp_path):
    raw,audio,trace,digest=media
    output=tmp_path/'synchronized.mp4'
    report=mux_episode_video(raw,audio,trace,output,digest)
    assert report['frame_count']==30 and report['first_raw_frame']==10
    assert report['trimmed_calibration_and_other_episode_frames']==10
    assert report['video_duration_seconds']==pytest.approx(.6,abs=1/48000)
    assert report['audio_delay_seconds']==.2 and report['warmup_frames']==10
    assert report['video_codec']=='h264' and report['audio_codec']=='aac'
    assert not raw.exists() and Path(report['manifest_path']).is_file()
    samples=decode_audio(output)
    assert np.sqrt(np.mean(samples[:int(.15*48000)]**2))<1e-4
    assert np.sqrt(np.mean(samples[int(.25*48000):int(.5*48000)]**2))>.05


def test_startup_failure_is_silent_and_keeps_raw_when_requested(media,tmp_path):
    raw,audio,trace,digest=media
    write_trace(trace,frames=5,warmup=5,start=10)
    output=tmp_path/'failed_episode.mp4'
    report=mux_episode_video(raw,audio,trace,output,digest,keep_raw=True)
    assert report['audio_status']=='silent_startup_failure'
    assert raw.exists() and report['frame_count']==5
    assert np.max(np.abs(decode_audio(output)))<1e-6
    with pytest.raises(FileExistsError): mux_episode_video(raw,audio,trace,output,digest)


def test_wrong_audio_sha_keeps_original_and_creates_no_output(media,tmp_path):
    raw,audio,trace,_=media
    output=tmp_path/'bad.mp4'
    with pytest.raises(ValueError,match='SHA256'):
        mux_episode_video(raw,audio,trace,output,'0'*64)
    assert raw.exists() and not output.exists() and not output.with_suffix('.sync.json').exists()


@pytest.mark.parametrize('field,value', [('video_frame_index',99),('episode_id','other'),('tick',999),('phase','warmup')])
def test_invalid_trace_rejected(tmp_path,field,value):
    trace=tmp_path/'trace.jsonl'
    write_trace(trace,frames=5,warmup=1,start=0)
    rows=[json.loads(s) for s in trace.read_text().splitlines()]
    rows[3][field]=value
    trace.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError): episode_video_mapping(trace)


def test_batch_preserves_each_episode_frames_and_removes_raw_only_after_all_exports(media, tmp_path):
    raw, audio, _, digest = media
    original = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(raw), '-f', 'rawvideo',
                                        '-pix_fmt', 'rgb24', 'pipe:1'])
    original = np.frombuffer(original, np.uint8).reshape(-1, 64, 64, 3)
    specs = []
    for index, (start, frames) in enumerate(((3, 12), (20, 20))):
        trace = tmp_path / f'episode{index}.jsonl'
        write_trace(trace, frames=frames, warmup=3, start=start)
        trace.write_text(trace.read_text().replace('episode:2', f'episode:{index+3}'))
        specs.append(dict(audio_path=audio, trace_path=trace, output_path=tmp_path/f'{index}.mp4',
                          expected_audio_sha256=digest))
    reports = mux_episode_videos(raw, specs)
    assert not raw.exists()
    for report in reports:
        assert not report['raw_retained']
        decoded = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', report['video_path'],
                                          '-map', '0:v:0', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'])
        decoded = np.frombuffer(decoded, np.uint8).reshape(-1, 64, 64, 3)
        expected = original[report['first_raw_frame']:report['end_raw_frame_exclusive']]
        assert decoded.shape == expected.shape
        assert np.abs(decoded.astype(float)-expected).mean() < 3


def test_batch_rejects_cross_episode_frame_overlap_before_export(media, tmp_path):
    raw, audio, first, digest = media
    second = tmp_path/'second.jsonl'
    write_trace(second, frames=10, warmup=2, start=20)
    second.write_text(second.read_text().replace('episode:2', 'episode:3'))
    specs = [dict(audio_path=audio, trace_path=trace, output_path=tmp_path/f'{i}.mp4',
                  expected_audio_sha256=digest) for i, trace in enumerate((first, second))]
    with pytest.raises(ValueError, match='Overlapping'):
        mux_episode_videos(raw, specs)
    assert raw.exists() and not (tmp_path/'0.mp4').exists()
