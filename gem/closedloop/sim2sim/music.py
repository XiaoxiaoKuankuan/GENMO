"""实时sim2sim的音乐清单、完整音频特征缓存与播放时钟。

清单可读取现有四来源selection.json、逐行manifest、音频文件夹或明确的音频路径列表；
文件夹递归扫描WAV/MP3，按相对路径排序，排除隐藏文件和macOS资源附属文件；
保留source/split/music_group_id和SHA，不能把同音乐动作片段当独立音乐。
任意WAV/MP3在物理episode开始前整首提取既有EDGE35，缓存键绑定音频内容、
提取源码和参数。无配对示范时目标活动度/活动门控/训练奖励均不适用。
音频callback仅复制预解码PCM；暂停、重播与切歌由物理控制边界更新游标，
不在20ms控制路径中解码或提取特征，不启动任何外部机器人设备。
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import os
import threading
from pathlib import Path
import numpy as np


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Track:
    audio: Path
    name: str
    source: str = 'external'
    split: str = 'not_applicable'
    group: str = ''
    features: Path | None = None
    audio_sha: str | None = None
    feature_sha: str | None = None
    motion: Path | None = None
    motion_sha: str | None = None
    sequence: str | None = None
    split_provenance: dict | None = None


def load_playlist(path, dataset_root):
    """读取清单、单首或目录；目录不猜测训练划分，也不按文件名认定独立音乐。"""
    path, root = Path(path).expanduser().resolve(), Path(dataset_root).expanduser().resolve()
    if not path.exists():raise FileNotFoundError(path)
    if path.is_dir():
        candidates=[]
        for directory, subdirs, files in os.walk(path,followlinks=False):
            subdirs[:]=sorted(d for d in subdirs if not d.startswith('.') and d!='__MACOSX')
            for name in files:
                if name.startswith('.') or Path(name).suffix.lower() not in ('.wav','.mp3'):continue
                candidate=Path(directory)/name
                if candidate.is_file():candidates.append(candidate)
        candidates.sort(key=lambda p:(p.relative_to(path).as_posix().casefold(),p.relative_to(path).as_posix()))
        tracks=[];seen=set()
        for candidate in candidates:
            resolved=candidate.resolve()
            if resolved in seen:continue
            seen.add(resolved);tracks.append(Track(resolved,candidate.stem))
        if not tracks:raise ValueError('音频文件夹没有可用的WAV/MP3: '+str(path))
        print(f'[MUSIC] 文件夹读取{len(tracks)}个音频文件，按相对路径排序: {path}',flush=True)
        return tracks
    if path.suffix.lower() in ('.wav','.mp3'):
        return [Track(path, path.stem)]
    if path.suffix.lower() == '.jsonl':
        rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        base = path.parent.parent if path.parent.name == 'manifests' else path.parent
    elif path.suffix.lower() == '.json':
        document = json.loads(path.read_text())
        rows = document if isinstance(document,list) else document.get('samples',document.get('tracks',document.get('tasks')))
        if rows is None:
            raise ValueError('音乐清单需要samples/tracks/tasks列表')
        base = path.parent
    else:
        rows = [s.strip() for s in path.read_text().splitlines() if s.strip() and not s.lstrip().startswith('#')]
        base = path.parent
    tracks = []
    for item in rows:
        if isinstance(item,str):
            audio = Path(item).expanduser()
            audio = audio if audio.is_absolute() else base/audio
            tracks.append(Track(audio.resolve(),audio.stem))
            continue
        row = item.get('row',item)
        source = item.get('source',row.get('source',base.name))
        directory = root/source if 'row' in item else base
        def resolve(key):
            if not row.get(key):return None
            value = Path(row[key]).expanduser()
            return (value if value.is_absolute() else directory/value).resolve()
        audio = resolve('audio_path') or resolve('audio')
        if audio is None:raise ValueError('清单条目缺少audio_path')
        tracks.append(Track(audio, row.get('music_token',audio.stem),source,
            item.get('split',row.get('split','unknown')), row.get('music_group_id',row.get('audio_key',audio.stem)),
            resolve('music_feature_path'),row.get('source_audio_sha256'),row.get('source_music_feature_sha256'),
            resolve('motion_path'),row.get('source_motion_sha256'),row.get('sequence_id'),
            {k:item[k] for k in ('pretraining_split','stage1_train_group_overlap','pretraining_unseen_claim') if k in item}))
    if not tracks:raise ValueError('音乐清单为空')
    for track in tracks:
        if not track.audio.is_file():raise FileNotFoundError(track.audio)
    return tracks


@dataclass
class PreparedMusic:
    track: Track
    features: np.ndarray
    duration: float
    identity: dict
    pcm: np.ndarray | None
    sample_rate: int
    paired_qpos: np.ndarray | None = None
    paired_joint_names: list | None = None


def prepare_music(track, cache, *, decode_audio=True):
    import torch
    import soundfile as sf
    from gem.utils import music_features as extractor
    sha = file_sha(track.audio)
    if track.audio_sha and sha != track.audio_sha:
        raise ValueError('清单音频SHA不匹配: '+str(track.audio))
    feature_sha = None
    if track.features is not None:
        feature_sha = file_sha(track.features)
        if not track.feature_sha or feature_sha != track.feature_sha:
            raise ValueError('已有音乐特征必须提供并满足SHA身份')
        values = torch.load(track.features,map_location='cpu',weights_only=False)
        features = np.asarray(values, dtype=np.float32)
        extraction = dict(method='manifest_EDGE35',path=str(track.features))
    else:
        import librosa
        key_data = dict(audio_sha256=sha,extractor_sha256=file_sha(extractor.__file__),fps=30,
                        aist_sequence_name=track.sequence,librosa_version=librosa.__version__)
        key = hashlib.sha256(json.dumps(key_data,sort_keys=True).encode()).hexdigest()
        cache = Path(cache);cache.mkdir(parents=True,exist_ok=True)
        cached = cache/(key+'.npz')
        if cached.exists():
            with np.load(cached,allow_pickle=False) as data:
                features=data['features'].copy();extraction=json.loads(str(data['metadata'].item()))
                if json.loads(str(data['key'].item())) != key_data:raise ValueError('音乐缓存身份错误')
                if hashlib.sha256(features.tobytes()).hexdigest()!=str(data['sha256'].item()):
                    raise ValueError('音乐缓存内容SHA错误')
            print(f'[MUSIC] 复用整曲EDGE35缓存: {track.audio.name}',flush=True)
        else:
            print(f'[MUSIC] 首次提取整曲EDGE35，完成后缓存: {track.audio.name}',flush=True)
            values, extraction=extractor.extract_edge_baseline35(track.audio,target_fps=30,aist_sequence_name=track.sequence)
            features=np.asarray(values,dtype=np.float32)
            # 独占发布，失败不留下可被误读的完整缓存。
            import tempfile
            with tempfile.NamedTemporaryFile(dir=cache,suffix='.partial',delete=False) as stream:
                tmp=Path(stream.name)
                np.savez_compressed(stream,features=features,metadata=json.dumps(extraction),
                    key=json.dumps(key_data),sha256=hashlib.sha256(features.tobytes()).hexdigest())
            try:os.replace(tmp,cached)
            finally:tmp.unlink(missing_ok=True)
    if features.ndim != 2 or features.shape[1] != 35 or not len(features) or not np.isfinite(features).all():
        raise ValueError('音乐特征必须为有限[N,35]，30Hz')
    try:
        info=sf.info(track.audio);duration=float(info.duration);rate=int(info.samplerate)
        pcm=sf.read(track.audio,dtype='float32',always_2d=True)[0] if decode_audio else None
    except (RuntimeError, sf.LibsndfileError):
        import librosa
        value,rate=librosa.load(track.audio,sr=48000,mono=False)
        pcm=np.asarray(value,dtype=np.float32).reshape(1,-1).T if value.ndim==1 else value.T
        duration=len(pcm)/rate
        if not decode_audio:pcm=None
    paired, names, motion_file_sha = None, None, None
    if track.motion is not None:
        motion=torch.load(track.motion,map_location='cpu',weights_only=False)
        # manifest记录转换前来源SHA，不能误当成包装后的.pt内容SHA。
        if track.motion_sha is None or motion.get('source_motion_sha256')!=track.motion_sha:
            raise ValueError('配对动作来源SHA不匹配')
        motion_file_sha=file_sha(track.motion)
        if motion['fps']!=30 or motion['quaternion_convention']!='wxyz':raise ValueError('配对动作合同不匹配')
        paired=np.asarray(motion['qpos'],dtype=np.float64);names=list(motion['joint_names'])
    # 旧清单常含同曲短动作片段。完整歌曲演示须显式提取整首，不能补零或重复。
    if len(features)/30 < duration-.1:
        if track.features is not None:
            from dataclasses import replace
            print(f'[MUSIC] 清单特征仅覆盖{len(features)/30:.2f}s，整曲{duration:.2f}s；'
                  '改用整曲EDGE35（已有缓存直接复用）',flush=True)
            full=prepare_music(replace(track,features=None,feature_sha=None),cache,decode_audio=decode_audio)
            full.track=track
            full.identity['manifest_partial_feature_sha256']=feature_sha
            full.identity['manifest_partial_feature_seconds']=len(features)/30
            full.identity['feature_selection']='explicit_full_audio_reextract_for_partial_manifest'
            return full
        raise ValueError(f'音乐特征仅{len(features)/30:.2f}s，音频{duration:.2f}s；请使用--audio整首提取')
    identity=dict(audio=str(track.audio),audio_sha256=sha,source=track.source,split=track.split,
        music_group_id=track.group,feature_file_sha256=feature_sha,
        feature_values_sha256=hashlib.sha256(features.tobytes()).hexdigest(),extraction=extraction,
        paired_motion=str(track.motion) if paired is not None else None,
        paired_source_motion_sha256=track.motion_sha,paired_motion_file_sha256=motion_file_sha,
        target_activity_applicable=paired is not None,
        paired_activity_coverage_seconds=len(paired)/30 if paired is not None else None,duration_seconds=duration)
    identity['split_provenance']=track.split_provenance
    identity['split_scope']='declared_manifest_split; no inference of unseen Stage1 music'
    return PreparedMusic(track,features,duration,identity,pcm,rate,paired,names)


class AudioPlayer:
    """回调采用独立样本游标；物理暂停时输出静音且游标不增长。"""
    def __init__(self, music, enabled=True):
        self.music=music;self.enabled=enabled;self.lock=threading.Lock()
        self.cursor=0;self.running=False;self.target_seconds=0.;self.underruns=0;self.stream=None
        if enabled:
            import sounddevice as sd
            if music.pcm is None:raise ValueError('播放前必须预解码PCM')
            self.stream=sd.OutputStream(samplerate=music.sample_rate,channels=music.pcm.shape[1],
                dtype='float32',blocksize=0,latency='low',callback=self._callback)
            self.stream.start()

    def _callback(self,outdata,frames,timing,status):
        outdata.fill(0)
        with self.lock:
            if status:self.underruns+=1
            if not self.running:return
            # 允许当前控制周期的20ms音频，绝不追赶而跳过物理经验。
            end=min(len(self.music.pcm),self.cursor+frames,
                    int((self.target_seconds+.02)*self.music.sample_rate))
            count=max(0,end-self.cursor)
            outdata[:count]=self.music.pcm[self.cursor:self.cursor+count]
            self.cursor+=count

    def update(self, seconds, running):
        with self.lock:
            self.target_seconds=max(0.,seconds);self.running=bool(running and seconds>=0)

    def snapshot(self):
        with self.lock:
            return dict(enabled=self.enabled,played_seconds=self.cursor/self.music.sample_rate,
                target_seconds=self.target_seconds,underflow_callbacks=self.underruns,
                drift_seconds=(self.cursor/self.music.sample_rate-self.target_seconds) if self.enabled else None)

    def close(self):
        self.update(self.target_seconds,False)
        if self.stream is not None:self.stream.stop();self.stream.close();self.stream=None
