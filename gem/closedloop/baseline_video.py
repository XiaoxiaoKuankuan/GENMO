"""为 Stage8 的真实 Isaac 视频按实际控制帧同步原始验证集音乐。

本模块只使用标准库及本机 ffmpeg/ffprobe，不导入模型、Isaac 或 GPU 库。
须在 Isaac worker 关闭视频 writer 后调用。worker 的原始视频可能含校准站立
和多个 reset，因此以指定 episode 的 trace.video_frame_index 裁出连续区间，
不能把整个文件简单延迟一秒配音。音频延迟严格等于该 episode 已执行的
warmup 帧数 / 50，音乐从验证清单所指音频的零时刻开始，不改变速度或节奏。

输出 H264/AAC MP4，音频在视频结束处截断；启动失败而没有 music 控制步时
只生成静音并明确记载。ffprobe 核对真实帧数、50fps、两种编码和音视频时长，
内容 SHA 与时间映射保存在旁边的 sync.json。拒绝覆盖任何已有文件；编码或
验收失败保留原始视频，成功后缺省只保留配音成品与元数据，避免重复的大型
中间产物。调用方可显式 keep_raw=True 保留诊断原片。
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

FPS = 50
CONTROL_TICKS = 12
AUDIO_RATE = 48000


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _executable(value):
    found = shutil.which(str(value))
    if not found:
        raise FileNotFoundError(f"Required multimedia executable is unavailable: {value}")
    return found


def _run(command, timeout):
    result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                            text=True, check=False, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"Multimedia command exited {result.returncode}: {result.stderr[-4000:]}")
    return result.stdout


def probe_media(path, *, ffprobe="ffprobe"):
    """读取封装/编码信息及实际解码帧数；无缓存、不写文件。"""
    return json.loads(_run([_executable(ffprobe), "-v", "error", "-count_frames", "-show_streams",
                            "-show_format", "-of", "json", str(Path(path).resolve())], timeout=120))


def _one_stream(probe, kind):
    streams = [stream for stream in probe.get("streams", []) if stream.get("codec_type") == kind]
    if len(streams) != 1:
        raise ValueError(f"Expected exactly one {kind} stream, got {len(streams)}")
    return streams[0]


def _duration(stream, fallback=None):
    raw = stream.get("duration", fallback)
    try:
        result = float(raw)
    except (ValueError, TypeError) as exc:
        raise ValueError("Missing media duration") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError("Media duration must be finite and positive")
    return result


def _frame_count(stream):
    raw = stream.get("nb_read_frames", stream.get("nb_frames"))
    try:
        result = int(raw)
    except (ValueError, TypeError) as exc:
        raise ValueError("Missing decoded video frame count") from exc
    if result <= 0:
        raise ValueError("Empty video")
    return result


def _check_rate(stream):
    if Fraction(stream.get("avg_frame_rate", "0/1")) != FPS:
        raise ValueError("Isaac baseline video must be exactly 50fps")


def episode_video_mapping(trace_path):
    """根据真实 trace 拒绝丢帧、跨episode、时间跳步及不连续warmup。"""
    rows = [json.loads(line) for line in Path(trace_path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("Cannot mux an episode without actual control frames")
    if rows[0].get("control_tick_begin") != 0:
        raise ValueError("Video trace must start at the episode's initial control boundary")
    identities = {(row.get("env_id"), row.get("episode_id")) for row in rows}
    if len(identities) != 1 or None in next(iter(identities)):
        raise ValueError("Video trace must belong to one explicit environment/episode")
    indices, warmup = [], 0
    music_started = False
    for index, row in enumerate(rows):
        frame = row.get("video_frame_index")
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            raise ValueError("Every actual control step needs an integer video_frame_index")
        if index and frame != indices[-1] + 1:
            raise ValueError("Episode video frames must be contiguous and unique")
        tick, begin = row.get("tick"), row.get("control_tick_begin")
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (tick, begin)):
            raise ValueError("Control ticks must be integers")
        if begin % CONTROL_TICKS or tick - begin != CONTROL_TICKS or (index and begin != rows[index-1]["tick"]):
            raise ValueError("Video trace does not contain contiguous actual 50Hz control steps")
        phase = row.get("phase")
        if phase == "warmup" and not music_started:
            warmup += 1
        elif phase == "music":
            music_started = True
        else:
            raise ValueError("Video episode requires an initial warmup followed by music steps")
        indices.append(frame)
    env_id, episode_id = next(iter(identities))
    return {"env_id": env_id, "episode_id": episode_id, "first_raw_frame": indices[0],
            "end_raw_frame_exclusive": indices[-1] + 1, "frame_count": len(rows),
            "warmup_frames": warmup, "music_frames": len(rows) - warmup,
            "audio_delay_seconds": warmup/FPS, "duration_seconds": len(rows)/FPS,
            "control_tick_begin": rows[0]["control_tick_begin"], "control_tick_end": rows[-1]["tick"]}


def mux_episode_video(raw_video, audio_path, trace_path, output_path, expected_audio_sha256,
                      *, keep_raw=False, ffmpeg="ffmpeg", ffprobe="ffprobe", _raw_info=None):
    """裁取一个已记录episode并同步验证集音频，成功返回可直接写入总报告的manifest。"""
    raw_video, audio_path, trace_path, output_path = (
        Path(value).expanduser().resolve() for value in (raw_video, audio_path, trace_path, output_path))
    manifest_path = output_path.with_suffix(".sync.json")
    for path in (raw_video, audio_path, trace_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("Synchronized output must use .mp4")
    if output_path in (raw_video, audio_path, trace_path) or manifest_path in (raw_video, audio_path, trace_path):
        raise ValueError("Output must not overwrite any input")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("Synchronized video or manifest already exists")
    if not isinstance(expected_audio_sha256, str) or len(expected_audio_sha256) != 64:
        raise ValueError("The selected val audio SHA256 must be explicit")
    audio_sha = _sha256(audio_path)
    if audio_sha != expected_audio_sha256.lower():
        raise ValueError("Audio SHA256 differs from selected val manifest")
    executable = _executable(ffmpeg)
    mapping = episode_video_mapping(trace_path)
    if _raw_info is not None and _raw_info["stat"] != (raw_video.stat().st_size, raw_video.stat().st_mtime_ns):
        raise RuntimeError("Raw video changed between episode exports")
    raw_probe = _raw_info["probe"] if _raw_info else probe_media(raw_video, ffprobe=ffprobe)
    raw_stream = _one_stream(raw_probe, "video")
    _check_rate(raw_stream)
    if mapping["end_raw_frame_exclusive"] > _frame_count(raw_stream):
        raise ValueError("Trace references frames absent from the closed raw video")
    audio_probe = probe_media(audio_path, ffprobe=ffprobe)
    audio_stream = _one_stream(audio_probe, "audio")
    audio_duration = _duration(audio_stream, audio_probe.get("format", {}).get("duration"))
    if mapping["music_frames"] / FPS > audio_duration + 1/30:
        raise ValueError("Original audio ends before the recorded music execution")
    raw_sha, trace_sha = (_raw_info["sha256"] if _raw_info else _sha256(raw_video)), _sha256(trace_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = f"{mapping['duration_seconds']:.8f}"
    # 输入侧精确seek从最近关键帧解码至目标50Hz控制帧，避免每首重解整批原片。
    video_filter = (f"[0:v:0]trim=start_frame=0:"
                    f"end_frame={mapping['frame_count']},setpts=PTS-STARTPTS[v]")
    audio_filter = ("[1:a:0]aresample=48000,asetpts=PTS-STARTPTS,"
                    + ("volume=0," if not mapping["music_frames"] else "")
                    + f"adelay={mapping['warmup_frames']*20}:all=1,apad,atrim=duration={duration}[a]")
    with tempfile.TemporaryDirectory(prefix=".stage8_mux_", dir=output_path.parent) as temporary:
        staged = Path(temporary) / "video.mp4"
        command = [executable, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
                   "-threads", "2", "-ss", f"{mapping['first_raw_frame']/FPS:.8f}",
                   "-i", str(raw_video), "-i", str(audio_path),
                   "-filter_complex_threads", "1", "-filter_complex", video_filter + ";" + audio_filter,
                   "-map", "[v]", "-map", "[a]", "-r", "50", "-c:v", "libx264", "-threads:v", "2",
                   "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                   "-ar", str(AUDIO_RATE), "-t", duration, "-movflags", "+faststart", str(staged)]
        _run(command, timeout=300)
        final_probe = probe_media(staged, ffprobe=ffprobe)
        video, audio = _one_stream(final_probe, "video"), _one_stream(final_probe, "audio")
        _check_rate(video)
        if video.get("codec_name") != "h264" or audio.get("codec_name") != "aac":
            raise RuntimeError("Muxed output is not H264/AAC")
        if _frame_count(video) != mapping["frame_count"]:
            raise RuntimeError("Muxing changed actual control frame count")
        video_duration, sound_duration = _duration(video), _duration(audio)
        if abs(video_duration-mapping["duration_seconds"]) > 1/AUDIO_RATE:
            raise RuntimeError("Muxed video duration differs from control frame count")
        if abs(sound_duration-video_duration) > .025:
            raise RuntimeError("Muxed audio/video durations differ beyond one AAC frame")
        for stream in (video, audio):
            if abs(float(stream.get("start_time", "0"))) > 1/AUDIO_RATE:
                raise RuntimeError("Muxed media does not start at time zero")
        report = {"version": "genmo.gmt_frozen_isaac.video_sync.v1", "status": "passed", **mapping,
                  "fps": FPS, "audio_sample_rate": int(audio["sample_rate"]),
                  "audio_status": "music_after_actual_warmup" if mapping["music_frames"] else "silent_startup_failure",
                  "raw_video": str(raw_video), "raw_sha256": raw_sha, "raw_retained": bool(keep_raw),
                  "audio_path": str(audio_path), "audio_sha256": audio_sha,
                  "trace_path": str(trace_path), "trace_sha256": trace_sha,
                  "video_path": str(output_path), "video_sha256": _sha256(staged),
                  "manifest_path": str(manifest_path), "video_codec": video["codec_name"],
                  "audio_codec": audio["codec_name"], "video_duration_seconds": video_duration,
                  "audio_duration_seconds": sound_duration, "width": video["width"], "height": video["height"],
                  "trimmed_calibration_and_other_episode_frames": _frame_count(raw_stream)-mapping["frame_count"]}
        # 同一文件系统硬链接原子发布，目标若被并发创建则失败，绝不覆盖。
        staged_manifest = Path(temporary) / "sync.json"
        staged_manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
        os.link(staged, output_path)
        try:
            os.link(staged_manifest, manifest_path)
        except BaseException:
            output_path.unlink()
            raise
    if not keep_raw:
        if _sha256(raw_video) != raw_sha:
            raise RuntimeError("Raw video changed during mux; refusing cleanup")
        raw_video.unlink()
    return report


def mux_episode_videos(raw_video, episodes, *, ffmpeg="ffmpeg", ffprobe="ffprobe"):
    """逐曲导出共享原片，验证全局帧区间不重叠；全部成功才删除本轮原片。"""
    raw = Path(raw_video).resolve()
    if not episodes:
        raise ValueError("At least one episode video is required")
    mappings = [episode_video_mapping(item["trace_path"]) for item in episodes]
    identities = [mapping["episode_id"] for mapping in mappings]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate episode in video batch")
    for previous, current in zip(mappings, mappings[1:]):
        if previous["end_raw_frame_exclusive"] > current["first_raw_frame"]:
            raise ValueError("Overlapping or unordered episode video frames")
    stat = raw.stat()
    info = {"stat": (stat.st_size, stat.st_mtime_ns), "probe": probe_media(raw, ffprobe=ffprobe),
            "sha256": _sha256(raw)}
    reports = []
    for episode in episodes:
        reports.append(mux_episode_video(raw_video=raw, **episode, keep_raw=True,
                       ffmpeg=ffmpeg, ffprobe=ffprobe, _raw_info=info))
    if (raw.stat().st_size, raw.stat().st_mtime_ns) != info["stat"] or _sha256(raw) != info["sha256"]:
        raise RuntimeError("Raw video changed during batch export; refusing cleanup")
    raw.unlink()
    for report in reports:
        report["raw_retained"] = False
        Path(report["manifest_path"]).write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    return reports
