#!/usr/bin/env python3
"""本机 GENMO → SONIC SMPL mode 2 → G1 MuJoCo 音乐演示统一入口。

本程序负责选段音频解码、EDGE35 特征、120/30 帧滚动 DDIM、跨窗姿态解码、
50 Hz SONIC 适配、声卡播放和可靠会话协调。控制策略与 MuJoCo 分别在独立进程
运行，媒体零点位于两秒准备动作之后；音乐速度和采样长度保持不变。后台生产
线程按 4/12 秒水位补帧，主线程持续发送心跳、记录状态并检测 100 ms 同步偏差。

默认绑定已经验证的 physics_v3 s100000 及当前 1762/994 维 SONIC release。
--launch-local 只启动本程序拥有的本机仿真子进程，退出时也只清理这些子进程。
故障停止音乐和物理推进并保存诊断；Ctrl+C 在尚未消费的未来位置安排一秒收尾。
--audio-output off 仅用于无声测试，报告不会将它计为声卡音视频同步验收。
--smpl-npz 可回放本入口保存的 generated_smpl.npz，以独立检查控制与同步。
每次运行创建独立目录，保存资产散列、命令、源 SMPL、每窗耗时及真实仿真状态。
"""

from __future__ import annotations

# 路径和动态库准备必须先于第三方导入。
# ruff: noqa: E402
import argparse
import hashlib
import json
import math
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 没有系统 PortAudio 时允许使用虚拟环境内的同名动态库；重启一次使动态加载器生效。
_audio_lib = Path(sys.prefix) / "lib"
if __name__ == "__main__" and (_audio_lib / "libportaudio.so.2").exists():
    _paths = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    if str(_audio_lib) not in _paths:
        os.execve(
            sys.executable,
            [sys.executable, "-B", *sys.argv],
            dict(os.environ, LD_LIBRARY_PATH=":".join([str(_audio_lib), *_paths])),
        )

import numpy as np
import soundfile as sf
import torch

from gem.network.endecoder import EnDecoder
from gem.runtime.music_only_trt import (
    SlidingDDIMGenerator,
    StreamingSmplDecoder,
    TensorRTStepRunner,
    derive_window_seed,
    padded_music_window,
    plan_sliding_windows,
    sha256_file,
)
from gem.runtime.sonic_music import AudioClock, PoseTimeline, SessionClient
from gem.utils.music_features import align_features_to_length, extract_edge_baseline35
from gem.utils.sonic.zmq_publisher import _pack_pose_message_compat

CHECKPOINT = (
    ROOT
    / "outputs/gem_smpl_music_only_4set_manual_q1_physics_v3_100k/version_0/checkpoints/s100000.ckpt"
)
CHECKPOINT_SHA = "98d70a145fb8f430ab557cdd0bde4af5f71e881b16f6976b385db6d172683136"
ENGINE_DIR = ROOT / "outputs/tensorrt/sonic_music_physics_v3_s100000/engines"


def parser():
    """命令行只暴露文件演示所需选项；真机网络接口不在本入口范围内。"""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--start-sec", type=float, default=0)
    p.add_argument("--duration-sec", type=float)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--engine", type=Path)
    p.add_argument("--ddim-steps", type=int, default=20)
    p.add_argument("--guidance-scale", type=float, default=2.5)
    p.add_argument("--sonic-root", type=Path, default=Path("/home/weili/GR00T-WholeBodyControl"))
    p.add_argument("--sim-python", type=Path, help="默认使用 SONIC 仓库的 .venv_sim/bin/python")
    p.add_argument("--launch-local", action="store_true")
    p.add_argument("--headless", action="store_true")
    p.add_argument(
        "--exit-on-finish",
        action="store_true",
        help="自然收尾后退出；默认保留站立画面，Ctrl+C 关闭",
    )
    p.add_argument("--audio-output", choices=("device", "off"), default="device")
    p.add_argument("--audio-device")
    p.add_argument("--sonic-endpoint", default="tcp://127.0.0.1:5560")
    p.add_argument("--sim-endpoint", default="tcp://127.0.0.1:5561")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--smpl-npz", type=Path)
    p.add_argument(
        "--audit-observations",
        action="store_true",
        help="保存实际编码器输入供 Python/C++ 一致性验收",
    )
    p.add_argument(
        "--stop-after", type=float, help="在指定音乐秒数执行用户停止，用于可复现收尾检查"
    )
    return p


def write_json(path, data):
    """先写临时文件再替换，避免进程中断留下半份报告。"""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def decode_audio(args, output):
    """统一解码成 48 kHz 双声道 PCM，特征和声卡都使用同一份精确选段。"""
    if not math.isfinite(args.start_sec) or args.start_sec < 0:
        raise ValueError("音频起始时间无效")
    if args.duration_sec is not None and (
        not math.isfinite(args.duration_sec) or args.duration_sec <= 0
    ):
        raise ValueError("音频选段时长无效")
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(args.audio.resolve(strict=True)),
        "-ss",
        str(args.start_sec),
    ]
    if args.duration_sec is not None:
        cmd += ["-t", str(args.duration_sec)]
    cmd += ["-f", "f32le", "-acodec", "pcm_f32le", "-ac", "2", "-ar", "48000", "pipe:1"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    pcm = np.frombuffer(raw, dtype="<f4").reshape(-1, 2).copy()
    if not len(pcm) or not np.isfinite(pcm).all():
        raise ValueError("解码后的音频为空或存在非有限数据")
    sf.write(output / "audio.wav", pcm, 48000, subtype="FLOAT")
    return pcm, 48000


class LocalProcesses:
    """记录并管理本入口创建的两个子进程，不终止既有服务或用户进程。"""

    def __init__(self):
        self.children, self.logs, self.commands = [], [], []

    def launch(self, args, output):
        for endpoint in (args.sonic_endpoint, args.sim_endpoint):
            if not endpoint.startswith("tcp://127.0.0.1:"):
                raise ValueError("音乐入口只接受本机 TCP 端点")
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", int(endpoint.rsplit(":", 1)[1])))
        sonic = args.sonic_root.resolve(strict=True)
        deploy = sonic / "gear_sonic_deploy"
        commands = [
            (
                "mujoco",
                sonic,
                [
                    str(args.sim_python or sonic / ".venv_sim/bin/python"),
                    "-B",
                    "gear_sonic/scripts/run_sim_loop.py",
                    "--music-endpoint",
                    args.sim_endpoint,
                    "--music-log-path",
                    str(output / "sim_state.jsonl"),
                    "--sim-frequency",
                    "200",
                ]
                + (["--no-enable-onscreen"] if args.headless else []),
            ),
            (
                "sonic",
                deploy,
                [
                    str(deploy / "target/release/g1_deploy_onnx_ref"),
                    "lo",
                    "policy/release/model_decoder.onnx",
                    "reference/example",
                    "--encoder-file",
                    "policy/release/model_encoder.onnx",
                    "--obs-config",
                    "policy/release/observation_config.yaml",
                    "--encoder-mode",
                    "2",
                    "--input-type",
                    "music",
                    "--disable-crc-check",
                    "--music-endpoint",
                    args.sonic_endpoint,
                    "--logs-dir",
                    str(output / "sonic"),
                ],
            ),
        ]
        env = dict(
            os.environ,
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONUNBUFFERED="1",
            OMP_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="2",
        )
        for name, cwd, command in commands:
            log = (output / f"{name}.log").open("x")
            self.logs.append(log)
            child = subprocess.Popen(
                command,
                cwd=cwd,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=env,
                start_new_session=True,
            )
            self.children.append((name, child))
            self.commands.append(dict(name=name, cwd=str(cwd), argv=command, pid=child.pid))

    def check(self):
        for name, child in self.children:
            if child.poll() is not None:
                raise RuntimeError(f"{name} 子进程退出，代码 {child.returncode}，请查看对应日志")

    def close(self):
        for _, child in reversed(self.children):
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
        for _, child in self.children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
        for log in self.logs:
            log.close()


def assets(args):
    """把实际加载文件和源代码提交写入同一清单，避免同名模型混用。"""
    sonic = args.sonic_root.resolve(strict=True)
    paths = [
        sonic / "gear_sonic_deploy/policy/release" / name
        for name in ("model_encoder.onnx", "model_decoder.onnx", "observation_config.yaml")
    ]
    paths += [
        sonic / "gear_sonic/data/human/human_joints_info.pkl",
        sonic / "gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml",
        sonic / "gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml",
        sonic / "gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml",
        sonic / "gear_sonic_deploy/target/release/g1_deploy_onnx_ref",
        args.audio.resolve(strict=True),
    ]
    result = {str(path): sha256_file(path) for path in paths}
    expected = (
        "013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3",
        "c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed",
        "466d05947c78af6c76388adfb86e3a2a77b2a1d921a64883ed3d085ebf58de1b",
    )
    if tuple(result[str(p)] for p in paths[:3]) != expected:
        raise ValueError("SONIC release 模型或观测配置与本次固定基线不符")
    for path in (
        ROOT / "gem/runtime/sonic_music.py",
        ROOT / "gem/runtime/music_only_trt.py",
        ROOT / "gem/network/endecoder.py",
        ROOT / "gem/network/stats_compose.py",
        Path(__file__),
        sonic / "gear_sonic/utils/mujoco_sim/music_session.py",
        sonic / "gear_sonic/utils/mujoco_sim/base_sim.py",
        sonic
        / "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/music_session.hpp",
    ):
        result[str(path)] = sha256_file(path)
    return result


def summarize(rows, windows, duration, audio_output, error, stopped):
    """分开报告采样监测、完整控制周期和生成耗时，未经测量的指标保持空值。"""
    active = [r for r in rows if r.get("media_active")]

    def distribution(values):
        if not values:
            return None
        return {
            "count": len(values),
            "mean": float(np.mean(values)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "max": float(np.max(values)),
        }

    live = [w["production_seconds"] for w in windows if w["during_playback"]]
    sync = distribution([abs(r["sync_ms"]) for r in active])
    return dict(
        completed=not error and not stopped,
        user_stopped=stopped,
        error=error,
        duration_seconds=duration,
        audio_output=audio_output,
        real_audio_validation=audio_output == "device",
        generation_live_seconds=distribution(live),
        sync_abs_ms=sync,
        sampled_control_compute_ms=distribution([r["sonic"]["compute_ms"] for r in active]),
        sim_abs_skew_ms=distribution([abs(r["sim"]["skew_ms"]) for r in active]),
        minimum_buffer_seconds=min((r["sonic"]["buffer_seconds"] for r in active), default=None),
        sync_target_pass=bool(sync and sync["p95"] <= 40 and sync["max"] <= 100 and not error),
        generation_target_pass=bool(live and np.mean(live) < 3 and np.percentile(live, 95) <= 2.4),
    )


def run(args):
    """统一协调准备、预缓冲、预约起播、在线生产、自然结束以及故障冻结。"""
    torch.set_num_threads(2)
    session_id = str(uuid.uuid4())
    output = (
        args.output_dir
        or ROOT
        / "outputs/sonic_music"
        / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + session_id[:8])
    ).resolve()
    output.mkdir(parents=True, exist_ok=False)
    print(f"session={session_id} output={output}", flush=True)
    processes = LocalProcesses()
    clients = [SessionClient(e, session_id) for e in (args.sonic_endpoint, args.sim_endpoint)]
    sonic, sim = clients
    quit_requested, producer_stop, playing = threading.Event(), threading.Event(), threading.Event()
    original_handler = signal.signal(signal.SIGINT, lambda *_: quit_requested.set())
    producer = None
    audio = None
    windows, rows, decoded_chunks, worker_errors, committed = [], [], [], [], []
    prefilled = threading.Event()
    error, user_stopped, prepared = "", False, []
    duration, epoch = 0.0, 0
    full_control_metrics = None
    gpu_handle = None
    telemetry = (output / "timeline.jsonl").open("x", encoding="utf-8")
    try:
        manifest = dict(
            session_id=session_id,
            args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            assets=assets(args),
            control_hz=50,
            physics_hz=200,
            source_hz=30,
            audio_start_frame=100,
            encoder_mode=2,
            translation_control=False,
        )
        for label, cwd in (("genmo", ROOT), ("sonic", args.sonic_root)):
            manifest[label + "_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=cwd, text=True
            ).strip()
        pcm, rate = decode_audio(args, output)
        duration = len(pcm) / rate
        manifest.update(
            audio_samples=len(pcm),
            audio_sample_rate=rate,
            duration_seconds=duration,
            selected_audio_sha256=sha256_file(output / "audio.wav"),
        )
        timeline = PoseTimeline(args.sonic_root, duration)
        frame_count = math.ceil(duration * 30 - 1e-9)
        plan = plan_sliding_windows(frame_count)
        if args.smpl_npz:
            saved = np.load(args.smpl_npz, allow_pickle=False)
            if len(saved["body_pose"]) != frame_count:
                raise ValueError("固定 SMPL 文件帧数必须与音频精确时长对应")
            manifest["fixed_smpl_sha256"] = sha256_file(args.smpl_npz)
        else:
            features, feature_info = extract_edge_baseline35(output / "audio.wav")
            features = align_features_to_length(features, frame_count, "trim_or_pad_last")
            manifest["features"] = feature_info
            engine = args.engine
            if engine is None:
                candidates = sorted(ENGINE_DIR.glob("*/music_only_denoiser.engine"))
                if len(candidates) != 1:
                    raise ValueError("默认模型需要唯一 TensorRT engine，请显式指定 --engine")
                engine = candidates[0]
            runner = TensorRTStepRunner(engine)
            if (
                runner.manifest["checkpoint_sha256"] != CHECKPOINT_SHA
                or sha256_file(CHECKPOINT) != CHECKPOINT_SHA
            ):
                raise ValueError("TensorRT engine 或 checkpoint 不属于选定的 physics_v3 s100000")
            manifest["engine"] = runner.manifest
            generator = SlidingDDIMGenerator(
                runner, device="cuda:0", steps=args.ddim_steps, guidance_scale=args.guidance_scale
            )
            endecoder = EnDecoder(
                stats_name="MM_V1_AMASS_LOCAL_BEDLAM_CAM",
                encode_type="gvhmr",
                feat_dim=151,
                clip_std=True,
            )
            endecoder.build_obs_indices_dict()
            decoder = StreamingSmplDecoder(endecoder, "cuda:0")
            manifest["endecoder"] = dict(
                stats_name="MM_V1_AMASS_LOCAL_BEDLAM_CAM",
                encode_type="gvhmr",
                clip_std=True,
                feature_dim=151,
                effective_stats_sha256=hashlib.sha256(
                    b"".join(
                        endecoder.stats_dict["gvhmr"][key].cpu().numpy().tobytes()
                        for key in ("mean", "std")
                    )
                ).hexdigest(),
            )
            generator.generate_window(
                padded_music_window(features, plan[0]), valid_length=plan[0].valid_length, seed=0
            )
            torch.cuda.synchronize()
        write_json(output / "manifest.json", manifest)
        if args.launch_local:
            processes.launch(args, output)
            manifest["processes"] = processes.commands
            write_json(output / "manifest.json", manifest)
        deadline = time.monotonic() + 120
        for client in clients:
            probe = SessionClient(client.endpoint, timeout_ms=100)
            try:
                while True:
                    processes.check()
                    try:
                        state = probe.call("status")
                        if state["state"] in ("playing", "armed"):
                            raise RuntimeError("端点已有正在播放的会话")
                        break
                    except TimeoutError:
                        if time.monotonic() > deadline or quit_requested.is_set():
                            raise RuntimeError("等待本机服务超时或用户取消") from None
                        time.sleep(0.1)
            finally:
                probe.close()
        sim.call("prepare")
        prepared.append(sim)
        sonic.call("prepare", audio_frames=timeline.target_frames, audio_start_frame=100)
        prepared.append(sonic)
        audio = AudioClock(pcm, rate, args.audio_output, args.audio_device)
        if audio.stream:
            import sounddevice as sd

            manifest["audio_device"] = dict(sd.query_devices(audio.stream.device))
            manifest["audio_latency_seconds"] = audio.stream.latency
        try:
            import pynvml

            pynvml.nvmlInit()
            gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as exc:
            manifest["gpu_monitor_unavailable"] = str(exc)

        def produce():
            """只提交每窗新帧；首两窗就绪后等待起播，再按水位生产。"""
            previous, filling = None, False
            try:
                for i, window in enumerate(plan):
                    if i >= 2:
                        prefilled.set()
                        while not producer_stop.is_set():
                            if playing.is_set():
                                buffered = sonic.call("status")["buffer_seconds"]
                                if buffered < 4:
                                    filling = True
                                if filling and buffered < 12:
                                    break
                                if buffered >= 12:
                                    filling = False
                            producer_stop.wait(0.05)
                    if producer_stop.is_set():
                        return
                    began = time.monotonic()
                    if args.smpl_npz:
                        start, end = window.start + window.new_start, window.end
                        params = {
                            k: torch.from_numpy(saved[k][start:end].copy())
                            for k in ("body_pose", "global_orient", "transl", "betas")
                        }
                    else:
                        generated = generator.generate_window(
                            padded_music_window(features, window),
                            valid_length=window.valid_length,
                            seed=derive_window_seed(args.seed, i),
                            known_x0=previous,
                        )
                        previous = generated[-30:].detach().clone()
                        params = decoder.decode_new(
                            generated, start=window.new_start, end=window.valid_length
                        )
                    payload = timeline.push(params, is_last=i == len(plan) - 1)
                    reply = sonic.append(payload)
                    committed.append(payload)
                    elapsed = time.monotonic() - began
                    decoded_chunks.append({k: v.detach().cpu().numpy() for k, v in params.items()})
                    row = dict(
                        window=i,
                        source_start=window.start,
                        source_new_frames=window.new_length,
                        production_seconds=elapsed,
                        during_playback=playing.is_set(),
                        committed_last_frame=reply["received_frame"],
                        buffer_seconds=reply["buffer_seconds"],
                    )
                    windows.append(row)
                    write_json(output / "windows.json", windows)
                tail = timeline.finish()
                sonic.append(tail)
                committed.append(tail)
                sonic.call("finish")
                prefilled.set()
            except Exception:
                worker_errors.append(traceback.format_exc())
                prefilled.set()

        producer = threading.Thread(target=produce, name="music-producer", daemon=True)
        producer.start()
        preparation_deadline = time.monotonic() + 120
        while True:
            processes.check()
            s, m = sonic.call("status"), sim.call("status")
            if worker_errors:
                raise RuntimeError(worker_errors[0])
            if quit_requested.is_set():
                raise RuntimeError("用户在起播前取消")
            if s["state"] == "fault" or m["state"] == "fault":
                raise RuntimeError(s.get("error") or m.get("error"))
            if prefilled.is_set() and s["control_ready"]:
                break
            if time.monotonic() > preparation_deadline:
                raise TimeoutError("动作预缓冲或控制器预热超时")
            time.sleep(0.05)
        epoch = time.monotonic_ns() + 800_000_000
        sonic.call("start", epoch_ns=epoch)
        sim.call("start", epoch_ns=epoch)
        audio.arm(epoch + 2_000_000_000)
        playing.set()
        manifest["epoch_ns"] = epoch
        write_json(output / "manifest.json", manifest)
        next_print, skew_since, finished = 0.0, None, False
        while True:
            processes.check()
            s, m = sonic.call("status", audit=args.audit_observations), sim.call("status")
            now = time.monotonic_ns()
            media = audio.position()
            if worker_errors:
                raise RuntimeError(worker_errors[0])
            if audio.error or s["state"] == "fault" or m["state"] == "fault":
                raise RuntimeError(audio.error or s.get("error") or m.get("error"))
            if now > epoch and now - s["control_ns"] > 100_000_000:
                raise RuntimeError("SONIC 实际控制输出超过 100 ms 未更新")
            active = s["used_frame"] >= 100 and 0 < media < duration and not user_stopped
            sync_ms = (
                ((s["used_frame"] - 100) / 50 - audio.position(s["control_ns"])) * 1000
                if active
                else 0.0
            )
            if active and abs(sync_ms) > 100:
                skew_since = skew_since or now
                if now - skew_since > 200_000_000:
                    raise RuntimeError("音频与实际参考帧持续偏差超过 100 ms")
            else:
                skew_since = None
            row = dict(
                monotonic_ns=now,
                audio_seconds=media,
                media_active=active,
                sync_ms=sync_ms,
                sonic=s,
                sim=m,
            )
            if gpu_handle is not None:
                row["gpu"] = dict(
                    utilization_pct=pynvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu,
                    memory_used_bytes=pynvml.nvmlDeviceGetMemoryInfo(gpu_handle).used,
                )
            rows.append(row)
            telemetry.write(json.dumps(row, ensure_ascii=False) + "\n")
            if time.monotonic() >= next_print:
                print(
                    f"music={media:.2f}/{duration:.2f}s buffer={s['buffer_seconds']:.2f}s mode=2 sync={sync_ms:+.1f}ms sim={m['skew_ms']:+.1f}ms state={s['state']}",
                    flush=True,
                )
                next_print = time.monotonic() + 0.5
            stop = quit_requested.is_set() or (
                args.stop_after is not None and media >= args.stop_after
            )
            if stop and not user_stopped and s["state"] != "finished":
                user_stopped = True
                producer_stop.set()
                producer.join(timeout=5)
                if producer.is_alive():
                    raise RuntimeError("用户停止时生成线程未能及时退出")
                s = sonic.call("status")
                cut = s["used_frame"] + 25
                if cut < s["received_frame"] - 59 and s["state"] == "playing":
                    tail = timeline.graceful_tail(cut)
                    sonic.call(
                        "stop", _pack_pose_message_compat(tail), graceful=True, start_frame=cut
                    )
                    committed[:] = [
                        {k: value[block["frame_index"] < cut] for k, value in block.items()}
                        for block in committed
                        if block["frame_index"][0] < cut
                    ]
                    committed.append(tail)
                    audio.fade(epoch + cut * 20_000_000)
                else:
                    audio.fade(now + 250_000_000)
            if s["state"] == "finished":
                if not finished:
                    full_control_metrics = sonic.call("status", metrics=True).get("control_metrics")
                    sim.call("finish")
                    audio.close()
                    finished = True
                    print("music completed; holding standing pose; Ctrl+C to close", flush=True)
                if args.exit_on_finish or user_stopped or quit_requested.is_set():
                    break
            time.sleep(0.025)
    except Exception:
        error = traceback.format_exc()
        print(error, file=sys.stderr, flush=True)
    finally:
        producer_stop.set()
        if audio and error:
            audio.close(fault=True)
        if prepared and error:
            for client in prepared:
                try:
                    client.call("stop", fault=True, reason=error.splitlines()[-1])
                except Exception:
                    pass
        if producer:
            producer.join(timeout=10)
        if audio:
            audio.close(fault=bool(error))
            if audio.fade_end_sample is not None:
                sf.write(output / "played_audio.wav", audio.pcm, audio.sample_rate, subtype="FLOAT")
        processes.close()
        for client in clients:
            client.close()
        telemetry.close()
        if decoded_chunks:
            np.savez_compressed(
                output / "generated_smpl.npz",
                **{
                    k: np.concatenate([chunk[k] for chunk in decoded_chunks])
                    for k in decoded_chunks[0]
                },
                fps=30,
            )
        if committed:
            np.savez_compressed(
                output / "reference_50hz.npz",
                **{k: np.concatenate([block[k] for block in committed]) for k in committed[0]},
                fps=50,
            )
        report = summarize(rows, windows, duration, args.audio_output, error, user_stopped)
        report["control_metrics"] = full_control_metrics
        report["generation_backend"] = "fixed_smpl" if args.smpl_npz else "tensorrt_fp16"
        if audio:
            report["audio_fade_end_seconds"] = (
                None if audio.fade_end_sample is None else audio.fade_end_sample / audio.sample_rate
            )
        if args.smpl_npz:
            report["generation_target_pass"] = None
        if gpu_handle is not None:
            report["gpu_peak_memory_bytes"] = max(
                (r["gpu"]["memory_used_bytes"] for r in rows), default=0
            )
            pynvml.nvmlShutdown()
        write_json(output / "report.json", report)
        signal.signal(signal.SIGINT, original_handler)
        print(
            f"report={output / 'report.json'} completed={report['completed']} error={bool(error)}",
            flush=True,
        )
    return 1 if error else 0


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
