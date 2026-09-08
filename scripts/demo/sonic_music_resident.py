"""GENMO 音乐部署的常驻控制台与站姿发布协调器。

本模块由 demo_music_sonic.py 的无音频启动或 --resident 选项调用，复用其生成、
声卡时钟、录制和三进程启动逻辑。整个运行期间只使用一个控制会话，不在换歌时
重启 SONIC、MuJoCo 或清空机器人的物理状态；TensorRT 模型也在歌曲之间复用。

独立心跳线程每 100 ms 发送十帧恒定 SMPL 站姿，并读取 MuJoCo 窗口事件。
默认站姿双臂外展 15 度；] 请求进入 SONIC，9 由仿真线程切换吊绳，P 停止表演。
终端可输入文件路径或 play 路径，stop 平滑收尾，status 查看状态，quit 退出。
输入线程只投递命令和停止事件，不执行生成或操作音频回调。无音乐、音乐准备、
自然结束以及主动停止之后均继续站姿闭环，只有退出或锁存故障才停止三个进程。

每首音乐保存独立报告，常驻目录保存公共进程日志、站姿状态和完整仿真日志。
歌曲日志按预约 epoch 从实际仿真日志提取，保留可复查的同步和控制证据。
"""

from __future__ import annotations

import copy
import json
import math
import queue
import shlex
import signal
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np

from gem.runtime.sonic_music import PoseTimeline, SessionClient
from gem.utils.sonic.zmq_publisher import _pack_pose_message_compat
from scripts.demo import demo_music_sonic as demo


class ResidentConsole:
    """分离歌曲执行、常驻站姿和用户输入，任何准备工作都不阻塞参考心跳。"""

    def __init__(self, args):
        if not math.isfinite(args.arm_open_degrees) or not 0 <= args.arm_open_degrees <= 45:
            raise ValueError("站姿手臂外展角必须位于 0 至 45 度")
        if not math.isfinite(args.hang_height) or not 0.78 <= args.hang_height <= 1.2:
            raise ValueError("常驻吊绳锚点高度必须位于 0.78 至 1.2 米")
        self.args = args
        self.session_id = str(uuid.uuid4())
        self.output = (
            args.output_dir
            or demo.ROOT
            / "outputs/sonic_music"
            / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_resident_" + self.session_id[:8])
        ).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.processes = demo.LocalProcesses()
        self.clients = [
            SessionClient(e, self.session_id) for e in (args.sonic_endpoint, args.sim_endpoint)
        ]
        self.sonic, self.sim = self.clients
        self.stop_requested, self.exit_requested, self.worker_stop = (
            threading.Event() for _ in range(3)
        )
        self.commands = queue.Queue(maxsize=8)
        self.command_lock = threading.RLock()
        self.busy, self.error, self.model = False, "", None
        self.standing_root = np.zeros(3)
        self.idle_builder = PoseTimeline(args.sonic_root, 1, arm_open_degrees=args.arm_open_degrees)
        self.idle_lock = threading.Lock()
        self.idle_binary = _pack_pose_message_compat(
            self.idle_builder.idle_packet(self.standing_root)
        )
        self.latest = None
        self.released_at = None
        self.worker = None
        self.initialized = []
        self.tracks = []
        self.sim_track_offset = 0

    def start(self):
        """建立空闲会话并开始站姿发送，等待用户起控及手动松绳。"""
        manifest = dict(
            session_id=self.session_id,
            resident=True,
            arm_open_degrees=self.args.arm_open_degrees,
            hang_height=self.args.hang_height,
            assets=demo.assets(self.args),
            argv=sys.argv,
        )
        if self.args.launch_local:
            self.processes.launch(self.args, self.output)
        manifest["processes"] = self.processes.commands
        demo.write_json(self.output / "manifest.json", manifest)
        deadline = time.monotonic() + 120
        for client in self.clients:
            probe = SessionClient(client.endpoint, timeout_ms=100)
            try:
                while True:
                    self.processes.check()
                    try:
                        reply = probe.call("status")
                        if reply.get("session_id"):
                            raise RuntimeError("端点已有会话，请退出其拥有者后再启动常驻控制台")
                        break
                    except TimeoutError:
                        if time.monotonic() > deadline or self.exit_requested.is_set():
                            raise RuntimeError("等待本地服务超时或用户退出") from None
                        self.worker_stop.wait(0.1)
            finally:
                probe.close()
        self.sonic.call("resident", self.idle_binary)
        self.initialized.append(self.sonic)
        self.sim.call("resident", hang_height=self.args.hang_height)
        self.initialized.append(self.sim)
        self.worker = threading.Thread(
            target=self._heartbeat, name="resident-standing", daemon=True
        )
        self.worker.start()
        print(f"resident={self.session_id} output={self.output}", flush=True)
        print("MuJoCo: ] = enable SONIC, 9 = release/toggle rope, P = stop performance", flush=True)
        print(
            "Console: play /path/song.wav | /path/song.wav | stop | status | ] | quit", flush=True
        )

    def _heartbeat(self):
        """持续刷新站姿、转发窗口事件并监测进程，避免歌曲准备阻断控制。"""
        enable_seen = stop_seen = 0
        next_stand = 0.0
        try:
            with (self.output / "resident_timeline.jsonl").open("x", encoding="utf-8") as log:
                while not self.worker_stop.is_set():
                    self.processes.check()
                    now = time.monotonic()
                    if now >= next_stand:
                        with self.idle_lock:
                            self.sonic.call("stand", self.idle_binary)
                        next_stand = now + 0.1
                    s, m = self.sonic.call("status"), self.sim.call("status")
                    if s["state"] == "fault" or m["state"] == "fault":
                        raise RuntimeError(s.get("error") or m.get("error"))
                    if m["enable_requests"] > enable_seen:
                        self.sonic.call("enable")
                        enable_seen = m["enable_requests"]
                        print("SONIC control requested; release rope with 9 in MuJoCo", flush=True)
                    if m["stop_requests"] > stop_seen:
                        self.request_stop()
                        stop_seen = m["stop_requests"]
                    if m["band_enabled"]:
                        self.released_at = None
                    elif self.released_at is None:
                        self.released_at = now
                    if s["control_ready"] and time.monotonic_ns() - s["control_ns"] > 100_000_000:
                        raise RuntimeError("SONIC 实际控制输出超过 100 ms 未更新")
                    self.latest = dict(
                        monotonic_ns=time.monotonic_ns(), sonic=s, sim=m, busy=self.busy
                    )
                    log.write(json.dumps(self.latest, ensure_ascii=False) + "\n")
                    self.worker_stop.wait(0.05)
        except Exception as exc:
            self.error = str(exc)
            self.stop_requested.set()
            print(f"resident fault: {exc}", file=sys.stderr, flush=True)
            for client in (self.sim, self.sonic):
                try:
                    client.call("stop", fault=True, reason=self.error)
                except Exception:
                    pass

    def ground_ready(self):
        """手动松绳后至少留一秒落地稳定，再允许预约音乐时间线。"""
        return self.released_at is not None and time.monotonic() - self.released_at >= 1

    def return_to_standing(self, root):
        """原子更新站姿朝向与控制器待机状态，不重启策略、不重置物理状态。"""
        with self.idle_lock:
            self.standing_root = np.asarray(root).copy()
            self.idle_binary = _pack_pose_message_compat(
                self.idle_builder.idle_packet(self.standing_root)
            )
            self.sonic.call("stand", self.idle_binary, return_to_idle=True)
        self.sim.call("finish")

    def save_sim_track(self, output, epoch):
        """从常驻状态日志提取本首音乐的真实状态，避免把其他歌曲混入评估。"""
        source = self.output / "sim_state.jsonl"
        if not epoch or not source.exists():
            return
        with (
            source.open(encoding="utf-8") as src,
            (output / "sim_state.jsonl").open("x", encoding="utf-8") as dst,
        ):
            src.seek(self.sim_track_offset)
            for line in src:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row["epoch_ns"] == epoch:
                    dst.write(line)

    def _read_commands(self):
        """阻塞式终端读取放在守护线程，stop/quit 直接设置正在表演的停止事件。"""
        for line in sys.stdin:
            command = line.strip()
            if not command:
                continue
            if command.lower() in ("stop", "p"):
                self.request_stop()
                if not self.busy:
                    print("already standing", flush=True)
            elif command.lower() in ("quit", "exit"):
                self.exit_requested.set()
                self.request_stop()
                return
            elif command in ("]", "9"):
                try:
                    self.sim.call("key", key=command)
                except Exception as exc:
                    print(f"input error: {exc}", flush=True)
            elif command.lower() == "status":
                self.print_status()
            elif self.busy:
                print("performance busy; use stop before selecting another file", flush=True)
            else:
                try:
                    path = Path(command).expanduser()
                    if not path.is_file():
                        words = shlex.split(command)
                        if words and words[0].lower() == "play":
                            words = words[1:]
                        if len(words) != 1:
                            raise ValueError("请输入完整文件路径；带空格的路径可加引号")
                        path = Path(words[0]).expanduser()
                    self.commands.put_nowait(path)
                except (ValueError, queue.Full) as exc:
                    print(f"input error: {exc}", flush=True)

    def request_stop(self):
        """停止也取消尚未开始的歌曲，避免 play 后立即 stop 时错过忙状态切换。"""
        with self.command_lock:
            self.stop_requested.set()
            while True:
                try:
                    self.commands.get_nowait()
                except queue.Empty:
                    break

    def print_status(self):
        """只显示用户操作所需状态，模型和协议诊断保留在文件日志中。"""
        if self.latest:
            s, m = self.latest["sonic"], self.latest["sim"]
            print(
                f"state={s['state']} control={'ON' if s['control_ready'] else 'WAIT_ ]'} rope={'ON' if m['band_enabled'] else 'OFF'} mode=2 idle_updates={s['idle_updates']}",
                flush=True,
            )

    def interrupt(self, *_):
        """表演期间 Ctrl+C 只请求收尾；待机时 Ctrl+C 退出常驻进程。"""
        if self.busy and not self.stop_requested.is_set():
            self.request_stop()
        else:
            self.exit_requested.set()
            self.request_stop()

    def serve(self):
        """顺序执行歌曲，同一 TensorRT 实例和三个进程跨歌曲保留。"""
        reader = threading.Thread(target=self._read_commands, name="resident-console", daemon=True)
        reader.start()
        if self.args.audio:
            self.commands.put(self.args.audio)
        while not self.exit_requested.is_set() and not self.error:
            try:
                with self.command_lock:
                    path = self.commands.get_nowait()
                    self.stop_requested.clear()
                    self.busy = True
            except queue.Empty:
                self.exit_requested.wait(0.1)
                continue
            if not path.is_file():
                print(f"audio file does not exist: {path}", flush=True)
                self.busy = False
                continue
            track = copy.copy(self.args)
            track.audio, track.resident = path.resolve(), True
            track.output_dir = (
                self.output / "tracks" / f"{len(self.tracks) + 1:04d}_{uuid.uuid4().hex[:8]}"
            )
            print(f"preparing={track.audio}; standing reference remains active", flush=True)
            try:
                source = self.output / "sim_state.jsonl"
                self.sim_track_offset = source.stat().st_size if source.exists() else 0
                code = demo.run(track, resident=self)
                self.tracks.append(dict(output=str(track.output_dir), returncode=code))
                demo.write_json(self.output / "tracks.json", self.tracks)
            finally:
                self.busy = False
                self.stop_requested.clear()
            self.print_status()
        return 1 if self.error else 0

    def close(self):
        """显式退出才停止常驻会话和本程序启动的子进程。"""
        self.worker_stop.set()
        if self.worker:
            self.worker.join(timeout=3)
        for client in reversed(self.initialized):
            try:
                client.call("stop", fault=bool(self.error), reason=self.error or "退出常驻控制台")
            except Exception:
                pass
        self.processes.close()
        for client in self.clients:
            client.close()
        demo.write_json(
            self.output / "resident_report.json",
            dict(session_id=self.session_id, error=self.error, tracks=self.tracks),
        )


def run_console(args):
    """常驻入口保证异常和退出均恢复信号处理器并关闭本程序拥有的资源。"""
    demo.torch.set_num_threads(2)
    console = ResidentConsole(args)
    handler = signal.signal(signal.SIGINT, console.interrupt)
    try:
        console.start()
        return console.serve()
    except Exception as exc:
        console.error = console.error or str(exc)
        print(f"resident error: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        console.close()
        signal.signal(signal.SIGINT, handler)
