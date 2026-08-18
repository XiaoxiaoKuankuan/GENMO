#!/usr/bin/env python3
"""通过真实 GMR-CPP 服务离线捕获完整的 BUMI3 重定向动作。

本脚本面向长于 Redis 默认 512 帧窗口的 GENMO SMPL-X 动作。它读取仓库标准
``smpl_params.pt``，逐帧执行与实时机器人链路完全相同的 SMPL-X FK 和 SMP1 UDP
发送，再启动 ``run_smplx_bumi3.sh``，让指定 IK JSON 与 BUMI3 MuJoCo 模型完成
真实 C++ GMR 求解。脚本不会在 Python 中仿写 IK，也不会修改输入动作、IK 配置、
机器人 XML 或 GMR-CPP 可执行文件。

GMR-CPP 的 Redis stream 固定使用近似 ``MAXLEN 512``。如果在动作结束后才读取，
长舞蹈的前半段会被静默淘汰。本实现从正式播放开始就用独立线程持续 XREAD，保存
所有已经发布的二进制帧；结束后再按照发送端墙钟时间轴插值为与源动作等长的
``qpos.npy [T,28]``。根节点四元数使用最短路径 SLERP，平移与 21 个关节角使用
线性插值。程序还检查 stream 首尾时间覆盖、原始帧数量、有限值、四元数单位范数
和输出帧数，任何不完整捕获都会失败而不会伪装成有效视频输入。

输出目录包含：原始 Redis 帧 ``gmr_stream_raw.npz``、MuJoCo 原生顺序的
``qpos.npy``、配置哈希与时间轴诊断 ``retarget_metadata.json``，以及 GMR 子进程
日志 ``gmr_run.log``。建议为批处理指定隔离 UDP 端口和唯一 Redis key；脚本只会
清理由自己管理的 key、stream 与 raw-bones key，不会停止其他 GMR Viewer。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import redis
import torch
from scipy.spatial.transform import Rotation, Slerp

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GMR_ROOT = Path("/home/weili/GMR-CPP_e1jump_lowdpi")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gem.gmr_udp_bridge import GMRUDPBridge  # noqa: E402
from gem.runtime.motion_streamer import frame_from_motion, load_smpl_motion  # noqa: E402
from gem.smplx_gmr_reference import SMPLXGMRReference  # noqa: E402
from scripts.demo.stream_smpl_params_to_gmr import (  # noqa: E402
    load_endecoder,
    send_frame_to_gmr,
)


def build_parser() -> argparse.ArgumentParser:
    """构造离线 GMR 捕获命令行。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--port", type=int, default=17006)
    parser.add_argument("--redis-key", default="genmo_gmr_capture")
    parser.add_argument("--redis-host", default="127.0.0.1")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--redis-db", type=int, default=0)
    parser.add_argument("--gmr-root", type=Path, default=DEFAULT_GMR_ROOT)
    parser.add_argument("--ik-config", type=Path)
    parser.add_argument("--robot-xml", type=Path)
    parser.add_argument("--ground-clearance", type=float, default=0.05)
    return parser


def sha256(path: Path) -> str:
    """流式计算文件 SHA-256，避免一次性读取大型 XML 依赖。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stop_process(process: subprocess.Popen[str]) -> None:
    """按 SIGINT、SIGTERM、SIGKILL 顺序收回本脚本启动的进程组。"""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=5.0)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=2.0)


class IncrementalStreamCapture:
    """在 Redis MAXLEN 淘汰之前持续消费全部 GMR 二进制帧。"""

    def __init__(self, client: redis.Redis, stream_key: str) -> None:
        self.client = client
        self.stream_key = stream_key
        self._entries: list[tuple[bytes, dict[bytes, bytes]]] = []
        self._last_id = "0-0"
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._finished_entries: list[tuple[bytes, dict[bytes, bytes]]] | None = None

    def start(self) -> None:
        self._thread.start()

    def _append(self, entries: list[tuple[bytes, dict[bytes, bytes]]]) -> None:
        if not entries:
            return
        with self._lock:
            self._entries.extend(entries)
            self._last_id = entries[-1][0].decode("ascii")

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                response = self.client.xread(
                    {self.stream_key: self._last_id}, count=256, block=100
                )
                for _, entries in response:
                    self._append(entries)
        except BaseException as exc:  # 异常由主线程重新抛出并执行统一清理。
            self._error = exc

    def finish(self) -> list[tuple[bytes, dict[bytes, bytes]]]:
        """停止阻塞读取、补收最后一批帧并返回按 Redis ID 排序的结果。"""
        if self._finished_entries is not None:
            return list(self._finished_entries)
        self._stop.set()
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            raise TimeoutError("Redis XREAD capture thread did not stop")
        if self._error is not None:
            raise RuntimeError("Redis XREAD capture failed") from self._error
        with self._lock:
            last_id = self._last_id
        # 部分 Redis/redis-py 组合不接受 ``(id`` 排他 XRANGE 语法。使用兼容的
        # 包含式查询，再移除已经由 XREAD 收到的最后一个 ID。
        last_id_bytes = last_id.encode("ascii")
        tail = [
            entry
            for entry in self.client.xrange(self.stream_key, min=last_id, max="+")
            if entry[0] != last_id_bytes
        ]
        self._append(tail)
        with self._lock:
            entries = list(self._entries)
        entries.sort(key=lambda item: tuple(int(value) for value in item[0].split(b"-")))
        identifiers = [identifier for identifier, _ in entries]
        if len(identifiers) != len(set(identifiers)):
            raise RuntimeError("incremental Redis capture contains duplicate IDs")
        self._finished_entries = entries
        return list(entries)


def resample_qpos(
    times_ms: np.ndarray,
    qpos: np.ndarray,
    target_times_ms: np.ndarray,
) -> np.ndarray:
    """把异步 Redis 帧重采样到源动作时间轴，旋转采用单位四元数 SLERP。"""
    order = np.argsort(times_ms, kind="stable")
    times_ms = times_ms[order]
    qpos = qpos[order]
    unique_times, unique_indices = np.unique(times_ms, return_index=True)
    times_ms = unique_times
    qpos = qpos[unique_indices]
    if len(times_ms) < 2:
        raise RuntimeError("GMR capture has fewer than two distinct Redis timestamps")

    # Redis 发布 tick 与发送 tick 相互独立，首尾允许存在不超过校验阈值的小相位差。
    # 只在已验证的首尾容差内钳制采样时刻，避免 Slerp 做未定义的外推。
    sample_times_ms = np.clip(target_times_ms, times_ms[0], times_ms[-1])
    output = np.empty((len(target_times_ms), 28), dtype=np.float64)
    for column in list(range(3)) + list(range(7, 28)):
        output[:, column] = np.interp(sample_times_ms, times_ms, qpos[:, column])

    quat_wxyz = np.asarray(qpos[:, 3:7], dtype=np.float64).copy()
    quat_wxyz /= np.linalg.norm(quat_wxyz, axis=1, keepdims=True).clip(1e-12)
    for index in range(1, len(quat_wxyz)):
        if float(np.dot(quat_wxyz[index - 1], quat_wxyz[index])) < 0.0:
            quat_wxyz[index] *= -1.0
    rotations = Rotation.from_quat(quat_wxyz[:, [1, 2, 3, 0]])
    out_xyzw = Slerp(times_ms, rotations)(sample_times_ms).as_quat()
    output[:, 3:7] = out_xyzw[:, [3, 0, 1, 2]]
    output[:, 3:7] /= np.linalg.norm(output[:, 3:7], axis=1, keepdims=True).clip(1e-12)
    return output.astype(np.float32)


def validate_paths(args: argparse.Namespace) -> dict[str, Path]:
    """解析并验证 GMR 启动脚本、IK、XML 与机器人顺序表。"""
    gmr_root = args.gmr_root.expanduser().resolve(strict=True)
    paths = {
        "root": gmr_root,
        "script": (gmr_root / "run_smplx_bumi3.sh").resolve(strict=True),
        "xml": (
            args.robot_xml
            if args.robot_xml is not None
            else gmr_root / "assets/bumi3/mjcf/bumi3.xml"
        ).expanduser().resolve(strict=True),
        "config": (
            args.ik_config
            if args.ik_config is not None
            else gmr_root / "config/ik_configs/smplx_to_bumi3_auto.json"
        ).expanduser().resolve(strict=True),
        "preset": (
            gmr_root / "config/robot_presets/bumi3.json"
        ).resolve(strict=True),
    }
    if not args.motion.expanduser().is_file():
        raise FileNotFoundError(args.motion)
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be in [1, 65535]")
    if not np.isfinite(args.fps) or args.fps <= 0:
        raise ValueError("--fps must be finite and positive")
    if not np.isfinite(args.ground_clearance) or args.ground_clearance < 0:
        raise ValueError("--ground-clearance must be finite and >= 0")
    if not args.redis_key or any(char.isspace() for char in args.redis_key):
        raise ValueError("--redis-key must be non-empty and contain no whitespace")
    return paths


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = validate_paths(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_log = args.output_dir / "gmr_run.log"

    client = redis.Redis(
        host=args.redis_host,
        port=args.redis_port,
        db=args.redis_db,
        socket_timeout=2.0,
    )
    if not client.ping():
        raise RuntimeError("Redis did not respond to PING")

    motion = load_smpl_motion(args.motion, shape_mode="zero")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    endecoder = load_endecoder(device)
    adapter = SMPLXGMRReference(user_yaw_deg=0.0, global_scale=1.0)
    bridge = GMRUDPBridge("127.0.0.1", args.port, debug=False)

    command = [
        str(paths["script"]),
        "--redis",
        "--always",
        "--port",
        str(args.port),
        "--redis-host",
        args.redis_host,
        "--redis-port",
        str(args.redis_port),
        "--redis-db",
        str(args.redis_db),
        "--redis-key",
        args.redis_key,
        "--ik-config",
        str(paths["config"]),
        "--xml",
        str(paths["xml"]),
        "--hz",
        f"{args.fps:g}",
        "--ttl-ms",
        "1000",
        "--stale-ms",
        "150",
        "--offset-to-ground",
        "--ground-clearance",
        f"{args.ground_clearance:g}",
    ]
    log_handle = run_log.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=paths["root"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )

    def copy_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            log_handle.write(line)
            log_handle.flush()

    output_thread = threading.Thread(target=copy_output, daemon=True)
    output_thread.start()
    stream_key = f"{args.redis_key}:stream"
    managed_keys = (args.redis_key, stream_key, f"{args.redis_key}_raw_bones")
    capture: IncrementalStreamCapture | None = None
    try:
        client.delete(*managed_keys)
        time.sleep(0.25)
        first_frame = frame_from_motion(motion, 0)
        warm_start = time.monotonic()
        warm_index = 0
        while client.xlen(stream_key) < 5:
            if process.poll() is not None:
                raise RuntimeError(
                    f"GMR exited during warm-up with code {process.returncode}"
                )
            if time.monotonic() - warm_start > 8.0:
                raise TimeoutError("GMR did not publish frames after 8 seconds of warm-up")
            delay = warm_start + warm_index / args.fps - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            send_frame_to_gmr(
                first_frame,
                endecoder,
                adapter,
                bridge,
                device=device,
                timestamp_ns=time.monotonic_ns(),
            )
            warm_index += 1

        time.sleep(0.1)
        client.delete(*managed_keys)
        capture = IncrementalStreamCapture(client, stream_key)
        capture.start()
        actual_start_ms = time.time_ns() / 1_000_000.0
        send_start = time.monotonic()
        for index in range(motion.num_frames):
            delay = send_start + index / args.fps - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            send_frame_to_gmr(
                frame_from_motion(motion, index),
                endecoder,
                adapter,
                bridge,
                device=device,
                timestamp_ns=time.monotonic_ns(),
            )
        actual_end_ms = time.time_ns() / 1_000_000.0
        time.sleep(0.35)

        entries = capture.finish()
        capture = None
        if not entries:
            raise RuntimeError("GMR Redis stream is empty after motion playback")
        redis_ids_ms = np.asarray(
            [float(identifier.split(b"-", 1)[0]) for identifier, _ in entries],
            dtype=np.float64,
        )
        payloads = np.stack(
            [np.frombuffer(fields[b"frame"], dtype="<f4") for _, fields in entries]
        )
        if payloads.ndim != 2 or payloads.shape[1] != 35:
            raise RuntimeError(f"Expected GMR frames [N,35], got {payloads.shape}")
        if not np.isfinite(payloads).all():
            raise RuntimeError("GMR output contains NaN or Inf")

        preset: dict[str, Any] = json.loads(paths["preset"].read_text(encoding="utf-8"))
        joint_map = np.asarray(preset["joint_ids_map"], dtype=np.int64)
        qpos_raw = np.empty((len(payloads), 28), dtype=np.float32)
        qpos_raw[:, :7] = payloads[:, 1:8]
        qpos_raw[:, 7 + joint_map] = payloads[:, 14:35]

        target_times_ms = actual_start_ms + (
            np.arange(motion.num_frames, dtype=np.float64) * 1000.0 / args.fps
        )
        tolerance_ms = max(150.0, 3.0 * 1000.0 / args.fps)
        start_lag_ms = float(redis_ids_ms[0] - target_times_ms[0])
        end_lead_ms = float(redis_ids_ms[-1] - target_times_ms[-1])
        if start_lag_ms > tolerance_ms or end_lead_ms < -tolerance_ms:
            raise RuntimeError(
                "GMR stream does not cover source timeline: "
                f"start_lag={start_lag_ms:.1f}ms end_lead={end_lead_ms:.1f}ms"
            )
        minimum_raw_frames = max(2, int(np.floor(motion.num_frames * 0.90)))
        if len(payloads) < minimum_raw_frames:
            raise RuntimeError(
                f"GMR capture has only {len(payloads)} raw frames; "
                f"expected at least {minimum_raw_frames}"
            )

        qpos = resample_qpos(redis_ids_ms, qpos_raw, target_times_ms)
        quat_norm = np.linalg.norm(qpos[:, 3:7], axis=1)
        if tuple(qpos.shape) != (motion.num_frames, 28):
            raise RuntimeError(f"Unexpected BUMI3 qpos shape: {qpos.shape}")
        if not np.isfinite(qpos).all() or not np.allclose(quat_norm, 1.0, atol=1e-4):
            raise RuntimeError("Resampled BUMI3 qpos failed finite/quaternion validation")

        exact_changes = np.r_[
            True,
            np.any(qpos_raw[1:].view(np.uint32) != qpos_raw[:-1].view(np.uint32), axis=1),
        ]
        np.save(args.output_dir / "qpos.npy", qpos)
        np.savez_compressed(
            args.output_dir / "gmr_stream_raw.npz",
            redis_ids_ms=redis_ids_ms,
            payloads=payloads,
            qpos_mujoco_order=qpos_raw,
            exact_pose_change=exact_changes,
        )
        metadata = {
            "source_motion": str(args.motion.expanduser().resolve()),
            "source_frames": motion.num_frames,
            "source_fps": motion.fps,
            "output_frames": int(qpos.shape[0]),
            "output_fps": args.fps,
            "raw_redis_frames": int(len(payloads)),
            "raw_exact_pose_changes": int(exact_changes.sum()),
            "timeline_start_lag_ms": start_lag_ms,
            "timeline_end_lead_ms": end_lead_ms,
            "timeline_tolerance_ms": tolerance_ms,
            "actual_start_epoch_ms": actual_start_ms,
            "actual_end_epoch_ms": actual_end_ms,
            "redis_first_epoch_ms": float(redis_ids_ms[0]),
            "redis_last_epoch_ms": float(redis_ids_ms[-1]),
            "gmr_command": command,
            "gmr_script": str(paths["script"]),
            "gmr_xml": str(paths["xml"]),
            "gmr_ik_config": str(paths["config"]),
            "gmr_grounding": (
                f"offset_to_ground, clearance={args.ground_clearance:g}m"
            ),
            "gmr_udp_port": args.port,
            "gmr_redis_key": args.redis_key,
            "gmr_script_sha256": sha256(paths["script"]),
            "gmr_xml_sha256": sha256(paths["xml"]),
            "gmr_ik_config_sha256": sha256(paths["config"]),
            "joint_names_mujoco_qpos_order": preset[
                "joint_names_mujoco_qpos_order"
            ],
            "joint_ids_map_from_publish_to_mujoco": preset["joint_ids_map"],
            "quaternion_convention": "wxyz",
            "capture_method": "incremental Redis XREAD, resampled on sender wall clock",
            "timeline_coverage_pass": True,
        }
        (args.output_dir / "retarget_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            f"[Retarget] {args.motion.name}: raw={len(payloads)} "
            f"changes={int(exact_changes.sum())} output={len(qpos)} "
            f"coverage=({start_lag_ms:.1f},{end_lead_ms:.1f})ms"
        )
    finally:
        if capture is not None:
            capture.finish()
        bridge.close()
        stop_process(process)
        output_thread.join(timeout=1.0)
        log_handle.close()
        client.delete(*managed_keys)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
