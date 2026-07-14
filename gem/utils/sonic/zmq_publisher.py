# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Asynchronous Protocol v3 publisher for SONIC pose streaming.

The SONIC subscriber expects one packed ZMQ message with this layout::

    [topic bytes][1280-byte JSON header][concatenated NumPy payloads]

``pack_pose_message`` from GR00T-WholeBodyControl is used when that package is
importable.  The local fallback mirrors its current Protocol v3 wire format.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from gem.utils.sonic.smpl_converter import SonicSMPLConverter

_HEADER_SIZE = 1280
_STOP = object()


def _pack_pose_message_compat(
    pose_data: Mapping[str, np.ndarray], topic: str = "pose", version: int = 3
) -> bytes:
    """Pack a message exactly like SONIC's ``pack_pose_message`` helper."""
    fields: list[dict[str, Any]] = []
    payloads: list[bytes] = []

    dtype_names = {
        np.dtype(np.float32): "f32",
        np.dtype(np.float64): "f64",
        np.dtype(np.int32): "i32",
        np.dtype(np.int64): "i64",
        np.dtype(np.bool_): "bool",
    }

    for name, value in pose_data.items():
        if not isinstance(value, np.ndarray):
            continue

        dtype_name = dtype_names.get(value.dtype)
        if dtype_name is None:
            value = value.astype(np.float32)
            dtype_name = "f32"

        # SONIC declares the payload little-endian in the JSON header.
        if value.dtype.byteorder == ">":
            value = value.astype(value.dtype.newbyteorder("<"))
        value = np.ascontiguousarray(value)

        fields.append({"name": name, "dtype": dtype_name, "shape": list(value.shape)})
        payloads.append(value.tobytes())

    header = {
        "v": version,
        "endian": "le",
        # The upstream v3 packer uses count=1 even for temporal chunks; the
        # receiver obtains the actual frame count from each field's shape.
        "count": 1,
        "fields": fields,
    }
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(header_json) > _HEADER_SIZE:
        raise ValueError(f"SONIC header too large: {len(header_json)} > {_HEADER_SIZE}")

    return topic.encode("utf-8") + header_json.ljust(_HEADER_SIZE, b"\x00") + b"".join(payloads)


def _resolve_pose_packer():
    """Prefer GR00T-WholeBodyControl's packer when it is on ``PYTHONPATH``."""
    try:
        from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message
    except ImportError:
        return _pack_pose_message_compat
    return pack_pose_message


class SonicPublisher:
    """Non-blocking GEM-SMPL to SONIC Protocol v3 publisher.

    ``publish_smpl`` is the producer-side API.  It only enqueues references to
    the already detached webcam outputs.  SONIC forward kinematics, SciPy
    quaternion conversion, packing, and ZMQ send all run on the consumer
    thread, keeping them off the GEM inference path.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5556,
        topic: str = "pose",
        queue_size: int = 2,
        sonic_repo_path: str | Path | None = None,
    ) -> None:
        if not host:
            raise ValueError("host must not be empty")
        if not 1 <= int(port) <= 65535:
            raise ValueError(f"port must be in [1, 65535], got {port}")
        if not topic:
            raise ValueError("topic must not be empty")
        if queue_size < 1:
            raise ValueError("queue_size must be at least 1")

        self.host = host
        self.port = int(port)
        self.topic = topic
        self.endpoint = f"tcp://{host}:{self.port}"
        if sonic_repo_path is None:
            sonic_repo_path = Path(__file__).resolve().parents[4] / "GR00T-WholeBodyControl"
        self.sonic_repo_path = Path(sonic_repo_path).expanduser()

        self._queue: queue.Queue[Any] = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._state_lock = threading.Lock()
        self._state = "disconnected"
        self._thread: threading.Thread | None = None
        self._worker_error: BaseException | None = None
        self._connected = False

        self._smpl_converter: SonicSMPLConverter | None = None
        self._next_frame_index = 0
        self._frames_sent = 0
        self._frames_dropped = 0
        self._next_log_frame = 100
        self._started_at = 0.0

    def connect(self) -> None:
        """Bind the PUB endpoint and start the background consumer thread.

        SONIC owns the SUB socket and connects to the publisher, so despite the
        public lifecycle method name, the ZMQ operation here must be ``bind``.
        """
        with self._state_lock:
            if self._state == "connected":
                return
            if self._state != "disconnected":
                raise RuntimeError(f"SONIC publisher is {self._state}")
            self._state = "starting"

        # A previous close may leave its sentinel in the queue if the worker
        # observed the stop event first. Start each connection with a clean
        # real-time stream and a fresh Protocol v3 frame sequence.
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._next_frame_index = 0
        self._frames_sent = 0
        self._frames_dropped = 0
        self._next_log_frame = 100

        try:
            import zmq  # Imported lazily so the original webcam demo stays independent.
        except ImportError as exc:
            with self._state_lock:
                self._state = "disconnected"
            raise RuntimeError(
                "SONIC streaming requires pyzmq. Install the project dependencies "
                "or run: pip install pyzmq"
            ) from exc

        # Register and start the worker while holding the lifecycle lock. This
        # closes the narrow window in which close() could otherwise return for
        # a not-yet-registered startup and a second connect() could begin.
        with self._state_lock:
            if self._state != "starting":
                if self._state == "closing" and self._thread is None:
                    self._state = "disconnected"
                raise RuntimeError("SONIC publisher startup was cancelled")
            self._stop_event.clear()
            self._ready_event.clear()
            self._worker_error = None
            self._thread = threading.Thread(
                target=self._worker_loop,
                args=(zmq,),
                name="sonic-zmq-publisher",
                daemon=True,
            )
            try:
                self._thread.start()
            except BaseException:
                self._thread = None
                self._state = "disconnected"
                raise

        # Startup is outside the inference loop. Waiting here makes bind/converter
        # failures visible immediately instead of silently losing every frame.
        if not self._ready_event.wait(timeout=30.0):
            self.close()
            raise RuntimeError("Timed out while starting the SONIC publisher")
        with self._state_lock:
            thread_alive = self._thread is not None and self._thread.is_alive()
            can_connect = (
                self._state == "starting"
                and not self._stop_event.is_set()
                and self._worker_error is None
                and thread_alive
            )
            if can_connect:
                self._state = "connected"
                self._connected = True
                self._started_at = 0.0
            error = self._worker_error

        if not can_connect:
            self.close()
            if error is not None:
                raise RuntimeError(f"Failed to start SONIC publisher at {self.endpoint}") from error
            raise RuntimeError("SONIC publisher startup was cancelled")
        print("[SONIC] ZMQ publisher started")

    def publish_smpl(
        self,
        body_params_incam: Mapping[str, Any],
        body_params_global: Mapping[str, Any],
    ) -> bool:
        """Queue one SMPL frame/chunk without waiting for the consumer.

        When the bounded queue is full, its oldest unsent item is discarded so
        real-time inference never waits behind stale poses.
        """
        body_pose = body_params_global.get("body_pose")
        if body_pose is None:
            body_pose = body_params_incam.get("body_pose")
        if body_pose is None:
            raise KeyError("body_pose is required in body parameters")
        num_values = (
            body_pose.numel() if isinstance(body_pose, torch.Tensor) else np.size(body_pose)
        )
        if num_values == 0 or num_values % 63:
            raise ValueError(f"body_pose must contain a multiple of 63 values, got {num_values}")
        num_frames = num_values // 63

        with self._state_lock:
            thread_alive = self._thread is not None and self._thread.is_alive()
            if self._state != "connected" or not thread_alive or self._stop_event.is_set():
                return False
            frame_index = np.arange(
                self._next_frame_index,
                self._next_frame_index + num_frames,
                dtype=np.int64,
            )
            self._next_frame_index += num_frames

        item = (dict(body_params_incam), dict(body_params_global), frame_index)
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            try:
                dropped_item = self._queue.get_nowait()
                if dropped_item is not _STOP:
                    with self._state_lock:
                        self._frames_dropped += len(dropped_item[2])
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                with self._state_lock:
                    self._frames_dropped += num_frames
                return False
        return True

    def close(self, timeout: float = 5.0) -> bool:
        """Stop the consumer and release its ZMQ resources."""
        with self._state_lock:
            previous_state = self._state
            self._connected = False
            self._stop_event.set()
            thread = self._thread
            if thread is None:
                # Keep a not-yet-registered connect() cancelled until that
                # caller observes the state; this prevents a second connect
                # from reusing its startup events in the meantime.
                self._state = (
                    "closing" if previous_state in {"starting", "closing"} else "disconnected"
                )
                return True
            self._state = "closing"

        try:
            self._queue.put_nowait(_STOP)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(_STOP)
            except queue.Full:
                pass

        if threading.current_thread() is not thread:
            thread.join(timeout=timeout)

        stopped = not thread.is_alive()
        with self._state_lock:
            if stopped:
                if self._thread is thread:
                    self._thread = None
                self._state = "disconnected"
            else:
                self._state = "closing"
        if not stopped:
            print("[SONIC] publisher shutdown timed out")
        return stopped

    def _worker_loop(self, zmq) -> None:
        context = None
        socket = None
        try:
            context = zmq.Context()
            socket = context.socket(zmq.PUB)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.SNDHWM, self._queue.maxsize)
            socket.bind(self.endpoint)

            self._smpl_converter = SonicSMPLConverter(self.sonic_repo_path)
            print("[SONIC] Using SONIC compute_human_joints")
            pack_pose_message = _resolve_pose_packer()
        except BaseException as exc:
            self._worker_error = exc
            self._ready_event.set()
            if socket is not None:
                socket.close(linger=0)
            if context is not None:
                context.term()
            self._mark_worker_stopped()
            return

        self._ready_event.set()

        try:
            while not self._stop_event.is_set():
                try:
                    item = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is _STOP:
                    break

                try:
                    pose_data = self._make_pose_data(*item)
                    if self._stop_event.is_set():
                        break
                    message = pack_pose_message(pose_data, topic=self.topic, version=3)
                    if self._stop_event.is_set():
                        break
                    socket.send(message, flags=zmq.NOBLOCK)
                    self._report_sent(pose_data)
                except zmq.Again:
                    # A slow/no subscriber must never back-pressure inference.
                    continue
                except Exception as exc:
                    self._worker_error = exc
                    print(f"\n[SONIC] publisher error: {exc}")
        except BaseException as exc:
            self._worker_error = exc
            print(f"\n[SONIC] publisher stopped: {exc}")
        finally:
            socket.close(linger=0)
            context.term()
            self._smpl_converter = None
            self._mark_worker_stopped()

    def _mark_worker_stopped(self) -> None:
        """Publish a terminal worker state without racing connect/close."""
        current_thread = threading.current_thread()
        with self._state_lock:
            self._connected = False
            if self._thread is current_thread:
                self._thread = None
            self._state = "disconnected"

    @torch.inference_mode()
    def _make_pose_data(
        self,
        body_params_incam: Mapping[str, Any],
        body_params_global: Mapping[str, Any],
        frame_index: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        if self._smpl_converter is None:
            raise RuntimeError("SONIC SMPL converter is not initialized")

        converted = self._smpl_converter.convert(
            body_params_global=body_params_global,
            body_params_incam=body_params_incam,
        )
        smpl_pose = converted["smpl_pose"]
        smpl_joints_local = converted["smpl_joints_local"]
        global_orient = converted["global_orient"]
        num_frames = len(smpl_pose)

        # SciPy emits scalar-last [x, y, z, w]; SONIC requires [w, x, y, z].
        quat_xyzw = Rotation.from_rotvec(global_orient.numpy()).as_quat()
        body_quat = quat_xyzw[:, [3, 0, 1, 2]].astype(np.float32, copy=False)

        if frame_index is None:
            with self._state_lock:
                frame_index = np.arange(
                    self._next_frame_index,
                    self._next_frame_index + num_frames,
                    dtype=np.int64,
                )
                self._next_frame_index += num_frames
        else:
            frame_index = np.ascontiguousarray(frame_index, dtype=np.int64).reshape(-1)
            if len(frame_index) != num_frames:
                raise ValueError(
                    f"frame_index has {len(frame_index)} frames; expected {num_frames}"
                )

        return {
            "smpl_pose": np.ascontiguousarray(smpl_pose.cpu().numpy(), dtype=np.float32),
            "smpl_joints": np.ascontiguousarray(smpl_joints_local.cpu().numpy(), dtype=np.float32),
            "body_quat": np.ascontiguousarray(body_quat, dtype=np.float32),
            # Protocol v3 requires G1 qpos/qvel. GEM supplies SMPL rather than
            # retargeted G1 joints, so these valid placeholders stay zero; the
            # SMPL fields carry the primary whole-body motion.
            "joint_pos": np.zeros((num_frames, 29), dtype=np.float32),
            "joint_vel": np.zeros((num_frames, 29), dtype=np.float32),
            "frame_index": frame_index,
        }

    def _report_sent(self, pose_data: Mapping[str, np.ndarray]) -> None:
        if self._frames_sent == 0:
            self._started_at = time.perf_counter()
        self._frames_sent += len(pose_data["frame_index"])
        if self._frames_sent < self._next_log_frame:
            return

        while self._next_log_frame <= self._frames_sent:
            self._next_log_frame += 100

        elapsed = max(time.perf_counter() - self._started_at, 1e-6)
        smpl_joints = pose_data["smpl_joints"]
        pelvis = smpl_joints[-1, SonicSMPLConverter.PELVIS_INDEX]
        left_wrist = smpl_joints[-1, SonicSMPLConverter.LEFT_WRIST_INDEX]
        right_wrist = smpl_joints[-1, SonicSMPLConverter.RIGHT_WRIST_INDEX]
        smpl_pose_shape = f"({','.join(map(str, pose_data['smpl_pose'].shape))})"
        smpl_joints_shape = f"({','.join(map(str, smpl_joints.shape))})"
        print(
            "\n[SONIC]\n"
            f"fps={self._frames_sent / elapsed:.1f}\n"
            f"frames={self._frames_sent}\n"
            f"smpl_pose_shape={smpl_pose_shape}\n"
            f"smpl_joint_shape={smpl_joints_shape}\n"
            f"smpl_pose:\nshape={smpl_pose_shape}\n"
            f"smpl_joints:\nshape={smpl_joints_shape}\n"
            f"pelvis position={np.array2string(pelvis, precision=4, separator=', ')}\n"
            f"left_wrist position={np.array2string(left_wrist, precision=4, separator=', ')}\n"
            f"right_wrist position={np.array2string(right_wrist, precision=4, separator=', ')}"
        )

    def __enter__(self) -> SonicPublisher:
        self.connect()
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()
