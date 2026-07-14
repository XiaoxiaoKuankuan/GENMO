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

import copy
import json
import queue
import threading
import time
import warnings
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from gem.utils.sonic.resampler import SMPLRealtimeResampler
from gem.utils.sonic.smpl_converter import InvalidSMPLFrameError, SonicSMPLConverter

_HEADER_SIZE = 1280
_STOP = object()
_TARGET_FPS = 50
_WINDOW_SIZE = 5
_SEND_LOOKAHEAD = 2
_PENDING_WINDOW_LIMIT = 2
_MAX_BAD_FRAME_SNAPSHOTS = 3


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


def _elbow_swing_euler(
    elbow_axis_angle: np.ndarray,
    decompose_rotation_aa,
) -> np.ndarray:
    """Return Pico-compatible elbow swing Euler angles with a zero-pose guard."""
    elbow_axis_angle = np.asarray(elbow_axis_angle, dtype=np.float64).reshape(-1, 3)
    swing_euler = np.zeros_like(elbow_axis_angle)
    valid = np.linalg.norm(elbow_axis_angle, axis=1) > 1e-8
    if not np.any(valid):
        return swing_euler

    _, swing_wxyz = decompose_rotation_aa(
        elbow_axis_angle[valid],
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        swing_euler[valid] = Rotation.from_quat(swing_wxyz[:, [1, 2, 3, 0]]).as_euler(
            "XYZ", degrees=False
        )
    return swing_euler


def _compute_g1_wrist_joint_pos(
    smpl_pose: np.ndarray,
    decompose_rotation_aa,
) -> np.ndarray:
    """Map Pico's SMPL elbow/wrist rotations into the six G1 wrist DoFs."""
    pose = np.asarray(smpl_pose, dtype=np.float64)
    if pose.ndim == 2:
        pose = pose[None]
    if pose.ndim != 3 or pose.shape[1:] != (21, 3):
        raise ValueError(f"smpl_pose must have shape [N,21,3], got {pose.shape}")

    left_elbow_swing = _elbow_swing_euler(pose[:, 17], decompose_rotation_aa)
    right_elbow_swing = _elbow_swing_euler(pose[:, 18], decompose_rotation_aa)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        left_wrist_euler = Rotation.from_rotvec(pose[:, 19]).as_euler("XYZ", degrees=False)
        right_wrist_euler = Rotation.from_rotvec(pose[:, 20]).as_euler("XYZ", degrees=False)

    joint_pos = np.zeros((len(pose), 29), dtype=np.float32)
    joint_pos[:, 23] = left_elbow_swing[:, 0] + left_wrist_euler[:, 0]
    joint_pos[:, 25] = left_wrist_euler[:, 1]
    joint_pos[:, 27] = left_elbow_swing[:, 2] + left_wrist_euler[:, 2]
    joint_pos[:, 24] = -(right_elbow_swing[:, 0] + right_wrist_euler[:, 0])
    joint_pos[:, 26] = -right_wrist_euler[:, 1]
    joint_pos[:, 28] = right_elbow_swing[:, 2] + right_wrist_euler[:, 2]
    if not np.isfinite(joint_pos).all():
        raise ValueError("G1 wrist mapping produced non-finite joint positions")
    return joint_pos


def _cpu_debug_copy(value: Any) -> Any:
    """Recursively snapshot body parameters without retaining GPU storage."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").clone()
    if isinstance(value, np.ndarray):
        try:
            return torch.as_tensor(value.copy()).clone()
        except (TypeError, ValueError):
            return value.copy()
    if isinstance(value, Mapping):
        return {key: _cpu_debug_copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_debug_copy(item) for item in value)
    if isinstance(value, list):
        return [_cpu_debug_copy(item) for item in value]
    return copy.deepcopy(value)


class SonicPublisher:
    """Non-blocking GEM-SMPL to SONIC Protocol v3 publisher.

    ``publish_smpl`` is the producer-side API.  It only enqueues references to
    the already detached webcam outputs. SONIC forward kinematics, resampling,
    wrist mapping, packing, and ZMQ send all run on the consumer
    thread, keeping them off the GEM inference path.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5556,
        topic: str = "pose",
        queue_size: int = 2,
        sonic_repo_path: str | Path | None = None,
        enable_yaw_calibration: bool = False,
        sonic_bad_frame_dir: str | Path | None = None,
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
        self.enable_yaw_calibration = bool(enable_yaw_calibration)
        self.sonic_bad_frame_dir = (
            Path(sonic_bad_frame_dir).expanduser() if sonic_bad_frame_dir is not None else None
        )
        self.target_fps = _TARGET_FPS
        self.window_size = _WINDOW_SIZE

        self._queue: queue.Queue[Any] = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._state_lock = threading.Lock()
        self._state = "disconnected"
        self._thread: threading.Thread | None = None
        self._worker_error: BaseException | None = None
        self._connected = False

        self._smpl_converter: SonicSMPLConverter | None = None
        self._resampler: SMPLRealtimeResampler | None = None
        self._frame_window: deque[dict[str, np.ndarray | int]] = deque(maxlen=self.window_size)
        self._decompose_rotation_aa = None
        self._next_frame_index = 0
        self._frames_sent = 0
        self._frames_dropped = 0
        self._next_log_frame = 100
        self._started_at = 0.0
        self._next_send_deadline: float | None = None
        self._lookahead_deadline: float | None = None
        self._last_send_started_at: float | None = None
        self._last_batch_discontinuity = False
        self._invalid_source_frames = 0
        self._consecutive_invalid_frames = 0
        self._last_invalid_warning_time = float("-inf")
        self._max_consecutive_invalid_frames = 30
        self._bad_frame_save_attempts = 0
        self._bad_frames_saved = 0

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
        self._frame_window.clear()
        self._next_send_deadline = None
        self._lookahead_deadline = None
        self._last_send_started_at = None
        self._last_batch_discontinuity = False
        self._invalid_source_frames = 0
        self._consecutive_invalid_frames = 0
        self._last_invalid_warning_time = float("-inf")

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

    def check_health(self) -> bool:
        """Raise in the caller when the asynchronous worker has failed."""
        with self._state_lock:
            error = self._worker_error
            state = self._state
            thread_alive = self._thread is not None and self._thread.is_alive()
            was_connected = self._connected

        if error is not None:
            raise RuntimeError("SONIC publisher worker failed") from error
        if was_connected and (state != "connected" or not thread_alive):
            raise RuntimeError("SONIC publisher worker stopped unexpectedly")
        return True

    def publish_smpl(
        self,
        body_params_incam: Mapping[str, Any],
        body_params_global: Mapping[str, Any],
        timestamp_ns: int | np.ndarray | None = None,
    ) -> bool:
        """Queue one SMPL frame/chunk without waiting for the consumer.

        When the bounded queue is full, its oldest unsent item is discarded so
        real-time inference never waits behind stale poses.
        """
        self.check_health()

        body_pose_candidates = (
            body_params_incam.get("body_pose"),
            body_params_global.get("body_pose"),
        )
        body_pose = body_pose_candidates[0]
        if body_pose is None:
            body_pose = body_pose_candidates[1]
        if body_pose is not None:
            frame_count_candidates = body_pose_candidates
            values_per_frame = 63
            num_values = (
                body_pose.numel() if isinstance(body_pose, torch.Tensor) else np.size(body_pose)
            )
            if num_values == 0 or num_values % 63:
                raise ValueError(
                    f"body_pose must contain a multiple of 63 values, got {num_values}"
                )
            num_frames = num_values // 63
        else:
            # Missing pose fields are recoverable source-frame errors. Infer the
            # batch size from the root when possible so the worker can count and
            # drop the batch through InvalidSMPLFrameError instead of raising on
            # the inference thread.
            root_candidates = (
                body_params_global.get("global_orient"),
                body_params_incam.get("global_orient"),
            )
            root = root_candidates[0]
            if root is None:
                root = root_candidates[1]
            if root is None:
                frame_count_candidates = ()
                values_per_frame = 3
                num_frames = 1
            else:
                frame_count_candidates = root_candidates
                values_per_frame = 3
                num_values = root.numel() if isinstance(root, torch.Tensor) else np.size(root)
                if num_values == 0 or num_values % 3:
                    raise ValueError(
                        "global_orient must contain a multiple of 3 values when "
                        f"body_pose is missing, got {num_values}"
                    )
                num_frames = num_values // 3

        # A finite fallback candidate can legitimately have a different batch
        # length. Accept timestamps matching any structurally valid candidate;
        # the consumer performs the final match after finite-value selection.
        timestamp_frame_counts = {num_frames}
        for candidate in frame_count_candidates:
            if candidate is None:
                continue
            candidate_values = (
                candidate.numel() if isinstance(candidate, torch.Tensor) else np.size(candidate)
            )
            if candidate_values > 0 and candidate_values % values_per_frame == 0:
                timestamp_frame_counts.add(candidate_values // values_per_frame)

        if timestamp_ns is None:
            timestamp_ns = time.monotonic_ns()
        raw_timestamps = np.asarray(timestamp_ns)
        if raw_timestamps.dtype == np.bool_ or not np.issubdtype(raw_timestamps.dtype, np.integer):
            raise TypeError("timestamp_ns must contain integer nanosecond timestamps")
        if np.issubdtype(raw_timestamps.dtype, np.unsignedinteger) and np.any(
            raw_timestamps > np.iinfo(np.int64).max
        ):
            raise ValueError("timestamp_ns exceeds the signed int64 range")
        timestamps = raw_timestamps.astype(np.int64, copy=False).reshape(-1)
        # Keep a scalar timestamp scalar until the consumer has selected the
        # finite body-pose candidate. The fallback candidate may have a
        # different batch length, and only the worker knows that final length.
        if len(timestamps) != 1 and len(timestamps) not in timestamp_frame_counts:
            expected_counts = ", ".join(map(str, sorted(timestamp_frame_counts)))
            raise ValueError(
                f"timestamp_ns has {len(timestamps)} values; expected 1 or one of "
                f"[{expected_counts}]"
            )

        with self._state_lock:
            thread_alive = self._thread is not None and self._thread.is_alive()
            if self._state != "connected" or not thread_alive or self._stop_event.is_set():
                return False

        item = (
            dict(body_params_incam),
            dict(body_params_global),
            timestamps,
            num_frames,
            False,
        )
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            dropped_input = False
            try:
                dropped_item = self._queue.get_nowait()
                if dropped_item is not _STOP:
                    dropped_input = True
                    with self._state_lock:
                        self._frames_dropped += dropped_item[3]
                else:
                    # close() also sets the stop event, so the worker does not
                    # depend on replacing a sentinel removed by this race.
                    return False
            except queue.Empty:
                pass
            if dropped_input:
                item = (*item[:4], True)
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
        pending_windows: deque[dict[str, np.ndarray]] = deque(maxlen=_PENDING_WINDOW_LIMIT)
        try:
            context = zmq.Context()
            socket = context.socket(zmq.PUB)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.SNDHWM, self._queue.maxsize)
            socket.bind(self.endpoint)

            self._smpl_converter = SonicSMPLConverter(
                self.sonic_repo_path,
                enable_yaw_calibration=self.enable_yaw_calibration,
            )
            self._resampler = SMPLRealtimeResampler(target_fps=self.target_fps)
            from gear_sonic.trl.utils.rotation_conversion import decompose_rotation_aa

            self._decompose_rotation_aa = decompose_rotation_aa
            print("[SONIC] Using SONIC compute_human_joints")
            print(f"[SONIC] SONIC repo path: {self._smpl_converter.sonic_repo_path}")
            print(
                "[SONIC] SONIC compute_human_joints: "
                f"{self._smpl_converter.compute_human_joints_source}"
            )
            print(f"[SONIC] Using Y-up -> Z-up: {self._smpl_converter.using_y_up_to_z_up}")
            print(
                "[SONIC] Using remove_smpl_base_rot: "
                f"{self._smpl_converter.using_remove_smpl_base_rot}"
            )
            print(f"[SONIC] Target FPS: {self.target_fps}")
            print(f"[SONIC] Window size: {self.window_size}")
            print(f"[SONIC] Yaw calibration: {self.enable_yaw_calibration}")
            pack_pose_message = _resolve_pose_packer()
        except BaseException as exc:
            self._set_worker_error(exc)
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
                    if self._send_due(pending_windows):
                        pose_data = pending_windows.popleft()
                        self._last_send_started_at = time.monotonic()
                        message = pack_pose_message(pose_data, topic=self.topic, version=3)
                        if self._stop_event.is_set():
                            break
                        try:
                            socket.send(message, flags=zmq.NOBLOCK)
                        except zmq.Again:
                            # A slow/no subscriber must never back-pressure inference.
                            pass
                        else:
                            self._report_sent(pose_data)
                        assert self._next_send_deadline is not None
                        self._next_send_deadline += 1.0 / self.target_fps
                        continue

                    timeout = self._input_wait_timeout(pending_windows)
                    try:
                        item = self._queue.get(timeout=timeout)
                    except queue.Empty:
                        if not pending_windows:
                            self._next_send_deadline = None
                            self._lookahead_deadline = None
                        continue
                    if item is _STOP:
                        break

                    body_params_incam, body_params_global, timestamps, num_frames, dropped = item
                    if dropped:
                        self._discard_stale_pending(pending_windows)
                    try:
                        pose_windows = self._make_pose_windows(
                            body_params_incam,
                            body_params_global,
                            timestamps,
                            num_frames,
                        )
                    except InvalidSMPLFrameError as exc:
                        self._record_invalid_source_frames(
                            exc,
                            body_params_incam,
                            body_params_global,
                            timestamps,
                            num_frames,
                        )
                        if self._consecutive_invalid_frames >= self._max_consecutive_invalid_frames:
                            raise RuntimeError(
                                "Too many consecutive invalid GEM-SMPL frames "
                                f"({self._consecutive_invalid_frames} >= "
                                f"{self._max_consecutive_invalid_frames})"
                            ) from exc
                        continue

                    self._consecutive_invalid_frames = 0
                    if self._last_batch_discontinuity:
                        pending_windows.clear()
                        self._next_send_deadline = None
                        self._lookahead_deadline = None
                    self._append_latest_windows(pending_windows, pose_windows)
                except Exception as exc:
                    self._set_worker_error(exc)
                    break
        except BaseException as exc:
            self._set_worker_error(exc)
        finally:
            socket.close(linger=0)
            context.term()
            self._smpl_converter = None
            self._resampler = None
            self._decompose_rotation_aa = None
            self._frame_window.clear()
            pending_windows.clear()
            self._mark_worker_stopped()

    def _set_worker_error(self, error: BaseException) -> None:
        with self._state_lock:
            self._worker_error = error
        print(f"\n[SONIC ERROR] {error}")

    def _record_invalid_source_frames(
        self,
        error: InvalidSMPLFrameError,
        body_params_incam: Mapping[str, Any],
        body_params_global: Mapping[str, Any],
        timestamps_ns: np.ndarray,
        num_frames: int,
    ) -> None:
        """Count, optionally snapshot, and rate-limit logging for a bad source batch."""
        self._invalid_source_frames += num_frames
        self._consecutive_invalid_frames += num_frames
        self._save_bad_frame(
            body_params_incam,
            body_params_global,
            timestamps_ns,
            error,
        )

        now = time.monotonic()
        if now - self._last_invalid_warning_time < 1.0:
            return
        self._last_invalid_warning_time = now
        print(
            "\n[SONIC WARNING] dropped invalid GEM frame\n"
            f"reason={error}\n"
            f"consecutive={self._consecutive_invalid_frames}\n"
            f"total_invalid={self._invalid_source_frames}"
        )

    def _save_bad_frame(
        self,
        body_params_incam: Mapping[str, Any],
        body_params_global: Mapping[str, Any],
        timestamps_ns: np.ndarray,
        error: InvalidSMPLFrameError,
    ) -> None:
        """Save at most three diagnostic inputs; failures remain recoverable."""
        if (
            self.sonic_bad_frame_dir is None
            or self._bad_frame_save_attempts >= _MAX_BAD_FRAME_SNAPSHOTS
        ):
            return

        snapshot_index = self._bad_frame_save_attempts
        self._bad_frame_save_attempts += 1
        output_path = self.sonic_bad_frame_dir / f"bad_smpl_frame_{snapshot_index:03d}.pt"
        try:
            self.sonic_bad_frame_dir.mkdir(parents=True, exist_ok=True)
            snapshot = {
                "timestamp_ns": torch.as_tensor(
                    np.asarray(timestamps_ns, dtype=np.int64).copy()
                ).clone(),
                "body_params_incam": _cpu_debug_copy(body_params_incam),
                "body_params_global": _cpu_debug_copy(body_params_global),
                "error": str(error),
            }
            torch.save(snapshot, output_path)
            self._bad_frames_saved += 1
            print(f"[SONIC WARNING] saved invalid GEM frame: {output_path}")
        except Exception as exc:
            print(f"[SONIC WARNING] failed to save invalid GEM frame: {exc}")

    def _send_due(self, pending_windows: deque[dict[str, np.ndarray]]) -> bool:
        """Return true for one 50 Hz send slot without replaying missed slots."""
        if not pending_windows:
            return False
        now = time.monotonic()
        period = 1.0 / self.target_fps
        if self._next_send_deadline is None:
            if len(pending_windows) >= _SEND_LOOKAHEAD:
                self._next_send_deadline = now
                self._lookahead_deadline = None
            else:
                if self._lookahead_deadline is None:
                    self._lookahead_deadline = now + _SEND_LOOKAHEAD * period
                if now < self._lookahead_deadline:
                    return False
                self._next_send_deadline = now
                self._lookahead_deadline = None

        effective_deadline = self._next_send_deadline
        if self._last_send_started_at is not None:
            effective_deadline = max(
                effective_deadline,
                self._last_send_started_at + period,
            )
        if now < effective_deadline:
            return False

        missed_slots = int((now - self._next_send_deadline) // period)
        for _ in range(min(missed_slots, max(0, len(pending_windows) - 1))):
            pending_windows.popleft()
        self._next_send_deadline += missed_slots * period
        return True

    def _input_wait_timeout(
        self,
        pending_windows: deque[dict[str, np.ndarray]],
    ) -> float:
        """Wait for source data only until the next wall-clock send slot."""
        deadline = self._next_send_deadline or self._lookahead_deadline
        if self._next_send_deadline is not None and self._last_send_started_at is not None:
            deadline = max(
                self._next_send_deadline,
                self._last_send_started_at + 1.0 / self.target_fps,
            )
        if deadline is not None:
            return max(0.0, min(deadline - time.monotonic(), 0.1))
        return 0.1

    def _append_latest_windows(
        self,
        pending_windows: deque[dict[str, np.ndarray]],
        pose_windows: list[dict[str, np.ndarray]],
    ) -> None:
        """Keep only the newest bounded set; stale history can never be replayed."""
        pending_windows.extend(pose_windows)

    def _discard_stale_pending(
        self,
        pending_windows: deque[dict[str, np.ndarray]],
    ) -> None:
        """Drop unsent packets after raw queue loss while preserving interpolation."""
        pending_windows.clear()
        self._next_send_deadline = None
        self._lookahead_deadline = None

    def _mark_worker_stopped(self) -> None:
        """Publish a terminal worker state without racing connect/close."""
        current_thread = threading.current_thread()
        with self._state_lock:
            self._connected = False
            if self._thread is current_thread:
                self._thread = None
            self._state = "disconnected"

    @torch.inference_mode()
    def _make_pose_windows(
        self,
        body_params_incam: Mapping[str, Any],
        body_params_global: Mapping[str, Any],
        timestamps_ns: np.ndarray,
        _num_source_frames: int,
    ) -> list[dict[str, np.ndarray]]:
        if self._smpl_converter is None:
            raise RuntimeError("SONIC SMPL converter is not initialized")
        if self._resampler is None:
            raise RuntimeError("SONIC realtime resampler is not initialized")
        if self._decompose_rotation_aa is None:
            raise RuntimeError("SONIC wrist mapper is not initialized")

        self._last_batch_discontinuity = False

        converted = self._smpl_converter.convert(
            body_params_global=body_params_global,
            body_params_incam=body_params_incam,
        )
        smpl_pose = np.ascontiguousarray(
            converted["smpl_pose"].detach().cpu().numpy(), dtype=np.float32
        )
        smpl_joints_local = np.ascontiguousarray(
            converted["smpl_joints_local"].detach().cpu().numpy(), dtype=np.float32
        )
        body_quat = np.ascontiguousarray(
            converted["body_quat_w"].detach().cpu().numpy(), dtype=np.float32
        )
        timestamps_ns = np.ascontiguousarray(timestamps_ns, dtype=np.int64).reshape(-1)
        converted_frames = len(smpl_pose)
        if len(timestamps_ns) == 1 and converted_frames > 1:
            end_timestamp = int(timestamps_ns[0])
            timestamps_ns = end_timestamp - np.arange(
                converted_frames - 1,
                -1,
                -1,
                dtype=np.int64,
            ) * int(round(1_000_000_000 / self.target_fps))
        elif len(timestamps_ns) != converted_frames:
            raise InvalidSMPLFrameError(
                "timestamp_ns no longer matches the finite fallback pose: "
                f"timestamps={len(timestamps_ns)}, pose_frames={converted_frames}"
            )
        if not (len(smpl_pose) == len(smpl_joints_local) == len(body_quat) == len(timestamps_ns)):
            raise ValueError("SONIC source pose fields and timestamps have different lengths")

        pose_windows: list[dict[str, np.ndarray]] = []
        for source_index in range(converted_frames):
            resampled = self._resampler.push(
                int(timestamps_ns[source_index]),
                smpl_pose[source_index],
                smpl_joints_local[source_index],
                body_quat[source_index],
            )
            if bool(resampled["discontinuity"]):
                self._frame_window.clear()
                self._next_send_deadline = None
                self._lookahead_deadline = None
                self._last_batch_discontinuity = True
                pose_windows.clear()

            resampled_pose = np.asarray(resampled["smpl_pose"], dtype=np.float32)
            resampled_joints = np.asarray(resampled["smpl_joints_local"], dtype=np.float32)
            resampled_body_quat = np.asarray(resampled["body_quat"], dtype=np.float32)
            for output_index in range(len(resampled_pose)):
                joint_pos = _compute_g1_wrist_joint_pos(
                    resampled_pose[output_index],
                    self._decompose_rotation_aa,
                )[0]
                self._frame_window.append(
                    {
                        "smpl_pose": resampled_pose[output_index],
                        "smpl_joints": resampled_joints[output_index],
                        "body_quat": resampled_body_quat[output_index],
                        "joint_pos": joint_pos,
                        "frame_index": self._next_frame_index,
                    }
                )
                self._next_frame_index += 1
                if len(self._frame_window) == self.window_size:
                    pose_windows.append(self._stack_pose_window())
        return pose_windows

    def _stack_pose_window(self) -> dict[str, np.ndarray]:
        """Snapshot the current five resampled frames for Protocol v3."""
        if len(self._frame_window) != self.window_size:
            raise RuntimeError("SONIC pose window is not full")
        frames = list(self._frame_window)
        return {
            "smpl_pose": np.ascontiguousarray(
                np.stack([frame["smpl_pose"] for frame in frames]), dtype=np.float32
            ),
            "smpl_joints": np.ascontiguousarray(
                np.stack([frame["smpl_joints"] for frame in frames]), dtype=np.float32
            ),
            "body_quat": np.ascontiguousarray(
                np.stack([frame["body_quat"] for frame in frames]), dtype=np.float32
            ),
            "joint_pos": np.ascontiguousarray(
                np.stack([frame["joint_pos"] for frame in frames]), dtype=np.float32
            ),
            "joint_vel": np.zeros((self.window_size, 29), dtype=np.float32),
            "frame_index": np.asarray([frame["frame_index"] for frame in frames], dtype=np.int64),
        }

    def _report_sent(self, pose_data: Mapping[str, np.ndarray]) -> None:
        if self._frames_sent == 0:
            self._started_at = time.perf_counter()
        # Each message advances the five-frame sliding window by one 50 Hz tick.
        self._frames_sent += 1
        if self._frames_sent < self._next_log_frame:
            return

        while self._next_log_frame <= self._frames_sent:
            self._next_log_frame += 100

        elapsed = max(time.perf_counter() - self._started_at, 1e-6)
        smpl_joints = pose_data["smpl_joints"]
        body_quat = pose_data["body_quat"][-1]
        pelvis = smpl_joints[-1, SonicSMPLConverter.PELVIS_INDEX]
        left_wrist = smpl_joints[-1, SonicSMPLConverter.LEFT_WRIST_INDEX]
        right_wrist = smpl_joints[-1, SonicSMPLConverter.RIGHT_WRIST_INDEX]
        smpl_pose_shape = f"({','.join(map(str, pose_data['smpl_pose'].shape))})"
        smpl_joints_shape = f"({','.join(map(str, smpl_joints.shape))})"
        print(
            "\n[SONIC]\n"
            f"fps={self._frames_sent / elapsed:.1f}\n"
            f"frames={self._frames_sent}\n"
            f"invalid_source_frames={self._invalid_source_frames}\n"
            f"consecutive_invalid_frames={self._consecutive_invalid_frames}\n"
            f"dropped_queue_frames={self._frames_dropped}\n"
            f"smpl_pose_shape={smpl_pose_shape}\n"
            f"smpl_joints_shape={smpl_joints_shape}\n"
            f"body_quat={np.array2string(body_quat, precision=4, separator=', ')}\n"
            f"pelvis={np.array2string(pelvis, precision=4, separator=', ')}\n"
            f"left_wrist={np.array2string(left_wrist, precision=4, separator=', ')}\n"
            f"right_wrist={np.array2string(right_wrist, precision=4, separator=', ')}"
        )

    def __enter__(self) -> SonicPublisher:
        self.connect()
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()
