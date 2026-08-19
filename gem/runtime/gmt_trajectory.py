# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""BUMI GMT ``trajectory_v1`` contract and rolling reference windows.

The legacy GMR Redis packet contains one 35-float reference frame.  GMT's
current BUMI policy, however, consumes a 21 x 52 command window.  This module
builds the native trajectory packet expected by ``MotionLoaderRedis`` so every
slot comes from its own point on a complete, already-retargeted BUMI qpos
timeline.
"""

from __future__ import annotations

import hashlib
import math
import secrets
import struct
import time
import zlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

TRAJECTORY_MAGIC = b"OMGBT001"
TRAJECTORY_VERSION = 1
TRAJECTORY_HEADER_FORMAT = "<8sHHIQQQqqfHHHH32sI"
TRAJECTORY_HEADER_SIZE = struct.calcsize(TRAJECTORY_HEADER_FORMAT)
TRAJECTORY_JOINT_COUNT = 21
TRAJECTORY_FRAME_DIM = 55
TRAJECTORY_HISTORY_FRAMES = 10
TRAJECTORY_PLAN_FRAMES = 100
TRAJECTORY_FRAME_COUNT = TRAJECTORY_HISTORY_FRAMES + TRAJECTORY_PLAN_FRAMES
TRAJECTORY_CURRENT_INDEX = TRAJECTORY_HISTORY_FRAMES
TRAJECTORY_PACKET_BYTES = TRAJECTORY_HEADER_SIZE + TRAJECTORY_FRAME_COUNT * TRAJECTORY_FRAME_DIM * 4

TRAJECTORY_ACK_MAGIC = b"OMGBTA01"
TRAJECTORY_ACK_VERSION = 1
TRAJECTORY_ACK_FORMAT = "<8sHHQQqqQ"
TRAJECTORY_ACK_SIZE = struct.calcsize(TRAJECTORY_ACK_FORMAT)

FLAG_FIXED_IDLE = 1 << 0
FLAG_TRANSITION = 1 << 1
FLAG_TEXT = 1 << 2
FLAG_AUDIO = 1 << 3
FLAG_ERROR = 1 << 4

BUMI_QPOS_DIM = 28


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def normalize_quat_wxyz(value: np.ndarray) -> np.ndarray:
    quat = np.asarray(value, dtype=np.float32).reshape(4)
    norm = float(np.linalg.norm(quat))
    if not math.isfinite(norm) or norm < 1e-8:
        raise ValueError("quaternion norm is too small or non-finite")
    quat = quat / np.float32(norm)
    return quat.astype(np.float32, copy=False)


def _continuous_quaternions(quaternions: np.ndarray) -> np.ndarray:
    result = np.stack([normalize_quat_wxyz(value) for value in quaternions], axis=0)
    for index in range(1, len(result)):
        if float(np.dot(result[index - 1], result[index])) < 0.0:
            result[index] *= -1.0
    return result


def _quat_conj_wxyz(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat, dtype=np.float32).reshape(4)
    return np.asarray((w, -x, -y, -z), dtype=np.float32)


def _quat_mul_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.asarray(left, dtype=np.float32).reshape(4)
    w2, x2, y2, z2 = np.asarray(right, dtype=np.float32).reshape(4)
    return np.asarray(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dtype=np.float32,
    )


def _quat_rotate_inverse_wxyz(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    pure = np.asarray((0.0, *np.asarray(vector, dtype=np.float32).reshape(3)), dtype=np.float32)
    return _quat_mul_wxyz(
        _quat_mul_wxyz(_quat_conj_wxyz(normalize_quat_wxyz(quat)), pure),
        normalize_quat_wxyz(quat),
    )[1:]


def _quat_to_rotvec_wxyz(quat: np.ndarray) -> np.ndarray:
    value = normalize_quat_wxyz(quat)
    if value[0] < 0.0:
        value = -value
    length = float(np.linalg.norm(value[1:]))
    if length < 1e-8:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * math.atan2(length, float(value[0]))
    return (value[1:] / np.float32(length) * np.float32(angle)).astype(np.float32)


def finite_difference(values: np.ndarray, fps: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim < 1 or array.shape[0] <= 0:
        raise ValueError("finite_difference requires a non-empty timeline")
    if not math.isfinite(float(fps)) or fps <= 0.0:
        raise ValueError("fps must be finite and > 0")
    output = np.zeros_like(array, dtype=np.float32)
    if len(array) == 1:
        return output
    output[0] = (array[1] - array[0]) * np.float32(fps)
    output[-1] = (array[-1] - array[-2]) * np.float32(fps)
    if len(array) > 2:
        output[1:-1] = (array[2:] - array[:-2]) * np.float32(fps / 2.0)
    return output


def body_angular_velocity(quaternions_wxyz: np.ndarray, fps: float) -> np.ndarray:
    quaternions = _continuous_quaternions(quaternions_wxyz)
    output = np.zeros((len(quaternions), 3), dtype=np.float32)
    if len(quaternions) == 1:
        return output
    for index in range(len(quaternions) - 1):
        delta_local = _quat_mul_wxyz(_quat_conj_wxyz(quaternions[index]), quaternions[index + 1])
        output[index] = _quat_to_rotvec_wxyz(delta_local) * np.float32(fps)
    output[-1] = output[-2]
    return output


def resample_qpos_timeline(
    qpos: np.ndarray, source_fps: float, target_fps: float = 50.0
) -> np.ndarray:
    """Resample BUMI qpos with linear translation/joints and quaternion SLERP."""
    timeline = np.asarray(qpos, dtype=np.float32)
    if timeline.ndim != 2 or timeline.shape[1] != BUMI_QPOS_DIM or len(timeline) <= 0:
        raise ValueError(f"qpos must have shape [T, {BUMI_QPOS_DIM}], got {timeline.shape}")
    if not np.isfinite(timeline).all():
        raise ValueError("qpos contains NaN or Inf")
    for name, value in (("source_fps", source_fps), ("target_fps", target_fps)):
        if not math.isfinite(float(value)) or value <= 0.0:
            raise ValueError(f"{name} must be finite and > 0")
    if len(timeline) == 1:
        output = timeline.copy()
        output[0, 3:7] = normalize_quat_wxyz(output[0, 3:7])
        return output
    if math.isclose(float(source_fps), float(target_fps), rel_tol=0.0, abs_tol=1e-8):
        output = timeline.copy()
        output[:, 3:7] = _continuous_quaternions(output[:, 3:7])
        return output

    from scipy.spatial.transform import Rotation, Slerp

    source_times = np.arange(len(timeline), dtype=np.float64) / float(source_fps)
    duration = float(source_times[-1])
    target_count = int(math.floor(duration * float(target_fps))) + 1
    target_times = np.arange(target_count, dtype=np.float64) / float(target_fps)
    target_times = np.clip(target_times, 0.0, duration)
    output = np.empty((target_count, BUMI_QPOS_DIM), dtype=np.float32)
    for dimension in (*range(3), *range(7, BUMI_QPOS_DIM)):
        output[:, dimension] = np.interp(target_times, source_times, timeline[:, dimension]).astype(
            np.float32
        )
    source_quat = _continuous_quaternions(timeline[:, 3:7])
    rotations = Rotation.from_quat(source_quat[:, (1, 2, 3, 0)])
    target_xyzw = Slerp(source_times, rotations)(target_times).as_quat().astype(np.float32)
    output[:, 3:7] = _continuous_quaternions(target_xyzw[:, (3, 0, 1, 2)])
    return output


def joint_order_sha256(joint_names: Iterable[str]) -> bytes:
    names = tuple(str(name).strip() for name in joint_names)
    if len(names) != TRAJECTORY_JOINT_COUNT or any(not name for name in names):
        raise ValueError("GMT joint order must contain 21 non-empty names")
    if len(set(names)) != len(names):
        raise ValueError("GMT joint order contains duplicates")
    return hashlib.sha256("\n".join(names).encode("utf-8")).digest()


@dataclass(frozen=True)
class GmtPolicyContract:
    path: Path
    joint_names: tuple[str, ...]
    default_joint_pos: np.ndarray
    joint_order_hash: bytes

    @classmethod
    def from_onnx(cls, path: str | Path) -> GmtPolicyContract:
        resolved = Path(path).expanduser().resolve(strict=True)
        import onnxruntime as ort

        session = ort.InferenceSession(str(resolved), providers=["CPUExecutionProvider"])
        input_shapes = {value.name: tuple(value.shape) for value in session.get_inputs()}
        expected_shapes = {
            "policy": (1, 69),
            "history_obs": (1, 690),
            "command_window": (1, 1092),
        }
        for name, shape in expected_shapes.items():
            if input_shapes.get(name) != shape:
                raise ValueError(
                    f"GMT policy input {name!r} must have shape {shape}, "
                    f"got {input_shapes.get(name)} in {resolved}"
                )
        metadata = session.get_modelmeta().custom_metadata_map
        try:
            names = tuple(part.strip() for part in metadata["joint_names"].split(","))
            defaults = np.asarray(
                [float(part) for part in metadata["default_joint_pos"].split(",")],
                dtype=np.float32,
            )
        except KeyError as exc:
            raise ValueError(f"GMT policy is missing metadata {exc.args[0]!r}") from exc
        _finite_array(defaults, (TRAJECTORY_JOINT_COUNT,), "default_joint_pos")
        return cls(resolved, names, defaults, joint_order_sha256(names))

    def native_to_gmt_indices(self, native_joint_names: Iterable[str]) -> np.ndarray:
        native = tuple(str(name).strip() for name in native_joint_names)
        if len(native) != TRAJECTORY_JOINT_COUNT or len(set(native)) != len(native):
            raise ValueError("native BUMI joint order must contain 21 unique names")
        lookup = {name: index for index, name in enumerate(native)}
        if set(native) != set(self.joint_names):
            raise ValueError("BUMI native and GMT policy joint names do not match")
        return np.asarray([lookup[name] for name in self.joint_names], dtype=np.int64)

    def default_in_native_order(self, native_joint_names: Iterable[str]) -> np.ndarray:
        """Return ONNX default joint positions in MuJoCo/native qpos order."""
        permutation = self.native_to_gmt_indices(native_joint_names)
        result = np.empty((TRAJECTORY_JOINT_COUNT,), dtype=np.float32)
        result[permutation] = self.default_joint_pos
        return result


def qpos_timeline_to_gmt_frames(
    qpos: np.ndarray, *, fps: float, native_to_gmt: np.ndarray
) -> np.ndarray:
    timeline = np.asarray(qpos, dtype=np.float32)
    if timeline.ndim != 2 or timeline.shape[1] != BUMI_QPOS_DIM or len(timeline) <= 0:
        raise ValueError(f"qpos must have shape [T, {BUMI_QPOS_DIM}], got {timeline.shape}")
    if not np.isfinite(timeline).all():
        raise ValueError("qpos contains NaN or Inf")
    permutation = np.asarray(native_to_gmt, dtype=np.int64)
    if permutation.shape != (TRAJECTORY_JOINT_COUNT,) or set(permutation.tolist()) != set(
        range(TRAJECTORY_JOINT_COUNT)
    ):
        raise ValueError("native_to_gmt must be a permutation of 0..20")
    root_position = timeline[:, :3]
    root_quaternion = _continuous_quaternions(timeline[:, 3:7])
    root_velocity_world = finite_difference(root_position, fps)
    root_velocity_body = np.stack(
        [
            _quat_rotate_inverse_wxyz(root_quaternion[index], root_velocity_world[index])
            for index in range(len(timeline))
        ],
        axis=0,
    )
    native_joint_position = timeline[:, 7:]
    native_joint_velocity = finite_difference(native_joint_position, fps)
    frames = np.concatenate(
        (
            root_position,
            root_quaternion,
            root_velocity_body,
            body_angular_velocity(root_quaternion, fps),
            native_joint_position[:, permutation],
            native_joint_velocity[:, permutation],
        ),
        axis=1,
    ).astype(np.float32, copy=False)
    if frames.shape != (len(timeline), TRAJECTORY_FRAME_DIM):
        raise AssertionError(f"unexpected GMT frame shape {frames.shape}")
    if not np.isfinite(frames).all():
        raise ValueError("derived GMT frames contain NaN or Inf")
    return frames


class IncrementalGmtFrameTimeline:
    """Incrementally derive GMT frames while preserving offline semantics.

    Appending qpos can only change the derivative stored for the previous tail
    frame.  Recomputing that one boundary frame plus the new suffix is enough
    to remain equivalent to :func:`qpos_timeline_to_gmt_frames`.  Published
    snapshots remain immutable because every append allocates new arrays.
    """

    def __init__(self, native_to_gmt: np.ndarray, *, fps: float = 50.0) -> None:
        self.native_to_gmt = np.asarray(native_to_gmt, dtype=np.int64).copy()
        if self.native_to_gmt.shape != (TRAJECTORY_JOINT_COUNT,) or set(
            self.native_to_gmt.tolist()
        ) != set(range(TRAJECTORY_JOINT_COUNT)):
            raise ValueError("native_to_gmt must be a permutation of 0..20")
        if not math.isfinite(float(fps)) or fps <= 0.0:
            raise ValueError("fps must be finite and > 0")
        self.fps = float(fps)
        self._qpos = np.empty((0, BUMI_QPOS_DIM), dtype=np.float32)
        self._frames = np.empty((0, TRAJECTORY_FRAME_DIM), dtype=np.float32)
        self._freeze()

    def _freeze(self) -> None:
        self._qpos.setflags(write=False)
        self._frames.setflags(write=False)

    @property
    def qpos(self) -> np.ndarray:
        return self._qpos

    @property
    def frames(self) -> np.ndarray:
        return self._frames

    def append(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        qpos = np.asarray(values, dtype=np.float32)
        if qpos.ndim != 2 or qpos.shape[1] != BUMI_QPOS_DIM or len(qpos) <= 0:
            raise ValueError(f"qpos chunk must have shape [T,{BUMI_QPOS_DIM}]")
        if not np.isfinite(qpos).all():
            raise ValueError("qpos chunk contains NaN or Inf")
        qpos = qpos.copy()
        old_count = len(self._qpos)
        if old_count == 0:
            next_qpos = qpos
            next_frames = qpos_timeline_to_gmt_frames(
                qpos, fps=self.fps, native_to_gmt=self.native_to_gmt
            )
        else:
            # Two prior qpos samples are sufficient for the centered linear
            # velocity at the old tail.  SO(3) angular velocity only needs the
            # immediately preceding sample, so it is covered by the same halo.
            halo = min(2, old_count)
            patch_qpos = np.concatenate((self._qpos[-halo:], qpos), axis=0)
            patch_frames = qpos_timeline_to_gmt_frames(
                patch_qpos, fps=self.fps, native_to_gmt=self.native_to_gmt
            )
            replace_from = old_count - 1
            patch_from = halo - 1
            next_qpos = np.concatenate((self._qpos, qpos), axis=0)
            next_frames = np.concatenate(
                (self._frames[:replace_from], patch_frames[patch_from:]), axis=0
            )
        if len(next_qpos) != len(next_frames):
            raise AssertionError("incremental GMT qpos/frame lengths diverged")
        next_qpos.setflags(write=False)
        next_frames.setflags(write=False)
        self._qpos = next_qpos
        self._frames = next_frames
        return self._qpos, self._frames


def rolling_window_indices(num_frames: int, cursor: int) -> np.ndarray:
    if num_frames <= 0:
        raise ValueError("num_frames must be > 0")
    if not 0 <= cursor < num_frames:
        raise ValueError(f"cursor {cursor} is outside [0, {num_frames})")
    indices = np.arange(
        cursor - TRAJECTORY_HISTORY_FRAMES,
        cursor + TRAJECTORY_PLAN_FRAMES,
        dtype=np.int64,
    )
    return np.clip(indices, 0, num_frames - 1)


@dataclass(frozen=True)
class GmtTrajectoryPacket:
    stream_id: int
    sequence: int
    published_unix_ns: int
    command_revision: int
    plan_id: int
    fps: float
    flags: int
    joint_order_hash: bytes
    frames: np.ndarray

    def encode(self) -> bytes:
        frames = _finite_array(
            self.frames,
            (TRAJECTORY_FRAME_COUNT, TRAJECTORY_FRAME_DIM),
            "trajectory frames",
        ).astype("<f4", copy=False)
        order_hash = bytes(self.joint_order_hash)
        if len(order_hash) != 32:
            raise ValueError("joint_order_hash must contain 32 bytes")
        if not math.isfinite(float(self.fps)) or self.fps <= 0.0:
            raise ValueError("trajectory fps must be finite and > 0")
        payload = frames.tobytes(order="C")
        header = struct.pack(
            TRAJECTORY_HEADER_FORMAT,
            TRAJECTORY_MAGIC,
            TRAJECTORY_VERSION,
            TRAJECTORY_HEADER_SIZE,
            int(self.flags),
            int(self.stream_id),
            int(self.sequence),
            int(self.published_unix_ns),
            int(self.command_revision),
            int(self.plan_id),
            float(self.fps),
            TRAJECTORY_FRAME_COUNT,
            TRAJECTORY_CURRENT_INDEX,
            TRAJECTORY_JOINT_COUNT,
            TRAJECTORY_FRAME_DIM,
            order_hash,
            zlib.crc32(payload) & 0xFFFFFFFF,
        )
        result = header + payload
        if len(result) != TRAJECTORY_PACKET_BYTES:
            raise AssertionError(f"unexpected trajectory packet size {len(result)}")
        return result

    @classmethod
    def decode(cls, blob: bytes | bytearray | memoryview) -> GmtTrajectoryPacket:
        value = bytes(blob)
        if len(value) != TRAJECTORY_PACKET_BYTES:
            raise ValueError(
                f"trajectory packet must contain {TRAJECTORY_PACKET_BYTES} bytes, got {len(value)}"
            )
        fields = struct.unpack(TRAJECTORY_HEADER_FORMAT, value[:TRAJECTORY_HEADER_SIZE])
        (
            magic,
            version,
            header_size,
            flags,
            stream_id,
            sequence,
            published_unix_ns,
            command_revision,
            plan_id,
            fps,
            frame_count,
            current_index,
            joint_count,
            frame_dim,
            order_hash,
            expected_crc,
        ) = fields
        if (magic, version, header_size) != (
            TRAJECTORY_MAGIC,
            TRAJECTORY_VERSION,
            TRAJECTORY_HEADER_SIZE,
        ):
            raise ValueError("unsupported trajectory magic/version/header")
        if (frame_count, current_index, joint_count, frame_dim) != (
            TRAJECTORY_FRAME_COUNT,
            TRAJECTORY_CURRENT_INDEX,
            TRAJECTORY_JOINT_COUNT,
            TRAJECTORY_FRAME_DIM,
        ):
            raise ValueError("trajectory dimensions do not match the GMT contract")
        if not math.isfinite(float(fps)) or fps <= 0.0:
            raise ValueError("trajectory fps must be finite and > 0")
        payload = value[TRAJECTORY_HEADER_SIZE:]
        if (zlib.crc32(payload) & 0xFFFFFFFF) != expected_crc:
            raise ValueError("trajectory payload CRC32 mismatch")
        frames = (
            np.frombuffer(payload, dtype="<f4")
            .reshape(TRAJECTORY_FRAME_COUNT, TRAJECTORY_FRAME_DIM)
            .copy()
        )
        if not np.isfinite(frames).all():
            raise ValueError("trajectory payload contains NaN or Inf")
        quaternion_norms = np.linalg.norm(frames[:, 3:7], axis=1)
        if np.any((quaternion_norms < 0.5) | (quaternion_norms > 1.5)):
            raise ValueError("trajectory payload contains an invalid root quaternion")
        frames[:, 3:7] /= quaternion_norms[:, None]
        return cls(
            int(stream_id),
            int(sequence),
            int(published_unix_ns),
            int(command_revision),
            int(plan_id),
            float(fps),
            int(flags),
            bytes(order_hash),
            frames,
        )


@dataclass(frozen=True)
class GmtTrajectoryAck:
    stream_id: int
    sequence: int
    command_revision: int
    plan_id: int
    received_unix_ns: int

    def encode(self) -> bytes:
        return struct.pack(
            TRAJECTORY_ACK_FORMAT,
            TRAJECTORY_ACK_MAGIC,
            TRAJECTORY_ACK_VERSION,
            TRAJECTORY_ACK_SIZE,
            int(self.stream_id),
            int(self.sequence),
            int(self.command_revision),
            int(self.plan_id),
            int(self.received_unix_ns),
        )

    @classmethod
    def decode(cls, blob: bytes | bytearray | memoryview) -> GmtTrajectoryAck:
        value = bytes(blob)
        if len(value) != TRAJECTORY_ACK_SIZE:
            raise ValueError(f"trajectory ACK must contain {TRAJECTORY_ACK_SIZE} bytes")
        magic, version, size, stream_id, sequence, revision, plan_id, received = struct.unpack(
            TRAJECTORY_ACK_FORMAT, value
        )
        if (magic, version, size) != (
            TRAJECTORY_ACK_MAGIC,
            TRAJECTORY_ACK_VERSION,
            TRAJECTORY_ACK_SIZE,
        ):
            raise ValueError("unsupported GMT trajectory ACK")
        return cls(int(stream_id), int(sequence), int(revision), int(plan_id), int(received))


def packet_for_cursor(
    frames: np.ndarray,
    cursor: int,
    *,
    fps: float,
    joint_order_hash: bytes,
    stream_id: int,
    sequence: int,
    command_revision: int = 0,
    plan_id: int = -1,
    flags: int = 0,
    published_unix_ns: int | None = None,
) -> GmtTrajectoryPacket:
    timeline = np.asarray(frames, dtype=np.float32)
    if timeline.ndim != 2 or timeline.shape[1] != TRAJECTORY_FRAME_DIM:
        raise ValueError(f"frames must have shape [T, {TRAJECTORY_FRAME_DIM}]")
    indices = rolling_window_indices(len(timeline), cursor)
    return GmtTrajectoryPacket(
        stream_id=int(stream_id),
        sequence=int(sequence),
        published_unix_ns=time.time_ns() if published_unix_ns is None else int(published_unix_ns),
        command_revision=int(command_revision),
        plan_id=int(plan_id),
        fps=float(fps),
        flags=int(flags),
        joint_order_hash=bytes(joint_order_hash),
        frames=timeline[indices],
    )


@dataclass(frozen=True)
class QposPlaybackTimeline:
    qpos: np.ndarray
    audio_start_frame: int
    audio_end_frame: int


def _intermediate_qpos(start: np.ndarray, end: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        return np.empty((0, BUMI_QPOS_DIM), dtype=np.float32)
    from scipy.spatial.transform import Rotation, Slerp

    first = _finite_array(start, (BUMI_QPOS_DIM,), "start qpos")
    last = _finite_array(end, (BUMI_QPOS_DIM,), "end qpos")
    alpha = np.arange(1, count + 1, dtype=np.float64) / float(count + 1)
    eased = alpha * alpha * (3.0 - 2.0 * alpha)
    output = first[None] + (last - first)[None] * eased[:, None].astype(np.float32)
    quaternions = _continuous_quaternions(np.stack((first[3:7], last[3:7])))
    xyzw = Slerp(
        np.asarray((0.0, 1.0)),
        Rotation.from_quat(quaternions[:, (1, 2, 3, 0)]),
    )(eased).as_quat()
    output[:, 3:7] = xyzw[:, (3, 0, 1, 2)]
    return output.astype(np.float32)


def build_playback_timeline(
    action_qpos: np.ndarray,
    idle_qpos: np.ndarray,
    *,
    fps: float = 50.0,
    blend_seconds: float = 0.8,
    return_seconds: float = 1.0,
) -> QposPlaybackTimeline:
    """Add real history/future idle context and smooth BUMI-space transitions."""
    for name, value in (
        ("fps", fps),
        ("blend_seconds", blend_seconds),
        ("return_seconds", return_seconds),
    ):
        if not math.isfinite(float(value)) or value <= 0.0:
            raise ValueError(f"{name} must be finite and > 0")
    action = np.asarray(action_qpos, dtype=np.float32)
    if action.ndim != 2 or action.shape[1] != BUMI_QPOS_DIM or len(action) <= 0:
        raise ValueError(f"action_qpos must have shape [T, {BUMI_QPOS_DIM}]")
    idle = _finite_array(idle_qpos, (BUMI_QPOS_DIM,), "idle_qpos").copy()
    if not np.isfinite(action).all():
        raise ValueError("action_qpos contains NaN or Inf")
    blend_count = max(1, int(round(blend_seconds * fps)))
    return_count = max(1, int(round(return_seconds * fps)))
    prefix = np.repeat(idle[None], TRAJECTORY_HISTORY_FRAMES, axis=0)
    blend = _intermediate_qpos(idle, action[0], blend_count)
    return_idle = idle.copy()
    return_idle[:2] = action[-1, :2]
    returning = _intermediate_qpos(action[-1], return_idle, return_count)
    suffix = np.repeat(return_idle[None], TRAJECTORY_PLAN_FRAMES, axis=0)
    audio_start = len(prefix) + len(blend)
    audio_end = audio_start + len(action)
    timeline = np.concatenate((prefix, blend, action, returning, suffix), axis=0)
    timeline[:, 3:7] = _continuous_quaternions(timeline[:, 3:7])
    return QposPlaybackTimeline(timeline.astype(np.float32), audio_start, audio_end)


class RedisTrajectoryPublisher:
    """Publish strict trajectory packets and validate GMT acknowledgements."""

    def __init__(
        self,
        client: Any,
        *,
        key: str = "gmt_online_frame_bumi",
        ttl_ms: int = 250,
        stream_id: int | None = None,
    ) -> None:
        if not key or any(character.isspace() for character in key):
            raise ValueError("Redis key must be non-empty and contain no whitespace")
        if ttl_ms <= 0:
            raise ValueError("ttl_ms must be > 0")
        self.client = client
        self.key = key
        self.ack_key = f"{key}_ack"
        self.ttl_ms = int(ttl_ms)
        self.stream_id = int(stream_id or secrets.randbits(63) or 1)
        self.sequence = 0
        self.last_packet: GmtTrajectoryPacket | None = None

    def publish(
        self,
        frames: np.ndarray,
        cursor: int,
        *,
        fps: float,
        joint_order_hash: bytes,
        command_revision: int = 0,
        plan_id: int = -1,
        flags: int = 0,
    ) -> GmtTrajectoryPacket:
        packet = packet_for_cursor(
            frames,
            cursor,
            fps=fps,
            joint_order_hash=joint_order_hash,
            stream_id=self.stream_id,
            sequence=self.sequence,
            command_revision=command_revision,
            plan_id=plan_id,
            flags=flags,
        )
        encoded = packet.encode()
        self.client.set(self.key, encoded, px=self.ttl_ms)
        self.last_packet = packet
        self.sequence += 1
        return packet

    def matching_ack(self) -> GmtTrajectoryAck | None:
        value = self.client.get(self.ack_key)
        if value is None:
            return None
        try:
            ack = GmtTrajectoryAck.decode(value)
        except ValueError:
            return None
        if ack.stream_id != self.stream_id:
            return None
        if self.last_packet is not None:
            if ack.sequence > self.last_packet.sequence:
                return None
            if (
                ack.command_revision != self.last_packet.command_revision
                or ack.plan_id != self.last_packet.plan_id
            ):
                return None
        return ack
