# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Contracts shared by the resident TensorRT console and BUMI safety bridge."""

from __future__ import annotations

import json
import math
import shlex
import struct
import subprocess
import threading
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import torch

from gem.gmr_udp_bridge import PACKET_BYTES as SMP1_PACKET_BYTES
from gem.gmr_udp_bridge import SMP1PacketEncoder
from gem.runtime.gmt_trajectory import (
    BUMI_QPOS_DIM,
    TRAJECTORY_PLAN_FRAMES,
)
from gem.smplx_gmr_reference import SMPLXGMRReference

ROBOT_STREAM_CONTRACT = "robot_stream_v1"
SOURCE_FPS = 30.0
TARGET_FPS = 50.0

GMR_REQUEST_MAGIC = b"GMRQ"
GMR_RESPONSE_MAGIC = b"GMRA"
GMR_PROTOCOL_VERSION = 1
GMR_OP_FRAME = 1
GMR_OP_RESET = 2
GMR_OP_QUIT = 3
GMR_STATUS_OK = 0
GMR_REQUEST_HEADER = struct.Struct("<4sHHIIII")
GMR_RESPONSE_HEADER = struct.Struct("<4sHHIIIIQ")


def has_complete_publish_context(num_frames: int, cursor: int) -> bool:
    """Require future99 for GMT plus one sample for centered tail velocity."""
    return (
        num_frames > 0
        and 0 <= cursor < num_frames
        and cursor + TRAJECTORY_PLAN_FRAMES < num_frames
    )


@dataclass(frozen=True, slots=True)
class RobotStreamChunk:
    request_id: str
    revision: int
    chunk_index: int
    absolute_start_frame: int
    frame_count: int
    total_frames: int
    is_last: bool
    checkpoint_sha256: str
    engine_sha256: str
    payload: bytes

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.revision < 0 or self.chunk_index < 0 or self.absolute_start_frame < 0:
            raise ValueError("revision, chunk_index and absolute_start_frame must be >= 0")
        if self.frame_count <= 0 or self.total_frames <= 0:
            raise ValueError("frame_count and total_frames must be > 0")
        if len(self.payload) != self.frame_count * SMP1_PACKET_BYTES:
            raise ValueError("SMP1 chunk payload size does not match frame_count")
        if self.absolute_start_frame + self.frame_count > self.total_frames:
            raise ValueError("chunk extends beyond total_frames")
        if self.is_last and self.absolute_start_frame + self.frame_count != self.total_frames:
            raise ValueError("last chunk must end exactly at total_frames")

    def header(self) -> dict[str, Any]:
        return {
            "contract_version": ROBOT_STREAM_CONTRACT,
            "command": "chunk",
            "request_id": self.request_id,
            "revision": self.revision,
            "chunk_index": self.chunk_index,
            "absolute_start_frame": self.absolute_start_frame,
            "frame_count": self.frame_count,
            "total_frames": self.total_frames,
            "source_fps": SOURCE_FPS,
            "is_last": self.is_last,
            "checkpoint_sha256": self.checkpoint_sha256,
            "engine_sha256": self.engine_sha256,
            "payload_crc32": zlib.crc32(self.payload) & 0xFFFF_FFFF,
        }

    def multipart(self) -> list[bytes]:
        encoded = json.dumps(self.header(), separators=(",", ":")).encode("utf-8")
        return [encoded, self.payload]

    @classmethod
    def from_multipart(cls, parts: list[bytes] | tuple[bytes, ...]) -> RobotStreamChunk:
        if len(parts) != 2:
            raise ValueError("robot_stream_v1 chunk must contain JSON header and payload")
        header = json.loads(parts[0].decode("utf-8"))
        if header.get("contract_version") != ROBOT_STREAM_CONTRACT:
            raise ValueError("unsupported robot stream contract")
        if header.get("command") != "chunk":
            raise ValueError("multipart message is not a chunk")
        if not math.isclose(float(header.get("source_fps", 0.0)), SOURCE_FPS):
            raise ValueError("robot stream source_fps must be 30")
        payload = bytes(parts[1])
        expected_crc = int(header["payload_crc32"])
        if zlib.crc32(payload) & 0xFFFF_FFFF != expected_crc:
            raise ValueError("robot stream payload CRC32 mismatch")
        return cls(
            request_id=str(header["request_id"]),
            revision=int(header["revision"]),
            chunk_index=int(header["chunk_index"]),
            absolute_start_frame=int(header["absolute_start_frame"]),
            frame_count=int(header["frame_count"]),
            total_frames=int(header["total_frames"]),
            is_last=bool(header["is_last"]),
            checkpoint_sha256=str(header["checkpoint_sha256"]),
            engine_sha256=str(header["engine_sha256"]),
            payload=payload,
        )

    def packets(self) -> list[bytes]:
        return [
            self.payload[index : index + SMP1_PACKET_BYTES]
            for index in range(0, len(self.payload), SMP1_PACKET_BYTES)
        ]


class RevisionTracker:
    """Reject stale, duplicated and non-contiguous robot stream chunks."""

    def __init__(self) -> None:
        self.revision = -1
        self.request_id: str | None = None
        self.next_chunk = 0
        self.next_frame = 0
        self.total_frames = 0

    def begin(self, request_id: str, revision: int, total_frames: int) -> None:
        if not request_id or revision <= self.revision or total_frames <= 0:
            raise ValueError("begin must use a non-empty request and a newer revision")
        self.revision = int(revision)
        self.request_id = request_id
        self.next_chunk = 0
        self.next_frame = 0
        self.total_frames = int(total_frames)

    def accept(self, chunk: RobotStreamChunk) -> None:
        if chunk.revision != self.revision or chunk.request_id != self.request_id:
            raise ValueError("stale request/revision chunk")
        if chunk.total_frames != self.total_frames:
            raise ValueError("chunk total_frames changed within one request")
        if chunk.chunk_index != self.next_chunk:
            raise ValueError("chunk index is duplicated or out of order")
        if chunk.absolute_start_frame != self.next_frame:
            raise ValueError("chunk frames are not contiguous")
        self.next_chunk += 1
        self.next_frame += chunk.frame_count
        if chunk.is_last and self.next_frame != self.total_frames:
            raise ValueError("last chunk did not complete the request")


def encode_gmr_request(
    operation: int,
    sequence: int,
    payload: bytes = b"",
    *,
    reset_iterations: int = 0,
) -> bytes:
    if operation not in {GMR_OP_FRAME, GMR_OP_RESET, GMR_OP_QUIT}:
        raise ValueError("unsupported GMR operation")
    if operation in {GMR_OP_FRAME, GMR_OP_RESET} and len(payload) != SMP1_PACKET_BYTES:
        raise ValueError("FRAME/RESET requires exactly one SMP1 packet")
    if operation == GMR_OP_QUIT and payload:
        raise ValueError("QUIT must not contain a payload")
    crc = zlib.crc32(payload) & 0xFFFF_FFFF
    return GMR_REQUEST_HEADER.pack(
        GMR_REQUEST_MAGIC,
        GMR_PROTOCOL_VERSION,
        operation,
        int(sequence) & 0xFFFF_FFFF,
        len(payload),
        int(reset_iterations),
        crc,
    ) + payload


def decode_gmr_response(header: bytes, payload: bytes) -> tuple[int, np.ndarray, int]:
    if len(header) != GMR_RESPONSE_HEADER.size:
        raise ValueError("invalid GMR response header size")
    magic, version, status, sequence, qpos_dim, payload_size, crc, elapsed_us = (
        GMR_RESPONSE_HEADER.unpack(header)
    )
    if magic != GMR_RESPONSE_MAGIC or version != GMR_PROTOCOL_VERSION:
        raise ValueError("unsupported GMR response magic/version")
    if status != GMR_STATUS_OK:
        message = payload.decode("utf-8", errors="replace")
        raise RuntimeError(f"GMR batch server status={status}: {message}")
    if qpos_dim != BUMI_QPOS_DIM or payload_size != BUMI_QPOS_DIM * 4:
        raise ValueError("GMR response qpos contract mismatch")
    if len(payload) != payload_size or zlib.crc32(payload) & 0xFFFF_FFFF != crc:
        raise ValueError("GMR response payload size/CRC mismatch")
    qpos = np.frombuffer(payload, dtype="<f4").copy()
    if qpos.shape != (BUMI_QPOS_DIM,) or not np.isfinite(qpos).all():
        raise ValueError("GMR response contains an invalid qpos")
    return int(sequence), qpos, int(elapsed_us)


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError(f"GMR batch server closed with {remaining} response bytes missing")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class GMRBatchClient:
    """Synchronous one-input/one-qpos client for the persistent GMR child."""

    def __init__(
        self,
        command: list[str],
        *,
        cwd: str | Path | None = None,
        popen=subprocess.Popen,
    ) -> None:
        if not command:
            raise ValueError("GMR batch command must not be empty")
        self.command = [str(value) for value in command]
        self.process = popen(
            self.command,
            cwd=None if cwd is None else str(cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            bufsize=0,
        )
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("failed to open GMR batch stdin/stdout")
        self._lock = threading.Lock()
        self._sequence = 0

    def _exchange(
        self,
        operation: int,
        payload: bytes = b"",
        *,
        reset_iterations: int = 0,
    ) -> tuple[np.ndarray, int]:
        with self._lock:
            if self.process.poll() is not None:
                raise RuntimeError(f"GMR batch server exited with {self.process.returncode}")
            sequence = self._sequence
            self._sequence = (self._sequence + 1) & 0xFFFF_FFFF
            request = encode_gmr_request(
                operation,
                sequence,
                payload,
                reset_iterations=reset_iterations,
            )
            self.process.stdin.write(request)
            self.process.stdin.flush()
            header = _read_exact(self.process.stdout, GMR_RESPONSE_HEADER.size)
            fields = GMR_RESPONSE_HEADER.unpack(header)
            response_payload = _read_exact(self.process.stdout, int(fields[5]))
            response_sequence, qpos, elapsed_us = decode_gmr_response(
                header, response_payload
            )
            if response_sequence != sequence:
                raise RuntimeError(
                    f"GMR response sequence {response_sequence} != request {sequence}"
                )
            return qpos, elapsed_us

    def frame(self, smp1_packet: bytes) -> tuple[np.ndarray, int]:
        return self._exchange(GMR_OP_FRAME, smp1_packet)

    def reset(
        self, idle_smp1_packet: bytes, *, iterations: int = 1000
    ) -> tuple[np.ndarray, int]:
        if iterations <= 0:
            raise ValueError("reset iterations must be > 0")
        return self._exchange(
            GMR_OP_RESET, idle_smp1_packet, reset_iterations=iterations
        )

    def close(self) -> None:
        with self._lock:
            if self.process.poll() is None:
                try:
                    sequence = self._sequence
                    self.process.stdin.write(encode_gmr_request(GMR_OP_QUIT, sequence))
                    self.process.stdin.flush()
                    self.process.wait(timeout=2.0)
                except (BrokenPipeError, subprocess.TimeoutExpired):
                    self.process.terminate()
            if self.process.poll() is None:
                self.process.kill()


class GMRMotionRetargetSession:
    """Warm a stateful GMR solver from the first real pose of each revision.

    GMR's nonlinear IK solution depends on its previous qpos.  Warming a new
    request from an unrelated idle pose can leave the solver on a joint-limit
    branch even though the SMPL motion is valid.  The original UDP service
    avoided that failure by retargeting a real incoming pose 1000 times before
    consuming the motion stream.  This helper preserves those semantics for
    the synchronous one-input/one-output batch protocol.

    The qpos returned by ``RESET`` is deliberately discarded.  The same SMP1
    packet is then submitted once through ``FRAME`` so every source frame still
    contributes exactly one qpos to the published timeline.
    """

    def __init__(self, *, warmup_iterations: int = 1000) -> None:
        if warmup_iterations <= 0:
            raise ValueError("warmup_iterations must be > 0")
        self.warmup_iterations = int(warmup_iterations)
        self._warmed_revision: int | None = None

    @property
    def warmed_revision(self) -> int | None:
        return self._warmed_revision

    def invalidate(self) -> None:
        """Require the next real motion packet to warm the solver again."""
        self._warmed_revision = None

    def retarget(
        self,
        client: GMRBatchClient,
        smp1_packet: bytes,
        *,
        revision: int,
    ) -> tuple[np.ndarray, int, int | None]:
        """Return one qpos and optional one-time warm-up duration in microseconds."""
        if revision < 0:
            raise ValueError("revision must be >= 0")
        if len(smp1_packet) != SMP1_PACKET_BYTES:
            raise ValueError("motion retarget requires exactly one SMP1 packet")

        warmup_elapsed_us: int | None = None
        if self._warmed_revision != revision:
            _, warmup_elapsed_us = client.reset(
                smp1_packet,
                iterations=self.warmup_iterations,
            )
            self._warmed_revision = int(revision)

        qpos, frame_elapsed_us = client.frame(smp1_packet)
        return qpos, int(frame_elapsed_us), warmup_elapsed_us


class IncrementalQposTimeline:
    """Append native 30 Hz qpos on the fixed global 50 Hz sample grid.

    Only the newly available target samples are interpolated.  The last source
    sample from the previous chunk is retained as the interpolation boundary;
    previously emitted target samples are final and are never recomputed.
    """

    def __init__(self) -> None:
        self._source_frames = 0
        self._last_source: np.ndarray | None = None
        self._target_frames = 0
        self._target_chunks: list[np.ndarray] = []

    @property
    def source_frames(self) -> int:
        return self._source_frames

    @property
    def target_frames(self) -> int:
        return self._target_frames

    def append(self, values: np.ndarray) -> np.ndarray:
        qpos = np.asarray(values, dtype=np.float32).copy()
        if qpos.ndim != 2 or qpos.shape[1] != BUMI_QPOS_DIM or len(qpos) <= 0:
            raise ValueError(f"qpos chunk must have shape [T,{BUMI_QPOS_DIM}]")
        if not np.isfinite(qpos).all():
            raise ValueError("qpos chunk contains NaN or Inf")

        # Normalize and make quaternion signs continuous across the chunk
        # boundary before constructing the local Slerp interval.
        for index in range(len(qpos)):
            quat = qpos[index, 3:7]
            norm = float(np.linalg.norm(quat))
            if not math.isfinite(norm) or norm < 1e-8:
                raise ValueError("quaternion norm is too small or non-finite")
            qpos[index, 3:7] = quat / np.float32(norm)
            previous = self._last_source if index == 0 else qpos[index - 1]
            if previous is not None and float(np.dot(previous[3:7], qpos[index, 3:7])) < 0.0:
                qpos[index, 3:7] *= -1.0

        old_source_frames = self._source_frames
        new_source_frames = old_source_frames + len(qpos)
        duration = (new_source_frames - 1) / SOURCE_FPS
        new_target_frames = int(math.floor(duration * TARGET_FPS)) + 1
        target_indices = np.arange(
            self._target_frames, new_target_frames, dtype=np.int64
        )

        if self._last_source is None:
            local_source = qpos
            local_start = 0
        else:
            local_source = np.concatenate((self._last_source[None], qpos), axis=0)
            local_start = old_source_frames - 1

        if len(target_indices) == 0:
            result = np.empty((0, BUMI_QPOS_DIM), dtype=np.float32)
        elif len(local_source) == 1:
            result = local_source.copy()
        else:
            from scipy.spatial.transform import Rotation, Slerp

            source_times = (
                local_start + np.arange(len(local_source), dtype=np.float64)
            ) / SOURCE_FPS
            target_times = target_indices.astype(np.float64) / TARGET_FPS
            target_times = np.clip(target_times, source_times[0], source_times[-1])
            result = np.empty((len(target_indices), BUMI_QPOS_DIM), dtype=np.float32)
            for dimension in (*range(3), *range(7, BUMI_QPOS_DIM)):
                result[:, dimension] = np.interp(
                    target_times, source_times, local_source[:, dimension]
                ).astype(np.float32)
            source_xyzw = local_source[:, (4, 5, 6, 3)]
            target_xyzw = Slerp(
                source_times, Rotation.from_quat(source_xyzw)
            )(target_times).as_quat().astype(np.float32)
            result[:, 3:7] = target_xyzw[:, (3, 0, 1, 2)]

        result.setflags(write=False)
        if len(result):
            self._target_chunks.append(result)
        self._source_frames = new_source_frames
        self._last_source = qpos[-1].copy()
        self._target_frames = new_target_frames
        return result

    def target(self) -> np.ndarray:
        if not self._target_chunks:
            return np.empty((0, BUMI_QPOS_DIM), dtype=np.float32)
        return np.concatenate(self._target_chunks, axis=0)

    def reset(self) -> None:
        self._source_frames = 0
        self._last_source = None
        self._target_frames = 0
        self._target_chunks.clear()


@torch.inference_mode()
def smpl_params_to_smp1_payload(
    smpl_params: dict[str, torch.Tensor],
    *,
    endecoder: torch.nn.Module,
    adapter: SMPLXGMRReference,
    encoder: SMP1PacketEncoder,
    absolute_start_frame: int,
) -> bytes:
    """Vectorized SMPL FK followed by the existing 14-target SMP1 contract."""
    required = {"body_pose": 63, "global_orient": 3, "transl": 3}
    frames: int | None = None
    for name, width in required.items():
        value = smpl_params.get(name)
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] != width:
            raise ValueError(f"smpl_params.{name} must have shape [T,{width}]")
        frames = len(value) if frames is None else frames
        if len(value) != frames or not torch.isfinite(value).all():
            raise ValueError("SMPL parameters have inconsistent or non-finite frames")
    assert frames is not None
    body_pose = smpl_params["body_pose"].unsqueeze(0)
    global_orient = smpl_params["global_orient"].unsqueeze(0)
    transl = smpl_params["transl"].unsqueeze(0)
    betas = torch.zeros(1, frames, 10, device=body_pose.device, dtype=body_pose.dtype)
    joints, _, fk_mat = endecoder.fk_v2(
        body_pose=body_pose,
        betas=betas,
        global_orient=global_orient,
        transl=transl,
        get_intermediate=True,
    )
    packets: list[bytes] = []
    for local_index in range(frames):
        absolute = absolute_start_frame + local_index
        adapted = adapter.adapt(
            joints[0, local_index, :22],
            fk_mat[0, local_index, :22, :3, :3],
            frame_id=absolute,
            timestamp_ns=int(round(absolute / SOURCE_FPS * 1e9)),
        )
        packets.append(
            encoder.pack_smplx_targets(
                adapted.scaled_targets,
                source_stamp_ns=int(round(absolute / SOURCE_FPS * 1e9)),
            )
        )
    return b"".join(packets)


@dataclass(frozen=True, slots=True)
class ConsoleCommand:
    name: str
    audio_path: Path | None = None
    duration_sec: float | None = None
    full: bool = False
    start_sec: float = 0.0
    seed: int = 42


def parse_console_line(line: str) -> ConsoleCommand | None:
    """Parse the resident ``robot>`` grammar without treating paths as commands."""
    stripped = line.strip()
    if not stripped:
        return None
    tokens = shlex.split(stripped)
    if not tokens:
        return None
    command = tokens[0].lower()
    if command in {"stand", "status", "quit", "exit", "shutdown", "help"}:
        if len(tokens) != 1:
            raise ValueError(f"{command} does not accept arguments")
        return ConsoleCommand("quit" if command == "exit" else command)
    if command == "play":
        tokens = tokens[1:]
    if not tokens:
        raise ValueError(
            'use: play "/absolute/song.wav" [seconds|full] [--start S --seed N]'
        )
    path = Path(tokens[0]).expanduser()
    option_start = 1
    if len(tokens) == 1 or tokens[1].startswith("--"):
        full = True
        duration = None
    else:
        duration_token = tokens[1].lower()
        full = duration_token == "full"
        duration = None if full else float(duration_token)
        option_start = 2
    if duration is not None and (not math.isfinite(duration) or duration <= 0.0):
        raise ValueError("music duration must be finite and > 0")
    start = 0.0
    seed = 42
    index = option_start
    while index < len(tokens):
        option = tokens[index]
        if index + 1 >= len(tokens):
            raise ValueError(f"missing value for {option}")
        value = tokens[index + 1]
        if option == "--start":
            start = float(value)
        elif option == "--seed":
            seed = int(value)
        else:
            raise ValueError(f"unknown play option: {option}")
        index += 2
    if not math.isfinite(start) or start < 0.0:
        raise ValueError("--start must be finite and >= 0")
    return ConsoleCommand(
        "play",
        audio_path=path,
        duration_sec=duration,
        full=full,
        start_sec=start,
        seed=seed,
    )


def heartbeat_expired(last_heartbeat: float, now: float | None = None, timeout: float = 1.5) -> bool:
    if timeout <= 0.0:
        raise ValueError("heartbeat timeout must be > 0")
    current = time.monotonic() if now is None else float(now)
    return current - float(last_heartbeat) > timeout
