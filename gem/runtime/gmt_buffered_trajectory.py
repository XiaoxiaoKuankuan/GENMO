"""BUMI 整段缓存轨迹的 Redis 发布协议。

完整的 float32[N,55] 轨迹只上传一次，使用独立魔数 OMGBF001、固定 50 Hz、关节顺序
哈希和 CRC 校验。后续 OMGBS001 小窗口包仅保持会话存活，不决定播放位置。GMT 必须
显式启用 buffered 模式，在每个策略更新中从本地缓存前进一帧，并通过 OMGBFA01 ACK
返回当前帧。旧 GMT 不认识这些魔数，会拒绝它们，不能静默退化为真实时间播放。

网络心跳仍用真实时间；仿真暂停时不推进轨迹。缓存带有租约，发布器消失后会自动过期，
控制端保留原有数据过期保护。本模块不改变旧 trajectory_v1 的编码和播放规则。
"""

from __future__ import annotations

import struct
import time
import zlib

import numpy as np

from gem.runtime.gmt_trajectory import (
    TRAJECTORY_ACK_FORMAT,
    TRAJECTORY_HEADER_FORMAT,
    TRAJECTORY_HEADER_SIZE,
    GmtTrajectoryAck,
    RedisTrajectoryPublisher,
    packet_for_cursor,
)

BUFFERED_CLIP_MAGIC = b"OMGBF001"
BUFFERED_CONTROL_MAGIC = b"OMGBS001"
BUFFERED_ACK_MAGIC = b"OMGBFA01"
BUFFERED_ACK_FORMAT = TRAJECTORY_ACK_FORMAT + "II"
BUFFERED_ACK_SIZE = struct.calcsize(BUFFERED_ACK_FORMAT)
MAX_BUFFERED_FRAMES = 65535


def encode_buffered_clip(frames, *, joint_order_hash, stream_id, revision, plan_id):
    """编码完整缓存；长度字段为 uint16，因此含过渡/尾垫最多 65535 帧。"""

    values = np.asarray(frames, dtype="<f4")
    if values.ndim != 2 or values.shape[1] != 55 or not 1 <= len(values) <= MAX_BUFFERED_FRAMES:
        raise ValueError("buffered clip must be float32[N,55], 1 <= N <= 65535")
    if not np.isfinite(values).all() or len(joint_order_hash) != 32:
        raise ValueError("buffered clip requires finite values and a 32-byte joint hash")
    norms = np.linalg.norm(values[:, 3:7], axis=1)
    if np.any((norms < 0.5) | (norms > 1.5)):
        raise ValueError("buffered clip contains invalid root quaternions")
    payload = values.tobytes(order="C")
    header = struct.pack(
        TRAJECTORY_HEADER_FORMAT,
        BUFFERED_CLIP_MAGIC,
        1,
        TRAJECTORY_HEADER_SIZE,
        0,
        stream_id,
        0,
        time.time_ns(),
        revision,
        plan_id,
        50.0,
        len(values),
        0,
        21,
        55,
        joint_order_hash,
        zlib.crc32(payload) & 0xFFFFFFFF,
    )
    return header + payload


class RedisBufferedTrajectoryPublisher(RedisTrajectoryPublisher):
    """发布不可变完整轨迹及真实时间心跳，只采纳 GMT 回报的播放位置。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip_key = f"{self.key}:clip:{self.stream_id}"
        self.clip_frames = None
        self.clip_identity = None
        self.next_lease_refresh = 0.0
        self.current_frame = 0

    def publish(
        self, frames, cursor, *, fps, joint_order_hash, command_revision=0, plan_id=-1, flags=0
    ):
        if fps != 50.0:
            raise ValueError("buffered playback requires 50 Hz")
        identity = (command_revision, plan_id, joint_order_hash)
        if self.clip_frames is None:
            encoded = encode_buffered_clip(
                frames,
                joint_order_hash=joint_order_hash,
                stream_id=self.stream_id,
                revision=command_revision,
                plan_id=plan_id,
            )
            self.client.set(self.clip_key, encoded, px=60_000)
            self.clip_frames = frames
            self.clip_identity = identity
            self.next_lease_refresh = time.monotonic() + 10.0
        elif frames is not self.clip_frames or identity != self.clip_identity:
            raise ValueError("buffered clip is immutable; a new plan needs a new publisher")
        if time.monotonic() >= self.next_lease_refresh:
            if not self.client.pexpire(self.clip_key, 60_000):
                raise RuntimeError("buffered clip lease disappeared")
            self.next_lease_refresh = time.monotonic() + 10.0
        # cursor 仅用于心跳包的诊断快照，接收端必须使用本地缓存帧号。
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
        self.client.set(self.key, BUFFERED_CONTROL_MAGIC + packet.encode()[8:], px=self.ttl_ms)
        self.last_packet = packet
        self.sequence += 1
        return packet

    def matching_ack(self):
        value = self.client.get(self.ack_key)
        if value is None or len(value) != BUFFERED_ACK_SIZE or self.last_packet is None:
            return None
        magic, version, size, stream, sequence, revision, plan, received, frame, count = (
            struct.unpack(BUFFERED_ACK_FORMAT, value)
        )
        if (
            magic != BUFFERED_ACK_MAGIC
            or version != 1
            or size != BUFFERED_ACK_SIZE
            or stream != self.stream_id
            or sequence > self.last_packet.sequence
            or revision != self.last_packet.command_revision
            or plan != self.last_packet.plan_id
            or self.clip_frames is None
            or count != len(self.clip_frames)
            or not self.current_frame <= frame < count
        ):
            return None
        self.current_frame = frame
        return GmtTrajectoryAck(stream, sequence, revision, plan, received)
