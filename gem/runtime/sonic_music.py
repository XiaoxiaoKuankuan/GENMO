"""SMPL 音乐部署的连续姿态适配、可靠会话客户端及声卡媒体时钟。

本模块将生成时钟和播放时钟分开：PoseTimeline 只接收一次性提交的 30 Hz 身体
参数，在全局 50 Hz 网格上用最短弧旋转插值，再执行 SONIC 的固定人体 FK。
协议只使用六个 G1 手腕参考，其他关节字段是 mode 2 不读取的占位数据。
SessionClient 的每次调用在调用线程内创建和关闭 REQ socket，锁保证跨生成线程
与监测线程的序号顺序；重试保持原请求与附件不变，避免 ZMQ socket 跨线程共享。
AudioClock 使用 PortAudio 输出时间估计已播放的采样位置，回调不做推理、网络或
文件操作。无声测试时钟必须显式选择，并在报告中标识为没有完成真实音频验收。
"""

from __future__ import annotations

import json
import math
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import zmq
from scipy.spatial.transform import Rotation

from gem.runtime.motion_streamer import synthetic_idle_motion
from gem.utils.sonic.resampler import _slerp_shortest
from gem.utils.sonic.smpl_converter import SonicSMPLConverter
from gem.utils.sonic.zmq_publisher import _compute_g1_wrist_joint_pos, _pack_pose_message_compat


class SessionClient:
    """串行、可重试的本地会话请求；超时不会更换请求编号。"""

    def __init__(self, endpoint: str, session_id: str = "", timeout_ms: int = 400):
        if not endpoint.startswith("tcp://127.0.0.1:") or not endpoint.rsplit(":", 1)[1].isdigit():
            raise ValueError("音乐客户端只允许本机 TCP 端点")
        self.endpoint, self.session_id = endpoint, session_id
        self.timeout_ms, self.seq = timeout_ms, 0
        self.context, self.lock = zmq.Context(), threading.Lock()

    def call(self, op: str, binary: bytes | None = None, **kwargs):
        with self.lock:
            request = dict(op=op, session_id=self.session_id, seq=self.seq, **kwargs)
            self.seq += 1
            parts = [json.dumps(request, separators=(",", ":")).encode()]
            if binary is not None:
                parts.append(binary)
            for attempt in range(3):
                socket = self.context.socket(zmq.REQ)
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
                socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
                try:
                    socket.connect(self.endpoint)
                    socket.send_multipart(parts)
                    reply = socket.recv_json()
                    if not reply.get("ok"):
                        raise RuntimeError(f"{self.endpoint} {op}: {reply.get('error')}")
                    return reply
                except zmq.Again:
                    if attempt == 2:
                        raise TimeoutError(f"{self.endpoint} {op} 请求超时") from None
                finally:
                    socket.close()

    def append(self, payload):
        indexes = payload["frame_index"]
        return self.call(
            "append",
            _pack_pose_message_compat(payload),
            start_frame=int(indexes[0]),
            end_frame=int(indexes[-1]),
        )

    def close(self):
        """调用方须先停止生产和监测线程，再关闭上下文。"""
        self.context.term()


class PoseTimeline:
    """在整首音频的采样网格上重采样；跨窗口保留插值端点和六腕连续角度。"""

    def __init__(self, sonic_root: Path, duration: float):
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("音频时长必须为正有限数")
        self.converter = SonicSMPLConverter(sonic_root, enable_yaw_calibration=False)
        from gear_sonic.trl.utils.rotation_conversion import decompose_rotation_aa

        self.decompose = decompose_rotation_aa
        self.target_frames = math.ceil(duration * 50 - 1e-9)
        self.source_count = self.output_count = self.sent_count = 0
        self.previous = None
        self.wrist_previous = None
        self.first_pose = self.last_pose = None
        self.pose_chunks = []
        self.bounds = np.zeros((29, 2), dtype=np.float64)
        xml = ET.parse(
            sonic_root / "gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml"
        )
        joints = {j.attrib.get("name"): j for j in xml.iter("joint")}
        for side, indices in (("left", (23, 25, 27)), ("right", (24, 26, 28))):
            for axis, index in zip(("roll", "pitch", "yaw"), indices):
                self.bounds[index] = [
                    float(v) for v in joints[f"{side}_wrist_{axis}_joint"].attrib["range"].split()
                ]

    @staticmethod
    def _interpolate(pose, root, times, source_start):
        """按 30 Hz 样本位置插值，只有最终音频尾段允许夹紧末帧。"""
        position = times * 30 - source_start
        low = np.floor(position + 1e-9).astype(int)
        alpha = np.clip(position - low, 0, 1)
        low = np.clip(low, 0, len(pose) - 1)
        high = np.minimum(low + 1, len(pose) - 1)
        body_q = Rotation.from_rotvec(pose.reshape(-1, 3)).as_quat().reshape(-1, 21, 4)
        root_q = Rotation.from_rotvec(root).as_quat()
        body = _slerp_shortest(body_q[low], body_q[high], alpha[:, None, None])
        orient = _slerp_shortest(root_q[low], root_q[high], alpha[:, None])
        return (
            Rotation.from_quat(body.reshape(-1, 4)).as_rotvec().reshape(-1, 63),
            Rotation.from_quat(orient).as_rotvec(),
        )

    def _pack(self, pose, root):
        """在输出帧率上执行 FK、手腕映射与限位，并赋予最终会话连续帧号。"""
        params = dict(
            body_pose=torch.as_tensor(pose, dtype=torch.float32),
            global_orient=torch.as_tensor(root, dtype=torch.float32),
        )
        converted = self.converter.convert(params, params)
        smpl_pose = converted["smpl_pose"].numpy()
        wrists = _compute_g1_wrist_joint_pos(smpl_pose, self.decompose)
        if self.wrist_previous is not None:
            wrists = np.unwrap(np.concatenate((self.wrist_previous[None], wrists)), axis=0)[1:]
        else:
            wrists = np.unwrap(wrists, axis=0)
        self.wrist_previous = wrists[-1].copy()
        wrists = np.clip(wrists, self.bounds[:, 0], self.bounds[:, 1]).astype(np.float32)
        n = len(pose)
        self.pose_chunks.append((self.sent_count, pose.copy(), root.copy()))
        result = dict(
            smpl_pose=smpl_pose.astype(np.float32),
            smpl_joints=converted["smpl_joints_local"].numpy().astype(np.float32),
            body_quat=converted["body_quat_w"].numpy().astype(np.float32),
            joint_pos=wrists,
            joint_vel=np.zeros((n, 29), dtype=np.float32),
            frame_index=np.arange(self.sent_count, self.sent_count + n, dtype=np.int64),
        )
        self.sent_count += n
        return result

    def push(self, params, is_last=False):
        """接收新提交帧；返回一个输出块，首块包含 1 秒站立和 1 秒起舞过渡。"""
        pose = params["body_pose"].detach().cpu().numpy().reshape(-1, 63)
        root = params["global_orient"].detach().cpu().numpy().reshape(-1, 3)
        if (
            not len(pose)
            or len(pose) != len(root)
            or not (np.isfinite(pose).all() and np.isfinite(root).all())
        ):
            raise ValueError("提交的 SMPL 参数形状错误或包含非有限数据")
        source_start = self.source_count
        self.source_count += len(pose)
        if self.previous is not None:
            pose = np.concatenate((self.previous[0], pose))
            root = np.concatenate((self.previous[1], root))
            source_start -= 1
        self.previous = pose[-1:].copy(), root[-1:].copy()
        stop = (
            self.target_frames
            if is_last
            else min(self.target_frames, math.floor((self.source_count - 1) * 50 / 30 + 1e-9) + 1)
        )
        times = np.arange(self.output_count, stop) / 50.0
        if not len(times):
            return None
        body, orient = self._interpolate(pose, root, times, source_start)
        self.output_count = stop
        self.last_pose = body[-1].copy(), orient[-1].copy()
        if self.first_pose is None:
            self.first_pose = body[0].copy(), orient[0].copy()
            idle = synthetic_idle_motion()
            idle_pose = idle.body_pose.numpy()[0]
            idle_root = orient[0]
            alpha = np.arange(1, 51) / 50
            alpha = alpha * alpha * (3 - 2 * alpha)
            p, r = self._blend(idle_pose, idle_root, body[0], orient[0], alpha)
            body = np.concatenate((np.repeat(idle_pose[None], 50, axis=0), p, body))
            orient = np.concatenate((np.repeat(idle_root[None], 50, axis=0), r, orient))
        return self._pack(body, orient)

    @staticmethod
    def _blend(p0, r0, p1, r1, alpha):
        """仅用于音乐区间之外的起舞和收尾，采用平滑进出权重。"""
        p = _slerp_shortest(
            Rotation.from_rotvec(p0.reshape(21, 3)).as_quat()[None],
            Rotation.from_rotvec(p1.reshape(21, 3)).as_quat()[None],
            alpha[:, None, None],
        )
        r = _slerp_shortest(
            Rotation.from_rotvec(r0).as_quat()[None],
            Rotation.from_rotvec(r1).as_quat()[None],
            alpha[:, None],
        )
        return Rotation.from_quat(p.reshape(-1, 4)).as_rotvec().reshape(-1, 63), Rotation.from_quat(
            r
        ).as_rotvec()

    def finish(self):
        """音频结束后追加 1 秒站立过渡及十帧未来窗口保护。"""
        if self.output_count != self.target_frames or self.last_pose is None:
            raise RuntimeError("音频参考尚未全部生成，不能收尾")
        idle = synthetic_idle_motion().body_pose.numpy()[0]
        alpha = np.arange(1, 51) / 50
        alpha = alpha * alpha * (3 - 2 * alpha)
        p, r = self._blend(*self.last_pose, idle, self.last_pose[1], alpha)
        return self._pack(
            np.concatenate((p, np.repeat(p[-1:], 10, axis=0))),
            np.concatenate((r, np.repeat(r[-1:], 10, axis=0))),
        )

    def graceful_tail(self, cut_frame):
        """显式用户停止时，从尚未播放的参考连续转入站立；调用前必须停止生成线程。"""
        for start, pose, root in self.pose_chunks:
            if start <= cut_frame < start + len(pose):
                p0, r0 = pose[cut_frame - start], root[cut_frame - start]
                break
        else:
            raise ValueError("停止帧尚未生成")
        alpha = np.linspace(0, 1, 50)
        p, r = self._blend(
            p0,
            r0,
            synthetic_idle_motion().body_pose.numpy()[0],
            r0,
            alpha * alpha * (3 - 2 * alpha),
        )
        self.sent_count, self.wrist_previous = int(cut_frame), None
        return self._pack(
            np.concatenate((p, np.repeat(p[-1:], 10, axis=0))),
            np.concatenate((r, np.repeat(r[-1:], 10, axis=0))),
        )


class AudioClock:
    """预打开音频设备，以 DAC 采样时间估计媒体进度；静音测试需要显式 off。"""

    def __init__(self, pcm, sample_rate, output="device", device=None):
        self.pcm = np.ascontiguousarray(pcm, dtype=np.float32)
        self.sample_rate, self.output = int(sample_rate), output
        self.epoch_ns = 0
        self.next_sample = 0
        self.anchor = None
        self.dac_sample = (0, 0)
        self.error = ""
        self.fade_end_sample = None
        self.stream = None
        self.pa_to_monotonic_ns = None
        if output == "device":
            import sounddevice as sd

            if isinstance(device, str) and device.isdigit():
                device = int(device)
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.pcm.shape[1],
                dtype="float32",
                blocksize=0,
                latency=0.05,
                device=device,
                callback=self._callback,
            )
            self.stream.start()
            # 在回调外校准两个时钟，选择调用跨度最小的一次，避免 Python 回调调度延迟
            # 被误当成声卡缓冲延迟；后续媒体位置仍由每块的 DAC 时间和采样编号更新。
            mappings = []
            for _ in range(8):
                before = time.monotonic_ns()
                portaudio_ns = round(self.stream.time * 1e9)
                after = time.monotonic_ns()
                mappings.append((after - before, (before + after) // 2 - portaudio_ns))
            self.pa_to_monotonic_ns = min(mappings)[1]

    def arm(self, epoch_ns):
        """起播前已有静音回调，预约音乐第一个采样的单调时钟时间。"""
        self.epoch_ns = int(epoch_ns)

    def fade(self, end_ns):
        """在回调之外准备 250 ms 淡出 PCM，保持之前的采样时间不变。"""
        end = min(
            len(self.pcm),
            max(self.next_sample, round((end_ns - self.epoch_ns) * self.sample_rate / 1e9)),
        )
        start = max(self.next_sample, end - round(0.25 * self.sample_rate))
        faded = self.pcm.copy()
        faded[start:end] *= np.linspace(1, 0, end - start, dtype=np.float32)[:, None]
        faded[end:] = 0
        self.pcm, self.fade_end_sample = faded, end

    def _callback(self, outdata, frames, timing, status):
        """仅填充 PCM 和采样位置；异常由主线程检查并协调冻结。"""
        outdata.fill(0)
        if status and self.epoch_ns:
            self.error = "音频输出发生 underflow 或设备异常"
        now_ns = time.monotonic_ns()
        dac_ns = now_ns + int((timing.outputBufferDacTime - timing.currentTime) * 1e9)
        if self.pa_to_monotonic_ns is not None:
            dac_ns = self.pa_to_monotonic_ns + round(timing.outputBufferDacTime * 1e9)
        if not self.epoch_ns:
            return
        offset = 0
        if self.anchor is None:
            offset = max(0, round((self.epoch_ns - dac_ns) * self.sample_rate / 1e9))
            if offset >= frames:
                return
            self.anchor = dac_ns + round(offset / self.sample_rate * 1e9)
        count = min(frames - offset, len(self.pcm) - self.next_sample)
        if count > 0:
            self.dac_sample = (self.next_sample, dac_ns + round(offset / self.sample_rate * 1e9))
            outdata[offset : offset + count] = self.pcm[self.next_sample : self.next_sample + count]
            self.next_sample += count

    def position(self, monotonic_ns=None):
        """返回已估计送到 DAC 的媒体秒数，而非已排入声卡队列的长度。"""
        now = time.monotonic_ns() if monotonic_ns is None else monotonic_ns
        sample, anchor = (0, self.epoch_ns) if self.output == "off" else self.dac_sample
        if not anchor:
            return 0.0
        return min(
            len(self.pcm) / self.sample_rate,
            max(0.0, sample / self.sample_rate + (now - anchor) / 1e9),
        )

    def close(self, fault=False):
        """故障立即清空音频输出；正常退出等待已经排队的尾部静音。"""
        if self.stream:
            self.stream.abort() if fault else self.stream.stop()
            self.stream.close()
            self.stream = None
