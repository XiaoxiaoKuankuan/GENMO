"""第九步奖励专用的配对示范活动度，不参与 Actor 或 Critic 的输入。

本模块从 ``stage9.bc_data_root`` 的 train 清单读取当前音乐严格对应的 BUMI 动作，
核对清单行、音乐身份、动作来源 SHA、元数据和关节名称，再把 30 Hz 示范关节角按时间
线性插值到 50 Hz 区间端点，用每个真实控制区间的后向位置差计算 rad/s 速度。奖励只
读取与实际执行窗口相同的最近 0.5 秒速度 RMS；reset 后尚不足半秒时双方使用已经
执行的相同数量区间，明确标记窗口未满。源第零帧作为第一个区间左端点，不伪造零速度。

音乐任务长度采用已有 N/30 秒约定；最后源姿态位于 (N-1)/30 秒，因此最后不足一帧的
任务尾部显式保持末帧，不外推未知动作，并记录 terminal_hold_interval_count。越过
音乐末尾、时间不在 50 Hz 网格、动作缺失或身份不符均报错，不以 GENMO 生成参考代替。

``source_motion_sha256`` 是转换前原始动作的身份，按原 reader 规则与 motion payload
核对；它不是 motion.pt 文件哈希。本模块另记录实际读取的 motion_file_sha256，明确
区分来源声明核验和当前内容身份，不冒称重新打开了不在数据包中的原始动作文件。所有
张量只在 CPU 读取；本模块不加载网络、不运行 FK/GPU，也不修改音乐采样器条件字段。
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import re
from numbers import Integral
from pathlib import Path

import numpy as np
import torch

from gem.closedloop.contracts import GMT_EXPECTED_JOINT_ORDER
from gem.closedloop.dppo.music_tasks import SOURCES


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TIMEBASE_HZ = 600
_MOTION_TICKS = 20
_CONTROL_TICKS = 12
ACTIVITY_VERSION = "genmo.closedloop.paired_activity.linear_interval_rms.v1"


def _sha(value, field):
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"paired activity {field} must be a lowercase SHA256")
    return value


def _path(root, value, field):
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"paired activity {field} must be a relative path")
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"paired activity {field} escapes dataset root")
    if not path.is_file():
        raise FileNotFoundError(f"paired activity missing {field}: {path}")
    return path


class PairedActivityTarget:
    """在指定控制区间结束 tick 返回同窗口的配对示范关节速度 RMS。"""

    def __init__(self, joint_positions, *, joint_names, source, music_start_tick=600,
                 window_s=0.5, dt=0.02):
        if isinstance(music_start_tick, bool) or not isinstance(music_start_tick, Integral) or music_start_tick < 0:
            raise ValueError("paired activity music_start_tick must be a nonnegative integer")
        if not math.isfinite(float(dt)) or not math.isclose(float(dt), .02, rel_tol=0, abs_tol=1e-12):
            raise ValueError("paired activity requires 50 Hz dt=0.02")
        if not math.isfinite(float(window_s)) or float(window_s) <= 0:
            raise ValueError("paired activity window_s must be positive and finite")
        steps = float(window_s) / float(dt)
        if round(steps) < 1 or not math.isclose(steps, round(steps), rel_tol=0, abs_tol=1e-9):
            raise ValueError("paired activity window must contain whole 50 Hz intervals")
        names = tuple(joint_names)
        expected = tuple(GMT_EXPECTED_JOINT_ORDER)
        if len(names) != len(expected) or len(set(names)) != len(expected) or set(names) != set(expected):
            raise ValueError("paired activity requires the exact 21 BUMI/GMT joint names")
        positions = np.asarray(joint_positions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 21 or len(positions) < 2 or not np.isfinite(positions).all():
            raise ValueError("paired activity joint positions must be finite [T>=2,21] rad")
        if not isinstance(source, dict) or not source:
            raise ValueError("paired activity source provenance must be provided")
        self.music_start_tick = int(music_start_tick)
        self.window_steps = int(round(steps))
        self.dt = float(dt)
        self.num_frames = len(positions)
        self._positions = positions[:, [names.index(name) for name in expected]].copy()
        self._positions.setflags(write=False)
        self.source = copy.deepcopy(source)
        self.source.update({"activity_version": ACTIVITY_VERSION, "source_joint_order": list(names),
                            "joint_order": list(expected), "position_unit": "rad", "velocity_unit": "rad/s",
                            "source_fps": 30, "control_fps": 50, "timebase_hz": _TIMEBASE_HZ,
                            "music_start_tick": self.music_start_tick, "source_num_frames": self.num_frames,
                            "window_s": float(window_s), "dt": self.dt,
                            "velocity_operator": "linear_q_at_50hz_endpoints_then_backward_interval_difference",
                            "tail_policy": "hold_last_source_pose_until_music_duration_N_over_30",
                            "supervision_scope": "reward_only"})

    def _interpolate(self, elapsed_ticks):
        sample = np.asarray(elapsed_ticks, dtype=np.float64) / _MOTION_TICKS
        left = np.minimum(np.floor(sample).astype(np.int64), self.num_frames - 1)
        right = np.minimum(left + 1, self.num_frames - 1)
        weight = np.clip(sample - left, 0, 1)[:, None]
        return self._positions[left] * (1 - weight) + self._positions[right] * weight

    def __call__(self, tick):
        if isinstance(tick, bool) or not isinstance(tick, Integral):
            raise ValueError("paired activity tick must be an integer")
        elapsed = int(tick) - self.music_start_tick
        if elapsed <= 0 or elapsed % _CONTROL_TICKS:
            raise ValueError("paired activity requires a positive completed 50 Hz control interval")
        if elapsed > (self.num_frames * _MOTION_TICKS // _CONTROL_TICKS) * _CONTROL_TICKS:
            raise ValueError("paired activity tick exceeds paired music duration")
        count = min(elapsed // _CONTROL_TICKS, self.window_steps)
        endpoints = np.arange(elapsed - count * _CONTROL_TICKS, elapsed + 1, _CONTROL_TICKS, dtype=np.int64)
        velocity = np.diff(self._interpolate(endpoints), axis=0) / self.dt
        joint_rms = np.sqrt(np.mean(velocity ** 2, axis=0))
        return {"valid": True, "activity_rad_s": float(np.sqrt(np.mean(velocity ** 2))),
                "per_joint_rms_rad_s": joint_rms.tolist(), "window_count": int(count),
                "window_complete": count == self.window_steps,
                "window_begin_tick": self.music_start_tick + int(endpoints[0]), "window_end_tick": int(tick),
                "window_duration_s": count * self.dt,
                "terminal_hold_interval_count": int(np.sum(endpoints[1:] > (self.num_frames - 1) * _MOTION_TICKS)),
                "source": copy.deepcopy(self.source)}


def load_paired_activity(data_root, sample, *, music_start_tick=600, window_s=0.5, dt=0.02):
    """严格加载当前 train 音乐对应的动作；缺失/不匹配直接抛错。"""
    root = Path(data_root).expanduser().resolve()
    if not isinstance(sample, dict) or not isinstance(sample.get("row"), dict):
        raise ValueError("paired activity requires a selected train music sample")
    folder, row = sample.get("dataset"), sample["row"]
    if not isinstance(row.get("sample_id"), str) or not row["sample_id"] or not isinstance(sample.get("group_id"), str) or not sample["group_id"]:
        raise ValueError("paired activity requires explicit sample and music group identities")
    if not isinstance(folder, str) or folder not in SOURCES:
        raise ValueError("paired activity dataset must name one of the four training sources")
    # 本地选取数据通过这四个明确的目录 symlink 复用既有数据；解析后的目录是文件边界。
    source_root = (root / folder).resolve()
    manifest_path = _path(source_root, "manifests/train.jsonl", "train manifest")
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha != _sha(sample.get("manifest_sha256"), "manifest_sha256"):
        raise ValueError("paired activity selected train manifest SHA mismatch")
    rows = [json.loads(line) for line in manifest_bytes.decode().splitlines() if line.strip()]
    matches = [item for item in rows if item.get("sample_id") == row.get("sample_id")]
    if len(matches) != 1 or matches[0] != row or row.get("split") != "train" or row.get("fps") != 30:
        raise ValueError("paired activity row must exactly match its unique train manifest row")
    if isinstance(row.get("num_frames"), bool) or not isinstance(row.get("num_frames"), Integral) or row["num_frames"] < 2:
        raise ValueError("paired activity requires at least two aligned source frames")
    if row.get("quality_accepted") is not True:
        raise ValueError("paired activity requires an accepted training motion")
    if row.get("resplit_provenance", {}).get("group_id") != sample.get("group_id"):
        raise ValueError("paired activity music group identity mismatch")
    for name in ("source_motion_sha256", "source_music_feature_sha256", "source_audio_sha256"):
        _sha(row.get(name), name)
    music_path = _path(source_root, row.get("music_feature_path"), "music_feature_path")
    if hashlib.sha256(music_path.read_bytes()).hexdigest() != row["source_music_feature_sha256"]:
        raise ValueError("paired activity music feature SHA mismatch")
    info_path = _path(source_root, "meta/dataset_info.json", "dataset_info")
    info_bytes = info_path.read_bytes()
    info = json.loads(info_bytes)
    expected = {"contract_version": "genmo.bumi_music.v1", "fps": 30, "robot_name": "bumi",
                "qpos_order": "mujoco_native", "quaternion_convention": "wxyz"}
    for key, value in {**expected, "qpos_dim": 28, "joint_dim": 21, "quality_filter_applied": True}.items():
        if info.get(key) != value:
            raise ValueError(f"paired activity dataset metadata mismatch: {key}")
    if row.get("dataset") != info.get("dataset_name"):
        raise ValueError("paired activity dataset name differs from manifest")
    motion_path = _path(source_root, row.get("motion_path"), "motion_path")
    motion_bytes = motion_path.read_bytes()
    payload = torch.load(io.BytesIO(motion_bytes), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("paired activity motion payload must be a dictionary")
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"paired activity motion metadata mismatch: {key}")
    for key in ("source_mjcf_sha256", "quality_config_sha256", "retarget_config_sha256"):
        if payload.get(key) != _sha(info.get(key), key):
            raise ValueError(f"paired activity source metadata SHA mismatch: {key}")
    if payload.get("source_motion_sha256") != row["source_motion_sha256"]:
        raise ValueError("paired activity source_motion_sha256 mismatch")
    if payload.get("source_sample_id") != row["sample_id"] or payload.get("quality_accepted") is not True:
        raise ValueError("paired activity payload sample identity or quality mismatch")
    if payload.get("joint_names") != info.get("joint_names"):
        raise ValueError("paired activity payload joint order differs from dataset metadata")
    for key in ("ground_semantics", "root_z_adjusted"):
        if key not in info or payload.get(key) != info[key]:
            raise ValueError(f"paired activity source metadata mismatch: {key}")
    if "qpos" not in payload:
        raise ValueError("paired activity motion payload has no qpos")
    qpos = torch.as_tensor(payload["qpos"]).detach().cpu().numpy()
    if qpos.ndim != 2 or qpos.shape[1] != 28 or len(qpos) != row.get("num_frames") or not np.isfinite(qpos).all():
        raise ValueError("paired activity qpos shape, frame alignment or finite-value validation failed")
    if np.max(np.abs(np.linalg.norm(qpos[:, 3:7], axis=1) - 1)) > 1e-3:
        raise ValueError("paired activity root quaternion is not normalized")
    provenance = {"dataset": folder, "sample_id": row["sample_id"], "group_id": sample["group_id"],
                  "split": "train", "data_root": str(root), "resolved_source_root": str(source_root),
                  "manifest_path": str(manifest_path),
                  "manifest_sha256": manifest_sha, "motion_path": str(motion_path),
                  "motion_file_sha256": hashlib.sha256(motion_bytes).hexdigest(),
                  "source_motion_sha256": row["source_motion_sha256"],
                  "source_motion_sha_validation": "payload_equals_verified_train_manifest_declaration",
                  "source_music_feature_sha256": row["source_music_feature_sha256"],
                  "source_audio_sha256": row["source_audio_sha256"],
                  "dataset_info_sha256": hashlib.sha256(info_bytes).hexdigest()}
    return PairedActivityTarget(qpos[:, 7:], joint_names=payload["joint_names"], source=provenance,
                                music_start_tick=music_start_tick, window_s=window_s, dt=dt)
