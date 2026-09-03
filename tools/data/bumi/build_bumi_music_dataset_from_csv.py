#!/usr/bin/env python3
"""把自建 BUMI CSV/WAV 原子发布为可与正式四库联合训练的数据集。

输入目录必须包含 ``dance_2_csv``、``dance_2_音频``、``dance_3_csv`` 和
``dance_3_音频``。工具先严格解析 ``bumi_<歌曲>_<fps>fps.csv``，核对唯一 WAV、
28 列表头、帧率、finite、xyzw 单位四元数、关节限位、根高度、有限差分动力学和
绑定 kinematics 的 FK 贴地规则；CSV 的 21 个关节按配置中的源关节名重排到目标
kinematics 顺序，绝不假设新旧 BUMI3 MJCF 的 qpos 顺序相同。配置中排除的
APT啦啦操会进入审计报告，但绝不会进入正式 manifest。

通过项将根四元数改为连续 wxyz；50 Hz 的根位置/关节用线性插值、根旋转用最短弧
SLERP 重采样到 30 Hz。目标帧数取动作和 WAV 在 30 Hz 下都能完整覆盖的整数帧下界，
输出 WAV 精确裁成 ``T / 30`` 秒后才提取 EDGE baseline35，并将 EDGE35 显式裁成
``[T,35]``。随后把仅由新旧 MJCF 关节上限微差造成的至多 0.01 rad 源越界裁回当前
限位，用同一 kinematics 对 qpos28 做 FK，给整条 root Z 加常量，使所有 body origin
的最小 Z 为零，并在完整序列上重算版本化左右脚接触标签。

motion、EDGE35、裁后 WAV、train manifest、dataset_info、逐条质量报告和转换报告均
先写入同盘 staging；完整 strict reader 与三类 SHA 验证通过后才原子发布。工具拒绝
覆盖已有输出，也不会修改原始 CSV/WAV。离线质量 PASS 只表示数据契约和运动学规则
通过，不代表 SONIC/GMT 动力学可跟踪、平衡或实物安全。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import tempfile
import wave
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gem.datasets.music_dance.music_dance_bumi import (  # noqa: E402
    BUMI_MUSIC_CONTRACT_VERSION,
    BumiMusicDatasetReader,
    resolve_contract_path,
    sha256_file,
)
from gem.robots.bumi.contacts import (  # noqa: E402
    BUMI_CONTACT_CONTRACT_VERSION,
    derive_bumi_foot_contact,
)
from gem.robots.bumi.kinematics import BumiKinematics  # noqa: E402
from gem.utils.music_features import extract_edge_baseline35  # noqa: E402
from tools.data.bumi.build_bumi_music_dataset_from_sonic_npz import (  # noqa: E402
    _slerp_pairs,
    make_quaternion_continuous_np,
    normalize_body_origin_ground,
)

CSV_QUALITY_CONTRACT_VERSION = "genmo.bumi_csv_quality_config.v2"
CSV_SOURCE_CONTRACT_VERSION = "genmo.bumi_csv_qpos_xyzw_named.v2"
CSV_RESAMPLE_CONTRACT_VERSION = "genmo.bumi_csv_to_30hz.v1"
GROUND_SEMANTICS = "legacy_body_origin_min_zero"
OUTPUT_JOINT_LIMIT_TOLERANCE_RAD = 1.0e-4
CSV_NAME = re.compile(r"^bumi_(.+)_(30|50)fps$")


@dataclass(frozen=True)
class CsvQualityConfig:
    dataset_name: str
    expected_kinematics_sha256: str
    allowed_fps: tuple[int, ...]
    csv_header: tuple[str, ...]
    source_joint_names: tuple[str, ...]
    wav_sample_rate: int
    wav_channels: int
    wav_sample_width_bytes: int
    excluded_songs: tuple[str, ...]
    expected_candidates: int
    expected_accepted: int
    hard: Mapping[str, float | int]
    soft: Mapping[str, float | int]
    dynamics: Mapping[str, float]
    floor: Mapping[str, float | int]


@dataclass(frozen=True)
class SourcePair:
    part: str
    song: str
    fps: int
    csv_path: Path
    audio_path: Path
    audio_alias_used: bool

    @property
    def sample_id(self) -> str:
        return f"{self.part}__{self.song}"


@dataclass
class AuditedPair:
    source: SourcePair
    qpos: np.ndarray
    audio_frames: int
    audio_duration_sec: float
    source_motion_sha256: str
    source_audio_sha256: str
    quality: dict[str, Any]
    target_frames: int


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} 必须是 mapping")
    return value


def _positive_int(parent: Mapping[str, Any], key: str) -> int:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} 必须是正整数")
    return value


def _finite_float(parent: Mapping[str, Any], key: str) -> float:
    value = float(parent.get(key, float("nan")))
    if not math.isfinite(value):
        raise ValueError(f"{key} 必须是有限数值")
    return value


def load_quality_config(path: str | Path) -> CsvQualityConfig:
    """严格加载自建 CSV/WAV 质量配置。"""

    config_path = Path(path).expanduser().resolve(strict=True)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("contract_version") != CSV_QUALITY_CONTRACT_VERSION:
        raise ValueError(f"质量配置必须是 {CSV_QUALITY_CONTRACT_VERSION}")
    source = _mapping(raw, "source")
    selection = _mapping(raw, "selection")
    hard_raw = _mapping(raw, "hard_thresholds")
    soft_raw = _mapping(raw, "soft_policy")
    dynamics_raw = _mapping(raw, "dynamics")
    floor_raw = _mapping(raw, "floor_style")
    header = tuple(map(str, source.get("csv_header", ())))
    if len(header) != 28 or len(set(header)) != 28:
        raise ValueError("source.csv_header 必须是 28 个不重复字段")
    allowed_fps = tuple(int(value) for value in source.get("allowed_fps", ()))
    if allowed_fps != (30, 50):
        raise ValueError("source.allowed_fps 必须严格为 [30, 50]")
    if source.get("quaternion_convention") != "xyzw":
        raise ValueError("source.quaternion_convention 必须为 xyzw")
    source_joint_names = tuple(map(str, source.get("source_joint_names", ())))
    if len(source_joint_names) != 21 or len(set(source_joint_names)) != 21:
        raise ValueError("source.source_joint_names 必须是 21 个不重复关节名")
    excluded = tuple(map(str, selection.get("excluded_songs", ())))
    if not excluded or len(excluded) != len(set(excluded)):
        raise ValueError("selection.excluded_songs 必须非空且不重复")
    hard_keys = (
        "minimum_frames",
        "quaternion_norm_error_max",
        "joint_limit_violation_max",
        "root_height_min_absolute",
        "root_height_max_absolute",
    )
    soft_keys = ("exceed_ratio_max", "consecutive_exceed_frames", "severe_multiplier")
    dynamics_keys = (
        "joint_velocity_l2",
        "joint_acceleration_l2",
        "joint_jerk_l2",
        "root_linear_velocity",
        "root_angular_velocity",
    )
    floor_keys = (
        "root_low_height",
        "root_low_tilt_degrees",
        "torso_ground_height",
        "upper_body_ground_height",
        "gate_root_height",
        "gate_tilt_degrees",
        "ankles_airborne_height",
        "reject_consecutive_frames",
        "review_ratio",
        "review_min_frames",
        "low_root_review_height",
        "low_root_review_consecutive_frames",
    )
    hard = {key: _finite_float(hard_raw, key) for key in hard_keys}
    soft = {key: _finite_float(soft_raw, key) for key in soft_keys}
    dynamics = {key: _finite_float(dynamics_raw, key) for key in dynamics_keys}
    floor = {key: _finite_float(floor_raw, key) for key in floor_keys}
    if int(hard["minimum_frames"]) < 4:
        raise ValueError("minimum_frames 不能小于 4")
    if not 0.0 <= soft["exceed_ratio_max"] <= 1.0:
        raise ValueError("exceed_ratio_max 必须位于 [0,1]")
    if soft["severe_multiplier"] <= 1.0:
        raise ValueError("severe_multiplier 必须大于 1")
    digest = str(raw.get("expected_kinematics_sha256", ""))
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("expected_kinematics_sha256 必须是小写 SHA256")
    return CsvQualityConfig(
        dataset_name=str(raw.get("dataset_name", "")),
        expected_kinematics_sha256=digest,
        allowed_fps=allowed_fps,
        csv_header=header,
        source_joint_names=source_joint_names,
        wav_sample_rate=_positive_int(source, "wav_sample_rate"),
        wav_channels=_positive_int(source, "wav_channels"),
        wav_sample_width_bytes=_positive_int(source, "wav_sample_width_bytes"),
        excluded_songs=excluded,
        expected_candidates=_positive_int(selection, "expected_candidates"),
        expected_accepted=_positive_int(selection, "expected_accepted"),
        hard=hard,
        soft=soft,
        dynamics=dynamics,
        floor=floor,
    )


def discover_source_pairs(source_root: Path, config: CsvQualityConfig) -> list[SourcePair]:
    """按两个舞蹈分区发现一一对应的 CSV/WAV，并拒绝孤儿文件。"""

    root = source_root.expanduser().resolve(strict=True)
    pairs: list[SourcePair] = []
    consumed_audio: set[Path] = set()
    for part in ("dance_2", "dance_3"):
        csv_root = root / f"{part}_csv"
        audio_root = root / f"{part}_音频"
        if not csv_root.is_dir() or not audio_root.is_dir():
            raise FileNotFoundError(f"缺少自建数据目录: {csv_root} 或 {audio_root}")
        csv_paths = sorted(csv_root.glob("*.csv"))
        audio_paths = sorted(audio_root.glob("*.wav"))
        audio_by_stem = {path.stem: path for path in audio_paths}
        if len(audio_by_stem) != len(audio_paths):
            raise ValueError(f"{audio_root} WAV stem 重复")
        for csv_path in csv_paths:
            match = CSV_NAME.fullmatch(csv_path.stem)
            if match is None:
                raise ValueError(f"CSV 文件名不符合 bumi_<歌曲>_<fps>fps: {csv_path}")
            song, fps_text = match.groups()
            fps = int(fps_text)
            if fps not in config.allowed_fps:
                raise ValueError(f"{csv_path}: 不支持 {fps} Hz")
            if part == "dance_2" and fps != 30:
                raise ValueError(f"{csv_path}: dance_2 必须为 30 Hz")
            direct = audio_by_stem.get(song)
            alias = audio_by_stem.get(f"{song}_{fps}fps")
            if direct is not None and alias is not None:
                raise ValueError(f"{csv_path}: 同时存在直接 WAV 和帧率别名 WAV")
            audio_path = direct if direct is not None else alias
            if audio_path is None:
                raise FileNotFoundError(f"{csv_path}: 找不到歌曲 WAV")
            if audio_path in consumed_audio:
                raise ValueError(f"WAV 被多个动作复用且未显式建组: {audio_path}")
            consumed_audio.add(audio_path)
            pairs.append(
                SourcePair(
                    part=part,
                    song=song,
                    fps=fps,
                    csv_path=csv_path,
                    audio_path=audio_path,
                    audio_alias_used=direct is None,
                )
            )
        orphans = set(audio_paths) - consumed_audio
        if orphans:
            raise ValueError(
                f"{audio_root}: 存在未配对 WAV: {[path.name for path in sorted(orphans)]}"
            )
    sample_ids = [pair.sample_id for pair in pairs]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("自建数据 sample_id 重复")
    if len(pairs) != config.expected_candidates:
        raise ValueError(
            f"候选数量不一致: expected={config.expected_candidates}, actual={len(pairs)}"
        )
    return pairs


def load_csv_qpos(
    pair: SourcePair,
    config: CsvQualityConfig,
    target_joint_names: Sequence[str],
) -> np.ndarray:
    """读取 CSV，把 xyzw 变为 wxyz，并按名字转换到目标 MJCF 关节顺序。"""

    with pair.csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = tuple(next(csv.reader(handle)))
    if header != config.csv_header:
        raise ValueError(f"{pair.csv_path}: CSV 表头不匹配")
    value = np.loadtxt(pair.csv_path, delimiter=",", skiprows=1, dtype=np.float64)
    if value.ndim == 1:
        value = value[None]
    if value.ndim != 2 or value.shape[1] != 28:
        raise ValueError(f"{pair.csv_path}: 数据必须为 [T,28]，实际 {value.shape}")
    if len(value) < int(config.hard["minimum_frames"]) or not np.isfinite(value).all():
        raise ValueError(f"{pair.csv_path}: 帧数不足或包含 NaN/Inf")
    quaternion_xyzw = value[:, 3:7]
    quaternion_wxyz = make_quaternion_continuous_np(quaternion_xyzw[:, [3, 0, 1, 2]])
    target_names = tuple(map(str, target_joint_names))
    if len(target_names) != 21 or len(set(target_names)) != 21:
        raise ValueError("target_joint_names 必须是 21 个不重复关节名")
    if set(target_names) != set(config.source_joint_names):
        raise ValueError(
            "CSV 源关节集合与目标 kinematics 不一致: "
            f"source_only={sorted(set(config.source_joint_names) - set(target_names))}, "
            f"target_only={sorted(set(target_names) - set(config.source_joint_names))}"
        )
    source_index = {name: index for index, name in enumerate(config.source_joint_names)}
    reorder = [source_index[name] for name in target_names]
    target_joints = value[:, 7:][:, reorder]
    qpos = np.concatenate((value[:, :3], quaternion_wxyz, target_joints), axis=-1)
    return np.ascontiguousarray(qpos, dtype=np.float64)


def read_wav_metadata(path: Path, config: CsvQualityConfig) -> tuple[int, float]:
    """验证源 WAV PCM 契约并返回采样帧数和秒数。"""

    with wave.open(str(path), "rb") as handle:
        if handle.getframerate() != config.wav_sample_rate:
            raise ValueError(f"{path}: sample rate 必须为 {config.wav_sample_rate}")
        if handle.getnchannels() != config.wav_channels:
            raise ValueError(f"{path}: channels 必须为 {config.wav_channels}")
        if handle.getsampwidth() != config.wav_sample_width_bytes:
            raise ValueError(f"{path}: sample width 必须为 {config.wav_sample_width_bytes}")
        if handle.getcomptype() != "NONE":
            raise ValueError(f"{path}: 必须是未压缩 PCM WAV")
        frames = int(handle.getnframes())
    if frames <= 0:
        raise ValueError(f"{path}: WAV 为空")
    return frames, frames / config.wav_sample_rate


def _longest_true_run(mask: np.ndarray) -> int:
    longest = current = 0
    for value in np.asarray(mask, dtype=np.bool_).reshape(-1):
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return longest


def _signal_metrics(values: np.ndarray, threshold: float, config: CsvQualityConfig) -> dict:
    sequence = np.asarray(values, dtype=np.float64).reshape(-1)
    exceed = sequence > threshold
    maximum = float(np.max(sequence)) if sequence.size else 0.0
    p95 = float(np.percentile(sequence, 95.0)) if sequence.size else 0.0
    ratio = float(np.mean(exceed)) if sequence.size else 0.0
    run = _longest_true_run(exceed)
    severe = maximum > threshold * float(config.soft["severe_multiplier"])
    broad = p95 > threshold and ratio > float(config.soft["exceed_ratio_max"])
    sustained = run >= int(config.soft["consecutive_exceed_frames"])
    status = "REJECT" if severe else ("REVIEW" if broad or sustained else "PASS")
    return {
        "max": maximum,
        "p95": p95,
        "exceed_ratio": ratio,
        "max_consecutive_exceed_frames": run,
        "threshold": threshold,
        "status": status,
    }


def evaluate_qpos_quality(
    qpos: np.ndarray,
    *,
    fps: int,
    kinematics: BumiKinematics,
    config: CsvQualityConfig,
) -> dict[str, Any]:
    """执行 CSV 可复算的 auto025 关节、动态、根姿态与 FK 贴地检查。"""

    value = np.asarray(qpos, dtype=np.float64)
    quaternion = value[:, 3:7]
    quaternion_norm_error = float(np.max(np.abs(np.linalg.norm(quaternion, axis=-1) - 1.0)))
    adjacent = np.sum(quaternion[:-1] * quaternion[1:], axis=-1)
    joints = value[:, 7:]
    lower = kinematics.joint_lower_limits.detach().cpu().numpy().astype(np.float64)
    upper = kinematics.joint_upper_limits.detach().cpu().numpy().astype(np.float64)
    violation = np.maximum(np.maximum(lower[None] - joints, joints - upper[None]), 0.0)
    per_joint_violation = np.max(violation, axis=0)
    root = value[:, :3]
    qdot = np.clip(np.abs(np.sum(quaternion[:-1] * quaternion[1:], axis=-1)), 0.0, 1.0)
    signals = {
        "joint_velocity_l2": np.linalg.norm(np.diff(joints, axis=0) * fps, axis=-1),
        "joint_acceleration_l2": np.linalg.norm(np.diff(joints, n=2, axis=0) * fps**2, axis=-1),
        "joint_jerk_l2": np.linalg.norm(np.diff(joints, n=3, axis=0) * fps**3, axis=-1),
        "root_linear_velocity": np.linalg.norm(np.diff(root, axis=0) * fps, axis=-1),
        "root_angular_velocity": 2.0 * np.arccos(qdot) * fps,
    }
    dynamics = {
        name: _signal_metrics(signal, float(config.dynamics[name]), config)
        for name, signal in signals.items()
    }
    with torch.no_grad():
        body = (
            kinematics.forward_kinematics(torch.from_numpy(value.astype(np.float32)))["body_pos_w"]
            .detach()
            .cpu()
            .numpy()
        )
    body_index = kinematics.body_name_to_index
    root_height = root[:, 2]
    root_tilt = np.degrees(
        np.arccos(np.clip(1.0 - 2.0 * (quaternion[:, 1] ** 2 + quaternion[:, 2] ** 2), -1.0, 1.0))
    )
    torso_height = np.mean(
        body[:, [body_index["l_arm_pitch_link"], body_index["r_arm_pitch_link"]], 2], axis=1
    )
    upper_names = (
        "l_arm_pitch_link",
        "l_arm_roll_link",
        "l_arm_yaw_link",
        "l_elbow_pitch_link",
        "r_arm_pitch_link",
        "r_arm_roll_link",
        "r_arm_yaw_link",
        "r_elbow_pitch_link",
    )
    upper_height = np.min(body[:, [body_index[name] for name in upper_names], 2], axis=1)
    ankle_height = np.min(
        body[:, [body_index["l_ankle_roll_link"], body_index["r_ankle_roll_link"]], 2], axis=1
    )
    floor_evidence = (
        (
            (root_height < float(config.floor["root_low_height"]))
            & (root_tilt > float(config.floor["root_low_tilt_degrees"]))
        )
        | (torso_height < float(config.floor["torso_ground_height"]))
        | (upper_height < float(config.floor["upper_body_ground_height"]))
    )
    floor_gate = (
        (root_height < float(config.floor["gate_root_height"]))
        | (root_tilt > float(config.floor["gate_tilt_degrees"]))
        | (ankle_height > float(config.floor["ankles_airborne_height"]))
    )
    floor_mask = floor_evidence & floor_gate
    floor_count = int(np.count_nonzero(floor_mask))
    floor_ratio = float(np.mean(floor_mask))
    floor_run = _longest_true_run(floor_mask)
    low_root_run = _longest_true_run(root_height < float(config.floor["low_root_review_height"]))
    reasons: list[str] = []
    if quaternion_norm_error > float(config.hard["quaternion_norm_error_max"]):
        reasons.append("ROOT_QUATERNION_NORM_REJECT")
    max_violation = float(np.max(per_joint_violation))
    if max_violation > float(config.hard["joint_limit_violation_max"]):
        reasons.append("SOURCE_JOINT_LIMIT_REJECT")
    if float(np.min(root_height)) < float(config.hard["root_height_min_absolute"]):
        reasons.append("ROOT_HEIGHT_BELOW_REJECT")
    if float(np.max(root_height)) > float(config.hard["root_height_max_absolute"]):
        reasons.append("ROOT_HEIGHT_ABOVE_REJECT")
    for name, metric in dynamics.items():
        if metric["status"] != "PASS":
            reasons.append(f"{name.upper()}_{metric['status']}")
    if floor_run >= int(config.floor["reject_consecutive_frames"]):
        reasons.append("FLOOR_STYLE_REJECT")
    elif floor_count >= int(config.floor["review_min_frames"]) and floor_ratio >= float(
        config.floor["review_ratio"]
    ):
        reasons.append("FLOOR_STYLE_REVIEW")
    if low_root_run >= int(config.floor["low_root_review_consecutive_frames"]) and not any(
        reason == "FLOOR_STYLE_REJECT" for reason in reasons
    ):
        reasons.append("LOW_ROOT_REVIEW")
    status = (
        "REJECT"
        if any(reason.endswith("REJECT") for reason in reasons)
        else ("REVIEW" if reasons else "PASS")
    )
    return {
        "status": status,
        "quality_accepted": status == "PASS",
        "reason_codes": reasons,
        "metrics": {
            "num_frames": int(len(value)),
            "fps": int(fps),
            "finite": bool(np.isfinite(value).all()),
            "root_quaternion_norm_max_error": quaternion_norm_error,
            "root_quaternion_adjacent_dot_min": float(np.min(adjacent)) if adjacent.size else 1.0,
            "root_quaternion_sign_flip_count": int(np.count_nonzero(adjacent < 0.0)),
            "joint_limit_violation_max": max_violation,
            "joint_limit_violation_max_by_joint": {
                name: float(amount)
                for name, amount in zip(kinematics.joint_order, per_joint_violation, strict=True)
                if amount > 0.0
            },
            "root_height_min": float(np.min(root_height)),
            "root_height_max": float(np.max(root_height)),
            "body_origin_ground_min": float(np.min(body[..., 2])),
            "dynamics": dynamics,
            "floor_style": {
                "frame_count": floor_count,
                "frame_ratio": floor_ratio,
                "max_consecutive_frames": floor_run,
                "low_root_max_consecutive_frames": low_root_run,
                "root_tilt_max_degrees": float(np.max(root_tilt)),
            },
        },
    }


def target_frame_count(
    source_frames: int, source_fps: int, audio_frames: int, audio_sample_rate: int
) -> int:
    """取动作和音频都能完整覆盖的 30 Hz 整数帧下界。"""

    motion_cap = math.floor(source_frames * 30 / source_fps + 1.0e-9)
    audio_cap = math.floor(audio_frames * 30 / audio_sample_rate + 1.0e-9)
    target = min(motion_cap, audio_cap)
    if target <= 0:
        raise ValueError("动作/音频共同 30 Hz 帧数必须为正")
    return target


def resample_qpos_to_30hz(qpos: np.ndarray, source_fps: int, target_frames: int) -> torch.Tensor:
    """把连续 wxyz qpos 从 30/50 Hz 重采样到严格 ``[T,28]``。"""

    source = np.asarray(qpos, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 28 or len(source) <= 0:
        raise ValueError(f"qpos 必须为 [T,28]，实际 {source.shape}")
    if source_fps not in {30, 50} or target_frames <= 0:
        raise ValueError("source_fps 必须为 30/50，target_frames 必须为正")
    source_position = np.arange(target_frames, dtype=np.float64) * source_fps / 30.0
    if float(source_position[-1]) > len(source) - 1 + 1.0e-9:
        raise ValueError("目标 30 Hz 时间网格超出源动作")
    lower = np.floor(source_position).astype(np.int64)
    upper = np.minimum(lower + 1, len(source) - 1)
    alpha = source_position - lower
    linear_source = np.concatenate((source[:, :3], source[:, 7:]), axis=-1)
    linear = linear_source[lower] * (1.0 - alpha[:, None]) + linear_source[upper] * alpha[:, None]
    quaternion = _slerp_pairs(source[lower, 3:7], source[upper, 3:7], alpha)
    result = np.concatenate((linear[:, :3], quaternion, linear[:, 3:]), axis=-1)
    if result.shape != (target_frames, 28) or not np.isfinite(result).all():
        raise RuntimeError(f"30 Hz qpos 构造失败: {result.shape}")
    return torch.from_numpy(result.astype(np.float32, copy=False)).contiguous()


def write_cropped_wav(
    source: Path, destination: Path, target_frames: int, config: CsvQualityConfig
) -> int:
    """将 WAV 精确裁成 ``target_frames / 30`` 秒并保留 PCM 参数。"""

    if config.wav_sample_rate % 30 != 0:
        raise ValueError("WAV sample rate 必须能被 30 整除才能做精确帧对齐")
    required = target_frames * (config.wav_sample_rate // 30)
    with wave.open(str(source), "rb") as reader:
        params = reader.getparams()
        if required > reader.getnframes():
            raise ValueError(f"{source}: 不足以裁出 {target_frames} 个 30 Hz 帧")
        payload = reader.readframes(required)
    expected_bytes = required * params.nchannels * params.sampwidth
    if len(payload) != expected_bytes:
        raise ValueError(f"{source}: WAV payload 字节数不足")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(payload)
    with wave.open(str(destination), "rb") as check:
        if check.getnframes() != required:
            raise RuntimeError(f"裁后 WAV 帧数校验失败: {destination}")
    return required


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _audit_pairs(
    pairs: Sequence[SourcePair],
    *,
    kinematics: BumiKinematics,
    config: CsvQualityConfig,
    quality_config_sha256: str,
) -> tuple[list[AuditedPair], list[dict[str, Any]]]:
    accepted: list[AuditedPair] = []
    report_rows: list[dict[str, Any]] = []
    for pair in pairs:
        qpos = load_csv_qpos(pair, config, kinematics.joint_order)
        audio_frames, audio_duration = read_wav_metadata(pair.audio_path, config)
        target_frames = target_frame_count(
            len(qpos), pair.fps, audio_frames, config.wav_sample_rate
        )
        source_motion_sha = sha256_file(pair.csv_path)
        source_audio_sha = sha256_file(pair.audio_path)
        base = {
            "report_contract_version": CSV_QUALITY_CONTRACT_VERSION,
            "sample_id": pair.sample_id,
            "part": pair.part,
            "song": pair.song,
            "source_fps": pair.fps,
            "source_num_frames": int(len(qpos)),
            "source_motion_duration_sec": len(qpos) / pair.fps,
            "source_audio_duration_sec": audio_duration,
            "source_csv": pair.csv_path.name,
            "source_audio": pair.audio_path.name,
            "audio_alias_used": pair.audio_alias_used,
            "source_motion_sha256": source_motion_sha,
            "source_audio_sha256": source_audio_sha,
            "quality_config_sha256": quality_config_sha256,
            "target_frames": target_frames,
            "target_duration_sec": target_frames / 30.0,
        }
        if pair.song in config.excluded_songs:
            report_rows.append(
                {
                    **base,
                    "status": "EXCLUDED",
                    "quality_accepted": False,
                    "reason_codes": ["EXPLICIT_SONG_EXCLUSION"],
                    "metrics": {},
                }
            )
            continue
        quality = evaluate_qpos_quality(qpos, fps=pair.fps, kinematics=kinematics, config=config)
        report_rows.append({**base, **quality})
        if quality["status"] != "PASS":
            continue
        accepted.append(
            AuditedPair(
                source=pair,
                qpos=qpos,
                audio_frames=audio_frames,
                audio_duration_sec=audio_duration,
                source_motion_sha256=source_motion_sha,
                source_audio_sha256=source_audio_sha,
                quality=quality,
                target_frames=target_frames,
            )
        )
    if len(accepted) != config.expected_accepted:
        counts = Counter(row["status"] for row in report_rows)
        raise ValueError(
            f"质量筛选通过数不一致: expected={config.expected_accepted}, "
            f"actual={len(accepted)}, status={dict(counts)}"
        )
    return accepted, report_rows


def convert_dataset(
    *,
    source_root: Path,
    output_root: Path,
    kinematics_path: Path,
    quality_config_path: Path,
    retarget_config_path: Path,
    reference_root: Path | None = None,
    feature_extractor: Callable[..., tuple[torch.Tensor, dict[str, Any]]] = extract_edge_baseline35,
) -> dict[str, Any]:
    """全有或全无地构建一个正式 ``mine_bumi`` train 数据根。"""

    source_root = source_root.expanduser().resolve(strict=True)
    output_root = output_root.expanduser().resolve()
    kinematics_path = kinematics_path.expanduser().resolve(strict=True)
    quality_config_path = quality_config_path.expanduser().resolve(strict=True)
    retarget_config_path = retarget_config_path.expanduser().resolve(strict=True)
    resolved_reference_root = (
        None if reference_root is None else reference_root.expanduser().resolve(strict=True)
    )
    if output_root.exists():
        raise FileExistsError(f"拒绝覆盖已有输出: {output_root}")
    config = load_quality_config(quality_config_path)
    kinematics = BumiKinematics(kinematics_path)
    if kinematics.kinematics_sha256 != config.expected_kinematics_sha256:
        raise ValueError(
            "质量配置绑定的 kinematics SHA 不匹配: "
            f"expected={config.expected_kinematics_sha256}, actual={kinematics.kinematics_sha256}"
        )
    quality_sha = sha256_file(quality_config_path)
    retarget_sha = sha256_file(retarget_config_path)
    pairs = discover_source_pairs(source_root, config)
    accepted, quality_rows = _audit_pairs(
        pairs,
        kinematics=kinematics,
        config=config,
        quality_config_sha256=quality_sha,
    )
    reference_rows: dict[str, dict[str, Any]] = {}
    if resolved_reference_root is not None:
        reference_manifest = resolved_reference_root / "manifests" / "train.jsonl"
        if not reference_manifest.is_file():
            raise FileNotFoundError(f"参考数据缺少 train manifest: {reference_manifest}")
        for line_number, line in enumerate(
            reference_manifest.read_text(encoding="utf-8").splitlines(), 1
        ):
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            if not sample_id or sample_id in reference_rows:
                raise ValueError(f"{reference_manifest}:{line_number}: sample_id 为空或重复")
            reference_rows[sample_id] = row
        expected_ids = {item.source.sample_id for item in accepted}
        if set(reference_rows) != expected_ids:
            raise ValueError(
                "参考数据 sample_id 与本次 PASS 集不完全一致: "
                f"missing={sorted(expected_ids - set(reference_rows))}, "
                f"extra={sorted(set(reference_rows) - expected_ids)}"
            )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent))
    try:
        quality_report_path = staging / "reports" / "quality_report.jsonl"
        _write_jsonl(quality_report_path, quality_rows)
        quality_report_sha = sha256_file(quality_report_path)
        manifest_rows: list[dict[str, Any]] = []
        sequence_reports: list[dict[str, Any]] = []
        ground_offsets: list[float] = []
        joint_limit_clips: list[float] = []
        total_source_motion_sec = 0.0
        total_source_audio_sec = 0.0
        total_target_frames = 0
        for item in accepted:
            pair = item.source
            sample_id = pair.sample_id
            motion_relative = Path("motions") / f"{sample_id}.pt"
            feature_relative = Path("musicfeat_v2") / f"{sample_id}_musicfeat_fps30.pt"
            audio_relative = Path("audio") / f"{sample_id}.wav"
            audio_output = staging / audio_relative
            if resolved_reference_root is None:
                output_audio_frames = write_cropped_wav(
                    pair.audio_path, audio_output, item.target_frames, config
                )
                music, music_metadata = feature_extractor(audio_output, target_fps=30)
                music = torch.as_tensor(music).detach().cpu().float()
            else:
                reference_row = reference_rows[sample_id]
                if int(reference_row.get("num_frames", -1)) != item.target_frames:
                    raise ValueError(
                        f"{sample_id}: 参考帧数{reference_row.get('num_frames')}与目标"
                        f"{item.target_frames}不一致"
                    )
                original_audio_sha = reference_row.get("original_source_audio_sha256")
                if original_audio_sha not in (None, item.source_audio_sha256):
                    raise ValueError(f"{sample_id}: 参考数据绑定的原始 WAV SHA 不一致")
                reference_audio = resolve_contract_path(
                    resolved_reference_root,
                    reference_row.get("audio_path"),
                    "audio_path",
                    sample_id,
                )
                reference_feature = resolve_contract_path(
                    resolved_reference_root,
                    reference_row.get("music_feature_path"),
                    "music_feature_path",
                    sample_id,
                )
                audio_output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(reference_audio, audio_output)
                output_audio_frames, output_audio_duration = read_wav_metadata(audio_output, config)
                if not math.isclose(
                    output_audio_duration,
                    item.target_frames / 30.0,
                    rel_tol=0.0,
                    abs_tol=1.0 / config.wav_sample_rate,
                ):
                    raise ValueError(f"{sample_id}: 参考 WAV 时长与目标 qpos 不一致")
                music = (
                    torch.as_tensor(
                        torch.load(reference_feature, map_location="cpu", weights_only=False)
                    )
                    .detach()
                    .cpu()
                    .float()
                )
                music_metadata = {
                    "feature_frames": int(len(music)),
                    "feature_source": "verified_reference_dataset",
                    "reference_root": str(resolved_reference_root),
                }
            if music.ndim != 2 or music.shape[1] != 35 or not bool(torch.isfinite(music).all()):
                raise ValueError(f"{sample_id}: EDGE35 必须为 finite [T,35]，实际 {music.shape}")
            if len(music) < item.target_frames:
                raise ValueError(
                    f"{sample_id}: EDGE35 不足: actual={len(music)}, target={item.target_frames}"
                )
            music = music[: item.target_frames].contiguous()
            feature_output = staging / feature_relative
            feature_output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(music, feature_output)
            qpos = resample_qpos_to_30hz(item.qpos, pair.fps, item.target_frames)
            lower = kinematics.joint_lower_limits.detach().cpu()
            upper = kinematics.joint_upper_limits.detach().cpu()
            unclipped_joints = qpos[:, 7:].clone()
            qpos[:, 7:] = torch.maximum(torch.minimum(unclipped_joints, upper), lower)
            joint_limit_clip_max_rad = float((qpos[:, 7:] - unclipped_joints).abs().amax())
            joint_limit_clips.append(joint_limit_clip_max_rad)
            qpos, ground_before, ground_after = normalize_body_origin_ground(qpos, kinematics)
            contact = derive_bumi_foot_contact(
                qpos,
                kinematics,
                valid_mask=torch.ones(len(qpos), dtype=torch.bool),
                fps=30,
                ground_height=None,
                estimate_ground_mask=torch.tensor(True),
            )
            ground_offsets.append(ground_before)
            motion_payload = {
                "contract_version": BUMI_MUSIC_CONTRACT_VERSION,
                "source_motion_contract_version": CSV_SOURCE_CONTRACT_VERSION,
                "qpos": qpos,
                "fps": 30,
                "robot_name": "bumi",
                "joint_names": list(kinematics.joint_order),
                "quaternion_convention": "wxyz",
                "qpos_order": "mujoco_native",
                "source_dataset": "mine",
                "source_sample_id": sample_id,
                "source_motion_sha256": item.source_motion_sha256,
                "source_audio_sha256": item.source_audio_sha256,
                "source_mjcf_sha256": kinematics.source_mjcf_sha256,
                "retarget_config_sha256": retarget_sha,
                "quality_config_sha256": quality_sha,
                "quality_report_sha256": quality_report_sha,
                "quality_accepted": True,
                "retarget_quality": {
                    "status": "PASS",
                    "joint_limit_violation_max": item.quality["metrics"][
                        "joint_limit_violation_max"
                    ],
                    "reason_codes": item.quality["reason_codes"],
                },
                "source_fps": pair.fps,
                "source_num_frames": int(len(item.qpos)),
                "resample_contract_version": CSV_RESAMPLE_CONTRACT_VERSION,
                "root_z_adjustment_m": -ground_before,
                "body_origin_ground_before_adjustment_m": ground_before,
                "body_origin_ground_after_adjustment_m": ground_after,
                "ground_semantics": GROUND_SEMANTICS,
                "root_z_adjusted": True,
                "joint_order_conversion": {
                    "source_joint_names": list(config.source_joint_names),
                    "target_joint_names": list(kinematics.joint_order),
                    "method": "exact_name_reorder",
                },
                "joint_limit_clip_max_rad": joint_limit_clip_max_rad,
                "foot_contact": contact.contact.contiguous(),
                "foot_contact_contract_version": BUMI_CONTACT_CONTRACT_VERSION,
                "foot_contact_source": "derived_from_full_qpos_fk_estimated_legacy_ground",
                "foot_contact_ground_height_m": float(contact.ground_height),
            }
            motion_output = staging / motion_relative
            motion_output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(motion_payload, motion_output)
            feature_sha = sha256_file(feature_output)
            output_audio_sha = sha256_file(audio_output)
            row = {
                "sample_id": sample_id,
                "sequence_id": sample_id,
                "music_group_id": sample_id,
                "audio_key": sample_id,
                "dataset": config.dataset_name,
                "motion_path": motion_relative.as_posix(),
                "music_feature_path": feature_relative.as_posix(),
                "audio_path": audio_relative.as_posix(),
                "fps": 30,
                "num_frames": item.target_frames,
                "split": "train",
                "quality_accepted": True,
                "source_motion_sha256": item.source_motion_sha256,
                "source_music_feature_sha256": feature_sha,
                "source_audio_sha256": output_audio_sha,
                "original_source_audio_sha256": item.source_audio_sha256,
                "source_fps": pair.fps,
                "source_num_frames": int(len(item.qpos)),
                "song_name": pair.song,
                "source_part": pair.part,
            }
            manifest_rows.append(row)
            total_source_motion_sec += len(item.qpos) / pair.fps
            total_source_audio_sec += item.audio_duration_sec
            total_target_frames += item.target_frames
            sequence_reports.append(
                {
                    "sample_id": sample_id,
                    "source_fps": pair.fps,
                    "source_num_frames": int(len(item.qpos)),
                    "source_motion_duration_sec": len(item.qpos) / pair.fps,
                    "source_audio_duration_sec": item.audio_duration_sec,
                    "target_frames": item.target_frames,
                    "target_duration_sec": item.target_frames / 30.0,
                    "output_audio_sample_frames": output_audio_frames,
                    "edge_raw_frames": int(music_metadata.get("feature_frames", len(music))),
                    "edge_output_frames": int(len(music)),
                    "root_z_adjustment_m": -ground_before,
                    "joint_limit_clip_max_rad": joint_limit_clip_max_rad,
                    "foot_contact_ratio_left": float(contact.contact[:, 0].float().mean()),
                    "foot_contact_ratio_right": float(contact.contact[:, 1].float().mean()),
                    "source_motion_sha256": item.source_motion_sha256,
                    "original_source_audio_sha256": item.source_audio_sha256,
                    "output_audio_sha256": output_audio_sha,
                    "music_feature_sha256": feature_sha,
                    "motion_payload_sha256": sha256_file(motion_output),
                }
            )
        manifest_rows.sort(key=lambda row: str(row["sample_id"]))
        sequence_reports.sort(key=lambda row: str(row["sample_id"]))
        _write_jsonl(staging / "manifests" / "train.jsonl", manifest_rows)
        dataset_info = {
            "contract_version": BUMI_MUSIC_CONTRACT_VERSION,
            "robot_name": "bumi",
            "dataset_name": config.dataset_name,
            "source_dataset": "mine",
            "qpos_dim": 28,
            "joint_dim": 21,
            "joint_names": list(kinematics.joint_order),
            "quaternion_convention": "wxyz",
            "qpos_order": "mujoco_native",
            "fps": 30,
            "source_fps": [30, 50],
            "quality_filter_applied": True,
            "quality_joint_limit_violation_max_rad": float(
                config.hard["joint_limit_violation_max"]
            ),
            "reader_joint_limit_tolerance_rad": OUTPUT_JOINT_LIMIT_TOLERANCE_RAD,
            "mjcf_sha256": kinematics.source_mjcf_sha256,
            "source_mjcf_sha256": kinematics.source_mjcf_sha256,
            "feature_kinematics_source_mjcf_sha256": kinematics.source_mjcf_sha256,
            "kinematics_sha256": kinematics.kinematics_sha256,
            "retarget_config_sha256": retarget_sha,
            "quality_config_sha256": quality_sha,
            "quality_report_sha256": quality_report_sha,
            "source_motion_contract_version": CSV_SOURCE_CONTRACT_VERSION,
            "source_joint_names": list(config.source_joint_names),
            "joint_order_conversion_method": "exact_name_reorder",
            "output_joint_limit_policy": "clip_to_current_kinematics_limits",
            "resample_contract_version": CSV_RESAMPLE_CONTRACT_VERSION,
            "ground_semantics": GROUND_SEMANTICS,
            "root_z_adjusted": True,
            "split_counts": {"train": len(manifest_rows), "val": 0, "test": 0},
            "excluded_songs": list(config.excluded_songs),
        }
        _write_json(staging / "meta" / "dataset_info.json", dataset_info)
        conversion_report = {
            "status": "passed",
            "contract_version": BUMI_MUSIC_CONTRACT_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_files_modified": False,
            "source_root": str(source_root),
            "reference_root": (
                None if resolved_reference_root is None else str(resolved_reference_root)
            ),
            "music_audio_reused_from_reference": resolved_reference_root is not None,
            "output_root": str(output_root),
            "dataset_name": config.dataset_name,
            "candidate_sequences": len(pairs),
            "accepted_sequences": len(manifest_rows),
            "excluded_sequences": len(pairs) - len(manifest_rows),
            "excluded_songs": list(config.excluded_songs),
            "source_fps_counts": dict(Counter(str(item.source.fps) for item in accepted)),
            "source_motion_seconds": total_source_motion_sec,
            "source_audio_seconds": total_source_audio_sec,
            "output_frames": total_target_frames,
            "output_seconds": total_target_frames / 30.0,
            "output_hours": total_target_frames / 30.0 / 3600.0,
            "quality_config_sha256": quality_sha,
            "quality_report_sha256": quality_report_sha,
            "retarget_config_sha256": retarget_sha,
            "kinematics_sha256": kinematics.kinematics_sha256,
            "source_mjcf_sha256": kinematics.source_mjcf_sha256,
            "root_z_adjustment_m": {
                "min": min(-value for value in ground_offsets),
                "max": max(-value for value in ground_offsets),
                "mean": sum(-value for value in ground_offsets) / len(ground_offsets),
            },
            "joint_limit_clip_max_rad": max(joint_limit_clips),
            "sequences": sequence_reports,
        }
        _write_json(staging / "reports" / "conversion_report.json", conversion_report)
        reader = BumiMusicDatasetReader(
            staging,
            config.dataset_name,
            "train",
            kinematics,
            strict_alignment=True,
            strict_contract=True,
            require_quality_filter=True,
            joint_limit_tolerance=OUTPUT_JOINT_LIMIT_TOLERANCE_RAD,
            validate_payloads_on_init=True,
            validate_source_hashes_on_init=True,
        )
        if len(reader.rows) != config.expected_accepted:
            raise RuntimeError("正式 reader 验证后的序列数不匹配")
        os.replace(staging, output_root)
        return conversion_report
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--kinematics", required=True, type=Path)
    parser.add_argument(
        "--quality-config",
        type=Path,
        default=(REPO_ROOT / "configs/bumi/quality_filter_csv_mine_robot_retargeter_fe934_v2.yaml"),
    )
    parser.add_argument("--retarget-config", required=True, type=Path)
    parser.add_argument(
        "--reference-root",
        type=Path,
        help=(
            "可选的同一99条旧正式数据根；只复用经SHA和帧数验证的WAV/EDGE35，"
            "qpos、关节顺序、落地和接触仍全部重新生成"
        ),
    )
    args = parser.parse_args()
    report = convert_dataset(
        source_root=args.source_root,
        output_root=args.output_root,
        kinematics_path=args.kinematics,
        quality_config_path=args.quality_config,
        retarget_config_path=args.retarget_config,
        reference_root=args.reference_root,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
