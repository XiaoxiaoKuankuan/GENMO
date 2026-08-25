#!/usr/bin/env python3
"""把手工 q1 + 自动 PASS 的 50 Hz SONIC NPZ 发布为 GENMO BUMI 音乐数据集。

本工具处理 ``sonic_isaaclab_bumi3_motion_v1`` 七字段 NPZ，而不是旧版 30 Hz
pickle。它以质量报告中的 PASS 为唯一准入列表，逐条复核源 NPZ SHA 和 50 Hz
契约；用权威人体数据的原始 train/val/test、EDGE35 长度和 WAV 配对确定目标序列，
再把根位置、wxyz 根四元数和 21 关节从 50 Hz 重采样到严格等长的 30 Hz qpos28。

源关节是 GMR preset 的 Isaac-Lab publish order，输出关节则按当前 GENMO 固定
kinematics 的 MuJoCo-native order 通过完整关节名重排。根四元数先消除 q/-q 跳变，
再做最短弧 SLERP；位置和关节做线性插值。实际 3,163 条源数据证明 30→50 Hz 离线
导出使用不含右端点的时间网格，即 ``ceil((T30-1)*50/30)`` 帧；因此 30 Hz 最后一帧
使用 50 Hz 末帧保持，不会外推未知姿态。输出前使用 GENMO kinematics 做 FK，并对
每条轨迹施加一个常量
root-Z 偏移，使全部 body origin 的全局最小 Z 精确为 0，从而与现有
``legacy_body_origin_min_zero`` 历史地面规范兼容；qpos30/contact 训练会按该语义用 GT
FK 足底低分位估计等效地面，该平移会被明确记录
为 ``root_z_adjusted=true``，不会伪装成未经调整的源轨迹。

构建过程只读源动作、人体数据、EDGE35 和 WAV，在同盘优先硬链接音乐资产；motion、
manifest、dataset_info 和报告全部先进入 staging，完整成功后才原子发布。工具不会
覆盖既有数据版本，也不会把 REVIEW/REJECT、缺失音频或长度不一致样本静默混入。
qpos30 不作为另一份轨迹缓存写入：训练仍以 qpos28 为权威状态，由固定 kinematics 和
codec 在线确定性编码为 30D，link 全部通过 FK 得到；随后由独立工具只基于 train split
重算 qpos30 统计量。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gem.datasets.music_dance.music_dance_bumi import (  # noqa: E402
    BUMI_MUSIC_CONTRACT_VERSION,
)
from gem.robots.bumi.kinematics import BumiKinematics  # noqa: E402
from gem.robots.bumi.legacy_motion import sha256_file  # noqa: E402
from tools.data.bumi.build_bumi_music_dataset import (  # noqa: E402
    DATASET_SPECS,
    _check_digest,
    _mapping,
    _materialize,
    _music_tensor,
    _parse_mapping,
    _relative_file,
    _write_json,
    _write_jsonl,
    load_human_indices,
    pairing_fields,
)
from tools.data.bumi.filter_sonic_npz_motions import (  # noqa: E402
    REPORT_VERSION,
    load_config,
    load_motion_npz,
    verify_assets,
)

SOURCE_CONTRACT_VERSION = "sonic_isaaclab_bumi3_motion_v1"
RESAMPLE_CONTRACT_VERSION = "genmo.bumi_50hz_to_30hz.v1"
GROUND_SEMANTICS = "legacy_body_origin_min_zero"
_AIST_ARMFIX_SUFFIX = "_armfix"


def _read_quality_report(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 并只返回完整、唯一且已接受的 PASS 记录。"""

    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: 必须是 JSON object")
            if row.get("report_contract_version") != REPORT_VERSION:
                raise ValueError(f"{path}:{line_number}: 质量报告契约不匹配")
            sample_id = str(row.get("sample_id", ""))
            if not sample_id or sample_id in seen:
                raise ValueError(f"{path}:{line_number}: sample_id 为空或重复: {sample_id!r}")
            seen.add(sample_id)
            if row.get("status") == "PASS":
                if row.get("quality_accepted") is not True:
                    raise ValueError(f"{sample_id}: PASS 但 quality_accepted 不为 true")
                rows.append(row)
    if not rows:
        raise ValueError(f"{path} 中没有 PASS")
    return rows


def canonical_human_sample_id(dataset: str, source_sample_id: str) -> str:
    """把已知人工修复 variant 映射回权威人体/音乐 sample basename。"""

    if dataset == "aistpp" and source_sample_id.endswith(_AIST_ARMFIX_SUFFIX):
        result = source_sample_id[: -len(_AIST_ARMFIX_SUFFIX)]
        if not result:
            raise ValueError("AIST++ _armfix 名称缺少基础 sample_id")
        return result
    return source_sample_id


def deduplicate_quality_variants(
    rows: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """按权威 sample 去重，并只允许已知 AIST ``_armfix`` 显式替代基础版。"""

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        dataset, source_sample_id = _split_report_sample_id(str(row["sample_id"]))
        canonical = canonical_human_sample_id(dataset, source_sample_id)
        grouped.setdefault((dataset, canonical), []).append(row)
    selected: list[dict[str, Any]] = []
    superseded: list[dict[str, str]] = []
    for (dataset, canonical), variants in sorted(grouped.items()):
        if len(variants) == 1:
            selected.append(variants[0])
            continue
        source_ids = {_split_report_sample_id(str(row["sample_id"]))[1]: row for row in variants}
        expected_ids = {canonical, f"{canonical}{_AIST_ARMFIX_SUFFIX}"}
        if dataset != "aistpp" or set(source_ids) != expected_ids:
            raise ValueError(
                f"不支持的规范 sample_id 重复: {dataset}/{canonical}, variants={sorted(source_ids)}"
            )
        preferred = f"{canonical}{_AIST_ARMFIX_SUFFIX}"
        selected.append(source_ids[preferred])
        superseded.append(
            {
                "dataset": dataset,
                "canonical_sample_id": canonical,
                "selected_variant_id": preferred,
                "superseded_variant_id": canonical,
            }
        )
    return selected, superseded


def _split_report_sample_id(value: str) -> tuple[str, str]:
    if "/" not in value:
        raise ValueError(f"质量 sample_id 必须为 DATASET/basename: {value!r}")
    dataset, sample_id = value.split("/", 1)
    if dataset not in DATASET_SPECS:
        raise ValueError(f"未知数据集: {dataset!r}")
    if not sample_id or Path(sample_id).name != sample_id:
        raise ValueError(f"质量 sample basename 非法: {sample_id!r}")
    return dataset, sample_id


def expected_50hz_frames(target_30hz_frames: int) -> int:
    """按持续时间保持规则计算 30→50 Hz 离线序列帧数。"""

    frames = int(target_30hz_frames)
    if frames <= 0:
        raise ValueError("target_30hz_frames 必须为正数")
    return max(1, math.ceil((frames - 1) * 50 / 30))


def make_quaternion_continuous_np(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """归一化 wxyz 四元数并消除逐帧 q/-q 符号跳变。"""

    value = np.asarray(quaternion_wxyz, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 4 or value.shape[0] <= 0:
        raise ValueError(f"quaternion 必须为 [T,4]，实际 {value.shape}")
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    if not np.isfinite(value).all() or np.any(norm < 1.0e-8):
        raise ValueError("quaternion 包含非有限值或零范数")
    result = value / norm
    for index in range(1, len(result)):
        if float(np.dot(result[index - 1], result[index])) < 0.0:
            result[index] *= -1.0
    return result


def _slerp_pairs(q0: np.ndarray, q1: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """对同形状 wxyz 四元数对执行向量化最短弧 SLERP。"""

    dot = np.sum(q0 * q1, axis=-1)
    q1_short = q1.copy()
    negative = dot < 0.0
    q1_short[negative] *= -1.0
    dot = np.abs(dot).clip(0.0, 1.0)
    angle = np.arccos(dot)
    sin_angle = np.sin(angle)
    use_linear = sin_angle < 1.0e-7
    a = np.asarray(alpha, dtype=np.float64)
    weight0 = np.empty_like(a)
    weight1 = np.empty_like(a)
    weight0[use_linear] = 1.0 - a[use_linear]
    weight1[use_linear] = a[use_linear]
    stable = ~use_linear
    weight0[stable] = np.sin((1.0 - a[stable]) * angle[stable]) / sin_angle[stable]
    weight1[stable] = np.sin(a[stable] * angle[stable]) / sin_angle[stable]
    result = weight0[:, None] * q0 + weight1[:, None] * q1_short
    return result / np.linalg.norm(result, axis=-1, keepdims=True)


def resample_sonic_qpos_to_30hz(
    arrays: Mapping[str, np.ndarray],
    *,
    source_joint_order: Iterable[str],
    target_joint_order: Iterable[str],
    target_frames: int,
) -> torch.Tensor:
    """从七字段 50 Hz 数据构造严格 ``[target_frames,28]`` 的 30 Hz qpos。"""

    source_names = tuple(map(str, source_joint_order))
    target_names = tuple(map(str, target_joint_order))
    if len(source_names) != 21 or len(target_names) != 21:
        raise ValueError("源/目标关节顺序都必须为 21 个名称")
    if set(source_names) != set(target_names):
        raise ValueError("源/目标关节名称集合不一致")
    source_frames = int(arrays["joint_pos"].shape[0])
    expected = expected_50hz_frames(target_frames)
    if source_frames != expected:
        raise ValueError(
            f"50/30 Hz 持续时间帧数不一致: expected_50hz={expected}, "
            f"actual_50hz={source_frames}, target_30hz={target_frames}"
        )
    root_pos = np.asarray(arrays["body_pos_w"][:, 0, :], dtype=np.float64)
    root_quat = make_quaternion_continuous_np(arrays["body_quat_w"][:, 0, :])
    source_joint = np.asarray(arrays["joint_pos"], dtype=np.float64)
    reorder = [source_names.index(name) for name in target_names]
    values = np.concatenate((root_pos, source_joint[:, reorder]), axis=-1)

    target_time = np.arange(int(target_frames), dtype=np.float64) / 30.0
    source_position = np.minimum(target_time * 50.0, source_frames - 1)
    lower = np.floor(source_position).astype(np.int64)
    upper = np.minimum(lower + 1, source_frames - 1)
    alpha = source_position - lower
    linear = values[lower] * (1.0 - alpha[:, None]) + values[upper] * alpha[:, None]
    quaternion = _slerp_pairs(root_quat[lower], root_quat[upper], alpha)
    qpos = np.concatenate((linear[:, :3], quaternion, linear[:, 3:]), axis=-1)
    if qpos.shape != (int(target_frames), 28) or not np.isfinite(qpos).all():
        raise RuntimeError(f"30 Hz qpos 构造失败: {qpos.shape}")
    return torch.from_numpy(qpos.astype(np.float32, copy=False)).contiguous()


def normalize_body_origin_ground(
    qpos: torch.Tensor, kinematics: BumiKinematics
) -> tuple[torch.Tensor, float, float]:
    """施加常量 root-Z 偏移，使 GENMO FK 的 body-origin 全局最小 Z 为零。"""

    value = qpos.detach().cpu().float().clone()
    with torch.no_grad():
        before = float(kinematics.forward_kinematics(value)["body_pos_w"][..., 2].amin().item())
        value[:, 2] -= before
        after = float(kinematics.forward_kinematics(value)["body_pos_w"][..., 2].amin().item())
    if abs(after) > 2.0e-5:
        raise RuntimeError(f"root-Z 归一化后 body-origin ground={after:.8g}，不接近 0")
    return value.contiguous(), before, after


def convert_datasets(
    *,
    source_root: Path,
    quality_report: Path,
    quality_config: Path,
    source_robot_xml: Path,
    source_preset: Path,
    source_kinematics: Path,
    retarget_config: Path,
    target_kinematics: Path,
    human_roots: Mapping[str, Path],
    audio_roots: Mapping[str, Path],
    output_root: Path,
    expected_total: int | None = 2986,
) -> dict[str, Any]:
    """完整 staged 构建四数据集，并在全部成功后原子发布。"""

    source_root = source_root.expanduser().resolve(strict=True)
    quality_report = quality_report.expanduser().resolve(strict=True)
    quality_config = quality_config.expanduser().resolve(strict=True)
    retarget_config = retarget_config.expanduser().resolve(strict=True)
    target_kinematics = target_kinematics.expanduser().resolve(strict=True)
    output_root = output_root.expanduser().resolve()
    source_config = load_config(quality_config)
    verified_assets = verify_assets(
        source_config,
        preset_path=source_preset,
        robot_xml_path=source_robot_xml,
        kinematics_path=source_kinematics,
    )
    quality_sha = sha256_file(quality_config)
    report_sha = sha256_file(quality_report)
    retarget_sha = sha256_file(retarget_config)
    target_kin = BumiKinematics(target_kinematics)
    quality_pass_rows = _read_quality_report(quality_report)
    if expected_total is not None and len(quality_pass_rows) != int(expected_total):
        raise ValueError(
            f"PASS 记录数不一致: expected={expected_total}, actual={len(quality_pass_rows)}"
        )
    selected, superseded_variants = deduplicate_quality_variants(quality_pass_rows)
    human_indices = load_human_indices(human_roots)

    if output_root.exists():
        raise FileExistsError(f"拒绝覆盖已有数据版本: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent))
    split_rows: dict[str, dict[str, list[dict[str, Any]]]] = {
        dataset: {split: [] for split in ("train", "val", "test")} for dataset in DATASET_SPECS
    }
    seen_canonical: set[tuple[str, str]] = set()
    counters: Counter[str] = Counter()
    materialization: Counter[str] = Counter()
    hash_cache: dict[Path, str] = {}
    materialized: dict[Path, str] = {}
    music_frames: dict[Path, int] = {}
    ground_offsets: dict[str, list[float]] = {name: [] for name in DATASET_SPECS}

    def digest(path: Path) -> str:
        if path not in hash_cache:
            hash_cache[path] = sha256_file(path)
        return hash_cache[path]

    def materialize_once(source: Path, destination: Path, source_sha: str) -> str:
        previous = materialized.get(destination)
        if previous is not None:
            if previous != source_sha:
                raise ValueError(f"目标文件碰撞且内容不同: {destination}")
            return "existing"
        mode = _materialize(source, destination)
        materialized[destination] = source_sha
        return mode

    try:
        for quality in selected:
            dataset, source_sample_id = _split_report_sample_id(str(quality["sample_id"]))
            sample_id = canonical_human_sample_id(dataset, source_sample_id)
            canonical_key = (dataset, sample_id)
            if canonical_key in seen_canonical:
                raise ValueError(f"规范 sample_id 碰撞: {dataset}/{sample_id}")
            seen_canonical.add(canonical_key)
            if _check_digest(quality.get("quality_config_sha256"), "quality config") != quality_sha:
                raise ValueError(f"{dataset}/{source_sample_id}: quality config SHA 不匹配")
            if quality.get("source_motion_contract_version") != SOURCE_CONTRACT_VERSION:
                raise ValueError(f"{dataset}/{source_sample_id}: 源动作契约不匹配")
            source_motion = _relative_file(
                source_root,
                quality.get("source_relative_path"),
                f"{dataset}/{source_sample_id} source_relative_path",
            )
            source_motion_sha = digest(source_motion)
            if source_motion_sha != _check_digest(quality.get("source_sha256"), "source motion"):
                raise ValueError(f"{dataset}/{source_sample_id}: 源 NPZ SHA 不匹配")
            arrays = load_motion_npz(source_motion, source_config)

            human = human_indices[dataset].get(sample_id)
            if human is None:
                raise ValueError(f"{dataset}/{source_sample_id}: 无权威人体 sample={sample_id}")
            if float(human.get("fps", 30)) != 30.0:
                raise ValueError(f"{dataset}/{sample_id}: 人体数据不是 30 FPS")
            split = str(human.get("split", ""))
            if split not in {"train", "val", "test"}:
                raise ValueError(f"{dataset}/{sample_id}: split 非法: {split!r}")
            target_frames = int(human.get("num_frames", -1))
            pair = pairing_fields(dataset, sample_id, human)
            feature_source = _relative_file(
                human_roots[dataset],
                human.get("music_feature_path"),
                f"{dataset}/{sample_id} music_feature_path",
            )
            if feature_source not in music_frames:
                music_frames[feature_source] = int(
                    _music_tensor(feature_source, sample_id).shape[0]
                )
            if music_frames[feature_source] != target_frames:
                raise ValueError(
                    f"{dataset}/{sample_id}: EDGE35/human 帧数不一致: "
                    f"{music_frames[feature_source]} != {target_frames}"
                )
            audio_source = audio_roots[dataset] / f"{pair['audio_key']}.wav"
            if not audio_source.is_file():
                raise FileNotFoundError(f"{dataset}/{sample_id}: WAV 缺失: {audio_source}")

            qpos = resample_sonic_qpos_to_30hz(
                arrays,
                source_joint_order=source_config.joint_order,
                target_joint_order=target_kin.joint_order,
                target_frames=target_frames,
            )
            qpos, ground_before, ground_after = normalize_body_origin_ground(qpos, target_kin)
            ground_offsets[dataset].append(ground_before)
            dataset_root = staging / DATASET_SPECS[dataset]["output"]
            motion_relative = Path("motions") / f"{sample_id}.pt"
            feature_relative = Path("musicfeat_v2") / feature_source.name
            audio_relative = Path("audio") / f"{pair['audio_key']}.wav"
            feature_sha = digest(feature_source)
            audio_sha = digest(audio_source)
            motion_payload = {
                "contract_version": BUMI_MUSIC_CONTRACT_VERSION,
                "source_motion_contract_version": SOURCE_CONTRACT_VERSION,
                "qpos": qpos,
                "fps": 30,
                "robot_name": "bumi",
                "joint_names": list(target_kin.joint_order),
                "quaternion_convention": "wxyz",
                "qpos_order": "mujoco_native",
                "source_dataset": dataset,
                "source_sample_id": sample_id,
                "source_variant_id": source_sample_id,
                "source_motion_sha256": source_motion_sha,
                "source_mjcf_sha256": source_config.robot_xml_sha256,
                "retarget_config_sha256": retarget_sha,
                "quality_config_sha256": quality_sha,
                "quality_report_sha256": report_sha,
                "quality_accepted": True,
                "retarget_quality": {
                    "status": "PASS",
                    "joint_limit_violation_max": quality["metrics"]["joint_limit_violation_max"],
                    "reason_codes": list(quality.get("reason_codes", ())),
                },
                "source_fps": 50,
                "resample_contract_version": RESAMPLE_CONTRACT_VERSION,
                "source_num_frames": int(arrays["joint_pos"].shape[0]),
                "root_z_adjustment_m": -ground_before,
                "body_origin_ground_before_adjustment_m": ground_before,
                "body_origin_ground_after_adjustment_m": ground_after,
                "root_z_adjusted": True,
                "ground_semantics": GROUND_SEMANTICS,
            }
            motion_destination = dataset_root / motion_relative
            motion_destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(motion_payload, motion_destination)
            materialization[
                materialize_once(feature_source, dataset_root / feature_relative, feature_sha)
            ] += 1
            materialization[
                materialize_once(audio_source, dataset_root / audio_relative, audio_sha)
            ] += 1
            row = {
                "sample_id": sample_id,
                "sequence_id": pair["sequence_id"],
                "music_group_id": pair["music_group_id"],
                "audio_key": pair["audio_key"],
                "dataset": DATASET_SPECS[dataset]["contract_name"],
                "motion_path": motion_relative.as_posix(),
                "music_feature_path": feature_relative.as_posix(),
                "audio_path": audio_relative.as_posix(),
                "fps": 30,
                "num_frames": target_frames,
                "split": split,
                "quality_accepted": True,
                "source_variant_id": source_sample_id,
                "source_motion_sha256": source_motion_sha,
                "source_music_feature_sha256": feature_sha,
                "source_audio_sha256": audio_sha,
            }
            for field in (
                "person_id",
                "dance_style",
                "music_genre",
                "coarse_style",
                "fine_style",
                "song_name",
                "pair_id",
                "role",
                "song_id",
                "take_id",
                "group_id",
            ):
                if field in human:
                    row[field] = human[field]
            split_rows[dataset][split].append(row)
            counters[f"{dataset}:{split}"] += 1
            counters[f"{dataset}:total"] += 1

        actual_splits: Counter[str] = Counter()
        for dataset, by_split in split_rows.items():
            dataset_root = staging / DATASET_SPECS[dataset]["output"]
            for split, rows in by_split.items():
                rows.sort(key=lambda item: str(item["sample_id"]))
                _write_jsonl(dataset_root / "manifests" / f"{split}.jsonl", rows)
                actual_splits[split] += len(rows)
            info = {
                "contract_version": BUMI_MUSIC_CONTRACT_VERSION,
                "robot_name": "bumi",
                "dataset_name": DATASET_SPECS[dataset]["contract_name"],
                "source_dataset": dataset,
                "qpos_dim": 28,
                "joint_dim": 21,
                "joint_names": list(target_kin.joint_order),
                "quaternion_convention": "wxyz",
                "qpos_order": "mujoco_native",
                "fps": 30,
                "source_fps": 50,
                "quality_filter_applied": True,
                "quality_joint_limit_violation_max_rad": source_config.joint_limit_violation_max,
                "reader_joint_limit_tolerance_rad": source_config.joint_limit_violation_max,
                "mjcf_sha256": source_config.robot_xml_sha256,
                "source_mjcf_sha256": source_config.robot_xml_sha256,
                "feature_kinematics_source_mjcf_sha256": target_kin.source_mjcf_sha256,
                "kinematics_sha256": target_kin.kinematics_sha256,
                "retarget_config_sha256": retarget_sha,
                "quality_config_sha256": quality_sha,
                "quality_report_sha256": report_sha,
                "source_preset_sha256": source_config.preset_sha256,
                "source_kinematics_sha256": source_config.kinematics_sha256,
                "resample_contract_version": RESAMPLE_CONTRACT_VERSION,
                "ground_semantics": GROUND_SEMANTICS,
                "root_z_adjusted": True,
                "split_counts": {name: len(rows) for name, rows in by_split.items()},
            }
            _write_json(dataset_root / "meta" / "dataset_info.json", info)

        report = {
            "status": "passed",
            "contract_version": BUMI_MUSIC_CONTRACT_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_files_modified": False,
            "quality_report_sha256": report_sha,
            "quality_config_sha256": quality_sha,
            "source_mjcf_sha256": source_config.robot_xml_sha256,
            "retarget_config_sha256": retarget_sha,
            "kinematics_sha256": target_kin.kinematics_sha256,
            "feature_kinematics_source_mjcf_sha256": target_kin.source_mjcf_sha256,
            "verified_source_assets": verified_assets,
            "source_fps": 50,
            "fps": 30,
            "resample_contract_version": RESAMPLE_CONTRACT_VERSION,
            "ground_semantics": GROUND_SEMANTICS,
            "root_z_adjusted": True,
            "quality_pass_records": len(quality_pass_rows),
            "superseded_quality_variants": superseded_variants,
            "total_sequences": len(seen_canonical),
            "split_counts": dict(sorted(actual_splits.items())),
            "dataset_counts": {
                dataset: {
                    "total": counters[f"{dataset}:total"],
                    **{split: counters[f"{dataset}:{split}"] for split in ("train", "val", "test")},
                }
                for dataset in DATASET_SPECS
            },
            "unique_music_features": sum(
                len({row["music_feature_path"] for rows in values.values() for row in rows})
                for values in split_rows.values()
            ),
            "materialization": dict(materialization),
            "root_ground_offset_m": {
                dataset: {
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                }
                for dataset, values in ground_offsets.items()
            },
        }
        for dataset in DATASET_SPECS:
            _write_json(
                staging / DATASET_SPECS[dataset]["output"] / "reports" / "conversion_report.json",
                {**report, "dataset": dataset, "dataset_counts": report["dataset_counts"][dataset]},
            )
        _write_json(staging / "conversion_report.json", report)
        os.replace(staging, output_root)
        return report
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--quality-report", required=True, type=Path)
    parser.add_argument("--quality-config", required=True, type=Path)
    parser.add_argument("--source-robot-xml", required=True, type=Path)
    parser.add_argument("--source-preset", required=True, type=Path)
    parser.add_argument("--source-kinematics", required=True, type=Path)
    parser.add_argument("--retarget-config", required=True, type=Path)
    parser.add_argument("--target-kinematics", required=True, type=Path)
    parser.add_argument("--human-root", action="append", required=True, type=_parse_mapping)
    parser.add_argument("--audio-root", action="append", required=True, type=_parse_mapping)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--expected-total", type=int, default=2986)
    args = parser.parse_args()
    report = convert_datasets(
        source_root=args.source_root,
        quality_report=args.quality_report,
        quality_config=args.quality_config,
        source_robot_xml=args.source_robot_xml,
        source_preset=args.source_preset,
        source_kinematics=args.source_kinematics,
        retarget_config=args.retarget_config,
        target_kinematics=args.target_kinematics,
        human_roots=_mapping(args.human_root, "--human-root"),
        audio_roots=_mapping(args.audio_root, "--audio-root"),
        output_root=args.output_root,
        expected_total=args.expected_total,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
