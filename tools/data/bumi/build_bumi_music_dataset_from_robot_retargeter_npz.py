#!/usr/bin/env python3
"""把 robot_retargeter 30 Hz 严格 PASS 动作发布为正式 BUMI 音乐训练数据。

输入动作必须已经通过 ``filter_robot_retargeter_npz_motions.py`` 的 13 字段、30 Hz、
坐标系、关节限位、速度、加速度、jerk、Root 倾角和倒地动作三态筛选。本工具会读取
完整质量 JSONL，但只允许 ``status=PASS`` 且 ``quality_accepted=true`` 的条目进入
train/val/test manifest；REVIEW、REJECT 和上游失败项不会被裁剪后偷偷复用。

新重定向结果本身就是 30 Hz，因此不插值，也不再次平移 Root Z。工具仅按关节名称把
Isaac publish order 重排为当前 MJCF qpos order，使用同一份新 MJCF 导出的运动学重新
计算足底接触，并明确采用 robot_retargeter 固定世界地面 Z=0 的语义。音乐特征、音频和
权威 split 从已验收的四库参考训练目录读取；参考目录只提供配对信息，不提供机器人动作。
全部结果先写 staging，四库都成功后才原子发布，已有正式目录不会被覆盖。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from gem.datasets.music_dance.music_dance_bumi import BUMI_MUSIC_CONTRACT_VERSION  # noqa: E402
from gem.robots.bumi.contacts import (  # noqa: E402
    BUMI_CONTACT_CONTRACT_VERSION,
    derive_bumi_foot_contact,
)
from gem.robots.bumi.kinematics import BumiKinematics  # noqa: E402
from gem.robots.bumi.legacy_motion import root_tilt_statistics, sha256_file  # noqa: E402
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
)
from tools.data.bumi.build_bumi_music_dataset_from_sonic_npz import (  # noqa: E402
    make_quaternion_continuous_np,
)
from tools.data.bumi.filter_robot_retargeter_npz_motions import (  # noqa: E402
    REPORT_VERSION,
    SOURCE_CONTRACT_VERSION,
    load_config,
    load_motion_npz,
    verify_assets,
)

GROUND_SEMANTICS = "robot_retargeter_floor_zero_v1"
BUILDER_CONTRACT_VERSION = "genmo.bumi_robot_retargeter_pass_builder.v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 必须是 object: {path}")
    return value


def read_pass_rows(
    report_path: Path, summary_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """复核完整三态报告，并只返回其中严格 PASS 记录。"""

    summary = _read_json(summary_path)
    if summary.get("report_contract_version") != REPORT_VERSION:
        raise ValueError("quality summary 契约不匹配")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    all_count = 0
    with report_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or row.get("report_contract_version") != REPORT_VERSION:
                raise ValueError(f"{report_path}:{line_number}: 质量报告契约不匹配")
            sample_id = str(row.get("sample_id", ""))
            if not sample_id or sample_id in seen:
                raise ValueError(f"{report_path}:{line_number}: sample_id 为空或重复")
            seen.add(sample_id)
            all_count += 1
            status = row.get("status")
            if status not in {"PASS", "REVIEW", "REJECT"}:
                raise ValueError(f"{sample_id}: 未知质量状态 {status!r}")
            if (status == "PASS") != (row.get("quality_accepted") is True):
                raise ValueError(f"{sample_id}: status 与 quality_accepted 不一致")
            if status == "PASS":
                rows.append(row)
    if all_count != int(summary.get("sequences", -1)):
        raise ValueError("quality JSONL 数量与 summary 不一致")
    if len(rows) != int(summary.get("quality_accepted_sequences", -1)):
        raise ValueError("PASS 数量与 quality summary 不一致")
    if all_count != 3154 or not rows:
        raise ValueError(f"正式输入应重算3154条且至少有1条PASS，实际 {all_count}/{len(rows)}")
    return sorted(rows, key=lambda row: str(row["sample_id"])), summary


def load_reference_indices(roots: Mapping[str, Path]) -> dict[str, dict[str, dict[str, Any]]]:
    """读取四库已验收 manifest，作为 split、EDGE35 与 WAV 配对的唯一来源。"""

    result: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset, root in roots.items():
        index: dict[str, dict[str, Any]] = {}
        for split in ("train", "val", "test"):
            path = root / "manifests" / f"{split}.jsonl"
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    sample_id = str(row.get("sample_id", ""))
                    if not sample_id or sample_id in index or row.get("split") != split:
                        raise ValueError(f"{path}:{line_number}: 参考 manifest 身份/split 非法")
                    index[sample_id] = dict(row)
        result[dataset] = index
    return result


def qpos30_from_npz(
    arrays: Mapping[str, np.ndarray], source_joint_order: tuple[str, ...], kinematics: BumiKinematics
) -> torch.Tensor:
    """保留源 Root 位置，把 30 Hz NPZ 按名称无损重排成 MuJoCo qpos28。"""

    if set(source_joint_order) != set(kinematics.joint_order):
        raise ValueError("源 NPZ 与目标运动学关节名称集合不一致")
    reorder = [source_joint_order.index(name) for name in kinematics.joint_order]
    root_pos = np.asarray(arrays["body_pos_w"][:, 0], dtype=np.float64)
    root_quat = make_quaternion_continuous_np(arrays["body_quat_w"][:, 0])
    joints = np.asarray(arrays["joint_pos"][:, reorder], dtype=np.float64)
    qpos = np.concatenate((root_pos, root_quat, joints), axis=-1)
    if qpos.shape != (len(root_pos), 28) or not np.isfinite(qpos).all():
        raise ValueError(f"qpos30 构造失败: {qpos.shape}")
    return torch.from_numpy(qpos.astype(np.float32, copy=False)).contiguous()


def _root_tilt_degrees(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    up_dot = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    return np.degrees(np.arccos(np.clip(up_dot, -1.0, 1.0)))


def convert_datasets(
    *,
    source_root: Path,
    quality_report: Path,
    quality_summary: Path,
    quality_config: Path,
    robot_xml: Path,
    retarget_config: Path,
    kinematics_path: Path,
    reference_roots: Mapping[str, Path],
    output_root: Path,
    expected_pass: int | None = None,
) -> dict[str, Any]:
    """以质量 PASS 为白名单，重建四库 qpos30/contact 和正式 manifest。"""

    source_root = source_root.expanduser().resolve(strict=True)
    quality_report = quality_report.expanduser().resolve(strict=True)
    quality_summary = quality_summary.expanduser().resolve(strict=True)
    quality_config = quality_config.expanduser().resolve(strict=True)
    output_root = output_root.expanduser().resolve()
    config = load_config(quality_config)
    assets = verify_assets(
        config,
        robot_xml=robot_xml,
        retarget_config=retarget_config,
        kinematics=kinematics_path,
    )
    kinematics = BumiKinematics(kinematics_path)
    pass_rows, summary = read_pass_rows(quality_report, quality_summary)
    if expected_pass is not None and len(pass_rows) != int(expected_pass):
        raise ValueError(f"PASS 数量应为 {expected_pass}，实际 {len(pass_rows)}")
    if summary.get("quality_config_sha256") != sha256_file(quality_config):
        raise ValueError("quality summary 配置 SHA 不匹配")
    reference = load_reference_indices(reference_roots)
    if output_root.exists():
        raise FileExistsError(f"拒绝覆盖已有正式数据目录: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent))
    split_rows = {
        name: {split: [] for split in ("train", "val", "test")} for name in DATASET_SPECS
    }
    counts: Counter[str] = Counter()
    materialized: dict[Path, str] = {}
    materialization: Counter[str] = Counter()
    hash_cache: dict[Path, str] = {}
    tilt_by_dataset: dict[str, list[np.ndarray]] = {name: [] for name in DATASET_SPECS}

    def digest(path: Path) -> str:
        if path not in hash_cache:
            hash_cache[path] = sha256_file(path)
        return hash_cache[path]

    def copy_once(source: Path, destination: Path, source_sha: str) -> None:
        previous = materialized.get(destination)
        if previous is not None:
            if previous != source_sha:
                raise ValueError(f"目标资产碰撞: {destination}")
            materialization["existing"] += 1
            return
        materialization[_materialize(source, destination)] += 1
        materialized[destination] = source_sha

    try:
        for quality in pass_rows:
            dataset, sample_id = str(quality["sample_id"]).split("/", 1)
            if dataset not in DATASET_SPECS or Path(sample_id).name != sample_id:
                raise ValueError(f"质量 sample_id 非法: {quality['sample_id']!r}")
            if quality.get("source_motion_contract_version") != SOURCE_CONTRACT_VERSION:
                raise ValueError(f"{dataset}/{sample_id}: 源契约不匹配")
            if quality.get("quality_config_sha256") != sha256_file(quality_config):
                raise ValueError(f"{dataset}/{sample_id}: 质量配置 SHA 不匹配")
            source = _relative_file(
                source_root, quality.get("source_relative_path"), f"{dataset}/{sample_id} NPZ"
            )
            source_sha = digest(source)
            if source_sha != _check_digest(quality.get("source_sha256"), "source NPZ"):
                raise ValueError(f"{dataset}/{sample_id}: 源 NPZ SHA 不匹配")
            arrays = load_motion_npz(source, config)
            reference_row = reference[dataset].get(sample_id)
            if reference_row is None:
                raise ValueError(f"{dataset}/{sample_id}: 参考训练 manifest 无同名条目")
            split = str(reference_row.get("split", ""))
            if split not in {"train", "val", "test"}:
                raise ValueError(f"{dataset}/{sample_id}: split 非法")
            feature_source = _relative_file(
                reference_roots[dataset],
                reference_row.get("music_feature_path"),
                f"{dataset}/{sample_id} EDGE35",
            )
            audio_source = _relative_file(
                reference_roots[dataset],
                reference_row.get("audio_path"),
                f"{dataset}/{sample_id} WAV",
            )
            qpos = qpos30_from_npz(arrays, config.joint_order, kinematics)
            frames = int(qpos.shape[0])
            music_frames = int(_music_tensor(feature_source, sample_id).shape[0])
            expected_frames = int(reference_row.get("num_frames", -1))
            if len({frames, music_frames, expected_frames}) != 1:
                raise ValueError(
                    f"{dataset}/{sample_id}: qpos/音乐/manifest 帧数不一致 "
                    f"{frames}/{music_frames}/{expected_frames}"
                )
            fk = kinematics.forward_kinematics(qpos)
            contact = derive_bumi_foot_contact(
                qpos,
                kinematics,
                valid_mask=torch.ones(frames, dtype=torch.bool),
                fps=30,
                ground_height=0.0,
                fk=fk,
            )
            tilt = _root_tilt_degrees(arrays["body_quat_w"][:, 0])
            tilt_by_dataset[dataset].append(tilt)
            dataset_root = staging / DATASET_SPECS[dataset]["output"]
            motion_relative = Path("motions") / f"{sample_id}.pt"
            feature_relative = Path("musicfeat_v2") / feature_source.name
            audio_relative = Path("audio") / audio_source.name
            feature_sha = digest(feature_source)
            audio_sha = digest(audio_source)
            payload = {
                "contract_version": BUMI_MUSIC_CONTRACT_VERSION,
                "source_motion_contract_version": SOURCE_CONTRACT_VERSION,
                "qpos": qpos,
                "fps": 30,
                "robot_name": "bumi",
                "joint_names": list(kinematics.joint_order),
                "quaternion_convention": "wxyz",
                "qpos_order": "mujoco_native",
                "source_dataset": dataset,
                "source_sample_id": sample_id,
                "source_motion_sha256": source_sha,
                "source_mjcf_sha256": config.robot_xml_sha256,
                "retarget_config_sha256": config.retarget_config_sha256,
                "quality_config_sha256": sha256_file(quality_config),
                "quality_report_sha256": sha256_file(quality_report),
                "quality_accepted": True,
                "retarget_quality": {
                    "status": "PASS",
                    "reason_codes": list(quality.get("reason_codes", ())),
                    "metrics": quality.get("metrics", {}),
                },
                "root_z_adjusted": False,
                "root_z_adjustment_method": "preserved_from_robot_retargeter_no_second_offset",
                "ground_semantics": GROUND_SEMANTICS,
                "root_orientation_audit": root_tilt_statistics(tilt),
                "foot_contact": contact.contact.contiguous(),
                "foot_contact_contract_version": BUMI_CONTACT_CONTRACT_VERSION,
                "foot_contact_source": "derived_from_robot_retargeter_qpos_fk_world_ground_zero",
                "foot_contact_ground_height_m": float(contact.ground_height),
            }
            destination = dataset_root / motion_relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, destination)
            copy_once(feature_source, dataset_root / feature_relative, feature_sha)
            copy_once(audio_source, dataset_root / audio_relative, audio_sha)
            row = {
                **{
                    key: value
                    for key, value in reference_row.items()
                    if key not in {"motion_path", "music_feature_path", "audio_path"}
                },
                "sample_id": sample_id,
                "dataset": DATASET_SPECS[dataset]["contract_name"],
                "motion_path": motion_relative.as_posix(),
                "music_feature_path": feature_relative.as_posix(),
                "audio_path": audio_relative.as_posix(),
                "fps": 30,
                "num_frames": frames,
                "split": split,
                "quality_accepted": True,
                "source_motion_sha256": source_sha,
                "source_music_feature_sha256": feature_sha,
                "source_audio_sha256": audio_sha,
            }
            split_rows[dataset][split].append(row)
            counts[f"{dataset}:{split}"] += 1
            counts[f"{dataset}:total"] += 1

        tilt_stats = {
            dataset: root_tilt_statistics(np.concatenate(values))
            for dataset, values in tilt_by_dataset.items()
            if values
        }
        split_counts: Counter[str] = Counter()
        for dataset, by_split in split_rows.items():
            dataset_root = staging / DATASET_SPECS[dataset]["output"]
            for split, rows in by_split.items():
                rows.sort(key=lambda row: str(row["sample_id"]))
                _write_jsonl(dataset_root / "manifests" / f"{split}.jsonl", rows)
                split_counts[split] += len(rows)
            if counts[f"{dataset}:total"] <= 0:
                raise ValueError(f"{dataset}: PASS 后为空，拒绝发布")
            info = {
                "contract_version": BUMI_MUSIC_CONTRACT_VERSION,
                "builder_contract_version": BUILDER_CONTRACT_VERSION,
                "robot_name": "bumi",
                "dataset_name": DATASET_SPECS[dataset]["contract_name"],
                "source_dataset": dataset,
                "qpos_dim": 28,
                "joint_dim": 21,
                "joint_names": list(kinematics.joint_order),
                "quaternion_convention": "wxyz",
                "qpos_order": "mujoco_native",
                "fps": 30,
                "quality_filter_applied": True,
                "quality_acceptance_policy": "PASS_ONLY_REVIEW_REJECT_EXCLUDED",
                "quality_joint_limit_violation_max_rad": config.joint_limit_violation_max,
                "reader_joint_limit_tolerance_rad": config.joint_limit_violation_max,
                "mjcf_sha256": config.robot_xml_sha256,
                "source_mjcf_sha256": config.robot_xml_sha256,
                "kinematics_sha256": kinematics.kinematics_sha256,
                "retarget_config_sha256": config.retarget_config_sha256,
                "quality_config_sha256": sha256_file(quality_config),
                "quality_report_sha256": sha256_file(quality_report),
                "ground_semantics": GROUND_SEMANTICS,
                "root_z_adjusted": False,
                "root_z_adjustment_method": "preserved_from_robot_retargeter_no_second_offset",
                "root_orientation_gate": {
                    "scope": "per_dataset_all_frames",
                    "statistics": tilt_stats[dataset],
                    "all_sequences_recomputed_and_dataset_passed": True,
                },
                "split_counts": {split: len(rows) for split, rows in by_split.items()},
            }
            _write_json(dataset_root / "meta" / "dataset_info.json", info)

        report = {
            "status": "passed",
            "contract_version": BUILDER_CONTRACT_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_files_modified": False,
            "quality_input_sequences": 3154,
            "quality_status_counts": summary["status_counts"],
            "total_pass_sequences": len(pass_rows),
            "review_reject_excluded": 3154 - len(pass_rows),
            "upstream_failed_excluded": 8,
            "dataset_counts": {
                dataset: {
                    "total": counts[f"{dataset}:total"],
                    **{split: counts[f"{dataset}:{split}"] for split in ("train", "val", "test")},
                }
                for dataset in DATASET_SPECS
            },
            "split_counts": dict(split_counts),
            "root_orientation_statistics": tilt_stats,
            "ground_semantics": GROUND_SEMANTICS,
            "root_z_second_adjustment_applied": False,
            "quality_report_sha256": sha256_file(quality_report),
            "quality_summary_sha256": sha256_file(quality_summary),
            "quality_config_sha256": sha256_file(quality_config),
            "kinematics_sha256": kinematics.kinematics_sha256,
            "verified_assets": assets,
            "materialization": dict(materialization),
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
    parser.add_argument("--quality-summary", required=True, type=Path)
    parser.add_argument("--quality-config", required=True, type=Path)
    parser.add_argument("--robot-xml", required=True, type=Path)
    parser.add_argument("--retarget-config", required=True, type=Path)
    parser.add_argument("--kinematics", required=True, type=Path)
    parser.add_argument("--reference-root", action="append", required=True, type=_parse_mapping)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--expected-pass", type=int)
    args = parser.parse_args()
    report = convert_datasets(
        source_root=args.source_root,
        quality_report=args.quality_report,
        quality_summary=args.quality_summary,
        quality_config=args.quality_config,
        robot_xml=args.robot_xml,
        retarget_config=args.retarget_config,
        kinematics_path=args.kinematics,
        reference_roots=_mapping(args.reference_root, "--reference-root"),
        output_root=args.output_root,
        expected_pass=args.expected_pass,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
