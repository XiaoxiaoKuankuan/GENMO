#!/usr/bin/env python3
"""将已发布的 BUMI3、SMPL-X 和 WAV 整理为可交付的同名多模态数据包。

这个工具面向数据管理和离线交付，不参与训练数据的生成。它以四库正式 manifest
作为唯一配对依据，对每个 ``sample_id`` 同时定位统一 Z-up 后的 SMPL-X NPZ、
重定向 BUMI3 PT 和对应的原始 WAV，再将三者物化为完全相同的文件主名。
输出的 ``source_video_mp4`` 只创建四个数据集空目录，不伪造或占位任何视频文件。

工具在写入前验证坐标系、SMPL-X neutral/16 维 betas、30 Hz 帧数、BUMI3 qpos28、
左右足接触、关节顺序和 WAV PCM 元数据；所有复制文件都在 ``dataset_contract.json``
中记录来源路径、大小和 SHA256。目录先写入同父级临时 staging，只有全部校验通过
才原子改名为正式目录，随后生成包含唯一一级目录的 ``tar.gz``。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

DATASETS = {
    "aioz_gdance": {"formal_folder": "AIOZ-GDANCE", "manifest_name": "aioz_gdance_bumi"},
    "aistpp": {"formal_folder": "AIST++", "manifest_name": "aistpp_bumi"},
    "compas3d": {"formal_folder": "CoMPAS3D", "manifest_name": "compas3d_bumi"},
    "finedance": {"formal_folder": "FineDance", "manifest_name": "finedance_bumi"},
}
DATA_TYPES = ("bumi3_motion", "human_smplx_motion", "music_wav", "source_video_mp4")
SMPLX_BODY_JOINT_ORDER = (
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)
EXPECTED_BUMI_JOINT_ORDER = (
    "l_leg_pitch_joint",
    "l_leg_roll_joint",
    "l_leg_yaw_joint",
    "l_knee_pitch_joint",
    "l_ankle_pitch_joint",
    "l_ankle_roll_joint",
    "r_leg_pitch_joint",
    "r_leg_roll_joint",
    "r_leg_yaw_joint",
    "r_knee_pitch_joint",
    "r_ankle_pitch_joint",
    "r_ankle_roll_joint",
    "waist_yaw_joint",
    "l_arm_pitch_joint",
    "l_arm_roll_joint",
    "l_arm_yaw_joint",
    "l_elbow_pitch_joint",
    "r_arm_pitch_joint",
    "r_arm_roll_joint",
    "r_arm_yaw_joint",
    "r_elbow_pitch_joint",
)


def sha256_file(path: Path) -> str:
    """分块计算文件 SHA256，避免大 WAV 一次读入内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取非空 JSONL，并在精确行号上报告结构错误。"""

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def scalar_text(value: Any) -> str:
    """把 NPZ 的标量字符串/字节统一为小写文本。"""

    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"expected scalar metadata, got {array.shape}")
    item = array.reshape(()).item()
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    return str(item).strip().lower()


def wav_metadata(path: Path) -> dict[str, Any]:
    """用 Python 标准库读取 PCM WAV 契约，不依赖服务器 ffprobe。"""

    with wave.open(str(path), "rb") as handle:
        if handle.getcomptype() != "NONE":
            raise ValueError(f"{path}: only uncompressed PCM WAV is supported")
        sample_rate = int(handle.getframerate())
        frame_count = int(handle.getnframes())
        return {
            "codec": "pcm" if handle.getcomptype() == "NONE" else handle.getcomptype(),
            "channels": int(handle.getnchannels()),
            "sample_width_bytes": int(handle.getsampwidth()),
            "sample_rate_hz": sample_rate,
            "num_audio_frames": frame_count,
            "duration_sec": frame_count / sample_rate,
        }


def audit_source_inventory(formal_root: Path, smplx_root: Path) -> dict[str, dict[str, int]]:
    """对四库全部 manifest 执行路径存在性审计，不只检查原型抽样。"""

    inventory: dict[str, dict[str, int]] = {}
    for dataset, spec in DATASETS.items():
        dataset_root = formal_root / spec["formal_folder"]
        rows: list[dict[str, Any]] = []
        split_counts: dict[str, int] = {}
        for split in ("train", "val", "test"):
            manifest = dataset_root / "manifests" / f"{split}.jsonl"
            split_rows = read_jsonl(manifest)
            rows.extend(split_rows)
            split_counts[split] = len(split_rows)
        sample_ids = [str(row["sample_id"]) for row in rows]
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError(f"{dataset}: duplicate sample_id in formal manifests")
        missing: list[str] = []
        for row in rows:
            sample_id = str(row["sample_id"])
            required = (
                smplx_root / dataset / f"{sample_id}.npz",
                dataset_root / str(row["motion_path"]),
                dataset_root / str(row["audio_path"]),
            )
            missing.extend(str(path) for path in required if not path.is_file())
        if missing:
            raise FileNotFoundError(
                f"{dataset}: {len(missing)} paired source files missing; first={missing[0]}"
            )
        inventory[dataset] = {
            "paired_samples": len(rows),
            "smplx_npz_files": len(list((smplx_root / dataset).glob("*.npz"))),
            "bumi3_pt_files": len(list((dataset_root / "motions").glob("*.pt"))),
            "unique_wav_files": len(list((dataset_root / "audio").glob("*.wav"))),
            "train_samples": split_counts["train"],
            "val_samples": split_counts["val"],
            "test_samples": split_counts["test"],
            "missing_paired_files": 0,
        }
    return inventory


def safe_torch_load(path: Path) -> dict[str, Any]:
    """只加载受信任正式训练根的 tensor/基础类型字典。"""

    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # 兼容旧 PyTorch。
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a dictionary payload")
    return value


def validate_smplx(path: Path, expected_frames: int) -> dict[str, Any]:
    """验证统一后 SMPL-X 字段、坐标、neutral 模型和帧数。"""

    with np.load(path, allow_pickle=False) as payload:
        required = {
            "root_orient",
            "pose_body",
            "trans",
            "betas",
            "mocap_frame_rate",
            "gender",
            "coordinate_system",
            "source_coordinate_system",
            "coordinate_transform",
            "coordinate_system_was_assumed",
        }
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"{path}: missing SMPL-X fields {missing}")
        shapes = {
            "root_orient": (expected_frames, 3),
            "pose_body": (expected_frames, 63),
            "trans": (expected_frames, 3),
            "betas": (16,),
        }
        for field, expected_shape in shapes.items():
            value = np.asarray(payload[field])
            if value.shape != expected_shape or not np.isfinite(value).all():
                raise ValueError(
                    f"{path}: {field} expected finite {expected_shape}, got {value.shape}"
                )
        if float(np.asarray(payload["mocap_frame_rate"]).reshape(())) != 30.0:
            raise ValueError(f"{path}: SMPL-X FPS must be 30")
        expected_scalars = {
            "gender": "neutral",
            "coordinate_system": "right_handed_z_up_metric",
            "source_coordinate_system": "right_handed_y_up_metric",
            "coordinate_transform": "rotate_global_root_and_translation_plus_90deg_about_x",
        }
        for field, expected_value in expected_scalars.items():
            if scalar_text(payload[field]) != expected_value:
                raise ValueError(f"{path}: {field} must be {expected_value!r}")
        if bool(np.asarray(payload["coordinate_system_was_assumed"]).reshape(())):
            raise ValueError(f"{path}: formal SMPL-X coordinate system must not be assumed")
        return {
            "num_frames": expected_frames,
            "fps": 30.0,
            "gender": "neutral",
            "num_betas": 16,
            "coordinate_system": "right_handed_z_up_metric",
            "source_coordinate_system": "right_handed_y_up_metric",
            "coordinate_transform": "rotate_global_root_and_translation_plus_90deg_about_x",
            "fields": {
                field: {
                    "shape": list(np.asarray(payload[field]).shape),
                    "dtype": str(payload[field].dtype),
                }
                for field in ("root_orient", "pose_body", "trans", "betas")
            },
        }


def validate_bumi(path: Path, expected_frames: int) -> dict[str, Any]:
    """验证 BUMI3 qpos28、足接触、顺序和重定向契约。"""

    payload = safe_torch_load(path)
    qpos = torch.as_tensor(payload.get("qpos")).detach().cpu()
    contact = torch.as_tensor(payload.get("foot_contact")).detach().cpu()
    if tuple(qpos.shape) != (expected_frames, 28) or not bool(torch.isfinite(qpos).all()):
        raise ValueError(f"{path}: qpos must be finite [{expected_frames},28]")
    if tuple(contact.shape) != (expected_frames, 2) or not bool(torch.isfinite(contact).all()):
        raise ValueError(f"{path}: foot_contact must be finite [{expected_frames},2]")
    if bool(((contact < 0.0) | (contact > 1.0)).any()):
        raise ValueError(f"{path}: foot_contact must be in [0,1]")
    expected_metadata = {
        "fps": 30,
        "robot_name": "bumi",
        "quaternion_convention": "wxyz",
        "qpos_order": "mujoco_native",
        "ground_semantics": "gmr_foot_sole_ground_zero_v1",
        "foot_contact_contract_version": "genmo.bumi_foot_contact.fk_sole_hysteresis.v1",
    }
    for field, expected_value in expected_metadata.items():
        if payload.get(field) != expected_value:
            raise ValueError(f"{path}: {field} must be {expected_value!r}")
    if tuple(map(str, payload.get("joint_names", ()))) != EXPECTED_BUMI_JOINT_ORDER:
        raise ValueError(f"{path}: BUMI3 joint order mismatch")
    return {
        "num_frames": expected_frames,
        "fps": 30,
        "qpos_shape": list(qpos.shape),
        "qpos_dtype": str(qpos.dtype),
        "foot_contact_shape": list(contact.shape),
        "foot_contact_dtype": str(contact.dtype),
        "quaternion_convention": "wxyz",
        "qpos_order": "mujoco_native",
        "joint_names": list(EXPECTED_BUMI_JOINT_ORDER),
        "ground_semantics": expected_metadata["ground_semantics"],
        "foot_contact_contract_version": expected_metadata["foot_contact_contract_version"],
    }


def select_rows(
    formal_root: Path, split: str, samples_per_dataset: int
) -> dict[str, list[dict[str, Any]]]:
    """按时长对齐误差和 sample_id 稳定选样；0 表示收录全部。"""

    selected: dict[str, list[dict[str, Any]]] = {}
    audio_cache: dict[Path, dict[str, Any]] = {}
    requested_splits = ("train", "val", "test") if split == "all" else (split,)
    for dataset, spec in DATASETS.items():
        dataset_root = formal_root / spec["formal_folder"]
        rows: list[dict[str, Any]] = []
        for current_split in requested_splits:
            manifest = dataset_root / "manifests" / f"{current_split}.jsonl"
            if manifest.is_file():
                rows.extend(read_jsonl(manifest))
        if not rows:
            raise ValueError(f"{dataset}: no manifest rows for split={split}")
        scored: list[tuple[float, str, dict[str, Any]]] = []
        for row in rows:
            if row.get("dataset") != spec["manifest_name"]:
                raise ValueError(f"{dataset}: manifest dataset mismatch for {row.get('sample_id')}")
            audio_path = dataset_root / str(row["audio_path"])
            if audio_path not in audio_cache:
                audio_cache[audio_path] = wav_metadata(audio_path)
            motion_duration = int(row["num_frames"]) / float(row["fps"])
            error = abs(audio_cache[audio_path]["duration_sec"] - motion_duration)
            scored.append((error, str(row["sample_id"]), row))
        scored.sort(key=lambda item: (item[0], item[1]))
        selected[dataset] = [item[2] for item in scored]
        if samples_per_dataset > 0:
            selected[dataset] = selected[dataset][:samples_per_dataset]
    return selected


def copy_verified(source: Path, target: Path) -> dict[str, Any]:
    """复制单个文件并验证前后 SHA256 完全一致。"""

    target.parent.mkdir(parents=True, exist_ok=True)
    source_digest = sha256_file(source)
    shutil.copy2(source, target)
    target_digest = sha256_file(target)
    if source_digest != target_digest:
        raise ValueError(f"copy SHA256 mismatch: {source} -> {target}")
    return {
        "path": str(target),
        "size_bytes": target.stat().st_size,
        "sha256": target_digest,
        "source_path": str(source),
    }


def package_dataset(args: argparse.Namespace) -> dict[str, Any]:
    """完成选样、契约验证、原子目录发布和 tar.gz 打包。"""

    formal_root = args.formal_bumi_root.expanduser().resolve()
    smplx_root = args.normalized_smplx_root.expanduser().resolve()
    output_parent = args.output_parent.expanduser().resolve()
    neutral_model = args.smplx_neutral_model.expanduser().resolve()
    bumi_mjcf = args.bumi_mjcf.expanduser().resolve()
    retarget_config = args.retarget_config.expanduser().resolve()
    for required in (formal_root, smplx_root, neutral_model, bumi_mjcf, retarget_config):
        if not required.exists():
            raise FileNotFoundError(required)
    if args.samples_per_dataset < 0:
        raise ValueError("--samples-per-dataset must be >= 0")
    if not args.package_name or Path(args.package_name).name != args.package_name:
        raise ValueError("--package-name must be one directory name")

    output_parent.mkdir(parents=True, exist_ok=True)
    output_root = output_parent / args.package_name
    archive_path = output_parent / f"{args.package_name}.tar.gz"
    if output_root.exists() or archive_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_root} or {archive_path}")

    source_inventory = audit_source_inventory(formal_root, smplx_root)
    selected = select_rows(formal_root, args.split, args.samples_per_dataset)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.package_name}.staging-", dir=output_parent))
    try:
        for data_type in DATA_TYPES:
            for dataset in DATASETS:
                (staging / data_type / dataset).mkdir(parents=True, exist_ok=True)

        samples: list[dict[str, Any]] = []
        for dataset, rows in selected.items():
            spec = DATASETS[dataset]
            source_dataset_root = formal_root / spec["formal_folder"]
            for row in rows:
                sample_id = str(row["sample_id"])
                num_frames = int(row["num_frames"])
                smplx_source = smplx_root / dataset / f"{sample_id}.npz"
                bumi_source = source_dataset_root / str(row["motion_path"])
                audio_source = source_dataset_root / str(row["audio_path"])
                for source in (smplx_source, bumi_source, audio_source):
                    if not source.is_file():
                        raise FileNotFoundError(source)

                smplx_contract = validate_smplx(smplx_source, num_frames)
                bumi_contract = validate_bumi(bumi_source, num_frames)
                audio_contract = wav_metadata(audio_source)
                outputs = {
                    "human_smplx_motion": copy_verified(
                        smplx_source,
                        staging / "human_smplx_motion" / dataset / f"{sample_id}.npz",
                    ),
                    "bumi3_motion": copy_verified(
                        bumi_source,
                        staging / "bumi3_motion" / dataset / f"{sample_id}.pt",
                    ),
                    "music_wav": copy_verified(
                        audio_source,
                        staging / "music_wav" / dataset / f"{sample_id}.wav",
                    ),
                }
                for value in outputs.values():
                    value["path"] = str(Path(value["path"]).relative_to(staging))
                output_stems = {Path(value["path"]).stem for value in outputs.values()}
                if output_stems != {sample_id}:
                    raise ValueError(f"{dataset}/{sample_id}: output stems are not identical")
                samples.append(
                    {
                        "dataset": dataset,
                        "sample_id": sample_id,
                        "sequence_id": str(row["sequence_id"]),
                        "split": str(row["split"]),
                        "num_frames": num_frames,
                        "motion_duration_sec": num_frames / 30.0,
                        "audio_key": str(row["audio_key"]),
                        "audio_duration_sec": audio_contract["duration_sec"],
                        "motion_audio_duration_abs_error_sec": abs(
                            audio_contract["duration_sec"] - num_frames / 30.0
                        ),
                        "smplx_contract": smplx_contract,
                        "bumi3_contract": bumi_contract,
                        "wav_contract": audio_contract,
                        "files": outputs,
                        "source_video_mp4": {
                            "available": False,
                            "expected_future_path": f"source_video_mp4/{dataset}/{sample_id}.mp4",
                        },
                        "source_manifest_row": row,
                    }
                )

        contract = {
            "schema": "genmo.bumi3_smplx_music_delivery.v1",
            "package_name": args.package_name,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": {
                "kind": "prototype" if args.samples_per_dataset > 0 else "full_release",
                "split": args.split,
                "samples_per_dataset": args.samples_per_dataset,
                "total_samples": len(samples),
                "source_videos_included": False,
            },
            "pairing_contract": {
                "key": "sample_id",
                "rule": "the NPZ, PT, WAV, and future MP4 use an identical filename stem",
                "paths": {
                    "human_smplx_motion": "human_smplx_motion/{dataset}/{sample_id}.npz",
                    "bumi3_motion": "bumi3_motion/{dataset}/{sample_id}.pt",
                    "music_wav": "music_wav/{dataset}/{sample_id}.wav",
                    "source_video_mp4": "source_video_mp4/{dataset}/{sample_id}.mp4",
                },
            },
            "directory_contract": {
                "level_1": args.package_name,
                "level_2": list(DATA_TYPES),
                "level_3_datasets": list(DATASETS),
                "metadata_json": "dataset_contract.json",
                "source_video_directories_are_intentionally_empty": True,
            },
            "source_inventory": source_inventory,
            "coordinate_contract": {
                "source_public_smplx": "right_handed_y_up_metric",
                "packaged_smplx": "right_handed_z_up_metric",
                "packaged_bumi3": "right_handed_z_up_metric",
                "transform_before_human_fk_contact_root_z_and_ik": (
                    "rotate_global_root_and_translation_plus_90deg_about_x"
                ),
                "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
                "handedness": "right_handed",
                "up_axis": "+Z",
            },
            "smplx_contract": {
                "format": "numpy_npz",
                "fps": 30.0,
                "model_type": "smplx",
                "gender": "neutral",
                "required_external_body_model": neutral_model.name,
                "body_model_included_in_package": False,
                "body_model_sha256": sha256_file(neutral_model),
                "num_betas": 16,
                "use_pca": False,
                "root_orient": "float32 [T,3], global axis-angle, radians",
                "pose_body": "float32 [T,63], 21 local axis-angle rotations, radians",
                "trans": "float32 [T,3], global root translation, metres",
                "betas": "float32 [16], one fixed dimensionless shape vector per sequence",
                "body_pose_joint_order": list(SMPLX_BODY_JOINT_ORDER),
                "hands_jaw_eyes_expression_for_retargeting": "zeros",
            },
            "bumi3_contract": {
                "format": "pytorch_dictionary_pt",
                "fps": 30,
                "qpos": "float32 [T,28]",
                "qpos_slices": {
                    "root_position": {"slice": [0, 3], "unit": "metre", "frame": "world"},
                    "root_quaternion": {
                        "slice": [3, 7],
                        "order": "wxyz",
                        "meaning": "MuJoCo free-joint body orientation in world",
                    },
                    "joint_position": {"slice": [7, 28], "unit": "radian"},
                },
                "joint_order": list(EXPECTED_BUMI_JOINT_ORDER),
                "foot_contact": "float32 [T,2], order [left_foot,right_foot], values in [0,1]",
                "ground_semantics": "gmr_foot_sole_ground_zero_v1",
                "mjcf_filename": bumi_mjcf.name,
                "mjcf_included_in_package": False,
                "mjcf_sha256": sha256_file(bumi_mjcf),
                "retarget_config_filename": retarget_config.name,
                "retarget_config_included_in_package": False,
                "retarget_config_sha256": sha256_file(retarget_config),
            },
            "music_wav_contract": {
                "container": "wav",
                "codec": "PCM for all packaged prototype files",
                "sample_rate_channels_and_bit_depth": "recorded per sample because sources differ",
                "unit": "integer PCM amplitude",
                "audio_is_copied_without_resampling": True,
            },
            "datasets": {
                dataset: {
                    "formal_folder": spec["formal_folder"],
                    "manifest_dataset_name": spec["manifest_name"],
                    "original_smplx_coordinate_system": "right_handed_y_up_metric",
                    "packaged_coordinate_system": "right_handed_z_up_metric",
                    "selected_samples": sum(sample["dataset"] == dataset for sample in samples),
                }
                for dataset, spec in DATASETS.items()
            },
            "samples": sorted(samples, key=lambda sample: (sample["dataset"], sample["sample_id"])),
        }
        contract_path = staging / "dataset_contract.json"
        contract_path.write_text(
            json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        for data_type in ("human_smplx_motion", "bumi3_motion", "music_wav"):
            for dataset in DATASETS:
                expected = sum(sample["dataset"] == dataset for sample in samples)
                actual = sum(path.is_file() for path in (staging / data_type / dataset).iterdir())
                if actual != expected:
                    raise ValueError(
                        f"{data_type}/{dataset}: expected {expected} files, got {actual}"
                    )
        for dataset in DATASETS:
            if any((staging / "source_video_mp4" / dataset).iterdir()):
                raise ValueError(f"source_video_mp4/{dataset} must be empty")

        os.replace(staging, output_root)
        archive_temporary = output_parent / f".{args.package_name}.tar.gz.tmp-{os.getpid()}"
        try:
            with tarfile.open(archive_temporary, "w:gz") as archive:
                archive.add(output_root, arcname=args.package_name, recursive=True)
            os.replace(archive_temporary, archive_path)
            with tarfile.open(archive_path, "r:gz") as archive:
                archived_names = archive.getnames()
            if not archived_names or any(
                Path(name).parts[0] != args.package_name for name in archived_names
            ):
                raise ValueError("archive must contain exactly one named level-1 package root")
        finally:
            if archive_temporary.exists():
                archive_temporary.unlink()
        return {
            "status": "passed",
            "package_root": str(output_root),
            "package_archive": str(archive_path),
            "archive_size_bytes": archive_path.stat().st_size,
            "archive_sha256": sha256_file(archive_path),
            "contract_sha256": sha256_file(output_root / "dataset_contract.json"),
            "total_samples": len(samples),
            "samples_by_dataset": {
                dataset: sum(sample["dataset"] == dataset for sample in samples)
                for dataset in DATASETS
            },
        }
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        if output_root.exists():
            shutil.rmtree(output_root)
        if archive_path.exists():
            archive_path.unlink()
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-bumi-root", required=True, type=Path)
    parser.add_argument("--normalized-smplx-root", required=True, type=Path)
    parser.add_argument("--smplx-neutral-model", required=True, type=Path)
    parser.add_argument("--bumi-mjcf", required=True, type=Path)
    parser.add_argument("--retarget-config", required=True, type=Path)
    parser.add_argument("--output-parent", required=True, type=Path)
    parser.add_argument("--package-name", default="bumi3_smplx_music_dataset_sample_v1")
    parser.add_argument("--samples-per-dataset", type=int, default=1)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="train")
    return parser.parse_args()


def main() -> None:
    report = package_dataset(parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
