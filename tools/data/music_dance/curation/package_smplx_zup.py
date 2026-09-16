#!/usr/bin/env python3
"""将四库完整人体审阅包转换为 30 Hz、Z-up 的 UMR 可读交付包。

本工具只读取具有 master 索引和 SHA256 的正式 Y-up 人体 NPZ，严格核对四库数量、
文件集合、字段、帧数、有限值和 30 Hz。世界根旋转左乘 X 轴 +90°，平移同步从
(x,y,z) 变为 (x,-z,y)，身体局部轴角不变；这与已有 3162 条标准化输入的参数
变换约定一致，不额外修正人体模型的骨盆原点、不贴地、不缩放、不重采样。

为兼容当前 UMR 的固定形状模型，betas 保存源首帧的 10 维参数并补 6 个零；
source_betas 则逐元素保留原始 [T,10] 数组，包括 AIOZ 的逐帧形状变化。gender
沿用源审阅包的 neutral 契约，不将它解释为原始演员性别。输入没有的手指、面部
姿态不伪造。完整源索引、数据集与音乐配对标识、字段说明和校验和一并交付。

转换文件写入独立临时 staging，重读并逐帧验证旋转矩阵、平移、身体姿态、形状
保真后，生成带唯一顶层目录的 tar.gz。压缩使用至多 4 个 pigz 线程，缺少 pigz
时退回标准库；归档内所有文件的 SHA256 和 gzip 完整性通过后才发布目录和归档。
已有目标一律拒绝覆盖。临时 staging 在成功或失败时自动清理，生产源数据不修改。
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

COUNTS = {"aistpp": 1020, "aioz_gdance": 6011, "finedance": 183, "compas3d": 72}
SCHEMA = "genmo.smplx_zup_delivery.v1"
Y_UP = "right_handed_y_up_metric"
Z_UP = "right_handed_z_up_metric"
TRANSFORM = "rotate_global_root_and_translation_plus_90deg_about_x"
BETA_POLICY = "source_first_frame_10_plus_6_zero_padding"
ROTATION = Rotation.from_euler("x", 90.0, degrees=True)
MATRIX = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
BODY_JOINTS = (
    "left_hip right_hip spine1 left_knee right_knee spine2 left_ankle right_ankle "
    "spine3 left_foot right_foot neck left_collar right_collar head left_shoulder "
    "right_shoulder left_elbow right_elbow left_wrist right_wrist"
).split()


def sha256_file(path: Path) -> str:
    """流式计算文件校验和，避免将归档整体读入内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def validate_source(data: Any, row: dict[str, Any]) -> int:
    """拒绝错坐标、错 FPS、非有限值及与索引不一致的源动作。"""
    frames = int(data["num_frames"])
    if frames < 1 or frames != row["num_frames"]:
        raise ValueError("source frame count differs from master")
    if float(data["fps"]) != 30.0 or float(row["fps"]) != 30.0:
        raise ValueError("source fps must be 30")
    if str(data["coordinate_system"]) != Y_UP:
        raise ValueError("source must explicitly declare right-handed metric Y-up")
    for key in ("dataset", "sample_id", "review_id"):
        if str(data[key]) != row[key]:
            raise ValueError(f"source {key} differs from master")
    if "gender" in data and str(data["gender"]).lower() != "neutral":
        raise ValueError("source review package requires neutral gender")
    for key, shape in (("pose", (frames, 66)), ("transl", (frames, 3)), ("betas", (frames, 10))):
        value = data[key]
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"invalid source {key}: expected finite float32 {shape}")
    return frames


def convert_motion(data: Any, row: dict[str, Any]) -> dict[str, np.ndarray]:
    """转换全局参数；源逐帧形状单独保留，静态形状明确采用首帧。"""
    frames = validate_source(data, row)
    pose = data["pose"]
    payload = {
        "root_orient": (ROTATION * Rotation.from_rotvec(pose[:, :3]))
        .as_rotvec()
        .astype(np.float32),
        "pose_body": pose[:, 3:66].copy(),
        "trans": (data["transl"] @ MATRIX.T).astype(np.float32),
        "betas": np.pad(data["betas"][0], (0, 6)),
        "source_betas": data["betas"].copy(),
        "mocap_frame_rate": np.asarray(30.0),
        "fps": np.asarray(30.0, dtype=np.float32),
        "num_frames": np.asarray(frames, dtype=np.int64),
        "gender": np.asarray("neutral"),
        "gender_source": np.asarray("source_review_package_neutral_contract"),
        "coordinate_system": np.asarray(Z_UP),
        "source_coordinate_system": np.asarray(Y_UP),
        "coordinate_transform": np.asarray(TRANSFORM),
        "coordinate_system_was_assumed": np.asarray(False),
        "output_up": np.asarray("z"),
        "betas_policy": np.asarray(BETA_POLICY),
        "source_betas_time_varying": np.asarray(bool(np.any(data["betas"] != data["betas"][0]))),
        "source_npz_sha256": np.asarray(row["review_sha256"]),
        "schema": np.asarray(SCHEMA),
    }
    for key in ("dataset", "sample_id", "review_id", "split", "music_key"):
        payload[key] = np.asarray(row[key])
    return payload


def validate_conversion(source: Any, output: Any) -> dict[str, float]:
    """逐帧核对实际落盘参数的变换，不仅依靠坐标元数据判断成功。"""
    frames = len(source["pose"])
    for key, shape in (
        ("root_orient", (frames, 3)),
        ("pose_body", (frames, 63)),
        ("trans", (frames, 3)),
        ("betas", (16,)),
        ("source_betas", (frames, 10)),
    ):
        value = output[key]
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"invalid output {key}")
    if not np.array_equal(output["pose_body"], source["pose"][:, 3:66]):
        raise ValueError("body pose changed")
    if not np.array_equal(output["source_betas"], source["betas"]):
        raise ValueError("original per-frame betas were not preserved")
    if not np.array_equal(output["betas"], np.pad(source["betas"][0], (0, 6))):
        raise ValueError("static beta policy mismatch")
    if not np.array_equal(output["trans"] @ MATRIX, source["transl"]):
        raise ValueError("translation round-trip mismatch")
    expected_rotation = MATRIX @ Rotation.from_rotvec(source["pose"][:, :3]).as_matrix()
    root_error = float(
        np.max(np.abs(expected_rotation - Rotation.from_rotvec(output["root_orient"]).as_matrix()))
    )
    if root_error > 5e-7:
        raise ValueError(f"root rotation matrix error {root_error}")
    for key, expected in (
        ("coordinate_system", Z_UP),
        ("source_coordinate_system", Y_UP),
        ("coordinate_transform", TRANSFORM),
        ("gender", "neutral"),
        ("output_up", "z"),
        ("betas_policy", BETA_POLICY),
    ):
        if str(output[key]) != expected:
            raise ValueError(f"output {key} mismatch")
    if float(output["mocap_frame_rate"]) != 30.0 or int(output["num_frames"]) != frames:
        raise ValueError("output temporal contract mismatch")
    return {"root_matrix_max_abs_error": root_error}


def verify_archive(archive_path: Path, package_name: str, hashes: dict[str, str]) -> int:
    """完整流式解压校验每个文件和 gzip 尾部；拒绝目录逃逸、链接及额外成员。"""
    seen: set[str] = set()
    with gzip.open(archive_path, "rb") as stream:
        with tarfile.open(fileobj=stream, mode="r|") as archive:
            for member in archive:
                parts = Path(member.name).parts
                if not parts or parts[0] != package_name or ".." in parts or not member.isfile():
                    raise ValueError(f"unsafe archive member: {member.name}")
                relative = Path(*parts[1:]).as_posix()
                if relative in seen or relative not in hashes:
                    raise ValueError(f"unexpected archive member: {member.name}")
                handle = archive.extractfile(member)
                if handle is None:
                    raise ValueError(f"unreadable archive member: {member.name}")
                digest = hashlib.sha256()
                while chunk := handle.read(4 * 1024 * 1024):
                    digest.update(chunk)
                if digest.hexdigest() != hashes[relative]:
                    raise ValueError(f"archive SHA256 mismatch: {member.name}")
                seen.add(relative)
        while stream.read(4 * 1024 * 1024):
            pass
    if seen != set(hashes):
        raise ValueError("archive file inventory mismatch")
    return len(seen)


def create_archive(package: Path, archive_path: Path, hashes: dict[str, str]) -> str:
    """有界并行压缩，归档仅包含已校验的普通文件。"""
    pigz = shutil.which("pigz")
    if pigz:
        with archive_path.open("wb") as output:
            process = subprocess.Popen(
                [pigz, "-6", "-p", "4", "-c"], stdin=subprocess.PIPE, stdout=output
            )
            try:
                with tarfile.open(fileobj=process.stdin, mode="w|") as archive:
                    for relative in sorted(hashes):
                        archive.add(
                            package / relative,
                            arcname=f"{package.name}/{relative}",
                            recursive=False,
                        )
            finally:
                if process.stdin is not None:
                    process.stdin.close()
                result = process.wait()
            if result != 0:
                raise RuntimeError(f"pigz exited {result}")
        return "pigz -6 -p 4"
    with tarfile.open(archive_path, "w:gz", compresslevel=6) as archive:
        for relative in sorted(hashes):
            archive.add(package / relative, arcname=f"{package.name}/{relative}", recursive=False)
    return "python gzip level 6"


def package_dataset(
    source_root: Path,
    output_parent: Path,
    package_name: str,
    expected_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """校验、转换、打包、复核并发布独立目录，拒绝覆盖任何已有目标。"""
    expected_counts = COUNTS if expected_counts is None else expected_counts
    source_root = source_root.expanduser().resolve()
    output_parent = output_parent.expanduser().resolve()
    if Path(package_name).name != package_name or package_name in {"", ".", ".."}:
        raise ValueError("package_name must be one safe path component")
    if output_parent == source_root or source_root in output_parent.parents:
        raise ValueError("output must be outside the source tree")
    targets = [
        output_parent / package_name,
        output_parent / f"{package_name}.tar.gz",
        output_parent / f"{package_name}.tar.gz.sha256",
        output_parent / f"{package_name}.release.json",
    ]
    if any(path.exists() or path.is_symlink() for path in targets):
        raise FileExistsError("refusing to overwrite an existing delivery target")
    master = source_root / "index/master.jsonl"
    master_bytes = master.read_bytes()
    rows = [json.loads(line) for line in master_bytes.decode().splitlines() if line.strip()]
    if Counter(row["dataset"] for row in rows) != expected_counts:
        raise ValueError("source dataset counts mismatch")
    inventory: set[str] = set()
    for row in rows:
        sample = row["sample_id"]
        if Path(sample).name != sample or sample in {"", ".", ".."}:
            raise ValueError("unsafe sample id")
        relative = f"motions/{row['dataset']}/{sample}.npz"
        if row["review_motion_path"] != relative or relative in inventory:
            raise ValueError("invalid or duplicate source path")
        if not (source_root / relative).resolve().is_relative_to(source_root):
            raise ValueError("source path escapes source tree")
        inventory.add(relative)
    actual = {
        p.relative_to(source_root).as_posix() for p in (source_root / "motions").rglob("*.npz")
    }
    if actual != inventory:
        raise ValueError("source NPZ inventory differs from master")

    output_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{package_name}.staging-", dir=output_parent) as td:
        staging = Path(td)
        package = staging / package_name
        (package / "index").mkdir(parents=True)
        (package / "index/source_master.jsonl").write_bytes(master_bytes)
        fingerprints = source_root / "index/source_fingerprints.json"
        if fingerprints.is_file():
            shutil.copyfile(fingerprints, package / "index/source_fingerprints.json")
        totals: dict[str, Any] = {
            "counts": {},
            "frames": {},
            "varying_betas": {},
            "root_matrix_max_abs_error": 0.0,
        }
        output_rows = []
        source_bytes_total = 0
        for i, row in enumerate(rows, 1):
            relative = row["review_motion_path"]
            raw = (source_root / relative).read_bytes()
            source_bytes_total += len(raw)
            if hashlib.sha256(raw).hexdigest() != row["review_sha256"]:
                raise ValueError(f"source SHA256 mismatch: {relative}")
            with np.load(io.BytesIO(raw), allow_pickle=False) as source:
                payload = convert_motion(source, row)
                path = package / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **payload)
                with np.load(path, allow_pickle=False) as output:
                    metrics = validate_conversion(source, output)
                dataset = row["dataset"]
                for key, value in (
                    ("counts", 1),
                    ("frames", int(payload["num_frames"])),
                    ("varying_betas", int(payload["source_betas_time_varying"])),
                ):
                    totals[key][dataset] = totals[key].get(dataset, 0) + value
                totals["root_matrix_max_abs_error"] = max(
                    totals["root_matrix_max_abs_error"], metrics["root_matrix_max_abs_error"]
                )
            output_rows.append(
                {
                    **row,
                    "source_review_motion_path": relative,
                    "source_review_sha256": row["review_sha256"],
                    "motion_path": relative,
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                    "coordinate_system": Z_UP,
                    "coordinate_transform": TRANSFORM,
                    "betas_policy": BETA_POLICY,
                    "source_betas_time_varying": bool(payload["source_betas_time_varying"]),
                }
            )
            if i % 250 == 0 or i == len(rows):
                print(
                    json.dumps(
                        {"stage": "converted_and_verified", "completed": i, "total": len(rows)}
                    ),
                    flush=True,
                )
        (package / "index/manifest.jsonl").write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output_rows
            )
        )
        repo_root = Path(__file__).resolve().parents[4]
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
        contract = {
            "schema": SCHEMA,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_root": str(source_root),
            "source_master_sha256": hashlib.sha256(master_bytes).hexdigest(),
            "producer_git_commit": commit,
            "producer_script_sha256": sha256_file(Path(__file__)),
            "total_samples": len(rows),
            "total_frames": sum(totals["frames"].values()),
            "total_hours": sum(totals["frames"].values()) / 30.0 / 3600.0,
            "fps": 30.0,
            "coordinate_system": Z_UP,
            "source_coordinate_system": Y_UP,
            "coordinate_transform": TRANSFORM,
            "rotation_matrix": MATRIX.tolist(),
            "root_rotation_formula": "R_out = Rx(+90deg) @ R_in",
            "translation_formula": "trans_out = trans_in @ Rx(+90deg).T",
            "pelvis_origin_compensation_applied": False,
            "ground_alignment_applied": False,
            "resampling_applied": False,
            "model_type": "smplx",
            "gender": "neutral",
            "gender_source": "source_review_package_neutral_contract",
            "body_pose_joint_order": BODY_JOINTS,
            "betas_policy": BETA_POLICY,
            "fields": {
                "root_orient": "float32 [T,3], global axis-angle, radians",
                "pose_body": "float32 [T,63], local axis-angle, radians",
                "trans": "float32 [T,3], metres",
                "betas": "float32 [16], source first frame plus zero padding",
                "source_betas": "float32 [T,10], exact original per-frame shape parameters",
                "mocap_frame_rate": "float64 scalar 30.0",
                "gender": "string scalar neutral",
                "output_up": "string scalar z",
            },
            "umr": {
                "direct_loader": "load_smplx_npz_motion",
                "uses_first_10_betas": True,
                "body_model": "SMPLX_NEUTRAL.pkl supplied separately",
                "body_only": True,
            },
            "scope": {
                "audio_included": False,
                "video_included": False,
                "body_model_included": False,
                "robot_motion_included": False,
                "hand_face_parameters_included": False,
            },
            "validation": totals,
        }
        write_json(package / "config.json", contract)
        write_json(
            package / "validation_report.json",
            {
                "status": "passed",
                "all_source_sha256_verified": True,
                "source_bytes": source_bytes_total,
                "full_frame_parameter_checks": True,
                "body_pose_unchanged": True,
                "source_betas_unchanged": True,
                **totals,
            },
        )
        (package / "README.md").write_text(
            "# 四库 30 Hz Z-up 人体动作交付包\n\n"
            f"共 {len(rows)} 条、{contract['total_frames']} 帧，完整保留 train/val/test；详细约定见 config.json。\n\n"
            "motions/<dataset>/<sample_id>.npz 为人体动作。root_orient 是世界根轴角 [T,3]，"
            "pose_body 是 21 个局部关节轴角 [T,63]，trans 是米制平移 [T,3]。全部 30 Hz、右手 Z-up。"
            "源 Y-up 根旋转和平移已绕 X 轴 +90°，不要重复旋转或只修改 FPS。未贴地、未缩放、未重采样。\n\n"
            "betas [16] 使用源首帧 10 维参数并补 6 个零，与已有 3162 条输入一致；"
            "source_betas [T,10] 完整保留原逐帧形状。AIOZ 的形状变化不会被当前 UMR 固定形状前向采用，"
            "如需使用原逐帧形状，应自行适配读取和前向，不能把静态版本称为原始网格逐帧等价。\n\n"
            "gender=neutral 来自统一人体审阅包约定，不是原始演员性别。手、脸、眼睛及表情参数未包含。"
            "使用者需自行提供 SMPL-X neutral 模型；当前 UMR 可直接读取本 NPZ，取 betas 前10维，"
            "使用 output_up=z。转换沿用已有根参数坐标约定，不额外施加模型骨盆原点补偿；"
            "人体或机器人落地高度、接触与重定向效果应在目标流水线中单独验证。\n\n"
            "index/manifest.jsonl 保存输出路径、SHA256、源索引、split 和音乐配对标识；"
            "index/source_master.jsonl 是原索引副本，含源服务器路径。音乐文件、视频、机器人动作"
            "和人体模型均不随包分发。源 review_coordinate_system 描述转换前的数据，"
            "输出 coordinate_system 描述当前 NPZ。\n\n"
            "解压后在包根执行 `sha256sum -c SHA256SUMS` 校验全部文件。"
            "本交付仅完成数据转换及参数完整性验证，未执行全量 UMR 重定向。\n"
        )
        hashes = {
            p.relative_to(package).as_posix(): sha256_file(p)
            for p in sorted(package.rglob("*"))
            if p.is_file()
        }
        (package / "SHA256SUMS").write_text(
            "".join(f"{digest}  {relative}\n" for relative, digest in sorted(hashes.items()))
        )
        hashes["SHA256SUMS"] = sha256_file(package / "SHA256SUMS")
        archive_path = staging / f"{package_name}.tar.gz"
        print(json.dumps({"stage": "compressing", "files": len(hashes)}), flush=True)
        compressor = create_archive(package, archive_path, hashes)
        print(json.dumps({"stage": "verifying_archive"}), flush=True)
        verified_files = verify_archive(archive_path, package_name, hashes)
        archive_sha = sha256_file(archive_path)
        result = {
            "status": "passed",
            "package_root": str(targets[0]),
            "archive": str(targets[1]),
            "archive_bytes": archive_path.stat().st_size,
            "archive_sha256": archive_sha,
            "archive_verified_files": verified_files,
            "compressor": compressor,
            "total_samples": len(rows),
            "total_frames": contract["total_frames"],
            **totals,
        }
        (staging / targets[2].name).write_text(f"{archive_sha}  {targets[1].name}\n")
        write_json(staging / targets[3].name, result)
        # 所有验证完成后再发布；不覆盖已有目录、归档或旁侧清单。
        for target in targets:
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"target appeared during packaging: {target}")
        for original, target in zip(
            [package, archive_path, staging / targets[2].name, staging / targets[3].name], targets
        ):
            os.rename(original, target)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-parent", type=Path, required=True)
    parser.add_argument("--package-name", default="music_smplx_4set_7286_30hz_zup_v1")
    args = parser.parse_args()
    print(
        json.dumps(
            package_dataset(args.source_root, args.output_parent, args.package_name),
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
