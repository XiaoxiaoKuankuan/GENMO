#!/usr/bin/env python3
"""构建 KIT-ML → AMASS 的可追溯文本动作目录及 GENMO 30 Hz 源数据。

读取 KIT 官方 ZIP 或解压目录，保留全部 ID、原始文本和来源 metadata；使用用户指定的
Mathux/AMASS-Annotation-Unifier 映射，不通过相似文件名猜测动作。缺文件、无映射、
无文本、MMM/AMASS 原始帧数或时长不一致均进入报告，不会被当成训练样本。

动作到齐后，复用 GENMO 已有的轴角 SLERP/平移线性重采样函数，保留完整的 156 维
SMPL+H 姿态、形状、性别和可选 DMPL；沿用 AMASS 原世界坐标，不做地面偏移或轴变换。
输出帧数为 round(T * 30 / source_fps)，时间网格严格为 k/30，超出源末帧的不足一帧
区间保持末帧。文本是整段标注，原始与目标时长分别保存，避免把量化误差当成新标注。

也可读取 GENMO 已有的 smplxpose_v2.pth：仅接受明确标记 model=smplx 的 66 维身体
姿态，按 AMASS 子集/被试/完整动作名精确对应 stageii 与 poses 名称，不补造手脸姿态。
该来源按现有 AmassDataset 的 30 Hz 契约解释，并以 MMM 原始时长交叉检查；它不包含
原始帧率，不能再做原始帧数相等的判断，允许的时长差仅一个源帧加一个 30 Hz 帧。

metadata.json 包含所有原始 ID（未完成动作的 end 为 null），metadata_ready.json
只包含已验证并导出动作的条目。这是重定向前的人体源数据，不冒充完整 SMPL-X 拟合、
BUMI qpos 数据、T5 embedding 或可以直接喂给现有训练 Dataset 的成品。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.data.music_dance.aioz.common import (  # noqa: E402
    _resample_axis_angle,
    _resample_linear,
)

UNIFIER_URL = "https://github.com/Mathux/AMASS-Annotation-Unifier"
TARGET_FPS = 30.0


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    value = PurePosixPath(relative)
    if value.is_absolute() or ".." in value.parts or "\\" in relative:
        raise ValueError(f"非法相对路径: {relative}")
    result = root.joinpath(*value.parts)
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"路径越界: {relative}")
    return result


class KitSource:
    def __init__(self, root: Path):
        self.root = root
        self.archive = zipfile.ZipFile(root) if root.is_file() else None
        names = (
            self.archive.namelist()
            if self.archive
            else (str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
        )
        self.files = {}
        for name in names:
            base = PurePosixPath(name).name
            if not base.endswith(("_annotations.json", "_meta.json", "_mmm.xml")):
                continue
            if base in self.files:
                raise ValueError(f"KIT 源文件 basename 重复: {base}")
            self.files[base] = name

    def close(self):
        if self.archive:
            self.archive.close()

    def read(self, name: str) -> bytes:
        full = self.files[name]
        return self.archive.read(full) if self.archive else (self.root / full).read_bytes()

    def ids(self) -> list[str]:
        ids = sorted(
            name.removesuffix("_meta.json") for name in self.files if name.endswith("_meta.json")
        )
        if not ids or any(not value.isdigit() for value in ids):
            raise ValueError("KIT 源目录没有有效的数字 ID")
        for value in ids:
            for suffix in ("_annotations.json", "_mmm.xml"):
                if value + suffix not in self.files:
                    raise ValueError(f"KIT 源文件缺失: {value + suffix}")
        return ids

    def mmm_timing(self, key: str) -> dict:
        tree = ET.fromstring(self.read(key + "_mmm.xml"))
        motions = tree.findall("Motion")
        if len(motions) != 1:
            raise ValueError("MMM 必须恰有一个 Motion")
        times = np.array(
            [float(x.text) for x in motions[0].findall("MotionFrames/MotionFrame/Timestep")]
        )
        if len(times) < 2 or not np.isfinite(times).all():
            raise ValueError("MMM 时间戳数量不足或不有限")
        delta = np.diff(times)
        step = float(np.median(delta))
        if step <= 0 or not np.allclose(delta, step, atol=1e-5, rtol=1e-3):
            raise ValueError("MMM 时间戳不是严格递增的均匀采样")
        return {
            "frames": len(times),
            "fps": 1.0 / step,
            "start_time": float(times[0]),
            "last_timestamp": float(times[-1]),
            "duration": float(times[-1] - times[0] + step),
        }


def resolve_mapping(key: str, published: dict, original: dict, amass_root: Path, available=None):
    def exists(relative):
        path = safe_path(amass_root, relative)
        return relative in available if available is not None else path.is_file()

    if key in published:
        relative = published[key]
        return relative, "unifier_published", "available" if exists(relative) else "missing_amass"
    if key not in original:
        return None, "none", "unmapped"
    item = original[key]
    order = {"kit": ("KIT", "CMU", "EKUT"), "cmu": ("CMU", "EKUT", "KIT")}[item["identifier"]]
    candidates = [f"{subset}/{item['path']}" for subset in order]
    found = [p for p in candidates if exists(p)]
    if len(found) > 1:
        return None, "unifier_exact_candidate", "ambiguous_mapping"
    if found:
        return found[0], "unifier_exact_candidate", "available"
    return candidates[0], "unifier_missing_candidate", "unmapped"


def validate_motion(data: dict) -> dict:
    poses, trans = data["poses"], data["trans"]
    fps = float(np.asarray(data["mocap_framerate"]).item())
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("AMASS mocap_framerate 非法")
    pose_dim = 66 if str(data.get("source_model_type", "smplh")) == "smplx" else 156
    if poses.ndim != 2 or poses.shape[1] != pose_dim or len(poses) < 2:
        raise ValueError(f"要求明确模型身份的 poses[T,{pose_dim}] 且 T>=2，实际 {poses.shape}")
    if trans.shape != (len(poses), 3):
        raise ValueError("AMASS trans 与 poses 帧数/维度不一致")
    if data["betas"].ndim != 1 or not data["betas"].size:
        raise ValueError("AMASS betas 必须为非空静态向量")
    gender = np.asarray(data["gender"]).item()
    if isinstance(gender, bytes):
        gender = gender.decode("utf-8")
    if gender not in ("male", "female", "neutral"):
        raise ValueError(f"AMASS gender 非法: {gender}")
    for key, value in data.items():
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError(f"AMASS {key} 含 NaN/Inf")
        if (
            key not in ("poses", "trans", "dmpls", "betas", "missing_parameters")
            and value.ndim
            and value.shape[0] == len(poses)
        ):
            raise ValueError(f"未定义的逐帧字段，不能静默重采样: {key}")
    if "dmpls" in data and (data["dmpls"].ndim != 2 or data["dmpls"].shape[0] != len(poses)):
        raise ValueError("AMASS dmpls 帧数/维度不一致")
    return data


def load_motion(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return validate_motion({key: archive[key] for key in archive.files})


class GenmoSource:
    """读取已有 30 Hz SMPL-X 身体参数，禁止模糊 basename 匹配。"""

    def __init__(self, path: Path):
        self.data = torch.load(path, map_location="cpu", weights_only=False)
        self.sha256 = file_sha256(path)
        self.index = {}
        for key, record in self.data.items():
            parts = PurePosixPath(key).parts
            positions = [i for i, part in enumerate(parts) if part in ("KIT", "CMU", "EKUT")]
            if not positions:
                continue
            tail = parts[positions[-1] :]
            if len(tail) != 3 or not tail[-1].endswith("_stageii.npz"):
                raise ValueError(f"不支持的 GENMO 来源路径结构: {key}")
            relative = str(PurePosixPath(*tail)).removesuffix("_stageii.npz") + "_poses.npz"
            if relative in self.index:
                raise ValueError(f"GENMO 来源映射碰撞: {relative}")
            if record.get("file_name") != key:
                raise ValueError(f"GENMO 内嵌 file_name 与字典键不一致: {key}")
            self.index[relative] = key

    def load(self, relative: str) -> dict:
        key = self.index[relative]
        record = self.data[key]
        if record.get("model") != "smplx":
            raise ValueError(f"GENMO 模型身份不是 smplx: {key}")
        data = {
            target: record[original].detach().cpu().numpy().copy()
            for target, original in (("poses", "pose"), ("trans", "trans"), ("betas", "beta"))
        }
        data.update(
            gender=np.array(record["gender"]),
            mocap_framerate=np.array(30.0),
            source_model_type=np.array("smplx"),
            source_key=np.array(key),
            source_pose_components=np.array("global_orient3+body_pose63"),
            full_pose_available=np.array(False),
            missing_parameters=np.array(
                [
                    "left_hand_pose",
                    "right_hand_pose",
                    "jaw_pose",
                    "leye_pose",
                    "reye_pose",
                    "expression",
                ]
            ),
        )
        return validate_motion(data)


def resample_motion(data: dict) -> dict:
    fps = float(data["mocap_framerate"])
    frames = len(data["poses"])
    result = {key: value.copy() for key, value in data.items()}
    if not math.isclose(fps, TARGET_FPS, rel_tol=0, abs_tol=1e-9):
        dimensions = data["poses"].shape[1]
        result["poses"] = _resample_axis_angle(
            data["poses"].reshape(frames, dimensions // 3, 3), fps, TARGET_FPS
        ).reshape(-1, dimensions)
        for key in ("trans", "dmpls"):
            if key in data:
                result[key] = _resample_linear(data[key], fps, TARGET_FPS)
    result.update(
        mocap_framerate=np.array(TARGET_FPS),
        source_mocap_framerate=np.array(fps),
        source_num_frames=np.array(frames),
        source_model_type=data.get("source_model_type", np.array("smplh")),
        coordinate_transform=np.array("identity"),
    )
    return result


def write_motion(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **data)
        checked = load_motion(temporary)
        if float(checked["mocap_framerate"]) != TARGET_FPS:
            raise ValueError("重载时帧率不是 30 Hz")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build(args) -> dict:
    source = KitSource(args.kitml_root)
    mapping_path = args.mapping_root / "amass-path2kitml.json"
    original_path = args.mapping_root / "kitml_amass_path.json"
    published = json.loads(mapping_path.read_text())
    original = json.loads(original_path.read_text())
    genmo_path = getattr(args, "amass_genmo_file", None)
    genmo = GenmoSource(genmo_path) if genmo_path else None
    amass_root = args.amass_root or genmo_path.parent
    rows = []
    try:
        for index, key in enumerate(source.ids()):
            texts = json.loads(source.read(key + "_annotations.json"))
            meta = json.loads(source.read(key + "_meta.json"))
            if not isinstance(texts, list) or any(
                not isinstance(x, str) or not x.strip() for x in texts
            ):
                raise ValueError(f"{key}: 文本不是有效字符串列表")
            if meta["nb_annotations"] != len(texts):
                raise ValueError(f"{key}: 文本数量与官方 metadata 不一致")
            relative, method, status = resolve_mapping(
                key, published, original, amass_root, genmo.index if genmo else None
            )
            row = {
                "motion_id": f"kitml_{key}",
                "kitml_id": key,
                "amass_path": relative,
                "texts": texts,
                "start": 0.0,
                "end": None,
                "fps": TARGET_FPS,
                "motion_path": None,
                "mapping_method": method,
                "status": status,
                "kitml_metadata": meta,
                "source_model_type": "smplx" if genmo else "smplh",
                "coordinate_transform": "identity",
                "annotation_scope": "whole_motion",
            }
            try:
                timing = source.mmm_timing(key)
                row["mmm"] = timing
                if status == "available":
                    path = safe_path(amass_root, relative)
                    data = genmo.load(relative) if genmo else load_motion(path)
                    frames = len(data["poses"])
                    fps = float(data["mocap_framerate"])
                    duration = frames / fps
                    row.update(
                        source_fps=fps,
                        source_num_frames=frames,
                        source_start_time=0.0,
                        source_end_time=duration,
                        source_sha256=genmo.sha256 if genmo else file_sha256(path),
                        alignment_duration_delta_sec=duration - timing["duration"],
                    )
                    if genmo:
                        row.update(
                            source_container=str(genmo_path.resolve()),
                            source_key=genmo.index[relative],
                            source_fps_evidence="GENMO AmassDataset fixed 30 Hz contract, cross-checked with MMM timestamps",
                            original_amass_fps=None,
                            full_pose_available=False,
                            missing_parameters=data["missing_parameters"].tolist(),
                            alignment_method="exact_subset_subject_clip_and_duration",
                        )
                        mismatch = (
                            abs(duration - timing["duration"]) > 1 / fps + 1 / timing["fps"] + 1e-6
                        )
                    else:
                        row["alignment_method"] = "exact_path_framecount_and_duration"
                        mismatch = (
                            frames != timing["frames"]
                            or abs(duration - timing["duration"])
                            > max(1 / fps, 1 / timing["fps"]) + 1e-6
                        )
                    if mismatch:
                        row["status"] = "alignment_mismatch"
                    elif not texts:
                        row["status"] = "no_text"
                    else:
                        motion = resample_motion(data)
                        target = f"motions_30hz/{row['motion_id']}_poses.npz"
                        write_motion(args.output_root / target, motion)
                        count = len(motion["poses"])
                        row.update(
                            status="ready",
                            motion_path=target,
                            num_frames=count,
                            end=count / TARGET_FPS,
                            duration_rounding_sec=count / TARGET_FPS - duration,
                            last_sample_hold_sec=max(
                                0.0, (count - 1) / TARGET_FPS - (frames - 1) / fps
                            ),
                        )
                elif not texts:
                    row["status"] = "no_text"
            except (ValueError, KeyError, OSError, ET.ParseError, zipfile.BadZipFile) as exc:
                row.update(status="invalid_source", error=str(exc))
            row["annotations"] = [
                {
                    "annotation_id": f"kitml_{key}_{i}",
                    "caption": text,
                    "start_time": 0.0,
                    "end_time": row["end"],
                    "source_start_time": 0.0,
                    "source_end_time": row.get("source_end_time"),
                }
                for i, text in enumerate(texts)
            ]
            rows.append(row)
            if (index + 1) % 250 == 0:
                print(f"已审计 {index + 1} 条 KIT-ML", flush=True)
    finally:
        source.close()
    ready = [row for row in rows if row["status"] == "ready"]
    groups = defaultdict(list)
    for row in rows:
        if row["amass_path"]:
            groups[row["amass_path"]].append(row["motion_id"])
    report = {
        "schema_version": 1,
        "stage": "source_prepared" if ready else "awaiting_amass",
        "kitml_root": str(args.kitml_root.resolve()),
        "amass_root": str(amass_root.resolve()),
        "amass_genmo_file": str(genmo_path.resolve()) if genmo else None,
        "source_container_sha256": genmo.sha256 if genmo else None,
        "output_root": str(args.output_root.resolve()),
        "unifier_url": UNIFIER_URL,
        "unifier_commit": args.unifier_commit,
        "mapping_sha256": file_sha256(mapping_path),
        "original_mapping_sha256": file_sha256(original_path),
        "total_motions": len(rows),
        "total_captions": sum(len(row["texts"]) for row in rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "published_mapping_counts": dict(Counter(p.split("/")[0] for p in published.values())),
        "ready_motions": len(ready),
        "ready_captions": sum(len(row["texts"]) for row in ready),
        "ready_duration_hours": sum(row["end"] for row in ready) / 3600,
        "same_source_motion_groups": {
            key: value for key, value in groups.items() if len(value) > 1
        },
        "training_ready": False,
        "boundary": "Human motion source only; requires retargeting/model adapter and training packaging; never split identical amass_path across train/val/test",
    }
    write_json(args.output_root / "metadata.json", rows)
    write_json(args.output_root / "metadata_ready.json", ready)
    write_json(args.report_root / "summary.json", report)
    write_json(
        args.report_root / "not_ready.json",
        [
            {
                "motion_id": r["motion_id"],
                "amass_path": r["amass_path"],
                "status": r["status"],
                "error": r.get("error"),
            }
            for r in rows
            if r["status"] != "ready"
        ],
    )
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kitml-root", type=Path, required=True, help="官方 ZIP 或解压目录")
    parser.add_argument(
        "--mapping-root", type=Path, required=True, help="Unifier 的 kitml_process 目录"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--amass-root", type=Path, help="包含 KIT/、CMU/、EKUT/ 的 SMPL+H G 根目录")
    source.add_argument(
        "--amass-genmo-file",
        type=Path,
        help="用户已有的可信 GENMO smplxpose_v2.pth，30 Hz SMPL-X 身体参数",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--unifier-commit", required=True)
    args = parser.parse_args(argv)
    report = build(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
