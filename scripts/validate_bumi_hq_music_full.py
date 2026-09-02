#!/usr/bin/env python3
"""批量验证四个公开舞蹈库和可选自建数据库的高质量音乐生成效果。

本工具把人工评分、长音乐生成、BUMI 世界系轨迹、MuJoCo 视频和质量审计串成一个可恢复的
离线流程。公开四库只选择评分 CSV 中 ``score=1`` 的动作，再按数据集规则映射到音频并去重；
自建库可选读取正式发布 manifest 中 ``quality_accepted=true`` 的 30 Hz BUMI 轨迹与裁齐音频；
每首音频保留全部对应的高质量动作名称，方便从页面和 JSON 追溯选择依据。各数据集可使用
不同的抽样数量；生成既支持完整音乐，也支持为网页验收设置统一的前缀时长。生成阶段复用
一次固定 120 帧的 BUMI ONNX 会话，通过 120/30 独立预测、世界对齐和几何感知
overlap-add 输出连续世界系 qpos28。

质量报告同时给出严格 XML 原始限位和部署容差后的限位结果。每个超限关节都记录 0/1 基编号、
名称、方向、最大超限弧度、发生帧、实际值、XML 边界和超限帧数；另外保留脚部穿地、根节点
倾斜、速度和节拍对齐等运动学指标。最终 ``index.html`` 按 FineDance、CoMPAS3D、
AIOZ-GDance、AIST++、自建数据库分组展示带声音视频及逐关节统计。该评测不执行 GMT，也不代表真实机器人
动力学可跟踪性。中断后再次运行会复用身份一致且已通过媒体校验的正式结果；若推理产物已经
原子落盘、但渲染阶段因外部资源缺失而失败，也会复用身份一致的动作产物，只补做指标、渲染
与报告，避免重复执行昂贵的扩散推理。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gem.robots.bumi.endecoder import BumiEndecoder  # noqa: E402
from gem.robots.bumi.metrics import (  # noqa: E402
    compute_bumi_kinematic_metrics,
    metrics_to_json,
)
from gem.robots.bumi.postprocess import BUMI_FOOT_LOCK_CONTRACT_VERSION  # noqa: E402
from gem.runtime.bumi_music_deploy import (  # noqa: E402
    BUMI_SLIDING_QPOS_CONTRACT_VERSION,
    BumiOrtStepRunner,
    BumiSlidingQposGenerator,
)
from gem.runtime.bumi_music_onnx import BUMI_ONNX_CONTRACT_VERSION  # noqa: E402
from gem.runtime.music_only_trt import plan_sliding_windows  # noqa: E402
from gem.utils.music_features import extract_edge_baseline35  # noqa: E402

DATASETS = (
    ("finedance", "FineDance", "bumi3_finedance_ratings.csv"),
    ("compas3d", "CoMPAS3D", "bumi3_compas3d_ratings.csv"),
    ("aioz_gdance", "AIOZ-GDance", "bumi3_aioz_gdance.csv"),
    ("aistpp", "AIST++", "bumi3_aistpp_ratings.csv"),
)
MINE_DATASET = ("mine_bumi", "自建数据库", None)
REPORT_DATASETS = DATASETS + (MINE_DATASET,)
DATASET_LABELS = {key: label for key, label, _ in DATASETS}
DATASET_LABELS[MINE_DATASET[0]] = MINE_DATASET[1]
RATING_COLUMNS = ("motion_name", "score")


def sha256_file(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def audio_key_for_motion(dataset: str, motion_name: str) -> str:
    """把四库动作文件名严格映射到共享音频键。"""

    if Path(motion_name).name != motion_name or not motion_name.endswith(".npz"):
        raise ValueError(f"{dataset}: 非法动作文件名 {motion_name!r}")
    if dataset == "finedance":
        return Path(motion_name).stem
    if dataset == "compas3d":
        match = re.fullmatch(r"(.+)_(?:leader|follower)\.npz", motion_name)
    elif dataset == "aioz_gdance":
        match = re.fullmatch(r"(.+)_dancer_\d+\.npz", motion_name)
    elif dataset == "aistpp":
        match = re.search(r"_(m[A-Z]{2}\d)_", motion_name)
    else:
        raise ValueError(f"不支持的数据集：{dataset}")
    if match is None:
        raise ValueError(f"{dataset}: 无法从动作名映射音频 {motion_name!r}")
    return match.group(1)


def audio_path_for_key(audio_root: Path, dataset: str, audio_key: str) -> Path:
    relative = (
        Path("aistpp") / "wav" / f"{audio_key}.wav"
        if dataset == "aistpp"
        else Path(dataset) / f"{audio_key}.wav"
    )
    return (audio_root / relative).resolve()


def select_hq_audio(
    ratings_root: Path,
    audio_root: Path,
    *,
    per_dataset: int | dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """读取 score=1，按音频去重并返回每库指定数量的可追溯选择。"""

    dataset_keys = {dataset for dataset, _, _ in DATASETS}
    if isinstance(per_dataset, int):
        if per_dataset <= 0:
            raise ValueError("per_dataset 必须为正整数")
        limits = dict.fromkeys(dataset_keys, per_dataset)
    elif isinstance(per_dataset, dict):
        if set(per_dataset) != dataset_keys:
            raise ValueError(f"per_dataset 必须精确覆盖 {sorted(dataset_keys)}")
        limits = {key: int(value) for key, value in per_dataset.items()}
        if any(value <= 0 for value in limits.values()):
            raise ValueError("各数据集选择数量必须为正整数")
    else:
        raise TypeError("per_dataset 必须为整数或数据集到整数的映射")
    selected: list[dict[str, Any]] = []
    dataset_summary: dict[str, Any] = {}
    for dataset, label, filename in DATASETS:
        path = (ratings_root / filename).resolve(strict=True)
        grouped: dict[str, list[str]] = defaultdict(list)
        total_rows = 0
        high_quality_rows = 0
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != RATING_COLUMNS:
                raise ValueError(
                    f"{path}: CSV 列必须严格为 {RATING_COLUMNS}，实际为 {reader.fieldnames}"
                )
            seen: set[str] = set()
            for line_number, row in enumerate(reader, start=2):
                total_rows += 1
                motion_name = str(row["motion_name"]).strip()
                if motion_name in seen:
                    raise ValueError(f"{path}:{line_number}: 动作名重复 {motion_name!r}")
                seen.add(motion_name)
                try:
                    score = int(str(row["score"]).strip())
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_number}: score 不是整数") from exc
                if score != 1:
                    continue
                high_quality_rows += 1
                grouped[audio_key_for_motion(dataset, motion_name)].append(motion_name)

        available: list[dict[str, Any]] = []
        missing: list[str] = []
        for audio_key in sorted(grouped):
            audio_path = audio_path_for_key(audio_root, dataset, audio_key)
            if not audio_path.is_file():
                missing.append(str(audio_path))
                continue
            names = sorted(grouped[audio_key])
            available.append(
                {
                    "dataset": dataset,
                    "dataset_label": label,
                    "audio_key": audio_key,
                    "audio": str(audio_path),
                    "representative_motion": names[0],
                    "high_quality_motion_count": len(names),
                    "high_quality_motion_names": names,
                }
            )
        requested_count = limits[dataset]
        chosen_count = min(requested_count, len(available))
        if chosen_count <= 1:
            chosen = available[:chosen_count]
        else:
            # 均匀覆盖完整音频键列表，避免字典序前十首集中在同一舞种或同一 Pair。
            positions = [
                round(index * (len(available) - 1) / (chosen_count - 1))
                for index in range(chosen_count)
            ]
            chosen = [available[position] for position in positions]
        selected.extend(chosen)
        dataset_summary[dataset] = {
            "dataset_label": label,
            "rating_rows": total_rows,
            "score_1_action_count": high_quality_rows,
            "distinct_score_1_audio_count": len(grouped),
            "available_score_1_audio_count": len(available),
            "missing_audio_count": len(missing),
            "missing_audio_examples": missing[:10],
            "requested_audio_count": requested_count,
            "selected_audio_count": len(chosen),
        }
    return selected, dataset_summary


def select_mine_bumi(
    manifest_path: Path,
    *,
    count: int,
    audio_keys: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """从正式自建库 manifest 的完整质量通过全集中均匀选择，并校验所选本地资产。

    manifest 是正式数据库的选择权威；本地只需缓存本次选中的音频和动作，因此选择位置必须
    先在全部 ``quality_accepted=true`` 条目上确定，再检查所选文件是否存在，不能先按本地
    已下载文件过滤后再抽样。这样在工作站只传回 5 个样本时，选择仍与服务器 99 条正式库
    的固定 manifest 行序全集一致，也不会把临时缓存误写成数据库总量。
    """

    if count <= 0:
        raise ValueError("mine_bumi count 必须为正整数")
    manifest_path = manifest_path.expanduser().resolve(strict=True)
    dataset_root = manifest_path.parent.parent.resolve(strict=True)
    rows = []
    seen: set[str] = set()
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("dataset") != "mine_bumi" or row.get("quality_accepted") is not True:
            continue
        audio_key = str(row.get("audio_key", ""))
        if not audio_key or audio_key in seen:
            raise ValueError(f"{manifest_path}:{line_number}: 自建库 audio_key 为空或重复")
        seen.add(audio_key)
        rows.append(row)
    if not rows:
        raise ValueError("自建库 manifest 没有 quality_accepted=true 条目")
    if audio_keys:
        if len(audio_keys) != count or len(set(audio_keys)) != len(audio_keys):
            raise ValueError("显式 mine_bumi audio key 必须去重且数量等于 mine-count")
        positions_by_key = {str(row["audio_key"]): index for index, row in enumerate(rows)}
        missing_keys = [key for key in audio_keys if key not in positions_by_key]
        if missing_keys:
            raise ValueError(f"显式 mine_bumi audio key 不在正式 manifest：{missing_keys}")
        positions = [positions_by_key[key] for key in audio_keys]
        selection_mode = "explicit_vetted_audio_keys"
    else:
        chosen_count = min(count, len(rows))
        positions = (
            [0]
            if chosen_count == 1
            else [
                int(index * (len(rows) - 1) / (chosen_count - 1)) for index in range(chosen_count)
            ]
        )
        selection_mode = "uniform_manifest_order"
    selected = []
    for position in positions:
        row = rows[position]
        audio_path = (dataset_root / str(row["audio_path"])).resolve()
        motion_path = (dataset_root / str(row["motion_path"])).resolve()
        if not audio_path.is_relative_to(dataset_root) or not motion_path.is_relative_to(
            dataset_root
        ):
            raise ValueError("自建库 manifest 路径越出数据根")
        if not audio_path.is_file() or not motion_path.is_file():
            raise FileNotFoundError(f"自建库所选正式资产未传回本地：{audio_path} / {motion_path}")
        selected.append(
            {
                "dataset": "mine_bumi",
                "dataset_label": MINE_DATASET[1],
                "audio_key": str(row["audio_key"]),
                "audio": str(audio_path),
                "representative_motion": motion_path.name,
                "source_motion": str(motion_path),
                "high_quality_motion_count": 1,
                "high_quality_motion_names": [motion_path.name],
                "selection_manifest_row": position,
                "source_part": row.get("source_part"),
                "song_name": row.get("song_name"),
            }
        )
    return selected, {
        "dataset_label": MINE_DATASET[1],
        "manifest_rows": len(rows),
        "quality_accepted_audio_count": len(rows),
        "requested_audio_count": count,
        "selected_audio_count": len(selected),
        "selection_positions_0based": positions,
        "selection_mode": selection_mode,
        "selected_local_assets_present": len(selected),
    }


def load_explicit_selection(
    manifest_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, int], str]:
    """加载已经人工冻结且绑定资产 SHA256 的跨数据集评测清单。

    显式清单用于某次训练绑定了新版正式 manifest、而旧评分表已不能代表实际数据契约的
    情况。清单中的相对路径必须留在清单目录内；音频与原动作都必须存在，并在提供期望
    SHA256 时逐文件核对。这样生成网页不会把旧数据版本的同名音频或动作误标成当前模型的
    验证样本，也允许调用方明确区分 held-out 样本和域外自建样本。
    """

    manifest_path = manifest_path.expanduser().resolve(strict=True)
    root = manifest_path.parent.resolve(strict=True)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("contract_version") != "genmo.bumi_hq_explicit_selection.v1":
        raise ValueError("显式选择清单 contract_version 不受支持")
    raw_items = payload.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("显式选择清单必须包含非空 items")

    allowed = tuple(dataset for dataset, _, _ in REPORT_DATASETS)
    allowed_set = set(allowed)
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    counts = {dataset: 0 for dataset in allowed}
    for index, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            raise ValueError(f"显式选择清单 items[{index}] 不是对象")
        dataset = str(raw.get("dataset", ""))
        audio_key = str(raw.get("audio_key", ""))
        if (
            dataset not in allowed_set
            or audio_key in {"", ".", ".."}
            or Path(audio_key).name != audio_key
        ):
            raise ValueError(f"显式选择清单 items[{index}] 数据集或 audio_key 非法")
        identity = (dataset, audio_key)
        if identity in seen:
            raise ValueError(f"显式选择清单存在重复样本：{identity}")
        seen.add(identity)

        resolved_paths: dict[str, Path] = {}
        for field in ("audio", "source_motion"):
            value = raw.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"显式选择清单 items[{index}] 缺少 {field}")
            candidate = Path(value).expanduser()
            candidate = (
                (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
            )
            if not candidate.is_file():
                raise FileNotFoundError(f"显式选择资产不存在：{candidate}")
            if not Path(value).is_absolute() and not candidate.is_relative_to(root):
                raise ValueError(f"显式选择相对路径越出清单目录：{value}")
            resolved_paths[field] = candidate

        expected_hashes = {
            "audio": raw.get("audio_sha256"),
            "source_motion": raw.get("source_motion_sha256"),
        }
        actual_hashes = {field: sha256_file(path) for field, path in resolved_paths.items()}
        for field, expected_sha256 in expected_hashes.items():
            if expected_sha256 is not None and str(expected_sha256) != actual_hashes[field]:
                raise ValueError(f"显式选择资产 SHA256 不匹配：{dataset}/{audio_key}/{field}")

        row = dict(raw)
        row.update(
            {
                "dataset": dataset,
                "dataset_label": DATASET_LABELS[dataset],
                "audio_key": audio_key,
                "audio": str(resolved_paths["audio"]),
                "source_motion": str(resolved_paths["source_motion"]),
                "representative_motion": str(
                    raw.get("representative_motion") or resolved_paths["source_motion"].name
                ),
                "high_quality_motion_count": int(raw.get("high_quality_motion_count", 1)),
                "high_quality_motion_names": list(
                    raw.get("high_quality_motion_names") or [resolved_paths["source_motion"].name]
                ),
                "audio_sha256": actual_hashes["audio"],
                "source_motion_sha256": actual_hashes["source_motion"],
                "selection_manifest_row": index,
            }
        )
        selected.append(row)
        counts[dataset] += 1

    requested = payload.get("per_dataset_limits")
    observed = {key: value for key, value in counts.items() if value}
    if requested is not None:
        requested = {str(key): int(value) for key, value in dict(requested).items()}
        if requested != observed:
            raise ValueError(f"显式选择清单配额不匹配：requested={requested}, observed={observed}")
    summary = {
        dataset: {
            "dataset_label": DATASET_LABELS[dataset],
            "requested_audio_count": count,
            "selected_audio_count": count,
            "selection_source": "explicit_sha256_bound_manifest",
        }
        for dataset, count in observed.items()
    }
    policy = str(
        payload.get("selection_policy")
        or "显式 SHA256 绑定清单，调用方负责记录数据 split 与选择依据"
    )
    return selected, summary, observed, policy


def truncate_music_features(
    features: torch.Tensor,
    metadata: dict[str, Any],
    *,
    max_duration_sec: float | None,
) -> tuple[torch.Tensor, dict[str, Any], float]:
    """按网页验证时长截取 EDGE35，同时保留原音乐时长和更新后的特征契约。"""

    if features.ndim != 2 or features.shape[1] != 35 or not bool(torch.isfinite(features).all()):
        raise ValueError(f"音乐特征必须是 finite [T,35]，实际为 {tuple(features.shape)}")
    updated = dict(metadata)
    original_duration_sec = float(updated["selected_duration_sec"])
    if max_duration_sec is None:
        return features, updated, original_duration_sec
    if not math.isfinite(max_duration_sec) or max_duration_sec < 4.0:
        raise ValueError("max_duration_sec 必须为空或至少为 4 秒的有限数")
    target_frames = min(len(features), round(max_duration_sec * 30.0))
    if target_frames < 120:
        raise ValueError("截取后的音乐特征少于固定 ONNX 窗口 120 帧")
    clipped = features[:target_frames].contiguous()
    updated.update(
        {
            "original_selected_duration_sec": original_duration_sec,
            "selected_duration_sec": len(clipped) / 30.0,
            "feature_frames": len(clipped),
            "validation_max_duration_sec": max_duration_sec,
        }
    )
    return clipped, updated, original_duration_sec


def analyze_joint_limits(
    qpos: torch.Tensor | np.ndarray,
    kinematics: Any,
    *,
    tolerance_rad: float,
) -> dict[str, Any]:
    """逐关节审计严格 XML 超限和部署容差后的超限。"""

    if not math.isfinite(tolerance_rad) or tolerance_rad < 0.0:
        raise ValueError("tolerance_rad 必须是有限非负数")
    values = np.asarray(torch.as_tensor(qpos).detach().cpu(), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 28 or len(values) <= 0:
        raise ValueError(f"qpos 必须为 [T,28]，实际为 {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("qpos 包含 NaN/Inf")
    lower = np.asarray(kinematics.joint_lower_limits.detach().cpu(), dtype=np.float64)
    upper = np.asarray(kinematics.joint_upper_limits.detach().cpu(), dtype=np.float64)
    names = tuple(map(str, kinematics.joint_order))
    joints = values[:, 7:]
    lower_excess = lower[None, :] - joints
    upper_excess = joints - upper[None, :]
    raw_excess = np.maximum(np.maximum(lower_excess, upper_excess), 0.0)
    tolerated_excess = np.maximum(raw_excess - float(tolerance_rad), 0.0)
    details = []
    for joint_index, joint_name in enumerate(names):
        column = raw_excess[:, joint_index]
        if float(column.max(initial=0.0)) <= 0.0:
            continue
        frame = int(np.argmax(column))
        lower_side = lower_excess[frame, joint_index] >= upper_excess[frame, joint_index]
        side = "lower" if lower_side else "upper"
        bound = lower[joint_index] if lower_side else upper[joint_index]
        details.append(
            {
                "joint_index_0based": joint_index,
                "joint_number_1based": joint_index + 1,
                "joint_name": joint_name,
                "side": side,
                "max_excess_rad": float(column[frame]),
                "max_excess_frame_0based": frame,
                "value_rad": float(joints[frame, joint_index]),
                "xml_bound_rad": float(bound),
                "raw_violating_frames": int(np.count_nonzero(column > 0.0)),
                "raw_violating_frame_rate": float(np.mean(column > 0.0)),
                "max_excess_after_tolerance_rad": float(
                    tolerated_excess[:, joint_index].max(initial=0.0)
                ),
                "violating_frames_after_tolerance": int(
                    np.count_nonzero(tolerated_excess[:, joint_index] > 0.0)
                ),
            }
        )
    details.sort(key=lambda row: (-row["max_excess_rad"], row["joint_index_0based"]))
    return {
        "xml_source_sha256": str(kinematics.source_mjcf_sha256),
        "tolerance_rad": float(tolerance_rad),
        "strict_xml_limit_exceeded": bool(np.any(raw_excess > 0.0)),
        "strict_xml_violating_joint_count": len(details),
        "strict_xml_violating_frame_count": int(np.count_nonzero(np.any(raw_excess > 0.0, axis=1))),
        "strict_xml_max_excess_rad": float(raw_excess.max(initial=0.0)),
        "tolerance_limit_exceeded": bool(np.any(tolerated_excess > 0.0)),
        "tolerance_violating_joint_count": int(
            np.count_nonzero(np.any(tolerated_excess > 0.0, axis=0))
        ),
        "tolerance_violating_frame_count": int(
            np.count_nonzero(np.any(tolerated_excess > 0.0, axis=1))
        ),
        "max_excess_after_tolerance_rad": float(tolerated_excess.max(initial=0.0)),
        "joints": details,
    }


def media_probe(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,codec_name,width,height,r_frame_rate:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    video = [row for row in streams if row.get("codec_type") == "video"]
    audio = [row for row in streams if row.get("codec_type") == "audio"]
    if len(video) != 1 or not audio or video[0].get("r_frame_rate") != "30/1":
        raise RuntimeError(f"媒体契约校验失败：{path}")
    return {
        "duration_sec": float(payload["format"]["duration"]),
        "video_codec": video[0].get("codec_name"),
        "audio_codec": audio[0].get("codec_name"),
        "width": int(video[0]["width"]),
        "height": int(video[0]["height"]),
        "fps": 30,
    }


def render_and_mux(
    *, artifact: Path, audio: Path, mjcf: Path, output: Path, width: int, height: int
) -> dict[str, Any]:
    silent = output.with_name(output.stem + ".silent.mp4")
    temporary = output.with_name(output.stem + ".tmp.mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    silent.unlink(missing_ok=True)
    temporary.unlink(missing_ok=True)
    try:
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "tools/eval/render_bumi_motion.py"),
                "--motion",
                str(artifact),
                "--mjcf",
                str(mjcf),
                "--output",
                str(silent),
                "--width",
                str(width),
                "--height",
                str(height),
            ],
            check=True,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(silent),
                "-i",
                str(audio),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                str(temporary),
            ],
            check=True,
        )
        probe = media_probe(temporary)
        temporary.replace(output)
        return probe
    finally:
        silent.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def build_summary(results: list[dict[str, Any]], *, tolerance_rad: float) -> dict[str, Any]:
    passed = [row for row in results if row.get("status") == "passed"]
    raw_samples = [row for row in passed if row["joint_limits"]["strict_xml_limit_exceeded"]]
    tolerance_samples = [row for row in passed if row["joint_limits"]["tolerance_limit_exceeded"]]
    aggregate: dict[tuple[int, str], dict[str, Any]] = {}
    for row in passed:
        for joint in row["joint_limits"]["joints"]:
            key = (joint["joint_index_0based"], joint["joint_name"])
            entry = aggregate.setdefault(
                key,
                {
                    "joint_index_0based": key[0],
                    "joint_number_1based": key[0] + 1,
                    "joint_name": key[1],
                    "strict_xml_exceed_sample_count": 0,
                    "tolerance_exceed_sample_count": 0,
                    "exceed_0_25_rad_sample_count": 0,
                    "max_excess_rad": -1.0,
                    "max_excess_sample": None,
                    "max_excess_frame_0based": None,
                },
            )
            entry["strict_xml_exceed_sample_count"] += 1
            if joint["max_excess_after_tolerance_rad"] > 0.0:
                entry["tolerance_exceed_sample_count"] += 1
            if joint["max_excess_rad"] > 0.25:
                entry["exceed_0_25_rad_sample_count"] += 1
            if joint["max_excess_rad"] > entry["max_excess_rad"]:
                entry["max_excess_rad"] = joint["max_excess_rad"]
                entry["max_excess_sample"] = f"{row['dataset']}/{row['audio_key']}"
                entry["max_excess_frame_0based"] = joint["max_excess_frame_0based"]

    def summarize_motion_quality(rows: list[dict[str, Any]]) -> dict[str, Any]:
        metric_keys = (
            "foot_penetration_max_m",
            "foot_penetration_mean_m",
            "root_tilt_max_rad",
            "joint_velocity_p95_radps",
            "root_angular_velocity_p95_radps",
            "beat_alignment_score",
        )
        metrics = {}
        for metric_key in metric_keys:
            values = [float(row["metrics"][metric_key]) for row in rows]
            if not values:
                metrics[metric_key] = {"mean": None, "max": None, "max_sample": None}
                continue
            max_index = int(np.argmax(values))
            metrics[metric_key] = {
                "mean": float(np.mean(values)),
                "max": values[max_index],
                "max_sample": f"{rows[max_index]['dataset']}/{rows[max_index]['audio_key']}",
            }
        return {
            "samples": len(rows),
            "total_frames": sum(int(row["frames"]) for row in rows),
            "total_duration_sec": sum(float(row["source_duration_sec"]) for row in rows),
            "foot_penetration_max_over_0_05m_samples": sum(
                float(row["metrics"]["foot_penetration_max_m"]) > 0.05 for row in rows
            ),
            "root_tilt_max_over_0_5rad_samples": sum(
                float(row["metrics"]["root_tilt_max_rad"]) > 0.5 for row in rows
            ),
            "metrics": metrics,
        }

    dataset_counts = {}
    for dataset, label, _ in REPORT_DATASETS:
        rows = [row for row in passed if row["dataset"] == dataset]
        dataset_counts[dataset] = {
            "dataset_label": label,
            "completed": len(rows),
            "strict_xml_exceed_samples": sum(
                bool(row["joint_limits"]["strict_xml_limit_exceeded"]) for row in rows
            ),
            "tolerance_exceed_samples": sum(
                bool(row["joint_limits"]["tolerance_limit_exceeded"]) for row in rows
            ),
            "motion_quality": summarize_motion_quality(rows),
        }
    tolerance_sensitivity = []
    for threshold in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25):
        count = sum(
            max(
                (joint["max_excess_rad"] for joint in row["joint_limits"]["joints"]),
                default=0.0,
            )
            > threshold
            for row in passed
        )
        tolerance_sensitivity.append(
            {"tolerance_rad": threshold, "exceed_sample_count": int(count)}
        )
    tolerance_exceed_details = []
    for row in passed:
        joints = []
        for joint in row["joint_limits"]["joints"]:
            if joint["max_excess_rad"] <= tolerance_rad:
                continue
            joints.append(
                {
                    key: joint[key]
                    for key in (
                        "joint_index_0based",
                        "joint_number_1based",
                        "joint_name",
                        "side",
                        "max_excess_rad",
                        "max_excess_after_tolerance_rad",
                        "max_excess_frame_0based",
                        "value_rad",
                        "xml_bound_rad",
                    )
                }
            )
        if joints:
            tolerance_exceed_details.append(
                {
                    "dataset": row["dataset"],
                    "audio_key": row["audio_key"],
                    "joints": joints,
                }
            )
    return {
        "contract_version": "genmo.bumi_hq_full_music_quality_summary.v1",
        "evaluated_samples": len(results),
        "completed_samples": len(passed),
        "failed_samples": len(results) - len(passed),
        "strict_xml_exceed_sample_count": len(raw_samples),
        "joint_limit_tolerance_rad": float(tolerance_rad),
        "tolerance_exceed_sample_count": len(tolerance_samples),
        "tolerance_sensitivity": tolerance_sensitivity,
        "tolerance_exceed_details": tolerance_exceed_details,
        "datasets": dataset_counts,
        "motion_quality": summarize_motion_quality(passed),
        "joint_aggregate": sorted(aggregate.values(), key=lambda row: row["joint_index_0based"]),
        "failed": [
            {"dataset": row["dataset"], "audio_key": row["audio_key"], "error": row.get("error")}
            for row in results
            if row.get("status") != "passed"
        ],
    }


def build_index(
    output_root: Path,
    results: list[dict[str, Any]],
    summary: dict[str, Any],
    selection: dict[str, Any],
) -> None:
    model_label = html.escape(str(selection["model"]["label"]))
    duration_label = (
        "完整音乐"
        if selection["generation"]["full_audio"]
        else f"前 {selection['generation']['max_duration_sec']:.1f} 秒"
    )
    root_postprocess = html.escape(
        str(selection["generation"].get("root_orientation_postprocess") or "无")
    )
    foot_lock_postprocess = html.escape(
        str(selection["generation"].get("foot_lock_postprocess") or "无")
    )
    cards = []
    for dataset, label, _ in REPORT_DATASETS:
        dataset_rows = [row for row in results if row["dataset"] == dataset]
        item_html = []
        for index, row in enumerate(dataset_rows, start=1):
            if row.get("status") != "passed":
                item_html.append(
                    f'<article class="card failed"><h3>{index}. {html.escape(row["audio_key"])}</h3>'
                    f"<p>生成失败：{html.escape(str(row.get('error', 'unknown')))}</p></article>"
                )
                continue
            limits = row["joint_limits"]
            badge = (
                "容差后超限"
                if limits["tolerance_limit_exceeded"]
                else f"{limits['tolerance_rad']:.2f}rad 容差内"
            )
            joint_rows = []
            for joint in limits["joints"]:
                joint_rows.append(
                    "<tr>"
                    f"<td>{joint['joint_number_1based']}</td>"
                    f"<td>{html.escape(joint['joint_name'])}</td>"
                    f"<td>{joint['side']}</td>"
                    f"<td>{joint['max_excess_rad']:.6f}</td>"
                    f"<td>{joint['max_excess_frame_0based']}</td>"
                    f"<td>{joint['raw_violating_frames']}</td>"
                    f"<td>{joint['max_excess_after_tolerance_rad']:.6f}</td>"
                    "</tr>"
                )
            if not joint_rows:
                joint_rows.append('<tr><td colspan="7">没有严格 XML 关节超限</td></tr>')
            video = html.escape(row["video_relative"])
            report = html.escape(row["report_relative"])
            representative = html.escape(row["representative_motion"])
            metrics = row["metrics"]
            item_html.append(
                f'<article class="card"><h3>{index}. {html.escape(row["audio_key"])} '
                f'<span class="badge">{badge}</span></h3>'
                f'<video controls preload="metadata" src="{video}"></video>'
                f"<p>高质量动作代表：<code>{representative}</code>；{duration_label} "
                f"{row['source_duration_sec']:.2f}s，生成 {row['frames']} 帧。</p>"
                f"<p>严格 XML 最大超限 {limits['strict_xml_max_excess_rad']:.6f} rad；"
                f"容差 {limits['tolerance_rad']:.3f} rad 后最大超限 "
                f"{limits['max_excess_after_tolerance_rad']:.6f} rad；脚部最大穿地 "
                f"{metrics['foot_penetration_max_m']:.4f}m；根节点最大倾斜 "
                f'{metrics["root_tilt_max_rad"]:.4f}rad。 <a href="{report}">JSON 报告</a></p>'
                '<details><summary>逐关节超限明细</summary><div class="table-wrap"><table>'
                "<thead><tr><th>编号(1-based)</th><th>关节名</th><th>方向</th>"
                "<th>最大超限(rad)</th><th>帧(0-based)</th><th>超限帧数</th>"
                "<th>容差后超限(rad)</th></tr></thead><tbody>"
                + "".join(joint_rows)
                + "</tbody></table></div></details></article>"
            )
        cards.append(f"<section><h2>{label}</h2>{''.join(item_html)}</section>")
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BUMI {model_label} 多库高质量音乐验证</title>
<style>
body{{margin:0;background:#10131a;color:#edf1f7;font:15px/1.55 system-ui,sans-serif}}main{{max-width:1500px;margin:auto;padding:28px}}
a{{color:#78c5ff}}code{{overflow-wrap:anywhere}}.summary,.card{{background:#191f2b;border:1px solid #30394a;border-radius:12px;padding:16px;margin:14px 0}}
section{{margin-top:34px}}video{{width:100%;max-height:620px;background:#000;border-radius:8px}}.badge{{font-size:12px;padding:3px 7px;background:#3b465b;border-radius:12px}}
.failed{{border-color:#a74d4d}}.table-wrap{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;margin-top:10px}}th,td{{padding:7px;border:1px solid #3a4354;text-align:left;white-space:nowrap}}
@media(min-width:1000px){{section{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}section h2{{grid-column:1/-1}}.card{{margin:0}}}}
</style></head><body><main>
<h1>BUMI {model_label} 多库人工高质量音乐验证</h1>
<div class="summary"><p>共评测 {summary["evaluated_samples"]} 首，完成 {summary["completed_samples"]} 首；严格 XML 超限样本 {summary["strict_xml_exceed_sample_count"]} 首，部署容差 {summary["joint_limit_tolerance_rad"]:.3f}rad 后仍超限 {summary["tolerance_exceed_sample_count"]} 首。</p>
<p>总音频 {summary["motion_quality"]["total_duration_sec"] / 60.0:.2f} 分钟、{summary["motion_quality"]["total_frames"]} 帧；脚部最大穿地超过 5cm 的样本 {summary["motion_quality"]["foot_penetration_max_over_0_05m_samples"]} 首，根节点最大倾斜超过 0.5rad 的样本 {summary["motion_quality"]["root_tilt_max_over_0_5rad_samples"]} 首。</p>
	<p>根姿态后处理：<code>{root_postprocess}</code>；FK 足底锁定：<code>{foot_lock_postprocess}</code>。页面播放轨迹只根据 contact head 修正 root XY 来降低脚滑；root Z、root 四元数和全部关节保持模型原值，artifact 同时保存 <code>qpos_raw</code>、contact logits 和逐帧 XY 修正供审计。</p>
<p>选择只来自人工评分 <code>score=1</code>，完整规则见 <a href="selection.json">selection.json</a>，汇总见 <a href="quality_summary.json">quality_summary.json</a>。所有编号同时在报告中提供 0-based 和 1-based。</p></div>
{"".join(cards)}
</main></body></html>"""
    (output_root / "index.html").write_text(document, encoding="utf-8")


def completed_result(
    item: dict[str, Any],
    output_root: Path,
    *,
    checkpoint_sha256: str,
    onnx_sha256: str,
    kinematics: Any,
    tolerance_rad: float,
    seed: int,
    cfg_scale: float,
    ddim_steps: int,
    max_duration_sec: float | None,
    sliding_contract_version: str,
    root_orientation_postprocess: str | None,
) -> dict[str, Any] | None:
    report_path = output_root / "reports" / item["dataset"] / f"{item['audio_key']}.json"
    if not report_path.is_file():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    video = output_root / report.get("video_relative", "")
    artifact = output_root / report.get("artifact_relative", "")
    if (
        report.get("status") != "passed"
        or report.get("checkpoint_sha256") != checkpoint_sha256
        or report.get("onnx_sha256") != onnx_sha256
        or report.get("sliding_qpos_contract_version") != sliding_contract_version
        or report.get("root_orientation_postprocess") != root_orientation_postprocess
        or report.get("audio") != item["audio"]
        or not video.is_file()
        or not artifact.is_file()
    ):
        return None
    media_probe(video)
    try:
        artifact_payload = torch.load(artifact, map_location="cpu", weights_only=False)
    except TypeError:
        artifact_payload = torch.load(artifact, map_location="cpu")
    if not isinstance(artifact_payload, dict) or any(
        artifact_payload.get(key) != value
        for key, value in {
            "checkpoint_sha256": checkpoint_sha256,
            "onnx_sha256": onnx_sha256,
            "audio": item["audio"],
            "seed": seed,
            "cfg_scale": cfg_scale,
            "ddim_steps": ddim_steps,
            "max_duration_sec": max_duration_sec,
            "sliding_qpos_contract_version": sliding_contract_version,
            "root_orientation_postprocess": root_orientation_postprocess,
        }.items()
    ):
        return None
    if report.get("joint_limits", {}).get("tolerance_rad") != tolerance_rad:
        report["joint_limits"] = analyze_joint_limits(
            artifact_payload["qpos"], kinematics, tolerance_rad=tolerance_rad
        )
        report["joint_limit_reaudited"] = True
        atomic_json(report_path, report)
    return report


def reusable_artifact(
    artifact_path: Path,
    *,
    item: dict[str, Any],
    checkpoint_sha256: str,
    onnx_sha256: str,
    seed: int,
    cfg_scale: float,
    ddim_steps: int,
    max_duration_sec: float | None,
    sliding_contract_version: str,
    root_orientation_postprocess: str | None,
) -> dict[str, Any] | None:
    """校验并读取渲染失败前已原子落盘的动作产物。"""

    if not artifact_path.is_file():
        return None
    try:
        try:
            artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        except TypeError:
            artifact = torch.load(artifact_path, map_location="cpu")
    except Exception:
        return None
    expected = {
        "checkpoint_sha256": checkpoint_sha256,
        "onnx_sha256": onnx_sha256,
        "audio": item["audio"],
        "seed": seed,
        "cfg_scale": cfg_scale,
        "ddim_steps": ddim_steps,
        "max_duration_sec": max_duration_sec,
        "sliding_qpos_contract_version": sliding_contract_version,
        "root_orientation_postprocess": root_orientation_postprocess,
    }
    if not isinstance(artifact, dict) or any(
        artifact.get(key) != value for key, value in expected.items()
    ):
        return None
    qpos = artifact.get("qpos")
    if not isinstance(qpos, torch.Tensor) or qpos.ndim != 2 or qpos.shape[1] != 28:
        return None
    if len(qpos) <= 0 or not bool(torch.isfinite(qpos).all()):
        return None
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument("--ratings-root", type=Path)
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        help="使用显式 SHA256 绑定样本清单；此时不读取评分表或自建库 manifest。",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--onnx-metadata", type=Path)
    parser.add_argument("--kinematics", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--per-dataset", type=int, default=10)
    parser.add_argument("--finedance-count", type=int)
    parser.add_argument("--compas3d-count", type=int)
    parser.add_argument("--aioz-gdance-count", type=int)
    parser.add_argument("--aistpp-count", type=int)
    parser.add_argument("--mine-manifest", type=Path)
    parser.add_argument("--mine-count", type=int, default=0)
    parser.add_argument("--mine-audio-key", action="append", default=[])
    parser.add_argument(
        "--max-duration-sec",
        type=float,
        help="只评测每首音乐从起点开始的指定秒数；不传则使用完整音乐。",
    )
    parser.add_argument("--onnx-provider", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument("--cfg-scale", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--joint-limit-tolerance-rad", type=float, default=0.25)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--selection-only", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    if not 2 <= args.ddim_steps <= 1000:
        raise ValueError("ddim_steps 必须在 2..1000")
    if not math.isfinite(args.cfg_scale) or args.cfg_scale < 0.0:
        raise ValueError("cfg_scale 必须是有限非负数")
    if args.max_duration_sec is not None and (
        not math.isfinite(args.max_duration_sec) or args.max_duration_sec < 4.0
    ):
        raise ValueError("max_duration_sec 必须为空或至少为 4 秒的有限数")
    paths = {}
    for name in ("checkpoint", "onnx", "kinematics", "stats", "mjcf"):
        paths[name] = getattr(args, name).expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    explicit_selection_path = None
    if args.selection_manifest is not None:
        if args.audio_root is not None or args.ratings_root is not None or args.mine_count:
            raise ValueError(
                "--selection-manifest 不能与 --audio-root/--ratings-root/--mine-count 混用"
            )
        explicit_selection_path = args.selection_manifest.expanduser().resolve(strict=True)
        selected, dataset_summary, selection_limits, selection_policy = load_explicit_selection(
            explicit_selection_path
        )
    else:
        if args.audio_root is None or args.ratings_root is None:
            raise ValueError(
                "未使用 --selection-manifest 时必须提供 --audio-root 和 --ratings-root"
            )
        paths["audio_root"] = args.audio_root.expanduser().resolve(strict=True)
        paths["ratings_root"] = args.ratings_root.expanduser().resolve(strict=True)
        selection_limits = {
            "finedance": args.finedance_count or args.per_dataset,
            "compas3d": args.compas3d_count or args.per_dataset,
            "aioz_gdance": args.aioz_gdance_count or args.per_dataset,
            "aistpp": args.aistpp_count or args.per_dataset,
        }
        selected, dataset_summary = select_hq_audio(
            paths["ratings_root"], paths["audio_root"], per_dataset=selection_limits
        )
        if args.mine_count < 0:
            raise ValueError("mine-count 不能为负数")
        if args.mine_count:
            if args.mine_manifest is None:
                raise ValueError("mine-count 大于 0 时必须提供 --mine-manifest")
            mine_selected, mine_summary = select_mine_bumi(
                args.mine_manifest,
                count=args.mine_count,
                audio_keys=args.mine_audio_key or None,
            )
            selected.extend(mine_selected)
            dataset_summary["mine_bumi"] = mine_summary
            selection_limits["mine_bumi"] = args.mine_count
        selection_policy = "公开四库使用评分 CSV 的 score=1 并按音频键字典序抽样；自建库使用正式 manifest 的 quality_accepted=true 并按 manifest 固定行序均匀抽样"
    # qpos30 模型必须自行学会 root roll/pitch；验证链路禁止直立投影。
    root_orientation_postprocess = None
    metadata_path = (
        args.onnx_metadata.expanduser().resolve(strict=True)
        if args.onnx_metadata is not None
        else paths["onnx"].with_suffix(paths["onnx"].suffix + ".json").resolve(strict=True)
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("contract_version") != BUMI_ONNX_CONTRACT_VERSION:
        raise ValueError("ONNX metadata 不是 BUMI guided denoiser 合约")
    identities = {
        "checkpoint_sha256": sha256_file(paths["checkpoint"]),
        "onnx_sha256": sha256_file(paths["onnx"]),
        "kinematics_sha256": sha256_file(paths["kinematics"]),
        "stats_sha256": sha256_file(paths["stats"]),
        "mjcf_sha256": sha256_file(paths["mjcf"]),
    }
    expected = {
        "checkpoint_sha256": (metadata.get("checkpoint") or {}).get("sha256"),
        "kinematics_sha256": (metadata.get("kinematics") or {}).get("sha256"),
        "stats_sha256": (metadata.get("stats") or {}).get("sha256"),
    }
    for name, value in expected.items():
        if identities[name] != value:
            raise ValueError(f"ONNX 身份不匹配：{name}")
    checkpoint_step = int((metadata.get("checkpoint") or {}).get("global_step", -1))
    model_label = f"s{checkpoint_step:06d}" if checkpoint_step >= 0 else "checkpoint"
    selection = {
        "contract_version": "genmo.bumi_hq_full_music_selection.v5",
        "selection_policy": selection_policy,
        "explicit_selection_manifest": (
            None
            if explicit_selection_path is None
            else {
                "path": str(explicit_selection_path),
                "sha256": sha256_file(explicit_selection_path),
            }
        ),
        "per_dataset_limits": selection_limits,
        "dataset_summary": dataset_summary,
        "model": {
            **identities,
            "label": model_label,
            "checkpoint_global_step": checkpoint_step,
            "onnx_metadata": str(metadata_path),
        },
        "generation": {
            "full_audio": args.max_duration_sec is None,
            "max_duration_sec": args.max_duration_sec,
            "sliding_window_frames": 120,
            "overlap_frames": 30,
            "sliding_qpos_contract_version": BUMI_SLIDING_QPOS_CONTRACT_VERSION,
            "ddim_steps": args.ddim_steps,
            "cfg_scale": args.cfg_scale,
            "seed": args.seed,
            "root_orientation_postprocess": root_orientation_postprocess,
            "foot_lock_postprocess": BUMI_FOOT_LOCK_CONTRACT_VERSION,
        },
        "quality_audit": {
            "strict_xml_limits_reported": True,
            "configured_tolerance_rad": args.joint_limit_tolerance_rad,
            "sensitivity_thresholds_rad": [0.0, 0.05, 0.10, 0.15, 0.20, 0.25],
        },
        "items": selected,
    }
    atomic_json(output_root / "selection.json", selection)
    if args.selection_only:
        print(json.dumps(selection, indent=2, ensure_ascii=False))
        return 0

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但当前不可用")
    endecoder = (
        BumiEndecoder(
            kinematics_path=paths["kinematics"],
            stats_path=paths["stats"],
            enable_contact_targets=False,
        )
        .to(device)
        .eval()
    )
    if endecoder.kinematics.source_mjcf_sha256 != identities["mjcf_sha256"]:
        raise ValueError("--mjcf 与 kinematics JSON 的源 XML SHA256 不一致")
    runner = BumiOrtStepRunner(paths["onnx"], device=device, provider=args.onnx_provider)
    generator = BumiSlidingQposGenerator(
        runner,
        endecoder,
        device=device,
        steps=args.ddim_steps,
        guidance_scale=args.cfg_scale,
    )
    results: list[dict[str, Any]] = []
    total = len(selected)
    for number, item in enumerate(selected, start=1):
        existing = completed_result(
            item,
            output_root,
            checkpoint_sha256=identities["checkpoint_sha256"],
            onnx_sha256=identities["onnx_sha256"],
            kinematics=endecoder.kinematics,
            tolerance_rad=args.joint_limit_tolerance_rad,
            seed=args.seed,
            cfg_scale=args.cfg_scale,
            ddim_steps=args.ddim_steps,
            max_duration_sec=args.max_duration_sec,
            sliding_contract_version=BUMI_SLIDING_QPOS_CONTRACT_VERSION,
            root_orientation_postprocess=root_orientation_postprocess,
        )
        if existing is not None:
            results.append(existing)
            print(
                f"[{number:02d}/{total:02d}] 复用 {item['dataset_label']}/{item['audio_key']}",
                flush=True,
            )
            continue
        started = time.perf_counter()
        artifact_path = output_root / "artifacts" / item["dataset"] / f"{item['audio_key']}.pt"
        report_path = output_root / "reports" / item["dataset"] / f"{item['audio_key']}.json"
        video_path = output_root / "videos" / item["dataset"] / f"{item['audio_key']}.mp4"
        audio_path = Path(item["audio"])
        print(
            f"[{number:02d}/{total:02d}] 生成 {item['dataset_label']}/{item['audio_key']}",
            flush=True,
        )
        try:
            features, feature_metadata = extract_edge_baseline35(audio_path, target_fps=30)
            features, feature_metadata, original_duration_sec = truncate_music_features(
                features,
                feature_metadata,
                max_duration_sec=args.max_duration_sec,
            )
            artifact = reusable_artifact(
                artifact_path,
                item=item,
                checkpoint_sha256=identities["checkpoint_sha256"],
                onnx_sha256=identities["onnx_sha256"],
                seed=args.seed,
                cfg_scale=args.cfg_scale,
                ddim_steps=args.ddim_steps,
                max_duration_sec=args.max_duration_sec,
                sliding_contract_version=BUMI_SLIDING_QPOS_CONTRACT_VERSION,
                root_orientation_postprocess=root_orientation_postprocess,
            )
            if artifact is None:
                generated = generator.generate(features, seed=args.seed)
                raw_qpos = generated.qpos_raw.float().contiguous()
                qpos = generated.qpos.float().contiguous()
                contact_logits = generated.foot_contact_logits.float().contiguous()
                canonical = endecoder.codec.encode(qpos.to(device)).canonical_qpos.detach().cpu()
                windows = len(generated.chunks)
            else:
                qpos = artifact["qpos"].float().contiguous()
                raw_qpos = artifact.get("qpos_raw", qpos).float().contiguous()
                contact_logits = artifact["foot_contact_logits"].float().contiguous()
                canonical = artifact["qpos_canonical"].float().contiguous()
                windows = len(plan_sliding_windows(len(qpos)))
                print(
                    f"[{number:02d}/{total:02d}] 复用已生成动作，仅补渲染与审计",
                    flush=True,
                )
            metrics = metrics_to_json(
                compute_bumi_kinematic_metrics(
                    qpos.to(device),
                    endecoder.kinematics,
                    pred_contact_logits=contact_logits.to(device),
                    music_beats=features[:, 34].to(device),
                    ground_height=0.0,
                )
            )
            raw_metrics = metrics_to_json(
                compute_bumi_kinematic_metrics(
                    raw_qpos.to(device),
                    endecoder.kinematics,
                    pred_contact_logits=contact_logits.to(device),
                    music_beats=features[:, 34].to(device),
                    ground_height=0.0,
                )
            )
            limits = analyze_joint_limits(
                qpos,
                endecoder.kinematics,
                tolerance_rad=args.joint_limit_tolerance_rad,
            )
            if artifact is None:
                artifact = {
                    "contract_version": "genmo.bumi_hq_full_music_prediction.v3",
                    "robot_name": "bumi",
                    "fps": 30,
                    "qpos": qpos,
                    "qpos_raw": raw_qpos,
                    "foot_contact_logits": generated.foot_contact_logits,
                    "foot_lock_correction_xy": generated.foot_lock_correction_xy,
                    "foot_lock_active_contact": generated.foot_lock_active_contact,
                    "foot_lock_contract_version": generated.foot_lock_contract_version,
                    "qpos_canonical": canonical,
                    "joint_names": list(endecoder.kinematics.joint_order),
                    "quaternion_convention": "wxyz",
                    "qpos_order": "mujoco_native",
                    "audio": str(audio_path),
                    "feature_metadata": feature_metadata,
                    "high_quality_source": item,
                    "checkpoint_sha256": identities["checkpoint_sha256"],
                    "onnx_sha256": identities["onnx_sha256"],
                    "sliding_qpos_contract_version": BUMI_SLIDING_QPOS_CONTRACT_VERSION,
                    "kinematics_sha256": identities["kinematics_sha256"],
                    "seed": args.seed,
                    "cfg_scale": args.cfg_scale,
                    "ddim_steps": args.ddim_steps,
                    "max_duration_sec": args.max_duration_sec,
                    "root_orientation_postprocess": root_orientation_postprocess,
                    "foot_lock_postprocess": BUMI_FOOT_LOCK_CONTRACT_VERSION,
                }
                artifact_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_artifact = artifact_path.with_suffix(".tmp")
                torch.save(artifact, temporary_artifact)
                temporary_artifact.replace(artifact_path)
            media = render_and_mux(
                artifact=artifact_path,
                audio=audio_path,
                mjcf=paths["mjcf"],
                output=video_path,
                width=args.width,
                height=args.height,
            )
            source_duration = float(feature_metadata["selected_duration_sec"])
            if abs(media["duration_sec"] - source_duration) > 0.15:
                raise RuntimeError(
                    f"视频/音频时长偏差过大：video={media['duration_sec']}, audio={source_duration}"
                )
            report = {
                "contract_version": "genmo.bumi_hq_full_music_sample_report.v3",
                "status": "passed",
                **{key: item[key] for key in item},
                "audio": str(audio_path),
                "source_duration_sec": source_duration,
                "original_audio_duration_sec": original_duration_sec,
                "frames": len(qpos),
                "windows": windows,
                "artifact_relative": _relative(artifact_path, output_root),
                "video_relative": _relative(video_path, output_root),
                "report_relative": _relative(report_path, output_root),
                "checkpoint_sha256": identities["checkpoint_sha256"],
                "onnx_sha256": identities["onnx_sha256"],
                "sliding_qpos_contract_version": BUMI_SLIDING_QPOS_CONTRACT_VERSION,
                "seed": args.seed,
                "cfg_scale": args.cfg_scale,
                "ddim_steps": args.ddim_steps,
                "max_duration_sec": args.max_duration_sec,
                "root_orientation_postprocess": root_orientation_postprocess,
                "foot_lock_postprocess": BUMI_FOOT_LOCK_CONTRACT_VERSION,
                "metrics": metrics,
                "raw_model_metrics": raw_metrics,
                "joint_limits": limits,
                "media": media,
                "elapsed_seconds": time.perf_counter() - started,
                "scope": "运动学/关节限位离线评测；不代表 GMT 动力学可跟踪性",
            }
        except Exception as exc:
            report = {
                "contract_version": "genmo.bumi_hq_full_music_sample_report.v3",
                "status": "failed",
                **{key: item[key] for key in item},
                "audio": str(audio_path),
                "report_relative": _relative(report_path, output_root),
                "checkpoint_sha256": identities["checkpoint_sha256"],
                "onnx_sha256": identities["onnx_sha256"],
                "sliding_qpos_contract_version": BUMI_SLIDING_QPOS_CONTRACT_VERSION,
                "max_duration_sec": args.max_duration_sec,
                "root_orientation_postprocess": root_orientation_postprocess,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": time.perf_counter() - started,
            }
            print(f"[{number:02d}/{total:02d}] 失败：{report['error']}", flush=True)
        atomic_json(report_path, report)
        results.append(report)
        summary = build_summary(results, tolerance_rad=args.joint_limit_tolerance_rad)
        atomic_json(output_root / "quality_summary.json", summary)
        build_index(output_root, results, summary, selection)

    summary = build_summary(results, tolerance_rad=args.joint_limit_tolerance_rad)
    atomic_json(output_root / "quality_summary.json", summary)
    build_index(output_root, results, summary, selection)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["failed_samples"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
