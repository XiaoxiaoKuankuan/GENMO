#!/usr/bin/env python3
"""构建人工高质量原动作与 BUMI 音乐模型生成动作的同步对比网页。

输入是 ``validate_bumi_hq_music_full.py`` 已冻结的可变数量选择和模型生成产物。公开四库的
人工 ``score=1`` 七字段 50 Hz SONIC/Isaac-Lab BUMI NPZ 严格复用正式数据构建器的
50→30 Hz 语义：根位置和关节线性插值、wxyz 根四元数最短弧 SLERP、Isaac publish order
按完整名称重排到 GENMO/MuJoCo-native order，并按训练契约执行 body-origin 地面归一化；
自建数据库则直接读取其正式发布、已经裁齐和地面规范化的 30 Hz qpos28 PT，不进行二次
插值或二次落地。

每项按参考验收目录布局输出两个独立视频：``<key>_gmr_bumi3.mp4`` 是原数据集动作，
``<key>_generated.mp4`` 是完整音乐长度的模型生成动作；模型生成源发生变化时以临时硬链接/
副本原子刷新旧文件。网页提供成对播放、暂停、重播和时间轴同步，不再额外转码一份左右拼接
视频，避免完整音乐验证重复占用磁盘。音轨统一来自同一 WAV，生成视频始终覆盖完整音乐；
原动作按其真实长度渲染，因此 AIST++ 的短源片段不会被循环、拉伸或伪造成完整舞蹈，网页会
明确显示这一数据边界。报告仍在双方共同真实区间计算运动学指标和 0.25 rad 限位结果，并
额外生成不含本地绝对路径的 ``site_data.json``，供公开验证网页直接消费。该页面展示的是
GMR 离线重定向轨迹和模型生成轨迹，不代表 GMT/真实机器人动力学跟踪效果。
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gem.robots.bumi.kinematics import BumiKinematics  # noqa: E402
from gem.robots.bumi.metrics import (  # noqa: E402
    compute_bumi_kinematic_metrics,
    metrics_to_json,
)
from scripts.validate_bumi_hq_music_full import (  # noqa: E402
    REPORT_DATASETS,
    analyze_joint_limits,
    atomic_json,
    media_probe,
    sha256_file,
)
from tools.data.bumi.build_bumi_music_dataset_from_sonic_npz import (  # noqa: E402
    RESAMPLE_CONTRACT_VERSION,
    expected_50hz_frames,
    normalize_body_origin_ground,
    resample_sonic_qpos_to_30hz,
)
from tools.data.bumi.filter_sonic_npz_motions import (  # noqa: E402
    load_config,
    load_motion_npz,
)


def target_30hz_frames(source_50hz_frames: int) -> int:
    """反解正式不含右端点时间网格，拒绝无法精确对应 30 Hz 的源长度。"""

    source_frames = int(source_50hz_frames)
    if source_frames <= 0:
        raise ValueError("source_50hz_frames 必须为正数")
    center = max(1, round((source_frames - 1) * 30.0 / 50.0) + 1)
    candidates = [
        value
        for value in range(max(1, center - 3), center + 4)
        if expected_50hz_frames(value) == source_frames
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"50 Hz 帧数无法唯一反解到正式 30 Hz 时间网格：source={source_frames}, "
            f"candidates={candidates}"
        )
    return candidates[0]


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def materialize_file(source: Path, target: Path) -> str:
    """优先硬链接正式生成视频，并在源身份变化时原子刷新已有目标。"""

    source = source.resolve(strict=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    target_existed = target.is_file()
    if target_existed:
        if target.stat().st_size == source.stat().st_size and sha256_file(target) == sha256_file(
            source
        ):
            return "reused"
    temporary = target.with_name(f".{target.name}.materialize.tmp")
    temporary.unlink(missing_ok=True)
    try:
        try:
            os.link(source, temporary)
            method = "hardlink"
        except OSError:
            shutil.copy2(source, temporary)
            method = "copy"
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return f"refreshed_{method}" if target_existed else method


def save_original_artifact(
    *,
    source_motion: Path,
    quality_config: Any,
    kinematics: BumiKinematics,
    kinematics_path: Path,
    output: Path,
    item: dict[str, Any],
) -> tuple[dict[str, Any], torch.Tensor]:
    if source_motion.suffix == ".pt":
        source_payload = torch_load(source_motion)
        if not isinstance(source_payload, dict):
            raise ValueError("自建库正式动作必须是字典 PT")
        qpos = torch.as_tensor(source_payload.get("qpos")).detach().cpu().float().contiguous()
        if qpos.ndim != 2 or qpos.shape[1] != 28 or len(qpos) < 120:
            raise ValueError(f"自建库正式 qpos 必须为至少 120 帧的 [T,28]，实际 {qpos.shape}")
        if not bool(torch.isfinite(qpos).all()):
            raise ValueError("自建库正式 qpos 包含 NaN/Inf")
        if int(source_payload.get("fps", -1)) != 30:
            raise ValueError("自建库正式动作必须为 30 Hz")
        if source_payload.get("qpos_order") != "mujoco_native":
            raise ValueError("自建库正式动作不是 MuJoCo-native qpos 顺序")
        if source_payload.get("quaternion_convention") != "wxyz":
            raise ValueError("自建库正式动作不是 wxyz 四元数")
        if tuple(source_payload.get("joint_names") or ()) != tuple(kinematics.joint_order):
            raise ValueError("自建库正式动作关节顺序与目标 kinematics 不一致")
        if source_payload.get("source_mjcf_sha256") != kinematics.source_mjcf_sha256:
            raise ValueError("自建库正式动作绑定的 MJCF 与目标 kinematics 不一致")
        artifact = {
            **source_payload,
            "contract_version": "genmo.bumi_hq_original_comparison_motion.v1",
            "source_motion": str(source_motion),
            "source_motion_sha256": sha256_file(source_motion),
            "source_preprocessed_30hz": True,
            "target_frames_30hz": len(qpos),
            "target_kinematics": str(kinematics_path),
            "target_kinematics_sha256": kinematics.kinematics_sha256,
            "high_quality_source": item,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp")
        torch.save(artifact, temporary)
        temporary.replace(output)
        return artifact, qpos

    arrays = load_motion_npz(source_motion, quality_config)
    target_frames = target_30hz_frames(len(arrays["joint_pos"]))
    qpos = resample_sonic_qpos_to_30hz(
        arrays,
        source_joint_order=quality_config.joint_order,
        target_joint_order=kinematics.joint_order,
        target_frames=target_frames,
    )
    qpos, ground_before, ground_after = normalize_body_origin_ground(qpos, kinematics)
    artifact = {
        "contract_version": "genmo.bumi_hq_original_comparison_motion.v1",
        "robot_name": "bumi",
        "fps": 30,
        "qpos": qpos,
        "joint_names": list(kinematics.joint_order),
        "quaternion_convention": "wxyz",
        "qpos_order": "mujoco_native",
        "source_contract_version": quality_config.motion_contract_version,
        "resample_contract_version": RESAMPLE_CONTRACT_VERSION,
        "source_motion": str(source_motion),
        "source_motion_sha256": sha256_file(source_motion),
        "source_frames_50hz": len(arrays["joint_pos"]),
        "source_fps": 50,
        "target_frames_30hz": len(qpos),
        "target_kinematics": str(kinematics_path),
        "target_kinematics_sha256": kinematics.kinematics_sha256,
        "root_z_adjusted": True,
        "body_origin_ground_before_m": ground_before,
        "body_origin_ground_after_m": ground_after,
        "high_quality_source": item,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    torch.save(artifact, temporary)
    temporary.replace(output)
    return artifact, qpos


def render_original_video(
    *,
    artifact: Path,
    audio: Path,
    mjcf: Path,
    output: Path,
    duration_sec: float,
    width: int,
    height: int,
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
                "--max-frames",
                str(round(duration_sec * 30.0)),
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
                "-t",
                f"{duration_sec:.9f}",
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


def motion_metrics(qpos: torch.Tensor, kinematics: BumiKinematics) -> dict[str, Any]:
    return metrics_to_json(compute_bumi_kinematic_metrics(qpos, kinematics, ground_height=0.0))


def completed_report(
    report_path: Path,
    *,
    source_motion: Path,
    generated_source: Path,
    output_root: Path,
) -> dict[str, Any] | None:
    if not report_path.is_file():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("status") != "passed"
        or report.get("contract_version") != "genmo.bumi_hq_original_comparison_sample.v2"
        or report.get("source_motion_sha256") != sha256_file(source_motion)
        or report.get("generated_source_sha256") != sha256_file(generated_source)
    ):
        return None
    for field in (
        "original_video_relative",
        "generated_video_relative",
    ):
        path = output_root / report.get(field, "")
        if not path.is_file():
            return None
        media_probe(path)
    return report


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    passed = [row for row in results if row.get("status") == "passed"]
    root_postprocesses = {
        row.get("root_orientation_postprocess") for row in passed
    }
    dataset_rows = {}
    for dataset, label, _ in REPORT_DATASETS:
        rows = [row for row in passed if row["dataset"] == dataset]
        dataset_rows[dataset] = {
            "dataset_label": label,
            "completed": len(rows),
            "comparison_duration_sec": sum(row["comparison_duration_sec"] for row in rows),
            "original_video_duration_sec": sum(
                row["original_video_duration_sec"] for row in rows
            ),
            "generated_full_duration_sec": sum(row["generated_duration_sec"] for row in rows),
            "source_clip_shorter_than_audio": sum(
                row["source_clip_shorter_than_audio"] for row in rows
            ),
        }
    metric_comparison = {}
    for key in ("foot_penetration_max_m", "root_tilt_max_rad", "joint_velocity_p95_radps"):
        original = [float(row["original_metrics"][key]) for row in passed]
        generated = [float(row["generated_overlap_metrics"][key]) for row in passed]
        metric_comparison[key] = {
            "original_mean": float(np.mean(original)) if original else None,
            "generated_mean": float(np.mean(generated)) if generated else None,
            "original_max": float(np.max(original)) if original else None,
            "generated_max": float(np.max(generated)) if generated else None,
        }
    joint_limits = {}
    for label, field in (
        ("original", "original_joint_limits"),
        ("generated_same_interval", "generated_overlap_joint_limits"),
    ):
        joint_limits[label] = {
            "strict_xml_exceed_samples": sum(
                bool(row[field]["strict_xml_limit_exceeded"]) for row in passed
            ),
            "exceed_0_25_rad_samples": sum(
                bool(row[field]["tolerance_limit_exceeded"]) for row in passed
            ),
        }
    return {
        "contract_version": "genmo.bumi_hq_original_comparison_summary.v2",
        "evaluated": len(results),
        "completed": len(passed),
        "failed": len(results) - len(passed),
        "comparison_duration_sec": sum(row["comparison_duration_sec"] for row in passed),
        "original_video_duration_sec": sum(
            row["original_video_duration_sec"] for row in passed
        ),
        "generated_full_duration_sec": sum(row["generated_duration_sec"] for row in passed),
        "root_orientation_postprocess": (
            None
            if not root_postprocesses
            else next(iter(root_postprocesses))
            if len(root_postprocesses) == 1
            else "mixed"
        ),
        "dataset_summary": dataset_rows,
        "metric_comparison_same_interval": metric_comparison,
        "joint_limit_comparison": joint_limits,
        "failed_items": [
            {"dataset": row["dataset"], "audio_key": row["audio_key"], "error": row.get("error")}
            for row in results
            if row.get("status") != "passed"
        ],
    }


def build_site_data(
    results: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    items: list[dict[str, Any]],
    model_label: str,
) -> dict[str, Any]:
    """生成公开网页使用的最小可追溯数据，排除本地路径和服务端身份。"""

    selected = {(item["dataset"], item["audio_key"]): item for item in items}
    site_items = []
    for row in results:
        if row.get("status") != "passed":
            continue
        item = selected[(row["dataset"], row["audio_key"])]
        site_items.append(
            {
                "dataset": row["dataset"],
                "dataset_label": row["dataset_label"],
                "audio_key": row["audio_key"],
                "representative_motion": row["representative_motion"],
                "high_quality_motion_count": item["high_quality_motion_count"],
                "comparison_duration_sec": row["comparison_duration_sec"],
                "original_video_duration_sec": row["original_video_duration_sec"],
                "generated_duration_sec": row["generated_duration_sec"],
                "source_audio_duration_sec": row["source_audio_duration_sec"],
                "source_clip_shorter_than_audio": row["source_clip_shorter_than_audio"],
                "original_video": f"/{row['original_video_relative']}",
                "generated_video": f"/{row['generated_video_relative']}",
                "original_metrics": row["original_metrics"],
                "generated_metrics": row["generated_overlap_metrics"],
                "original_joint_limits": {
                    "strict_xml_limit_exceeded": row["original_joint_limits"][
                        "strict_xml_limit_exceeded"
                    ],
                    "tolerance_limit_exceeded": row["original_joint_limits"][
                        "tolerance_limit_exceeded"
                    ],
                    "max_excess_rad": row["original_joint_limits"]["strict_xml_max_excess_rad"],
                },
                "generated_joint_limits": {
                    "strict_xml_limit_exceeded": row["generated_overlap_joint_limits"][
                        "strict_xml_limit_exceeded"
                    ],
                    "tolerance_limit_exceeded": row["generated_overlap_joint_limits"][
                        "tolerance_limit_exceeded"
                    ],
                    "max_excess_rad": row["generated_overlap_joint_limits"][
                        "strict_xml_max_excess_rad"
                    ],
                },
            }
        )
    return {
        "contract_version": "genmo.bumi_hq_original_comparison_site.v2",
        "model_label": model_label,
        "root_orientation_postprocess": summary.get("root_orientation_postprocess"),
        "comparison_policy": "两个独立视频同页同步；生成覆盖完整音乐，原动作保留真实长度，短源片段不循环或拉伸",
        "summary": summary,
        "items": site_items,
    }


def build_index(
    output_root: Path,
    results: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    model_label: str,
) -> None:
    """生成与历史验收目录一致的双视频同步、筛选和懒加载页面。"""

    def metric_text(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.4f}"

    sections = []
    for dataset, label, _ in REPORT_DATASETS:
        rows = [row for row in results if row["dataset"] == dataset]
        cards = []
        for index, row in enumerate(rows, start=1):
            if row.get("status") != "passed":
                cards.append(
                    f'<article class="card failed"><h3>{index}. {html.escape(row["audio_key"])}</h3>'
                    f"<p>{html.escape(str(row.get('error')))}</p></article>"
                )
                continue
            original = row["original_metrics"]
            generated = row["generated_overlap_metrics"]
            short_note = (
                "原数据集只有该真实短片段，未循环或拉伸；右侧生成动作仍覆盖整首音乐。"
                if row["source_clip_shorter_than_audio"]
                else "原动作与整首音乐时长一致。"
            )
            query = html.escape(
                f"{dataset} {label} {row['audio_key']} {row['representative_motion']}".lower(),
                quote=True,
            )
            cards.append(
                f'<article class="card" data-dataset="{dataset}" data-query="{query}">'
                f'<h3>{index}. {html.escape(row["audio_key"])}</h3>'
                f'<p class="motion">人工高质量动作：<code>{html.escape(row["representative_motion"])}</code></p>'
                '<div class="pair" data-sync-pair>'
                '<div class="video-panel"><div class="video-label">原始 GMR BUMI3 动作</div>'
                f'<video class="original" controls muted playsinline preload="none" data-src="{html.escape(row["original_video_relative"])}"></video></div>'
                f'<div class="video-panel"><div class="video-label">{html.escape(model_label)} 模型生成（完整音乐）</div>'
                f'<video class="generated" controls playsinline preload="none" data-src="{html.escape(row["generated_video_relative"])}"></video></div></div>'
                '<div class="controls"><button type="button" data-action="play">同步播放</button>'
                '<button type="button" data-action="pause">暂停</button>'
                '<button type="button" data-action="restart">从头播放</button></div>'
                f"<p>共同真实对比区间 {row['comparison_duration_sec']:.2f}s；原动作视频 "
                f"{row['original_video_duration_sec']:.2f}s；模型生成/音乐 "
                f"{row['generated_duration_sec']:.2f}s。{short_note}</p>"
                "<table><thead><tr><th>同区间指标</th><th>原数据集</th><th>模型生成</th></tr></thead><tbody>"
                f"<tr><td>最大脚部穿地(m)</td><td>{original['foot_penetration_max_m']:.4f}</td><td>{generated['foot_penetration_max_m']:.4f}</td></tr>"
                f"<tr><td>最大根倾角(rad)</td><td>{original['root_tilt_max_rad']:.4f}</td><td>{generated['root_tilt_max_rad']:.4f}</td></tr>"
                f"<tr><td>关节速度P95(rad/s)</td><td>{original['joint_velocity_p95_radps']:.4f}</td><td>{generated['joint_velocity_p95_radps']:.4f}</td></tr>"
                "</tbody></table>"
                f'<p><a href="{html.escape(row["original_video_relative"])}">下载原动作</a> · '
                f'<a href="{html.escape(row["generated_video_relative"])}">下载模型生成</a> · '
                f'<a href="{html.escape(row["report_relative"])}">JSON报告</a></p></article>'
            )
        sections.append(
            f'<section data-section="{dataset}"><h2>{label}</h2>{"".join(cards)}</section>'
        )
    metrics = summary["metric_comparison_same_interval"]
    limits = summary["joint_limit_comparison"]
    document = f"""<!doctype html>
<!--
本页由 GENMO 高质量多库验证工具生成。每个样本保留两个独立视频：左侧是真实 GMR BUMI3
原动作，右侧是模型对完整音乐的生成动作。页面以右侧生成视频作为有声音的主时间轴，并在
原动作真实长度内同步左侧视频；AIST++ 短片段结束后不会循环或伪造剩余动作。
-->
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BUMI 原数据集与完整音乐生成对比</title><style>
:root{{--bg:#0e1219;--panel:#191f2b;--line:#30394a;--text:#edf1f7;--muted:#9ca9ba;--accent:#78c5ff}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 system-ui,sans-serif}}main{{max-width:1580px;margin:auto;padding:28px}}a{{color:var(--accent)}}code{{overflow-wrap:anywhere}}
.summary,.toolbar,.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;margin:14px 0}}.failed{{border-color:#a74d4d}}.toolbar{{position:sticky;top:0;z-index:3;display:flex;gap:8px;flex-wrap:wrap;align-items:center}}
.toolbar input{{min-width:260px;flex:1;background:#0f151e;color:var(--text);border:1px solid #3b4659;border-radius:8px;padding:9px 11px}}button{{cursor:pointer;background:#28364a;color:var(--text);border:1px solid #46556d;border-radius:8px;padding:8px 12px}}button.active{{background:#21679b;border-color:#53aeea}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}.video-panel{{min-width:0}}.video-label{{font-weight:700;margin:0 0 7px}}video{{display:block;width:100%;aspect-ratio:4/3;background:#000;border-radius:8px}}.controls{{display:flex;gap:8px;margin:10px 0;flex-wrap:wrap}}
.motion{{color:var(--muted)}}table{{border-collapse:collapse;width:100%}}th,td{{padding:7px;border:1px solid #3a4354;text-align:left}}section{{margin-top:34px}}
@media(max-width:850px){{main{{padding:14px}}.pair{{grid-template-columns:1fr}}.toolbar{{position:static}}}}</style></head><body><main>
<h1>BUMI 人工高质量原动作 vs {html.escape(model_label)} 模型生成</h1><div class="summary">
<p>完成 {summary["completed"]}/{summary["evaluated"]} 项；模型生成完整音乐总时长 {summary["generated_full_duration_sec"] / 60.0:.2f} 分钟，双方共同真实对比区间 {summary["comparison_duration_sec"] / 60.0:.2f} 分钟。左侧是 score=1 对应的原始 GMR BUMI3 轨迹，右侧是同一首完整音乐的模型生成轨迹。</p>
<p>同区间均值：最大脚部穿地 原始 {metric_text(metrics["foot_penetration_max_m"]["original_mean"])}m / 生成 {metric_text(metrics["foot_penetration_max_m"]["generated_mean"])}m；最大根倾角 原始 {metric_text(metrics["root_tilt_max_rad"]["original_mean"])}rad / 生成 {metric_text(metrics["root_tilt_max_rad"]["generated_mean"])}rad。</p>
<p>0.25rad 容差后关节超限：原动作 {limits["original"]["exceed_0_25_rad_samples"]}/{summary["completed"]}，模型生成同区间 {limits["generated_same_interval"]["exceed_0_25_rad_samples"]}/{summary["completed"]}；严格 XML 原始边界触碰/超出分别为 {limits["original"]["strict_xml_exceed_samples"]}/{summary["completed"]} 和 {limits["generated_same_interval"]["strict_xml_exceed_samples"]}/{summary["completed"]}。</p>
<p>AIST++ 原数据本身是短动作片段，左侧到真实末帧即结束；右侧仍播放完整音乐和完整生成动作。<a href="summary.json">汇总 JSON</a> · <a href="selection.json">选择与契约</a></p>
<p>生成动作根姿态后处理：<code>{html.escape(str(summary.get("root_orientation_postprocess") or "无"))}</code>；启用时保留模型全部关节和连续根 yaw，将 yaw 角速度限制为 4rad/s，移除导致横躺的根 roll/pitch，并只在足底穿地时向上抬根 Z；原始未投影轨迹保存在上游 artifact 的 <code>qpos_raw</code>。</p></div>
<div class="toolbar"><button class="active" data-filter="all">全部</button><button data-filter="finedance">FineDance</button><button data-filter="compas3d">CoMPAS3D</button><button data-filter="aioz_gdance">AIOZ-GDance</button><button data-filter="aistpp">AIST++</button><button data-filter="mine_bumi">自建数据库</button><input id="search" type="search" placeholder="搜索音乐键或动作名"></div>
{"".join(sections)}
<script>
const loadVideo=v=>{{if(!v.src){{v.src=v.dataset.src;v.load()}}}};
const videos=[...document.querySelectorAll('video[data-src]')];
if('IntersectionObserver'in window){{const observer=new IntersectionObserver(entries=>entries.forEach(entry=>{{if(entry.isIntersecting){{loadVideo(entry.target);observer.unobserve(entry.target)}}}}),{{rootMargin:'500px'}});videos.forEach(v=>observer.observe(v))}}else{{videos.forEach(loadVideo)}}
const clampOriginal=(original,time)=>{{if(!Number.isFinite(original.duration)||original.duration<=0)return 0;return Math.min(time,Math.max(0,original.duration-.04))}};
document.querySelectorAll('[data-sync-pair]').forEach(pair=>{{
 const original=pair.querySelector('.original'),generated=pair.querySelector('.generated');original.muted=true;
 const ensure=()=>{{loadVideo(original);loadVideo(generated)}};
 const seekOriginal=()=>{{if(original.readyState<1)return;const target=clampOriginal(original,generated.currentTime);if(Math.abs(original.currentTime-target)>.12)original.currentTime=target}};
 const play=restart=>{{ensure();if(restart){{generated.currentTime=0;if(original.readyState>=1)original.currentTime=0}}seekOriginal();generated.play().catch(()=>{{}});if(!Number.isFinite(original.duration)||generated.currentTime<original.duration-.04)original.play().catch(()=>{{}})}};
 const card=pair.closest('.card');card.querySelector('[data-action="play"]').addEventListener('click',()=>play(false));card.querySelector('[data-action="restart"]').addEventListener('click',()=>play(true));card.querySelector('[data-action="pause"]').addEventListener('click',()=>{{generated.pause();original.pause()}});
 generated.addEventListener('play',()=>{{seekOriginal();if(!Number.isFinite(original.duration)||generated.currentTime<original.duration-.04)original.play().catch(()=>{{}})}});generated.addEventListener('pause',()=>original.pause());generated.addEventListener('seeking',seekOriginal);
 setInterval(()=>{{if(generated.paused)return;if(Number.isFinite(original.duration)&&generated.currentTime>=original.duration-.04){{original.pause();return}}seekOriginal();if(original.paused)original.play().catch(()=>{{}})}},250);
}});
let active='all';const search=document.querySelector('#search');
const applyFilter=()=>{{const needle=search.value.trim().toLowerCase();document.querySelectorAll('.card').forEach(card=>{{card.hidden=!((active==='all'||card.dataset.dataset===active)&&(!needle||card.dataset.query.includes(needle)))}});document.querySelectorAll('section[data-section]').forEach(section=>{{section.hidden=![...section.querySelectorAll('.card')].some(card=>!card.hidden)}})}};
document.querySelectorAll('[data-filter]').forEach(button=>button.addEventListener('click',()=>{{active=button.dataset.filter;document.querySelectorAll('[data-filter]').forEach(x=>x.classList.toggle('active',x===button));applyFilter()}}));search.addEventListener('input',applyFilter);
</script></main></body></html>"""
    (output_root / "index.html").write_text(document, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--source-motion-root", type=Path, required=True)
    parser.add_argument("--quality-config", type=Path, required=True)
    parser.add_argument("--kinematics", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--joint-limit-tolerance-rad", type=float, default=0.25)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    return parser.parse_args()


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    if not math.isfinite(args.joint_limit_tolerance_rad) or args.joint_limit_tolerance_rad < 0.0:
        raise ValueError("joint-limit tolerance 必须为有限非负数")
    validation_root = args.validation_root.expanduser().resolve(strict=True)
    source_root = args.source_motion_root.expanduser().resolve(strict=True)
    config_path = args.quality_config.expanduser().resolve(strict=True)
    kinematics_path = args.kinematics.expanduser().resolve(strict=True)
    mjcf_path = args.mjcf.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    selection = json.loads((validation_root / "selection.json").read_text(encoding="utf-8"))
    items = selection.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("validation selection 必须包含至少一项")
    model_label = str((selection.get("model") or {}).get("label", "checkpoint"))
    if re.fullmatch(r"[A-Za-z0-9_.-]+", model_label) is None:
        raise ValueError("selection model.label 含有不安全字符")
    total = len(items)
    quality_config = load_config(config_path)
    kinematics = BumiKinematics(kinematics_path).eval()
    if kinematics.source_mjcf_sha256 != sha256_file(mjcf_path):
        raise ValueError("目标 kinematics 与 --mjcf SHA256 不一致")
    frozen_selection = {
        "contract_version": "genmo.bumi_hq_original_comparison_selection.v2",
        "source_validation": str(validation_root),
        "source_selection_sha256": sha256_file(validation_root / "selection.json"),
        "source_motion_root": str(source_root),
        "quality_config": str(config_path),
        "quality_config_sha256": sha256_file(config_path),
        "source_joint_order": list(quality_config.joint_order),
        "target_kinematics": str(kinematics_path),
        "target_kinematics_sha256": kinematics.kinematics_sha256,
        "mjcf": str(mjcf_path),
        "mjcf_sha256": sha256_file(mjcf_path),
        "model_label": model_label,
        "root_orientation_postprocess": (selection.get("generation") or {}).get(
            "root_orientation_postprocess"
        ),
        "comparison_policy": "公开四库原50Hz动作精确重采样到30Hz，自建库复用正式30Hz qpos28；生成视频覆盖完整音乐；原动作保留真实长度；网页双视频同步；不循环或拉伸短片段",
        "items": items,
    }
    atomic_json(output_root / "selection.json", frozen_selection)
    results: list[dict[str, Any]] = []
    for number, item in enumerate(items, start=1):
        dataset = item["dataset"]
        key = item["audio_key"]
        source_motion = Path(
            item.get("source_motion")
            or (source_root / dataset / item["representative_motion"])
        ).resolve(strict=True)
        generated_artifact_source = (validation_root / "artifacts" / dataset / f"{key}.pt").resolve(
            strict=True
        )
        generated_video_source = (validation_root / "videos" / dataset / f"{key}.mp4").resolve(
            strict=True
        )
        report_path = output_root / "reports" / dataset / f"{key}.json"
        existing = completed_report(
            report_path,
            source_motion=source_motion,
            generated_source=generated_artifact_source,
            output_root=output_root,
        )
        if existing is not None:
            results.append(existing)
            print(f"[{number:02d}/{total:02d}] 复用 {item['dataset_label']}/{key}", flush=True)
            continue
        artifact_path = output_root / "artifacts" / "original" / dataset / f"{key}.pt"
        sample_directory = output_root / dataset / key
        original_video = sample_directory / f"{key}_gmr_bumi3.mp4"
        generated_video = sample_directory / f"{key}_generated.mp4"
        started = time.perf_counter()
        print(f"[{number:02d}/{total:02d}] 对比 {item['dataset_label']}/{key}", flush=True)
        try:
            original_artifact, original_qpos = save_original_artifact(
                source_motion=source_motion,
                quality_config=quality_config,
                kinematics=kinematics,
                kinematics_path=kinematics_path,
                output=artifact_path,
                item=item,
            )
            generated_payload = torch_load(generated_artifact_source)
            generated_qpos = torch.as_tensor(generated_payload["qpos"]).float()
            if generated_qpos.ndim != 2 or generated_qpos.shape[1] != 28:
                raise ValueError("模型生成 artifact qpos 不是 [T,28]")
            comparison_frames = min(len(original_qpos), len(generated_qpos))
            if comparison_frames < 120:
                raise ValueError("原动作与模型生成的共同区间不足 120 帧")
            comparison_duration = comparison_frames / 30.0
            generated_duration = float(
                generated_payload["feature_metadata"]["selected_duration_sec"]
            )
            original_video_duration = min(len(original_qpos) / 30.0, generated_duration)
            source_audio_duration = float(
                generated_payload["feature_metadata"].get(
                    "original_selected_duration_sec", generated_duration
                )
            )
            audio_path = Path(item["audio"]).resolve(strict=True)
            original_probe = render_original_video(
                artifact=artifact_path,
                audio=audio_path,
                mjcf=mjcf_path,
                output=original_video,
                duration_sec=original_video_duration,
                width=args.width,
                height=args.height,
            )
            if abs(original_probe["duration_sec"] - original_video_duration) > 0.15:
                raise RuntimeError("原动作视频时长误差超过 0.15 秒")
            generated_materialization = materialize_file(generated_video_source, generated_video)
            generated_probe = media_probe(generated_video)
            if abs(generated_probe["duration_sec"] - generated_duration) > 0.15:
                raise RuntimeError("生成视频没有覆盖完整音乐")
            original_overlap = original_qpos[:comparison_frames]
            original_metrics = motion_metrics(original_overlap, kinematics)
            generated_metrics = motion_metrics(generated_qpos[:comparison_frames], kinematics)
            report = {
                "contract_version": "genmo.bumi_hq_original_comparison_sample.v2",
                "status": "passed",
                "dataset": dataset,
                "dataset_label": item["dataset_label"],
                "audio_key": key,
                "audio": str(audio_path),
                "representative_motion": item["representative_motion"],
                "source_motion": str(source_motion),
                "source_motion_sha256": sha256_file(source_motion),
                "generated_source_artifact": str(generated_artifact_source),
                "generated_source_sha256": sha256_file(generated_artifact_source),
                "source_frames": int(
                    original_artifact.get("source_num_frames", len(original_qpos))
                ),
                "source_fps": int(original_artifact.get("source_fps", 50)),
                "source_frames_50hz": original_artifact.get("source_frames_50hz"),
                "original_frames_30hz": len(original_qpos),
                "generated_full_frames_30hz": len(generated_qpos),
                "comparison_frames_30hz": comparison_frames,
                "comparison_duration_sec": comparison_duration,
                "original_video_duration_sec": original_video_duration,
                "generated_duration_sec": generated_duration,
                "root_orientation_postprocess": generated_payload.get(
                    "root_orientation_postprocess"
                ),
                "source_audio_duration_sec": source_audio_duration,
                "source_clip_shorter_than_audio": comparison_duration + 0.15 < generated_duration,
                "original_artifact_relative": relative(artifact_path, output_root),
                "original_video_relative": relative(original_video, output_root),
                "generated_video_relative": relative(generated_video, output_root),
                "report_relative": relative(report_path, output_root),
                "generated_materialization": generated_materialization,
                "original_media": original_probe,
                "generated_media": generated_probe,
                "joint_limit_tolerance_rad": args.joint_limit_tolerance_rad,
                "original_joint_limits": analyze_joint_limits(
                    original_overlap,
                    kinematics,
                    tolerance_rad=args.joint_limit_tolerance_rad,
                ),
                "generated_overlap_joint_limits": analyze_joint_limits(
                    generated_qpos[:comparison_frames],
                    kinematics,
                    tolerance_rad=args.joint_limit_tolerance_rad,
                ),
                "original_metrics": original_metrics,
                "generated_overlap_metrics": generated_metrics,
                "elapsed_seconds": time.perf_counter() - started,
                "scope": "离线GMR BUMI与模型生成运动学对比；不是GMT动力学结论",
            }
        except Exception as exc:
            report = {
                "contract_version": "genmo.bumi_hq_original_comparison_sample.v2",
                "status": "failed",
                "dataset": dataset,
                "dataset_label": item["dataset_label"],
                "audio_key": key,
                "source_motion_sha256": sha256_file(source_motion),
                "generated_source_sha256": sha256_file(generated_artifact_source),
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": time.perf_counter() - started,
            }
            print(f"[{number:02d}/{total:02d}] 失败：{report['error']}", flush=True)
        atomic_json(report_path, report)
        results.append(report)
        summary = summarize(results)
        atomic_json(output_root / "summary.json", summary)
        atomic_json(
            output_root / "site_data.json",
            build_site_data(results, summary, items=items, model_label=model_label),
        )
        build_index(output_root, results, summary, model_label=model_label)
    summary = summarize(results)
    atomic_json(output_root / "summary.json", summary)
    atomic_json(
        output_root / "site_data.json",
        build_site_data(results, summary, items=items, model_label=model_label),
    )
    build_index(output_root, results, summary, model_label=model_label)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
