#!/usr/bin/env python3
"""构建人工高质量原动作与 BUMI 音乐模型生成动作的同步对比网页。

输入是 ``validate_bumi_hq_music_full.py`` 已冻结的可变数量选择和模型生成产物，以及人工
``score=1`` 对应的七字段 50 Hz SONIC/Isaac-Lab BUMI NPZ。工具严格复用正式数据构建器的
50→30 Hz 语义：根位置和关节线性插值、wxyz 根四元数最短弧 SLERP、Isaac publish order
按完整名称重排到 GENMO/MuJoCo-native order，并按训练契约执行 body-origin 地面归一化。

每项输出原数据集 BUMI 视频、模型生成视频的自包含硬链接/副本，以及带标签的左右同步
对比视频；模型生成源发生变化时以临时硬链接/副本原子刷新旧文件。音轨统一来自同一 WAV。
对比时长取原动作与模型验证片段的共同真实长度，因此 AIST++ 的短源片段不会被循环或拉伸，
其他长动作也不会超过模型实际生成长度。网页会明确显示源片段和生成片段时长。报告还在
相同对比区间计算双方运动学指标和 0.25 rad 限位结果，并额外生成不含本地绝对路径的
``site_data.json``，供公开验证网页直接消费。该页面展示的是 GMR 离线重定向轨迹和模型
生成轨迹，不代表 GMT/真实机器人动力学跟踪效果。
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
    DATASETS,
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


def build_comparison_video(
    *,
    original: Path,
    generated: Path,
    audio: Path,
    output: Path,
    duration_sec: float,
    model_label: str,
    width: int,
    height: int,
) -> dict[str, Any]:
    temporary = output.with_name(output.stem + ".tmp.mp4")
    temporary.unlink(missing_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    filter_graph = (
        f"[0:v]scale={width}:{height},setpts=PTS-STARTPTS,"
        "drawtext=text='Original GMR BUMI':x=18:y=18:fontsize=24:"
        "fontcolor=white:box=1:boxcolor=black@0.65[left];"
        f"[1:v]scale={width}:{height},setpts=PTS-STARTPTS,"
        f"drawtext=text='{model_label} Generated':x=18:y=18:fontsize=24:"
        "fontcolor=white:box=1:boxcolor=black@0.65[right];"
        "[left][right]hstack=inputs=2[video]"
    )
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(original),
                "-i",
                str(generated),
                "-i",
                str(audio),
                "-filter_complex",
                filter_graph,
                "-map",
                "[video]",
                "-map",
                "2:a:0",
                "-t",
                f"{duration_sec:.9f}",
                "-r",
                "30",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-shortest",
                str(temporary),
            ],
            check=True,
        )
        probe = media_probe(temporary)
        if probe["width"] != width * 2 or probe["height"] != height:
            raise RuntimeError(f"对比视频尺寸错误：{probe}")
        temporary.replace(output)
        return probe
    finally:
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
        or report.get("source_motion_sha256") != sha256_file(source_motion)
        or report.get("generated_source_sha256") != sha256_file(generated_source)
    ):
        return None
    for field in (
        "original_video_relative",
        "generated_video_relative",
        "comparison_video_relative",
    ):
        path = output_root / report.get(field, "")
        if not path.is_file():
            return None
        media_probe(path)
    return report


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    passed = [row for row in results if row.get("status") == "passed"]
    dataset_rows = {}
    for dataset, label, _ in DATASETS:
        rows = [row for row in passed if row["dataset"] == dataset]
        dataset_rows[dataset] = {
            "dataset_label": label,
            "completed": len(rows),
            "comparison_duration_sec": sum(row["comparison_duration_sec"] for row in rows),
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
        "contract_version": "genmo.bumi_hq_original_comparison_summary.v1",
        "evaluated": len(results),
        "completed": len(passed),
        "failed": len(results) - len(passed),
        "comparison_duration_sec": sum(row["comparison_duration_sec"] for row in passed),
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
                "generated_duration_sec": row["generated_duration_sec"],
                "source_audio_duration_sec": row["source_audio_duration_sec"],
                "source_clip_shorter_than_audio": row["source_clip_shorter_than_audio"],
                "comparison_video": f"/{row['comparison_video_relative']}",
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
        "contract_version": "genmo.bumi_hq_original_comparison_site.v1",
        "model_label": model_label,
        "comparison_policy": "原动作与模型生成在同一音乐、同一时间轴上并排对比；短源片段不循环或拉伸",
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
    def metric_text(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.4f}"

    sections = []
    for dataset, label, _ in DATASETS:
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
                "；原数据集只有该短片段，未循环" if row["source_clip_shorter_than_audio"] else ""
            )
            cards.append(
                f'<article class="card"><h3>{index}. {html.escape(row["audio_key"])}</h3>'
                f'<video controls preload="metadata" src="{html.escape(row["comparison_video_relative"])}"></video>'
                f"<p>左：原数据集 GMR BUMI；右：{html.escape(model_label)} 模型生成。对比 "
                f"{row['comparison_duration_sec']:.2f}s，模型验证片段 "
                f"{row['generated_duration_sec']:.2f}s{short_note}。</p>"
                "<table><thead><tr><th>同区间指标</th><th>原数据集</th><th>模型生成</th></tr></thead><tbody>"
                f"<tr><td>最大脚部穿地(m)</td><td>{original['foot_penetration_max_m']:.4f}</td><td>{generated['foot_penetration_max_m']:.4f}</td></tr>"
                f"<tr><td>最大根倾角(rad)</td><td>{original['root_tilt_max_rad']:.4f}</td><td>{generated['root_tilt_max_rad']:.4f}</td></tr>"
                f"<tr><td>关节速度P95(rad/s)</td><td>{original['joint_velocity_p95_radps']:.4f}</td><td>{generated['joint_velocity_p95_radps']:.4f}</td></tr>"
                "</tbody></table>"
                f'<p><a href="{html.escape(row["original_video_relative"])}">只看原动作</a> · '
                f'<a href="{html.escape(row["generated_video_relative"])}">只看模型动作</a> · '
                f'<a href="{html.escape(row["report_relative"])}">JSON报告</a></p></article>'
            )
        sections.append(f"<section><h2>{label}</h2>{''.join(cards)}</section>")
    metrics = summary["metric_comparison_same_interval"]
    limits = summary["joint_limit_comparison"]
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BUMI 原数据集与模型生成对比</title><style>
body{{margin:0;background:#0e1219;color:#edf1f7;font:15px/1.55 system-ui,sans-serif}}main{{max-width:1500px;margin:auto;padding:28px}}a{{color:#78c5ff}}
.summary,.card{{background:#191f2b;border:1px solid #30394a;border-radius:12px;padding:16px;margin:14px 0}}.failed{{border-color:#a74d4d}}video{{width:100%;background:#000;border-radius:8px}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:7px;border:1px solid #3a4354;text-align:left}}section{{margin-top:34px}}
@media(min-width:1100px){{section{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}section h2{{grid-column:1/-1}}.card{{margin:0}}}}</style></head><body><main>
<h1>BUMI 人工高质量原动作 vs {html.escape(model_label)} 模型生成</h1><div class="summary">
<p>完成 {summary["completed"]}/{summary["evaluated"]} 项，同步对比总时长 {summary["comparison_duration_sec"] / 60.0:.2f} 分钟。左侧是 score=1 对应的原始 GMR BUMI 轨迹，右侧是相同音乐的模型生成轨迹。</p>
<p>同区间均值：最大脚部穿地 原始 {metric_text(metrics["foot_penetration_max_m"]["original_mean"])}m / 生成 {metric_text(metrics["foot_penetration_max_m"]["generated_mean"])}m；最大根倾角 原始 {metric_text(metrics["root_tilt_max_rad"]["original_mean"])}rad / 生成 {metric_text(metrics["root_tilt_max_rad"]["generated_mean"])}rad。</p>
<p>0.25rad 容差后关节超限：原动作 {limits["original"]["exceed_0_25_rad_samples"]}/{summary["completed"]}，模型生成同区间 {limits["generated_same_interval"]["exceed_0_25_rad_samples"]}/{summary["completed"]}；严格 XML 原始边界触碰/超出分别为 {limits["original"]["strict_xml_exceed_samples"]}/{summary["completed"]} 和 {limits["generated_same_interval"]["strict_xml_exceed_samples"]}/{summary["completed"]}。</p>
<p>AIST++ 原数据本身是短动作片段，对比不会循环原动作。<a href="summary.json">汇总 JSON</a> · <a href="selection.json">选择与契约</a></p></div>
{"".join(sections)}</main></body></html>"""
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
        "contract_version": "genmo.bumi_hq_original_comparison_selection.v1",
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
        "comparison_policy": "原50Hz动作精确重采样到30Hz；对比截到原动作与模型生成的共同真实长度；不循环或拉伸短片段",
        "items": items,
    }
    atomic_json(output_root / "selection.json", frozen_selection)
    results: list[dict[str, Any]] = []
    for number, item in enumerate(items, start=1):
        dataset = item["dataset"]
        key = item["audio_key"]
        source_motion = (source_root / dataset / item["representative_motion"]).resolve(strict=True)
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
        original_video = output_root / "videos" / "original" / dataset / f"{key}.mp4"
        generated_video = output_root / "videos" / "generated" / dataset / f"{key}.mp4"
        comparison_video = output_root / "videos" / "comparison" / dataset / f"{key}.mp4"
        started = time.perf_counter()
        print(f"[{number:02d}/{total:02d}] 对比 {item['dataset_label']}/{key}", flush=True)
        try:
            _, original_qpos = save_original_artifact(
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
                duration_sec=comparison_duration,
                width=args.width,
                height=args.height,
            )
            if abs(original_probe["duration_sec"] - comparison_duration) > 0.15:
                raise RuntimeError("原动作视频时长误差超过 0.15 秒")
            generated_materialization = materialize_file(generated_video_source, generated_video)
            comparison_probe = build_comparison_video(
                original=original_video,
                generated=generated_video,
                audio=audio_path,
                output=comparison_video,
                duration_sec=comparison_duration,
                model_label=model_label,
                width=args.width,
                height=args.height,
            )
            if abs(comparison_probe["duration_sec"] - comparison_duration) > 0.15:
                raise RuntimeError("对比视频时长误差超过 0.15 秒")
            original_overlap = original_qpos[:comparison_frames]
            original_metrics = motion_metrics(original_overlap, kinematics)
            generated_metrics = motion_metrics(generated_qpos[:comparison_frames], kinematics)
            report = {
                "contract_version": "genmo.bumi_hq_original_comparison_sample.v1",
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
                "source_frames_50hz": torch_load(artifact_path)["source_frames_50hz"],
                "original_frames_30hz": len(original_qpos),
                "generated_full_frames_30hz": len(generated_qpos),
                "comparison_frames_30hz": comparison_frames,
                "comparison_duration_sec": comparison_duration,
                "generated_duration_sec": generated_duration,
                "source_audio_duration_sec": source_audio_duration,
                "source_clip_shorter_than_audio": comparison_duration + 0.15 < generated_duration,
                "original_artifact_relative": relative(artifact_path, output_root),
                "original_video_relative": relative(original_video, output_root),
                "generated_video_relative": relative(generated_video, output_root),
                "comparison_video_relative": relative(comparison_video, output_root),
                "report_relative": relative(report_path, output_root),
                "generated_materialization": generated_materialization,
                "original_media": original_probe,
                "comparison_media": comparison_probe,
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
                "contract_version": "genmo.bumi_hq_original_comparison_sample.v1",
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
