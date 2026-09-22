#!/usr/bin/env python3
"""BUMI qpos28离线渲染与UMR质量报告的视频复核入口。

保留原单动作artifact入口，增加--quality-report模式：流式分析完整质量报告，选择
高/低质量动作各N条，以同一真实MJCF渲染双视角合集。视频叠加来源、指标、异常帧
标志与章节编号，保持原30Hz完整动作，不做贴地、平滑、时间裁剪或动力学推进。
使用PyAV逐帧写入H.264，避免在内存保存整个合集；输出经解码帧数校验后原子发布。
资产、选中机器人与人体SHA必须匹配原报告，历史配置从Git按SHA恢复为审计附件。
--quality-groups可选high_quality/low_quality/review；待复核组使用黄色明确标注，
沿用完整动作与文本身份核验，不改变筛选状态、不将REVIEW写入训练数据。
BONES-SEED使用同一源验证器核对50到30Hz时间线并显示原始文本；可额外逐条导出
完整视频与未修改的原生NPZ，独立视频逐个解码验证帧数后随合集原子发布。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_qpos(path: Path, key: str) -> tuple[np.ndarray, int, dict[str, Any]]:
    payload = torch_load(path)
    if not isinstance(payload, dict):
        raise ValueError(f"BUMI motion artifact must be a dictionary: {path}")
    expected = {
        "robot_name": "bumi",
        "quaternion_convention": "wxyz",
        "qpos_order": "mujoco_native",
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"{path}: {field} must be {value!r}, got {payload.get(field)!r}")
    qpos = torch.as_tensor(payload.get(key)).detach().cpu().double()
    if qpos.ndim != 2 or qpos.shape[1] != 28 or qpos.shape[0] <= 0:
        raise ValueError(f"{path}: {key} must have shape [T,28], got {qpos.shape}")
    if not bool(torch.isfinite(qpos).all()):
        raise ValueError(f"{path}: {key} contains NaN or Inf")
    norm = torch.linalg.vector_norm(qpos[:, 3:7], dim=-1)
    if float((norm - 1.0).abs().max()) > 1.0e-3:
        raise ValueError(f"{path}: root quaternion is not normalized wxyz")
    return qpos.numpy(), int(payload.get("fps", 30)), payload


def mujoco_joint_order(model: mujoco.MjModel) -> list[str]:
    ids = []
    for joint_id in range(model.njnt):
        address = int(model.jnt_qposadr[joint_id])
        if address >= 7:
            ids.append(joint_id)
    ids.sort(key=lambda value: int(model.jnt_qposadr[value]))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, value) for value in ids]
    if any(name is None for name in names):
        raise ValueError("Every BUMI actuated MJCF joint must have a name")
    return [str(name) for name in names]


def checked_umr_qpos(row, run, model, source_engine=None):
    """直接验证原始UMR数组与双输入指纹，不依赖旧报告中误写的辅助sequence字段。"""
    from tools.data.bumi.umr_text_preprocess import read_npz
    from tools.eval.bumi_quality_review import sha256

    root = Path(run["identity"]["paths"]["input_root"]).resolve()
    motion = (root / row["relative_path"]).resolve(strict=True)
    human = Path(row["human_path"]).resolve(strict=True)
    if not motion.is_relative_to(root):
        raise ValueError("动作路径越过原数据根目录")
    if sha256(motion) != row["source_sha256"] or sha256(human) != row["human_sha256"]:
        raise ValueError(f"动作或人体在筛选后改变: {motion}")
    if run["identity"]["paths"].get("dataset") in {"humanml3d", "bones_seed", "kitml"}:
        from tools.data.bumi.umr_text_preprocess import load_umr

        if source_engine is None:
            raise ValueError("文本动作渲染必须提供报告绑定的源验证器")
        ordered, metadata = load_umr(row, run["identity"]["paths"], source_engine)
        for field in (
            "source_motion_id",
            "source_sequence_key",
            "source_file",
            "canonical_source_id",
            "frames",
        ):
            if metadata[field] != row[field]:
                raise ValueError(f"文本动作源身份与报告不同: {field}")
        row["verified_source_sequence_key"] = metadata["source_sequence_key"]
        row["verified_robot_xml"] = str(source_engine.xml)
        if metadata.get("dataset") == "bones_seed":
            from tools.data.bumi.umr_text_preprocess import bones_text_catalog

            if not hasattr(source_engine, "bones_catalog"):
                source_engine.bones_catalog = bones_text_catalog(
                    run["identity"]["paths"]["metadata_csv"]
                )
            row["verified_captions"] = source_engine.bones_catalog.get(
                metadata["source_motion_id"], {}
            ).get("captions", [])
        elif metadata.get("dataset") == "kitml":
            row["verified_captions"] = source_engine.kitml_catalog[metadata["source_motion_id"]][
                "captions"
            ]
        else:
            row["verified_captions"] = source_engine.humanml_catalog[1][
                metadata["source_motion_id"]
            ]["captions"]
        names, expected = list(source_engine.kin.joint_order), mujoco_joint_order(model)
        if len(names) != 21 or set(names) != set(expected):
            raise ValueError("文本动作机器人关节名称不匹配")
        return np.concatenate(
            (ordered[:, :7], ordered[:, [names.index(n) + 7 for n in expected]]), axis=1
        )
    data = read_npz(motion, object_names=True)
    qpos = data["qpos"]
    if (
        qpos.shape != (row["frames"], 28)
        or qpos.dtype != np.float32
        or float(data["fps"].item()) != 30
        or not np.array_equal(data["frame_ids"], np.arange(len(qpos)))
        or np.max(np.abs(np.linalg.norm(qpos[:, 3:7], axis=1) - 1)) > 1e-3
    ):
        raise ValueError("UMR qpos/FPS/时间线/四元数合同错误")
    if (
        Path(str(data["source_data"].item())).resolve() != human
        or data["source_sequence_key"].item() != human.stem
        or Path(str(data["robot_xml"].item())).resolve()
        != Path(run["identity"]["paths"]["robot_xml"]).resolve()
        or data["robot_name"].item() != "bumi3"
        or data["source_format"].item() != "smplx_npz"
    ):
        raise ValueError("机器人NPZ原始身份与源文件不一致")
    source = read_npz(human)
    if source["source_file"].item() != row["source_file"]:
        raise ValueError("源人体的原始数据集来源与质量报告不符")
    names, expected = list(data["robot_joint_names"]), mujoco_joint_order(model)
    if len(names) != 21 or set(names) != set(expected):
        raise ValueError("机器人关节名称不匹配")
    row["verified_source_sequence_key"] = str(data["source_sequence_key"].item())
    row["verified_robot_xml"] = str(data["robot_xml"].item())
    return np.concatenate((qpos[:, :7], qpos[:, [names.index(n) + 7 for n in expected]]), axis=1)


def archive_config(run, output):
    """按报告SHA找到配置原文，避免用更新后的配置解释历史判定。"""
    path = Path(run["identity"]["paths"]["config"])
    expected = run["identity"]["config_sha256"]
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        relative = "configs/bumi/quality_filter_umr_text_30hz_v1.yaml"
        commits = subprocess.check_output(
            ["git", "log", "--all", "--format=%H", "--", relative], cwd=ROOT, text=True
        ).splitlines()
        for commit in commits:
            data = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
            if hashlib.sha256(data).hexdigest() == expected:
                break
        else:
            raise ValueError("无法从Git恢复报告绑定的原筛选配置")
    (output / "filter_config.yaml").write_bytes(data)


def montage_frames(qpos, row, group, index, count, model, data, renderer, width, height, font_path):
    from PIL import Image, ImageDraw, ImageFont

    heading = ImageFont.truetype(str(font_path), 27)
    font = ImageFont.truetype(str(font_path), 19)
    small = ImageFont.truetype(str(font_path), 17)
    cameras = [mujoco.MjvCamera(), mujoco.MjvCamera()]
    ground_id = model.geom("ground").id
    bottom, top = min(float(qpos[:, 2].min()) - 0.55, 0), max(float(qpos[:, 2].max()) + 0.65, 1)
    target_z = (bottom + top) / 2 if bottom < -0.65 or top > 1.8 else 0.48
    distance = max(2.25, (top - bottom) * 1.5)
    for camera, azimuth in zip(cameras, (135, 225)):
        camera.distance, camera.azimuth, camera.elevation = distance, azimuth, -17
    color = {
        "high_quality": (58, 220, 146),
        "low_quality": (255, 113, 105),
        "review": (255, 201, 74),
    }[group]
    m = row["review_metrics"]
    panel_height = height - 190
    for frame_index, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        views = []
        for view, camera in enumerate(cameras):
            camera.lookat[:] = (pose[0], pose[1], target_z)
            renderer.update_scene(data, camera=camera)
            if view == 1:
                for geom in renderer.scene.geoms[: renderer.scene.ngeom]:
                    if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and geom.objid == ground_id:
                        geom.rgba[3] = 0.22
            views.append(renderer.render().copy())
        canvas = Image.new("RGB", (width, height), (16, 23, 35))
        canvas.paste(Image.fromarray(np.concatenate(views, axis=1)), (0, 90))
        draw = ImageDraw.Draw(canvas)
        name = {
            "high_quality": "HIGH QUALITY / PASS",
            "low_quality": "LOW QUALITY / REJECT",
            "review": "REVIEW / NOT IN TRAINING",
        }[group]
        draw.text(
            (20, 10),
            f"{dict(humanml3d='HumanML3D', bones_seed='BONES-SEED-SMPL', kitml='KIT-ML').get(row.get('dataset'), 'MotionMillion')}    {name}    {index + 1:02d}/{count:02d}",
            font=heading,
            fill=color,
        )
        caption = row["source_motion_id"]
        if row.get("verified_captions"):
            caption += "  |  " + row["verified_captions"][0]["caption"]
        while draw.textlength(caption, font=font) > width - 45:
            caption = caption[:-4] + "..."
        draw.text((20, 49), caption, font=font, fill=(235, 240, 248))
        draw.line((width // 2, 90, width // 2, 90 + panel_height), fill=(80, 90, 110), width=2)
        draw.text(
            (16, 99),
            "VIEW A",
            font=small,
            fill=(255, 255, 255),
            stroke_width=1,
            stroke_fill=(20, 20, 20),
        )
        draw.text(
            (width // 2 + 16, 99),
            "VIEW B / transparent ground",
            font=small,
            fill=(255, 255, 255),
            stroke_width=1,
            stroke_fill=(20, 20, 20),
        )
        active = [
            reason
            for reason, intervals in row.get("issue_intervals", {}).items()
            if any(a <= frame_index < b for a, b in intervals)
        ]
        y = height - 93
        draw.text(
            (20, y),
            f"{row['selection_category']}  |  frame {frame_index + 1}/{len(qpos)}  |  30 FPS, original motion",
            font=font,
            fill=color,
        )
        line = (
            f"slide p95/max: {m['foot_slide_p95_m_s']:.2f}/{m['foot_slide_max_m_s']:.2f} m/s   "
            f"penetration: {m['penetration_m'] * 100:.1f} cm   collision extra: {m['collision_extra_m'] * 100:.1f} cm"
        )
        if row["selection_category"] == "motion_discontinuity":
            line = (
                f"root speed max: {m['root_speed_max_m_s']:.2f} m/s   "
                f"root angular speed max: {m['root_angular_speed_max_rad_s']:.2f} rad/s   "
                f"joint speed p95: {m['joint_speed_p95_rad_s']:.2f} rad/s"
            )
        if row["selection_category"] == "root_tilt":
            line = f"root tilt p95: {m['root_tilt_p95_deg']:.1f} deg  |  reject: >30 deg for >=15 frames"
        if row["selection_category"] == "airborne":
            intervals = row.get("issue_intervals", {}).get("LONG_AIRBORNE_REVIEW", [])
            longest = max((b - a for a, b in intervals), default=0)
            line = f"flagged both-feet airborne: longest {longest} frames ({longest / 30:.2f}s)  |  review: >=30 frames"
        draw.text((20, y + 30), line, font=small, fill=(222, 228, 237))
        status = (
            ("ISSUE NOW: " + " | ".join(active)) if active else "No flagged issue on this frame"
        )
        draw.text(
            (20, y + 60),
            status[:125],
            font=small,
            fill=(255, 180, 115) if active else (156, 174, 193),
        )
        if active:
            draw.rectangle(
                (1, 90, width - 2, height - 102),
                outline=color if group == "review" else (230, 74, 65),
                width=3,
            )
        yield np.asarray(canvas)


def render_montage(rows, group, run, model, path, args):
    import av

    data = mujoco.MjData(model)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width // 2)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height - 190)
    renderer = mujoco.Renderer(model, height=args.height - 190, width=args.width // 2)
    total, chapters = 0, []
    try:
        with av.open(str(path), mode="w", options={"movflags": "+faststart"}) as container:
            stream = container.add_stream("libx264", rate=30)
            stream.width, stream.height, stream.pix_fmt = args.width, args.height, "yuv420p"
            stream.options = {"crf": "23", "preset": "fast"}
            stream.thread_count = 4
            for index, row in enumerate(rows):
                qpos = checked_umr_qpos(row, run, model, getattr(args, "source_engine", None))
                start = total
                for pixels in montage_frames(
                    qpos,
                    row,
                    group,
                    index,
                    len(rows),
                    model,
                    data,
                    renderer,
                    args.width,
                    args.height,
                    args.font,
                ):
                    frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                    total += 1
                chapters.append(
                    dict(
                        index=index + 1,
                        source_motion_id=row["source_motion_id"],
                        start_seconds=start / 30,
                        end_seconds=total / 30,
                        frames=len(qpos),
                        category=row["selection_category"],
                    )
                )
                print(
                    json.dumps(
                        dict(stage="render", group=group, completed=index + 1, total=len(rows))
                    ),
                    flush=True,
                )
            for packet in stream.encode():
                container.mux(packet)
    finally:
        renderer.close()
    with av.open(str(path)) as container:
        video = container.streams.video[0]
        if (video.width, video.height, video.average_rate) != (args.width, args.height, 30):
            raise ValueError("输出视频尺寸/FPS不符")
        decoded = sum(1 for _ in container.decode(video=0))
    if decoded != total:
        raise ValueError("输出解码帧数与完整动作总帧数不同")
    return dict(
        path=path.name, frames=total, fps=30, duration_seconds=total / 30, chapters=chapters
    )


def export_review_clips(montage, chapters, rows, paths):
    """顺序解码合集并按已验证章节精确分片，保存完整视频及对应原始轨迹。"""
    import av

    from tools.eval.bumi_quality_review import sha256

    directory = montage.parent / (
        "pass"
        if montage.name.startswith("high_quality")
        else "reject"
        if montage.name.startswith("low_quality")
        else "review"
    )
    directory.mkdir()
    outputs = []
    with av.open(str(montage)) as source_video:
        frames = iter(source_video.decode(video=0))
        for chapter, row in zip(chapters, rows, strict=True):
            stem = f"{chapter['index']:02d}_{row['source_motion_id'].replace('/', '__')}"
            video_path, motion_path = directory / (stem + ".mp4"), directory / (stem + ".npz")
            with av.open(str(video_path), mode="w", options={"movflags": "+faststart"}) as out:
                stream = out.add_stream("libx264", rate=30)
                original = source_video.streams.video[0]
                stream.width, stream.height, stream.pix_fmt = (
                    original.width,
                    original.height,
                    "yuv420p",
                )
                stream.options = {"crf": "20", "preset": "fast"}
                stream.thread_count = 4
                for _ in range(chapter["frames"]):
                    frame = av.VideoFrame.from_ndarray(
                        next(frames).to_ndarray(format="rgb24"), format="rgb24"
                    )
                    for packet in stream.encode(frame):
                        out.mux(packet)
                for packet in stream.encode():
                    out.mux(packet)
            with av.open(str(video_path)) as check:
                if (
                    check.streams.video[0].average_rate != 30
                    or sum(1 for _ in check.decode(video=0)) != chapter["frames"]
                ):
                    raise ValueError("独立视频FPS或完整帧数错误")
            shutil.copy2(Path(paths["input_root"]) / row["relative_path"], motion_path)
            if sha256(motion_path) != row["source_sha256"]:
                raise ValueError("独立视频配套轨迹与筛选SHA不同")
            outputs.append(
                dict(
                    video=video_path.relative_to(montage.parent).as_posix(),
                    motion=motion_path.relative_to(montage.parent).as_posix(),
                    frames=chapter["frames"],
                    video_sha256=sha256(video_path),
                    motion_sha256=row["source_sha256"],
                )
            )
        if next(frames, None) is not None:
            raise ValueError("合集包含未被章节覆盖的额外帧")
    return outputs


def render_quality_review(args):
    from tools.data.bumi.umr_text_preprocess import source_contract_hashes
    from tools.data.bumi.umr_text_quality import QualityEngine, load_rules, verify_asset_files
    from tools.eval.bumi_quality_review import analyze_report, sha256, write_analysis_markdown

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("使用新的复核目录；验收新视频后再替换旧结果")
    if (
        args.width < 960
        or args.height < 540
        or args.width % 4
        or args.height % 2
        or args.per_group < 1
    ):
        raise ValueError("合集至少960×540，宽度须为4的倍数，数量须为正")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.staging-", dir=output.parent) as temp:
        staged = Path(temp) / "review"
        staged.mkdir()
        analysis, run = analyze_report(args.quality_report, args.per_group, args.quality_groups)
        if analysis.get("text_binding"):
            from tools.data.bumi.motionmillion_text import TextCatalog

            binding = analysis["text_binding"]
            catalog = TextCatalog(binding["text_catalog"])
            try:
                if (
                    catalog.metadata["database_sha256"] != binding["database_sha256"]
                    or sha256(catalog.root / "metadata.json") != binding["metadata_sha256"]
                ):
                    raise ValueError("文本目录与报告绑定SHA不符")
                for rows in analysis["groups"].values():
                    for row in rows:
                        text = catalog.lookup(row)
                        if text:
                            row["verified_captions"] = [{"caption": c} for c in text["captions"]]
                            row["verified_text_sha256"] = text["text_sha256"]
                            row["official_split"] = text["official_split"]
            finally:
                catalog.close()
        paths, assets = run["identity"]["paths"], run["identity"]["assets"]
        if (
            verify_asset_files(
                paths["robot_xml"], paths["kinematics"], paths["asset_manifest"], assets
            )
            != assets
        ):
            raise ValueError("机器人资产已改变")
        archive_config(run, staged)
        if source_contract_hashes(paths) != run["identity"].get("source_contract_hashes", {}):
            raise ValueError("报告绑定的源清单或文本已改变")
        args.source_engine = None
        if paths.get("dataset") in {"humanml3d", "bones_seed", "kitml"}:
            for field in ("retarget_config", "batch_config"):
                if paths.get(field) and sha256(paths[field]) != run["identity"][field + "_sha256"]:
                    raise ValueError(f"报告绑定的配置已改变: {field}")
            args.source_engine = QualityEngine(
                load_rules(staged / "filter_config.yaml"),
                paths["robot_xml"],
                paths["kinematics"],
                paths["retarget_config"],
                paths["batch_config"],
            )
        model = mujoco.MjModel.from_xml_path(paths["robot_xml"])
        if model.nq != 28:
            raise ValueError("MJCF不是qpos28机器人")
        analysis["videos"] = {}
        # 在正式渲染前保存分析快照，供同一任务观察进度；失败时随staging清理。
        (staged / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2))
        write_analysis_markdown(analysis, staged / "分析报告.md")
        for group, rows in analysis["groups"].items():
            name = f"{group}_{args.per_group}.mp4"
            result = render_montage(rows, group, run, model, staged / name, args)
            result.update(sha256=sha256(staged / name), bytes=(staged / name).stat().st_size)
            if getattr(args, "individual_clips", False):
                result["individual_clips"] = export_review_clips(
                    staged / name, result["chapters"], rows, paths
                )
            analysis["videos"][group] = result
        analysis["render_contract"] = dict(
            robot_xml=paths["robot_xml"],
            assets=assets,
            coordinate_system="z_up",
            qpos_modified=False,
            crop_count=0,
            fps=30,
            views=[135, 225],
            camera_follow="root XY only",
            physics="mj_forward only; no dynamics",
            code_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        )
        for group in analysis["groups"].values():
            for row in group:
                path = Path(paths["input_root"]) / row["relative_path"]
                if (
                    sha256(path) != row["source_sha256"]
                    or sha256(row["human_path"]) != row["human_sha256"]
                ):
                    raise ValueError("渲染期间源动作改变")
        if source_contract_hashes(paths) != run["identity"].get("source_contract_hashes", {}):
            raise ValueError("渲染期间源清单或文本改变")
        (staged / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2))
        (staged / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2))
        write_analysis_markdown(analysis, staged / "分析报告.md")
        (staged / "quality_summary.json").write_bytes(
            (args.quality_report / "quality_summary.json").read_bytes()
        )
        staged.rename(output)
    print(
        json.dumps(
            dict(output=str(output), videos=analysis["videos"]), ensure_ascii=False, indent=2
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path)
    parser.add_argument("--mjcf", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quality-report", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--per-group", type=int, default=30)
    parser.add_argument(
        "--individual-clips", action="store_true", help="同时导出每条完整视频和原生NPZ轨迹"
    )
    parser.add_argument(
        "--quality-groups",
        nargs="+",
        choices=["high_quality", "low_quality", "review"],
        default=["high_quality", "low_quality"],
        help="指定复核组；默认仍渲染PASS/REJECT两组",
    )
    parser.add_argument(
        "--font", type=Path, default=Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    )
    parser.add_argument("--qpos-key", default="qpos")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera")
    args = parser.parse_args()
    if args.quality_report:
        if not args.output_dir or args.motion or args.mjcf or args.output:
            parser.error("质量复核模式需要--quality-report/--output-dir，不混用单动作参数")
        render_quality_review(args)
        return
    if not (args.motion and args.mjcf and args.output):
        parser.error("单动作模式需要--motion/--mjcf/--output")
    qpos, fps, payload = load_qpos(args.motion.expanduser().resolve(), args.qpos_key)
    if fps != 30:
        raise ValueError(f"BUMI render input must be 30 FPS, got {fps}")
    model = mujoco.MjModel.from_xml_path(str(args.mjcf.expanduser().resolve()))
    if model.nq != 28:
        raise ValueError(f"BUMI MJCF nq must be 28, got {model.nq}")
    artifact_joint_names = tuple(map(str, payload.get("joint_names", ())))
    mjcf_joint_names = tuple(mujoco_joint_order(model))
    if artifact_joint_names != mjcf_joint_names:
        raise ValueError(
            "Motion joint_names do not exactly match MuJoCo-native MJCF qpos order; "
            "rendering refuses to guess a reorder"
        )
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    frames = []
    try:
        for frame_qpos in qpos:
            data.qpos[:] = frame_qpos
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=args.camera)
            frames.append(renderer.render().copy())
    finally:
        renderer.close()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output, np.stack(frames), fps=fps)
    print(f"Rendered {len(frames)} frames to {output}")


if __name__ == "__main__":
    main()
