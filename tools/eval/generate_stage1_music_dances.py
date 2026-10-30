#!/usr/bin/env python3
"""用第一阶段 BUMI 模型为任意完整音乐生成舞蹈、轨迹和带音乐的视频。

本入口复用训练模型加载器、Stage1 ONNX DDIM 运行器、严格因果 proprio48 历史构造器、
qpos30 编解码和既有 MuJoCo 视频渲染器，不使用旧纯音乐模型的五输入推理接口。运行前
核对 checkpoint、训练统计量、运动学、ONNX 元数据和源 MJCF 的 SHA256，确保生成与
渲染使用同一机器人资产。音乐通过既有 librosa EDGE35 提取器按完整解码时长处理，
轨迹取 ceil(时长×30) 帧，末尾不足一帧的时间明确保留，音视频不会裁成短演示。

新音乐没有动作示范：前 12 帧为按机器人真实脚底几何贴地的标准零位站姿，随后每个
120 帧窗口固定已有 12 帧重叠前缀，按 108 帧步长生成；历史只读取决策时刻及以前
自身生成的姿态。每个窗口记录独立噪声种子、前缀误差和帧范围，不复制任何训练动作。
初始站姿的接触输出是有限零占位，必须结合 contact_valid=false 使用，不作为预测。

每首音乐导出原生 qpos28 的 PT、NPZ、可读 CSV、完整 EDGE35 特征及生成元数据。
NPZ 另外保留 21 关节位置和速度、22 刚体世界位姿、接触概率和有序名称，便于后续
按名称映射；速度是离线有限差分，刚体位姿来自 FK，均不是动力学测量。原始生成动作
不做关节裁剪、足锁或地面平移。视频由既有渲染器逐帧 mj_forward，跟随根位置并配
原音频，属于运动学可视化。最后核验导出一致性、视频帧数、编码和时长，生成浏览索引。
输出目录必须不存在，避免覆盖先前交付；模型和已有工作树均只读。
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from gem.closedloop.stage1_dataset import CausalDemoProprio48Builder  # noqa: E402
from gem.closedloop.training import batch_to_device  # noqa: E402
from gem.robots.bumi.feature_codec import make_quaternion_continuous  # noqa: E402
from gem.robots.bumi.metrics import compute_bumi_kinematic_metrics  # noqa: E402
from gem.runtime.closedloop_stage1_onnx import CONTRACT_VERSION, Stage1OnnxSampler  # noqa: E402
from gem.utils.music_features import extract_edge_baseline35  # noqa: E402
from tools.eval.validate_closedloop_stage1 import (  # noqa: E402
    load_actor,
    render_video,
    sha256_file,
    world_qpos,
    write_json,
)


@torch.inference_mode()
def generate(actor, runtime, music, total, seed, directory):
    """以标准站姿启动，复用第一阶段已有 12 帧前缀的自主滚动语义。"""
    codec = actor.endecoder.codec
    kin = codec.kinematics
    builder = CausalDemoProprio48Builder(kin)
    generated = kin.make_standing_qpos().cpu().repeat(min(12, total), 1)
    contacts = torch.zeros(len(generated), 2)
    contact_valid = torch.zeros(len(generated), dtype=torch.bool)
    windows = []
    decision = 0
    device = next(actor.parameters()).device
    while len(generated) < total:
        count = min(120, total - decision)
        prefix = generated[decision : decision + 12]
        encoded = codec.encode(prefix)
        known = torch.zeros(120, 30)
        mask = torch.zeros(120, 30, dtype=torch.bool)
        mask[: len(prefix), 2:] = True
        mask[: len(prefix) - 1, :2] = True
        known[: len(prefix)] = encoded.physical_features
        known[~mask] = 0.0
        history, history_valid, history_times = builder.build_history(
            generated[: decision + 1], decision_frame=decision, history_steps=50
        )
        music_window = torch.zeros(120, 35)
        # sim2sim 可多生成少量真实模型尾帧供十帧 Goal/导数 halo 使用。
        # 音乐结束后 mask 为假，不复制末帧音乐，不外推名义动作。
        music_count = max(0, min(count, len(music)-decision))
        music_window[:music_count] = music[decision : decision + music_count]
        valid = torch.arange(120) < count
        conditions = batch_to_device(
            {
                "music_features": music_window[None],
                "music_valid": (torch.arange(120) < music_count)[None],
                "proprio_history": history[None],
                "proprio_history_valid": history_valid[None],
                "proprio_history_times": history_times[None],
                "known_qpos30": known[None],
                "known_qpos30_mask": mask[None],
                "future_valid": valid[None],
                "future_times": ((decision + torch.arange(120, dtype=torch.float64)) / 30)[None],
                "decision_time": torch.tensor([decision / 30], dtype=torch.float64),
            },
            device,
        )
        window_seed = seed + len(windows)
        noise = torch.randn((1, 120, 30), generator=torch.Generator().manual_seed(window_seed))
        result = runtime.sample(conditions, steps=20, guidance_scale=2.5, noise=noise.to(device))
        error = float(
            (
                result["qpos30"][conditions["known_qpos30_mask"]]
                - known[None].to(device)[conditions["known_qpos30_mask"]]
            )
            .abs()
            .max()
        )
        if error != 0.0:
            raise RuntimeError(f"已有动作前缀发生变化：{error}")
        segment = world_qpos(codec, result["qpos"][0, :count].cpu(), encoded.anchor)
        generated = torch.cat((generated, segment[len(prefix) :]), 0)
        contacts = torch.cat((contacts, result["contact_logits"][0, len(prefix) : count].cpu()), 0)
        contact_valid = torch.cat(
            (contact_valid, torch.ones(count - len(prefix), dtype=torch.bool))
        )
        windows.append(
            {
                "decision_frame": decision,
                "valid_frames": count,
                "seed": window_seed,
                "prefix_max_abs_error": error,
            }
        )
        decision = len(generated) - 12
        print(
            f"{directory.name}: 窗口 {len(windows)}，已生成 {len(generated)}/{total} 帧", flush=True
        )
    generated[:, 3:7] = make_quaternion_continuous(generated[:, 3:7])
    if not torch.isfinite(generated).all() or not torch.isfinite(contacts).all():
        raise FloatingPointError("轨迹或接触输出含非有限数值")
    return generated, contacts, contact_valid, windows


def export_trajectory(directory, actor, qpos, logits, contact_valid, metadata):
    """保存原始世界坐标轨迹，回读 NPZ/CSV 检查帧数、列序和数值一致性。"""
    kin = actor.endecoder.codec.kinematics
    joint_names = list(kin.joint_order)
    body_names = list(kin.body_order)
    fk = kin.forward_kinematics(qpos)
    probability = logits.sigmoid()
    probability[~contact_valid] = 0.0
    payload = {
        "robot_name": "bumi",
        "quaternion_convention": "wxyz",
        "qpos_order": "mujoco_native",
        "fps": 30,
        "qpos": qpos,
        "joint_names": joint_names,
        "foot_contact_logits": logits,
        "foot_contact": probability,
        "contact_valid": contact_valid,
        "metadata": metadata,
    }
    torch.save(payload, directory / "trajectory.pt")
    arrays = {
        "qpos": qpos.numpy(),
        "time_s": np.arange(len(qpos), dtype=np.float64) / 30,
        "fps": np.asarray(30, dtype=np.int64),
        "robot_name": np.asarray("bumi"),
        "quaternion_convention": np.asarray("wxyz"),
        "qpos_order": np.asarray("mujoco_native"),
        "joint_names": np.asarray(joint_names),
        "body_names": np.asarray(body_names),
        "joint_pos": qpos[:, 7:].numpy(),
        "joint_vel": np.gradient(qpos[:, 7:].numpy(), 1 / 30, axis=0),
        "body_pos_w": fk["body_pos_w"].numpy(),
        "body_quat_w": fk["body_quat_w"].numpy(),
        "foot_contact_logits": logits.numpy(),
        "foot_contact": probability.numpy(),
        "contact_valid": contact_valid.numpy(),
        "metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False)),
    }
    np.savez_compressed(directory / "trajectory.npz", **arrays)
    csv_array = np.column_stack(
        (arrays["time_s"], arrays["qpos"], arrays["foot_contact"], arrays["contact_valid"])
    )
    header = [
        "time_s",
        "root_x_m",
        "root_y_m",
        "root_z_m",
        "root_qw",
        "root_qx",
        "root_qy",
        "root_qz",
        *joint_names,
        "left_contact_probability",
        "right_contact_probability",
        "contact_valid",
    ]
    with (directory / "trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(csv_array)
    with np.load(directory / "trajectory.npz", allow_pickle=False) as loaded:
        if not np.array_equal(loaded["qpos"], arrays["qpos"]):
            raise RuntimeError("NPZ 回读与源轨迹不一致")
        if loaded["body_pos_w"].shape != (len(qpos), 22, 3):
            raise RuntimeError("FK 刚体导出形状错误")
        for name in ("qpos", "joint_vel", "body_pos_w", "body_quat_w", "foot_contact_logits"):
            if not np.isfinite(loaded[name]).all():
                raise FloatingPointError(f"NPZ 字段非有限：{name}")
    restored_csv = np.loadtxt(directory / "trajectory.csv", delimiter=",", skiprows=1)
    if restored_csv.shape != csv_array.shape or not np.array_equal(restored_csv, csv_array):
        raise RuntimeError("CSV 回读与源轨迹不一致")
    metrics = compute_bumi_kinematic_metrics(qpos, kin, ground_height=0.0)
    return {
        "frames": len(qpos),
        "quaternion_norm_max_error": float((qpos[:, 3:7].norm(dim=-1) - 1).abs().max()),
        "npz_csv_exact_roundtrip": True,
        "metrics": {key: float(value) for key, value in metrics.items()},
    }


def verify_video(path, frames):
    """使用 ffprobe 实际读取视频帧数及音轨，拒绝短片和编码错误。"""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    probe = json.loads(result.stdout)
    video = next(stream for stream in probe["streams"] if stream["codec_type"] == "video")
    audio = next(stream for stream in probe["streams"] if stream["codec_type"] == "audio")
    if int(video["nb_read_frames"]) != frames or video["avg_frame_rate"] != "30/1":
        raise RuntimeError("视频帧数或帧率与完整轨迹不一致")
    if video["codec_name"] != "h264" or audio["codec_name"] != "aac":
        raise RuntimeError("视频必须为 H.264/AAC")
    if abs(float(video["duration"]) - frames / 30) > 1 / 30:
        raise RuntimeError("视频时长与轨迹不一致")
    return {
        "frames": int(video["nb_read_frames"]),
        "fps": 30,
        "width": video["width"],
        "height": video["height"],
        "duration_sec": float(video["duration"]),
        "video_codec": video["codec_name"],
        "audio_codec": audio["codec_name"],
    }


def write_index(output, songs, model_step, seed):
    """生成只引用本目录实际成品的中文视频与轨迹浏览索引。"""
    cards = []
    for song in songs:
        name = html.escape(song["name"])
        directory = song["directory"]
        cards.append(
            f'<article><h2>{name}</h2><p>{song["frames"]} 帧 · {song["frames"] / 30:.2f} 秒 · 30 FPS</p><video controls preload="metadata" src="{directory}/dance.mp4"></video><p><a href="{directory}/dance.mp4">下载视频</a>　<a href="{directory}/trajectory.npz">NPZ 轨迹</a>　<a href="{directory}/trajectory.csv">CSV 轨迹</a>　<a href="{directory}/metadata.json">生成信息</a></p></article>'
        )
    (output / "index.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第一阶段舞蹈生成</title><style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:36px auto;padding:0 20px;background:#f6f7fa;color:#182131}article{background:white;border-radius:14px;padding:24px;margin:24px 0}video{width:100%;max-width:960px}a{color:#1455c0}</style>'
        + f'<h1>{len(songs)} 首音乐的第一阶段舞蹈生成</h1><p>模型：Stage1 s{model_step}；完整音乐，DDIM20 / CFG2.5 / seed{seed}。前0.4秒标准站姿启动，随后自主滚动生成。视频为运动学渲染。</p>'
        + "".join(cards)
        + "</html>",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "stats", "kinematics", "onnx", "mjcf", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--audio", type=Path, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for name in ("checkpoint", "stats", "kinematics", "onnx", "mjcf"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    args.audio = [path.resolve(strict=True) for path in args.audio]
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    actor, _, summary = load_actor(args.checkpoint, args.stats, args.kinematics, "cuda")
    identity = {
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "stats_sha256": sha256_file(args.stats),
        "kinematics_sha256": sha256_file(args.kinematics),
        "onnx_sha256": sha256_file(args.onnx),
        "mjcf_sha256": sha256_file(args.mjcf),
    }
    for name in ("stats", "kinematics"):
        if identity[f"{name}_sha256"] != summary["asset_identity"][f"{name}_sha256"]:
            raise RuntimeError(f"训练绑定资产不匹配：{name}")
    onnx_metadata = json.loads(args.onnx.with_suffix(args.onnx.suffix + ".json").read_text())
    if (
        onnx_metadata.get("contract_version") != CONTRACT_VERSION
        or onnx_metadata.get("checkpoint_sha256") != identity["checkpoint_sha256"]
    ):
        raise RuntimeError("ONNX 与第一阶段 checkpoint 身份不一致")
    if onnx_metadata.get("onnx_sha256") != identity["onnx_sha256"]:
        raise RuntimeError("ONNX 文件 SHA256 与导出记录不一致")
    kin_spec = json.loads(args.kinematics.read_text())
    if kin_spec["source_mjcf_sha256"] != identity["mjcf_sha256"]:
        raise RuntimeError("渲染 MJCF 与运动学源身份不一致")
    for name in ("stats", "kinematics"):
        if onnx_metadata["asset_identity"][f"{name}_sha256"] != identity[f"{name}_sha256"]:
            raise RuntimeError(f"ONNX 绑定资产不匹配：{name}")
    write_json(args.output / "checkpoint_capture.json", {**summary, "generation_assets": identity})
    runtime = Stage1OnnxSampler(args.onnx, actor, device="cuda")
    songs = []
    render_jobs = []
    with ThreadPoolExecutor(max_workers=1) as rendering:
        for number, audio in enumerate(args.audio, 1):
            start = time.monotonic()
            directory = args.output / f"{number:02d}"
            directory.mkdir()
            shutil.copy2(audio, directory / "audio.mp3")
            print(f"开始提取完整音乐特征：{audio.name}", flush=True)
            music, feature_metadata = extract_edge_baseline35(audio, target_fps=30)
            total = math.ceil(feature_metadata["selected_duration_sec"] * 30)
            if len(music) < total or total < 13:
                raise ValueError("音乐特征未覆盖完整音频，或音频不足启动段")
            torch.save(music, directory / "music_features.pt")
            qpos, logits, contact_valid, windows = generate(
                actor, runtime, music, total, args.seed, directory
            )
            metadata = {
                "name": audio.stem,
                "audio_path": str(audio),
                "audio_sha256": sha256_file(audio),
                "model_step": summary["global_step"],
                **identity,
                "feature_metadata": feature_metadata,
                "frames": total,
                "fps": 30,
                "trajectory_duration_sec": total / 30,
                "ddim_steps": 20,
                "cfg_scale": 2.5,
                "seed": args.seed,
                "startup_frames": 12,
                "startup_pose": "kinematics.make_standing_qpos_zero_joints_grounded",
                "demonstration_frames": 0,
                "history_source": "own_generated_kinematic_proxy_no_physics",
                "window_frames": 120,
                "overlap_frames": 12,
                "window_log": windows,
                "postprocessing": "quaternion_sign_continuity_only",
                "contact_order": ["left", "right"],
                "velocity_source": "offline_finite_difference_30Hz",
                "body_pose_source": "kinematic_forward_kinematics",
                "rendering": "mujoco_mj_forward_no_dynamics",
            }
            checks = export_trajectory(directory, actor, qpos, logits, contact_valid, metadata)
            metadata["generation_wall_seconds"] = time.monotonic() - start
            write_json(directory / "metadata.json", metadata)
            write_json(directory / "trajectory_checks.json", checks)
            songs.append(
                {
                    "name": audio.stem,
                    "directory": directory.name,
                    "frames": total,
                    "audio_duration_sec": feature_metadata["selected_duration_sec"],
                    "trajectory_checks": checks,
                }
            )
            render_jobs.append(
                rendering.submit(
                    render_video,
                    directory / "trajectory.pt",
                    directory / "audio.mp3",
                    args.mjcf,
                    directory / "dance.mp4",
                    directory / "render.log",
                )
            )
            print(f"{audio.name}: 轨迹导出完成，后台渲染完整视频", flush=True)
        for song, job in zip(songs, render_jobs):
            job.result()
            directory = args.output / song["directory"]
            song["video_checks"] = verify_video(directory / "dance.mp4", song["frames"])
            song["artifacts"] = {
                path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in directory.iterdir()
                if path.is_file()
            }
            print(f"{song['name']}: 视频核验通过", flush=True)
    write_index(args.output, songs, summary["global_step"], args.seed)
    write_json(
        args.output / "completion.json",
        {
            "status": "complete",
            "model_step": summary["global_step"],
            "model_sha256": identity["checkpoint_sha256"],
            "songs": songs,
        },
    )
    print(f"全部完成：{args.output}", flush=True)


if __name__ == "__main__":
    main()
