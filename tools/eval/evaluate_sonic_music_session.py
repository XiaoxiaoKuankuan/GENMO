#!/usr/bin/env python3
"""复核音乐会话的 SMPL 编码观测和真实动力学结果，支持生成可分享的带音乐录像。

观测检查独立使用 scipy 重建 1762 维 mode 2 输入：未启用模态填零，固定 SONIC
人体 FK、根朝向的机器人局部旋转和六腕数据必须与 C++ 实际输入逐元素一致。
仿真指标从保存的真实 qpos/qvel 和单调时钟计算，不将参考轨迹当成机器人动作。
可选录像按真实时间戳重绘保存的仿真状态，再接入同会话音频；它是状态回放录像，
不是重新运行控制，也不用于证明音频设备实际发声或替代人工视听复核。
"""

from __future__ import annotations

# 直接运行脚本时先建立仓库导入路径。
# ruff: noqa: E402
import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from gem.runtime.sonic_music import PoseTimeline


def evaluate(directory, render=False):
    """消费同一会话的清单、SMPL 与状态记录，输出可重复计算的证据。"""
    manifest = json.loads((directory / "manifest.json").read_text())
    epoch, duration = manifest["epoch_ns"], manifest["duration_seconds"]
    sonic = Path(manifest["args"]["sonic_root"])
    if (directory / "reference_50hz.npz").exists():
        with np.load(directory / "reference_50hz.npz") as saved:
            reference = {k: saved[k] for k in saved.files}
    else:
        timeline = PoseTimeline(sonic, duration)
        with np.load(directory / "generated_smpl.npz") as saved:
            params = {k: torch.from_numpy(saved[k]) for k in ("body_pose", "global_orient")}
        if len(params["body_pose"]) != math.ceil(duration * 30 - 1e-9):
            raise ValueError("旧会话源 SMPL 不完整，不能重建真实的未来参考")
        payload = timeline.push(params, is_last=True)
        ending = timeline.finish()
        reference = {k: np.concatenate((payload[k], ending[k])) for k in payload}
    errors, audit_count = [], 0
    for line in (directory / "timeline.jsonl").read_text().splitlines():
        row = json.loads(line)
        audit = row["sonic"].get("observation_audit")
        if not audit:
            continue
        frame = audit["frame"]
        if frame == -1 and manifest.get("resident"):
            # 常驻预约前仍消费站姿快照，对应本首前缀首帧，不能用负索引误读末尾朝向。
            indexes = np.zeros(10, dtype=int)
        else:
            indexes = frame + np.arange(10) if audit["play"] else np.repeat(frame, 10)
        expected = np.zeros(1762)
        expected[0] = 2
        expected[922:1642] = reference["smpl_joints"][indexes].ravel()

        def quat(q):
            return Rotation.from_quat(np.asarray(q)[..., [1, 2, 3, 0]])

        relative = (
            quat(audit["robot_quat"]).inv()
            * quat(audit["heading"])
            * quat(reference["body_quat"][indexes])
        )
        expected[1642:1702] = relative.as_matrix()[..., :2].ravel()
        expected[1702:] = reference["joint_pos"][indexes, 23:29].ravel()
        errors.append(float(np.max(np.abs(np.asarray(audit["encoder"]) - expected))))
        audit_count += 1
    states = [json.loads(line) for line in (directory / "sim_state.jsonl").read_text().splitlines()]
    active = [s for s in states if epoch <= s["monotonic_ns"] <= epoch + int((duration + 3) * 1e9)]
    qpos = np.asarray([s["qpos"] for s in active])
    wall = np.asarray([s["monotonic_ns"] for s in active], np.float64) * 1e-9
    sim = np.asarray([s["sim_time"] for s in active])
    result = dict(
        session_id=manifest["session_id"],
        observation_audit_count=audit_count,
        observation_max_abs=max(errors, default=None),
        observation_pass=max(errors) <= 2e-5 if errors else None,
        sample_count=len(active),
        minimum_base_height=float(qpos[:, 2].min()),
        max_base_tilt_deg=max(s["tilt_deg"] for s in active),
        band_enabled_during_playback=any(
            s["band_enabled"] for s in active if s["state"] == "playing"
        ),
        sim_time_reset=bool(np.any(np.diff(sim) <= 0)),
        physical_steps_per_wall_second=float((sim[-1] - sim[0]) / 0.005 / (wall[-1] - wall[0])),
        realtime_factor=float((sim[-1] - sim[0]) / (wall[-1] - wall[0])),
        horizontal_travel_m=float(np.linalg.norm(np.diff(qpos[:, :2], axis=0), axis=1).sum()),
        finite_state=bool(np.isfinite(qpos).all()),
    )
    result.update(tracking_metrics(sonic, active, reference, epoch, duration))
    if render:
        render_states(directory, manifest, states)
    if (directory / "dance_with_music.mp4").is_file():
        result["video"] = str(directory / "dance_with_music.mp4")
        result["video_type"] = "按真实时间戳重绘 qpos 的状态回放录像"
    (directory / "evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def tracking_metrics(sonic, states, reference, epoch, duration):
    """用膝屈曲信号估计局部跟踪相位；按真实足底接触点速度度量滑动。"""
    import mujoco

    scene = sonic / "gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"
    model, data = mujoco.MjModel.from_xml_path(str(scene)), None
    data = mujoco.MjData(model)
    feet = {model.body(side + "_ankle_roll_link").id for side in ("left", "right")}
    knees = [
        int(model.jnt_qposadr[model.joint(side + "_knee_joint").id]) for side in ("left", "right")
    ]
    speeds, real_angles, times = [], [], []
    jacobian = np.empty((3, model.nv))
    for state in states:
        t = (state["monotonic_ns"] - epoch) / 1e9
        if t < 2 or t >= duration + 2:
            continue
        data.qpos[:], data.qvel[:] = state["qpos"], state["qvel"]
        mujoco.mj_forward(model, data)
        real_angles.append(data.qpos[knees].copy())
        times.append(t)
        for contact in data.contact:
            body1, body2 = (
                int(model.geom_bodyid[contact.geom1]),
                int(model.geom_bodyid[contact.geom2]),
            )
            foot = (
                body1
                if body1 in feet and body2 == 0
                else body2
                if body2 in feet and body1 == 0
                else None
            )
            if foot is not None and contact.dist <= 0.001:
                mujoco.mj_jac(model, data, jacobian, None, contact.pos, foot)
                speeds.append(float(np.linalg.norm((jacobian @ data.qvel)[:2])))
    joints = reference["smpl_joints"]
    thigh = joints[:, [1, 2]] - joints[:, [4, 5]]
    shank = joints[:, [7, 8]] - joints[:, [4, 5]]
    cosine = np.sum(thigh * shank, axis=-1) / (
        np.linalg.norm(thigh, axis=-1) * np.linalg.norm(shank, axis=-1)
    )
    ref_angles = np.pi - np.arccos(np.clip(cosine, -1, 1))
    actual = np.asarray(real_angles)
    correlations = []
    lags = np.arange(-0.2, 0.401, 0.02)
    if len(actual) > 50 and np.min(np.std(actual, axis=0)) > 0.02:
        normalized = (actual - actual.mean(axis=0)) / np.maximum(actual.std(axis=0), 1e-6)
        for lag in lags:
            shifted = np.stack(
                [
                    np.interp(
                        np.asarray(times) - lag, np.arange(len(ref_angles)) / 50, ref_angles[:, j]
                    )
                    for j in range(2)
                ],
                axis=1,
            )
            shifted = (shifted - shifted.mean(axis=0)) / np.maximum(shifted.std(axis=0), 1e-6)
            correlations.append(float(np.mean(normalized * shifted)))
    best = int(np.argmax(correlations)) if correlations else None
    return dict(
        foot_contact_samples=len(speeds),
        foot_tangential_speed_rms_mps=float(np.sqrt(np.mean(np.square(speeds))))
        if speeds
        else None,
        foot_tangential_speed_p95_mps=float(np.percentile(speeds, 95)) if speeds else None,
        knee_phase_lag_ms=float(lags[best] * 1000) if best is not None else None,
        knee_phase_correlation=correlations[best] if best is not None else None,
        knee_phase_search_boundary=best in (0, len(lags) - 1) if best is not None else None,
        knee_phase_scope="双膝屈曲与 SMPL 几何膝角的相关估计，正数表示机器人较参考滞后；不代表全身统一延迟",
    )


def render_states(directory, manifest, states):
    """只绘制实际保存状态；保留音乐时长、两秒准备以及一秒收尾。"""
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    scene = (
        Path(manifest["args"]["sonic_root"])
        / "gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(scene))
    model.vis.global_.offwidth, model.vis.global_.offheight = 1280, 720
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=720, width=1280)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 3.3, 135, -15
    timestamps = np.array([s["monotonic_ns"] for s in states], dtype=np.int64)
    epoch, duration = manifest["epoch_ns"], manifest["duration_seconds"] + 3
    duration = min(duration, (timestamps[-1] - epoch) / 1e9)
    audio_path = directory / (
        "played_audio.wav" if (directory / "played_audio.wav").exists() else "audio.wav"
    )
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-n",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        "1280x720",
        "-framerate",
        "50",
        "-i",
        "pipe:0",
        "-i",
        str(audio_path),
        "-filter_complex",
        "[1:a]adelay=2000|2000,apad[a]",
        "-map",
        "0:v",
        "-map",
        "[a]",
        "-t",
        str(duration),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(directory / "dance_with_music.mp4"),
    ]
    child = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for index in range(math.ceil(duration * 50)):
            target = epoch + index * 20_000_000
            row = max(0, min(len(states) - 1, int(np.searchsorted(timestamps, target))))
            data.qpos[:] = states[row]["qpos"]
            data.qvel[:] = states[row]["qvel"]
            mujoco.mj_forward(model, data)
            camera.lookat[:] = data.qpos[:3]
            camera.lookat[2] = 0.8
            renderer.update_scene(data, camera=camera)
            child.stdin.write(renderer.render().tobytes())
    finally:
        child.stdin.close()
        renderer.close()
        code = child.wait()
        if code:
            raise RuntimeError(f"录像编码失败：{code}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    evaluate(args.session.resolve(strict=True), args.render)
