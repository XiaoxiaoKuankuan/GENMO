#!/usr/bin/env python3
"""只读汇总 Stage8 多音乐的真实 Isaac 评估，兼容视频与纯数据模式。

本工具不导入 Actor、Isaac 或训练入口，不读取动作监督标签，不启动 GPU。它读取
一个已结束实验的 run_summary.json、audit.json、gmt_identity.json、事件及各个
episode 的初态/逐控制步轨迹。逐条流式读取体积较大的 JSONL，只保存统计所需的
标量、很短的启动窗口和数组摘要，不把四十条完整物理/渲染轨迹同时载入内存。

报告按音频 SHA 及跨数据集共享的全局音乐组身份核验不同音乐，模式、数据划分与目标首数
由调用方明确指定；默认仍为旧二十首 val 双模式视频报告，不改变已有验收规则。
纯数据模式无需视频或源音频，但特征来源及独立组仍须可追溯；缺测项保留 null。
已明确系统异常的末步记录缺失和EOF无换行JSON残片允许只读恢复，保留原文件SHA、
最后有效时刻和残片摘要，归为基础设施异常；中间坏行仍拒绝，不补造物理步。
新增逐曲CSV、完成率Wilson区间、按库统计、时长覆盖、运动平滑性、物理诊断及
分离yaw终止阈值统计。完成率按实际达到min(时长上限,音乐特征长度)判定；基础
设施故障、启动失败、跟踪失败与自然结束分别计数，30秒率只统计足够长的音乐。
同曲多种子、跨库收录或不同音频裁剪 SHA 均不能冒充不同歌曲；同一全局组仅允许
每种指定模式一个 episode，并保留冲突组及源 episode 身份供审计，绝不自动删样。
显式多run集合由独立SHA绑定清单加载，每条轨迹/事件仍从原run读取并保留原编号；
只在报告增加source_run_id，不生成假的合并run_summary，也不平均各run的分位数。
startup 失败、music 失败和音乐自然结束分别统计，主误差只使用
music 控制步。warmup 前五步、末十步及 music 首一秒另外汇总实际姿态/速度/接触，
不引入稳定阈值，不丢弃 music 初段，也不将诊断冒充额外的终止判定。节拍结果复用
已有指标，缺失值保留 null；跨曲分位数从控制步重新汇总，绝不平均各曲 P95。
Markdown和视频卡片另列逐曲实际/参考幅度、关节与全局根误差、脚肘相对根高度
误差、节拍距离及原0.6rad姿态阈值越界占比，不增加三维末端误差或质量评分。

视频索引使用相对路径、HTML 转义和 preload=none。视频容器/音频验收来自已经完成
的 sync manifest，PhysX→USD 证据来自 audit；本工具不声称重新解码所有视频或
验证画面中的像素关键点。净接触力包括所有碰撞，computed/applied torque 只是
隐式 PD 估计。中文 Markdown、JSON 和本地 HTML 排他写入，不覆盖已有实验结果。
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import csv
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import re
import sys
from urllib.parse import quote

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from gem.closedloop.baseline_metrics import distribution, json_value  # noqa: E402
from gem.closedloop.evaluation_music import music_control_steps  # noqa: E402
from tools.eval.stage8_collection import load_collection  # noqa: E402


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _local(root, path):
    result = Path(path)
    result = (result if result.is_absolute() else root / result).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"Artifact escapes experiment directory: {path}")
    return result


def _array(value, shape, name):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"Invalid {name}: expected finite {shape}, got {array.shape}")
    return array


def _trace_rows(stream, integrity, *, digest=None):
    """只恢复EOF处无换行的坏JSON残片；中间坏行和完整换行坏行仍拒绝。

    只消费完整记录，不修补或猜测残片字段。残片长度、SHA及原始行号进入
    基础设施异常证据；digest仍覆盖完整原文件字节，不能用恢复视图改写SHA。
    """
    for line_number, line in enumerate(stream, 1):
        if digest is not None:
            digest.update(line)
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            if line.endswith(b"\n") or stream.read(1):
                raise ValueError(f"Invalid trace JSON at {stream.name}:{line_number}") from error
            integrity.update(complete=False, system_error=integrity.get("system_error") or "truncated_trace_final_line",
                             truncated_final_line={"line_number": line_number, "byte_count": len(line),
                                                   "sha256": hashlib.sha256(line).hexdigest(),
                                                   "error_type": type(error).__name__})
            return
        yield line_number, row


def _diagnostic(row, previous_qpos):
    qpos = _array(row["actual_qpos"], (28,), "actual qpos")
    velocity = row.get("actual_joint_vel_gmt", row.get("actual_joint_vel"))
    out = {"tick": int(row["tick"]), "root_z_m": float(qpos[2]),
           "root_vertical_speed_abs_m_s": None, "root_xy_speed_m_s": None,
           "joint_speed_rms_rad_s": None, "joint_speed_abs_max_rad_s": None,
           "foot_net_force_norm_n": None, "foot_net_force_z_n": None,
           "foot_min_support_clearance_m": None}
    if velocity is not None:
        velocity = _array(velocity, (21,), "actual joint velocity")
        out["joint_speed_rms_rad_s"] = float(np.sqrt(np.mean(velocity**2)))
        out["joint_speed_abs_max_rad_s"] = float(np.abs(velocity).max())
    if previous_qpos is not None:
        root_velocity = (qpos[:3]-previous_qpos[:3])*50
        out["root_vertical_speed_abs_m_s"] = float(abs(root_velocity[2]))
        out["root_xy_speed_m_s"] = float(np.linalg.norm(root_velocity[:2]))
    physical = row.get("physical_diagnostics")
    if physical is not None:
        if physical["control_tick"] != row["tick"]:
            raise ValueError("Physical diagnostic tick differs from trace")
        if physical["foot_body_names"] != ["l_ankle_roll_link", "r_ankle_roll_link"]:
            raise ValueError("Foot diagnostic names/order differs from BUMI4340")
        forces = _array(physical["foot_net_contact_forces_w_n"], (2, 3), "foot net force")
        out["foot_net_force_norm_n"] = np.linalg.norm(forces, axis=1).tolist()
        out["foot_net_force_z_n"] = forces[:, 2].tolist()
        out["foot_min_support_clearance_m"] = _array(physical["foot_min_support_clearance_m"], (2,), "support clearance").tolist()
    return out, qpos


def _window(rows):
    scalar_keys = ("root_z_m", "root_vertical_speed_abs_m_s", "root_xy_speed_m_s",
                   "joint_speed_rms_rad_s", "joint_speed_abs_max_rad_s")
    vector_keys = ("foot_net_force_norm_n", "foot_net_force_z_n", "foot_min_support_clearance_m")
    def stats(values):
        return {**distribution(values), "min": min(values) if values else None}
    return {"control_steps": len(rows), "duration_seconds": len(rows)/50,
            "first_tick": rows[0]["tick"] if rows else None, "last_tick": rows[-1]["tick"] if rows else None,
            **{key: stats([r[key] for r in rows if r[key] is not None]) for key in scalar_keys},
            **{key: [stats([r[key][i] for r in rows if r[key] is not None]) for i in range(2)] for key in vector_keys},
            "physical_diagnostic_steps": sum(r["foot_min_support_clearance_m"] is not None for r in rows)}


def _new_pool():
    return {"errors": defaultdict(list), "latency": defaultdict(list), "prefix": [],
            "motion": defaultdict(list), "physical": defaultdict(list), "boundaries": defaultdict(list),
            "buffers": [], "events": Counter(), "rejections": Counter()}


def _limits(identity):
    """限位来自本轮实际运行指纹，不根据维度猜测另一个机器人的配置。"""
    parameters = (identity.get("runtime_fingerprint") or {}).get("parameters", {})
    names = parameters.get("joint_names")
    result = {"joint_names": names}
    for name, shape in (("joint_pos_limits", (21, 2)), ("joint_vel_limits", (21,)), ("joint_effort_limits", (21,))):
        value = parameters.get(name)
        if value is not None and names and len(names) == 21:
            array = np.asarray(value, dtype=float)
            if array.shape == (1,)+shape:
                array = array[0]
            result[name] = _array(array, shape, name)
    return result


def _motion_step(row, state, motion, physical, limits):
    """仅music连续支持点派生速度/加速度/jerk；不跨reset或warmup边界差分。"""
    for kind, joints, velocity, root in (
        ("actual", row.get("actual_joint_pos_gmt"), row.get("actual_joint_vel_gmt"), row["actual_qpos"][:3]),
        ("reference", row.get("reference", {}).get("joint_pos"), row.get("reference", {}).get("joint_vel"),
         row.get("reference", {}).get("body_pos_w", [None])[0]),
    ):
        if joints is not None:
            state.setdefault(kind+"_joints", []).append(_array(joints, (21,), kind+" joints"))
        if velocity is not None:
            velocity = _array(velocity, (21,), kind+" velocity")
            for suffix, value in (("rms", np.sqrt(np.mean(velocity**2))), ("abs_max", np.abs(velocity).max())):
                motion[f"{kind}_joint_speed_{suffix}_rad_s"].append(float(value))
            previous = state.get(kind+"_velocity")
            if previous is not None:
                acceleration = (velocity-previous)*50.
                motion[kind+"_joint_acceleration_rms_rad_s2"].append(float(np.sqrt(np.mean(acceleration**2))))
                motion[kind+"_joint_acceleration_abs_max_rad_s2"].append(float(np.abs(acceleration).max()))
                old_acceleration = state.get(kind+"_acceleration")
                if old_acceleration is not None:
                    jerk = (acceleration-old_acceleration)*50.
                    motion[kind+"_joint_jerk_rms_rad_s3"].append(float(np.sqrt(np.mean(jerk**2))))
                    motion[kind+"_joint_jerk_abs_max_rad_s3"].append(float(np.abs(jerk).max()))
                state[kind+"_acceleration"] = acceleration
            state[kind+"_velocity"] = velocity
        else:
            # 某步缺速度后，下一条有效值不能与两步前的状态当作相邻20ms差分。
            state.pop(kind+"_velocity", None)
            state.pop(kind+"_acceleration", None)
        if root is not None:
            root = _array(root, (3,), kind+" root")
            previous = state.get(kind+"_root")
            if previous is not None:
                motion[kind+"_root_xy_speed_m_s"].append(float(np.linalg.norm(root[:2]-previous[:2])*50.))
            state[kind+"_root"] = root
        else:
            state.pop(kind+"_root", None)
    diagnostic = row.get("physical_diagnostics") or {}
    if diagnostic.get("joint_names") and limits.get("joint_names") and limits["joint_names"] != diagnostic["joint_names"]:
        raise ValueError("Physical diagnostics and runtime joint-limit names/order differ")
    force = diagnostic.get("foot_net_contact_forces_w_n")
    if force is not None:
        force = _array(force, (2, 3), "foot force")
        norms = np.linalg.norm(force, axis=1)
        for i, side in enumerate(("left", "right")):
            physical[side+"_foot_net_force_norm_n"].append(float(norms[i]))
            physical[side+"_foot_contact_above_1n"].append(float(norms[i] > 1.))
        physical["both_feet_contact_above_1n"].append(float(np.all(norms > 1.)))
        physical["no_foot_contact_above_1n"].append(float(np.all(norms <= 1.)))
    clearance = diagnostic.get("foot_min_support_clearance_m")
    if clearance is not None:
        clearance = _array(clearance, (2,), "foot clearance")
        physical["both_feet_clearance_above_5mm"].append(float(np.all(clearance > .005)))
        for i, side in enumerate(("left", "right")):
            physical[side+"_foot_support_clearance_m"].append(float(clearance[i]))
    contacts, names = diagnostic.get("net_contact_forces_w_n"), diagnostic.get("contact_body_names")
    if contacts is not None and names:
        contacts = _array(contacts, (len(names), 3), "all body contacts")
        nonfeet = [i for i, name in enumerate(names) if name not in diagnostic.get("foot_body_names", [])]
        physical["nonfoot_contact_above_1n"].append(float(np.any(np.linalg.norm(contacts[nonfeet], axis=1) > 1.)))
    for kind in ("computed", "applied"):
        torque = diagnostic.get(kind+"_joint_torque_nm")
        if torque is not None:
            torque = _array(torque, (21,), kind+" torque")
            physical[kind+"_torque_estimate_rms_nm"].append(float(np.sqrt(np.mean(torque**2))))
            physical[kind+"_torque_estimate_abs_max_nm"].append(float(np.abs(torque).max()))
            effort = limits.get("joint_effort_limits")
            if effort is not None and np.all(effort > 0):
                physical[kind+"_torque_estimate_limit_ratio_max"].append(float(np.max(np.abs(torque)/effort)))
    joints, velocity = row.get("actual_joint_pos_gmt"), row.get("actual_joint_vel_gmt")
    if joints is not None and "joint_pos_limits" in limits:
        joints = _array(joints, (21,), "limit joints")
        position = limits["joint_pos_limits"]
        violation = np.maximum(position[:, 0]-joints, joints-position[:, 1])
        physical["joint_position_limit_exceeded"].append(float(np.any(violation > 1e-5)))
        physical["joint_position_limit_violation_max_rad"].append(float(max(0., violation.max())))
    if velocity is not None and "joint_vel_limits" in limits and np.all(limits["joint_vel_limits"] > 0):
        physical["joint_speed_limit_ratio_max"].append(float(np.max(np.abs(velocity)/limits["joint_vel_limits"])))


def _trace(root, episode, pool, *, limits=None, thresholds=None):
    path = _local(root, episode["artifacts"]["trace"])
    initial = _json(path.parent / "initial.json")
    initial = initial.get("snapshot", initial)
    previous = initial.get("robot_qpos")
    previous = _array(previous, (28,), "initial actual qpos") if previous is not None else None
    previous_tick = int(initial["tick"])
    warmup_first, warmup_last, music_first = [], deque(maxlen=10), []
    phase_rows = {"warmup": [], "music": []}
    phase_counts = Counter()
    digest, first_frame, last_frame = hashlib.sha256(), None, None
    frame_count = 0
    state, extras, physical, first_crossings = {}, defaultdict(list), defaultdict(list), {}
    own_errors, boundaries = defaultdict(list), defaultdict(list)
    previous_reference = None
    trace_integrity = dict(episode.get("trace_integrity") or {})
    for name in ("errors", "latency"):
        pool.setdefault(name, defaultdict(list))
    with path.open("rb") as stream:
        for line_number, row in _trace_rows(stream, trace_integrity, digest=digest):
            if row["episode_id"] != episode["episode_id"] or row["tick"] != previous_tick+12:
                raise ValueError(f"Trace episode/tick mismatch at {path}:{line_number}")
            if row.get("control_tick_begin") != previous_tick:
                raise ValueError(f"Trace begin tick mismatch at {path}:{line_number}")
            phase = row["phase"]
            if phase not in phase_rows or (phase == "warmup" and phase_counts["music"]):
                raise ValueError("Trace requires warmup followed by music")
            diagnostic, previous = _diagnostic(row, previous)
            phase_counts[phase] += 1
            phase_rows[phase].append(diagnostic)
            if phase == "warmup":
                if len(warmup_first) < 5:
                    warmup_first.append(diagnostic)
                warmup_last.append(diagnostic)
            else:
                if len(music_first) < 50:
                    music_first.append(diagnostic)
                for key, value in row["errors"].items():
                    pool["errors"][key].append(float(value))
                    own_errors[key].append(float(value))
                    if key in (thresholds or {}) and value > thresholds[key] and key not in first_crossings:
                        first_crossings[key] = {"tick": int(row["tick"]), "music_seconds": phase_counts["music"]/50., "value": float(value)}
                _motion_step(row, state, extras, physical, limits or {})
                reference = row.get("reference")
                if previous_reference is not None and row.get("reference_plan_id") != previous_reference.get("reference_plan_id"):
                    old = previous_reference["reference"]
                    for key, unit in (("joint_pos", "rad"), ("joint_vel", "rad_s"), ("body_lin_vel_w", "m_s")):
                        if key in reference and key in old:
                            boundaries[key+"_adjacent_step_max_"+unit].append(float(np.max(np.abs(np.asarray(reference[key])-old[key]))))
                    if "body_pos_w" in reference and "body_pos_w" in old:
                        boundaries["root_position_adjacent_step_m"].append(float(np.linalg.norm(np.asarray(reference["body_pos_w"])[0]-old["body_pos_w"][0])))
                    if "body_quat_w" in reference and "body_quat_w" in old:
                        first, second = np.asarray(old["body_quat_w"])[0], np.asarray(reference["body_quat_w"])[0]
                        denominator = float(np.linalg.norm(first)*np.linalg.norm(second))
                        if denominator <= 0:
                            raise ValueError("Invalid zero-norm reference orientation")
                        cosine = abs(float(first@second))/denominator
                        boundaries["root_orientation_adjacent_step_rad"].append(float(2*np.arccos(np.clip(cosine, 0., 1.))))
                previous_reference = row if reference else None
                for key in ("gmt_inference_seconds", "physics_seconds", "video_capture_seconds", "step_seconds"):
                    if key in row:
                        pool["latency"][key].append(float(row[key]))
            frame = row.get("video_frame_index")
            if frame is not None:
                if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0 or (last_frame is not None and frame != last_frame+1):
                    raise ValueError("Video frame indexes are not contiguous")
                first_frame = frame if first_frame is None else first_frame
                last_frame, frame_count = frame, frame_count+1
            previous_tick = row["tick"]
    terminal_tick = int(episode["terminal_snapshot"]["tick"])
    known_incomplete = bool(trace_integrity.get("system_error"))
    if previous_tick != terminal_tick:
        if not known_incomplete or terminal_tick < previous_tick or (terminal_tick-previous_tick) % 12:
            raise ValueError("Trace does not reach terminal snapshot")
        trace_integrity.update(complete=False, recorded_trace_last_tick=previous_tick,
                               terminal_snapshot_tick=terminal_tick,
                               missing_trailing_control_step_records=(terminal_tick-previous_tick)//12)
    if phase_counts["music"] != episode["recorded_music_control_steps"]:
        if not known_incomplete:
            raise ValueError("Trace music frame count differs from summary")
        trace_integrity.update(complete=False, recovered_music_control_step_records=phase_counts["music"],
                               summary_music_control_step_records=episode["recorded_music_control_steps"])
    for name, collection in (("motion", extras), ("physical", physical), ("boundaries", boundaries)):
        pool.setdefault(name, defaultdict(list))
        for key, values in collection.items():
            pool[name][key].extend(values)
    amplitudes = {}
    for kind in ("actual", "reference"):
        joints = state.get(kind+"_joints", [])
        amplitudes[kind] = ((np.percentile(joints, 95, axis=0)-np.percentile(joints, 5, axis=0)).tolist() if joints else None)
    return {"trace_sha256": digest.hexdigest(), "control_steps": sum(phase_counts.values()),
            "trace_integrity": trace_integrity,
            "phase_counts": dict(phase_counts),
            "warmup_all": _window(phase_rows["warmup"]), "music_all": _window(phase_rows["music"]),
            "warmup_first_5": _window(warmup_first), "warmup_last_10": _window(list(warmup_last)),
            "music_first_1_second": _window(music_first),
            "initial_snapshot": initial,
            "extended": {"motion": {k: distribution(v) for k, v in extras.items()},
                         "physical": {k: {**distribution(v), "min": min(v)} for k, v in physical.items()},
                         "joint_amplitude_p95_minus_p05_rad": amplitudes,
                         "errors": {k: distribution(v) for k, v in own_errors.items()},
                         "boundary_changes": {k: distribution(v) for k, v in boundaries.items()},
                         "threshold_first_crossing": first_crossings,
                         "threshold_crossing_counts": {key: sum(x > value for x in own_errors.get(key, [])) for key, value in (thresholds or {}).items()},
                         "threshold_supported_steps": {key: len(own_errors.get(key, [])) for key in (thresholds or {})}},
            "video_frames": {"count": frame_count, "first": first_frame, "last": last_frame}}


def _beat_summary(episodes, which):
    valid, unavailable = [], Counter()
    for episode in episodes:
        music = episode.get("music") or {}
        metric = music.get(which)
        if not isinstance(metric, dict) or metric.get("status") != "available":
            unavailable[(metric or {}).get("status", music.get("status", "not_recorded"))] += 1
            continue
        distance, alignment, count = metric.get("mean_beat_distance_seconds"), metric.get("alignment"), music.get("music_beat_count")
        if distance is None or alignment is None or count is None or count <= 0:
            unavailable["invalid_available_metric"] += 1
        else:
            valid.append((float(distance), float(alignment), int(count)))
    beats = sum(v[2] for v in valid)
    weighted_distance = sum(d*n for d, _, n in valid)/beats if beats else None
    return {"available_episodes": len(valid), "unavailable_reason_counts": dict(unavailable),
            "available_music_beat_count": beats,
            "per_episode_mean_distance_seconds": distribution([v[0] for v in valid]),
            "per_episode_alignment": distribution([v[1] for v in valid]),
            "music_beat_weighted_distance_seconds": weighted_distance,
            "alignment_of_weighted_distance": math.exp(-weighted_distance/.1) if weighted_distance is not None else None,
            "interpretation": "每个音乐节拍到最近动作速度谷值的距离；加权alignment=exp(-加权平均距离/0.1)，不是各曲alignment平均，也不是音乐质量评分"}


def _videos(root, summary, episodes):
    entries = list(summary.get("videos", []))
    if summary.get("video") and not entries:
        entries.append(summary["video"])
    mapping, issues = {}, []
    for value in entries:
        value = dict(value)
        episode_id = value.get("episode_id")
        if episode_id in mapping or episode_id is None:
            issues.append(f"duplicate/missing video episode_id: {episode_id}")
            continue
        raw = value.get("path", value.get("video_path"))
        if raw is None:
            issues.append(f"{episode_id}: missing video path")
            continue
        path = _local(root, raw)
        value["relative_path"] = path.relative_to(root).as_posix()
        value["file_exists"] = path.is_file()
        manifest = value.get("manifest_path")
        manifest_path = _local(root, manifest) if manifest else path.with_suffix(".sync.json")
        if manifest_path.is_file():
            sync = _json(manifest_path)
            if sync.get("episode_id") != episode_id:
                issues.append(f"{episode_id}: sync episode_id mismatch")
            value["sync"] = sync
        elif "frame_count" in value:
            value["sync"] = dict(value)
        else:
            value["sync"] = None
            issues.append(f"{episode_id}: sync manifest missing")
        if not path.is_file():
            issues.append(f"{episode_id}: video missing")
        mapping[episode_id] = value
    intervals = []
    unknown = set(mapping)-{episode["episode_id"] for episode in episodes}
    if unknown:
        issues.append(f"unknown video episode IDs: {sorted(unknown)}")
    for episode in episodes:
        episode_id, trace = episode["episode_id"], episode["startup_diagnostics"]
        video = mapping.get(episode_id)
        if video is None:
            issues.append(f"{episode_id}: video not indexed")
            continue
        sync, frames = video.get("sync"), trace["video_frames"]
        if frames["count"] != trace["control_steps"]:
            issues.append(f"{episode_id}: missing trace video frame indexes")
        if sync is not None:
            expected = {"first_raw_frame": frames["first"], "end_raw_frame_exclusive": frames["last"]+1 if frames["last"] is not None else None,
                        "frame_count": trace["control_steps"], "warmup_frames": trace["phase_counts"].get("warmup", 0),
                        "music_frames": trace["phase_counts"].get("music", 0), "fps": 50,
                        "audio_sha256": episode["audio_sha256"], "trace_sha256": trace["trace_sha256"]}
            for key, value in expected.items():
                if sync.get(key) != value:
                    issues.append(f"{episode_id}: video sync {key} mismatch")
            if sync.get("status") != "passed":
                issues.append(f"{episode_id}: video container/audio sync not passed")
            if sync.get("audio_delay_seconds") != expected["warmup_frames"]/50:
                issues.append(f"{episode_id}: audio warmup offset mismatch")
        if frames["first"] is not None:
            intervals.append((frames["first"], frames["last"], episode_id))
    gaps = []
    for old, new in zip(sorted(intervals), sorted(intervals)[1:]):
        if new[0] <= old[1]:
            issues.append(f"video raw frame intervals overlap: {old[2]} / {new[2]}")
        elif new[0] != old[1]+1:
            gaps.append({"previous_episode": old[2], "next_episode": new[2], "unindexed_frames": new[0]-old[1]-1})
    return mapping, {"status": "passed" if not issues and episodes else "failed", "indexed_episode_count": len(mapping),
                     "issues": issues, "raw_frame_gaps": gaps,
                     "evidence_boundary": "核对已完成的容器/音频sync清单与trace；本工具不重新解码视频或复算成品文件SHA"}


def _grounded_initialization(identity, episodes):
    provenance = identity.get("initial_ground_pose")
    issues, clearances = [], []
    if not provenance:
        return {"status": "not_recorded", "verified_resets": 0, "minimum_support_clearance_m": distribution([]),
                "issues": ["initial_ground_pose provenance missing"]}
    if provenance.get("clearance_m") != .001:
        issues.append("initial clearance setting is not 0.001 m")
    if re.fullmatch(r"[0-9a-f]{64}", str(provenance.get("sha256", ""))) is None:
        issues.append("initial pose provenance SHA missing/invalid")
    for ep in episodes:
        actual = ep["startup_diagnostics"]["initial_snapshot"].get("reset_ground_pose_actual", {})
        minimum, tolerance = actual.get("minimum_support_clearance_m"), actual.get("tolerance_m")
        valid = (actual.get("status") == "passed" and actual.get("provenance_sha256") == provenance.get("sha256")
                 and actual.get("physics_steps_added") == 0 and actual.get("history_updates_added") == 0
                 and actual.get("expected_clearance_m") == .001 and minimum is not None and tolerance is not None
                 and 0 <= tolerance <= 5e-5 and abs(minimum-.001) <= tolerance)
        if valid:
            clearances.append(minimum)
        else:
            issues.append(f"{ep['episode_id']}: missing/invalid actual reset 1 mm proof")
    return {"status": "passed" if not issues and episodes else "failed", "verified_resets": len(clearances),
            "configured_root_z_m": provenance.get("root_z_m"), "original_root_z_m": provenance.get("original_root_z_m"),
            "target_clearance_m": provenance.get("clearance_m"), "provenance_sha256": provenance.get("sha256"),
            "minimum_support_clearance_m": distribution(clearances), "issues": issues}


def _wilson(successes, total):
    """95% Wilson区间；样本为独立音乐条目，不能解释为随机抽样总体的无偏保证。"""
    if not total:
        return {"successes": successes, "total": 0, "fraction": None, "ci95_low": None, "ci95_high": None, "method": "Wilson"}
    z = 1.959963984540054
    proportion, denominator = successes/total, 1+z*z/total
    centre = (proportion+z*z/(2*total))/denominator
    radius = z*math.sqrt(proportion*(1-proportion)/total+z*z/(4*total*total))/denominator
    return {"successes": successes, "total": total, "fraction": proportion,
            "ci95_low": max(0., centre-radius), "ci95_high": min(1., centre+radius), "method": "Wilson"}


def _completion(original, maximum_seconds):
    row = original["sample"]["row"]
    feature_frames = row.get("music_feature_num_frames", row.get("num_frames"))
    feature_seconds = float(feature_frames)/30. if feature_frames is not None else None
    if feature_seconds is not None and (not math.isfinite(feature_seconds) or feature_seconds <= 0):
        raise ValueError("Music feature duration must be finite and positive")
    target = min(maximum_seconds, feature_seconds) if feature_seconds is not None else None
    # 控制时间网格向下取整与coordinator完全一致；余下不足20ms不能当失败。
    executable_target = music_control_steps(feature_frames, maximum_seconds)/50. if feature_frames is not None else None
    elapsed = float(original.get("music_duration_seconds") or 0.)
    reason = original.get("reason") or "unknown"
    integrity = original.get("trace_integrity") or {}
    if reason in {"infrastructure_error", "system_error", "worker_error", "exception", "interrupted"} or integrity.get("system_error"):
        category = "infrastructure_error"
    elif original.get("startup_failure"):
        category = "startup_failure"
    elif reason == "music_end":
        category = "natural_music_end"
    elif reason == "duration_limit":
        category = "duration_limit"
    else:
        category = "control_failure"
    reached = elapsed+1e-8 >= executable_target if executable_target is not None else None
    normal = category in {"natural_music_end", "duration_limit"} and not original.get("failed", False)
    normal_complete = normal and reached if reached is not None else None
    ratio = min(1., elapsed/executable_target) if executable_target else None
    return {"category": category, "feature_duration_seconds": feature_seconds,
            "maximum_seconds": maximum_seconds, "target_duration_seconds": target,
            "executable_target_seconds": executable_target, "music_duration_seconds": elapsed,
            "reached_target_duration": reached, "normal_complete": normal_complete,
            "duration_completion_fraction": ratio,
            "eligible_for_30s": feature_seconds >= 30. if feature_seconds is not None else None,
            "reached_30s_without_failure": normal and elapsed+1e-8 >= 30.,
            "failure_music_seconds": elapsed if category in {"startup_failure", "control_failure"} else None,
            "failure_total_seconds": original.get("total_duration_seconds") if category in {"startup_failure", "control_failure"} else None}


def _completion_group(episodes, requested=None):
    requested = len(episodes) if requested is None else requested
    completed = sum(e["completion"]["normal_complete"] is True for e in episodes)
    known = [e for e in episodes if e["completion"]["normal_complete"] is not None]
    eligible = [e for e in episodes if e["completion"]["eligible_for_30s"] is True]
    targets = [e["completion"]["executable_target_seconds"] for e in episodes]
    target_seconds = sum(t for t in targets if t is not None)
    covered = sum(min(e["music_duration_seconds"], e["completion"]["executable_target_seconds"])
                  for e in episodes if e["completion"]["executable_target_seconds"] is not None)
    return {"requested_episodes": requested, "attempted_episodes": len(episodes),
            "not_attempted_episodes": max(0, requested-len(episodes)),
            "ending_category_counts": dict(Counter(e["completion"]["category"] for e in episodes)),
            "normal_completion": _wilson(completed, len(known)),
            "normal_completion_fraction_of_requested": completed/requested if requested else None,
            "completion_unknown_duration_episodes": len(episodes)-len(known),
            "reached_30s_among_eligible": _wilson(sum(e["completion"]["reached_30s_without_failure"] for e in eligible), len(eligible)),
            "requested_music_seconds_for_attempted": target_seconds,
            "executed_music_seconds": sum(e["music_duration_seconds"] for e in episodes),
            "covered_target_music_seconds": covered,
            "total_execution_coverage_fraction": covered/target_seconds if target_seconds else None,
            "per_episode_duration_completion_fraction": distribution([e["completion"]["duration_completion_fraction"] for e in episodes if e["completion"]["duration_completion_fraction"] is not None]),
            "failure_music_seconds": distribution([e["completion"]["failure_music_seconds"] for e in episodes if e["completion"]["failure_music_seconds"] is not None]),
            "failure_reason_counts": dict(Counter(e.get("backend_reason") or e["reason"] for e in episodes if e["completion"]["category"] in {"startup_failure", "control_failure"})),
            "interpretation": "正常完成要求正常结束且达到本曲可执行目标；30秒率分母仅为特征长度>=30秒的已尝试条目；启动失败计入分母，基础设施错误单列。未尝试曲目不进入已尝试完成率。"}


def _pooled_diagnostics(pool):
    buffers = pool["buffers"]
    return {"motion_pooled_music_control_steps": {key: distribution(values) for key, values in pool["motion"].items()},
            "physical_pooled_music_control_steps": {key: {**distribution(values), "min": min(values) if values else None}
                                                    for key, values in pool["physical"].items()},
            "reference_boundary_changes": {key: distribution(values) for key, values in pool["boundaries"].items()},
            "latency_seconds": {key: distribution(values) for key, values in pool["latency"].items()},
            "control_timing_over_20ms": {key: {"supported_steps": len(pool["latency"].get(key, [])),
                "over_20ms_count": sum(value > .02 for value in pool["latency"].get(key, [])),
                "over_20ms_fraction": (sum(value > .02 for value in pool["latency"].get(key, []))/len(pool["latency"][key]))
                                      if pool["latency"].get(key) else None}
                for key in ("step_seconds", "gmt_inference_seconds", "physics_seconds")},
            "prefix_frames": distribution(pool["prefix"]),
            "prefix_frame_counts": dict(sorted(Counter(map(str, pool["prefix"])).items())),
            "prefix_over_18_fraction": sum(p > 18 for p in pool["prefix"])/len(pool["prefix"]) if pool["prefix"] else None,
            "reference_cache_remaining_seconds": {**distribution(buffers), "min": min(buffers) if buffers else None},
            "decision_missed_count": pool["events"]["decision_missed"],
            "request_rejected_count": pool["events"]["request_rejected"],
            "plan_rejected_count": pool["events"]["plan_rejected"],
            "rejection_codes": dict(pool["rejections"])}


def _protocol_summary(episodes):
    counts, warnings = Counter(), Counter()
    history_verified = history_mismatches = 0
    for episode in episodes:
        audit = episode["audit"]
        counts.update(audit.get("counts", {}))
        warnings.update(audit.get("warnings", []))
        initial, terminal = audit.get("initial_history_counts", {}), audit.get("terminal_history_counts", {})
        controls = audit.get("counts", {}).get("control_steps")
        if controls is not None and all(key in initial and key in terminal for key in ("gmt", "proprio")):
            history_verified += 1
            history_mismatches += any(terminal[key]-initial[key] != controls for key in ("gmt", "proprio"))
    return {"audited_episode_status_counts": dict(Counter(e["audit"].get("status", "not_recorded") for e in episodes)),
            "counts": dict(counts), "history_count_verified_episodes": history_verified,
            "history_count_mismatch_episodes": history_mismatches if history_verified else None,
            "warnings": dict(warnings),
            "plan_ownership_and_cross_episode_check": "passed" if episodes and all(e["audit"].get("status") == "passed" for e in episodes) else "not_all_passed",
            "interpretation": "协议审计逐步重建参考消费时间/来源计划，检查无跨episode执行及25控制/100物理时序；history单独汇总初末计数，未记录不当0。"}


def _amplitude_summary(episodes):
    """跨曲幅度按曲统计，不将各曲不同关节的幅度混称为控制步分布。"""
    result, ratios = {}, []
    for kind in ("actual", "reference"):
        vectors = [e["startup_diagnostics"]["extended"]["joint_amplitude_p95_minus_p05_rad"][kind] for e in episodes]
        vectors = [v for v in vectors if v is not None]
        result[kind] = {"per_episode_mean_joint_amplitude_rad": distribution([np.mean(v) for v in vectors]),
                        "per_episode_max_joint_amplitude_rad": distribution([max(v) for v in vectors]),
                        "mean_amplitude_by_joint_rad": np.mean(vectors, axis=0).tolist() if vectors else None}
    for episode in episodes:
        values = episode["startup_diagnostics"]["extended"]["joint_amplitude_p95_minus_p05_rad"]
        if values["actual"] is not None and values["reference"] is not None and np.mean(values["reference"]) > 1e-10:
            ratios.append(float(np.mean(values["actual"])/np.mean(values["reference"])))
    result["actual_over_reference_mean_amplitude_ratio"] = distribution(ratios)
    return result


def _partial_episodes(root, summary):
    """异常退出时恢复已落盘证据；未结束trace只记基础设施错误，不补造物理步。"""
    source = list(summary.get("episodes", []))
    indexed = {e["episode_id"] for e in source}
    for initial_path in sorted((root/"episodes").glob("*/initial.json")):
        initial = _json(initial_path)
        snapshot = initial.get("snapshot", initial)
        if snapshot["episode_id"] in indexed:
            continue
        detail = initial_path.with_name("summary.json")
        if detail.is_file():
            episode = _json(detail)
        else:
            trace = initial_path.with_name("trace.jsonl")
            if not trace.is_file() or "sample" not in initial:
                continue
            music_count, count, last_tick = 0, 0, snapshot["tick"]
            trace_integrity = {"complete": False, "system_error": "episode_summary_missing"}
            with trace.open("rb") as stream:
                for _, row in _trace_rows(stream, trace_integrity):
                    count += 1
                    music_count += row.get("phase") == "music"
                    last_tick = row["tick"]
            episode = {"episode_id": snapshot["episode_id"], "sample": initial["sample"], "seed": initial["seed"], "mode": initial["mode"],
                       "reason": "infrastructure_error", "backend_reason": None, "failed": True, "startup_failure": False,
                       "music_duration_seconds": music_count/50., "warmup_duration_seconds": (count-music_count)/50., "total_duration_seconds": count/50.,
                       "recorded_music_control_steps": music_count, "artifacts": {"trace": str(trace.relative_to(root))},
                       "terminal_snapshot": {**snapshot, "tick": last_tick}, "trace_integrity": trace_integrity}
        source.append(episode)
        indexed.add(episode["episode_id"])
    return [e for e in source if e.get("mode") != "calibration"]


def build_music_sweep_report(run_dir, *, expected_music=20, expected_modes=("paused", "latency"),
                             expected_split="val", require_video=True, maximum_seconds=30., collection_manifest=None):
    """构建 JSON 可序列化报告，不写文件；调用方须等 worker 和视频封装均结束。"""
    root = Path(run_dir).resolve()
    collection = None
    if collection_manifest is not None:
        if require_video:
            raise ValueError("Explicit collections currently require no-video mode")
        manifest_path = Path(collection_manifest)
        contexts, collection = load_collection(manifest_path if manifest_path.is_absolute() else root/manifest_path)
        identity = contexts[0]["identity"]
        # 这里只整理内存中的统计上下文；不落盘、不伪造合并 run_summary，所有原始
        # episode/plan 身份与实际 trace 路径仍归属于各自 source_run_id。
        summary = {"termination": contexts[0]["summary"].get("termination", {}), "exit_code": 0,
            "selection": [e["sample"] for context in contexts for e in context["episodes"]],
            "source_verification": {"unchanged": True, "source_count": len(contexts)},
            "acceptance": {"collection_source_checks_passed": True, "source_runs":
                {context["source_run_id"]: context["summary"].get("acceptance") for context in contexts}}}
    else:
        summary = _json(root / "run_summary.json")
        audit = _json(root / "audit.json") if (root / "audit.json").is_file() else {}
        identity = _json(root / "gmt_identity.json") if (root / "gmt_identity.json").is_file() else {}
        contexts = [{"source_run_id": None, "root": root, "summary": summary, "audit": audit,
                     "identity": identity, "episodes": _partial_episodes(root, summary)}]
    source = [(context, episode) for context in contexts for episode in context["episodes"]]
    pools, dataset_pools = defaultdict(_new_pool), defaultdict(_new_pool)
    termination = summary.get("termination", {})
    thresholds = {key: float(value) for key, value in termination.items() if key.endswith("_error_m") or key.endswith("_error_rad")}
    if termination.get("orientation_mode") == "separated_yaw":
        thresholds.pop("global_orientation_error_rad", None)
    limits = _limits(identity)
    episodes, issues, ids = [], [], set()
    for context, original in source:
        episode_id = original["episode_id"]
        source_id, source_root = context["source_run_id"], context["root"]
        identity_key = (source_id, episode_id)
        if identity_key in ids:
            raise ValueError(f"Duplicate source/episode id: {identity_key}")
        ids.add(identity_key)
        audit_by_id = {e["episode_id"]: e for e in context["audit"].get("episodes", [])}
        sample = original["sample"]
        row = sample["row"]
        sha = row.get("source_audio_sha256", "")
        if sha and re.fullmatch(r"[0-9a-f]{64}", sha) is None:
            raise ValueError(f"Invalid audio SHA for {episode_id}")
        if re.fullmatch(r"[0-9a-f]{64}", sha) is None and require_video:
            raise ValueError(f"Missing audio SHA for {episode_id}")
        episode_pool = _new_pool()
        trace = _trace(source_root, original, episode_pool, limits=limits, thresholds=thresholds)
        if collection is not None and trace["trace_sha256"] != context["selected_by_id"][episode_id]["trace_sha256"]:
            raise ValueError(f"Selected source trace SHA mismatch: {source_id}/{episode_id}")
        original = {**original, "trace_integrity": trace["trace_integrity"]}
        ep = {key: original.get(key) for key in ("episode_id", "mode", "seed", "reason", "backend_reason", "failed", "startup_failure",
                "music_duration_seconds", "warmup_duration_seconds", "total_duration_seconds", "replan_count", "errors", "music", "motion",
                "reference_boundaries", "protected_reference", "prefix_frames", "prefix_over_18_count", "reference_buffer", "event_counts", "rejection_codes",
                "original_threshold_diagnostics", "latency_seconds", "trace_integrity")}
        ep.update(dataset=sample.get("dataset", original.get("dataset")), sample_id=row["sample_id"], group_id=sample.get("group_id"),
                  audio_sha256=sha, music_feature_sha256=row.get("source_music_feature_sha256"), manifest_sha256=sample.get("manifest_sha256"),
                  split=row.get("split"), startup_diagnostics=trace, completion=_completion(original, maximum_seconds),
                  audit=audit_by_id.get(episode_id, {"status": "not_recorded"}))
        if collection is not None:
            ep.update(source_run_id=source_id, source_run_dir=str(source_root),
                      source_trace_relative_path=str(original["artifacts"]["trace"]))
        detail_path = _local(source_root, original["artifacts"]["trace"]).with_name("summary.json")
        ep["detail_json_relative_path"] = (Path(os.path.relpath(detail_path, root)).as_posix()
                                           if detail_path.is_file() else "music_sweep_report.json")
        if ep["split"] != expected_split:
            issues.append(f"{episode_id}: split is not {expected_split}")
        if not ep["group_id"]:
            issues.append(f"{episode_id}: independent music group ID missing")
        if expected_split == "train" and ep["completion"]["feature_duration_seconds"] is None:
            issues.append(f"{episode_id}: music feature duration missing")
        if trace["trace_integrity"].get("system_error"):
            issues.append(f"{episode_id}: incomplete trace infrastructure evidence: {trace['trace_integrity']['system_error']}")
        for target in (pools[original["mode"]], dataset_pools[(original["mode"], ep["dataset"])]):
            for category in ("errors", "latency", "motion", "physical", "boundaries"):
                for key, values in episode_pool[category].items():
                    target[category][key].extend(values)
        episodes.append(ep)
    index = {(ep.get("source_run_id"), ep["episode_id"]): ep for ep in episodes}
    event_counts = defaultdict(Counter)
    for context in contexts:
        path = context["root"] / "events.jsonl"
        if not path.is_file():
            issues.append(f"{context['source_run_id']}: events.jsonl missing: pooled request timing/prefix unavailable")
            continue
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                event = json.loads(line)
                ep = index.get((context["source_run_id"], event.get("episode_id")))
                if ep is None:
                    continue
                mode, name = ep["mode"], event["event"]
                event_counts[mode][name] += 1
                for pool in (pools[mode], dataset_pools[(mode, ep["dataset"])]):
                    pool["events"][name] += 1
                    if name == "plan_prepared":
                        pool["prefix"].append(int(event["prefix_frames"]))
                        for key in ("end_to_end_seconds", "actor_seconds", "prepare_seconds"):
                            if event.get(key) is not None:
                                pool["latency"][key].append(float(event[key]))
                        if all(event.get(key) is not None for key in ("end_to_end_seconds", "actor_seconds", "prepare_seconds")):
                            pool["latency"]["coordinator_rpc_overhead_seconds"].append(max(0., event["end_to_end_seconds"]-event["actor_seconds"]-event["prepare_seconds"]))
                    if name in ("request_rejected", "plan_rejected"):
                        pool["rejections"][event.get("code", "unknown")] += 1
                    if name == "advance" and event.get("phase") == "music" and event.get("reference_valid_end_tick") is not None:
                        pool["buffers"].append((event["reference_valid_end_tick"]-event["end_tick"])/600.)
    songs = defaultdict(list)
    for ep in episodes:
        key = ep["audio_sha256"] or f"group:{ep['group_id']}"
        songs[key].append(ep)
    for sha, entries in songs.items():
        counts = Counter(e["mode"] for e in entries)
        if counts != Counter({mode: 1 for mode in expected_modes}):
            issues.append(f"audio {sha}: expected one episode per mode, got {dict(counts)}")
        if len({e["seed"] for e in entries}) != 1:
            issues.append(f"audio {sha}: paused/latency seeds differ")
        if len({(e["dataset"], e["group_id"], e["sample_id"]) for e in entries}) != 1:
            issues.append(f"audio {sha}: paused/latency source sample differs")
    if len(songs) != expected_music:
        issues.append(f"expected {expected_music} unique audio SHA, got {len(songs)}")
    expected_episodes = expected_music*len(expected_modes)
    if len(episodes) != expected_episodes:
        issues.append(f"expected {expected_episodes} episodes, got {len(episodes)}")
    # Stage1 resplit 的 group_id 已跨数据集统一：数据集前缀或音频裁剪 SHA 不能使
    # 同一首音乐成为两个独立样本。双模式复用同一曲是合法的，其模式矩阵单独核验。
    groups = defaultdict(list)
    for ep in episodes:
        if ep["group_id"]:
            groups[ep["group_id"]].append(ep)
    independent_groups = set(groups)
    group_conflicts = []
    for group_id, entries in groups.items():
        counts = Counter(e["mode"] for e in entries)
        sources = {(e["dataset"], e["sample_id"], e["audio_sha256"]) for e in entries}
        if counts != Counter({mode: 1 for mode in expected_modes}) or len(sources) != 1:
            group_conflicts.append({"group_id": group_id, "mode_counts": dict(counts),
                "episodes": [{key: e.get(key) for key in ("source_run_id", "episode_id", "dataset", "sample_id", "audio_sha256", "mode")}
                             for e in entries]})
            issues.append(f"global music group {group_id}: expected one source and one episode per mode, "
                          f"got {len(sources)} sources and {dict(counts)}")
    if len(independent_groups) != expected_music:
        issues.append(f"expected {expected_music} independent music groups, got {len(independent_groups)}")
    if require_video:
        videos, video_audit = _videos(root, summary, episodes)
    else:
        videos, video_audit = {}, {"status": "not_requested", "indexed_episode_count": 0, "issues": [], "raw_frame_gaps": [],
                                  "evidence_boundary": "本轮明确关闭渲染和视频；不作视觉验收或视觉通过声明。"}
    modes = {}
    for mode in sorted({e["mode"] for e in episodes}):
        chosen = [e for e in episodes if e["mode"] == mode]
        startup = [e for e in chosen if e["startup_failure"]]
        music_started = [e for e in chosen if e["startup_diagnostics"]["phase_counts"].get("music", 0)]
        music_failed = [e for e in music_started if e["completion"]["category"] == "control_failure"]
        exposure = sum(e["music_duration_seconds"] for e in chosen)
        protected = [e.get("protected_reference") or {} for e in chosen]
        known_protection = all(p.get("modification_count") is not None for p in protected)
        pool = pools[mode]
        def maximum_available(key):
            values = [(p.get(key) or {}).get("max") for p in protected]
            values = [value for value in values if value is not None]
            return max(values) if values else None
        modes[mode] = {"episodes": len(chosen), "startup_failures": len(startup),
            "startup_failure_fraction": len(startup)/len(chosen),
            "startup_backend_reason_counts": dict(Counter(e["backend_reason"] or e["reason"] for e in startup)),
            "music_started_episodes": len(music_started), "music_failures": len(music_failed),
            "music_failure_fraction_of_started": len(music_failed)/len(music_started) if music_started else None,
            "music_failure_reason_counts": dict(Counter(e["backend_reason"] or e["reason"] for e in music_failed)),
            "ending_reason_counts": dict(Counter(e["reason"] for e in chosen)),
            "music_exposure_seconds": exposure, "music_failures_per_minute": len(music_failed)/(exposure/60) if exposure else None,
            "music_duration_seconds": distribution([e["music_duration_seconds"] for e in chosen]),
            "errors_pooled_music_control_steps": {k: distribution(v) for k, v in pool["errors"].items()},
            "latency_seconds": {k: distribution(v) for k, v in pool["latency"].items()},
            "beat": {kind: _beat_summary(chosen, kind) for kind in ("actual", "reference")},
            "protected_reference": {"modification_count": sum(p["modification_count"] for p in protected) if known_protection else None,
                "position_recompute_max": maximum_available("recomputed_position_error"),
                "velocity_recompute_max": maximum_available("recomputed_velocity_error")},
            "prefix_frames": distribution(pool["prefix"]), "prefix_over_18_count": sum(p > 18 for p in pool["prefix"]),
            "events": dict(event_counts[mode]), "replans": sum(e["replan_count"] or 0 for e in chosen),
            "reference_underrun_count": sum((e.get("reference_buffer") or {}).get("reference_underrun", False) for e in chosen),
            "protocol_passed_episodes": sum(e["audit"].get("status") == "passed" for e in chosen),
            "visual_passed_episodes": sum(e["audit"].get("visual_evidence", {}).get("status") == "passed" for e in chosen)}
        modes[mode].update(_pooled_diagnostics(pool))
        modes[mode]["protocol"] = _protocol_summary(chosen)
        modes[mode]["amplitude"] = _amplitude_summary(chosen)
        modes[mode]["completion"] = _completion_group(chosen, expected_music)
        modes[mode]["by_dataset"] = {}
        selected_counts = Counter(s["dataset"] for s in summary.get("selection", []))
        for dataset in sorted({e["dataset"] for e in chosen} | set(selected_counts)):
            subset = [e for e in chosen if e["dataset"] == dataset]
            modes[mode]["by_dataset"][dataset] = {
                "completion": _completion_group(subset, selected_counts.get(dataset, len(subset))),
                "amplitude": _amplitude_summary(subset), "protocol": _protocol_summary(subset),
                "music_duration_seconds": distribution([e["music_duration_seconds"] for e in subset]),
                "errors_pooled_music_control_steps": {key: distribution(values) for key, values in dataset_pools[(mode, dataset)]["errors"].items()},
                "beat": {kind: _beat_summary(subset, kind) for kind in ("actual", "reference")},
                **_pooled_diagnostics(dataset_pools[(mode, dataset)])}
    for ep in episodes:
        ep["video"] = videos.get(ep["episode_id"])
    all_protocol = bool(episodes) and all(e["audit"].get("status") == "passed" for e in episodes)
    all_visual = (bool(episodes) and all(e["audit"].get("visual_evidence", {}).get("status") == "passed" for e in episodes)) if require_video else None
    grounded = _grounded_initialization(identity, episodes)
    report = {"schema": "genmo.closedloop_music_sweep_report.v1", "run_dir": str(root),
        "coverage": {"unique_audio_sha256_count": len({e["audio_sha256"] for e in episodes if e["audio_sha256"]}),
                     "unique_music_group_count": len(independent_groups), "evaluation_episode_count": len(episodes),
                     "global_music_group_conflicts": group_conflicts,
                     "expected_music_count": expected_music, "expected_episode_count": expected_episodes,
                     "expected_modes": list(expected_modes), "expected_split": expected_split,
                     "requested_episodes": expected_episodes, "attempted_episodes": len(episodes),
                     "not_attempted_episodes": max(0, expected_episodes-len(episodes)),
                     "complete": not issues, "issues": issues},
        "evidence": {"protocol_all_passed": all_protocol, "visual_all_passed": all_visual,
                     "video_required": require_video,
                     "video_index": video_audit,
                     "grounded_initialization": grounded,
                     "collection_complete": not issues and all_protocol and summary.get("exit_code", 0) == 0 and not (root/"failure.json").is_file() and (not require_video or (all_visual and video_audit["status"] == "passed")) and grounded["status"] == "passed",
                     "interpretation": "采集/证据完整不代表跟踪成功或音乐质量通过；失败和实际执行时长必须单独阅读"},
        "gmt_identity": {k: identity.get(k) for k in ("policy_sha256", "asset_sha256", "physics_device", "control_hz", "physics_hz", "runtime_fingerprint", "initial_ground_pose", "initial_ground_pose_actual")},
        "source_verification": summary.get("source_verification"), "run_acceptance": summary.get("acceptance"),
        "collection": collection,
        "termination": termination, "maximum_seconds": maximum_seconds,
        "run_infrastructure_error": _json(root/"failure.json") if (root/"failure.json").is_file() else None,
        "modes": modes, "episodes": episodes,
        "semantics": {"startup": "固定真实1秒warmup；前5/后10步和music首1秒均是诊断，无额外稳定阈值，不清空/复制history。H=50的真实历史仍包含最初控制响应，不能宣称纯稳态历史或零瞬态",
            "metrics": "主误差仅music，保留music首1秒；分位数按实际控制步汇总，较长episode贡献更多控制步",
            "boundaries": "reference_boundaries为参考来源切换时相邻控制步变化，不是同一时刻保护区改写",
            "contacts": "净接触力包含全部碰撞对象，不是纯地面力；支撑球高度为实际link pose的几何量",
            "torques": "computed/applied仅implicit PD估计，不是实际PhysX求解/施加力矩",
            "physics_timing": "physics_seconds包住整个env.step_control：动作处理、4个物理子步、scene状态更新、GMT观测/历史更新和终止计算；启用GUI/RTX且到达render_interval时，还包含仿真内部sim.render。它不是独立纯PhysX求解耗时",
            "video_timing": "video_capture_seconds是step_control之后单独计时的相机渲染/采集、22-body同步与固定镜头核验、帧SHA和编码writer写入/提交段；不含writer最终close/封装耗时",
            "step_timing": "step_seconds覆盖该控制步整段，还含推进前观测读取、实际GENMO历史追加、误差/物理诊断和参考查询等，因此不等于GMT推理+physics+video三项的简单相加；不含外层RPC传输、协调器JSONL写盘或最终视频封装",
            "latency": "以上是当前实际代码计时口径，未拆出独立纯物理求解耗时，不用带渲染的控制步耗时声称无渲染部署吞吐",
            "old_thresholds": "原阈值越界仅作诊断，不能解释为严格阈值下重跑得到的失败率",
            "comparison": "数据划分、终止阈值、渲染和初始化配置不同的实验不能直接合并成功率；本轮真实运行阈值见termination",
            "completion": "正常完成率使用每条min(上限,特征长度)并按50Hz向下取整；短曲自然结束是完成，但不进入足够长音乐的30秒率。Wilson区间按音乐条目计算；只有全局音乐组核验通过时才满足独立条目口径，且不代表训练集测评是未见数据泛化。",
            "smoothness": "速度、加速度和jerk仅用music连续有效支持点；加速度=diff(速度)*50，jerk=diff(加速度)*50。逐步RMS/max分位数不等于把所有关节元素展开后的分位数。",
            "thresholds": "non_yaw_orientation_error_rad与yaw_error_rad沿用本轮后端定义；global_orientation_error_rad仍作完整四元数误差诊断，不与新分离阈值混用。缺失的诊断不补零。",
            "music_source": "节拍来自已核验EDGE35特征的节拍通道，独立于是否读取/渲染音频；无特征节拍或无运动节拍时保留不可用原因。"}}
    return json_value(report)


def _fmt(value, digits=4):
    if value is None:
        return "未记录"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def _md(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _episode_quality(episode):
    """展示已有指标；幅度仅对21个关节各自的P95−P05作均值，不重定义误差。"""
    result = {}
    for kind in ("actual", "reference"):
        amplitudes = ((episode.get("motion") or {}).get(kind) or {}).get("joint_amplitude_p95_minus_p05_rad")
        result[f"{kind}_amplitude_rad"] = (float(_array(amplitudes, (21,), f"{kind} joint amplitude").mean())
                                              if amplitudes is not None else None)
        metric = (episode.get("music") or {}).get(kind)
        result[f"{kind}_beat_distance_s"] = (metric.get("mean_beat_distance_seconds")
                                              if metric and metric.get("status") == "available" else None)
    for field in ("joint_position_rmse_rad", "root_position_error_m", "end_effector_relative_height_error_m"):
        result[field] = ((episode.get("errors") or {}).get(field) or {}).get("p95")
    old = ((episode.get("original_threshold_diagnostics") or {}).get("metrics") or {}).get("global_orientation_error_rad")
    result["old_orientation_crossing_fraction"] = (old.get("music_crossing_fraction")
                                                   if old and old.get("threshold") == .6 else None)
    return result


def _percent(value):
    return "未记录" if value is None else f"{100*value:.2f}%"


def render_markdown(report):
    """中文结果表保留逐曲失败和不可用beat，避免只给一个平均分。"""
    coverage, evidence = report["coverage"], report["evidence"]
    lines = [f"# Stage8 {coverage['expected_music_count']} 首音乐评估报告", "", f"实验目录：`{report['run_dir']}`", "",
             f"不同音频 SHA：{coverage['unique_audio_sha256_count']}；实际 episode：{coverage['evaluation_episode_count']}；目标：{coverage['expected_music_count']} 首／{coverage['expected_episode_count']} 次。", "",
             f"数据划分：{coverage['expected_split']}；独立音乐组：{coverage['unique_music_group_count']}；请求/尝试/未尝试：{coverage['requested_episodes']}/{coverage['attempted_episodes']}/{coverage['not_attempted_episodes']}；单曲上限：{report['maximum_seconds']} s。", "",
             f"本轮实际终止配置：`{json.dumps(report['termination'], ensure_ascii=False)}`。", "",
             f"矩阵完整：{coverage['complete']}；协议全通过：{evidence['protocol_all_passed']}；视觉全通过：{evidence['visual_all_passed']}；视频索引：{evidence['video_index']['status']}。", "",
             "采集完整与跟踪成功分开判断。固定 1 s 真实 warmup，未增加稳定阈值；music 首 1 s 仍计入全部主指标。", "",
             f"1 mm初态：{evidence['grounded_initialization']['status']}，实际验证reset数：{evidence['grounded_initialization']['verified_resets']}；配置根Z：{_fmt(evidence['grounded_initialization'].get('configured_root_z_m'))} m。", "",
             "## 分模式结果", "", "|模式|启动失败/总数|music失败/已启动|music总秒数|每分钟失败|重规划|保护区修改|", "|---|---:|---:|---:|---:|---:|---:|"]
    if report.get("collection"):
        collection = report["collection"]
        lines[4:4] = ["本报告是显式多次运行的派生集合，按原始trace重新汇总；不是一次进程连续执行全部曲目。", "",
            f"集合清单SHA：`{collection['manifest_sha256']}`；源运行数：{collection['source_count']}；原运行冻结与源码证据全部通过。", ""]
    for mode, value in report["modes"].items():
        lines.append(f"|{mode}|{value['startup_failures']}/{value['episodes']}|{value['music_failures']}/{value['music_started_episodes']}|{_fmt(value['music_exposure_seconds'],2)}|{_fmt(value['music_failures_per_minute'])}|{value['replans']}|{_fmt(value['protected_reference']['modification_count'])}|")
    lines.extend(["", "## 完成率与按库覆盖", "",
                  "正常完成要求达到min(30s,特征长度)并以duration_limit或music_end正常结束；控制网格不足20ms尾段向下取整。短曲完整播放计成功，但不进入30秒率分母。失败类别互斥，未尝试与基础设施错误不伪装为跟踪失败。95%区间为Wilson区间。", "",
                  "|模式/数据集|请求/尝试|正常完成|完成率及95%区间|≥30秒曲目完成30秒|总时长覆盖|启动/控制/基础设施失败|自然结束/上限结束|", "|---|---:|---:|---|---|---:|---|---|"])
    for mode, value in report["modes"].items():
        groups = [("全部", value["completion"])] + [(name, result["completion"]) for name, result in value["by_dataset"].items()]
        for name, completion in groups:
            normal, long, counts = completion["normal_completion"], completion["reached_30s_among_eligible"], completion["ending_category_counts"]
            ci = f"{_percent(normal['fraction'])} [{_percent(normal['ci95_low'])}, {_percent(normal['ci95_high'])}]"
            failures = '/'.join(str(counts.get(k, 0)) for k in ("startup_failure", "control_failure", "infrastructure_error"))
            ends = '/'.join(str(counts.get(k, 0)) for k in ("natural_music_end", "duration_limit"))
            lines.append(f"|{mode}/{_md(name)}|{completion['requested_episodes']}/{completion['attempted_episodes']}|{normal['successes']}/{normal['total']}|{ci}|{long['successes']}/{long['total']} ({_percent(long['fraction'])})|{_percent(completion['total_execution_coverage_fraction'])}|{failures}|{ends}|")
    for mode, value in report["modes"].items():
        lines.extend(["", f"**{mode} 失败原因**：启动 `{json.dumps(value['startup_backend_reason_counts'],ensure_ascii=False)}`；music `{json.dumps(value['music_failure_reason_counts'],ensure_ascii=False)}`。", "",
                      "|music逐步误差|样本数|均值|P95|最大|", "|---|---:|---:|---:|---:|"])
        for key, stats in value["errors_pooled_music_control_steps"].items():
            lines.append(f"|{key}|{stats['count']}|{_fmt(stats['mean'])}|{_fmt(stats['p95'])}|{_fmt(stats['max'])}|")
        lines.extend(["", "|节拍对象|有效episode|有效音乐节拍|按节拍加权距离(s)|对应alignment|不可用原因|", "|---|---:|---:|---:|---:|---|"])
        for kind, stats in value["beat"].items():
            lines.append(f"|{kind}|{stats['available_episodes']}|{stats['available_music_beat_count']}|{_fmt(stats['music_beat_weighted_distance_seconds'])}|{_fmt(stats['alignment_of_weighted_distance'])}|{_md(stats['unavailable_reason_counts'])}|")
        lines.extend(["", "|耗时(s)|样本数|均值|P95|最大|", "|---|---:|---:|---:|---:|"])
        for key, stats in value["latency_seconds"].items():
            lines.append(f"|{key}|{stats['count']}|{_fmt(stats['mean'])}|{_fmt(stats['p95'])}|{_fmt(stats['max'])}|")
        lines.extend(["", f"前缀分布：`{json.dumps(value['prefix_frame_counts'])}`；P>18占比：{_percent(value['prefix_over_18_fraction'])}；漏决策：{value['decision_missed_count']}；请求拒绝：{value['request_rejected_count']}；计划拒绝：{value['plan_rejected_count']}；缓存最小余量：{_fmt(value['reference_cache_remaining_seconds']['min'])} s。", "",
                      f"协议计数：`{json.dumps(value['protocol'], ensure_ascii=False)}`。", ""])
        for label, field in (("运动幅度/平滑性逐步统计", "motion_pooled_music_control_steps"),
                             ("物理诊断逐步统计", "physical_pooled_music_control_steps"),
                             ("参考来源切换的相邻步变化", "reference_boundary_changes")):
            lines.extend([f"**{mode} {label}**", "", "|指标|样本数|均值|P50|P95|P99|最大|", "|---|---:|---:|---:|---:|---:|---:|"])
            for key, stats in value[field].items():
                lines.append(f"|{key}|{stats['count']}|{_fmt(stats['mean'])}|{_fmt(stats['p50'])}|{_fmt(stats['p95'])}|{_fmt(stats['p99'])}|{_fmt(stats['max'])}|")
        for dataset, result in value["by_dataset"].items():
            lines.extend(["", f"**{mode}/{_md(dataset)} 逐步误差**", "", "|指标|控制步数|均值|P50|P95|P99|最大|", "|---|---:|---:|---:|---:|---:|---:|"])
            for key, stats in result["errors_pooled_music_control_steps"].items():
                lines.append(f"|{key}|{stats['count']}|{_fmt(stats['mean'])}|{_fmt(stats['p50'])}|{_fmt(stats['p95'])}|{_fmt(stats['p99'])}|{_fmt(stats['max'])}|")
    lines.extend(["", "## 逐曲完成与失败时刻", "", "|数据/样本|模式|特征/目标/实际秒数|完成比例|结束类别|失败music时刻(s)|新阈值首次越界|", "|---|---|---|---:|---|---:|---|"])
    for ep in report["episodes"]:
        complete = ep["completion"]
        crossings = ep["startup_diagnostics"]["extended"]["threshold_first_crossing"]
        lines.append(f"|{_md(ep['dataset'])}/{_md(ep['sample_id'])}|{ep['mode']}|{_fmt(complete['feature_duration_seconds'],2)}/{_fmt(complete['target_duration_seconds'],2)}/{_fmt(ep['music_duration_seconds'],2)}|{_percent(complete['duration_completion_fraction'])}|{complete['category']}|{_fmt(complete['failure_music_seconds'],2)}|{_md(json.dumps(crossings,ensure_ascii=False))}|")
    lines.extend(["", "## 逐曲结果与视频", "", "|数据/样本|模式/seed|music秒数|终止原因|保护区修改|视频|", "|---|---|---:|---|---:|---|"])
    for ep in report["episodes"]:
        video = ep.get("video")
        link = f"[打开]({quote(video['relative_path'])})" if video and video["file_exists"] else "未记录"
        lines.append(f"|{_md(ep['dataset'])}/{_md(ep['sample_id'])}|{ep['mode']}/{ep['seed']}|{_fmt(ep['music_duration_seconds'],2)}|{_md(ep['reason'])} / {_md(ep['backend_reason'])}|{_fmt((ep.get('protected_reference') or {}).get('modification_count'))}|{link}|")
    lines.extend(["", "## 逐曲幅度与跟踪诊断", "",
                  "幅度为21个关节各自P95−P05的均值，实际与参考分别计算。所有误差和越界占比仅统计music控制步。脚肘指标只表示相对根坐标系的高度误差，不是三维末端距离。旧0.6rad姿态阈值越界是诊断，不是严格阈值下实测失败率。节拍缺失保留“未记录”。", "",
                  "|样本/模式/seed|实际幅度均值(rad)|参考幅度均值(rad)|关节误差P95(rad)|全局根位置误差P95(m)|脚肘相对高度误差P95(m)|实际/参考beat距离(s)|旧0.6rad越界占比|详细JSON|",
                  "|---|---:|---:|---:|---:|---:|---|---:|---|"])
    for ep in report["episodes"]:
        quality = _episode_quality(ep)
        detail = quote(ep.get("detail_json_relative_path", "music_sweep_report.json"))
        lines.append(f"|{_md(ep['sample_id'])}/{ep['mode']}/{ep['seed']}|{_fmt(quality['actual_amplitude_rad'])}|{_fmt(quality['reference_amplitude_rad'])}|{_fmt(quality['joint_position_rmse_rad'])}|{_fmt(quality['root_position_error_m'])}|{_fmt(quality['end_effector_relative_height_error_m'])}|{_fmt(quality['actual_beat_distance_s'])} / {_fmt(quality['reference_beat_distance_s'])}|{_percent(quality['old_orientation_crossing_fraction'])}|[详情]({detail})|")
    lines.extend(["", "## 启动与首秒实测诊断", "", "以下均为实际控制步，不设置静稳合格线。左右脚力包含所有碰撞。root速度由相邻实际位置差分得到。", "",
                  "|样本/模式|窗口|控制步|关节速度RMS P95(rad/s)|根竖直速度最大(m/s)|左/右净力Z均值(N)|左/右球底高度最小(m)|", "|---|---|---:|---:|---:|---|---|"])
    for ep in report["episodes"]:
        for key, label in (("warmup_first_5", "warmup前5"), ("warmup_last_10", "warmup后10"), ("music_first_1_second", "music首1s")):
            window = ep["startup_diagnostics"][key]
            force = " / ".join(_fmt((s or {}).get("mean")) for s in window["foot_net_force_z_n"])
            clearance = " / ".join(_fmt((s or {}).get("min")) for s in window["foot_min_support_clearance_m"])
            lines.append(f"|{_md(ep['sample_id'])}/{ep['mode']}|{label}|{window['control_steps']}|{_fmt((window['joint_speed_rms_rad_s'] or {}).get('p95'))}|{_fmt((window['root_vertical_speed_abs_m_s'] or {}).get('max'))}|{force}|{clearance}|")
    lines.extend(["", "## 解释边界", ""])
    lines.extend(f"- {text}" for text in report["semantics"].values())
    all_issues = coverage["issues"] + evidence["video_index"]["issues"] + evidence["grounded_initialization"]["issues"]
    if all_issues:
        lines.extend(["", "## 未满足项", ""])
        lines.extend(f"- {_md(issue)}" for issue in all_issues)
    return "\n".join(lines)+"\n"


def render_video_index(report):
    """离线 HTML 无脚本/外部资源；所有视频仍在原实验相对路径。"""
    cards = []
    for ep in report["episodes"]:
        esc = lambda value: html.escape(str(value), quote=True)
        video = ep.get("video")
        tag = (f'<video controls preload="none" src="{esc(quote(video["relative_path"]))}"></video>'
               if video and video["file_exists"] else ('<p class="missing">视频未记录或文件不存在</p>' if report["evidence"]["video_required"] else '<p>本轮纯数据评估，未请求视频。</p>'))
        quality = _episode_quality(ep)
        pairs = [
            ("实际/参考关节幅度均值(rad)", f"{_fmt(quality['actual_amplitude_rad'])} / {_fmt(quality['reference_amplitude_rad'])}"),
            ("关节误差 P95(rad)", _fmt(quality["joint_position_rmse_rad"])),
            ("全局根位置误差 P95(m)", _fmt(quality["root_position_error_m"])),
            ("脚肘相对根高度误差 P95(m)", _fmt(quality["end_effector_relative_height_error_m"])),
            ("实际/参考 beat距离(s)", f"{_fmt(quality['actual_beat_distance_s'])} / {_fmt(quality['reference_beat_distance_s'])}"),
            ("旧0.6rad姿态阈值越界占比", _percent(quality["old_orientation_crossing_fraction"])),
        ]
        diagnostics = '<dl>'+''.join(f'<dt>{esc(label)}</dt><dd>{esc(value)}</dd>' for label, value in pairs)+'</dl>'
        detail = esc(quote(ep.get("detail_json_relative_path", "music_sweep_report.json")))
        cards.append(f'<article><h2>{esc(ep["dataset"])} / {esc(ep["sample_id"])}</h2>'
                     f'<p>{esc(ep["mode"])} · seed {esc(ep["seed"])} · music {esc(ep["music_duration_seconds"])} s</p>'
                     f'<p>终止：{esc(ep["reason"])}；后端：{esc(ep["backend_reason"])}</p>{tag}{diagnostics}'
                     f'<p><a href="{detail}">详细 JSON：{esc((ep["source_run_id"]+"/") if ep.get("source_run_id") else "")}{esc(ep["episode_id"])}</a></p></article>')
    title = "Stage8 音乐视频索引" if report["evidence"]["video_required"] else "Stage8 音乐数据索引"
    introduction = "逐曲实际 Isaac 渲染。前 1 秒为真实 warmup；启动失败视频可能只有静音。" if report["evidence"]["video_required"] else "本轮仅统计真实 Isaac 轨迹，未渲染视频。前 1 秒为真实 warmup。"
    if report.get("collection"):
        introduction += f"这是{report['collection']['source_count']}次原始运行的显式派生集合，逐条保留source_run_id和原episode编号。"
    return ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{title}</title><style>body{{font:16px sans-serif;max-width:1280px;margin:2rem auto;padding:0 1rem;background:#f4f5f7;color:#20252a}}'
            'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:1rem}article{background:white;padding:1rem;border-radius:8px}'
            'h2{font-size:1.05rem;overflow-wrap:anywhere}video{width:100%;max-height:480px}.missing{color:#a20}p{line-height:1.5}'
            'dl{display:grid;grid-template-columns:1fr auto;gap:.4rem;font-size:.85rem}dt{overflow-wrap:anywhere}dd{margin:0;text-align:right}</style>'
            f'<h1>{title}</h1><p>{introduction}'
            '失败、节拍、跟踪误差和保护区证据请结合 <a href="music_sweep_report.md">详细报告</a> 与 '
            '<a href="music_sweep_report.json">JSON</a>。幅度为21关节各自P95−P05的均值；脚肘项是相对根高度误差，'
            '不是三维末端距离。旧阈值越界不等于严格阈值实测失败率。视频按需加载，不创建网络服务。</p><main>'
            + ''.join(cards) + '</main></html>\n')


def render_csv(report):
    """逐次尝试一行；数值缺测留空，不以0替代，所有列均来自同次实验。"""
    rows = []
    for episode in report["episodes"]:
        completion = episode["completion"]
        row = {key: episode.get(key) for key in ("source_run_id", "episode_id", "source_trace_relative_path", "dataset", "group_id", "sample_id", "split", "mode", "seed", "reason", "backend_reason", "audio_sha256", "music_feature_sha256", "manifest_sha256", "music_duration_seconds", "warmup_duration_seconds", "total_duration_seconds", "replan_count", "prefix_over_18_count")}
        row.update({key: completion.get(key) for key in ("category", "feature_duration_seconds", "target_duration_seconds", "executable_target_seconds", "normal_complete", "duration_completion_fraction", "eligible_for_30s", "reached_30s_without_failure", "failure_music_seconds", "failure_total_seconds")})
        extended = episode["startup_diagnostics"]["extended"]
        for category in ("errors", "motion", "physical", "boundary_changes"):
            for metric, stats in extended[category].items():
                for stat in ("count", "mean", "p50", "p95", "p99", "max", "min"):
                    if stat in stats:
                        row[f"{category}.{metric}.{stat}"] = stats[stat]
        for metric, values in extended["threshold_first_crossing"].items():
            row[f"threshold.{metric}.first_music_seconds"] = values["music_seconds"]
            row[f"threshold.{metric}.first_value"] = values["value"]
        for metric, count in extended["threshold_crossing_counts"].items():
            row[f"threshold.{metric}.crossing_count"] = count
            row[f"threshold.{metric}.supported_steps"] = extended["threshold_supported_steps"][metric]
        for kind, amplitudes in extended["joint_amplitude_p95_minus_p05_rad"].items():
            row[f"{kind}.joint_amplitude_mean_rad"] = float(np.mean(amplitudes)) if amplitudes is not None else None
            for i, value in enumerate(amplitudes or []):
                row[f"{kind}.joint_{i:02d}_amplitude_rad"] = value
        for metric, stats in (episode.get("latency_seconds") or {}).items():
            for stat, value in stats.items():
                row[f"latency.{metric}.{stat}"] = value
        for stat, value in (episode.get("prefix_frames") or {}).items():
            row[f"prefix_frames.{stat}"] = value
        row["protected_modification_count"] = (episode.get("protected_reference") or {}).get("modification_count")
        row["protocol_status"] = episode["audit"].get("status")
        for kind in ("actual", "reference"):
            metric = (episode.get("music") or {}).get(kind) or {}
            row[f"beat.{kind}.status"] = metric.get("status", (episode.get("music") or {}).get("status", "not_recorded"))
            row[f"beat.{kind}.distance_seconds"] = metric.get("mean_beat_distance_seconds") if metric.get("status") == "available" else None
        rows.append(row)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def write_music_sweep_report(run_dir, *, expected_music=20, expected_modes=("paused", "latency"),
                            expected_split="val", require_video=True, maximum_seconds=30., collection_manifest=None):
    root = Path(run_dir).resolve()
    outputs = {"json": root/"music_sweep_report.json", "markdown": root/"music_sweep_report.md",
               "csv": root/"music_sweep_rows.csv", "html": root/("music_sweep_videos.html" if require_video else "music_sweep_data.html")}
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError("Music sweep report already exists; refusing overwrite")
    report = build_music_sweep_report(root, expected_music=expected_music, expected_modes=expected_modes,
                                      expected_split=expected_split, require_video=require_video, maximum_seconds=maximum_seconds,
                                      collection_manifest=collection_manifest)
    contents = {"json": json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n",
                "markdown": render_markdown(report), "html": render_video_index(report), "csv": render_csv(report)}
    for key, path in outputs.items():
        with path.open("x", encoding="utf-8") as stream:
            stream.write(contents[key])
    return report, {key: str(path) for key, path in outputs.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-music", type=int, default=20)
    parser.add_argument("--expected-modes", nargs="+", choices=("paused", "latency"), default=("paused", "latency"))
    parser.add_argument("--expected-split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--no-video", action="store_true", help="明确不要求视频证据；不影响物理/协议核验")
    parser.add_argument("--maximum-seconds", type=float, default=30.)
    parser.add_argument("--collection-manifest", type=Path, help="显式多run集合清单；不创建/要求假的合并run_summary")
    args = parser.parse_args(argv)
    if args.expected_music <= 0:
        parser.error("expected-music must be positive")
    if not math.isfinite(args.maximum_seconds) or args.maximum_seconds <= 0:
        parser.error("maximum-seconds must be finite and positive")
    if len(set(args.expected_modes)) != len(args.expected_modes):
        parser.error("expected-modes must not contain duplicates")
    report, outputs = write_music_sweep_report(args.run_dir, expected_music=args.expected_music,
        expected_modes=tuple(args.expected_modes), expected_split=args.expected_split,
        require_video=not args.no_video, maximum_seconds=args.maximum_seconds, collection_manifest=args.collection_manifest)
    print(json.dumps({"coverage": report["coverage"], "evidence": report["evidence"], "outputs": outputs}, ensure_ascii=False))
    return 0 if report["evidence"]["collection_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
