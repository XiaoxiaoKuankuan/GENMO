#!/usr/bin/env python3
"""只读汇总 Stage8 二十首音乐、两种时序模式的真实 Isaac 视频评估。

本工具不导入 Actor、Isaac 或训练入口，不读取动作监督标签，不启动 GPU。它读取
一个已结束实验的 run_summary.json、audit.json、gmt_identity.json、事件及各个
episode 的初态/逐控制步轨迹。逐条流式读取体积较大的 JSONL，只保存统计所需的
标量、很短的启动窗口和数组摘要，不把四十条完整物理/渲染轨迹同时载入内存。

报告按音频 SHA 校验二十首不同音乐，每首 paused/latency 各一次；同曲多种子不能
冒充不同歌曲。startup 失败、music 失败和音乐自然结束分别统计，主误差只使用
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
import hashlib
import html
import json
import math
from pathlib import Path
import re
import sys
from urllib.parse import quote

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from gem.closedloop.baseline_metrics import distribution, json_value  # noqa: E402


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


def _trace(root, episode, pool):
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
    for name in ("errors", "latency"):
        pool.setdefault(name, defaultdict(list))
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            digest.update(line)
            if not line.strip():
                continue
            row = json.loads(line)
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
    if previous_tick != episode["terminal_snapshot"]["tick"]:
        raise ValueError("Trace does not reach terminal snapshot")
    if phase_counts["music"] != episode["recorded_music_control_steps"]:
        raise ValueError("Trace music frame count differs from summary")
    return {"trace_sha256": digest.hexdigest(), "control_steps": sum(phase_counts.values()),
            "phase_counts": dict(phase_counts),
            "warmup_all": _window(phase_rows["warmup"]), "music_all": _window(phase_rows["music"]),
            "warmup_first_5": _window(warmup_first), "warmup_last_10": _window(list(warmup_last)),
            "music_first_1_second": _window(music_first),
            "initial_snapshot": initial,
            "video_frames": {"count": frame_count, "first": first_frame, "last": last_frame}}


def _beat_summary(episodes, which):
    valid, unavailable = [], Counter()
    for episode in episodes:
        music = episode.get("music", {})
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


def build_music_sweep_report(run_dir, *, expected_music=20, expected_modes=("paused", "latency")):
    """构建 JSON 可序列化报告，不写文件；调用方须等 worker 和视频封装均结束。"""
    root = Path(run_dir).resolve()
    summary = _json(root / "run_summary.json")
    audit = _json(root / "audit.json") if (root / "audit.json").is_file() else {}
    identity = _json(root / "gmt_identity.json") if (root / "gmt_identity.json").is_file() else {}
    source = [e for e in summary["episodes"] if e.get("mode") != "calibration"]
    audit_by_id = {e["episode_id"]: e for e in audit.get("episodes", [])}
    pools = defaultdict(lambda: {"errors": defaultdict(list), "latency": defaultdict(list), "prefix": []})
    episodes, issues, ids = [], [], set()
    for original in source:
        episode_id = original["episode_id"]
        if episode_id in ids:
            raise ValueError(f"Duplicate episode id: {episode_id}")
        ids.add(episode_id)
        sample = original["sample"]
        row = sample["row"]
        sha = row.get("source_audio_sha256", "")
        if re.fullmatch(r"[0-9a-f]{64}", sha) is None:
            raise ValueError(f"Missing audio SHA for {episode_id}")
        trace = _trace(root, original, pools[original["mode"]])
        ep = {key: original.get(key) for key in ("episode_id", "mode", "seed", "reason", "backend_reason", "failed", "startup_failure",
                "music_duration_seconds", "warmup_duration_seconds", "total_duration_seconds", "replan_count", "errors", "music", "motion",
                "reference_boundaries", "protected_reference", "prefix_frames", "prefix_over_18_count", "reference_buffer", "event_counts", "rejection_codes",
                "original_threshold_diagnostics", "latency_seconds", "trace_integrity")}
        ep.update(dataset=sample.get("dataset", original.get("dataset")), sample_id=row["sample_id"], group_id=sample.get("group_id"),
                  audio_sha256=sha, music_feature_sha256=row.get("source_music_feature_sha256"), manifest_sha256=sample.get("manifest_sha256"),
                  split=row.get("split"), startup_diagnostics=trace,
                  audit=audit_by_id.get(episode_id, {"status": "not_recorded"}))
        detail_path = _local(root, original["artifacts"]["trace"]).with_name("summary.json")
        ep["detail_json_relative_path"] = (detail_path.relative_to(root).as_posix()
                                           if detail_path.is_file() else "music_sweep_report.json")
        if ep["split"] != "val":
            issues.append(f"{episode_id}: split is not val")
        episodes.append(ep)
    index = {ep["episode_id"]: ep for ep in episodes}
    event_counts = defaultdict(Counter)
    path = root / "events.jsonl"
    if path.is_file():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                event = json.loads(line)
                ep = index.get(event.get("episode_id"))
                if ep is None:
                    continue
                mode, name = ep["mode"], event["event"]
                event_counts[mode][name] += 1
                if name == "plan_prepared":
                    pools[mode]["prefix"].append(int(event["prefix_frames"]))
                    for key in ("end_to_end_seconds", "actor_seconds", "prepare_seconds"):
                        if event.get(key) is not None:
                            pools[mode]["latency"][key].append(float(event[key]))
    else:
        issues.append("events.jsonl missing: pooled request timing/prefix unavailable")
    songs = defaultdict(list)
    for ep in episodes:
        songs[ep["audio_sha256"]].append(ep)
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
    videos, video_audit = _videos(root, summary, episodes)
    modes = {}
    for mode in sorted({e["mode"] for e in episodes}):
        chosen = [e for e in episodes if e["mode"] == mode]
        startup = [e for e in chosen if e["startup_failure"]]
        music_started = [e for e in chosen if e["startup_diagnostics"]["phase_counts"].get("music", 0)]
        music_failed = [e for e in music_started if e["failed"]]
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
    for ep in episodes:
        ep["video"] = videos.get(ep["episode_id"])
    all_protocol = bool(episodes) and all(e["audit"].get("status") == "passed" for e in episodes)
    all_visual = bool(episodes) and all(e["audit"].get("visual_evidence", {}).get("status") == "passed" for e in episodes)
    grounded = _grounded_initialization(identity, episodes)
    report = {"schema": "genmo.closedloop_music_sweep_report.v1", "run_dir": str(root),
        "coverage": {"unique_audio_sha256_count": len(songs), "evaluation_episode_count": len(episodes),
                     "expected_music_count": expected_music, "expected_episode_count": expected_episodes,
                     "expected_modes": list(expected_modes), "complete": not issues, "issues": issues},
        "evidence": {"protocol_all_passed": all_protocol, "visual_all_passed": all_visual,
                     "video_index": video_audit,
                     "grounded_initialization": grounded,
                     "collection_complete": not issues and all_protocol and all_visual and video_audit["status"] == "passed" and grounded["status"] == "passed",
                     "interpretation": "采集/证据完整不代表跟踪成功或音乐质量通过；失败和实际执行时长必须单独阅读"},
        "gmt_identity": {k: identity.get(k) for k in ("policy_sha256", "asset_sha256", "physics_device", "control_hz", "physics_hz", "runtime_fingerprint", "initial_ground_pose", "initial_ground_pose_actual")},
        "source_verification": summary.get("source_verification"), "run_acceptance": summary.get("acceptance"),
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
            "comparison": "grounded初始化改变了实验条件，应作为新版本基线，不混入旧悬空初态的总体平均"}}
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
    lines = ["# Stage8 二十首音乐评估报告", "", f"实验目录：`{report['run_dir']}`", "",
             f"不同音频 SHA：{coverage['unique_audio_sha256_count']}；实际 episode：{coverage['evaluation_episode_count']}；目标：{coverage['expected_music_count']} 首／{coverage['expected_episode_count']} 次。", "",
             f"矩阵完整：{coverage['complete']}；协议全通过：{evidence['protocol_all_passed']}；视觉全通过：{evidence['visual_all_passed']}；视频索引：{evidence['video_index']['status']}。", "",
             "采集完整与跟踪成功分开判断。固定 1 s 真实 warmup，未增加稳定阈值；music 首 1 s 仍计入全部主指标。", "",
             f"1 mm初态：{evidence['grounded_initialization']['status']}，实际验证reset数：{evidence['grounded_initialization']['verified_resets']}；配置根Z：{_fmt(evidence['grounded_initialization'].get('configured_root_z_m'))} m。", "",
             "## 分模式结果", "", "|模式|启动失败/总数|music失败/已启动|music总秒数|每分钟失败|重规划|保护区修改|", "|---|---:|---:|---:|---:|---:|---:|"]
    for mode, value in report["modes"].items():
        lines.append(f"|{mode}|{value['startup_failures']}/{value['episodes']}|{value['music_failures']}/{value['music_started_episodes']}|{_fmt(value['music_exposure_seconds'],2)}|{_fmt(value['music_failures_per_minute'])}|{value['replans']}|{_fmt(value['protected_reference']['modification_count'])}|")
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
               if video and video["file_exists"] else '<p class="missing">视频未记录或文件不存在</p>')
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
                     f'<p><a href="{detail}">详细 JSON：{esc(ep["episode_id"])}</a></p></article>')
    return ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>Stage8 音乐视频索引</title><style>body{font:16px sans-serif;max-width:1280px;margin:2rem auto;padding:0 1rem;background:#f4f5f7;color:#20252a}'
            'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:1rem}article{background:white;padding:1rem;border-radius:8px}'
            'h2{font-size:1.05rem;overflow-wrap:anywhere}video{width:100%;max-height:480px}.missing{color:#a20}p{line-height:1.5}'
            'dl{display:grid;grid-template-columns:1fr auto;gap:.4rem;font-size:.85rem}dt{overflow-wrap:anywhere}dd{margin:0;text-align:right}</style>'
            '<h1>Stage8 音乐视频索引</h1><p>逐曲实际 Isaac 渲染。前 1 秒为真实 warmup；启动失败视频可能只有静音。'
            '失败、节拍、跟踪误差和保护区证据请结合 <a href="music_sweep_report.md">详细报告</a> 与 '
            '<a href="music_sweep_report.json">JSON</a>。幅度为21关节各自P95−P05的均值；脚肘项是相对根高度误差，'
            '不是三维末端距离。旧阈值越界不等于严格阈值实测失败率。视频按需加载，不创建网络服务。</p><main>'
            + ''.join(cards) + '</main></html>\n')


def write_music_sweep_report(run_dir, *, expected_music=20):
    root = Path(run_dir).resolve()
    outputs = {"json": root/"music_sweep_report.json", "markdown": root/"music_sweep_report.md", "html": root/"music_sweep_videos.html"}
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError("Music sweep report already exists; refusing overwrite")
    report = build_music_sweep_report(root, expected_music=expected_music)
    contents = {"json": json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n",
                "markdown": render_markdown(report), "html": render_video_index(report)}
    for key, path in outputs.items():
        with path.open("x", encoding="utf-8") as stream:
            stream.write(contents[key])
    return report, {key: str(path) for key, path in outputs.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-music", type=int, default=20)
    args = parser.parse_args(argv)
    if args.expected_music <= 0:
        parser.error("expected-music must be positive")
    report, outputs = write_music_sweep_report(args.run_dir, expected_music=args.expected_music)
    print(json.dumps({"coverage": report["coverage"], "evidence": report["evidence"], "outputs": outputs}, ensure_ascii=False))
    return 0 if report["evidence"]["collection_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
