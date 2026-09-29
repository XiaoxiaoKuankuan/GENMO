#!/usr/bin/env python3
"""只读复核 Stage8 已结束基线的时间、计划归属与历史推进证据。

本工具不启动模型、仿真或训练，不改写原始事件／逐步轨迹，只读取运行产物并
排他创建独立 audit.json。以事件顺序重建每个参考时刻的 plan_id 来源，逐控制步
验证 600 Hz 整数时钟、21 点窗口、50 Hz 步进、4 个 PhysX 子步及无 episode
串接。初态与末态历史计数按实际记录作差，不猜测 reset 是否额外计数。

当前已有运行的 advance 事件中 proprio_history_update_count 为 null，是日志
字段名遗漏；本工具明确记录这项证据边界，使用末态 history_append_count 和
逐时刻严格 append 实现提供的独立计数核验，不回填或冒充现场事件计数。
保护区修改计数和重算误差依据每次真实 commit ACK 检查；这与独立重放源姿态
再比较完整六数组是不同证明层，报告不会把前者说成后者。

可选视觉审计逐帧独立重算 22 个 PhysX/USD link 位姿误差，核对同 tick 的实际
物理诊断、固定相机、连续录像帧编号与 RGB SHA。旧数值基线缺少这些字段时
保留协议审计结论，但绝不标为视觉通过。接触力、支撑球、关节运动与执行器
隐式 PD 力矩估计按 warmup/music 分开汇总；净接触力包含所有碰撞，不能当成纯地面力。
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

CLOCK_HZ, CONTROL_TICKS = 600, 12
FIELDS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                yield number, json.loads(line)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, message):
    _require(isinstance(value, int) and not isinstance(value, bool), message)
    return value


def _audit_episode(root: Path, summary: dict, events: list[dict]) -> dict:
    trace_path = root / summary["artifacts"]["trace"]
    initial = _read_json(trace_path.parent / "initial.json")["snapshot"]
    terminal = summary["terminal_snapshot"]
    episode = summary["episode_id"]
    _require(initial["episode_id"] == terminal["episode_id"] == episode, "initial/terminal episode mismatch")
    expected_tick = _integer(initial["tick"], "initial tick is not integer")
    _require(expected_tick == 0, "evaluation episode did not start at zero")
    source_owner = initial["plan_id"]
    installed_plan_id = source_owner
    commitments = []
    pending = {}
    trace_iter = iter(_jsonl(trace_path))
    errors = {"max_protected_position_recompute_error": 0., "max_protected_velocity_recompute_error": 0.}
    counts = Counter()
    warnings = set()
    last_row = None
    macro_steps = Counter()
    terminal_event_seen = False
    gmt_initial = _integer(initial["gmt_history_update_count"], "missing initial GMT history count")
    proprio_initial = _integer(initial["history_append_count"], "missing initial proprio history count")

    def owner_at(tick):
        owner = source_owner
        for boundary, plan_id in commitments:
            if tick > boundary:
                owner = plan_id
        return owner

    def arrival(event):
        request_tick = _integer(event["request_tick"], "noninteger request tick")
        measured = float(event["end_to_end_seconds"])
        _require(math.isfinite(measured) and measured >= 0, "invalid measured latency")
        expected = request_tick if summary["mode"] == "paused" else math.ceil((request_tick + measured*CLOCK_HZ)/CONTROL_TICKS)*CONTROL_TICKS
        _require(event["arrival_tick"] == expected, "arrival does not match measured latency/control boundary")
        _require(event["effective_tick"] == expected, "plan applied/rejected at wrong control boundary")
        _require(event["effective_tick"] == expected_tick, "plan event rewrote past or future simulation time")

    for event in events:
        kind = event["event"]
        if terminal_event_seen:
            _require(kind not in {"advance", "plan_committed"}, "work continued after terminal transition")
        if kind == "plan_prepared":
            _require(event["request_tick"] == expected_tick, "request did not use current actual state time")
            _require(event["request_tick"] % 300 == 0, "request not on 0.5 s decision grid")
            _require(event["parent_plan_id"] == installed_plan_id, "prepared plan used a stale parent")
            protected, p = event["protected_end_tick"], event["prefix_frames"]
            _require(protected % CONTROL_TICKS == 0, "protected boundary not at 50 Hz")
            _require(protected >= math.ceil(event["deadline_tick"]/CONTROL_TICKS)*CONTROL_TICKS+120, "protection misses arrival plus GMT lookahead")
            source_last = event["request_tick"]+(p-1)*20
            required_last = math.ceil((protected+CONTROL_TICKS)/20)*20
            _require(12 <= p < 120 and source_last >= required_last, "prefix misses difference/interpolation halo")
            pending[event["plan_id"]] = event
            counts["prepared_plans"] += 1
        elif kind == "plan_committed":
            arrival(event)
            _require(event["plan_id"] in pending, "commit has no corresponding prepared plan")
            prepared = pending.pop(event["plan_id"])
            ack = event["acknowledgement"]
            _require(ack["episode_id"] == episode and ack["plan_id"] == event["plan_id"], "commit identity mismatch")
            _require(ack["parent_plan_id"] == prepared["parent_plan_id"], "commit parent mismatch")
            _require(event["effective_tick"] <= event["deadline_tick"], "late plan installed")
            _require(ack["protected_modification_count"] == 0, "committed protected arrays were modified")
            _require(ack["protected_end_tick"] >= prepared["protected_end_tick"], "commit protection shrank")
            for key, limit in (("max_position_error", 1e-5), ("max_velocity_error", 1e-4)):
                value = ack[key]
                _require(math.isfinite(value) and 0 <= value <= limit, "invalid protected recomputation error")
            errors["max_protected_position_recompute_error"] = max(errors["max_protected_position_recompute_error"], ack["max_position_error"])
            errors["max_protected_velocity_recompute_error"] = max(errors["max_protected_velocity_recompute_error"], ack["max_velocity_error"])
            commitments.append((ack["protected_end_tick"], event["plan_id"]))
            installed_plan_id = event["plan_id"]
            counts["commits"] += 1
        elif kind == "plan_rejected":
            arrival(event)
            _require(event["plan_id"] in pending, "rejection has no prepared plan")
            pending.pop(event["plan_id"])
            if event.get("code") == "late_plan":
                _require(event["effective_tick"] > event["deadline_tick"], "on-time plan incorrectly logged as late")
            counts["rejected_plans"] += 1
        elif kind == "advance":
            begin, end = event["begin_tick"], event["end_tick"]
            controls, physics = event["executed_control_steps"], event["executed_physics_steps"]
            _require(begin == expected_tick, "advance begin skips/repeats physical time")
            _require(0 <= controls <= event["requested_control_steps"] <= 25, "invalid bounded advance count")
            _require(end-begin == controls*CONTROL_TICKS and physics == controls*4, "control/physics clock mismatch")
            _require(controls == event["requested_control_steps"] or event["done"], "short transition without terminal state")
            if event.get("gmt_history_update_count") is not None:
                _require(event["gmt_history_update_count"]-gmt_initial == end//CONTROL_TICKS, "GMT history duplicated/missed update")
            if event.get("proprio_history_update_count") is None:
                warnings.add("proprio_history_event_count_missing: event field is null; terminal history_append_count verified separately")
            else:
                _require(event["proprio_history_update_count"]-proprio_initial == end//CONTROL_TICKS, "proprio history duplicated/missed update")
            for tick in range(begin+CONTROL_TICKS, end+1, CONTROL_TICKS):
                try:
                    line, row = next(trace_iter)
                except StopIteration as exc:
                    raise AssertionError("executed control step has no trace") from exc
                _require(row["episode_id"] == episode and row["env_id"] == initial["env_id"], f"trace {line}: episode/env mismatch")
                _require(row["tick"] == tick and row["control_tick_begin"] == tick-CONTROL_TICKS, f"trace {line}: clock mismatch")
                _require(row["reference_tick"] == tick, f"trace {line}: feedback reference time mismatch")
                _require(row["reference_plan_id"] == owner_at(tick), f"trace {line}: feedback owner mismatch")
                expected_window = list(range(tick-132, tick+109, CONTROL_TICKS))
                _require(row["consumed_reference_ticks"] == expected_window, f"trace {line}: consumed window timing mismatch")
                _require(row["consumed_plan_ids"] == [owner_at(t) for t in expected_window], f"trace {line}: consumed plan identity mismatch")
                for field in FIELDS:
                    _require(np.isfinite(np.asarray(row["reference"][field], dtype=float)).all(), f"trace {line}: reference nonfinite")
                _require(np.isfinite(np.asarray(row["actual_qpos"], dtype=float)).all(), f"trace {line}: actual pose nonfinite")
                _require(row["phase"] == event["phase"], f"trace {line}: phase mismatch")
                _require(row.get("decision_id") == event.get("decision_id"), f"trace {line}: decision mismatch")
                last_row = row
                counts["control_steps"] += 1
            expected_tick = end
            counts["physics_steps"] += physics
            counts["advance_calls"] += 1
            if controls == 25:
                counts["full_25_control_100_physics_calls"] += 1
            if event["phase"] == "music":
                macro_steps[event["decision_id"]] += controls
            if event["done"]:
                terminal_event_seen = True
                counts["terminal_transitions"] += 1
                _require(end == terminal["tick"], "terminal transition continued before episode snapshot")
        elif kind == "pending_invalidated":
            _require(event["plan_id"] in pending, "invalidated plan was not pending")
            pending.pop(event["plan_id"])

    _require(next(trace_iter, None) is None, "trace contains unaccounted control steps")
    _require(expected_tick == terminal["tick"], "terminal tick differs from executed trace")
    _require(terminal["history_append_count"] == terminal["tick"]//CONTROL_TICKS+1, "terminal proprio history count mismatch")
    _require(terminal["history_append_count"]-proprio_initial == counts["control_steps"], "proprio initial/final delta mismatch")
    _require(terminal["gmt_history_update_count"]-gmt_initial == counts["control_steps"], "GMT initial/final delta mismatch")
    _require(summary["executed_control_steps"] == counts["control_steps"], "summary control count mismatch")
    _require(summary["executed_physics_steps"] == counts["physics_steps"], "summary physics count mismatch")
    if terminal.get("done"):
        _require(terminal_event_seen, "terminal snapshot lacks terminal advance event")
        _require(last_row is None or last_row["tick"] == terminal["tick"], "terminal trace not pre-reset final state")
    for decision, count in macro_steps.items():
        _require(0 <= count <= 25, "more than 25 controls in one upper interval")
        if count < 25:
            _require(decision == max(macro_steps), "incomplete nonfinal upper interval")
        else:
            counts["complete_upper_intervals_25_100"] += 1
    if summary["mode"] != "calibration":
        _require(not pending, "ended episode retained unaccounted prepared plans")
    return {"episode_id": episode, "mode": summary["mode"], "dataset": summary.get("dataset"),
            "status": "passed", "counts": dict(counts), "macro_control_counts": dict(macro_steps),
            "initial_history_counts": {"gmt": gmt_initial, "proprio": proprio_initial},
            "terminal_history_counts": {"gmt": terminal["gmt_history_update_count"], "proprio": terminal["history_append_count"]},
            **errors, "warnings": sorted(warnings)}


def _finite_array(value, shape, label):
    array = np.asarray(value, dtype=np.float64)
    _require(array.shape == shape and np.isfinite(array).all(), f"{label}: invalid shape or nonfinite")
    return array


def _pose_errors(physical, rendered):
    def rotations(quaternions):
        norm = np.linalg.norm(quaternions, axis=1)
        _require(np.all(np.abs(norm-1.) <= 1e-3), "render pose quaternion is not normalized")
        w, x, y, z = (quaternions/norm[:, None]).T
        return np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                         2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                         2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), axis=1).reshape(-1,3,3)
    position = np.linalg.norm(physical[:, :3]-rendered[:, :3], axis=-1)
    matrix_difference = rotations(physical[:, 3:])-rotations(rendered[:, 3:])
    angle = 2*np.arcsin(np.clip(np.linalg.norm(matrix_difference, axis=(1,2))/(2*np.sqrt(2)), 0, 1))
    return float(position.max()), float(angle.max())


def _range_stats(values, *, absolute=False):
    array = np.asarray(values, dtype=float)
    if not array.size:
        return None
    if absolute:
        array = np.abs(array)
    return {"min": float(array.min()), "p50": float(np.percentile(array,50)),
            "p95": float(np.percentile(array,95)), "max": float(array.max())}


def _phase_physics_summary(values):
    if not values["ticks"]:
        return {"frames": 0, "duration_seconds": 0., "actual_joint_motion": None}
    joints = np.asarray(values["joints"])
    root = np.asarray(values["root"])
    contact = np.asarray(values["forces"])
    clearance = np.asarray(values["clearance"])
    norms = np.linalg.norm(contact, axis=-1)
    delta = np.diff(joints, axis=0)
    return {"frames": len(joints), "duration_seconds": len(joints)/50.,
            "first_tick": values["ticks"][0], "last_tick": values["ticks"][-1],
            "actual_joint_motion": {"order": "GMT Isaac joint_names",
                "peak_to_peak_rad": np.ptp(joints, axis=0).tolist(),
                "p95_minus_p05_rad": (np.percentile(joints,95,axis=0)-np.percentile(joints,5,axis=0)).tolist(),
                "max_adjacent_delta_rad": float(np.abs(delta).max()) if len(delta) else None,
                "exact_repeated_adjacent_joint_frames": int(np.all(delta==0,axis=1).sum()) if len(delta) else 0},
            "actual_root": {"xyz_min_m": root.min(axis=0).tolist(), "xyz_max_m": root.max(axis=0).tolist(),
                            "xy_path_m": float(np.linalg.norm(np.diff(root[:,:2],axis=0),axis=1).sum())},
            "foot_order": ["l_ankle_roll_link", "r_ankle_roll_link"],
            "foot_net_contact_force_norm_n": [_range_stats(norms[:,i]) for i in range(2)],
            "foot_net_contact_force_z_n": [_range_stats(contact[:,i,2]) for i in range(2)],
            "foot_min_support_clearance_m": [_range_stats(clearance[:,i]) for i in range(2)],
            "both_feet_clearance_above_5mm_frames": int(np.all(clearance>.005,axis=1).sum()),
            "either_foot_net_contact_norm_above_1n_frames": int(np.any(norms>1.,axis=1).sum()),
            "applied_joint_torque_estimate_abs_nm": _range_stats(values["applied"],absolute=True),
            "computed_joint_torque_estimate_abs_nm": _range_stats(values["computed"],absolute=True)}


def _audit_render_physics(root, summary):
    """独立于协议结论审查可选视觉字段；缺证据与证据失败都不能算视觉通过。"""
    path = root / summary["artifacts"]["trace"]
    phases = {phase: defaultdict(list) for phase in ("warmup", "music")}
    visual_issues, physics_issues = [], []
    total = video_expected = render_count = physical_count = 0
    first_index = previous_index = None
    previous_hash = None
    unique_hashes = set()
    repeated_hashes = 0
    first_camera = None
    expected_body_names = expected_joint_names = expected_asset_sha = None
    identity_path = root / "gmt_identity.json"
    if identity_path.exists():
        identity = _read_json(identity_path)
        expected_asset_sha = identity.get("asset_sha256")
        binding = identity.get("runtime_fingerprint", {}).get("parameters", {})
        expected_body_names, expected_joint_names = binding.get("body_names"), binding.get("joint_names")
    maxima = {"position_m": 0., "orientation_rad": 0., "physics_capture_position_m": 0., "physics_capture_orientation_rad": 0.}
    render_body_names = None
    for line, row in _jsonl(path):
        total += 1
        physical = row.get("physical_diagnostics")
        render = row.get("render_pose_sync")
        frame_index, frame_hash = row.get("video_frame_index"), row.get("video_frame_sha256")
        wants_video = frame_index is not None or frame_hash is not None or render is not None
        if wants_video:
            video_expected += 1
        if physical is not None:
            try:
                _require(physical["control_tick"] == row["tick"], "physical diagnostic tick mismatch")
                _require(physical["env_index"] == row["env_id"], "physical diagnostic env mismatch")
                names, joints = physical["body_names"], physical["joint_names"]
                _require(len(names)==22 and len(set(names))==22, "physical body names must contain 22 unique links")
                _require(len(joints)==21 and len(set(joints))==21, "physical joint names must contain 21 unique joints")
                if expected_body_names is not None:
                    _require(names==expected_body_names and joints==expected_joint_names, "physical order differs from runtime binding")
                if expected_asset_sha is not None:
                    _require(physical["urdf_sha256"]==expected_asset_sha, "physical URDF differs from runtime binding")
                _finite_array(physical["body_link_pos_w"], (22,3), "actual body position")
                quaternions = _finite_array(physical["body_link_quat_w"], (22,4), "actual body quaternion")
                _require(np.max(np.abs(np.linalg.norm(quaternions,axis=1)-1.))<=1e-3, "actual body quaternions not normalized")
                foot_names = physical["foot_body_names"]
                _require(foot_names==["l_ankle_roll_link","r_ankle_roll_link"], "foot names/order mismatch")
                force = _finite_array(physical["foot_net_contact_forces_w_n"], (2,3), "actual foot forces")
                contact_names = physical["contact_body_names"]
                all_force = _finite_array(physical["net_contact_forces_w_n"], (len(contact_names),3), "all-body contact forces")
                _require(len(set(contact_names))==len(contact_names), "contact body names are duplicated")
                _require(np.array_equal(force,all_force[[contact_names.index(n) for n in foot_names]]), "foot force name mapping mismatch")
                clearance = _finite_array(physical["foot_min_support_clearance_m"], (2,), "actual support clearance")
                for i,name in enumerate(foot_names):
                    spheres = np.asarray(physical["foot_support_sphere_clearance_m"][name],dtype=float)
                    _require(spheres.shape==(8,) and np.isfinite(spheres).all(), "actual support sphere set invalid")
                    _require(clearance[i]==float(spheres.min()), "minimum support clearance disagrees with spheres")
                _require(physical["contact_is_ground_only"] is False, "net contact mislabelled as ground-only")
                # 先完整校验单帧，再整体加入统计，避免错误帧留下不同长度的数组。
                fields = {"joints":_finite_array(row["actual_joint_pos_gmt"], (21,), "actual joint positions"),
                          "root":_finite_array(row["actual_qpos"],(28,),"actual qpos")[:3],
                          "forces":force, "clearance":clearance,
                          "applied":_finite_array(physical["applied_joint_torque_nm"],(21,),"applied torque estimate"),
                          "computed":_finite_array(physical["computed_joint_torque_nm"],(21,),"computed torque estimate"),
                          "ticks":row["tick"]}
                values = phases[row["phase"]]
                for name,value in fields.items():
                    values[name].append(value)
                physical_count += 1
            except (AssertionError,KeyError,ValueError,TypeError) as error:
                physics_issues.append(f"trace {line}: {error}")
        if not wants_video:
            continue
        try:
            _require(isinstance(render,dict), "video lacks render_pose_sync evidence")
            _require(render["status"]=="passed", "render pose status is not passed")
            _require(render["control_tick"]==row["tick"], "render tick mismatch")
            _require(render.get("physics_steps_added")==0 and render.get("history_updates_added")==0, "render check changed simulation/history")
            names=render["body_names"]
            _require(len(names)==22 and len(set(names))==22, "render requires 22 unique body names")
            if render_body_names is None:
                render_body_names=names
            _require(names==render_body_names, "render body order changes between frames")
            phys=_finite_array(render["physics_body_pose_wxyz"],(22,7),"render check physical poses")
            usd=_finite_array(render["usd_body_pose_wxyz"],(22,7),"USD poses")
            position,orientation=_pose_errors(phys,usd)
            _require(position<=1e-4 and orientation<=1e-4, "independent 22-body render pose comparison failed")
            for name,value in (("max_position_error_m",position),("max_orientation_error_rad",orientation)):
                _require(math.isclose(float(render[name]),value,abs_tol=1e-10,rel_tol=1e-5), "reported render maximum differs from independent calculation")
            route=render["render_sync_state"]
            _require(route["route"]=="usd" and route["fabric_enabled"] is False and route["update_to_usd"] is True, "render evidence does not use verified USD sync route")
            _require(isinstance(physical,dict) and physical["control_tick"]==row["tick"], "render frame lacks same-tick physical diagnostics")
            _require(physical["body_names"]==names, "render/physical diagnostic names differ")
            captured=np.c_[physical["body_link_pos_w"],physical["body_link_quat_w"]]
            capture_position,capture_orientation=_pose_errors(_finite_array(captured,(22,7),"physical capture"),phys)
            _require(capture_position<=1e-6 and capture_orientation<=1e-6, "render evidence differs from same-step actual physical capture")
            camera=render["camera"]
            _require(camera["status"]=="available", "camera state unavailable")
            _finite_array(camera["world_transform"],(4,4),"actual camera transform")
            _require(all(math.isfinite(float(camera[k])) and float(camera[k])>0 for k in ("focal_length","horizontal_aperture","vertical_aperture")), "invalid camera optics")
            if first_camera is None:
                first_camera=camera
            _require(camera==first_camera, "camera extrinsics/intrinsics are not fixed")
            index=_integer(frame_index,"video frame index missing/noninteger")
            _require(index>=0 and (previous_index is None or index==previous_index+1), "video frame indexes skip/repeat")
            _require(isinstance(frame_hash,str) and re.fullmatch(r"[0-9a-f]{64}",frame_hash) is not None, "RGB frame SHA256 missing/invalid")
            if first_index is None:
                first_index=index
            previous_index=index
            repeated_hashes += int(previous_hash==frame_hash)
            previous_hash=frame_hash
            unique_hashes.add(frame_hash)
            maxima["position_m"]=max(maxima["position_m"],position)
            maxima["orientation_rad"]=max(maxima["orientation_rad"],orientation)
            maxima["physics_capture_position_m"]=max(maxima["physics_capture_position_m"],capture_position)
            maxima["physics_capture_orientation_rad"]=max(maxima["physics_capture_orientation_rad"],capture_orientation)
            render_count += 1
        except (AssertionError,KeyError,ValueError,TypeError) as error:
            visual_issues.append(f"trace {line}: {error}")
    if video_expected and (physics_issues or physical_count!=total):
        visual_issues.append("Video physical diagnostics are invalid or incomplete")
    if not video_expected:
        visual_status="not_recorded"
    elif visual_issues or render_count!=total or video_expected!=total:
        visual_status="failed" if render_count or render_body_names is not None else "incomplete"
    else:
        visual_status="passed"
    if not physical_count and not physics_issues:
        physics_status="not_recorded"
    elif physics_issues or physical_count!=total:
        physics_status="failed"
    else:
        physics_status="passed"
    phase_summary={name:_phase_physics_summary(values) for name,values in phases.items()}
    return {"visual_evidence": {"status":visual_status,"expected_frames":video_expected,"verified_frames":render_count,
                "first_frame_index":first_index,"last_frame_index":previous_index,
                "unique_rgb_frame_hashes":len(unique_hashes),"adjacent_repeated_rgb_hashes":repeated_hashes,
                "fixed_camera":first_camera if visual_status=="passed" else None,
                "max_position_error_m":maxima["position_m"] if render_count else None,
                "max_orientation_error_rad":maxima["orientation_rad"] if render_count else None,
                "max_same_tick_physical_capture_position_error_m":maxima["physics_capture_position_m"] if render_count else None,
                "max_same_tick_physical_capture_orientation_error_rad":maxima["physics_capture_orientation_rad"] if render_count else None,
                "issue_count":len(visual_issues),"issues":visual_issues[:20],
                "evidence_boundary":"22 same-tick PhysX/USD link poses plus fixed camera and captured RGB hashes; not independent landmark projection into decoded video"},
            "physical_diagnostics": {"status":physics_status,"verified_frames":physical_count,
                "phases":phase_summary,"issue_count":len(physics_issues),"issues":physics_issues[:20],
                "force_semantics":"Net contact against all colliders, not ground-only; final physics state, not four-substep average",
                "torque_semantics":"Implicit PD computed/applied torque estimates from the control calculation before the final physics substep; not measured reaction torque or proof of actual PhysX submitted/applied torque"}}


def audit_experiment(output_dir) -> dict[str, Any]:
    """返回审计结果而不写文件，供总入口完成 worker 退出后合并 acceptance。"""
    root = output_dir
    root = Path(root).resolve()
    report_path = root / "run_summary.json"
    if not report_path.exists():
        report_path = root / "report.json"
    report = _read_json(report_path)
    events = defaultdict(list)
    for _, row in _jsonl(root / "events.jsonl"):
        if row.get("episode_id") is not None:
            events[row["episode_id"]].append(row)
    results = []
    for episode in report["episodes"]:
        try:
            result = _audit_episode(root, episode, events[episode["episode_id"]])
            result.update(_audit_render_physics(root, episode))
            results.append(result)
        except (AssertionError, KeyError, ValueError, TypeError) as error:
            results.append({"episode_id": episode.get("episode_id"), "mode": episode.get("mode"),
                            "status": "failed", "error": str(error)})
    visual = [r["visual_evidence"] for r in results if "visual_evidence" in r and r["visual_evidence"]["status"] != "not_recorded"]
    visual_status = ("not_recorded" if not visual else "passed" if all(v["status"]=="passed" for v in visual) else "failed")
    return {"schema": "genmo.closedloop_stage8_audit.v1", "run": str(root),
            "visual_evidence": {"status":visual_status,"episodes_with_video":len(visual),
                "verified_frames":sum(v["verified_frames"] for v in visual),
                "note":"Protocol status is independent; absent render evidence never means visual validation passed"},
            "status": "passed" if results and all(r["status"] == "passed" for r in results) else "failed",
            "episodes": results,
            "evidence_boundary": {
                "protected_arrays": "Runtime commit modification counts and independently recomputed prefix errors; not independent full six-array replay",
                "proprio_history": "Null advance event counters are not backfilled; strict initial/final counts and trace timing are checked. Combine with ActualProprioHistory strict per-step append checks and independent backend repeated-observe validation; no per-advance proprio counter claim",
                "timing": "Simulation 600 Hz clock and measured-latency arrival schedule; no claim of wall-time deployment throughput"}}


# 保留独立测试／早期调用方的同义入口。
audit_run = audit_experiment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = audit_experiment(args.run_dir)
    output = args.output or args.run_dir / "audit.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "episodes": len(result["episodes"]), "output": str(output)}, ensure_ascii=False))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
