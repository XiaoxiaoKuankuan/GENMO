"""Stage8 多音乐离线报告的轻量合成日志测试，不运行模型、Isaac 或 GPU。

临时夹具提供小型JSONL、验收清单和占位视频文件，只验证报告对于独立音乐身份、
40格模式矩阵、启动/音乐分段、空节拍、精确控制步分位数、逐曲展示原指标语义、
相对视频与JSON路径和不覆盖
的处理。新增200首train单latency无视频矩阵、按实际音乐长度计算的完成率与Wilson
区间、启动/控制/基础设施故障分类、独立组与节拍缺测、50Hz差分支持点、关节限位、
分离yaw阈值首次越界和中断后落盘恢复测试。占位视频不会被当作真实渲染证明；真实视频解码由baseline_video负责。
全部产物限定pytest临时目录，既有实验数据不修改。
"""
import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.eval.report_stage8_music_sweep import (
    build_music_sweep_report, render_markdown, render_video_index, write_music_sweep_report,
)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def make_run(root, songs=1, *, music_steps=60, startup_failure=None, modes=("paused", "latency")):
    root.mkdir(exist_ok=True)
    episodes, audit_episodes, videos, events = [], [], [], []
    frame_index = 0
    for song in range(songs):
        sha = hashlib.sha256(f"song {song}".encode()).hexdigest()
        for mode in modes:
            episode_id = f"{song}:{mode}"
            startup = episode_id == startup_failure
            step_count = music_steps[song] if isinstance(music_steps, list) else music_steps
            warmup_count, music_count = (7, 0) if startup else (50, step_count)
            directory = root / "episodes" / episode_id.replace(":", "_")
            directory.mkdir(parents=True)
            qpos = [0., 0., .48, 1., 0., 0., 0.]+[0.]*21
            write_json(directory / "initial.json", {"snapshot": {"episode_id": episode_id, "tick": 0, "robot_qpos": qpos,
                       "reset_ground_pose_actual": {"status":"passed", "minimum_support_clearance_m":.001,
                           "provenance_sha256":"d"*64,"physics_steps_added":0,"history_updates_added":0,
                           "expected_clearance_m":.001,"tolerance_m":5e-5}}})
            rows = []
            first_frame = frame_index
            for step in range(warmup_count+music_count):
                phase = "warmup" if step < warmup_count else "music"
                physical = {"control_tick": (step+1)*12, "foot_body_names": ["l_ankle_roll_link", "r_ankle_roll_link"],
                            "foot_net_contact_forces_w_n": [[0., 0., 100.+step], [0., 0., 90.+step]],
                            "foot_min_support_clearance_m": [.001-step*.000001, .002]}
                # 前warmup幅度很大，便于检测是否误混入music主误差。
                error = 10000. if phase == "warmup" else float(step-warmup_count)
                rows.append({"episode_id": episode_id, "tick": (step+1)*12, "control_tick_begin": step*12,
                    "actual_qpos": qpos, "actual_joint_vel_gmt": [float(step)]*21, "phase": phase,
                    "physical_diagnostics": physical, "errors": {"root_height_error_m": error},
                    "gmt_inference_seconds": .001, "physics_seconds": .01, "video_capture_seconds": .1,
                    "step_seconds": .111, "video_frame_index": frame_index})
                frame_index += 1
            trace = directory / "trace.jsonl"
            trace.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            metric = {"status": "available", "motion_beat_count": 3, "mean_beat_distance_seconds": .04,
                      "alignment": float(np.exp(-.4))}
            music = {"status": "available", "music_beat_count": 2, "actual": copy.deepcopy(metric), "reference": copy.deepcopy(metric)}
            if startup:
                music = {"status": "insufficient_duration", "actual": None, "reference": None}
            ep = {"episode_id": episode_id, "mode": mode, "seed": 42, "dataset": "Mine",
                "sample": {"dataset": "Mine", "group_id": f"group{song}", "manifest_sha256": "a"*64,
                    "row": {"sample_id": f"歌曲{song}<test>", "split": "val", "source_audio_sha256": sha,
                            "source_music_feature_sha256": "b"*64}},
                "artifacts": {"trace": str(trace.relative_to(root))},
                "terminal_snapshot": {"episode_id": episode_id, "tick": (warmup_count+music_count)*12},
                "recorded_music_control_steps": music_count, "recorded_control_steps": len(rows),
                "reason": "startup_failure" if startup else "duration_limit", "backend_reason": "global_anchor_ori" if startup else None,
                "failed": startup, "startup_failure": startup,
                "music_duration_seconds": music_count/50, "warmup_duration_seconds": warmup_count/50,
                "total_duration_seconds": len(rows)/50, "replan_count": 0 if startup else 2,
                "music": music, "protected_reference": {"modification_count": 0,
                    "recomputed_position_error": {"count": 0 if startup else 2, "max": None if startup else 0.},
                    "recomputed_velocity_error": {"count": 0 if startup else 2, "max": None if startup else 0.}},
                "reference_buffer": {"reference_underrun": False}, "trace_integrity": {"complete": True}}
            episodes.append(ep)
            audit_episodes.append({"episode_id": episode_id, "status": "passed", "visual_evidence": {"status": "passed"}})
            video = directory / "music.mp4"
            video.write_bytes(b"fixture only; no rendered-video claim")
            sync = {"episode_id": episode_id, "status": "passed", "first_raw_frame": first_frame,
                    "end_raw_frame_exclusive": frame_index, "frame_count": len(rows), "warmup_frames": warmup_count,
                    "music_frames": music_count, "fps": 50, "audio_sha256": sha,
                    "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(), "audio_delay_seconds": warmup_count/50}
            write_json(video.with_suffix(".sync.json"), sync)
            videos.append({"episode_id": episode_id, "path": str(video.relative_to(root))})
            if not startup:
                events.append({"event": "plan_prepared", "episode_id": episode_id, "prefix_frames": 12,
                               "end_to_end_seconds": .02, "actor_seconds": .01, "prepare_seconds": .005})
    summary = {"episodes": episodes, "videos": videos}
    write_json(root / "run_summary.json", summary)
    write_json(root / "audit.json", {"status": "passed", "episodes": audit_episodes})
    write_json(root / "gmt_identity.json", {"policy_sha256": "c"*64,
               "initial_ground_pose": {"clearance_m": .001,"sha256":"d"*64,"root_z_m":.48,"original_root_z_m":.65}})
    (root / "events.jsonl").write_text(''.join(json.dumps(e)+'\n' for e in events))
    return summary


def test_complete_twenty_song_forty_episode_matrix(tmp_path):
    make_run(tmp_path, songs=20, music_steps=3)
    report = build_music_sweep_report(tmp_path)
    assert report["coverage"]["complete"]
    assert report["coverage"]["unique_audio_sha256_count"] == 20
    assert report["coverage"]["evaluation_episode_count"] == 40
    assert report["evidence"]["collection_complete"]
    assert len(report["episodes"]) == 40
    assert report["modes"]["paused"]["errors_pooled_music_control_steps"]["root_height_error_m"]["count"] == 60
    assert report["gmt_identity"]["initial_ground_pose"]["clearance_m"] == .001
    assert report["evidence"]["grounded_initialization"]["verified_resets"] == 40


def test_phase_windows_pooling_video_mapping_and_safe_html(tmp_path):
    make_run(tmp_path)
    report = build_music_sweep_report(tmp_path, expected_music=1)
    assert report["evidence"]["collection_complete"]
    episode = report["episodes"][0]
    diag = episode["startup_diagnostics"]
    assert diag["warmup_first_5"]["control_steps"] == 5
    assert diag["warmup_last_10"]["first_tick"] == 41*12
    assert diag["music_first_1_second"]["control_steps"] == 50
    assert diag["music_first_1_second"]["first_tick"] == 51*12
    assert diag["warmup_last_10"]["foot_net_force_z_n"][0]["mean"] == 144.5
    assert diag["warmup_first_5"]["foot_min_support_clearance_m"][0]["min"] == pytest.approx(.000996)
    errors = report["modes"]["paused"]["errors_pooled_music_control_steps"]["root_height_error_m"]
    assert errors["p95"] == pytest.approx(np.percentile(np.arange(60),95))
    assert errors["max"] == 59
    assert report["modes"]["paused"]["beat"]["actual"]["music_beat_weighted_distance_seconds"] == .04
    markup = render_video_index(report)
    assert 'preload="none"' in markup and '&lt;test&gt;' in markup
    assert '<test>' not in markup and str(tmp_path) not in markup
    markdown = render_markdown(report)
    assert 'warmup前5' in markdown and 'music首1s' in markdown
    assert 'implicit PD' in markdown


def test_startup_failures_are_separate_and_no_beats_remain_missing(tmp_path):
    make_run(tmp_path, startup_failure="0:paused")
    report = build_music_sweep_report(tmp_path, expected_music=1)
    mode = report["modes"]["paused"]
    assert mode["startup_failures"] == 1 and mode["music_failures"] == 0
    assert mode["music_failure_fraction_of_started"] is None
    assert mode["music_failures_per_minute"] is None
    assert mode["startup_backend_reason_counts"] == {"global_anchor_ori": 1}
    assert mode["protected_reference"]["position_recompute_max"] is None
    beat = mode["beat"]["actual"]
    assert beat["available_episodes"] == 0
    assert beat["music_beat_weighted_distance_seconds"] is None
    assert beat["unavailable_reason_counts"] == {"insufficient_duration": 1}
    assert report["evidence"]["collection_complete"]  # 完整记录失败仍是完整采集，不代表跟踪成功。


def test_pooled_percentiles_and_beat_weights_not_mean_episode_statistics(tmp_path):
    summary = make_run(tmp_path, songs=2)
    ep = summary["episodes"][2]
    trace = tmp_path / ep["artifacts"]["trace"]
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    for row in rows:
        if row["phase"] == "music":row["errors"]["root_height_error_m"] = 100.
    trace.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    manifest = (tmp_path / summary["videos"][2]["path"]).with_suffix(".sync.json")
    sync = json.loads(manifest.read_text())
    sync["trace_sha256"] = hashlib.sha256(trace.read_bytes()).hexdigest()
    write_json(manifest, sync)
    ep["music"]["music_beat_count"] = 8
    ep["music"]["actual"]["mean_beat_distance_seconds"] = .10
    ep["music"]["actual"]["alignment"] = float(np.exp(-1.))
    write_json(tmp_path / "run_summary.json", summary)
    result = build_music_sweep_report(tmp_path, expected_music=2)
    assert result["evidence"]["collection_complete"]
    metrics = result["modes"]["paused"]
    assert metrics["errors_pooled_music_control_steps"]["root_height_error_m"]["p95"] == 100.
    assert metrics["beat"]["actual"]["music_beat_weighted_distance_seconds"] == pytest.approx(.088)


@pytest.mark.parametrize("damage", ["duplicate_audio", "wrong_mode", "seed", "audio_sync", "missing_video", "old_audit"])
def test_malformed_matrix_or_video_evidence_not_marked_complete(tmp_path, damage):
    summary = make_run(tmp_path, songs=2, music_steps=2)
    if damage == "duplicate_audio":
        sha = summary["episodes"][0]["sample"]["row"]["source_audio_sha256"]
        for ep in summary["episodes"][2:]:ep["sample"]["row"]["source_audio_sha256"] = sha
    elif damage == "wrong_mode":summary["episodes"][1]["mode"] = "paused"
    elif damage == "seed":summary["episodes"][1]["seed"] = 43
    elif damage == "audio_sync":
        path = tmp_path / summary["videos"][0]["path"]
        sync = json.loads(path.with_suffix(".sync.json").read_text())
        sync["audio_delay_seconds"] = 0
        write_json(path.with_suffix(".sync.json"), sync)
    elif damage == "missing_video":(tmp_path / summary["videos"][0]["path"]).unlink()
    elif damage == "old_audit":
        audit = json.loads((tmp_path / "audit.json").read_text())
        for ep in audit["episodes"]:ep.pop("visual_evidence")
        write_json(tmp_path / "audit.json", audit)
    write_json(tmp_path / "run_summary.json", summary)
    result = build_music_sweep_report(tmp_path, expected_music=2)
    assert not result["evidence"]["collection_complete"]
    if damage == "old_audit":
        assert result["evidence"]["protocol_all_passed"]
        assert not result["evidence"]["visual_all_passed"]


def test_write_is_exclusive_and_old_single_video_shape_supported(tmp_path):
    summary = make_run(tmp_path)
    # 单视频字段仍能读取；缺少另一个episode视频必须按缺证据报告。
    summary["video"] = summary.pop("videos")[0]
    write_json(tmp_path / "run_summary.json", summary)
    result, outputs = write_music_sweep_report(tmp_path, expected_music=1)
    assert result["episodes"][0]["video"]["file_exists"]
    assert result["episodes"][1]["video"] is None
    assert all(Path(path).is_file() for path in outputs.values())
    original = (tmp_path / "music_sweep_report.json").read_bytes()
    with pytest.raises(FileExistsError):write_music_sweep_report(tmp_path, expected_music=1)
    assert (tmp_path / "music_sweep_report.json").read_bytes() == original


def test_missing_nonfinite_and_outside_artifacts_rejected(tmp_path):
    summary = make_run(tmp_path)
    summary["videos"][0]["path"] = "../../outside.mp4"
    write_json(tmp_path / "run_summary.json", summary)
    with pytest.raises(ValueError, match="escapes experiment"):
        build_music_sweep_report(tmp_path, expected_music=1)


def test_initial_one_mm_proof_requires_actual_reset_and_no_hidden_step(tmp_path):
    summary = make_run(tmp_path)
    initial = tmp_path / summary["episodes"][0]["artifacts"]["trace"]
    initial = initial.parent / "initial.json"
    payload = json.loads(initial.read_text())
    payload["snapshot"]["reset_ground_pose_actual"]["physics_steps_added"] = 1
    write_json(initial, payload)
    report = build_music_sweep_report(tmp_path, expected_music=1)
    assert not report["evidence"]["collection_complete"]
    assert report["evidence"]["grounded_initialization"]["status"] == "failed"
    assert report["evidence"]["grounded_initialization"]["verified_resets"] == 1


def test_per_music_quality_display_uses_existing_metrics_and_relative_json(tmp_path):
    summary = make_run(tmp_path)
    detail = (tmp_path / summary["episodes"][0]["artifacts"]["trace"]).with_name("summary.json")
    write_json(detail, summary["episodes"][0])
    report = build_music_sweep_report(tmp_path, expected_music=1)
    ep = report["episodes"][0]
    ep["motion"] = {"actual":{"joint_amplitude_p95_minus_p05_rad": [.4]*21},
                    "reference":{"joint_amplitude_p95_minus_p05_rad": [.8]*21}}
    ep["errors"] = {"joint_position_rmse_rad":{"p95":.1234},
                    "root_position_error_m":{"p95":.5678},
                    "end_effector_relative_height_error_m":{"p95":.0456}}
    ep["music"]["actual"]["mean_beat_distance_seconds"] = .0678
    ep["music"]["reference"]["mean_beat_distance_seconds"] = .0891
    ep["original_threshold_diagnostics"] = {"metrics":{"global_orientation_error_rad":{
        "threshold":.6,"music_crossing_fraction":.22266666666666668}}}
    markdown, webpage = render_markdown(report), render_video_index(report)
    for text in (markdown, webpage):
        for expected in ("0.4000", "0.8000", "0.1234", "0.5678", "0.0456", "0.0678", "0.0891", "22.27%"):
            assert expected in text
        assert "三维末端距离" in text
        assert "episodes/0_paused/summary.json" in text
    assert "逐曲幅度与跟踪诊断" in markdown
    assert "详细 JSON：0:paused" in webpage
    assert 'href="episodes/0_paused/summary.json"' in webpage
    # 有名为姿态越界的字段仍须确认它确实采用原0.6rad阈值，不能偷换1.2rad。
    ep["original_threshold_diagnostics"]["metrics"]["global_orientation_error_rad"]["threshold"] = 1.2
    assert "22.27%" not in render_markdown(report)
    assert "22.27%" not in render_video_index(report)


def test_per_music_missing_beat_is_not_displayed_as_zero_distance(tmp_path):
    make_run(tmp_path)
    report = build_music_sweep_report(tmp_path, expected_music=1)
    ep = report["episodes"][0]
    ep["music"]["actual"] = {"status":"no_motion_beats", "mean_beat_distance_seconds":0.}
    ep["music"]["reference"]["mean_beat_distance_seconds"] = .0891
    assert "未记录 / 0.0891" in render_markdown(report)
    assert "未记录 / 0.0891" in render_video_index(report)


def train_no_video(root, *, songs=1, music_steps=5, frames=None, startup_failure=None):
    summary = make_run(root, songs=songs, music_steps=music_steps, modes=("latency",), startup_failure=startup_failure)
    for i, ep in enumerate(summary["episodes"]):
        ep["sample"]["row"].update(split="train", num_frames=frames[i] if isinstance(frames, list) else (frames or 3))
        ep["sample"]["dataset"] = "Mine" if i % 2 == 0 else "AIST++"
        ep["music"] = {"status": "no_music_beats", "actual": None, "reference": None}
        path = root / ep["artifacts"]["trace"]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row.pop("video_frame_index")
            row.pop("video_capture_seconds")
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    summary.pop("videos")
    summary["selection"] = [e["sample"] for e in summary["episodes"]]
    summary["termination"] = {"orientation_mode": "separated_yaw", "root_height_error_m": .2,
        "non_yaw_orientation_error_rad": .6, "yaw_error_rad": 1.5, "end_effector_relative_height_error_m": .15}
    write_json(root / "run_summary.json", summary)
    audit = json.loads((root/"audit.json").read_text())
    for ep in audit["episodes"]:
        ep["visual_evidence"] = {"status": "not_recorded"}
    write_json(root/"audit.json", audit)
    return summary


def train_report(root, count):
    return build_music_sweep_report(root, expected_music=count, expected_modes=("latency",), expected_split="train", require_video=False)


def test_two_hundred_train_groups_single_latency_without_video_or_audio(tmp_path):
    summary = train_no_video(tmp_path, songs=200)
    # 无源音频仍可按独立音乐组核验；缺音频不会被误报成视频不完整。
    for ep in summary["episodes"]:
        ep["sample"]["row"].pop("source_audio_sha256")
    write_json(tmp_path/"run_summary.json", summary)
    result, outputs = write_music_sweep_report(tmp_path, expected_music=200, expected_modes=("latency",), expected_split="train", require_video=False)
    assert result["coverage"]["complete"]
    assert result["coverage"]["unique_music_group_count"] == 200
    assert result["coverage"]["unique_audio_sha256_count"] == 0
    assert result["evidence"]["collection_complete"]
    assert result["evidence"]["visual_all_passed"] is None
    assert result["evidence"]["video_index"]["status"] == "not_requested"
    completion = result["modes"]["latency"]["completion"]
    assert completion["normal_completion"]["successes"] == 200
    assert completion["normal_completion"]["fraction"] == 1.
    assert completion["normal_completion"]["ci95_low"] == pytest.approx(.9811547, abs=1e-6)
    assert completion["reached_30s_among_eligible"]["total"] == 0
    assert completion["total_execution_coverage_fraction"] == pytest.approx(1.)
    beat = result["modes"]["latency"]["beat"]["actual"]
    assert beat["available_episodes"] == 0 and beat["music_beat_weighted_distance_seconds"] is None
    assert beat["unavailable_reason_counts"] == {"no_music_beats": 200}
    assert len(list(csv.DictReader(Path(outputs["csv"]).open()))) == 200
    webpage = Path(outputs["html"]).read_text()
    assert "<video" not in webpage and "纯数据评估" in webpage
    assert "200 首音乐" in Path(outputs["markdown"]).read_text()


def test_completion_denominators_short_music_startup_control_and_infra_are_distinct(tmp_path):
    summary = train_no_video(tmp_path, songs=5, music_steps=[1500, 500, 0, 350, 100],
        frames=[1200, 300, 1200, 1200, 1200], startup_failure="2:latency")
    for ep, reason, failed in zip(summary["episodes"], ["duration_limit", "music_end", "startup_failure", "global_anchor_yaw", "infrastructure_error"], [False, False, True, True, True]):
        ep.update(reason=reason, failed=failed)
    write_json(tmp_path/"run_summary.json", summary)
    report = train_report(tmp_path, 5)
    completion = report["modes"]["latency"]["completion"]
    assert completion["normal_completion"]["successes"] == 2
    assert completion["normal_completion"]["total"] == 5
    assert completion["reached_30s_among_eligible"]["successes"] == 1
    assert completion["reached_30s_among_eligible"]["total"] == 4
    assert completion["ending_category_counts"] == {"duration_limit": 1, "natural_music_end": 1, "startup_failure": 1, "control_failure": 1, "infrastructure_error": 1}
    assert completion["requested_music_seconds_for_attempted"] == 130.
    assert completion["executed_music_seconds"] == 49.
    assert completion["total_execution_coverage_fraction"] == pytest.approx(49/130)
    assert completion["failure_music_seconds"]["mean"] == 3.5
    assert report["modes"]["latency"]["by_dataset"]["Mine"]["completion"]["attempted_episodes"] == 3
    assert report["episodes"][3]["completion"]["failure_music_seconds"] == 7.


def test_smoothness_has_supported_points_and_threshold_strict_comparison(tmp_path):
    summary = train_no_video(tmp_path)
    ep = summary["episodes"][0]
    path = tmp_path/ep["artifacts"]["trace"]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for i, row in enumerate(rows):
        row["actual_joint_pos_gmt"] = [0.]*21
        row["actual_joint_vel_gmt"] = [float(max(0, i-50)**2)]*21
        row["reference"] = {"joint_pos": [0.]*21, "joint_vel": [0.]*21, "body_pos_w": [[0., 0., .48]]}
        row["reference_plan_id"] = "plan1"
        row["errors"] = {"root_height_error_m": .2, "non_yaw_orientation_error_rad": .6,
                         "yaw_error_rad": 1.5 if i <= 51 else 1.51, "global_orientation_error_rad": 1.6}
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    result = train_report(tmp_path, 1)
    extended = result["episodes"][0]["startup_diagnostics"]["extended"]
    assert extended["threshold_first_crossing"]["yaw_error_rad"]["music_seconds"] == .06
    assert "root_height_error_m" not in extended["threshold_first_crossing"]
    assert "non_yaw_orientation_error_rad" not in extended["threshold_first_crossing"]
    assert "global_orientation_error_rad" not in extended["threshold_crossing_counts"]
    assert extended["threshold_crossing_counts"]["yaw_error_rad"] == 3
    motion = result["modes"]["latency"]["motion_pooled_music_control_steps"]
    assert motion["actual_joint_speed_rms_rad_s"]["count"] == 5
    assert motion["actual_joint_acceleration_rms_rad_s2"]["count"] == 4
    assert motion["actual_joint_jerk_rms_rad_s3"]["count"] == 3
    assert motion["actual_joint_jerk_rms_rad_s3"]["mean"] == pytest.approx(5000.)
    assert result["modes"]["latency"]["physical_pooled_music_control_steps"]["both_feet_contact_above_1n"]["mean"] == 1.


def test_interrupted_run_recovers_finished_and_partial_disk_evidence(tmp_path):
    summary = train_no_video(tmp_path, songs=2)
    first, second = summary["episodes"]
    write_json((tmp_path/first["artifacts"]["trace"]).with_name("summary.json"), first)
    second_initial = (tmp_path/second["artifacts"]["trace"]).with_name("initial.json")
    initial = json.loads(second_initial.read_text())
    initial.update(sample=second["sample"], mode="latency", seed=42)
    write_json(second_initial, initial)
    summary["episodes"] = []
    summary["exit_code"] = 1
    write_json(tmp_path/"run_summary.json", summary)
    write_json(tmp_path/"failure.json", {"type": "TimeoutError", "message": "fixture worker timeout"})
    report = train_report(tmp_path, 3)
    assert report["coverage"]["attempted_episodes"] == 2
    assert report["coverage"]["not_attempted_episodes"] == 1
    assert not report["evidence"]["collection_complete"]
    assert report["episodes"][1]["completion"]["category"] == "infrastructure_error"
    assert report["modes"]["latency"]["completion"]["normal_completion_fraction_of_requested"] == pytest.approx(1/3)
    assert report["run_infrastructure_error"]["type"] == "TimeoutError"


def test_runtime_limits_and_contact_torque_diagnostics_have_correct_order_and_units(tmp_path):
    summary = train_no_video(tmp_path)
    identity = json.loads((tmp_path/"gmt_identity.json").read_text())
    names = [f"j{i}" for i in range(21)]
    identity["runtime_fingerprint"] = {"parameters": {"joint_names": names,
        "joint_pos_limits": [[[-1., 1.]]*21], "joint_vel_limits": [[2.]*21], "joint_effort_limits": [[4.]*21]}}
    write_json(tmp_path/"gmt_identity.json", identity)
    path = tmp_path/summary["episodes"][0]["artifacts"]["trace"]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row["actual_joint_pos_gmt"] = [1.1]+[0.]*20
        row["actual_joint_vel_gmt"] = [3.]*21
        row["physical_diagnostics"].update(joint_names=names, computed_joint_torque_nm=[8.]*21,
            applied_joint_torque_nm=[4.]*21, contact_body_names=["base_link", "l_ankle_roll_link", "r_ankle_roll_link"],
            net_contact_forces_w_n=[[0., 0., 2.], [0., 0., 100.], [0., 0., 100.]])
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    result = train_report(tmp_path, 1)
    physical = result["modes"]["latency"]["physical_pooled_music_control_steps"]
    assert physical["joint_position_limit_exceeded"]["mean"] == 1.
    assert physical["joint_position_limit_violation_max_rad"]["max"] == pytest.approx(.1)
    assert physical["joint_speed_limit_ratio_max"]["max"] == 1.5
    assert physical["computed_torque_estimate_limit_ratio_max"]["max"] == 2.
    assert physical["applied_torque_estimate_limit_ratio_max"]["max"] == 1.
    assert physical["nonfoot_contact_above_1n"]["mean"] == 1.
    rows[-1]["physical_diagnostics"]["joint_names"] = list(reversed(names))
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(ValueError, match="names/order differ"):
        train_report(tmp_path, 1)
