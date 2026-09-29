"""Stage8 记录器的有界 CPU 测试，所有输出限定在 pytest 临时目录。

覆盖 warmup 排除、原阈值诊断、实际步数、计划二进制证据、Unicode 来源、
30/50 Hz 音乐节拍时间对齐、缺测 null、覆盖拒绝及静态图。输入是明确测试夹具，
不伪造真实闭环验收；本文件不加载策略、不启动仿真，也不读取动作 target。
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from gem.closedloop.baseline_metrics import BaselineRecorder, distribution, json_value


ERROR_KEYS = ("root_height_error_m", "global_orientation_error_rad", "end_effector_relative_height_error_m",
              "joint_position_rmse_rad", "joint_velocity_rmse_rad_s", "root_position_error_m", "body_position_rmse_m")


def snapshot(tick=0, **kwargs):
    return {"episode_id": "episode:1", "env_id": 0, "tick": tick, "done": False, "reason": None, **kwargs}


def sample(**kwargs):
    return {"dataset": "Mine", "row": {"sample_id": "测试音乐", "motion_path": "do-not-open.pt"}, **kwargs}


def row(tick, *, error=.1, phase="music", speed=1., plan="旧计划"):
    joint = np.full(21, tick/600)
    velocity = np.full(21, speed)
    qpos = np.r_[tick/600, 0., .8, 1., 0., 0., 0., joint[::-1]]
    body = np.zeros((22, 3)); body[:, 0] = tick/600; body[:, 2] = .8
    quaternion = np.zeros((22, 4)); quaternion[:, 0] = 1.
    return {"episode_id": "episode:1", "env_id": 0, "tick": tick, "phase": phase,
            "actual_qpos": qpos, "actual_joint_vel": velocity[::-1],
            "actual_joint_pos_gmt": joint, "actual_joint_vel_gmt": velocity,
            "reference": {"joint_pos": joint, "joint_vel": velocity, "body_pos_w": body,
                          "body_quat_w": quaternion, "body_lin_vel_w": np.ones((22, 3)),
                          "body_ang_vel_w": np.zeros((22, 3))},
            "reference_plan_id": plan, "consumed_reference_ticks": np.arange(tick-132, tick+109, 12),
            "consumed_plan_ids": np.array([plan]*21), "errors": {key: error for key in ERROR_KEYS},
            "applied_action": np.zeros(21), "gmt_inference_seconds": .001,
            "physics_seconds": .003, "step_seconds": .004}


@pytest.fixture
def recorder(tmp_path):
    value = BaselineRecorder(tmp_path / "run")
    yield value
    value.close()


def test_music_metrics_exclude_warmup_and_keep_diagnostics(recorder):
    recorder.start_episode(sample(), 42, "paused", snapshot())
    for tick in range(12, 601, 12):
        recorder.step(row(tick, error=10., phase="warmup"))
    for tick in range(612, 673, 12):
        recorder.step(row(tick, error=.1))
    terminal = snapshot(672, original_threshold_first_crossing={"root_height_error_m": 12},
                        original_threshold_crossing_counts={"root_height_error_m": 50})
    summary = recorder.finish_episode(terminal, "duration_limit")
    assert summary["errors"]["root_height_error_m"]["mean"] == pytest.approx(.1)
    assert summary["duration_seconds"] == pytest.approx(.12)
    assert summary["warmup_duration_seconds"] == 1
    assert summary["executed_control_steps"] == 56 and summary["executed_physics_steps"] == 224
    diagnostic = summary["original_threshold_diagnostics"]["metrics"]["root_height_error_m"]
    assert diagnostic["first_crossing_tick"] == 12 and diagnostic["all_phase_crossing_count"] == 50
    assert diagnostic["music_crossing_fraction"] == 0
    assert summary["music"]["status"] == "insufficient_duration"
    assert summary["music"]["actual"] is None
    report = recorder.summarize()
    assert report["modes"]["paused"]["failure_fraction"] == 0
    assert report["modes"]["paused"]["errors"]["root_height_error_m"]["count"] == 6
    assert recorder.summarize() == report
    assert (recorder.output_dir / "report.json").exists()
    assert len(summary["artifacts"]["plots"]["paths"]) == 1


def test_startup_failure_not_hidden_and_empty_metrics_are_null(recorder):
    recorder.start_episode(sample(), 42, "latency", snapshot())
    recorder.step(row(12, phase="warmup", error=2))
    summary = recorder.finish_episode(snapshot(12, done=True, reason="anchor_ori"), "startup_failure")
    assert summary["startup_failure"] and summary["failed"]
    assert summary["backend_reason"] == "anchor_ori"
    assert summary["duration_seconds"] == 0
    assert summary["motion"]["actual"] is None
    assert summary["latency_seconds"]["gmt_inference_seconds"]["mean"] is None
    report = recorder.summarize()
    assert report["modes"]["latency"]["startup_failures"] == 1
    assert report["modes"]["latency"]["failures_per_music_minute"] is None


def test_duplicate_steps_cross_episode_and_nonfinite_data_rejected(recorder):
    recorder.start_episode(sample(), 42, "paused", snapshot(600))
    recorder.step(row(612))
    with pytest.raises(ValueError, match="Duplicate"):
        recorder.step(row(612))
    bad = row(624); bad["episode_id"] = "reset-episode"
    with pytest.raises(ValueError, match="episode"):
        recorder.step(bad)
    bad = row(624); bad["actual_qpos"][0] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        recorder.step(bad)
    recorder.finish_episode(snapshot(612), "duration_limit")
    trace = next((recorder.output_dir / "episodes").glob("*/trace.jsonl"))
    assert len(trace.read_text().splitlines()) == 1


def test_plan_npz_metadata_events_and_boundary_statistics(recorder):
    recorder.start_episode(sample(), 43, "paused", snapshot(600))
    generated = {"episode_id": "episode:1", "plan_id": "计划:1", "prefix_frames": np.int64(20),
                 "qpos_world": np.zeros((120, 28)), "qpos30": np.zeros((120, 30)),
                 "contact": np.zeros((120, 2))}
    recorder.plan(generated)
    with pytest.raises(FileExistsError):
        recorder.plan(generated)
    recorder.event("plan_prepared", prefix_frames=20, request_tick=600,
                   end_to_end_seconds=.123, actor_seconds=.12, prepare_seconds=.002)
    recorder.event("plan_committed", acknowledgement={"protected_modification_count": 0,
                   "max_position_error": 1e-7, "max_velocity_error": 2e-6})
    recorder.event("decision_missed", reason="request_in_flight")
    recorder.event("plan_rejected", code="late_plan")
    recorder.step(row(612, plan="旧计划"))
    recorder.step(row(624, plan="计划:1", speed=2.))
    summary = recorder.finish_episode(snapshot(624), "duration_limit")
    assert summary["generated_plan_count"] == 1 and summary["replan_count"] == 1
    assert summary["prefix_over_18_fraction"] == 1
    assert summary["event_counts"]["decision_missed"] == 1
    assert summary["rejection_codes"] == {"late_plan": 1}
    assert summary["protected_reference"]["modification_count"] == 0
    assert summary["reference_boundaries"]["count"] == 1
    assert summary["motion"]["actual"]["joint_acceleration_abs_rad_s2"]["max"] == 50
    saved = next((recorder.output_dir / "episodes").glob("*/plans/*.npz"))
    with np.load(saved, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["qpos_world"], generated["qpos_world"])
    assert json.loads(saved.with_suffix(".json").read_text())["plan_id"] == "计划:1"
    assert len(summary["artifacts"]["plots"]["paths"]) == 2


def test_music_beat_mapping_uses_time_instead_of30hz_frame_number(recorder):
    music = np.zeros((90, 35), dtype=np.float32)
    music[[15, 45], 34] = 1  # 0.5 s、1.5 s 对应 50 Hz 的第25、75帧。
    recorder.start_episode(sample(music_features=music), 42, "paused", snapshot(600))
    for index in range(1, 101):
        speed = abs(index - 25) if index < 50 else abs(index - 75)
        recorder.step(row(600+12*index, speed=speed))
    summary = recorder.finish_episode(snapshot(1800), "music_end")
    assert summary["music"]["music_beat_count"] == 2
    for key in ("actual", "reference"):
        assert summary["music"][key]["alignment"] == pytest.approx(1.)
        assert summary["music"][key]["mean_beat_distance_seconds"] == 0


def test_no_beats_is_unavailable_not_perfect_or_zero_score(recorder):
    recorder.start_episode(sample(music_features=np.zeros((90, 35))), 42, "paused", snapshot(600))
    for index in range(1, 51):
        recorder.step(row(600+index*12, speed=1))
    summary = recorder.finish_episode(snapshot(1200), "music_end")
    assert summary["music"]["actual"]["alignment"] is None
    assert summary["music"]["actual"]["status"] == "no_music_beats"


def test_refuse_overwrite_existing_run(tmp_path):
    first = BaselineRecorder(tmp_path)
    first.close()
    with pytest.raises(FileExistsError):
        BaselineRecorder(tmp_path)


def test_json_missing_stats_and_nonfinite_are_explicit():
    assert json_value({"value": np.float64(2)}) == {"value": 2.}
    with pytest.raises(ValueError):
        json_value(np.array([np.inf]))
    assert json_value([np.nan], nonfinite="null") == [None]
    assert distribution([])["mean"] is None


def test_calibration_not_counted_as_eval_failure_or_latency(recorder):
    recorder.start_episode(sample(), 42, "calibration", snapshot())
    recorder.step(row(12, phase="warmup"))
    recorder.event("plan_prepared", prefix_frames=12, request_tick=12, end_to_end_seconds=100)
    recorder.finish_episode(snapshot(12), "calibration_complete")
    report = recorder.summarize()
    assert report["evaluation_episode_count"] == 0 and report["calibration_episode_count"] == 1
    assert report["modes"] == {}


def test_executed_but_unrecorded_terminal_step_is_exposure_and_system_error(recorder):
    recorder.start_episode(sample(), 42, "paused", snapshot(600))
    recorder.step(row(612))
    recorder.event("advance", phase="music", end_tick=624, reference_valid_end_tick=2376)
    summary = recorder.finish_episode(snapshot(624, done=True, reason="nonfinite", reference_valid_end_tick=2376), "nonfinite")
    assert summary["elapsed_music_seconds"] == .04
    assert summary["music_control_steps"] == 2 and summary["recorded_music_control_steps"] == 1
    assert summary["trace_integrity"]["system_error"] == "executed_steps_missing_trace"
    assert summary["trace_integrity"]["missing_control_step_records"] == 1
    assert summary["reference_buffer"]["remaining_seconds_at_advance_end"]["count"] == 1
    assert recorder.summarize()["modes"]["paused"]["incomplete_trace_episodes"] == 1
