"""冻结 Stage8 基线的逐步证据记录、分阶段统计和静态报告。

记录器只消费协调器收到的真实控制步、候选计划和事件，不读取动作 target，不推进
仿真或观测历史。主指标只统计 phase=music；启动阶段保留独立时长与失败归因。
每个 episode 保存逐控制步 JSONL、生成计划 NPZ/元数据、终止快照和静态图，根目录
保存事件流及总报告。文件均排他创建，拒绝覆盖已有同名证据；NumPy 转换显式处理，
原始观测/计划里的 NaN/Inf 直接拒绝，无数据的统计值使用 JSON null，不以零冒充。

音乐指标复用 BUMI 的 _derive_motion_beats 和 _beat_alignment，按 50 Hz 实际时刻
映射原 EDGE35 第 35 列节拍。无音乐、少于 1 秒、无音乐节拍或无运动节拍时返回 null。
历史阈值的越界次数只作诊断，绝不称为旧阈值下的失败率。参考来源切换处的一步变化
与同一时刻保护区重算误差分别报告，避免把自然运动速度当成重规划跳变。
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

CLOCK_HZ = 600
CONTROL_TICKS = 12
FPS = 50
NORMAL_ENDINGS = {"duration_limit", "music_end", "calibration_complete"}
ORIGINAL_THRESHOLDS = {
    "root_height_error_m": .2, "global_orientation_error_rad": .6,
    "end_effector_relative_height_error_m": .15,
}


def json_value(value: Any, *, nonfinite="reject"):
    """显式转换 NumPy/Path；观测与证据默认拒绝 NaN，派生缺测指标可选 null。"""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return json_value(value.tolist(), nonfinite=nonfinite)
    if isinstance(value, np.generic):
        return json_value(value.item(), nonfinite=nonfinite)
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Report mapping keys must be strings")
        return {key: json_value(item, nonfinite=nonfinite) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item, nonfinite=nonfinite) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        if nonfinite == "null":
            return None
        raise ValueError("Non-finite value in baseline evidence")
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported report type: {type(value).__name__}")


def distribution(values):
    """不将空集合填零；标明每个分布实际支持的样本数量。"""
    array = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(array).all():
        raise ValueError("Non-finite metric samples")
    if not len(array):
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    return {"count": int(len(array)), "mean": float(array.mean()),
            "p50": float(np.percentile(array, 50)), "p95": float(np.percentile(array, 95)),
            "p99": float(np.percentile(array, 99)), "max": float(array.max())}


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(json_value(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _stack(rows, key, nested=None):
    if not rows:
        return None
    values = [row[key] if nested is None else row[nested][key] for row in rows]
    result = np.asarray(values, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"Non-finite {key}")
    return result


def _motion_statistics(rows):
    if not rows:
        return {"actual": None, "reference": None}
    result = {}
    for kind, joints, velocity, roots in (
        ("actual", _stack(rows, "actual_joint_pos_gmt"), _stack(rows, "actual_joint_vel_gmt"),
         _stack(rows, "actual_qpos")[:, :3]),
        ("reference", _stack(rows, "joint_pos", "reference"), _stack(rows, "joint_vel", "reference"),
         _stack(rows, "body_pos_w", "reference")[:, 0, :]),
    ):
        acceleration = np.diff(velocity, axis=0) * FPS
        result[kind] = {
            "joint_order": "GMT Isaac contract order",
            "joint_amplitude_p95_minus_p05_rad": (np.percentile(joints, 95, axis=0) - np.percentile(joints, 5, axis=0)).tolist(),
            "root_xyz_range_m": np.ptp(roots, axis=0).tolist(),
            "joint_speed_abs_rad_s": distribution(np.abs(velocity)),
            "joint_acceleration_abs_rad_s2": distribution(np.abs(acceleration)),
            "root_step_displacement_m": distribution(np.linalg.norm(np.diff(roots, axis=0), axis=-1)),
        }
    result["actual"]["native_joint_amplitude_p95_minus_p05_rad"] = (
        np.percentile(_stack(rows, "actual_qpos")[:, 7:], 95, axis=0)
        - np.percentile(_stack(rows, "actual_qpos")[:, 7:], 5, axis=0)).tolist()
    result["actual"]["native_joint_order"] = "GENMO source_joint_names from run manifest"
    return result


def _boundary_statistics(rows):
    values = defaultdict(list)
    for old, new in zip(rows, rows[1:]):
        if old["reference_plan_id"] == new["reference_plan_id"]:
            continue
        a, b = old["reference"], new["reference"]
        values["joint_position_step_max_rad"].append(float(np.max(np.abs(np.asarray(b["joint_pos"]) - a["joint_pos"]))))
        values["joint_velocity_step_max_rad_s"].append(float(np.max(np.abs(np.asarray(b["joint_vel"]) - a["joint_vel"]))))
        values["root_position_step_m"].append(float(np.linalg.norm(np.asarray(b["body_pos_w"])[0] - np.asarray(a["body_pos_w"])[0])))
        values["body_linear_velocity_step_max_m_s"].append(float(np.max(np.abs(np.asarray(b["body_lin_vel_w"]) - a["body_lin_vel_w"]))))
        values["joint_acceleration_step_max_rad_s2"].append(values["joint_velocity_step_max_rad_s"][-1] * FPS)
        first, second = np.asarray(a["body_quat_w"])[0], np.asarray(b["body_quat_w"])[0]
        cosine = abs(float(first @ second)) / float(np.linalg.norm(first) * np.linalg.norm(second))
        values["root_orientation_step_rad"].append(float(2*np.arccos(np.clip(cosine, 0, 1))))
    return {"interpretation": "Adjacent physical control-step changes where reference origin changes; not same-time replan deltas",
            "count": len(values.get("root_position_step_m", [])),
            "metrics": {key: distribution(value) for key, value in values.items()}}


class BaselineRecorder:
    """一个有界实验的记录器；每次只允许一个活动 episode。"""

    def __init__(self, output_dir, data_root=None):
        self.output_dir = Path(output_dir)
        self.data_root = Path(data_root).resolve() if data_root is not None else None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for name in ("events.jsonl", "episodes", "report.json"):
            if (self.output_dir / name).exists():
                raise FileExistsError(f"Refusing to overwrite baseline output: {self.output_dir / name}")
        self._event_stream = (self.output_dir / "events.jsonl").open("x", encoding="utf-8")
        (self.output_dir / "episodes").mkdir()
        self._active = None
        self._summaries = []
        self._pooled = defaultdict(lambda: {"errors": defaultdict(list), "latency": defaultdict(list), "prefix": [], "events": Counter()})
        self._closed = False
        self._final_report = None

    def _load_music(self, sample):
        if "music_features" in sample:
            self.set_music(sample["music_features"])
            return
        if self.data_root is None:
            return
        row = sample.get("row", sample)
        relative = row.get("music_feature_path")
        if relative is None:
            return
        root = self.data_root / sample["dataset"]
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Music path escapes selected dataset")
        if not path.exists():
            self._active["music_status"] = "music_feature_missing"
            return
        expected = row.get("source_music_feature_sha256")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected is not None and expected != digest:
            raise ValueError("Music feature SHA does not match manifest")
        import torch
        features = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(features, torch.Tensor):
            raise ValueError("Music feature file must be raw EDGE35 Tensor")
        self.set_music(features)
        self._active["music_source"] = {"path": str(path), "sha256": digest}

    def start_episode(self, sample, seed, mode, snapshot):
        if self._closed or self._active is not None or self._final_report is not None:
            raise RuntimeError("Recorder is closed or an episode is still active")
        safe_sample = {key: value for key, value in sample.items() if key != "music_features"}
        initial = json_value(snapshot)
        index = len(self._summaries)
        episode_dir = self.output_dir / "episodes" / f"{index:03d}_{mode}_seed{int(seed)}"
        episode_dir.mkdir(exist_ok=False)
        stream = (episode_dir / "trace.jsonl").open("x", encoding="utf-8")
        self._active = {"sample": json_value(safe_sample), "seed": int(seed), "mode": str(mode),
                        "snapshot": initial, "directory": episode_dir, "stream": stream,
                        "rows": [], "events": [], "plans": [], "music": None,
                        "music_start_tick": 600, "music_status": "music_not_provided",
                        "last_tick": int(initial["tick"]), "wall_start": time.perf_counter()}
        _write_json(episode_dir / "initial.json", {"sample": safe_sample, "seed": seed, "mode": mode, "snapshot": initial})
        self._load_music(sample)
        self.event("episode_started", episode_id=initial["episode_id"], mode=mode, seed=int(seed), tick=initial["tick"])

    def set_music(self, features, music_start_tick=600):
        if self._active is None:
            raise RuntimeError("set_music requires active episode")
        if hasattr(features, "detach"):
            features = features.detach().cpu().numpy()
        features = np.asarray(features)
        if features.ndim != 2 or features.shape[1] != 35 or len(features) == 0 or not np.isfinite(features).all():
            raise ValueError("Music must be finite EDGE35 [T,35]")
        self._active["music"] = features.copy()
        self._active["music_start_tick"] = int(music_start_tick)
        self._active["music_status"] = "available"

    def step(self, row):
        if self._active is None:
            raise RuntimeError("step requires active episode")
        value = json_value(row)
        active = self._active
        if value["episode_id"] != active["snapshot"]["episode_id"]:
            raise ValueError("Step crosses episode boundary")
        if value["phase"] not in ("warmup", "music"):
            raise ValueError("Step phase must be warmup or music")
        if int(value["tick"]) != active["last_tick"] + CONTROL_TICKS:
            raise ValueError("Duplicate, missing, or nonmonotonic control step")
        active["stream"].write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        active["stream"].flush()
        active["rows"].append(value)
        active["last_tick"] = int(value["tick"])
        if value["phase"] == "music" and active["mode"] != "calibration":
            pool = self._pooled[active["mode"]]
            for key, metric in value["errors"].items():
                pool["errors"][key].append(metric)
            for key in ("gmt_inference_seconds", "physics_seconds", "step_seconds"):
                if key in value:
                    pool["latency"][key].append(value[key])

    def event(self, name, **payload):
        if self._closed:
            raise RuntimeError("Recorder closed")
        row = {"event": str(name), **json_value(payload)}
        if self._active is not None:
            row.setdefault("episode_id", self._active["snapshot"]["episode_id"])
            row.setdefault("mode", self._active["mode"])
            self._active["events"].append(row)
            if self._active["mode"] != "calibration":
                pool = self._pooled[self._active["mode"]]
                pool["events"][name] += 1
                if name == "plan_prepared":
                    pool["prefix"].append(int(row["prefix_frames"]))
                    for key in ("end_to_end_seconds", "actor_seconds", "prepare_seconds"):
                        if row.get(key) is not None:
                            pool["latency"][key].append(row[key])
        self._event_stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        self._event_stream.flush()

    def plan(self, generated):
        if self._active is None:
            raise RuntimeError("plan requires active episode")
        if generated.get("episode_id") != self._active["snapshot"]["episode_id"]:
            raise ValueError("Generated plan belongs to another episode")
        plan_id = str(generated["plan_id"])
        if plan_id in self._active["plans"]:
            raise FileExistsError(f"Plan already recorded: {plan_id}")
        arrays, metadata = {}, {}
        for key, value in generated.items():
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            if isinstance(value, np.ndarray):
                if value.dtype.hasobject or value.dtype.kind not in "biufUS":
                    raise TypeError(f"Unsupported plan array dtype: {value.dtype}")
                if value.dtype.kind in "f" and not np.isfinite(value).all():
                    raise ValueError("Non-finite generated plan")
                arrays[key] = value
            else:
                metadata[key] = json_value(value)
        directory = self._active["directory"] / "plans"
        directory.mkdir(exist_ok=True)
        name = f"{len(self._active['plans']):04d}_{hashlib.sha256(plan_id.encode()).hexdigest()[:12]}"
        with (directory / f"{name}.npz").open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        _write_json(directory / f"{name}.json", {**metadata, "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in arrays.items()}})
        self._active["plans"].append(plan_id)

    def _music_metrics(self, rows):
        active = self._active
        result = {"status": active["music_status"], "minimum_duration_seconds": 1.0,
                  "music_source": active.get("music_source"), "actual": None, "reference": None}
        if not rows or len(rows) < FPS:
            result["status"] = "insufficient_duration"
            return result
        music = active["music"]
        if music is None:
            return result
        import torch
        from gem.robots.bumi.metrics import _derive_motion_beats, _beat_alignment
        ticks = np.asarray([row["tick"] for row in rows], dtype=np.int64)
        start = active["music_start_tick"]
        valid = (ticks >= start) & (ticks <= start + (len(music)-1)*20)
        # 30 Hz 标记按实际相对时刻映射到最近 50 Hz 栅格，不按帧号混用频率。
        beat_ticks = start + np.rint(np.flatnonzero(music[:, 34] > .5) * 20 / CONTROL_TICKS).astype(np.int64) * CONTROL_TICKS
        music_beats = torch.from_numpy(np.isin(ticks, beat_ticks))[None]
        support = valid.copy()
        if len(support) >= 3:
            support[1:-1] &= valid[:-2] & valid[2:]
        support[[0, -1]] = False
        valid_tensor = torch.from_numpy(support)[None]
        result["music_beat_count"] = int((music_beats & valid_tensor).sum())
        for name, speeds in (("actual", _stack(rows, "actual_joint_vel")),
                             ("reference", _stack(rows, "joint_vel", "reference"))):
            beats = _derive_motion_beats(torch.from_numpy(np.abs(speeds))[None], valid_tensor)
            count = int(beats.sum())
            if result["music_beat_count"] == 0 or count == 0:
                result[name] = {"motion_beat_count": count, "mean_beat_distance_seconds": None,
                                "alignment": None, "status": "no_music_beats" if result["music_beat_count"] == 0 else "no_motion_beats"}
                continue
            distance, alignment = _beat_alignment(music_beats, beats, valid_tensor, fps=FPS)
            result[name] = {"motion_beat_count": count, "mean_beat_distance_seconds": float(distance),
                            "alignment": float(alignment), "status": "available"}
        return result

    def _plots(self, rows, events):
        directory = self._active["directory"]
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return {"status": "matplotlib_unavailable"}
        result = []
        if rows:
            fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            times = [(row["tick"] - self._active["music_start_tick"])/CLOCK_HZ for row in rows]
            for key in ("root_position_error_m", "joint_position_rmse_rad"):
                axes[0].plot(times, [row["errors"][key] for row in rows], label=key)
            axes[0].legend(fontsize=8)
            axes[0].set_ylabel("Tracking error (m / rad)")
            axes[1].plot(times, [row.get("gmt_inference_seconds", 0)*1000 for row in rows], label="GMT inference")
            axes[1].plot(times, [row.get("physics_seconds", 0)*1000 for row in rows], label="Physics")
            axes[1].legend(fontsize=8)
            axes[1].set_ylabel("Wall time (ms)")
            axes[1].set_xlabel("Music time (s)")
            fig.tight_layout()
            target = directory / "tracking_and_control_latency.png"
            if target.exists():
                plt.close(fig)
                raise FileExistsError(target)
            fig.savefig(target, dpi=120)
            plt.close(fig)
            result.append(str(target.relative_to(self.output_dir)))
        plans = [event for event in events if event["event"] == "plan_prepared"]
        if plans:
            fig, axis = plt.subplots(figsize=(10, 3))
            axis.plot([event["request_tick"]/CLOCK_HZ for event in plans], [event["end_to_end_seconds"]*1000 for event in plans], marker=".")
            axis.set(xlabel="Episode time (s)", ylabel="Plan end-to-end latency (ms)")
            fig.tight_layout()
            target = directory / "planning_latency.png"
            if target.exists():
                plt.close(fig)
                raise FileExistsError(target)
            fig.savefig(target, dpi=120)
            plt.close(fig)
            result.append(str(target.relative_to(self.output_dir)))
        return {"status": "written" if result else "no_samples", "paths": result}

    def finish_episode(self, snapshot, reason):
        if self._active is None:
            raise RuntimeError("No active episode")
        active = self._active
        if snapshot["episode_id"] != active["snapshot"]["episode_id"]:
            raise ValueError("Terminal snapshot crosses episode boundary")
        terminal = json_value(snapshot, nonfinite="null")
        rows = [row for row in active["rows"] if row["phase"] == "music"]
        events = active["events"]
        counts = Counter(event["event"] for event in events)
        prepared = [event for event in events if event["event"] == "plan_prepared"]
        commits = [event for event in events if event["event"] == "plan_committed"]
        prefixes = [event["prefix_frames"] for event in prepared]
        metrics = {key: distribution([row["errors"][key] for row in rows]) for key in (rows[0]["errors"] if rows else ())}
        latencies = {key: distribution([row[key] for row in rows if key in row]) for key in ("gmt_inference_seconds", "physics_seconds", "step_seconds")}
        for key in ("end_to_end_seconds", "actor_seconds", "prepare_seconds"):
            latencies[key] = distribution([event[key] for event in prepared if event.get(key) is not None])
        original = {}
        for key, threshold in ORIGINAL_THRESHOLDS.items():
            crossing = terminal.get("original_threshold_first_crossing", {}).get(key)
            all_count = terminal.get("original_threshold_crossing_counts", {}).get(key, 0)
            music_count = sum(row["errors"][key] > threshold for row in rows)
            original[key] = {"threshold": threshold, "first_crossing_tick": crossing,
                             "all_phase_crossing_count": all_count,
                             "music_crossing_count": music_count,
                             "music_crossing_fraction": music_count/len(rows) if rows else None}
        failed = reason not in NORMAL_ENDINGS
        steps = (int(snapshot["tick"]) - int(active["snapshot"]["tick"])) // CONTROL_TICKS
        music_started = bool(rows) or any(event.get("phase") == "music" for event in events)
        music_steps = max(0, (int(snapshot["tick"]) - max(int(active["snapshot"]["tick"]), active["music_start_tick"])) // CONTROL_TICKS) if music_started else 0
        trace_discrepancy = steps - len(active["rows"])
        buffers = [(event["reference_valid_end_tick"] - event["end_tick"])/CLOCK_HZ
                   for event in events if event["event"] == "advance" and event.get("phase") == "music"
                   and event.get("reference_valid_end_tick") is not None]
        terminal_buffer = ((terminal["reference_valid_end_tick"] - terminal["tick"])/CLOCK_HZ
                           if "reference_valid_end_tick" in terminal else None)
        overhead = [max(0., event["end_to_end_seconds"] - event["actor_seconds"] - event["prepare_seconds"])
                    for event in prepared if event.get("actor_seconds") is not None and event.get("prepare_seconds") is not None]
        latencies["coordinator_rpc_overhead_seconds"] = distribution(overhead)
        summary = {"episode_id": snapshot["episode_id"], "sample": active["sample"], "seed": active["seed"],
                   "dataset": active["sample"].get("dataset"),
                   "mode": active["mode"], "reason": str(reason), "backend_reason": terminal.get("reason"),
                   "failed": failed, "phase": "music" if music_started else "warmup", "startup_failure": failed and not music_started,
                   "duration_seconds": music_steps/FPS, "music_duration_seconds": music_steps/FPS,
                   "elapsed_music_seconds": music_steps/FPS,
                   "warmup_duration_seconds": (steps-music_steps)/FPS, "total_duration_seconds": steps/FPS,
                   "executed_control_steps": steps, "executed_physics_steps": 4*steps,
                   "recorded_control_steps": len(active["rows"]), "music_control_steps": music_steps,
                   "recorded_music_control_steps": len(rows),
                   "trace_integrity": {"missing_control_step_records": trace_discrepancy,
                       "complete": trace_discrepancy == 0,
                       "system_error": None if trace_discrepancy == 0 else "executed_steps_missing_trace"},
                   "wall_seconds": time.perf_counter()-active["wall_start"],
                   "generated_plan_count": len(active["plans"]), "replan_count": len(commits),
                   "errors": metrics, "latency_seconds": latencies,
                   "motion": _motion_statistics(rows), "music": self._music_metrics(rows),
                   "reference_boundaries": _boundary_statistics(rows),
                   "protected_reference": {"modification_count": sum(event["acknowledgement"].get("protected_modification_count", 0) for event in commits),
                       "recomputed_position_error": distribution([event["acknowledgement"]["max_position_error"] for event in commits if "max_position_error" in event["acknowledgement"]]),
                       "recomputed_velocity_error": distribution([event["acknowledgement"]["max_velocity_error"] for event in commits if "max_velocity_error" in event["acknowledgement"]])},
                   "prefix_frames": distribution(prefixes), "prefix_over_18_count": sum(p > 18 for p in prefixes),
                   "prefix_over_18_fraction": sum(p > 18 for p in prefixes)/len(prefixes) if prefixes else None,
                   "reference_buffer": {"remaining_seconds_at_advance_end": distribution(buffers),
                       "terminal_remaining_seconds": terminal_buffer,
                       "terminal_gmt_lookahead_margin_seconds": terminal_buffer-.2 if terminal_buffer is not None else None,
                       "reference_underrun": str(reason) == "reference_underrun"},
                   "event_counts": dict(counts), "rejection_codes": dict(Counter(event.get("code", "unknown") for event in events if event["event"] in ("plan_rejected", "request_rejected"))),
                   "original_threshold_diagnostics": {"interpretation": "Threshold crossing only; not measured strict-threshold episode failure rate", "metrics": original},
                   "artifacts": {"trace": str((active["directory"] / "trace.jsonl").relative_to(self.output_dir))},
                   "terminal_snapshot": terminal}
        summary["artifacts"]["plots"] = self._plots(rows, events)
        self.event("episode_finished", episode_id=snapshot["episode_id"], reason=str(reason), failed=failed,
                   music_control_steps=len(rows), total_control_steps=steps)
        active["stream"].close()
        _write_json(active["directory"] / "summary.json", summary)
        self._summaries.append(summary)
        self._active = None
        return copy.deepcopy(summary)

    def summarize(self):
        if self._active is not None:
            raise RuntimeError("Finish active episode before final report")
        if self._final_report is not None:
            return copy.deepcopy(self._final_report)
        evaluation = [episode for episode in self._summaries if episode["mode"] != "calibration"]
        modes = {}
        for mode in sorted({episode["mode"] for episode in evaluation}):
            selected = [episode for episode in evaluation if episode["mode"] == mode]
            failures = sum(episode["failed"] for episode in selected)
            exposure = sum(episode["music_duration_seconds"] for episode in selected)
            pool = self._pooled[mode]
            modes[mode] = {"episodes": len(selected), "failures": failures,
                           "failure_fraction": failures/len(selected),
                           "startup_failures": sum(episode["startup_failure"] for episode in selected),
                           "music_exposure_seconds": exposure,
                           "music_failures": sum(episode["failed"] and not episode["startup_failure"] for episode in selected),
                           "failures_per_music_minute": sum(episode["failed"] and not episode["startup_failure"] for episode in selected)/(exposure/60) if exposure else None,
                           "incomplete_trace_episodes": sum(not episode["trace_integrity"]["complete"] for episode in selected),
                           "reason_counts": dict(Counter(episode["reason"] for episode in selected)),
                           "errors": {key: distribution(value) for key, value in pool["errors"].items()},
                           "latency_seconds": {key: distribution(value) for key, value in pool["latency"].items()},
                           "prefix_frames": distribution(pool["prefix"]), "event_counts": dict(pool["events"])}
        report = {"schema": "genmo.closedloop_stage8_baseline.v1", "episodes": copy.deepcopy(self._summaries),
                  "evaluation_episode_count": len(evaluation), "calibration_episode_count": len(self._summaries)-len(evaluation),
                  "modes": modes, "metric_scope": "Music-phase control steps; warmup is reported separately",
                  "validation_boundary": "This report describes the recorded bounded runs; no training or deployment claim"}
        _write_json(self.output_dir / "report.json", report)
        self._final_report = report
        return copy.deepcopy(report)

    def close(self):
        if self._closed:
            return
        if self._active is not None:
            self._active["stream"].close()
            self._active = None
        self._event_stream.close()
        self._closed = True
