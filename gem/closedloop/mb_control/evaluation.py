"""全曲真实物理评价与音乐/动作退化诊断，绝不合并模型想象奖励。

主baseline为原Stage1+固定SM；条件消融必须加载对应条件训练的checkpoint。记录实际
失败/时长、原始根误差、八项与SM六项、幅度/速度、节拍和跨种子多样性。音乐是既有
EDGE35输入，不生成新配对库；七库SM预训练的动作可见性与生成器音乐split分别报告。
"""

from pathlib import Path
import json
import time

import numpy as np
import torch

from .application import seed_all, task_inputs


def motion_metrics(trajectory, music):
    if not trajectory:
        return {"status": "no_physical_steps"}
    ticks = np.asarray([r[0] for r in trajectory])
    state = torch.cat([r[1] for r in trajectory])
    q, qd = state[:, 13:34], state[:, 34:55]
    result = {"joint_range_mean_rad": float((q.max(0).values - q.min(0).values).mean()),
              "joint_speed_rms_rad_s": float(qd.square().mean().sqrt()),
              "root_travel_m": float(torch.diff(state[:, :3], dim=0).norm(dim=-1).sum())}
    # 沿用既有音乐节拍口径：EDGE35第最后一维为beat标记，映射到实际50Hz时刻。
    from gem.robots.bumi.metrics import _derive_motion_beats, _beat_alignment
    beat_source = np.flatnonzero(music[0, :, -1].cpu().numpy() > .5)
    beat_ticks = np.rint(beat_source * 20 / 12).astype(int) * 12
    beats = torch.as_tensor(np.isin(ticks, beat_ticks))[None]
    valid = torch.as_tensor((ticks >= 0) & (ticks <= (music.shape[1] - 1) * 20))[None]
    if valid.shape[1]:
        valid[:, 0] = False
        valid[:, -1] = False
    motion_beats = _derive_motion_beats(qd.abs()[None], valid)
    result.update(music_beat_count=int(beats.sum()), motion_beat_count=int(motion_beats.sum()))
    if (beats & valid).any() and motion_beats.any():
        distance, alignment = _beat_alignment(beats, motion_beats, valid, fps=50.)
        result.update(beat_distance_seconds=float(distance.mean()), beat_alignment=float(alignment.mean()))
    else:
        result.update(beat_distance_seconds=None, beat_alignment=None, beat_status="insufficient_beats")
    return result


def evaluate_tasks(collector, tasks, output, *, mode, device, max_seconds=None):
    output = Path(output)
    rows, diversity, latency = [], {}, []
    for task in tasks:
        for seed in task.get("seeds", [42]):
            seed_all(seed)
            music, initial = task_inputs(task, collector.adapter, device)
            collector.reset(music, initial)
            start = time.perf_counter()
            reason = "administrative_truncation"
            while not collector.terminal:
                intervals = collector.collect(1)
                reason = intervals[-1].reason
                if max_seconds is not None and collector.state.tick / 600 >= max_seconds:
                    reason = "administrative_truncation"
                    break
            metrics = collector.metrics
            averages = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]} if metrics else {}
            latencies = [e["actual_ready_tick"] - e["request_tick"] for e in collector.events]
            latency.extend(latencies)
            row = {"task": task["id"], "seed": seed, "mode": mode, "reason": reason,
                   "complete": reason == "music_end", "physical_seconds": collector.state.tick / 600,
                   "music_seconds": music.shape[1] / 30, "wall_seconds": time.perf_counter() - start,
                   "physical_metrics_mean": averages, "motion": motion_metrics(collector.trajectory, music),
                   "commits": sum(e["accepted"] for e in collector.events),
                   "timeouts": sum(not e["accepted"] for e in collector.events),
                   "control_wall_overruns": collector.overruns, "latency_ticks": latencies,
                   "events": collector.events, "cancelled_requests": collector.cancelled_requests,
                   "generator_unseen_music": task.get("generator_unseen_music"),
                   "controller_unseen_motion": task.get("controller_unseen_motion"),
                   "evidence": "real_isaaclab3_newton_mujoco_warp"}
            rows.append(row)
            if collector.trajectory:
                state = torch.cat([v[1] for v in collector.trajectory])
                diversity.setdefault(task["id"], []).append(state[:, 13:34])
                # 只保存真实紧凑轨迹；没有扩散链/每请求完整状态JSON。
                safe_id = str(task["id"]).replace("/", "_").replace("..", "_")
                np.savez_compressed(output / f"{safe_id}_seed{seed}_physical.npz",
                                    ticks=np.asarray([v[0] for v in collector.trajectory]), state=state.numpy())
            with (output / "physical_results.jsonl").open("a") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    pairwise = {}
    for name, motions in diversity.items():
        distances = []
        for i, a in enumerate(motions):
            for b in motions[:i]:
                n = min(len(a), len(b))
                distances.append(float((a[:n] - b[:n]).square().mean().sqrt()))
        pairwise[name] = {"paired_overlap_joint_rmse_rad": float(np.mean(distances)) if distances else None,
                          "pairs": len(distances), "note": "按相同音乐时刻比较；失败后的缺失不补齐"}
    summary = {"mode": mode, "episodes": len(rows), "completed": sum(r["complete"] for r in rows),
               "diversity": pairwise, "model_reward": None, "physical_results": rows,
               "quality_improvement_claim": False}
    (output / "physical_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    if latency:
        (output / "deployment_latency.json").write_text(json.dumps({
            "scope": "single_robot_forecast_genmo_reference_full_path", "mode": mode,
            "forecast_enabled": collector.worker.forecast_enabled,
            "frozen_sha256": collector.adapter.bundle.metadata["sha256"],
            "ddim_steps": 20, "motion_frames": 60, "precision": "fp32",
            "latency_ticks": latency, "clock_hz": 600, "control_wall_overruns": sum(r["control_wall_overruns"] for r in rows)}, indent=2))
    return summary
