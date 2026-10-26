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

from .application import seed_all, task_inputs, task_identity


def evaluation_contract(cfg, tasks, frozen_sha256):
    """允许预先运行更新前物理基线；比较实际条件和任务，不比较源码版本。"""
    teacher = None
    if cfg.get("teacher",{}).get("enabled"):
        from supermimic.bridge.bundle import file_digest
        teacher = {"seed":cfg["teacher"]["seed"],"schema":"frozen_stage1_music_teacher.v1",
            "assets":{k:file_digest(cfg["stage1"][k]) for k in ("checkpoint","stats","kinematics")}}
    return {"model": cfg["model"], "clock": cfg["clock"], "rewards": cfg["rewards"], "teacher":teacher,
            "physics_determinism": cfg.get("execution", {}).get("physics_determinism", "not_guaranteed"),
            "frozen_sha256": frozen_sha256,
            "tasks": [{"id": t["id"], "identity": task_identity(t), "seeds": t.get("seeds", [42])} for t in tasks]}


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
                   "actor_version": collector.actor_version,
                   "source": task.get("source"), "split": task.get("split"),
                   "complete": reason == "music_end", "physical_seconds": collector.state.tick / 600,
                   "music_seconds": music.shape[1] / 30, "wall_seconds": time.perf_counter() - start,
                   "physical_metrics_mean": averages, "motion": motion_metrics(collector.trajectory, music),
                   "commits": sum(e["accepted"] for e in collector.events),
                   "timeouts": sum(not e["accepted"] for e in collector.events),
                   "control_wall_overruns": collector.overruns, "latency_ticks": latencies,
                   "events": collector.events, "cancelled_requests": collector.cancelled_requests,
                   "contacts": ({"source": "mujoco_warp_solver_active_contacts", **collector.contact_counts}
                                if collector.record_contact_metrics else None),
                   "generator_unseen_music": task.get("generator_unseen_music"),
                   "controller_unseen_motion": task.get("controller_unseen_motion"),
                   "evidence": "real_isaaclab3_newton_mujoco_warp"}
            rows.append(row)
            if collector.trajectory:
                state = torch.cat([v[1] for v in collector.trajectory])
                reference_q = torch.cat([v[5] for v in collector.trajectory])
                row["reference_motion"] = {"joint_range_mean_rad":float((reference_q.amax(0)-reference_q.amin(0)).mean()),
                    "joint_speed_rms_rad_s":float((torch.diff(reference_q,dim=0)*50).square().mean().sqrt())}
                teacher_arrays = {}
                if collector.state.teacher_reference is not None:
                    clock = torch.tensor([v[0] for v in collector.trajectory],device=collector.state.state.device)
                    teacher_arrays["fixed_stage1_teacher_world_qpos28"] = collector.state.teacher_reference.source_at(clock).squeeze(0).cpu().numpy()
                    teacher_arrays["fixed_stage1_teacher_joint_names"] = np.asarray(collector.actor.stage1.endecoder.kinematics.joint_order)
                diversity.setdefault(task["id"], []).append(state[:, 13:34])
                # 同一文件兼容SM现有渲染器：真实状态与当帧原始世界参考逐帧配对。
                # 不保存扩散链，不重新对齐影子，不使用未来覆盖后的缓存补造参考。
                safe_id = str(task["id"]).replace("/", "_").replace("..", "_")
                n = len(state)
                alive = np.ones((n, 1), dtype=bool)
                if reason == "physical_failure":
                    alive[-1] = False
                np.savez_compressed(output / f"{safe_id}_seed{seed}_physical.npz",
                    ticks=np.asarray([v[0] for v in collector.trajectory]), state=state.numpy(),
                    root_p=state[:, None, :3].numpy(), root_quat=state[:, None, 3:7].numpy(),
                    q=state[:, None, 13:34].numpy(), alive=alive, f=np.arange(n)[:, None],
                    ref_root_p=torch.cat([v[2] for v in collector.trajectory]).numpy(),
                    ref_root_R=torch.cat([v[4] for v in collector.trajectory]).numpy(),
                    ref_q=torch.cat([v[5] for v in collector.trajectory]).numpy(),
                    names=np.asarray([task["id"]]), clip_start=np.asarray([0]),
                    robot_id=np.asarray(collector.adapter.bundle.spec.robot_id),
                    joint_names=np.asarray(collector.adapter.bundle.spec.joint_names),
                    actor_version=np.asarray(collector.actor_version),
                    termination_reason=np.asarray(reason), music_seconds=np.asarray(music.shape[1]/30),
                    physics_backend=np.asarray("isaaclab3_newton_mujoco_warp"),**teacher_arrays)
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
