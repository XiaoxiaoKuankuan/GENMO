"""固定名义参考/音乐/初态的真实评价与音乐诊断。

评价只接受真实转移，保持原离散节拍函数，并增加双向距离、唯一匹配F1、密度、活动
幅度、jerk、脚滑和完整控制路径分位延迟。脚滑需要求解器真实接触标签；没有标签时
明确未测，不能用参考contact或足高冒充。三组比较校验相同名义资产与采集/更新预算。
"""

import numpy as np
import torch


def beat_report(qd, ticks, audio, tolerance_s=.1):
    from gem.robots.bumi.metrics import _derive_motion_beats, _beat_alignment
    qd, ticks = qd.detach().cpu(), ticks.detach().cpu()
    valid = ((ticks >= audio.start_tick) & (ticks <= audio.end_tick))[None]
    indices = ((ticks-audio.start_tick)//20).clamp(0, audio.features.shape[1]-1)
    valid &= audio.valid[:1, indices.to(audio.valid.device)].cpu()
    motion_mask = _derive_motion_beats(qd.abs()[None], valid)
    motion_ticks = ticks[motion_mask[0]]
    beat_ticks = audio.beat_ticks[0].cpu()
    beat_ticks = beat_ticks[(beat_ticks >= ticks[0]) & (beat_ticks <= ticks[-1])]
    music_mask = torch.zeros_like(valid)
    if len(beat_ticks):
        nearest = (ticks[:, None]-beat_ticks[None]).abs().argmin(0)
        music_mask[0, nearest] = True
    distance, alignment = _beat_alignment(music_mask, motion_mask, valid, fps=50)
    duration = max((int(ticks[-1])-int(ticks[0]))/600, .02)
    result = dict(music_beat_count=len(beat_ticks), motion_beat_count=len(motion_ticks),
                  motion_beats_per_second=len(motion_ticks)/duration,
                  legacy_mean_distance_s=float(distance) if len(beat_ticks) and len(motion_ticks) else None,
                  legacy_alignment=float(alignment) if len(beat_ticks) and len(motion_ticks) else None)
    if not len(beat_ticks) or not len(motion_ticks):
        return result | dict(precision=0. if len(beat_ticks) else None, recall=0. if len(beat_ticks) else None,
                             f1=0. if len(beat_ticks) else None, audio_to_motion_s=None, motion_to_audio_s=None)
    distances = (motion_ticks[:, None]-beat_ticks[None]).abs().float()/600
    # 有序一对一容差匹配，避免高频多造速度谷提高recall却逃避precision。
    i = j = matches = 0
    while i < len(motion_ticks) and j < len(beat_ticks):
        delta = int(motion_ticks[i]-beat_ticks[j])/600
        if abs(delta) <= tolerance_s:
            matches += 1
            i += 1
            j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    p, r = matches/len(motion_ticks), matches/len(beat_ticks)
    return result | dict(precision=p, recall=r, f1=2*p*r/max(p+r, 1e-12),
                         audio_to_motion_s=float(distances.amin(0).mean()), motion_to_audio_s=float(distances.amin(1).mean()))


def summarize(rows, *, contacts=None, runtime=None):
    if not rows or any(r.start.origin != "real" or r.end.origin != "real" for r in rows):
        raise ValueError("物理评价需要非空真实转移，不接受模型回报")
    states = torch.cat([rows[0].start.executed_state] + [r.end.executed_state for r in rows])
    ticks = torch.tensor([rows[0].start.tick]+[r.end.tick for r in rows])
    q, qd = states[:, 13:34], states[:, 34:]
    acceleration = torch.diff(qd, dim=0)/.02
    jerk = torch.diff(qd, n=2, dim=0)/.02**2
    latency = np.array([r.wall_seconds for r in rows])
    keys = ["body_pos", "body_ori", "body_linvel", "body_angvel", "root_pos", "root_ori",
            "error_body_pos", "error_body_ori", "error_body_linvel", "error_body_angvel",
            "error_root_pos", "error_root_ori", "track", "action_rate", "joint_limit", "residual_saturation",
            "halo_max_abs", "target_qd_max_abs", "music", "activity"]
    terms = {k: float(torch.cat([r.raw[k].reshape(-1) for r in rows]).mean()) for k in keys}
    slip = None
    if contacts is not None and runtime is not None:
        from supermimic.mdp import unflatten_state
        body_v = runtime.kin.forward(*unflatten_state(states[1:]))[2]
        names = tuple(runtime.bundle.spec.tracked_bodies)
        foot_idx = [names.index(n) for n in ("l_ankle_roll_link", "r_ankle_roll_link")]
        mask = torch.tensor([[c["left_foot"], c["right_foot"]] for c in contacts], device=body_v.device)
        speed = body_v[:, foot_idx, :2].norm(dim=-1)
        slip = float(speed[mask].mean()) if mask.any() else 0.
    return dict(steps=len(rows), reason=rows[-1].reason, physical_failure=rows[-1].reason == "physical_failure",
                supported_reference_complete=rows[-1].reason in ("music_end", "reference_end"),
                full_music_complete=rows[-1].reason == "music_end", final_tick=rows[-1].end.tick,
                total_reward=float(sum(r.reward.sum() for r in rows)), sm_terms=terms,
                root_position_rmse_m=terms["error_root_pos"]**.5, root_orientation_rmse_rad=terms["error_root_ori"]**.5,
                music=beat_report(qd, ticks, rows[0].start.audio),
                mean_abs_joint_speed=float(qd.abs().mean()), joint_range_mean=float((q.amax(0)-q.amin(0)).mean()),
                acceleration_rms=float(acceleration.square().mean().sqrt()),
                jerk_rms=float(jerk.square().mean().sqrt()) if jerk.numel() else None,
                contact_foot_slip_mps=slip, foot_slip_status="solver_contact_measured" if slip is not None else "not_measured",
                latency_ms={f"p{p}": float(np.percentile(latency, p)*1000) for p in (50, 95, 99)},
                deadline_misses=int((latency > .02).sum()),
                latency_scope="Adapter+reference+FK+Actor+physics+reward+history; fixed_cache_no_GENMO_concurrency")


def validate_comparison(reports):
    if set(reports) != {"zero", "no_music", "music"}:
        raise ValueError("配对比较需要zero/no_music/music三组")
    baseline = reports["zero"]["task_contract"]
    if any(r["task_contract"] != baseline for r in reports.values()):
        raise ValueError("音乐/噪声/初态/名义计划/物理合同不一致")
    a, b = reports["no_music"]["budget"], reports["music"]["budget"]
    if a != b:
        raise ValueError("有/无音乐训练必须匹配真实采集量和Adapter更新数")
    return {name: r["episodes"] for name, r in reports.items()}


def quality_comparison(baseline, current, thresholds):
    """同任务真实配对监控：完成、名义跟踪和音乐任一显著退化即拒绝候选。"""
    by_task = {r["task"]: r for r in baseline}
    cases = []
    for row in current:
        old = by_task[row["task"]]
        checks = dict(completion=int(row["full_music_complete"]) >= int(old["full_music_complete"]),
                      tracking=row["sm_terms"]["track"] >= old["sm_terms"]["track"]*(1-thresholds["tracking_relative_drop"]),
                      music_reward=row["sm_terms"]["music"] >= old["sm_terms"]["music"]-thresholds["music_absolute_drop"])
        if old["music"]["f1"] is not None:
            checks["music_f1"] = row["music"]["f1"] is not None and row["music"]["f1"] >= old["music"]["f1"]-thresholds["f1_absolute_drop"]
        cases.append(dict(task=row["task"], checks=checks, passed=all(checks.values())))
    return dict(passed=bool(cases) and all(c["passed"] for c in cases), cases=cases, thresholds=thresholds)
