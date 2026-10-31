"""对音乐和英文文本生成结果做同任务、同种子、同时长的只读离线比较。

输入是已经生成的 BUMI qpos28，不调用模型、不改权重，也不寻找重建损失最优模型。
复用机器人 FK 与既有运动学指标，额外计算关节活动、根 jerk、实际 EDGE35 音乐
拍点和运动速度谷的单次匹配 F1/双向距离。脚滑采用统一的离地高度门限，避免用
低速度反推接触再评低速度的循环；它始终只是运动学 proxy，没有接触力或物理真值。

manifest 中每个任务列出所有比较模型的文件。缺失、损坏、时长或种子冲突均保留在
报告中，公平聚合仅使用所有模型均有效的完整配对。原有 PT 的 generation_contract
可以验证条件与种子；没有这些元数据的旧导出只记录 manifest 声明，绝不称已验证。
盲评复制去模型名和来源路径的标准 NPZ，随机编号，另存揭盲表；文本语义评分保持
空白等待人工，既不套 SMPL 语义指标，也不把平滑、节拍或运动学有效性当作语义分。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from .kinematics import BumiKinematics, sha256_file
from .metrics import _derive_motion_beats, compute_bumi_kinematic_metrics, metrics_to_json

FPS = 30
SCHEMA = "genmo.bumi.multimodal_generation_evaluation.v1"
_SCORE_FIELDS = (
    "category_match",
    "direction_match",
    "instruction_match",
    "naturalness",
    "reviewer_notes",
)


def _path(value, parent):
    path = Path(value).expanduser()
    return (parent / path if not path.is_absolute() else path).resolve()


def _json(path, payload):
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def read_manifest(path):
    """统一时间网格；任务标识唯一，split 只能 val/test，缺失模型交由报告显式列出。"""
    path = Path(path).expanduser().resolve(strict=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    tasks = payload["tasks"] if isinstance(payload, dict) else payload
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("manifest 必须包含非空 tasks 列表")
    names, seen, normalized = set(), set(), []
    for raw in tasks:
        task = dict(raw)
        if (
            not isinstance(task.get("task_id"), str)
            or not task["task_id"]
            or task["task_id"] in seen
        ):
            raise ValueError("每个 task_id 必须为唯一非空字符串")
        seen.add(task["task_id"])
        if task.get("split") not in ("val", "test"):
            raise ValueError("离线评价只接受 val/test，不能把训练任务混入留出评价")
        kind = task.get("condition_kind")
        has_text = isinstance(task.get("text"), str) and bool(task["text"].strip())
        has_music = bool(task.get("music_features"))
        if kind not in ("music", "text") or has_text == has_music or has_music != (kind == "music"):
            raise ValueError("任务必须明确音乐或非空文本，且只提供一种条件")
        if not isinstance(task.get("seed"), int) or isinstance(task["seed"], bool):
            raise ValueError("每个任务必须声明整数 seed")
        seconds = float(task["duration_s"])
        if not math.isfinite(seconds) or seconds < 0.02:
            raise ValueError("duration_s 必须有限且至少0.02秒")
        task["duration_s"] = seconds
        task["duration_ticks"] = round(seconds * 600)
        # 闭区间内实际30Hz样本；额外模型尾帧不进入指标，也不插值制造运动学样本。
        task["evaluated_frames"] = task["duration_ticks"] // 20 + 1
        ground = float(task.get("ground_height_m", 0.0))
        if not math.isfinite(ground):
            raise ValueError("ground_height_m 必须有限")
        task["ground_height_m"] = ground
        models = task.get("models")
        if not isinstance(models, dict) or any(not isinstance(k, str) or not k for k in models):
            raise ValueError("models 必须是非空模型标签到文件路径的映射")
        task["models"] = {k: str(_path(v, path.parent)) if v else None for k, v in models.items()}
        names.update(models)
        if has_music:
            task["music_features"] = str(_path(task["music_features"], path.parent))
        else:
            task["text"] = task["text"].strip()
        normalized.append(task)
    if isinstance(payload, dict) and payload.get("model_labels") is not None:
        declared = payload["model_labels"]
        if not isinstance(declared, list) or any(not isinstance(k, str) or not k for k in declared):
            raise ValueError("model_labels 必须是模型名列表")
        if not names.issubset(set(declared)):
            raise ValueError("任务模型未包含在 model_labels 中")
        names = set(declared)
    if not names:
        raise ValueError("manifest 没有任何模型标签")
    return path, normalized, sorted(names)


def _music(task):
    if task["condition_kind"] == "text":
        return None, None
    path = Path(task["music_features"])
    if path.suffix == ".npy":
        array = torch.from_numpy(np.load(path, allow_pickle=False))
    else:
        array = torch.load(path, map_location="cpu", weights_only=False)
    array = torch.as_tensor(array, dtype=torch.float32)
    if array.ndim != 2 or array.shape[1] != 35 or len(array) < 2 or not torch.isfinite(array).all():
        raise ValueError("音乐必须为有限的原始30Hz EDGE35 [T,35]，不能转置或自动补帧")
    if len(array) * 20 < task["duration_ticks"]:
        raise ValueError("实际音乐特征未覆盖声明的评价时长")
    return array[: task["evaluated_frames"]], sha256_file(path)


def load_trajectory(path, task, kin, music_sha256=None):
    """读取具名 qpos；仅重排关节，无修剪限位、平滑、重定向或推测FPS。"""
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            payload = {key: data[key].copy() for key in data.files}
    else:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("轨迹必须是含 qpos/world_qpos、joint_names 的字典或 NPZ")
    contract = payload.get("generation_contract", {})
    # 当前统一Stage1 reference.pt没有独立fps字段，其world_qpos合同固定30Hz。
    fps = payload.get("fps", FPS if "world_qpos" in payload and contract else None)
    if fps is None or float(fps) != FPS:
        raise ValueError("轨迹必须显式 fps=30，或使用当前Stage1 world_qpos/generation_contract合同")
    if int(payload.get("start_tick", 0)) != 0:
        raise ValueError("配对轨迹必须从任务0时刻开始")
    names = tuple(str(name) for name in payload["joint_names"])
    if len(names) != 21 or len(set(names)) != 21 or set(names) != set(kin.joint_order):
        raise ValueError("joint_names 必须恰好是BUMI的21个具名关节")
    qpos = torch.as_tensor(payload.get("world_qpos", payload.get("qpos")), dtype=torch.float32)
    if qpos.ndim == 3 and qpos.shape[0] == 1:
        qpos = qpos[0]
    if qpos.ndim != 2 or qpos.shape[1] != 28 or len(qpos) < task["evaluated_frames"]:
        raise ValueError("轨迹必须为完整覆盖任务时长的 [T,28] qpos，不能补末帧")
    source_frames = len(qpos)
    qpos = qpos[: task["evaluated_frames"]].clone()
    order = [names.index(name) for name in kin.joint_order]
    qpos[:, 7:] = qpos[:, 7:][:, order]
    checked = []
    expected = {
        "seed": task["seed"],
        "duration_ticks": task["duration_ticks"],
        "condition_kind": task["condition_kind"],
    }
    if task["condition_kind"] == "text":
        expected["text_sha256"] = hashlib.sha256(task["text"].encode()).hexdigest()
    else:
        expected["music_sha256"] = music_sha256
    for key, value in expected.items():
        if key in contract:
            if contract[key] != value:
                raise ValueError(f"生成合同与配对任务不一致: {key}")
            checked.append(key)
    declared_kin = contract.get("assets", {}).get("kinematics")
    if declared_kin is not None and declared_kin != kin.kinematics_sha256:
        raise ValueError("生成合同的运动学资产与评价FK不一致")
    if "seed" in payload and int(payload["seed"]) != task["seed"]:
        raise ValueError("轨迹seed与配对任务不一致")
    if "seed" in payload:
        checked.append("seed")
    if "execution_duration_s" in payload:
        if round(float(payload["execution_duration_s"]) * 600) != task["duration_ticks"]:
            raise ValueError("轨迹可执行时长与配对任务不一致")
        checked.append("execution_duration_s")
    return qpos, dict(
        source_frames=source_frames,
        discarded_tail_frames=source_frames - len(qpos),
        generation_fields_verified=checked,
        kinematics_identity_verified=declared_kin is not None,
        unverified_manifest_fields=sorted(set(expected) - set(checked)),
        source_sha256=sha256_file(path),
    )


def beat_metrics(music_indices, motion_indices, *, tolerance_s=0.06):
    """使用一对一时间容差匹配，拒绝把一个运动事件重复给多个音乐拍点加分。"""
    music = torch.as_tensor(music_indices, dtype=torch.float64).reshape(-1).sort().values / FPS
    motion = torch.as_tensor(motion_indices, dtype=torch.float64).reshape(-1).sort().values / FPS
    nm, npred = len(music), len(motion)
    i = j = matches = 0
    while i < nm and j < npred:
        delta = float(motion[j] - music[i])
        if abs(delta) <= tolerance_s + 1e-12:
            matches += 1
            i += 1
            j += 1
        elif delta < 0:
            j += 1
        else:
            i += 1
    precision = matches / npred if npred else 0.0
    recall = matches / nm if nm else 0.0
    forward = reverse = None
    if nm and npred:
        # searchsorted只保留两个相邻候选，整曲也不构造拍点数平方矩阵。
        def nearest(a, b):
            right = torch.searchsorted(b, a).clamp_max(len(b) - 1)
            left = (right - 1).clamp_min(0)
            return float(torch.minimum((a - b[left]).abs(), (a - b[right]).abs()).mean())

        forward, reverse = nearest(music, motion), nearest(motion, music)
    return dict(
        music_beat_count=nm,
        motion_beat_count=npred,
        matched_beat_count=matches,
        beat_precision=precision if nm else None,
        beat_recall=recall if nm else None,
        beat_f1=(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        if nm
        else None,
        music_to_motion_distance_s=forward,
        motion_to_music_distance_s=reverse,
        bidirectional_beat_distance_s=None if forward is None else (forward + reverse) / 2,
        beat_status="ok" if nm and npred else "no_motion_beats" if nm else "no_music_beats",
        beat_tolerance_s=tolerance_s,
    )


@torch.no_grad()
def trajectory_metrics(qpos, kin, *, music=None, ground_height=0.0):
    """完整任务窗的运动学质量；没有足底真实接触、文本语义或动力学成功标签。"""
    finite = torch.isfinite(qpos)
    metrics = dict(finite_fraction=float(finite.float().mean()), all_finite=bool(finite.all()))
    if not metrics["all_finite"]:
        return metrics, "nonfinite_qpos"
    norm = qpos[:, 3:7].double().norm(dim=-1)
    metrics["root_quaternion_norm_max_error"] = float((norm - 1).abs().max())
    if bool((norm < 1e-8).any()) or metrics["root_quaternion_norm_max_error"] > 0.001:
        return metrics, "invalid_root_quaternion"
    fk = kin.forward_kinematics(qpos)
    soles = kin.aggregate_sole_by_foot(fk["body_pos_w"], fk["body_quat_w"])
    support = soles["foot_bottom_height"] <= ground_height + 0.025
    values = metrics_to_json(
        compute_bumi_kinematic_metrics(
            qpos, kin, target_contact=support, ground_height=ground_height
        )
    )
    # 单条轨迹的diversity=0没有跨种子含义，不能混入本评价。
    values.pop("diversity_joint_rms_rad")
    slide_intervals = int((support[1:] & support[:-1]).sum())
    for key in tuple(values):
        if key.startswith("foot_sliding_"):
            value = values.pop(key)
            values["height_gated_proxy_" + key] = value if slide_intervals else None
    metrics.update(values)
    metrics.update(
        height_gated_support_fraction=float(support.float().mean()),
        height_gated_foot_intervals=slide_intervals,
        joint_activity_range_mean_rad=float((qpos[:, 7:].amax(0) - qpos[:, 7:].amin(0)).mean()),
    )
    speed = torch.diff(qpos[:, 7:], dim=0).abs() * FPS
    metrics["joint_activity_mean_abs_velocity_radps"] = (
        float(speed.mean()) if speed.numel() else None
    )
    jerk = torch.diff(qpos[:, :3], n=3, dim=0) * FPS**3
    root_jerk = jerk.norm(dim=-1)
    metrics.update(
        root_jerk_mean_mps3=float(root_jerk.mean()) if len(jerk) else None,
        root_jerk_p95_mps3=float(torch.quantile(root_jerk, 0.95)) if len(jerk) else None,
        root_jerk_max_mps3=float(root_jerk.max()) if len(jerk) else None,
    )
    for order, names in ((1, ("velocity",)), (2, ("acceleration",)), (3, ("jerk",))):
        if len(qpos) <= order:
            for key in metrics:
                if any("_" + name + "_" in key for name in names):
                    metrics[key] = None
    if music is None:
        metrics.update({key: None for key in beat_metrics([], [])})
        metrics["beat_status"] = "not_applicable_text"
        metrics["beat_evaluated_frames"] = None
        metrics["beat_evaluated_last_time_s"] = None
    else:
        padded = torch.cat((speed[:1], speed), dim=0) if len(speed) else torch.zeros(len(qpos), 21)
        # runtime允许duration=N/30与N帧EDGE35；末个运动边界点没有音乐特征。
        # 仅实际音乐覆盖的轴提取运动拍点，不补音乐末帧，也不用覆盖外运动判定末端极值。
        covered = len(music)
        events = _derive_motion_beats(
            padded[None, :covered], torch.ones(1, covered, dtype=torch.bool)
        )[0]
        metrics.update(beat_metrics(torch.where(music[:, 34] > 0.5)[0], torch.where(events)[0]))
        metrics.update(
            beat_evaluated_frames=covered, beat_evaluated_last_time_s=(covered - 1) / FPS
        )
    nonfinite = [
        key
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and not math.isfinite(value)
    ]
    if nonfinite:
        for key in nonfinite:
            metrics[key] = None
        metrics["nonfinite_derived_metrics"] = nonfinite
        return metrics, "nonfinite_derived_metrics"
    return metrics, None


def _aggregate(rows, labels, paired):
    result = {}
    for label in labels:
        result[label] = {}
        for split in ("val", "test"):
            for kind in ("music", "text"):
                selected = [
                    r
                    for r in rows
                    if r["model"] == label
                    and r["split"] == split
                    and r["condition_kind"] == kind
                    and (not paired or r["paired_complete"])
                ]
                valid = [r for r in selected if r["status"] == "ok"]
                keys = {k for r in valid for k in r["metrics"]}
                means, counts = {}, {}
                for key in sorted(keys):
                    values = [
                        r["metrics"][key]
                        for r in valid
                        if isinstance(r["metrics"].get(key), (int, float))
                    ]
                    means[key] = sum(values) / len(values) if values else None
                    counts[key] = len(values)
                result[label][split + "/" + kind] = dict(
                    total=len(selected),
                    valid=len(valid),
                    invalid=len(selected) - len(valid),
                    mean=means,
                    contributing_tasks=counts,
                )
    return result


def evaluate_manifest(manifest_path, kinematics_path, output_root, *, blind_seed=42):
    """写全新评价目录；输入只读，任一模型缺失照常出报告并标记配对不完整。"""
    manifest, tasks, labels = read_manifest(manifest_path)
    kin = BumiKinematics(kinematics_path)
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    assets = output / "blind_assets"
    assets.mkdir()
    rows, blind, reveal, pairing, blind_status = [], [], [], [], []
    rng = random.Random(blind_seed)
    for task in tasks:
        task_rows, clips = [], []
        try:
            music, music_sha = _music(task)
            condition_error = None
        except Exception as exc:
            music, music_sha = None, None
            condition_error = f"condition:{type(exc).__name__}:{exc}"
        for label in labels:
            path = task["models"].get(label)
            row = dict(
                task_id=task["task_id"],
                split=task["split"],
                condition_kind=task["condition_kind"],
                seed=task["seed"],
                duration_s=task["duration_s"],
                duration_ticks=task["duration_ticks"],
                evaluated_frames=task["evaluated_frames"],
                evaluated_last_time_s=(task["evaluated_frames"] - 1) / FPS,
                model=label,
                path=path,
                status="invalid",
                error=None,
                metrics={},
            )
            try:
                if not path:
                    raise ValueError("missing_model_output")
                if condition_error:
                    raise ValueError(condition_error)
                qpos, provenance = load_trajectory(path, task, kin, music_sha)
                row["provenance"] = provenance
                metrics, error = trajectory_metrics(
                    qpos, kin, music=music, ground_height=task["ground_height_m"]
                )
                row.update(metrics=metrics, error=error, status="invalid" if error else "ok")
                if not error:
                    clips.append((label, path, qpos))
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}:{exc}"
            task_rows.append(row)
        complete = all(r["status"] == "ok" for r in task_rows)
        for row in task_rows:
            row["paired_complete"] = complete
        rows.extend(task_rows)
        pairing.append(
            dict(
                task_id=task["task_id"],
                complete=complete,
                invalid_models={r["model"]: r["error"] for r in task_rows if r["status"] != "ok"},
            )
        )
        blind_status.append(
            dict(
                task_id=task["task_id"],
                paired_complete=complete,
                expected_model_count=len(labels),
                available_clips=len(clips),
                missing_model_count=len(labels) - len(clips),
            )
        )
        rng.shuffle(clips)
        music_public = None
        task_code = f"task_{len(pairing):05d}"
        if music is not None:
            music_public = f"blind_assets/{task_code}_edge35.npy"
            np.save(output / music_public, music.numpy(), allow_pickle=False)
        for label, path, qpos in clips:
            clip_id = f"clip_{rng.getrandbits(96):024x}"
            relative = f"blind_assets/{clip_id}.npz"
            np.savez_compressed(
                output / relative,
                qpos=qpos.numpy(),
                joint_names=np.asarray(kin.joint_order),
                fps=FPS,
            )
            item = dict(
                task_id=task["task_id"],
                clip_id=clip_id,
                trajectory=relative,
                split=task["split"],
                condition_kind=task["condition_kind"],
                seed=task["seed"],
                duration_s=task["duration_s"],
                evaluated_last_time_s=(len(qpos) - 1) / FPS,
                text=task.get("text", ""),
                music_features=music_public,
                paired_complete=complete,
                missing_model_count=len(labels) - len(clips),
                **{name: None for name in _SCORE_FIELDS},
            )
            blind.append(item)
            reveal.append(
                dict(
                    task_id=task["task_id"],
                    clip_id=clip_id,
                    model=label,
                    source_path=path,
                    source_sha256=sha256_file(path),
                )
            )
    report = dict(
        schema=SCHEMA,
        manifest=str(manifest),
        manifest_sha256=sha256_file(manifest),
        kinematics_sha256=kin.kinematics_sha256,
        model_labels=labels,
        pairing=pairing,
        results=rows,
        all_valid_per_model=_aggregate(rows, labels, False),
        paired_per_model=_aggregate(rows, labels, True),
        evidence="kinematic_generation_only_no_physics_or_semantic_success",
        missing_outputs_are_failures=True,
        selection="independent_of_raw_reconstruction_best_no_automatic_winner",
        metric_contract=dict(
            fps=FPS,
            window="30Hz_samples_at_or_before_rounded_task_end_no_tail",
            joint_order=list(kin.joint_order),
            aggregation="equal_task_mean_with_per_metric_counts_split_and_modality_separate",
            foot_contact="height_le_ground_plus_0.025m_only_no_measured_contact_force",
            foot_sliding="kinematic_proxy_only_no_velocity_gate",
            motion_beats="existing_mean_absolute_joint_speed_local_minima",
            beat_matching="one_to_one_sorted_within_0.06s",
            absent_music_beats="null_not_perfect_score",
            semantic_scores="blank_manual_review_only",
            metadata="absent_generation_identity_is_declared_not_verified",
        ),
    )
    _json(output / "metrics.json", report)
    _json(
        output / "blind_tasks.json",
        dict(
            schema=SCHEMA,
            tasks=blind,
            task_status=blind_status,
            instructions="仅分享blind_tasks与blind_assets；metrics和reveal_key含模型身份。语义评分留空待人工，文本节拍不适用。",
        ),
    )
    _json(output / "reveal_key.json", dict(schema=SCHEMA, blind_seed=blind_seed, clips=reveal))
    fields = list(blind[0]) if blind else ["task_id", "clip_id", "trajectory", *_SCORE_FIELDS]
    with (output / "blind_tasks.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(blind)
    return report
