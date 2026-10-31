"""音乐与文本 Adapter 任务的条件身份、时长及评价聚合。

任务时间使用 600 Hz 整数 tick，显式时长独立于音频是否存在；文本不创建伪音频
文件。条件摘要仅绑定影响生成结果的输入，不把源码版本当恢复门禁。旧音乐任务
保留音频长度默认值，新联合任务必须声明时长与源动作分组；相同源动作的不同
描述不能跨训练与验证。评价先分别统计各模态，再等权聚合，节拍只统计音乐任务。
"""

import hashlib
import json
import math
from pathlib import Path


def condition_kind(task):
    return task.get("condition_kind", "music")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_end_tick(task, audio=None):
    start = int(task.get("task_start_tick", 0))
    if "duration_ticks" in task:
        duration = task["duration_ticks"]
        if isinstance(duration, bool) or int(duration) != duration:
            raise ValueError("duration_ticks必须为整数600Hz tick")
        duration = int(duration)
    elif "duration_s" in task:
        seconds = float(task["duration_s"])
        if not math.isfinite(seconds):
            raise ValueError("任务时长必须有限")
        duration = round(seconds*600)
    elif audio is not None and "condition_kind" not in task:
        return audio.end_tick
    else:
        raise ValueError("联合任务必须声明duration_s或duration_ticks")
    if "duration_s" in task and round(float(task["duration_s"])*600) != duration:
        raise ValueError("duration_s与duration_ticks不一致")
    if duration < 12 or start != 0:
        raise ValueError("固定缓存任务必须从0开始且至少包含一个50Hz控制步")
    return start+duration


def condition_contract(task):
    kind = condition_kind(task)
    if kind not in ("music", "text"):
        raise ValueError("当前联合训练仅支持music或text条件")
    values = dict(condition_kind=kind)
    if kind == "music":
        values.update(music_sha256=file_sha256(task["music_features"]),
                      music_start_tick=int(task.get("music_start_tick", 0)))
    else:
        caption = task.get("text")
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError("文本任务需要非空text")
        if task.get("music_features"):
            raise ValueError("text任务不得同时声明音乐；首版不训练未配对融合")
        values["text_sha256"] = hashlib.sha256(caption.encode("utf-8")).hexdigest()
    if "condition_kind" in task:
        values["task_end_tick"] = task_end_tick(task)
    values["condition_sha256"] = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
    return values


def modality_metrics(episodes):
    """缺测保持None，文本节拍保持不适用；任务数量不会改变两模态的相对权重。"""
    groups = {}
    for kind in ("music", "text"):
        rows = [r for r in episodes if r.get("condition_kind", "music") == kind]
        if not rows:
            continue
        def mean(key):
            values = [r.get(key) for r in rows if r.get(key) is not None]
            return sum(values)/len(values) if values else None
        beats = [r.get("music", {}).get("f1") for r in rows] if kind == "music" else []
        beats = [v for v in beats if v is not None]
        groups[kind] = dict(count=len(rows),
            completion_rate=sum(r.get("full_task_complete", r.get("full_music_complete", False)) for r in rows)/len(rows),
            mean_root_position_rmse_m=mean("root_position_rmse_m"),
            mean_full_horizon_tracking=mean("full_horizon_tracking"),
            mean_jerk_rms=mean("jerk_rms"),
            mean_contact_foot_slip_mps=mean("contact_foot_slip_mps"),
            mean_beat_f1=sum(beats)/len(beats) if beats else None)
    return groups
