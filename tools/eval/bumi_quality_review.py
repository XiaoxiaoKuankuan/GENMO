"""分析已有BUMI UMR全量质量报告，并确定性挑选可追溯的视频复核样本。

本模块由既有render_bumi_motion.py调用，不重新筛选或改写原始报告。逐文件流式读取
JSONL，核对完成状态、汇总SHA、记录数、状态、帧数与源字节数，区分质量淘汰与训练
长度限制。每种原因按动作去重计数，不把多个碰撞对或左右脚的计数直接相加。

高质量样本从120..300帧的活动PASS中按移动、转向、低姿态、关节活动等指标分组；
低质量样本从60..300帧REJECT中覆盖脚滑、碰撞、穿地和突变。选择按来源目录均衡，
使用原始来源ID排除重复，保留完整动作，不宣称这些目的性样本代表全库分布。
仅保存少量候选池，避免把56万条完整指标加载入内存。实际渲染仍须校验动作SHA。
"""

from __future__ import annotations

import hashlib
import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reason_family(code):
    for prefix, family in (
        ("FOOT_SLIDE", "foot_slide"),
        ("SELF_COLLISION", "self_collision"),
        ("FOOT_PENETRATION", "ground_penetration"),
        ("ROOT_HEIGHT", "root_height"),
        ("LONG_AIRBORNE", "airborne"),
        ("SOURCE_JOINT_LIMIT", "joint_limits"),
    ):
        if code.startswith(prefix):
            return family
    if "VELOCITY" in code or "ACCELERATION" in code or "JERK" in code:
        return "motion_discontinuity"
    return "other"


def review_metrics(row):
    metrics = row["metrics"]
    feet = metrics["feet"]
    dynamics = metrics["dynamics"]
    return dict(
        foot_slide_p95_m_s=max(feet[s]["slide"]["p95"] for s in ("left", "right")),
        foot_slide_max_m_s=max(feet[s]["slide"]["max"] for s in ("left", "right")),
        penetration_m=max(0.0, -min(feet[s]["min_surface_height_m"] for s in ("left", "right"))),
        collision_extra_m=max(
            (m["extra_depth"]["max"] for m in metrics["collisions"].values() if not m["allowed"]),
            default=0.0,
        ),
        root_travel_m=metrics["root_travel_m"],
        root_height_min_m=metrics["root_height_min"],
        root_tilt_p95_deg=metrics["floor_style"]["root_tilt_p95_degrees"],
        joint_speed_p95_rad_s=dynamics["joint_velocity_l2"]["p95"],
        root_speed_max_m_s=dynamics["root_linear_velocity"]["max"],
        root_angular_speed_max_rad_s=dynamics["root_angular_velocity"]["max"],
        repeated_pose_fraction=metrics["repeated_pose_pair_fraction"],
    )


def high_category(metrics):
    if metrics["root_height_min_m"] < 0.30 or metrics["root_tilt_p95_deg"] > 45:
        return "low_posture"
    if metrics["root_travel_m"] >= 0.7:
        return "travel"
    if metrics["root_angular_speed_max_rad_s"] >= 1:
        return "turning"
    if metrics["joint_speed_p95_rad_s"] >= 8:
        return "active_limbs"
    return "limb_motion"


def candidate_scores(row):
    """低分优先；高组避免静止投机，低组按触发REJECT的原因分层。"""
    m = review_metrics(row)
    if row["status"] == "PASS" and row["training_eligible"]:
        if not 120 <= row["frames"] <= 300 or m["repeated_pose_fraction"] > 0.1:
            return []
        if m["joint_speed_p95_rad_s"] < 1 or m["root_travel_m"] > 8:
            return []
        score = (
            m["foot_slide_p95_m_s"] / 0.25
            + m["penetration_m"] / 0.005
            + m["collision_extra_m"] / 0.01
            + row["metrics"]["dynamics"]["joint_jerk_l2"]["p95"] / 60000
        )
        return [("high_quality", high_category(m), score)]
    if row["status"] != "REJECT" or not 60 <= row["frames"] <= 300:
        return []
    families = {
        reason_family(code) for code, level in row["reason_statuses"].items() if level == "REJECT"
    }
    scores = []
    if "foot_slide" in families:
        scores.append(("low_quality", "foot_slide", -m["foot_slide_p95_m_s"]))
    if "self_collision" in families:
        scores.append(("low_quality", "self_collision", -m["collision_extra_m"]))
    if families & {"ground_penetration", "root_height"}:
        scores.append(
            ("low_quality", "ground_penetration", -max(m["penetration_m"], -m["root_height_min_m"]))
        )
    if "motion_discontinuity" in families:
        value = max(v["max"] / v["threshold"] for v in row["metrics"]["dynamics"].values())
        scores.append(("low_quality", "motion_discontinuity", -value))
    return scores


class CandidatePool:
    """每组/类型/来源目录只保留固定数量候选；目录均衡在最终挑选时执行。"""

    def __init__(self, capacity=30):
        self.capacity = capacity
        self.pools = defaultdict(list)

    def add(self, row):
        tie = hashlib.sha256(row["relative_path"].encode()).hexdigest()
        for group, category, score in candidate_scores(row):
            heap = self.pools[group, category, row["folder"]]
            item = (-score, tie, row)
            if len(heap) < self.capacity:
                heapq.heappush(heap, item)
            elif item[:2] > heap[0][:2]:
                heapq.heapreplace(heap, item)

    def choose(self, group, count):
        categories = (
            ["travel", "turning", "low_posture", "active_limbs", "limb_motion"]
            if group == "high_quality"
            else ["foot_slide", "self_collision", "ground_penetration", "motion_discontinuity"]
        )
        weights = [1] * 5 if group == "high_quality" else [10, 10, 6, 4]
        quotas = [count * w // sum(weights) for w in weights]
        for i in range(count - sum(quotas)):
            quotas[i % len(quotas)] += 1
        pools = defaultdict(list)
        for (g, category, _), heap in self.pools.items():
            if g == group:
                pools[category].extend((-score, tie, row) for score, tie, row in heap)
        chosen, used, folders = [], set(), Counter()

        def take(category):
            candidates = [
                item for item in pools[category] if item[2]["canonical_source_id"] not in used
            ]
            if not candidates:
                return False
            score, _, row = min(
                candidates, key=lambda item: (folders[item[2]["folder"]], item[0], item[1])
            )
            chosen.append(
                dict(
                    row,
                    selection_category=category,
                    selection_score=score,
                    review_metrics=review_metrics(row),
                )
            )
            used.add(row["canonical_source_id"])
            folders[row["folder"]] += 1
            return True

        for category, quota in zip(categories, quotas):
            for _ in range(quota):
                if not take(category):
                    break
        while len(chosen) < count:
            if not any(take(category) for category in categories):
                raise ValueError(f"{group}可用的不同来源动作不足{count}条")
        return chosen


def analyze_report(root, count=30):
    root = Path(root).resolve(strict=True)
    run = json.loads((root / "run.json").read_text())
    summary = json.loads((root / "quality_summary.json").read_text())
    if run["state"] != "complete" or run["partial_scan"]:
        raise ValueError("只接受已完成的全量报告")
    if sha256(root / "quality_summary.json") != run["summary_sha256"]:
        raise ValueError("质量汇总SHA不符")
    if summary["run_fingerprint"] != run["fingerprint"]:
        raise ValueError("报告运行身份不符")
    if sha256(root / "train_candidates.jsonl") != summary["candidate_manifest_sha256"]:
        raise ValueError("训练候选清单SHA不符")
    statuses, frames, lengths, eligible_folders, source_keys = (Counter() for _ in range(5))
    cross = defaultdict(Counter)
    families, reject_families = defaultdict(Counter), Counter()
    intersections, folders, sources = Counter(), defaultdict(Counter), Counter()
    total_bytes, eligible, sequence_key_mismatches = 0, 0, 0
    pool, hashes = CandidatePool(), {}
    for path in sorted((root / "reports").glob("*.jsonl")):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for line in stream:
                digest.update(line)
                row = json.loads(line)
                status, n = row["status"], row["frames"]
                statuses[status] += 1
                frames[status] += n
                total_bytes += row.get("source_bytes", 0)
                folders[row["folder"]][status] += 1
                sources[row.get("source_motion_id", "unknown").split("/", 1)[0]] += 1
                source_keys[row.get("source_sequence_key", "missing")] += 1
                if (
                    row.get("human_path")
                    and row.get("source_sequence_key") != Path(row["human_path"]).stem
                ):
                    sequence_key_mismatches += 1
                band = (
                    "under_60"
                    if n < 60
                    else "60_119"
                    if n < 120
                    else "120_300"
                    if n <= 300
                    else "over_300"
                )
                lengths[band] += 1
                cross[status][band] += 1
                if row["training_eligible"]:
                    if status != "PASS" or not 60 <= n <= 300:
                        raise ValueError("训练资格与状态/帧数不一致")
                    eligible += 1
                    eligible_folders[row["folder"]] += 1
                groups = {reason_family(c) for c in row["reason_codes"]}
                families[status].update(groups)
                rejected = {
                    reason_family(c)
                    for c, level in row.get("reason_statuses", {}).items()
                    if level == "REJECT"
                }
                reject_families.update(rejected)
                if status == "REJECT":
                    intersections[" + ".join(sorted(rejected))] += 1
                if status in {"PASS", "REJECT"}:
                    pool.add(row)
        hashes[path.name] = digest.hexdigest()
        print(
            json.dumps(dict(stage="analyze", file=path.name, processed=sum(statuses.values()))),
            flush=True,
        )
    expected = {k: v for k, v in summary["status_counts"].items() if k != "TRAIN_ELIGIBLE"}
    expected_frames = {
        k: v for k, v in summary["frames_by_status"].items() if k != "TRAIN_ELIGIBLE"
    }
    if (
        dict(statuses) != expected
        or dict(frames) != expected_frames
        or total_bytes != summary["source_bytes"]
        or dict(folders) != summary["by_folder"]
        or sum(statuses.values()) != run["indexed_records"]
        or eligible != summary["status_counts"]["TRAIN_ELIGIBLE"]
    ):
        raise ValueError("逐条报告与汇总记录数/状态/帧数/来源大小不一致")
    analysis = dict(
        schema="genmo.bumi_quality_review.v1",
        report_root=str(root),
        run_fingerprint=run["fingerprint"],
        original_summary=summary,
        status_counts=dict(statuses),
        length_distribution=dict(lengths),
        status_by_length=dict(cross),
        eligible_by_folder=dict(eligible_folders),
        reason_families_by_status=dict(families),
        reject_trigger_families=dict(reject_families),
        rejection_intersections=dict(intersections),
        source_namespaces=dict(sources),
        report_source_sequence_keys_top=source_keys.most_common(5),
        source_sequence_key_mismatches=sequence_key_mismatches,
        report_file_sha256=hashes,
        selection_policy="目的性复核样本；高组分层选活动PASS，低组按REJECT原因分层；完整序列，非随机总体估计",
        groups={g: pool.choose(g, count) for g in ("high_quality", "low_quality")},
    )
    return analysis, run


def write_analysis_markdown(analysis, output):
    """写可随两段视频移交的中文分析、口径与逐动作时间索引。"""
    summary = analysis["original_summary"]
    total = summary["processed_records"]
    passed = summary["status_counts"]["PASS"]
    eligible = summary["status_counts"]["TRAIN_ELIGIBLE"]
    lines = [
        "# MotionMillion UMR 动作质量分析与60条视频复核",
        "",
        f"报告身份：`{analysis['run_fingerprint']}`。全量逐条报告的数量、状态、帧数、字节数均与原汇总一致。",
        "",
        "## 全量结果",
        "",
        "| 项目 | 动作数 | 占输入比例 |",
        "|---|---:|---:|",
    ]
    for label, count in [
        ("原始输入", total),
        ("质量PASS", passed),
        ("REVIEW", summary["status_counts"].get("REVIEW", 0)),
        ("REJECT", summary["status_counts"].get("REJECT", 0)),
        ("INVALID", summary["status_counts"].get("INVALID", 0)),
        ("ERROR", summary["status_counts"].get("ERROR", 0)),
        ("完整60–300帧PASS训练候选", eligible),
    ]:
        lines.append(f"| {label} | {count:,} | {count / total:.2%} |")
    length = analysis["status_by_length"]["PASS"]
    hours = sum(v for k, v in summary["hours_by_status"].items() if k != "TRAIN_ELIGIBLE")
    lines += [
        "",
        f"质量PASS中有{passed - eligible:,}条因长度不适用而未进入训练候选：不足60帧{length.get('under_60', 0):,}条，超过300帧{length.get('over_300', 0):,}条。长度不适用不等同于动作质量差。",
        f"输入共{hours:.2f}小时，机器人NPZ {summary['source_bytes'] / 1e9:.3f} GB；训练候选{summary['hours_by_status']['TRAIN_ELIGIBLE']:.2f}小时，对应原NPZ {summary['eligible_source_bytes'] / 1e9:.3f} GB。候选仍需文本、T5、split绑定后才能称为训练release。",
        "",
        "## 质量问题",
        "",
        "以下按动作计数，合并左右脚和多个碰撞对；原因可能重叠，不能相加作为淘汰总数。",
        "",
        "| REJECT触发原因 | 动作数 | 占REJECT比例 |",
        "|---|---:|---:|",
    ]
    labels = dict(
        foot_slide="脚滑",
        self_collision="自碰撞",
        ground_penetration="脚部穿地",
        root_height="根高度越界",
        motion_discontinuity="速度/旋转/关节突变",
        airborne="持续悬空",
    )
    rejected = summary["status_counts"].get("REJECT", 0)
    for family, count in sorted(
        analysis["reject_trigger_families"].items(), key=lambda item: -item[1]
    ):
        lines.append(f"| {labels.get(family, family)} | {count:,} | {count / rejected:.2%} |")
    lines += [
        "",
        "## 来源目录差异",
        "",
        "| 目录 | 总数 | PASS比例 | 训练候选 | 候选比例 |",
        "|---|---:|---:|---:|---:|",
    ]
    for folder, values in summary["by_folder"].items():
        n = sum(values.values())
        kept = analysis["eligible_by_folder"].get(folder, 0)
        lines.append(
            f"| {folder} | {n:,} | {values.get('PASS', 0) / n:.2%} | {kept:,} | {kept / n:.2%} |"
        )
    lines += [
        "",
        "## 解释与边界",
        "",
        f"- 来源命名空间统计：`{json.dumps(analysis['source_namespaces'], ensure_ascii=False)}`。结论对应这批实际转换数据，不能泛化为MotionMillion全部来源。",
        "- 筛选是数值与运动学门禁；凸包碰撞近似、接触候选脚滑与悬空规则仍需结合视频复核。姿态低不直接等于质量差，完整动作没有再次贴地或平滑。",
        "- 高组从有活动的PASS中按移动、转向、低姿态和关节活动分组，低组覆盖不同REJECT原因；两组均平衡目录并去重。视频样本用于看清差异，不是随机抽样估计全库比例。",
        "- 视频为双视角、原始30Hz、完整动作顺序拼接。视角B地面半透明，便于观察穿地；相机变化不改变qpos。红框只标识当前帧命中原报告异常区间。",
    ]
    if analysis["source_sequence_key_mismatches"]:
        lines.append(
            f"- 旧报告{analysis['source_sequence_key_mismatches']:,}条的辅助source_sequence_key字段与人体文件名不一致，源于旧适配器循环变量覆盖。原始source_motion_id、文件SHA和质量计算不受该字段影响；本次视频另存verified_source_sequence_key并逐条核验原NPZ。原报告未改写，后续输出的代码已修正。"
        )
    for group, video in analysis["videos"].items():
        title = "高质量30例" if group == "high_quality" else "低质量30例"
        lines += [
            "",
            f"## {title}",
            "",
            f"[打开视频]({video['path']})：{video['duration_seconds']:.2f}秒，{video['frames']:,}帧，SHA256 `{video['sha256']}`。",
            "",
            "| 序号 | 起点 | 时长 | 选样类型 | 原动作ID |",
            "|---:|---:|---:|---|---|",
        ]
        for row in video["chapters"]:
            sec = row["start_seconds"]
            start = f"{int(sec) // 60:02d}:{sec % 60:05.2f}"
            lines.append(
                f"| {row['index']} | {start} | {row['frames'] / 30:.2f}s | {row['category']} | `{row['source_motion_id']}` |"
            )
    Path(output).write_text("\n".join(lines) + "\n")
