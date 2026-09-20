"""BUMI质量报告分析与视频样本选择的回归测试。

使用明确合成的报告，验证质量状态与长度统计分离、错误SHA/汇总拒绝、去重与来源
均衡、拒绝原因按实际严重程度分组、高质量选择不被静止样本占据。所有报告只写入
pytest临时目录；视频像素/FPS/帧数另外通过真实MuJoCo渲染和解码验收。
"""

import copy
import json
from collections import Counter

import pytest

from tools.eval.bumi_quality_review import (
    CandidatePool,
    analyze_report,
    candidate_scores,
    sha256,
    write_analysis_markdown,
)


def example(number=0, status="PASS", folder="folder0"):
    foot = dict(min_surface_height_m=0, slide=dict(p95=0.01, max=0.02))
    signal = dict(p95=2.0, max=3.0, threshold=30.0)
    return dict(
        relative_path=f"{folder}/{number}.npz",
        canonical_source_id=f"source/{number}",
        source_motion_id=f"Mirror_MotionGV/{folder}/{number}",
        source_sequence_key=str(number),
        folder=folder,
        status=status,
        frames=180,
        source_bytes=100,
        training_eligible=status == "PASS",
        reason_codes=[],
        reason_statuses={},
        metrics=dict(
            feet=dict(left=copy.deepcopy(foot), right=copy.deepcopy(foot)),
            collisions={},
            root_travel_m=1.0,
            root_height_min=0.45,
            floor_style=dict(root_tilt_p95_degrees=5),
            repeated_pose_pair_fraction=0,
            dynamics={
                k: copy.deepcopy(signal)
                for k in (
                    "joint_velocity_l2",
                    "joint_jerk_l2",
                    "root_linear_velocity",
                    "root_angular_velocity",
                )
            },
        ),
    )


def test_quality_selection_is_active_distinct_and_balanced():
    pool = CandidatePool()
    for i in range(90):
        pool.add(example(i, folder=f"folder{i % 10}"))
    selected = pool.choose("high_quality", 30)
    assert len({r["canonical_source_id"] for r in selected}) == 30
    assert set(Counter(r["folder"] for r in selected).values()) == {3}
    static = example(100)
    static["metrics"]["dynamics"]["joint_velocity_l2"]["p95"] = 0
    assert candidate_scores(static) == []


def test_reject_categories_use_reject_reason_not_secondary_review():
    row = example(status="REJECT")
    row["reason_statuses"] = {
        "FOOT_SLIDE_left_REVIEW": "REVIEW",
        "SELF_COLLISION_arm / leg": "REJECT",
    }
    row["reason_codes"] = list(row["reason_statuses"])
    assert [c for _, c, _ in candidate_scores(row)] == ["self_collision"]
    row["reason_statuses"] = {"ROOT_TILT_SUSTAINED": "REJECT"}
    assert [c for _, c, _ in candidate_scores(row)] == ["root_tilt"]


def report_fixture(tmp_path):
    rows = [example(), example(1, status="REJECT")]
    rows[1]["reason_codes"] = ["FOOT_SLIDE_left_REJECT"]
    rows[1]["reason_statuses"] = {"FOOT_SLIDE_left_REJECT": "REJECT"}
    (tmp_path / "reports").mkdir()
    path = tmp_path / "reports/folder0.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    candidates = tmp_path / "train_candidates.jsonl"
    candidates.write_text(json.dumps(rows[0]) + "\n")
    summary = dict(
        run_fingerprint="fixture",
        processed_records=2,
        hours_by_status={"PASS": 1 / 600, "REJECT": 1 / 600, "TRAIN_ELIGIBLE": 1 / 600},
        eligible_source_bytes=100,
        status_counts={"PASS": 1, "REJECT": 1, "TRAIN_ELIGIBLE": 1},
        frames_by_status={"PASS": 180, "REJECT": 180, "TRAIN_ELIGIBLE": 180},
        source_bytes=200,
        by_folder={"folder0": {"PASS": 1, "REJECT": 1}},
        candidate_manifest_sha256=sha256(candidates),
    )
    (tmp_path / "quality_summary.json").write_text(json.dumps(summary))
    run = dict(
        state="complete",
        partial_scan=False,
        fingerprint="fixture",
        indexed_records=2,
        summary_sha256=sha256(tmp_path / "quality_summary.json"),
    )
    (tmp_path / "run.json").write_text(json.dumps(run))
    return path


def test_report_analysis_and_tamper_detection(tmp_path):
    path = report_fixture(tmp_path)
    result, _ = analyze_report(tmp_path, count=1)
    assert result["status_by_length"] == {"PASS": {"120_300": 1}, "REJECT": {"120_300": 1}}
    assert result["reject_trigger_families"] == {"foot_slide": 1}
    assert all(len(group) == 1 for group in result["groups"].values())
    with path.open("a") as out:
        out.write(json.dumps(example(2)) + "\n")
    with pytest.raises(ValueError, match="逐条报告"):
        analyze_report(tmp_path, count=1)


def test_report_rejects_changed_summary(tmp_path):
    report_fixture(tmp_path)
    path = tmp_path / "quality_summary.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="SHA"):
        analyze_report(tmp_path, count=1)


def test_humanml_namespace_composition_and_report_title(tmp_path):
    path = report_fixture(tmp_path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for index, row in enumerate(rows):
        row.update(dataset="humanml3d", mirrored=bool(index))
        row["source_motion_id"] = "M000002__seg_1000_7000" if index else "000002"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    run_path = tmp_path / "run.json"
    run = json.loads(run_path.read_text())
    run["identity"] = {"paths": {"dataset": "humanml3d"}}
    run_path.write_text(json.dumps(run))
    analysis, _ = analyze_report(tmp_path, count=1)
    assert analysis["source_namespaces"] == {"humanml3d": 2}
    assert analysis["composition_by_status"] == {
        "PASS": {"original": 1, "full_source": 1},
        "REJECT": {"mirrored": 1, "subclip": 1},
    }
    analysis["videos"] = {}
    output = tmp_path / "analysis.md"
    write_analysis_markdown(analysis, output)
    assert output.read_text().startswith("# HumanML3D UMR 动作质量分析与2条视频复核")
    assert "MotionMillion" not in output.read_text()
