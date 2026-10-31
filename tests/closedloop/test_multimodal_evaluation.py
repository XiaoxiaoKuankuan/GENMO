"""多模态离线质量评价的数值、配对完整性和人工盲评边界回归。

测试使用真实BUMI FK资产和合成的标准qpos28，验证同时间窗、具名重排、独立元数据
核对、一对一拍点匹配及无接触真值的高度门限脚滑proxy；不把合成站姿当模型质量。
另经当前真实联合Actor生成函数构造reference.pt，检查其合同可被新评价入口直接消费。
缺失或坏文件不能从报告中消失，短窗没有足够差分样本必须返回空值，盲评包禁止泄露
模型名、原文件路径和自动语义分；CLI不完整配对必须返回非零状态且保留完整报告。
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gem.robots.bumi.kinematics import BumiKinematics
from gem.robots.bumi.multimodal_evaluation import (
    beat_metrics,
    evaluate_manifest,
    trajectory_metrics,
)

pytest_plugins = ["tests.closedloop.test_stage1_multimodal_actor"]

KIN = (
    Path(__file__).resolve().parents[2]
    / "configs/bumi/bumi_kinematics_robot_retargeter_fe934_v1.json"
)


@pytest.fixture
def kin():
    return BumiKinematics(KIN)


def _npz(path, qpos, kin, **extra):
    np.savez_compressed(
        path, qpos=qpos.numpy(), joint_names=np.asarray(kin.joint_order), fps=30, **extra
    )


def _task(task_id="wave", **extra):
    return dict(
        task_id=task_id,
        split="val",
        condition_kind="text",
        text="wave the right hand",
        seed=7,
        duration_s=1.0,
        models={},
        **extra,
    )


def _manifest(tmp_path, tasks, **extra):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps(dict(tasks=tasks, **extra)), encoding="utf-8")
    return path


def test_beats_one_to_one_bidirectional_and_absence():
    values = beat_metrics([10, 11], [10])
    assert values["matched_beat_count"] == 1
    assert values["beat_precision"] == 1 and values["beat_recall"] == 0.5
    assert values["beat_f1"] == pytest.approx(2 / 3)
    assert values["music_to_motion_distance_s"] == pytest.approx(1 / 60)
    assert values["motion_to_music_distance_s"] == 0
    assert beat_metrics([], [])["beat_f1"] is None
    empty = beat_metrics([10], [])
    assert empty["beat_f1"] == 0 and empty["bidirectional_beat_distance_s"] is None


def test_kinematic_units_height_gate_and_short_window(kin):
    qpos = kin.make_standing_qpos().repeat(31, 1)
    t = torch.arange(31) / 30
    qpos[:, 0] += t * 0.3
    metrics, error = trajectory_metrics(qpos, kin)
    assert error is None
    assert metrics["height_gated_proxy_foot_sliding_mean_mps"] == pytest.approx(0.3, abs=1e-6)
    assert metrics["joint_activity_mean_abs_velocity_radps"] == 0
    assert metrics["beat_f1"] is None and metrics["beat_status"] == "not_applicable_text"
    cubic = kin.make_standing_qpos().repeat(31, 1)
    cubic[:, 0] = t**3
    measured, _ = trajectory_metrics(cubic, kin)
    assert measured["root_jerk_mean_mps3"] == pytest.approx(6, abs=0.002)
    short, _ = trajectory_metrics(qpos[:1], kin)
    assert short["joint_jerk_p95_radps3"] is None and short["root_jerk_max_mps3"] is None
    assert short["height_gated_proxy_foot_sliding_mean_mps"] is None
    assert "foot_sliding_mean_mps" not in short
    high = qpos.clone()
    high[:, 2] += 10
    unsupported, _ = trajectory_metrics(high, kin)
    assert unsupported["height_gated_foot_intervals"] == 0
    assert unsupported["height_gated_proxy_foot_sliding_mean_mps"] is None


def test_paired_tail_crop_joint_order_and_blind_package(tmp_path, kin):
    qpos = kin.make_standing_qpos().repeat(40, 1)
    qpos[31:, 7:] = 1e6  # halo不纳入一秒任务评价
    _npz(tmp_path / "SECRET_shared.npz", qpos, kin, seed=7, execution_duration_s=1.0)
    reverse = qpos.clone()
    reverse[:, 7:] = reverse[:, 7:].flip(-1)
    np.savez_compressed(
        tmp_path / "SECRET_baseline.npz",
        qpos=reverse.numpy(),
        fps=30,
        joint_names=np.asarray(tuple(reversed(kin.joint_order))),
    )
    task = _task()
    task["models"] = {
        "SECRET_joint": "SECRET_shared.npz",
        "SECRET_independent": "SECRET_baseline.npz",
    }
    path = _manifest(tmp_path, [task])
    before = {p.name: p.read_bytes() for p in tmp_path.glob("*.npz")}
    result = evaluate_manifest(path, KIN, tmp_path / "out")
    assert result["pairing"][0]["complete"]
    for row in result["results"]:
        assert row["evaluated_frames"] == 31 and row["metrics"]["joint_limit_violation_rate"] == 0
        assert row["provenance"]["discarded_tail_frames"] == 9
    public = (tmp_path / "out/blind_tasks.json").read_text()
    assert "SECRET" not in public and str(tmp_path) not in public
    blind = json.loads(public)["tasks"]
    assert len(blind) == 2 and all(item["instruction_match"] is None for item in blind)
    assert "SECRET" not in (tmp_path / "out/blind_tasks.csv").read_text()
    assert "SECRET_joint" in (tmp_path / "out/reveal_key.json").read_text()
    for item in blind:
        with np.load(tmp_path / "out" / item["trajectory"], allow_pickle=False) as asset:
            assert set(asset.files) == {"qpos", "joint_names", "fps"}
            assert asset["qpos"].shape == (31, 28)
    assert before == {p.name: p.read_bytes() for p in tmp_path.glob("*.npz")}
    with pytest.raises(FileExistsError):
        evaluate_manifest(path, KIN, tmp_path / "out")


def test_missing_corrupt_nonfinite_and_mismatched_seed_stay_visible(tmp_path, kin):
    task = _task()
    qpos = kin.make_standing_qpos().repeat(31, 1)
    _npz(tmp_path / "good.npz", qpos, kin)
    _npz(tmp_path / "wrong.npz", qpos, kin, seed=9)
    qpos[10, 7] = float("nan")
    _npz(tmp_path / "nonfinite.npz", qpos, kin)
    (tmp_path / "broken.pt").write_text("not a checkpoint")
    task["models"] = dict(
        good="good.npz",
        absent=None,
        corrupt="broken.pt",
        nonfinite="nonfinite.npz",
        wrong="wrong.npz",
    )
    result = evaluate_manifest(_manifest(tmp_path, [task]), KIN, tmp_path / "out")
    assert len(result["results"]) == 5
    assert sum(row["status"] == "ok" for row in result["results"]) == 1
    assert not result["pairing"][0]["complete"]
    assert len(result["pairing"][0]["invalid_models"]) == 4
    assert result["paired_per_model"]["good"]["val/text"]["total"] == 0
    nf = next(row for row in result["results"] if row["model"] == "nonfinite")
    assert nf["metrics"]["finite_fraction"] < 1 and nf["error"] == "nonfinite_qpos"


def test_global_models_missing_and_bad_music_not_silently_dropped(tmp_path, kin):
    qpos = kin.make_standing_qpos().repeat(31, 1)
    _npz(tmp_path / "motion.npz", qpos, kin)
    task = _task()
    task["models"] = dict(one="motion.npz")
    path = _manifest(tmp_path, [task], model_labels=["one", "two"])
    from tools.evaluate_bumi_multimodal_generation import main

    assert (
        main(["--manifest", str(path), "--kinematics", str(KIN), "--output", str(tmp_path / "out")])
        == 2
    )
    report = json.loads((tmp_path / "out/metrics.json").read_text())
    assert report["results"][1]["error"].endswith("missing_model_output")
    task.pop("text")
    task.update(condition_kind="music", music_features="bad.pt")
    torch.save(torch.zeros(35, 31), tmp_path / "bad.pt")
    report = evaluate_manifest(_manifest(tmp_path, [task]), KIN, tmp_path / "bad_out")
    assert report["results"][0]["status"] == "invalid"
    assert "condition:" in report["results"][0]["error"]


def test_actual_shared_actor_text_reference_contract(tmp_path, joint_actor):
    from gem.runtime.multimodal_stage1 import generate_sequence

    actor, batch = joint_actor
    result = generate_sequence(
        actor,
        total_frames=40,
        steps=2,
        seed=7,
        text_features=batch["text_features"][1:2],
        text_attention_mask=batch["text_attention_mask"][1:2],
    )
    task = _task()
    result["generation_contract"] = dict(
        seed=7,
        duration_ticks=600,
        condition_kind="text",
        text_sha256=hashlib.sha256(task["text"].encode()).hexdigest(),
    )
    torch.save(result, tmp_path / "reference.pt")
    task["models"] = dict(shared="reference.pt")
    report = evaluate_manifest(_manifest(tmp_path, [task]), KIN, tmp_path / "out")
    row = report["results"][0]
    assert row["status"] == "ok" and row["provenance"]["unverified_manifest_fields"] == []
    assert row["metrics"]["beat_status"] == "not_applicable_text"
    task["seed"] = 8
    wrong = evaluate_manifest(_manifest(tmp_path, [task]), KIN, tmp_path / "wrong")
    assert "seed" in wrong["results"][0]["error"]


def test_music_metrics_use_edge_column_and_text_beats_remain_na(tmp_path, kin):
    qpos = kin.make_standing_qpos().repeat(31, 1)
    _npz(tmp_path / "standing.npz", qpos, kin)
    music = torch.zeros(30, 35)
    music[[5, 20], 34] = 1
    torch.save(music, tmp_path / "edge.pt")
    task = _task()
    task.pop("text")
    task.update(condition_kind="music", music_features="edge.pt", models=dict(model="standing.npz"))
    report = evaluate_manifest(_manifest(tmp_path, [task]), KIN, tmp_path / "out")
    metrics = report["results"][0]["metrics"]
    assert metrics["music_beat_count"] == 2 and metrics["motion_beat_count"] == 0
    assert metrics["beat_f1"] == 0 and metrics["beat_status"] == "no_motion_beats"
    assert metrics["music_to_motion_distance_s"] is None
    assert metrics["beat_evaluated_frames"] == 30
    assert report["results"][0]["evaluated_frames"] == 31


def test_whole_runtime_reference_contract_enters_evaluation(tmp_path, joint_actor, monkeypatch):
    from gem.runtime import multimodal_stage1

    actor, _ = joint_actor
    monkeypatch.setattr(multimodal_stage1, "load_actor", lambda *args: (actor, {}))
    music = torch.zeros(40, 35)
    music[[5, 20], 34] = 1
    torch.save(music, tmp_path / "edge.pt")
    config = dict(sample_steps=2)
    for name in ("checkpoint", "stats", "kinematics"):
        path = tmp_path / name
        path.write_text(name)
        config[name] = str(path)
    config["kinematics"] = str(KIN)
    task = dict(
        task_id="music",
        split="test",
        condition_kind="music",
        duration_s=1.0,
        seed=7,
        music_features=str(tmp_path / "edge.pt"),
        models={"shared": "reference.pt"},
    )
    generated = multimodal_stage1.generate_stage1_reference(
        config, task, device="cpu", required_end_tick=732
    )
    torch.save(generated, tmp_path / "reference.pt")
    report = evaluate_manifest(_manifest(tmp_path, [task]), KIN, tmp_path / "out")
    row = report["results"][0]
    assert row["status"] == "ok" and row["provenance"]["unverified_manifest_fields"] == []
    assert row["metrics"]["music_beat_count"] == 2
    assert row["provenance"]["discarded_tail_frames"] > 0
