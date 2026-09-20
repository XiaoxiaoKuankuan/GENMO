"""UMR文本全量筛选及PASS构建的CPU回归测试。

使用真实fe934机器人XML/网格、标准运动学JSON和临时合成UMR/SMPL-X文件，验证原生
字段、名字重排、Y-up来源、时间线、数值异常、有效膝限位、脚滑/悬空、默认碰撞扣除、
有限队列多进程、断点续跑、源SHA变化和正式构建质量门禁。文本特征明确为测试替身，
不把这些测试当成真实生成质量或动力学验证。全部运行产物仅写pytest的tmp_path。
"""

import json
import os
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

from gem.datasets.pure_motion.bumi_text import BumiTextDataset, caption_hash
from gem.robots.bumi.kinematics import sha256_file
from tools.data.bumi.prepare_bumi_text import build
from tools.data.bumi.umr_text_preprocess import (
    DEFAULT_CONFIG,
    DEFAULT_KINEMATICS,
    InputContractError,
    QualityGate,
    evaluate_row,
    init_worker,
    load_umr,
    run_filter,
)
from tools.data.bumi.umr_text_quality import (
    POSTURE_CODES,
    QualityEngine,
    foot_diagnostics,
    load_rules,
)


@pytest.fixture(scope="module")
def engine():
    umr = Path(os.environ.get("GENMO_UMR_ROOT", "/home/weili/UMR"))
    if not (umr / "assets/bumi3/asset_manifest.json").is_file():
        pytest.skip("真实UMR BUMI资产不存在；设置GENMO_UMR_ROOT")
    return QualityEngine(
        load_rules(DEFAULT_CONFIG),
        umr / "assets/bumi3/mjcf/bumi3_retarget.xml",
        DEFAULT_KINEMATICS,
        umr / "robot_configs/humanoid_retarget_bumi3_example.json",
        umr / "humanoid_retarget_defaults_batch_bumi3.json",
    )


def grounded(engine, frames=60):
    qpos = np.tile(engine.kin.default_qpos.numpy(), (frames, 1))
    qpos[:, 7:] = np.clip(
        qpos[:, 7:], engine.config.joint_lower_limits, engine.config.joint_upper_limits
    )
    result = engine.evaluate(qpos)
    bottom = min(
        result["metrics"]["feet"][side]["min_surface_height_m"] for side in ("left", "right")
    )
    qpos[:, 2] -= bottom
    return qpos


@pytest.fixture
def bundle(tmp_path, engine):
    umr = engine.xml.parents[3]
    paths = dict(
        input_root=str(tmp_path / "output"),
        source_root=str(tmp_path / "human"),
        config=str(DEFAULT_CONFIG),
        robot_xml=str(engine.xml),
        kinematics=str(DEFAULT_KINEMATICS),
        asset_manifest=str(umr / "assets/bumi3/asset_manifest.json"),
        retarget_config=str(umr / "robot_configs/humanoid_retarget_bumi3_example.json"),
        batch_config=str(umr / "humanoid_retarget_defaults_batch_bumi3.json"),
    )
    base = grounded(engine)
    rows, results = [], []

    def make(frames=60, folder="folder0", reorder=False):
        index = len(rows)
        key = f"{index:06d}_smplx"
        source = Path(paths["source_root"]) / folder / (key + ".npz")
        source.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            source,
            pose_aa=np.zeros((frames, 66), np.float32),
            trans=np.zeros((frames, 3), np.float32),
            fps=np.array([30.0], np.float32),
            output_up="y",
            source_format="motionmillion_272",
            source_file=f"/test/motion_272rpr_unpacked/MotionGV/fixture/{index:06d}.npy",
        )
        path = Path(paths["input_root"]) / folder / "bumi3" / (key + "_bumi3.npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        qpos = np.tile(base[:1], (frames, 1))
        names = list(engine.kin.joint_order)
        if reorder:
            qpos[:, 7:] = qpos[:, 7:][:, ::-1]
            names.reverse()
        np.savez_compressed(
            path,
            qpos=qpos,
            fps=np.array([30.0], np.float32),
            frame_ids=np.arange(frames, dtype=np.int32),
            robot_xml=str(engine.xml),
            robot_name="bumi3",
            robot_joint_names=np.array(names, dtype=object),
            source_data=str(source),
            source_sequence_key=key,
            source_format="smplx_npz",
            smpl_scale=np.array([0.573], np.float32),
            ground_z=np.array([0.0], np.float32),
            zero_source_finger_pose=np.array([True]),
        )
        row = dict(
            relative_path=path.relative_to(paths["input_root"]).as_posix(),
            folder=folder,
            human_path=str(source),
            upstream_status="ok",
        )
        rows.append(row)
        results.append(dict(motion=str(source), out=str(path), status="ok"))
        for f in {r["folder"] for r in rows}:
            selected = [r for r in results if Path(r["out"]).parent.parent.name == f]
            (Path(paths["input_root"]) / f / "bumi3/batch_summary.json").write_text(
                json.dumps(dict(results=selected))
            )
        return row, path

    def args(**overrides):
        values = {k: Path(v) for k, v in paths.items()}
        values.update(
            output=tmp_path / "report",
            workers=1,
            folders=None,
            limit=None,
            expected_records=len(rows),
            resume=False,
        )
        values.update(overrides)
        return Namespace(**values)

    return paths, make, args, rows


def rewrite(path, **changes):
    with np.load(path, allow_pickle=True) as archive:
        arrays = {k: archive[k] for k in archive.files}
    arrays.update(changes)
    np.savez_compressed(path, **arrays)


def test_real_asset_limits_quaternion_sign_and_collision_baseline(engine):
    qpos = grounded(engine)
    qpos[::2, 3:7] *= -1
    decision = engine.evaluate(qpos)
    assert decision["status"] == "PASS"
    assert decision["metrics"]["dynamics"]["root_angular_velocity"]["max"] < 1e-6
    assert not any(x.startswith("SELF_COLLISION") for x in decision["reason_codes"])
    assert decision["metrics"]["reference_collision_depths"]
    qpos[:, engine.kin.joint_order.index("l_knee_pitch_joint") + 7] = 0
    bad = engine.evaluate(qpos)
    assert "SOURCE_JOINT_LIMIT" in bad["reason_codes"]
    assert bad["issue_intervals"]["SOURCE_JOINT_LIMIT"] == [[0, 60]]


def test_shortest_arc_root_jump_and_posture_diagnostic(engine):
    qpos = grounded(engine)
    qpos[30:, 0] += 2
    result = engine.evaluate(qpos)
    assert "ROOT_LINEAR_VELOCITY_SEVERE" in result["reason_codes"]
    assert result["issue_intervals"]["ROOT_LINEAR_VELOCITY_SEVERE"] == [[29, 31]]
    qpos = grounded(engine)
    qpos[:, 2] = 0.25
    qpos[:, 3:7] = [2**-0.5, 2**-0.5, 0, 0]
    result = engine.evaluate(qpos)
    assert not POSTURE_CODES.intersection(result["reason_codes"])
    assert POSTURE_CODES.intersection(result["diagnostic_reasons"])


def test_slide_is_not_hidden_by_contact_speed_gate():
    rules = load_rules(DEFAULT_CONFIG)
    heights = np.zeros((60, 2))
    centers = np.zeros((60, 2, 3))
    centers[:, :, 0] = np.arange(60)[:, None] / 30
    metrics, flags = foot_diagnostics(heights, centers, rules)
    assert metrics["left"]["support_edge_fraction"] == 1
    assert any(code == "FOOT_SLIDE_left_REJECT" for code, _, _ in flags)
    _, flags = foot_diagnostics(heights + 0.06, centers * 0, rules)
    assert any(code == "LONG_AIRBORNE_REVIEW" and status == "REVIEW" for code, status, _ in flags)


def test_low_posture_nonfoot_support_does_not_imply_airborne_failure(engine, monkeypatch):
    # 隔离足部诊断信号，使用真实躯干FK检查策略合成；不声称此合成姿态整体质量通过。
    qpos = grounded(engine)
    qpos[:, 2] = 0.06
    qpos[:, 3:7] = [2**-0.5, 2**-0.5, 0, 0]
    import tools.data.bumi.umr_text_quality as quality

    monkeypatch.setattr(
        quality,
        "foot_diagnostics",
        lambda *args: ({}, [("LONG_AIRBORNE_REVIEW", "REVIEW", np.ones(60, dtype=bool))]),
    )
    result = engine.evaluate(qpos)
    assert result["metrics"]["feet"]["nonfoot_support_inferred_fraction"] == 1
    assert "LONG_AIRBORNE_REVIEW" not in result["reason_codes"]
    assert "FEET_AIRBORNE_WITH_NONFOOT_SUPPORT" in result["diagnostic_reasons"]


def test_native_loading_reorders_and_keeps_root(bundle, engine):
    paths, make, _, _ = bundle
    row, _ = make(reorder=True)
    qpos, meta = load_umr(row, paths, engine)
    np.testing.assert_allclose(qpos, grounded(engine))
    assert meta["source_up"] == "y" and meta["output_up"] == "z"
    assert meta["source_motion_id"] == "MotionGV/fixture/000000"


@pytest.mark.parametrize(
    "case", ["nan", "fps", "frames", "names", "source_up", "source_nan", "truncated"]
)
def test_invalid_inputs_are_not_quality_rejections(bundle, case):
    paths, make, _, _ = bundle
    row, path = make()
    if case == "nan":
        with np.load(path) as z:
            value = z["qpos"].copy()
        value[2, 7] = np.nan
        rewrite(path, qpos=value)
    elif case == "fps":
        rewrite(path, fps=np.array([25.0], np.float32))
    elif case == "frames":
        rewrite(path, frame_ids=np.arange(60, dtype=np.float32))
    elif case == "names":
        rewrite(path, robot_joint_names=np.array(["waist_yaw_joint"] * 21))
    elif case == "source_up":
        rewrite(row["human_path"], output_up="z")
    elif case == "source_nan":
        rewrite(row["human_path"], trans=np.full((60, 3), np.inf, np.float32))
    else:
        path.write_bytes(path.read_bytes()[:100])
    init_worker(paths)
    decision, cached = evaluate_row((row, None))
    assert decision["status"] == "INVALID", decision
    assert not decision["training_eligible"] and not cached


def test_unexpected_failure_has_error_status(bundle, monkeypatch):
    paths, make, args, _ = bundle
    row, _ = make()
    init_worker(paths)

    def fail(*args):
        raise RuntimeError("test-only implementation failure")

    monkeypatch.setattr(QualityEngine, "evaluate", fail)
    result, _ = evaluate_row((row, None))
    assert result["status"] == "ERROR" and result["reason_codes"] == ["EXECUTION_ERROR"]
    options = args()
    assert run_filter(options) == 2
    with pytest.raises(InputContractError, match="ERROR"):
        QualityGate(options.output)
    monkeypatch.undo()
    options.resume = True
    assert run_filter(options) == 0
    assert json.loads((options.output / "run.json").read_text())["resumed_records"] == 0


def test_full_scan_resume_changed_source_and_lengths(bundle):
    _, make, args, _ = bundle
    row, _ = make()
    make(frames=15)
    make(frames=301, folder="folder1")
    options = args(workers=2)
    assert run_filter(options) == 0
    summary = json.loads((options.output / "quality_summary.json").read_text())
    assert summary["status_counts"] == {"PASS": 3, "TRAIN_ELIGIBLE": 1}
    assert summary["training_exclusion_counts"] == {"LENGTH_OUTSIDE_60_300": 2}
    assert summary["frames_by_status"]["PASS"] == 376
    options.resume = True
    assert run_filter(options) == 0
    assert json.loads((options.output / "run.json").read_text())["resumed_records"] == 3
    rewrite(row["human_path"], output_up="z")
    assert run_filter(options) == 0
    summary = json.loads((options.output / "quality_summary.json").read_text())
    assert summary["status_counts"]["INVALID"] == 1
    assert json.loads((options.output / "run.json").read_text())["resumed_records"] == 2


def test_resume_rejects_config_changes_partial_reports_and_missing_folder(bundle, tmp_path):
    _, make, args, _ = bundle
    make()
    make()
    options = args(limit=1)
    assert run_filter(options) == 0
    with pytest.raises(InputContractError, match="部分"):
        QualityGate(options.output)
    config = tmp_path / "rules.yaml"
    config.write_text(DEFAULT_CONFIG.read_text() + "\n# changed identity\n")
    options.resume, options.config = True, config
    with pytest.raises(ValueError, match="变化"):
        run_filter(options)
    options = args(output=tmp_path / "missing_report")
    (options.input_root / "folder1/bumi3").mkdir(parents=True)
    with pytest.raises(ValueError, match="缺少batch_summary"):
        run_filter(options)


def test_catalog_missing_and_extra_outputs(bundle):
    _, make, args, _ = bundle
    _, missing = make()
    _, valid = make()
    missing.unlink()
    extra = valid.with_name("unrecorded_bumi3.npz")
    extra.write_bytes(valid.read_bytes())
    options = args(expected_records=3)
    assert run_filter(options) == 0
    result = json.loads((options.output / "quality_summary.json").read_text())
    assert result["status_counts"]["INVALID"] == 2
    assert result["status_counts"]["TRAIN_ELIGIBLE"] == 1


def make_conversion(tmp_path, qpos_path, row):
    tmp_path.mkdir(parents=True, exist_ok=True)
    text = "synthetic test only: a robot is standing"
    source_id = row["source_motion_id"]
    embedding = tmp_path / "embedding.pt"
    torch.save(
        dict(
            motion_id=source_id,
            captions=[text],
            encoder="t5-3b",
            max_text_len=150,
            embeddings=torch.ones((1, 150, 1024), dtype=torch.float16),
            attention_mask=torch.ones((1, 150), dtype=torch.bool),
        ),
        embedding,
    )
    ref = dict(
        format="bumi_text_t5_v1",
        path=str(embedding),
        sha256=sha256_file(embedding),
        record_index=0,
        text_index=0,
        motion_id=source_id,
        caption_sha256=caption_hash(text),
    )
    record = dict(
        dataset="motionmillion",
        motion_id=source_id,
        split="train",
        qpos_path=str(qpos_path),
        fps=30,
        captions=[text],
        caption_ids=["0"],
        embeddings=[ref],
        provenance=dict(
            source_id=source_id,
            interval_seconds=[0, 2],
            retargeter="UMR",
            retarget_version="synthetic-test",
        ),
    )
    payload = dict(
        schema="genmo.bumi_text_conversion.v1",
        kinematics=dict(path=str(DEFAULT_KINEMATICS), sha256=sha256_file(DEFAULT_KINEMATICS)),
        records=[record],
    )
    source = tmp_path / "conversion.json"
    source.write_text(json.dumps(payload))
    return source, payload


def test_pass_only_build_rechecks_identity_hash_and_publishes_atomically(bundle, tmp_path):
    _, make, args, _ = bundle
    _, path = make()
    options = args()
    run_filter(options)
    gate = QualityGate(options.output)
    row = gate.lookup(path)
    gate.close()
    source, payload = make_conversion(tmp_path, path, row)
    result = build(source, tmp_path / "release", quality_report=options.output)
    assert result["counts"] == {"train/motionmillion": 1}
    dataset = BumiTextDataset(tmp_path / "release", "train")
    record = dataset.read_record(0)
    assert record["frames"] == 60 and record["qpos"].shape == (60, 28)
    assert record["foot_contact"].shape == (60, 2)
    payload["records"][0]["provenance"]["source_id"] = "wrong-source"
    source.write_text(json.dumps(payload))
    with pytest.raises(InputContractError, match="来源ID"):
        build(source, tmp_path / "failed_release", quality_report=options.output)
    assert not (tmp_path / "failed_release").exists()
    assert not list(tmp_path.glob(".failed_release.staging-*"))
    rewrite(path, ground_z=np.array([0.1], np.float32))
    gate = QualityGate(options.output)
    try:
        with pytest.raises(InputContractError, match="改变"):
            gate.lookup(path)
    finally:
        gate.close()


def test_mirror_lineage_blocks_cross_split_training_leak(bundle, tmp_path):
    _, make, args, _ = bundle
    _, first = make()
    mirror, second = make()
    rewrite(
        mirror["human_path"],
        source_file="/test/motion_272rpr_unpacked/Mirror_MotionGV/fixture/000000.npy",
    )
    options = args()
    run_filter(options)
    gate = QualityGate(options.output)
    try:
        _, train = make_conversion(tmp_path / "train", first, gate.lookup(first))
        _, held = make_conversion(tmp_path / "val", second, gate.lookup(second))
    finally:
        gate.close()
    held["records"][0]["split"] = "val"
    train["records"].extend(held["records"])
    conversion = tmp_path / "combined.json"
    conversion.write_text(json.dumps(train))
    result = build(conversion, tmp_path / "release", quality_report=options.output)
    assert result["counts"] == {"val/motionmillion": 1}
    assert result["excluded"][0]["reason"] == "train_overlaps_held_out"
    assert len(BumiTextDataset(tmp_path / "release", "val")) == 1
