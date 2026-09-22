"""UMR文本全量筛选及PASS构建的CPU回归测试。

使用真实fe934机器人XML/网格、标准运动学JSON和临时合成UMR/SMPL-X文件，验证原生
字段、名字重排、Y-up来源、时间线、数值异常、有效膝限位、脚滑/悬空、默认碰撞扣除、
有限队列多进程、断点续跑、源SHA变化和正式构建质量门禁。文本特征明确为测试替身，
不把这些测试当成真实生成质量或动力学验证。全部运行产物仅写pytest的tmp_path。
HumanML3D回归覆盖迁移路径、双输入SHA、原文本绑定、Z-up不重复旋转、镜像保留和
子片段时间范围，以及训练派生包禁止伪造held-out身份。
"""

import csv
import json
import os
import shutil
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

from gem.datasets.pure_motion.bumi_text import BumiTextDataset, caption_hash
from gem.robots.bumi.kinematics import sha256_file
from tools.data.bumi.prepare_bumi_text import build, humanml_conversion, preflight, statistics
from tools.data.bumi.umr_text_preprocess import (
    DEFAULT_CONFIG,
    DEFAULT_KINEMATICS,
    InputContractError,
    QualityGate,
    bones_text_catalog,
    evaluate_row,
    humanml_lineage,
    init_worker,
    kitml_catalog,
    load_umr,
    publish_bones_pass,
    publish_umr_text_pass,
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
    assert any(code == "FOOT_SLIDE_left_REVIEW" for code, _, _ in flags)
    assert not any(code.endswith("REJECT") for code, _, _ in flags)
    _, flags = foot_diagnostics(heights, centers * 2, rules)
    assert not any(code.endswith("REJECT") for code, _, _ in flags)
    _, flags = foot_diagnostics(heights, centers * 3.1, rules)
    assert any(code == "FOOT_SLIDE_left_REJECT" for code, _, _ in flags)
    _, flags = foot_diagnostics(heights + 0.06, centers * 0, rules)
    assert any(code == "LONG_AIRBORNE_REVIEW" and status == "REVIEW" for code, status, _ in flags)


def test_bones_source_timeline_text_and_all_pass_publication(bundle, engine, tmp_path):
    """真实模型验证BONES时间线、源身份、缺失文本和超300帧PASS的完整发布。"""
    paths, make, _, _ = bundle
    paths = dict(paths, dataset="bones_seed")
    original = tmp_path / "smpl_filtered"
    original.mkdir()
    paths["original_source_root"] = str(original)
    csv_path = tmp_path / "metadata.csv"
    paths["metadata_csv"] = str(csv_path)
    rows = []
    for count in (100, 501):
        frames = int(np.floor((count - 1) / 50 * 30 + 1e-9)) + 1
        row, path = make(frames=frames)
        key = Path(row["human_path"]).stem.removesuffix("_smplx")
        raw_path = original / (key + ".pkl")
        raw_path.write_bytes(b"source identity fixture only")
        np.savez_compressed(
            row["human_path"],
            pose_aa=np.zeros((count, 72), np.float32),
            poses=np.zeros((count, 24, 3), np.float32),
            root_orient=np.zeros((count, 3), np.float32),
            pose_body=np.zeros((count, 63), np.float32),
            trans=np.zeros((count, 3), np.float32),
            trans_orig=np.zeros((count, 3), np.float32),
            betas=np.zeros(10, np.float32),
            fps=np.float32(50),
            mocap_framerate=np.float32(50),
            mocap_frame_rate=np.float32(50),
            output_up="y",
            samp_output_up="y",
            source_format="bumi_smpl_pkl",
            source_file=str(raw_path),
            model_type="smplx",
            gender="neutral",
        )
        rewrite(path, source_fps=np.array([50], np.float32), target_fps=np.array([30], np.float32))
        rows.append((row, path, key))
    paths["input_root"] = str(Path(paths["input_root"]) / "folder0")
    fields = ["filename", "is_mirror"] + [f"content_natural_desc_{i}" for i in range(1, 5)]
    with csv_path.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            dict(filename=rows[0][2], is_mirror="False", content_natural_desc_1="Standing still.")
        )
    for row, path, _ in rows:
        row["relative_path"] = path.relative_to(paths["input_root"]).as_posix()
        qpos, meta = load_umr(row, paths, engine)
        assert len(qpos) == meta["frames"] and meta["source_fps"] == 50
        assert meta["source_up"] == "y" and meta["output_up"] == "z"
        with np.load(path, allow_pickle=True) as payload:
            np.testing.assert_array_equal(qpos, payload["qpos"])
    for field in ("output_up", "samp_output_up"):
        rewrite(rows[0][0]["human_path"], **{field: "z"})
        with pytest.raises(InputContractError, match="格式或坐标"):
            load_umr(rows[0][0], paths, engine)
        rewrite(rows[0][0]["human_path"], **{field: "y"})
    rewrite(rows[0][1], source_fps=np.array([30], np.float32))
    with pytest.raises(InputContractError, match="重采样帧率"):
        load_umr(rows[0][0], paths, engine)
    rewrite(rows[0][1], source_fps=np.array([50], np.float32))
    # 保留原始summary绝对路径，验证迁移目录必须显式绑定且qpos不变。
    recorded_output = Path(paths["input_root"]) / "bumi3"
    relocated = tmp_path / "relocated"
    shutil.copytree(Path(paths["input_root"]), relocated)
    paths["input_root"] = str(relocated)
    paths["recorded_output_root"] = str(recorded_output)
    args = Namespace(
        **{k: Path(v) if k != "dataset" else v for k, v in paths.items()},
        output=tmp_path / "report",
        workers=1,
        folders=None,
        limit=None,
        expected_records=2,
        resume=False,
    )
    assert run_filter(args) == 0
    result = publish_bones_pass(args.output, tmp_path / "release")
    assert result["status_counts"] == {"PASS": 2}
    assert result["text_counts"]["with_text"] == 1
    assert result["pass_missing_text_ids"] == [rows[1][2]]
    published = [
        json.loads(line)
        for line in (tmp_path / "release/manifests/pass.jsonl").read_text().splitlines()
    ]
    assert [r["frames"] for r in published] == [60, 301]
    assert published[0]["captions"][0]["caption"] == "Standing still."
    assert published[1]["captions"] == []
    with pytest.raises(FileExistsError):
        publish_bones_pass(args.output, tmp_path / "release")
    with csv_path.open("a") as out:
        out.write(f"{rows[0][2]},False,Duplicate,,,\n")
    with pytest.raises(InputContractError, match="身份为空或重复"):
        bones_text_catalog(csv_path)


def test_kitml_whitelist_timeline_text_and_publication(bundle, engine, tmp_path):
    """真实模型验证KIT白名单、Z-up不旋转、时钟/文本身份及长动作PASS完整发布。"""
    original_paths, make, _, _ = bundle
    source_root = tmp_path / "kit/motions_30hz"
    source_root.mkdir(parents=True)
    root = tmp_path / "kit_output"
    (root / "bumi3").mkdir(parents=True)
    metadata = source_root.parent / "metadata_ready.json"
    paths = dict(
        original_paths,
        dataset="kitml",
        input_root=str(root),
        source_root=str(source_root),
        metadata_json=str(metadata),
    )
    records, results, rows = [], [], []
    for i, frames in enumerate((60, 301), 1):
        _, original = make(frames=frames, reorder=True)
        key = f"kitml_{i:05d}"
        source = source_root / (key + "_poses.npz")
        source_key = f"inputs/smplx_amass/KIT/example_{i}.npz"
        np.savez_compressed(
            source,
            poses=np.zeros((frames, 66), np.float32),
            trans=np.zeros((frames, 3), np.float32),
            betas=np.zeros(10, np.float32),
            gender="neutral",
            mocap_framerate=30.0,
            source_model_type="smplx",
            source_key=source_key,
            source_pose_components="global_orient3+body_pose63",
            coordinate_transform="identity",
            full_pose_available=False,
            source_clock_method="inferred_integer_decimation_from_mmm",
            source_mocap_framerate=100 / 3,
            source_num_frames=frames + 10,
        )
        path = root / "bumi3" / (key + "_poses_bumi3.npz")
        shutil.copy2(original, path)
        rewrite(
            path,
            source_data=str(source),
            source_sequence_key=source.stem,
            source_fps=np.array([30], np.float32),
            target_fps=np.array([30], np.float32),
        )
        record = dict(
            motion_id=key,
            kitml_id=f"{i:05d}",
            motion_path=f"motions_30hz/{source.name}",
            status="ready",
            fps=30,
            source_model_type="smplx",
            coordinate_transform="identity",
            annotation_scope="whole_motion",
            num_frames=frames,
            start=0,
            end=frames / 30,
            texts=["A person stands."],
            annotations=[dict(caption="A person stands.", start_time=0, end_time=frames / 30)],
            source_key=source_key,
            amass_path=f"KIT/example_{i}_poses.npz",
            source_clock_method="inferred_integer_decimation_from_mmm",
            source_fps=100 / 3,
            source_num_frames=frames + 10,
        )
        records.append(record)
        results.append(dict(motion=str(source), out=str(path), status="ok"))
        rows.append(
            dict(
                relative_path=path.relative_to(root).as_posix(),
                folder=root.name,
                human_path=str(source),
                upstream_status="ok",
            )
        )
    metadata.write_text(json.dumps(records))
    engine.__dict__.pop("kitml_catalog", None)
    for row in rows:
        qpos, meta = load_umr(row, paths, engine)
        assert meta["source_up"] == meta["output_up"] == "z"
        assert meta["caption_count"] == 1 and meta["source_frames"] == len(qpos)
        np.testing.assert_array_equal(qpos, grounded(engine, len(qpos)))
    source = Path(rows[0]["human_path"])
    for field, wrong, restored, pattern in (
        ("mocap_framerate", 50.0, 30.0, "来源、Z-up"),
        ("output_up", "y", "z", "来源、Z-up"),
        ("source_num_frames", 999, 70, "来源时钟"),
        ("source_key", "wrong", records[0]["source_key"], "来源、Z-up"),
    ):
        rewrite(source, **{field: wrong})
        with pytest.raises(InputContractError, match=pattern):
            load_umr(rows[0], paths, engine)
        rewrite(source, **{field: restored})
    metadata.write_text(json.dumps(records + records[:1]))
    with pytest.raises(InputContractError, match="身份重复"):
        kitml_catalog(paths)
    metadata.write_text(json.dumps(records))
    summary_path = root / "bumi3/batch_summary.json"
    summary_path.write_text(json.dumps(dict(results=results[:1])))
    args = Namespace(
        **{k: Path(v) if k != "dataset" else v for k, v in paths.items()},
        output=tmp_path / "incomplete_report",
        workers=1,
        folders=None,
        limit=None,
        expected_records=2,
        resume=False,
    )
    with pytest.raises(InputContractError, match="白名单集合不同"):
        run_filter(args)
    summary_path.write_text(json.dumps(dict(results=results)))
    args.output = tmp_path / "kit_report"
    assert run_filter(args) == 0
    info = publish_umr_text_pass(args.output, tmp_path / "kit_pass", dataset="kitml")
    assert info["status_counts"] == {"PASS": 2} and info["pass_frames"] == 361
    assert info["text_counts"]["pass_with_text"] == 2
    published = [
        json.loads(line)
        for line in (tmp_path / "kit_pass/manifests/pass.jsonl").read_text().splitlines()
    ]
    assert [row["frames"] for row in published] == [60, 301]
    assert published[0]["captions"] == records[0]["annotations"]
    assert published[0]["canonical_source_id"] == "kitml:" + records[0]["source_key"]


@pytest.mark.parametrize(
    "degrees,frames,rejected", [(29.9, 20, False), (30.1, 14, False), (30.1, 15, True)]
)
def test_root_tilt_threshold_and_duration(engine, degrees, frames, rejected):
    qpos = grounded(engine, frames=frames)
    angle = np.deg2rad(degrees) / 2
    qpos[:, 3:7] = [np.cos(angle), np.sin(angle), 0, 0]
    result = engine.evaluate(qpos)
    assert ("ROOT_TILT_SUSTAINED" in result["reason_codes"]) == rejected
    if rejected:
        assert result["reason_statuses"]["ROOT_TILT_SUSTAINED"] == "REJECT"
        assert result["issue_intervals"]["ROOT_TILT_SUSTAINED"] == [[0, frames]]
    qpos[:, 3:7] = [np.cos(angle), 0, 0, np.sin(angle)]
    assert "ROOT_TILT_SUSTAINED" not in engine.evaluate(qpos)["reason_codes"]


@pytest.mark.parametrize(
    "depth,frames,level",
    [
        (0.009, 20, None),
        (0.02, 10, None),
        (0.02, 11, "REVIEW"),
        (0.06, 10, None),
        (0.06, 11, "REJECT"),
    ],
)
def test_collision_requires_more_than_ten_frames(engine, monkeypatch, depth, frames, level):
    qpos = grounded(engine, frames=30)
    pair = "l_arm_yaw_link / r_arm_yaw_link"
    baseline = engine.reference_depths.get(pair, 0)
    calls = iter(range(len(qpos)))
    monkeypatch.setattr(
        engine, "collision_depths", lambda: {pair: baseline + depth} if next(calls) < frames else {}
    )
    result = engine.evaluate(qpos)
    assert result["reason_statuses"].get("SELF_COLLISION_" + pair) == level


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
    assert meta["source_sequence_key"] == Path(row["human_path"]).stem


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


@pytest.fixture
def humanml_bundle(tmp_path, bundle, engine):
    paths, _, args, _ = bundle
    root, source = Path(paths["input_root"]), Path(paths["source_root"])
    robot = root / "out_umr/bumi3"
    robot.mkdir(parents=True)
    (root / "source_metadata").mkdir()
    (source / "motions").mkdir(parents=True)
    old_source, old_output = Path("/old/humanml"), Path("/old/umr/out_umr/bumi3")
    old_xml = Path("/old/UMR/assets/bumi3/mjcf/bumi3_retarget.xml")
    manifest, results, texts = [], [], {}
    for key in ("000002", "M000002", "000002__seg_1000_3000"):
        human = source / "motions" / (key + ".npz")
        np.savez_compressed(
            human,
            root_orient=np.zeros((60, 3), np.float32),
            pose_body=np.zeros((60, 63), np.float32),
            trans=np.zeros((60, 3), np.float32),
            betas=np.zeros(10, np.float32),
            gender="neutral",
            mocap_frame_rate=np.float32(30),
            output_up="z",
        )
        path = robot / (key + "_bumi3.npz")
        np.savez_compressed(
            path,
            qpos=grounded(engine),
            fps=np.float32(30),
            frame_ids=np.arange(60, dtype=np.int32),
            robot_xml=str(old_xml),
            robot_name="bumi3",
            robot_joint_names=np.array(engine.kin.joint_order),
            source_data=str(old_source / "motions" / human.name),
            source_sequence_key=key,
            source_format="smplx_npz",
            smpl_scale=np.float32(0.573),
            ground_z=np.float32(0.01),
            zero_source_finger_pose=np.bool_(True),
        )
        manifest.append(
            dict(
                motion_id=key,
                file=f"motions/{key}.npz",
                frames=60,
                caption_count=1,
                bytes=human.stat().st_size,
                sha256=sha256_file(human),
            )
        )
        texts[key] = [dict(caption="synthetic test only: a robot is standing", tokens=[])]
        results.append(
            dict(
                motion=str(old_source / "motions" / human.name),
                out=str(old_output / path.name),
                status="ok",
            )
        )
    metadata = dict(
        schema="humanml3d_umr_npz_v1",
        coordinate_system="right_handed_z_up",
        fps=30,
        dataset="GENMO HumanML3D training derivative",
        motion_count=3,
        total_frames=180,
        output_directory=str(old_source),
    )
    (source / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in manifest))
    (source / "metadata.json").write_text(json.dumps(metadata))
    (source / "texts.json").write_text(json.dumps(texts))
    for name in ("manifest.jsonl", "metadata.json", "texts.json"):
        (root / "source_metadata" / name).write_bytes((source / name).read_bytes())
    (robot / "batch_summary.json").write_text(json.dumps(dict(results=results)))
    files = sorted(p for p in root.rglob("*") if p.is_file())
    (root / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(p)}  {p.relative_to(root).as_posix()}\n" for p in files)
    )
    options = args(
        dataset="humanml3d",
        recorded_output_root=old_output,
        recorded_robot_xml=old_xml,
        expected_records=3,
    )
    return options, robot, source


def test_humanml_full_source_check_mirrors_segments_and_training_build(
    humanml_bundle, tmp_path, engine
):
    from tools.eval.render_bumi_motion import checked_umr_qpos

    options, robot, _ = humanml_bundle
    assert run_filter(options) == 0
    summary = json.loads((options.output / "quality_summary.json").read_text())
    assert summary["status_counts"] == {"PASS": 3, "TRAIN_ELIGIBLE": 3}
    conversion = tmp_path / "conversion.json"
    assert humanml_conversion(options.output, conversion)["records"] == 3
    payload = json.loads(conversion.read_text())
    gate = QualityGate(options.output)
    try:
        for record in payload["records"]:
            path = Path(record["qpos_path"])
            row = gate.lookup(path)
            assert row["source_up"] == row["output_up"] == "z"
            _, embedding_payload = make_conversion(tmp_path / record["motion_id"], path, row)
            record["embeddings"] = embedding_payload["records"][0]["embeddings"]
            qpos, _ = gate.read_candidate(path, record)
            np.testing.assert_array_equal(qpos.numpy(), grounded(engine))
            rendered = checked_umr_qpos(row, gate.run, gate.engine.model, gate.engine)
            np.testing.assert_array_equal(rendered, qpos.numpy())
            assert row["verified_source_sequence_key"] == record["motion_id"]
            assert row["verified_captions"][0]["caption"] == record["captions"][0]
            with pytest.raises(ValueError, match="源身份"):
                checked_umr_qpos(
                    dict(row, source_motion_id="wrong"), gate.run, gate.engine.model, gate.engine
                )
        bad = dict(payload["records"][0], split="val")
        with pytest.raises(InputContractError, match="训练集"):
            gate.validate_identity(gate.lookup(Path(bad["qpos_path"])), bad)
        bad = dict(payload["records"][0], captions=["wrong caption"])
        with pytest.raises(InputContractError, match="caption"):
            gate.read_candidate(Path(bad["qpos_path"]), bad)
    finally:
        gate.close()
    conversion.write_text(json.dumps(payload))
    report = build(conversion, tmp_path / "release", quality_report=options.output)
    assert report["counts"] == {"train/humanml3d": 3}
    dataset = BumiTextDataset(tmp_path / "release", "train")
    assert len(dataset) == 3
    intervals = {
        dataset.read_record(i)["motion_id"]: dataset.read_record(i)["provenance"][
            "interval_seconds"
        ]
        for i in range(3)
    }
    assert intervals["000002__seg_1000_3000"] == [1.0, 3.0]
    stats = statistics(tmp_path / "release", tmp_path / "stats.json")
    assert stats["dataset"] == stats["data_identity"]["dataset"] == "humanml3d"
    assert stats["records"] == 3
    assert preflight(tmp_path / "release", limit=0)["data_identity"]["dataset"] == "humanml3d"
    options.resume = True
    assert run_filter(options) == 0
    assert json.loads((options.output / "run.json").read_text())["resumed_records"] == 3


@pytest.mark.parametrize(
    "change", ["source_missing", "source_tampered", "robot_tampered", "old_path"]
)
def test_humanml_rejects_missing_tampered_or_mismapped_inputs(humanml_bundle, change):
    options, robot, source = humanml_bundle
    human = source / "motions/000002.npz"
    output = robot / "000002_bumi3.npz"
    if change == "source_missing":
        human.unlink()
    elif change == "source_tampered":
        rewrite(human, output_up="y")
    elif change == "robot_tampered":
        rewrite(output, ground_z=np.float32(2))
    else:
        options.recorded_robot_xml = Path("/wrong/model.xml")
    assert run_filter(options) == 0
    summary = json.loads((options.output / "quality_summary.json").read_text())
    assert summary["status_counts"]["INVALID"] == (3 if change == "old_path" else 1)


def test_humanml_short_source_boundary_preserves_actual_caption_interval():
    value = humanml_lineage("003037__seg_1000_5000", 118, parent_frames=148)
    assert value["source_end_clipped"]
    assert value["interval_seconds"] == [1.0, 1 + 118 / 30]
    assert value["annotation_interval_seconds"] == [1.0, 5.0]
    with pytest.raises(InputContractError, match="时间范围"):
        humanml_lineage("003037__seg_1000_5000", 118, parent_frames=300)
