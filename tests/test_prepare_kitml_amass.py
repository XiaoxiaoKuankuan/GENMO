"""KIT-ML 文本对齐和 AMASS 30 Hz 转换的数据契约测试。

使用临时目录中的可解析 MMM 时间戳、SMPL+H 数组和原始文本验证真实失败边界：
缺失动作不得伪造时长或可训练状态；帧数/时长错配不得导出；跨 ±π 的旋转必须沿
最短路径插值；手部、DMPL、形状与性别必须保留。所有产物由 pytest tmp_path 管理。
"""

import json
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from tools.data.kitml.prepare_kitml_amass import (
    GenmoSource,
    build,
    load_motion,
    resample_motion,
    resolve_mapping,
    safe_path,
)


def motion(frames=100, fps=100.0):
    return dict(
        poses=np.zeros((frames, 156), dtype=np.float64),
        trans=np.arange(frames)[:, None] / fps * np.array([[1, 2, 3]]),
        betas=np.arange(16, dtype=float),
        gender=np.array("female"),
        mocap_framerate=np.array(fps),
        dmpls=np.ones((frames, 8)),
    )


def fixture(tmp_path, frames=100, fps=100.0, texts=None):
    root = tmp_path / "kit"
    root.mkdir()
    if texts is None:
        texts = ["A person walks.", "a person takes several steps"]
    (root / "00001_annotations.json").write_text(json.dumps(texts))
    (root / "00001_meta.json").write_text(json.dumps({"nb_annotations": len(texts)}))
    xml = (
        "<MMM><Motion><MotionFrames>"
        + "".join(
            f"<MotionFrame><Timestep>{i / fps:.9f}</Timestep></MotionFrame>" for i in range(frames)
        )
        + "</MotionFrames></Motion></MMM>"
    )
    (root / "00001_mmm.xml").write_text(xml)
    mapping = tmp_path / "mapping"
    mapping.mkdir()
    (mapping / "amass-path2kitml.json").write_text(json.dumps({"00001": "KIT/1/walk_poses.npz"}))
    (mapping / "kitml_amass_path.json").write_text(
        json.dumps({"00001": {"path": "1/walk_poses.npz", "identifier": "kit"}})
    )
    amass = tmp_path / "amass"
    (amass / "KIT/1").mkdir(parents=True)
    args = SimpleNamespace(
        kitml_root=root,
        amass_root=amass,
        mapping_root=mapping,
        output_root=tmp_path / "output",
        report_root=tmp_path / "reports",
        unifier_commit="test",
    )
    return args, amass / "KIT/1/walk_poses.npz"


def test_missing_amass_preserves_all_texts_and_unknown_end(tmp_path):
    args, _ = fixture(tmp_path)
    result = build(args)
    assert result["ready_motions"] == 0
    assert result["status_counts"] == {"missing_amass": 1}
    rows = json.loads((args.output_root / "metadata.json").read_text())
    assert rows[0]["motion_id"] == "kitml_00001"
    assert len(rows[0]["texts"]) == 2
    assert rows[0]["end"] is None
    assert rows[0]["annotations"][0]["end_time"] is None
    assert not list(args.output_root.rglob("*.npz"))


def test_zip_build_preserves_smplh_and_real_caption_intervals(tmp_path):
    args, path = fixture(tmp_path)
    np.savez(path, **motion())
    archive = tmp_path / "kit.zip"
    with zipfile.ZipFile(archive, "w") as out:
        for file in args.kitml_root.iterdir():
            out.write(file, f"kit/{file.name}")
    args.kitml_root = archive
    report = build(args)
    assert report["ready_motions"] == 1
    assert not report["training_ready"]
    rows = json.loads((args.output_root / "metadata_ready.json").read_text())
    row = rows[0]
    assert row["source_end_time"] == row["end"] == 1.0
    assert row["source_num_frames"] == 100 and row["num_frames"] == 30
    output = load_motion(args.output_root / row["motion_path"])
    assert output["poses"].shape == (30, 156)
    assert output["dmpls"].shape == (30, 8)
    np.testing.assert_allclose(output["betas"], np.arange(16))
    np.testing.assert_allclose(output["trans"][:, 0], np.arange(30) / 30, atol=1e-7)
    assert output["gender"].item() == "female"
    assert row["annotations"][0]["caption"] == "A person walks."


@pytest.mark.parametrize("frames,fps", [(99, 100), (100, 120)])
def test_alignment_mismatch_blocks_motion(tmp_path, frames, fps):
    args, path = fixture(tmp_path)
    np.savez(path, **motion(frames, fps))
    result = build(args)
    assert result["status_counts"] == {"alignment_mismatch": 1}
    assert not list(args.output_root.rglob("*.npz"))


def test_slerp_wrap_and_hand_pose_retention():
    data = motion(2, 2.0)
    data["poses"][:, 2] = np.deg2rad([179, -179])
    data["poses"][:, -1] = 0.6
    out = resample_motion(data)
    middle = Rotation.from_rotvec(out["poses"][7, :3]).as_matrix()
    assert middle[0, 0] < -0.999
    np.testing.assert_allclose(out["poses"][:, -1], 0.6)
    assert len(out["poses"]) == 30
    np.testing.assert_allclose(out["trans"][16:], np.broadcast_to(data["trans"][-1], (14, 3)))


@pytest.mark.parametrize("problem", ["nan", "smplx", "dmpls"])
def test_invalid_source_never_becomes_ready(tmp_path, problem):
    args, path = fixture(tmp_path)
    data = motion()
    if problem == "nan":
        data["trans"][0, 0] = np.nan
    elif problem == "smplx":
        data["poses"] = np.zeros((100, 165))
    else:
        data["dmpls"] = np.zeros((99, 8))
    np.savez(path, **data)
    assert build(args)["status_counts"] == {"invalid_source": 1}


def test_no_caption_excluded(tmp_path):
    args, path = fixture(tmp_path, texts=[])
    np.savez(path, **motion())
    assert build(args)["status_counts"] == {"no_text": 1}


def test_mapping_cannot_escape_root_or_guess_ambiguous_dataset(tmp_path):
    with pytest.raises(ValueError):
        safe_path(tmp_path, "../wrong.npz")
    original = {"00001": {"path": "1/sequence_poses.npz", "identifier": "kit"}}
    for family in ("KIT", "EKUT"):
        path = tmp_path / family / "1/sequence_poses.npz"
        path.parent.mkdir(parents=True)
        path.touch()
    assert resolve_mapping("00001", {}, original, tmp_path)[2] == "ambiguous_mapping"
    assert (
        resolve_mapping("00001", {"00001": "CMU/1/sequence_poses.npz"}, original, tmp_path)[2]
        == "missing_amass"
    )


@pytest.mark.parametrize("target_frames,expected", [(30, "ready"), (60, "alignment_mismatch")])
def test_existing_genmo_exact_mapping_and_duration_gate(tmp_path, target_frames, expected):
    args, _ = fixture(tmp_path)
    key = "inputs/smplx_amass/smplxn_raw/KIT/KIT/1/walk_stageii.npz"
    record = {
        "pose": torch.zeros(target_frames, 66),
        "trans": torch.zeros(target_frames, 3),
        "beta": torch.zeros(10),
        "gender": "neutral",
        "model": "smplx",
        "file_name": key,
    }
    args.amass_genmo_file = tmp_path / "smplxpose_v2.pth"
    args.amass_root = None
    torch.save({key: record}, args.amass_genmo_file)
    assert build(args)["status_counts"] == {expected: 1}
    if expected == "ready":
        rows = json.loads((args.output_root / "metadata_ready.json").read_text())
        row = rows[0]
        assert row["source_model_type"] == "smplx"
        assert row["source_key"] == key
        assert row["original_amass_fps"] is None
        out = load_motion(args.output_root / row["motion_path"])
        assert out["poses"].shape == (30, 66)
        assert out["full_pose_available"].item() is False
        assert "left_hand_pose" in out["missing_parameters"]


def test_existing_genmo_embedded_source_identity_cannot_drift(tmp_path):
    path = tmp_path / "source.pth"
    key = "inputs/smplxn_raw/KIT/KIT/1/walk_stageii.npz"
    torch.save({key: {"file_name": "other_stageii.npz"}}, path)
    with pytest.raises(ValueError, match="file_name"):
        GenmoSource(path)
