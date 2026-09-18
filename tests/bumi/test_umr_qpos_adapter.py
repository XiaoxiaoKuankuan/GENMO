"""验证 UMR 接入现有筛选及训练 reader 的关键数据边界。

测试使用真实 fe934 运动学和临时合成动作，覆盖具名关节重排、拒绝不完整时间线、
拒绝错误四元数，以及从统一质量报告到 train manifest 的完整原子发布。集成测试
还会篡改已筛选源文件，确认发布器拒绝过期报告且不留下正式或 staging 目录。
所有文件由 pytest 的临时目录管理，不生成正式训练 checkpoint 或数据。
"""

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

from gem.robots.bumi.kinematics import BumiKinematics
from gem.robots.bumi.legacy_motion import sha256_file
from tools.data.bumi import umr_qpos_adapter as adapter
from tools.data.bumi.filter_robot_retargeter_npz_motions import DEFAULT_CONFIG, load_config

KIN = (
    Path(__file__).resolve().parents[2]
    / "configs/bumi/bumi_kinematics_robot_retargeter_fe934_v1.json"
)


def source(tmp_path, *, reverse=False, mutation=None):
    kin = BumiKinematics(KIN)
    root = tmp_path / "source"
    root.mkdir()
    xml = tmp_path / "robot.xml"
    xml.write_text("verified in production separately")
    qpos = kin.default_qpos.repeat(120, 1).numpy()
    # 采用非对称关节值验证重排，不能让全零掩盖错序。
    qpos[:, 7:] = ((kin.joint_lower_limits + kin.joint_upper_limits) * 0.1).numpy()
    names = list(kin.joint_order)
    expected = qpos.copy()
    if reverse:
        names.reverse()
        qpos[:, 7:] = qpos[:, 7:][:, ::-1]
    ids = np.arange(120)
    if mutation == "frames":
        ids[50] += 1
    if mutation == "quat":
        qpos[:, 3] = 2
    if mutation == "names":
        names[0] = names[1]
    path = root / "example_bumi3.npz"
    human = tmp_path / "original/aistpp/example.npz"
    human.parent.mkdir(parents=True)
    np.savez(human, dataset="aistpp", sample_id="example", fps=30., num_frames=120,
             coordinate_system="right_handed_z_up_metric")
    np.savez(
        path,
        qpos=qpos,
        fps=np.array([30.0]),
        frame_ids=ids,
        robot_joint_names=np.asarray(names, dtype=object),
        robot_xml=str(xml),
        robot_name="bumi3",
        source_format="smplx_npz",
        source_sequence_key="example",
        source_data=str(human),
    )
    return kin, path, xml, expected


def test_umr_maps_joint_names_without_moving_root(tmp_path):
    kin, path, xml, expected = source(tmp_path, reverse=True)
    actual, _ = adapter.load_umr_qpos(path, kin, xml)
    np.testing.assert_array_equal(actual.numpy(), expected)
    arrays = adapter.qpos_arrays(actual, kin, load_config(DEFAULT_CONFIG))
    assert arrays["body_pos_w"].shape == (120, 22, 3)
    np.testing.assert_allclose(arrays["body_pos_w"][:, 0], expected[:, :3], atol=1e-6)


@pytest.mark.parametrize("mutation", ["frames", "quat", "names"])
def test_umr_rejects_broken_contract(tmp_path, mutation):
    kin, path, xml, _ = source(tmp_path, mutation=mutation)
    with pytest.raises(ValueError):
        adapter.load_umr_qpos(path, kin, xml)


def test_prefixed_umr_alias_keeps_original_dataset_identity(tmp_path):
    kin, path, xml, expected = source(tmp_path)
    original = tmp_path / "human" / "aistpp" / "example.npz"
    original.parent.mkdir(parents=True)
    np.savez(original, dataset="aistpp", sample_id="example", fps=30., num_frames=120,
             coordinate_system="right_handed_z_up_metric")
    alias = tmp_path / "aistpp__example.npz"
    alias.symlink_to(original)
    with np.load(path, allow_pickle=True) as archive:
        payload = {key: archive[key] for key in archive.files}
    payload.update(source_sequence_key="aistpp__example", source_data=str(alias))
    np.savez(path, **payload)
    qpos, _ = adapter.load_source(
        {"dataset": "aistpp", "sample_id": "aistpp/example", "source_relative_path": path.name},
        path.parent, None, kin, xml,
    )
    np.testing.assert_array_equal(qpos.numpy(), expected)


def test_full_publish_and_source_tamper_rejection(tmp_path, monkeypatch):
    kin, path, xml, _ = source(tmp_path)
    manifest = tmp_path / "selection.json"
    manifest.write_text(
        json.dumps(
            {
                "n": 1,
                "files": [path.name],
                "paths": [str(path)],
                "clips": [{"dataset": "aistpp", "file": path.name, "rank": 1, "score_m": 0.0}],
            }
        )
    )
    row = adapter.input_rows(manifest, path.parent, None)[0]
    adapter._init_worker(str(KIN))
    decision = adapter._evaluate(
        (row, path.parent, None, xml, load_config(DEFAULT_CONFIG), sha256_file(DEFAULT_CONFIG))
    )
    assert decision["status"] == "PASS", decision
    quality = tmp_path / "quality.jsonl"
    quality.write_text(json.dumps(decision) + "\n")
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "report_contract_version": adapter.REPORT_VERSION,
                "quality_config_sha256": sha256_file(DEFAULT_CONFIG),
                "mine_root": None,
                "selection_manifest": str(manifest),
                "selection_manifest_sha256": sha256_file(manifest),
                "sequences": 1,
                "quality_accepted_sequences": 1,
                "status_counts": {"PASS": 1},
            }
        )
    )
    music, audio = tmp_path / "music.pt", tmp_path / "audio.wav"
    torch.save(torch.zeros(120, 35), music)
    audio.write_bytes(b"reference audio identity")
    monkeypatch.setattr(adapter, "verify_assets", lambda *a: {"retarget_config_sha256": "a" * 64})
    monkeypatch.setattr(
        adapter,
        "_references",
        lambda *a: {
            "aistpp": {
                "example": {
                    "sample_id": "example",
                    "sequence_id": "example",
                    "music_group_id": "song",
                    "audio_key": "song",
                    "split": "train",
                    "num_frames": 120,
                    "_feature": str(music),
                    "_audio": str(audio),
                }
            }
        },
    )
    args = Namespace(
        source_root=path.parent,
        mine_root=None,
        output_root=tmp_path / "published",
        quality_config=DEFAULT_CONFIG,
        robot_xml=xml,
        retarget_config=xml,
        kinematics=KIN,
        quality_summary=summary,
        quality_report=quality,
        expected_pass=1,
    )
    report = adapter.build_main(args, {}, {})
    assert report["total_pass_sequences"] == 1
    payload = torch.load(args.output_root / "AIST++/motions/example.pt", weights_only=False)
    assert payload["ground_semantics"] == adapter.GROUND
    assert payload["root_z_second_adjustment_applied"] is False
    args.output_root = tmp_path / "tampered"
    with path.open("ab") as handle:
        handle.write(b"changed after screening")
    with pytest.raises(ValueError, match="SHA"):
        adapter.build_main(args, {}, {})
    assert not args.output_root.exists()
    assert not list(tmp_path.glob(".tampered.staging-*"))
