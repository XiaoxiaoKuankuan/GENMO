"""验证 GMR 人工 q1 发布适配器的集合、哈希、安全门、Root Z 和原子失败契约。

测试使用四个极小的合成语料样本构造完整 release audit、人工选择索引与 PKL SHA 清单，
既覆盖成功发布后的 provenance/足底地面语义，也覆盖任一嵌入式安全门失败时不产生正式
目录或 staging 残留。这样正式 3162 条数据的接入规则可在不依赖大数据文件时持续回归。
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import yaml

from gem.robots.bumi.legacy_motion import LEGACY_BUMI_BODY_ORDER, LEGACY_BUMI_JOINT_ORDER
from tools.data.bumi import prepare_gmr_manual_q1_selected_root as adapter

DATASETS = ("aistpp", "aioz_gdance", "finedance", "compas3d")


def _sha(path: Path) -> str:
    return adapter.sha256_file(path)


def _fixture_tree(tmp_path: Path, *, safety: bool = True) -> dict[str, Path]:
    motion_root = tmp_path / "gmr"
    index_path = tmp_path / "selected.jsonl"
    selection_manifest = tmp_path / "motion_sha256.txt"
    pkl_manifest = tmp_path / "pkl_sha256.txt"
    source_mjcf = tmp_path / "bumi3.xml"
    source_mjcf.write_text("<mujoco model='synthetic'/>", encoding="utf-8")
    retarget = tmp_path / "smplx_to_bumi3_auto.json"
    retarget.write_text("{}\n", encoding="utf-8")
    quality_config = tmp_path / "quality.yaml"
    quality_config.write_text(
        yaml.safe_dump(
            {
                "contract_version": "genmo.bumi_gmr_manual_q1_quality_gate.v1",
                "source": {
                    "pipeline_version": "bumi3_temporal_bounded_v1",
                    "mjcf_sha256": _sha(source_mjcf),
                    "joint_order": list(LEGACY_BUMI_JOINT_ORDER),
                },
                "selection": {
                    "expected_total": 4,
                    "expected_dataset_counts": {name: 1 for name in DATASETS},
                    "expected_split_counts": {"train": 4, "val": 0, "test": 0},
                },
                "gmr_acceptance": {
                    "required_flags": [
                        "finite",
                        "xml_urdf_joint_limit_contract",
                        "joint_position_velocity_acceleration",
                        "upper_body_frame_delta",
                        "root_height",
                        "root_acceleration_not_worse",
                        "safety_overall",
                    ]
                },
                "root_z": {
                    "ground_semantics": "gmr_foot_sole_ground_zero_v1",
                    "required_method": "foot_contact_bounded_qp",
                    "max_sole_penetration_m": 0.0021,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    index_rows = []
    manifest_rows = []
    selection_manifest_rows = []
    for dataset in DATASETS:
        sample_id = f"{dataset}_sample"
        path = motion_root / dataset / f"{sample_id}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        acceptance = {
            "finite": safety,
            "xml_urdf_joint_limit_contract": True,
            "joint_position_velocity_acceleration": True,
            "upper_body_frame_delta": True,
            "fk_drift": dataset != "finedance",
            "root_height": True,
            "root_acceleration_not_worse": True,
            "safety_overall": safety,
            "fidelity_overall": dataset != "finedance",
            "overall": dataset != "finedance" and safety,
        }
        payload = {
            "fps": 30.0,
            "root_pos": np.zeros((4, 3), dtype=np.float32),
            "root_rot": np.tile(np.asarray([0.0, 0.0, 0.0, 1.0]), (4, 1)),
            "dof_pos": np.zeros((4, 21), dtype=np.float32),
            "dof_names": list(LEGACY_BUMI_JOINT_ORDER),
            "local_body_pos": np.zeros((4, 25, 3), dtype=np.float32),
            "link_body_list": list(LEGACY_BUMI_BODY_ORDER),
            "quality": {
                "pipeline_version": "bumi3_temporal_bounded_v1",
                "aligned_fps": 30.0,
                "joint_limit_contract": {"pass": True},
                "trajectory": {"constraint_pass": True},
                "root_height": {"method": "foot_contact_bounded_qp"},
                "final_root_audit": {"finite": True, "max_sole_penetration": 0.002},
                "acceptance": acceptance,
            },
        }
        with path.open("wb") as handle:
            pickle.dump(payload, handle)
        manifest_rows.append(f"{_sha(path)}  {dataset}/{sample_id}.pkl\n")
        index_rows.append(
            {
                "dataset": dataset,
                "sample_id": sample_id,
                "score": 1,
                "source_index_hash_match": True,
                "num_frames": 4,
                "split": "train",
                "actual_sha256": "a" * 64,
            }
        )
        selection_manifest_rows.append(f"{'a' * 64}  motions/{dataset}/{sample_id}.npz\n")
    index_path.write_text("".join(json.dumps(row) + "\n" for row in index_rows), encoding="utf-8")
    pkl_manifest.write_text("".join(manifest_rows), encoding="utf-8")
    selection_manifest.write_text("".join(selection_manifest_rows), encoding="utf-8")
    release_audit = tmp_path / "release.json"
    release_audit.write_text(
        json.dumps(
            {
                "schema": "gmr_bumi3_manual_q1_release_audit_v1",
                "selection": {
                    "selected_unique_clips": 4,
                    "selected_total_frames": 16,
                    "source_sha256_verified": 4,
                },
                "dataset_contract": {
                    "output_pkl_count": 4,
                    "loaded_total_frames": 16,
                    "fps": 30.0,
                    "dof_names": list(LEGACY_BUMI_JOINT_ORDER),
                    "pkl_manifest_sha256": _sha(pkl_manifest),
                },
                "acceptance_counts": {"safety_overall": 4},
                "per_dataset": {name: {"clips": 1} for name in DATASETS},
            }
        ),
        encoding="utf-8",
    )
    return {
        "motion_root": motion_root,
        "selection_index": index_path,
        "selection_sha256_manifest": selection_manifest,
        "release_audit": release_audit,
        "pkl_sha256_manifest": pkl_manifest,
        "source_mjcf": source_mjcf,
        "retarget_config": retarget,
        "quality_config": quality_config,
    }


def test_prepare_selected_root_preserves_manual_set_and_root_contract(
    tmp_path: Path, monkeypatch
) -> None:
    arguments = _fixture_tree(tmp_path)
    monkeypatch.setattr(adapter, "EXPECTED_SOURCE_MJCF_SHA256", _sha(arguments["source_mjcf"]))
    output = tmp_path / "selected_root"
    report = adapter.prepare_selected_root(
        **arguments,
        output_root=output,
        expected_total=4,
    )
    assert report["total_sequences"] == 4
    assert report["total_frames"] == 16
    assert report["ground_semantics"] == "gmr_foot_sole_ground_zero_v1"
    assert report["root_z_adjusted"] is True
    assert report["second_root_z_adjustment_applied"] is False
    assert report["fidelity_counts"] == {"false": 1, "true": 3}
    rows = [
        json.loads(line)
        for line in (output / "manifests" / "selected.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 4
    assert all(row["manual_rating_score"] == 1 for row in rows)
    assert all(row["gmr_safety_overall"] is True for row in rows)
    assert (output / "meta" / "source_mjcf.snapshot.xml").is_file()


def test_failed_embedded_safety_gate_leaves_no_partial_output(tmp_path: Path, monkeypatch) -> None:
    arguments = _fixture_tree(tmp_path, safety=False)
    monkeypatch.setattr(adapter, "EXPECTED_SOURCE_MJCF_SHA256", _sha(arguments["source_mjcf"]))
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="required GMR safety gates failed"):
        adapter.prepare_selected_root(
            **arguments,
            output_root=output,
            expected_total=4,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".rejected.staging-*"))
