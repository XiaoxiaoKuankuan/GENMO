from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from gem.robots.bumi.legacy_motion import (
    LEGACY_BUMI_BODY_ORDER,
    LEGACY_BUMI_JOINT_ORDER,
    sha256_file,
)
from tools.data.bumi import build_bumi_music_dataset as builder

SAMPLES = {
    "aistpp": "gBR_sBM_cAll_d04_mBR0_ch01",
    "aioz_gdance": "clip_01_0_120_dancer_00",
    "finedance": "001",
    "compas3d": "Pair1_song1_take1_leader",
}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _legacy(path: Path, frames: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    root = np.zeros((frames, 3), dtype=np.float32)
    root[:, 0] = np.arange(frames) * 0.02
    root[:, 2] = 0.7
    quat = np.zeros((frames, 4), dtype=np.float32)
    quat[:, 3] = 1.0  # legacy xyzw
    payload = {
        "fps": 30,
        "root_pos": root,
        "root_rot": quat,
        "dof_pos": np.zeros((frames, 21), dtype=np.float32),
        "local_body_pos": np.zeros((frames, 25, 3), dtype=np.float32),
        "link_body_list": list(LEGACY_BUMI_BODY_ORDER),
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def _formal_test_kinematics(source: Path, destination: Path) -> Path:
    spec = json.loads(source.read_text(encoding="utf-8"))
    spec["source_mjcf_sha256"] = sha256_file(destination)
    spec["joint_order"] = list(LEGACY_BUMI_JOINT_ORDER)
    spec["joint_name_to_qpos_index"] = {
        name: index for index, name in enumerate(LEGACY_BUMI_JOINT_ORDER)
    }
    spec["joint_name_to_qpos_address"] = {
        name: index + 7 for index, name in enumerate(LEGACY_BUMI_JOINT_ORDER)
    }
    path = destination.parent / "kinematics.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path


def _source_tree(tmp_path: Path, test_kinematics_path: Path):
    selected_root = tmp_path / "selected"
    source_mjcf = tmp_path / "bumi.xml"
    source_mjcf.write_text("<mujoco model='synthetic'/>", encoding="utf-8")
    kinematics = _formal_test_kinematics(test_kinematics_path, source_mjcf)
    ik = tmp_path / "ik.json"
    ik.write_text("{}", encoding="utf-8")
    quality = tmp_path / "quality.yaml"
    quality.write_text("contract_version: synthetic\n", encoding="utf-8")
    human_roots = {name: tmp_path / "human" / name for name in SAMPLES}
    audio_roots = {name: tmp_path / "audio" / name for name in SAMPLES}
    for root in (*human_roots.values(), *audio_roots.values()):
        root.mkdir(parents=True)

    aist = human_roots["aistpp"]
    torch.save(
        {SAMPLES["aistpp"]: {"smpl_pose_global": np.zeros((4, 72), np.float32)}},
        aist / "annot_aist_30fps.pt",
    )
    torch.save([SAMPLES["aistpp"]], aist / "train.pt")
    torch.save([], aist / "val.pt")
    torch.save([], aist / "test.pt")
    aist_feature = aist / "musicfeat_v2" / f"{SAMPLES['aistpp']}_musicfeat_fps30.pt"
    aist_feature.parent.mkdir()
    torch.save(torch.zeros(4, 35), aist_feature)
    (audio_roots["aistpp"] / "mBR0.wav").write_bytes(b"aist")

    rows = {
        "aioz_gdance": {
            "sample_id": SAMPLES["aioz_gdance"],
            "group_id": "clip_01_0_120",
            "music_feature_path": "musicfeat_v2/clip_01_0_120_musicfeat_fps30.pt",
        },
        "finedance": {
            "sample_id": SAMPLES["finedance"],
            "song_name": "Same Song!",
            "music_feature_path": "musicfeat_v2/001_musicfeat_fps30.pt",
        },
        "compas3d": {
            "sample_id": SAMPLES["compas3d"],
            "sequence_id": "Pair1_song1_take1",
            "song_id": "song1",
            "role": "leader",
            "music_feature_path": "musicfeat_v2/Pair1_song1_take1_musicfeat_fps30.pt",
        },
    }
    audio_keys = {
        "aioz_gdance": "clip_01_0_120",
        "finedance": "001",
        "compas3d": "Pair1_song1_take1",
    }
    for dataset, row in rows.items():
        root = human_roots[dataset]
        full = {**row, "fps": 30.0, "num_frames": 4, "split": "train"}
        for split in ("train", "val", "test"):
            _write_jsonl(root / "manifests" / f"{split}.jsonl", [full] if split == "train" else [])
        feature = root / str(row["music_feature_path"])
        feature.parent.mkdir(exist_ok=True)
        torch.save(torch.zeros(4, 35), feature)
        (audio_roots[dataset] / f"{audio_keys[dataset]}.wav").write_bytes(dataset.encode())

    selected_rows = []
    for dataset, sample_id in SAMPLES.items():
        motion = selected_root / "motions" / dataset / f"{sample_id}.pkl"
        _legacy(motion)
        selected_rows.append(
            {
                "dataset": dataset,
                "sample_id": f"{dataset}/{sample_id}",
                "motion_path": f"motions/{dataset}/{sample_id}.pkl",
                "quality_accepted": True,
                "quality_status": "PASS",
                "quality_config_sha256": sha256_file(quality),
                "source_mjcf_sha256": sha256_file(source_mjcf),
                "source_sha256": sha256_file(motion),
            }
        )
    _write_jsonl(selected_root / "manifests" / "selected.jsonl", selected_rows)
    return {
        "selected_root": selected_root,
        "human_roots": human_roots,
        "audio_roots": audio_roots,
        "source_mjcf": source_mjcf,
        "ik_config": ik,
        "quality_config": quality,
        "kinematics_path": kinematics,
    }


def test_converter_preserves_names_root_and_atomic_contract(
    tmp_path: Path, test_kinematics_path: Path, monkeypatch
) -> None:
    arguments = _source_tree(tmp_path, test_kinematics_path)
    monkeypatch.setattr(
        builder, "EXPECTED_SOURCE_MJCF_SHA256", sha256_file(arguments["source_mjcf"])
    )
    output = tmp_path / "formal"
    report = builder.convert_datasets(
        **arguments,
        output_root=output,
        expected_total=4,
        expected_splits={"train": 4, "val": 0, "test": 0},
        expected_dataset_counts={name: 1 for name in SAMPLES},
        expected_unique_music_features=4,
    )
    assert report["total_sequences"] == 4
    assert report["split_counts"] == {"test": 0, "train": 4, "val": 0}
    expected_dirs = {
        "aistpp": "AIST++",
        "aioz_gdance": "AIOZ-GDANCE",
        "finedance": "FineDance",
        "compas3d": "CoMPAS3D",
    }
    for dataset, directory in expected_dirs.items():
        sample_id = SAMPLES[dataset]
        motion_path = output / directory / "motions" / f"{sample_id}.pt"
        assert motion_path.is_file()
        motion = torch.load(motion_path, map_location="cpu", weights_only=False)
        assert motion["qpos"].shape == (4, 28)
        torch.testing.assert_close(motion["qpos"][:, 0], torch.arange(4) * 0.02)
        torch.testing.assert_close(motion["qpos"][:, 2], torch.full((4,), 0.7))
        torch.testing.assert_close(
            motion["qpos"][:, 3:7], torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(4, 1)
        )
        assert motion["root_z_adjusted"] is False


def test_converter_sha_failure_leaves_no_published_or_staging_tree(
    tmp_path: Path, test_kinematics_path: Path, monkeypatch
) -> None:
    arguments = _source_tree(tmp_path, test_kinematics_path)
    monkeypatch.setattr(
        builder, "EXPECTED_SOURCE_MJCF_SHA256", sha256_file(arguments["source_mjcf"])
    )
    manifest = arguments["selected_root"] / "manifests" / "selected.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    rows[0]["source_sha256"] = "0" * 64
    _write_jsonl(manifest, rows)
    output = tmp_path / "failed_formal"
    with pytest.raises(ValueError, match="source motion SHA mismatch"):
        builder.convert_datasets(
            **arguments,
            output_root=output,
            expected_total=4,
            expected_splits={"train": 4, "val": 0, "test": 0},
            expected_dataset_counts={name: 1 for name in SAMPLES},
            expected_unique_music_features=4,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".failed_formal.staging-*"))


def test_converter_propagates_versioned_root_z_selection_contract(
    tmp_path: Path, test_kinematics_path: Path, monkeypatch
) -> None:
    arguments = _source_tree(tmp_path, test_kinematics_path)
    monkeypatch.setattr(
        builder, "EXPECTED_SOURCE_MJCF_SHA256", sha256_file(arguments["source_mjcf"])
    )
    selection_info = {
        "contract_version": "genmo.bumi_gmr_manual_q1_selection.v1",
        "ground_semantics": "gmr_foot_sole_ground_zero_v1",
        "root_z_adjusted": True,
        "root_z_adjustment_method": "foot_contact_bounded_qp",
    }
    info_path = arguments["selected_root"] / "meta" / "selection_info.json"
    info_path.parent.mkdir(parents=True)
    info_path.write_text(json.dumps(selection_info), encoding="utf-8")
    output = tmp_path / "formal_gmr"
    report = builder.convert_datasets(
        **arguments,
        output_root=output,
        expected_total=4,
        expected_splits={"train": 4, "val": 0, "test": 0},
        expected_dataset_counts={name: 1 for name in SAMPLES},
        expected_unique_music_features=4,
    )
    assert report["ground_semantics"] == "gmr_foot_sole_ground_zero_v1"
    assert report["root_z_adjusted"] is True
    motion = torch.load(
        output / "AIST++" / "motions" / f"{SAMPLES['aistpp']}.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert motion["ground_semantics"] == "gmr_foot_sole_ground_zero_v1"
    assert motion["root_z_adjustment_method"] == "foot_contact_bounded_qp"
