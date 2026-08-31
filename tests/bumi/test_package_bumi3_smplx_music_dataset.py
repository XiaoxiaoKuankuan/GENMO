"""验证 BUMI3/SMPL-X/WAV 同名数据包的核心交付契约。

测试在 pytest ``tmp_path`` 中构造四个最小正式数据根，包含显式 Y-up→Z-up 溯源的
neutral SMPL-X NPZ、MuJoCo qpos28/contact PT、PCM WAV 和三个 split manifest。
它不使用生产数据或真实模型文件；目标是确认四库路径审计、同一 sample_id
主名、空视频目录、JSON 字段、SHA256、原子发布和单一顶层 tar.gz 结构能端到端成立。
"""

from __future__ import annotations

import argparse
import json
import tarfile
import wave
from pathlib import Path

import numpy as np
import torch

from tools.data.bumi import package_bumi3_smplx_music_dataset as packager


def _write_wav(path: Path, *, seconds: int = 2, sample_rate: int = 8000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * seconds * sample_rate)


def _write_smplx(path: Path, num_frames: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        root_orient=np.zeros((num_frames, 3), dtype=np.float32),
        pose_body=np.zeros((num_frames, 63), dtype=np.float32),
        trans=np.zeros((num_frames, 3), dtype=np.float32),
        betas=np.zeros(16, dtype=np.float32),
        mocap_frame_rate=np.array(30.0, dtype=np.float64),
        gender=np.array("neutral"),
        coordinate_system=np.array("right_handed_z_up_metric"),
        source_coordinate_system=np.array("right_handed_y_up_metric"),
        coordinate_transform=np.array("rotate_global_root_and_translation_plus_90deg_about_x"),
        coordinate_system_was_assumed=np.array(False),
    )


def _write_bumi(path: Path, dataset: str, sample_id: str, num_frames: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    qpos = torch.zeros(num_frames, 28, dtype=torch.float32)
    qpos[:, 3] = 1.0
    torch.save(
        {
            "qpos": qpos,
            "foot_contact": torch.zeros(num_frames, 2, dtype=torch.float32),
            "fps": 30,
            "robot_name": "bumi",
            "quaternion_convention": "wxyz",
            "qpos_order": "mujoco_native",
            "joint_names": list(packager.EXPECTED_BUMI_JOINT_ORDER),
            "ground_semantics": "gmr_foot_sole_ground_zero_v1",
            "foot_contact_contract_version": ("genmo.bumi_foot_contact.fk_sole_hysteresis.v1"),
            "source_dataset": dataset,
            "source_sample_id": sample_id,
        },
        path,
    )


def _write_manifest(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")


def test_package_one_same_named_sample_per_dataset(tmp_path: Path) -> None:
    formal_root = tmp_path / "formal"
    smplx_root = tmp_path / "smplx"
    output_parent = tmp_path / "delivery"
    num_frames = 60

    for dataset, spec in packager.DATASETS.items():
        sample_id = f"{dataset}_sample"
        dataset_root = formal_root / spec["formal_folder"]
        motion_path = dataset_root / "motions" / f"{sample_id}.pt"
        audio_path = dataset_root / "audio" / "shared_song.wav"
        _write_bumi(motion_path, dataset, sample_id, num_frames)
        _write_smplx(smplx_root / dataset / f"{sample_id}.npz", num_frames)
        _write_wav(audio_path)
        row = {
            "dataset": spec["manifest_name"],
            "sample_id": sample_id,
            "sequence_id": sample_id,
            "split": "train",
            "fps": 30,
            "num_frames": num_frames,
            "motion_path": f"motions/{sample_id}.pt",
            "audio_path": "audio/shared_song.wav",
            "audio_key": "shared_song",
        }
        _write_manifest(dataset_root / "manifests" / "train.jsonl", row)
        (dataset_root / "manifests" / "val.jsonl").write_text("", encoding="utf-8")
        (dataset_root / "manifests" / "test.jsonl").write_text("", encoding="utf-8")

    neutral_model = tmp_path / "SMPLX_NEUTRAL.npz"
    bumi_mjcf = tmp_path / "bumi3.xml"
    retarget_config = tmp_path / "smplx_to_bumi3_auto.json"
    neutral_model.write_bytes(b"neutral-model-contract")
    bumi_mjcf.write_text("<mujoco/>", encoding="utf-8")
    retarget_config.write_text("{}\n", encoding="utf-8")

    report = packager.package_dataset(
        argparse.Namespace(
            formal_bumi_root=formal_root,
            normalized_smplx_root=smplx_root,
            smplx_neutral_model=neutral_model,
            bumi_mjcf=bumi_mjcf,
            retarget_config=retarget_config,
            output_parent=output_parent,
            package_name="bumi3_smplx_music_dataset_sample_v1",
            samples_per_dataset=1,
            split="train",
        )
    )

    package_root = Path(report["package_root"])
    contract = json.loads((package_root / "config.json").read_text())
    assert report["status"] == "passed"
    assert report["total_samples"] == 4
    assert contract["scope"]["source_videos_included"] is False
    assert contract["smplx_contract"]["gender"] == "neutral"
    assert contract["bumi3_contract"]["qpos"] == "float32 [T,28]"
    for dataset in packager.DATASETS:
        expected_stem = f"{dataset}_sample"
        assert {
            path.stem for path in (package_root / "human_smplx_motion" / dataset).iterdir()
        } == {expected_stem}
        assert {path.stem for path in (package_root / "bumi3_motion" / dataset).iterdir()} == {
            expected_stem
        }
        assert {path.stem for path in (package_root / "music_wav" / dataset).iterdir()} == {
            expected_stem
        }
        assert list((package_root / "source_video_mp4" / dataset).iterdir()) == []

    with tarfile.open(report["package_archive"], "r:gz") as archive:
        assert {Path(name).parts[0] for name in archive.getnames()} == {
            "bumi3_smplx_music_dataset_sample_v1"
        }
