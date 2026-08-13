"""CPU-only tests for the custom partial AIST++ annotation builder."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

import gem.datasets.aistpp.aistplusplus as aist_dataset_module
import tools.data.aistpp.build_annot_aist_30fps as builder
from gem.datasets.aistpp.aistplusplus import AISTPlusPlusSmplDataset
from tools.data.aistpp.build_annot_aist_30fps import (
    atomic_save_outputs,
    build_camera_tensors,
    build_partial_splits,
    camera_space_smpl,
    choose_minitrain,
    compute_tight_bboxes,
    downsample_motion_indices,
    normalize_aist_camera_extrinsics,
    normalize_aist_translation,
    normalize_smpl_poses,
    select_camera,
    select_keypoint_frames,
    validate_music_features,
    view_to_index,
)


class _DummyBodyModel:
    def get_skeleton(self, betas: torch.Tensor) -> torch.Tensor:
        return torch.zeros(55, 3, dtype=torch.float32)


def _camera(name: str = "c01") -> dict:
    return {
        "name": name,
        "size": [640, 480],
        "matrix": [[500.0, 0.0, 320.0], [0.0, 510.0, 240.0], [0.0, 0.0, 1.0]],
        "distortions": [0.0, 0.0, 0.0, 0.0, 0.0],
        "rotation": [0.0, 0.0, np.pi / 2],
        "translation": [1.0, 2.0, 3.0],
    }


def _record(length: int = 4) -> dict:
    pose = np.zeros((length, 72), dtype=np.float32)
    trans = np.zeros((length, 3), dtype=np.float32)
    return {
        "smpl_pose_global": pose.copy(),
        "smpl_trans_global": trans.copy(),
        "smpl_pose": pose.copy(),
        "smpl_trans": trans.copy(),
        "bbox_xyxy": np.tile(np.array([[10, 20, 100, 200]], dtype=np.float32), (length, 1)),
        "intrinsics": torch.eye(3, dtype=torch.float32),
        "T_w2c": torch.eye(4, dtype=torch.float32),
        "contact_supervision_valid": True,
        "height": 480,
        "width": 640,
    }


def _valid_music(length: int) -> torch.Tensor:
    feature = torch.zeros(length, 35, dtype=torch.float32)
    feature[:, 0] = torch.arange(length, dtype=torch.float32)
    feature[::2, 33] = 1
    feature[1::2, 34] = 1
    return feature


def test_pose_normalization_accepts_24x3_and_72() -> None:
    structured = np.arange(5 * 24 * 3, dtype=np.float64).reshape(5, 24, 3)
    flat = normalize_smpl_poses(structured)
    assert flat.shape == (5, 72)
    assert flat.dtype == np.float32 and flat.flags.c_contiguous
    already_flat = normalize_smpl_poses(flat)
    assert np.array_equal(already_flat, flat)


def test_downsampling_uses_source_frame_zero_and_stride_two() -> None:
    assert np.array_equal(downsample_motion_indices(7, 60, 30), [0, 2, 4, 6])
    with pytest.raises(ValueError, match="integer"):
        downsample_motion_indices(7, 50, 30)


def test_keypoint_tail_clamp_and_excess_difference() -> None:
    keypoints = np.zeros((9, 5, 17, 3), dtype=np.float32)
    for frame in range(5):
        keypoints[:, frame, :, 0] = frame
    selected, clamped, kp_frames = select_keypoint_frames(
        keypoints, "c01", np.array([0, 2, 4, 6]), 7, 2
    )
    assert kp_frames == 5 and clamped == 1
    assert np.all(selected[-1, :, 0] == 4)
    with pytest.raises(ValueError, match="exceeds"):
        select_keypoint_frames(keypoints, "c01", np.array([0]), 8, 2)


@pytest.mark.parametrize("number", range(1, 10))
def test_view_indices(number: int) -> None:
    assert view_to_index(f"c{number:02d}") == number - 1


def test_camera_selection_uses_exact_name_not_first_item() -> None:
    payload = [_camera("c03"), _camera("c01"), _camera("c02")]
    assert select_camera(payload, "c01")["name"] == "c01"
    with pytest.raises(ValueError, match="exactly one"):
        select_camera(payload, "c09")


def test_rodrigues_t_w2c_and_world_to_camera_convention() -> None:
    intrinsics, transform, width, height = build_camera_tensors(_camera())
    assert (width, height) == (640, 480)
    assert intrinsics.shape == (3, 3)
    world = torch.tensor([1.0, 0.0, 0.0])
    expected = transform[:3, :3] @ world + transform[:3, 3]
    homogeneous = transform @ torch.tensor([1.0, 0.0, 0.0, 1.0])
    assert torch.allclose(expected, homogeneous[:3])
    assert torch.allclose(expected, torch.tensor([1.0, 3.0, 3.0]), atol=1e-5)


def _frame(points: list[tuple[float, float]], confidence: float = 1.0) -> np.ndarray:
    result = np.zeros((17, 3), dtype=np.float32)
    for index, (x, y) in enumerate(points):
        result[index] = (x, y, confidence)
    return result


def test_tight_bbox_from_coco_keypoints() -> None:
    keypoints = np.stack([_frame([(10, 20), (30, 40), (15, 35), (25, 22)])])
    boxes, invalid = compute_tight_bboxes(keypoints, 100, 80, 0.1, 4)
    assert invalid == 0
    assert np.array_equal(boxes[0], [10, 20, 30, 40])


def test_bbox_interpolates_middle_and_fills_ends() -> None:
    invalid = np.zeros((17, 3), dtype=np.float32)
    first = _frame([(10, 10), (20, 30), (12, 20), (18, 25)])
    last = _frame([(30, 20), (50, 60), (35, 30), (45, 55)])
    keypoints = np.stack([invalid, first, invalid, last, invalid])
    boxes, missing = compute_tight_bboxes(keypoints, 100, 80, 0.1, 4)
    assert missing == 3
    assert np.array_equal(boxes[0], boxes[1])
    assert np.array_equal(boxes[-1], boxes[-2])
    assert np.allclose(boxes[2], (boxes[1] + boxes[3]) / 2)


def test_all_invalid_bboxes_raise() -> None:
    with pytest.raises(ValueError, match="no valid bbox"):
        compute_tight_bboxes(np.zeros((4, 17, 3)), 100, 80, 0.1, 4)


def test_camera_space_root_uses_current_transform() -> None:
    pose = np.zeros((2, 72), dtype=np.float32)
    trans = np.array([[0, 0, 0], [1, 2, 3]], dtype=np.float32)
    _, transform, _, _ = build_camera_tensors(_camera())
    camera_pose, camera_trans = camera_space_smpl(
        pose, trans, transform, torch.zeros(3)
    )
    assert camera_pose.shape == pose.shape
    expected = (transform[:3, :3] @ torch.from_numpy(trans).T).T + transform[:3, 3]
    assert np.allclose(camera_trans, expected.numpy(), atol=1e-5)


def test_aist_scene_translation_is_normalized_by_sequence_scaling() -> None:
    translation = np.array([[100.0, -50.0, 25.0]], dtype=np.float32)
    normalized = normalize_aist_translation(translation, 100.0)
    assert normalized.dtype == np.float32
    assert normalized.flags.c_contiguous
    assert np.allclose(normalized, [[1.0, -0.5, 0.25]])

    transform = torch.eye(4, dtype=torch.float32)
    transform[:3, 3] = torch.tensor([100.0, 200.0, -300.0])
    normalized_transform = normalize_aist_camera_extrinsics(transform, 100.0)
    assert torch.allclose(normalized_transform[:3, 3], torch.tensor([1.0, 2.0, -3.0]))
    assert torch.allclose(normalized_transform[:3, :3], transform[:3, :3])
    assert torch.allclose(transform[:3, 3], torch.tensor([100.0, 200.0, -300.0]))


def test_aist_scene_normalization_uses_negative_scale_magnitude() -> None:
    translation = np.array([[100.0, -50.0, 25.0]], dtype=np.float32)
    assert np.allclose(
        normalize_aist_translation(translation, -100.0),
        [[1.0, -0.5, 0.25]],
    )


@pytest.mark.parametrize("scaling", [0.0, float("nan"), float("inf")])
def test_aist_scene_normalization_rejects_invalid_scaling(scaling: float) -> None:
    with pytest.raises(ValueError, match="finite and non-zero"):
        normalize_aist_translation(np.zeros((1, 3), dtype=np.float32), scaling)
    with pytest.raises(ValueError, match="finite and non-zero"):
        normalize_aist_camera_extrinsics(torch.eye(4), scaling)


def test_music_validation_rejects_shape_and_length(tmp_path: Path) -> None:
    good = tmp_path / "good.pt"
    torch.save(_valid_music(4), good)
    assert validate_music_features(good, 4).shape == (4, 35)
    wrong_shape = tmp_path / "shape.pt"
    torch.save(torch.zeros(4, 34), wrong_shape)
    with pytest.raises(ValueError, match=r"\[L, 35\]"):
        validate_music_features(wrong_shape, 4)
    with pytest.raises(ValueError, match="length mismatch"):
        validate_music_features(good, 5)


def test_partial_split_excludes_val_test_and_minitrain_is_deterministic() -> None:
    built = {"a", "b", "c", "d", "e"}
    train, val, test = build_partial_splits(built, {"b", "x"}, {"c", "y"})
    assert train == ["a", "d", "e"]
    assert val == ["b"] and test == ["c"]
    annot = {name: _record(130 if name != "d" else 100) for name in built}
    assert choose_minitrain(train, annot, 2, 120) == ["a", "e"]


def test_atomic_save_can_be_reloaded(tmp_path: Path) -> None:
    annot = {"sequence": _record(4)}
    splits = {"train": ["sequence"], "val": [], "test": [], "minitrain": ["sequence"]}
    paths = {name: tmp_path / f"{name}.pt" for name in splits}
    paths["annot"] = tmp_path / "annot.pt"
    atomic_save_outputs(annot, splits, paths, overwrite=False)
    assert builder.safe_torch_load(paths["annot"])["sequence"]["smpl_pose"].shape == (4, 72)
    assert builder.safe_torch_load(paths["train"]) == ["sequence"]
    assert not list(tmp_path.glob("*.tmp"))


def test_dataset_custom_and_default_filenames_are_backward_compatible(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(aist_dataset_module, "make_smplx", lambda _kind: _DummyBodyModel())
    annot = {"sequence": _record(4)}
    torch.save(annot, tmp_path / "custom_annot.pt")
    torch.save(["sequence"], tmp_path / "custom_split.pt")
    custom = AISTPlusPlusSmplDataset(
        root=tmp_path,
        split="train",
        annot_file="custom_annot.pt",
        split_file="custom_split.pt",
    )
    assert custom.split == "train" and custom.idx2meta == ["sequence"]

    torch.save(annot, tmp_path / "annot_aist_30fps.pt")
    torch.save(["sequence"], tmp_path / "train.pt")
    default = AISTPlusPlusSmplDataset(root=tmp_path, split="train")
    assert default.annot_file is None and default.split_file is None
    assert default.idx2meta == ["sequence"]


def test_music_only_eval_dataset_uses_center_clip_without_raw_audio(
    tmp_path: Path, monkeypatch
) -> None:
    """The validation loader must not require audio_array/ or audio/*.mp3."""
    monkeypatch.setattr(aist_dataset_module, "make_smplx", lambda _kind: _DummyBodyModel())
    sequence = "gBR_sBM_cAll_d04_mBR0_ch01"
    torch.save({sequence: _record(240)}, tmp_path / "annot_aist_30fps.pt")
    torch.save([sequence], tmp_path / "val.pt")
    music_dir = tmp_path / "musicfeat_v2"
    music_dir.mkdir()
    torch.save(_valid_music(240), music_dir / f"{sequence}_musicfeat_fps30.pt")

    dataset = AISTPlusPlusSmplDataset(
        root=tmp_path,
        split="val",
        feat_version="v2",
        strict_music_alignment=True,
        load_raw_music_audio=False,
        eval_motion_frames=120,
        eval_clip_mode="center",
        music_only_conditioning=True,
        enable_contact_supervision=True,
    )
    item = dataset[0]
    assert item["length"] == 120
    assert item["meta"]["start_end"] == (60, 180)
    assert item["music_embed"].shape == (120, 35)
    assert item["music_embed"][0, 0].item() == 60
    assert item["music_array"].shape == (120, 1024)
    assert torch.count_nonzero(item["music_array"]) == 0
    assert not item["mask"]["has_2d_mask"].any()
    assert item["mask"]["has_music_mask"].all()
    assert item["mask"]["invalid_contact"] is False


def test_music_only_respects_per_sequence_invalid_contact_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(aist_dataset_module, "make_smplx", lambda _kind: _DummyBodyModel())
    sequence = "gBR_sBM_cAll_d04_mBR0_ch01"
    record = _record(120)
    record["contact_supervision_valid"] = False
    torch.save({sequence: record}, tmp_path / "annot_aist_30fps.pt")
    torch.save([sequence], tmp_path / "train.pt")
    music_dir = tmp_path / "musicfeat_v2"
    music_dir.mkdir()
    torch.save(_valid_music(120), music_dir / f"{sequence}_musicfeat_fps30.pt")

    dataset = AISTPlusPlusSmplDataset(
        root=tmp_path,
        split="train",
        feat_version="v2",
        enable_contact_supervision=True,
    )
    item = dataset[0]
    assert item["meta"]["contact_supervision_valid"] is False
    assert item["mask"]["invalid_contact"] is True


def _write_synthetic_tree(root: Path) -> tuple[Path, Path]:
    annotations = root / "annotations"
    music_root = root / "musicfeat_v2"
    for name in ("motions", "keypoints2d", "cameras", "splits"):
        (annotations / name).mkdir(parents=True, exist_ok=True)
    music_root.mkdir()
    sequences = ["gAA_sBM_cAll_d01_mAA0_ch01", "gAA_sBM_cAll_d01_mAA0_ch02"]
    camera_payload = [_camera(f"c{index:02d}") for index in range(9, 0, -1)]
    (annotations / "cameras" / "environment.json").write_text(
        json.dumps(camera_payload), encoding="utf-8"
    )
    (annotations / "cameras" / "mapping.txt").write_text(
        "".join(f"{sequence} environment\n" for sequence in sequences), encoding="utf-8"
    )
    (annotations / "ignore_list.txt").write_text("", encoding="utf-8")
    for split in ("crossmodal_train", "crossmodal_val", "crossmodal_test"):
        (annotations / "splits" / f"{split}.txt").write_text("", encoding="utf-8")
    for sequence in sequences:
        motion = {
            "smpl_poses": np.zeros((8, 72), dtype=np.float32),
            "smpl_scaling": np.array([100.0], dtype=np.float32),
            "smpl_trans": np.zeros((8, 3), dtype=np.float32),
        }
        with (annotations / "motions" / f"{sequence}.pkl").open("wb") as file:
            pickle.dump(motion, file)
        keypoints = np.zeros((9, 8, 17, 3), dtype=np.float32)
        keypoints[..., :4, 0] = np.array([10, 30, 15, 25])
        keypoints[..., :4, 1] = np.array([20, 40, 35, 25])
        keypoints[..., :4, 2] = 1
        with (annotations / "keypoints2d" / f"{sequence}.pkl").open("wb") as file:
            pickle.dump({"keypoints2d": keypoints}, file)
    torch.save(_valid_music(4), music_root / f"{sequences[0]}_musicfeat_fps30.pt")
    return annotations, music_root


def test_dry_run_skips_missing_music_and_writes_only_reports(
    tmp_path: Path, monkeypatch
) -> None:
    annotations, music_root = _write_synthetic_tree(tmp_path)
    monkeypatch.setattr(builder, "make_smplx", lambda _kind: _DummyBodyModel())
    output = tmp_path / "output"
    report = tmp_path / "report"
    assert (
        builder.main(
            [
                "--annotations-root",
                str(annotations),
                "--musicfeat-dir",
                str(music_root),
                "--output-root",
                str(output),
                "--report-dir",
                str(report),
                "--dry-run",
            ]
        )
        == 0
    )
    assert not output.exists()
    summary = json.loads((report / "build_summary.json").read_text(encoding="utf-8"))
    assert summary["successfully_built_count"] == 1
    assert summary["skipped_missing_musicfeat"] == 1
    assert summary["status"] == "dry_run_complete"
