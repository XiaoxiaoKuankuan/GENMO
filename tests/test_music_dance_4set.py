"""Contracts for the four-dataset music-only training path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from gem.datamodule.mocap_trainX_testY import collate_fn
from gem.datasets.music_dance.music_dance_smpl import (
    MusicDanceSmplDataset,
    duration_repeat_count,
    select_training_window,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _compose(exp: str):
    with initialize_config_dir(
        version_base="1.3", config_dir=str(REPO_ROOT / "configs")
    ):
        return compose(config_name="train", overrides=[f"exp={exp}"])


def test_four_dataset_music_only_config_composes_without_changing_baseline() -> None:
    cfg = _compose("gem_smpl_music_only_4set")
    assert list(cfg.train_datasets) == [
        "aistpp_train",
        "aioz_gdance_train",
        "finedance_train",
        "compas3d_train",
    ]
    assert list(cfg.pipeline.args.train_modes) == ["diffusion"]
    assert list(cfg.pipeline.args.in_attr) == ["encoded_music"]
    assert cfg.pipeline.args.encoded_music_dim == 35
    assert cfg.model.model_cfg.text_encoder is None
    assert cfg.network.model_cfg.denoiser.encode_text is False
    assert all(
        value.duration_aware_sampling is True
        for value in cfg.train_datasets.values()
    )

    baseline = _compose("gem_smpl_music_only")
    assert list(baseline.train_datasets) == ["aistpp_train"]
    assert baseline.train_datasets.aistpp_train.get(
        "duration_aware_sampling", False
    ) is False


@pytest.mark.parametrize(
    ("frames", "expected"), [(1, 1), (119, 1), (120, 1), (121, 1), (240, 2), (359, 2), (360, 3)]
)
def test_duration_repeat_count_is_complete_window_proportional(
    frames: int, expected: int
) -> None:
    assert duration_repeat_count(frames, 120) == expected


def test_training_window_includes_last_legal_crop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("numpy.random.randint", lambda low, high: high - 1)
    assert select_training_window(119, 120) == (0, 119)
    assert select_training_window(120, 120) == (0, 120)
    assert select_training_window(121, 120) == (1, 120)
    assert select_training_window(240, 120) == (120, 120)


class _DummySmpl:
    def get_skeleton(self, _betas: torch.Tensor) -> torch.Tensor:
        return torch.zeros(24, 3, dtype=torch.float32)


def _write_canonical_root(root: Path, *, name: str, length: int = 240) -> None:
    (root / "motions").mkdir(parents=True)
    (root / "musicfeat_v2").mkdir()
    (root / "manifests").mkdir()
    pose = torch.zeros(length, 66, dtype=torch.float32)
    motion = {
        "global_orient": pose[:, :3].clone(),
        "body_pose": pose[:, 3:].clone(),
        "transl": torch.stack(
            [torch.arange(length) / 1000, torch.zeros(length), torch.zeros(length)], dim=-1
        ),
        "betas": torch.zeros(length, 10),
    }
    music = torch.zeros(length, 35)
    music[::2, 33] = 1
    music[1::2, 34] = 1
    torch.save(motion, root / "motions" / f"{name}.pt")
    torch.save(music, root / "musicfeat_v2" / f"{name}.pt")
    row = {
        "sample_id": name,
        "motion_path": f"motions/{name}.pt",
        "music_feature_path": f"musicfeat_v2/{name}.pt",
        "fps": 30,
        "num_frames": length,
        "split": "train",
    }
    (root / "manifests" / "train.jsonl").write_text(
        json.dumps(row) + "\n", encoding="utf-8"
    )


def test_manifest_dataset_emits_exact_music_only_batch_contract(tmp_path: Path) -> None:
    _write_canonical_root(tmp_path, name="dance")
    dataset = MusicDanceSmplDataset(
        tmp_path,
        dataset_name="fixture",
        duration_aware_sampling=True,
    )
    dataset._smpl_model = _DummySmpl()
    assert len(dataset) == 2
    sample = dataset[0]
    assert sample["music_embed"].shape == (120, 35)
    assert sample["music_beats"].shape == (120,)
    assert torch.equal(sample["music_beats"], sample["music_embed"][:, 34])
    for key, width in {
        "body_pose": 63,
        "global_orient": 3,
        "transl": 3,
        "betas": 10,
    }.items():
        assert sample["smpl_params_w"][key].shape == (120, width)
        assert sample["smpl_params_c"][key].shape == (120, width)
    assert sample["f_imgseq"].shape == (120, 1024)
    assert sample["kp2d"].shape == (120, 17, 3)
    assert sample["K_fullimg"].shape == (120, 3, 3)
    assert sample["T_w2c"].shape == (120, 4, 4)
    assert sample["mask"]["has_music_mask"].all()
    for key in (
        "has_img_mask",
        "has_2d_mask",
        "has_cam_mask",
        "has_audio_mask",
    ):
        assert not sample["mask"][key].any()
    assert sample["mask"]["invalid_contact"] is False
    tensors = [
        sample["music_embed"],
        sample["smpl_params_w"]["body_pose"],
        sample["smpl_params_w"]["transl"],
        sample["smpl_params_c"]["transl"],
        sample["R_c2gv"],
        sample["K_fullimg"],
        sample["T_w2c"],
    ]
    assert all(torch.isfinite(value).all() for value in tensors)

    cfg = OmegaConf.create(
        {
            "max_motion_frames": 120,
            "default_frame_feature_dim": {
                "music_array": [1024],
                "music_embed": [35],
                "music_beats": [],
                "audio_array": [],
                "use_det_kp": [],
            },
            "default_seq_feature_dim": {"text_embed": [50, 1024]},
            "default_seq_feature_length_multiplier": {"audio_array": 600},
            "default_feature_val": {
                "caption": "",
                "music_fps": 30,
                "audio_fps": 30,
                "has_text": False,
            },
            "default_feature_type": {},
        }
    )
    batch = collate_fn([sample, sample], mode="train", collate_cfg=cfg)
    assert batch["music_embed"].shape == (2, 120, 35)
    assert batch["smpl_params_w"]["body_pose"].shape == (2, 120, 63)
    assert batch["mask"]["has_music_mask"].all()
    assert not batch["has_text"].any()


def test_short_sequence_is_synchronously_padded_to_one_full_window(tmp_path: Path) -> None:
    _write_canonical_root(tmp_path, name="short", length=102)
    dataset = MusicDanceSmplDataset(
        tmp_path,
        dataset_name="fixture",
        duration_aware_sampling=True,
    )
    dataset._smpl_model = _DummySmpl()
    sample = dataset[0]
    assert len(dataset) == 1
    assert sample["length"] == 120
    assert sample["meta"]["source_crop_length"] == 102
    assert sample["music_embed"].shape == (120, 35)
    assert sample["smpl_params_w"]["body_pose"].shape == (120, 63)
    assert sample["mask"]["valid"].all()
    assert sample["mask"]["has_music_mask"].all()
    assert torch.equal(sample["music_embed"][101], sample["music_embed"][119])
    assert torch.equal(
        sample["smpl_params_w"]["transl"][101],
        sample["smpl_params_w"]["transl"][119],
    )
