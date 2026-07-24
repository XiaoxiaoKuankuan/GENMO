from __future__ import annotations

import json
import pickle

import torch

from gem.datasets.pure_motion.motionxpp import MotionXppDataset
from tools.data.motionxpp.common import atomic_write_jsonl


class _DummySMPL:
    def __call__(self, body_pose, betas, global_orient, transl):
        joints = torch.zeros(body_pose.shape[0], 24, 3, dtype=body_pose.dtype)
        joints[..., 1] = 1.0
        return joints

    def get_skeleton(self, betas):
        return torch.zeros(1, 3, dtype=betas.dtype)


def _record(frames: int, caption: str) -> dict:
    pose = torch.zeros(frames, 66, dtype=torch.float32)
    pose[:, 1] = torch.linspace(0, 0.1, frames)
    trans = torch.zeros(frames, 3, dtype=torch.float32)
    trans[:, 0] = torch.linspace(0, 1, frames)
    trans[:, 1] = 1
    return {
        "pose": pose.contiguous(),
        "trans": trans.contiguous(),
        "beta": torch.zeros(10, dtype=torch.float32),
        "gender": "neutral",
        "fps": 30.0,
        "text_data": [{"caption": caption, "source": "toy"}],
        "source_subset": "toy",
        "source_path": f"toy/{caption}.npy",
        "source_group": f"toy:{caption}",
        "content_hash": caption,
        "motion_hash": caption,
    }


def _artifacts(tmp_path):
    support = tmp_path / "genmo_support"
    motion_rel = "shards/train/motion_00000.pth"
    motion_path = support / motion_rel
    motion_path.parent.mkdir(parents=True)
    motion = {
        "toy__long": _record(150, "Long motion."),
        "toy__short": _record(40, "Short motion."),
    }
    torch.save(motion, motion_path)
    rows = [
        {
            "motion_id": key,
            "shard_path": motion_rel,
            "record_key": key,
            "frames": value["pose"].shape[0],
            "caption_count": 1,
            "subset": "toy",
            "source_group": value["source_group"],
            "content_hash": value["content_hash"],
            "motion_hash": value["motion_hash"],
            "fps": 30.0,
        }
        for key, value in motion.items()
    ]
    manifest = support / "manifests/train.jsonl"
    atomic_write_jsonl(manifest, rows)

    embedding_root = tmp_path / "t5_embeddings_v1_half"
    embedding_rel = "shards/train/embed_00000.pth"
    embedding_path = embedding_root / embedding_rel
    embedding_path.parent.mkdir(parents=True)
    torch.save(
        {
            key: torch.full((1, 50, 1024), index + 1, dtype=torch.float16)
            for index, key in enumerate(motion)
        },
        embedding_path,
    )
    embedding_manifest = embedding_root / "manifests/train.json"
    embedding_manifest.parent.mkdir(parents=True)
    embedding_manifest.write_text(
        json.dumps(
            {
                "split": "train",
                "motion_to_shard": {
                    key: {
                        "shard_path": embedding_rel,
                        "record_key": key,
                        "caption_count": 1,
                    }
                    for key in motion
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest, embedding_manifest


def _dataset(tmp_path, monkeypatch, **kwargs):
    monkeypatch.setattr(
        "gem.datasets.pure_motion.base_dataset.make_smplx",
        lambda *args, **kw: _DummySMPL(),
    )
    motion_manifest, embedding_manifest = _artifacts(tmp_path)
    return MotionXppDataset(
        root=tmp_path,
        manifest_path=motion_manifest,
        embedding_manifest_path=embedding_manifest,
        split="train",
        motion_frames=120,
        cam_augmentation="static",
        shard_cache_size=2,
        random_seed=7,
        **kwargs,
    )


def test_dataset_random_crop_and_text_contract(tmp_path, monkeypatch):
    dataset = _dataset(tmp_path, monkeypatch)
    sample = dataset[0]
    assert sample["length"] == 120
    assert sample["smpl_params_w"]["body_pose"].shape == (120, 63)
    assert sample["text_embed"].shape == (50, 1024)
    assert sample["text_embed"].dtype == torch.float32
    assert sample["caption"] == "Long motion."
    assert sample["has_text"] is True
    assert sample["mask"]["valid"].sum() == 120
    assert sample["mask"]["has_2d_mask"].sum() == 0
    assert sample["mask"]["2d_only"] is False
    assert "music_embed" not in sample
    assert "audio_array" not in sample


def test_dataset_short_motion_padding_and_valid_length(tmp_path, monkeypatch):
    dataset = _dataset(tmp_path, monkeypatch)
    sample = dataset[1]
    assert sample["length"] == 40
    assert sample["smpl_params_w"]["body_pose"].shape == (120, 63)
    assert sample["mask"]["valid"].sum() == 40
    assert torch.count_nonzero(sample["smpl_params_w"]["body_pose"][40:]) == 0


def test_dataset_pickle_drops_worker_shard_caches(tmp_path, monkeypatch):
    dataset = _dataset(tmp_path, monkeypatch)
    dataset[0]
    assert dataset._motion_cache
    assert dataset._embedding_cache
    restored = pickle.loads(pickle.dumps(dataset))
    assert not restored._motion_cache
    assert not restored._embedding_cache
    assert restored._rng is None


def test_keypoint_condition_refuses_uncalibrated_data(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "gem.datasets.pure_motion.base_dataset.make_smplx",
        lambda *args, **kw: _DummySMPL(),
    )
    motion_manifest, embedding_manifest = _artifacts(tmp_path)
    try:
        MotionXppDataset(
            root=tmp_path,
            manifest_path=motion_manifest,
            embedding_manifest_path=embedding_manifest,
            condition_on_keypoints=True,
        )
    except NotImplementedError as exc:
        assert "calibrated" in str(exc)
    else:
        raise AssertionError("uncalibrated keypoint conditioning must be rejected")
