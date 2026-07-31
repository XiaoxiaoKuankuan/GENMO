"""CPU-only tests for the GEM-SMPL server training preflight."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import open_dict

import tools.train.preflight_gem_smpl as preflight
from gem.datasets.rich.rich_utils import load_rich_artifact


def _music(length: int) -> torch.Tensor:
    value = torch.zeros(length, 35, dtype=torch.float32)
    value[::2, 33] = 1
    value[1::2, 34] = 1
    return value


def _make_aist(root: Path, *, extra_music: bool = False) -> None:
    root.mkdir(parents=True)
    music = root / "musicfeat_v2"
    music.mkdir()
    splits = {"train": ["a", "b"], "val": ["v"], "test": ["t"], "minitrain": ["a"]}
    annot = {
        name: {"smpl_pose_global": np.zeros((4, 72), dtype=np.float32)}
        for name in splits["train"] + splits["val"] + splits["test"]
    }
    torch.save(annot, root / "annot_aist_30fps.pt")
    for name, values in splits.items():
        torch.save(values, root / f"{name}.pt")
    for name in annot:
        torch.save(_music(4), music / f"{name}_musicfeat_fps30.pt")
    if extra_music:
        torch.save(_music(4), music / "extra_musicfeat_fps30.pt")


def test_official_aist_split_counts_and_contract(tmp_path: Path) -> None:
    root = tmp_path / "AIST++"
    _make_aist(root)
    result = preflight.check_aist_artifacts(
        root,
        expected_annot=4,
        expected_train=2,
        expected_val=1,
        expected_test=1,
        expected_minitrain=1,
    )
    assert result["annot"] == 4 and result["split_union_equals_annot"] is True


def test_extra_aist_music_features_are_allowed(tmp_path: Path) -> None:
    root = tmp_path / "AIST++"
    _make_aist(root, extra_music=True)
    result = preflight.check_aist_artifacts(
        root,
        expected_annot=4,
        expected_train=2,
        expected_val=1,
        expected_test=1,
        expected_minitrain=1,
    )
    assert result["music_feature_count"] == 5
    assert result["extra_music_feature_count"] == 1


def test_humanml3d_key_mismatch_fails(tmp_path: Path) -> None:
    motion = tmp_path / "motion.pth"
    embedding = tmp_path / "embedding.pth"
    torch.save({"motion": {"text_data": [{"caption": "walk", "tokens": ["walk"]}]}}, motion)
    torch.save({"different": torch.zeros(1, 50, 1024)}, embedding)
    with pytest.raises(preflight.PreflightError, match="keys differ"):
        preflight.check_humanml3d_artifacts(motion, embedding, expected_motion_count=1)


def _make_beat2(root: Path, *, include_audio: bool = True) -> None:
    item = {"video_id": "clip", "subset": "beat_english_v2.0.0", "length": 120}
    torch.save(
        {"train": [item], "val": [item], "test": [item], "minitrain": [item]},
        root / "all_splits.pth",
    )
    motion = root / item["subset"] / "smplxflame_30" / "clip.npz"
    audio = root / item["subset"] / "wave16k" / "clip.wav"
    motion.parent.mkdir(parents=True)
    audio.parent.mkdir(parents=True)
    motion.write_bytes(b"npz placeholder")
    if include_audio:
        audio.write_bytes(b"wav placeholder")


def test_beat2_valid_index_and_non_exact_train_flag(tmp_path: Path) -> None:
    _make_beat2(tmp_path)
    result = preflight.check_beat2_artifacts(tmp_path)
    assert result["train"] == 1
    assert result["checked_indexed_pairs"] == 4
    assert result["non_exact_beat2_train_count"] is True


def test_beat2_missing_indexed_pair_fails(tmp_path: Path) -> None:
    _make_beat2(tmp_path, include_audio=False)
    with pytest.raises(preflight.PreflightError, match="indexed pair missing"):
        preflight.check_beat2_artifacts(tmp_path)


def test_server_config_excludes_3dpw_occ_and_accepts_missing_optional_dir() -> None:
    cfg = preflight.compose_training_config(Path.cwd(), "gem_smpl_server")
    result = preflight.validate_server_config(cfg)
    assert "3dpw_occ_v1" not in cfg.train_datasets
    assert result["regression_only"] is False
    optional = Path("inputs/3DPW/hmr4d_support/imgfeats/3dpw_occ_train")
    assert not optional.exists()


def test_server_config_defaults_to_eight_gpus_and_batch_128_per_gpu() -> None:
    cfg = preflight.compose_training_config(Path.cwd(), "gem_smpl_server")
    assert cfg.pl_trainer.devices == 8
    assert cfg.pl_trainer.num_nodes == 1
    assert cfg.pl_trainer.strategy == "auto"
    assert cfg.pl_trainer.precision == "16-mixed"
    assert cfg.data.loader_opts.train.batch_size == 128


def test_server_config_rejects_3dpw_occ() -> None:
    cfg = preflight.compose_training_config(Path.cwd(), "gem_smpl_server")
    with open_dict(cfg.train_datasets):
        cfg.train_datasets["3dpw_occ_v1"] = {"_target_": "fake.Dataset"}
    with pytest.raises(preflight.PreflightError, match="train datasets"):
        preflight.validate_server_config(cfg)


def _mixed_batch() -> dict:
    batch_size = 2
    return {
        "B": batch_size,
        "length": torch.tensor([120, 120]),
        "text_embed": torch.zeros(batch_size, 50, 1024),
        "music_embed": torch.zeros(batch_size, 120, 35),
        "f_imgseq": torch.zeros(batch_size, 120, 1024),
        "audio_array": torch.zeros(batch_size, 72000),
        "mask": {"valid": torch.ones(batch_size, 120, dtype=torch.bool)},
        "meta": [{"data_name": "aist++"}, {"data_name": "beat2"}],
    }


def test_mixed_batch_shape_validation() -> None:
    result = preflight.validate_mixed_batch(_mixed_batch(), 2)
    assert result["text_embed"] == [2, 50, 1024]
    assert result["music_embed"] == [2, 120, 35]
    assert result["sources"] == ["aist++", "beat2"]


def test_recursive_finite_check_rejects_nan() -> None:
    with pytest.raises(preflight.PreflightError, match="NaN or Inf"):
        preflight.assert_all_finite({"nested": {"value": torch.tensor([float("nan")])}}, "fake")


def test_json_report_is_written_atomically(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    preflight._write_report(path, {"status": "passed", "count": 2})
    assert json.loads(path.read_text()) == {"status": "passed", "count": 2}
    assert not path.with_name("report.json.tmp").exists()


def test_strict_failure_returns_nonzero_and_writes_report(tmp_path: Path, monkeypatch) -> None:
    def fail(_args):
        raise preflight.PreflightError("injected failure")

    monkeypatch.setattr(preflight, "run_preflight", fail)
    report = tmp_path / "report.json"
    result = preflight.main(
        ["--repo-root", str(tmp_path), "--report", str(report), "--strict"]
    )
    assert result == 1
    payload = json.loads(report.read_text())
    assert payload["status"] == "failed" and "injected failure" in payload["error"]


def test_original_gem_smpl_yaml_is_unchanged() -> None:
    digest = hashlib.sha256(Path("configs/exp/gem_smpl.yaml").read_bytes()).hexdigest()
    assert digest == preflight.ORIGINAL_GEM_SMPL_SHA256


def test_rich_trusted_artifact_loader_supports_numpy_payload(tmp_path: Path) -> None:
    path = tmp_path / "rich.pt"
    torch.save({"scalar": np.float64(1.25)}, path)
    assert float(load_rich_artifact(path)["scalar"]) == 1.25
